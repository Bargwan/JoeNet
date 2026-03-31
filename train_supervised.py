import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


class JoeHDF5Dataset(Dataset):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None  # Delay opening to prevent multiprocessing lock issues

    def __len__(self):
        # Open briefly just to get the total number of recorded steps
        with h5py.File(self.filepath, 'r') as f:
            return f['current_size'][()]

    def __getitem__(self, idx):
        # Open the file on the first fetch
        if self.file is None:
            self.file = h5py.File(self.filepath, 'r')

        # 1. Read ONLY the specific row from the hard drive
        np_spatial = self.file['spatial'][idx]
        np_scalar = self.file['scalar'][idx]
        np_mask = self.file['action_mask'][idx]
        np_oracle = self.file['oracle_truth'][idx]
        np_score = self.file['terminal_score'][idx]
        np_policy = self.file['policy'][idx]

        # 2. Apply preprocessing on the fly
        t_spatial = torch.tensor(np_spatial, dtype=torch.float32)
        presence_channels = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        t_spatial[presence_channels, :, :] /= 2.0

        t_scalar = torch.tensor(np_scalar, dtype=torch.float32)
        t_mask = torch.tensor(np_mask, dtype=torch.bool)
        t_oracle = torch.tensor(np_oracle, dtype=torch.float32).clamp(min=0.0, max=1.0)
        t_score = torch.tensor(np_score, dtype=torch.float32)

        t_policy = torch.tensor(np_policy, dtype=torch.float32).clamp(min=0.0, max=1.0)
        t_policy = t_policy / (t_policy.sum(dim=-1, keepdim=True) + 1e-8)

        return t_spatial, t_scalar, t_mask, t_oracle, t_score, t_policy


def train_supervised_epoch(model, dataloader, optimizer, device, actor_weights=None):
    """
    Executes one epoch of Behavioral Cloning across all three network heads simultaneously.
    """
    model.train()

    # Define the composite loss functions
    actor_criterion = nn.CrossEntropyLoss(weight=actor_weights)
    critic_criterion = nn.MSELoss()
    oracle_criterion = nn.BCELoss()

    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_oracle_loss = 0.0
    total_combined_loss = 0.0

    for batch in dataloader:
        t_spatial, t_scalar, t_mask, t_oracle_truth, t_score, t_policy = [b.to(device) for b in
                                                                          batch]

        optimizer.zero_grad()

        # Forward Pass
        logits, ev, oracle_probs = model(t_spatial, t_scalar, action_mask=t_mask)

        # 1. Actor Loss (Behavioral Cloning the Heuristic Policy)
        # Note: PyTorch's CrossEntropyLoss automatically applies LogSoftmax to the raw logits
        actor_loss = actor_criterion(logits, t_policy)

        # 2. Critic Loss (Predicting the Asymmetric Terminal Score)
        critic_loss = critic_criterion(ev, t_score)

        # 3. Oracle Loss (Predicting the exact hidden opponent hands)
        oracle_loss = oracle_criterion(oracle_probs, t_oracle_truth)

        # The Composite Loss
        loss = actor_loss + critic_loss + oracle_loss

        # Backward Pass & Optimize
        loss.backward()
        optimizer.step()

        total_actor_loss += actor_loss.item()
        total_critic_loss += critic_loss.item()
        total_oracle_loss += oracle_loss.item()
        total_combined_loss += loss.item()

    num_batches = len(dataloader)

    # Return metrics for the orchestration script to log
    return {
        'actor_loss': total_actor_loss / num_batches,
        'critic_loss': total_critic_loss / num_batches,
        'oracle_loss': total_oracle_loss / num_batches,
        'total_loss': total_combined_loss / num_batches
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from network import JoeNet
    import os

    # 1. Configuration
    data_path = "joe_phase1_sandbox.h5"
    save_path = "models/joenet_phase2_cloned.pth"
    batch_size = 4096
    epochs = 10
    learning_rate = 1e-3

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Cannot find training data at {data_path}. Did you run the generator?")

    # 2. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Phase 2: Behavioral Cloning ---")
    print(f"Device: {device}")

    # 3. Load Data
    print(f"Connecting to HDF5 buffer (Lazy Loading)...")
    dataset = JoeHDF5Dataset(data_path)
    # Important: Leave num_workers=0 (default) when streaming from HDF5 to avoid thread locking
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Dataset Connected: {len(dataset)} total steps.")

    print("Calculating dynamic class weights (Chunked to save RAM)...")
    class_counts = torch.zeros(58, dtype=torch.float64)

    with h5py.File(data_path, 'r') as f:
        size = f['current_size'][()]
        chunk_size = 500000
        for i in range(0, size, chunk_size):
            end = min(i + chunk_size, size)
            # Load just one chunk of the policy array at a time
            policy_chunk = torch.tensor(f['policy'][i:end], dtype=torch.float32)
            class_counts += policy_chunk.sum(dim=0).double()

    total_actions = class_counts.sum()
    actor_weights = total_actions / (58.0 * (class_counts + 1.0))
    actor_weights = torch.clamp(actor_weights, max=10.0).to(device).float()

    # 4. Initialize Model & Optimizer
    model = JoeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize TensorBoard Writer for Phase 2
    writer = SummaryWriter(log_dir="runs/joenet_phase2_supervised")

    # 5. Training Loop
    for epoch in range(epochs):
        metrics = train_supervised_epoch(model, dataloader, optimizer, device, actor_weights)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Actor Loss: {metrics['actor_loss']:.4f} | "
              f"Critic Loss: {metrics['critic_loss']:.4f} | "
              f"Oracle Loss: {metrics['oracle_loss']:.4f}")

        # --- TENSORBOARD BROADCAST ---
        writer.add_scalar('Loss/Actor', metrics['actor_loss'], epoch + 1)
        writer.add_scalar('Loss/Critic', metrics['critic_loss'], epoch + 1)
        writer.add_scalar('Loss/Oracle', metrics['oracle_loss'], epoch + 1)
        writer.add_scalar('Loss/Total', metrics['total_loss'], epoch + 1)

    writer.close()

    # 6. Save the final weights
    torch.save(model.state_dict(), save_path)
    print(f"--- Training Complete! Weights saved to {save_path} ---")