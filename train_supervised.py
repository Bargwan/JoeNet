import h5py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset


def load_buffer_to_ram(filepath: str) -> TensorDataset:
    """
    Loads the entire HDF5 Replay Buffer into RAM and casts it to
    GPU-ready PyTorch Tensors to eliminate Disk I/O bottlenecks.
    """
    with h5py.File(filepath, 'r') as f:
        size = f['current_size'][()]

        # Read the entire active buffer block directly into RAM
        np_spatial = f['spatial'][:size]
        np_scalar = f['scalar'][:size]
        np_mask = f['action_mask'][:size]
        np_oracle = f['oracle_truth'][:size]
        np_score = f['terminal_score'][:size]
        np_policy = f['policy'][:size]

    # Cast to PyTorch tensors
    # CNNs require float32 inputs, so we cast the int8 spatial tensor here
    t_spatial = torch.tensor(np_spatial, dtype=torch.float32)
    # We must scale the presence channels by 2.0 to match the live Arena preprocessing
    presence_channels = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    t_spatial[:, presence_channels, :, :] /= 2.0
    t_scalar = torch.tensor(np_scalar, dtype=torch.float32)
    t_mask = torch.tensor(np_mask, dtype=torch.bool)

    # BCE Loss strictly requires targets between 0.0 and 1.0.
    # Because players can hold duplicates (value = 2), we clamp it to a binary presence mask.
    t_oracle = torch.tensor(np_oracle, dtype=torch.float32).clamp(min=0.0, max=1.0)
    t_score = torch.tensor(np_score, dtype=torch.float32)

    # CrossEntropyLoss ALSO strictly requires soft targets to be between 0.0 and 1.0.
    # We clamp to erase any float32 precision drift (e.g., 1.0000001) from Phase 1.
    t_policy = torch.tensor(np_policy, dtype=torch.float32).clamp(min=0.0, max=1.0)
    t_policy = t_policy / (t_policy.sum(dim=-1, keepdim=True) + 1e-8)

    return TensorDataset(t_spatial, t_scalar, t_mask, t_oracle, t_score, t_policy)


def train_supervised_epoch(model, dataloader, optimizer, device):
    """
    Executes one epoch of Behavioral Cloning across all three network heads simultaneously.
    """
    model.train()

    # Define the composite loss functions
    actor_criterion = nn.CrossEntropyLoss()
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
    batch_size = 512
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
    print(f"Loading HDF5 buffer into RAM...")
    dataset = load_buffer_to_ram(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Dataset Loaded: {len(dataset)} total steps.")

    # 4. Initialize Model & Optimizer
    model = JoeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 5. Training Loop
    for epoch in range(epochs):
        metrics = train_supervised_epoch(model, dataloader, optimizer, device)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Actor Loss: {metrics['actor_loss']:.4f} | "
              f"Critic Loss: {metrics['critic_loss']:.4f} | "
              f"Oracle Loss: {metrics['oracle_loss']:.4f}")

    # 6. Save the final weights
    torch.save(model.state_dict(), save_path)
    print(f"--- Training Complete! Weights saved to {save_path} ---")