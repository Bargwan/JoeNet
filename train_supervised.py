import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
from network import JoeNet
import os
from torch.utils.tensorboard import SummaryWriter


class JoeHDF5Dataset(Dataset):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None

    def __len__(self):
        with h5py.File(self.filepath, 'r') as f:
            return f['current_size'][()]

    def __getitem__(self, indices):
        # NEW: 'indices' is now a full list of 4,096 random numbers, not a single integer.
        if self.file is None:
            self.file = h5py.File(self.filepath, 'r')

        # HDF5 'fancy indexing' requires the requested row numbers to be in strictly increasing order
        sorted_indices = sorted(indices)

        # 1. BULK READ: Pull all 4,096 rows from the SSD in a single, lightning-fast operation
        np_spatial = self.file['spatial'][sorted_indices]
        np_scalar = self.file['scalar'][sorted_indices]
        np_mask = self.file['action_mask'][sorted_indices]
        np_oracle = self.file['oracle_truth'][sorted_indices]
        np_score = self.file['terminal_score'][sorted_indices]
        np_policy = self.file['policy'][sorted_indices]

        # 2. Batch Preprocessing
        t_spatial = torch.tensor(np_spatial, dtype=torch.float32)
        presence_channels = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        t_spatial[:, presence_channels, :, :] /= 2.0  # Added batch dimension (:)

        t_scalar = torch.tensor(np_scalar, dtype=torch.float32)
        t_mask = torch.tensor(np_mask, dtype=torch.bool)
        t_oracle = torch.tensor(np_oracle, dtype=torch.float32).clamp(min=0.0, max=1.0)
        t_score = torch.tensor(np_score, dtype=torch.float32)

        t_policy = torch.tensor(np_policy, dtype=torch.float32).clamp(min=0.0, max=1.0)
        t_policy = t_policy / (t_policy.sum(dim=-1, keepdim=True) + 1e-8)

        return t_spatial, t_scalar, t_mask, t_oracle, t_score, t_policy


def train_supervised_epoch(model, dataloader, optimizer, device, actor_weights=None):
    model.train()
    actor_criterion = nn.CrossEntropyLoss(weight=actor_weights)
    critic_criterion = nn.MSELoss()
    oracle_criterion = nn.BCELoss()

    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_oracle_loss = 0.0
    total_combined_loss = 0.0
    total_entropy = 0.0
    num_batches = 0 # Track local batches

    for batch in dataloader:
        t_spatial, t_scalar, t_mask, t_oracle_truth, t_score, t_policy = [b.to(device) for b in batch]

        optimizer.zero_grad()
        logits, ev, oracle_probs = model(t_spatial, t_scalar, action_mask=t_mask)

        # --- CALCULATE WEIGHT SOFTNESS (ENTROPY) ---
        # 1. Convert raw logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        # 2. H(P) = -sum(p * log(p))
        # We add 1e-8 to the log to prevent it from exploding if a probability is exactly 0
        batch_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
        total_entropy += batch_entropy.item()

        actor_loss = actor_criterion(logits, t_policy)
        critic_loss = critic_criterion(ev, t_score)
        oracle_loss = oracle_criterion(oracle_probs, t_oracle_truth)

        loss = actor_loss + critic_loss + oracle_loss
        loss.backward()
        optimizer.step()

        total_actor_loss += actor_loss.item()
        total_critic_loss += critic_loss.item()
        total_oracle_loss += oracle_loss.item()
        total_combined_loss += loss.item()
        num_batches += 1

    return (total_actor_loss, total_critic_loss, total_oracle_loss,
            total_combined_loss, total_entropy, num_batches)


if __name__ == "__main__":
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

    # 3. Determine Dataset Size
    with h5py.File(data_path, 'r') as f:
        total_size = f['current_size'][()]
    print(f"Dataset Size: {total_size:,} total steps.")

    print("Calculating dynamic class weights (Chunked to save RAM)...")
    class_counts = torch.zeros(58, dtype=torch.float64)
    with h5py.File(data_path, 'r') as f:
        weight_chunk = 500000
        for i in range(0, total_size, weight_chunk):
            end = min(i + weight_chunk, total_size)
            policy_chunk = torch.tensor(f['policy'][i:end], dtype=torch.float32)
            class_counts += policy_chunk.sum(dim=0).double()

    total_actions = class_counts.sum()
    actor_weights = total_actions / (58.0 * (class_counts + 1.0))
    actor_weights = torch.clamp(actor_weights, max=10.0).to(device).float()

    # 4. Initialize Model & Optimizer
    model = JoeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir="runs/joenet_phase2_supervised")

    # NEW: Import standard PyTorch RAM utilities
    from torch.utils.data import TensorDataset, DataLoader

    # 5. The Chunked Training Loop
    CHUNK_SIZE = 1_000_000  # Load 1 million rows into RAM at a time (~3.5 GB footprint)

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        ep_act_loss, ep_crit_loss, ep_ora_loss, ep_tot_loss, ep_entropy = 0.0, 0.0, 0.0, 0.0, 0.0
        ep_batches = 0

        # Stream the 24GB file in 1-million-row blocks
        for start_idx in range(0, total_size, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, total_size)
            print(f"  Loading RAM Chunk: {start_idx:,} to {end_idx:,}...")

            with h5py.File(data_path, 'r') as f:
                np_spatial = f['spatial'][start_idx:end_idx]
                np_scalar = f['scalar'][start_idx:end_idx]
                np_mask = f['action_mask'][start_idx:end_idx]
                np_oracle = f['oracle_truth'][start_idx:end_idx]
                np_score = f['terminal_score'][start_idx:end_idx]
                np_policy = f['policy'][start_idx:end_idx]

            # Preprocess directly into RAM
            t_spatial = torch.tensor(np_spatial, dtype=torch.float32)
            presence_channels = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            t_spatial[:, presence_channels, :, :] /= 2.0

            t_scalar = torch.tensor(np_scalar, dtype=torch.float32)
            t_mask = torch.tensor(np_mask, dtype=torch.bool)
            t_oracle = torch.tensor(np_oracle, dtype=torch.float32).clamp(min=0.0, max=1.0)
            t_score = torch.tensor(np_score, dtype=torch.float32)
            t_policy = torch.tensor(np_policy, dtype=torch.float32).clamp(min=0.0, max=1.0)
            t_policy = t_policy / (t_policy.sum(dim=-1, keepdim=True) + 1e-8)

            # Create a rapid-fire DataLoader purely out of the RAM chunk
            chunk_dataset = TensorDataset(t_spatial, t_scalar, t_mask, t_oracle, t_score, t_policy)
            chunk_loader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True)

            # Train on the chunk
            c_act, c_crit, c_ora, c_tot, c_ent, c_batches = train_supervised_epoch(
                model, chunk_loader, optimizer, device, actor_weights)

            ep_act_loss += c_act
            ep_crit_loss += c_crit
            ep_ora_loss += c_ora
            ep_tot_loss += c_tot
            ep_entropy += c_ent
            ep_batches += c_batches

            # Explicitly delete the variables to free the 3.5GB of RAM for the next chunk
            del np_spatial, t_spatial, chunk_dataset, chunk_loader

        # Final Epoch Averages
        avg_act = ep_act_loss / ep_batches
        avg_crit = ep_crit_loss / ep_batches
        avg_ora = ep_ora_loss / ep_batches
        avg_tot = ep_tot_loss / ep_batches
        avg_ent = ep_entropy / ep_batches

        print(
            f"Epoch {epoch + 1} | Actor: {avg_act:.4f} | Critic: {avg_crit:.4f} |"
            f" Oracle: {avg_ora:.4f} | Entropy: {avg_ent:.4f}")

        # Broadcast to TensorBoard
        writer.add_scalar('Loss/Actor', avg_act, epoch + 1)
        writer.add_scalar('Loss/Critic', avg_crit, epoch + 1)
        writer.add_scalar('Loss/Oracle', avg_ora, epoch + 1)
        writer.add_scalar('Loss/Total', avg_tot, epoch + 1)
        writer.add_scalar('Health/Actor_Entropy', avg_ent, epoch + 1)

        # --- Checkpoint Save ---
        epoch_save_path = f"models/joenet_phase2_ep{epoch + 1}.pth"
        torch.save(model.state_dict(), epoch_save_path)
        print(f"  [Checkpoint] Epoch {epoch + 1} saved to {epoch_save_path}")

    writer.close()
    torch.save(model.state_dict(), save_path)
    print(f"\n--- Training Complete! Weights saved to {save_path} ---")