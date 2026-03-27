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
    t_scalar = torch.tensor(np_scalar, dtype=torch.float32)
    t_mask = torch.tensor(np_mask, dtype=torch.bool)

    # BCE Loss requires float32 targets, so we cast the int8 oracle truth here
    t_oracle = torch.tensor(np_oracle, dtype=torch.float32)
    t_score = torch.tensor(np_score, dtype=torch.float32)
    t_policy = torch.tensor(np_policy, dtype=torch.float32)

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