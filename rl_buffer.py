import torch
import numpy as np

class RolloutBuffer:
    """
    A lightweight, fast RAM buffer for storing step-by-step
    RL transitions during a single game rollout.
    """
    def __init__(self):
        self.clear()

    def add(self, spatial: np.ndarray, scalar: np.ndarray, mask: np.ndarray,
            action: int, reward: float, is_terminal: bool, oracle_truth: np.ndarray = None):
        """Stores a single step of the game into RAM."""
        self.spatial.append(spatial)
        self.scalar.append(scalar)
        self.mask.append(mask)
        self.action.append(action)
        self.reward.append(reward)
        self.is_terminal.append(is_terminal)
        self.oracle_truth.append(oracle_truth)

    def clear(self):
        """Wipes the buffer clean for the next game."""
        self.spatial = []
        self.scalar = []
        self.mask = []
        self.action = []
        self.reward = []
        self.is_terminal = []
        self.oracle_truth = []

    def __len__(self):
        return len(self.spatial)

    def get_tensors(self) -> dict:
        """
        Stacks the stored steps into batched PyTorch tensors ready for the GPU.
        """
        if len(self) == 0:
            raise ValueError("Cannot convert an empty buffer to tensors.")

        # Stacking into numpy arrays first is significantly faster than
        # appending PyTorch tensors directly in a loop.
        np_spatial = np.stack(self.spatial)
        np_scalar = np.stack(self.scalar)
        np_mask = np.stack(self.mask)
        np_action = np.array(self.action, dtype=np.int64)
        np_reward = np.array(self.reward, dtype=np.float32)
        np_is_terminal = np.array(self.is_terminal, dtype=np.bool_)
        np_oracle = np.stack(self.oracle_truth)

        # Cast to PyTorch tensors with strict Data Types matching our network contracts
        return {
            'spatial': torch.tensor(np_spatial, dtype=torch.float32),
            'scalar': torch.tensor(np_scalar, dtype=torch.float32),
            'mask': torch.tensor(np_mask, dtype=torch.bool),
            'action': torch.tensor(np_action, dtype=torch.long),
            'reward': torch.tensor(np_reward, dtype=torch.float32),
            'is_terminal': torch.tensor(np_is_terminal, dtype=torch.bool),
            'oracle_truth': torch.tensor(np_oracle, dtype=torch.float32)
        }