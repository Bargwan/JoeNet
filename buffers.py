import h5py
import numpy as np


class JoeReplayBuffer:
    """
    HDF5-backed Circular Replay Buffer for JoeNet Actor-Critic & Belief State training.
    Directly streams tensors to the hard drive to prevent RAM exhaustion.
    """

    def __init__(self, filepath="replay_memory.h5", max_size=100000):
        self.filepath = filepath
        self.max_size = max_size

        # Open in 'append' mode to safely create or attach to an existing file
        self.file = h5py.File(filepath, 'a')

        # Initialize datasets based strictly on Phase 2 Information Set specifications
        if 'spatial' not in self.file:
            self.file.create_dataset('spatial', shape=(max_size, 13, 4, 14), dtype=np.int8)
            self.file.create_dataset('scalar', shape=(max_size, 28), dtype=np.float32)
            self.file.create_dataset('action_mask', shape=(max_size, 58), dtype=np.bool_)

            # New Actor-Critic/Belief State variables
            self.file.create_dataset('oracle_truth', shape=(max_size, 3, 4, 14), dtype=np.int8)
            self.file.create_dataset('terminal_score', shape=(max_size, 1), dtype=np.float32)
            self.file.create_dataset('policy', shape=(max_size, 58), dtype=np.float32)

            self.file.create_dataset('current_size', shape=(), dtype=np.int32, data=0)
            self.file.create_dataset('ptr', shape=(), dtype=np.int32, data=0)

    def add(self, spatial, scalar, action_mask, oracle_truth, terminal_score, policy):
        """Streams a single experience tuple directly to disk."""
        ptr = self.file['ptr'][()]
        current_size = self.file['current_size'][()]

        # Write directly to the hard drive at the current pointer index
        self.file['spatial'][ptr] = spatial
        self.file['scalar'][ptr] = scalar
        self.file['action_mask'][ptr] = action_mask
        self.file['oracle_truth'][ptr] = oracle_truth
        self.file['terminal_score'][ptr] = terminal_score
        self.file['policy'][ptr] = policy

        # Update pointer (loop back to 0 if we hit max_size)
        self.file['ptr'][()] = (ptr + 1) % self.max_size

        # Update current_size limit
        if current_size < self.max_size:
            self.file['current_size'][()] = current_size + 1

    def add_batch(self, spatial, scalar, action_mask, oracle_truth, terminal_score, policy):
        """Streams an entire batch of experiences directly to disk at once."""
        batch_size = spatial.shape[0]
        ptr = self.file['ptr'][()]
        current_size = self.file['current_size'][()]

        space_left = self.max_size - ptr

        if batch_size <= space_left:
            # Fits cleanly in the remaining contiguous block
            self.file['spatial'][ptr:ptr + batch_size] = spatial
            self.file['scalar'][ptr:ptr + batch_size] = scalar
            self.file['action_mask'][ptr:ptr + batch_size] = action_mask
            self.file['oracle_truth'][ptr:ptr + batch_size] = oracle_truth
            self.file['terminal_score'][ptr:ptr + batch_size] = terminal_score
            self.file['policy'][ptr:ptr + batch_size] = policy

            self.file['ptr'][()] = (ptr + batch_size) % self.max_size
        else:
            # Wrap-around block (Hits the end of the buffer and wraps to the start)
            chunk1 = space_left
            chunk2 = batch_size - space_left

            # First chunk to the end of the buffer
            self.file['spatial'][ptr:self.max_size] = spatial[:chunk1]
            self.file['scalar'][ptr:self.max_size] = scalar[:chunk1]
            self.file['action_mask'][ptr:self.max_size] = action_mask[:chunk1]
            self.file['oracle_truth'][ptr:self.max_size] = oracle_truth[:chunk1]
            self.file['terminal_score'][ptr:self.max_size] = terminal_score[:chunk1]
            self.file['policy'][ptr:self.max_size] = policy[:chunk1]

            # Second chunk to the beginning
            self.file['spatial'][0:chunk2] = spatial[chunk1:]
            self.file['scalar'][0:chunk2] = scalar[chunk1:]
            self.file['action_mask'][0:chunk2] = action_mask[chunk1:]
            self.file['oracle_truth'][0:chunk2] = oracle_truth[chunk1:]
            self.file['terminal_score'][0:chunk2] = terminal_score[chunk1:]
            self.file['policy'][0:chunk2] = policy[chunk1:]

            self.file['ptr'][()] = chunk2

        if current_size < self.max_size:
            self.file['current_size'][()] = min(self.max_size, current_size + batch_size)

    def sample(self, batch_size):
        """Extracts a randomized batch from disk into RAM."""
        current_size = self.file['current_size'][()]
        if current_size < batch_size:
            raise ValueError(
                f"Not enough data to sample batch of {batch_size}. Current size: {current_size}")

        # 1. Select random indices
        indices = np.random.choice(current_size, batch_size, replace=False)

        # 2. HDF5 requires indices to be in strictly increasing order for "fancy indexing"
        indices.sort()

        # 3. Pull only the requested batch into RAM
        b_spatial = self.file['spatial'][indices]
        b_scalar = self.file['scalar'][indices]
        b_mask = self.file['action_mask'][indices]
        b_oracle = self.file['oracle_truth'][indices]
        b_score = self.file['terminal_score'][indices]
        b_policy = self.file['policy'][indices]

        return b_spatial, b_scalar, b_mask, b_oracle, b_score, b_policy

    def close(self):
        """Safely closes the file connection."""
        if self.file:
            self.file.close()