import os
import unittest
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import h5py

from buffers import JoeReplayBuffer
from train_supervised import train_supervised_epoch


def mock_load_buffer(filepath):
    """
    Replaces the old load_buffer_to_ram for testing purposes.
    Simulates the preprocessing step that happens in the chunked loop.
    """
    with h5py.File(filepath, 'r') as f:
        # Grab the exact number of populated rows
        size = f['current_size'][()]

        np_spatial = f['spatial'][:size]
        np_scalar = f['scalar'][:size]
        np_mask = f['action_mask'][:size]
        np_oracle = f['oracle_truth'][:size]
        np_score = f['terminal_score'][:size]
        np_policy = f['policy'][:size]

    t_spatial = torch.tensor(np_spatial, dtype=torch.float32)
    presence_channels = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    t_spatial[:, presence_channels, :, :] /= 2.0

    t_scalar = torch.tensor(np_scalar, dtype=torch.float32)
    t_mask = torch.tensor(np_mask, dtype=torch.bool)
    t_oracle = torch.tensor(np_oracle, dtype=torch.float32).clamp(min=0.0, max=1.0)
    t_score = torch.tensor(np_score, dtype=torch.float32)
    t_policy = torch.tensor(np_policy, dtype=torch.float32).clamp(min=0.0, max=1.0)
    t_policy = t_policy / (t_policy.sum(dim=-1, keepdim=True) + 1e-8)

    return TensorDataset(t_spatial, t_scalar, t_mask, t_oracle, t_score, t_policy)


class MockJoeNet(torch.nn.Module):
    """Mocking the split architecture so the modes can target specific heads."""

    def __init__(self):
        super().__init__()
        self.dummy_weight = torch.nn.Parameter(torch.ones(1))

    # --- Mocking the Sub-Modules ---
    def oracle(self, spatial, scalar):
        batch_size = spatial.shape[0]
        return torch.sigmoid(
            torch.ones((batch_size, 3, 4, 14), requires_grad=True) * self.dummy_weight)

    def critic(self, expanded_spatial, scalar):
        batch_size = expanded_spatial.shape[0]
        return torch.ones((batch_size, 1), requires_grad=True) * self.dummy_weight

    def actor(self, expanded_spatial, scalar, action_mask):
        batch_size = expanded_spatial.shape[0]
        return torch.ones((batch_size, 58), requires_grad=True) * self.dummy_weight


class TestSupervisedTrainingLoop(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_supervised_buffer.h5"
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

        # Pre-fill a tiny buffer with 10 rows
        buffer = JoeReplayBuffer(filepath=self.test_file, max_size=100)
        for _ in range(10):
            buffer.add(
                spatial=np.zeros((13, 4, 14), dtype=np.int8),
                scalar=np.zeros(28, dtype=np.float32),
                action_mask=np.ones(58, dtype=np.bool_),
                oracle_truth=np.ones((3, 4, 14), dtype=np.int8),
                terminal_score=np.array([-10.5], dtype=np.float32),
                policy=np.ones(58, dtype=np.float32) / 58.0  # Uniform distribution
            )
        buffer.close()

    def tearDown(self):
        if os.path.exists(self.test_file):
            try:
                os.remove(self.test_file)
            except PermissionError:
                pass

    def test_ram_dataloader_conversion(self):
        dataset = mock_load_buffer(self.test_file)

        self.assertIsInstance(dataset, TensorDataset)
        self.assertEqual(len(dataset), 10, "Dataset did not load all 10 rows.")

        spatial, scalar, mask, oracle, score, policy = dataset[0]

        self.assertEqual(spatial.dtype, torch.float32)
        self.assertEqual(oracle.dtype, torch.float32)

    def test_oracle_mode_backward_pass(self):
        """Verify Phase 2A strictly isolates and updates the Oracle."""
        dataset = mock_load_buffer(self.test_file)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

        model = MockJoeNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        initial_weight = model.dummy_weight.item()

        metrics = train_supervised_epoch(model, dataloader, optimizer, torch.device('cpu'),
                                         mode='oracle')

        # Tuple: (act_loss, crit_loss, ora_loss, tot_loss, entropy, num_batches)
        self.assertEqual(len(metrics), 6)
        self.assertGreater(metrics[2], 0.0, "Oracle loss should be calculated.")
        self.assertEqual(metrics[0], 0.0, "Actor loss should be ignored in Oracle mode.")

        final_weight = model.dummy_weight.item()
        self.assertNotEqual(initial_weight, final_weight, "Oracle weights did not update!")

    def test_decision_mode_backward_pass(self):
        """Verify Phase 2B strictly isolates and updates the Actor/Critic."""
        dataset = mock_load_buffer(self.test_file)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

        model = MockJoeNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        initial_weight = model.dummy_weight.item()

        metrics = train_supervised_epoch(model, dataloader, optimizer, torch.device('cpu'),
                                         mode='decision')

        # Tuple: (act_loss, crit_loss, ora_loss, tot_loss, entropy, num_batches)
        self.assertEqual(len(metrics), 6)
        self.assertGreater(metrics[0], 0.0, "Actor loss should be calculated.")
        self.assertGreater(metrics[1], 0.0, "Critic loss should be calculated.")
        self.assertEqual(metrics[2], 0.0, "Oracle loss should be ignored in Decision mode.")

        final_weight = model.dummy_weight.item()
        self.assertNotEqual(initial_weight, final_weight, "Decision weights did not update!")


if __name__ == '__main__':
    unittest.main()