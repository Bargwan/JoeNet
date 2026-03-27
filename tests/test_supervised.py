import os
import unittest
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import h5py

from buffers import JoeReplayBuffer
from train_supervised import load_buffer_to_ram, train_supervised_epoch


# Mocking the JoeNet architecture for the test to isolate the training loop logic
class MockJoeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # A dummy parameter so the optimizer has something to update
        self.dummy_weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, spatial, scalar, action_mask):
        batch_size = spatial.shape[0]
        # Return mock outputs matching the exact shapes defined in the spec
        mock_logits = torch.ones((batch_size, 58), requires_grad=True) * self.dummy_weight
        mock_ev = torch.ones((batch_size, 1), requires_grad=True) * self.dummy_weight
        mock_oracle = torch.sigmoid(
            torch.ones((batch_size, 3, 4, 14), requires_grad=True) * self.dummy_weight)
        return mock_logits, mock_ev, mock_oracle


class TestSupervisedTrainingLoop(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_supervised_buffer.h5"
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

        # Pre-fill a tiny buffer with 10 rows of perfectly formatted Phase 1 data
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
        """
        Verify the helper function successfully pulls the HDF5 data entirely into RAM,
        casts to the correct PyTorch dtypes, and builds a TensorDataset.
        """
        dataset = load_buffer_to_ram(self.test_file)

        self.assertIsInstance(dataset, TensorDataset)
        self.assertEqual(len(dataset), 10, "Dataset did not load all 10 rows.")

        spatial, scalar, mask, oracle, score, policy = dataset[0]

        # Verify the PyTorch dtypes are perfect for GPU processing
        self.assertEqual(spatial.dtype, torch.float32, "Spatial must be cast to float32 for CNNs")
        self.assertEqual(scalar.dtype, torch.float32)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(oracle.dtype, torch.float32, "Oracle Truth must be float32 for BCE Loss")
        self.assertEqual(score.dtype, torch.float32)
        self.assertEqual(policy.dtype, torch.float32)

    def test_composite_loss_backward_pass(self):
        """
        Verify the training epoch successfully calculates the 3 composite losses
        (CE, MSE, BCE) and successfully flows gradients backward to update the weights.
        """
        dataset = load_buffer_to_ram(self.test_file)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

        model = MockJoeNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        initial_weight = model.dummy_weight.item()

        # ACT: Run one supervised training epoch
        metrics = train_supervised_epoch(model, dataloader, optimizer, device=torch.device('cpu'))

        # ASSERT: Metrics returned
        self.assertIn('actor_loss', metrics)
        self.assertIn('critic_loss', metrics)
        self.assertIn('oracle_loss', metrics)
        self.assertIn('total_loss', metrics)

        # ASSERT: Weights updated (proving loss.backward() and optimizer.step() executed)
        final_weight = model.dummy_weight.item()
        self.assertNotEqual(initial_weight, final_weight,
                            "Model weights did not update! Backward pass failed.")


if __name__ == '__main__':
    unittest.main()