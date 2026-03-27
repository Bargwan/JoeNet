import unittest
import torch
import numpy as np

# This import will fail initially because we haven't written it yet!
from rl_buffer import RolloutBuffer


class TestRolloutBuffer(unittest.TestCase):

    def setUp(self):
        self.buffer = RolloutBuffer()

    def test_add_and_clear_step(self):
        """Verify that step packages can be added and the buffer can be cleanly wiped."""
        # 1. SETUP: Create dummy state data
        spatial = np.zeros((13, 4, 14), dtype=np.float32)
        scalar = np.zeros(28, dtype=np.float32)
        mask = np.ones(58, dtype=bool)
        action = 5
        reward = 1.5
        is_terminal = False

        # 2. ACT
        self.buffer.add(spatial, scalar, mask, action, reward, is_terminal)

        # 3. ASSERT: Buffer size increased
        self.assertEqual(len(self.buffer), 1)

        # 4. ACT & ASSERT: Clearing the buffer resets size to 0
        self.buffer.clear()
        self.assertEqual(len(self.buffer), 0)

    def test_tensor_conversion(self):
        """Verify the buffer correctly stacks individual numpy arrays into batched PyTorch tensors."""
        # 1. SETUP: Dummy data
        spatial = np.zeros((13, 4, 14), dtype=np.float32)
        scalar = np.zeros(28, dtype=np.float32)
        mask = np.ones(58, dtype=bool)

        # Add 3 mock steps to simulate a short game
        dummy_oracle = np.zeros((3, 4, 14), dtype=np.int8)
        for i in range(3):
            self.buffer.add(spatial, scalar, mask, action=i, reward=float(i),
                            is_terminal=(i == 2), oracle_truth=dummy_oracle)

        # 2. ACT: Extract as tensors
        tensors = self.buffer.get_tensors()

        # 3. ASSERT: The output must be a dictionary of PyTorch tensors
        self.assertIsInstance(tensors, dict)
        self.assertIsInstance(tensors['spatial'], torch.Tensor)

        # ASSERT: Check correct batched dimensions (B=3)
        self.assertEqual(tensors['spatial'].shape, (3, 13, 4, 14))
        self.assertEqual(tensors['scalar'].shape, (3, 28))
        self.assertEqual(tensors['mask'].shape, (3, 58))
        self.assertEqual(tensors['action'].shape, (3,))
        self.assertEqual(tensors['reward'].shape, (3,))
        self.assertEqual(tensors['is_terminal'].shape, (3,))

        # ASSERT: Verify data integrity was preserved during stacking
        self.assertEqual(tensors['reward'][-1].item(), 2.0,
                         "Reward data corrupted during tensorification")
        self.assertTrue(tensors['is_terminal'][-1].item(),
                        "Terminal flag corrupted during tensorification")


if __name__ == '__main__':
    unittest.main()