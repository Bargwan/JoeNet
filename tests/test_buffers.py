import os
import unittest

import h5py
import numpy as np

# We will build this new unified class in buffers.py next
from buffers import JoeReplayBuffer


class TestHDF5ReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_replay_buffer.h5"
        # Ensure a clean slate before each test
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def tearDown(self):
        # Clean up the file after the test finishes
        if os.path.exists(self.test_file):
            try:
                os.remove(self.test_file)
            except PermissionError:
                pass

    def test_buffer_schema_and_storage(self):
        """
        Step 5.2: Verify the HDF5 Buffer strictly enforces the Section 7 Schema,
        streams to disk successfully, and samples batches correctly.
        """
        # 1. Initialize Buffer with a small max_size for testing
        buffer = JoeReplayBuffer(filepath=self.test_file, max_size=100)

        # 2. ACT: Create dummy data matching our EXACT Phase 2 Information Set specs
        spatial = np.ones((13, 4, 14), dtype=np.int8)
        scalar = np.ones(28, dtype=np.float32) * 0.5
        action_mask = np.zeros(58, dtype=np.bool_)
        action_mask[0:2] = True

        # New Actor-Critic/Belief State Specifics
        oracle_truth = np.ones((3, 4, 14), dtype=np.int8) * 2
        terminal_score = np.array([45.0], dtype=np.float32)
        policy = np.zeros(58, dtype=np.float32)
        policy[0] = 0.8
        policy[1] = 0.2

        # Stream 10 identical experiences directly to the HDF5 disk file
        for _ in range(10):
            buffer.add(spatial, scalar, action_mask, oracle_truth, terminal_score, policy)

        # 3. ASSERT: File on disk exists and has correct pre-allocated shapes/dtypes
        with h5py.File(self.test_file, 'r') as f:
            self.assertEqual(f['spatial'].dtype, np.int8, "Spatial dataset must be int8")
            self.assertEqual(f['scalar'].dtype, np.float32, "Scalar dataset must be float32")
            self.assertEqual(f['action_mask'].dtype, np.bool_, "Action mask must be bool_")
            self.assertEqual(f['oracle_truth'].dtype, np.int8, "Oracle Truth must be int8")
            self.assertEqual(f['terminal_score'].dtype, np.float32,
                             "Terminal Score must be float32")
            self.assertEqual(f['policy'].dtype, np.float32, "Policy must be float32")
            self.assertEqual(f['current_size'][()], 10, "Buffer did not track size correctly")

        # 4. ACT: Sample a randomized batch of 4 directly from disk
        batch = buffer.sample(batch_size=4)

        # 5. ASSERT: Batch shapes and data integrity
        b_spatial, b_scalar, b_mask, b_oracle, b_score, b_policy = batch

        self.assertEqual(b_spatial.shape, (4, 13, 4, 14))
        self.assertEqual(b_scalar.shape, (4, 28))
        self.assertEqual(b_mask.shape, (4, 58))
        self.assertEqual(b_oracle.shape, (4, 3, 4, 14))
        self.assertEqual(b_score.shape, (4, 1))
        self.assertEqual(b_policy.shape, (4, 58))

        # Verify the data wasn't corrupted during the Disk I/O roundtrip
        self.assertEqual(b_oracle[0, 0, 0, 0], 2, "Oracle data corrupted during save/load")
        self.assertAlmostEqual(b_score[0, 0], 45.0, msg="Terminal Score corrupted during save/load")
        self.assertAlmostEqual(b_policy[0, 0], 0.8, msg="Policy corrupted during save/load")

        buffer.close()


if __name__ == '__main__':
    unittest.main()