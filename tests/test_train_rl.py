import unittest
from unittest.mock import MagicMock

# We will build these in the Green phase
from train_rl import calculate_pbrs_reward, RLRunner


class TestTrainRL(unittest.TestCase):

    def test_calculate_pbrs_reward_normal_step(self):
        """
        Verify standard PBRS reward calculation: R = Phi(s') - Phi(s)
        """
        current_phi = 10.0
        next_phi = 15.0

        reward = calculate_pbrs_reward(
            current_phi,
            next_phi,
            is_terminal=False,
            terminal_score=0.0
        )

        # 15.0 - 10.0 = 5.0
        self.assertEqual(reward, 5.0, "PBRS step reward must be the difference in potential.")

    def test_calculate_pbrs_reward_terminal_step(self):
        """
        Verify terminal PBRS anchoring.
        At the terminal state, the future potential is 0, and the environment
        delivers the true Asymmetric Terminal Score.
        Formula: Terminal_Score - Current_Phi
        """
        current_phi = 12.0

        # In Joe, negative points are good. A score of -40.0 is a massive win.
        terminal_score = -40.0

        reward = calculate_pbrs_reward(
            current_phi,
            next_phi=0.0,  # Should be mathematically ignored
            is_terminal=True,
            terminal_score=terminal_score
        )

        # -40.0 - 12.0 = -52.0
        self.assertEqual(reward, -52.0, "Terminal PBRS must rigidly anchor to the true score.")

    def test_rl_runner_update_trigger(self):
        """
        Verify the RLRunner correctly hands the RolloutBuffer to the RLTrainer
        and clears the buffer after a game concludes.
        """
        mock_model = MagicMock()
        mock_trainer = MagicMock()
        mock_buffer = MagicMock()

        # Fake that the buffer has 10 steps of data
        mock_buffer.__len__.return_value = 10
        mock_buffer.get_tensors.return_value = {"dummy": "tensors"}

        runner = RLRunner(mock_model, mock_trainer, mock_buffer)

        # ACT: Simulate the end of an episode
        # We pass a dummy next_value of 0.0 for the terminal boundary
        runner.trigger_update(next_value=0.0)

        # ASSERT: The pipeline must flow perfectly
        mock_buffer.get_tensors.assert_called_once()
        mock_trainer.update.assert_called_once_with({"dummy": "tensors"}, 0.0)
        mock_buffer.clear.assert_called_once()


import numpy as np
# Add EpisodeTracker to your imports from train_rl
from train_rl import calculate_pbrs_reward, RLRunner, EpisodeTracker


class TestEpisodeTracker(unittest.TestCase):

    def test_delayed_step_caching(self):
        """
        Verify that the tracker caches a step and waits for the NEXT turn
        to calculate Phi(s') before pushing to the RolloutBuffer.
        """
        mock_buffer = MagicMock()
        tracker = EpisodeTracker(mock_buffer)

        # 1. SETUP: Dummy data for Turn 1
        spatial_1 = np.zeros((13, 4, 14))
        scalar_1 = np.zeros(28)
        mask_1 = np.ones(58, dtype=bool)
        action_1 = 10
        phi_1 = 5.0
        truth_1 = np.zeros((3, 4, 14))

        # 2. ACT: Agent 0 takes a turn
        tracker.cache_step(spatial_1, scalar_1, mask_1, action_1, phi_1, truth_1)

        # ASSERT: The buffer must NOT be updated yet! We don't know the next state.
        mock_buffer.add.assert_not_called()

        # 3. SETUP: Dummy data for Turn 2 (Agent 0's next turn)
        spatial_2 = np.ones((13, 4, 14))
        scalar_2 = np.ones(28)
        mask_2 = np.ones(58, dtype=bool)
        action_2 = 12
        phi_2 = 8.0
        truth_2 = np.zeros((3, 4, 14))

        # 4. ACT: Agent 0 takes their next turn. This should trigger the push
        # of the FIRST cached step into the buffer.
        tracker.cache_step(spatial_2, scalar_2, mask_2, action_2, phi_2, truth_2)

        # ASSERT: The buffer should now receive Turn 1's data, with a reward calculated
        # using Turn 2's potential (8.0 - 5.0 = 3.0)
        mock_buffer.add.assert_called_once()

        # Extract the arguments passed to buffer.add()
        args, kwargs = mock_buffer.add.call_args

        # Verify the pushed action was action_1
        self.assertEqual(kwargs.get('action', args[3] if len(args) > 3 else None), 10)

        # Verify the PBRS reward was calculated correctly (phi_2 - phi_1)
        self.assertEqual(kwargs.get('reward', args[4] if len(args) > 4 else None), 3.0)
        self.assertFalse(kwargs.get('is_terminal', args[5] if len(args) > 5 else False))

    def test_terminal_step_flush(self):
        """
        Verify that when the game ends, the tracker flushes the final cached
        step into the buffer using the Asymmetric Terminal Score.
        """
        mock_buffer = MagicMock()
        tracker = EpisodeTracker(mock_buffer)

        # 1. SETUP: Cache a step
        tracker.cache_step(np.zeros((13, 4, 14)), np.zeros(28), np.ones(58, dtype=bool), 5,
                           current_phi=10.0, oracle_truth=np.zeros((3, 4, 14)))

        # 2. ACT: The round ends. We flush the cache with the terminal score.
        terminal_score = -20.0  # Agent won by 20 points
        tracker.flush_terminal(terminal_score)

        # 3. ASSERT: Buffer updated with terminal flag and anchored reward (-20.0 - 10.0 = -30.0)
        mock_buffer.add.assert_called_once()
        args, kwargs = mock_buffer.add.call_args

        self.assertEqual(kwargs.get('reward', args[4] if len(args) > 4 else None), -30.0)
        self.assertTrue(kwargs.get('is_terminal', args[5] if len(args) > 5 else True))


if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()