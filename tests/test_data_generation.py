import os
import unittest
import numpy as np

from game_context import GameContext
from agents import HeuristicAgent
from buffers import JoeReplayBuffer
from config import JoeConfig


class TestDataGenerationPipeline(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_pipeline_buffer.h5"
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def tearDown(self):
        if os.path.exists(self.test_file):
            try:
                os.remove(self.test_file)
            except PermissionError:
                pass

    def test_end_to_end_sandbox_generation(self):
        """
        Verify the pipeline successfully extracts tensors from GameContext,
        generates an Agent policy, and flushes the formatted data to the HDF5 disk.
        """
        # 1. Setup
        config = JoeConfig()
        ctx = GameContext(num_players=4, config=config)
        agent = HeuristicAgent(random_seed=42)
        buffer = JoeReplayBuffer(filepath=self.test_file, max_size=100)

        # 2. Episode Storage
        episode_memory = []

        # 3. ACT: Manually simulate the 4 sequential decision phases of a standard turn
        ctx.execute_deal()

        # Give Player 0 an artificial discard pile to interact with
        ctx.execute_discard(1, 0)

        mock_turn_sequence = [
            'pickup_decision',
            'go_down_decision',
            'table_play_phase',
            'discard_phase'
        ]

        current_player_idx = 0

        for state_id in mock_turn_sequence:
            # A. Extract Information Set
            tensors = ctx.get_input_tensor(current_player_idx, state_id)
            mask = ctx.get_action_mask(current_player_idx, state_id)
            oracle = ctx.get_oracle_truth(current_player_idx)

            # B. Agent Selects Action
            action_idx = agent.select_action(state_id, ctx, current_player_idx, mask)

            # Convert action to a one-hot policy target
            policy = np.zeros(58, dtype=np.float32)
            if action_idx >= 0:
                policy[action_idx] = 1.0

            # C. Store step in RAM temporarily
            episode_memory.append({
                'spatial': tensors['spatial'],
                'scalar': tensors['scalar'],
                'mask': mask,
                'oracle': oracle,
                'policy': policy,
                'player_idx': current_player_idx
            })

        # 4. End of Round Processing
        ctx.calculate_scores()

        # Flush episode memory to the HDF5 disk buffer
        for step in episode_memory:
            p_idx = step['player_idx']
            terminal_score = np.array([float(ctx.players[p_idx].score)], dtype=np.float32)

            buffer.add(
                spatial=step['spatial'],
                scalar=step['scalar'],
                action_mask=step['mask'],
                oracle_truth=step['oracle'],
                terminal_score=terminal_score,
                policy=step['policy']
            )

        buffer.close()

        # 5. ASSERT
        # The buffer must have captured exactly 4 actions
        self.assertEqual(len(episode_memory), 4, "Pipeline failed to record the turn sequence.")

        # Verify the file actually took the data
        import h5py
        with h5py.File(self.test_file, 'r') as f:
            self.assertEqual(f['current_size'][()], 4, "Disk buffer size mismatch!")

    def test_multiprocessing_worker_standalone(self):
        """
        Verify the isolated multiprocessing worker successfully plays
        a full game and returns the correctly formatted RAM data.
        """
        from generate_sandbox_data import _worker_generate_game

        # 1. ACT: Run the worker synchronously for a single game
        # Using a fixed seed for predictability
        game_data = _worker_generate_game(game_seed=42)

        # 2. ASSERT: It should return a list of dictionaries representing the episode
        self.assertIsInstance(game_data, list)
        self.assertGreater(len(game_data), 0, "Worker failed to generate any data.")

        # 3. ASSERT: Check the schema of the returned steps
        first_step = game_data[0]
        self.assertIn('spatial', first_step)
        self.assertIn('scalar', first_step)
        self.assertIn('mask', first_step)
        self.assertIn('oracle', first_step)
        self.assertIn('terminal_score', first_step)
        self.assertIn('policy', first_step)

        # Verify the target is a NumPy array
        self.assertIsInstance(first_step['terminal_score'], np.ndarray)

if __name__ == '__main__':
    unittest.main()