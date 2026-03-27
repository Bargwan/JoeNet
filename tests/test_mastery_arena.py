import unittest
import torch
import numpy as np

# We will build these in the Green phase
from mastery_arena import MasteryTournament, DualBrainAgent
from game_context import GameContext


class TestMasteryArena(unittest.TestCase):

    def test_dual_brain_agent_routing(self):
        """
        Verify the DualBrainAgent routes Player 0's turns to the RL Model,
        and Players 1, 2, and 3's turns to the Baseline Heuristic Model.
        """

        # Create dummy models that return specific identifying scalars of shape (1, 58)
        class DummyRLModel:
            def __call__(self, spatial, scalar, mask):
                logits = torch.zeros((1, 58))
                logits[0, 0] = 99.0  # Strongly prefer action 0
                return logits, torch.tensor([[1.0]]), torch.tensor([[[0.0]]])

        class DummyBaselineModel:
            def __call__(self, spatial, scalar, mask):
                logits = torch.zeros((1, 58))
                logits[0, 0] = -99.0  # Strongly avoid action 0
                logits[0, 1] = 99.0  # Prefer action 1 instead
                return logits, torch.tensor([[0.0]]), torch.tensor([[[1.0]]])

        rl_model = DummyRLModel()
        baseline_model = DummyBaselineModel()

        # Initialize the dual-brain router
        agent = DualBrainAgent(rl_model, baseline_model, device="cpu", temperature=0.1)

        # Create dummy inputs
        spatial = np.zeros((13, 4, 14))
        scalar = np.zeros(28)
        mask = np.ones(58, dtype=bool)

        # 1. Test Player 0 (The RL Challenger)
        action_0 = agent.compute_action_for_player(player_idx=0, spatial=spatial, scalar=scalar,
                                                   mask=mask)
        # Because logits are 99.0, action 0 must always be chosen
        self.assertEqual(action_0, 0, "Player 0 must be routed to the RL model.")

        # 2. Test Player 1 (The Baseline Opponent)
        action_1 = agent.compute_action_for_player(player_idx=1, spatial=spatial, scalar=scalar,
                                                   mask=mask)
        # Because logits are -99.0 for index 0 and 0.0 elsewhere (default), action 0 should NOT be chosen
        self.assertNotEqual(action_1, 0, "Player 1 must be routed to the Baseline model.")

    def test_tournament_score_tracking(self):
        """
        Verify the MasteryTournament correctly accumulates game wins
        and calculates the final win rate for the RL Challenger.
        """
        tournament = MasteryTournament(games=10, agent=None)

        # Simulate 10 games.
        # In Joe, negative points are good.
        # Player 0 (RL) wins 6 games.

        # Win 1
        tournament._record_game_result([-20, 10, 40, 50])
        # Win 2
        tournament._record_game_result([-50, -10, 0, 10])
        # Loss (Player 1 wins)
        tournament._record_game_result([10, -30, 0, 40])
        # Win 3
        tournament._record_game_result([-100, 0, 0, 0])
        # Loss (Player 2 wins)
        tournament._record_game_result([0, 10, -5, 20])
        # Win 4
        tournament._record_game_result([-5, 0, 10, 20])
        # Win 5
        tournament._record_game_result([-1, 10, 10, 10])
        # Loss (Player 3 wins)
        tournament._record_game_result([10, 20, 30, -50])
        # Win 6
        tournament._record_game_result([-10, 0, 0, 0])
        # Loss (Player 1 wins)
        tournament._record_game_result([50, -100, 10, 10])

        stats = tournament.get_statistics()

        self.assertEqual(stats['rl_wins'], 6)
        self.assertEqual(stats['baseline_wins'], 4)
        self.assertEqual(stats['rl_win_rate'], 60.0, "Win rate must be correctly calculated.")


if __name__ == '__main__':
    unittest.main()