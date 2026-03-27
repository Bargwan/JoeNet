import unittest
from unittest.mock import patch, MagicMock

from agents import RandomAgent, HeuristicAgent
from game_context import GameContext
from cards import Card, Suit, Rank

# The new Phase 6 implementation
from evaluate_arena import TournamentConfig, TournamentRunner, apply_engine_action


class TestEvaluationArena(unittest.TestCase):

    def test_config_validation(self):
        """Verify the config correctly enforces 3 or 4 player setups."""
        # Must fail if only 2 agents are provided
        with self.assertRaises(ValueError):
            TournamentConfig(name="Invalid", agents=[RandomAgent(), RandomAgent()])

        # Must succeed with 3 agents
        config3 = TournamentConfig(name="3-Player Test", agents=[RandomAgent()] * 3, num_games=1)
        self.assertEqual(len(config3.agents), 3)

    def test_evaluation_metrics_structure(self):
        """
        Step 6.3: Verify the TournamentRunner accepts a config, runs headless games,
        and returns a dictionary containing the exact required Phase 6 metrics.
        """
        # 1. SETUP: Create a tiny 2-game, 1-round tournament
        config = TournamentConfig(
            name="Test Micro-Tournament",
            agents=[RandomAgent(), RandomAgent(), RandomAgent(), RandomAgent()],
            num_games=2,
            rounds_per_game=1
        )

        runner = TournamentRunner(config)

        # 2. ACT: Run the simulation
        results = runner.simulate()

        # 3. ASSERT: Check for the exact keys defined in the Phase 6 plan
        self.assertIsInstance(results, dict)
        self.assertIn("tournament_win_rate", results)
        self.assertIn("round_win_rate", results)
        self.assertIn("avg_point_diff", results)
        self.assertIn("strategic_detonations", results)
        self.assertIn("simulation_time", results)

        # Verify types
        self.assertIsInstance(results["tournament_win_rate"], float)
        self.assertIsInstance(results["strategic_detonations"], float)

    def test_apply_engine_action_dispatch(self):
        """
        Verify that apply_engine_action correctly routes the raw NN output logits (0-57)
        to the appropriate fast_engine methods and translates absolute card indices to relative list indices.
        """
        ctx = GameContext(num_players=4)
        mock_engine = MagicMock()

        # Setup: Give Player 0 a 10 of Hearts
        # Absolute Logit Formula: 6 + (Suit * 13) + Rank
        # Suit.HEARTS = 1, Rank.TEN = 9 -> 6 + 13 + 9 = 28
        target_card = Card(Suit.HEARTS, Rank.TEN)
        ctx.players[0].receive_cards([target_card])

        # Test A: Control Logit (Logit 0 -> resolve_pickup(0))
        apply_engine_action(mock_engine, ctx, state_id='pickup_decision', player_idx=0,
                            action_idx=0)
        mock_engine.resolve_pickup.assert_called_with(0)

        # Test B: Control Logit (Logit 4 -> resolve_go_down(4))
        apply_engine_action(mock_engine, ctx, state_id='go_down_decision', player_idx=0,
                            action_idx=4)
        mock_engine.resolve_go_down.assert_called_with(4)

        # Test C: Card Translation Logit (Logit 28 -> perform_discard(0))
        # It must translate logit 28 into the relative physical hand list index (0)
        apply_engine_action(mock_engine, ctx, state_id='discard_phase', player_idx=0, action_idx=28)
        mock_engine.perform_discard.assert_called_with(0)

        # Test D: Invalid Card Logit Exception
        # Trying to play a card the player doesn't actually hold (e.g. Logit 50)
        with self.assertRaises(ValueError):
            apply_engine_action(mock_engine, ctx, state_id='discard_phase', player_idx=0,
                                action_idx=50)

    @patch('evaluate_arena.apply_engine_action')
    def test_transient_state_round_win_tracking(self, mock_apply):
        """
        Regression Test: Proves that a round win and Strategic Detonation
        are tracked correctly by monitoring current_round_idx, circumventing the
        transient hand-size bug where the engine instantly refills hands to 11.
        """
        # 1. SETUP: Create a tournament with Mock agents
        mock_agent = MagicMock()
        config = TournamentConfig(
            name="Transient State Test",
            agents=[mock_agent, mock_agent, mock_agent, mock_agent],
            num_games=1,
            rounds_per_game=1
        )

        # Force the Mock Agent to always pick action 4 (GO_DOWN) when asked
        mock_agent.select_action.return_value = 4

        runner = TournamentRunner(config)

        # We hook into `apply_engine_action` to simulate the fast_engine transitions
        def side_effect_apply(engine, ctx, state_id, player_idx, action_idx):
            # Force the active player to be Agent 0 so it gets the win credit
            ctx.active_player_idx = 0

            # FIX: Point the instance to the correct class-level state object!
            # Do NOT mutate the .id property of the singleton!
            if state_id == 'pickup_decision':
                engine.current_state = engine.go_down_decision
            elif state_id == 'go_down_decision':
                engine.current_state = engine.discard_phase
            elif state_id == 'discard_phase':
                ctx.current_round_idx += 1
                ctx.players[0].hand_list = [MagicMock()] * 11
                engine.current_state = engine.game_over

        mock_apply.side_effect = side_effect_apply

        # 2. ACT
        results = runner.simulate()

        # 3. ASSERT: The tracking must survive the transient state reset
        self.assertEqual(
            results['round_win_rate'], 100.0,
            "Failed to track round win! The runner is likely still falling for the transient hand-size trap."
        )
        self.assertEqual(
            results['strategic_detonations'], 1.0,
            "Failed to track strategic detonation!"
        )


if __name__ == '__main__':
    unittest.main()