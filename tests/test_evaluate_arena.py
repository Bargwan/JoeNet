import unittest
from unittest.mock import patch, MagicMock

from agents import RandomAgent, HeuristicAgent
from game_context import GameContext
from cards import Card, Suit, Rank
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
        Verify the TournamentRunner accepts a config, runs headless games,
        and returns a dictionary of the summary metrics.
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

        # 3. ASSERT: Check the summary metric keys
        self.assertIsInstance(results, dict)
        self.assertIn("tournament_win_rate", results)
        self.assertIn("round_win_rate", results)
        self.assertIn("avg_point_diff", results)
        self.assertIn("strategic_detonations", results)
        self.assertIn("simulation_time", results)

        # Verify types
        self.assertIsInstance(results["tournament_win_rate"], float)
        self.assertIsInstance(results["strategic_detonations"], float)

    def test_apply_engine_action_converts_numbers_to_actions(self):
        """
        Verify the temporary adapter turns an agent's action number into the
        matching action object and plays it with engine.apply().
        """
        from actions import Discard, GoDown, PickStock

        ctx = GameContext(num_players=4)
        mock_engine = MagicMock()
        player = ctx.players[0]
        player.hand_list.clear()
        player.private_hand.fill(0)
        # Two sets (7s and 9s) for the [3, 3] objective, plus a 10 of Hearts
        player.receive_cards([Card(s, r) for r in (Rank.SEVEN, Rank.NINE)
                              for s in (Suit.SPADES, Suit.CLUBS, Suit.DIAMONDS)]
                             + [Card(Suit.HEARTS, Rank.TEN)])

        # Control action 0 -> PickStock
        apply_engine_action(mock_engine, ctx, 'pickup_decision', 0, 0)
        mock_engine.apply.assert_called_with(PickStock())

        # GO_DOWN with no agent -> the first way to go down
        apply_engine_action(mock_engine, ctx, 'go_down_decision', 0, 4)
        mock_engine.apply.assert_called_with(GoDown((('set', Rank.SEVEN), ('set', Rank.NINE))))

        # GO_DOWN with an agent -> the agent's meld choice
        chooser = MagicMock()
        chooser.choose_go_down_melds.return_value = [('set', Rank.NINE), ('set', Rank.SEVEN)]
        apply_engine_action(mock_engine, ctx, 'go_down_decision', 0, 4, agent=chooser)
        mock_engine.apply.assert_called_with(GoDown((('set', Rank.NINE), ('set', Rank.SEVEN))))

        # Discard action 28 = 6 + (Suit * 13) + Rank = 6 + 13 + 9 -> Discard 10 of Hearts
        apply_engine_action(mock_engine, ctx, 'discard_phase', 0, 28)
        mock_engine.apply.assert_called_with(Discard(Suit.HEARTS, Rank.TEN))

        # Discarding a card the player doesn't hold (action 50) raises
        with self.assertRaises(ValueError):
            apply_engine_action(mock_engine, ctx, 'discard_phase', 0, 50)

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
        def side_effect_apply(engine, ctx, state_id, player_idx, action_idx, agent=None):
            # Force the active player to be Agent 0 so it gets the win credit
            ctx.active_player_idx = 0

            # Point at the engine's own state objects (never change their .id)
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
            results['down_and_out_wins'], 1.0,
            "Failed to track strategic detonation!"
        )


    def test_each_round_is_scored_exactly_once(self):
        """A full game scores 7 rounds; the final round must not be scored again at game over."""
        import random

        from evaluate_arena import _worker_simulate_single_game

        random.seed(0)
        config = TournamentConfig(name="Scoring", agents=[HeuristicAgent(random_seed=i)
                                                          for i in range(4)], num_games=1)
        with patch.object(GameContext, 'calculate_scores', autospec=True,
                          side_effect=GameContext.calculate_scores) as scorer:
            _worker_simulate_single_game(config)
        self.assertEqual(scorer.call_count, 7)


if __name__ == '__main__':
    unittest.main()