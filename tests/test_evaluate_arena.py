import unittest
from unittest.mock import patch, MagicMock

from agents import RandomAgent, HeuristicAgent
from game_context import GameContext
from evaluate_arena import TournamentConfig, TournamentRunner
from fast_engine import JoeEngine


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

    @patch.object(JoeEngine, 'apply', autospec=True)
    def test_round_win_tracking_survives_the_next_deal(self, mock_apply):
        """
        Verify a round win and a down-and-out win are credited from the round
        index, even though the next round's deal refills the hands at once.
        """
        from actions import GoDown

        # 1. SETUP: Mock agents that always choose to go down
        mock_agent = MagicMock()
        mock_agent.select_action.return_value = GoDown(())
        config = TournamentConfig(
            name="Round Win Tracking Test",
            agents=[mock_agent, mock_agent, mock_agent, mock_agent],
            num_games=1,
            rounds_per_game=1
        )
        runner = TournamentRunner(config)

        # Stand in for engine.apply: walk agent 0 through pickup -> go down -> discard -> out
        def fake_apply(engine, action):
            ctx = engine.ctx
            state_id = engine.state_id
            ctx.active_player_idx = 0  # agent 0 makes every following decision

            # Point at the engine's own state objects (never change their .id)
            if state_id == 'pickup_decision':
                engine.current_state = engine.go_down_decision
            elif state_id == 'go_down_decision':
                engine.current_state = engine.discard_phase
            elif state_id == 'discard_phase':
                ctx.current_round_idx += 1
                ctx.last_round_result = 'out'
                ctx.players[0].hand_list = [MagicMock()] * 11  # as if already redealt
                engine.current_state = engine.game_over

        mock_apply.side_effect = fake_apply

        # 2. ACT
        results = runner.simulate()

        # 3. ASSERT
        self.assertEqual(results['round_win_rate'], 100.0, "Failed to track the round win!")
        self.assertEqual(results['down_and_out_wins'], 1.0, "Failed to track the down-and-out win!")

    def test_round_cap_log_records_the_rounds_final_hands(self):
        """The round-cap log shows the hands the round ended with, not the next deal."""
        import os
        import tempfile

        from evaluate_arena import _log_round_cap_hands

        ctx = GameContext(num_players=3)
        ctx.execute_deal()
        ctx.total_actions = 300  # 100 circuits of 3 players
        final_hand = sorted(ctx.players[0].hand_list, key=lambda c: (int(c.suit), int(c.rank)))
        ctx.calculate_scores()
        ctx.execute_deal()  # the engine deals the next round straight away

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                _log_round_cap_hands(ctx)
                log = open("round_cap_tracker.log", encoding="utf-8").read()
            finally:
                os.chdir(cwd)

        self.assertIn("Circuit 100", log)
        self.assertIn("Round 1 Objective", log)
        self.assertIn(", ".join(str(c) for c in final_hand), log)

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