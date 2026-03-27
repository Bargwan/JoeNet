import unittest
import numpy as np

# We will build these in agents.py next
from agents import RandomAgent, HeuristicAgent
from game_context import GameContext
from config import JoeConfig
from cards import Card, Suit, Rank


class TestBaselineAgents(unittest.TestCase):

    def setUp(self):
        """
        Sets up a mock configuration with an explicit objective map
        to ensure the engine's DFS knows what to search for.
        """
        self.config = JoeConfig()
        # Mocking the round objectives:
        # Round 0: [3,3] (2 sets, 0 runs)
        # Round 2: [4,4] (0 sets, 2 runs)
        self.config.objective_map = {
            0: (2, 0),
            2: (0, 2)
        }

    def test_random_agent_mask_compliance(self):
        """Verify RandomAgent strictly selects actions where the mask is True."""
        agent = RandomAgent()

        # Create a mock mask where only GO_DOWN (4) and WAIT (5) are valid
        mask = np.zeros(58, dtype=bool)
        mask[4] = True
        mask[5] = True

        choices = set()
        # Run 100 times to ensure it respects the mask and actually randomizes
        for _ in range(100):
            action = agent.select_action(state_id='go_down_decision', ctx=None, player_idx=0,
                                         action_mask=mask)
            self.assertTrue(mask[action], f"RandomAgent chose an invalid action index: {action}")
            choices.add(action)

        self.assertIn(4, choices, "RandomAgent never chose GO_DOWN.")
        self.assertIn(5, choices, "RandomAgent never chose WAIT.")

    def test_heuristic_agent_smart_discard_objective_aware(self):
        """Verify Discard heuristic respects specific round objectives."""
        agent = HeuristicAgent()
        ctx = GameContext(num_players=4, config=self.config)
        p0 = ctx.players[0]

        p0.hand_list.clear()
        p0.private_hand.fill(0)

        card_7h1 = Card(Suit.HEARTS, Rank.SEVEN)
        card_7h2 = Card(Suit.HEARTS, Rank.SEVEN)
        card_9s = Card(Suit.SPADES, Rank.NINE)
        card_10s = Card(Suit.SPADES, Rank.TEN)

        p0.receive_cards([card_7h1, card_7h2, card_9s, card_10s])

        idx_7h = agent._get_discard_action_idx(card_7h1)
        idx_9s = agent._get_discard_action_idx(card_9s)
        idx_10s = agent._get_discard_action_idx(card_10s)

        mask = np.zeros(58, dtype=bool)
        mask[idx_7h] = True
        mask[idx_9s] = True
        mask[idx_10s] = True

        # --- Scenario A: Round 0 ([3,3] - Sets Only) ---
        ctx.current_round_idx = 0
        action = agent.select_action('discard_phase', ctx, player_idx=0, action_mask=mask)
        self.assertIn(
            action,
            [idx_9s, idx_10s],
            "In a [3,3] round, agent should discard run-synergy cards and keep the 7s."
        )

        # --- Scenario B: Round 2 ([4,4] - Runs Only) ---
        ctx.current_round_idx = 2
        action = agent.select_action('discard_phase', ctx, player_idx=0, action_mask=mask)
        self.assertEqual(
            action,
            idx_7h,
            "In a [4,4] round, agent should discard set-synergy cards and keep the Spades."
        )

    def test_heuristic_agent_may_i_logic_objective_aware(self):
        """Verify May-I heuristic only calls if the discard strictly completes a required meld."""
        agent = HeuristicAgent()
        ctx = GameContext(num_players=4, config=self.config)
        p0 = ctx.players[0]

        mask = np.zeros(58, dtype=bool)
        mask[2] = True
        mask[3] = True

        def reset_hand_and_receive(cards):
            p0.hand_list.clear()
            p0.private_hand.fill(0)
            p0.receive_cards(cards)

        # --- Scenario A: Round 0 ([3,3] - Sets Only) ---
        ctx.current_round_idx = 0

        ctx.discard_pile = [Card(Suit.SPADES, Rank.EIGHT)]
        reset_hand_and_receive([
            Card(Suit.SPADES, Rank.NINE),
            Card(Suit.SPADES, Rank.TEN),
            Card(Suit.SPADES, Rank.JACK)
        ])
        action = agent.select_action('may_i_decision', ctx, player_idx=0, action_mask=mask)
        self.assertEqual(action, 3, "In a [3,3] round, agent should PASS on a run-completing card.")

        ctx.discard_pile = [Card(Suit.HEARTS, Rank.SEVEN)]
        reset_hand_and_receive([Card(Suit.CLUBS, Rank.SEVEN), Card(Suit.DIAMONDS, Rank.SEVEN)])
        action = agent.select_action('may_i_decision', ctx, player_idx=0, action_mask=mask)
        self.assertEqual(action, 2, "In a [3,3] round, agent should CALL on a set-completing card.")

    # =========================================================================
    # NEW STEP 5.1 TESTS: Dynamic Panic Probability
    # =========================================================================

    def test_dynamic_panic_early_game_baseline(self):
        """Verify the baseline go_down probability is a flat 50% in safe conditions."""
        agent = HeuristicAgent()
        ctx = GameContext(num_players=4)

        # 20 actions / 4 players = Circuit 5 (Early game)
        ctx.total_actions = 20

        prob = agent._calculate_go_down_probability(ctx, 0)

        self.assertEqual(prob, 0.5,
                         "Baseline go_down probability should be 0.5 in safe conditions.")

    def test_dynamic_panic_turn_pressure(self):
        """Verify the probability of going down scales linearly as circuits approach max_turns."""
        agent = HeuristicAgent()
        ctx = GameContext(num_players=4)

        # MOCK: Explicitly set max_turns so the math is guaranteed, regardless of default config
        ctx.config.max_turns = 50

        # 160 actions / 4 players = Circuit 40. (40 out of 50 = 80% pressure)
        ctx.total_actions = 160

        prob = agent._calculate_go_down_probability(ctx, 0)

        self.assertEqual(prob, 0.8,
                         "Turn pressure should dynamically increase go_down probability to 0.8.")

    def test_dynamic_panic_opponent_threat(self):
        """Verify the probability spikes inversely to the size of a downed opponent's hand."""
        agent = HeuristicAgent()
        ctx = GameContext(num_players=4)

        # 20 actions / 4 players = Circuit 5 (Safe on circuits)
        ctx.total_actions = 20

        # MOCK THREAT: Op1 is down and only has 2 cards left!
        # Math: 1.0 - (2 cards * 0.1) = 0.8 threat level
        ctx.players[1].is_down = True
        ctx.players[1].hand_list = [Card(Suit.SPADES, Rank.TWO), Card(Suit.HEARTS, Rank.THREE)]

        prob = agent._calculate_go_down_probability(ctx, 0)

        self.assertAlmostEqual(prob, 0.8, places=2,
                               msg="Opponent threat should dynamically scale probability!")

if __name__ == '__main__':
    unittest.main()