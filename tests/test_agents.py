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
        from agents import HeuristicAgent
        from game_context import GameContext

        agent = HeuristicAgent()
        ctx = GameContext(num_players=4)

        # 12 actions / 4 players = Circuit 3 (True early game)
        # Pressure = 3 / 8.0 = 0.375. max(0.5, 0.375) == 0.5.
        ctx.total_actions = 12

        prob = agent._calculate_go_down_probability(ctx, 0)

        self.assertEqual(prob, 0.5,
                         "Baseline go_down probability should be 0.5 in safe conditions.")

    def test_dynamic_panic_turn_pressure(self):
        """Verify the probability of going down scales linearly based on the dynamic patience threshold."""
        from agents import HeuristicAgent
        from game_context import GameContext

        agent = HeuristicAgent()
        ctx = GameContext(num_players=4)

        # --- THE FIX ---
        # 1. Set to Round 3 (Index 2) -> Patience Threshold is now 10.0 turns (8.0 + 2)
        ctx.current_round_idx = 2

        # 2. Set to 32 actions -> Circuit 8.
        # Pressure = 8 / 10.0 = 0.8
        ctx.total_actions = 32

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

        # =========================================================================
        # NEW STEP 5.1 TESTS: Partial Meld Acceptance Logic
        # =========================================================================

        def test_heuristic_agent_pickup_accepts_partials(self):
            """
            Verify Standard Pickup accepts a card that extends a partial meld,
            even if it does not complete the objective.
            """
            agent = HeuristicAgent()
            ctx = GameContext(num_players=4, config=self.config)
            p0 = ctx.players[0]

            # Scenario: Round 2 ([4,4] - Runs Only)
            ctx.current_round_idx = 2

            # Hand: 5 and 6 of Spades (Partial run of 2)
            p0.hand_list.clear()
            p0.private_hand.fill(0)
            p0.receive_cards([Card(Suit.SPADES, Rank.FIVE), Card(Suit.SPADES, Rank.SIX)])

            # Discard: 7 of Spades (Extends partial to 3, but does NOT complete a 4-card run)
            ctx.discard_pile = [Card(Suit.SPADES, Rank.SEVEN)]

            # Action Mask: 0 (Draw Stock), 1 (Pickup Discard)
            mask = np.zeros(58, dtype=bool)
            mask[0] = True
            mask[1] = True

            action = agent.select_action('pickup_decision', ctx, player_idx=0, action_mask=mask)
            self.assertEqual(
                action,
                1,
                "Agent should pick up the 7 of Spades on its turn to build a partial 3-card run."
            )

        def test_heuristic_agent_may_i_refuses_partials(self):
            """
            Verify May-I strictly refuses a card that only extends a partial meld,
            adhering to the rule that May-I is only for final completion.
            """
            agent = HeuristicAgent()
            ctx = GameContext(num_players=4, config=self.config)
            p0 = ctx.players[0]

            # Scenario: Round 2 ([4,4] - Runs Only)
            ctx.current_round_idx = 2

            # Hand: 5 and 6 of Spades (Partial run of 2)
            p0.hand_list.clear()
            p0.private_hand.fill(0)
            p0.receive_cards([Card(Suit.SPADES, Rank.FIVE), Card(Suit.SPADES, Rank.SIX)])

            # Discard: 7 of Spades (Extends partial to 3, but does NOT complete a 4-card run)
            ctx.discard_pile = [Card(Suit.SPADES, Rank.SEVEN)]

            # Action Mask: 2 (May-I Call), 3 (May-I Pass)
            mask = np.zeros(58, dtype=bool)
            mask[2] = True
            mask[3] = True

            action = agent.select_action('may_i_decision', ctx, player_idx=0, action_mask=mask)
            self.assertEqual(
                action,
                3,
                "Agent should PASS on the May-I because 5-6-7 does not complete the 4-card objective."
            )

        def test_heuristic_agent_pickup_accepts_one_gap(self):
            """
            Verify Option B logic: Agent accepts a card that creates a 1-gap
            (inside straight) because it is within a rank distance of 2.
            """
            agent = HeuristicAgent()
            ctx = GameContext(num_players=4, config=self.config)
            p0 = ctx.players[0]

            # Scenario: Round 2 ([4,4] - Runs Only)
            ctx.current_round_idx = 2

            # Hand: 5 and 6 of Spades (Partial run of 2)
            p0.hand_list.clear()
            p0.private_hand.fill(0)
            p0.receive_cards([Card(Suit.SPADES, Rank.FIVE), Card(Suit.SPADES, Rank.SIX)])

            # Discard: 8 of Spades (Creates a 1-gap: 5-6-[-]-8)
            ctx.discard_pile = [Card(Suit.SPADES, Rank.EIGHT)]

            # Action Mask: 0 (Draw Stock), 1 (Pickup Discard)
            mask = np.zeros(58, dtype=bool)
            mask[0] = True
            mask[1] = True

            action = agent.select_action('pickup_decision', ctx, player_idx=0, action_mask=mask)
            self.assertEqual(
                action,
                1,
                "Agent should pick up the 8 of Spades on its turn to build a 1-gapped run."
            )

class TestHeuristicTablePlay(unittest.TestCase):
    """HeuristicAgent places Key Cards on runs when it can, otherwise on sets."""

    def setUp(self):
        from game_context import TableRun
        self.ctx = GameContext(num_players=4)
        self.player = self.ctx.players[0]
        self.player.hand_list.clear()
        self.player.private_hand.fill(0)
        self.player.is_down = True
        # A set of 7s and the run ♠3-4-5-6 on the table.
        for suit in (Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS):
            self.ctx.table_sets[suit, Rank.SEVEN] = 1
        self.ctx.runs_on_table.append(TableRun(Suit.SPADES, Rank.THREE, Rank.SIX))
        for rank in range(Rank.THREE, Rank.SEVEN):
            self.ctx.table_runs[Suit.SPADES, rank] = 1

    def choose(self):
        mask = self.ctx.get_action_mask(0, 'table_play_phase')
        return HeuristicAgent(random_seed=0).select_action('table_play_phase', self.ctx, 0, mask)

    def test_prefers_a_run_over_a_set(self):
        from game_context import RUN_HIGH_END, place_action
        self.player.receive_cards([Card(Suit.SPADES, Rank.SEVEN)])
        self.assertEqual(self.choose(), place_action(Suit.SPADES, Rank.SEVEN, RUN_HIGH_END))

    def test_uses_a_set_when_no_run_fits(self):
        from game_context import ON_SET, place_action
        self.player.receive_cards([Card(Suit.HEARTS, Rank.SEVEN)])
        self.assertEqual(self.choose(), place_action(Suit.HEARTS, Rank.SEVEN, ON_SET))

    def test_ends_table_play_when_nothing_fits(self):
        from game_context import END_TABLE_PLAY
        self.player.receive_cards([Card(Suit.HEARTS, Rank.KING)])
        self.assertEqual(self.choose(), END_TABLE_PLAY)


class TestHeuristicGoDownChoice(unittest.TestCase):
    """HeuristicAgent goes down with the melds that leave the least deadwood."""

    def ctx_with_hand(self, cards):
        ctx = GameContext(num_players=4)
        ctx.current_round_idx = 0  # [3, 3]
        player = ctx.players[0]
        player.hand_list.clear()
        player.private_hand.fill(0)
        player.receive_cards(cards)
        return ctx

    @staticmethod
    def three(rank):
        return [Card(s, rank) for s in (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS)]

    def test_keeps_the_lowest_value_cards_in_hand(self):
        # Laying 9s and Kings leaves three 7s (15 points), the lowest possible.
        ctx = self.ctx_with_hand(self.three(Rank.SEVEN) + self.three(Rank.NINE)
                                 + self.three(Rank.KING))
        melds = HeuristicAgent(random_seed=0).choose_go_down_melds(ctx, 0)
        self.assertEqual(sorted(melds), [('set', Rank.NINE), ('set', Rank.KING)])

    def test_counts_cards_it_can_place_straight_afterwards(self):
        # A fourth 7 makes laying the 7s better: the extra 7 goes onto the set.
        ctx = self.ctx_with_hand(self.three(Rank.SEVEN) + [Card(Suit.CLUBS, Rank.SEVEN)]
                                 + self.three(Rank.NINE) + self.three(Rank.TWO))
        melds = HeuristicAgent(random_seed=0).choose_go_down_melds(ctx, 0)
        self.assertEqual(sorted(melds), [('set', Rank.SEVEN), ('set', Rank.NINE)])

    def test_breaks_ties_at_random(self):
        # 9s, 10s and Jacks are all worth 10 points: every option leaves 30 points.
        ctx = self.ctx_with_hand(self.three(Rank.NINE) + self.three(Rank.TEN)
                                 + self.three(Rank.JACK))
        chosen = {tuple(sorted(HeuristicAgent(random_seed=s).choose_go_down_melds(ctx, 0)))
                  for s in range(20)}
        self.assertGreater(len(chosen), 1)


if __name__ == '__main__':
    unittest.main()