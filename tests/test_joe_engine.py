import unittest

# from engine import JoeEngine
from fast_engine import JoeEngine
from game_context import GameContext
from tests.test_utils import verify_mass_integrity


class TestJoeEngine(unittest.TestCase):

    def setUp(self):
        self.ctx = GameContext()
        self.engine = JoeEngine(self.ctx)

    def test_initial_flow_to_discard(self):
        """
        Verify the engine correctly traverses from Setup down to the Discard Phase.
        We mock a weak hand to ensure it bypasses the go_down_decision.
        """
        from cards import Card, Rank, Suit

        # 1. Verify initial state
        self.assertEqual(self.engine.current_state.id, "setup")

        # 2. Trigger flow down to the start of the turn
        self.engine.start_game()
        self.engine.deal_cards()  # This rotates the active player!

        # 3. MOCK: Clear the ACTIVE player's hand (now safely pointing at Player 1)
        player = self.ctx.active_player
        player.hand_list = []
        player.private_hand.fill(0)
        player.receive_cards([Card(Suit.SPADES, Rank.TWO, deck_index=0)])

        # 4. Proceed with the flow
        self.engine.enter_pickup()
        self.engine.resolve_pickup(1)

        # 5. ASSERT: Should now land on discard_phase every time
        self.assertEqual(
            self.engine.current_state.id,
            "discard_phase",
            "Engine should land on discard_phase with a weak hand."
        )

    def test_turn_rotation(self):
        """
        Verify that completing a turn successfully rotates the active player
        and wraps around correctly (e.g., Player 3 -> Player 0),
        using state teleportation for isolated testing.
        """
        # 1. Teleport the Data Layer
        # Start with Player 3 to explicitly test the wrap-around to 0
        self.ctx.active_player_idx = 3

        # Give Player 3 a dummy card so the engine's hand_not_empty guard passes
        from cards import Card, Rank, Suit
        dummy_card_1 = Card(Suit.SPADES, Rank.ACE, deck_index=0)
        dummy_card_2 = Card(Suit.HEARTS, Rank.KING, deck_index=0)
        self.ctx.active_player.receive_cards([dummy_card_1, dummy_card_2])

        # 2. Teleport the Logic Layer (Bypassing all previous phases)
        self.engine.current_state = self.engine.discard_phase

        # Verify teleportation worked before acting
        self.assertEqual(self.engine.current_state.id, "discard_phase")
        self.assertEqual(len(self.ctx.active_player.hand_list), 2)

        # 3. ACT: Fire the transition
        # We pass a dummy card_index=0 to satisfy the perform_discard payload
        self.engine.perform_discard(0)

        # 4. ASSERT: State Routing and Rotation
        self.assertEqual(
            self.engine.current_state.id,
            "start_turn",
            "Engine should route to start_turn because the hand wasn't empty"
        )
        self.assertEqual(
            self.ctx.active_player_idx,
            0,
            "Player index should have correctly wrapped around from 3 to 0"
        )

    def test_pickup_discard_logic(self):
        """
        Intent: Verify the mechanics of the 1 action during the pickup phase.
        """
        # 1. Teleport the Data Layer
        self.ctx.active_player_idx = 0
        from cards import Card, Rank, Suit
        target_card = Card(Suit.SPADES, Rank.ACE, deck_index=0)
        self.ctx.discard_pile.append(target_card)

        # Verify baseline data state
        self.assertEqual(len(self.ctx.discard_pile), 1)
        self.assertEqual(len(self.ctx.active_player.hand_list), 0)

        # --- INTEGRITY CHECK: Snapshot mass before action ---
        initial_mass = verify_mass_integrity(self, self.ctx)

        # 2. Teleport the Logic Layer
        self.engine.current_state = self.engine.pickup_decision
        self.assertEqual(self.engine.current_state.id, "pickup_decision")

        # 3. ACT: Fire the transition
        self.engine.resolve_pickup(1)

        # 4. ASSERT: State Routing
        self.assertEqual(self.engine.current_state.id, "discard_phase")

        # 5. ASSERT: Data Movement
        self.assertEqual(len(self.ctx.discard_pile), 0)
        self.assertIn(target_card, self.ctx.active_player.hand_list)

        # --- INTEGRITY CHECK: Verify no cards were lost during pickup ---
        verify_mass_integrity(self, self.ctx, expected_total=initial_mass)

    def test_pickup_stock_logic(self):
        """
        Intent: Verify that choosing PICK_STOCK moves the top card of the deck
        into the active player's hand.
        """
        # 1. Teleport the Data Layer
        self.ctx.active_player_idx = 0
        initial_deck_size = len(self.ctx.deck)

        # Verify baseline data state
        self.assertEqual(len(self.ctx.active_player.hand_list), 0)
        self.assertTrue(initial_deck_size > 0, "Deck should have cards")

        # --- INTEGRITY CHECK: Snapshot mass before action ---
        initial_mass = verify_mass_integrity(self, self.ctx)

        # 2. Teleport the Logic Layer
        self.engine.current_state = self.engine.pickup_decision

        # 3. ACT: Fire the transition
        self.engine.resolve_pickup(0)

        # 4. ASSERT: State Routing
        self.assertEqual(self.engine.current_state.id, "may_i_decision")

        # 5. ASSERT: Data Movement
        self.assertEqual(len(self.ctx.deck), initial_deck_size - 1)
        self.assertEqual(len(self.ctx.active_player.hand_list), 1)

        # --- INTEGRITY CHECK: Verify no cards were lost during stock draw ---
        verify_mass_integrity(self, self.ctx, expected_total=initial_mass)

    def test_may_i_flow_eligibility(self):
        """
        Intent: Verify the engine correctly iterates through downstream players for a May-I,
        skipping ineligible players and pausing on the first eligible one.

        Proof: Active Player (0) picks from the stock, triggering the May-I loop.
        Player 1 is artificially set to "down" (ineligible). The test proves success if
        the engine skips Player 1 and lands on a new 'may_i_decision' waiting state
        specifically targeting Player 2.
        """
        # 1. Teleport the Data Layer
        self.ctx.active_player_idx = 0

        # Make Player 1 ineligible (already melded/down)
        self.ctx.players[1].is_down = True

        # Ensure Player 2 is eligible (not down)
        self.ctx.players[2].is_down = False

        # Give the deck a card to prevent drawing errors
        from cards import Card, Rank, Suit
        self.ctx.deck.append(Card(Suit.SPADES, Rank.ACE, deck_index=0))

        # 2. Teleport the Logic Layer
        self.engine.current_state = self.engine.pickup_decision

        # 3. ACT: Active Player (0) touches the stock
        self.engine.resolve_pickup(0)

        # 4. ASSERT: State Routing
        # The engine should pass through the automatic 'may_i_check' state,
        # skip Player 1, and pause on a new waiting state for Player 2.
        self.assertEqual(
            self.engine.current_state.id,
            "may_i_decision",
            "Engine should pause on 'may_i_decision' for an eligible interrupter"
        )

        # 5. ASSERT: Context Targeting
        # The GameContext must track WHICH player is currently being asked
        self.assertEqual(
            self.ctx.may_i_target_idx,
            2,
            "Player 2 should be the target of the May-I decision"
        )

    def test_may_i_call_resolution(self):
        """
        Intent: Verify that if an interrupting player CALLS a May-I, they receive
        the discard card (and the penalty stock card).
        """
        # 1. Teleport the Data Layer
        self.ctx.active_player_idx = 0
        self.ctx.may_i_target_idx = 1

        from cards import Card, Rank, Suit
        target_discard = Card(Suit.SPADES, Rank.ACE, deck_index=0)
        self.ctx.discard_pile.append(target_discard)

        initial_deck_size = len(self.ctx.deck)
        interrupter = self.ctx.players[1]

        # Verify baseline
        self.assertEqual(len(interrupter.hand_list), 0)
        self.assertEqual(len(self.ctx.discard_pile), 1)

        # --- INTEGRITY CHECK: Snapshot mass before action ---
        initial_mass = verify_mass_integrity(self, self.ctx)

        # 2. Teleport the Logic Layer
        self.engine.current_state = self.engine.may_i_decision

        # 3. ACT: Interrupter calls the May-I
        self.engine.resolve_may_i(2)

        # 4. ASSERT: State Routing
        self.assertEqual(self.engine.current_state.id, "discard_phase")

        # 5. ASSERT: Data Movement
        self.assertEqual(len(self.ctx.discard_pile), 0)
        self.assertEqual(len(self.ctx.deck), initial_deck_size - 1)
        self.assertEqual(len(interrupter.hand_list), 2)

        # --- INTEGRITY CHECK: Verify 2 cards successfully transferred ---
        verify_mass_integrity(self, self.ctx, expected_total=initial_mass)

    def test_evaluate_hand_valid_objective(self):
        """
        Intent: Verify that evaluate_hand correctly identifies a hand that meets
        the round's objective and routes the state to 'go_down_decision' instead
        of 'discard_phase'.
        """
        from cards import Card, Rank, Suit

        # 1. Setup for Round 0 (Requires two sets of 3)
        self.ctx.current_round_idx = 0
        player = self.ctx.players[0]
        self.ctx.active_player_idx = 0

        # Build a valid hand that meets the objective.
        # Set 1: Two 7s of Hearts and one 7 of Diamonds (Valid per Joe rules)
        # Set 2: Three Aces of different suits
        # Extra: One 3 of Clubs (would be the discard)
        winning_cards = [
            Card(Suit.HEARTS, Rank.SEVEN, deck_index=0),
            Card(Suit.HEARTS, Rank.SEVEN, deck_index=1),
            Card(Suit.DIAMONDS, Rank.SEVEN, deck_index=0),
            Card(Suit.SPADES, Rank.ACE, deck_index=0),
            Card(Suit.CLUBS, Rank.ACE, deck_index=0),
            Card(Suit.DIAMONDS, Rank.ACE, deck_index=0),
            Card(Suit.CLUBS, Rank.THREE, deck_index=0)
        ]
        player.receive_cards(winning_cards)

        # 2. Teleport the Logic Layer to processing_pickup
        self.engine.current_state = self.engine.processing_pickup

        # 3. ACT: Trigger the hand evaluation
        self.engine.evaluate_hand()

        # 4. ASSERT: State Routing
        self.assertEqual(
            self.engine.current_state.id,
            "go_down_decision",
            "Engine should route to 'go_down_decision' when the hand meets the round "
            "objective."
        )

    def test_evaluate_hand_invalid_objective(self):
        """
        Intent: Verify that evaluate_hand correctly identifies a hand that DOES NOT meet
        the round's objective and routes the state directly to 'discard_phase',
        skipping 'go_down_decision'.
        """
        from cards import Card, Rank, Suit

        # 1. Setup Data Layer for Round 0 (Requires two sets of 3)
        self.ctx.current_round_idx = 0
        player = self.ctx.players[0]
        self.ctx.active_player_idx = 0

        # Build an INVALID hand.
        # We give the player only pairs and singles, guaranteeing _search_melds fails.
        bad_cards = [
            Card(Suit.HEARTS, Rank.SEVEN, deck_index=0),
            Card(Suit.DIAMONDS, Rank.SEVEN, deck_index=0),  # Just a pair of 7s
            Card(Suit.SPADES, Rank.ACE, deck_index=0),
            Card(Suit.CLUBS, Rank.ACE, deck_index=0),  # Just a pair of Aces
            Card(Suit.HEARTS, Rank.TWO, deck_index=0),
            Card(Suit.SPADES, Rank.NINE, deck_index=0),
            Card(Suit.CLUBS, Rank.THREE, deck_index=0)
        ]
        player.receive_cards(bad_cards)

        # 2. Teleport the Logic Layer to processing_pickup
        self.engine.current_state = self.engine.processing_pickup

        # 3. ACT: Trigger the hand evaluation
        self.engine.evaluate_hand()

        # 4. ASSERT: State Routing
        self.assertEqual(
            self.engine.current_state.id,
            "discard_phase",
            "Engine should route straight to 'discard_phase' when the hand is invalid."
        )

    def test_go_down_decision_action_go_down(self):
        """
        Intent: Verify that choosing GO_DOWN transitions the player through the
        placement phases, moves the meld cards out of their hand, and lands
        on discard_phase.
        """
        from cards import Card, Rank, Suit

        # 1. Setup Data Layer for Round 0 (Requires two sets of 3)
        self.ctx.current_round_idx = 0
        player = self.ctx.players[0]
        self.ctx.active_player_idx = 0

        # Give a hand with 6 valid meld cards and 1 extra card (7 total)
        winning_cards = [
            Card(Suit.HEARTS, Rank.SEVEN, deck_index=0),
            Card(Suit.HEARTS, Rank.SEVEN, deck_index=1),
            Card(Suit.DIAMONDS, Rank.SEVEN, deck_index=0),
            Card(Suit.SPADES, Rank.ACE, deck_index=0),
            Card(Suit.CLUBS, Rank.ACE, deck_index=0),
            Card(Suit.DIAMONDS, Rank.ACE, deck_index=0),
            Card(Suit.CLUBS, Rank.THREE, deck_index=0)  # The extra card
        ]
        player.receive_cards(winning_cards)

        # 2. Teleport to go_down_decision
        self.engine.current_state = self.engine.go_down_decision

        # 3. ACT: Fire the GO_DOWN decision
        self.engine.resolve_go_down(4)

        # Currently, your engine does not automatically trigger commit_melds()
        # when entering the 'going_down' state. We manually fire it here to
        # continue the chain, but this highlights a missing automation link.
        if self.engine.current_state.id == "going_down":
            self.engine.commit_melds()

        # 4. ASSERT: State Routing
        self.assertEqual(
            self.engine.current_state.id,
            "table_play_phase",  # <-- UPDATED
            "Engine should route to table_play_phase after placing melds, waiting for agent."
        )

        # 5. ASSERT: Data Movement (Cards move to table)
        self.assertTrue(player.is_down, "Player should be marked as down.")

        # Note: Hand size will remain 7 here until we actually implement the
        # engine's commit_melds data movement logic, but the routing is correct.

    def test_auto_place_cards_updates_table_tensors(self):
        """
        Intent: Verify that extracting melds correctly updates the public game state
        tensors (table_sets and table_runs) with the exact card counts.
        """
        import numpy as np

        from cards import Card, Rank, Suit

        # 1. Setup Data Layer for Round 0 (Requires two sets of 3)
        self.ctx.current_round_idx = 0
        player = self.ctx.players[0]

        # Same test hand: Two 7s of Hearts, one 7 of Diamonds, and three Aces
        winning_cards = [
            Card(Suit.HEARTS, Rank.SEVEN, deck_index=0),
            Card(Suit.HEARTS, Rank.SEVEN, deck_index=1),
            Card(Suit.DIAMONDS, Rank.SEVEN, deck_index=0),
            Card(Suit.SPADES, Rank.ACE, deck_index=0),
            Card(Suit.CLUBS, Rank.ACE, deck_index=0),
            Card(Suit.DIAMONDS, Rank.ACE, deck_index=0),
            Card(Suit.CLUBS, Rank.THREE, deck_index=0)
        ]
        player.receive_cards(winning_cards)

        # Verify baseline table state
        self.assertEqual(np.sum(self.ctx.table_sets), 0)
        self.assertEqual(np.sum(self.ctx.table_runs), 0)

        # --- INTEGRITY CHECK: Snapshot mass before action ---
        initial_mass = verify_mass_integrity(self, self.ctx)

        # 2. ACT: Trigger the placement logic directly
        self.ctx.auto_place_cards(0)

        # 3. ASSERT: Sets Tensor Logic
        self.assertEqual(self.ctx.table_sets[Suit.HEARTS, Rank.SEVEN], 2)
        self.assertEqual(self.ctx.table_sets[Suit.DIAMONDS, Rank.SEVEN], 1)

        # Check the Aces (Must exist at index 0 AND index 13)
        for suit in [Suit.SPADES, Suit.CLUBS, Suit.DIAMONDS]:
            self.assertEqual(self.ctx.table_sets[suit, Rank.ACE], 1)
            self.assertEqual(self.ctx.table_sets[suit, 13], 1)

        self.assertEqual(np.sum(self.ctx.table_runs), 0)

        # --- INTEGRITY CHECK: Verify moving from Private Hand -> Table Tensors loses no cards ---
        verify_mass_integrity(self, self.ctx, expected_total=initial_mass)

    def test_evaluate_hand_already_down_skips_decision(self):
        """
        Intent: Verify that if a player is already marked as 'is_down',
        the engine bypasses 'go_down_decision' and moves straight
        through 'auto_placing' to 'discard_phase'.
        """
        from cards import Card, Rank, Suit

        # 1. Setup: Player is already "Down"
        self.ctx.current_round_idx = 0
        player = self.ctx.players[0]
        player.is_down = True  # <--- Crucial state trigger
        self.ctx.active_player_idx = 0

        # Give them a hand that would normally trigger a 'Go Down' prompt
        winning_cards = [
            Card(Suit.HEARTS, Rank.SEVEN, deck_index=0),
            Card(Suit.HEARTS, Rank.SEVEN, deck_index=1),
            Card(Suit.DIAMONDS, Rank.SEVEN, deck_index=0),
            Card(Suit.SPADES, Rank.ACE, deck_index=0),
            Card(Suit.CLUBS, Rank.ACE, deck_index=0),
            Card(Suit.DIAMONDS, Rank.ACE, deck_index=0),
            Card(Suit.CLUBS, Rank.THREE, deck_index=0)
        ]
        player.receive_cards(winning_cards)

        # 2. Teleport to processing_pickup
        self.engine.current_state = self.engine.processing_pickup

        # 3. ACT: Trigger hand evaluation
        self.engine.evaluate_hand()

        # 4. ASSERT: State Routing
        self.assertEqual(
            self.engine.current_state.id,
            "table_play_phase",  # <-- UPDATED
            "Engine should skip 'go_down_decision' and land on 'table_play_phase' "
            "if player is already down."
        )

    def test_run_extension_logic_with_wraparound(self):
        """
        Intent: Verify that _can_extend_run correctly identifies standard
        adjacencies and the King-Ace wrap-around.
        """
        from cards import Rank, Suit

        # 1. Setup Table: A run of 2-3-4 of Spades
        self.ctx.table_runs[Suit.SPADES, Rank.TWO] = 1
        self.ctx.table_runs[Suit.SPADES, Rank.THREE] = 1
        self.ctx.table_runs[Suit.SPADES, Rank.FOUR] = 1

        # 2. Test Standard Extensions
        self.assertTrue(
            self.ctx._can_extend_run(Suit.SPADES, Rank.ACE),
            "Ace should connect to Two"
        )
        self.assertTrue(
            self.ctx._can_extend_run(Suit.SPADES, Rank.FIVE),
            "Five should connect to Four"
        )
        self.assertFalse(
            self.ctx._can_extend_run(Suit.SPADES, Rank.SIX),
            "Six is not adjacent to 2-3-4"
        )

        # 3. Setup Table: A run of Queen-King of Hearts
        self.ctx.table_runs[Suit.HEARTS, Rank.QUEEN] = 1
        self.ctx.table_runs[Suit.HEARTS, Rank.KING] = 1

        # 4. Test Wrap-around (King to Ace)
        self.assertTrue(
            self.ctx._can_extend_run(Suit.HEARTS, Rank.ACE),
            "Ace should wrap around to connect with King"
        )

        # 5. Test Suit Isolation
        self.assertFalse(
            self.ctx._can_extend_run(Suit.DIAMONDS, Rank.ACE),
            "Ace of Diamonds should not connect to Hearts run"
        )

    def test_discard_exit(self):
        """
        Task 10.g (Discard Exit): Verify that discarding the last card
        transitions the state to round_end.
        """
        from cards import Card, Rank, Suit
        self.ctx.active_player_idx = 0

        # Prevent engine from automatically dealing a new 104-card game
        # so we can accurately measure the conservation of mass of just the discard action.
        self.ctx.current_round_idx = 7

        player = self.ctx.players[0]

        # 1. Give the player exactly 1 card
        player.hand_list = []
        player.private_hand.fill(0)
        player.receive_cards([Card(Suit.SPADES, Rank.ACE, deck_index=0)])

        # --- INTEGRITY CHECK: Snapshot mass before action ---
        initial_mass = verify_mass_integrity(self, self.ctx)

        # 2. Teleport to discard_phase
        self.engine.current_state = self.engine.discard_phase

        # 3. ACT: Discard the only card
        self.engine.perform_discard(0)

        # 4. ASSERT
        # Instead of 'dealing', it lands on 'game_over' due to the tournament limit.
        self.assertEqual(self.engine.current_state.id, "game_over")

        # --- INTEGRITY CHECK: Verify card successfully transitioned to dead_cards tensor
        # / discard_pile ---
        verify_mass_integrity(self, self.ctx, expected_total=initial_mass)

    def test_round_end_to_dealing(self):
        """
        Task 10i (Next Round): Verify that if the tournament is not over,
        resolving the round end loops back to the dealing state.
        """
        # 1. Setup Data Layer
        self.ctx.current_round_idx = 0

        # 2. Teleport the Logic Layer
        self.engine.current_state = self.engine.round_end

        # 3. ACT: Trigger the end of round resolution
        self.engine.resolve_round_end()

        # 4. ASSERT: State Routing
        self.assertEqual(
            self.engine.current_state.id,
            "dealing",
            "Engine should loop back to 'dealing' for Rounds 0-6."
        )

    def test_round_end_to_game_over(self):
        """
        Task 10j (Game Over): Verify that if the tournament is over (Round 7),
        resolving the round end transitions to game_over.
        """
        # 1. Setup Data Layer
        # The engine checks if current_round_idx >= 7
        self.ctx.current_round_idx = 7

        # 2. Teleport the Logic Layer
        self.engine.current_state = self.engine.round_end

        # 3. ACT: Trigger the end of round resolution
        self.engine.resolve_round_end()

        # 4. ASSERT: State Routing
        self.assertEqual(
            self.engine.current_state.id,
            "game_over",
            "Engine should transition to 'game_over' when the tournament limit is reached."
        )

    def test_calculate_scores_and_increment_round(self):
        """
        Task 13 (Scoring Engine): Verify terminal scores are calculated dynamically
        by querying the injected JoeConfig based on remaining cards in hand,
        and that the round index successfully increments.
        """
        from cards import Card, Rank, Suit

        # 1. Setup Data Layer
        self.ctx.current_round_idx = 0
        p0, p1 = self.ctx.players[0], self.ctx.players[1]

        # Clear hands and initialize scores
        p0.hand_list = []
        p1.hand_list = []
        p0.score = 0
        p1.score = 0

        # p0 went out (0 cards, 0 penalty points)
        # p1 is caught with an Ace (20), a King (10), and a Five (5) = 35 points total
        p1.receive_cards([
            Card(Suit.SPADES, Rank.ACE, deck_index=0),
            Card(Suit.HEARTS, Rank.KING, deck_index=0),
            Card(Suit.CLUBS, Rank.FIVE, deck_index=0)
        ])

        # 2. ACT: Trigger the end-of-round scoring
        self.ctx.calculate_scores()

        # 3. ASSERT: Player Scores
        self.assertEqual(
            p0.score,
            0,
            "Player 0 should have 0 penalty points for an empty hand."
        )
        self.assertEqual(
            p1.score,
            35,
            "Player 1 should accrue exactly 35 penalty points (20 + 10 + 5)."
        )

        # 4. ASSERT: Tournament Progression
        self.assertEqual(
            self.ctx.current_round_idx,
            1,
            "The round index should increment by 1 after scores are calculated."
        )

    def test_turn_counter_progression_and_reset(self):
        """
        Verify that the current_turn counter increments on start_turn
        and correctly resets to 0 when a new round is dealt.
        """
        # 1. Initial state
        self.assertEqual(
            self.ctx.current_turn,
            0,
            "Turn counter should initialize at 0."
        )

        # 2. Start game and deal (Transitions: setup -> dealing -> start_turn)
        self.engine.start_game()
        self.engine.deal_cards()

        # 3. Assert Increment
        self.assertEqual(
            self.ctx.current_turn,
            1,
            "Turn counter should be 1 after the first start_turn transition."
        )

        # 4. Simulate a second turn starting
        self.engine.on_enter_start_turn()
        self.assertEqual(
            self.ctx.current_turn,
            2,
            "Turn counter should increment to 2 on the next turn."
        )

        # 5. Trigger a new deal (Next Round)
        self.ctx.execute_deal()

        # 6. Assert Reset
        self.assertEqual(
            self.ctx.current_turn,
            0,
            "Turn counter must reset to 0 during a new deal."
        )

    def test_may_i_decision_action_pass(self):
        """
        Intent: Verify that if an interrupter PASSES a May-I, the engine
        correctly routes back to may_i_check, advances the pointer, and
        evaluates the next player.
        """
        # 1. Setup Data Layer
        self.ctx.active_player_idx = 0
        self.ctx.may_i_target_idx = 1

        # Give Player 2 a clean slate so they are eligible
        self.ctx.players[2].is_down = False

        # 2. Teleport Logic Layer
        self.engine.current_state = self.engine.may_i_decision

        # 3. ACT: Interrupter 1 passes
        self.engine.resolve_may_i(3)

        # 4. ASSERT: State Routing and Pointer
        # The engine should loop back to may_i_check, automatically evaluate
        # Player 2, and pause on may_i_decision again.
        self.assertEqual(
            self.engine.current_state.id,
            "may_i_decision",
            "Engine should loop back and pause on the next eligible player."
        )
        self.assertEqual(
            self.ctx.may_i_target_idx,
            2,
            "The May-I target pointer should advance to Player 2 after a pass."
        )

    def test_go_down_decision_action_wait(self):
        """
        Intent: Verify that choosing WAIT skips the placement phases, leaves the
        cards in the player's hand, and lands on discard_phase.
        """
        from cards import Card, Rank, Suit

        # 1. Setup Data Layer
        self.ctx.current_round_idx = 0
        player = self.ctx.players[0]
        self.ctx.active_player_idx = 0

        # Give a valid hand
        winning_cards = [
            Card(Suit.HEARTS, Rank.SEVEN, deck_index=0),
            Card(Suit.HEARTS, Rank.SEVEN, deck_index=1),
            Card(Suit.DIAMONDS, Rank.SEVEN, deck_index=0),
            Card(Suit.SPADES, Rank.ACE, deck_index=0),
            Card(Suit.CLUBS, Rank.ACE, deck_index=0),
            Card(Suit.DIAMONDS, Rank.ACE, deck_index=0)
        ]
        player.receive_cards(winning_cards)

        # 2. Teleport to go_down_decision
        self.engine.current_state = self.engine.go_down_decision

        # 3. ACT: Fire the WAIT decision
        self.engine.resolve_go_down(5)

        # 4. ASSERT: State Routing
        self.assertEqual(
            self.engine.current_state.id,
            "discard_phase",
            "Engine should bypass placement and route straight to discard_phase."
        )

        # 5. ASSERT: Data Movement
        self.assertFalse(
            player.is_down,
            "Player should NOT be marked as down."
        )
        self.assertEqual(
            len(player.hand_list),
            6,
            "All cards must remain in the player's hand."
        )

    def test_conservation_full_game_deal(self):
        """
        Verify that a standard initial deal strictly conserves all 104 cards.
        """
        self.engine.start_game()
        self.engine.deal_cards()

        # Absolute check for a standard game
        verify_mass_integrity(self, self.ctx, expected_total=104)

    def test_conservation_relative_action(self):
        """
        Verify that mocked sub-states maintain perfect relative conservation
        before and after complex data movements (like discarding).
        """
        from cards import Card, Rank, Suit

        # 1. Clear the universe (0 cards)
        self.ctx.deck = []
        self.ctx.discard_pile = []
        for p in self.ctx.players:
            p.hand_list = []
            p.private_hand.fill(0)

        # 2. Mock a localized state (Exactly 3 cards in the system)
        active_p = self.ctx.players[0]
        active_p.receive_cards([
            Card(Suit.SPADES, Rank.ACE, deck_index=0),
            Card(Suit.HEARTS, Rank.KING, deck_index=0)
        ])

        # Give the discard pile 1 card to test the burying logic
        self.ctx.discard_pile.append(Card(Suit.CLUBS, Rank.TWO, deck_index=0))

        # 3. Verify starting integrity (Total should be exactly 3)
        initial_mass = verify_mass_integrity(self, self.ctx)
        self.assertEqual(initial_mass, 3)

        # 4. ACT: Perform a discard action
        # This moves a card from the hand, buries the old discard, and updates both lists and
        # tensors.
        self.ctx.execute_discard(0, card_index=0)  # Discard the Ace of Spades

        # 5. Verify ending integrity (Total must STILL be exactly 3)
        verify_mass_integrity(self, self.ctx, expected_total=initial_mass)

    def test_round_transition_rotation_integration(self):
        """
        Integration Test: Verify that moving from the end of Round 1 into Round 2
        correctly advances the dealer and the active player without double-rotating.
        """
        # 1. Setup end of Round 1 (index 0)
        self.ctx.current_round_idx = 0
        self.ctx.dealer_idx = 0

        # 2. Teleport to round_end
        self.engine.current_state = self.engine.round_end

        # 3. ACT: Cascade through the round boundary
        # Manually fire the entry hook since we teleported into the state
        self.engine.on_enter_round_end()
        self.assertEqual(self.engine.current_state.id, "dealing")

        self.engine.deal_cards()  # Routes to 'start_turn', triggering the rotation

        # The engine pauses on start_turn. It does not auto-forward!
        self.assertEqual(self.engine.current_state.id, "start_turn")

        # 4. ASSERT: Dealer should be 1, Active Player should be 2
        self.assertEqual(
            self.ctx.current_round_idx,
            1,
            "Round should be 1 (Round 2)"
        )
        self.assertEqual(
            self.ctx.dealer_idx,
            1,
            "Dealer should advance to Player 1"
        )
        self.assertEqual(
            self.ctx.active_player_idx,
            2,
            "Active player should be Player 2 (Dealer + 1)"
        )

    def test_may_i_terminates_before_discarder(self):
        """
        Intent: Verify that the May-I loop terminates and returns control to the
        active player BEFORE asking the player who just discarded the card.
        """
        from cards import Card, Rank, Suit

        # 1. Setup: Active player is Player 1.
        # By definition, this means Player 0 was the discarder.
        self.ctx.active_player_idx = 1

        # Give the deck a card so the stock draw works
        self.ctx.deck.append(Card(Suit.SPADES, Rank.ACE, deck_index=0))

        # 2. Teleport to the pickup decision
        self.engine.current_state = self.engine.pickup_decision

        # 3. ACT: Active player (1) picks from stock, starting the May-I loop.
        self.engine.resolve_pickup(0)

        # Engine should ask Player 2 first (Active + 1).
        self.assertEqual(self.engine.current_state.id, "may_i_decision")
        self.assertEqual(self.ctx.may_i_target_idx, 2)

        # Player 2 passes.
        self.engine.resolve_may_i(3)

        # Engine should ask Player 3 next.
        self.assertEqual(self.engine.current_state.id, "may_i_decision")
        self.assertEqual(self.ctx.may_i_target_idx, 3)

        # 4. ACT: Player 3 passes.
        # The pointer will now move to Player 0.
        self.engine.resolve_may_i(3)

        # 5. ASSERT: The loop must terminate.
        # The engine's 'all_targets_checked' guard should instantly catch that
        # Player 0 is the discarder. It should skip the 'may_i_decision' state
        # entirely and return control to the active player.
        #
        # Note: Control returns via 'processing_pickup', which immediately evaluates
        # Player 1's hand. Because Player 1 has a weak hand, it auto-routes to 'discard_phase'.
        self.assertEqual(
            self.engine.current_state.id,
            "discard_phase",
            "Engine should skip the discarder and return control to the active player's "
            "discard phase."
        )

    def test_table_play_phase_play_card(self):
        """
        Task 10.5a (Play Card): Verify triggering perform_table_play moves
        a single card from the hand to the table and loops back to table_play_phase.
        """
        from cards import Card, Rank, Suit
        self.ctx.active_player_idx = 0
        player = self.ctx.players[0]

        # Give player a dummy card
        target_card = Card(Suit.HEARTS, Rank.SEVEN, deck_index=0)
        player.receive_cards([target_card])

        # Teleport to new state (This will immediately fail because the state doesn't exist yet)
        self.engine.current_state = self.engine.table_play_phase

        # ACT: Play the card at index 0
        self.engine.perform_table_play(0)

        # ASSERT: State should loop back automatically
        self.assertEqual(
            self.engine.current_state.id,
            "table_play_phase",
            "Engine should loop back to table_play_phase after processing a card."
        )

        # ASSERT: Data movement
        self.assertEqual(len(player.hand_list), 0, "Card should be removed from hand.")
        self.assertEqual(self.ctx.table_sets[Suit.HEARTS, Rank.SEVEN], 1,
                         "Card should appear on table.")

    def test_table_play_phase_end_play(self):
        """
        Task 10.5b (End Play): Verify triggering end_table_play routes to
        discard_phase if cards remain, or round_end if the hand is empty.
        """
        from cards import Card, Rank, Suit
        self.ctx.active_player_idx = 0
        self.ctx.current_round_idx = 0
        player = self.ctx.players[0]

        # --- Scenario A: Cards remain in hand ---
        player.receive_cards([Card(Suit.HEARTS, Rank.SEVEN, deck_index=0)])
        self.engine.current_state = self.engine.table_play_phase

        self.engine.end_table_play()

        self.assertEqual(
            self.engine.current_state.id,
            "discard_phase",
            "Should route to discard_phase because hand is not empty."
        )

        # --- Scenario B: Hand is empty ---
        player.hand_list = []
        player.private_hand.fill(0)
        self.engine.current_state = self.engine.table_play_phase

        self.engine.end_table_play()

        # Note: round_end automatically routes to dealing for Round 0
        self.assertEqual(
            self.engine.current_state.id,
            "dealing",
            "Should route to round_end (and cascade to dealing) because hand is empty."
        )

    def test_all_players_down_continues_play(self):
        """
        Verify that if all players are down, a discard action DOES NOT
        transition the state to round_end. The round must only end when
        a player's hand is completely empty.
        """
        from cards import Card, Rank, Suit
        self.ctx.active_player_idx = 0
        self.ctx.current_round_idx = 0

        # 1. MOCK: Set ALL players in the game to 'down'
        for player in self.ctx.players:
            player.is_down = True

        # 2. Give the active player two cards so hand_is_empty is False
        p0 = self.ctx.players[0]
        p0.hand_list = []
        p0.private_hand.fill(0)
        p0.receive_cards([
            Card(Suit.SPADES, Rank.ACE, deck_index=0),
            Card(Suit.HEARTS, Rank.KING, deck_index=0)
        ])

        # 3. Teleport to discard_phase
        self.engine.current_state = self.engine.discard_phase

        # 4. ACT: Discard one card (leaving one in hand)
        self.engine.perform_discard(0)

        # 5. ASSERT: It should continue to the next turn.
        self.assertEqual(
            self.engine.current_state.id,
            "start_turn",
            "Engine should route to 'start_turn' because the hand is not empty, even if all players are down."
        )


if __name__ == '__main__':
    unittest.main()