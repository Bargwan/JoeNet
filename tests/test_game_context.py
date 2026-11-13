import unittest

import numpy as np

from cards import Card, Rank, Suit
from config import JoeConfig
from game_context import GameContext
from player import Player
from tests.test_utils import verify_mass_integrity


class TestGameContextIntegrity(unittest.TestCase):
    def setUp(self):
        self.ctx = GameContext()

    def test_execute_discard_integrity(self):
        """Verify mass is conserved when burying cards in dead_cards tensor."""
        self.ctx.execute_deal()
        initial_mass = verify_mass_integrity(self, self.ctx)

        # Discard the first card in Player 0's hand
        self.ctx.execute_discard(player_idx=0, card_index=0)

        verify_mass_integrity(self, self.ctx, expected_total=initial_mass)

    def test_execute_pickup_stock_integrity(self):
        """Verify mass is conserved when drawing from the deck."""
        self.ctx.execute_deal()
        initial_mass = verify_mass_integrity(self, self.ctx)

        self.ctx.execute_pickup_stock()

        verify_mass_integrity(self, self.ctx, expected_total=initial_mass)

    def test_auto_place_cards_integrity(self):
        """
        Verify mass is conserved during complex extension logic
        (Private Hand -> Table Tensors).
        """
        # 1. Setup Table with a "hook" (Set of Kings)
        self.ctx.table_sets[Suit.SPADES, Rank.KING] = 1
        self.ctx.table_sets[Suit.HEARTS, Rank.KING] = 1
        self.ctx.table_sets[Suit.CLUBS, Rank.KING] = 1

        # 2. Give player the 4th King to extend the set
        p0 = self.ctx.players[0]
        p0.is_down = True  # only down players place Key Cards
        p0.receive_cards([Card(Suit.DIAMONDS, Rank.KING)])

        initial_mass = verify_mass_integrity(self, self.ctx)

        # 3. ACT: Extension phase
        # This moves the Diamond King from Player 0's hand to the Table Sets tensor
        self.ctx.auto_place_cards(player_idx=0)

        # 4. ASSERT
        verify_mass_integrity(self, self.ctx, expected_total=initial_mass)
        self.assertEqual(len(p0.hand_list), 0, "Card should have moved to table.")
        self.assertEqual(self.ctx.table_sets[Suit.DIAMONDS, Rank.KING], 1)

    def test_replenish_deck_on_empty_stock(self):
        """
        Verify that drawing from an empty stock correctly recycles
        the discard pile, preserves the top discard, and wipes the dead_cards tensor.
        """
        # 1. SETUP: Force an empty stock
        self.ctx.deck.clear()

        # Create a mock discard pile with exactly 4 cards
        cards = [
            Card(Suit.SPADES, Rank.TWO),
            Card(Suit.HEARTS, Rank.THREE),
            Card(Suit.CLUBS, Rank.FOUR),
            Card(Suit.DIAMONDS, Rank.FIVE)  # This is the top card!
        ]
        self.ctx.discard_pile.extend(cards)

        # Simulate that those buried cards were tracked in the dead_cards tensor
        self.ctx.dead_cards[0, 1] = 1
        self.ctx.dead_cards[1, 2] = 1

        # 2. ACT: Active player (0) attempts to draw from the empty stock
        self.ctx.execute_pickup_stock()

        # 3. ASSERT: Data Movement
        # The player successfully received 1 card
        self.assertEqual(
            len(self.ctx.active_player.hand_list),
            1,
            "Player did not receive a card"
        )

        # The discard pile retains ONLY the top card (the 5 of Diamonds)
        self.assertEqual(len(self.ctx.discard_pile), 1,
                         "Discard pile should have exactly 1 card left")
        self.assertEqual(self.ctx.discard_pile[0].rank, Rank.FIVE,
                         "Top discard card was not preserved!")

        # The deck now has 2 cards remaining (4 original - 1 kept in discard - 1 drawn by player)
        self.assertEqual(len(self.ctx.deck), 2, "Deck was not replenished correctly")

        # 4. ASSERT: Tensor Integrity
        # The dead cards tensor MUST be completely zeroed out since those cards are back in play
        self.assertEqual(
            np.sum(self.ctx.dead_cards), 0, "Dead cards tensor was not wiped clean!"
        )


class TestGameContext(unittest.TestCase):

    def test_initialization(self):
        """
        Verify the context creates the correct number of players and a fresh deck.
        """
        ctx = GameContext()

        # 1. Verify Players
        self.assertEqual(
            len(ctx.players),
            4,
            "Game should have 4 players by default")
        self.assertIsInstance(
            ctx.players[0],
            Player,
            "Context should hold Player objects"
        )

        # 2. Verify Deck
        # Deck should be initialized (104 cards)
        self.assertEqual(len(ctx.deck), 104, "Deck should start with 104 cards")

        # 3. Verify Discard Pile
        self.assertEqual(len(ctx.discard_pile), 0, "Discard pile should start empty")

    def test_execute_deal(self):
        """
        Verify cards move from Deck -> Players + Discard Pile
        """
        ctx = GameContext()

        # ACT: Deal the cards
        ctx.execute_deal()

        # 1. Verify Hand Size (11 cards per player)
        for i, player in enumerate(ctx.players):
            self.assertEqual(len(player.hand_list),
                             11,
                             f"Player {i} should have 11 cards")

        # 2. Verify Discard Pile (1 card to start the pile)
        self.assertEqual(
            len(ctx.discard_pile),
            1,
            "One card should be flipped to discard"
        )

        # 3. Verify Deck Depletion
        # Initial (104) - Players (4*11=44) - Discard (1) = 59
        expected_remaining = 104 - 44 - 1
        self.assertEqual(len(ctx.deck), expected_remaining, "Deck count incorrect after deal")

    def setUp(self):
        self.config = JoeConfig()
        self.context = GameContext(num_players=4, config=self.config)
        self.player_idx = 0
        self.player = self.context.players[self.player_idx]

    def test_objective_3_3_valid_sets(self):
        """Test [3,3] objective using the player's 4x14 spatial tensor."""
        # Set 1: Three 7s (Two Hearts, One Diamond)
        set1_cards = [
            Card(Suit.HEARTS, Rank.SEVEN),
            Card(Suit.HEARTS, Rank.SEVEN, deck_index=1),  # Deck 1 for duplicate
            Card(Suit.DIAMONDS, Rank.SEVEN)
        ]

        # Set 2: Three Aces (Spades, Clubs, Diamonds) - Tests the Ace index overlap!
        set2_cards = [
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.CLUBS, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.ACE)
        ]

        # Deadwood
        deadwood = [
            Card(Suit.HEARTS, Rank.TWO),
            Card(Suit.SPADES, Rank.FOUR),
            Card(Suit.CLUBS, Rank.NINE),
            Card(Suit.DIAMONDS, Rank.KING),
            Card(Suit.HEARTS, Rank.FIVE)
        ]

        # Act: Inject cards through the Player object to populate the tensor
        self.player.receive_cards(set1_cards + set2_cards + deadwood)
        self.context.current_round_idx = 0

        is_valid = self.context.check_hand_objective(self.player_idx)

        # Assert
        self.assertTrue(is_valid, "Tensor should resolve two valid sets.")

    def test_objective_4_4_wrap_around_failure(self):
        """Test [4,4] objective. Proves that K-A-2-3 wrap-around runs are illegal."""
        run1 = [
            Card(Suit.SPADES, Rank.JACK), Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.SPADES, Rank.KING), Card(Suit.SPADES, Rank.ACE)
        ]

        # Illegal wrap-around run
        run2_illegal = [
            Card(Suit.CLUBS, Rank.KING), Card(Suit.CLUBS, Rank.ACE),
            Card(Suit.CLUBS, Rank.TWO), Card(Suit.CLUBS, Rank.THREE)
        ]

        self.player.receive_cards(run1 + run2_illegal + [Card(Suit.HEARTS, Rank.TWO)] * 3)

        # Round 3 is index 2 -> Objective: [4,4]
        self.context.current_round_idx = 2

        is_valid = self.context.check_hand_objective(self.player_idx)
        self.assertFalse(is_valid, "A wrapping run (K-A-2) should fail the [4,4] objective.")

    def test_objective_3_3_invalid_short(self):
        """Test that [3,3] fails if only one set is present."""
        set1_cards = [
            Card(Suit.HEARTS, Rank.SEVEN),
            Card(Suit.CLUBS, Rank.SEVEN),
            Card(Suit.DIAMONDS, Rank.SEVEN)
        ]

        deadwood = [
            Card(Suit.HEARTS, Rank.TWO), Card(Suit.SPADES, Rank.FOUR),
            Card(Suit.CLUBS, Rank.NINE), Card(Suit.DIAMONDS, Rank.KING),
            Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.TWO),
            Card(Suit.CLUBS, Rank.THREE), Card(Suit.DIAMONDS, Rank.FOUR)
        ]

        self.player.receive_cards(set1_cards + deadwood)
        self.context.current_round_idx = 0

        is_valid = self.context.check_hand_objective(self.player_idx)
        self.assertFalse(is_valid, "Tensor with one set should fail [3,3].")

    def test_objective_3_4_greedy_trap(self):
        """
        Test that a 5-card run correctly leaves the needed card behind for a set.
        Hand: 4H, 5H, 6H, 7H, 8H (5-card run) + 4S, 4C (waiting for the 4H).
        The DFS must extract 5-6-7-8 for the run, leaving the 4H for the set.
        """
        # 5-card Run in Hearts
        run_of_five = [
            Card(Suit.HEARTS, Rank.FOUR), Card(Suit.HEARTS, Rank.FIVE),
            Card(Suit.HEARTS, Rank.SIX), Card(Suit.HEARTS, Rank.SEVEN),
            Card(Suit.HEARTS, Rank.EIGHT)
        ]

        # Two 4s waiting for the 4 of Hearts
        set_of_fours = [
            Card(Suit.SPADES, Rank.FOUR), Card(Suit.CLUBS, Rank.FOUR)
        ]

        # 4 random deadwood cards
        deadwood = [Card(Suit.DIAMONDS, Rank.KING)] * 4

        self.player.receive_cards(run_of_five + set_of_fours + deadwood)
        self.context.current_round_idx = 1  # [3,4]

        is_valid = self.context.check_hand_objective(self.player_idx)
        self.assertTrue(
            is_valid,
            "DFS should avoid the greedy trap and leave the 4H for the set."
        )

    def test_objective_3_4_valid_intersection(self):
        """Test [3,4] objective. Proves the DFS handles overlapping cards correctly."""
        # The Run: 4-5-6-7 of Hearts
        run_cards = [
            Card(Suit.HEARTS, Rank.FOUR), Card(Suit.HEARTS, Rank.FIVE),
            Card(Suit.HEARTS, Rank.SIX), Card(Suit.HEARTS, Rank.SEVEN)
        ]

        # The Set: Three 7s (Spades, Clubs, Diamonds)
        # Note: The player DOES NOT have a second 7 of Hearts.
        # The DFS must correctly separate the 7 of Hearts to the Run, and the others to the Set.
        set_cards = [
            Card(Suit.SPADES, Rank.SEVEN),
            Card(Suit.CLUBS, Rank.SEVEN),
            Card(Suit.DIAMONDS, Rank.SEVEN)
        ]

        self.player.receive_cards(run_cards + set_cards + [Card(Suit.HEARTS, Rank.TWO)] * 4)

        # Round 2 is index 1 -> Objective: [3,4]
        self.context.current_round_idx = 1

        is_valid = self.context.check_hand_objective(self.player_idx)
        self.assertTrue(is_valid, "[3,4] objective should pass with exact card isolation.")

    def test_objective_4_4_mega_run_split(self):
        """
        Test that an 8-card contiguous sequence is correctly split into two 4-card runs.
        Hand: 4 through Jack of Hearts.
        """
        # 8-card Mega Run in Hearts
        mega_run = [
            Card(Suit.HEARTS, Rank.FOUR), Card(Suit.HEARTS, Rank.FIVE),
            Card(Suit.HEARTS, Rank.SIX), Card(Suit.HEARTS, Rank.SEVEN),
            Card(Suit.HEARTS, Rank.EIGHT), Card(Suit.HEARTS, Rank.NINE),
            Card(Suit.HEARTS, Rank.TEN), Card(Suit.HEARTS, Rank.JACK)
        ]

        # 3 random deadwood cards
        deadwood = [Card(Suit.CLUBS, Rank.TWO)] * 3

        self.player.receive_cards(mega_run + deadwood)
        self.context.current_round_idx = 2  # [4,4]

        is_valid = self.context.check_hand_objective(self.player_idx)
        self.assertTrue(
            is_valid,
            "DFS should natively split an 8-card run into two 4-card runs."
        )

    def test_objective_3_4_4_three_melds(self):
        """
        Test that the recursive DFS scales correctly to 3-meld objectives.
        Round 5 Objective: [3,4,4] (1 Set, 2 Runs).
        """
        # Set of 9s
        set_cards = [
            Card(Suit.SPADES, Rank.NINE),
            Card(Suit.CLUBS, Rank.NINE),
            Card(Suit.DIAMONDS, Rank.NINE)
        ]

        # Run 1: Low Clubs
        run1 = [
            Card(Suit.CLUBS, Rank.TWO),
            Card(Suit.CLUBS, Rank.THREE),
            Card(Suit.CLUBS, Rank.FOUR),
            Card(Suit.CLUBS, Rank.FIVE)
        ]

        # Run 2: High Diamonds
        run2 = [
            Card(Suit.DIAMONDS, Rank.TEN),
            Card(Suit.DIAMONDS, Rank.JACK),
            Card(Suit.DIAMONDS, Rank.QUEEN),
            Card(Suit.DIAMONDS, Rank.KING)
        ]

        self.player.receive_cards(set_cards + run1 + run2)  # Exactly 11 cards
        self.context.current_round_idx = 5  # [3,4,4]

        is_valid = self.context.check_hand_objective(self.player_idx)
        self.assertTrue(is_valid, "DFS should successfully resolve a depth-3 meld objective.")

    def test_go_down_mass_conservation(self):
        """
        Verify that moving melds to the table preserves the total card count.
        """
        from cards import Card, Rank, Suit
        # 1. Setup minimal universe
        self.context.deck = []
        self.context.discard_pile = []
        self.player.hand_list = []
        self.player.private_hand.fill(0)

        # 2. Give player 6 cards (Two sets of 3)
        cards = [
            Card(Suit.SPADES, Rank.ACE, deck_index=0), Card(Suit.HEARTS, Rank.ACE, deck_index=0),
            Card(Suit.CLUBS, Rank.ACE, deck_index=0), Card(Suit.SPADES, Rank.TWO, deck_index=0),
            Card(Suit.HEARTS, Rank.TWO, deck_index=0), Card(Suit.CLUBS, Rank.TWO, deck_index=0)
        ]
        self.player.receive_cards(cards)

        # 3. Snapshot and ACT
        # Use your existing helper logic to verify mass is exactly 6
        initial_mass = np.sum(self.player.private_hand[:, 0:13])
        self.context.current_round_idx = 0  # Objective [3,3]
        self.context.go_down(self.player_idx)

        # 4. ASSERT
        final_player_mass = np.sum(self.player.private_hand[:, 0:13])
        table_mass = np.sum(self.context.table_sets[:, 0:13])

        self.assertEqual(
            initial_mass,
            final_player_mass + table_mass,
            "Mass lost or created during go_down move."
        )
        self.assertEqual(len(self.player.hand_list), 0, "List and Tensor desynced.")

    def test_sync_ace_underflow_protection(self):
        """
        Verify that _sync_ace safely prevents integer underflow (e.g., 0 - 1 = 255)
        and overflow when modifying uint8 numpy tensors.
        """
        ctx = GameContext()
        # Create a mock uint8 tensor identical to the engine's private_hand
        mock_tensor = np.zeros((4, 14), dtype=np.uint8)

        suit_idx = Suit.HEARTS.value
        ace_low_idx = Rank.ACE.value  # 0
        ace_high_idx = 13

        # --- 1. Test Underflow Protection (0 - 1) ---
        # If unprotected, a uint8 0 minus 1 becomes 255.
        ctx._sync_ace(mock_tensor, suit=suit_idx, rank=ace_low_idx, increment=False)

        self.assertEqual(mock_tensor[suit_idx, ace_low_idx], 0, "Ace Low underflowed to 255!")
        self.assertEqual(mock_tensor[suit_idx, ace_high_idx], 0, "Ace High underflowed to 255!")

        # --- 2. Test Normal Increment (0 + 1) ---
        ctx._sync_ace(mock_tensor, suit=suit_idx, rank=ace_low_idx, increment=True)

        self.assertEqual(mock_tensor[suit_idx, ace_low_idx], 1, "Ace Low failed to increment.")
        self.assertEqual(mock_tensor[suit_idx, ace_high_idx], 1, "Ace High failed to increment.")

        # --- 3. Test Overflow Cap (10 + 1) ---
        # Force the tensor to the logical maximum
        mock_tensor[suit_idx, ace_low_idx] = 10
        mock_tensor[suit_idx, ace_high_idx] = 10

        ctx._sync_ace(mock_tensor, suit=suit_idx, rank=ace_low_idx, increment=True)

        self.assertEqual(mock_tensor[suit_idx, ace_low_idx], 10, "Ace Low exceeded safety cap.")
        self.assertEqual(mock_tensor[suit_idx, ace_high_idx], 10, "Ace High exceeded safety cap.")



class TestHandPenalty(unittest.TestCase):

    def test_hand_penalty_scores_each_rank_band(self):
        """Aces score 20, 8 through King 10, and 2 through 7 score 5."""
        ctx = GameContext(num_players=3)
        ctx.players[0].hand_list.clear()
        ctx.players[0].receive_cards([
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.CLUBS, Rank.EIGHT),
            Card(Suit.DIAMONDS, Rank.SEVEN),
            Card(Suit.SPADES, Rank.TWO),
        ])
        self.assertEqual(ctx.hand_penalty(0), 20 + 10 + 10 + 5 + 5)


class TestOracleTruth(unittest.TestCase):

    def test_oracle_truth_4_player(self):
        """Verify the Oracle correctly maps 3 opponents relative to the active player."""
        ctx = GameContext(num_players=4)
        ctx.deck.clear()

        # Give specific cards to the opponents of Player 0
        ctx.players[1].receive_cards([Card(Suit.SPADES, Rank.ACE)])  # Op1 (Left)
        ctx.players[2].receive_cards([Card(Suit.HEARTS, Rank.KING)])  # Op2 (Across)
        ctx.players[3].receive_cards([Card(Suit.CLUBS, Rank.TWO)])  # Op3 (Right)

        # Act
        truth = ctx.get_oracle_truth(player_idx=0)

        # Assert
        self.assertEqual(truth.shape, (3, 4, 14), "Oracle truth must be strictly (3, 4, 14)")

        self.assertEqual(truth[0, Suit.SPADES.value, Rank.ACE.value], 1,
                         "Op1 missing Ace of Spades")
        self.assertEqual(truth[1, Suit.HEARTS.value, Rank.KING.value], 1,
                         "Op2 missing King of Hearts")
        self.assertEqual(truth[2, Suit.CLUBS.value, Rank.TWO.value], 1, "Op3 missing Two of Clubs")

    def test_oracle_truth_3_player_padding(self):
        """Verify the Oracle strictly zero-pads the Op3 channel in 3-player games."""
        ctx = GameContext(num_players=3)
        ctx.deck.clear()

        # Give specific cards to the opponents of Player 0
        ctx.players[1].receive_cards([Card(Suit.SPADES, Rank.ACE)])  # Op1 (Left)
        ctx.players[2].receive_cards([Card(Suit.HEARTS, Rank.KING)])  # Op2 (Right)

        # Act
        truth = ctx.get_oracle_truth(player_idx=0)

        # Assert
        self.assertEqual(truth.shape, (3, 4, 14),
                         "Oracle truth must remain (3, 4, 14) for architecture shape consistency")
        self.assertEqual(truth[0, Suit.SPADES.value, Rank.ACE.value], 1)
        self.assertEqual(truth[1, Suit.HEARTS.value, Rank.KING.value], 1)

        # The critical zero-padding check
        self.assertEqual(np.sum(truth[2]), 0,
                         "Op3 channel (Index 2) MUST be completely empty in a 3-player game.")

if __name__ == '__main__':
    unittest.main()