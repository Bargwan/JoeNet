import unittest
import numpy as np
from game_context import GameContext
from cards import Card, Suit, Rank


class TestActionMasking(unittest.TestCase):
    def setUp(self):
        self.ctx = GameContext(num_players=4)
        self.ctx.deck.clear()
        self.player_idx = 0
        self.player = self.ctx.players[self.player_idx]

    def test_pickup_phase_mask(self):
        """Verify only Stock (0) and Discard (1) are legal during pickup."""
        self.ctx.discard_pile.append(Card(Suit.SPADES, Rank.ACE))

        mask = self.ctx.get_action_mask(self.player_idx, state_id='pickup_decision')

        self.assertEqual(mask.shape, (58,), "Mask must be exactly 58 booleans.")
        self.assertTrue(mask[0], "Pick Stock should be True")
        self.assertTrue(mask[1], "Pick Discard should be True")
        self.assertEqual(np.sum(mask), 2, "Only 2 actions should be legal.")

    def test_pickup_phase_empty_discard_mask(self):
        """Verify Pick Discard is illegal if the discard pile is empty."""
        # Discard pile is explicitly empty
        mask = self.ctx.get_action_mask(self.player_idx, state_id='pickup_decision')

        self.assertTrue(mask[0], "Pick Stock should be True")
        self.assertFalse(mask[1], "Pick Discard should be False (Empty pile)")
        self.assertEqual(np.sum(mask), 1, "Only 1 action should be legal.")

    def test_may_i_decision_mask(self):
        """Verify only Call (2) and Pass (3) are legal."""
        mask = self.ctx.get_action_mask(self.player_idx, state_id='may_i_decision')

        self.assertTrue(mask[2], "Call May-I should be True")
        self.assertTrue(mask[3], "Pass should be True")
        self.assertEqual(np.sum(mask), 2)

    def test_go_down_decision_mask(self):
        """Verify only Go Down (4) and Wait (5) are legal."""
        mask = self.ctx.get_action_mask(self.player_idx, state_id='go_down_decision')

        self.assertTrue(mask[4], "Go Down should be True")
        self.assertTrue(mask[5], "Wait should be True")
        self.assertEqual(np.sum(mask), 2)

    def test_discard_phase_mask(self):
        """
        Verify only the cards currently held in the physical hand are legal to discard.
        Cards are mapped to indices 6 through 57.
        """
        # Give the player exactly 2 distinct cards
        self.player.receive_cards([
            Card(Suit.HEARTS, Rank.TWO),  # Offset: 6 + (1*13) + 1 = 20
            Card(Suit.CLUBS, Rank.KING)  # Offset: 6 + (2*13) + 12 = 44
        ])

        mask = self.ctx.get_action_mask(self.player_idx, state_id='discard_phase')

        self.assertTrue(mask[20], "2 of Hearts should be legal to discard.")
        self.assertTrue(mask[44], "King of Clubs should be legal to discard.")
        self.assertEqual(np.sum(mask), 2, "Only the 2 cards in hand should be legal.")
        self.assertFalse(mask[0] or mask[1] or mask[2] or mask[3] or mask[4] or mask[5],
                         "Control actions must be False during discard.")

    def test_table_play_phase_mask(self):
        """
        Verify cards in hand are legal to play, AND End_Table_Play (5) is legal.
        """
        self.player.receive_cards([Card(Suit.DIAMONDS, Rank.FIVE)])  # Offset: 6 + (3*13) + 4 = 49

        mask = self.ctx.get_action_mask(self.player_idx, state_id='table_play_phase')

        self.assertTrue(mask[5], "End Table Play should be True")
        self.assertTrue(mask[49], "5 of Diamonds should be True")
        self.assertEqual(np.sum(mask), 2)


if __name__ == '__main__':
    unittest.main()