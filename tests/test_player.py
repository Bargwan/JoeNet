import unittest

from cards import Card, Rank, Suit
from player import Player


class TestPlayerLogic(unittest.TestCase):

    def test_nn_structure_and_deal(self):
        p = Player(player_id=0)

        # 1. Create Test Cards using strict constructor
        # Ace of Spades (Deck 0) -> ID 0
        ace_spades = Card(Suit.SPADES, Rank.ACE, deck_index=0)

        # Five of Diamonds (Deck 0) -> ID 43
        # Calc: 0 + (3 * 13) + 4 = 43
        five_diamonds = Card(Suit.DIAMONDS, Rank.FIVE, deck_index=0)

        cards_to_deal = [ace_spades, five_diamonds]

        # 2. Action: Deal
        p.receive_cards(cards_to_deal)

        # 3. Verify NN State (Logic remains the same)
        # Ace maps to 0 and 13
        self.assertEqual(p.private_hand[Suit.SPADES, 0], 1)
        self.assertEqual(p.private_hand[Suit.SPADES, 13], 1)

        # Rank 5 maps to Index 4
        self.assertEqual(p.private_hand[Suit.DIAMONDS, 4], 1)


if __name__ == '__main__':
    unittest.main()