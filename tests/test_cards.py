import unittest

from cards import Card, Rank, Suit, create_double_deck


class TestCardIdentity(unittest.TestCase):
    def test_deterministic_ids(self):
        # 1. Test Ace of Spades, Deck 0 (Should be ID 0)
        c1 = Card(Suit.SPADES, Rank.ACE, deck_index=0)
        self.assertEqual(c1.id, 0)

        # 2. Test King of Diamonds, Deck 0 (Should be ID 51)
        # 0 + (3 * 13) + 12 = 39 + 12 = 51
        c2 = Card(Suit.DIAMONDS, Rank.KING, deck_index=0)
        self.assertEqual(c2.id, 51)

        # 3. Test Ace of Spades, Deck 1 (Should be ID 52)
        c3 = Card(Suit.SPADES, Rank.ACE, deck_index=1)
        self.assertEqual(c3.id, 52)

        # 4. Test King of Diamonds, Deck 1 (Should be ID 103)
        c4 = Card(Suit.DIAMONDS, Rank.KING, deck_index=1)
        self.assertEqual(c4.id, 103)

    def test_deck_generation(self):
        deck = create_double_deck()
        self.assertEqual(len(deck), 104)
        # Verify strict ordering
        self.assertEqual(deck[0].id, 0)
        self.assertEqual(deck[103].id, 103)
        # Verify uniqueness
        all_ids = {c.id for c in deck}
        self.assertEqual(len(all_ids), 104)


if __name__ == '__main__':
    unittest.main()