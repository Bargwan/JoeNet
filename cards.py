from enum import IntEnum


class Suit(IntEnum):
    SPADES = 0
    HEARTS = 1
    CLUBS = 2
    DIAMONDS = 3


class Rank(IntEnum):
    ACE = 0
    TWO = 1
    THREE = 2
    FOUR = 3
    FIVE = 4
    SIX = 5
    SEVEN = 6
    EIGHT = 7
    NINE = 8
    TEN = 9
    JACK = 10
    QUEEN = 11
    KING = 12


class Card:
    __slots__ = ['suit', 'rank', 'deck_index', 'id']
    _SUIT_SYMBOLS = "♠♡♣♢"
    _RANK_SYMBOLS = "A23456789TJQK"

    def __init__(self, suit, rank, deck_index=0):
        """
        Strictly enforces ID generation based on Suit/Rank/Deck.
        deck_index: 0 for the first deck, 1 for the second deck.
        """
        if deck_index not in (0, 1):
            raise ValueError("deck_index must be 0 or 1")

        self.suit = suit
        self.rank = rank
        self.deck_index = deck_index

        # Enforce Canonical ID: 0-51 (Deck 0) and 52-103 (Deck 1)
        # Formula: (Deck * 52) + (Suit * 13) + Rank
        self.id = (deck_index * 52) + (int(suit) * 13) + int(rank)

    def __str__(self):
        return f"{self._RANK_SYMBOLS[self.rank]}{self._SUIT_SYMBOLS[self.suit]}"

    def __repr__(self):
        return f"{self._RANK_SYMBOLS[self.rank]}{self._SUIT_SYMBOLS[self.suit]}(ID:{self.id})"

    def __eq__(self, other):
        return self.id == other.id


def create_double_deck():
    cards = []
    for deck in (0, 1):
        for s in Suit:
            for r in Rank:
                cards.append(Card(s, r, deck_index=deck))
    return cards