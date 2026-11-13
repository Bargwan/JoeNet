"""
The actions a player can take in Joe, as small read-only records.

GameContext.legal_actions() lists the actions allowed at each decision; an agent
(or a UI) picks one of them. Cards are identified by suit and rank: the two
copies of a card in the double deck are interchangeable under the rules.
"""
from dataclasses import dataclass
from enum import IntEnum
from typing import Union

from cards import Card, Rank, Suit


class Destination(IntEnum):
    """Where a Key Card is placed. These three cover every choice that changes the game."""
    ON_SET = 0
    RUN_LOW_END = 1
    RUN_HIGH_END = 2


def _card_name(suit, rank):
    return str(Card(Suit(suit), Rank(rank)))


# --- Pickup decision ---

@dataclass(frozen=True)
class PickStock:
    """Decline the discard and draw from the stock (May-Is are offered first)."""

    def __str__(self):
        return "Pick up from the stock"


@dataclass(frozen=True)
class PickDiscard:
    """Take the top discard."""

    def __str__(self):
        return "Pick up the discard"


# --- May-I decision (made by the player being offered the May-I) ---

@dataclass(frozen=True)
class CallMayI:
    """Take the top discard plus the top stock card."""

    def __str__(self):
        return "Call May-I"


@dataclass(frozen=True)
class Pass:
    """Decline the May-I."""

    def __str__(self):
        return "Pass"


# --- Go-down decision ---

@dataclass(frozen=True)
class GoDown:
    """
    Go down with these melds (GameContext's meld format): ('run', suit, start)
    and ('set', rank). The order doesn't matter: melds are stored sorted, so two
    GoDowns with the same melds are equal.
    """
    melds: tuple

    def __post_init__(self):
        object.__setattr__(self, 'melds', tuple(sorted(tuple(m) for m in self.melds)))

    def __str__(self):
        parts = []
        for meld in self.melds:
            if meld[0] == 'run':
                _, suit, start = meld
                cards = [_card_name(suit, pos % 13) for pos in (start, start + 3)]
                parts.append(f"run {cards[0]}-{cards[1]}")
            else:
                parts.append(f"set of {Rank(meld[1]).name.title()}s")
        return "Go down with " + ", ".join(parts)


@dataclass(frozen=True)
class Wait:
    """Hold the objective for now and go on to the discard."""

    def __str__(self):
        return "Wait"


# --- Table play (after going down) ---

@dataclass(frozen=True)
class Place:
    """Place a Key Card on a meld at the given destination."""
    suit: Suit
    rank: Rank
    destination: Destination

    def __str__(self):
        where = {Destination.ON_SET: "a set",
                 Destination.RUN_LOW_END: "the low end of a run",
                 Destination.RUN_HIGH_END: "the high end of a run"}[self.destination]
        return f"Place {_card_name(self.suit, self.rank)} on {where}"


@dataclass(frozen=True)
class EndTablePlay:
    """Stop placing cards (and go on to the discard, unless the hand is empty)."""

    def __str__(self):
        return "End table play"


# --- Discard decision ---

@dataclass(frozen=True)
class Discard:
    """Discard a card."""
    suit: Suit
    rank: Rank

    def __str__(self):
        return f"Discard {_card_name(self.suit, self.rank)}"


Action = Union[PickStock, PickDiscard, CallMayI, Pass, GoDown, Wait, Place, EndTablePlay,
               Discard]
