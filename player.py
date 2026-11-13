import numpy as np

from cards import Rank


class Player:
    """One player's hand, score and round status."""

    def __init__(self, player_id=0):
        self.player_id = player_id

        # The hand is recorded twice, and the two are kept in step:
        # - `hand_list`: the Card objects themselves (each copy of the double deck is distinct)
        # - `private_hand`: counts per card for fast meld searches
        #       - Shape: (4, 14) -> 4 suits x 14 columns
        #       - Column 0 = Ace (low), 1 = Two, ..., 12 = King, 13 = Ace (high)

        self.hand_list = []
        self.private_hand = np.zeros((4, 14), dtype=np.int8)

        self.is_down = False
        self.score = 0
        self.may_is_used = 0

    # --- ACTION METHODS ---

    def receive_cards(self, cards):
        """Adds cards (a list of Card) to the hand."""
        for card in cards:
            self._add_card_to_state(card)

    def _add_card_to_state(self, card):
        """Internal helper to update both list and tensor."""
        # 1. Logic Update
        self.hand_list.append(card)

        # 2. Tensor Update
        rank_idx = int(card.rank)  # 0-12
        suit_idx = int(card.suit)

        # Increment standard position (A-Low ... King)
        self.private_hand[suit_idx, rank_idx] += 1

        # Special Case: Aces also map to Index 13 (High)
        if card.rank == Rank.ACE:
            self.private_hand[suit_idx, 13] += 1

    def discard(self, suit, rank):
        """
        Removes one card of this suit and rank from the hand (any copy) and
        returns it. Used for every card leaving the hand: discards, going down
        and placing Key Cards.
        """
        card_to_discard = next((c for c in self.hand_list
                                if c.suit == suit and c.rank == rank), None)

        if card_to_discard is None:
            raise ValueError(
                f"Player {self.player_id} tried to discard {suit}-{rank} but doesn't have it!"
            )

        self.hand_list.remove(card_to_discard)

        # Update Tensor
        rank_idx = int(rank)
        self.private_hand[suit, rank_idx] -= 1

        # Special Case: Remove High Ace if Ace
        if rank == Rank.ACE:
            self.private_hand[suit, 13] -= 1

        return card_to_discard

    def clone(self):
        """A deep copy of the player's state (used by GameContext.clone)."""
        new_player = Player(self.player_id)

        # Lists and NumPy arrays must be explicitly copied to break references
        new_player.hand_list = list(self.hand_list)
        new_player.private_hand = self.private_hand.copy()  # Fast NumPy duplication

        new_player.is_down = self.is_down
        new_player.score = self.score
        new_player.may_is_used = self.may_is_used

        return new_player