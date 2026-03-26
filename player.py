import numpy as np
from cards import Rank

class Player:
    def __init__(self, player_id=0):
        self.player_id = player_id

        # The state of a player's hand is simultaneously recorded in two separate properties:
        # - the `hand_list` which preserves the ID, so knowledge of each distinct card in a 2
        #   double deck
        # - the `private_hand` which is a NumPy array counting cards at the rank/suit coordinates
        #   and optimised for passing to a Neural Network as a tensor input.
        #       - Shape: (4, 14) -> 4 Suits x 14 Ranks
        #       - Index 0=Ace(Low), 1=Two, ..., 12=King, 13=Ace(High)

        self.hand_list = []
        self.private_hand = np.zeros((4, 14), dtype=np.int8)

        self.is_down = False
        self.score = 0
        self.may_is_used = 0

    # --- ACTION METHODS ---

    def receive_cards(self, cards):
        """
        cards: List[Card]
        """
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
        Finds card object, removes it, and updates NN tensor.
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
        """
        Creates a deep copy of the player's state.
        Used exclusively for high-speed RAM cloning in the fast_engine.
        """
        new_player = Player(self.player_id)

        # Lists and NumPy arrays must be explicitly copied to break references
        new_player.hand_list = list(self.hand_list)
        new_player.private_hand = self.private_hand.copy()  # Fast NumPy duplication

        new_player.is_down = self.is_down
        new_player.score = self.score
        new_player.may_is_used = self.may_is_used

        return new_player