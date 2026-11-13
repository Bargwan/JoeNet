import itertools
import random
from dataclasses import dataclass

import numpy as np

from cards import create_double_deck
from config import JoeConfig
from player import Player

ACE_LOW = 0
ACE_HIGH = 13


@dataclass(slots=True)
class TableRun:
    """
    One run on the table, stored by its end positions.
    Positions follow the 14-column tensor layout: 0 = Ace (low), 1 = Two, ...,
    12 = King, 13 = Ace (high). A run can only grow at its ends (rules.md Section 6).
    """
    suit: int
    low: int
    high: int


def _run_positions(rank: int) -> tuple:
    """The run positions a card of this rank can occupy. An Ace (0 or 13) can be either end."""
    if rank in (ACE_LOW, ACE_HIGH):
        return (ACE_LOW, ACE_HIGH)
    return (rank,)


class GameContext:
    """
    Manages data for the game.
    Acts as the 'Source of Truth' and facilitates data movement between
    the Deck, the Discard Pile, and the Players.
    """

    def __init__(self, num_players=4, config=None):
        self.num_players = num_players
        self.config = config if config is not None else JoeConfig()
        self.deck = create_double_deck()  # Returns 104 cards
        self.discard_pile = []

        # Initialize Players
        # (Player 0, Player 1, Player 2, Player 3)
        self.players = [Player(player_id=i) for i in range(num_players)]

        # Public Game State Tensors (Spatial layers 1 and 2)
        self.table_sets = np.zeros((4, 14), dtype=np.int8)
        self.table_runs = np.zeros((4, 14), dtype=np.int8)
        # Each run on the table as its own record; table_runs above is their running total.
        self.runs_on_table: list[TableRun] = []
        self.dead_cards = np.zeros((4, 14), dtype=np.int8) # Channel 12
        self.player_discard_counts = np.zeros((4, 4, 14), dtype=np.int8)
        self.player_pickup_counts = np.zeros((4, 4, 14), dtype=np.int8)

        # Game State Markers
        self.active_player_idx = 0
        self.dealer_idx = 0
        self.current_round_idx = 0
        self.total_actions = 0
        self.may_i_target_idx = None

    # --- Turn Management ---

    @property
    def active_player(self):
        """Returns the Player object for the current turn."""
        return self.players[self.active_player_idx]

    @property
    def current_circuit(self):
        """Returns the number of full board circuits completed (strategic turns)."""
        return self.total_actions // self.num_players

    def execute_deal(self):
        """
        Resets the board, shuffles deck, deals 11 cards to each player, and flips one to discard.
        """
        self.total_actions = 0

        # --- FULL BOARD RESET FOR NEW ROUND ---
        self.deck = create_double_deck()
        self.discard_pile.clear()

        self.table_sets.fill(0)
        self.table_runs.fill(0)
        self.runs_on_table.clear()
        self.dead_cards.fill(0)

        self.player_discard_counts.fill(0)
        self.player_pickup_counts.fill(0)

        for player in self.players:
            player.hand_list.clear()
            player.private_hand.fill(0)
            player.is_down = False
            player.may_is_used = 0
            # Note: player.score is intentionally preserved across rounds!

        # 1. Shuffle
        random.shuffle(self.deck)

        # 2. Deal 11 cards to each player
        for _ in range(11):
            for player in self.players:
                if not self.deck:
                    raise ValueError("Ran out of cards during deal!")
                card = self.deck.pop(0)
                player.receive_cards([card])

        # 3. Flip top card to discard
        if self.deck:
            initial_discard = self.deck.pop(0)
            self.discard_pile.append(initial_discard)

    def rotate_player(self):
        self.active_player_idx = (self.active_player_idx + 1) % self.num_players

    def rotate_dealer(self):
        self.dealer_idx = (self.dealer_idx + 1) % self.num_players
        # Active player usually starts to the left of dealer
        self.active_player_idx = self.dealer_idx

    def advance_action_counter(self):
        """Increments the global action counter for absolute engine safety."""
        self.total_actions += 1

    def calculate_scores(self):
        """
        Tallies penalty points for all cards remaining in players' hands
        based on the injected config, adds them to player scores,
        and advances the tournament round.
        """
        for player in self.players:
            round_penalty = 0
            for card in player.hand_list:
                rank_val = int(card.rank)

                if rank_val == 0:  # ACE
                    round_penalty += self.config.points_ace
                elif 7 <= rank_val <= 12:  # EIGHT (7) through KING (12)
                    round_penalty += self.config.points_eight_to_king
                elif 1 <= rank_val <= 6:  # TWO (1) through SEVEN (6)
                    round_penalty += self.config.points_two_to_seven

            # Add the accrued penalty points to the player's total score
            # (Assumes your Player class has a 'score' attribute initialized to 0)
            player.score += round_penalty

        # Advance the tournament boundary marker
        self.current_round_idx += 1

    def _replenish_deck_if_empty(self):
        """Recycles the dead cards into a new stock pile if the deck runs out."""
        if not self.deck and len(self.discard_pile) > 1:
            # Save the top card to keep the discard pile alive
            top_card = self.discard_pile.pop()

            # The rest of the dead cards become the new deck
            self.deck = self.discard_pile[:]
            self.discard_pile = [top_card]

            # Shuffle to prevent deterministic infinite loops
            import random
            random.shuffle(self.deck)

            # Wipe the dead_cards tensor (Channel 20) because they are resurrected!
            self.dead_cards.fill(0)

    def execute_pickup_discard(self):
        """Moves the top card of the discard pile to the active player's hand."""
        if not self.discard_pile:
            raise ValueError("Cannot pick up from an empty discard pile.")

        # Pop from the end of the list. Channel 3 is now effectively empty.
        # We do NOT resurrect from self.dead_cards.
        card = self.discard_pile.pop()
        self.active_player.receive_cards([card])

        self._record_action(
            self.player_pickup_counts,
            self.active_player_idx,
            int(card.suit),
            int(card.rank)
        )

    def execute_pickup_stock(self):
        """Moves the top card of the deck to the active player's hand."""
        self._replenish_deck_if_empty()
        if not self.deck:
            raise ValueError("Cannot pick up from an empty deck.")

        # Pop from the front of the list (top of the deck)
        card = self.deck.pop()
        self.active_player.receive_cards([card])

    def execute_may_i_call(self):
        """Transfers the discard and a penalty stock card to the interrupter."""
        if not self.discard_pile:
            raise ValueError("Cannot May-I an empty discard pile.")

        self._replenish_deck_if_empty()
        if not self.deck:
            raise ValueError("Cannot draw penalty card; deck is empty.")

        # Grab the prize and the penalty
        discard_card = self.discard_pile.pop()
        penalty_card = self.deck.pop(0)

        # Give both to the target player
        interrupter = self.players[self.may_i_target_idx]
        interrupter.receive_cards([discard_card, penalty_card])

        interrupter.may_is_used += 1

        self._record_action(
            self.player_pickup_counts,
            self.may_i_target_idx,
            int(discard_card.suit),
            int(discard_card.rank)
        )

    # --- Logic Helpers (Placeholders for later tasks) ---
    def check_hand_objective(self, player_idx):
        hand_tensor = self.players[player_idx].private_hand
        req_sets, req_runs = self.config.objective_map[self.current_round_idx]

        return self.find_melds(hand_tensor, req_sets, req_runs) is not None

    def _record_action(self, count_tensor, player_idx, suit, rank):
        """Records a public action into the provided presence tensor."""
        self._sync_ace(count_tensor[player_idx], suit, rank, increment=True)

    def _sync_ace(self, tensor, suit, rank, increment=False):
        """Helper to keep Ace-Low (0) and Ace-High (13) synced during matrix math."""

        # 1. Convert to standard Python int before math to bypass NumPy scalar warnings
        current_val = int(tensor[suit, rank])

        if increment:
            # Cap at 10 to prevent positive overflow
            tensor[suit, rank] = min(current_val + 1, 10)
            if rank == 0:
                tensor[suit, 13] = min(int(tensor[suit, 13]) + 1, 10)
            elif rank == 13:
                tensor[suit, 0] = min(int(tensor[suit, 0]) + 1, 10)
        else:
            # Floor at 0 to prevent negative underflow (the 255 hallucination)
            tensor[suit, rank] = max(current_val - 1, 0)
            if rank == 0:
                tensor[suit, 13] = max(int(tensor[suit, 13]) - 1, 0)
            elif rank == 13:
                tensor[suit, 0] = max(int(tensor[suit, 0]) - 1, 0)

    def find_melds(self, tensor, sets_needed, runs_needed):
        """
        Recursive DFS for minimum-size melds (3-card sets, 4-card runs) meeting an objective.

        Returns None if the hand cannot meet it, otherwise a list with one tuple per meld:
          ('run', suit, start)  - 4 cards of `suit` at positions start..start+3
                                  (position 0 = low Ace, 13 = high Ace)
          ('set', rank, suits)  - 3 cards of `rank`, one per entry in `suits`
        """
        if sets_needed == 0 and runs_needed == 0:
            return []

        # 1. Try extracting Runs
        if runs_needed > 0:
            for suit in range(4):
                for start in range(11):
                    if np.all(tensor[suit, start:start + 4] >= 1):
                        new_tensor = tensor.copy()
                        for r in range(start, start + 4):
                            self._sync_ace(new_tensor, suit, r)

                        rest = self.find_melds(new_tensor, sets_needed, runs_needed - 1)
                        if rest is not None:
                            return [('run', suit, start)] + rest

        # 2. Try extracting Sets
        if sets_needed > 0:
            for rank in range(13):
                if np.sum(tensor[:, rank]) >= 3:
                    available_suits = []
                    for suit in range(4):
                        for _ in range(tensor[suit, rank]):
                            available_suits.append(suit)

                    for combo in set(itertools.combinations(available_suits, 3)):
                        new_tensor = tensor.copy()
                        for suit in combo:
                            self._sync_ace(new_tensor, suit, rank)

                        rest = self.find_melds(new_tensor, sets_needed - 1, runs_needed)
                        if rest is not None:
                            return [('set', rank, combo)] + rest

        return None

    def go_down(self, player_idx):
        """Moves the required objective melds to the table."""
        player = self.players[player_idx]
        req_sets, req_runs = self.config.objective_map[self.current_round_idx]

        melds = self.find_melds(player.private_hand, req_sets, req_runs)
        if melds is None:
            return

        for kind, a, b in melds:
            if kind == 'run':
                suit, start = a, b
                for pos in range(start, start + 4):
                    player.discard(suit, pos % 13)  # position 13 is the Ace
                    self._sync_ace(self.table_runs, suit, pos % 13, increment=True)
                self.runs_on_table.append(TableRun(suit, start, start + 3))
            else:
                rank, suits = a, b
                for suit in suits:
                    player.discard(suit, rank)
                    self._sync_ace(self.table_sets, suit, rank, increment=True)

    def auto_place_cards(self, player_idx):
        """
        The Extension Phase: Logic to handle both the initial objective
        if not yet down, followed by deadwood extensions.
        """
        player = self.players[player_idx]

        # 1. If this is a direct call and the player isn't down yet,
        # try to commit the initial batch first.
        if not player.is_down:
            self.go_down(player_idx)
            player.is_down = True

        # 2. Now perform the card-by-card scan for everything else
        self._place_key_cards(player_idx)

    def _place_key_cards(self, player_idx):
        """
        Places every Key Card in the player's hand. Repeats until nothing more fits,
        because placing one card can open a run end for another.
        """
        player = self.players[player_idx]
        placed = True
        while placed:
            placed = False
            for card in list(player.hand_list):
                if self.is_key_card(int(card.suit), int(card.rank)):
                    player.discard(int(card.suit), int(card.rank))
                    self._place_on_table(int(card.suit), int(card.rank))
                    placed = True

    def is_key_card(self, suit, rank):
        """True if a card can legally be placed on some meld on the table."""
        return bool(np.any(self.table_sets[:, rank % 13] > 0)) or self._can_extend_run(suit, rank)

    def _extendable_run(self, suit, rank):
        """
        Returns (run, position) for the first table run this card extends, or None.
        Runs grow only at their ends; a low Ace end (position 0) and a high Ace end
        (position 13) are closed.
        """
        for run in self.runs_on_table:
            if run.suit != suit:
                continue
            for pos in _run_positions(rank):
                if pos == run.low - 1 or pos == run.high + 1:
                    return run, pos
        return None

    def _can_extend_run(self, suit, rank):
        """True if this card can be placed at the end of a run on the table."""
        return self._extendable_run(suit, rank) is not None

    def _place_on_table(self, suit, rank):
        """
        Puts a card (already removed from a hand) onto a meld.
        A set of that rank takes priority over a run (review item 3, pending decision).
        """
        rank = rank % 13
        if np.any(self.table_sets[:, rank] > 0):
            self._sync_ace(self.table_sets, suit, rank, increment=True)
            return

        found = self._extendable_run(suit, rank)
        if found is None:
            raise ValueError(f"Card {suit}-{rank} does not fit any meld on the table.")
        run, pos = found
        if pos < run.low:
            run.low = pos
        else:
            run.high = pos
        self._sync_ace(self.table_runs, suit, rank, increment=True)

    def execute_discard(self, player_idx, card_index):
        """Moves a card from the player's hand to the discard pile, burying the old top card."""
        player = self.players[player_idx]
        card_to_discard = player.hand_list[card_index]

        # 1. Remove from player's hand
        player.discard(int(card_to_discard.suit), int(card_to_discard.rank))

        # 2. Bury the current top card into the dead_cards tensor (Channel 20)
        if self.discard_pile:
            old_top = self.discard_pile[-1]
            self._sync_ace(self.dead_cards, int(old_top.suit), int(old_top.rank), increment=True)

        # 3. Place new card on top of the list (Representing Channel 3)
        self.discard_pile.append(card_to_discard)
        self._record_action(
            self.player_discard_counts,
            player_idx,
            int(card_to_discard.suit),
            int(card_to_discard.rank)
        )

    def execute_table_play(self, player_idx: int, card_index: int):
        """Moves one Key Card from the player's hand onto a meld on the table."""
        player = self.players[player_idx]
        card = player.hand_list[card_index]
        suit, rank = int(card.suit), int(card.rank)

        if not self.is_key_card(suit, rank):
            raise ValueError(f"{card} is not a Key Card and cannot be placed.")

        player.discard(suit, rank)
        self._place_on_table(suit, rank)

    def start_may_i_checks(self):
        """Initializes the May-I loop to the first downstream player."""
        self.may_i_target_idx = (self.active_player_idx + 1) % self.num_players

    def advance_may_i_target(self):
        """Moves the target pointer to the next downstream player."""
        self.may_i_target_idx = (self.may_i_target_idx + 1) % self.num_players

    def is_may_i_target_eligible(self):
        """Checks if the currently targeted player can legally call a May-I."""
        if self.may_i_target_idx is None:
            return False

        target_player = self.players[self.may_i_target_idx]

        # Rule 1: Cannot May-I if already down
        if target_player.is_down:
            return False

        # --- Rule 2: Cannot May-I if out of tokens ---
        # Checks your config if the variable exists, otherwise strictly enforces 3
        limit = getattr(self.config, 'max_may_is', 3)
        if target_player.may_is_used >= limit:
            return False

        return True

    def all_may_i_targets_checked(self):
        """
        Returns True if the loop has reached the player who discarded the card.
        The discarder is always the player immediately preceding the active player.
        """
        discarder_idx = (self.active_player_idx - 1) % self.num_players
        return self.may_i_target_idx == discarder_idx

    def clone(self):
        """
        Creates a deep copy of the entire game state in RAM.
        Optimized for high-speed Actor-Critic fast_engine data generation.
        """
        # Create a new instance passing the shared immutable config
        new_ctx = GameContext(num_players=self.num_players, config=self.config)

        # Copy Lists
        new_ctx.deck = list(self.deck)
        new_ctx.discard_pile = list(self.discard_pile)

        # Copy Players using their internal clone method
        new_ctx.players = [p.clone() for p in self.players]

        # Copy NumPy Tensors
        new_ctx.table_sets = self.table_sets.copy()
        new_ctx.table_runs = self.table_runs.copy()
        new_ctx.runs_on_table = [TableRun(r.suit, r.low, r.high) for r in self.runs_on_table]
        new_ctx.dead_cards = self.dead_cards.copy()

        new_ctx.player_discard_counts = self.player_discard_counts.copy()
        new_ctx.player_pickup_counts = self.player_pickup_counts.copy()

        # Copy State Markers
        new_ctx.active_player_idx = self.active_player_idx
        new_ctx.dealer_idx = self.dealer_idx
        new_ctx.current_round_idx = self.current_round_idx
        new_ctx.total_actions = self.total_actions

        if hasattr(self, 'may_i_target_idx'):
            new_ctx.may_i_target_idx = self.may_i_target_idx

        return new_ctx

    def get_oracle_truth(self, player_idx: int) -> np.ndarray:
        """
        Extracts the absolute ground truth of the opponents' hidden hands.
        Shape: (3, 4, 14) representing Op1 (Left), Op2 (Across), Op3 (Right).
        In a 3-player game, Op3 (index 2) is strictly zero-padded.
        """
        oracle_tensor = np.zeros((3, 4, 14), dtype=np.int8)

        for i in range(1, 4):  # Relative opponents: 1, 2, 3
            # Strict Zero-Padding for missing 4th player
            if i == 3 and self.num_players == 3:
                continue

            target_idx = (player_idx + i) % self.num_players

            # Map Opponent 1 to index 0, Opponent 2 to index 1, Opponent 3 to index 2
            oracle_tensor[i - 1] = self.players[target_idx].private_hand

        return oracle_tensor

    def hand_penalty(self, player_idx: int) -> int:
        """Penalty points for the cards currently in a player's hand (rules.md Section 9)."""
        total = 0
        for card in self.players[player_idx].hand_list:
            rank_val = int(card.rank)
            if rank_val == 0:  # Ace
                total += self.config.points_ace
            elif 7 <= rank_val <= 12:  # 8 through King
                total += self.config.points_eight_to_king
            else:  # 2 through 7
                total += self.config.points_two_to_seven
        return total

    def get_action_mask(self, player_idx: int, state_id: str) -> np.ndarray:
        """
        Generates a 58-element boolean array masking illegal actions for the given phase.
        Indices:
        0: Pick_Stock, 1: Pick_Discard, 2: Call_MayI, 3: Pass
        4: Go_Down, 5: Wait / End_Table_Play
        6-57: Card identifiers (Discard or Play)
        """
        state_id = state_id.lower().replace(' ', '_').replace('-', '_')
        mask = np.zeros(58, dtype=bool)
        player = self.players[player_idx]

        if state_id == 'pickup_decision':
            mask[0] = True  # Pick Stock is always legal
            if len(self.discard_pile) > 0:
                mask[1] = True  # Pick Discard requires a card in the pile

        elif state_id == 'may_i_decision':
            mask[2] = True  # Call May-I
            mask[3] = True  # Pass

        elif state_id == 'go_down_decision':
            mask[4] = True  # Go Down
            mask[5] = True  # Wait

        elif state_id == 'discard_phase':
            # Control actions are completely disabled.
            # Only cards currently in the physical hand are legal to discard.
            for card in player.hand_list:
                # Calculate the exact offset: 6 + (Suit * 13) + Rank
                card_idx = 6 + (int(card.suit) * 13) + int(card.rank)
                mask[card_idx] = True


        elif state_id == 'table_play_phase':
            mask[5] = True  # End Table Play
            # Cards in hand are ONLY legal if they actually fit an existing meld
            for card in player.hand_list:
                suit, rank = int(card.suit), int(card.rank)
                if self.is_key_card(suit, rank):
                    card_idx = 6 + (int(card.suit) * 13) + int(card.rank)
                    mask[card_idx] = True

        else:
            # We strictly enforce known states to prevent silent tensor failures
            raise ValueError(f"Cannot generate action mask for unknown state: {state_id}")

        return mask