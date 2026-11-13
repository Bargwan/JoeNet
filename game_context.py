import random
from dataclasses import dataclass

import numpy as np

from cards import create_double_deck
from config import JoeConfig
from player import Player

ACE_LOW = 0
ACE_HIGH = 13

# --- Action numbering ---
# Agents choose actions by number from a boolean mask (get_action_mask).
PICK_STOCK, PICK_DISCARD, CALL_MAY_I, PASS, GO_DOWN, WAIT = range(6)
END_TABLE_PLAY = WAIT  # The same action ends table play.

# Where a Key Card goes when placed on the table.
ON_SET, RUN_LOW_END, RUN_HIGH_END = range(3)

DISCARD_BASE = 6                   # 6..57: discard a card
PLACE_BASE = DISCARD_BASE + 52     # 58..213: place a card on a set / run low end / run high end
NUM_ACTIONS = PLACE_BASE + 3 * 52


def card_code(suit: int, rank: int) -> int:
    """0..51 for a card's suit and rank (an Ace is rank 0)."""
    return int(suit) * 13 + int(rank) % 13


def discard_action(suit: int, rank: int) -> int:
    return DISCARD_BASE + card_code(suit, rank)


def place_action(suit: int, rank: int, destination: int) -> int:
    return PLACE_BASE + destination * 52 + card_code(suit, rank)


def decode_card_action(action: int) -> tuple:
    """
    Returns (suit, rank, destination) for a discard or place action.
    destination is None for a discard.
    """
    if DISCARD_BASE <= action < PLACE_BASE:
        destination, code = None, action - DISCARD_BASE
    elif PLACE_BASE <= action < NUM_ACTIONS:
        destination, code = divmod(action - PLACE_BASE, 52)
    else:
        raise ValueError(f"Action {action} is not a card action.")
    return code // 13, code % 13, destination


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

    # --- Melds ---
    # A meld is a tuple:
    #   ('run', suit, start) - 4 cards of `suit` at positions start..start+3
    #                          (position 0 = low Ace, 13 = high Ace)
    #   ('set', rank)        - 3 cards of `rank`. Which suits doesn't matter: any
    #                          leftover card of that rank can go straight onto the set.
    # Melds are minimum size; longer melds are made by placing Key Cards afterwards.

    def find_melds(self, tensor, sets_needed, runs_needed):
        """One combination of melds meeting the objective, or None if the hand can't."""
        return next(self._meld_combinations(tensor, sets_needed, runs_needed), None)

    def find_all_melds(self, tensor, sets_needed, runs_needed):
        """Every distinct combination of melds meeting the objective (possibly empty)."""
        return list(self._meld_combinations(tensor, sets_needed, runs_needed))

    def _meld_combinations(self, tensor, sets_needed, runs_needed):
        """Yields each distinct combination of melds once, in a fixed order."""
        candidates = self._candidate_melds(tensor)
        yield from self._combine_melds(tensor, sets_needed, runs_needed, candidates, 0)

    def _combine_melds(self, tensor, sets_needed, runs_needed, candidates, start):
        if sets_needed == 0 and runs_needed == 0:
            yield []
            return
        # Candidates are tried in list order, never going back to an earlier one,
        # so each combination appears once. Repeating the same candidate is allowed
        # (two sets of 7s, or two identical runs).
        for i in range(start, len(candidates)):
            meld = candidates[i]
            if (meld[0] == 'run' and runs_needed == 0) or (meld[0] == 'set' and sets_needed == 0):
                continue
            remaining = self._remove_meld(tensor, meld)
            if remaining is None:
                continue
            for rest in self._combine_melds(remaining,
                                            sets_needed - (meld[0] == 'set'),
                                            runs_needed - (meld[0] == 'run'),
                                            candidates, i):
                yield [meld] + rest

    def _candidate_melds(self, tensor):
        """Every meld the hand could make on its own: runs first, then sets."""
        runs = [('run', suit, start) for suit in range(4) for start in range(11)
                if np.all(tensor[suit, start:start + 4] >= 1)]
        sets = [('set', rank) for rank in range(13) if tensor[:, rank].sum() >= 3]
        return runs + sets

    def _set_suits(self, tensor, rank):
        """The suits of the 3 cards a set of `rank` uses: the most plentiful suits first."""
        counts = [int(tensor[suit, rank]) for suit in range(4)]
        suits = []
        for _ in range(3):
            suit = max(range(4), key=lambda s: counts[s])
            if counts[suit] == 0:
                return None
            counts[suit] -= 1
            suits.append(suit)
        return suits

    def _meld_cards(self, tensor, meld):
        """The (suit, rank) of each card a meld takes from this hand tensor, or None if missing."""
        if meld[0] == 'run':
            _, suit, start = meld
            if not (0 <= start <= 10 and np.all(tensor[suit, start:start + 4] >= 1)):
                return None
            return [(suit, pos % 13) for pos in range(start, start + 4)]
        _, rank = meld
        suits = self._set_suits(tensor, rank)
        return None if suits is None else [(suit, rank) for suit in suits]

    def _remove_meld(self, tensor, meld):
        """A copy of the hand tensor without the meld's cards, or None if they aren't there."""
        cards = self._meld_cards(tensor, meld)
        if cards is None:
            return None
        remaining = tensor.copy()
        for suit, rank in cards:
            self._sync_ace(remaining, suit, rank)
        return remaining

    def go_down(self, player_idx, melds=None):
        """
        Lays the player's objective melds on the table and marks them as down.
        `melds` is the player's choice (see the meld format above); if None, the
        first valid combination is used. Raises ValueError, changing nothing, if
        the choice does not exactly meet the objective with cards in hand.
        """
        player = self.players[player_idx]
        req_sets, req_runs = self.config.objective_map[self.current_round_idx]

        if melds is None:
            melds = self.find_melds(player.private_hand, req_sets, req_runs)
            if melds is None:
                raise ValueError(f"Player {player_idx} cannot meet the objective.")
        melds = [tuple(m) for m in melds]

        kinds = [m[0] for m in melds]
        if kinds.count('set') != req_sets or kinds.count('run') != req_runs \
                or len(kinds) != req_sets + req_runs:
            raise ValueError(f"{melds} does not match the objective "
                             f"({req_sets} sets, {req_runs} runs).")

        # Work out every card first (runs before sets, so a set never takes a card a
        # run needs), and only then move anything.
        hand = player.private_hand.copy()
        laid = []
        for meld in sorted(melds, key=lambda m: m[0] != 'run'):
            cards = self._meld_cards(hand, meld)
            if cards is None:
                raise ValueError(f"Player {player_idx} does not hold the cards for {meld}.")
            for suit, rank in cards:
                self._sync_ace(hand, suit, rank)
            laid.append((meld, cards))

        for meld, cards in laid:
            table = self.table_runs if meld[0] == 'run' else self.table_sets
            for suit, rank in cards:
                player.discard(suit, rank)
                self._sync_ace(table, suit, rank, increment=True)
            if meld[0] == 'run':
                _, suit, start = meld
                self.runs_on_table.append(TableRun(suit, start, start + 3))
        player.is_down = True

    def auto_place_cards(self, player_idx):
        """Goes down (if not already) with the first valid melds, then places every Key Card."""
        if not self.players[player_idx].is_down:
            self.go_down(player_idx)
        self._place_key_cards(player_idx)

    def _place_key_cards(self, player_idx):
        """
        Places every Key Card in the player's hand, preferring runs over sets.
        Repeats until nothing more fits, because placing one card can open a
        run end for another.
        """
        player = self.players[player_idx]
        placed = True
        while placed:
            placed = False
            for card in list(player.hand_list):
                suit, rank = int(card.suit), int(card.rank)
                destinations = self.placement_destinations(suit, rank)
                if destinations:
                    player.discard(suit, rank)
                    self._place_on_table(suit, rank, destinations[-1])  # runs listed after sets
                    placed = True

    def is_key_card(self, suit, rank):
        """True if a card can legally be placed on some meld on the table."""
        return bool(self.placement_destinations(suit, rank))

    def placement_destinations(self, suit, rank):
        """
        The distinct places a card can go: ON_SET, RUN_LOW_END and/or RUN_HIGH_END.
        Choosing between two sets of the same rank, or two runs that share the
        same end, never changes what can be placed later, so these three cover
        every meaningful choice.
        """
        destinations = []
        if np.any(self.table_sets[:, rank % 13] > 0):
            destinations.append(ON_SET)
        for end in (RUN_LOW_END, RUN_HIGH_END):
            if self._find_run_end(suit, rank, end) is not None:
                destinations.append(end)
        return destinations

    def _find_run_end(self, suit, rank, end):
        """
        Returns (run, position) for the first table run this card extends at the
        given end, or None. A low Ace end (position 0) and a high Ace end
        (position 13) are closed.
        """
        for run in self.runs_on_table:
            if run.suit != suit:
                continue
            for pos in _run_positions(rank):
                if end == RUN_LOW_END and pos == run.low - 1:
                    return run, pos
                if end == RUN_HIGH_END and pos == run.high + 1:
                    return run, pos
        return None

    def _can_extend_run(self, suit, rank):
        """True if this card can be placed at the end of a run on the table."""
        destinations = self.placement_destinations(suit, rank)
        return RUN_LOW_END in destinations or RUN_HIGH_END in destinations

    def _place_on_table(self, suit, rank, destination):
        """Puts a card (already removed from a hand) onto a meld at the chosen destination."""
        rank = rank % 13
        if destination == ON_SET:
            self._sync_ace(self.table_sets, suit, rank, increment=True)
            return

        run, pos = self._find_run_end(suit, rank, destination)
        if destination == RUN_LOW_END:
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

    def execute_table_play(self, player_idx: int, card_index: int, destination: int):
        """Moves one Key Card from the player's hand onto the chosen meld destination."""
        player = self.players[player_idx]
        card = player.hand_list[card_index]
        suit, rank = int(card.suit), int(card.rank)

        if destination not in self.placement_destinations(suit, rank):
            raise ValueError(f"{card} cannot be placed at destination {destination}.")

        player.discard(suit, rank)
        self._place_on_table(suit, rank, destination)

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
        Boolean mask of the legal actions for the given phase (NUM_ACTIONS long).
        See the action numbering constants at the top of this module.
        """
        state_id = state_id.lower().replace(' ', '_').replace('-', '_')
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        player = self.players[player_idx]

        if state_id == 'pickup_decision':
            mask[PICK_STOCK] = True
            if self.discard_pile:
                mask[PICK_DISCARD] = True

        elif state_id == 'may_i_decision':
            mask[CALL_MAY_I] = True
            mask[PASS] = True

        elif state_id == 'go_down_decision':
            mask[GO_DOWN] = True
            mask[WAIT] = True

        elif state_id == 'discard_phase':
            for card in player.hand_list:
                mask[discard_action(card.suit, card.rank)] = True

        elif state_id == 'table_play_phase':
            mask[END_TABLE_PLAY] = True
            for card in player.hand_list:
                suit, rank = int(card.suit), int(card.rank)
                for destination in self.placement_destinations(suit, rank):
                    mask[place_action(suit, rank, destination)] = True

        else:
            raise ValueError(f"Cannot generate action mask for unknown state: {state_id}")

        return mask
