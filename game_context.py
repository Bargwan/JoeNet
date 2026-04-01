import itertools
import random

import numpy as np

from cards import create_double_deck
from config import JoeConfig
from player import Player


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

        success, _, _ = self._search_melds(hand_tensor, req_sets, req_runs)
        return success

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

    def _search_melds(self, tensor, sets_needed, runs_needed):
        """
        Recursive DFS to find valid meld combinations.
        Returns: (success_bool, extracted_sets_tensor, extracted_runs_tensor)
        """
        if sets_needed == 0 and runs_needed == 0:
            return (
                True,
                np.zeros((4, 14), dtype=np.int8),
                np.zeros((4, 14), dtype=np.int8)
                )

        # 1. Try extracting Runs
        if runs_needed > 0:
            for suit in range(4):
                for start_rank in range(11):
                    if np.all(tensor[suit, start_rank:start_rank + 4] >= 1):
                        new_tensor = tensor.copy()
                        for r in range(start_rank, start_rank + 4):
                            self._sync_ace(new_tensor, suit, r)

                        success, ext_sets, ext_runs = self._search_melds(new_tensor,
                                                                         sets_needed,
                                                                         runs_needed - 1)
                        if success:
                            # Reconstruct the run for the return tensor
                            for r in range(start_rank, start_rank + 4):
                                ext_runs[suit, r] += 1
                                if r == 0:
                                    ext_runs[suit, 13] += 1
                                elif r == 13:
                                    ext_runs[suit, 0] += 1
                            return True, ext_sets, ext_runs

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

                        success, ext_sets, ext_runs = self._search_melds(new_tensor,
                                                                         sets_needed - 1,
                                                                         runs_needed)
                        if success:
                            # Reconstruct the set for the return tensor
                            for suit in combo:
                                ext_sets[suit, rank] += 1
                                if rank == 0:
                                    ext_sets[suit, 13] += 1
                                elif rank == 13:
                                    ext_sets[suit, 0] += 1
                            return True, ext_sets, ext_runs

        return False, None, None

        # If we reach here, this branch failed. Backtrack.
        return False, None

    def go_down(self, player_idx):
        """Moves the required objective melds to the table."""
        player = self.players[player_idx]
        req_sets, req_runs = self.config.objective_map[self.current_round_idx]

        # Search for the cards that meet the round objective
        success, ext_sets, ext_runs = self._search_melds(player.private_hand, req_sets, req_runs)

        if success:
            # Update public table tensors
            self.table_sets += ext_sets
            self.table_runs += ext_runs

            # Remove specific cards from player list and tensor
            combined = ext_sets + ext_runs
            for suit in range(4):
                for rank in range(13):
                    for _ in range(combined[suit, rank]):
                        player.discard(suit, rank)

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
        Iterates through the player's remaining hand and identifies
        any single cards that can extend existing sets or runs on the table.
        """
        player = self.players[player_idx]

        # We must iterate over a copy of the list because we will be modifying the hand
        for card in list(player.hand_list):
            suit = int(card.suit)
            rank = int(card.rank)

            # 1. Check Table Sets (Spatial Layer 1)
            # If any card of this rank exists in Table Sets, we can add to it
            if np.any(self.table_sets[:, rank] > 0):
                player.discard(suit, rank)
                self.table_sets[suit, rank] += 1
                if rank == 0:
                    self.table_sets[suit, 13] += 1  # Sync High Ace
                continue  # Card is placed, move to next hand card

            # 2. Check Table Runs (Spatial Layer 2)
            # If the same suit has an adjacent rank in Table Runs, we can extend it
            elif self._can_extend_run(suit, rank):
                player.discard(suit, rank)
                self.table_runs[suit, rank] += 1
                if rank == 0:
                    self.table_runs[suit, 13] += 1  # Sync High Ace
                elif rank == 12:
                    self.table_runs[suit, 0] += 1  # Sync Low Ace if King placed

    def _can_extend_run(self, suit, rank):
        """
        Checks if a specific suit/rank coordinate is adjacent to an existing run.
        Strictly prevents King/Two wrap-around by checking for Ghost Aces.
        """
        # Rank 1 (Two)
        if rank == 1:
            if self.table_runs[suit, 2] > 0: return True
            # Can connect to Ace Low ONLY if King is not present
            if self.table_runs[suit, 0] > 0 and self.table_runs[suit, 12] == 0: return True
            return False

        # Rank 12 (King)
        if rank == 12:
            if self.table_runs[suit, 11] > 0: return True
            # Can connect to Ace High ONLY if Two is not present
            if self.table_runs[suit, 13] > 0 and self.table_runs[suit, 1] == 0: return True
            return False

        # Rank 0 or 13 (Ace)
        if rank == 0 or rank == 13:
            # Can connect to Two ONLY if King is not present
            if self.table_runs[suit, 1] > 0 and self.table_runs[suit, 12] == 0: return True
            # Can connect to King ONLY if Two is not present
            if self.table_runs[suit, 12] > 0 and self.table_runs[suit, 1] == 0: return True
            return False

        # Standard Case: 3 through Queen (Ranks 2 to 11)
        if 1 < rank < 12:
            if self.table_runs[suit, rank - 1] > 0 or self.table_runs[suit, rank + 1] > 0:
                return True

        return False

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
        player = self.players[player_idx]
        card_to_play = player.hand_list.pop(card_index)

        # Re-sync private hand
        player.private_hand.fill(0)
        for c in player.hand_list:
            suit, rank = int(c.suit), int(c.rank)
            player.private_hand[suit, rank] += 1
            if rank == 0:
                player.private_hand[suit, 13] += 1

        suit, rank = int(card_to_play.suit), int(card_to_play.rank)

        # Route to the correct public tensor
        if np.any(self.table_sets[:, rank] > 0):
            self.table_sets[suit, rank] += 1
            if rank == 0:
                self.table_sets[suit, 13] += 1
        elif self._can_extend_run(suit, rank):
            self.table_runs[suit, rank] += 1
            if rank == 0:
                self.table_runs[suit, 13] += 1

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

    def get_input_tensor(self, player_idx: int, state_id: str):
        """
        Generates the Information Set tensors for the Neural Network.
        Returns: {'spatial': ndarray(13, 4, 14, int8), 'scalar': ndarray(28, float32)}
        """
        # --- NEW STRICT GUARD ---
        valid_states = ['pickup_decision', 'may_i_decision', 'go_down_decision', 'table_play_phase',
                        'discard_phase']
        if state_id not in valid_states:
            raise ValueError(
                f"CRITICAL: get_input_tensor called with invalid or missing state_id: '{state_id}'")

        # 1. Initialize LEAN 13-channel spatial tensor
        spatial = np.zeros((13, 4, 14), dtype=np.int8)
        scalar = np.zeros(28, dtype=np.float32)

        target_player = self.players[player_idx]

        # ==========================================
        # SPATIAL TENSOR (Shape: 13, 4, 14)
        # ==========================================

        # Channel 0: Private Hand
        spatial[0] = target_player.private_hand

        # Channel 1 & 2: Table Melds
        spatial[1] = self.table_sets
        spatial[2] = self.table_runs

        # Channel 3: Top Discard
        if self.discard_pile:
            top_card = self.discard_pile[-1]
            suit = int(top_card.suit)
            rank = int(top_card.rank)
            self._sync_ace(spatial[3], suit, rank, increment=True)

        # Channels 4-7: Discard History (Presence only)
        # Channels 8-11: Pickup History (Presence only)
        for i in range(4):
            # Zero-pad Op3 if it's a 3-player game
            if i == 3 and self.num_players == 3:
                continue

            target_idx = (player_idx + i) % self.num_players

            # Map Discards to 4, 5, 6, 7
            spatial[4 + i] = self.player_discard_counts[target_idx]

            # Map Pickups to 8, 9, 10, 11
            spatial[8 + i] = self.player_pickup_counts[target_idx]

        # Channel 12: Dead Cards
        spatial[12] = self.dead_cards

        # ==========================================
        # SCALAR TENSOR (Shape: 23)
        # ==========================================

        # --- SCALAR TENSOR (28 Features) ---
        scalar = np.zeros(28, dtype=np.float32)

        # 0-6: Round Objective (One-Hot)
        scalar[self.current_round_idx] = 1.0

        # 7: Circuit Progress (Strategic Turn Number)
        max_circuits = getattr(self.config, 'max_turns', 15)
        scalar[7] = min(self.current_circuit / 100.0, 1.0)

        # 8: Stock Depth
        scalar[8] = len(self.deck) / 104.0

        # Relative Player Loop (Scores, Hand Sizes, May-Is)
        for i in range(4):
            # Zero-pad Op3 if it's a 3-player game
            if i == 3 and self.num_players == 3:
                continue

            target_idx = (player_idx + i) % self.num_players
            p = self.players[target_idx]

            scalar[9 + i] = np.sqrt(p.score) / np.sqrt(1470.0)
            scalar[13 + i] = len(p.hand_list) / 18.0
            scalar[17 + i] = p.may_is_used / 3.0

        # 21: Is Down?
        scalar[21] = 1.0 if self.players[player_idx].is_down else 0.0

        # 22: Is 3-Player Game?
        scalar[22] = 1.0 if self.num_players == 3 else 0.0

        # 23-27: Active Phase One-Hot
        if state_id == 'pickup_decision':
            scalar[23] = 1.0
        elif state_id == 'may_i_decision':
            scalar[24] = 1.0
        elif state_id == 'go_down_decision':
            scalar[25] = 1.0
        elif state_id == 'table_play_phase':
            scalar[26] = 1.0
        elif state_id == 'discard_phase':
            scalar[27] = 1.0

        return {
            'spatial': spatial,
            'scalar': scalar
        }

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
                if np.any(self.table_sets[:, rank] > 0) or self._can_extend_run(suit, rank):
                    card_idx = 6 + (int(card.suit) * 13) + int(card.rank)
                    mask[card_idx] = True

        else:
            # We strictly enforce known states to prevent silent tensor failures
            raise ValueError(f"Cannot generate action mask for unknown state: {state_id}")

        return mask