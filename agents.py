import random
import numpy as np
import onnxruntime as ort

# Global cache to hold models in the multiprocessing worker's local RAM
_global_onnx_sessions = {}


class Agent:
    def select_action(self, state_id, ctx, player_idx, action_mask):
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, random_seed=None):
        self.rng = random.Random(random_seed)

    def select_action(self, state_id, ctx, player_idx, action_mask):
        """Selects a completely random action from the allowed mask."""
        valid_actions = np.where(action_mask)[0]
        return self.rng.choice(valid_actions)


class HeuristicAgent(Agent):
    """
    The baseline rule-based bot.
    For the ProbabilisticAgent, this class primarily serves as a fallback handler
    for strict, rule-bound phases (like playing cards to the table) where
    complex EV simulation is unnecessary.
    """

    def __init__(self, random_seed=None):
        self.rng = random.Random(random_seed)

    def select_action(self, state_id, ctx, player_idx, action_mask):
        """
        Executes heuristic logic for the current phase.
        Inherited Purpose: The ProbabilisticAgent falls back to this method
        specifically for the 'table_play_phase'. If a card can be played on the
        table, it is always mathematically optimal to do so to reduce deadwood,
        so we let this greedy heuristic handle it instantly rather than simulating it.
        """
        player = ctx.players[player_idx] if ctx and ctx.players else None
        hand = player.hand_list if player else []
        discard_top = ctx.discard_pile[-1] if ctx and ctx.discard_pile else None

        # --- Phase: PICKUP DECISION ---
        if state_id == 'pickup_decision' and discard_top:
            is_down = getattr(player, 'is_down', False)

            # 1. Key-Card Hoarding: If we have the objective but aren't down, grab playable cards
            if action_mask[1] and not is_down and ctx.check_hand_objective(player_idx):
                if self._is_playable_on_table(discard_top, ctx):
                    return 1

            # 2. Standard Pickup Logic
            if is_down:
                if action_mask[1] and self._is_playable_on_table(discard_top, ctx):
                    return 1
                elif action_mask[0]:
                    return 0
            else:
                if action_mask[1]:
                    if self._completes_objective(discard_top, player_idx, ctx):
                        return 1
                    elif self._is_useful_pickup(discard_top, player_idx, ctx):
                        return 1

            # 3. Fallback (Draw from stock)
            if action_mask[0]:
                return 0
            elif action_mask[1]:
                return 1

        # --- Phase: MAY-I DECISION ---
        elif state_id == 'may_i_decision' and discard_top:
            if action_mask[2]:
                # 1. Instantly advances meld progress
                if self._completes_objective(discard_top, player_idx, ctx):
                    return 2

                # 2. Strategic Capacity Expansion for Late Rounds (3+)
                if ctx.current_round_idx >= 3 and len(hand) <= 11:
                    if self._is_useful_pickup(discard_top, player_idx, ctx):
                        return 2

            if action_mask[3]:
                return 3

        # --- Phase: GO DOWN DECISION ---
        elif state_id == 'go_down_decision':
            if action_mask[4] and ctx.check_hand_objective(player_idx):
                # Pragmatic Win: Go down if it leaves us with <= 5 penalty cards
                if self._calculate_projected_deadwood(ctx, player_idx) <= 5:
                    return 4

                # Otherwise apply patience pressure
                if action_mask[5]:
                    go_down_prob = self._calculate_go_down_probability(ctx, player_idx)
                    return 4 if self.rng.random() <= go_down_prob else 5
                return 4
            elif action_mask[5]:
                return 5

        # --- Phase: TABLE PLAY DECISION ---
        elif state_id == 'table_play_phase':
            valid_actions = np.where(action_mask)[0]
            card_plays = [a for a in valid_actions if a >= 6]
            if card_plays:
                return card_plays[0]
            elif 5 in valid_actions:
                return 5
            elif 3 in valid_actions:
                return 3

        # --- Phase: DISCARD DECISION ---
        elif state_id == 'discard_phase' and hand:
            is_down = getattr(player, 'is_down', False)
            req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]
            base_progress = self._get_objective_progress(player.private_hand, req_sets, req_runs,
                                                         ctx)

            best_discard_idx = -1
            lowest_synergy = float('inf')

            # Activate the Gridlock Breaker slightly earlier (Circuit 15)
            is_stuck = ctx.current_circuit > 15

            for card in hand:
                action_idx = self._get_discard_action_idx(card)
                if action_idx < len(action_mask) and action_mask[action_idx]:

                    synergy = self._calculate_synergy(card, hand, ctx, player_idx)

                    # THE GRIDLOCK BREAKER (Entropy Injection)
                    if is_stuck and 0 < synergy < 2000:
                        synergy += self.rng.uniform(-40, 40)

                    # Absolute Objective Immunity
                    if not is_down and self._breaks_objective(card, player_idx, ctx, base_progress):
                        synergy += 10000

                    # Penalize high point values to dump heavy deadwood
                    deadwood_val = self._get_deadwood_value(card, ctx)
                    synergy -= (deadwood_val / 100.0)

                    if synergy < lowest_synergy:
                        lowest_synergy = synergy
                        best_discard_idx = action_idx

            if best_discard_idx != -1:
                return best_discard_idx

        # --- FALLBACK ---
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) > 0:
            return self.rng.choice(valid_actions)
        return -1

    # ==========================================
    # INTERNAL EVALUATION HELPERS
    # ==========================================

    def _is_useful_pickup(self, target_card, player_idx, ctx):
        """Evaluates outstanding needs and prioritizes sequences/pairs accordingly."""
        player = ctx.players[player_idx]
        hand = player.hand_list
        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]

        has_sets = ctx._search_melds(player.private_hand, sets_needed=req_sets, runs_needed=0)[0]
        has_runs = ctx._search_melds(player.private_hand, sets_needed=0, runs_needed=req_runs)[0]

        needs_sets = (req_sets > 0) and not has_sets
        needs_runs = (req_runs > 0) and not has_runs

        if needs_runs and not needs_sets:
            for card in hand:
                if card.suit == target_card.suit and card.rank == target_card.rank:
                    return False  # Refuse duplicate from discard for runs

        run_synergy_found = False
        set_synergy_found = False

        for card in hand:
            if needs_runs and card.suit == target_card.suit:
                if 0 < abs(card.rank.value - target_card.rank.value) <= 2:
                    run_synergy_found = True
            if needs_sets and card.rank == target_card.rank:
                set_synergy_found = True

        if needs_runs and needs_sets:
            if req_runs >= req_sets:
                return run_synergy_found
            return run_synergy_found or set_synergy_found
        elif needs_runs:
            return run_synergy_found
        elif needs_sets:
            return set_synergy_found

        return False

    def _calculate_synergy(self, target_card, hand_list, ctx, player_idx):
        """Refined synergy that dynamically evaluates outstanding needs and Key Cards."""
        player = ctx.players[player_idx]
        has_objective = ctx.check_hand_objective(player_idx)
        is_down = getattr(player, 'is_down', False)

        # 1. Key Card Phase (Objective met or already down)
        if has_objective or is_down:
            if self._is_playable_on_table(target_card, ctx):
                num_players = len(ctx.players)
                next_player = ctx.players[(player_idx + 1) % num_players]
                if getattr(next_player, 'is_down', False) and len(next_player.hand_list) <= 2:
                    return 4000  # Radioactive
                return 2000  # Standard Key Card
            return 0  # Garbage if not a key card

        # 2. Drafting Phase (Pre-Objective)
        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]
        has_sets = ctx._search_melds(player.private_hand, sets_needed=req_sets, runs_needed=0)[0]
        has_runs = ctx._search_melds(player.private_hand, sets_needed=0, runs_needed=req_runs)[0]

        needs_sets = (req_sets > 0) and not has_sets
        needs_runs = (req_runs > 0) and not has_runs

        if needs_runs and not needs_sets:
            duplicate_count = sum(
                1 for c in hand_list if c.suit == target_card.suit and c.rank == target_card.rank)
            if duplicate_count > 1:
                return -500  # Toxic deadwood

        synergy = 0
        for other in hand_list:
            if other is target_card:
                continue
            if needs_sets and other.rank == target_card.rank:
                synergy += 9
            if needs_runs and other.suit == target_card.suit:
                rank_diff = abs(other.rank.value - target_card.rank.value)
                if 0 < rank_diff <= 2:
                    synergy += 10

        return int(synergy)

    def _calculate_projected_deadwood(self, ctx, player_idx):
        player = ctx.players[player_idx]
        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]

        success, ext_sets, ext_runs = ctx._search_melds(player.private_hand, req_sets, req_runs)
        if not success:
            return len(player.hand_list)

        objective_tensor = ext_sets + ext_runs
        remaining_count = 0

        for card in player.hand_list:
            suit, rank = int(card.suit), int(card.rank)
            if objective_tensor[suit, rank] > 0:
                objective_tensor[suit, rank] -= 1
                continue
            if self._is_playable_on_table(card, ctx):
                continue
            remaining_count += 1

        return remaining_count

    def _calculate_go_down_probability(self, ctx, player_idx):
        prob = 0.8
        patience_threshold = 8.0 + ctx.current_round_idx
        turn_pressure = ctx.current_circuit / patience_threshold
        prob = max(prob, turn_pressure)

        for i, p in enumerate(ctx.players):
            if i != player_idx and getattr(p, 'is_down', False):
                cards_left = len(p.hand_list)
                prob = max(prob, 1.0 - (cards_left * 0.1))

        return min(1.0, prob)

    def _completes_objective(self, discard_card, player_idx, ctx):
        player = ctx.players[player_idx]
        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]

        if ctx._search_melds(player.private_hand, sets_needed=req_sets, runs_needed=req_runs)[0]:
            return False

        base_progress = self._get_objective_progress(player.private_hand, req_sets, req_runs, ctx)
        temp_tensor = player.private_hand.copy()
        suit, rank = discard_card.suit.value, discard_card.rank.value
        ctx._sync_ace(temp_tensor, suit, rank, increment=True)

        new_progress = self._get_objective_progress(temp_tensor, req_sets, req_runs, ctx)
        return new_progress > base_progress

    def _breaks_objective(self, candidate_card, player_idx, ctx, base_progress):
        if base_progress == 0:
            return False
        player = ctx.players[player_idx]
        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]

        temp_tensor = player.private_hand.copy()
        suit, rank = candidate_card.suit.value, candidate_card.rank.value
        ctx._sync_ace(temp_tensor, suit, rank, increment=False)

        new_progress = self._get_objective_progress(temp_tensor, req_sets, req_runs, ctx)
        return new_progress < base_progress

    def _get_objective_progress(self, tensor, req_sets, req_runs, ctx):
        for total in range(req_sets + req_runs, 0, -1):
            for s in range(min(total, req_sets), -1, -1):
                r = total - s
                if r <= req_runs:
                    if ctx._search_melds(tensor, sets_needed=s, runs_needed=r)[0]:
                        return total
        return 0

    def _get_deadwood_value(self, card, ctx):
        """
        Calculates the literal penalty point value of a single card.
        Uses: Aces=20, Face Cards=10, Number Cards=5.
        Rationale: Used by the math engine to quickly look up raw point values
        without needing matrix multiplication.
        """
        rank_val = int(card.rank)
        if rank_val == 0 or rank_val == 13:
            return ctx.config.points_ace
        elif 7 <= rank_val <= 12:
            return ctx.config.points_eight_to_king
        return ctx.config.points_two_to_seven

    def _is_playable_on_table(self, card, ctx):
        """
        Determines if a specific card can be legally appended to any face-up
        meld currently on the table.
        Rationale: Identifies "Key Cards" which effectively have 0 penalty
        risk if the bot successfully goes down.
        """
        suit, rank = int(card.suit), int(card.rank)
        if np.any(ctx.table_sets[:, rank] > 0):
            return True
        if ctx._can_extend_run(suit, rank):
            return True
        return False

    def _get_discard_action_idx(self, card):
        """
        Maps a physical Card object to its index in the 58-element action_mask.
        Math: (Suit * 13) + Rank + 6.
        """
        return (card.suit.value * 13) + card.rank.value + 6


class OpenHandAgent(HeuristicAgent):
    """
    A Phase 1 Data Generator that plays with 'open hands'.
    Uses Edge-Detection Tensor Math to evaluate the true lifespan of
    connected clusters, and never hallucinates its own cards.
    """

    def _get_card_availability(self, suit, rank, ctx, player_idx):
        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]
        num_players = len(ctx.players)
        upstream_idx = (player_idx - 1) % num_players

        deck_size = max(1.0, float(len(ctx.deck)))
        stock_density = 60.0 / deck_size

        total_weight = 2.0 * stock_density

        table_count = ctx.table_sets[suit, rank] + ctx.table_runs[suit, rank]
        total_weight -= (table_count * stock_density)

        dead_count = ctx.dead_cards[suit, rank]
        total_weight -= (dead_count * stock_density)

        if deck_size < 15 and dead_count > 0:
            total_weight += (dead_count * 0.2)

        # FIX 1: DEDUCT OUR OWN HAND! (Cures the Hallucination)
        my_held = ctx.players[player_idx].private_hand[suit, rank]
        total_weight -= (my_held * stock_density)

        for i, opp in enumerate(ctx.players):
            if i == player_idx: continue

            held_count = opp.private_hand[suit, rank]
            is_building = False

            if req_sets > 0 and np.sum(opp.private_hand[:, rank]) >= 2:
                is_building = True

            if req_runs > 0 and not is_building:
                min_idx = max(0, rank - 2)
                max_idx = min(14, rank + 3)
                if np.sum(opp.private_hand[suit, min_idx:max_idx]) >= 2:
                    is_building = True

            if held_count > 0:
                total_weight -= (held_count * stock_density)
                for _ in range(held_count):
                    if is_building:
                        total_weight += 0.0
                    elif i == upstream_idx:
                        total_weight += 2.0
                    else:
                        total_weight += 0.5
            else:
                if is_building:
                    total_weight -= (1.0 * stock_density)

        return max(0.0, total_weight)

    def _calculate_synergy(self, target_card, hand_list, ctx, player_idx):
        player = ctx.players[player_idx]
        has_objective = ctx.check_hand_objective(player_idx)
        is_down = getattr(player, 'is_down', False)

        if has_objective or is_down:
            if self._is_playable_on_table(target_card, ctx):
                next_player_idx = (player_idx + 1) % len(ctx.players)
                for i, opp in enumerate(ctx.players):
                    if i == player_idx: continue
                    if getattr(opp, 'is_down', False) and len(opp.hand_list) <= 2:
                        may_is_left = getattr(opp, 'may_is', 0)
                        if i == next_player_idx or may_is_left > 0:
                            return 5000
                return 2000
            return 0

            # 1. Grab the mathematically proven baseline structure value
        base_synergy = super()._calculate_synergy(target_card, hand_list, ctx, player_idx)

        # If the baseline sees this as deadwood or a toxic duplicate (<= 0), don't waste time
        if base_synergy <= 0:
            return int(base_synergy)

        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]
        has_sets = ctx._search_melds(player.private_hand, sets_needed=req_sets, runs_needed=0)[0]
        has_runs = ctx._search_melds(player.private_hand, sets_needed=0, runs_needed=req_runs)[0]

        needs_sets = (req_sets > 0) and not has_sets
        needs_runs = (req_runs > 0) and not has_runs

        suit = target_card.suit.value
        rank = target_card.rank.value

        final_synergy = float(base_synergy)
        is_dead = True

        # 2. Evaluate SETS Fertility
        if needs_sets:
            # Check current cluster size (including the target card)
            matches_in_hand = sum(1 for c in hand_list if c.rank == rank and c is not target_card)
            cluster_size = matches_in_hand + 1

            outs = 0
            for s in range(4):
                outs += self._get_card_availability(s, rank, ctx, player_idx)

            if cluster_size >= 3:
                is_dead = False
                final_synergy += 500  # Immunity: Completed Set!
            elif outs > 0:
                is_dead = False
                final_synergy += (outs * 5.0)

        # 3. Evaluate RUNS Fertility (Contiguous Edge Detection)
        if needs_runs:
            # Trace the tensor to find the actual physical boundaries
            left_edge = rank
            while left_edge > 0 and player.private_hand[suit, left_edge - 1] > 0:
                left_edge -= 1

            right_edge = rank
            while right_edge < 13 and player.private_hand[suit, right_edge + 1] > 0:
                right_edge += 1

            cluster_size = right_edge - left_edge + 1

            outs = 0
            if left_edge > 0:
                outs += self._get_card_availability(suit, left_edge - 1, ctx, player_idx)
            if right_edge < 13:
                outs += self._get_card_availability(suit, right_edge + 1, ctx, player_idx)

            if cluster_size >= 4:
                is_dead = False
                final_synergy += 500  # Immunity: Completed Run!
            elif outs > 0:
                is_dead = False
                final_synergy += (outs * 5.0)

        # 4. Risk-Adjusted Aversion (Fatality by Fertility Fix)
        # We offset the fertility bonus by the card's heavy penalty weight
        deadwood_val = self._get_deadwood_value(target_card, ctx)
        final_synergy -= (deadwood_val * 0.5)

        # 5. The True Veto: If it has 0 physical outs and isn't a completed meld, it's dead.
        if is_dead:
            return 0

        return int(final_synergy)

    def _is_useful_pickup(self, target_card, player_idx, ctx):
        is_useful = super()._is_useful_pickup(target_card, player_idx, ctx)
        if not is_useful:
            return False

        player = ctx.players[player_idx]
        expected_synergy = self._calculate_synergy(target_card, player.hand_list, ctx, player_idx)

        if expected_synergy <= 0:
            return False

        return True


class ProbabilisticAgent(HeuristicAgent):
    """
    A unified Expected Value (EV) engine that navigates imperfect information.
    It evaluates every decision by generating hypothetical future states,
    calculating the probability of achieving the round objective, and weighing
    the expected point reward against the risk of an opponent ending the round.
    """

    def __init__(self, random_seed=None, expected_horizon_3p=14.5, expected_horizon_4p=10.9):
        super().__init__(random_seed)
        self.expected_horizon_3p = expected_horizon_3p
        self.expected_horizon_4p = expected_horizon_4p

    def _get_penalty_weights(self, ctx):
        """
        Generates a cached (4, 14) tensor where each index holds the normalized
        penalty value of that specific card (e.g., 0.20 for Aces, 0.05 for Twos).

        Rationale: Matrix multiplication is significantly faster than looping.
        By dividing by 100 here, the engine can calculate the point value of
        an entire 14-card hand in a single C-optimized NumPy operation.
        """
        if not hasattr(self, '_cached_penalty_weights'):
            weights = np.zeros((4, 14), dtype=np.float32)
            # Divide by 100 here to save operations during the evaluation loop
            weights[:, 0] = ctx.config.points_ace / 100.0
            weights[:, 1:7] = ctx.config.points_two_to_seven / 100.0
            weights[:, 7:13] = ctx.config.points_eight_to_king / 100.0
            self._cached_penalty_weights = weights

        return self._cached_penalty_weights

    def _get_draw_probability(self, required, available, unknown_cards, horizon=15):
        """
        Calculates the cumulative hypergeometric probability of successfully
        drawing the `required` amount of a specific card over a given `horizon`.

        Rationale: Converts a singular draw chance (e.g., 2% chance on the next turn)
        into a realistic trajectory (e.g., 30% chance to find the card before the
        game ends). This prevents the bot from becoming mathematically paralyzed
        by the low odds of single draws.
        """
        if required == 0: return 1.0
        if available < required: return 0.0
        if unknown_cards <= 0: return 0.0

        # Chance to NOT draw a specific card in 1 draw
        p_miss_single = max(0.0, (unknown_cards - available) / unknown_cards)

        # Chance to NOT draw it over the entire horizon
        p_miss_all = p_miss_single ** horizon

        # Chance to draw AT LEAST 1 copy
        p_hit_at_least_one = 1.0 - p_miss_all

        # If we need multiple copies, we multiply the probabilities.
        return p_hit_at_least_one ** required

    def _get_expected_stock_value(self, ctx, hand_tensor, player_idx):
        """
        Calculates the exact average point value of a single card pulled blindly
        from the stock pile at this specific microsecond.

        Rationale: Used to price the "cost" of a May-I penalty card. If all
        Aces and Face cards are visible on the table, the stock is "safe" (~5 pts).
        If no Aces have been played, the stock is "radioactive" (~9+ pts).
        """
        available = self._get_available_tensor(ctx, hand_tensor, player_idx)
        weights = self._get_penalty_weights(ctx)

        total_unknown_cards = np.sum(available)
        if total_unknown_cards == 0:
            return 8.46  # Fallback to theoretical average if deck is empty

        # (Sum of all remaining points) / (Number of remaining cards)
        # Weights are already divided by 100, so we multiply by 100 to get points
        avg_points = (np.sum(available * weights) / total_unknown_cards) * 100.0
        return avg_points

    def _evaluate_hand_state(self, hand_tensor, ctx, player_idx):
        """
        THE MASTER EQUATION.
        Evaluates the literal Expected Value (EV) of a hand state in penalty points.

        Calculates two timelines:
        1. Timeline A (Imminent Threat): The EV if an opponent ends the game THIS turn.
           (Results in taking full Deadwood or Total Hand points).
        2. Timeline B (Future Trajectory): The EV if the game continues naturally.
           (Results in blended risk and claiming the Opponents' Penalty Points).

        Rationale: This equation balances aggressive drafting (Timeline B) against
        sudden-death survival (Timeline A), scaled dynamically by the physical
        threat level (`p_lose`) of the opponents.
        """
        is_down = ctx.players[player_idx].is_down

        if is_down:
            # If we are already down, our melds are on the table. We have won the
            # objective. Our entire physical hand is just deadwood and key cards.
            p_win = 1.0
            slotted_mask = np.zeros((4, 14), dtype=np.int8)
        else:
            req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]
            p_win, slotted_mask = self._find_best_seed_allocation(
                hand_tensor, req_sets, req_runs, ctx, player_idx
            )
            ace_max = np.maximum(slotted_mask[:, 0], slotted_mask[:, 13])
            slotted_mask[:, 0] = ace_max
            slotted_mask[:, 13] = ace_max

        penalty_weights = self._get_penalty_weights(ctx)
        total_hand_points = np.sum(hand_tensor * penalty_weights) * 100.0

        deadwood_tensor = np.clip(hand_tensor - slotted_mask, 0, None)

        for suit in range(4):
            for rank in range(14):
                if deadwood_tensor[suit, rank] > 0:
                    if np.any(ctx.table_sets[:, rank] > 0) or ctx._can_extend_run(suit, rank):
                        deadwood_tensor[suit, rank] = 0
                        # Ensure the Ace's alter-ego is also zeroed out!
                        if rank == 0: deadwood_tensor[suit, 13] = 0
                        if rank == 13: deadwood_tensor[suit, 0] = 0

        deadwood_points = np.sum(deadwood_tensor * penalty_weights) * 100.0

        available_tensor = self._get_available_tensor(ctx, hand_tensor, player_idx)
        unknown_deck_size = float(np.sum(available_tensor[:, 0:13]))
        unknown_deck_points = np.sum(available_tensor * penalty_weights) * 100.0
        avg_unknown_card_val = unknown_deck_points / max(1.0, unknown_deck_size)

        # ==========================================
        # 1. THE AVALANCHE THREAT
        # ==========================================
        p_lose = self._calculate_avalanche_threat(ctx, player_idx, available_tensor,
                                                  unknown_deck_size)

        # ==========================================
        # 2. THE REWARD CALCULATION
        # ==========================================
        opponents_expected_penalty = 0.0

        for i, opp in enumerate(ctx.players):
            if i != player_idx:
                opp_hand_size = len(opp.hand_list)

                known_held_tensor = np.clip(
                    ctx.player_pickup_counts[i] - ctx.player_discard_counts[i], 0, None)
                total_known_cards = float(np.sum(known_held_tensor[:, 0:13]))

                if total_known_cards > opp_hand_size:
                    opponents_expected_penalty += opp_hand_size * avg_unknown_card_val
                else:
                    known_points = np.sum(known_held_tensor * penalty_weights) * 100.0
                    num_unknown_cards = float(opp_hand_size) - total_known_cards
                    unknown_points = num_unknown_cards * avg_unknown_card_val
                    opponents_expected_penalty += (known_points + unknown_points)

        # ==========================================
        # 3. THE MASTER EQUATION
        # ==========================================
        current_physical_penalty = deadwood_points if is_down else total_hand_points
        future_blended_penalty = (p_win * deadwood_points) + ((1.0 - p_win) * total_hand_points)

        expected_penalty_risk = (p_lose * current_physical_penalty) + (
                    (1.0 - p_lose) * future_blended_penalty)

        # TOTAL REWARD scales perfectly with the total points in the opponents' hands
        expected_reward = (1.0 - p_lose) * (p_win * opponents_expected_penalty)

        expected_value = expected_reward - expected_penalty_risk

        return expected_value

    def _parse_all_valid_seeds(self, hand_tensor, req_sets, req_runs):
        """
        Scans the hand tensor using 4-card sliding windows and frequency counts
        to identify every possible partial meld (seed) that contributes to the round objective.

        Rationale: The bot must identify its foundational puzzle pieces before
        it can ask the probability engine how hard it will be to finish them.
        """
        seeds = []

        # ==========================================
        # 1. PARSE SET SEEDS
        # ==========================================
        if req_sets > 0:
            # Sum across all 4 suits to get the total count of each rank
            rank_counts = np.sum(hand_tensor, axis=0)

            # Iterate 1 to 13 (Ignoring 0 so we don't double-count the Ace)
            for rank in range(1, 14):
                held_count = rank_counts[rank]
                if held_count > 0:
                    distance = max(0, 3 - held_count)
                    seeds.append({
                        'type': 'set',
                        'distance': distance,
                        'target_rank': rank,
                        'held_count': held_count
                    })

        # ==========================================
        # 2. PARSE RUN SEEDS (NumPy Sliding Window)
        # ==========================================
        if req_runs > 0:
            for suit in range(4):
                # 11 valid windows: indices 0-3, 1-4, ... 10-13
                for start_idx in range(11):
                    # Slice the 4-card window directly from the tensor
                    window = hand_tensor[suit, start_idx:start_idx + 4]

                    # np.count_nonzero perfectly ignores duplicate cards of the same rank!
                    held_unique_ranks = np.count_nonzero(window)

                    if held_unique_ranks > 0:
                        distance = 4 - held_unique_ranks

                        # Find exactly which indices in the window are 0 (missing)
                        missing_offsets = np.where(window == 0)[0]
                        missing_ranks = [start_idx + offset for offset in missing_offsets]

                        seeds.append({
                            'type': 'run',
                            'distance': distance,
                            'suit': suit,
                            'target_window_start': start_idx,  # Maps to rank (0 = Ace Low)
                            'missing_ranks': missing_ranks,
                            'held_unique_ranks': held_unique_ranks
                        })

        return seeds

    def _get_available_tensor(self, ctx, hand_tensor, player_idx):
        """
        Subtracts globally visible cards, our hand, and cards legally known to be
        in opponents' hands from a master (2x) double deck.

        Rationale: Creates the absolute ground-truth "Outs" matrix. By tracking
        opponent pickups, the bot avoids hallucinating that a card is in the deck
        when it physically watched an opponent pick it up.
        """
        # 1. Start with the globally visible cards and our own hand
        known_tensor = (ctx.table_sets +
                        ctx.table_runs +
                        ctx.dead_cards +
                        hand_tensor).copy()

        # 2. Add the cards we KNOW opponents are holding
        for i in range(len(ctx.players)):
            if i != player_idx:
                # Calculate exactly what they've picked up minus what they've discarded
                known_held = np.clip(ctx.player_pickup_counts[i] - ctx.player_discard_counts[i], 0,
                                     None)
                known_tensor += known_held

        # 3. Double deck means max 2 of any specific card (Suit + Rank)
        # np.clip ensures we don't drop below 0 if matrix math gets weird
        return np.clip(2 - known_tensor, 0, 2)

    def _calculate_seed_probability(self, seed, ctx, hand_tensor, player_idx):
        """
        Evaluates a specific seed (e.g., a Run missing one card) and calculates
        the exact $P(win)$ of completing it.

        Rationale: Acts as the bridge between the structural parser (`_parse_all_valid_seeds`)
        and the statistical engine (`_get_draw_probability`), factoring in
        empirical time limits (the 3P vs 4P horizon).
        """
        if seed['distance'] == 0:
            return 1.0

        available_tensor = self._get_available_tensor(ctx, hand_tensor, player_idx)
        unknown_deck_size = float(np.sum(available_tensor[:, 0:13]))

        if len(ctx.players) == 3:
            horizon = self.expected_horizon_3p
        else:
            horizon = self.expected_horizon_4p

        if seed['type'] == 'set':
            target_rank = seed['target_rank']
            live_draws = float(np.sum(available_tensor[:, target_rank]))
            return self._get_draw_probability(seed['distance'], live_draws, unknown_deck_size,
                                              horizon=horizon)

        elif seed['type'] == 'run':
            suit = seed['suit']
            p_run = 1.0

            for missing_rank in seed['missing_ranks']:
                specific_draws = float(available_tensor[suit, missing_rank])
                p_hole = self._get_draw_probability(1, specific_draws, unknown_deck_size,
                                                    horizon=horizon)
                p_run *= p_hole

            return p_run

        return 0.0

    def _find_best_seed_allocation(self, hand_tensor, req_sets, req_runs, ctx, player_idx):
        """
        Dynamic Greedy Algorithm.
        Achieves optimal seed allocation without the exponential compute overhead of a DFS.
        """
        p_win = 1.0
        slotted_mask = np.zeros((4, 14), dtype=np.int8)
        working_hand = hand_tensor.copy()

        sets_found = 0
        runs_found = 0

        # Loop strictly bounds to the maximum requirements (e.g., 3 iterations)
        while sets_found < req_sets or runs_found < req_runs:

            # 1. Parse valid seeds strictly from the REMAINING working hand
            current_seeds = self._parse_all_valid_seeds(working_hand, req_sets - sets_found,
                                                        req_runs - runs_found)

            if not current_seeds:
                break  # The hand is physically empty or has no valid seeds left

            # 2. Calculate accurate EV based on the remaining cards
            for seed in current_seeds:
                seed['ev'] = self._calculate_seed_probability(seed, ctx, working_hand, player_idx)

            # 3. Greedily pick the absolute highest probability seed
            best_seed = max(current_seeds, key=lambda x: x['ev'])

            # 4. Physically consume the cards from the working hand
            if best_seed['type'] == 'set':
                sets_found += 1
                rank = best_seed['target_rank']
                # Consume all available cards of this rank to avoid leaving stragglers
                for s in range(4):
                    while working_hand[s, rank] > 0:
                        ctx._sync_ace(working_hand, s, rank, increment=False)
                        slotted_mask[s, rank] += 1

            elif best_seed['type'] == 'run':
                runs_found += 1
                suit = best_seed['suit']
                start = best_seed['target_window_start']
                for r in range(start, start + 4):
                    if working_hand[suit, r] > 0:
                        ctx._sync_ace(working_hand, suit, r, increment=False)
                        slotted_mask[suit, r] += 1

            # Accumulate the total probability
            p_win *= best_seed['ev']

        # 5. Apply Additive Smoothing for unfulfilled requirements
        missing_sets = req_sets - sets_found
        missing_runs = req_runs - runs_found

        p_win *= (0.01 ** missing_sets)
        p_win *= (0.01 ** missing_runs)

        return p_win, slotted_mask

    def _calculate_avalanche_threat(self, ctx, player_idx, available_tensor, unknown_deck_size):
        """
        Calculates the 1-turn sudden-death probability (p_lose) of an opponent going out.
        """
        physical_table_outs = 0.0
        for s in range(4):
            for r in range(1, 14):  # Exclude 0 to avoid double-counting Aces
                if np.any(ctx.table_sets[:, r] > 0) or ctx._can_extend_run(s, r):
                    physical_table_outs += float(available_tensor[s, r])

        p_draw_out_single = physical_table_outs / max(1.0, unknown_deck_size)
        p_safe = 1.0

        for i, opp in enumerate(ctx.players):
            if i != player_idx:
                opp_hand_size = len(opp.hand_list)

                if getattr(opp, 'is_down', False):
                    if opp_hand_size == 0:
                        opp_threat = 1.0
                    else:
                        p_hold_avalanche = (float(opp_hand_size) / max(1.0, unknown_deck_size)) ** (
                                    opp_hand_size - 1)
                        opp_threat = p_draw_out_single * p_hold_avalanche
                else:
                    opp_threat = 0.02

                p_safe *= (1.0 - opp_threat)

        p_lose = 1.0 - p_safe
        return max(0.01, min(0.99, p_lose))

    def select_action(self, state_id, ctx, player_idx, action_mask):
        """
        Overrides the rule-based heuristics with a 1-Ply Stochastic Lookahead.

        - Discard Phase: Simulates the EV of the hand *after* dropping every legally discardable card.
        - Pickup Phase: Simulates the EV of a Full Turn (Pickup + Best Discard).
        - May-I Phase: Compares baseline EV against the EV of expanding the hand,
          minus the dynamic expected risk of the blind stock card.
        - Go Down Phase: Deterministically locks in melds to secure EV safety.

        Rationale: Converts static rules into dynamic, localized math. The bot
        "feels" the weight of every action by looking exactly one turn into the future.
        """
        player = ctx.players[player_idx]
        hand_tensor = player.private_hand
        discard_top = ctx.discard_pile[-1] if ctx.discard_pile else None

        # ==========================================
        # 1. PREDATORY DISCARD (1-Ply Lookahead)
        # ==========================================
        if state_id == 'discard_phase':
            best_discard_idx = -1
            best_future_ev = -float('inf')

            # Simulate the future for every legally discardable card
            for card in player.hand_list:
                action_idx = self._get_discard_action_idx(card)
                if action_idx < len(action_mask) and action_mask[action_idx]:

                    # Create the hypothetical future (Hand MINUS this card)
                    hypo_tensor = hand_tensor.copy()
                    suit, rank = int(card.suit), int(card.rank)
                    ctx._sync_ace(hypo_tensor, suit, rank, increment=False)

                    # Ask the EV Engine how good this resulting hand is
                    future_ev = self._evaluate_hand_state(hypo_tensor, ctx, player_idx)

                    if future_ev > best_future_ev:
                        best_future_ev = future_ev
                        best_discard_idx = action_idx

            if best_discard_idx != -1:
                return best_discard_idx

        # ==========================================
        # 2. PICKUP DECISION (Full Turn Lookahead)
        # ==========================================
        elif state_id == 'pickup_decision' and discard_top:
            if action_mask[1]:
                baseline_ev = self._evaluate_hand_state(hand_tensor, ctx, player_idx)

                # Step 1: Simulate the Pickup
                hypo_tensor = hand_tensor.copy()
                suit, rank = int(discard_top.suit), int(discard_top.rank)
                ctx._sync_ace(hypo_tensor, suit, rank, increment=True)

                # Step 2: Simulate the Mandatory Discard
                best_post_discard_ev = -float('inf')

                # Scan the hypothetical hand and simulate dropping every available card
                for s in range(4):
                    for r in range(1, 14):
                        if hypo_tensor[s, r] > 0:
                            post_discard_tensor = hypo_tensor.copy()
                            ctx._sync_ace(post_discard_tensor, s, r, increment=False)

                            ev = self._evaluate_hand_state(post_discard_tensor, ctx, player_idx)
                            if ev > best_post_discard_ev:
                                best_post_discard_ev = ev

                # THE UPDATE: We are now dealing in literal Penalty Points.
                # Only pick up the discard if it statistically saves us at least 1 overall point.
                if best_post_discard_ev > baseline_ev + 1.0:
                    return 1

            if action_mask[0]:
                return 0

        # ==========================================
        # 3. MAY-I DECISION (The Expansion Valve)
        # ==========================================
        elif state_id == 'may_i_decision' and discard_top and action_mask[2]:

            if hasattr(ctx, 'may_i_target_idx') and ctx.may_i_target_idx is not None:
                player_idx = ctx.may_i_target_idx

            hand_tensor = ctx.players[player_idx].private_hand
            baseline_ev = self._evaluate_hand_state(hand_tensor, ctx, player_idx)

            hypo_tensor = hand_tensor.copy()
            suit, rank = int(discard_top.suit), int(discard_top.rank)
            ctx._sync_ace(hypo_tensor, suit, rank, increment=True)

            pickup_ev = self._evaluate_hand_state(hypo_tensor, ctx, player_idx)
            stock_pts = self._get_expected_stock_value(ctx, hand_tensor, player_idx)

            # --- THE AVALANCHE THREAT ---
            available_est = self._get_available_tensor(ctx, hand_tensor, player_idx)
            unknown_size_est = float(np.sum(available_est[:, 0:13]))

            p_lose_est = self._calculate_avalanche_threat(ctx, player_idx, available_est,
                                                          unknown_size_est)

            ev_cost_of_blind_card = stock_pts * p_lose_est

            if (pickup_ev - ev_cost_of_blind_card) > baseline_ev + 2.0:
                return 2

            if action_mask[3]:
                return 3

        # ==========================================
        # 4. GO DOWN DECISION (The Tactical Sandbag)
        # ==========================================
        elif state_id == 'go_down_decision':
            if action_mask[4] and action_mask[5]:

                # Option A: Wait (Sandbag). Our EV relies on keeping the table locked.
                wait_ev = self._evaluate_hand_state(hand_tensor, ctx, player_idx)

                # Option B: Go Down. We simulate physically placing the melds.
                hypo_ctx = ctx.clone()
                hypo_ctx.auto_place_cards(player_idx)
                hypo_hand_tensor = hypo_ctx.players[player_idx].private_hand

                # Re-evaluate. The engine will notice our deadwood dropped, BUT it will also
                # detect the new 'sparks' on the table and spike the opponents' Avalanche Threat!
                go_down_ev = self._evaluate_hand_state(hypo_hand_tensor, hypo_ctx, player_idx)

                # If unlocking the table destroys our reward probability more than it
                # saves us in penalty risk, the bot will naturally choose to sandbag!
                if go_down_ev > wait_ev:
                    return 4
                else:
                    return 5

            elif action_mask[4]:
                return 4
            return 5

        # ==========================================
        # 5. FALLBACK (Table Play & Edge Cases)
        # ==========================================
        # Let the base HeuristicAgent handle playing on table melds,
        # as those are strictly rule-bound actions that don't require EV simulation.
        return super().select_action(state_id, ctx, player_idx, action_mask)


class BayesianAgent(ProbabilisticAgent):
    """
        An advanced Player-Profiling engine that evolves the ProbabilisticAgent
        by replacing uniform deck assumptions with Bayesian Inference.

        While the ProbabilisticAgent treats all unknown cards as a randomized
        uniform distribution, the BayesianAgent maintains a persistent
        'Observation Tensor' to track the specific 'scents' of opponents' hands.
        By analyzing pickup and discard history, it builds a non-uniform
        probability heatmap of the hidden state.

        Key Objectives:
        1. Cures the 'Average Card Fallacy': Recognizes that the deck is not
           randomly distributed but filtered by the heuristic biases of
           opponents.
        2. Identifies 'Toxic Seeds': Detects when an opponent is building a
           specific suit or rank, allowing the bot to pivot its own strategy
           or intentionally hoard 'sparks' to starve the opponent.
        3. Optimizes 4-Player Horizons: Mathematically slows down the
           'Heuristic Swarm' by denying high-probability connections,
           forcing the game into a turn-count window where EV-logic can
           reliably dominate.

        Implementation Details:
        - Inherits the 1-Ply Stochastic Lookahead and Master EV Equation
          from ProbabilisticAgent.
        - Overrides `_get_available_tensor` to return a weighted probability
          matrix (Heatmap) instead of a flat frequency count.
        - Introduces a persistent 'Observation Memory' to update player
          suit/rank probabilities after every turn.
    """
    pass


class ONNXAgent(Agent):
    """
    An adapter that allows evaluate_arena.py to seamlessly play an ONNX neural network.
    Uses lazy-loading to survive being pickled across multiprocessing boundaries.
    """

    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path

    def select_action(self, state_id, ctx, player_idx, action_mask):
        global _global_onnx_sessions

        # 1. Lazy-Load the Model (Happens exactly ONCE per CPU core)
        if self.onnx_path not in _global_onnx_sessions:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            _global_onnx_sessions[self.onnx_path] = ort.InferenceSession(self.onnx_path,
                                                                         sess_options)

        session = _global_onnx_sessions[self.onnx_path]

        # 2. Extract and format the tensors
        nn_inputs = ctx.get_input_tensor(player_idx, state_id)
        spatial_ort = np.expand_dims(nn_inputs['spatial'], axis=0).astype(np.float32)
        scalar_ort = np.expand_dims(nn_inputs['scalar'], axis=0).astype(np.float32)
        mask_ort = np.expand_dims(action_mask, axis=0).astype(np.bool_)

        ort_inputs = {
            'spatial': spatial_ort,
            'scalar': scalar_ort,
            'mask': mask_ort
        }

        # 3. Inference
        outputs = session.run(None, ort_inputs)
        logits = outputs[0][0]

        # 4. Mask and Select
        logits[~action_mask] = -1e9
        return int(np.argmax(logits))