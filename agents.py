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
    The unified 'Ruthless Closer' baseline bot.
    Aggressively drafts partial melds, hoards key cards, and expands hand capacity late game.
    """

    def __init__(self, random_seed=None):
        self.rng = random.Random(random_seed)

    def select_action(self, state_id, ctx, player_idx, action_mask):
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
        rank_val = int(card.rank)
        if rank_val == 0 or rank_val == 13:
            return ctx.config.points_ace
        elif 7 <= rank_val <= 12:
            return ctx.config.points_eight_to_king
        return ctx.config.points_two_to_seven

    def _is_playable_on_table(self, card, ctx):
        suit, rank = int(card.suit), int(card.rank)
        if np.any(ctx.table_sets[:, rank] > 0):
            return True
        if ctx._can_extend_run(suit, rank):
            return True
        return False

    def _get_discard_action_idx(self, card):
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
    # Static (4, 14) tensor mapping the penalty values of every card.
    # Divided by 100 directly here so we don't have to do it during evaluation.
    # Note: Index 13 (High Ace) is 0 to prevent double-counting with Index 0 (Low Ace).

    def _get_penalty_weights(self, ctx):
        """Generates and caches the penalty weight tensor from the dynamic config."""
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
        Calculates the true probability (0.0 to 1.0) of drawing 'required' copies
        of a card given 'available' copies in an 'unknown_cards' deck.
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
        """Calculates the literal average point value of all unknown cards."""
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
        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]
        all_seeds = self._parse_all_valid_seeds(hand_tensor, req_sets, req_runs)

        # 1. Get True Probabilities
        for seed in all_seeds:
            seed['ev'] = self._calculate_seed_probability(seed, ctx, hand_tensor, player_idx)

        all_seeds.sort(key=lambda x: x['ev'], reverse=True)

        slotted_sets_p = []
        slotted_runs_p = []
        slotted_mask = np.zeros((4, 14), dtype=np.int8)
        working_hand = hand_tensor.copy()

        # 2. Strict Consumption Slotting (Unchanged)
        for seed in all_seeds:
            if seed['type'] == 'set' and len(slotted_sets_p) < req_sets:
                target_rank = seed['target_rank']
                actual_held = np.sum(working_hand[:, target_rank])
                if actual_held > 0:
                    slotted_sets_p.append(seed['ev'])
                    cards_to_consume = min(3, int(actual_held))
                    consumed = 0
                    for s in range(4):
                        while working_hand[s, target_rank] > 0 and consumed < cards_to_consume:
                            working_hand[s, target_rank] -= 1
                            slotted_mask[s, target_rank] += 1
                            consumed += 1

            elif seed['type'] == 'run' and len(slotted_runs_p) < req_runs:
                suit = seed['suit']
                start = seed['target_window_start']
                window = working_hand[suit, start:start + 4]
                actual_held = np.count_nonzero(window)
                if actual_held > 0:
                    slotted_runs_p.append(seed['ev'])
                    for i in range(4):
                        if working_hand[suit, start + i] > 0:
                            working_hand[suit, start + i] -= 1
                            slotted_mask[suit, start + i] += 1

        # 3. Multiplicative Win Probability (0.0 to 1.0)
        p_sets = np.prod(slotted_sets_p) if slotted_sets_p else 0.01
        p_runs = np.prod(slotted_runs_p) if slotted_runs_p else 0.01
        p_win = p_sets * p_runs

        # If an Ace was consumed at either Index 0 or 13, ensure both indices reflect it
        ace_max = np.maximum(slotted_mask[:, 0], slotted_mask[:, 13])
        slotted_mask[:, 0] = ace_max
        slotted_mask[:, 13] = ace_max

        # 4. REAL POINT CALCULATIONS
        penalty_weights = self._get_penalty_weights(ctx)

        # Calculate literal point values of our hand
        total_hand_points = np.sum(hand_tensor * penalty_weights) * 100.0

        deadwood_tensor = np.clip(hand_tensor - slotted_mask, 0, None)
        deadwood_points = np.sum(deadwood_tensor * penalty_weights) * 100.0

        # 5. THE TRUE PROBABILISTIC VICTORY REWARD
        available_tensor = self._get_available_tensor(ctx, hand_tensor, player_idx)
        unknown_deck_size = float(np.sum(available_tensor))
        unknown_deck_points = np.sum(available_tensor * penalty_weights) * 100.0

        avg_unknown_card_val = unknown_deck_points / max(1.0, unknown_deck_size)

        opponents_expected_penalty = 0.0
        for i, opp in enumerate(ctx.players):
            if i != player_idx:
                # Calculate known cards physically tracked into their hand
                # np.clip prevents negative values if they discard a card they were dealt initially
                known_held_tensor = np.clip(
                    ctx.player_pickup_counts[i] - ctx.player_discard_counts[i], 0, None)

                # Count how many public cards we mathematically think they are holding
                total_known_cards = float(np.sum(known_held_tensor))

                # Edge case safeguard: If they went "down", they played cards to the table.
                # If our known count exceeds their physical hand size, fallback to average
                # to prevent overestimating their penalty damage.
                if total_known_cards > len(opp.hand_list):
                    opponents_expected_penalty += len(opp.hand_list) * avg_unknown_card_val
                else:
                    # Calculate exact points for the cards we KNOW they hold
                    known_points = np.sum(known_held_tensor * penalty_weights) * 100.0

                    # Calculate average points for the rest of their hidden hand
                    num_unknown_cards = float(len(opp.hand_list)) - total_known_cards
                    unknown_points = num_unknown_cards * avg_unknown_card_val

                    opponents_expected_penalty += (known_points + unknown_points)

        # 6. THE RELATIVE MASTER EQUATION
        # True Expected Value = (Points we expect to inflict) - (Points we expect to take)
        expected_value = (p_win * opponents_expected_penalty) - (
                    (1.0 - p_win) * total_hand_points) - (p_win * deadwood_points)

        return expected_value

    def _parse_all_valid_seeds(self, hand_tensor, req_sets, req_runs):
        """
        Scans the (4, 14) hand tensor to find valid structural seeds.
        Returns a list of dictionaries containing seed metrics.
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
        Creates a (4, 14) matrix representing all cards physically left to draw.
        Integrates public opponent pickups to prevent hallucinating dead outs.
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
        """Returns a strict 0.0 to 1.0 probability of completing this seed."""
        if seed['distance'] == 0:
            return 1.0

            # --- PASS player_idx HERE ---
        available_tensor = self._get_available_tensor(ctx, hand_tensor, player_idx)
        unknown_deck_size = float(np.sum(available_tensor[:, 0:13]))

        if seed['type'] == 'set':
            target_rank = seed['target_rank']
            live_draws = float(np.sum(available_tensor[:, target_rank]))
            return self._get_draw_probability(seed['distance'], live_draws, unknown_deck_size)

        elif seed['type'] == 'run':
            suit = seed['suit']
            p_run = 1.0

            for missing_rank in seed['missing_ranks']:
                specific_draws = float(available_tensor[suit, missing_rank])
                p_hole = self._get_draw_probability(1, specific_draws, unknown_deck_size)
                p_run *= p_hole

            return p_run

        return 0.0

    def select_action(self, state_id, ctx, player_idx, action_mask):
        """
        Overrides the Heuristic Action Selection with a 1-Ply Stochastic Lookahead.
        Now uses True Probability & Expected Penalty Points.
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
            baseline_ev = self._evaluate_hand_state(hand_tensor, ctx, player_idx)

            # Step 1: Simulate the pickup of the targeted discard
            hypo_tensor = hand_tensor.copy()
            suit, rank = int(discard_top.suit), int(discard_top.rank)
            ctx._sync_ace(hypo_tensor, suit, rank, increment=True)

            # Step 2: Ask the EV engine how much better the hand becomes with that card
            # (Note: May-I doesn't force a discard yet, so we don't simulate a discard here)
            pickup_ev = self._evaluate_hand_state(hypo_tensor, ctx, player_idx)

            # Step 3: DYNAMIC RISK ASSESSMENT
            # Calculate the literal expected debt of the random penalty card from the stock
            # Inside the 'may_i_decision' block in select_action:
            expected_penalty_damage = self._get_expected_stock_value(ctx, hand_tensor, player_idx)

            # Step 4: Decision Logic
            # Only May-I if the hand improvement pays off the literal
            # expected debt of the random card + a 5-point margin for round-end risk.
            if (pickup_ev - expected_penalty_damage) > baseline_ev + 5.0:
                return 2

            if action_mask[3]:
                return 3

        # ==========================================
        # 4. FALLBACK (Going Down & Table Play)
        # ==========================================
        # Let the base HeuristicAgent handle going down and playing on table melds,
        # as those are strictly rule-bound actions that don't require EV simulation.
        return super().select_action(state_id, ctx, player_idx, action_mask)


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