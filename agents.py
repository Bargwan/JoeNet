import random
import numpy as np


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
    def __init__(self, random_seed=None):
        """
        Initializes the agent with an isolated random number generator
        to ensure reproducible sandbagging during TDD.
        """
        self.rng = random.Random(random_seed)

    def _calculate_go_down_probability(self, ctx, player_idx):
        """
        Calculates a dynamic 0.0 to 1.0 probability of locking in melds.
        Takes the maximum threat level between baseline, turn counts, and opponent proximity to winning.
        """
        # Baseline probability to go down (50% sandbag rate)
        prob = 0.5

        # 1. Turn Pressure (Scales from 0.0 to 1.0)
        # Using the new 'current_circuit' which represents full board loops
        max_turns = getattr(ctx.config, 'max_turns', 15)
        turn_pressure = ctx.current_circuit / max_turns
        prob = max(prob, turn_pressure)

        # 2. Opponent Threat Pressure
        for i, p in enumerate(ctx.players):
            if i != player_idx and getattr(p, 'is_down', False):
                # Threat scales inversely with their hand size.
                # e.g., 4 cards = 0.6 probability. 1 card = 0.9 probability.
                cards_left = len(p.hand_list)
                opp_threat = 1.0 - (cards_left * 0.1)
                prob = max(prob, opp_threat)

        # Strictly bound between 0.0 and 1.0
        return min(1.0, prob)

    def select_action(self, state_id, ctx, player_idx, action_mask):
        """
        Uses objective-aware heuristics to make decisions for Pickup, May-I,
        Go Down, Table Play, and Discard phases.
        """
        player = ctx.players[player_idx] if ctx and ctx.players else None
        hand = player.hand_list if player else []
        discard_top = ctx.discard_pile[-1] if ctx and ctx.discard_pile else None

        # --- Phase: PICKUP DECISION ---
        if state_id == 'pickup_decision' and discard_top:
            is_down = getattr(player, 'is_down', False)

            if is_down:
                if action_mask[1] and self._is_playable_on_table(discard_top, ctx):
                    return 1
                elif action_mask[0]:
                    return 0
            else:
                if action_mask[1] and self._completes_objective(discard_top, player_idx, ctx):
                    return 1

            if action_mask[0]:
                return 0
            elif action_mask[1]:
                return 1

        # --- Phase: MAY-I DECISION ---
        elif state_id == 'may_i_decision' and discard_top:
            if action_mask[2] and self._completes_objective(discard_top, player_idx, ctx):
                return 2
            elif action_mask[3]:
                return 3

        # --- Phase: GO DOWN DECISION (Step 5.1: Dynamic Panic) ---
        elif state_id == 'go_down_decision':
            if action_mask[4] and ctx.check_hand_objective(player_idx):
                if action_mask[5]:
                    go_down_prob = self._calculate_go_down_probability(ctx, player_idx)

                    # Roll the dice!
                    if self.rng.random() <= go_down_prob:
                        return 4  # GO_DOWN (Panic / Play Safe)
                    else:
                        return 5  # WAIT (Sandbag / Risk It)
                return 4
            elif action_mask[5]:
                return 5

        # --- Phase: TABLE PLAY DECISION ---
        elif state_id == 'table_play_phase':
            valid_actions = np.where(action_mask)[0]
            # Greedily play any valid card (actions 6 through 57)
            card_plays = [a for a in valid_actions if a >= 6]
            if card_plays:
                return card_plays[0]
            # If no cards can be played, Wait/Pass (usually index 5 or 3)
            elif 5 in valid_actions:
                return 5
            elif 3 in valid_actions:
                return 3

        # --- Phase: DISCARD DECISION ---
        elif state_id == 'discard_phase' and hand:
            best_discard_idx = -1
            lowest_synergy = float('inf')

            is_down = getattr(player, 'is_down', False)

            # OPTIMIZATION 1: Calculate the baseline hand progress ONCE per phase
            req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]
            base_progress = self._get_objective_progress(player.private_hand, req_sets,
                                                         req_runs, ctx)

            for card in hand:
                action_idx = self._get_discard_action_idx(card)

                if action_idx < len(action_mask) and action_mask[action_idx]:
                    synergy = self._calculate_synergy(card, hand, ctx)

                    # OPTIMIZATION 2: Fast-fail! Isolated cards (synergy 0) physically CANNOT break an objective.
                    if synergy > 0:
                        if self._breaks_objective(card, player_idx, ctx, base_progress):
                            synergy += 5000

                    if is_down and self._is_playable_on_table(card, ctx):
                        synergy += 1000

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
        else:
            return ctx.config.points_two_to_seven

    def _calculate_synergy(self, target_card, hand_list, ctx):
        synergy = 0
        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]

        for other in hand_list:
            if other is target_card:
                continue

            if req_sets > 0 and other.rank == target_card.rank:
                synergy += 9

            elif req_runs > 0 and other.suit == target_card.suit:
                rank_diff = abs(other.rank.value - target_card.rank.value)
                if 0 < rank_diff <= 2:
                    synergy += 10

        return synergy

    def _is_playable_on_table(self, card, ctx):
        suit, rank = int(card.suit), int(card.rank)
        if np.any(ctx.table_sets[:, rank] > 0):
            return True
        if ctx._can_extend_run(suit, rank):
            return True
        return False

    def _get_discard_action_idx(self, card):
        return (card.suit.value * 13) + card.rank.value + 6