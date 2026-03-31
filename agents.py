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


class LegacyHeuristicAgent(Agent):
    """
    The strict baseline bot. Only picks up discards if they perfectly
    complete the round objective or are playable on the table.
    """
    def __init__(self, random_seed=None):
        self.rng = random.Random(random_seed)

    def _calculate_go_down_probability(self, ctx, player_idx):
        prob = 0.8  # (Or 0.8 if you kept the aggressive dial from earlier)

        # --- Dynamic Round-Based Patience ---
        # Base expectation is 8 turns for Round 1 (Index 0),
        # scaling up to 14 turns for Round 7 (Index 6).
        patience_threshold = 8.0 + ctx.current_round_idx
        turn_pressure = ctx.current_circuit / patience_threshold
        prob = max(prob, turn_pressure)

        # (Opponent threat logic remains unchanged)
        for i, p in enumerate(ctx.players):
            if i != player_idx and getattr(p, 'is_down', False):
                cards_left = len(p.hand_list)
                opp_threat = 1.0 - (cards_left * 0.1)
                prob = max(prob, opp_threat)

        return min(1.0, prob)

    def select_action(self, state_id, ctx, player_idx, action_mask):
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
                # LEGACY LOGIC: Strict objective completion only
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

        # --- Phase: GO DOWN DECISION ---
        elif state_id == 'go_down_decision':
            if action_mask[4] and ctx.check_hand_objective(player_idx):
                if action_mask[5]:
                    go_down_prob = self._calculate_go_down_probability(ctx, player_idx)
                    if self.rng.random() <= go_down_prob:
                        return 4  # GO_DOWN
                    else:
                        return 5  # WAIT
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
            best_discard_idx = -1
            lowest_synergy = float('inf')
            is_down = getattr(player, 'is_down', False)

            req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]
            base_progress = self._get_objective_progress(player.private_hand, req_sets, req_runs, ctx)

            for card in hand:
                action_idx = self._get_discard_action_idx(card)

                if action_idx < len(action_mask) and action_mask[action_idx]:
                    synergy = self._calculate_synergy(card, hand, ctx)

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


class HeuristicAgent(LegacyHeuristicAgent):
    """
    The upgraded Option B bot. Inherits all logic from the Legacy bot,
    but overrides the pickup phase to aggressively grab partial melds and 1-gaps.
    """
    def select_action(self, state_id, ctx, player_idx, action_mask):
        # We only want to intercept the pickup decision.
        # All other decisions fall back to the Legacy logic.
        if state_id == 'pickup_decision':
            player = ctx.players[player_idx] if ctx and ctx.players else None
            hand = player.hand_list if player else []
            discard_top = ctx.discard_pile[-1] if ctx and ctx.discard_pile else None

            if discard_top:
                is_down = getattr(player, 'is_down', False)

                if is_down:
                    if action_mask[1] and self._is_playable_on_table(discard_top, ctx):
                        return 1
                    elif action_mask[0]:
                        return 0
                else:
                    # NEW LOGIC (Option B): Accept cards that complete objectives OR extend partials/gaps
                    if action_mask[1]:
                        if self._completes_objective(discard_top, player_idx, ctx):
                            return 1
                        elif self._is_useful_pickup(discard_top, player_idx,
                                                    ctx):  # <-- New signature
                            return 1

                # Fallback (usually draw from stock)
                if action_mask[0]:
                    return 0
                elif action_mask[1]:
                    return 1

        # If it's not a pickup decision, use the exact same logic as the Legacy bot
        return super().select_action(state_id, ctx, player_idx, action_mask)

    def _is_useful_pickup(self, target_card, player_idx, ctx):
        """
        Option B (Fuzzy/Gapped): Checks if a card provides immediate
        synergy to the current hand based on the round's objective.
        """
        hand_list = ctx.players[player_idx].hand_list
        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]

        for card in hand_list:
            # Set Synergy: Needs to match rank
            if req_sets > 0 and card.rank == target_card.rank:
                return True

            # Run Synergy: Needs to be same suit and within a rank distance of 2 (allows 1-gaps)
            if req_runs > 0 and card.suit == target_card.suit:
                rank_diff = abs(card.rank.value - target_card.rank.value)
                if 0 < rank_diff <= 2:
                    return True

        return False

class OptionAAgent(HeuristicAgent):
    """
    The Strict Partial Bot.
    Accepts contiguous partials (rank distance == 1). Strictly ignores 1-gaps.
    """
    def _is_useful_pickup(self, target_card, player_idx, ctx):
        hand_list = ctx.players[player_idx].hand_list
        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]

        for card in hand_list:
            if req_sets > 0 and card.rank == target_card.rank:
                return True

            if req_runs > 0 and card.suit == target_card.suit:
                rank_diff = abs(card.rank.value - target_card.rank.value)
                if rank_diff == 1:  # <--- OPTION A: Strict adjacency only!
                    return True

        return False


class KeyCardAwareHeuristicAgent(HeuristicAgent):
    """
    The 'Ruthless Closer' Bot.
    Identifies instant-win scenarios and protects cards that can be
    played on the table (Key Cards) even after the objective is met.
    """

    def _is_useful_pickup(self, target_card, player_idx, ctx):
        """
        Dynamic Run-Biased Pickup:
        Evaluates the outstanding needs of the hand and strictly prioritizes sequences
        if Runs are still missing, while ignoring synergies for already-completed objectives.
        """
        player = ctx.players[player_idx]
        hand = player.hand_list
        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]

        # 1. Pulse Check: What do we ACTUALLY still need?
        # We test the hand against the set and run requirements independently.
        has_sets = ctx._search_melds(player.private_hand, sets_needed=req_sets, runs_needed=0)[0]
        has_runs = ctx._search_melds(player.private_hand, sets_needed=0, runs_needed=req_runs)[0]

        needs_sets = (req_sets > 0) and not has_sets
        needs_runs = (req_runs > 0) and not has_runs

        if needs_runs and not needs_sets:
            for card in hand:
                if card.suit == target_card.suit and card.rank == target_card.rank:
                    return False  # Refuse to pick up duplicate from the discard for runs

        run_synergy_found = False
        set_synergy_found = False

        # 2. Scan the hand only for the synergies we actually need
        for card in hand:
            if needs_runs and card.suit == target_card.suit:
                if 0 < abs(card.rank.value - target_card.rank.value) <= 2:
                    run_synergy_found = True

            if needs_sets and card.rank == target_card.rank:
                set_synergy_found = True

        # 3. Dynamic Prioritization
        if needs_runs and needs_sets:
            # We need both. Are runs the heavier requirement? Prioritize them.
            if req_runs >= req_sets:
                return run_synergy_found
            return run_synergy_found or set_synergy_found

        elif needs_runs:
            # We only need runs. Completely blind the bot to pairs.
            return run_synergy_found

        elif needs_sets:
            # We only need sets. Completely blind the bot to sequences.
            return set_synergy_found

        return False

    def select_action(self, state_id, ctx, player_idx, action_mask):
        # --- PHASE: PICKUP DECISION (Key-Card Hoarding) ---
        if state_id == 'pickup_decision':
            player = ctx.players[player_idx]
            discard_top = ctx.discard_pile[-1] if ctx.discard_pile else None

            if discard_top and action_mask[1]:
                is_down = getattr(player, 'is_down', False)

                # THE FIX: If we have the objective but aren't down yet,
                # aggressively snatch any cards we know we can play on the table later!
                if not is_down and ctx.check_hand_objective(player_idx):
                    if self._is_playable_on_table(discard_top, ctx):
                        return 1  # PICK_DISCARD

            # If not a Key Card hoarding scenario, fall through to standard pickup logic
            # (which handles completing objectives and picking up partials).

        # --- PHASE: MAY-I DECISION (Strategic Capacity Expansion) ---
        elif state_id == 'may_i_decision':
            discard_top = ctx.discard_pile[-1] if ctx.discard_pile else None

            if discard_top and action_mask[2]:
                player = ctx.players[player_idx]

                # 1. Legacy Behavior: Does it instantly advance a meld progress?
                if self._completes_objective(discard_top, player_idx, ctx):
                    return 2  # MAY_I

                # 2. THE FIX: Hand Expansion for Late Rounds
                # In Rounds 4-7 (index 3+), objectives require 10-12 cards.
                # If we haven't May-I'd yet (hand size is still baseline 11)...
                if ctx.current_round_idx >= 3 and len(player.hand_list) <= 11:
                    # Use our Pulse Check! If it's a valuable sequence piece, take the penalty!
                    if self._is_useful_pickup(discard_top, player_idx, ctx):
                        return 2  # MAY_I

            # Otherwise, decline
            if action_mask[3]:
                return 3

        # --- PHASE: GO DOWN DECISION ---
        elif state_id == 'go_down_decision':
            if action_mask[4] and ctx.check_hand_objective(player_idx):

                # --- THE MISSING FIX: Relax to a 'Pragmatic Win' ---
                # Change this from <= 1 to <= 5
                if self._calculate_projected_deadwood(ctx, player_idx) <= 5:
                    return 4  # GO_DOWN

                # Otherwise, use the standard patience/pressure logic
                if action_mask[5]:
                    go_down_prob = self._calculate_go_down_probability(ctx, player_idx)
                    return 4 if self.rng.random() <= go_down_prob else 5
                return 4

        # --- PHASE: DISCARD DECISION (Key-Card Awareness) ---
        elif state_id == 'discard_phase':
            player = ctx.players[player_idx]
            hand = player.hand_list
            is_down = getattr(player, 'is_down', False)
            req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]
            base_progress = self._get_objective_progress(player.private_hand, req_sets,
                                                         req_runs, ctx)

            best_discard_idx = -1
            lowest_synergy = float('inf')

            # --- THE FIX: Detect Gridlock ---
            # If we've been circling the drain for 15 turns, panic.
            is_stuck = ctx.current_circuit > 20

            for card in hand:
                action_idx = self._get_discard_action_idx(card)
                if action_idx < len(action_mask) and action_mask[action_idx]:

                    synergy = self._calculate_synergy_smart(card, hand, ctx, player_idx)

                    # --- THE FIX: Tiered Synergy Decay ---
                    # Only decay normal hoarded cards (synergy < 2000).
                    # Key Cards (2000) and Radioactive Cards (4000) are immune to decay!
                    if is_stuck and synergy > 0 and synergy < 2000:
                        synergy *= 0.1

                    # --- THE FIX: Absolute Objective Immunity ---
                    # We boost this from 5000 to 10000.
                    # This guarantees the bot will ALWAYS discard a Key Card (2000 or 4000)
                    # before it ever breaks its own objective (10000+).
                    if not is_down and self._breaks_objective(card, player_idx, ctx, base_progress):
                        synergy += 10000

                    # Penalize high point values to encourage dumping high cards
                    deadwood_val = self._get_deadwood_value(card, ctx)
                    synergy -= (deadwood_val / 100.0)

                    if synergy < lowest_synergy:
                        lowest_synergy = synergy
                        best_discard_idx = action_idx

            if best_discard_idx != -1:
                return best_discard_idx

        # Fall back to standard HeuristicAgent for anything not explicitly overridden
        return super().select_action(state_id, ctx, player_idx, action_mask)

    def _calculate_synergy_smart(self, target_card, hand_list, ctx, player_idx):
        """Refined synergy that dynamically evaluates outstanding needs."""
        player = ctx.players[player_idx]
        has_objective = ctx.check_hand_objective(player_idx)
        is_down = getattr(player, 'is_down', False)

        # 1. Are we ready to play on the table? (Objective met OR already down)
        if has_objective or is_down:
            # ONLY hoard Key Cards if we are actually in a position to use them
            if self._is_playable_on_table(target_card, ctx):
                num_players = len(ctx.players)
                next_player = ctx.players[(player_idx + 1) % num_players]

                # If the next player is down and has 1 or 2 cards left, this card is RADIOACTIVE.
                if getattr(next_player, 'is_down', False) and len(next_player.hand_list) <= 2:
                    return 4000
                return 2000  # Standard Key Card

            # If it's NOT a key card, and we have our objective/are down, it's garbage
            return 0

            # 2. --- Outstanding-Aware Pulse Check (Pre-Objective) ---
        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]

        # What do we actually still need?
        has_sets = ctx._search_melds(player.private_hand, sets_needed=req_sets, runs_needed=0)[0]
        has_runs = ctx._search_melds(player.private_hand, sets_needed=0, runs_needed=req_runs)[0]

        needs_sets = (req_sets > 0) and not has_sets
        needs_runs = (req_runs > 0) and not has_runs

        # --- Object-Identity Safe Duplicate Purge ---
        if needs_runs and not needs_sets:
            # Count how many of this exact logical card exist in the hand
            duplicate_count = sum(
                1 for c in hand_list if c.suit == target_card.suit and c.rank == target_card.rank)

            # If there's more than one, both copies become toxic deadwood
            if duplicate_count > 1:
                return -500

        synergy = 0

        # 4. Only assign synergy to the meld types we are actively missing
        for other in hand_list:
            if other is target_card:
                continue

            # Only hoard pairs if we STILL need Sets
            if needs_sets and other.rank == target_card.rank:
                synergy += 9

            # Only hoard sequences if we STILL need Runs
            if needs_runs and other.suit == target_card.suit:
                rank_diff = abs(other.rank.value - target_card.rank.value)
                if 0 < rank_diff <= 2:
                    synergy += 10

        return synergy

    def _calculate_projected_deadwood(self, ctx, player_idx):
        """Simulates the table extension phase to see how many cards will remain in hand."""
        player = ctx.players[player_idx]
        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]

        # Search for the cards that would be consumed by the 'Go Down' action
        success, ext_sets, ext_runs = ctx._search_melds(player.private_hand, req_sets, req_runs)
        if not success:
            return len(player.hand_list)

        # Calculate what's left after the objective is placed on the table
        objective_tensor = ext_sets + ext_runs
        remaining_count = 0

        for card in player.hand_list:
            suit, rank = int(card.suit), int(card.rank)

            # If this card is part of the objective, it's gone
            if objective_tensor[suit, rank] > 0:
                objective_tensor[suit, rank] -= 1
                continue

            # If it's a key card for the table, it will be played (it's gone)
            if self._is_playable_on_table(card, ctx):
                continue

            # Otherwise, it stays in hand as penalty points
            remaining_count += 1

        return remaining_count

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