"""
Agents that play Joe. At each decision an agent is given the legal actions
(GameContext.legal_actions(), see actions.py) and returns one of them.

- RandomAgent:        picks any legal action.
- HeuristicAgent:     the rule-based baseline.
- ProbabilisticAgent: chooses by expected value, using draw probabilities.
- BayesianAgent:      planned; not implemented yet.
"""
import math
import random

import numpy as np

from actions import (
    CallMayI,
    Destination,
    Discard,
    EndTablePlay,
    GoDown,
    Pass,
    PickDiscard,
    PickStock,
    Place,
    Wait,
)


class Agent:
    """Base class for agents."""

    def select_action(self, state_id, ctx, player_idx, legal):
        """
        Returns the action to take: one of `legal`, the list from
        ctx.legal_actions(player_idx, state_id). `state_id` names the decision,
        e.g. 'pickup_decision'.
        """
        raise NotImplementedError


class RandomAgent(Agent):
    """Picks a legal action uniformly at random."""

    def __init__(self, random_seed=None):
        self.rng = random.Random(random_seed)

    def select_action(self, state_id, ctx, player_idx, legal):
        """Picks one of the legal actions at random."""
        return self.rng.choice(legal)


class HeuristicAgent(Agent):
    """
    The rule-based baseline, meant to play like a basic human player.
    ProbabilisticAgent inherits from it and uses its decisions wherever it has
    no expected-value logic of its own (e.g. placing Key Cards on the table).
    """

    def __init__(self, random_seed=None):
        self.rng = random.Random(random_seed)

    def select_action(self, state_id, ctx, player_idx, legal):
        """
        Chooses an action with simple rules for each decision:
        - Pickup: take the discard if it fits the table (when down, or when holding
          the objective) or helps the objective; otherwise draw from the stock.
        - May-I: call if the discard advances the objective, or from Round 4 on,
          if it is useful and the hand has 11 cards or fewer.
        - Go down: go down if that leaves 5 or fewer cards that can't be placed;
          otherwise go down with a probability that rises with time and threat.
        - Table play: place every Key Card, preferring runs (at random) to sets.
        - Discard: discard the card with the lowest synergy score.
        """
        player = ctx.players[player_idx]
        hand = player.hand_list
        discard_top = ctx.discard_pile[-1] if ctx.discard_pile else None

        # --- Phase: PICKUP DECISION ---
        if state_id == 'pickup_decision' and discard_top:
            can_take_discard = PickDiscard() in legal

            # 1. Holding the objective but not down: take a discard that fits the table
            if can_take_discard and not player.is_down and ctx.check_hand_objective(player_idx):
                if self._is_playable_on_table(discard_top, ctx):
                    return PickDiscard()

            # 2. Standard pickup logic
            if player.is_down:
                if can_take_discard and self._is_playable_on_table(discard_top, ctx):
                    return PickDiscard()
            elif can_take_discard:
                if self._advances_objective(discard_top, player_idx, ctx):
                    return PickDiscard()
                if self._is_useful_pickup(discard_top, player_idx, ctx):
                    return PickDiscard()

            # 3. Otherwise draw from the stock
            return PickStock()

        # --- Phase: MAY-I DECISION ---
        elif state_id == 'may_i_decision' and discard_top:
            # 1. The discard completes another meld towards the objective
            if self._advances_objective(discard_top, player_idx, ctx):
                return CallMayI()

            # 2. From Round 4 ([3, 3, 3]) on, also call for merely useful cards
            if ctx.current_round_idx >= 3 and len(hand) <= 11:
                if self._is_useful_pickup(discard_top, player_idx, ctx):
                    return CallMayI()

            return Pass()

        # --- Phase: GO DOWN DECISION ---
        elif state_id == 'go_down_decision':
            if any(isinstance(a, GoDown) for a in legal):
                # Go down if it leaves 5 or fewer cards that can't be placed
                if self._calculate_projected_deadwood(ctx, player_idx) <= 5:
                    return self._choose_go_down(ctx, player_idx, legal)

                # Otherwise apply patience pressure
                go_down_prob = self._calculate_go_down_probability(ctx, player_idx)
                if self.rng.random() <= go_down_prob:
                    return self._choose_go_down(ctx, player_idx, legal)
            return Wait()

        # --- Phase: TABLE PLAY DECISION ---
        elif state_id == 'table_play_phase':
            # Place every Key Card, one per decision. Prefer runs (chosen at random),
            # which tends to end rounds sooner; otherwise use a set.
            placements = sorted((a for a in legal if isinstance(a, Place)),
                                key=lambda a: (a.destination, a.suit, a.rank))
            run_plays = [a for a in placements if a.destination != Destination.ON_SET]
            set_plays = [a for a in placements if a.destination == Destination.ON_SET]
            if run_plays:
                return self.rng.choice(run_plays)
            if set_plays:
                return set_plays[0]
            return EndTablePlay()

        # --- Phase: DISCARD DECISION ---
        elif state_id == 'discard_phase' and hand:
            req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]
            base_progress = self._get_objective_progress(player.private_hand, req_sets, req_runs,
                                                         ctx)

            best_discard = None
            lowest_synergy = float('inf')

            # After 15 circuits, add noise to break repeated discard cycles
            is_stuck = ctx.current_circuit > 15

            for card in hand:
                synergy = self._calculate_synergy(card, hand, ctx, player_idx)

                if is_stuck and 0 < synergy < 2000:
                    synergy += self.rng.uniform(-40, 40)

                # Never discard a card that the objective progress depends on
                if not player.is_down and self._breaks_objective(card, player_idx, ctx,
                                                                 base_progress):
                    synergy += 10000

                # Penalize high point values to dump heavy deadwood
                deadwood_val = self._get_deadwood_value(card, ctx)
                synergy -= (deadwood_val / 100.0)

                if synergy < lowest_synergy:
                    lowest_synergy = synergy
                    best_discard = Discard(card.suit, card.rank)

            if best_discard is not None:
                return best_discard

        # --- FALLBACK ---
        return self.rng.choice(legal)

    def _choose_go_down(self, ctx, player_idx, legal):
        """
        The GoDown (of those in `legal`) that leaves the lowest penalty value in
        hand once every placeable card has been placed. Ties are broken at random.
        """
        options = [a for a in legal if isinstance(a, GoDown)]
        if len(options) == 1:
            return options[0]

        scored = []
        for option in options:
            future = ctx.clone()
            future.go_down(player_idx, list(option.melds))
            future._place_key_cards(player_idx)
            scored.append((future.hand_penalty(player_idx), option))

        lowest = min(score for score, _ in scored)
        return self.rng.choice([option for score, option in scored if score == lowest])

    # ==========================================
    # DECISION HELPERS
    # ==========================================

    def _is_useful_pickup(self, target_card, player_idx, ctx):
        """
        True if the card is near a meld the hand still needs: same rank as a held
        card (sets), or same suit within 2 ranks of a held card (runs). A duplicate
        of a held card is refused when only runs are still needed.
        """
        player = ctx.players[player_idx]
        hand = player.hand_list
        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]

        has_sets = ctx.find_melds(player.private_hand, req_sets, 0) is not None
        has_runs = ctx.find_melds(player.private_hand, 0, req_runs) is not None

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
        """
        How much the hand wants to keep this card; the discard logic drops the
        lowest. Once the objective is held or the player is down, Key Cards score
        2000 (4000 if the next player is down with 2 or fewer cards, so feeding
        them is avoided) and everything else 0. Before that, the score counts
        nearby same-rank / same-suit cards, and duplicates score -500 when only
        runs are needed.
        """
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
        has_sets = ctx.find_melds(player.private_hand, req_sets, 0) is not None
        has_runs = ctx.find_melds(player.private_hand, 0, req_runs) is not None

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
        """
        Number of cards (not points) that would stay in hand after going down
        with the first valid melds and placing every Key Card. If the objective
        isn't met, the whole hand.
        """
        player = ctx.players[player_idx]
        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]

        melds = ctx.find_melds(player.private_hand, req_sets, req_runs)
        if melds is None:
            return len(player.hand_list)

        # Cards left after laying the melds, minus any that could go on the table.
        remaining = player.private_hand
        for meld in melds:
            remaining = ctx._remove_meld(remaining, meld)

        remaining_count = 0
        for suit in range(4):
            for rank in range(13):
                if remaining[suit, rank] > 0 and not ctx.is_key_card(suit, rank):
                    remaining_count += int(remaining[suit, rank])
        return remaining_count

    def _calculate_go_down_probability(self, ctx, player_idx):
        """
        Chance of going down now: at least 0.5, rising with circuits played
        (relative to 8 + round index) and with how few cards a down opponent holds.
        """
        prob = 0.5
        patience_threshold = 8.0 + ctx.current_round_idx
        turn_pressure = ctx.current_circuit / patience_threshold
        prob = max(prob, turn_pressure)

        for i, p in enumerate(ctx.players):
            if i != player_idx and getattr(p, 'is_down', False):
                cards_left = len(p.hand_list)
                prob = max(prob, 1.0 - (cards_left * 0.1))

        return min(1.0, prob)

    def _advances_objective(self, discard_card, player_idx, ctx):
        """
        True if adding this card lets the hand complete more of the objective's
        melds than before. False if the objective is already met.
        """
        player = ctx.players[player_idx]
        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]

        if ctx.find_melds(player.private_hand, req_sets, req_runs) is not None:
            return False

        base_progress = self._get_objective_progress(player.private_hand, req_sets, req_runs, ctx)
        temp_tensor = player.private_hand.copy()
        suit, rank = discard_card.suit.value, discard_card.rank.value
        ctx._sync_ace(temp_tensor, suit, rank, increment=True)

        new_progress = self._get_objective_progress(temp_tensor, req_sets, req_runs, ctx)
        return new_progress > base_progress

    def _breaks_objective(self, candidate_card, player_idx, ctx, base_progress):
        """True if removing this card lowers the objective progress below `base_progress`."""
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
        """The largest number of the objective's melds this hand can complete."""
        for total in range(req_sets + req_runs, 0, -1):
            for s in range(min(total, req_sets), -1, -1):
                r = total - s
                if r <= req_runs:
                    if ctx.find_melds(tensor, s, r) is not None:
                        return total
        return 0

    def _get_deadwood_value(self, card, ctx):
        """Penalty points for one card: Ace 20, 8 to King 10, 2 to 7 5 (from the config)."""
        rank_val = int(card.rank)
        if rank_val == 0 or rank_val == 13:
            return ctx.config.points_ace
        elif 7 <= rank_val <= 12:
            return ctx.config.points_eight_to_king
        return ctx.config.points_two_to_seven

    def _is_playable_on_table(self, card, ctx):
        """
        True if the card could legally be placed on some meld on the table
        (i.e. it would be a Key Card). Same test as GameContext.is_key_card.
        """
        suit, rank = int(card.suit), int(card.rank)
        if np.any(ctx.table_sets[:, rank] > 0):
            return True
        if ctx._can_extend_run(suit, rank):
            return True
        return False


class ProbabilisticAgent(HeuristicAgent):
    """
    Chooses pickups, May-Is, going down and discards by expected value (EV) in
    penalty points. Each option is scored by evaluating the hand it would leave:
    the chance of completing the objective from draw probabilities over the
    remaining turns, weighed against the chance an opponent ends the round first.
    Table play is left to HeuristicAgent. See probabilistic_agent_theory.md.
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
        Calculates the cumulative probability of successfully drawing at least the
        `required` amount of a specific card over a given `horizon`.

        Uses a Binomial approximation of the hypergeometric distribution to accurately
        sum the exact probabilities of drawing k cards.

        Rationale: Converts a singular draw chance (e.g., 2% chance on the next turn)
        into a realistic trajectory (e.g., 30% chance to find the card before the
        game ends). Crucially, this mathematically grounds the bot, preventing it
        from overestimating its chances of pulling multiple perfect cards (like an
        entire 3-card set) from a completely empty hand.
        """
        if required == 0: return 1.0
        if available < required: return 0.0
        if unknown_cards <= 0: return 0.0

        # Binomial approximation of hypergeometric
        p_success = available / unknown_cards
        p_fail = 1.0 - p_success

        # We want P(X >= required).
        # Mathematically: 1.0 - P(X = 0) - P(X = 1) ... - P(X = required - 1)
        cumulative_miss_prob = 0.0
        int_horizon = int(max(1, horizon))

        for k in range(int(required)):
            # If horizon is smaller than k, we physically can't draw enough cards
            if int_horizon < k:
                cumulative_miss_prob += 0.0
                continue

            nCr = math.comb(int_horizon, k)
            p_exact_k = nCr * (p_success ** k) * (p_fail ** max(0, int_horizon - k))
            cumulative_miss_prob += p_exact_k

        return max(0.0, 1.0 - cumulative_miss_prob)

    def _get_expected_stock_value(self, ctx, hand_tensor, player_idx):
        """
        Average penalty value of a card not yet seen (stock or opponents' unknown
        cards), used as the value of a blind stock card.

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

    def _evaluate_hand_state(self, hand_tensor, ctx, player_idx, horizon_offset=0):
        """
        THE MASTER EQUATION.
        Evaluates the literal Expected Value (EV) of a hand state in penalty points.

        Calculates two timelines:
        1. Timeline A (Imminent Threat): The EV if an opponent ends the round THIS turn.
           (Results in taking full Deadwood or Total Hand points).
        2. Timeline B (Future Trajectory): The EV if the game continues naturally.
           (Results in blended risk and claiming the Opponents' Penalty Points).

        Rationale: This equation balances aggressive drafting (Timeline B) against
        sudden-death survival (Timeline A), scaled dynamically by the physical
        threat level (`p_lose`) of the opponents.
        """
        is_down = ctx.players[player_idx].is_down

        if is_down:
            p_win = 1.0
            slotted_mask = np.zeros((4, 14), dtype=np.int8)
        else:
            req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]

            p_win, slotted_mask = self._find_best_seed_allocation(
                hand_tensor, req_sets, req_runs, ctx, player_idx, horizon_offset=horizon_offset
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
        # 3. THE MASTER EQUATION (Pairwise Difference)
        # ==========================================
        num_opponents = len(ctx.players) - 1

        current_physical_penalty = deadwood_points if is_down else total_hand_points
        future_blended_penalty = (p_win * deadwood_points) + (
                    (1.0 - p_win) * total_hand_points)

        # Our penalty is weighed against each opponent separately, so it counts once per opponent
        expected_penalty_risk = num_opponents * ((p_lose * current_physical_penalty) + (
                    (1.0 - p_lose) * future_blended_penalty))

        # Reward remains the sum of the opponents' penalties
        expected_reward = (1.0 - p_lose) * (p_win * opponents_expected_penalty)

        # Pairwise EV automatically restores the 1:1 pot odds ratio
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
        Copies (0-2) of each card not yet accounted for: two of every card, minus
        the table melds, buried discards, our hand, and cards opponents are known
        to hold (picked up and not discarded since). The top discard and the
        opponents' unseen cards are counted as available.
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

    def _calculate_seed_probability(self, seed, ctx, hand_tensor, player_idx, horizon_offset=0):
        """
        Chance of completing a partial meld (seed) within the turns expected to
        remain this round (expected round length minus circuits played, plus
        `horizon_offset`).
        """
        if seed['distance'] == 0:
            return 1.0

        available_tensor = self._get_available_tensor(ctx, hand_tensor, player_idx)
        unknown_deck_size = float(np.sum(available_tensor[:, 0:13]))

        # --- DYNAMIC TICKING CLOCK ---
        if len(ctx.players) == 3:
            total_horizon = self.expected_horizon_3p
        else:
            total_horizon = self.expected_horizon_4p

        # Subtract the current circuit to find true remaining turns,
        # then add the virtual 'May-I' offset.
        horizon = max(1.0, (total_horizon - ctx.current_circuit) + horizon_offset)

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

    def _find_best_seed_allocation(self, hand_tensor, req_sets,
                                   req_runs, ctx, player_idx, horizon_offset=0):
        """
        Greedily picks, one meld at a time, the seed most likely to be completed,
        and multiplies their chances into p_win (greedy, so not guaranteed best).
        Returns (p_win, mask of the cards the chosen seeds use).
        """
        p_win = 1.0
        slotted_mask = np.zeros((4, 14), dtype=np.int8)
        working_hand = hand_tensor.copy()

        sets_found = 0
        runs_found = 0

        # Loop strictly bounds to the maximum requirements (e.g., 3 iterations)
        while sets_found < req_sets or runs_found < req_runs:

            current_seeds = self._parse_all_valid_seeds(working_hand, req_sets - sets_found,
                                                        req_runs - runs_found)

            if not current_seeds:
                break

            for seed in current_seeds:
                seed['ev'] = self._calculate_seed_probability(seed, ctx, working_hand, player_idx,
                                                              horizon_offset=horizon_offset)

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

        # 5. Each meld with no seed at all multiplies p_win by 0.01
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

    def select_action(self, state_id, ctx, player_idx, legal):
        """
        Overrides the rule-based heuristics with a 1-Ply Stochastic Lookahead.

        - Discard Phase: Simulates the EV of the hand *after* dropping every legally discardable card.
        - Pickup Phase: Simulates the EV of a Full Turn (Pickup + Best Discard).
        - May-I Phase: Compares baseline EV against the EV of expanding the hand,
          minus the dynamic expected risk of the blind stock card.
        - Go Down Phase: Compares the EV of waiting with the EV of going down and
          placing every Key Card (which also opens the table to opponents).

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
            best_discard = None
            best_future_ev = -float('inf')

            # Simulate the future for every card in hand
            for card in player.hand_list:
                # Create the hypothetical future (Hand MINUS this card)
                hypo_tensor = hand_tensor.copy()
                suit, rank = int(card.suit), int(card.rank)
                ctx._sync_ace(hypo_tensor, suit, rank, increment=False)

                # Ask the EV Engine how good this resulting hand is
                future_ev = self._evaluate_hand_state(hypo_tensor, ctx, player_idx)

                if future_ev > best_future_ev:
                    best_future_ev = future_ev
                    best_discard = Discard(card.suit, card.rank)

            if best_discard is not None:
                return best_discard

        # ==========================================
        # 2. PICKUP DECISION (Full Turn Lookahead)
        # ==========================================
        elif state_id == 'pickup_decision' and discard_top:
            if PickDiscard() in legal:
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

                # Only pick up the discard if it is worth at least 1 penalty point.
                if best_post_discard_ev > baseline_ev + 1.0:
                    return PickDiscard()

            return PickStock()

        # ==========================================
        # 3. MAY-I DECISION (The Virtual Turn Expansion)
        # ==========================================
        elif state_id == 'may_i_decision' and discard_top:
            # player_idx is the player being offered the May-I
            hand_tensor = ctx.players[player_idx].private_hand

            # 1. Baseline: Hand as it is, with 0 offset
            baseline_ev = self._evaluate_hand_state(hand_tensor, ctx, player_idx,
                                                    horizon_offset=0)

            hypo_tensor = hand_tensor.copy()
            suit, rank = int(discard_top.suit), int(discard_top.rank)
            ctx._sync_ace(hypo_tensor, suit, rank, increment=True)

            # 2. Pickup: Hand + extra card, WITH the +1 'Virtual Turn' offset
            pickup_ev = self._evaluate_hand_state(hypo_tensor, ctx, player_idx,
                                                  horizon_offset=1)

            stock_pts = self._get_expected_stock_value(ctx, hand_tensor, player_idx)

            # --- THE AVALANCHE THREAT ---
            available_est = self._get_available_tensor(ctx, hand_tensor, player_idx)
            unknown_size_est = float(np.sum(available_est[:, 0:13]))

            p_lose_est = self._calculate_avalanche_threat(ctx, player_idx, available_est,
                                                          unknown_size_est)

            # The expected cost remains bound to a strict 1-turn sudden death evaluation
            ev_cost_of_blind_card = stock_pts * p_lose_est

            # 3. Call if the extra card is worth more than the blind stock card costs
            if (pickup_ev - ev_cost_of_blind_card) > baseline_ev:
                return CallMayI()
            return Pass()

        # ==========================================
        # 4. GO DOWN DECISION (The Tactical Sandbag)
        # ==========================================
        elif state_id == 'go_down_decision':
            if any(isinstance(a, GoDown) for a in legal):

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
                    return self._choose_go_down(ctx, player_idx, legal)
            return Wait()

        # ==========================================
        # 5. FALLBACK (Table Play & Edge Cases)
        # ==========================================
        # Let the base HeuristicAgent handle playing on table melds,
        # as those are strictly rule-bound actions that don't require EV simulation.
        return super().select_action(state_id, ctx, player_idx, legal)


class BayesianAgent(ProbabilisticAgent):
    """
    Planned; not implemented yet. Intended design:

    Belief tensor
        4 suits x 13 ranks x (N players + 3) layers: one per player, plus the
        stock, the discard pile and the table melds. Each cell is the expected
        number of copies (0-2) of that card in that place. Every card's copies
        sum to 2 across the layers, and each player's layer sums to their hand size.

    Updates
        - After the deal: each unknown card is spread over the opponents' hands
          and the stock as for 11 cards dealt without replacement from 104.
        - Own hand, the top discard and the table melds are known exactly.
        - An opponent's pickup or May-I places that card in their hand, and a
          likelihood matrix for the round objective (52 x 52, P(other card in hand
          | card picked up)) raises the chance of related cards in their hand.
        - An opponent's discard uses the matching likelihood the other way: cards
          related to the discard become less likely in their hand.
        - After every update, all layers are rebalanced so both sums still hold.
        The likelihood matrices are to be tuned later by optimisation.

    Decisions
        - Pickup: score every possible meld (208 sets, 44 runs) by how likely the
          agent is to complete it. Cards on the table are gone, buried discards are
          locked until the stock is rebuilt, cards in upstream opponents' partial
          melds are unlikely, and own cards are certain. Take the discard if it
          raises the hand's potential more than an average stock card would.
        - Discard: the same in reverse.
        - May-I: as pickup, but counting the penalty card and needing a larger gain.
        - Going down: weigh opponents' estimated progress and, once down, deadwood.
        - Placing Key Cards: avoid unlocking an opponent's hand, or occasionally
          unlock one to make another opponent bust.
    """
    pass
