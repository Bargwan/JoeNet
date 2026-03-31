import numpy as np
from typing import List
from game_context import GameContext
from cards import Card


class RewardCalculator:
    """
    Calculates State Potential (Phi) based on the structural distance
    to the current round objective.
    """

    def __init__(self, ctx: GameContext):
        self.ctx = ctx
        # The 'Exchange Rate' between cards and penalty points.
        # Set to 25.0 based on the average value of an Ace/Face card combo.
        self.win_hunger = 25.0

    def calculate_distance_to_win(self, player_idx: int) -> int:
        """
        Calculates how many more cards are needed to fulfill the round objective.
        A 'partial' (e.g., a pair or 2-card run) reduces distance.
        """
        player = self.ctx.players[player_idx]

        # Translate the objective_map tuple (e.g., (2, 0)) into a list of slot sizes (e.g., [3, 3])
        req_sets, req_runs = self.ctx.config.objective_map[self.ctx.current_round_idx]
        objective = [3] * req_sets + [4] * req_runs

        # Total cards required for this round (e.g., [3, 3] = 6)
        total_required = sum(objective)

        # Find the best allocation of current hand cards into the objective slots
        cards_found = self._find_max_objective_cards(player.hand_list, objective)

        return max(0, total_required - cards_found)

    def calculate_state_potential(self, player_idx: int) -> float:
        """
        The PBRS Potential Function Phi(s).
        Formula: -(Distance * win_hunger)
        """
        distance = self.calculate_distance_to_win(player_idx)
        return float(-(distance * self.win_hunger))

    def calculate_asymmetric_score(self, margin: float) -> float:
        """
        Terminal scoring logic using multipliers from config.
        Margin > 0: Trailing (Catch-up). Margin < 0: Leading (Pull-ahead).
        """
        if margin > 0:
            return margin * self.ctx.config.catch_up_multiplier
        elif margin < 0:
            return margin * self.ctx.config.pull_ahead_multiplier
        return 0.0

    def _calculate_active_deadwood(self, player_idx: int) -> float:
        """Helper to tally physical penalty points currently in hand."""
        player = self.ctx.players[player_idx]
        deadwood = 0.0
        for card in player.hand_list:
            rank_val = int(card.rank)
            if rank_val == 0:  # ACE
                deadwood += self.ctx.config.points_ace
            elif 7 <= rank_val <= 12:  # 8 through King
                deadwood += self.ctx.config.points_eight_to_king
            else:  # 2 through 7
                deadwood += self.ctx.config.points_two_to_seven
        return deadwood

    def _find_max_objective_cards(self, hand: List[Card], objective: List[int]) -> int:
        """
        Combinatorial search to find the subset of cards that fills the
        most 'slots' in the objective.
        """
        if not hand or not objective:
            return 0

        # 1. Identify all potential components (Sets of 2+, Runs of 2+)
        # For 'Distance' logic, we count pairs and sequences of 2 as progress.
        potential_sets = self._get_potential_sets(hand)
        potential_runs = self._get_potential_runs(hand)

        # 2. Recursive search to find best non-overlapping assignment
        return self._backtrack_fill(objective, potential_sets, potential_runs)

    def _get_potential_sets(self, hand):
        ranks = {}
        for c in hand:
            ranks[c.rank] = ranks.get(c.rank, 0) + 1
        # Returns list of card counts for every rank where we have at least a pair
        return [count for count in ranks.values() if count >= 2]

    def _get_potential_runs(self, hand):
        # Simplification: Find sequences of same suit >= 2 cards
        run_counts = []
        hand_sorted = sorted(hand, key=lambda x: (x.suit.value, x.rank.value))

        if not hand_sorted: return []

        current_run = [hand_sorted[0]]
        for i in range(1, len(hand_sorted)):
            prev, curr = hand_sorted[i - 1], hand_sorted[i]
            if curr.suit == prev.suit and curr.rank.value == prev.rank.value + 1:
                current_run.append(curr)
            elif curr.suit == prev.suit and curr.rank.value == prev.rank.value:
                continue  # Duplicate card doesn't break run but doesn't extend it
            else:
                if len(current_run) >= 2:
                    run_counts.append(len(current_run))
                current_run = [curr]
        if len(current_run) >= 2:
            run_counts.append(len(current_run))
        return run_counts

    def _backtrack_fill(self, remaining_obj, sets, runs):
        """Greedily assigns the largest partials to the largest objective slots."""
        if not remaining_obj:
            return 0

        # For distance-to-win, we can simplify:
        # Sort objective largest first. Fill with largest matching partials.
        obj = sorted(remaining_obj, reverse=True)
        total_fill = 0

        # This is a simplified greedy approach for the potential function.
        # It counts how many cards contribute to required 3s or 4s.
        used_sets = sorted(sets, reverse=True)
        used_runs = sorted(runs, reverse=True)

        for slot_size in obj:
            if slot_size == 3 and used_sets:
                found = used_sets.pop(0)
                total_fill += min(3, found)
            elif slot_size == 4 and used_runs:
                found = used_runs.pop(0)
                total_fill += min(4, found)
            elif slot_size == 4 and used_sets:
                # A set can't fill a run slot
                continue
            elif slot_size == 3 and used_runs:
                # A run can't fill a set slot
                continue

        return total_fill