import numpy as np
from game_context import GameContext


class RewardCalculator:
    """
    Handles the deterministic mathematical logic required for Potential-Based Reward Shaping (PBRS)
    and Asymmetric Terminal Scoring to resolve the Credit Assignment Problem.
    """

    def __init__(self, ctx: GameContext):
        self.ctx = ctx

    def calculate_card_outs(self, suit: int, rank: int, player_idx: int,
                            oracle_probs: np.ndarray) -> float:
        """
        Step 3.1: Combinatorial Outs Calculator (Ukeire)
        Calculates how many copies of a specific card are available in the unknown universe,
        discounting cards the Oracle predicts opponents are holding.
        """
        # Joe uses a double deck, meaning exactly 2 of every card exist in the universe
        total_in_universe = 2.0

        # 1. Deduct Known Absolute Truths
        known_count = 0.0

        # Cards in observing player's private hand
        known_count += float(self.ctx.players[player_idx].private_hand[suit, rank])

        # Cards publicly visible on the table
        known_count += float(self.ctx.table_sets[suit, rank])
        known_count += float(self.ctx.table_runs[suit, rank])

        # Cards dead in the discard pile (iterating the list safely captures both buried and top card)
        known_count += sum(
            1.0 for c in self.ctx.discard_pile if int(c.suit) == suit and int(c.rank) == rank)

        # 2. Deduct Oracle Predictions
        # Sums the 0.0 to 1.0 probability across all 3 opponent channels for this specific card
        predicted_held = float(np.sum(oracle_probs[:, suit, rank]))

        # 3. Calculate Effective Outs
        effective_outs = total_in_universe - known_count - predicted_held

        # Mathematically floor at 0.0 to prevent the Oracle's hallucinations
        # (e.g., predicting 3 Aces exist) from creating negative outs.
        return max(effective_outs, 0.0)

    def calculate_asymmetric_score(self, margin: float) -> float:
        """
        Step 3.3: Asymmetric Terminal Scoring
        Scales the raw point differential using the config's tournament-aware multipliers.
        Negative margins (Trailing) use catch_up_multiplier.
        Positive margins (Leading) use pull_ahead_multiplier.
        """
        if margin < 0:
            return margin * self.ctx.config.catch_up_multiplier
        elif margin > 0:
            return margin * self.ctx.config.pull_ahead_multiplier
        else:
            return 0.0

    def _calculate_active_deadwood(self, player_idx: int) -> float:
        """
        Helper to calculate the exact penalty points currently sitting in a player's hand.
        """
        player = self.ctx.players[player_idx]
        deadwood = 0.0

        for card in player.hand_list:
            rank_val = int(card.rank)
            if rank_val == 0:  # ACE
                deadwood += self.ctx.config.points_ace
            elif 7 <= rank_val <= 12:  # EIGHT through KING
                deadwood += self.ctx.config.points_eight_to_king
            elif 1 <= rank_val <= 6:  # TWO through SEVEN
                deadwood += self.ctx.config.points_two_to_seven

        return deadwood

    def calculate_danger_score(self, player_idx: int, suit: int, rank: int,
                               active_deadwood: float, oracle_probs: np.ndarray) -> float:
        """
        Step 3.2: Danger Score Calculator (Betaori)
        Calculates the asymmetric tournament risk of discarding a specific card.
        If the margin is negative (Agent is winning), the Danger becomes a bonus,
        incentivizing the Detonation Strategy.
        """
        total_danger = 0.0
        agent = self.ctx.players[player_idx]

        # If the agent holds the card and the opponent goes out, the agent eats their deadwood
        agent_projected_score = agent.score + active_deadwood

        for i in range(1, 4):
            # Strict Zero-Padding for missing 4th player
            if i == 3 and self.ctx.num_players == 3:
                continue

            target_idx = (player_idx + i) % self.ctx.num_players
            opponent = self.ctx.players[target_idx]

            # The probability this specific opponent needs this card to go out
            p_needs_card = float(oracle_probs[i - 1, suit, rank])

            if p_needs_card == 0.0:
                continue

            # If the opponent goes out, they take 0 penalty points for this round
            opponent_projected_score = opponent.score + 0.0

            # Calculate the Tournament Margin
            raw_margin = agent_projected_score - opponent_projected_score

            # Scale the margin using the Asymmetric Multipliers
            asym_margin = self.calculate_asymmetric_score(raw_margin)

            # Accumulate the danger
            total_danger += p_needs_card * asym_margin

        return total_danger
