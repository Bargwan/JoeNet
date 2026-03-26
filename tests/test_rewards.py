import unittest
import numpy as np

from game_context import GameContext
from cards import Card, Suit, Rank
from config import JoeConfig
# We will build this class next!
from reward import RewardCalculator


class TestUkeireCalculator(unittest.TestCase):
    def setUp(self):
        self.ctx = GameContext(num_players=4)
        self.ctx.deck.clear()
        self.reward_calc = RewardCalculator(self.ctx)

    def test_effective_outs_single_card(self):
        """
        Verify the base math: Total in Double Deck (2) - Known Public - Oracle Guesses
        Target Card: Ace of Spades
        """
        target_suit = Suit.SPADES.value
        target_rank = Rank.ACE.value

        # 1. Setup Public Knowledge
        # Put 1 Ace of Spades in the discard pile (Publicly dead)
        self.ctx.discard_pile.append(Card(Suit.SPADES, Rank.ACE, deck_index=0))

        # 2. Setup Oracle Predictions (Shape: 3, 4, 14)
        mock_oracle = np.zeros((3, 4, 14), dtype=np.float32)

        # Op1 (Left) is 100% sure they have an Ace of Spades
        mock_oracle[0, target_suit, target_rank] = 1.0

        # Op2 (Across) is 50% sure they have an Ace of Spades
        mock_oracle[1, target_suit, target_rank] = 0.5

        # 3. ACT: Calculate effective outs
        outs = self.reward_calc.calculate_card_outs(
            suit=target_suit,
            rank=target_rank,
            player_idx=0,
            oracle_probs=mock_oracle
        )

        self.assertEqual(outs, 0.0,
                         "Outs should floor at 0.0 if predictions exceed remaining cards.")

    def test_effective_outs_partial_availability(self):
        """
        Verify fractional outs when a card is partially available.
        Target Card: King of Hearts
        """
        target_suit = Suit.HEARTS.value
        target_rank = Rank.KING.value

        # 1. Setup Public Knowledge: Player 0 already holds 1 King of Hearts
        self.ctx.players[0].receive_cards([Card(Suit.HEARTS, Rank.KING, deck_index=0)])

        # 2. Setup Oracle Predictions
        mock_oracle = np.zeros((3, 4, 14), dtype=np.float32)
        # Op3 (Right) might have one (25% chance)
        mock_oracle[2, target_suit, target_rank] = 0.25

        # 3. ACT
        # Math: 2 (Total) - 1 (In Hand) - 0.25 (Op3) = 0.75 Effective Outs
        outs = self.reward_calc.calculate_card_outs(
            suit=target_suit,
            rank=target_rank,
            player_idx=0,
            oracle_probs=mock_oracle
        )

        self.assertEqual(outs, 0.75, "Expected exactly 0.75 effective outs.")


class TestAsymmetricScoring(unittest.TestCase):
    def setUp(self):
        # Inject a config to strictly test the multipliers
        self.config = JoeConfig(catch_up_multiplier=2.0, pull_ahead_multiplier=0.5)
        self.ctx = GameContext(num_players=4, config=self.config)
        self.reward_calc = RewardCalculator(self.ctx)

    def test_asymmetric_multipliers(self):
        """
        Step 3.3: Verify the piecewise reward delta correctly applies the
        config multipliers to negative (trailing) and positive (leading) margins.
        """
        # Trailing by 50 points -> -50 * 2.0 = -100.0
        val_trailing = self.reward_calc.calculate_asymmetric_score(-50)
        self.assertEqual(val_trailing, -100.0)

        # Leading by 20 points -> 20 * 0.5 = 10.0
        val_leading = self.reward_calc.calculate_asymmetric_score(20)
        self.assertEqual(val_leading, 10.0)

        # Tied -> 0 * anything = 0.0
        val_tied = self.reward_calc.calculate_asymmetric_score(0)
        self.assertEqual(val_tied, 0.0)

    def test_reward_delta_crosses_zero(self):
        """
        Step 3.3: Verify that calculating (End_Value - Start_Value) naturally
        splits the multiplier when a player overtakes an opponent.
        """
        # User Scenario: Trailing by 40, ends up winning by 60 (100 point swing)
        # 40 points * 2.0 = 80
        # 60 points * 0.5 = 30
        # Expected Total Reward = 110.0

        start_val = self.reward_calc.calculate_asymmetric_score(-40)
        end_val = self.reward_calc.calculate_asymmetric_score(60)

        reward_delta = end_val - start_val
        self.assertEqual(reward_delta, 110.0, "Delta math failed to split the multiplier!")


class TestBetaoriDangerScore(unittest.TestCase):
    def setUp(self):
        # Using the same config: Trailing = 2.0x, Leading = 0.5x
        self.config = JoeConfig(catch_up_multiplier=2.0, pull_ahead_multiplier=0.5)
        self.ctx = GameContext(num_players=4, config=self.config)
        self.ctx.deck.clear()
        self.reward_calc = RewardCalculator(self.ctx)
        self.player_idx = 0

    def test_calculate_active_deadwood(self):
        """Verify the calculator accurately tallies the physical hand penalty."""
        cards = [
            Card(Suit.SPADES, Rank.ACE),  # 20
            Card(Suit.HEARTS, Rank.KING),  # 10
            Card(Suit.CLUBS, Rank.FIVE)  # 5
        ]
        self.ctx.players[self.player_idx].receive_cards(cards)

        deadwood = self.reward_calc._calculate_active_deadwood(self.player_idx)
        self.assertEqual(deadwood, 35.0)

    def test_danger_score_safe_detonation(self):
        """
        Detonate on a TRAILING player.
        Agent Tourney Score: 10. Agent Deadwood: 5.  (Projected: 15)
        Op Tourney Score: 100. Op goes out: 0.       (Projected: 100)
        Raw Margin: 15 - 100 = -85 (Agent is winning massively).
        Asymmetric Margin: -85 * 2.0 (catch-up multiplier) = -170.
        """
        self.ctx.players[0].score = 10
        self.ctx.players[1].score = 100

        active_deadwood = 5.0
        # Op1 needs it 100% of the time
        oracle_probs = np.zeros((3, 4, 14), dtype=np.float32)
        oracle_probs[0, Suit.SPADES.value, Rank.ACE.value] = 1.0

        danger = self.reward_calc.calculate_danger_score(
            player_idx=0,
            suit=Suit.SPADES.value,
            rank=Rank.ACE.value,
            active_deadwood=active_deadwood,
            oracle_probs=oracle_probs
        )

        # Danger should be intensely negative (a massive reward bonus for detonating)
        self.assertEqual(danger, -170.0, "Failed to incentivize detonating on a trailing player.")

    def test_danger_score_suicidal_detonation(self):
        """
        Attempting to detonate on the LEADING player.
        Agent Tourney Score: 50. Agent Deadwood: 5. (Projected: 55)
        Op Tourney Score: 0.     Op goes out: 0.    (Projected: 0)
        Raw Margin: 55 - 0 = +55 (Agent is losing to this player).
        Asymmetric Margin: +55 * 0.5 (pull-ahead multiplier) = +27.5.
        """
        self.ctx.players[0].score = 50
        self.ctx.players[1].score = 0

        active_deadwood = 5.0

        oracle_probs = np.zeros((3, 4, 14), dtype=np.float32)
        oracle_probs[0, Suit.SPADES.value, Rank.ACE.value] = 1.0

        danger = self.reward_calc.calculate_danger_score(
            player_idx=0,
            suit=Suit.SPADES.value,
            rank=Rank.ACE.value,
            active_deadwood=active_deadwood,
            oracle_probs=oracle_probs
        )

        # Danger should be POSITIVE. The math strictly forbids feeding the leader.
        self.assertEqual(danger, 27.5, "Failed to penalize feeding the tournament leader!")

    def test_danger_score_zero_probability(self):
        """If the Oracle is 100% certain the opponent doesn't need it, Danger is 0.0."""
        self.ctx.players[0].score = 50
        self.ctx.players[1].score = 0

        active_deadwood = 5.0
        oracle_probs = np.zeros((3, 4, 14), dtype=np.float32)  # All 0.0

        danger = self.reward_calc.calculate_danger_score(
            player_idx=0,
            suit=Suit.SPADES.value,
            rank=Rank.ACE.value,
            active_deadwood=active_deadwood,
            oracle_probs=oracle_probs
        )

        self.assertEqual(danger, 0.0)

if __name__ == '__main__':
    unittest.main()