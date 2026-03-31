import unittest

from game_context import GameContext
from cards import Card, Suit, Rank
from config import JoeConfig
from reward import RewardCalculator


class TestDistanceToWin(unittest.TestCase):
    def setUp(self):
        self.ctx = GameContext(num_players=4)
        self.ctx.deck.clear()
        self.reward_calc = RewardCalculator(self.ctx)

        # Lock in a win_hunger constant for consistent testing
        self.reward_calc.win_hunger = 25.0

    def test_distance_empty_hand(self):
        """
        Round 0 Objective: Two 3s (6 cards total required).
        An empty hand should have a maximum distance of 6.
        """
        self.ctx.current_round_idx = 0
        self.ctx.players[0].hand_list.clear()

        distance = self.reward_calc.calculate_distance_to_win(player_idx=0)
        self.assertEqual(distance, 6, "Empty hand should equal the total objective requirement.")

    def test_distance_partial_set(self):
        """
        Round 0 Objective: Two 3s (6 cards total).
        Holding a Pair uses 2 valid cards. Distance should be 4.
        """
        self.ctx.current_round_idx = 0
        cards = [Card(Suit.SPADES, Rank.EIGHT), Card(Suit.HEARTS, Rank.EIGHT)]
        self.ctx.players[0].receive_cards(cards)

        distance = self.reward_calc.calculate_distance_to_win(player_idx=0)
        self.assertEqual(distance, 4, "Failed to calculate distance with a single pair.")

    def test_distance_completed_set_and_partial(self):
        """
        Round 0 Objective: Two 3s (6 cards total).
        Holding a 3-set and a pair uses 5 valid cards. Distance should be 1.
        """
        self.ctx.current_round_idx = 0
        cards = [
            Card(Suit.SPADES, Rank.NINE), Card(Suit.HEARTS, Rank.NINE), Card(Suit.CLUBS, Rank.NINE),
            Card(Suit.SPADES, Rank.KING), Card(Suit.HEARTS, Rank.KING)
        ]
        self.ctx.players[0].receive_cards(cards)

        distance = self.reward_calc.calculate_distance_to_win(player_idx=0)
        self.assertEqual(distance, 1, "Failed to calculate distance with a full set and a pair.")

    def test_distance_partial_run(self):
        """
        Round 2 Objective: Two 4s (8 cards total).
        Holding a 3-card sequence uses 3 valid cards. Distance should be 5.
        """
        self.ctx.current_round_idx = 2
        cards = [
            Card(Suit.SPADES, Rank.FIVE), Card(Suit.SPADES, Rank.SIX), Card(Suit.SPADES, Rank.SEVEN)
        ]
        self.ctx.players[0].receive_cards(cards)

        distance = self.reward_calc.calculate_distance_to_win(player_idx=0)
        self.assertEqual(distance, 5, "Failed to calculate distance for a partial run.")

    def test_state_potential_calculation(self):
        """
        Verify the final Phi(s) math converts distance into a negative potential score.
        Formula: -(Distance * win_hunger)
        """
        self.ctx.current_round_idx = 0
        # Pair = 2 cards. Distance = 4.
        cards = [Card(Suit.SPADES, Rank.EIGHT), Card(Suit.HEARTS, Rank.EIGHT)]
        self.ctx.players[0].receive_cards(cards)

        # Expected: -(4 * 25.0) = -100.0
        potential = self.reward_calc.calculate_state_potential(player_idx=0)
        self.assertEqual(potential, -100.0, "State Potential Phi(s) math is incorrect.")


class TestAsymmetricScoring(unittest.TestCase):
    # (Keep your existing TestAsymmetricScoring class here untouched.
    # Terminal scoring is still valid and required for the final game state).
    def setUp(self):
        self.config = JoeConfig(catch_up_multiplier=2.0, pull_ahead_multiplier=0.5)
        self.ctx = GameContext(num_players=4, config=self.config)
        self.reward_calc = RewardCalculator(self.ctx)

    def test_asymmetric_multipliers(self):
        val_trailing = self.reward_calc.calculate_asymmetric_score(50)
        self.assertEqual(val_trailing, 100.0)

        val_leading = self.reward_calc.calculate_asymmetric_score(-20)
        self.assertEqual(val_leading, -10.0)

        val_tied = self.reward_calc.calculate_asymmetric_score(0)
        self.assertEqual(val_tied, 0.0)

    def test_reward_delta_crosses_zero(self):
        start_val = self.reward_calc.calculate_asymmetric_score(40)
        end_val = self.reward_calc.calculate_asymmetric_score(-60)

        reward_delta = end_val - start_val
        self.assertEqual(reward_delta, -110.0)


if __name__ == '__main__':
    unittest.main()