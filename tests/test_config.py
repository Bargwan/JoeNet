import unittest

from config import JoeConfig
from game_context import GameContext


class TestConfigurationInjection(unittest.TestCase):

    def test_joe_config_unscaled_values(self):
        """Verify config initializes with unscaled terminal point values."""
        config = JoeConfig()

        # Checking the standard Rummy unscaled point values
        self.assertEqual(config.points_ace, 20)
        self.assertEqual(config.points_eight_to_king, 10)
        self.assertEqual(config.points_two_to_seven, 5)

    def test_game_context_accepts_config(self):
        """Verify GameContext accepts a JoeConfig instance upon initialization."""
        custom_config = JoeConfig()
        custom_config.points_ace = 50  # Mutate for testing

        # Act
        context = GameContext(num_players=4, config=custom_config)

        # Assert
        self.assertIsNotNone(context.config)
        self.assertEqual(context.config.points_ace,
                         50,
                         "GameContext did not store the injected config.")

    def test_asymmetric_multipliers(self):
        """
        Ensures that the RL asymmetric scoring multipliers are present,
        have the correct defaults, and can be overridden.
        """
        # Test Defaults
        default_config = JoeConfig()
        self.assertTrue(hasattr(default_config, "catch_up_multiplier"),
                        "Missing catch_up_multiplier")
        self.assertTrue(hasattr(default_config, "pull_ahead_multiplier"),
                        "Missing pull_ahead_multiplier")

        self.assertEqual(default_config.catch_up_multiplier, 2.0,
                         "catch_up_multiplier default should be 2.0")
        self.assertEqual(default_config.pull_ahead_multiplier, 0.5,
                         "pull_ahead_multiplier default should be 0.5")

        # Test Overrides
        custom_config = JoeConfig(catch_up_multiplier=3.0, pull_ahead_multiplier=0.1)
        self.assertEqual(custom_config.catch_up_multiplier, 3.0)
        self.assertEqual(custom_config.pull_ahead_multiplier, 0.1)

    def test_turn_and_action_limits(self):
        """
        Verify config cleanly separates board circuits (turns)
        from individual decisions (actions).
        """
        config = JoeConfig()

        self.assertTrue(hasattr(config, 'max_turns'))
        self.assertTrue(hasattr(config, 'max_actions'))

        # A 104-card double deck, minus 44 dealt cards = 60 cards in stock.
        # ~15 circuits of a 4-player table will naturally exhaust the stock pile.
        self.assertEqual(config.max_turns, 30)

        # Actions will greatly outnumber turns due to May-Is and multi-phase decisions.
        self.assertGreater(config.max_actions, config.max_turns * 4,
                           "Action limit must be vastly larger than the turn limit!")


if __name__ == '__main__':
    unittest.main()