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

    def test_turn_and_action_limits(self):
        """
        Verify config cleanly separates board circuits (turns)
        from individual decisions (actions).
        """
        config = JoeConfig()

        self.assertTrue(hasattr(config, 'max_turns'))

        # A 104-card double deck, minus 44 dealt cards = 60 cards in stock.
        # ~15 circuits of a 4-player table will naturally exhaust the stock pile.
        self.assertEqual(config.max_turns, 100)



if __name__ == '__main__':
    unittest.main()