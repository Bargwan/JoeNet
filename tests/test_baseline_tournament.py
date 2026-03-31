import unittest
from unittest.mock import patch, MagicMock

# Import the function we want to test
from baseline_tournament import run_baseline


class TestBaselineTournament(unittest.TestCase):

    @patch('baseline_tournament.TournamentRunner')
    @patch('baseline_tournament.torch.load')
    @patch('baseline_tournament.JoeNet')
    def test_run_baseline_wiring(self, mock_joenet_cls, mock_torch_load, mock_runner_cls):
        """
        Verifies that the baseline script correctly initializes the model,
        loads the weights, and executes the TournamentRunner without actually
        running a massive simulation.
        """
        # 1. SETUP MOCKS
        mock_model_instance = mock_joenet_cls.return_value
        mock_runner_instance = mock_runner_cls.return_value

        # 2. ACT
        run_baseline()

        # 3. ASSERT: Model Initialization & Weight Loading
        mock_joenet_cls.assert_called_once()
        mock_torch_load.assert_called_once()
        mock_model_instance.load_state_dict.assert_called_once()

        # 4. ASSERT: Tournament Execution
        mock_runner_cls.assert_called_once()

        # Verify that the runner was initialized with a config containing exactly 4 agents
        config_arg = mock_runner_cls.call_args[0][0]
        self.assertEqual(len(config_arg.agents), 4,
                         "TournamentConfig must be initialized with exactly 4 agents.")

        # Verify the simulation was actually triggered
        mock_runner_instance.simulate_parallel.assert_called_once()


if __name__ == '__main__':
    unittest.main()