import unittest
import torch
import numpy as np
from unittest.mock import MagicMock

# We will implement this exploration logic in the NeuralAgent
from neural_agent import NeuralAgent


class TestNeuralAgentExploration(unittest.TestCase):

    def setUp(self):
        # Mock a dummy PyTorch model
        self.mock_model = MagicMock()
        self.device = torch.device("cpu")
        self.agent = NeuralAgent(self.mock_model, device=self.device)

    def test_temperature_zero_is_greedy(self):
        """Verify that temperature=0.0 forces a strict argmax (greedy) selection."""
        # Setup: Action 1 is the clear favorite, Action 2 is illegal
        raw_logits = torch.tensor([[1.0, 10.0, 5.0, -10.0]])
        mask = np.array([True, True, True, False])

        # ACT: Compute action with zero temperature
        action = self.agent.compute_action_with_exploration(raw_logits, mask, temperature=0.0)

        # ASSERT: Must perfectly pick the highest valid logit (Index 1)
        self.assertEqual(action, 1)

    def test_temperature_injects_entropy(self):
        """Verify that temperature > 0.0 allows sub-optimal actions to be selected."""
        # Setup: Action 0 and 1 are extremely close in value
        raw_logits = torch.tensor([[10.0, 9.9, -50.0]])
        mask = np.array([True, True, False])

        picked_actions = set()

        # ACT: Sample 100 times with a high temperature
        for _ in range(100):
            action = self.agent.compute_action_with_exploration(raw_logits, mask, temperature=2.0)
            picked_actions.add(action)

        # ASSERT: It should have randomly tried both Action 0 and Action 1
        self.assertIn(0, picked_actions, "Failed to explore the optimal action.")
        self.assertIn(1, picked_actions, "Failed to explore the sub-optimal action.")

    def test_strict_mask_adherence_during_exploration(self):
        """Verify that even with extreme temperature, masked actions are mathematically impossible."""
        # Setup: Action 2 has a massive neural preference, but is strictly ILLEGAL
        raw_logits = torch.tensor([[-5.0, -5.0, 1000.0]])
        mask = np.array([True, True, False])  # Action 2 is masked out!

        picked_actions = set()

        # ACT: Sample 100 times with extreme temperature
        for _ in range(100):
            action = self.agent.compute_action_with_exploration(raw_logits, mask, temperature=100.0)
            picked_actions.add(action)

        # ASSERT: It must NEVER pick Action 2
        self.assertNotIn(2, picked_actions,
                         "CRITICAL: Agent explored into an illegal masked action!")
        self.assertIn(0, picked_actions)
        self.assertIn(1, picked_actions)


if __name__ == '__main__':
    unittest.main()