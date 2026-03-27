import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import numpy as np

# We will build this next
from neural_agent import NeuralAgent


class MockActorCriticNet(nn.Module):
    """A fake JoeNet that perfectly mimics our new Actor-Critic multi-head architecture."""
    def __init__(self):
        super().__init__()

    def forward(self, spatial, scalar, action_mask=None):
        # Create a dummy output tensor of shape (Batch=1, Logits=58)
        logits = torch.ones((1, 58), dtype=torch.float32) * -100.0

        # Primary favorite (e.g., a "junk" floating logit for an illegal move)
        logits[0, 4] = 100.0

        # Secondary favorite (e.g., the actual best legal move)
        logits[0, 10] = 50.0

        # Mimic JoeNet's native ActorNet masking behavior
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        # Dummy outputs for EV (Critic) and Oracle (Belief State)
        ev = torch.zeros((1, 1))
        oracle = torch.zeros((1, 3, 4, 14))

        # MUST return the 3-part tuple defined in our spec
        return logits, ev, oracle


class TestNeuralAgent(unittest.TestCase):

    def setUp(self):
        self.dummy_model = MockActorCriticNet()
        self.agent = NeuralAgent(model=self.dummy_model, device='cpu')

        # Create a mock GameContext that returns expected tensor shapes
        self.mock_ctx = MagicMock()
        self.mock_ctx.get_input_tensor.return_value = {
            'spatial': np.zeros((13, 4, 14), dtype=np.int8),
            'scalar': np.zeros(28, dtype=np.float32)
        }

        # Create a dummy mask (all True)
        self.dummy_mask = np.ones(58, dtype=np.bool_)

    def test_eval_mode_enforced(self):
        """Verify the agent automatically locks the network into evaluation mode."""
        self.assertFalse(
            self.dummy_model.training,
            "NeuralAgent failed to put the model in .eval() mode upon initialization!"
        )

    def test_multi_headed_forward_pass_and_argmax(self):
        """
        Verify the agent successfully processes the multi-head forward pass,
        extracts ONLY the logits, and returns the correct integer action.
        """
        chosen_action = self.agent.select_action(
            state_id='dummy_state',
            ctx=self.mock_ctx,
            player_idx=0,
            action_mask=self.dummy_mask
        )

        # ASSERT: Because all actions are legal in dummy_mask, it should pick the absolute max (Index 4)
        self.assertEqual(
            chosen_action, 4,
            f"Expected agent to choose action 4, but got {chosen_action}"
        )

        # ASSERT: The return type must be a standard Python int, not a PyTorch tensor
        self.assertIsInstance(
            chosen_action, int,
            "Agent must return a standard Python integer, not a Tensor!"
        )

    def test_junk_logit_masking(self):
        """
        Regression Test: Verify that the agent correctly passes the action mask
        to the model to prevent 'floating junk' from causing illegal moves.
        """
        # Create a custom mask where everything is legal EXCEPT index 4
        custom_mask = np.ones(58, dtype=np.bool_)
        custom_mask[4] = False

        chosen_action = self.agent.select_action(
            state_id='dummy_state',
            ctx=self.mock_ctx,
            player_idx=0,
            action_mask=custom_mask
        )

        # Index 4 has a raw logit of 100.0. Index 10 has a raw logit of 50.0.
        # Because index 4 is masked, the model squashes it to -1e9, making index 10 the winner.
        self.assertEqual(
            chosen_action, 10,
            f"Agent chose action {chosen_action} instead of the legal fallback 10!"
        )

    def test_state_id_passed_to_tensor_generator(self):
        """
        Regression Test: Verify that the agent strictly passes the state_id
        to the context to ensure the one-hot phase tensor is correctly encoded.
        """
        self.agent.select_action(
            state_id='test_state_id',
            ctx=self.mock_ctx,
            player_idx=2,
            action_mask=self.dummy_mask
        )
        self.mock_ctx.get_input_tensor.assert_called_once_with(2, 'test_state_id')

    @patch('torch.no_grad')
    def test_no_grad_context_used(self, mock_no_grad):
        """
        Verify inference is strictly wrapped in torch.no_grad() to prevent
        massive GPU memory leaks during long tournaments.
        """
        self.agent.select_action(
            state_id='dummy_state',
            ctx=self.mock_ctx,
            player_idx=0,
            action_mask=self.dummy_mask
        )
        # Verify the context manager was actually invoked
        self.assertTrue(mock_no_grad.called, "Agent did not use torch.no_grad() during inference!")


if __name__ == '__main__':
    unittest.main()