import unittest
import torch
from torch.optim import Adam

from network import JoeNet
from rl_trainer import RLTrainer


class TestRLTrainer(unittest.TestCase):

    def setUp(self):
        # Initialize a raw, untrained JoeNet
        self.model = JoeNet()
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.trainer = RLTrainer(self.model, self.optimizer, gamma=0.99)

    def test_rl_update_flows_gradients(self):
        """
        Verify that passing a batch of rollout tensors through the RLTrainer
        successfully computes the Policy Gradient and TD Error, and physically
        updates the network's weights.
        """
        # 1. SETUP: Snapshot the Actor and Critic weights BEFORE the update
        pre_actor_weight = self.model.actor.output_head.weight.clone()
        pre_critic_weight = self.model.critic.output_head.weight.clone()

        # Create a dummy batch of 4 steps (simulating a RolloutBuffer payload)
        batch_size = 4
        dummy_tensors = {
            'spatial': torch.rand(batch_size, 13, 4, 14),
            'scalar': torch.rand(batch_size, 28),
            'mask': torch.ones(batch_size, 58, dtype=torch.bool),
            'action': torch.tensor([5, 12, 3, 50], dtype=torch.long),
            'reward': torch.tensor([0.5, -1.0, 2.5, 10.0], dtype=torch.float32),
            'is_terminal': torch.tensor([False, False, False, True], dtype=torch.bool),
            'oracle_truth': torch.zeros(batch_size, 3, 4, 14, dtype=torch.float32)  # <-- ADDED
        }

        # A dummy Critic prediction for the state occurring immediately after the final step
        dummy_next_value = torch.tensor([0.0], dtype=torch.float32)

        # 2. ACT: Trigger the RL update step
        metrics = self.trainer.update(dummy_tensors, dummy_next_value)

        # 3. ASSERT: Ensure metrics were returned
        self.assertIsInstance(metrics, dict)
        self.assertIn('actor_loss', metrics)
        self.assertIn('critic_loss', metrics)

        # 4. ASSERT: Prove the gradients flowed and the weights physically changed
        post_actor_weight = self.model.actor.output_head.weight
        post_critic_weight = self.model.critic.output_head.weight

        self.assertFalse(
            torch.equal(pre_actor_weight, post_actor_weight),
            "CRITICAL: Actor weights did not change! Policy Gradient is broken."
        )
        self.assertFalse(
            torch.equal(pre_critic_weight, post_critic_weight),
            "CRITICAL: Critic weights did not change! TD Loss is broken."
        )


if __name__ == '__main__':
    unittest.main()