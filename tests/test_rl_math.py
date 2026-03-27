import unittest
import torch

# We will build this in the Green phase
from rl_math import compute_td_targets_and_advantages


class TestRLMath(unittest.TestCase):

    def test_td_target_and_advantage_calculation(self):
        """
        Verify the core math for Actor-Critic updates:
        TD Target = Reward + (Gamma * Next_Value * (1 - Terminal))
        Advantage = TD_Target - Current_Value
        """
        # 1. SETUP: A simulated 3-step sequence
        # Step 0: Normal move
        # Step 1: Normal move
        # Step 2: Terminal move (Game Over)
        rewards = torch.tensor([1.0, 2.0, 10.0], dtype=torch.float32)
        values = torch.tensor([0.5, 1.5, 8.0], dtype=torch.float32)
        is_terminals = torch.tensor([False, False, True], dtype=torch.bool)

        # The value of the state *after* the final step (used for bootstrapping if not terminal)
        # Since step 2 is terminal, this should be mathematically ignored by the function.
        next_value = torch.tensor([5.0], dtype=torch.float32)

        gamma = 0.9  # Discount factor

        # 2. ACT
        td_targets, advantages = compute_td_targets_and_advantages(
            rewards, values, is_terminals, next_value, gamma
        )

        # 3. ASSERT: Calculate the expected math manually

        # --- Step 2 (Terminal) ---
        # Target: 10.0 + (0.9 * 5.0 * 0) = 10.0
        # Advantage: 10.0 - 8.0 = 2.0
        self.assertAlmostEqual(td_targets[2].item(), 10.0, places=4)
        self.assertAlmostEqual(advantages[2].item(), 2.0, places=4)

        # --- Step 1 (Normal) ---
        # Target: 2.0 + (0.9 * 8.0) = 9.2
        # Advantage: 9.2 - 1.5 = 7.7
        self.assertAlmostEqual(td_targets[1].item(), 9.2, places=4)
        self.assertAlmostEqual(advantages[1].item(), 7.7, places=4)

        # --- Step 0 (Normal) ---
        # Target: 1.0 + (0.9 * 1.5) = 2.35
        # Advantage: 2.35 - 0.5 = 1.85
        self.assertAlmostEqual(td_targets[0].item(), 2.35, places=4)
        self.assertAlmostEqual(advantages[0].item(), 1.85, places=4)

        # Verify output shapes match the input batch size exactly
        self.assertEqual(td_targets.shape, (3,))
        self.assertEqual(advantages.shape, (3,))


if __name__ == '__main__':
    unittest.main()