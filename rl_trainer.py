import torch
import torch.nn.functional as F
from rl_math import compute_td_targets_and_advantages


class RLTrainer:
    """
    Orchestrates the Reinforcement Learning backpropagation step
    using Policy Gradients (Actor) and TD Error (Critic).
    """

    def __init__(self, model, optimizer, gamma=0.99):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma

    def update(self, batch_tensors: dict, next_value: torch.Tensor) -> dict:
        """
        Computes the losses and updates the network weights.
        """
        # Dynamically grab the device the model is currently sitting on
        device = next(self.model.parameters()).device

        # Extract and immediately push all tensors to the correct device
        spatial = batch_tensors['spatial'].to(device)
        scalar = batch_tensors['scalar'].to(device)
        mask = batch_tensors['mask'].to(device)
        actions = batch_tensors['action'].to(device)
        rewards = batch_tensors['reward'].to(device)
        is_terminals = batch_tensors['is_terminal'].to(device)

        # Clamp to enforce strict binary presence (0.0 or 1.0) for the BCE Loss
        oracle_truths = batch_tensors['oracle_truth'].to(device).clamp(min=0.0, max=1.0)

        # 1. Forward Pass to get current network predictions
        # We assume JoeNet returns a tuple: (logits, values, oracle_probs)
        outputs = self.model(spatial, scalar, mask)
        logits = outputs[0]
        values = outputs[1].squeeze(-1)  # Flatten (B, 1) to (B,)
        oracle_probs = outputs[2]

        # 2. Compute TD Targets and Advantages
        # CRITICAL: We detach() the values here. We do not want the Actor's
        # policy gradient to flow backward through the Critic's historical predictions.
        td_targets, advantages = compute_td_targets_and_advantages(
            rewards, values.detach(), is_terminals, next_value, self.gamma
        )

        # 3. Critic Loss (Mean Squared Error)
        # The Critic learns to push its Value predictions closer to the TD Target
        critic_loss = F.mse_loss(values, td_targets)

        # 4. Actor Loss (Policy Gradient)
        # Convert raw logits to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Extract the log probabilities of the exact actions the agent actually took
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Standard RL Stability Trick: Normalize advantages to have mean 0 and std 1
        if advantages.shape[0] > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy Gradient formula: Maximize expected reward -> Minimize -log(P(a|s)) * Advantage
        actor_loss = -(action_log_probs * advantages).mean()
        oracle_loss = F.binary_cross_entropy(oracle_probs, oracle_truths)

        # 5. Backpropagation
        # --- FIXED: Reconnect the Oracle with a massive defensive weight ---
        oracle_weight = 1  # Scales the ~0.65 loss to ~6500.0 to fight the Critic
        total_loss = actor_loss + critic_loss + (oracle_loss * oracle_weight)

        self.optimizer.zero_grad()
        total_loss.backward()

        # --- FIXED: Gradient Clipping to prevent massive penalty spikes from exploding weights ---
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

        self.optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'oracle_loss': oracle_loss.item()
        }