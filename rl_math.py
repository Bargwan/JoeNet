import torch


def compute_td_targets_and_advantages(
        rewards: torch.Tensor,
        values: torch.Tensor,
        is_terminals: torch.Tensor,
        next_value: torch.Tensor,
        gamma: float = 0.99
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes 1-Step Temporal Difference (TD) targets and Advantages
    using vectorized PyTorch operations.

    Args:
        rewards: Tensor of shape (B,) containing the step rewards.
        values: Tensor of shape (B,) containing the Critic's value estimates.
        is_terminals: Boolean Tensor of shape (B,) indicating if the round ended.
        next_value: Tensor containing the Critic's estimate of the state *after* the final step.
        gamma: The discount factor for future rewards.

    Returns:
        td_targets: Tensor of shape (B,)
        advantages: Tensor of shape (B,)
    """
    # 1. Align the "Next Values"
    # Create a tensor to hold V(t+1) for every step in our batch
    next_values = torch.zeros_like(values)

    # For all steps except the last, V(t+1) is simply the Critic's value at the next index
    next_values[:-1] = values[1:]

    # For the final step in the batch, V(t+1) is the explicitly provided next_value
    next_values[-1] = next_value

    # 2. Mask out terminal states
    # If a state is terminal, there is no future! We multiply by 0.0 to kill the gamma term.
    # We use ~is_terminals (logical NOT) and cast it to a float multiplier.
    non_terminal_mask = (~is_terminals).float()

    # 3. Calculate TD Targets: Reward + (Gamma * Next_Value * Non_Terminal)
    td_targets = rewards + (gamma * next_values * non_terminal_mask)

    # 4. Calculate Advantages: TD Target - Current Value
    advantages = td_targets - values

    return td_targets, advantages