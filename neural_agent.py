import torch
import numpy as np

from agents import Agent


class NeuralAgent(Agent):
    """
    Wraps the PyTorch JoeNet (Actor-Critic) so it can compete in the Evaluation Arena.
    """

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)

        # Strictly enforce evaluation mode to disable Dropout/BatchNorm layers
        self.model.eval()

    def select_action(self, state_id, ctx, player_idx, action_mask):
        # 1. Grab raw tensors from the game state
        tensors = ctx.get_input_tensor(player_idx, state_id)

        # 2. Preprocess spatial tensor (Cast to float32 and scale presence channels to 0.0 - 1.0)
        spatial_np = tensors['spatial'].astype(np.float32)
        presence_channels = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        spatial_np[presence_channels, :, :] /= 2.0

        # 3. Cast to PyTorch tensors and add the Batch Dimension (B=1)
        spatial = torch.tensor(spatial_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        scalar = torch.tensor(tensors['scalar'], dtype=torch.float32).unsqueeze(0).to(self.device)
        mask = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)

        # 4. Strict no_grad context to prevent massive memory leaks during tournaments
        with torch.no_grad():
            # Forward pass through the Actor-Critic multi-head architecture
            # We unpack the tuple and ignore the Critic (EV) and Oracle for action selection
            logits, ev, oracle_probs = self.model(spatial, scalar, action_mask=mask)

        # 5. Extract the best move strictly from the masked Actor logits
        # Since JoeNet naturally squashes masked illegal moves to -1e9, argmax is perfectly safe
        best_action_idx = torch.argmax(logits, dim=1).item()

        return int(best_action_idx)