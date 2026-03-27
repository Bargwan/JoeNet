import torch
import torch.nn.functional as F


class DualBrainAgent:
    """
    A routing agent that hosts two separate neural networks in RAM.
    Player 0 acts as the 'Challenger' using the RL model.
    Players 1, 2, and 3 act as the 'Baseline' using the Heuristic clone.
    """

    def __init__(self, rl_model, baseline_model, device="cpu", temperature=0.1):
        self.rl_model = rl_model
        self.baseline_model = baseline_model
        self.device = device
        self.temperature = temperature

    def compute_action_for_player(self, player_idx: int, spatial, scalar, mask) -> int:
        # Route to the correct brain!
        model = self.rl_model if player_idx == 0 else self.baseline_model

        # Convert NumPy arrays to PyTorch tensors
        spatial_t = torch.tensor(spatial, dtype=torch.float32).unsqueeze(0).to(self.device)
        scalar_t = torch.tensor(scalar, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Forward pass: outputs [Logits, Value, Oracle]
            outputs = model(spatial_t, scalar_t, mask_t)
            logits = outputs[0].squeeze(0)

        mask_t = mask_t.squeeze(0)

        # Strictly enforce the action space by setting illegal moves to -infinity
        # The ActorNet outputs 58 masked logits.
        logits[~mask_t] = -1e9

        # Phase 4 Exploitation: Near-greedy action selection
        if self.temperature < 1e-3:
            return torch.argmax(logits).item()
        else:
            probs = F.softmax(logits / self.temperature, dim=-1)
            return torch.multinomial(probs, 1).item()


class MasteryTournament:
    """
    Tracks the macro-level win/loss statistics over a series of games
    to empirically prove superhuman performance.
    """

    def __init__(self, games: int, agent: DualBrainAgent):
        self.games = games
        self.agent = agent

        self.rl_wins = 0
        self.baseline_wins = 0

    def _record_game_result(self, final_scores: list):
        """
        In Joe, the goal is to have the fewest penalty points.
        Negative points are mathematically better.
        """
        winning_score = min(final_scores)

        # Player 0 is our RL Challenger
        if final_scores[0] == winning_score:
            self.rl_wins += 1
        else:
            self.baseline_wins += 1

    def get_statistics(self) -> dict:
        total_decisive = self.rl_wins + self.baseline_wins
        win_rate = (self.rl_wins / total_decisive * 100.0) if total_decisive > 0 else 0.0

        return {
            'rl_wins': self.rl_wins,
            'baseline_wins': self.baseline_wins,
            'rl_win_rate': round(win_rate, 2)
        }