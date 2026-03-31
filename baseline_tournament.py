import torch
import warnings

from network import JoeNet
from neural_agent import NeuralAgent
from agents import HeuristicAgent
from evaluate_arena import TournamentConfig, TournamentRunner

# Suppress PyTorch UserWarnings about non-writable tensors during evaluation
warnings.filterwarnings("ignore", category=UserWarning)

def run_baseline():
    # FORCE CPU: Multiprocessing workers should do local CPU inference
    # to avoid CUDA context crashes and IPC pickling errors on Windows.
    device = torch.device("cpu")
    print(f"Loading Cloned JoeNet on {device}...")

    # 1. Initialize the raw PyTorch model
    model = JoeNet()

    # 2. Load the trained Phase 2 weights
    weights_path = "models/joenet_phase2_cloned.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

    # 3. Wrap it in the standardized Arena Agent interface
    neural_agent = NeuralAgent(model, device=device)

    # 4. Setup the Tournament (Agent 0 is our Neural Net, Agents 1-3 are Heuristic Bots)
    agents = [
        neural_agent,     # Agent 0 (The Clone)
        HeuristicAgent(), # Agent 1 (The Teacher)
        HeuristicAgent(), # Agent 2 (The Teacher)
        HeuristicAgent()  # Agent 3 (The Teacher)
    ]

    config = TournamentConfig(
        name="Phase 2 Neural Clone vs Heuristic Baseline",
        agents=agents,
        num_games=1000,  # A standard 1000-game (7,000 round) sample
        rounds_per_game=7
    )

    # 5. Execute!
    runner = TournamentRunner(config)
    runner.simulate_parallel()


if __name__ == '__main__':
    run_baseline()