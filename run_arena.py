import argparse
import os
import torch

from evaluate_arena import TournamentConfig, TournamentRunner
from agents import HeuristicAgent, ProbabilisticAgent, ONNXAgent
from neural_agent import NeuralAgent
from network import JoeNet


def main():
    parser = argparse.ArgumentParser(description="Unified JoeNet Evaluation Arena")
    parser.add_argument("--mode", type=str, required=True,
                        choices=['benchmark', 'pytorch', 'probabilistic', 'onnx'],
                        help="Which agent to test against the baseline.")
    parser.add_argument("--games", type=int, default=100, help="Number of games to simulate.")
    parser.add_argument("--players", type=int, default=4, choices=[3, 4], help="Number of players.")
    parser.add_argument("--model_path", type=str, default="", help="Path to .pth or .onnx file.")

    args = parser.parse_args()
    print(f"\nInitializing {args.players}P Arena | Mode: {args.mode.upper()} | Games: {args.games}")

    agents = []

    # --- 1. Load the Challenger (Agent 0) based on Mode ---
    if args.mode == 'benchmark':
        agents.append(HeuristicAgent(random_seed=101))

    elif args.mode == 'probabilistic':
        agents.append(ProbabilisticAgent(random_seed=101))

    elif args.mode == 'pytorch':
        device = torch.device("cpu")
        model = JoeNet()
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        agents.append(NeuralAgent(model, device=device))

    elif args.mode == 'onnx':
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Missing ONNX model at {args.model_path}.")
        agents.append(ONNXAgent(args.model_path))

    # --- 2. Fill the rest of the table with Baseline Heuristics ---
    for i in range(1, args.players):
        agents.append(HeuristicAgent(random_seed=100 * (i + 1)))

    # --- 3. Execute ---
    config = TournamentConfig(
        name=f"{args.mode.upper()} EVALUATION",
        agents=agents,
        num_games=args.games,
        rounds_per_game=7
    )

    runner = TournamentRunner(config)
    runner.simulate_parallel()


if __name__ == '__main__':
    main()