import argparse
import os
from evaluate_arena import TournamentConfig, TournamentRunner
from agents import HeuristicAgent, ONNXAgent


def run_ultimate_mastery(onnx_path, num_games=100, num_players=4):
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Missing ONNX model at {onnx_path}.")

    print(f"Initializing Phase 3 Arena: {num_players}P | {num_games} Games")

    # =========================================================
    # THE UNIFIED TOURNAMENT
    # =========================================================
    agents = [ONNXAgent(onnx_path)]  # Player 0: The Challenger

    # Fill the rest of the table with Independent Heuristic Teachers
    for i in range(1, num_players):
        agents.append(HeuristicAgent(random_seed=100 * i))

    config = TournamentConfig(
        name=f"MASTERY EVALUATION ({num_players}P)",
        agents=agents,
        num_games=num_games,
        rounds_per_game=7
    )

    # Let the Gold Standard arena handle everything else!
    runner = TournamentRunner(config)
    runner.simulate_parallel()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the JoeNet Arena")
    parser.add_argument("--onnx", type=str, required=True, help="Path to the ONNX model")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--players", type=int, default=4, help="Number of players (3 or 4)")

    args = parser.parse_args()
    run_ultimate_mastery(args.onnx, args.games, args.players)