from evaluate_arena import TournamentConfig, TournamentRunner
from agents import HeuristicAgent


def main():
    print("Initializing Baseline Heuristic 'Mexican Standoff' Test...")

    # =========================================================
    # TOURNAMENT: 3 Heuristic Agents (Pure Baseline)
    # =========================================================
    agents = [
        HeuristicAgent(random_seed=101),  # Player 0: Heuristic
        HeuristicAgent(random_seed=202),  # Player 1: Heuristic
        HeuristicAgent(random_seed=303),  # Player 2: Heuristic
        # HeuristicAgent(random_seed=404)   # Player 3: Heuristic
    ]

    config = TournamentConfig(
        name="BASELINE TEST: 4 Pure Heuristics",
        agents=agents,
        num_games=1000,
        rounds_per_game=7
    )

    runner = TournamentRunner(config)
    runner.simulate_parallel()


if __name__ == '__main__':
    main()