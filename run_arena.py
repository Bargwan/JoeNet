import argparse

from agents import HeuristicAgent, ProbabilisticAgent
from evaluate_arena import TournamentConfig, TournamentRunner

CHALLENGERS = {
    "benchmark": HeuristicAgent,
    "probabilistic": ProbabilisticAgent,
}


def main():
    parser = argparse.ArgumentParser(description="Joe evaluation arena")
    parser.add_argument("--mode", type=str, required=True, choices=sorted(CHALLENGERS),
                        help="Which agent to test against the HeuristicAgent baseline.")
    parser.add_argument("--games", type=int, default=100, help="Number of games to simulate.")
    parser.add_argument("--players", type=int, default=4, choices=[3, 4], help="Number of players.")

    args = parser.parse_args()
    print(f"\nInitializing {args.players}P Arena | Mode: {args.mode.upper()} | Games: {args.games}")

    # Agent 0 is the challenger; the rest of the table is baseline heuristics.
    agents = [CHALLENGERS[args.mode](random_seed=101)]
    for i in range(1, args.players):
        agents.append(HeuristicAgent(random_seed=100 * (i + 1)))

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
