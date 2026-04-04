from evaluate_arena import TournamentConfig, TournamentRunner
from agents import HeuristicAgent, OpenHandAgent, ProbabilisticAgent


def run_scenario(name, agents, num_games=1000):
    config = TournamentConfig(
        name=name,
        agents=agents,
        num_games=num_games,
        rounds_per_game=7,
        log_stalemates=True
    )
    runner = TournamentRunner(config)
    runner.simulate_parallel()

def run_full_suite(num_games=1000):
    print("\n" + "="*60)
    print("   INITIATING OPEN-HAND EVALUATION SUITE   ")
    print("="*60)

    # ---------------------------------------------------------
    # TEST 1: The Teacher vs The Baselines (4-Player)
    # ---------------------------------------------------------
    # Tests the omniscient drafting and threat detection in a crowded table.
    agents_test_1 = [
        ProbabilisticAgent(random_seed=101),  # Agent 0: The Omniscient Teacher
        HeuristicAgent(random_seed=102),
        HeuristicAgent(random_seed=103),
        HeuristicAgent(random_seed=104)
    ]
    run_scenario("TEST 1: 1 PROBABILISTIC vs 3 HEURISTIC (4P)", agents_test_1, num_games)

    # ---------------------------------------------------------
    # TEST 2: The Teacher vs The Baselines (3-Player)
    # ---------------------------------------------------------
    # Tests how well it exploits a slightly looser card economy.
    agents_test_2 = [
        ProbabilisticAgent(random_seed=201),  # Agent 0: The Omniscient Teacher
        HeuristicAgent(random_seed=202),
        HeuristicAgent(random_seed=203)
    ]
    run_scenario("TEST 2: 1 PROBABILISTIC vs 2 HEURISTIC (3P)", agents_test_2, num_games)

if __name__ == '__main__':
    # Recommend running 100 first to verify, then 1000 for statistical significance
    run_full_suite(100)