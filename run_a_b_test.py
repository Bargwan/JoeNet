from evaluate_arena import TournamentConfig, TournamentRunner
from agents import HeuristicAgent, KeyCardAwareHeuristicAgent

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
    # print("\n" + "="*60)
    # print("   INITIATING HEURISTIC EVALUATION SUITE   ")
    # print("="*60)
    #
    # # ---------------------------------------------------------
    # # TEST 1: (1 State-Aware vs 3 Legacy)
    # # ---------------------------------------------------------
    # agents_test_1 = [
    #     KeyCardAwareHeuristicAgent(random_seed=101),  # Agent 0: The Challenger
    #     HeuristicAgent(random_seed=102),
    #     HeuristicAgent(random_seed=103),
    #     HeuristicAgent(random_seed=104)
    # ]
    # run_scenario("TEST 1: 1 KEYCARD-AWARE vs 3 LEGACY", agents_test_1, num_games)
    #
    # # ---------------------------------------------------------
    # # TEST 2: The Outcast (1 Legacy vs 3 State-Aware)
    # # ---------------------------------------------------------
    # # Will the Legacy bot be crushed by the fast-playing table?
    # agents_test_2 = [
    #     HeuristicAgent(random_seed=201),            # Agent 0: The Hoarder
    #     KeyCardAwareHeuristicAgent(random_seed=202),
    #     KeyCardAwareHeuristicAgent(random_seed=203),
    #     KeyCardAwareHeuristicAgent(random_seed=204)
    # ]
    # run_scenario("TEST 2: 1 LEGACY vs 3 KEYCARD-AWARE", agents_test_2, num_games)

    # ---------------------------------------------------------
    # TEST 3: The New Baseline (4 State-Aware)
    # ---------------------------------------------------------
    # Does the 100-turn gridlock completely disappear?
    agents_test_3 = [
        KeyCardAwareHeuristicAgent(random_seed=301),  # Agent 0: Baseline Check
        KeyCardAwareHeuristicAgent(random_seed=302),
        KeyCardAwareHeuristicAgent(random_seed=303),
        # KeyCardAwareHeuristicAgent(random_seed=304)
    ]
    run_scenario("TEST 3: 3 KEYCARD-AWARE", agents_test_3, num_games)

if __name__ == '__main__':
    run_full_suite(1000)