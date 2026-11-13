"""
The evaluation arena: plays full games between agents (using fast_engine) and
reports results from the point of view of agent 0, the agent being tested.
"""
import multiprocessing
import time
from dataclasses import dataclass

from tqdm import tqdm

from actions import (
    CallMayI,
    Discard,
    EndTablePlay,
    GoDown,
    Pass,
    PickDiscard,
    PickStock,
    Place,
    Wait,
)
from fast_engine import JoeEngine
from game_context import (
    CALL_MAY_I,
    END_TABLE_PLAY,
    GO_DOWN,
    PASS,
    PICK_DISCARD,
    PICK_STOCK,
    WAIT,
    GameContext,
    discard_action,
    place_action,
)


def _log_round_cap_hands(ctx):
    """Dumps every player's hand to a log file when a round hits the max_turns safety net."""
    log_file = "round_cap_tracker.log"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"=== ROUND CAP REACHED (Circuit {ctx.current_circuit}) ===\n")
        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx - 1]
        f.write(f"Round {ctx.current_round_idx} Objective: {req_sets} Sets, {req_runs} Runs\n\n")

        for p in ctx.players:
            # Sorted by suit, then rank
            sorted_hand = sorted(p.hand_list, key=lambda c: (int(c.suit), int(c.rank)))
            hand_str = ", ".join([str(c) for c in sorted_hand])

            has_obj = ctx.check_hand_objective(p.player_id)

            f.write(f"Player {p.player_id} (Is Down: {p.is_down}):\n")
            f.write(f"  Objective Met: {has_obj}\n")
            f.write(f"  Hand ({len(p.hand_list)} cards): {hand_str}\n")
            f.write("-" * 40 + "\n")

        f.write("\n\n")

def apply_engine_action(engine: JoeEngine, ctx: GameContext, state_id: str, player_idx: int,
                        action_idx: int, agent=None):
    """
    Temporary adapter while agents still return action numbers: converts the
    number to the matching action object and plays it with engine.apply().
    For GO_DOWN, the agent (if given) chooses the melds.
    """
    engine.apply(_action_from_number(ctx, state_id, player_idx, action_idx, agent))


def _action_from_number(ctx, state_id, player_idx, number, agent):
    legal = ctx.legal_actions(player_idx, state_id)
    if state_id == 'go_down_decision' and number == GO_DOWN:
        melds = agent.choose_go_down_melds(ctx, player_idx) if agent is not None else None
        if melds is None:
            return next(a for a in legal if isinstance(a, GoDown))
        return GoDown(tuple(tuple(m) for m in melds))
    for action in legal:
        if _action_number(action, state_id) == number:
            return action
    raise ValueError(f"Action number {number} is not legal in {state_id}.")


def _action_number(action, state_id):
    """The old action number for an action object (temporary, see apply_engine_action)."""
    if isinstance(action, Place):
        return place_action(action.suit, action.rank, action.destination)
    if isinstance(action, Discard):
        return discard_action(action.suit, action.rank)
    if isinstance(action, (Wait, EndTablePlay)):
        return WAIT if state_id == 'go_down_decision' else END_TABLE_PLAY
    return {PickStock: PICK_STOCK, PickDiscard: PICK_DISCARD, CallMayI: CALL_MAY_I,
            Pass: PASS, GoDown: GO_DOWN}[type(action)]


# =========================================================================
# MULTIPROCESSING WORKER
# Must be at module level so multiprocessing can pickle it.
# =========================================================================
def _worker_simulate_single_game(config):
    """
    Plays one full game and returns agent 0's raw results:
    - tournament_wins: 1 if agent 0 finished with the lowest score (ties count).
    - round_wins: rounds agent 0 won by going out.
    - point_differentials: agent 0's final score minus the opponents' average.
    - down_and_out_wins: round wins where agent 0 went down and out on the same turn.
    - strategic_detonations: rounds another player won straight after agent 0's
      discard, without discarding themselves, where agent 0 was ahead of the
      winner when it discarded and lost fewer points than the other losers.
    - stalemates, round_caps: rounds ended by a stalemate or by the turn cap.
    - round_turns, round_circuits: turns and full circuits played in each round.
    """
    target_tournament_wins = 0
    target_round_wins = 0
    point_differentials = []
    down_and_out_wins = 0
    true_strategic_detonations = 0
    stalemates = 0
    round_caps = 0

    round_turns = []
    round_circuits = []

    num_players = len(config.agents)

    ctx = GameContext(num_players=num_players)
    engine = JoeEngine(ctx)
    engine.advance()

    agent_0_went_down_this_turn = False
    last_discarder_idx = -1
    agent_0_projected_score_at_discard = 0.0

    # The engine scores each round itself; the loop ends when the game is over.
    while (decision := engine.decider()) is not None:
        state_id, active_idx = decision
        if state_id == 'pickup_decision' and active_idx == 0:
            agent_0_went_down_this_turn = False  # agent 0's turn is starting

        agent = config.agents[active_idx]

        mask = ctx.get_action_mask(active_idx, state_id)
        action_idx = agent.select_action(state_id, ctx, active_idx, mask)

        if active_idx == 0 and state_id == 'go_down_decision' and action_idx == GO_DOWN:
            agent_0_went_down_this_turn = True

        if state_id == 'discard_phase':
            last_discarder_idx = active_idx
            if active_idx == 0:
                agent_0_deadwood = ctx.hand_penalty(0)
                agent_0_projected_score_at_discard = ctx.players[0].score + agent_0_deadwood

        prev_round_idx = ctx.current_round_idx
        scores_before = [p.score for p in ctx.players]

        turns_before = ctx.total_actions
        circuits_before = ctx.current_circuit

        apply_engine_action(engine, ctx, state_id, active_idx, action_idx, agent)

        if ctx.current_round_idx > prev_round_idx:
            round_turns.append(turns_before)
            round_circuits.append(circuits_before)
            if ctx.last_round_result == 'stalemate':
                stalemates += 1
                continue
            if ctx.last_round_result == 'cap':
                round_caps += 1
                if config.log_round_caps:
                    _log_round_cap_hands(ctx)
                continue
            if active_idx == 0:
                target_round_wins += 1
                if agent_0_went_down_this_turn:
                    down_and_out_wins += 1
            else:
                if last_discarder_idx == 0:
                    winner_idx = active_idx
                    opponent_projected = scores_before[winner_idx]
                    negative_danger_achieved = (agent_0_projected_score_at_discard < opponent_projected)
                    scores_after = [p.score for p in ctx.players]
                    round_penalties = [scores_after[i] - scores_before[i] for i in range(num_players)]
                    agent_0_penalty = round_penalties[0]
                    other_losers_penalties = [round_penalties[i] for i in range(1, num_players) if i != winner_idx]
                    avg_collateral_damage = sum(other_losers_penalties) / len(other_losers_penalties) if other_losers_penalties else 0.0

                    if negative_danger_achieved and (agent_0_penalty < avg_collateral_damage):
                        true_strategic_detonations += 1

    final_scores = [p.score for p in ctx.players]
    if final_scores[0] == min(final_scores):
        target_tournament_wins += 1

    avg_opp_score = sum(final_scores[1:]) / (num_players - 1)
    point_differentials.append(final_scores[0] - avg_opp_score)

    return {
        'tournament_wins': target_tournament_wins,
        'round_wins': target_round_wins,
        'point_differentials': point_differentials,
        'down_and_out_wins': down_and_out_wins,
        'strategic_detonations': true_strategic_detonations,
        'stalemates': stalemates,
        'round_caps': round_caps,
        'round_turns': round_turns,
        'round_circuits': round_circuits,
    }


@dataclass
class TournamentConfig:
    """
    An arena run: the agents in seat order (agent 0 is the one evaluated) and
    how many games to play. Every game is a full 7 rounds; rounds_per_game is
    only used to turn round counts into rates.
    """
    name: str
    agents: list
    num_games: int = 100
    rounds_per_game: int = 7
    log_round_caps: bool = False  # Log the hands of rounds ended by the max_turns safety net

    def __post_init__(self):
        if not (3 <= len(self.agents) <= 4):
            raise ValueError("A tournament must have exactly 3 or 4 agents.")


class TournamentRunner:
    """Runs a TournamentConfig's games and reports agent 0's results."""

    def __init__(self, config: TournamentConfig):
        self.config = config

    def simulate(self) -> dict:
        """Plays one game in this process (useful for debugging)."""
        return self._aggregate_and_print(
            [_worker_simulate_single_game(self.config)], time.time())

    def simulate_parallel(self) -> dict:
        """Plays all the games across CPU cores, one game per task."""
        print(f"\n{'=' * 50}")
        print(f"   {self.config.name.upper()} (PARALLEL)")
        print(f"{'=' * 50}")

        start_time = time.time()

        num_workers = max(1, multiprocessing.cpu_count() - 1)

        # One task per game, so idle cores pick up the next game
        tasks = [self.config for _ in range(self.config.num_games)]

        print(f"Distributing {self.config.num_games} games across {num_workers} cores...")

        with multiprocessing.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_worker_simulate_single_game, tasks),
                total=self.config.num_games,
                desc="Evaluating Arena",
                unit="game",
                ncols=80
            ))

        return self._aggregate_and_print(results, start_time)

    def _aggregate_and_print(self, results: list, start_time: float) -> dict:
        """Combines the games' results, prints them and returns the summary metrics."""
        total_tournament_wins = sum(r['tournament_wins'] for r in results)
        total_round_wins = sum(r['round_wins'] for r in results)
        total_down_and_out = sum(r['down_and_out_wins'] for r in results)
        total_detonations = sum(r['strategic_detonations'] for r in results)
        total_stalemates = sum(r['stalemates'] for r in results)
        total_round_caps = sum(r['round_caps'] for r in results)

        all_point_diffs = []
        all_turns = []
        all_circuits = []
        for r in results:
            all_point_diffs.extend(r['point_differentials'])
            all_turns.extend(r['round_turns'])
            all_circuits.extend(r['round_circuits'])

        total_rounds_played = self.config.num_games * self.config.rounds_per_game
        elapsed = time.time() - start_time

        avg_turns = sum(all_turns) / len(all_turns) if all_turns else 0.0
        avg_circuits = sum(all_circuits) / len(all_circuits) if all_circuits else 0.0

        final_metrics = {
            "tournament_win_rate": (
                                               total_tournament_wins / self.config.num_games) * 100.0 if self.config.num_games > 0 else 0.0,
            "round_win_rate": (
                                          total_round_wins / total_rounds_played) * 100.0 if total_rounds_played > 0 else 0.0,
            "avg_point_diff": sum(all_point_diffs) / len(
                all_point_diffs) if all_point_diffs else 0.0,
            "down_and_out_wins": float(total_down_and_out),
            "strategic_detonations": float(total_detonations),
            "simulation_time": elapsed,
            "stalemates": float(total_stalemates),
            "round_caps": float(total_round_caps),
        }

        print("\n--- BASELINE RESULTS (Agent 0) ---")
        print(f"Tournament Win Rate:   {final_metrics['tournament_win_rate']:.1f}%")
        print(f"Round Win Rate:        {final_metrics['round_win_rate']:.1f}%")
        print(f"Avg Point Diff:        {final_metrics['avg_point_diff']:.1f} pts")
        print(f"Avg Turns / Round:     {avg_turns:.1f}")
        print(f"Avg Circuits / Round:  {avg_circuits:.1f}")
        print(f"Down and Out Wins:     {final_metrics['down_and_out_wins']:.0f}")
        print(f"Strategic Detonations: {final_metrics['strategic_detonations']:.0f}")
        print(f"Stalemates:            {final_metrics['stalemates']:.0f}")
        print(f"Rounds Hit Turn Cap:   {final_metrics['round_caps']:.0f}")
        print(f"Time:                  {elapsed:.2f}s")
        print("=" * 50)

        return final_metrics