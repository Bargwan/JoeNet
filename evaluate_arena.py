import time
import multiprocessing
from dataclasses import dataclass

from tqdm import tqdm

from fast_engine import JoeEngine
from game_context import GameContext
from reward import RewardCalculator


def _get_list_index_from_action(player, action_idx):
    """Translates an absolute neural network logit (6-57) to a relative hand list index."""
    target_suit = (action_idx - 6) // 13
    target_rank = (action_idx - 6) % 13

    for i, card in enumerate(player.hand_list):
        if int(card.suit) == target_suit and int(card.rank) == target_rank:
            return i
    raise ValueError(f"CRITICAL: Card for action {action_idx} not found in player's hand!")


def _log_stalemate_hands(ctx):
    """Dumps the exact hand states of all players to a log file when a stalemate occurs."""
    log_file = "stalemate_tracker.log"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"=== STALEMATE DETECTED (Circuit {ctx.current_circuit}) ===\n")
        req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]
        f.write(f"Round {ctx.current_round_idx} Objective: {req_sets} Sets, {req_runs} Runs\n\n")

        for p in ctx.players:
            # --- THE FIX: Sort by Suit, then Rank, and use the Card.__str__ ---
            sorted_hand = sorted(p.hand_list, key=lambda c: (int(c.suit), int(c.rank)))
            hand_str = ", ".join([str(c) for c in sorted_hand])

            has_obj = ctx.check_hand_objective(p.player_id)

            f.write(f"Player {p.player_id} (Is Down: {p.is_down}):\n")
            f.write(f"  Objective Met: {has_obj}\n")
            f.write(f"  Hand ({len(p.hand_list)} cards): {hand_str}\n")
            f.write("-" * 40 + "\n")

        f.write("\n\n")

def apply_engine_action(engine: JoeEngine, ctx: GameContext, state_id: str, player_idx: int,
                        action_idx: int):
    """Routes absolute logits to the fast_engine using strictly integers and translated list indices."""
    if state_id == 'pickup_decision':
        engine.resolve_pickup(action_idx)
    elif state_id == 'may_i_decision':
        engine.resolve_may_i(action_idx)
    elif state_id == 'go_down_decision':
        engine.resolve_go_down(action_idx)
    elif state_id == 'table_play_phase':
        if action_idx == 5:
            engine.end_table_play()
        else:
            list_idx = _get_list_index_from_action(ctx.players[player_idx], action_idx)
            engine.perform_table_play(list_idx)
    elif state_id == 'discard_phase':
        list_idx = _get_list_index_from_action(ctx.players[player_idx], action_idx)
        engine.perform_discard(list_idx)


# =========================================================================
# MULTIPROCESSING WORKER
# Must be at the top level of the module to avoid Windows Pickling errors
# =========================================================================
def _worker_simulate_single_game(config):
    """Executes EXACTLY ONE game and returns the raw accumulated metrics."""
    target_tournament_wins = 0
    target_round_wins = 0
    point_differentials = []
    down_and_out_wins = 0
    true_strategic_detonations = 0
    stalemates = 0

    round_actions = []
    round_turns = []

    num_players = len(config.agents)

    ctx = GameContext(num_players=num_players)
    engine = JoeEngine(ctx)
    engine.start_game()

    agent_0_went_down_this_turn = False
    last_discarder_idx = -1
    agent_0_projected_score_at_discard = 0.0

    while True:
        raw_state_id = engine.current_state.id
        state_id = raw_state_id.lower().replace(' ', '_').replace('-', '_')

        if state_id == 'game_over':
            ctx.calculate_scores()
            break

        if ctx.current_circuit >= ctx.config.max_turns:
            if config.log_stalemates:
                _log_stalemate_hands(ctx)

            stalemates += 1
            round_actions.append(ctx.total_actions)
            round_turns.append(ctx.current_circuit)

            ctx.calculate_scores()
            if ctx.current_round_idx >= config.rounds_per_game:
                break
            else:
                engine.current_state = engine.dealing
                engine.on_enter_dealing()
                continue

        if state_id == 'dealing':
            engine.deal_cards()
            continue

        if state_id == 'start_turn':
            if ctx.active_player_idx == 0:
                agent_0_went_down_this_turn = False
            engine.enter_pickup()
            continue

        active_idx = ctx.may_i_target_idx if state_id == 'may_i_decision' else ctx.active_player_idx
        agent = config.agents[active_idx]

        mask = ctx.get_action_mask(active_idx, state_id)
        action_idx = agent.select_action(state_id, ctx, active_idx, mask)

        if active_idx == 0 and state_id == 'go_down_decision' and action_idx == 4:
            agent_0_went_down_this_turn = True

        if state_id == 'discard_phase':
            last_discarder_idx = active_idx
            if active_idx == 0:
                calc = RewardCalculator(ctx)
                agent_0_deadwood = calc._calculate_active_deadwood(0)
                agent_0_projected_score_at_discard = ctx.players[0].score + agent_0_deadwood

        prev_round_idx = ctx.current_round_idx
        scores_before = [p.score for p in ctx.players]

        actions_before = ctx.total_actions
        turns_before = ctx.current_circuit

        apply_engine_action(engine, ctx, state_id, active_idx, action_idx)

        if ctx.current_round_idx > prev_round_idx:
            round_actions.append(actions_before)
            round_turns.append(turns_before)
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
        'round_actions': round_actions,
        'round_turns': round_turns
    }


@dataclass
class TournamentConfig:
    name: str
    agents: list
    num_games: int = 100
    rounds_per_game: int = 7
    log_stalemates: bool = False  # <-- NEW: Defaults to False so it doesn't spam standard runs

    def __post_init__(self):
        if not (3 <= len(self.agents) <= 4):
            raise ValueError("A tournament must have exactly 3 or 4 agents.")


class TournamentRunner:
    """
    A high-performance, headless arena loop for evaluating JoeNet Phase 6 Metrics.
    Always evaluates performance from the perspective of Agent 0.
    """

    def __init__(self, config: TournamentConfig):
        self.config = config

    def simulate(self) -> dict:
        """Sequential simulation (Useful for debugging single games)"""
        return self._aggregate_and_print(
            [_worker_simulate_single_game(self.config)], time.time())

    def simulate_parallel(self) -> dict:
        """High-speed multiprocessed simulation using dynamic queue."""
        print(f"\n{'=' * 50}")
        print(f"   {self.config.name.upper()} (PARALLEL)")
        print(f"{'=' * 50}")

        start_time = time.time()

        # --- PYTORCH IPC SAFETY BLOCK ---
        original_devices = []
        for agent in self.config.agents:
            if hasattr(agent, 'model'):
                try:
                    original_devices.append(next(agent.model.parameters()).device)
                    agent.model.to('cpu')
                    agent.model.eval()
                except Exception:
                    original_devices.append(None)
            else:
                original_devices.append(None)

        num_workers = max(1, multiprocessing.cpu_count() - 1)

        # --- NEW: Create exactly 1 task per game for dynamic load balancing ---
        tasks = [self.config for _ in range(self.config.num_games)]

        print(f"Distributing {self.config.num_games} games across {num_workers} cores...")

        # --- NEW: Execute parallel workers with TQDM ---
        with multiprocessing.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_worker_simulate_single_game, tasks),
                total=self.config.num_games,
                desc="Evaluating Arena",
                unit="game",
                ncols=80
            ))

        # --- RESTORE PYTORCH DEVICES ---
        for agent, device in zip(self.config.agents, original_devices):
            if hasattr(agent, 'model') and device is not None:
                agent.model.to(device)

        return self._aggregate_and_print(results, start_time)

    def _aggregate_and_print(self, results: list, start_time: float) -> dict:
        """Helper to combine worker results and print the final metrics."""
        total_tournament_wins = sum(r['tournament_wins'] for r in results)
        total_round_wins = sum(r['round_wins'] for r in results)
        total_down_and_out = sum(r['down_and_out_wins'] for r in results)
        total_detonations = sum(r['strategic_detonations'] for r in results)
        total_stalemates = sum(r['stalemates'] for r in results)

        all_point_diffs = []
        all_actions = []  # NEW
        all_turns = []
        for r in results:
            all_point_diffs.extend(r['point_differentials'])
            all_actions.extend(r['round_actions'])  # NEW
            all_turns.extend(r['round_turns'])

        total_rounds_played = self.config.num_games * self.config.rounds_per_game
        elapsed = time.time() - start_time

        avg_actions = sum(all_actions) / len(all_actions) if all_actions else 0.0
        avg_turns = sum(all_turns) / len(all_turns) if all_turns else 0.0

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
            "stalemates": float(total_stalemates)
        }

        print("\n--- BASELINE RESULTS (Agent 0) ---")
        print(f"Tournament Win Rate:   {final_metrics['tournament_win_rate']:.1f}%")
        print(f"Round Win Rate:        {final_metrics['round_win_rate']:.1f}%")
        print(f"Avg Point Diff:        {final_metrics['avg_point_diff']:.1f} pts")
        print(f"Avg Actions / Round:   {avg_actions:.1f}")  # NEW
        print(f"Avg Turns / Round:     {avg_turns:.1f}")  # NEW
        print(f"Down and Out Wins:     {final_metrics['down_and_out_wins']:.0f}")
        print(f"Strategic Detonations: {final_metrics['strategic_detonations']:.0f}")
        print(f"Stalemates:            {final_metrics['stalemates']:.0f}")
        print(f"Time:                  {elapsed:.2f}s")
        print("=" * 50)

        return final_metrics