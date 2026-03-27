import time
from dataclasses import dataclass, field
from typing import List

from fast_engine import JoeEngine
from game_context import GameContext


def _get_list_index_from_action(player, action_idx):
    """Translates an absolute neural network logit (6-57) to a relative hand list index."""
    target_suit = (action_idx - 6) // 13
    target_rank = (action_idx - 6) % 13

    for i, card in enumerate(player.hand_list):
        if int(card.suit) == target_suit and int(card.rank) == target_rank:
            return i
    raise ValueError(f"CRITICAL: Card for action {action_idx} not found in player's hand!")


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


@dataclass
class TournamentConfig:
    name: str
    agents: list
    num_games: int = 100
    rounds_per_game: int = 7

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
        print(f"\n{'=' * 50}")
        print(f"   {self.config.name.upper()}")
        print(f"{'=' * 50}")

        start_time = time.time()

        # Phase 6 Target Metrics (Agent 0 perspective)
        target_tournament_wins = 0
        target_round_wins = 0
        point_differentials = []
        strategic_detonations = 0

        num_players = len(self.config.agents)

        for game in range(self.config.num_games):
            ctx = GameContext(num_players=num_players)
            engine = JoeEngine(ctx)
            engine.start_game()

            # Temporary trackers for the current game
            game_detonations = 0
            agent_0_went_down_this_turn = False

            while True:
                state_id = engine.current_state.id

                if state_id == 'game_over':
                    break

                # Circuit failsafe prevents infinite loops
                if ctx.current_circuit >= ctx.config.max_turns:
                    ctx.calculate_scores()

                    # If this forces the game to end, break
                    if ctx.current_round_idx >= self.config.rounds_per_game:
                        break
                    else:
                        engine.current_state = engine.dealing
                        engine.on_enter_dealing()
                        continue

                if state_id == 'dealing':
                    engine.deal_cards()
                    continue

                if state_id == 'start_turn':
                    # Reset the turn-specific detonation tracker
                    if ctx.active_player_idx == 0:
                        agent_0_went_down_this_turn = False
                    engine.enter_pickup()
                    continue

                # Determine active player and query their agent
                active_idx = ctx.may_i_target_idx if state_id == 'may_i_decision' else ctx.active_player_idx
                agent = self.config.agents[active_idx]

                mask = ctx.get_action_mask(active_idx, state_id)
                action_idx = agent.select_action(state_id, ctx, active_idx, mask)

                # Track if Agent 0 goes down THIS turn for the Detonation metric
                if active_idx == 0 and state_id == 'go_down_decision' and action_idx == 4:
                    agent_0_went_down_this_turn = True

                # Save the round index BEFORE the action
                prev_round_idx = ctx.current_round_idx

                # Apply the action
                apply_engine_action(engine, ctx, state_id, active_idx, action_idx)

                # Check if the action natively triggered a round transition
                if ctx.current_round_idx > prev_round_idx:
                    if active_idx == 0:
                        target_round_wins += 1

                        # Phase 6 Detonation: Won the round on the exact same turn they went down
                        if agent_0_went_down_this_turn:
                            game_detonations += 1

            # --- Game Concluded: Compile Tournament Metrics ---
            scores = [p.score for p in ctx.players]

            # 1. Tournament Win %
            if scores[0] == min(scores):
                target_tournament_wins += 1

            # 2. Average Point Differential (Target Score minus Avg Opponent Score)
            # A negative differential is good (means Agent 0 had fewer penalty points)
            avg_opp_score = sum(scores[1:]) / (num_players - 1)
            point_differentials.append(scores[0] - avg_opp_score)

            # 3. Strategic Detonations
            strategic_detonations += game_detonations

        # Calculate final aggregated stats
        total_rounds_played = self.config.num_games * self.config.rounds_per_game
        elapsed = time.time() - start_time

        results = {
            "tournament_win_rate": (
                                               target_tournament_wins / self.config.num_games) * 100.0 if self.config.num_games > 0 else 0.0,
            "round_win_rate": (
                                          target_round_wins / total_rounds_played) * 100.0 if total_rounds_played > 0 else 0.0,
            "avg_point_diff": sum(point_differentials) / len(
                point_differentials) if point_differentials else 0.0,
            "strategic_detonations": float(strategic_detonations),
            "simulation_time": elapsed
        }

        print("\n--- BASELINE RESULTS (Agent 0) ---")
        print(f"Tournament Win Rate:   {results['tournament_win_rate']:.1f}%")
        print(f"Round Win Rate:        {results['round_win_rate']:.1f}%")
        print(f"Avg Point Diff:        {results['avg_point_diff']:.1f} pts")
        print(f"Strategic Detonations: {results['strategic_detonations']:.0f}")
        print(f"Time:                  {elapsed:.2f}s")
        print("=" * 50)

        return results