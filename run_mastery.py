import os
import time
import math
import numpy as np
import onnxruntime as ort

from game_context import GameContext
from fast_engine import JoeEngine
from mastery_arena import MasteryTournament
from evaluate_arena import apply_engine_action
from agents import HeuristicAgent


def run_ultimate_mastery_onnx(num_games=1000):
    print("==================================================")
    print(" INITIALIZING ULTIMATE MASTERY TOURNAMENT (ONNX)  ")
    print("==================================================")

    # 1. Load the ONNX Model
    onnx_path = "models/joenet_phase3.onnx"
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Missing ONNX model at {onnx_path}. Did you run export_onnx.py?")

    # Set up ORT session (uses CPU by default, which is incredibly fast for batch-size 1)
    ort_session = ort.InferenceSession(onnx_path)

    # 2. Load the True Python Heuristic Agents
    heuristic_bot = HeuristicAgent()
    tournament = MasteryTournament(games=num_games, agent=None)

    # --- Advanced Metric Trackers ---
    adv_stats = {
        'rl_round_wins': 0,
        'heuristic_round_wins': 0,
        'stalemates': 0,
        'round_actions': [],
        'round_turns': []
    }

    print(f"\n--- Commencing {num_games}-Game Deathmatch ---")
    print("Player 0: Phase 3 RL Challenger (ONNX C++ Runtime)")
    print("Players 1, 2, 3: The True Heuristic Agents (Python)")

    start_time = time.time()

    for game in range(1, num_games + 1):
        ctx = GameContext(num_players=4)
        engine = JoeEngine(ctx)
        engine.start_game()

        while True:
            state_id = engine.current_state.id

            if state_id == 'game_over':
                ctx.calculate_scores()
                final_scores = [float(p.score) for p in ctx.players]
                tournament._record_game_result(final_scores)
                break

            if state_id == 'dealing':
                engine.deal_cards()
                continue

            if state_id == 'start_turn':
                engine.enter_pickup()
                continue

            # --- Failsafe / Stalemate Tracker ---
            if ctx.current_circuit >= ctx.config.max_turns:
                adv_stats['stalemates'] += 1
                adv_stats['round_actions'].append(ctx.total_actions)
                adv_stats['round_turns'].append(ctx.current_circuit)

                ctx.calculate_scores()
                if ctx.current_round_idx >= 7:
                    final_scores = [float(p.score) for p in ctx.players]
                    tournament._record_game_result(final_scores)
                    break
                else:
                    engine.current_state = engine.dealing
                    engine.on_enter_dealing()
                    continue

            active_idx = ctx.may_i_target_idx if state_id == 'may_i_decision' else ctx.active_player_idx
            mask_np = ctx.get_action_mask(active_idx, state_id)

            # --- THE MIXED ROUTER (ONNX EDITION) ---
            if active_idx == 0:
                nn_inputs = ctx.get_input_tensor(active_idx, state_id)

                # Expand dimensions to create a batch size of 1 for ORT
                spatial_ort = np.expand_dims(nn_inputs['spatial'], axis=0).astype(np.float32)
                scalar_ort = np.expand_dims(nn_inputs['scalar'], axis=0).astype(np.float32)
                mask_ort = np.expand_dims(mask_np, axis=0).astype(np.bool_)

                ort_inputs = {
                    'spatial': spatial_ort,
                    'scalar': scalar_ort,
                    'mask': mask_ort
                }

                # Blast it through the C++ Runtime
                outputs = ort_session.run(None, ort_inputs)
                logits = outputs[0][
                    0]  # Grab the first output (logits) and strip the batch dimension

                # Enforce action space and select greedy action
                logits[~mask_np] = -1e9
                action_idx = int(np.argmax(logits))
            else:
                action_idx = heuristic_bot.select_action(state_id, ctx, active_idx, mask_np)

            # Apply action
            apply_engine_action(engine, ctx, state_id, active_idx, action_idx)

            # --- Track Natural Round Ends (Detonations) ---
            new_state = engine.current_state.id
            if new_state == 'dealing' or new_state == 'game_over':
                winner_idx = next((i for i, p in enumerate(ctx.players) if len(p.hand_list) == 0),
                                  -1)

                if winner_idx != -1:
                    if winner_idx == 0:
                        adv_stats['rl_round_wins'] += 1
                    else:
                        adv_stats['heuristic_round_wins'] += 1

                    adv_stats['round_actions'].append(ctx.total_actions)
                    adv_stats['round_turns'].append(ctx.current_circuit)

        if game % 100 == 0:
            stats = tournament.get_statistics()
            avg_actions = np.mean(adv_stats['round_actions']) if adv_stats['round_actions'] else 0
            avg_turns = np.mean(adv_stats['round_turns']) if adv_stats['round_turns'] else 0

            print(
                f"Game {game:04d} | RL Wins: {stats['rl_wins']} | Heur Wins: {stats['baseline_wins']} | Win Rate: {stats['rl_win_rate']}%")
            print(
                f"  -> Round Stats | Avg Actions: {avg_actions:.1f} | Avg Turns: {avg_turns:.1f} | RL Dets: {adv_stats['rl_round_wins']} | Heur Dets: {adv_stats['heuristic_round_wins']} | Stalemates: {adv_stats['stalemates']}")

    # --- Final Statistics Output ---
    elapsed = time.time() - start_time
    stats = tournament.get_statistics()
    wins = stats['rl_wins']
    p_hat = wins / num_games
    p_0 = 0.25
    standard_error = math.sqrt((p_0 * (1.0 - p_0)) / num_games)
    z_score = (p_hat - p_0) / standard_error if standard_error > 0 else 0.0

    avg_actions = np.mean(adv_stats['round_actions']) if adv_stats['round_actions'] else 0
    avg_turns = np.mean(adv_stats['round_turns']) if adv_stats['round_turns'] else 0
    total_rounds = len(adv_stats['round_actions'])
    total_dets = adv_stats['rl_round_wins'] + adv_stats['heuristic_round_wins']

    print("\n==================================================")
    print("          ULTIMATE TOURNAMENT COMPLETE            ")
    print("==================================================")
    print(f"Total Games Played:      {num_games}")
    print(f"RL Challenger Wins:      {wins}")
    print(f"Heuristic Baselines:     {stats['baseline_wins']}")
    print(f"Final RL Game Win Rate:  {stats['rl_win_rate']}%")
    print(f"Time Elapsed:            {elapsed:.2f} seconds")
    print("--------------------------------------------------")
    print("ROUND-LEVEL METRICS:")
    print(f"Total Rounds Played:     {total_rounds}")
    print(f"Avg Actions / Round:     {avg_actions:.1f}")
    print(f"Avg Turns / Round:       {avg_turns:.1f}")
    if total_rounds > 0:
        print(
            f"Total Detonations:       {total_dets} ({(total_dets / total_rounds * 100):.1f}% of rounds)")
        print(f"  -> RL Detonations:     {adv_stats['rl_round_wins']}")
        print(f"  -> Heuristic Dets:     {adv_stats['heuristic_round_wins']}")
        print(
            f"Stalemates (Max Turns):  {adv_stats['stalemates']} ({(adv_stats['stalemates'] / total_rounds * 100):.1f}% of rounds)")
    print("==================================================")
    print(f"Statistical Z-Score:     {z_score:.3f}")

    if z_score >= 2.326:
        print("\n🏆 UNDENIABLE MASTERY (99% Confidence) 🏆")
        print("The RL Agent has mathematically crushed the baseline.")
    elif z_score >= 1.645:
        print("\n🎉 SUPERHUMAN CONFIRMED (95% Confidence) 🎉")
        print("The RL Agent is statistically superior to the baseline.")
    else:
        print("\n⚠️ NO STATISTICAL SIGNIFICANCE ⚠️")
        print("The RL win rate falls within the margin of error for random chance.")
        print("More Phase 3 training is required to prove dominance.")


if __name__ == '__main__':
    run_ultimate_mastery_onnx(num_games=1000)