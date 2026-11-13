import os
import random

from evaluate_arena import apply_engine_action

# --- SHATTER NUMPY THREAD LOCKS ---
# This prevents numpy from spawning massive internal threads inside our multiprocessing workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import signal
import multiprocessing
import numpy as np

from config import JoeConfig
from game_context import GameContext
from fast_engine import JoeEngine
from agents import OpenHandAgent as HeuristicAgent
from buffers import JoeReplayBuffer


def _mute_worker_keyboard_interrupt():
    """Forces child processes to ignore Ctrl-C, leaving it to the main thread."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def _worker_generate_game(args):
    """
    Top-level worker function. Runs a single, isolated game.
    """
    game_seed, num_players = args
    config = JoeConfig()
    ctx = GameContext(num_players=num_players, config=config)
    engine = JoeEngine(ctx)

    # Offset seed by game number so each agent is uniquely randomized
    agents = [HeuristicAgent(random_seed=(game_seed * num_players) + i) for i in
              range(num_players)]

    game_data = []
    episode_memory = []

    def flush_memory():
        """Flushes the round's memory to the worker's local game list."""
        nonlocal episode_memory
        if not episode_memory:
            return

        for step in episode_memory:
            p_idx = step['player_idx']
            terminal_score = np.array([float(ctx.players[p_idx].score)], dtype=np.float32)

            # Store the data in standard RAM lists for IPC transfer
            game_data.append({
                'spatial': step['spatial'],
                'scalar': step['scalar'],
                'mask': step['mask'],
                'oracle': step['oracle'],
                'terminal_score': terminal_score,
                'policy': step['policy']
            })
        episode_memory.clear()

    # Kick off the state machine
    engine.start_game()

    while True:
        state_id = engine.state_id

        # --- 1. Terminal / Flush States ---
        if state_id == 'game_over':
            flush_memory()
            break

        if ctx.current_circuit >= config.max_turns:
            episode_memory.clear()
            break

        if ctx.total_actions >= config.max_actions:
            episode_memory.clear()
            break

        # --- 2. Auto-Stepper for Non-Decision States ---
        if state_id == 'dealing':
            flush_memory()
            engine.deal_cards()
            continue

        elif state_id == 'start_turn':
            engine.enter_pickup()
            continue

        # --- 3. Neural Decision States ---
        decision_states = ['pickup_decision', 'may_i_decision', 'go_down_decision',
                           'table_play_phase', 'discard_phase']

        if state_id in decision_states:
            current_player_idx = ctx.may_i_target_idx if state_id == 'may_i_decision' else ctx.active_player_idx

            tensors = ctx.get_input_tensor(current_player_idx, state_id)
            mask = ctx.get_action_mask(current_player_idx, state_id)
            oracle = ctx.get_oracle_truth(current_player_idx)

            agent = agents[current_player_idx]
            action_idx = agent.select_action(state_id, ctx, current_player_idx, mask)

            policy = np.zeros(58, dtype=np.float32)
            if action_idx >= 0:
                policy[action_idx] = 1.0

            episode_memory.append({
                'spatial': tensors['spatial'],
                'scalar': tensors['scalar'],
                'mask': mask,
                'oracle': oracle,
                'policy': policy,
                'player_idx': current_player_idx
            })

            # Execute Action
            apply_engine_action(engine, ctx, state_id, current_player_idx, action_idx)

    # Return the entire game's tensor footprint to the main thread
    return game_data


def generate_data(num_games=1000, max_size=500000, output_file="joe_phase1_sandbox.h5"):
    num_cores = max(1, multiprocessing.cpu_count() - 1)

    print(f"--- Starting JoeNet Sandbox Generation (PARALLEL) ---")
    print(f"Target Games:  {num_games} (Up to {num_games * 7} Rounds)")
    print(f"Output Buffer: {output_file}")
    print(f"Workers:       {num_cores} CPU Cores")

    buffer = JoeReplayBuffer(filepath=output_file, max_size=max_size)

    total_steps_recorded = 0
    start_time = time.time()

    # The iterable just passes a unique game seed ID to each worker
    iterable = [(i, random.choice([3, 4])) for i in range(num_games)]

    with multiprocessing.Pool(processes=num_cores,
                              initializer=_mute_worker_keyboard_interrupt) as pool:
        try:
            # imap_unordered yields results as soon as any worker finishes, drastically improving speed
            result_iterator = pool.imap_unordered(_worker_generate_game, iterable, chunksize=1)

            for i, game_data in enumerate(result_iterator):
                if not game_data:
                    continue

                # Stack the individual steps into massive batch arrays
                b_spatial = np.stack([step['spatial'] for step in game_data])
                b_scalar = np.stack([step['scalar'] for step in game_data])
                b_mask = np.stack([step['mask'] for step in game_data])
                b_oracle = np.stack([step['oracle'] for step in game_data])
                b_score = np.stack([step['terminal_score'] for step in game_data])
                b_policy = np.stack([step['policy'] for step in game_data])

                # 4. The Main Thread writes the ENTIRE game to disk in one I/O operation
                buffer.add_batch(
                    spatial=b_spatial,
                    scalar=b_scalar,
                    action_mask=b_mask,
                    oracle_truth=b_oracle,
                    terminal_score=b_score,
                    policy=b_policy
                )

                total_steps_recorded += len(game_data)

                # Print progress periodically
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"Games {i + 1}/{num_games} | Total Steps: {total_steps_recorded} | Time: {elapsed:.2f}s")

        except KeyboardInterrupt:
            print("\n\n[Ctrl-C] Manual interrupt detected! Safely shutting down workers...")
            pool.terminate()
            pool.join()
        finally:
            buffer.close()
            print(f"--- Generation Complete! Saved to {output_file} ---")


if __name__ == "__main__":
    generate_data(num_games=20000, max_size=20000000)