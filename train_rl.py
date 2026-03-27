import os
import torch
from torch.optim import Adam
import numpy as np
import time

from network import JoeNet
from neural_agent import NeuralAgent
from game_context import GameContext
from fast_engine import JoeEngine
from rl_buffer import RolloutBuffer
from evaluate_arena import apply_engine_action
from rl_trainer import RLTrainer


def calculate_pbrs_reward(current_phi: float, next_phi: float, is_terminal: bool,
                          terminal_score: float) -> float:
    """
    Calculates the step-by-step reward using Potential-Based Reward Shaping (PBRS).
    Formula: R = Phi(s') - Phi(s)

    If the state is terminal, there is no future potential. The reward rigidly
    anchors to the environment's true Asymmetric Terminal Score.
    """
    if is_terminal:
        # At the end of the game, the future potential is conceptually 0.0.
        # We replace the future potential with the actual game score.
        return terminal_score - current_phi

    # Standard step-by-step potential difference
    return next_phi - current_phi


class RLRunner:
    """
    Orchestrates the live self-play games and triggers the RLTrainer
    updates using the experiences stored in the RolloutBuffer.
    """

    def __init__(self, model, trainer, buffer):
        self.model = model
        self.trainer = trainer
        self.buffer = buffer

    def trigger_update(self, next_value: torch.Tensor):
        """
        Extracts the rollout batch, updates the neural network,
        and wipes the buffer clean for the next game.
        """
        # 1. Extract the batched tensors from the RAM buffer
        batch_tensors = self.buffer.get_tensors()

        # 2. Pass the batch and the terminal boundary value to the Trainer
        metrics = self.trainer.update(batch_tensors, next_value)

        # 3. Wipe the buffer immediately to free RAM and prevent stale data
        self.buffer.clear()

        # 4. Send the metrics back to the main loop!
        return metrics


class EpisodeTracker:
    """
    Caches an agent's step and waits until their NEXT action to calculate
    the PBRS reward (Phi(s') - Phi(s)) before pushing it to the RolloutBuffer.
    """

    def __init__(self, buffer):
        self.buffer = buffer
        self.cached_step = None

    def cache_step(self, spatial, scalar, mask, action, current_phi, oracle_truth):
        """
        If a step is already cached, calculating the new current_phi means we
        finally have the 'next_phi' for the old step. We push the old step to
        the buffer, and cache the new one.
        """
        if self.cached_step is not None:
            # FIX: We now unpack 6 variables, including old_truth!
            old_spatial, old_scalar, old_mask, old_action, old_phi, old_truth = self.cached_step

            # Calculate PBRS reward for the PREVIOUS step using the CURRENT potential
            reward = calculate_pbrs_reward(
                current_phi=old_phi,
                next_phi=current_phi,
                is_terminal=False,
                terminal_score=0.0
            )

            # Add to buffer, now including oracle_truth=old_truth
            self.buffer.add(old_spatial, old_scalar, old_mask, old_action, reward,
                            is_terminal=False, oracle_truth=old_truth)

        # Cache the new step (now 6 items long) to wait for the next board rotation
        self.cached_step = (spatial, scalar, mask, action, current_phi, oracle_truth)

    def flush_terminal(self, terminal_score: float):
        """
        When the game ends, flush the final cached step into the buffer
        using the Asymmetric Terminal Score as the absolute anchor.
        """
        if self.cached_step is not None:
            # FIX: We now unpack 6 variables here too!
            old_spatial, old_scalar, old_mask, old_action, old_phi, old_truth = self.cached_step

            reward = calculate_pbrs_reward(
                current_phi=old_phi,
                next_phi=0.0,  # Mathematically ignored due to terminal flag
                is_terminal=True,
                terminal_score=terminal_score
            )

            # Add to buffer, now including oracle_truth=old_truth
            self.buffer.add(old_spatial, old_scalar, old_mask, old_action, reward,
                            is_terminal=True, oracle_truth=old_truth)
            self.cached_step = None


def calculate_state_potential(ctx, player_idx, oracle_probs):
    """
    Placeholder for your Phase 3 PBRS Math: (Combinatorial Outs) - (Danger Score)
    Replace this with your actual reward.py import!
    """
    return 0.0


def run_rl_training(episodes=1000, temperature=1.5):
    print("==================================================")
    print("   INITIALIZING PHASE 3: EXPLORATORY RL SELF-PLAY ")
    print("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Initialize the Neural Architecture
    model = JoeNet().to(device)

    # Load the Phase 2 Cloned Weights to bootstrap the RL
    weights_path = "models/joenet_phase2_cloned.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("-> Successfully loaded Phase 2 Behavioral Clone weights.")
    else:
        print("-> WARNING: Phase 2 weights not found. Starting from random initialization!")

    optimizer = Adam(model.parameters(), lr=1e-4)

    # 2. Initialize the RL Infrastructure
    buffer = RolloutBuffer()
    trainer = RLTrainer(model, optimizer, gamma=0.99)
    runner = RLRunner(model, trainer, buffer)

    # All 4 players share the exact same brain!
    shared_agent = NeuralAgent(model, device)

    # 3. The Massive Self-Play Loop
    for episode in range(1, episodes + 1):
        ctx = GameContext(num_players=4)
        engine = JoeEngine(ctx)
        engine.start_game()

        # Each player needs their own tracker to solve the Delayed Reward Trap
        trackers = {i: EpisodeTracker(buffer) for i in range(4)}

        start_time = time.time()

        while True:
            state_id = engine.current_state.id

            if state_id == 'game_over':
                # The game is finished. Flush all 4 trackers with their terminal scores!
                ctx.calculate_scores()

                for i in range(4):
                    # In Joe, negative points are good. We invert the penalty score
                    # so the neural network learns to maximize it.
                    terminal_reward = -float(ctx.players[i].score)
                    trackers[i].flush_terminal(terminal_reward)
                break

            # Circuit failsafe prevents infinite loops
            if ctx.current_circuit >= ctx.config.max_turns:
                ctx.calculate_scores()
                if ctx.current_round_idx >= 7:
                    break
                else:
                    engine.current_state = engine.dealing
                    engine.on_enter_dealing()
                    continue

            if state_id == 'dealing':
                engine.deal_cards()
                continue

            if state_id == 'start_turn':
                engine.enter_pickup()
                continue

            # Determine whose turn it is
            active_idx = ctx.may_i_target_idx if state_id == 'may_i_decision' else ctx.active_player_idx

            # 1. Network Observation (Forward Pass for logits and oracle predictions)
            # Use the actual Phase 2 builder that returns the dictionary
            nn_inputs = ctx.get_input_tensor(active_idx, state_id)
            spatial_np = nn_inputs['spatial']
            scalar_np = nn_inputs['scalar']

            mask_np = ctx.get_action_mask(active_idx, state_id)

            # Convert to tensors for the network
            spatial_t = torch.tensor(spatial_np, dtype=torch.float32).unsqueeze(0).to(device)
            scalar_t = torch.tensor(scalar_np, dtype=torch.float32).unsqueeze(0).to(device)
            mask_t = torch.tensor(mask_np, dtype=torch.bool).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, values, oracle_probs = model(spatial_t, scalar_t, mask_t)

            # 2. Action Selection (With Entropy Injection!)
            action_idx = shared_agent.compute_action_with_exploration(
                logits.squeeze(0), mask_np, temperature=temperature
            )

            # 3. Calculate the Current Potential: Phi(s)
            # We use the Oracle's prediction to discount the combinatorial outs
            current_phi = calculate_state_potential(ctx, active_idx,
                                                    oracle_probs.squeeze(0).cpu().numpy())

            truth_np = ctx.get_oracle_truth(active_idx)

            # 4. Cache the step! (This safely handles the delayed reward trap)
            trackers[active_idx].cache_step(spatial_np, scalar_np, mask_np,
                                            action_idx, current_phi, truth_np)

            # 5. Apply the action to the engine
            apply_engine_action(engine, ctx, state_id, active_idx, action_idx)

        # --- EPISODE CONCLUDED: TRIGGER BACKPROPAGATION ---

        # We pass 0.0 as the next_value because all flushed steps are anchored to the terminal score
        metrics = runner.trigger_update(
            next_value=torch.tensor([0.0], dtype=torch.float32).to(device))

        elapsed = time.time() - start_time
        print(
            f"Episode {episode:04d} | Time: {elapsed:.2f}s |"
            f" Actor Loss: {metrics['actor_loss']:.4f} |"
            f" Critic Loss: {metrics['critic_loss']:.4f} |"
            f" Oracle Loss: {metrics['oracle_loss']:.4f}")

        # Save weights periodically
        if episode % 100 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/joenet_phase3_rl_ckpt_{episode}.pth")

    print("\n--- Phase 3 Training Complete! ---")
    torch.save(model.state_dict(), "models/joenet_phase3_rl_final.pth")


if __name__ == '__main__':
    run_rl_training(episodes=1000, temperature=1.5)