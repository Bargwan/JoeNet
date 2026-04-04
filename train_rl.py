import os
from datetime import datetime
import random
import subprocess
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import time


from export_onnx import export_joenet_to_onnx, JoeNetDeployment
from network import JoeNet
from neural_agent import NeuralAgent
from game_context import GameContext
from fast_engine import JoeEngine
from reward import RewardCalculator
from rl_buffer import RolloutBuffer
from evaluate_arena import apply_engine_action, _log_stalemate_hands
from rl_trainer import RLTrainer
from agents import HeuristicAgent

def calculate_pbrs_reward(current_phi: float, next_phi: float, is_terminal: bool,
                          terminal_score: float, gamma: float = 0.99) -> float:
    """Includes the missing Gamma discount factor for the Andrew Ng PBRS proof."""
    if is_terminal:
        return terminal_score - current_phi
    return (gamma * next_phi) - current_phi

class RLRunner:
    def __init__(self, model, trainer, buffer):
        self.model = model
        self.trainer = trainer
        self.buffer = buffer

    def trigger_update(self, next_value: torch.Tensor):
        if len(self.buffer) == 0:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'oracle_loss': 0.0}

        batch_tensors = self.buffer.get_tensors()
        metrics = self.trainer.update(batch_tensors, next_value)
        self.buffer.clear()
        return metrics


from typing import Dict, Tuple, Optional
import numpy as np


class MultiPlayerTracker:
    """Tracks separate experience trajectories for all players simultaneously."""

    def __init__(self, buffer, num_players: int, gamma: float = 0.99):
        self.buffer = buffer
        self.gamma = gamma

        # Explicit type hinting clears the PyCharm static analysis warning
        self.cached_steps: Dict[
            int, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray]]] = {
            i: None for i in range(num_players)
        }

    def cache_step(self, player_idx: int, spatial: np.ndarray, scalar: np.ndarray,
                   mask: np.ndarray, action: int, current_phi: float, oracle_truth: np.ndarray):

        if self.cached_steps[player_idx] is not None:
            old_spatial, old_scalar, old_mask, old_action, old_phi, old_truth = self.cached_steps[
                player_idx]
            # Dense step reward using Gamma
            reward = calculate_pbrs_reward(old_phi, current_phi, False, 0.0, self.gamma)
            self.buffer.add(old_spatial, old_scalar, old_mask, old_action, reward,
                            is_terminal=False, oracle_truth=old_truth)

        # Cache the new step
        self.cached_steps[player_idx] = (spatial, scalar, mask, action, current_phi, oracle_truth)

    def flush_round_boundary(self):
        """
        Breaks the PBRS chain between rounds so the network doesn't get
        penalized when its hand is wiped by the dealer.
        """
        for i in range(len(self.cached_steps)):
            if self.cached_steps[i] is not None:
                old_spatial, old_scalar, old_mask, old_action, old_phi, old_truth = \
                self.cached_steps[i]

                # Close out the round's PBRS mathematically (treating hand as 0 mass),
                # but DO NOT flag it as terminal for the actual game trajectory.
                reward = calculate_pbrs_reward(old_phi, 0.0, True, 0.0, self.gamma)

                self.buffer.add(old_spatial, old_scalar, old_mask, old_action, reward,
                                is_terminal=False, oracle_truth=old_truth)

                # Wipe the cache so Round N+1 starts with a fresh delta calculation
                self.cached_steps[i] = None

    def flush_terminal(self, player_idx: int, terminal_score: float):
        if self.cached_steps[player_idx] is not None:
            old_spatial, old_scalar, old_mask, old_action, old_phi, old_truth = self.cached_steps[
                player_idx]
            # Terminal step reward using actual game score
            reward = calculate_pbrs_reward(old_phi, 0.0, True, terminal_score, self.gamma)
            self.buffer.add(old_spatial, old_scalar, old_mask, old_action, reward,
                            is_terminal=True, oracle_truth=old_truth)
            self.cached_steps[player_idx] = None

def calculate_state_potential(ctx, player_idx, oracle_probs=None):
    """
    Calculates the Phi(s) using the existing structural distance logic.
    """
    calc = RewardCalculator(ctx)
    # The RewardCalculator already evaluates the entire hand's potential at once
    return calc.calculate_state_potential(player_idx)


def count_completed_objectives(ctx, player_idx) -> tuple:
    """Helper to count completed objectives using the communal table reality."""
    req_sets, req_runs = ctx.config.objective_map[ctx.current_round_idx]
    player = ctx.players[player_idx]

    # If the engine says they are down, the objectives are permanently met.
    if player.is_down:
        return req_sets, req_runs

    # Otherwise, scan their physical secret hand
    calc = RewardCalculator(ctx)
    sets = calc._get_potential_sets(player.hand_list)
    runs = calc._get_potential_runs(player.hand_list)

    valid_sets = sum(1 for s in sets if s >= 3)
    valid_runs = sum(1 for r in runs if r >= 4)

    return min(valid_sets, req_sets), min(valid_runs, req_runs)


def calculate_safe_potential(my_hand, oracle_probs, completed_sets, completed_runs) -> float:
    """The PyTorch Spatial Potential Function (Communal Table Safe)."""
    device = my_hand.device

    # 1. Pure Secret Hand Assets
    hand_size = torch.clamp(torch.sum(my_hand), min=1.0)

    # 2. Base Availability & Fertility
    # oracle_probs is 4D: [Batch, Opponents, Suits, Ranks]
    # Multiply across dim=1 (the 3 opponents) to find probability NO opponent has it
    availability = torch.prod(1.0 - oracle_probs, dim=1, keepdim=True)

    # Sum across dim=2 (the 4 suits) to find Set Fertility
    set_fertility = availability.sum(dim=2, keepdim=True) - availability

    set_fertility_masked = set_fertility.clone()
    # Mask out the 4th dimension (Ranks) for Ace High
    set_fertility_masked[:, :, :, 13] = 0.0

    run_kernel = torch.tensor([[[[0.5, 1.0, 0.0, 1.0, 0.5]]]], device=device)
    padded_avail = F.pad(availability, (2, 2, 0, 0), mode='constant', value=0.0)
    run_fertility = F.conv2d(padded_avail, run_kernel)

    # 3. Average Potentials (Protects May-I and Go Down)
    avg_set_potential = torch.sum(my_hand * set_fertility_masked) / hand_size
    avg_run_potential = torch.sum(my_hand * run_fertility) / hand_size

    # 4. Static Weights & The Staircase
    spatial_potential = (avg_set_potential * 1.0) + (avg_run_potential * 1.5)
    objective_bonus = (completed_sets * 30.0) + (completed_runs * 50.0)

    return (spatial_potential + objective_bonus).item()


def run_rl_training(episodes=1000, temperature=1.5):
    print("==================================================")
    print("   INITIALIZING PHASE 3: THE CRUCIBLE (MIXED)     ")
    print("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = JoeNet().to(device)
    weights_path = "models/joenet_phase2b_decision_ep8.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        print("-> Successfully loaded Phase 2 Dual Oracle weights.")
    else:
        print("-> WARNING: Phase 2 weights not found. Starting from random initialization!")

        # --- NEW: PERMANENTLY FREEZE BOTH ORACLES ---
    for param in model.oracle_3p.parameters():
        param.requires_grad = False
    for param in model.oracle_4p.parameters():
        param.requires_grad = False

        # Initialize Optimizer ONLY for Actor and Critic
    optimizer = Adam([
        {'params': model.actor.parameters(), 'lr': 1e-4},
        {'params': model.critic.parameters(), 'lr': 1e-4}
    ])
    buffer = RolloutBuffer()
    trainer = RLTrainer(model, optimizer, gamma=0.99)
    runner = RLRunner(model, trainer, buffer)

    rl_agent = NeuralAgent(model, device)

    # Initialize TensorBoard Writer
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_name = f"JoeNet_RL_{timestamp}"
    log_dir = os.path.join("runs", log_name)

    writer = SummaryWriter(log_dir=log_dir)

    # --- NEW: Batch Accumulation Parameters ---
    update_frequency = 10  # Accumulate 10 full games before updating the network
    games_since_update = 0
    last_metrics = {'actor_loss': 0.0, 'critic_loss': 0.0, 'oracle_loss_3p': 0.0,
                    'oracle_loss_4p': 0.0}

    for episode in range(1, episodes + 1):
        current_temp = max(0.1, 0.8 - (episode / 700.0))
        stalemate_occurred = False
        tb_terminal_reward = 0.0  # Just for Tensorboard logging

        model.eval()
        num_players = random.choice([3, 4])
        ctx = GameContext(num_players=num_players)
        engine = JoeEngine(ctx)
        engine.start_game()

        # --- NEW: Track ALL players ---
        tracker = MultiPlayerTracker(buffer, num_players)
        start_time = time.time()

        while True:
            raw_state_id = engine.current_state.id
            state_id = raw_state_id.lower().replace(' ', '_').replace('-', '_')

            # tournament end
            if state_id == 'game_over':
                # REMOVED: ctx.calculate_scores()
                # The fast_engine already calculated Round 7 perfectly.
                # player.score now contains the TRUE 7-round cumulative deadwood.

                total_deadwood = sum(ctx.players[p].score for p in range(num_players))

                for i in range(num_players):
                    my_score = float(ctx.players[i].score)
                    avg_opponent_score = (total_deadwood - my_score) / max(1, (num_players - 1))

                    # The Ultimate Tournament Reward
                    terminal_reward = avg_opponent_score - my_score
                    tracker.flush_terminal(i, terminal_reward)

                    if i == 0:
                        tb_terminal_reward = terminal_reward
                break

            # round end
            if state_id == 'round_end':
                # Break the PBRS chain before the engine wipes the hands
                tracker.flush_round_boundary()
                engine.resolve_round_end()
                continue

            if ctx.current_circuit >= ctx.config.max_turns:
                stalemate_penalty = -500.0
                for i in range(num_players):
                    tracker.flush_terminal(i, stalemate_penalty)

                stalemate_occurred = True
                tb_terminal_reward = stalemate_penalty
                _log_stalemate_hands(ctx)
                break

            if state_id == 'dealing':
                engine.deal_cards()
                continue
            if state_id == 'start_turn':
                engine.enter_pickup()
                continue

            active_idx = ctx.may_i_target_idx if state_id == 'may_i_decision' else ctx.active_player_idx
            mask_np = ctx.get_action_mask(active_idx, state_id)

            nn_inputs = ctx.get_input_tensor(active_idx, state_id)
            spatial_np = nn_inputs['spatial']
            scalar_np = nn_inputs['scalar']

            spatial_t = torch.tensor(spatial_np, dtype=torch.float32).unsqueeze(0).to(device)
            presence_channels = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            spatial_t[:, presence_channels, :, :] /= 2.0
            scalar_t = torch.tensor(scalar_np, dtype=torch.float32).unsqueeze(0).to(device)
            mask_t = torch.tensor(mask_np, dtype=torch.bool).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, values, oracle_probs = model(spatial_t, scalar_t, mask_t)

            action_idx = rl_agent.compute_action_with_exploration(
                logits.squeeze(0), mask_np, temperature=current_temp
            )

            # --- NEW: ALL clones cache their data ---
            my_hand_t = spatial_t[:, 0:1, :, :]

            c_sets, c_runs = count_completed_objectives(ctx, active_idx)

            current_phi = calculate_safe_potential(
                my_hand=my_hand_t,
                oracle_probs=oracle_probs,
                completed_sets=c_sets,
                completed_runs=c_runs
            )

            truth_np = ctx.get_oracle_truth(active_idx)
            tracker.cache_step(
                active_idx,
                spatial_np.copy(),
                scalar_np.copy(),
                mask_np.copy(),
                action_idx,
                current_phi,
                truth_np.copy()
            )

            apply_engine_action(engine, ctx, state_id, active_idx, action_idx)

        # ==========================================
        # BATCH ACCUMULATION LOGIC
        # ==========================================
        games_since_update += 1

        if games_since_update >= update_frequency:
            model.train()
            model.oracle_3p.eval()
            model.oracle_4p.eval()

            # Trigger update on the massive accumulated buffer
            last_metrics = runner.trigger_update(
                next_value=torch.tensor([0.0], dtype=torch.float32).to(device)
            )
            games_since_update = 0

            # CRITICAL: Return model to eval mode for the next rollout
            model.eval()

        elapsed = time.time() - start_time

        # Use last_metrics for printing and TensorBoard
        if episode % 10 == 0:
            active_oracle_loss = last_metrics['oracle_loss_3p'] if num_players == 3 else \
            last_metrics['oracle_loss_4p']
            print(f"Ep {episode:04d} ({num_players}P) | Time: {elapsed:.1f}s | "
                  f"ActLoss: {last_metrics['actor_loss']:.4f} | CritLoss: {last_metrics['critic_loss']:.4f} | "
                  f"Oracle: {active_oracle_loss:.4f}")

        # --- NEW: Save Intermediate Checkpoints ---
        if episode % 50 == 0:
            checkpoint_path = f"models/joenet_phase3_ep{episode}.pth"
            torch.save(model.state_dict(), checkpoint_path)

        if episode % 100 == 0:
            pth_path = f"models/joenet_phase3_ep{episode}.pth"
            onnx_path = f"models/joenet_phase3_ep{episode}.onnx"

            print(f"\n[Checkpoint] Saving PyTorch Weights...")
            torch.save(model.state_dict(), pth_path)

            print(f"[Checkpoint] Compiling ONNX Graph...")
            # --- FIXED: Wrap the model for deployment before exporting ---
            deployment_model = JoeNetDeployment(model)
            deployment_model.eval()
            export_joenet_to_onnx(deployment_model, save_path=onnx_path)

            print(f"\n[EVALUATION] Launching 100-Game 3-Player Arena...")
            # subprocess.run blocks the script until the arena finishes
            subprocess.run(
                ["uv", "run", "python", "run_mastery.py", "--onnx", onnx_path, "--games", "100",
                 "--players", "3"])

            print(f"\n[EVALUATION] Launching 100-Game 4-Player Arena...")
            subprocess.run(
                ["uv", "run", "python", "run_mastery.py", "--onnx", onnx_path, "--games", "100",
                 "--players", "4"])

            print("\n[Checkpoint] Evaluation Complete. Resuming Training...\n")

            # CRITICAL: We must force PyTorch back into training mode!
            # export_joenet_to_onnx sets the model to .eval(), which disables learning.
            model.train()

        # 3. TensorBoard Broadcasting
        # Core Losses
        writer.add_scalar('Loss/Actor', last_metrics['actor_loss'], episode)
        writer.add_scalar('Loss/Critic', last_metrics['critic_loss'], episode)

        # --- FIXED: Only broadcast the Oracle loss that was actually active this episode ---
        if num_players == 3:
            writer.add_scalar('Loss/Oracle_3P', last_metrics['oracle_loss_3p'], episode)
        elif num_players == 4:
            writer.add_scalar('Loss/Oracle_4P', last_metrics['oracle_loss_4p'], episode)

        # Performance & Health
        writer.add_scalar('Performance/Time_Elapsed_Secs', elapsed, episode)
        writer.add_scalar(f'Reward/Terminal_Score_{num_players}P', tb_terminal_reward, episode)

        writer.add_scalar(f'Health/Stalemates_{num_players}P', 1 if stalemate_occurred else 0,
                          episode)
        writer.add_scalar(f'Health/Turns_Per_Round_{num_players}P', ctx.current_circuit, episode)

    print("\n--- Phase 3 Training Complete! ---")
    torch.save(model.state_dict(), "models/joenet_phase3_rl_final.pth")


if __name__ == '__main__':
    run_rl_training(episodes=1000, temperature=1.5)