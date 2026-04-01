import os
import random
import subprocess
import torch
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
                          terminal_score: float) -> float:
    if is_terminal:
        return terminal_score - current_phi
    return next_phi - current_phi


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


class EpisodeTracker:
    def __init__(self, buffer):
        self.buffer = buffer
        self.cached_step = None

    def cache_step(self, spatial, scalar, mask, action, current_phi, oracle_truth):
        if self.cached_step is not None:
            old_spatial, old_scalar, old_mask, old_action, old_phi, old_truth = self.cached_step
            reward = calculate_pbrs_reward(old_phi, current_phi, False, 0.0)
            self.buffer.add(old_spatial, old_scalar, old_mask, old_action, reward,
                            is_terminal=False, oracle_truth=old_truth)

        self.cached_step = (spatial, scalar, mask, action, current_phi, oracle_truth)

    def flush_terminal(self, terminal_score: float):
        if self.cached_step is not None:
            old_spatial, old_scalar, old_mask, old_action, old_phi, old_truth = self.cached_step
            reward = calculate_pbrs_reward(old_phi, 0.0, True, terminal_score)
            self.buffer.add(old_spatial, old_scalar, old_mask, old_action, reward,
                            is_terminal=True, oracle_truth=old_truth)
            self.cached_step = None


def calculate_state_potential(ctx, player_idx, oracle_probs=None):
    """
    Calculates the Phi(s) using the existing structural distance logic.
    """
    calc = RewardCalculator(ctx)
    # The RewardCalculator already evaluates the entire hand's potential at once
    return calc.calculate_state_potential(player_idx)

def run_rl_training(episodes=1000, temperature=1.5):
    print("==================================================")
    print("   INITIALIZING PHASE 3: THE CRUCIBLE (MIXED)     ")
    print("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = JoeNet().to(device)
    weights_path = "models/joenet_phase2_ep7.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("-> Successfully loaded Phase 2 Behavioral Clone weights.")
    else:
        print("-> WARNING: Phase 2 weights not found. Starting from random initialization!")

    # train_rl.py (Replacement for Line 92)
    optimizer = Adam([
        {'params': model.oracle.parameters(), 'lr': 1e-6},  # The Seer's Guard (Micro-LR)
        {'params': model.actor.parameters(), 'lr': 1e-4},  # Standard Policy LR
        {'params': model.critic.parameters(), 'lr': 1e-4}  # Standard Value LR
    ])
    buffer = RolloutBuffer()
    trainer = RLTrainer(model, optimizer, gamma=0.99)
    runner = RLRunner(model, trainer, buffer)

    rl_agent = NeuralAgent(model, device)

    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir="runs/joenet_phase3_crucible")
    for episode in range(1, episodes + 1):
        # Starts at 1.5 (heavy exploration) and smoothly decays to 0.5 (heavy exploitation)
        current_temp = max(0.5, 1.5 - (episode / 800.0))
        stalemate_occurred = False
        terminal_reward = 0.0

        # --- NEW: Dynamic Player Counts ---
        # num_players = random.choice([3, 4])
        num_players = 4
        ctx = GameContext(num_players=num_players)
        engine = JoeEngine(ctx)
        engine.start_game()

        # --- NEW: We ONLY track the RL Agent (Player 0) ---
        tracker = EpisodeTracker(buffer)

        # --- FIXED: Independent Teacher Bots ---
        heuristic_bots = {i: HeuristicAgent() for i in range(1, num_players)}
        start_time = time.time()

        while True:
            raw_state_id = engine.current_state.id
            state_id = raw_state_id.lower().replace(' ', '_').replace('-', '_')

            if state_id == 'game_over':
                ctx.calculate_scores()
                # ONLY flush Player 0's score into the buffer
                terminal_reward = -float(ctx.players[0].score)
                tracker.flush_terminal(terminal_reward)
                break

            if ctx.current_circuit >= ctx.config.max_turns:
                # --- Handling stalemates ---
                # Assign a flat penalty slightly worse than a normal loss
                stalemate_penalty = -50.0
                tracker.flush_terminal(stalemate_penalty)

                # Flag it for TensorBoard
                stalemate_occurred = True
                terminal_reward = stalemate_penalty

                # Log the hands to see exactly who was hoarding what!
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

            # ==========================================
            # THE MIXED ROUTER
            # ==========================================
            if active_idx == 0:
                # --- PLAYER 0: THE RL LEARNER ---
                nn_inputs = ctx.get_input_tensor(active_idx, state_id)
                spatial_np = nn_inputs['spatial']
                scalar_np = nn_inputs['scalar']

                spatial_t = torch.tensor(spatial_np, dtype=torch.float32).unsqueeze(0).to(device)
                scalar_t = torch.tensor(scalar_np, dtype=torch.float32).unsqueeze(0).to(device)
                mask_t = torch.tensor(mask_np, dtype=torch.bool).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits, values, oracle_probs = model(spatial_t, scalar_t, mask_t)

                # Player 0 explores based on the decaying temperature
                action_idx = rl_agent.compute_action_with_exploration(
                    logits.squeeze(0), mask_np, temperature=current_temp
                )

                # ONLY cache Player 0 for training
                current_phi = calculate_state_potential(ctx, active_idx,
                                                        oracle_probs.squeeze(0).cpu().numpy())
                truth_np = ctx.get_oracle_truth(active_idx)
                tracker.cache_step(spatial_np, scalar_np, mask_np, action_idx, current_phi,
                                   truth_np)

            else:
                # --- PLAYERS 1, 2, 3: THE HEURISTIC TEACHERS ---
                # CORRECTED: Query the specific independent bot
                action_idx = heuristic_bots[active_idx].select_action(state_id, ctx, active_idx,
                                                                      mask_np)

            # Execute the chosen action (RL or Heuristic)
            apply_engine_action(engine, ctx, state_id, active_idx, action_idx)

        # Trigger Backpropagation based purely on Player 0's experiences
        metrics = runner.trigger_update(
            next_value=torch.tensor([0.0], dtype=torch.float32).to(device))

        elapsed = time.time() - start_time
        if episode % 10 == 0:
            print(
                f"Ep {episode:04d} ({num_players}P) | Time: {elapsed:.1f}s | ActLoss: {metrics['actor_loss']:.4f} | CritLoss: {metrics['critic_loss']:.4f}")

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
        writer.add_scalar('Loss/Actor', metrics['actor_loss'], episode)
        writer.add_scalar('Loss/Critic', metrics['critic_loss'], episode)
        writer.add_scalar('Loss/Oracle', metrics['oracle_loss'], episode)

        # Performance & Health
        writer.add_scalar('Performance/Time_Elapsed_Secs', elapsed, episode)
        writer.add_scalar(f'Reward/Terminal_Score_{num_players}P', terminal_reward, episode)

        writer.add_scalar(f'Health/Stalemates_{num_players}P', 1 if stalemate_occurred else 0,
                          episode)
        writer.add_scalar(f'Health/Turns_Per_Round_{num_players}P', ctx.current_circuit, episode)

    print("\n--- Phase 3 Training Complete! ---")
    torch.save(model.state_dict(), "models/joenet_phase3_rl_final.pth")


if __name__ == '__main__':
    run_rl_training(episodes=1000, temperature=1.5)