import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, TensorDataset
from network import JoeNet, expand_spatial_with_oracle
import os
import argparse
from torch.utils.tensorboard import SummaryWriter


def train_supervised_epoch(model, dataloader, optimizer, device, mode, actor_weights=None):
    model.train()
    actor_criterion = nn.CrossEntropyLoss(weight=actor_weights)
    critic_criterion = nn.MSELoss()
    oracle_criterion = nn.BCELoss()

    total_actor_loss, total_critic_loss = 0.0, 0.0
    total_oracle_3p_loss, total_oracle_4p_loss = 0.0, 0.0
    total_combined_loss, total_entropy = 0.0, 0.0
    num_batches = 0
    batches_3p, batches_4p = 0, 0

    for batch in dataloader:
        t_spatial, t_scalar, t_mask, t_oracle_truth, t_score, t_policy = [b.to(device) for b in
                                                                          batch]
        optimizer.zero_grad()

        is_3_player = t_scalar[:, 22].bool()
        mask_3p = is_3_player
        mask_4p = ~is_3_player

        if mode == 'oracle':
            loss = 0.0

            if mask_3p.any():
                probs_3p = model.oracle_3p(t_spatial[mask_3p], t_scalar[mask_3p])
                adjusted_probs_3p = probs_3p[:, :2, :, :]
                adjusted_truth_3p = t_oracle_truth[mask_3p][:, :2, :, :]
                loss_3p = oracle_criterion(adjusted_probs_3p, adjusted_truth_3p)
                loss += loss_3p
                total_oracle_3p_loss += loss_3p.item()
                batches_3p += 1

            if mask_4p.any():
                probs_4p = model.oracle_4p(t_spatial[mask_4p], t_scalar[mask_4p])
                loss_4p = oracle_criterion(probs_4p, t_oracle_truth[mask_4p])
                loss += loss_4p
                total_oracle_4p_loss += loss_4p.item()
                batches_4p += 1

            loss.backward()
            optimizer.step()
            total_combined_loss += loss.item()

        elif mode == 'decision':
            # Phase 2B: Isolate Policy/Value with Frozen Vision Specialists
            with torch.no_grad():
                # Initialize an empty tensor to hold the combined predictions
                batch_size = t_spatial.shape[0]
                oracle_probs = torch.zeros((batch_size, 3, 4, 14), device=device)

                # Fill the tensor using the respective specialists
                if mask_3p.any():
                    oracle_probs[mask_3p] = model.oracle_3p(t_spatial[mask_3p], t_scalar[mask_3p])
                if mask_4p.any():
                    oracle_probs[mask_4p] = model.oracle_4p(t_spatial[mask_4p], t_scalar[mask_4p])

            expanded_spatial = expand_spatial_with_oracle(t_spatial, oracle_probs)
            ev = model.critic(expanded_spatial, t_scalar)
            logits = model.actor(expanded_spatial, t_scalar, action_mask=t_mask)

            # Entropy calculation
            probs = torch.softmax(logits, dim=-1)
            batch_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
            total_entropy += batch_entropy.item()

            actor_loss = actor_criterion(logits, t_policy)
            critic_loss = critic_criterion(ev, t_score)

            loss = actor_loss + critic_loss
            loss.backward()
            optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_combined_loss += loss.item()

        num_batches += 1

    return (total_actor_loss, total_critic_loss, total_oracle_3p_loss, total_oracle_4p_loss,
            total_combined_loss, total_entropy, num_batches, batches_3p, batches_4p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JoeNet Phase 2 Training")
    parser.add_argument('--mode', type=str, required=True, choices=['oracle', 'decision'],
                        help="Train 'oracle' first, then 'decision'")
    parser.add_argument('--oracle_weights', type=str, default='',
                        help="Path to saved Oracle weights (required for decision mode)")
    args = parser.parse_args()

    data_path = "joe_phase1_sandbox.h5"
    batch_size = 4096
    epochs = 10

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find training data at {data_path}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Phase 2: Behavioral Cloning ({args.mode.upper()} MODE) ---")
    print(f"Device: {device}")

    with h5py.File(data_path, 'r') as f:
        total_size = f['current_size'][()]
    print(f"Dataset Size: {total_size:,} total steps.")

    actor_weights = None
    if args.mode == 'decision':
        print("Calculating dynamic class weights for Actor...")
        class_counts = torch.zeros(58, dtype=torch.float64)
        with h5py.File(data_path, 'r') as f:
            weight_chunk = 500000
            for i in range(0, total_size, weight_chunk):
                end = min(i + weight_chunk, total_size)
                policy_chunk = torch.tensor(f['policy'][i:end], dtype=torch.float32)
                class_counts += policy_chunk.sum(dim=0).double()

        total_actions = class_counts.sum()
        actor_weights = total_actions / (58.0 * (class_counts + 1.0))
        actor_weights = torch.clamp(actor_weights, max=10.0).to(device).float()

    model = JoeNet().to(device)

    if args.mode == 'oracle':
        learning_rate = 1e-4
        # --- FIXED: Combine parameters from both specialists into one optimizer ---
        optimizer = torch.optim.Adam(
            list(model.oracle_3p.parameters()) + list(model.oracle_4p.parameters()),
            lr=learning_rate
        )
        writer = SummaryWriter(log_dir="runs/joenet_phase2a_oracle")
        save_prefix = "models/joenet_phase2a_oracle"
    else:
        if not args.oracle_weights or not os.path.exists(args.oracle_weights):
            raise ValueError("Decision mode requires valid --oracle_weights path.")

        print(f"Loading Frozen Dual Oracles from {args.oracle_weights}...")
        model.load_state_dict(torch.load(args.oracle_weights, map_location=device), strict=False)

        # --- FIXED: Freeze both specialists ---
        for param in model.oracle_3p.parameters():
            param.requires_grad = False
        for param in model.oracle_4p.parameters():
            param.requires_grad = False

        learning_rate = 1e-3
        optimizer = torch.optim.Adam(
            list(model.actor.parameters()) + list(model.critic.parameters()),
            lr=learning_rate
        )
        writer = SummaryWriter(log_dir="runs/joenet_phase2b_decision")
        save_prefix = "models/joenet_phase2b_decision"

    CHUNK_SIZE = 1_000_000

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        # --- FIXED: Epoch level accumulators ---
        ep_act, ep_crit, ep_tot, ep_ent, ep_batches = 0.0, 0.0, 0.0, 0.0, 0
        ep_ora_3p, ep_ora_4p, ep_batches_3p, ep_batches_4p = 0.0, 0.0, 0, 0

        for start_idx in range(0, total_size, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, total_size)
            print(f"  Loading RAM Chunk: {start_idx:,} to {end_idx:,}...")

            with h5py.File(data_path, 'r') as f:
                np_spatial = f['spatial'][start_idx:end_idx]
                np_scalar = f['scalar'][start_idx:end_idx]
                np_mask = f['action_mask'][start_idx:end_idx]
                np_oracle = f['oracle_truth'][start_idx:end_idx]
                np_score = f['terminal_score'][start_idx:end_idx]
                np_policy = f['policy'][start_idx:end_idx]

            t_spatial = torch.tensor(np_spatial, dtype=torch.float32)
            presence_channels = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            t_spatial[:, presence_channels, :, :] /= 2.0

            t_scalar = torch.tensor(np_scalar, dtype=torch.float32)
            t_mask = torch.tensor(np_mask, dtype=torch.bool)
            t_oracle = torch.tensor(np_oracle, dtype=torch.float32).clamp(min=0.0, max=1.0)
            t_score = torch.tensor(np_score, dtype=torch.float32)
            t_policy = torch.tensor(np_policy, dtype=torch.float32).clamp(min=0.0, max=1.0)
            t_policy = t_policy / (t_policy.sum(dim=-1, keepdim=True) + 1e-8)

            chunk_dataset = TensorDataset(t_spatial, t_scalar, t_mask, t_oracle, t_score, t_policy)
            chunk_loader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True)

            c_act, c_crit, c_ora_3p, c_ora_4p, c_tot, c_ent, c_batches, c_b3p, c_b4p = train_supervised_epoch(
                model, chunk_loader, optimizer, device, args.mode, actor_weights)

            ep_act += c_act
            ep_crit += c_crit
            ep_ora_3p += c_ora_3p
            ep_ora_4p += c_ora_4p
            ep_tot += c_tot
            ep_ent += c_ent
            ep_batches += c_batches
            ep_batches_3p += c_b3p
            ep_batches_4p += c_b4p

            del np_spatial, t_spatial, chunk_dataset, chunk_loader

        avg_act = ep_act / max(1, ep_batches)
        avg_crit = ep_crit / max(1, ep_batches)
        avg_ora_3p = ep_ora_3p / max(1, ep_batches_3p)
        avg_ora_4p = ep_ora_4p / max(1, ep_batches_4p)
        avg_tot = ep_tot / max(1, ep_batches)
        avg_ent = ep_ent / max(1, ep_batches)

        print(f"Epoch {epoch + 1} | Actor: {avg_act:.4f} | Critic: {avg_crit:.4f} | "
              f"Oracle_3P: {avg_ora_3p:.4f} | Oracle_4P: {avg_ora_4p:.4f} | Entropy: {avg_ent:.4f}")

        if args.mode == 'decision':
            writer.add_scalar('Loss/Actor', avg_act, epoch + 1)
            writer.add_scalar('Loss/Critic', avg_crit, epoch + 1)
            writer.add_scalar('Health/Actor_Entropy', avg_ent, epoch + 1)
        else:
            writer.add_scalar('Loss/Oracle_3P', avg_ora_3p, epoch + 1)
            writer.add_scalar('Loss/Oracle_4P', avg_ora_4p, epoch + 1)

        writer.add_scalar('Loss/Total', avg_tot, epoch + 1)

        epoch_save_path = f"{save_prefix}_ep{epoch + 1}.pth"
        torch.save(model.state_dict(), epoch_save_path)

    writer.close()
    final_save_path = f"{save_prefix}_final.pth"
    torch.save(model.state_dict(), final_save_path)
    print(f"\n--- Training Complete! Weights saved to {final_save_path} ---")