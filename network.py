import torch
import torch.nn as nn

class SpatialHead(nn.Module):
    """
    Spatial CNN Head using asymmetric kernels.
    Includes Batch Normalization and an expanded receptive field.
    Designed to accept either 13 channels (Oracle) or 16 channels (Actor/Critic).
    """

    def __init__(self, in_channels: int = 13):
        super().__init__()

        # --- Branch A: Run Detector (Sequence Logic) ---
        self.branch_a = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 3), padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 3), padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 3), padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 3), padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # --- Branch B: Set Detector (Suit Agnostic Logic) ---
        self.branch_b = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 1), padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # --- The Merger ---
        # Branch A outputs (32 * 4 * 6) = 768 features
        # Branch B outputs (32 * 1 * 14) = 448 features
        # 768 + 448 = 1216 features
        self.fc_project = nn.Linear(in_features=1216, out_features=256)
        self.layer_norm = nn.LayerNorm(256)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_a = self.branch_a(x)
        out_a_flat = torch.flatten(out_a, start_dim=1)

        # Collapse the suit dimension for the set detector
        x_summed = torch.sum(x, dim=2, keepdim=True)
        out_b = self.branch_b(x_summed)
        out_b_flat = torch.flatten(out_b, start_dim=1)

        merged = torch.cat([out_a_flat, out_b_flat], dim=1)
        latent_spatial = self.activation(self.layer_norm(self.fc_project(merged)))

        return latent_spatial


class ScalarHead(nn.Module):
    """
    Scalar MLP Head processing the 28-feature tensor.
    """

    def __init__(self, in_features: int = 28):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OracleNet(nn.Module):
    """
    Task 4.2: Belief State Perception.
    Predicts the exact hidden cards held by all 3 opponents based on public history.
    """

    def __init__(self):
        super().__init__()

        # Oracle strictly views the 13 public channels
        self.spatial_head = SpatialHead(in_channels=13)
        self.scalar_head = ScalarHead(in_features=28)

        # WIDENED FUNNEL: Prevent the 168-output bottleneck
        self.fusion = nn.Sequential(
            nn.Linear(320, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )

        # Output exactly 168 features (3 opponents * 4 suits * 14 ranks)
        self.output_head = nn.Linear(512, 168)

    def forward(self, spatial_x: torch.Tensor, scalar_x: torch.Tensor) -> torch.Tensor:
        latent_spatial = self.spatial_head(spatial_x)
        latent_scalar = self.scalar_head(scalar_x)

        merged = torch.cat([latent_spatial, latent_scalar], dim=1)
        fused = self.fusion(merged)

        flat_logits = self.output_head(fused)

        # Reshape to (Batch, 3, 4, 14) and apply Sigmoid to generate raw probabilities
        reshaped_logits = flat_logits.view(-1, 3, 4, 14)
        return torch.sigmoid(reshaped_logits)



def expand_spatial_with_oracle(spatial_x: torch.Tensor, oracle_probs: torch.Tensor) -> torch.Tensor:
    """
    The Concatenation Trick: Merges the 13-channel public board with the
    3-channel Oracle predictions to create the 16-channel vision for Actor/Critic.
    """
    # Dim 1 is the Channel dimension (B, C, H, W)
    return torch.cat([spatial_x, oracle_probs], dim=1)



class CriticNet(nn.Module):
    """
    Task 4.3: Value Estimation.
    Calculates the Expected Value (EV) of the current board state in terminal points.
    """

    def __init__(self):
        super().__init__()

        # Critic views the 16-channel expanded board (13 Public + 3 Oracle)
        self.spatial_head = SpatialHead(in_channels=16)
        self.scalar_head = ScalarHead(in_features=28)

        self.fusion = nn.Sequential(
            nn.Linear(320, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # Outputs a single linear scalar representing the point delta
        self.output_head = nn.Linear(128, 1)

    def forward(self, expanded_spatial_x: torch.Tensor, scalar_x: torch.Tensor) -> torch.Tensor:
        latent_spatial = self.spatial_head(expanded_spatial_x)
        latent_scalar = self.scalar_head(scalar_x)

        merged = torch.cat([latent_spatial, latent_scalar], dim=1)
        fused = self.fusion(merged)

        return self.output_head(fused)


class ActorNet(nn.Module):
    """
    Task 4.3: Policy Formulation.
    Generates the 58-logit policy distribution for action selection, strictly masked.
    """

    def __init__(self):
        super().__init__()

        # Actor views the 16-channel expanded board (13 Public + 3 Oracle)
        self.spatial_head = SpatialHead(in_channels=16)
        self.scalar_head = ScalarHead(in_features=28)

        self.fusion = nn.Sequential(
            nn.Linear(320, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=0.2) # <--- NEW: Anti-Saturation layer before the final decision
        )

        # Outputs the unified 58 action logits
        self.output_head = nn.Linear(128, 58)

    def forward(self, expanded_spatial_x: torch.Tensor, scalar_x: torch.Tensor,
                action_mask: torch.Tensor = None) -> torch.Tensor:
        latent_spatial = self.spatial_head(expanded_spatial_x)
        latent_scalar = self.scalar_head(scalar_x)

        merged = torch.cat([latent_spatial, latent_scalar], dim=1)
        fused = self.fusion(merged)

        logits = self.output_head(fused)

        # Strict Late-Stage Masking
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        return logits


class JoeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # The Vision Specialists
        self.oracle_3p = OracleNet()
        self.oracle_4p = OracleNet()

        # The Decision Makers (Shared)
        self.actor = ActorNet()
        self.critic = CriticNet()

    def forward(self, spatial, scalar, mask):
        # Extract the 3-Player flag (assuming batch dimension: scalar[:, 22])
        is_3_player = scalar[:, 22].bool()

        # Create an empty tensor to hold the mixed batch results
        batch_size = spatial.shape[0]
        oracle_probs = torch.zeros((batch_size, 3, 4, 14), device=spatial.device)

        # Route the 3-player games to the 3P Specialist
        if is_3_player.any():
            # FIXED: Oracles require both spatial and scalar inputs
            probs_3p = self.oracle_3p(spatial[is_3_player], scalar[is_3_player])

            # --- ENFORCE ABSOLUTE ZERO ---
            # Manually kill any Sigmoid floating-point noise in the 3rd opponent slot
            probs_3p[:, 2, :, :] = 0.0

            oracle_probs[is_3_player] = probs_3p

        # Route the 4-player games to the 4P Specialist
        if (~is_3_player).any():
            # FIXED: Oracles require both spatial and scalar inputs
            oracle_probs[~is_3_player] = self.oracle_4p(spatial[~is_3_player], scalar[~is_3_player])

        # --- FIXED: The Concatenation Trick ---
        # Expand the spatial tensor from 13 channels to 16 channels BEFORE passing to Actor/Critic
        expanded_spatial = expand_spatial_with_oracle(spatial, oracle_probs)

        # Feed the combined 16-channel results to the shared Actor and Critic
        # FIXED: Pass the mask to the Actor
        actor_logits = self.actor(expanded_spatial, scalar, action_mask=mask)
        critic_value = self.critic(expanded_spatial, scalar)

        return actor_logits, critic_value, oracle_probs