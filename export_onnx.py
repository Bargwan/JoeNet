import torch


class JoeNetDeployment(torch.nn.Module):
    """A lightweight wrapper that strips out the Critic for high-speed inference."""

    def __init__(self, base_model):
        super().__init__()
        self.oracle = base_model.oracle
        self.actor = base_model.actor

    def forward(self, spatial, scalar, mask):
        # 1. Oracle predicts hands
        oracle_probs = self.oracle(spatial, scalar)

        # 2. Manual concatenation (avoids importing external functions)
        expanded_spatial = torch.cat([spatial, oracle_probs], dim=1)

        # 3. Actor generates policy (Critic is completely bypassed)
        logits = self.actor(expanded_spatial, scalar, mask)

        return logits


def export_joenet_to_onnx(model, save_path="joenet.onnx"):
    """
    Exports the PyTorch JoeNet model to an ONNX graph for high-speed inference.
    """
    # 1. Force evaluation mode (disables dropout, locks batchnorm, etc.)
    model.eval()

    # --- NEW: Dynamically detect the model's device ---
    device = next(model.parameters()).device

    # 2. Create dummy inputs matching the exact tensor contracts (and move them to the device)
    batch_size = 1
    spatial = torch.randn(batch_size, 13, 4, 14, dtype=torch.float32, device=device)
    scalar = torch.randn(batch_size, 28, dtype=torch.float32, device=device)
    mask = torch.ones(batch_size, 58, dtype=torch.bool, device=device)

    # 3. Define dynamic axes (Removed 'value' and 'oracle')
    dynamic_axes = {
        'spatial': {0: 'batch_size'},
        'scalar': {0: 'batch_size'},
        'mask': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    }

    # 4. Export the graph
    torch.onnx.export(
        model,
        (spatial, scalar, mask),
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['spatial', 'scalar', 'mask'],
        output_names=['logits'],  # <--- ONLY EXPORTING LOGITS
        dynamic_axes=dynamic_axes
    )


if __name__ == "__main__":
    import os
    from network import JoeNet

    print("==================================================")
    print("           COMPILING JOENET TO ONNX               ")
    print("==================================================")

    # Load the victorious Phase 3 weights
    device = torch.device("cpu")  # Exporting is usually safest on CPU
    model = JoeNet().to(device)

    weights_path = "models/joenet_phase3_rl_final.pth"
    onnx_path = "models/joenet_phase3.onnx"

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

        # --- NEW: Wrap the model for deployment ---
        deployment_model = JoeNetDeployment(model)
        deployment_model.eval()

        export_joenet_to_onnx(deployment_model, onnx_path)
        print(f"-> Successfully exported LEAN Phase 3 model to: {onnx_path}")
    else:
        print(f"Error: Could not find weights at {weights_path}")