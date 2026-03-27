import torch


def export_joenet_to_onnx(model, save_path="joenet.onnx"):
    """
    Exports the PyTorch JoeNet model to an ONNX graph for high-speed inference.
    """
    # 1. Force evaluation mode (disables dropout, locks batchnorm, etc.)
    model.eval()

    # 2. Create dummy inputs matching the exact tensor contracts
    batch_size = 1
    spatial = torch.randn(batch_size, 13, 4, 14, dtype=torch.float32)
    scalar = torch.randn(batch_size, 28, dtype=torch.float32)
    mask = torch.ones(batch_size, 58, dtype=torch.bool)

    # 3. Define dynamic axes
    # This tells ONNX that the 0th dimension (batch size) can change,
    # allowing us to use this same file for single-agent inference or batch processing later.
    dynamic_axes = {
        'spatial': {0: 'batch_size'},
        'scalar': {0: 'batch_size'},
        'mask': {0: 'batch_size'},
        'logits': {0: 'batch_size'},
        'value': {0: 'batch_size'},
        'oracle': {0: 'batch_size'}
    }

    # 4. Export the graph
    torch.onnx.export(
        model,
        (spatial, scalar, mask),
        save_path,
        export_params=True,
        opset_version=14,  # Opset 14 is highly stable and fully supported by Godot/ORT
        do_constant_folding=True,  # Optimizes constant math out of the graph
        input_names=['spatial', 'scalar', 'mask'],
        output_names=['logits', 'value', 'oracle'],
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
        model.load_state_dict(torch.load(weights_path, map_location=device))
        export_joenet_to_onnx(model, onnx_path)
        print(f"-> Successfully exported Phase 3 model to: {onnx_path}")
    else:
        print(f"Error: Could not find weights at {weights_path}")