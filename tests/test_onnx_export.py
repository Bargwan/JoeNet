import unittest
import os
import torch
import numpy as np
import onnxruntime as ort

# We will build this in the Green phase
from export_onnx import export_joenet_to_onnx
from network import JoeNet


class TestONNXExport(unittest.TestCase):
    def test_onnx_export_and_inference(self):
        """
        Verify that the exported ONNX model produces mathematically identical
        outputs to the PyTorch model given the exact same tensor inputs.
        """
        # 1. SETUP: Initialize a blank model and set to evaluation mode
        model = JoeNet()
        model.eval()

        # Create dummy inputs matching the exact tensor specifications
        batch_size = 1
        spatial = torch.randn(batch_size, 13, 4, 14, dtype=torch.float32)
        scalar = torch.randn(batch_size, 28, dtype=torch.float32)
        mask = torch.ones(batch_size, 58, dtype=torch.bool)

        # 2. ACT: PyTorch Forward Pass
        with torch.no_grad():
            pt_logits, pt_value, pt_oracle = model(spatial, scalar, mask)

        # 3. ACT: Export to ONNX
        onnx_path = "temp_joenet_test.onnx"
        export_joenet_to_onnx(model, onnx_path)
        self.assertTrue(os.path.exists(onnx_path), "ONNX file was not created.")

        # 4. ACT: ONNX Runtime Forward Pass
        session = ort.InferenceSession(onnx_path)

        # ORT requires inputs to be explicitly mapped by their graph names
        ort_inputs = {
            session.get_inputs()[0].name: spatial.numpy(),
            session.get_inputs()[1].name: scalar.numpy(),
            session.get_inputs()[2].name: mask.numpy()
        }
        ort_logits, ort_value, ort_oracle = session.run(None, ort_inputs)

        # 5. ASSERT: Compare Outputs
        # We allow a tiny floating-point tolerance (1e-04) because C++ ONNX Runtime
        # calculates math slightly differently than Python PyTorch.
        np.testing.assert_allclose(pt_logits.numpy(), ort_logits, rtol=1e-04, atol=1e-05,
                                   err_msg="Logits mismatch!")
        np.testing.assert_allclose(pt_value.numpy(), ort_value, rtol=1e-04, atol=1e-05,
                                   err_msg="Value mismatch!")
        np.testing.assert_allclose(pt_oracle.numpy(), ort_oracle, rtol=1e-04, atol=1e-05,
                                   err_msg="Oracle mismatch!")

        # Cleanup
        if os.path.exists(onnx_path):
            os.remove(onnx_path)


if __name__ == '__main__':
    unittest.main()