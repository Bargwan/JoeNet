import unittest
import torch

from network import SpatialHead, ScalarHead, OracleNet, expand_spatial_with_oracle


class TestSharedRepresentationHeads(unittest.TestCase):

    def test_spatial_head_13_channels_oracle(self):
        """
        Verify SpatialHead processes the standard 13-channel public tensor
        down to a 256-feature latent space for the OracleNet.
        """
        spatial_head = SpatialHead(in_channels=13)
        dummy_spatial = torch.zeros((2, 13, 4, 14), dtype=torch.float32)

        latent_vector = spatial_head(dummy_spatial)

        self.assertEqual(
            latent_vector.shape,
            (2, 256),
            "Oracle SpatialHead failed to output (Batch, 256)"
        )

    def test_spatial_head_16_channels_actor_critic(self):
        """
        Verify SpatialHead processes the 16-channel expanded tensor
        (Public + Oracle Concatenation Trick) down to the same 256-feature space.
        """
        spatial_head = SpatialHead(in_channels=16)
        dummy_expanded = torch.zeros((2, 16, 4, 14), dtype=torch.float32)

        latent_vector = spatial_head(dummy_expanded)

        self.assertEqual(
            latent_vector.shape,
            (2, 256),
            "Actor/Critic SpatialHead failed to output (Batch, 256) on 16 channels"
        )

        # Verify the CNN flattening math remains stable regardless of in_channels
        linear_layers = [m for m in spatial_head.modules() if isinstance(m, torch.nn.Linear)]
        self.assertEqual(
            linear_layers[0].in_features,
            1216,
            "The flattened CNN branches must concatenate to exactly 1216 features!"
        )

    def test_scalar_head_dimensions(self):
        """
        Verify the Scalar MLP Head processes the 28-feature tensor
        down to a 64-feature latent space.
        """
        scalar_head = ScalarHead(in_features=28)
        dummy_scalar = torch.zeros((2, 28), dtype=torch.float32)

        latent_scalar = scalar_head(dummy_scalar)

        self.assertEqual(
            latent_scalar.shape,
            (2, 64),
            "ScalarHead failed to output (Batch, 64)"
        )


class TestOracleNetAndConcat(unittest.TestCase):

    def test_oracle_net_dimensions_and_activation(self):
        """
        Verify OracleNet outputs the strict (B, 3, 4, 14) shape and
        applies the Sigmoid activation to bound probabilities.
        """
        oracle = OracleNet()
        dummy_spatial = torch.zeros((2, 13, 4, 14), dtype=torch.float32)
        dummy_scalar = torch.zeros((2, 28), dtype=torch.float32)

        # ACT: Forward Pass
        probs = oracle(dummy_spatial, dummy_scalar)

        # ASSERT: Shape must exactly match the 3 Opponents
        self.assertEqual(
            probs.shape,
            (2, 3, 4, 14),
            "OracleNet output shape must be (Batch, 3, 4, 14)"
        )

        # ASSERT: Sigmoid activation strictly bounds between 0.0 and 1.0
        self.assertTrue(
            torch.all(probs >= 0.0) and torch.all(probs <= 1.0),
            "OracleNet probabilities must be strictly bounded between 0.0 and 1.0"
        )

    def test_the_concat_trick(self):
        """
        Verify the utility function perfectly merges the 13-channel board
        and the 3-channel predictions into the final 16-channel Expanded Tensor.
        """
        # 1. Setup mock tensors
        dummy_spatial = torch.zeros((2, 13, 4, 14), dtype=torch.float32)

        # We fill the mock oracle with 1.0s so we can track it after the merge
        dummy_oracle_probs = torch.ones((2, 3, 4, 14), dtype=torch.float32)

        # 2. ACT: Execute the Concat Trick
        expanded = expand_spatial_with_oracle(dummy_spatial, dummy_oracle_probs)

        # 3. ASSERT: Dimensions
        self.assertEqual(
            expanded.shape,
            (2, 16, 4, 14),
            "Expanded spatial tensor must be strictly 16 channels"
        )

        # 4. ASSERT: Data integrity
        # Check that the last 3 channels (Indices 13, 14, 15) are exactly the oracle probs
        sum_of_oracle_channels = torch.sum(expanded[:, 13:16, :, :])
        expected_sum = 2 * 3 * 4 * 14  # Batch(2) * Channels(3) * Suits(4) * Ranks(14)

        self.assertEqual(
            sum_of_oracle_channels,
            expected_sum,
            "Oracle probabilities were not correctly concatenated to the end of the tensor"
        )

if __name__ == '__main__':
    unittest.main()