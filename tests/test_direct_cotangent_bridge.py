from __future__ import annotations

import unittest

import torch

from blocket_league.direct_activation_lens import FrozenSoftPixelActivationLens
from blocket_league.direct_cotangent_bridge import (
    PixelChangeProbeBank,
    assert_no_physical_input_api,
    cotangent_poisson_bridge,
)
from blocket_league.direct_poisson_ph import DirectPoissonPHConfig, DirectPoissonPortHamiltonian
from blocket_league.direct_visual_poisson_ph import (
    DirectVisualPoissonPH,
    WholeStreamEncoderConfig,
    WholeStreamFrozenEncoder,
)
from blocket_league.end_to_end_ph_experiment import LatentPatchTransformerRenderer
from blocket_league.pixel_direct_model import DirectPixelTransformer, PixelDirectConfig


class DirectCotangentBridgeTests(unittest.TestCase):
    @staticmethod
    def _write_basis(lens, batch_size: int, port_size: int) -> torch.Tensor:
        raw = torch.randn(batch_size, *lens.activation_shape, port_size)
        flat = raw.flatten(1, 3)
        orthonormal, _ = torch.linalg.qr(flat, mode="reduced")
        return orthonormal.reshape_as(raw).detach()

    def _model(self) -> DirectVisualPoissonPH:
        backbone = DirectPixelTransformer(
            PixelDirectConfig(
                image_size=8,
                patch_size=4,
                palette_size=3,
                history_frames=2,
                pixel_embedding_size=2,
                hidden_size=6,
                depth=2,
                heads=2,
                mlp_ratio=2.0,
            )
        )
        encoder = WholeStreamFrozenEncoder(
            backbone,
            WholeStreamEncoderConfig(2, readout_hidden_size=6, lens_block=0),
        )
        renderer = LatentPatchTransformerRenderer(
            2,
            image_size=8,
            patch_size=4,
            palette_size=3,
            hidden_size=6,
            depth=1,
            heads=2,
        )
        core = DirectPoissonPortHamiltonian(
            DirectPoissonPHConfig(
                state_size=2,
                port_size=1,
                hidden_size=6,
                hidden_layers=1,
                coupling_layers=2,
                implicit_iterations=2,
                implicit_tolerance=0.0,
            )
        )
        return DirectVisualPoissonPH(encoder, renderer, core)

    def test_pixels_only_cotangent_is_pulled_back_before_poisson_sharp(self) -> None:
        torch.manual_seed(51)
        model = self._model()
        lens = FrozenSoftPixelActivationLens(
            model.encoder.backbone,
            intervention_block=0,
            horizons=(1,),
        )
        raw_probe = torch.randn(1, 3, 8, 8)
        probes = PixelChangeProbeBank(raw_probe)
        contexts = torch.randint(0, 3, (2, 2, 8, 8))
        result = cotangent_poisson_bridge(
            model,
            lens,
            probes,
            contexts,
            horizons=(1,),
            ridge=1e-3,
            extracted_write_basis=self._write_basis(lens, 2, 1),
        )
        self.assertTrue(bool(torch.isfinite(result.total)))
        self.assertEqual(result.activation_covectors[1].shape, (2, 2 * 4 * 6, 1))
        self.assertEqual(result.state_covectors[1].shape, (2, 2, 1))
        self.assertEqual(result.poisson_port_priors[1].shape, (2, 2, 1))
        result.total.backward()
        self.assertTrue(any(
            parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
            for parameter in model.encoder.readout.parameters()
        ))
        self.assertTrue(any(
            parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
            for parameter in model.core.port_network.parameters()
        ))
        self.assertTrue(all(parameter.grad is None for parameter in model.encoder.backbone.parameters()))

    def test_public_bridge_has_no_physical_input(self) -> None:
        assert_no_physical_input_api()

    def test_shuffled_bridge_only_permutes_precomputed_coupling_targets(self) -> None:
        torch.manual_seed(88)
        model = self._model()
        lens = FrozenSoftPixelActivationLens(
            model.encoder.backbone,
            intervention_block=0,
            horizons=(1,),
        )
        probes = PixelChangeProbeBank(torch.randn(1, 3, 8, 8))
        contexts = torch.randint(0, 3, (2, 2, 8, 8))
        paired = cotangent_poisson_bridge(
            model,
            lens,
            probes,
            contexts,
            horizons=(1,),
            ridge=1e-3,
            extracted_write_basis=self._write_basis(lens, 2, 1),
        )
        permutation = torch.tensor([1, 0], dtype=torch.long)
        shuffled = cotangent_poisson_bridge(
            model,
            lens,
            probes,
            contexts,
            horizons=(1,),
            ridge=1e-3,
            target_batch_permutation=permutation,
            extracted_write_basis=self._write_basis(lens, 2, 1),
        )

        # The visual derivative calculation itself is identical.  Only the
        # already-pulled-back target row coupled to B(x_i) changes.
        torch.testing.assert_close(
            shuffled.activation_covectors[1], paired.activation_covectors[1]
        )
        torch.testing.assert_close(
            shuffled.state_covectors[1], paired.state_covectors[1][permutation]
        )
        torch.testing.assert_close(
            shuffled.poisson_port_priors[1],
            paired.poisson_port_priors[1][permutation],
        )
        torch.testing.assert_close(
            shuffled.pullback_compatibility, paired.pullback_compatibility
        )
        torch.testing.assert_close(
            shuffled.horizon_consistency, paired.horizon_consistency
        )
        expected_total = (
            shuffled.port_alignment
            + shuffled.tangent_pushforward_alignment
            + 0.25 * shuffled.pullback_compatibility
            + 0.25 * shuffled.horizon_consistency
            + 0.10 * shuffled.port_isotropy
        )
        torch.testing.assert_close(shuffled.total, expected_total)

        with self.assertRaisesRegex(ValueError, "no fixed point"):
            cotangent_poisson_bridge(
                model,
                lens,
                probes,
                contexts,
                horizons=(1,),
                ridge=1e-3,
                target_batch_permutation=torch.arange(2),
                extracted_write_basis=self._write_basis(lens, 2, 1),
            )


if __name__ == "__main__":
    unittest.main()
