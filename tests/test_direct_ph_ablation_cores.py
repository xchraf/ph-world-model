from __future__ import annotations

import unittest

import torch

from blocket_league.direct_cotangent_bridge import PixelChangeProbeBank
from blocket_league.direct_experiment_training import (
    DirectSystemSpec,
    DirectTrainingConfig,
    build_direct_bundle,
)
from blocket_league.direct_ph_ablation_cores import (
    ConstantPortHamiltonian,
    SkewOnlyPortHamiltonian,
)
from blocket_league.direct_jacobian_port_extractor import (
    EmpiricalTangentConfig,
    make_synthetic_empirical_tangent_artifact_for_tests,
)
from blocket_league.direct_poisson_ph import DirectPoissonPHConfig
from blocket_league.pixel_direct_model import DirectPixelTransformer, PixelDirectConfig


class DirectPHAblationCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DirectPoissonPHConfig(
            state_size=4,
            port_size=2,
            hidden_size=8,
            hidden_layers=1,
            coupling_layers=2,
            implicit_iterations=3,
            implicit_tolerance=0.0,
        )
        self.state = torch.randn(3, 4)

    def test_skew_only_is_skew_but_not_forced_to_satisfy_jacobi(self) -> None:
        core = SkewOnlyPortHamiltonian(self.config)
        interconnection = core.interconnection(self.state)
        torch.testing.assert_close(
            interconnection + interconnection.transpose(-1, -2),
            torch.zeros_like(interconnection),
        )
        self.assertGreater(float(core.jacobi_tensor(self.state).abs().max()), 1e-9)

    def test_constant_port_is_identical_at_all_states(self) -> None:
        core = ConstantPortHamiltonian(self.config)
        torch.testing.assert_close(
            core.port(self.state), core.port(self.state + 10.0), atol=0.0, rtol=0.0
        )
        next_state = core.step(self.state, torch.randn(3, 2))
        self.assertTrue(bool(torch.isfinite(next_state).all()))

    def test_variant_specific_parameters_do_not_shift_matched_initialization(self) -> None:
        backbone = DirectPixelTransformer(
            PixelDirectConfig(
                image_size=4,
                patch_size=2,
                palette_size=3,
                history_frames=2,
                pixel_embedding_size=2,
                hidden_size=6,
                depth=2,
                heads=2,
                mlp_ratio=2.0,
            )
        )
        probes = PixelChangeProbeBank(
            torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4)
        )
        train_config = DirectTrainingConfig(
            steps=1,
            micro_batch_size=2,
            lens_batch_size=1,
            validation_batches=1,
            state_hidden_size=6,
            renderer_hidden_size=6,
            renderer_depth=1,
            renderer_heads=2,
            ph_hidden_size=6,
            ph_hidden_layers=1,
            coupling_layers=2,
            implicit_iterations=2,
            write_hidden_size=4,
            write_hidden_layers=1,
            port_tangent_channel_rank=4,
            port_tangent_neighbors=2,
            lens_horizons=(1,),
        )
        system = DirectSystemSpec("toy", 2, 1, 0.05, lens_block=0)
        empirical_tangent = make_synthetic_empirical_tangent_artifact_for_tests(
            history_frames=backbone.config.history_frames,
            patch_count=backbone.config.grid_size**2,
            hidden_size=backbone.config.hidden_size,
            config=EmpiricalTangentConfig(
                channel_rank=train_config.port_tangent_channel_rank,
                neighbors=train_config.port_tangent_neighbors,
                support_floor_ratio=train_config.port_support_floor_ratio,
            ),
            seed=443,
        )

        def construct(variant: str, device: torch.device):
            torch.manual_seed(12_345)
            cuda_before = (
                torch.cuda.get_rng_state(device).clone()
                if device.type == "cuda"
                else None
            )
            bundle = build_direct_bundle(
                backbone,
                system,
                PixelChangeProbeBank(probes.basis.clone()),
                train_config,
                device,
                empirical_tangent=empirical_tangent,
                variant=variant,
            )
            cpu_after = torch.get_rng_state().clone()
            cuda_after = (
                torch.cuda.get_rng_state(device).clone()
                if device.type == "cuda"
                else None
            )
            if cuda_before is not None:
                self.assertTrue(torch.equal(cuda_before, cuda_after))
            return bundle, cpu_after

        def assert_modules_equal(left: torch.nn.Module, right: torch.nn.Module) -> None:
            left_state = left.state_dict()
            right_state = right.state_dict()
            self.assertEqual(tuple(left_state), tuple(right_state))
            for name in left_state:
                self.assertTrue(
                    torch.equal(left_state[name], right_state[name]),
                    msg=f"matched initialization differs at {name}",
                )

        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            with self.subTest(device=str(device)):
                bundles = {
                    variant: construct(variant, device)
                    for variant in ("full", "skew_only", "constant_port")
                }
                full, full_rng = bundles["full"]
                skew, skew_rng = bundles["skew_only"]
                constant, constant_rng = bundles["constant_port"]
                self.assertTrue(torch.equal(full_rng, skew_rng))
                self.assertTrue(torch.equal(full_rng, constant_rng))

                for extract in (
                    lambda bundle: bundle.model.encoder,
                    lambda bundle: bundle.model.renderer,
                    lambda bundle: bundle.model.effort_inference,
                    lambda bundle: bundle.write_field,
                    lambda bundle: bundle.model.core.energy_network,
                    lambda bundle: bundle.model.core.resistance_network,
                    lambda bundle: bundle.response_frame,
                    lambda bundle: bundle.cotangent_frame,
                ):
                    assert_modules_equal(extract(full), extract(skew))
                    assert_modules_equal(extract(full), extract(constant))
                assert_modules_equal(
                    full.model.core.coordinate_map,
                    constant.model.core.coordinate_map,
                )
                assert_modules_equal(
                    full.model.core.port_network,
                    skew.model.core.port_network,
                )


if __name__ == "__main__":
    unittest.main()
