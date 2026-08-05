from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch

from blocket_league.direct_cotangent_bridge import PixelChangeProbeBank
from blocket_league.direct_experiment_training import (
    DirectSystemSpec,
    DirectTrainingConfig,
    _atomic_torch_save,
    _atomic_json_save,
    _aggregated_pixels_only_lens_validation,
    _permute_correspondence_target,
    build_direct_bundle,
    jacobian_lens_terms,
)
from blocket_league.direct_jacobian_port_extractor import (
    EmpiricalTangentConfig,
    make_synthetic_empirical_tangent_artifact_for_tests,
)
from blocket_league.passive_jacobian_ph_model import module_tensor_hash
from blocket_league.pixel_direct_model import DirectPixelTransformer, PixelDirectConfig


class DirectExperimentTrainingTests(unittest.TestCase):
    @staticmethod
    def _empirical_tangent(backbone, config, *, seed=404):
        return make_synthetic_empirical_tangent_artifact_for_tests(
            history_frames=backbone.config.history_frames,
            patch_count=backbone.config.grid_size**2,
            hidden_size=backbone.config.hidden_size,
            config=EmpiricalTangentConfig(
                channel_rank=config.port_tangent_channel_rank,
                neighbors=config.port_tangent_neighbors,
                support_floor_ratio=config.port_support_floor_ratio,
            ),
            seed=seed,
        )

    def test_atomic_checkpoint_publish_preserves_previous_file_on_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.pt"
            _atomic_torch_save({"step": 1}, path)
            original = path.read_bytes()
            with mock.patch(
                "blocket_league.direct_experiment_training.torch.save",
                side_effect=RuntimeError("simulated timeout"),
            ):
                with self.assertRaises(RuntimeError):
                    _atomic_torch_save({"step": 2}, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(
                torch.load(path, map_location="cpu", weights_only=True)["step"], 1
            )

    def test_atomic_json_publish_preserves_previous_seal_on_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            _atomic_json_save({"step": 1}, path)
            original = path.read_bytes()
            with mock.patch.object(
                Path, "write_text", side_effect=RuntimeError("simulated timeout")
            ):
                with self.assertRaises(RuntimeError):
                    _atomic_json_save({"step": 2}, path)
            self.assertEqual(path.read_bytes(), original)

    def test_lens_validation_uses_disjoint_groups_means_and_worst_case_minima(self) -> None:
        group_count = 3
        group_size = 2
        rows = torch.arange(group_count * group_size, dtype=torch.long)
        suite = {
            "pixelContexts": rows[:, None, None, None, None].expand(
                -1, 1, 2, 1, 1
            ).clone(),
            "frames": torch.zeros(group_count * group_size, 2, 1, 1),
        }
        config = DirectTrainingConfig(
            steps=1,
            micro_batch_size=group_size,
            lens_batch_size=group_size,
            validation_batches=group_count,
        )
        seen_rows = []
        minimum_signal = (0.8, 0.3, 0.5)
        minimum_frozen_rank = (4.0, 2.0, 3.0)
        minimum_ph_rank = (7.0, 1.0, 5.0)

        def fake_lens_terms(_bundle, contexts, **kwargs):
            selected = tuple(int(value) for value in contexts[:, 0, 0, 0])
            seen_rows.append((selected, kwargs))
            group = selected[0] // group_size
            scalar = torch.tensor(float(group + 1))
            return (
                {
                    "bridge": scalar,
                    "oddness": 2.0 * scalar,
                    "manifoldCycle": 3.0 * scalar,
                },
                {
                    "responseAlignment": 4.0 * scalar,
                    "writeFirstOrderSignal": torch.tensor(minimum_signal[group]),
                    "minimumFrozenResponseSingularValue": torch.tensor(
                        minimum_frozen_rank[group]
                    ),
                    "minimumPHResponseSingularValue": torch.tensor(
                        minimum_ph_rank[group]
                    ),
                },
            )

        with mock.patch(
            "blocket_league.direct_experiment_training.jacobian_lens_terms",
            side_effect=fake_lens_terms,
        ):
            result = _aggregated_pixels_only_lens_validation(
                object(),
                suite,
                config,
                torch.device("cpu"),
                variant="shuffled_lens",
            )

        self.assertEqual(
            tuple(item[0] for item in seen_rows),
            ((0, 1), (2, 3), (4, 5)),
        )
        self.assertTrue(all(item[1]["shuffled"] for item in seen_rows))
        self.assertTrue(all(item[1]["horizons"] == (1, 2, 4) for item in seen_rows))
        self.assertEqual(result["lensBridge"], 2.0)
        self.assertEqual(result["lensOddness"], 4.0)
        self.assertEqual(result["lensManifoldCycle"], 6.0)
        self.assertEqual(result["lensResponseAlignment"], 8.0)
        self.assertAlmostEqual(result["lensWriteFirstOrderSignal"], 0.3, places=6)
        self.assertEqual(result["lensMinimumFrozenResponseSingularValue"], 2.0)
        self.assertEqual(result["lensMinimumPHResponseSingularValue"], 1.0)
        self.assertEqual(result["lensValidationGroups"], 3.0)
        self.assertEqual(result["lensValidationContexts"], 6.0)

        with self.assertRaisesRegex(ValueError, "distinct trajectories"):
            _aggregated_pixels_only_lens_validation(
                object(),
                {name: value[:5] for name, value in suite.items()},
                config,
                torch.device("cpu"),
                variant="full",
            )

    def test_shuffle_is_only_a_fixed_point_free_target_permutation(self) -> None:
        target = torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2)

        paired = _permute_correspondence_target(target, shuffled=False)
        shuffled = _permute_correspondence_target(target, shuffled=True)

        self.assertIs(paired, target)
        torch.testing.assert_close(shuffled, target.roll(1, dims=0))
        self.assertTrue(bool((shuffled != target).reshape(3, -1).any(dim=1).all()))
        torch.testing.assert_close(
            shuffled.flatten().sort().values,
            target.flatten().sort().values,
        )
        torch.testing.assert_close(
            torch.linalg.matrix_norm(shuffled),
            torch.linalg.matrix_norm(target).roll(1),
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            _permute_correspondence_target(target[:1], shuffled=True)

    def test_all_variants_share_exact_common_initialization_and_rng_lineage(self) -> None:
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
        ).eval().requires_grad_(False)
        raw_probe = torch.randn(1, 3, 4, 4)
        config = DirectTrainingConfig(
            steps=1,
            micro_batch_size=2,
            lens_batch_size=2,
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
            port_tangent_channel_rank=4,
            port_tangent_neighbors=2,
            lens_horizons=(1,),
        )
        empirical_tangent = self._empirical_tangent(backbone, config)
        variants = (
            "full",
            "no_jacobian",
            "single_horizon",
            "shuffled_lens",
            "skew_only",
            "constant_port",
        )
        bundles = {}
        terminal_rng_states = {}
        for variant in variants:
            torch.manual_seed(9_174)
            bundles[variant] = build_direct_bundle(
                backbone,
                DirectSystemSpec("toy", 2, 1, 0.05, lens_block=0),
                PixelChangeProbeBank(raw_probe.clone()),
                config,
                torch.device("cpu"),
                empirical_tangent=empirical_tangent,
                variant=variant,
            )
            terminal_rng_states[variant] = torch.get_rng_state().clone()

        def common_hashes(variant: str) -> dict[str, str]:
            bundle = bundles[variant]
            return {
                "encoder": module_tensor_hash(bundle.model.encoder),
                "renderer": module_tensor_hash(bundle.model.renderer),
                "effortInference": module_tensor_hash(bundle.model.effort_inference),
                "writeField": module_tensor_hash(bundle.write_field),
                "responseFrame": module_tensor_hash(bundle.response_frame),
                "cotangentFrame": module_tensor_hash(bundle.cotangent_frame),
                "lens": module_tensor_hash(bundle.lens),
                "probes": module_tensor_hash(bundle.probes),
            }

        reference_hashes = common_hashes("full")
        reference_rng = terminal_rng_states["full"]
        reference_core = bundles["full"].model.core.state_dict()
        for variant in variants[1:]:
            self.assertEqual(common_hashes(variant), reference_hashes)
            self.assertTrue(torch.equal(terminal_rng_states[variant], reference_rng))
            observed_core = bundles[variant].model.core.state_dict()
            shared_core_keys = set(reference_core).intersection(observed_core)
            self.assertTrue(shared_core_keys)
            for name in shared_core_keys:
                self.assertTrue(
                    torch.equal(reference_core[name], observed_core[name]),
                    msg=f"shared core tensor {name!r} differs for {variant}",
                )

    def test_joint_tangent_and_cotangent_bridge_has_gradients(self) -> None:
        torch.manual_seed(72)
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
        probes = PixelChangeProbeBank(torch.randn(1, 3, 4, 4))
        config = DirectTrainingConfig(
            steps=1,
            micro_batch_size=2,
            lens_batch_size=2,
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
            port_tangent_channel_rank=4,
            port_tangent_neighbors=2,
            lens_horizons=(1,),
        )
        empirical_tangent = self._empirical_tangent(backbone, config, seed=405)
        bundle = build_direct_bundle(
            backbone,
            DirectSystemSpec("toy", 2, 1, 0.05, lens_block=0),
            probes,
            config,
            torch.device("cpu"),
            empirical_tangent=empirical_tangent,
            variant="full",
        )
        contexts = torch.randint(0, 3, (2, 2, 4, 4))
        terms, metrics = jacobian_lens_terms(
            bundle, contexts, horizons=(1,), ridge=1e-3
        )
        loss = terms["bridge"] + terms["oddness"] + terms["manifoldCycle"]
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIn("cotangentAlignment", metrics)
        self.assertEqual(sum(p.numel() for p in bundle.write_field.parameters()), 0)
        self.assertTrue(any(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in bundle.model.encoder.readout.parameters()
        ))
        self.assertTrue(any(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in bundle.model.core.port_network.parameters()
        ))
        # K_h U and G_h^pH are both direct-state responses.  The renderer is
        # intentionally absent from this preregistered tangent bridge.
        self.assertTrue(all(
            parameter.grad is None for parameter in bundle.model.renderer.parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None for parameter in bundle.model.encoder.backbone.parameters()
        ))

        # The shuffled control executes the same tangent and cotangent losses.
        # Only their precomputed target rows are permuted.
        bundle.model.zero_grad(set_to_none=True)
        shuffled_terms, shuffled_metrics = jacobian_lens_terms(
            bundle,
            contexts,
            horizons=(1,),
            ridge=1e-3,
            shuffled=True,
        )
        self.assertEqual(set(shuffled_terms), set(terms))
        self.assertEqual(set(shuffled_metrics), set(metrics))
        self.assertTrue(all(bool(torch.isfinite(value)) for value in shuffled_terms.values()))
        self.assertTrue(all(bool(torch.isfinite(value)) for value in shuffled_metrics.values()))
        self.assertGreater(
            float(
                shuffled_metrics["cotangentCompatibility"].detach()
                + shuffled_metrics["persistentCotangentFrameAlignment"].detach()
            ),
            0.0,
        )
        shuffled_loss = sum(shuffled_terms.values())
        shuffled_loss.backward()
        self.assertEqual(sum(p.numel() for p in bundle.write_field.parameters()), 0)
        self.assertTrue(any(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in bundle.model.core.port_network.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
