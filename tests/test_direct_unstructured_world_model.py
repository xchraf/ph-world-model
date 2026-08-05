from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import torch

from blocket_league.action_free_latent_effort import (
    UnstructuredLatentEffortDynamics,
)
from blocket_league.direct_cotangent_bridge import PixelChangeProbeBank
from blocket_league.direct_experiment_training import (
    DirectSystemSpec,
    DirectTrainingConfig,
    build_direct_bundle,
)
from blocket_league.direct_physical_evaluation import (
    _validate_primary_planner_isolation,
)
from blocket_league.direct_jacobian_port_extractor import (
    EmpiricalTangentConfig,
    make_synthetic_empirical_tangent_artifact_for_tests,
)
from blocket_league.direct_unstructured_world_model import (
    IndependentUnstructuredArchitecture,
    build_independent_unstructured_bundle,
    capture_homologous_initialization,
    independent_named_parameters,
    independent_tangent_lens_terms,
    independent_video_objective,
    unstructured_dynamics_parameter_count,
)
from blocket_league.direct_unstructured_training import (
    build_fresh_independent_baseline,
    train_independent_unstructured_world_model,
    validate_independent_checkpoint,
)
from blocket_league.direct_visual_poisson_ph import DirectVideoLossConfig
from blocket_league.passive_jacobian_ph_model import module_tensor_hash
from blocket_league.pixel_direct_model import DirectPixelTransformer, PixelDirectConfig


class IndependentUnstructuredWorldModelTests(unittest.TestCase):
    @staticmethod
    def _backbone() -> DirectPixelTransformer:
        return DirectPixelTransformer(
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

    @staticmethod
    def _config() -> DirectTrainingConfig:
        return DirectTrainingConfig(
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
            write_hidden_layers=1,
            port_tangent_channel_rank=4,
            port_tangent_neighbors=2,
            lens_horizons=(1,),
        )

    @staticmethod
    def _empirical_tangent(backbone, config, *, seed=808):
        tangent_config = EmpiricalTangentConfig(
            channel_rank=config.port_tangent_channel_rank,
            neighbors=config.port_tangent_neighbors,
            support_floor_ratio=config.port_support_floor_ratio,
        )
        artifact = make_synthetic_empirical_tangent_artifact_for_tests(
            history_frames=backbone.config.history_frames,
            patch_count=backbone.config.grid_size**2,
            hidden_size=backbone.config.hidden_size,
            config=tangent_config,
            seed=seed,
        )
        return artifact, tangent_config

    def _models(self):
        torch.manual_seed(91)
        backbone = self._backbone()
        config = self._config()
        system = DirectSystemSpec("toy", 2, 1, 0.05, lens_block=0)
        probes = PixelChangeProbeBank(torch.randn(1, 3, 4, 4))
        empirical_tangent, tangent_config = self._empirical_tangent(backbone, config)
        torch.manual_seed(10_771)
        structured = build_direct_bundle(
            backbone,
            system,
            probes,
            config,
            torch.device("cpu"),
            empirical_tangent=empirical_tangent,
        )
        target = sum(
            parameter.numel()
            for module in (
                structured.model,
                structured.write_field,
                structured.response_frame,
                structured.cotangent_frame,
            )
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        initialization = capture_homologous_initialization(
            encoder=structured.model.encoder,
            renderer=structured.model.renderer,
            effort_inference=structured.model.effort_inference,
            write_field=structured.write_field,
            response_frame=structured.response_frame,
            reference_initialization_seed=10_771,
        )
        architecture = IndependentUnstructuredArchitecture(
            state_size=2,
            port_size=1,
            dt=0.05,
            lens_block=0,
            state_hidden_size=6,
            renderer_hidden_size=6,
            renderer_depth=1,
            renderer_heads=2,
            dynamics_hidden_layers=1,
            write_hidden_size=4,
            write_hidden_layers=1,
            lens_horizons=(1,),
            initialization_seed=10_771,
        )
        independent = build_independent_unstructured_bundle(
            backbone,
            architecture,
            empirical_tangent=empirical_tangent,
            probes=probes,
            tangent_config=tangent_config,
            target_trainable_parameters=target,
            homologous_initialization=initialization,
            device=torch.device("cpu"),
        )
        return structured, independent, target

    def test_parameter_formula_matches_the_actual_state_dependent_drift_and_port(self) -> None:
        dynamics = UnstructuredLatentEffortDynamics(
            8, 2, 17, dt=0.05, hidden_layers=3
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in dynamics.parameters()),
            unstructured_dynamics_parameter_count(8, 2, 17, 3),
        )
        state = torch.randn(5, 8)
        self.assertEqual(dynamics.drift(state).shape, (5, 8))
        self.assertEqual(dynamics.port(state).shape, (5, 8, 2))

    def test_world_models_are_independent_homologously_initialized_and_matched(self) -> None:
        structured, independent, target = self._models()
        self.assertLessEqual(independent.relative_parameter_gap, 0.01)
        self.assertEqual(independent.target_trainable_parameters, target)
        self.assertEqual(
            independent.trainable_parameters,
            sum(value.numel() for _, value in independent_named_parameters(independent)),
        )
        structured_ids = {
            id(value)
            for value in structured.model.parameters()
            if value.requires_grad
        } | {id(value) for value in structured.write_field.parameters()}
        independent_ids = {id(value) for _, value in independent_named_parameters(independent)}
        self.assertTrue(structured_ids.isdisjoint(independent_ids))
        self.assertIsNot(structured.model.encoder, independent.model.encoder)
        self.assertIs(
            structured.model.encoder.backbone,
            independent.model.encoder.backbone,
        )
        self.assertIsNot(structured.model.renderer, independent.model.renderer)
        self.assertIsNot(
            structured.model.effort_inference,
            independent.model.effort_inference,
        )
        self.assertIsNot(structured.write_field, independent.write_field)
        self.assertEqual(
            module_tensor_hash(structured.model.renderer),
            module_tensor_hash(independent.model.renderer),
        )
        self.assertEqual(
            module_tensor_hash(structured.model.effort_inference),
            module_tensor_hash(independent.model.effort_inference),
        )
        self.assertEqual(
            module_tensor_hash(structured.write_field),
            module_tensor_hash(independent.write_field),
        )
        independent.model.encoder.assert_backbone_frozen()
        _validate_primary_planner_isolation(
            structured.model.encoder,
            structured.model.renderer,
            structured.model.core,
            independent.model.encoder,
            independent.model.renderer,
            independent.model.dynamics,
        )
        with self.assertRaisesRegex(ValueError, "module object"):
            _validate_primary_planner_isolation(
                structured.model.encoder,
                structured.model.renderer,
                structured.model.core,
                independent.model.encoder,
                structured.model.renderer,
                independent.model.dynamics,
            )

    def test_tangent_bridge_is_latent_multi_horizon_and_has_no_renderer_or_poisson(self) -> None:
        _, independent, _ = self._models()
        contexts = torch.randint(0, 3, (2, 2, 4, 4))
        terms, metrics = independent_tangent_lens_terms(
            independent, contexts, horizons=(1,)
        )
        self.assertEqual(set(terms), {"bridge", "oddness", "manifoldCycle"})
        self.assertNotIn("cotangentAlignment", metrics)
        self.assertIn("minimumUnstructuredResponseSingularValue", metrics)
        sum(terms.values()).backward()
        self.assertTrue(
            any(value.grad is not None for value in independent.model.dynamics.parameters())
        )
        self.assertEqual(sum(p.numel() for p in independent.write_field.parameters()), 0)
        self.assertTrue(
            all(value.grad is None for value in independent.model.renderer.parameters())
        )
        self.assertTrue(
            all(value.grad is None for value in independent.model.encoder.backbone.parameters())
        )

    def test_joint_objective_updates_own_visual_chain_and_all_non_ph_terms(self) -> None:
        _, independent, _ = self._models()
        contexts = torch.randint(0, 3, (2, 3, 2, 4, 4))
        frames = torch.randint(0, 3, (2, 3, 4, 4))
        states = independent.model.encode(contexts)
        lens_terms, _ = independent_tangent_lens_terms(
            independent, contexts[:, 0], horizons=(1,), encoded_states=states[:, 0]
        )
        loss, metrics = independent_video_objective(
            independent,
            contexts,
            frames,
            torch.ones(3),
            DirectVideoLossConfig(rollout_horizons=(1, 2)),
            lens_terms=lens_terms,
            encoded_states=states,
        )
        expected = {
            "reconstruction",
            "rolloutPixel",
            "rolloutLatent",
            "innovation",
            "whitening",
            "portFrameTransport",
            "portFrameHolonomy",
            "portRankOrientation",
            "jacobianBridge",
            "writeOddness",
            "manifoldCycle",
        }
        self.assertTrue(expected.issubset(metrics))
        self.assertNotIn("energyGauge", metrics)
        self.assertNotIn("chartConditioning", metrics)
        self.assertNotIn("chainRulePenalty", metrics)
        loss.backward()
        for module in (
            independent.model.encoder.pool_score,
            independent.model.encoder.readout,
            independent.model.renderer,
            independent.model.effort_inference,
            independent.model.dynamics,
        ):
            self.assertTrue(any(value.grad is not None for value in module.parameters()))
        self.assertEqual(sum(p.numel() for p in independent.write_field.parameters()), 0)
        self.assertTrue(
            all(value.grad is None for value in independent.model.encoder.backbone.parameters())
        )

    def test_fresh_builder_has_no_trained_structured_input_and_reconstructs_seed(self) -> None:
        backbone = self._backbone()
        config = self._config()
        system = DirectSystemSpec("toy", 2, 1, 0.05, lens_block=0)
        probes = PixelChangeProbeBank(torch.randn(1, 3, 4, 4))
        empirical_tangent, _ = self._empirical_tangent(backbone, config, seed=809)
        first = build_fresh_independent_baseline(
            backbone,
            system,
            probes,
            config,
            torch.device("cpu"),
            empirical_tangent=empirical_tangent,
            reference_initialization_seed=77_003,
        )
        # Ambient RNG and an unrelated mutated structured model cannot affect
        # the fresh reconstruction because neither is an API input.
        torch.manual_seed(9_999_999)
        unrelated = build_direct_bundle(
            backbone,
            system,
            PixelChangeProbeBank(probes.basis.clone()),
            config,
            torch.device("cpu"),
            empirical_tangent=empirical_tangent,
        )
        with torch.no_grad():
            for value in unrelated.model.renderer.parameters():
                value.add_(100.0)
        second = build_fresh_independent_baseline(
            backbone,
            system,
            probes,
            config,
            torch.device("cpu"),
            empirical_tangent=empirical_tangent,
            reference_initialization_seed=77_003,
        )
        self.assertEqual(
            first.homologous_initialization_hashes,
            second.homologous_initialization_hashes,
        )
        self.assertEqual(
            module_tensor_hash(first.model.renderer),
            module_tensor_hash(second.model.renderer),
        )
        self.assertNotEqual(
            module_tensor_hash(unrelated.model.renderer),
            module_tensor_hash(second.model.renderer),
        )

    def test_pixels_only_atomic_checkpoint_rejects_all_structured_leaks(self) -> None:
        backbone = self._backbone()
        config = self._config()
        system = DirectSystemSpec("toy", 2, 1, 0.05, lens_block=0)
        probes = PixelChangeProbeBank(torch.randn(1, 3, 4, 4))
        empirical_tangent, _ = self._empirical_tangent(backbone, config, seed=810)
        source_sha = "e" * 64
        data_seal = {
            "system": "toy",
            "fitAggregateSha256": "a" * 64,
            "fitSanitizedTensorSha256": "b" * 64,
            "validationAggregateSha256": "c" * 64,
            "validationSanitizedTensorSha256": "d" * 64,
        }
        pixels = {
            "pixelContexts": torch.randint(0, 3, (4, 3, 2, 4, 4)),
            "frames": torch.randint(0, 3, (4, 3, 4, 4)),
        }
        loss_config = DirectVideoLossConfig(rollout_horizons=(1, 2))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_root = root / "trainer" / "toy"
            archive_root.mkdir(parents=True)
            archive_paths = {
                "fit": archive_root / "fit-pixels.pt",
                "validation": archive_root / "validation-pixels.pt",
            }
            for path in archive_paths.values():
                torch.save({"frames": torch.zeros(1)}, path)
            bundle = build_fresh_independent_baseline(
                backbone,
                system,
                probes,
                config,
                torch.device("cpu"),
                empirical_tangent=empirical_tangent,
                reference_initialization_seed=10_771,
            )
            summary = train_independent_unstructured_world_model(
                bundle,
                pixels,
                {name: value.clone() for name, value in pixels.items()},
                torch.ones(3),
                system,
                root / "output",
                config,
                loss_config,
                data_seal=data_seal,
                pixel_archive_paths=archive_paths,
                source_tree_sha256=source_sha,
            )
            self.assertEqual(summary["optimizationTensorKeys"], ["pixelContexts", "frames"])
            payload = torch.load(root / "output" / "best.pt", weights_only=True)
            self.assertNotIn("model", payload)
            self.assertNotIn("fullModelHash", payload)
            self.assertNotIn("structuredModelHash", payload)
            self.assertNotIn("encodedPixelStates", payload["optimizationTensorKeys"])
            for field in (
                "encoderPoolScore",
                "encoderReadout",
                "renderer",
                "dynamics",
                "effortInference",
                "writeField",
                "responseFrame",
            ):
                self.assertTrue(all("backbone" not in name.lower() for name in payload[field]))

            verifier = build_fresh_independent_baseline(
                backbone,
                system,
                probes,
                config,
                torch.device("cpu"),
                empirical_tangent=empirical_tangent,
                reference_initialization_seed=10_771,
            )
            optimizer = torch.optim.AdamW(
                [value for _, value in independent_named_parameters(verifier)],
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            validate_independent_checkpoint(
                payload,
                verifier,
                optimizer,
                system=system,
                train_config=config,
                loss_config=loss_config,
                data_seal=data_seal,
                source_tree_sha256=source_sha,
                include_training_state=False,
            )
            for forbidden, value in (
                ("encodedPixelStates", torch.zeros(1)),
                ("fullModelHash", "f" * 64),
                ("structuredModelHash", "f" * 64),
                ("backbone", {"weight": torch.zeros(1)}),
            ):
                tampered = copy.deepcopy(payload)
                tampered[forbidden] = value
                with self.assertRaisesRegex(ValueError, "schema|forbidden"):
                    validate_independent_checkpoint(
                        tampered,
                        verifier,
                        optimizer,
                        system=system,
                        train_config=config,
                        loss_config=loss_config,
                        data_seal=data_seal,
                        source_tree_sha256=source_sha,
                        include_training_state=False,
                    )


if __name__ == "__main__":
    unittest.main()
