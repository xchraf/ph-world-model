from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from blocket_league.action_free_excitation import (
    HiddenExcitationConfig,
    action_free_environment_config_sha256,
    hidden_excitation_config_sha256,
)
from blocket_league.direct_cotangent_bridge import PixelChangeProbeBank
from blocket_league.direct_action_free_data import (
    ActionFreeBackboneTrainConfig,
    PixelsOnlyManifest,
    class_weights,
    make_optimization_suite,
    sanitized_pixel_tensor_sha256,
)
from blocket_league.direct_experiment_training import (
    DIRECT_SYSTEMS,
    DirectTrainingConfig,
    _named_optimized_parameters,
    build_direct_bundle,
)
from blocket_league.direct_jacobian_port_extractor import (
    EmpiricalTangentConfig,
    build_empirical_tangent_artifact,
)
from blocket_league.direct_jacobian_port_precompute import (
    JacobianPortPrecomputeConfig,
    build_empirical_tangent_from_pixels,
)
from blocket_league.direct_unstructured_training import (
    build_fresh_independent_baseline,
    train_independent_unstructured_world_model,
)
from blocket_league.learner_source_bundle import build_learner_source_bundle
from blocket_league.direct_jacobian_poisson_ph_experiment import ExperimentFConfig
from blocket_league.direct_ph_structure_audits import LensAuditEvidence
from blocket_league.passive_jacobian_ph_model import module_tensor_hash
from blocket_league.direct_postfreeze_runner import (
    ControlShard,
    Gate4CollectionConfig,
    PostFreezePaths,
    REGISTERED_CONTROL_EPISODES,
    REQUIRED_POSTFREEZE_VARIANTS,
    _DIRECT_CHECKPOINT_KEYS,
    _expected_nonbackbone_optimizer_parameter_names,
    _file_sha256,
    _freeze_bundle,
    _gate4_batch,
    _gate4_path_provenance,
    _gate4_sealed_extractor_sha256,
    audit_gate1_postfreeze,
    build_frozen_activation_world_model,
    compose_single_seed_outcome,
    load_postfreeze_system,
    merge_control_shards,
    registered_control_shard_ranges,
    validate_direct_checkpoint_metadata,
)
from blocket_league.direct_visual_poisson_ph import DirectVideoLossConfig
from blocket_league.direct_physical_evaluation import (
    ControlResult,
    SYSTEMS,
    fixed_interfaces,
    linear_interface_protocol,
)
from blocket_league.pixel_direct_model import (
    DirectPixelTransformer,
    PixelDirectConfig,
    pixel_direct_config_for_preset,
)
from blocket_league.source_provenance import build_source_manifest
from blocket_league.runtime_firewall_trace import RuntimeFirewallTrace


class PostFreezeMetadataTests(unittest.TestCase):
    def test_activation_runner_routes_the_sealed_pixel_probe_bank(self) -> None:
        source = inspect.getsource(build_frozen_activation_world_model)
        self.assertIn("bundle.write_field", source)
        self.assertIn("bundle.probes", source)
        self.assertNotIn("StateConditionedActivationWriteField", source)

    def _metadata(self) -> dict[str, object]:
        system = DIRECT_SYSTEMS["pendulum"]
        config = DirectTrainingConfig(
            steps=3,
            state_hidden_size=8,
            renderer_hidden_size=8,
            renderer_depth=1,
            renderer_heads=2,
            ph_hidden_size=8,
            ph_hidden_layers=1,
            coupling_layers=2,
            write_hidden_size=4,
            write_hidden_layers=1,
        )
        payload = {key: None for key in _DIRECT_CHECKPOINT_KEYS}
        payload.update(
            {
                "kind": "direct_jacobian_poisson_port_hamiltonian",
                "actionChannels": 0,
                "physicalStateChannels": 0,
                "optimizationTensorKeys": ["pixelContexts", "frames"],
                "system": asdict(system),
                "variant": "full",
                "step": 3,
                "bestValidation": 1.25,
                "bestStructureEligible": True,
                "trainConfig": asdict(config),
                "lossConfig": asdict(DirectVideoLossConfig()),
                "backboneHash": "b" * 64,
                "probeHash": "c" * 64,
                "dataSeal": {
                    "system": "pendulum",
                    "fitAggregateSha256": "a" * 64,
                    "fitSanitizedTensorSha256": "b" * 64,
                    "validationAggregateSha256": "c" * 64,
                    "validationSanitizedTensorSha256": "d" * 64,
                },
                "optimizedParameterNames": ["model.readout.weight"],
                "sourceTreeSha256": "d" * 64,
            }
        )
        return payload

    def test_registered_loader_variants_are_exact(self) -> None:
        self.assertEqual(
            REQUIRED_POSTFREEZE_VARIANTS,
            (
                "full",
                "no_jacobian",
                "single_horizon",
                "shuffled_lens",
                "skew_only",
                "constant_port",
            ),
        )

    def test_direct_metadata_is_fail_closed(self) -> None:
        payload = self._metadata()
        validate_direct_checkpoint_metadata(
            payload,
            system_name="pendulum",
            variant="full",
            backbone_hash="b" * 64,
            source_tree_sha256="d" * 64,
        )
        payload["actionChannels"] = 1
        with self.assertRaisesRegex(ValueError, "forbidden physical"):
            validate_direct_checkpoint_metadata(
                payload,
                system_name="pendulum",
                variant="full",
                backbone_hash="b" * 64,
                source_tree_sha256="d" * 64,
            )

    def test_direct_metadata_rejects_extra_sidecar_key(self) -> None:
        payload = self._metadata()
        payload["simulatorState"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            validate_direct_checkpoint_metadata(
                payload,
                system_name="pendulum",
                variant="full",
                backbone_hash="b" * 64,
                source_tree_sha256="d" * 64,
            )

    def test_gate4_collection_budget_is_locked(self) -> None:
        valid = Gate4CollectionConfig()
        self.assertEqual(valid.samples, 128)
        self.assertEqual(valid.random_draws, 16)
        self.assertEqual(valid.horizons, (1, 2, 4))
        with self.assertRaisesRegex(ValueError, "128"):
            Gate4CollectionConfig(samples=127)
        with self.assertRaisesRegex(ValueError, "16"):
            Gate4CollectionConfig(random_draws=15)

    def test_loader_reconstructs_all_bundles_and_rejects_manifest_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_manifest = build_source_manifest()
            source_tree_sha256 = source_manifest["treeSha256"]
            learner_root = root / "learner-source"
            learner_manifest = build_learner_source_bundle(
                Path(__file__).resolve().parents[1],
                source_manifest,
                learner_root,
            )
            learner_cache = root / "learner-cache"
            learner_cache.mkdir()
            sanitized_root = root / "sanitized"
            output = root / "output"
            (sanitized_root / "trainer-mount" / "pendulum").mkdir(parents=True)
            (sanitized_root / "heldout" / "pendulum").mkdir(parents=True)
            (sanitized_root / "seals" / "pendulum").mkdir(parents=True)
            (output / "backbone").mkdir(parents=True)
            (output / "baseline-independent").mkdir(parents=True)

            pixels = torch.randint(0, 9, (32, 10, 8, 8), dtype=torch.uint8)
            test_manifest = PixelsOnlyManifest(
                system="pendulum",
                trajectories=32,
                frames_per_trajectory=10,
                image_size=8,
                aggregate_sha256="c" * 64,
                sanitized_tensor_sha256=sanitized_pixel_tensor_sha256(pixels),
            )
            torch.save(
                {"pixels": pixels, "manifest": asdict(test_manifest)},
                sanitized_root / "heldout" / "pendulum" / "test-pixels.pt",
            )
            fit_manifest = PixelsOnlyManifest(
                system="pendulum",
                trajectories=32,
                frames_per_trajectory=10,
                image_size=8,
                aggregate_sha256="a" * 64,
                sanitized_tensor_sha256=sanitized_pixel_tensor_sha256(pixels),
            )
            validation_manifest = PixelsOnlyManifest(
                system="pendulum",
                trajectories=32,
                frames_per_trajectory=10,
                image_size=8,
                aggregate_sha256="d" * 64,
                sanitized_tensor_sha256=sanitized_pixel_tensor_sha256(pixels),
            )
            for split, manifest in (
                ("fit", fit_manifest),
                ("validation", validation_manifest),
            ):
                torch.save(
                    {"pixels": pixels.clone(), "manifest": asdict(manifest)},
                    sanitized_root
                    / "trainer-mount"
                    / "pendulum"
                    / f"{split}-pixels.pt",
                )
            producer_trace = RuntimeFirewallTrace(
                sanitized_root / "seals" / "pendulum" / "firewall-trace.jsonl",
                stage="producer:pendulum",
                source_tree_sha256=source_tree_sha256,
            )
            for _ in range(96):
                producer_trace.record_tensor_payload(
                    phase="producer",
                    role="raw_action_erased_video",
                    tensors={"frames": torch.zeros(10, 8, 8, 3)},
                )
            producer_runtime_seal = producer_trace.snapshot().to_dict()
            producer_trace.close()
            excitation_config = HiddenExcitationConfig(frames=10, image_size=8)
            (sanitized_root / "seals" / "pendulum" / "manifest.json").write_text(
                json.dumps(
                    {
                        "system": "pendulum",
                        "generationEnvironmentSha256": (
                            action_free_environment_config_sha256(
                                "pendulum", image_size=8
                            )
                        ),
                        "splits": {
                            "fit": asdict(fit_manifest),
                            "validation": asdict(validation_manifest),
                            "test": asdict(test_manifest),
                        },
                        "producerSeedSerialized": False,
                        "physicalCommandsSerialized": False,
                        "simulatorStatesSerialized": False,
                        "sourceTreeSha256": source_tree_sha256,
                        "runtimeTrace": producer_runtime_seal,
                        "hiddenExcitationConfig": asdict(excitation_config),
                        "hiddenExcitationConfigSha256": (
                            hidden_excitation_config_sha256(excitation_config)
                        ),
                    }
                ),
                encoding="utf-8",
            )
            model_config = pixel_direct_config_for_preset(
                "tiny",
                image_size=8,
                patch_size=4,
                palette_size=9,
                history_frames=2,
            )
            backbone = DirectPixelTransformer(model_config).eval().requires_grad_(False)
            backbone_hash = module_tensor_hash(backbone)
            torch.save(
                {
                    "kind": "passive_direct_pixel_world_model",
                    "system": "pendulum",
                    "actionChannels": 0,
                    "optimizationTensorKeys": ["pixels"],
                    "pixelsOnlyManifest": asdict(fit_manifest),
                    "model": dict(backbone.state_dict()),
                    "model_config": model_config.to_dict(),
                    "train_config": asdict(ActionFreeBackboneTrainConfig(steps=1)),
                    "step": 1,
                },
                output / "backbone" / "checkpoint.pt",
            )

            runtime_entries = []

            def write_runtime_trace(
                phase: str, relative_path: str, tensor_keys: tuple[str, ...]
            ) -> dict[str, object]:
                path = output / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                trace = RuntimeFirewallTrace(
                    path,
                    stage=phase,
                    source_tree_sha256=source_tree_sha256,
                )
                trace.record_file_read(
                    learner_root / "learner-source-manifest.json",
                    role="learner_source_manifest",
                    serialized_keys=tuple(sorted(learner_manifest)),
                    semantic_sha256=learner_manifest["treeSha256"],
                )
                trace.record_recursive_manifest(
                    learner_root, role="learner_source_bundle"
                )
                trace.record_recursive_manifest(
                    learner_cache, role="learner_cache:0"
                )
                trace.record_mount_manifest(
                    sanitized_root / "trainer-mount", role="trainer_mount_root"
                )
                trace.record_file_read(
                    sanitized_root
                    / "trainer-mount"
                    / "pendulum"
                    / "fit-pixels.pt",
                    role="trainer_archive:fit",
                    serialized_keys=("manifest", "pixels"),
                )
                optimized = torch.nn.Parameter(torch.ones(1))
                protected = torch.nn.Parameter(
                    torch.zeros(1), requires_grad=False
                )
                trace.record_optimizer(
                    phase=phase,
                    named_parameters={"optimized.weight": optimized},
                    protected_parameters=(
                        {}
                        if phase == "backbone"
                        else {"encoder.backbone.weight": protected}
                    ),
                )
                trace.record_gradient_batch(
                    phase=phase,
                    step=1,
                    tensors={key: torch.zeros(1) for key in tensor_keys},
                )
                trace.record_backbone_boundary(
                    phase=phase,
                    boundary="selected_checkpoint",
                    sha256=backbone_hash,
                )
                seal = trace.snapshot().to_dict()
                trace.close()
                entry = {
                    "phase": phase,
                    "relativePath": relative_path,
                    "seal": seal,
                }
                runtime_entries.append(entry)
                return seal

            backbone_runtime_seal = write_runtime_trace(
                "backbone", "backbone/firewall-trace.jsonl", ("pixels",)
            )

            direct_config = DirectTrainingConfig(
                steps=1,
                state_hidden_size=8,
                renderer_hidden_size=8,
                renderer_depth=1,
                renderer_heads=2,
                ph_hidden_size=8,
                ph_hidden_layers=1,
                coupling_layers=2,
                implicit_iterations=4,
                write_hidden_size=4,
                write_hidden_layers=1,
                port_tangent_channel_rank=4,
                port_tangent_neighbors=2,
            )
            loss_config = DirectVideoLossConfig()
            baseline_config = direct_config
            experiment_seed = 151_910_737
            experiment_config = ExperimentFConfig(
                seed=experiment_seed,
                fit_trajectories=32,
                validation_trajectories=32,
                test_trajectories=32,
                history_frames=2,
                transitions=8,
                cache_frames=10,
                image_size=8,
                patch_size=4,
                backbone_preset="tiny",
                variants=REQUIRED_POSTFREEZE_VARIANTS,
            )
            port_config = JacobianPortPrecomputeConfig(
                contexts=4,
                batch_size=2,
                lens_block=DIRECT_SYSTEMS["pendulum"].lens_block,
                horizons=direct_config.lens_horizons,
                channel_rank=direct_config.port_tangent_channel_rank,
                neighbors=direct_config.port_tangent_neighbors,
                support_floor_ratio=direct_config.port_support_floor_ratio,
            )
            distributed_config = {
                "kind": "direct_distributed_training_config",
                "system": "pendulum",
                "experimentConfig": asdict(experiment_config),
                "backboneConfig": asdict(ActionFreeBackboneTrainConfig(steps=1)),
                "portConfig": asdict(port_config),
                "directConfig": asdict(direct_config),
                "baselineConfig": asdict(baseline_config),
                "lossConfig": asdict(loss_config),
                "manifests": {
                    "fit": asdict(fit_manifest),
                    "validation": asdict(validation_manifest),
                },
                "sourceManifest": source_manifest,
                "learnerSourceManifest": learner_manifest,
                "actionGradientUpdates": 0,
                "physicalStateGradientUpdates": 0,
            }
            canonical_distributed_config = json.loads(
                json.dumps(distributed_config, allow_nan=False)
            )
            (output / "distributed-config.json").write_text(
                json.dumps(canonical_distributed_config), encoding="utf-8"
            )
            distributed_config_sha256 = hashlib.sha256(
                json.dumps(
                    canonical_distributed_config,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()

            port_trace = RuntimeFirewallTrace(
                output / "port-precompute" / "firewall-trace.jsonl",
                stage="jacobian-port-precompute:pendulum",
                source_tree_sha256=source_tree_sha256,
            )
            port_trace.record_file_read(
                learner_root / "learner-source-manifest.json",
                role="learner_source_manifest",
                serialized_keys=tuple(sorted(learner_manifest)),
                semantic_sha256=learner_manifest["treeSha256"],
            )
            port_trace.record_recursive_manifest(
                learner_root, role="learner_source_bundle"
            )
            port_trace.record_recursive_manifest(
                learner_cache, role="learner_cache:0"
            )
            port_trace.record_mount_manifest(
                sanitized_root / "trainer-mount", role="trainer_mount_root"
            )
            port_trace.record_file_read(
                sanitized_root
                / "trainer-mount"
                / "pendulum"
                / "fit-pixels.pt",
                role="trainer_archive:fit",
                serialized_keys=("manifest", "pixels"),
            )
            port_trace.record_backbone_boundary(
                phase="jacobian-port-precompute:pendulum",
                boundary="selected_checkpoint",
                sha256=backbone_hash,
            )
            empirical_tangent, port_summary = build_empirical_tangent_from_pixels(
                backbone,
                make_optimization_suite(pixels, model_config, transitions=8),
                system="pendulum",
                fit_sanitized_tensor_sha256=fit_manifest.sanitized_tensor_sha256,
                output_dir=output / "port-precompute",
                device=torch.device("cpu"),
                config=port_config,
                runtime_trace=port_trace,
                source_tree_sha256=source_tree_sha256,
            )
            port_trace.close()
            runtime_entries.append(
                {
                    "phase": "jacobian-port-precompute:pendulum",
                    "relativePath": "port-precompute/firewall-trace.jsonl",
                    "seal": port_summary["runtimeTrace"],
                }
            )
            artifact_path = output / "port-precompute" / "empirical-tangent.pt"
            artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            (output / "port-precompute-complete.json").write_text(
                json.dumps(
                    {
                        "kind": "direct_empirical_jacobian_port_precompute_complete",
                        "system": "pendulum",
                        "configSha256": distributed_config_sha256,
                        "backboneHash": backbone_hash,
                        "fitSanitizedTensorSha256": (
                            fit_manifest.sanitized_tensor_sha256
                        ),
                        "artifactSha256": artifact_sha256,
                        "sourceTreeSha256": source_tree_sha256,
                        "summary": port_summary,
                    }
                ),
                encoding="utf-8",
            )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(experiment_seed + 71)
                source_probes = PixelChangeProbeBank.from_pixel_frames(
                    make_optimization_suite(
                        pixels, model_config, transitions=8
                    )["frames"],
                    palette_size=9,
                    probe_size=1,
                )
            registered_probes = PixelChangeProbeBank(source_probes.basis.clone())
            data_seal = {
                "system": "pendulum",
                "fitAggregateSha256": fit_manifest.aggregate_sha256,
                "fitSanitizedTensorSha256": fit_manifest.sanitized_tensor_sha256,
                "validationAggregateSha256": validation_manifest.aggregate_sha256,
                "validationSanitizedTensorSha256": validation_manifest.sanitized_tensor_sha256,
            }
            variant_summaries = {}
            for variant in REQUIRED_POSTFREEZE_VARIANTS:
                directory = output / "direct" / variant
                directory.mkdir(parents=True)
                probe_copy = PixelChangeProbeBank(registered_probes.basis.clone())
                probe_copy.load_state_dict(registered_probes.state_dict())
                bundle = build_direct_bundle(
                    backbone,
                    DIRECT_SYSTEMS["pendulum"],
                    probe_copy,
                    direct_config,
                    torch.device("cpu"),
                    empirical_tangent=empirical_tangent,
                    variant=variant,
                )
                parameter_names = list(
                    _expected_nonbackbone_optimizer_parameter_names(bundle)
                )
                payload = {
                    key: None for key in _DIRECT_CHECKPOINT_KEYS
                }
                payload.update(
                    {
                        "kind": "direct_jacobian_poisson_port_hamiltonian",
                        "actionChannels": 0,
                        "physicalStateChannels": 0,
                        "optimizationTensorKeys": ["pixelContexts", "frames"],
                        "system": asdict(DIRECT_SYSTEMS["pendulum"]),
                        "variant": variant,
                        "step": 1,
                        "bestValidation": 0.5,
                        "bestStructureEligible": True,
                        "model": dict(bundle.model.state_dict()),
                        "writeField": dict(bundle.write_field.state_dict()),
                        "responseFrame": dict(bundle.response_frame.state_dict()),
                        "cotangentFrame": dict(bundle.cotangent_frame.state_dict()),
                        "probes": dict(bundle.probes.state_dict()),
                        "probeHash": module_tensor_hash(bundle.probes),
                        "dataSeal": data_seal,
                        "optimizedParameterNames": parameter_names,
                        "trainConfig": asdict(direct_config),
                        "lossConfig": asdict(
                            replace(
                                loss_config,
                                jacobian_bridge_weight=0.0,
                                oddness_weight=0.0,
                                manifold_cycle_weight=0.0,
                            )
                            if variant == "no_jacobian"
                            else replace(
                                loss_config,
                                chart_conditioning_weight=0.0,
                            )
                            if variant == "skew_only"
                            else loss_config
                        ),
                        "backboneHash": backbone_hash,
                        "sourceTreeSha256": source_tree_sha256,
                    }
                )
                torch.save(payload, directory / "best.pt")
                variant_runtime_seal = write_runtime_trace(
                    f"direct:{variant}",
                    f"direct/{variant}/firewall-trace.jsonl",
                    ("pixelContexts", "frames"),
                )
                variant_summaries[variant] = {
                    "system": "pendulum",
                    "variant": variant,
                    "bestStep": 1,
                    "bestValidation": 0.5,
                    "bestStructureEligible": True,
                    "backboneHashBefore": backbone_hash,
                    "backboneHashAfter": backbone_hash,
                    "actionGradientUpdates": 0,
                    "physicalStateGradientUpdates": 0,
                    "trainableParameters": sum(
                        parameter.numel()
                        for _, parameter in _named_optimized_parameters(bundle)
                    ),
                    "sourceTreeSha256": source_tree_sha256,
                    "seconds": 0.0,
                    "runtimeTrace": variant_runtime_seal,
                }

            independent = build_fresh_independent_baseline(
                backbone,
                DIRECT_SYSTEMS["pendulum"],
                registered_probes,
                direct_config,
                torch.device("cpu"),
                empirical_tangent=empirical_tangent,
                reference_initialization_seed=experiment_seed + 10_003,
            )
            fit_suite = make_optimization_suite(
                pixels, model_config, transitions=8
            )
            baseline_runtime_trace = RuntimeFirewallTrace(
                output / "baseline-independent" / "firewall-trace.jsonl",
                stage="baseline:independent_unstructured",
                source_tree_sha256=source_tree_sha256,
            )
            baseline_runtime_trace.record_file_read(
                learner_root / "learner-source-manifest.json",
                role="learner_source_manifest",
                serialized_keys=tuple(sorted(learner_manifest)),
                semantic_sha256=learner_manifest["treeSha256"],
            )
            baseline_runtime_trace.record_recursive_manifest(
                learner_root, role="learner_source_bundle"
            )
            baseline_runtime_trace.record_recursive_manifest(
                learner_cache, role="learner_cache:0"
            )
            baseline_summary = train_independent_unstructured_world_model(
                independent,
                fit_suite,
                make_optimization_suite(pixels.clone(), model_config, transitions=8),
                class_weights(fit_suite["frames"], 9, torch.device("cpu")),
                DIRECT_SYSTEMS["pendulum"],
                output / "baseline-independent",
                direct_config,
                loss_config,
                data_seal=data_seal,
                pixel_archive_paths={
                    split: sanitized_root
                    / "trainer-mount"
                    / "pendulum"
                    / f"{split}-pixels.pt"
                    for split in ("fit", "validation")
                },
                source_tree_sha256=source_tree_sha256,
                runtime_trace=baseline_runtime_trace,
            )
            baseline_runtime_trace.close()
            runtime_entries.append(
                {
                    "phase": "baseline:independent_unstructured",
                    "relativePath": "baseline-independent/firewall-trace.jsonl",
                    "seal": baseline_summary["runtimeTrace"],
                }
            )
            summary = {
                "kind": "direct_jacobian_poisson_ph_training_complete",
                "system": "pendulum",
                "experimentConfig": asdict(experiment_config),
                "backboneConfig": asdict(ActionFreeBackboneTrainConfig(steps=1)),
                "portConfig": asdict(port_config),
                "directConfig": asdict(direct_config),
                "baselineConfig": asdict(baseline_config),
                "lossConfig": asdict(loss_config),
                "manifests": {
                    "fit": asdict(fit_manifest),
                    "validation": asdict(validation_manifest),
                },
                "sourceManifest": source_manifest,
                "sourceTreeSha256": source_tree_sha256,
                "learnerSourceManifest": learner_manifest,
                "learnerSourceTreeSha256": learner_manifest["treeSha256"],
                "heldoutTestArchiveOpenedByTraining": False,
                "backbone": {"runtimeTrace": backbone_runtime_seal},
                "portPrecompute": port_summary,
                "backboneHash": backbone_hash,
                "variants": variant_summaries,
                "baseline": baseline_summary,
                "seconds": 0.0,
                "neuralParametersFrozenForPhysicalEvaluation": True,
                "actionGradientUpdates": 0,
                "physicalStateGradientUpdates": 0,
                "runtimeFirewallTraces": runtime_entries,
                "hiddenExcitationConfig": asdict(excitation_config),
                "hiddenExcitationConfigSha256": hidden_excitation_config_sha256(
                    excitation_config
                ),
            }
            (output / "training-complete.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )

            loaded = load_postfreeze_system(
                "pendulum",
                PostFreezePaths(sanitized_root, output),
                torch.device("cpu"),
            )
            self.assertEqual(set(loaded.variants), set(REQUIRED_POSTFREEZE_VARIANTS))
            loaded.assert_frozen_and_unchanged()
            gate1 = audit_gate1_postfreeze(loaded)
            self.assertTrue(gate1.auditable, gate1.failures)
            self.assertTrue(gate1.passed, gate1.to_dict())

            summary_path = output / "training-complete.json"
            sealed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            tampered_summary = json.loads(json.dumps(sealed_summary))
            tampered_summary["variants"]["constant_port"]["trainableParameters"] += 1
            summary_path.write_text(json.dumps(tampered_summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trainable-parameter count"):
                load_postfreeze_system(
                    "pendulum",
                    PostFreezePaths(sanitized_root, output),
                    torch.device("cpu"),
                )
            tampered_summary = json.loads(json.dumps(sealed_summary))
            tampered_summary["baseline"]["relativeParameterGap"] = 0.0
            summary_path.write_text(json.dumps(tampered_summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "baseline summary|lineage"):
                load_postfreeze_system(
                    "pendulum",
                    PostFreezePaths(sanitized_root, output),
                    torch.device("cpu"),
                )
            tampered_summary = json.loads(json.dumps(sealed_summary))
            tampered_summary["sourceTreeSha256"] = "0" * 64
            summary_path.write_text(json.dumps(tampered_summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source-tree SHA-256"):
                load_postfreeze_system(
                    "pendulum",
                    PostFreezePaths(sanitized_root, output),
                    torch.device("cpu"),
                )
            summary_path.write_text(json.dumps(sealed_summary), encoding="utf-8")

            completion_path = output / "port-precompute-complete.json"
            sealed_completion = json.loads(
                completion_path.read_text(encoding="utf-8")
            )
            tampered_completion = json.loads(json.dumps(sealed_completion))
            tampered_completion["summary"]["contexts"] += 1
            completion_path.write_text(
                json.dumps(tampered_completion), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "completion lineage"):
                load_postfreeze_system(
                    "pendulum",
                    PostFreezePaths(sanitized_root, output),
                    torch.device("cpu"),
                )
            completion_path.write_text(
                json.dumps(sealed_completion), encoding="utf-8"
            )

            artifact_path = output / "port-precompute" / "empirical-tangent.pt"
            sealed_artifact_bytes = artifact_path.read_bytes()
            tampered_artifact = torch.load(artifact_path, weights_only=True)
            tampered_artifact["config"] = dict(tampered_artifact["config"])
            tampered_artifact["config"]["contexts"] += 1
            torch.save(tampered_artifact, artifact_path)
            tampered_completion = json.loads(json.dumps(sealed_completion))
            tampered_completion["artifactSha256"] = hashlib.sha256(
                artifact_path.read_bytes()
            ).hexdigest()
            completion_path.write_text(
                json.dumps(tampered_completion), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "config mismatch"):
                load_postfreeze_system(
                    "pendulum",
                    PostFreezePaths(sanitized_root, output),
                    torch.device("cpu"),
                )
            artifact_path.write_bytes(sealed_artifact_bytes)
            completion_path.write_text(
                json.dumps(sealed_completion), encoding="utf-8"
            )

            producer_seal_path = (
                sanitized_root / "seals" / "pendulum" / "manifest.json"
            )
            producer_seal = json.loads(producer_seal_path.read_text(encoding="utf-8"))
            producer_seal["generationEnvironmentSha256"] = "0" * 64
            producer_seal_path.write_text(json.dumps(producer_seal), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "producer boundary seal"):
                load_postfreeze_system(
                    "pendulum",
                    PostFreezePaths(sanitized_root, output),
                    torch.device("cpu"),
                )
            producer_seal["generationEnvironmentSha256"] = (
                action_free_environment_config_sha256("pendulum", image_size=8)
            )
            producer_seal_path.write_text(json.dumps(producer_seal), encoding="utf-8")

            full_path = output / "direct" / "full" / "best.pt"
            full_payload = torch.load(full_path, weights_only=True)
            names = full_payload["optimizedParameterNames"]
            names[0], names[1] = names[1], names[0]
            torch.save(full_payload, full_path)
            with self.assertRaisesRegex(ValueError, "parameter-name seal"):
                load_postfreeze_system(
                    "pendulum",
                    PostFreezePaths(sanitized_root, output),
                    torch.device("cpu"),
                )
            names[0], names[1] = names[1], names[0]
            torch.save(full_payload, full_path)

            archive = torch.load(
                sanitized_root / "heldout" / "pendulum" / "test-pixels.pt",
                weights_only=True,
            )
            archive["pixels"][0, 0, 0, 0] ^= 1
            torch.save(
                archive,
                sanitized_root / "heldout" / "pendulum" / "test-pixels.pt",
            )
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_postfreeze_system(
                    "pendulum",
                    PostFreezePaths(sanitized_root, output),
                    torch.device("cpu"),
                )


class DetachedGate4CollectionTests(unittest.TestCase):
    @staticmethod
    def _tiny_bundle():
        torch.manual_seed(71)
        backbone = DirectPixelTransformer(
            PixelDirectConfig(
                image_size=8,
                patch_size=4,
                palette_size=9,
                history_frames=2,
                pixel_embedding_size=3,
                hidden_size=8,
                depth=2,
                heads=2,
                mlp_ratio=2.0,
            )
        ).eval().requires_grad_(False)
        basis = torch.randn(1, 9, 8, 8)
        tangent_config = EmpiricalTangentConfig(
            channel_rank=2,
            neighbors=2,
            support_floor_ratio=0.02,
        )
        source_activation = torch.randn(4, 2, 4, 8)
        predicted_successor = source_activation + 0.05 * torch.randn_like(
            source_activation
        )
        observed_successor = predicted_successor + 0.05 * torch.randn_like(
            source_activation
        )
        empirical_tangent = build_empirical_tangent_artifact(
            source_activation,
            observed_successor,
            predicted_successor,
            tangent_config,
        )
        bundle = build_direct_bundle(
            backbone,
            DIRECT_SYSTEMS["pendulum"],
            PixelChangeProbeBank(basis),
            DirectTrainingConfig(
                steps=1,
                state_hidden_size=8,
                renderer_hidden_size=8,
                renderer_depth=1,
                renderer_heads=2,
                ph_hidden_size=8,
                ph_hidden_layers=1,
                coupling_layers=2,
                implicit_iterations=4,
                write_hidden_size=4,
                write_hidden_layers=1,
                port_tangent_channel_rank=tangent_config.channel_rank,
                port_tangent_neighbors=tangent_config.neighbors,
                port_support_floor_ratio=tangent_config.support_floor_ratio,
            ),
            torch.device("cpu"),
            empirical_tangent=empirical_tangent,
            variant="full",
        )
        _freeze_bundle(bundle)
        return bundle

    def test_batch_collects_latent_state_responses_and_detaches_every_tensor(self) -> None:
        bundle = self._tiny_bundle()
        contexts = torch.randint(0, 9, (1, 2, 8, 8))
        generator = torch.Generator().manual_seed(91)
        with patch.object(
            bundle.model.renderer,
            "forward",
            side_effect=AssertionError("renderer must not enter Gate-4 retention"),
        ):
            evidence = _gate4_batch(bundle, contexts, Gate4CollectionConfig(), generator)
        self.assertIsInstance(evidence, LensAuditEvidence)
        self.assertEqual(tuple(sorted(evidence.lens_responses or {})), (1, 2, 4))
        for horizon in (1, 2, 4):
            lens = (evidence.lens_responses or {})[horizon]
            ph = (evidence.ph_responses or {})[horizon]
            self.assertEqual(lens.shape, (1, 2, 1))
            self.assertEqual(ph.shape, lens.shape)
            self.assertFalse(lens.requires_grad)
            self.assertIsNone(lens.grad_fn)
            self.assertFalse(ph.requires_grad)
            self.assertIsNone(ph.grad_fn)
        self.assertEqual(evidence.random_write_effect_norms.shape, (1, 1, 16))
        self.assertEqual(evidence.adjoint_jvp_inner_products.shape, (1, 3))
        self.assertEqual(evidence.adjoint_vjp_inner_products.shape, (1, 3))
        self.assertEqual(evidence.adjoint_jvp_norm_bounds.shape, (1, 3))
        self.assertEqual(evidence.adjoint_vjp_norm_bounds.shape, (1, 3))
        torch.testing.assert_close(
            evidence.adjoint_jvp_inner_products,
            evidence.adjoint_vjp_inner_products,
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            evidence.explicit_state_jacobian_products,
            evidence.independent_state_jvp_products,
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            evidence.extracted_port_gram_matrices,
            torch.ones(1, 1, 1),
            atol=2e-5,
            rtol=2e-5,
        )
        self.assertGreater(float(evidence.extracted_port_singular_values.min()), 0.0)
        self.assertEqual(
            evidence.extracted_port_reported_orthonormality_defects.shape,
            (1,),
        )
        self.assertEqual(evidence.extracted_projected_signal_ratios.shape, (1,))
        self.assertEqual(evidence.extracted_neighbor_indices.shape, (1, 2))
        self.assertEqual(evidence.extracted_neighbor_fit_population, 4)
        self.assertEqual(len(evidence.path_fingerprint_sha256), 64)
        self.assertEqual(evidence.path_code_sha256, evidence.sealed_path_code_sha256)
        self.assertEqual(
            evidence.path_backbone_sha256, evidence.sealed_backbone_sha256
        )
        self.assertEqual(
            evidence.path_extractor_sha256, evidence.sealed_extractor_sha256
        )
        self.assertEqual(
            evidence.path_source_tree_sha256, evidence.sealed_source_tree_sha256
        )
        self.assertTrue(evidence.random_writes_norm_matched)
        self.assertIn("frozen_transformer_suffix", evidence.retention_path_kind)
        for value in (
            evidence.positive_effects,
            evidence.negative_effects,
            evidence.baseline_effects,
            evidence.random_write_effect_norms,
            evidence.adjoint_jvp_inner_products,
            evidence.adjoint_vjp_inner_products,
            evidence.adjoint_jvp_norm_bounds,
            evidence.adjoint_vjp_norm_bounds,
            evidence.explicit_state_jacobian_products,
            evidence.independent_state_jvp_products,
            evidence.extracted_port_gram_matrices,
            evidence.extracted_port_singular_values,
            evidence.extracted_port_reported_orthonormality_defects,
            evidence.extracted_projected_signal_ratios,
            evidence.extracted_neighbor_indices,
        ):
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)
        self.assertEqual(
            module_tensor_hash(bundle.model.encoder.backbone),
            bundle.model.encoder.sealed_backbone_hash,
        )

    def test_path_provenance_falsifies_mutated_empirical_extractor_buffer(self) -> None:
        bundle = self._tiny_bundle()
        sealed = module_tensor_hash(bundle.write_field)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "direct.pt"
            torch.save(
                {"writeField": dict(bundle.write_field.state_dict())},
                checkpoint_path,
            )
            frozen = SimpleNamespace(
                bundle=bundle,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=_file_sha256(checkpoint_path),
            )
            self.assertEqual(_gate4_sealed_extractor_sha256(frozen), sealed)
            original = _gate4_path_provenance(
                bundle,
                sealed_extractor_sha256=sealed,
            )
            with torch.no_grad():
                bundle.write_field.channel_basis[0, 0].add_(0.25)
            self.assertEqual(_gate4_sealed_extractor_sha256(frozen), sealed)
            corrupted = _gate4_path_provenance(
                bundle,
                sealed_extractor_sha256=sealed,
            )
            self.assertEqual(original.path_extractor_sha256, sealed)
            self.assertNotEqual(corrupted.path_extractor_sha256, sealed)
            self.assertNotEqual(
                corrupted.path_fingerprint_sha256,
                original.path_fingerprint_sha256,
            )
            torch.save(
                {"writeField": dict(bundle.write_field.state_dict())},
                checkpoint_path,
            )
            with self.assertRaisesRegex(ValueError, "changed after post-freeze"):
                _gate4_sealed_extractor_sha256(frozen)


class ControlShardingAndOutcomeTests(unittest.TestCase):
    @staticmethod
    def _result(start: int, stop: int, interface: str = "native") -> ControlResult:
        length = stop - start
        errors = {
            name: tuple(float(start + index + offset) for index in range(length))
            for offset, name in enumerate(
                (
                    "structured",
                    "unstructured",
                    "activation",
                    "no_jacobian",
                    "shuffled_lens",
                    "coast",
                    "random",
                )
            )
        }
        return ControlResult(
            errors=errors,
            interface_name=interface,
            episodes=length,
            control_steps=3,
            planner_budget={"candidatesPerDecision": 8, "iterationsPerDecision": 2},
        )

    def test_registered_ranges_and_merge_cover_exactly_64_episodes(self) -> None:
        ranges = registered_control_shard_ranges(7)
        self.assertEqual(ranges[0], (0, 7))
        self.assertEqual(ranges[-1][1], REGISTERED_CONTROL_EPISODES)
        shards = [
            ControlShard(
                "native",
                start,
                stop,
                REGISTERED_CONTROL_EPISODES,
                self._result(start, stop),
            )
            for start, stop in ranges
        ]
        merged = merge_control_shards(shards)
        self.assertEqual(merged.episodes, 64)
        self.assertTrue(all(len(values) == 64 for values in merged.errors.values()))

    def test_merge_rejects_missing_shard(self) -> None:
        ranges = registered_control_shard_ranges(8)
        shards = [
            ControlShard("native", start, stop, 64, self._result(start, stop))
            for start, stop in ranges[:-1]
        ]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            merge_control_shards(shards)

    def test_replayable_merge_preserves_and_requires_one_physical_protocol(self) -> None:
        native = linear_interface_protocol(
            SYSTEMS["pendulum"], fixed_interfaces("pendulum")["native"]
        )

        def traced(start: int, stop: int, protocol=native) -> ControlResult:
            length = stop - start
            names = (
                "structured",
                "unstructured",
                "activation",
                "no_jacobian",
                "shuffled_lens",
                "coast",
                "random",
            )
            return ControlResult(
                errors={name: tuple(0.5 for _ in range(length)) for name in names},
                interface_name="native",
                episodes=length,
                control_steps=3,
                planner_budget={"candidatesPerDecision": 8, "iterationsPerDecision": 2},
                episode_identifiers=tuple(
                    f"pendulum-{index}" for index in range(start, stop)
                ),
                interface_command_traces={
                    name: tuple(
                        tuple((0.0,) for _ in range(3)) for _ in range(length)
                    )
                    for name in names
                },
                planner_seed_schedule_sha256=f"{start // 32 + 1}" * 64,
                physical_protocol=protocol,
            )

        shards = [
            ControlShard("native", start, stop, 64, traced(start, stop))
            for start, stop in ((0, 32), (32, 64))
        ]
        merged = merge_control_shards(shards)
        self.assertEqual(merged.physical_protocol, native)
        self.assertEqual(len(merged.episode_identifiers), 64)
        self.assertTrue(merged.interface_command_traces)

        unseen = linear_interface_protocol(
            SYSTEMS["pendulum"], fixed_interfaces("pendulum")["unseen"]
        )
        mismatched = [
            shards[0],
            ControlShard("native", 32, 64, 64, traced(32, 64, unseen)),
        ]
        with self.assertRaisesRegex(ValueError, "physical protocols differ"):
            merge_control_shards(mismatched)

    def test_final_composition_fails_on_any_missing_gate(self) -> None:
        result = compose_single_seed_outcome(
            {"pendulum": {"gate1": {"passed": True}}}
        )
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["outcome"],
            "direct_jacobian_poisson_ph_breakthrough_not_supported_single_seed",
        )
        missing = result["systems"]["blocket"]["gates"]["gate8"]
        self.assertFalse(missing["auditable"])

    def test_final_composition_requires_both_systems_and_all_eight_gates(self) -> None:
        complete = {
            system: {
                f"gate{gate}": {"passed": True}
                for gate in range(1, 9)
            }
            for system in ("pendulum", "blocket")
        }
        result = compose_single_seed_outcome(complete)
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["outcome"],
            "direct_jacobian_poisson_ph_breakthrough_supported_single_seed_two_systems",
        )


if __name__ == "__main__":
    unittest.main()
