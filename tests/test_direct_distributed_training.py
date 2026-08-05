from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from blocket_league.direct_action_free_data import (
    ActionFreeBackboneTrainConfig,
    PixelsOnlyManifest,
    class_weights,
    make_optimization_suite,
    sanitized_pixel_tensor_sha256,
)
from blocket_league.direct_cotangent_bridge import PixelChangeProbeBank
from blocket_league.direct_distributed_training import (
    _load_registered_probes,
    _seal_config,
    _write_or_validate_probes,
    load_sealed_config,
)
from blocket_league.action_free_excitation import (
    private_producer_seed_from_file as _private_producer_seed,
)
from blocket_league.direct_experiment_training import (
    DIRECT_SYSTEMS,
    DirectTrainingConfig,
)
from blocket_league.direct_jacobian_port_extractor import (
    EmpiricalTangentConfig,
    make_synthetic_empirical_tangent_artifact_for_tests,
)
from blocket_league.direct_jacobian_port_precompute import (
    JacobianPortPrecomputeConfig,
)
from blocket_league.direct_unstructured_training import (
    build_fresh_independent_baseline,
    train_independent_unstructured_world_model,
)
from blocket_league.experiment_f_contract import (
    ExperimentFConfig,
    REGISTERED_VARIANTS,
)
from blocket_league.direct_visual_poisson_ph import DirectVideoLossConfig
from blocket_league.tensor_provenance import module_tensor_hash
from blocket_league.pixel_direct_model import DirectPixelTransformer, PixelDirectConfig
from blocket_league.source_provenance import build_source_manifest


class DistributedTrainingSealTests(unittest.TestCase):
    def test_private_128_bit_producer_seed_is_file_derived_and_permission_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "producer-seed.hex"
            path.write_text("0123456789abcdef0123456789abcdef\n", encoding="ascii")
            path.chmod(0o600)
            pendulum = _private_producer_seed(path, system="pendulum")
            blocket = _private_producer_seed(path, system="blocket")
            self.assertNotEqual(pendulum, blocket)
            self.assertGreaterEqual(pendulum, 0)
            self.assertLess(pendulum, 1 << 63)
            self.assertEqual(
                pendulum, _private_producer_seed(path, system="pendulum")
            )
            path.chmod(0o644)
            with self.assertRaises(PermissionError):
                _private_producer_seed(path, system="pendulum")

    def test_config_and_probe_seals_are_reusable_but_not_mutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pixels = torch.randint(0, 9, (2, 10, 8, 8), dtype=torch.uint8)
            manifests = {
                split: PixelsOnlyManifest(
                    system="pendulum",
                    trajectories=2,
                    frames_per_trajectory=10,
                    image_size=8,
                    aggregate_sha256=character * 64,
                    sanitized_tensor_sha256=sanitized_pixel_tensor_sha256(pixels),
                )
                for split, character in (("fit", "a"), ("validation", "b"))
            }
            experiment = ExperimentFConfig(
                fit_trajectories=2,
                validation_trajectories=2,
                test_trajectories=2,
                history_frames=2,
                transitions=8,
                cache_frames=10,
                image_size=8,
                patch_size=4,
                backbone_preset="nano",
                variants=REGISTERED_VARIANTS,
            )
            backbone = ActionFreeBackboneTrainConfig(steps=1)
            direct = DirectTrainingConfig(
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
            port = JacobianPortPrecomputeConfig(
                contexts=2,
                batch_size=1,
                lens_block=DIRECT_SYSTEMS["pendulum"].lens_block,
                horizons=direct.lens_horizons,
                channel_rank=direct.port_tangent_channel_rank,
                neighbors=direct.port_tangent_neighbors,
                support_floor_ratio=direct.port_support_floor_ratio,
            )
            baseline = direct
            loss = DirectVideoLossConfig()
            sealed = _seal_config(
                root,
                "pendulum",
                experiment,
                backbone,
                port,
                direct,
                baseline,
                loss,
                manifests,
            )
            loaded = load_sealed_config(root)
            self.assertEqual(loaded.sha256, sealed.sha256)
            self.assertEqual(loaded.source_manifest, sealed.source_manifest)
            self.assertEqual(
                loaded.source_tree_sha256,
                build_source_manifest()["treeSha256"],
            )
            model_config = PixelDirectConfig(
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
            _, registered_hash = _write_or_validate_probes(
                root, sealed, pixels, model_config
            )
            self.assertEqual(
                registered_hash,
                module_tensor_hash(_load_registered_probes(root, sealed)),
            )

            path = root / "distributed-config.json"
            canonical = json.loads(path.read_text(encoding="utf-8"))
            tampered = json.loads(json.dumps(canonical))
            tampered["sourceManifest"]["files"][0]["sha256"] = "0" * 64
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source manifest"):
                load_sealed_config(root)
            path.write_text(json.dumps(canonical), encoding="utf-8")
            tampered = json.loads(json.dumps(canonical))
            tampered["directConfig"]["steps"] = 2
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed on resume"):
                _seal_config(
                    root,
                    "pendulum",
                    experiment,
                    backbone,
                    port,
                    direct,
                    baseline,
                    loss,
                    manifests,
                )

    def test_baseline_checkpoint_is_order_independent_and_seed_sealed(self) -> None:
        torch.manual_seed(11)
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
        probes = PixelChangeProbeBank(torch.randn(1, 9, 8, 8))
        config = DirectTrainingConfig(
            steps=1,
            micro_batch_size=2,
            lens_batch_size=2,
            validation_every=1,
            validation_batches=1,
            checkpoint_every=1,
            log_every=1,
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
            lens_horizons=(1,),
            seed=917,
        )
        pixels = torch.randint(0, 9, (32, 4, 8, 8), dtype=torch.uint8)
        suite = make_optimization_suite(pixels, backbone.config, transitions=2)
        data_seal = {
            "system": "pendulum",
            "fitAggregateSha256": "a" * 64,
            "fitSanitizedTensorSha256": "b" * 64,
            "validationAggregateSha256": "c" * 64,
            "validationSanitizedTensorSha256": "d" * 64,
        }
        loss = DirectVideoLossConfig(rollout_horizons=(1, 2))
        empirical_tangent = make_synthetic_empirical_tangent_artifact_for_tests(
            history_frames=backbone.config.history_frames,
            patch_count=backbone.config.grid_size**2,
            hidden_size=backbone.config.hidden_size,
            config=EmpiricalTangentConfig(
                channel_rank=config.port_tangent_channel_rank,
                neighbors=config.port_tangent_neighbors,
                support_floor_ratio=config.port_support_floor_ratio,
            ),
            seed=944,
        )
        module_fields = (
            "encoderPoolScore",
            "encoderReadout",
            "renderer",
            "dynamics",
            "effortInference",
            "writeField",
            "responseFrame",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archives = root / "archives"
            archives.mkdir()
            archive_paths = {}
            for split in ("fit", "validation"):
                path = archives / f"{split}-pixels.pt"
                torch.save({"pixels": pixels, "manifest": {}}, path)
                archive_paths[split] = path

            def run(name: str):
                bundle = build_fresh_independent_baseline(
                    backbone,
                    DIRECT_SYSTEMS["pendulum"],
                    probes,
                    config,
                    torch.device("cpu"),
                    empirical_tangent=empirical_tangent,
                    reference_initialization_seed=77_003,
                )
                train_independent_unstructured_world_model(
                    bundle,
                    suite,
                    suite,
                    class_weights(suite["frames"], 9, torch.device("cpu")),
                    DIRECT_SYSTEMS["pendulum"],
                    root / name,
                    config,
                    loss,
                    data_seal=data_seal,
                    pixel_archive_paths=archive_paths,
                    source_tree_sha256="e" * 64,
                )
                return torch.load(root / name / "best.pt", weights_only=True)

            first = run("monolithic")
            torch.manual_seed(99_999)
            second = run("distributed")
            self.assertEqual(first["trainConfig"], asdict(config))
            self.assertEqual(second["sourceTreeSha256"], "e" * 64)
            self.assertNotIn("encodedPixelStates", second)
            self.assertNotIn("model", second)
            for field in module_fields:
                self.assertEqual(set(first[field]), set(second[field]))
                for name in first[field]:
                    self.assertTrue(torch.equal(first[field][name], second[field][name]))

            # Strict resume reconstructs from the registered seed and paired
            # best/last artifacts; ambient RNG cannot change the selected model.
            resumed = run("monolithic")
            for field in module_fields:
                for name in first[field]:
                    self.assertTrue(torch.equal(first[field][name], resumed[field][name]))

            (root / "monolithic" / "last.pt").unlink()
            with self.assertRaisesRegex(ValueError, "paired best.pt and last.pt"):
                run("monolithic")

class MesoheliosScriptTests(unittest.TestCase):
    scripts = Path(__file__).resolve().parents[1] / "scripts" / "mesohelios"

    def test_shell_syntax_and_strict_training_mounts(self) -> None:
        names = (
            "experiment-f-producer.sbatch",
            "experiment-f-backbone.sbatch",
            "experiment-f-port.sbatch",
            "experiment-f-variant.sbatch",
            "experiment-f-baseline.sbatch",
            "experiment-f-finalize.sbatch",
            "experiment-f-postfreeze-parent.sbatch",
            "experiment-f-postfreeze-prepare.sbatch",
            "experiment-f-control-shard.sbatch",
            "experiment-f-postfreeze-finalize.sbatch",
            "experiment-f-control-performance-probe.sbatch",
            "submit-experiment-f.sh",
        )
        subprocess.run(
            ["bash", "-n", *(str(self.scripts / name) for name in names)],
            check=True,
        )
        for name in (
            "experiment-f-backbone.sbatch",
            "experiment-f-port.sbatch",
            "experiment-f-variant.sbatch",
            "experiment-f-baseline.sbatch",
            "experiment-f-finalize.sbatch",
        ):
            text = (self.scripts / name).read_text(encoding="utf-8")
            self.assertIn("--containall", text)
            self.assertIn("--no-mount home,cwd,hostfs", text)
            bind_lines = "\n".join(
                line for line in text.splitlines() if "--bind" in line
            )
            self.assertIn("$trainer_system:/trainer-mount/$system:ro", bind_lines)
            self.assertNotIn("heldout", bind_lines)
            self.assertNotIn("$sanitized_root", bind_lines)
            self.assertNotIn("producer-private", bind_lines)
            self.assertIn("$learner_root:/workspace:ro", bind_lines)
            self.assertNotIn("$repo_root:/workspace:ro", bind_lines)
            self.assertIn("PYTHONSAFEPATH=1", text)
            self.assertIn("PYTHONDONTWRITEBYTECODE=1", text)
            self.assertIn("--learner-bundle-root /workspace", text)
            self.assertIn("--learner-cache /learner-cache", text)
        backbone_text = (
            self.scripts / "experiment-f-backbone.sbatch"
        ).read_text(encoding="utf-8")
        launcher_text = (
            self.scripts / "submit-experiment-f.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--source-manifest /workspace/source-manifest.json", backbone_text)
        self.assertIn("source_provenance.py", launcher_text)
        self.assertIn("blocket_league.learner_source_bundle build", launcher_text)
        self.assertIn("status --porcelain=v1", launcher_text)
        producer_text = (
            self.scripts / "experiment-f-producer.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("$producer_private:/producer-private:ro", producer_text)
        self.assertIn(
            "python -m blocket_league.direct_experiment_f_producer", producer_text
        )
        self.assertNotIn(
            "direct_distributed_training producer", producer_text
        )
        self.assertIn("--producer-seed-file /producer-private/producer-seed.hex", producer_text)
        self.assertNotIn("F_PRODUCER_SEED_BASE", producer_text)
        self.assertNotIn("--producer-seed \"", producer_text)
        self.assertIn("secrets.token_hex(16)", launcher_text)

        # The admissible scientific run is immutable at the launcher surface.
        # Shape-exact performance probes are deliberately separate and may be
        # tuned, but producer/backbone jobs must pass registered literals.
        scientific_variables = (
            "F_FIT_TRAJECTORIES",
            "F_VALIDATION_TRAJECTORIES",
            "F_TEST_TRAJECTORIES",
            "F_TRANSITIONS",
            "F_CACHE_FRAMES",
            "F_IMAGE_SIZE",
            "F_PATCH_SIZE",
            "F_BACKBONE_PRESET",
            "F_BACKBONE_STEPS",
            "F_DIRECT_STEPS",
            "F_BASELINE_STEPS",
            "F_MICRO_BATCH_SIZE",
            "F_LENS_BATCH_SIZE",
            "F_IMPLICIT_ITERATIONS",
        )
        for variable in scientific_variables:
            self.assertNotIn(f"${{{variable}:-", producer_text)
            self.assertNotIn(f"${{{variable}:-", backbone_text)
            self.assertIn(variable, launcher_text)

        overridden = subprocess.run(
            ["bash", str(self.scripts / "submit-experiment-f.sh"), "--dry-run"],
            check=False,
            text=True,
            capture_output=True,
            env={
                **os.environ,
                "F_RUN_ROOT": "/tmp/experiment-f-static-dry-run",
                "F_DIRECT_STEPS": "1",
            },
        )
        self.assertEqual(overridden.returncode, 2)
        self.assertIn("Refusing scientific override F_DIRECT_STEPS", overridden.stderr)

    def test_variant_array_maps_once_to_each_system_variant(self) -> None:
        observed = set()
        script = self.scripts / "experiment-f-variant.sbatch"
        for task_id in range(12):
            environment = dict(os.environ)
            environment.update(
                {
                    "SLURM_ARRAY_TASK_ID": str(task_id),
                    "F_RUN_ROOT": "/tmp/experiment-f-static-dry-run",
                    "F_LEARNER_BUNDLE_ROOT": "/tmp/experiment-f-static-learner",
                    "F_LEARNER_SOURCE_TREE_SHA256": "a" * 64,
                }
            )
            completed = subprocess.run(
                ["bash", str(script), "--dry-run"],
                check=True,
                text=True,
                capture_output=True,
                env=environment,
            )
            prefix = completed.stdout.split(" command=", 1)[0]
            fields = dict(item.split("=", 1) for item in prefix.split() if "=" in item)
            observed.add((fields["system"], fields["variant"]))
        self.assertEqual(
            observed,
            {
                (system, variant)
                for system in ("pendulum", "blocket")
                for variant in REGISTERED_VARIANTS
            },
        )

    def test_dependency_dag_and_postfreeze_are_fail_closed(self) -> None:
        launcher = self.scripts / "submit-experiment-f.sh"
        completed = subprocess.run(
            ["bash", str(launcher), "--dry-run", "--include-postfreeze"],
            check=True,
            text=True,
            capture_output=True,
            env={**os.environ, "F_RUN_ROOT": "/tmp/experiment-f-static-dry-run"},
        )
        self.assertEqual(completed.stdout.count("sbatch --parsable"), 9)
        self.assertGreaterEqual(completed.stdout.count("--dependency=afterok:"), 8)
        self.assertIn("experiment-f-postfreeze-prepare.sbatch", completed.stdout)
        self.assertIn("experiment-f-control-shard.sbatch", completed.stdout)
        self.assertIn("experiment-f-postfreeze-finalize.sbatch", completed.stdout)
        scripts = "\n".join(
            (self.scripts / name).read_text(encoding="utf-8")
            for name in (
                "experiment-f-postfreeze-prepare.sbatch",
                "experiment-f-control-shard.sbatch",
                "experiment-f-postfreeze-finalize.sbatch",
            )
        )
        self.assertIn("direct_postfreeze_complete prepare-system", scripts)
        self.assertIn("direct_postfreeze_complete control-shard", scripts)
        self.assertIn("direct_postfreeze_complete finalize", scripts)
        self.assertNotIn("python -m blocket_league.direct_postfreeze_runner", scripts)
        self.assertIn("--require-gates 1,2,3,4,5,6,7,8,9", scripts)
        baseline = (self.scripts / "experiment-f-baseline.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("direct/full/best.pt", baseline)
        self.assertIn("backbone-complete.json", baseline)
        for name, directive in (
            ("experiment-f-backbone.sbatch", "#SBATCH --array=0-1%1"),
            ("experiment-f-port.sbatch", "#SBATCH --array=0-1%1"),
            ("experiment-f-variant.sbatch", "#SBATCH --array=0-11%1"),
            ("experiment-f-baseline.sbatch", "#SBATCH --array=0-1%1"),
            ("experiment-f-postfreeze-prepare.sbatch", "#SBATCH --array=0-1%1"),
            ("experiment-f-control-shard.sbatch", "#SBATCH --array=0-63%1"),
        ):
            self.assertIn(
                directive, (self.scripts / name).read_text(encoding="utf-8")
            )

    def test_control_array_maps_exactly_once_to_all_registered_shards(self) -> None:
        script = self.scripts / "experiment-f-control-shard.sbatch"
        observed = set()
        for task_id in range(64):
            completed = subprocess.run(
                ["bash", str(script), "--dry-run", "--task-id", str(task_id)],
                check=True,
                text=True,
                capture_output=True,
                env={**os.environ, "F_RUN_ROOT": "/tmp/experiment-f-static-dry-run"},
            )
            prefix = completed.stdout.split(" command=", 1)[0]
            fields = dict(item.split("=", 1) for item in prefix.split() if "=" in item)
            observed.add(
                (
                    fields["system"],
                    fields["interface"],
                    int(fields["start"]),
                    int(fields["stop"]),
                )
            )
        self.assertEqual(
            observed,
            {
                (system, interface, start, start + 4)
                for system in ("pendulum", "blocket")
                for interface in ("native", "unseen")
                for start in range(0, 64, 4)
            },
        )


if __name__ == "__main__":
    unittest.main()
