from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from blocket_league.learner_source_bundle import (
    LEARNER_MODULE_ALLOWLIST,
    build_learner_source_bundle,
    learner_import_closure,
    validate_code_free_cache,
    verify_learner_source_bundle,
)
from blocket_league.source_provenance import build_source_manifest


ROOT = Path(__file__).resolve().parents[1]


class LearnerSourceBundleTests(unittest.TestCase):
    def _bundle(self, temporary: str) -> tuple[Path, dict[str, object]]:
        destination = Path(temporary) / "learner"
        manifest = build_learner_source_bundle(
            ROOT, build_source_manifest(ROOT), destination
        )
        return destination, manifest

    def test_exact_bundle_imports_training_without_simulator_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle, manifest = self._bundle(temporary)
            observed = verify_learner_source_bundle(
                bundle,
                expected_full_source_tree_sha256=manifest["fullSourceTreeSha256"],
            )
            self.assertEqual(observed, manifest)
            program = """
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import blocket_league.direct_distributed_training
forbidden = sorted(name for name in sys.modules if name in {
    'blocket_league.env',
    'blocket_league.action_free_excitation',
    'blocket_league.passive_control_systems',
    'blocket_league.direct_experiment_f_producer',
    'blocket_league.passive_jacobian_ph_model',
})
print(json.dumps(forbidden))
"""
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(bundle)
            environment["PYTHONSAFEPATH"] = "1"
            result = subprocess.run(
                [sys.executable, "-I", "-c", program],
                cwd=bundle,
                env=environment,
                check=True,
                text=True,
                capture_output=True,
            )
            # Isolated mode ignores PYTHONPATH, so execute from the bundle and
            # insert its exact root explicitly, never the development tree.
            if result.stdout.strip() != "[]":  # pragma: no cover - diagnostic
                self.fail(result.stdout)

    def test_ast_closure_is_the_reviewed_allowlist(self) -> None:
        self.assertEqual(
            learner_import_closure(ROOT), LEARNER_MODULE_ALLOWLIST
        )

    def test_extra_source_file_and_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle, _ = self._bundle(temporary)
            package = bundle / "blocket_league"
            package.chmod(0o755)
            injected_directory = package / "__pycache__"
            injected_directory.mkdir()
            with self.assertRaisesRegex(ValueError, "directory contents are not exact"):
                verify_learner_source_bundle(bundle)
            injected_directory.rmdir()
            (package / "env.py").write_text("LEAK = True\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contents are not exact"):
                verify_learner_source_bundle(bundle)
            (package / "env.py").unlink()
            target = package / "pixel_palette.py"
            target.chmod(0o644)
            target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_learner_source_bundle(bundle)

    def test_symbolic_bundle_entry_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle, _ = self._bundle(temporary)
            package = bundle / "blocket_league"
            package.chmod(0o755)
            target = package / "pixel_palette.py"
            target.chmod(0o644)
            target.unlink()
            target.symlink_to(package / "tensor_provenance.py")
            with self.assertRaisesRegex(ValueError, "symbolic"):
                verify_learner_source_bundle(bundle)

    def test_code_free_cache_rejects_python_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            cache.mkdir()
            (cache / "kernel.bin").write_bytes(b"compiled")
            self.assertEqual(validate_code_free_cache(cache)["pythonCodeFiles"], [])
            (cache / "inject.pth").write_text("/producer\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not code-free"):
                validate_code_free_cache(cache)
            (cache / "inject.pth").unlink()
            (cache / "link").symlink_to(cache / "kernel.bin")
            with self.assertRaisesRegex(ValueError, "not code-free"):
                validate_code_free_cache(cache)

    def test_forbidden_import_is_detected_by_ast_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "blocket_league"
            package.mkdir()
            for module in LEARNER_MODULE_ALLOWLIST:
                source = ROOT / "blocket_league" / f"{module}.py"
                destination = package / f"{module}.py"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            shutil.copyfile(ROOT / "blocket_league" / "env.py", package / "env.py")
            entry = package / "direct_distributed_training.py"
            entry.write_text(
                entry.read_text(encoding="utf-8") + "\nfrom .env import PALETTE\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "forbidden learner import"):
                learner_import_closure(root)

    def test_slurm_bootstrap_verifies_before_import(self) -> None:
        scripts = ROOT / "scripts" / "mesohelios"
        codes = []
        for name in (
            "experiment-f-backbone.sbatch",
            "experiment-f-port.sbatch",
            "experiment-f-variant.sbatch",
            "experiment-f-baseline.sbatch",
            "experiment-f-finalize.sbatch",
        ):
            text = (scripts / name).read_text(encoding="utf-8")
            match = re.search(
                r"bootstrap_code='(.*?)'\n(?:[ \t]*\n)*command=", text, re.S
            )
            self.assertIsNotNone(match, name)
            codes.append(match.group(1))
        self.assertTrue(all(code == codes[0] for code in codes[1:]))
        with tempfile.TemporaryDirectory() as temporary:
            bundle, manifest = self._bundle(temporary)
            command = [
                sys.executable,
                "-I",
                "-c",
                codes[0],
                manifest["treeSha256"],
                str(bundle),
                "blocket_league.direct_distributed_training",
                "--help",
            ]
            accepted = subprocess.run(
                command, text=True, capture_output=True, check=False
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            target = bundle / "blocket_league" / "pixel_palette.py"
            target.chmod(0o644)
            target.write_text(
                target.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            rejected = subprocess.run(
                command, text=True, capture_output=True, check=False
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("learner source byte mismatch", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
