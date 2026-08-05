from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from blocket_league.source_provenance import (
    build_source_manifest,
    validate_source_manifest_schema,
    verify_source_manifest,
    write_or_verify_source_manifest,
)


class SourceProvenanceTests(unittest.TestCase):
    def _tree(self, root: Path) -> None:
        contents = {
            "blocket_league/a.py": "VALUE = 1\n",
            "scripts/mesohelios/run.sh": "#!/bin/sh\nexit 0\n",
            "tests/test_a.py": "def test_a(): pass\n",
            "docs/direct_jacobian_poisson_ph_experiment.md": "sealed protocol\n",
            "pyproject.toml": "[project]\nname='sealed'\n",
            "uv.lock": "version = 1\n",
            # These mutable/runtime locations are intentionally outside the
            # registered source selection.
            ".git/HEAD": "mutable\n",
            "results/output.json": "{}\n",
        }
        for relative, text in contents.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    def test_manifest_recomputes_exact_tree_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._tree(root)
            manifest = build_source_manifest(root)
            self.assertEqual(validate_source_manifest_schema(manifest), manifest["treeSha256"])
            self.assertEqual(verify_source_manifest(manifest, root), manifest["treeSha256"])
            self.assertNotIn(".git/HEAD", {item["path"] for item in manifest["files"]})
            self.assertNotIn("results/output.json", {item["path"] for item in manifest["files"]})

            (root / "blocket_league/a.py").write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from sealed manifest"):
                verify_source_manifest(manifest, root)

            tampered = json.loads(json.dumps(manifest))
            tampered["files"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "canonical tree"):
                validate_source_manifest_schema(tampered)

    def test_launch_manifest_is_immutable_once_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._tree(root)
            path = root / "launch-source.json"
            first = write_or_verify_source_manifest(path, root)
            second = write_or_verify_source_manifest(path, root)
            self.assertEqual(first, second)
            (root / "tests/test_a.py").write_text("def test_b(): pass\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "launch source manifest differs"):
                write_or_verify_source_manifest(path, root)


if __name__ == "__main__":
    unittest.main()
