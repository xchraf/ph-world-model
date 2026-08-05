from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from blocket_league.direct_postfreeze_complete import (
    _atomic_json,
    _final_artifact,
    _final_core,
    _verify_final_artifact_envelope,
    parse_args,
)


class CompletePostFreezeCLITests(unittest.TestCase):
    @staticmethod
    def _system(passed: bool) -> dict:
        return {
            "system": "unused",
            "trainingLineageSha256": "a" * 64,
            "physicalSha256": "b" * 64,
            "gates": {
                f"gate{gate}": {"passed": passed} for gate in range(1, 9)
            },
            "controlEvidence": {"episodeIdentifiers": ("one", "two")},
        }

    def test_gate9_is_derived_only_from_both_complete_system_conjunctions(self) -> None:
        complete = _final_core(
            {system: self._system(True) for system in ("pendulum", "blocket")}
        )
        self.assertTrue(complete["gate9"]["passed"])
        self.assertEqual(
            complete["outcome"],
            "direct_jacobian_poisson_ph_breakthrough_supported_single_seed_two_systems",
        )
        failed = _final_core(
            {"pendulum": self._system(True), "blocket": self._system(False)}
        )
        self.assertFalse(failed["gate9"]["passed"])
        self.assertFalse(failed["passed"])
        self.assertEqual(
            failed["outcome"],
            "direct_jacobian_poisson_ph_breakthrough_not_supported_single_seed",
        )

    def test_cli_refuses_any_partial_gate_list(self) -> None:
        parsed = parse_args(
            [
                "run",
                "/sanitized",
                "/training",
                "/results",
                "--require-gates",
                "1,2,3,4,5,6,7,8,9",
            ]
        )
        self.assertEqual(parsed.require_gates, tuple(range(1, 10)))
        self.assertEqual(parsed.device, "cuda")
        verified = parse_args(
            [
                "verify",
                "/results/final-outcome.json",
                "/sanitized",
                "/training",
                "--device",
                "cuda",
                "--require-gates",
                "1,2,3,4,5,6,7,8,9",
            ]
        )
        self.assertEqual(verified.device, "cuda")
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "verify",
                    "/results/final-outcome.json",
                    "/sanitized",
                    "/training",
                    "--require-gates",
                    "1,2,3,4,5,6,7,8",
                ]
            )

    def test_final_json_hash_survives_write_read_and_rejects_tamper(self) -> None:
        systems = {
            system: self._system(True) for system in ("pendulum", "blocket")
        }
        artifact = _final_artifact(systems)
        # The pre-hash normalization must match what JSON actually persists.
        self.assertIsInstance(
            artifact["systems"]["pendulum"]["controlEvidence"][
                "episodeIdentifiers"
            ],
            list,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final-outcome.json"
            _atomic_json(path, artifact)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            verified = _verify_final_artifact_envelope(loaded)
            self.assertEqual(verified["outcome"], artifact["outcome"])

            loaded["systems"]["pendulum"]["controlEvidence"][
                "episodeIdentifiers"
            ][0] = "tampered"
            with self.assertRaisesRegex(ValueError, "digest/provenance"):
                _verify_final_artifact_envelope(loaded)

    def test_distributed_stage_cli_keeps_one_exact_four_episode_shard(self) -> None:
        prepared = parse_args(
            [
                "prepare-system",
                "/sanitized",
                "/training",
                "/results",
                "--system",
                "pendulum",
            ]
        )
        self.assertEqual(prepared.system, "pendulum")
        shard = parse_args(
            [
                "control-shard",
                "/sanitized",
                "/training",
                "/results",
                "--system",
                "blocket",
                "--interface",
                "unseen",
                "--start",
                "60",
                "--stop",
                "64",
            ]
        )
        self.assertEqual(
            (shard.system, shard.interface, shard.start, shard.stop),
            ("blocket", "unseen", 60, 64),
        )
        finalized = parse_args(
            [
                "finalize",
                "/sanitized",
                "/training",
                "/results",
                "--require-gates",
                "1,2,3,4,5,6,7,8,9",
            ]
        )
        self.assertEqual(finalized.require_gates, tuple(range(1, 10)))


if __name__ == "__main__":
    unittest.main()
