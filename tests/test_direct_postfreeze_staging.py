from __future__ import annotations

from dataclasses import replace
import unittest

from blocket_league.direct_postfreeze_evidence_io import canonical_evidence_sha256
from blocket_league.direct_postfreeze_staging import (
    PREPARED_KIND,
    PreparedSystemArtifact,
)


class PreparedPostFreezeArtifactTests(unittest.TestCase):
    @staticmethod
    def _artifact() -> PreparedSystemArtifact:
        evidence = {f"gate{gate}": {"raw": gate} for gate in range(1, 6)}
        evidence["gate5Energy"] = {"raw": "physical-energy-semantics"}
        core = {
            "kind": PREPARED_KIND,
            "system": "pendulum",
            "trainingLineageSha256": "a" * 64,
            "physicalSha256": "b" * 64,
            "evidence": evidence,
        }
        return PreparedSystemArtifact(
            system_name="pendulum",
            training_lineage_sha256="a" * 64,
            physical_sha256="b" * 64,
            evidence=evidence,
            artifact_sha256=canonical_evidence_sha256(core),
        )

    def test_prepared_artifact_binds_every_gate_and_rejects_tampering(self) -> None:
        artifact = self._artifact()
        self.assertEqual(artifact.to_payload()["kind"], PREPARED_KIND)
        changed = dict(artifact.evidence)
        changed["gate4"] = {"raw": 999}
        with self.assertRaisesRegex(ValueError, "digest"):
            replace(artifact, evidence=changed)

    def test_prepared_artifact_requires_gates_and_energy_subaudit(self) -> None:
        artifact = self._artifact()
        missing = dict(artifact.evidence)
        missing.pop("gate5")
        with self.assertRaisesRegex(ValueError, "exactly Gates"):
            replace(artifact, evidence=missing)
        missing_energy = dict(artifact.evidence)
        missing_energy.pop("gate5Energy")
        with self.assertRaisesRegex(ValueError, "energy sub-audit"):
            replace(artifact, evidence=missing_energy)


if __name__ == "__main__":
    unittest.main()
