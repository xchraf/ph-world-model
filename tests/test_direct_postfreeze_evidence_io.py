from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from blocket_league.direct_physical_evaluation import (
    ControlResult,
    control_trace_sha256,
    fixed_interfaces,
    linear_interface_protocol,
    SYSTEMS,
)
from blocket_league.direct_postfreeze_evidence_io import (
    AuthenticatedControlShard,
    REGISTERED_CONTROL_EPISODE_SEED,
    REGISTERED_CONTROL_NAMES,
    REGISTERED_CONTROL_PLANNER_SEED,
    canonical_evidence_sha256,
    load_control_shard,
    save_control_shard,
)


class TypedPostFreezeEvidenceIOTests(unittest.TestCase):
    @staticmethod
    def _control_result() -> ControlResult:
        names = REGISTERED_CONTROL_NAMES
        commands = {
            name: tuple(
                tuple((0.01 * (controller + episode + step),) for step in range(3))
                for episode in range(2)
            )
            for controller, name in enumerate(names)
        }
        return ControlResult(
            errors={
                name: (0.5 + controller * 0.01, 0.6 + controller * 0.01)
                for controller, name in enumerate(names)
            },
            interface_name="native",
            episodes=2,
            control_steps=3,
            planner_budget={
                "candidatesPerDecision": 512,
                "iterationsPerDecision": 4,
                "elitesPerIteration": 64,
                "horizon": 24,
                "candidateEvaluationsPerDecision": 2048,
                "pairedCandidateNoiseAcrossLearnedPlanners": 1,
                "activationRolloutMicroBatch": 32,
            },
            episode_identifiers=("pendulum-a", "pendulum-b"),
            interface_command_traces=commands,
            planner_seed_schedule_sha256="9" * 64,
            physical_protocol=linear_interface_protocol(
                SYSTEMS["pendulum"], fixed_interfaces("pendulum")["native"]
            ),
        )

    @classmethod
    def _shard(cls) -> AuthenticatedControlShard:
        result = cls._control_result()
        values = {
            "system_name": "pendulum",
            "interface_name": "native",
            "start": 0,
            "stop": 2,
            "total_episodes": 64,
            "episode_seed": REGISTERED_CONTROL_EPISODE_SEED,
            "planner_seed": REGISTERED_CONTROL_PLANNER_SEED,
            "training_lineage_sha256": "a" * 64,
            "physical_sha256": "b" * 64,
            "neural_hashes_before": {"full": "c" * 64},
            "neural_hashes_after": {"full": "c" * 64},
            "result": result,
            "trace_sha256": control_trace_sha256(result),
            "artifact_sha256": "0" * 64,
        }
        provisional = AuthenticatedControlShard.__new__(AuthenticatedControlShard)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        values["artifact_sha256"] = canonical_evidence_sha256(
            provisional.core_payload()
        )
        return AuthenticatedControlShard(**values)

    def test_control_shard_roundtrip_and_raw_trace_tampering_fail_closed(self) -> None:
        shard = self._shard()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shard.pt"
            save_control_shard(path, shard)
            loaded = load_control_shard(path)
            self.assertEqual(loaded.artifact_sha256, shard.artifact_sha256)
            self.assertEqual(loaded.trace_sha256, shard.trace_sha256)

            payload = torch.load(path, weights_only=True)
            structured = list(
                payload["result"]["interfaceCommandTraces"]["structured"]
            )
            first_episode = list(structured[0])
            first_episode[0] = (0.9,)
            structured[0] = tuple(first_episode)
            payload["result"]["interfaceCommandTraces"]["structured"] = tuple(
                structured
            )
            torch.save(payload, path)
            with self.assertRaises(ValueError) as error:
                load_control_shard(path)
            self.assertRegex(
                str(error.exception),
                r"(provenance|sealed linear interface domain)",
            )

    def test_canonical_digest_binds_tensor_values_and_types(self) -> None:
        first = {"tensor": torch.tensor((1.0, 2.0)), "count": 2}
        second = {"tensor": torch.tensor((1.0, 3.0)), "count": 2}
        self.assertNotEqual(
            canonical_evidence_sha256(first), canonical_evidence_sha256(second)
        )
        self.assertNotEqual(
            canonical_evidence_sha256({"value": 1}),
            canonical_evidence_sha256({"value": 1.0}),
        )


if __name__ == "__main__":
    unittest.main()
