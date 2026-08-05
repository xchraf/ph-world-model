import json

import pytest

from blocket_league.direct_full_only_realizability import _candidate_outcome
from blocket_league.direct_full_only_realizability import _prepare_result_directory


def test_full_only_outcome_never_claims_complete_breakthrough() -> None:
    result = _candidate_outcome(
        {"native": {"passed": True}, "unseen": {"passed": True}}
    )
    assert result["physicalRealizabilityCandidatePass"] is True
    assert result["completeBreakthroughClaimAllowed"] is False
    assert "closed_loop_control" in result["missingRequiredEvidence"]


def test_full_only_outcome_requires_both_interfaces() -> None:
    result = _candidate_outcome(
        {"native": {"passed": True}, "unseen": {"passed": False}}
    )
    assert result["physicalRealizabilityCandidatePass"] is False
    assert result["interpretation"].endswith("not_supported")


def test_result_directory_allows_only_precreated_source_manifest(tmp_path) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    manifest = result_dir / "evaluator-source-manifest.json"
    manifest.write_text(json.dumps({"sealed": True}), encoding="utf-8")
    _prepare_result_directory(result_dir, manifest)

    (result_dir / "stale.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="stale.json"):
        _prepare_result_directory(result_dir, manifest)
