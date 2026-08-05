from dataclasses import asdict
import hashlib

import pytest
import torch

from blocket_league.direct_jacobian_port_extractor import (
    EmpiricalTangentConfig,
    build_empirical_tangent_artifact,
)
from blocket_league.direct_jacobian_port_precompute import (
    JacobianPortPrecomputeConfig,
    _selected_transition_rows,
    build_empirical_tangent_from_pixels,
    load_empirical_tangent_artifact,
)
from blocket_league.direct_action_free_data import make_optimization_suite
from blocket_league.pixel_direct_model import DirectPixelTransformer, PixelDirectConfig
from blocket_league.runtime_firewall_trace import (
    RuntimeFirewallTrace,
    verify_runtime_trace,
)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def test_response_blind_transition_selection_uses_only_pixel_suite():
    contexts = torch.arange(3 * 5 * 2 * 2 * 2, dtype=torch.uint8).reshape(
        3, 5, 2, 2, 2
    )
    frames = torch.zeros(3, 5, 2, 2, dtype=torch.uint8)
    source, successor, indices = _selected_transition_rows(
        {"pixelContexts": contexts, "frames": frames}, 7
    )
    assert source.shape == successor.shape == (7, 2, 2, 2)
    torch.testing.assert_close(source, contexts[:, :-1].flatten(0, 1)[indices])
    torch.testing.assert_close(successor, contexts[:, 1:].flatten(0, 1)[indices])
    assert int(torch.unique(indices).numel()) == 7
    with pytest.raises(ValueError):
        _selected_transition_rows(
            {"pixelContexts": contexts, "frames": frames, "action": frames}, 7
        )


def test_tangent_artifact_loader_is_exact_and_fail_closed(tmp_path):
    generator = torch.Generator().manual_seed(9)
    source = torch.randn(6, 2, 3, 4, generator=generator)
    predicted = source + 0.1 * torch.randn(source.shape, generator=generator)
    observed = predicted + torch.randn(source.shape, generator=generator)
    precompute = JacobianPortPrecomputeConfig(
        contexts=6,
        batch_size=2,
        lens_block=1,
        channel_rank=2,
        neighbors=2,
    )
    artifact = build_empirical_tangent_artifact(
        source,
        observed,
        predicted,
        EmpiricalTangentConfig(channel_rank=2, neighbors=2),
    )
    indices = torch.arange(6, dtype=torch.long)
    digest = "a" * 64
    payload = {
        "kind": "frozen_empirical_jacobian_tangent_v1",
        "system": "toy",
        "actionChannels": 0,
        "physicalStateChannels": 0,
        "sourceSchema": ["pixelContexts", "frames"],
        "config": asdict(precompute),
        "fitSanitizedTensorSha256": digest,
        "sourceTreeSha256": "b" * 64,
        "backboneHash": "c" * 64,
        "selectedTransitionIndices": indices,
        "selectedTransitionIndicesSha256": _tensor_sha256(indices),
        "channelBasis": artifact.channel_basis,
        "featureLocations": artifact.feature_locations,
        "featureMean": artifact.feature_mean,
        "featureScale": artifact.feature_scale,
        "innovationSupport": artifact.innovation_support,
        "innovationChannelEigenvalues": artifact.innovation_channel_eigenvalues,
        "sourceActivationTensorSha256": artifact.source_tensor_sha256,
    }
    path = tmp_path / "empirical-tangent.pt"
    torch.save(payload, path)
    loaded = load_empirical_tangent_artifact(
        path,
        expected_system="toy",
        expected_fit_sanitized_tensor_sha256=digest,
        expected_source_tree_sha256="b" * 64,
        expected_backbone_hash="c" * 64,
        expected_config=precompute,
    )
    torch.testing.assert_close(loaded.channel_basis, artifact.channel_basis)
    tampered = dict(payload)
    tampered["actionChannels"] = 1
    torch.save(tampered, path)
    with pytest.raises(ValueError, match="actionChannels"):
        load_empirical_tangent_artifact(
            path,
            expected_system="toy",
            expected_fit_sanitized_tensor_sha256=digest,
            expected_source_tree_sha256="b" * 64,
            expected_backbone_hash="c" * 64,
            expected_config=precompute,
        )


def test_precompute_constructs_no_optimizer_and_runs_no_backward(
    tmp_path, monkeypatch
):
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
    backbone = DirectPixelTransformer(model_config).eval().requires_grad_(False)
    pixels = torch.randint(0, 9, (4, 4, 8, 8), dtype=torch.uint8)
    suite = make_optimization_suite(pixels, model_config, transitions=2)
    config = JacobianPortPrecomputeConfig(
        contexts=2,
        batch_size=1,
        lens_block=1,
        horizons=(1,),
        channel_rank=2,
        neighbors=1,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("port precompute attempted optimization/backpropagation")

    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    trace_path = tmp_path / "firewall-trace.jsonl"
    trace = RuntimeFirewallTrace(
        trace_path,
        stage="jacobian-port-precompute:toy",
        source_tree_sha256="a" * 64,
    )
    _, summary = build_empirical_tangent_from_pixels(
        backbone,
        suite,
        system="toy",
        fit_sanitized_tensor_sha256="b" * 64,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        config=config,
        runtime_trace=trace,
        source_tree_sha256="a" * 64,
    )
    seal = trace.snapshot().to_dict()
    trace.close()
    records = verify_runtime_trace(trace_path, seal)
    assert summary["runtimeTrace"]["events"] <= seal["events"]
    assert not any(
        record["event"] in {"optimizer_constructed", "gradient_batch"}
        for record in records
    )
    port_payloads = [
        record
        for record in records
        if record["event"] == "tensor_payload"
        and record["payload"].get("phase")
        == "jacobian_port_precompute_no_optimizer"
    ]
    assert len(port_payloads) == 1
