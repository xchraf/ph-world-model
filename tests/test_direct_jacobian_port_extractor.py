import pytest
import torch

from blocket_league.direct_jacobian_port_extractor import (
    EmpiricalTangentAccumulator,
    EmpiricalTangentConfig,
    FrozenEmpiricalJacobianActivationPort,
    FrozenJacobianActivationPort,
    build_empirical_tangent_artifact,
    polar_riesz_write_basis,
)


def test_polar_riesz_port_is_parameter_free_oriented_and_orthonormal():
    generator = torch.Generator().manual_seed(151_910_737)
    first = torch.randn(3, 20, 2, generator=generator, requires_grad=True)
    second = 2.5 * first.detach() + 0.05 * torch.randn(
        3, 20, 2, generator=generator
    )
    extractor = FrozenJacobianActivationPort(
        history_frames=2,
        patch_count=2,
        hidden_size=5,
        port_size=2,
    )
    result = extractor({1: first, 4: second})
    extractor.assert_parameter_free()
    assert result.write_basis.shape == (3, 2, 2, 5, 2)
    assert not result.write_basis.requires_grad
    flat = result.write_basis.flatten(1, 3)
    identity = torch.eye(2).expand(3, 2, 2)
    torch.testing.assert_close(flat.transpose(-1, -2) @ flat, identity)
    assert float(result.orthonormality_defect.max()) < 1e-5
    # The polar factor is oriented: U^T G is symmetric positive semidefinite.
    cross = flat.transpose(-1, -2) @ result.aggregate_covectors
    torch.testing.assert_close(cross, cross.transpose(-1, -2), atol=2e-6, rtol=2e-6)
    assert float(torch.linalg.eigvalsh(cross).min()) >= -2e-6


def test_horizon_scaling_does_not_change_extracted_port():
    generator = torch.Generator().manual_seed(23)
    covectors = torch.randn(4, 13, 2, generator=generator)
    reference = polar_riesz_write_basis({1: covectors, 2: covectors})
    rescaled = polar_riesz_write_basis({1: 17.0 * covectors, 2: 0.3 * covectors})
    torch.testing.assert_close(
        reference.write_basis, rescaled.write_basis, atol=2e-6, rtol=2e-6
    )


def test_extractor_rejects_malformed_or_nonfinite_covectors():
    with pytest.raises(ValueError):
        polar_riesz_write_basis({})
    with pytest.raises(ValueError):
        polar_riesz_write_basis({0: torch.ones(1, 4, 1)})
    with pytest.raises(ValueError):
        polar_riesz_write_basis(
            {1: torch.ones(1, 4, 1), 2: torch.ones(2, 4, 1)}
        )
    invalid = torch.ones(1, 4, 1)
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        polar_riesz_write_basis({1: invalid})


def test_empirical_extractor_uses_fit_only_tangent_and_current_jacobian():
    generator = torch.Generator().manual_seed(804)
    samples, times, patches, channels = 7, 2, 3, 4
    source = torch.randn(
        samples, times, patches, channels, generator=generator
    )
    predicted = source + 0.02 * torch.randn(
        source.shape, generator=generator
    )
    observed = predicted.clone()
    # Innovations occupy exactly the first two channel directions but have
    # context-dependent time/patch support.
    observed[..., :2] += torch.randn(
        samples, times, patches, 2, generator=generator
    )
    config = EmpiricalTangentConfig(
        channel_rank=2, neighbors=2, support_floor_ratio=0.01
    )
    artifact = build_empirical_tangent_artifact(
        source.detach(), observed.detach(), predicted.detach(), config
    )
    extractor = FrozenEmpiricalJacobianActivationPort(
        artifact,
        history_frames=times,
        patch_count=patches,
        hidden_size=channels,
        port_size=1,
        config=config,
    )
    extractor.assert_frozen_parameter_free()
    query = source[:2].detach()
    ambient = times * patches * channels
    jacobian = torch.randn(2, ambient, 1, generator=generator, requires_grad=True)
    result = extractor({1: jacobian, 2: 3.0 * jacobian}, query)
    assert not result.jacobian.write_basis.requires_grad
    assert result.neighbor_indices.shape == (2, 2)
    assert result.local_support.shape == (2, times, patches)
    assert bool((result.projected_signal_ratio > 0.0).all())
    assert bool((result.projected_signal_ratio <= 1.0 + 1e-5).all())
    flat = result.jacobian.write_basis.flatten(1, 3)
    torch.testing.assert_close(
        (flat.square().sum(dim=1)), torch.ones(2, 1), atol=2e-6, rtol=2e-6
    )
    # Every output channel lies inside the artifact's empirical PCA span.
    shaped = result.jacobian.write_basis
    basis = artifact.channel_basis
    reconstructed = torch.einsum(
        "btpdm,dr,er->btpem",
        shaped,
        basis,
        basis,
    )
    torch.testing.assert_close(shaped, reconstructed, atol=2e-5, rtol=2e-5)


def test_fit_row_can_be_excluded_from_its_own_tangent_neighborhood():
    generator = torch.Generator().manual_seed(17)
    source = torch.randn(5, 1, 2, 3, generator=generator)
    predicted = source.clone()
    observed = predicted + torch.randn(source.shape, generator=generator)
    config = EmpiricalTangentConfig(channel_rank=2, neighbors=2)
    artifact = build_empirical_tangent_artifact(source, observed, predicted, config)
    extractor = FrozenEmpiricalJacobianActivationPort(
        artifact,
        history_frames=1,
        patch_count=2,
        hidden_size=3,
        port_size=1,
        config=config,
    )
    covectors = {1: torch.randn(2, 6, 1, generator=generator)}
    excluded = torch.tensor([0, 1], dtype=torch.long)
    result = extractor(covectors, source[:2], excluded_fit_rows=excluded)
    assert not bool((result.neighbor_indices == excluded[:, None]).any())


def test_streaming_tangent_statistics_match_closed_form_subspace():
    generator = torch.Generator().manual_seed(41)
    source = torch.randn(8, 2, 3, 5, generator=generator)
    predicted = source + 0.1 * torch.randn(source.shape, generator=generator)
    observed = predicted + torch.randn(source.shape, generator=generator)
    config = EmpiricalTangentConfig(channel_rank=3, neighbors=2)
    closed = build_empirical_tangent_artifact(source, observed, predicted, config)
    accumulator = EmpiricalTangentAccumulator(config)
    accumulator.update(source[:3], observed[:3], predicted[:3])
    accumulator.update(source[3:], observed[3:], predicted[3:])
    streamed = accumulator.finalize()
    closed_projector = closed.channel_basis @ closed.channel_basis.T
    streamed_projector = streamed.channel_basis @ streamed.channel_basis.T
    torch.testing.assert_close(
        closed_projector, streamed_projector, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        closed.feature_locations, streamed.feature_locations, atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        closed.innovation_support,
        streamed.innovation_support,
        atol=2e-6,
        rtol=2e-6,
    )
    assert len(streamed.source_tensor_sha256) == 64
