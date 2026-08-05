"""Parameter-free activation ports extracted from a frozen Jacobian lens.

The direct experiment must not learn an activation write field jointly with
the port-Hamiltonian state.  Such a field could co-adapt with ``E``, ``J`` and
``B`` and would support only the weaker statement "learned under Jacobian
constraints".  This module instead turns the frozen lens covectors into a
write basis by a deterministic Euclidean Riesz map followed by a polar
factor.  It has no parameter, no simulator input and no physical label.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class JacobianPortExtraction:
    """A sealed local activation port and diagnostics of its extraction."""

    write_basis: torch.Tensor
    aggregate_covectors: torch.Tensor
    singular_values: torch.Tensor
    orthonormality_defect: torch.Tensor
    horizon_relative_weights: dict[int, torch.Tensor]


@dataclass(frozen=True)
class EmpiricalTangentConfig:
    """Pixels-only tangent envelope used to reject off-manifold writes."""

    channel_rank: int = 16
    neighbors: int = 32
    support_floor_ratio: float = 0.02

    def __post_init__(self) -> None:
        if type(self.channel_rank) is not int or self.channel_rank < 1:
            raise ValueError("channel_rank must be a positive integer")
        if type(self.neighbors) is not int or self.neighbors < 1:
            raise ValueError("neighbors must be a positive integer")
        if (
            type(self.support_floor_ratio) not in (int, float)
            or not math.isfinite(float(self.support_floor_ratio))
            or not 0.0 <= self.support_floor_ratio < 1.0
        ):
            raise ValueError("support_floor_ratio must lie in [0,1)")


@dataclass(frozen=True)
class EmpiricalTangentArtifact:
    """Fit-only sufficient statistics for a non-parametric local tangent."""

    channel_basis: torch.Tensor
    feature_locations: torch.Tensor
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    innovation_support: torch.Tensor
    innovation_channel_eigenvalues: torch.Tensor
    source_tensor_sha256: str


@dataclass(frozen=True)
class EmpiricalJacobianPortExtraction:
    """Exact Jacobian port together with its fit-neighbour provenance."""

    jacobian: JacobianPortExtraction
    neighbor_indices: torch.Tensor
    local_support: torch.Tensor
    projected_signal_ratio: torch.Tensor


class EmpiricalTangentAccumulator:
    """Streaming sufficient statistics for a large fit activation suite."""

    def __init__(self, config: EmpiricalTangentConfig = EmpiricalTangentConfig()) -> None:
        self.config = config
        self._shape: tuple[int, int, int] | None = None
        self._channel_sum: torch.Tensor | None = None
        self._channel_outer: torch.Tensor | None = None
        self._token_count = 0
        self._features: list[torch.Tensor] = []
        self._supports: list[torch.Tensor] = []
        self._digest = hashlib.sha256()

    def update(
        self,
        source_activations: torch.Tensor,
        observed_successor_activations: torch.Tensor,
        predicted_successor_activations: torch.Tensor,
    ) -> None:
        if (
            source_activations.shape != observed_successor_activations.shape
            or source_activations.shape != predicted_successor_activations.shape
            or source_activations.ndim != 4
            or source_activations.shape[0] < 1
        ):
            raise ValueError("streaming tangent batches must share [B,T,P,D] shape")
        if not all(
            value.is_floating_point()
            and bool(torch.isfinite(value).all())
            and not value.requires_grad
            and value.grad_fn is None
            for value in (
                source_activations,
                observed_successor_activations,
                predicted_successor_activations,
            )
        ):
            raise ValueError("streaming tangent activations must be finite and detached")
        shape = tuple(source_activations.shape[1:])
        if self._shape is None:
            self._shape = shape
            channel_size = shape[-1]
            if self.config.channel_rank > channel_size:
                raise ValueError("channel_rank exceeds the activation channel size")
            self._channel_sum = torch.zeros(channel_size, dtype=torch.float64)
            self._channel_outer = torch.zeros(
                channel_size, channel_size, dtype=torch.float64
            )
        elif shape != self._shape:
            raise ValueError("streaming tangent activation shape changed")

        source_cpu = source_activations.detach().float().cpu().contiguous()
        observed_cpu = observed_successor_activations.detach().float().cpu().contiguous()
        predicted_cpu = predicted_successor_activations.detach().float().cpu().contiguous()
        innovation = observed_cpu - predicted_cpu
        tokens = innovation.flatten(0, 2).double()
        assert self._channel_sum is not None and self._channel_outer is not None
        self._channel_sum += tokens.sum(dim=0)
        self._channel_outer += tokens.T @ tokens
        self._token_count += int(tokens.shape[0])
        self._features.append(_activation_features(source_cpu))
        self._supports.append(innovation.square().mean(dim=-1).sqrt())
        for value in (source_cpu, observed_cpu, predicted_cpu):
            self._digest.update(str(value.dtype).encode("ascii"))
            self._digest.update(str(tuple(value.shape)).encode("ascii"))
            self._digest.update(value.view(torch.uint8).numpy().tobytes())

    def finalize(self) -> EmpiricalTangentArtifact:
        if (
            self._shape is None
            or self._channel_sum is None
            or self._channel_outer is None
            or not self._features
            or self._token_count < 2
        ):
            raise ValueError("cannot finalize an empty tangent accumulator")
        features = torch.cat(self._features, dim=0)
        support = torch.cat(self._supports, dim=0)
        if features.shape[0] <= self.config.neighbors:
            raise ValueError("fit activations must outnumber the registered neighbors")
        mean = self._channel_sum / self._token_count
        covariance = (
            self._channel_outer
            - self._token_count * torch.outer(mean, mean)
        ) / (self._token_count - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        channel_size = covariance.shape[0]
        order = torch.arange(
            channel_size - 1,
            channel_size - self.config.channel_rank - 1,
            -1,
        )
        selected_values = eigenvalues[order].clamp_min(0.0).float()
        basis = eigenvectors[:, order].float()
        pivots = basis.abs().argmax(dim=0)
        signs = torch.sign(
            basis[pivots, torch.arange(self.config.channel_rank)]
        )
        basis = basis * torch.where(signs == 0.0, torch.ones_like(signs), signs)
        feature_mean = features.mean(dim=0)
        feature_scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
        return EmpiricalTangentArtifact(
            channel_basis=basis.contiguous(),
            feature_locations=((features - feature_mean) / feature_scale).contiguous(),
            feature_mean=feature_mean.contiguous(),
            feature_scale=feature_scale.contiguous(),
            innovation_support=support.contiguous(),
            innovation_channel_eigenvalues=selected_values.contiguous(),
            source_tensor_sha256=self._digest.hexdigest(),
        )


def _tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _activation_features(activation: torch.Tensor) -> torch.Tensor:
    if activation.ndim != 4:
        raise ValueError("activation must have shape [sample,time,patch,channel]")
    flattened = activation.flatten(1, 2).float()
    return torch.cat(
        (
            flattened.mean(dim=1),
            flattened.std(dim=1, unbiased=False),
        ),
        dim=-1,
    )


def build_empirical_tangent_artifact(
    source_activations: torch.Tensor,
    observed_successor_activations: torch.Tensor,
    predicted_successor_activations: torch.Tensor,
    config: EmpiricalTangentConfig = EmpiricalTangentConfig(),
) -> EmpiricalTangentArtifact:
    r"""Build a closed-form fit-only activation tangent from pixels.

    ``observed - predicted`` is the innovation left by a zero-write frozen
    video rollout.  Every argument is an activation of the same sealed video
    backbone; no latent state, action, object mask or simulator quantity is
    accepted.  A channel covariance over all time/patch tokens gives a compact
    empirical channel tangent.  Per-context innovation energy retains the
    spatiotemporal support needed for a local, non-parametric envelope.
    """

    if (
        source_activations.shape != observed_successor_activations.shape
        or source_activations.shape != predicted_successor_activations.shape
        or source_activations.ndim != 4
    ):
        raise ValueError(
            "source, observed-successor and predicted-successor activations "
            "must share [sample,time,patch,channel] shape"
        )
    if not all(
        value.is_floating_point()
        and bool(torch.isfinite(value).all())
        and not value.requires_grad
        and value.grad_fn is None
        for value in (
            source_activations,
            observed_successor_activations,
            predicted_successor_activations,
        )
    ):
        raise ValueError("tangent-artifact activations must be finite and detached")
    sample_count, _, _, channel_size = source_activations.shape
    if sample_count <= config.neighbors:
        raise ValueError("fit activations must outnumber the registered neighbors")
    if config.channel_rank > channel_size:
        raise ValueError("channel_rank cannot exceed the activation channel size")

    innovation = (
        observed_successor_activations.float()
        - predicted_successor_activations.float()
    )
    token_innovations = innovation.flatten(0, 2)
    channel_mean = token_innovations.mean(dim=0, keepdim=True)
    centered = token_innovations - channel_mean
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.arange(
        channel_size - 1,
        channel_size - config.channel_rank - 1,
        -1,
        device=eigenvalues.device,
    )
    selected_values = eigenvalues[order].clamp_min(0.0)
    basis = eigenvectors[:, order]
    # Eigh signs are arbitrary.  Anchor each component deterministically to
    # the largest-magnitude channel so artifact hashes are reproducible.
    pivots = basis.abs().argmax(dim=0)
    signs = torch.sign(basis[pivots, torch.arange(config.channel_rank, device=basis.device)])
    basis = basis * torch.where(signs == 0.0, torch.ones_like(signs), signs)

    features = _activation_features(source_activations)
    feature_mean = features.mean(dim=0)
    feature_scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    locations = (features - feature_mean) / feature_scale
    support = innovation.square().mean(dim=-1).sqrt()
    return EmpiricalTangentArtifact(
        channel_basis=basis.detach().contiguous(),
        feature_locations=locations.detach().contiguous(),
        feature_mean=feature_mean.detach().contiguous(),
        feature_scale=feature_scale.detach().contiguous(),
        innovation_support=support.detach().contiguous(),
        innovation_channel_eigenvalues=selected_values.detach().contiguous(),
        source_tensor_sha256=_tensor_sha256(
            source_activations,
            observed_successor_activations,
            predicted_successor_activations,
        ),
    )


def make_synthetic_empirical_tangent_artifact_for_tests(
    *,
    history_frames: int,
    patch_count: int,
    hidden_size: int,
    config: EmpiricalTangentConfig,
    seed: int = 151_910_737,
) -> EmpiricalTangentArtifact:
    """Construct a tiny shape-correct artifact for tests/timing probes only.

    Registered training entry points never call this helper.  Its explicit
    name prevents a random placeholder from being mistaken for pixels-only
    scientific evidence.
    """

    if type(seed) is not int:
        raise ValueError("synthetic tangent seed must be an integer")
    sample_count = config.neighbors + 2
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (sample_count, history_frames, patch_count, hidden_size)
    source = torch.randn(shape, generator=generator)
    predicted = source + 0.01 * torch.randn(shape, generator=generator)
    observed = predicted + 0.10 * torch.randn(shape, generator=generator)
    return build_empirical_tangent_artifact(
        source, observed, predicted, config
    )


def _validated_covectors(
    activation_covectors: Mapping[int, torch.Tensor],
) -> tuple[tuple[int, ...], tuple[int, int, int]]:
    if type(activation_covectors) is not dict or not activation_covectors:
        raise ValueError("activation_covectors must be a non-empty plain dict")
    horizons = tuple(sorted(activation_covectors))
    if any(type(horizon) is not int or horizon < 1 for horizon in horizons):
        raise ValueError("activation-covector horizons must be positive integers")
    reference_shape: tuple[int, int, int] | None = None
    for horizon in horizons:
        covectors = activation_covectors[horizon]
        if type(covectors) is not torch.Tensor or covectors.ndim != 3:
            raise ValueError(
                "each activation-covector family must have shape "
                "[batch,activation,port]"
            )
        if not covectors.is_floating_point():
            raise TypeError("activation covectors must be floating point")
        if covectors.shape[0] < 1 or covectors.shape[1] < 1 or covectors.shape[2] < 1:
            raise ValueError("activation-covector dimensions must be positive")
        shape = tuple(covectors.shape)
        if reference_shape is None:
            reference_shape = shape
        elif shape != reference_shape:
            raise ValueError("all horizons must have the same covector shape")
        if not bool(torch.isfinite(covectors).all()):
            raise ValueError("activation covectors must be finite")
    assert reference_shape is not None
    return horizons, reference_shape


def polar_riesz_write_basis(
    activation_covectors: Mapping[int, torch.Tensor],
    *,
    epsilon: float = 1e-12,
) -> JacobianPortExtraction:
    r"""Extract an oriented orthonormal tangent port from frozen VJPs.

    For horizon ``k``, let ``G_k`` contain the activation covectors
    ``d observable / d activation`` as columns.  The frozen residual stream
    carries its ordinary Euclidean metric, so its Riesz map identifies each
    covector with a tangent vector.  We first give every registered horizon
    equal Frobenius weight,

    ``G = mean_k G_k / max(||G_k||_F, epsilon)``,

    and then take the thin polar factor ``U = left @ right`` of ``G``.  Unlike
    an arbitrary SVD basis, this is the closest orthonormal matrix to ``G``
    and therefore preserves the fixed pixel-probe orientation.  Sign changes
    of singular vectors cancel in the product.  The returned tensors are
    detached by construction: no trainable model can rotate this port.
    """

    if not isinstance(epsilon, float) or not torch.isfinite(torch.tensor(epsilon)):
        raise ValueError("epsilon must be a finite float")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    horizons, (batch_size, activation_size, port_size) = _validated_covectors(
        activation_covectors
    )
    detached = {
        horizon: activation_covectors[horizon].detach()
        for horizon in horizons
    }
    relative_weights: dict[int, torch.Tensor] = {}
    normalized = []
    for horizon in horizons:
        covectors = detached[horizon]
        norm = torch.linalg.matrix_norm(
            covectors, ord="fro", dim=(-2, -1), keepdim=True
        )
        scale = norm.clamp_min(epsilon).reciprocal()
        relative_weights[horizon] = scale.squeeze(-1).squeeze(-1)
        normalized.append(covectors * scale)
    aggregate = torch.stack(normalized, dim=0).mean(dim=0)

    # The left/right product is the unique thin polar factor whenever G has
    # full column rank.  It remains a deterministic orthonormal completion at
    # a rank-deficient sample, which is surfaced through singular_values and
    # rejected by the held-out extraction gate rather than hidden here.
    left, singular_values, right_transpose = torch.linalg.svd(
        aggregate, full_matrices=False
    )
    basis = left @ right_transpose
    identity = torch.eye(
        port_size, dtype=basis.dtype, device=basis.device
    ).expand(batch_size, port_size, port_size)
    orthonormality = torch.linalg.matrix_norm(
        basis.transpose(-1, -2) @ basis - identity,
        ord="fro",
        dim=(-2, -1),
    )
    if basis.shape != (batch_size, activation_size, port_size):
        raise AssertionError("thin polar extraction returned an invalid shape")
    return JacobianPortExtraction(
        write_basis=basis.detach(),
        aggregate_covectors=aggregate.detach(),
        singular_values=singular_values.detach(),
        orthonormality_defect=orthonormality.detach(),
        horizon_relative_weights={
            horizon: weight.detach()
            for horizon, weight in relative_weights.items()
        },
    )


class FrozenJacobianActivationPort(nn.Module):
    """Zero-parameter module exposing only deterministic Jacobian extraction."""

    def __init__(
        self,
        *,
        history_frames: int,
        patch_count: int,
        hidden_size: int,
        port_size: int,
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        dimensions = (history_frames, patch_count, hidden_size, port_size)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("all frozen-port dimensions must be positive integers")
        self.history_frames = history_frames
        self.patch_count = patch_count
        self.hidden_size = hidden_size
        self.port_size = port_size
        self.epsilon = float(epsilon)

    @property
    def ambient_size(self) -> int:
        return self.history_frames * self.patch_count * self.hidden_size

    def forward(
        self,
        activation_covectors: Mapping[int, torch.Tensor],
    ) -> JacobianPortExtraction:
        extraction = polar_riesz_write_basis(
            activation_covectors, epsilon=self.epsilon
        )
        expected = (self.ambient_size, self.port_size)
        if tuple(extraction.write_basis.shape[-2:]) != expected:
            raise ValueError(
                f"expected flattened activation-port tail {expected}, got "
                f"{tuple(extraction.write_basis.shape[-2:])}"
            )
        shaped = extraction.write_basis.reshape(
            extraction.write_basis.shape[0],
            self.history_frames,
            self.patch_count,
            self.hidden_size,
            self.port_size,
        )
        return JacobianPortExtraction(
            write_basis=shaped,
            aggregate_covectors=extraction.aggregate_covectors,
            singular_values=extraction.singular_values,
            orthonormality_defect=extraction.orthonormality_defect,
            horizon_relative_weights=extraction.horizon_relative_weights,
        )

    def assert_parameter_free(self) -> None:
        if tuple(self.parameters()) or self.state_dict():
            raise AssertionError("the frozen Jacobian port acquired mutable tensors")


class FrozenEmpiricalJacobianActivationPort(nn.Module):
    r"""Exact Jacobian extraction inside a fit-only empirical tangent.

    The module stores only detached statistics computed from the fit pixels.
    For a new source activation it finds neighbours in a fixed frozen-feature
    metric, averages their innovation support, projects every frozen Jacobian
    covector onto the closed-form channel PCA tangent, and applies the polar
    Riesz extraction.  There are no trainable parameters and no interpolation
    of a previously learned port: the current context's Jacobian is always
    evaluated explicitly by the caller.
    """

    def __init__(
        self,
        artifact: EmpiricalTangentArtifact,
        *,
        history_frames: int,
        patch_count: int,
        hidden_size: int,
        port_size: int,
        config: EmpiricalTangentConfig = EmpiricalTangentConfig(),
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        if artifact.channel_basis.ndim != 2:
            raise ValueError("artifact channel_basis must be a matrix")
        if artifact.channel_basis.shape != (hidden_size, config.channel_rank):
            raise ValueError("artifact channel basis does not match the extractor")
        expected_features = 2 * hidden_size
        if (
            artifact.feature_locations.ndim != 2
            or artifact.feature_locations.shape[1] != expected_features
            or artifact.feature_mean.shape != (expected_features,)
            or artifact.feature_scale.shape != (expected_features,)
        ):
            raise ValueError("artifact frozen-feature statistics are malformed")
        if artifact.innovation_support.shape != (
            artifact.feature_locations.shape[0],
            history_frames,
            patch_count,
        ):
            raise ValueError("artifact innovation support has the wrong shape")
        if artifact.feature_locations.shape[0] <= config.neighbors:
            raise ValueError("artifact has too few fit contexts for neighbor exclusion")
        if artifact.innovation_channel_eigenvalues.shape != (config.channel_rank,):
            raise ValueError("artifact eigenvalue spectrum is malformed")
        if (
            type(artifact.source_tensor_sha256) is not str
            or len(artifact.source_tensor_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in artifact.source_tensor_sha256
            )
        ):
            raise ValueError("artifact source hash is malformed")
        dimensions = (history_frames, patch_count, hidden_size, port_size)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("all empirical-port dimensions must be positive integers")
        if epsilon <= 0.0 or not math.isfinite(float(epsilon)):
            raise ValueError("epsilon must be finite and positive")
        self.history_frames = history_frames
        self.patch_count = patch_count
        self.hidden_size = hidden_size
        self.port_size = port_size
        self.config = config
        self.epsilon = float(epsilon)
        self.source_tensor_sha256 = artifact.source_tensor_sha256
        self.register_buffer("channel_basis", artifact.channel_basis.detach().clone())
        self.register_buffer(
            "feature_locations", artifact.feature_locations.detach().clone()
        )
        self.register_buffer("feature_mean", artifact.feature_mean.detach().clone())
        self.register_buffer("feature_scale", artifact.feature_scale.detach().clone())
        self.register_buffer(
            "innovation_support", artifact.innovation_support.detach().clone()
        )
        self.register_buffer(
            "innovation_channel_eigenvalues",
            artifact.innovation_channel_eigenvalues.detach().clone(),
        )

    @property
    def ambient_size(self) -> int:
        return self.history_frames * self.patch_count * self.hidden_size

    def train(self, mode: bool = True):
        # This operator is a sealed artifact, not a train/eval-dependent
        # network.  Parent models may enter train mode without changing it.
        super().train(False)
        return self

    def _local_support(
        self,
        source_activation: torch.Tensor,
        *,
        excluded_fit_rows: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = _activation_features(source_activation)
        standardized = (features - self.feature_mean) / self.feature_scale
        distances = torch.cdist(
            standardized.float(), self.feature_locations.float(), p=2
        ).square()
        if excluded_fit_rows is not None:
            if (
                type(excluded_fit_rows) is not torch.Tensor
                or excluded_fit_rows.dtype != torch.long
                or excluded_fit_rows.device != distances.device
                or excluded_fit_rows.shape != (source_activation.shape[0],)
            ):
                raise ValueError(
                    "excluded_fit_rows must be a batch int64 tensor on the same device"
                )
            if bool(
                ((excluded_fit_rows < 0) | (excluded_fit_rows >= distances.shape[1])).any()
            ):
                raise ValueError("an excluded fit row lies outside the artifact")
            distances[
                torch.arange(distances.shape[0], device=distances.device),
                excluded_fit_rows,
            ] = torch.inf
        neighbors = torch.topk(
            distances,
            k=self.config.neighbors,
            dim=-1,
            largest=False,
            sorted=True,
        ).indices
        local = self.innovation_support[neighbors].mean(dim=1)
        mean_support = local.mean(dim=(-2, -1), keepdim=True).clamp_min(self.epsilon)
        normalized = local / mean_support
        normalized = normalized.clamp_min(self.config.support_floor_ratio)
        return normalized.detach(), neighbors.detach()

    def forward(
        self,
        activation_covectors: Mapping[int, torch.Tensor],
        source_activation: torch.Tensor,
        *,
        excluded_fit_rows: torch.Tensor | None = None,
    ) -> EmpiricalJacobianPortExtraction:
        horizons, shape = _validated_covectors(activation_covectors)
        expected_shape = (
            source_activation.shape[0],
            self.ambient_size,
            self.port_size,
        )
        if shape != expected_shape:
            raise ValueError(
                f"activation covectors must have shape {expected_shape}, got {shape}"
            )
        if source_activation.shape != (
            shape[0],
            self.history_frames,
            self.patch_count,
            self.hidden_size,
        ):
            raise ValueError("source activation does not match the extractor shape")
        if source_activation.requires_grad or source_activation.grad_fn is not None:
            raise ValueError("source activation must be detached from the backbone")
        support, neighbors = self._local_support(
            source_activation, excluded_fit_rows=excluded_fit_rows
        )
        weighted_covectors: dict[int, torch.Tensor] = {}
        original_energy = source_activation.new_zeros(shape[0])
        projected_energy = source_activation.new_zeros(shape[0])
        basis = self.channel_basis.to(
            device=source_activation.device, dtype=source_activation.dtype
        )
        support_weight = support.sqrt()[..., None, None]
        for horizon in horizons:
            covectors = activation_covectors[horizon].detach().reshape(
                shape[0],
                self.history_frames,
                self.patch_count,
                self.hidden_size,
                self.port_size,
            )
            coefficients = torch.einsum("btpdm,dr->btprm", covectors, basis)
            projected = torch.einsum("btprm,dr->btpdm", coefficients, basis)
            weighted = projected * support_weight
            weighted_covectors[horizon] = weighted.flatten(1, 3)
            original_energy = original_energy + covectors.square().sum(
                dim=(1, 2, 3, 4)
            )
            projected_energy = projected_energy + projected.square().sum(
                dim=(1, 2, 3, 4)
            )
        extraction = polar_riesz_write_basis(
            weighted_covectors, epsilon=self.epsilon
        )
        shaped = extraction.write_basis.reshape(
            shape[0],
            self.history_frames,
            self.patch_count,
            self.hidden_size,
            self.port_size,
        )
        extraction = JacobianPortExtraction(
            write_basis=shaped,
            aggregate_covectors=extraction.aggregate_covectors,
            singular_values=extraction.singular_values,
            orthonormality_defect=extraction.orthonormality_defect,
            horizon_relative_weights=extraction.horizon_relative_weights,
        )
        return EmpiricalJacobianPortExtraction(
            jacobian=extraction,
            neighbor_indices=neighbors,
            local_support=support,
            projected_signal_ratio=(
                projected_energy / original_energy.clamp_min(self.epsilon)
            ).detach(),
        )

    def assert_frozen_parameter_free(self) -> None:
        if tuple(self.parameters()):
            raise AssertionError("the empirical Jacobian port acquired parameters")
        if any(value.requires_grad for value in self.buffers()):
            raise AssertionError("the empirical tangent artifact acquired gradients")


__all__ = [
    "EmpiricalJacobianPortExtraction",
    "EmpiricalTangentArtifact",
    "EmpiricalTangentAccumulator",
    "EmpiricalTangentConfig",
    "FrozenEmpiricalJacobianActivationPort",
    "FrozenJacobianActivationPort",
    "JacobianPortExtraction",
    "build_empirical_tangent_artifact",
    "make_synthetic_empirical_tangent_artifact_for_tests",
    "polar_riesz_write_basis",
]
