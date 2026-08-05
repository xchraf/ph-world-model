from __future__ import annotations

from typing import Literal

import torch


Reduction = Literal["none", "mean", "sum"]


__all__ = [
    "cotangent_pullback_solve",
    "pullback_compatibility_residual",
    "poisson_sharp",
    "orthonormal_subspace_basis",
    "subspace_projector",
    "principal_angles",
    "grassmannian_loss",
]


def _covector_layout(
    operator: torch.Tensor,
    covectors: torch.Tensor,
    *,
    domain_size: int,
    name: str,
) -> Literal["vector", "family"]:
    """Validate aligned batch dimensions and identify the covector layout."""

    if covectors.ndim == operator.ndim - 1 and covectors.shape[-1] == domain_size:
        return "vector"
    if covectors.ndim == operator.ndim and covectors.shape[-2] == domain_size:
        return "family"
    raise ValueError(
        f"{name} must have shape [..., {domain_size}] for one covector or "
        f"[..., {domain_size}, k] for a family, with batch dimensions aligned "
        "with the operator"
    )


def _as_columns(covectors: torch.Tensor, layout: str) -> torch.Tensor:
    return covectors.unsqueeze(-1) if layout == "vector" else covectors


def _restore_layout(covectors: torch.Tensor, layout: str) -> torch.Tensor:
    return covectors.squeeze(-1) if layout == "vector" else covectors


def cotangent_pullback_solve(
    adapter_jacobian: torch.Tensor,
    activation_covectors: torch.Tensor,
    *,
    ridge: float = 1e-6,
    regularizer_metric: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Recover state covectors from activation covectors by ridge regression.

    Let ``x = E(h)`` and ``A = D_h E`` with shape ``[..., n, d]``.  A state
    covector ``alpha_x`` pulls back to activation space as ``A.T alpha_x``.
    This function solves

    ``min_alpha ||A.T alpha - alpha_h||^2 + ridge * alpha.T G alpha``

    through the normal equation

    ``(A A.T + ridge G) alpha = A alpha_h``.

    ``activation_covectors`` may contain one covector with shape ``[..., d]``
    or a column family with shape ``[..., d, k]``.  The result respectively
    has shape ``[..., n]`` or ``[..., n, k]``.

    ``G`` defaults to the identity.  For covariance under a general state
    coordinate change ``x' = C x``, pass the transformed metric
    ``G' = C G C.T`` together with ``A' = C A``.  The returned covector then
    transforms correctly as ``alpha_x' = C^{-T} alpha_x``.  The identity
    default alone is covariant only under orthogonal coordinate changes.
    All operations are differentiable with respect to the Jacobian,
    covectors, and metric.
    """

    if adapter_jacobian.ndim < 2:
        raise ValueError("adapter_jacobian must have shape [..., n, d]")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")

    state_size, activation_size = adapter_jacobian.shape[-2:]
    layout = _covector_layout(
        adapter_jacobian,
        activation_covectors,
        domain_size=activation_size,
        name="activation_covectors",
    )
    covector_columns = _as_columns(activation_covectors, layout)

    gram = adapter_jacobian @ adapter_jacobian.transpose(-1, -2)
    if regularizer_metric is None:
        regularizer_metric = torch.eye(
            state_size,
            dtype=adapter_jacobian.dtype,
            device=adapter_jacobian.device,
        )
    elif regularizer_metric.shape[-2:] != (state_size, state_size):
        raise ValueError(
            "regularizer_metric must have shape [..., n, n] matching the "
            "state dimension"
        )

    right_hand_side = adapter_jacobian @ covector_columns
    solution = torch.linalg.solve(
        gram + ridge * regularizer_metric,
        right_hand_side,
    )
    return _restore_layout(solution, layout)


def pullback_compatibility_residual(
    adapter_jacobian: torch.Tensor,
    state_covectors: torch.Tensor,
    activation_covectors: torch.Tensor,
    *,
    relative: bool = True,
    eps: float | None = None,
) -> torch.Tensor:
    r"""Measure how well ``A.T alpha_x`` reconstructs ``alpha_h``.

    The activation dimension is reduced while batch dimensions and an
    optional family dimension are retained.  Consequently, a single
    covector returns ``[...]`` and a family returns ``[..., k]``.  The
    relative residual is

    ``||alpha_h - A.T alpha_x|| / max(||alpha_h||, eps)``.

    It is a direct row-space compatibility audit: a large value means the
    state adapter discarded part of the activation-space observable.
    """

    if adapter_jacobian.ndim < 2:
        raise ValueError("adapter_jacobian must have shape [..., n, d]")
    state_size, activation_size = adapter_jacobian.shape[-2:]
    activation_layout = _covector_layout(
        adapter_jacobian,
        activation_covectors,
        domain_size=activation_size,
        name="activation_covectors",
    )
    state_layout = _covector_layout(
        adapter_jacobian,
        state_covectors,
        domain_size=state_size,
        name="state_covectors",
    )
    if activation_layout != state_layout:
        raise ValueError(
            "state_covectors and activation_covectors must both be single "
            "covectors or both be column families"
        )

    state_columns = _as_columns(state_covectors, state_layout)
    activation_columns = _as_columns(activation_covectors, activation_layout)
    reconstructed = adapter_jacobian.transpose(-1, -2) @ state_columns
    error = torch.linalg.vector_norm(
        reconstructed - activation_columns,
        dim=-2,
    )
    if relative:
        if eps is None:
            eps = torch.finfo(activation_covectors.dtype).eps
        denominator = torch.linalg.vector_norm(activation_columns, dim=-2)
        error = error / denominator.clamp_min(eps)
    return error.squeeze(-1) if activation_layout == "vector" else error


def poisson_sharp(
    interconnection: torch.Tensor,
    state_covectors: torch.Tensor,
) -> torch.Tensor:
    r"""Convert state covectors to generalized-force vector fields.

    The operation is ``-J alpha``.  With the canonical convention
    ``J = [[0, I], [-I, 0]]``, this maps ``dq`` to the positive ``p``
    direction, which is the tangent direction of a generalized force.

    A single covector has shape ``[..., n]`` and a family of port covectors
    has shape ``[..., n, k]``.  Skew symmetry or the Jacobi identity are not
    silently assumed here; they should be audited on ``interconnection``.
    """

    if interconnection.ndim < 2 or interconnection.shape[-2] != interconnection.shape[-1]:
        raise ValueError("interconnection must have shape [..., n, n]")
    state_size = interconnection.shape[-1]
    layout = _covector_layout(
        interconnection,
        state_covectors,
        domain_size=state_size,
        name="state_covectors",
    )
    columns = _as_columns(state_covectors, layout)
    vector_fields = -(interconnection @ columns)
    return _restore_layout(vector_fields, layout)


def orthonormal_subspace_basis(
    spanning_vectors: torch.Tensor,
    *,
    rank: int | None = None,
) -> torch.Tensor:
    r"""Return an orthonormal basis for a column span.

    ``spanning_vectors`` has shape ``[..., ambient, columns]``.  By default
    all columns are assumed independent and a differentiable reduced QR is
    used.  If there are more columns than ambient dimensions, or ``rank`` is
    smaller than the number of columns, the leading left singular vectors
    define the requested fixed-rank subspace.  An explicit rank should be
    supplied for rank-deficient or overcomplete data because a batch cannot
    represent different dynamically inferred ranks in one dense tensor.
    """

    if spanning_vectors.ndim < 2:
        raise ValueError("spanning_vectors must have shape [..., ambient, columns]")
    ambient_size, column_count = spanning_vectors.shape[-2:]
    maximum_rank = min(ambient_size, column_count)
    selected_rank = maximum_rank if rank is None else rank
    if selected_rank < 1 or selected_rank > maximum_rank:
        raise ValueError(f"rank must lie between 1 and {maximum_rank}")

    if column_count <= ambient_size and selected_rank == column_count:
        basis, _ = torch.linalg.qr(spanning_vectors, mode="reduced")
        return basis

    left_vectors, _, _ = torch.linalg.svd(spanning_vectors, full_matrices=False)
    return left_vectors[..., :selected_rank]


def subspace_projector(
    spanning_vectors: torch.Tensor,
    *,
    rank: int | None = None,
) -> torch.Tensor:
    """Return the orthogonal projector onto a fixed-rank column span."""

    basis = orthonormal_subspace_basis(spanning_vectors, rank=rank)
    return basis @ basis.transpose(-1, -2)


def principal_angles(
    first_span: torch.Tensor,
    second_span: torch.Tensor,
    *,
    first_rank: int | None = None,
    second_rank: int | None = None,
) -> torch.Tensor:
    r"""Return principal angles in radians between two column subspaces.

    For training, prefer :func:`grassmannian_loss`: ``acos`` has an
    ill-conditioned derivative for coincident subspaces, whereas the chordal
    Grassmann loss is smooth there.
    """

    if first_span.shape[-2] != second_span.shape[-2]:
        raise ValueError("the subspaces must have the same ambient dimension")
    first_basis = orthonormal_subspace_basis(first_span, rank=first_rank)
    second_basis = orthonormal_subspace_basis(second_span, rank=second_rank)
    cross_gram = first_basis.transpose(-1, -2) @ second_basis
    cosine = torch.linalg.svdvals(cross_gram).clamp(min=0.0, max=1.0)
    return torch.acos(cosine)


def grassmannian_loss(
    first_span: torch.Tensor,
    second_span: torch.Tensor,
    *,
    rank: int | None = None,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    r"""Compute the chordal Grassmann loss ``mean_i sin(theta_i)^2``.

    The two subspaces must have equal dimension.  With orthonormal bases
    ``Q1`` and ``Q2``, the unreduced batch loss is

    ``1 - ||Q1.T Q2||_F^2 / rank``.

    It is invariant to a change of basis within either span, bounded in
    ``[0, 1]`` up to floating-point precision, and avoids differentiating
    through principal-angle ``acos``.
    """

    if first_span.shape[-2] != second_span.shape[-2]:
        raise ValueError("the subspaces must have the same ambient dimension")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    if rank is None:
        first_dimension = min(first_span.shape[-2:])
        second_dimension = min(second_span.shape[-2:])
        if first_dimension != second_dimension:
            raise ValueError(
                "Grassmann distance requires equal-dimensional subspaces; "
                "pass an explicit shared rank"
            )
        rank = first_dimension

    first_basis = orthonormal_subspace_basis(first_span, rank=rank)
    second_basis = orthonormal_subspace_basis(second_span, rank=rank)
    cross_gram = first_basis.transpose(-1, -2) @ second_basis
    squared_cosines = cross_gram.square().sum(dim=(-2, -1))
    loss = (1.0 - squared_cosines / rank).clamp(min=0.0, max=1.0)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()
