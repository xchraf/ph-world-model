"""Cotangent Jacobian lens coupled directly to a Poisson latent port.

The bridge implemented here is the coordinate-correct replacement for treating
an activation gradient as an ordinary latent vector.  A visual observable
gradient is a covector.  It is pulled through ``D_h E`` into the direct state,
then converted to a generalized-force vector by the learned Poisson sharp map
``-J(x)``.  Only then is it compared with the learned port distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import Iterable, Mapping

import torch
from torch import nn

from .cotangent_jacobian_ports import (
    cotangent_pullback_solve,
    grassmannian_loss,
    poisson_sharp,
    pullback_compatibility_residual,
)
from .direct_activation_lens import (
    FrozenSoftPixelActivationLens,
    differentiable_attention_backend,
)
from .direct_visual_poisson_ph import DirectVisualPoissonPH


def _horizons(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(sorted(set(int(value) for value in values)))
    if not result or result[0] < 1:
        raise ValueError("horizons must contain positive integers")
    return result


class PixelChangeProbeBank(nn.Module):
    """Fixed pixels-only visual observables used to form true covectors.

    The basis is estimated before training from principal directions of
    one-hot frame differences.  It contains no object mask, centroid, physical
    coordinate, simulator state, or actuation label.
    """

    def __init__(self, basis: torch.Tensor) -> None:
        super().__init__()
        if basis.ndim != 4:
            raise ValueError("basis must have shape [probe,class,height,width]")
        flat = basis.flatten(1).float()
        orthonormal, _ = torch.linalg.qr(flat.T, mode="reduced")
        self.register_buffer("basis", orthonormal.T.reshape_as(basis).contiguous())

    @property
    def probe_size(self) -> int:
        return self.basis.shape[0]

    @classmethod
    def from_pixel_frames(
        cls,
        frames: torch.Tensor,
        *,
        palette_size: int,
        probe_size: int,
        maximum_differences: int = 1_024,
    ) -> "PixelChangeProbeBank":
        if frames.ndim < 4:
            raise ValueError("frames must include trajectory,time,height,width")
        if frames.shape[-3] < 2:
            raise ValueError("frames need at least two time points")
        height, width = frames.shape[-2:]
        previous = frames[..., :-1, :, :].reshape(-1, height, width)
        successor = frames[..., 1:, :, :].reshape(-1, height, width)
        sample_count = previous.shape[0]
        if sample_count > maximum_differences:
            # Subsample before any class expansion.  Expanding all 4096x8
            # registered transitions to int64 one-hot tensors would require
            # roughly ten gigabytes at 64x64x9.
            indices = torch.linspace(
                0,
                sample_count - 1,
                maximum_differences,
                device=frames.device,
            ).long()
            previous = previous[indices]
            successor = successor[indices]
        difference = torch.zeros(
            previous.shape[0],
            palette_size,
            height,
            width,
            dtype=torch.float32,
            device=frames.device,
        )
        source = torch.ones(
            previous.shape[0], 1, height, width,
            dtype=difference.dtype,
            device=difference.device,
        )
        difference.scatter_add_(1, successor.long().unsqueeze(1), source)
        difference.scatter_add_(1, previous.long().unsqueeze(1), -source)
        flat = difference.flatten(1)
        flat = flat - flat.mean(dim=0, keepdim=True)
        maximum_rank = min(flat.shape)
        if probe_size > maximum_rank:
            raise ValueError("probe_size exceeds the visual-change sample rank")
        # Randomized PCA avoids an intractable dense SVD of the 36k-dimensional
        # 64x64 palette space.  Its seed is inherited from the sealed single
        # seed, and only pixels enter the multiplication.
        approximation_rank = min(
            maximum_rank, max(probe_size + 4, 2 * probe_size)
        )
        _, _, right = torch.pca_lowrank(
            flat,
            q=approximation_rank,
            center=False,
            niter=4,
        )
        basis = right[:, :probe_size].T.reshape(
            probe_size, palette_size, frames.shape[-2], frames.shape[-1]
        )
        return cls(basis)

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        if probabilities.shape[-3:] != self.basis.shape[-3:]:
            raise ValueError("pixel probabilities do not match the sealed probe basis")
        return torch.einsum("bchw,mchw->bm", probabilities.float(), self.basis)


@dataclass(frozen=True)
class CotangentPoissonBridgeResult:
    total: torch.Tensor
    port_alignment: torch.Tensor
    pullback_compatibility: torch.Tensor
    horizon_consistency: torch.Tensor
    port_isotropy: torch.Tensor
    tangent_pushforward_alignment: torch.Tensor
    state: torch.Tensor
    activation_covectors: dict[int, torch.Tensor]
    state_covectors: dict[int, torch.Tensor]
    poisson_port_priors: dict[int, torch.Tensor]
    state_tangent_port: torch.Tensor


def activation_observable_covectors(
    lens: FrozenSoftPixelActivationLens,
    pixel_context: torch.Tensor,
    probes: PixelChangeProbeBank,
    *,
    horizons: Iterable[int] = (1, 2, 4),
    create_graph: bool = False,
) -> dict[int, torch.Tensor]:
    """Differentiate future pixels with respect to the full residual stream."""

    selected = _horizons(horizons)
    batch = pixel_context.shape[0]
    write = torch.zeros(
        batch,
        *lens.activation_shape,
        device=pixel_context.device,
        dtype=lens.backbone.pixel_embedding.weight.dtype,
        requires_grad=True,
    )
    current = lens.pixel_probabilities(pixel_context)
    predictions: dict[int, torch.Tensor] = {}
    for step in range(1, selected[-1] + 1):
        output = lens.soft_forward(
            current,
            residual_write=write if step == 1 else None,
        )
        next_pixels = output[:, -1]
        current = torch.cat((current[:, 1:], next_pixels[:, None]), dim=1)
        if step in selected:
            predictions[step] = next_pixels

    # Vectorize the independent horizon/probe VJPs into one autograd call.
    # Each seed still sums one visual observable over the data batch exactly
    # as the former nested loop did; ``is_grads_batched`` only evaluates those
    # seed vectors together.
    observables = torch.stack(
        tuple(probes(predictions[horizon]) for horizon in selected), dim=1
    )
    seed_count = len(selected) * probes.probe_size
    seeds = torch.eye(
        seed_count, dtype=observables.dtype, device=observables.device
    ).reshape(seed_count, 1, len(selected), probes.probe_size)
    seeds = seeds.expand(seed_count, batch, len(selected), probes.probe_size)
    gradients = torch.autograd.grad(
        observables,
        write,
        grad_outputs=seeds,
        create_graph=create_graph,
        retain_graph=create_graph,
        is_grads_batched=True,
    )[0]
    gradients = gradients.reshape(
        len(selected),
        probes.probe_size,
        batch,
        -1,
    ).permute(0, 2, 3, 1)
    return {
        horizon: gradients[index]
        for index, horizon in enumerate(selected)
    }


def cotangent_poisson_bridge(
    model: DirectVisualPoissonPH,
    lens: FrozenSoftPixelActivationLens,
    probes: PixelChangeProbeBank,
    pixel_context: torch.Tensor,
    *,
    horizons: Iterable[int] = (1, 2, 4),
    ridge: float = 1e-4,
    target_batch_permutation: torch.Tensor | None = None,
    activation_covectors: Mapping[int, torch.Tensor] | None = None,
    prefix_activation: torch.Tensor | None = None,
    extracted_write_basis: torch.Tensor | None = None,
) -> CotangentPoissonBridgeResult:
    """Pull visual covectors into state and align ``B`` after ``-J``.

    ``target_batch_permutation`` is used only by the registered shuffled-lens
    ablation.  It is required to be a fixed-point-free permutation and is
    applied *after* every visual Jacobian target has been computed.  Thus the
    ablation destroys state-to-Jacobian correspondence without changing the
    target values, losses, model capacity, or frozen-lens computation.
    """

    if model.core.config.port_size != probes.probe_size:
        raise ValueError("the probe rank must equal the latent port rank")
    if lens.backbone is not model.encoder.backbone:
        raise ValueError("lens and encoder must share the exact frozen backbone")
    if lens.intervention_block != model.encoder.config.lens_block:
        raise ValueError("lens and encoder intervention blocks differ")
    selected = _horizons(horizons)
    batch_size = pixel_context.shape[0]
    if target_batch_permutation is not None:
        expected = torch.arange(batch_size, device=pixel_context.device)
        if (
            target_batch_permutation.dtype != torch.long
            or target_batch_permutation.device != pixel_context.device
            or tuple(target_batch_permutation.shape) != (batch_size,)
        ):
            raise ValueError(
                "target_batch_permutation must be a length-batch int64 "
                "tensor on the pixel-context device"
            )
        if not torch.equal(target_batch_permutation.sort().values, expected):
            raise ValueError("target_batch_permutation must be a permutation")
        if bool((target_batch_permutation == expected).any()):
            raise ValueError(
                "target_batch_permutation must have no fixed point"
            )

    if prefix_activation is None:
        activation = model.encoder.prefix_activation(pixel_context)
    else:
        expected_activation = (pixel_context.shape[0], *lens.activation_shape)
        if (
            type(prefix_activation) is not torch.Tensor
            or tuple(prefix_activation.shape) != expected_activation
            or prefix_activation.requires_grad
            or prefix_activation.grad_fn is not None
        ):
            raise ValueError(
                "prefix_activation must be a detached activation of this pixel batch"
            )
        activation = prefix_activation
    activation = activation.detach().requires_grad_(True)
    # The bridge differentiates once to build D_h E and again when its loss is
    # optimized.  Flash attention does not expose that double backward on all
    # Torch backends, so this forward is deliberately executed by the exact
    # differentiable math kernel.
    with differentiable_attention_backend(activation):
        state, adapter_jacobian = model.encoder.state_and_jacobian_from_activation(
            activation, create_graph=True
        )
    # The frozen visual covectors are targets.  Detaching avoids needless
    # third derivatives through a backbone that has no trainable tensor.
    if activation_covectors is None:
        frozen_covectors = activation_observable_covectors(
            lens,
            pixel_context,
            probes,
            horizons=selected,
            create_graph=False,
        )
    else:
        if type(activation_covectors) is not dict or tuple(
            sorted(activation_covectors)
        ) != selected:
            raise ValueError(
                "precomputed activation covectors must contain exactly the horizons"
            )
        expected_covector_shape = (
            pixel_context.shape[0],
            math.prod(lens.activation_shape),
            probes.probe_size,
        )
        for horizon, covectors in activation_covectors.items():
            if (
                type(covectors) is not torch.Tensor
                or tuple(covectors.shape) != expected_covector_shape
                or covectors.requires_grad
                or covectors.grad_fn is not None
                or not bool(torch.isfinite(covectors).all())
            ):
                raise ValueError(
                    f"precomputed activation covectors for horizon {horizon} "
                    "are malformed or attached"
                )
        frozen_covectors = activation_covectors
    activation_covectors = {
        horizon: covector.detach()
        for horizon, covector in frozen_covectors.items()
    }
    interconnection = model.core.interconnection(state.float())
    learned_port = model.core.port(state.float())
    expected_write_shape = (pixel_context.shape[0], *lens.activation_shape, probes.probe_size)
    if (
        type(extracted_write_basis) is not torch.Tensor
        or tuple(extracted_write_basis.shape) != expected_write_shape
        or extracted_write_basis.requires_grad
        or extracted_write_basis.grad_fn is not None
        or not bool(torch.isfinite(extracted_write_basis).all())
    ):
        raise ValueError(
            "extracted_write_basis must be the detached exact Jacobian port"
        )
    state_tangent_port = torch.einsum(
        "bna,bam->bnm",
        adapter_jacobian.float(),
        extracted_write_basis.flatten(1, 3).float(),
    ) / model.core.config.dt
    tangent_pushforward_alignment = grassmannian_loss(
        learned_port, state_tangent_port
    )
    state_covectors: dict[int, torch.Tensor] = {}
    poisson_priors: dict[int, torch.Tensor] = {}
    alignments = []
    compatibilities = []
    isotropies = []
    for horizon in selected:
        paired_state_covector = cotangent_pullback_solve(
            adapter_jacobian.float(),
            activation_covectors[horizon].float(),
            ridge=ridge,
        )
        paired_prior = poisson_sharp(interconnection, paired_state_covector)
        if target_batch_permutation is None:
            state_covector = paired_state_covector
            prior = paired_prior
        else:
            state_covector = paired_state_covector[target_batch_permutation]
            prior = paired_prior[target_batch_permutation]
        state_covectors[horizon] = state_covector
        poisson_priors[horizon] = prior
        alignments.append(grassmannian_loss(learned_port, prior))
        compatibility = pullback_compatibility_residual(
            adapter_jacobian.float(),
            paired_state_covector,
            activation_covectors[horizon].float(),
        )
        compatibilities.append(compatibility.square().mean())
        # A configuration-covector family generating force ports is isotropic:
        # alpha^T B should vanish.  This is learned, never assigned to q/p axes.
        isotropies.append(
            torch.einsum("...im,...in->...mn", state_covector, learned_port)
            .square()
            .mean()
        )
    consistency_terms = []
    for first, second in zip(selected[:-1], selected[1:]):
        consistency_terms.append(
            grassmannian_loss(poisson_priors[first], poisson_priors[second])
        )
    zero = state.new_zeros(())
    alignment = torch.stack(alignments).mean()
    compatibility = torch.stack(compatibilities).mean()
    consistency = (
        torch.stack(consistency_terms).mean() if consistency_terms else zero
    )
    isotropy = torch.stack(isotropies).mean()
    total = (
        alignment
        + tangent_pushforward_alignment
        + 0.25 * compatibility
        + 0.25 * consistency
        + 0.10 * isotropy
    )
    return CotangentPoissonBridgeResult(
        total=total,
        port_alignment=alignment,
        pullback_compatibility=compatibility,
        horizon_consistency=consistency,
        port_isotropy=isotropy,
        tangent_pushforward_alignment=tangent_pushforward_alignment,
        state=state,
        activation_covectors=activation_covectors,
        state_covectors=state_covectors,
        poisson_port_priors=poisson_priors,
        state_tangent_port=state_tangent_port,
    )


def assert_no_physical_input_api() -> None:
    for function in (
        activation_observable_covectors,
        cotangent_poisson_bridge,
        PixelChangeProbeBank.forward,
    ):
        names = tuple(inspect.signature(function).parameters)
        for forbidden in ("action", "control", "force", "torque", "state_label"):
            if any(forbidden in name.lower() for name in names):
                raise AssertionError(f"{function.__qualname__} exposes {forbidden!r}")


__all__ = [
    "CotangentPoissonBridgeResult",
    "PixelChangeProbeBank",
    "activation_observable_covectors",
    "assert_no_physical_input_api",
    "cotangent_poisson_bridge",
]
