from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import nn

from .cotangent_jacobian_ports import grassmannian_loss
from .pixel_direct_model import DirectPixelTransformer


TensorResponse = Callable[[torch.Tensor], torch.Tensor]
LatentStepper = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


__all__ = [
    "ActivationWriteFieldConfig",
    "StateConditionedActivationWriteField",
    "MultiHorizonResponse",
    "PixelInterventionProxies",
    "FrozenSoftPixelActivationLens",
    "direct_dynamics_pulse_responses",
    "basis_invariant_response_loss",
    "grassmann_response_loss",
    "odd_symmetry_loss",
    "differentiable_attention_backend",
]


def _validated_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    result = tuple(sorted(set(int(horizon) for horizon in horizons)))
    if not result or result[0] < 1:
        raise ValueError("horizons must contain positive integers")
    return result


def _response_vector(value: torch.Tensor, response: TensorResponse | None) -> torch.Tensor:
    result = value if response is None else response(value)
    if result.ndim < 2 or result.shape[0] != value.shape[0]:
        raise ValueError("a response must preserve the leading batch dimension")
    return result.reshape(result.shape[0], -1)


@contextmanager
def differentiable_attention_backend(reference: torch.Tensor):
    """Use the differentiable math attention kernel for higher derivatives."""

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.MATH]):
            yield
    except (ImportError, AttributeError):  # pragma: no cover - old Torch fallback
        context = (
            torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            )
            if reference.device.type == "cuda"
            else nullcontext()
        )
        with context:
            yield


@dataclass(frozen=True)
class ActivationWriteFieldConfig:
    """Shape and capacity of a latent-conditioned residual-stream chart.

    ``history_frames * patch_count * hidden_size`` is the ambient activation
    dimension.  The field returns a rank-``port_size`` orthonormal basis in
    that full space.  It does not select an object, patch, time, or channel.
    """

    latent_size: int
    port_size: int
    history_frames: int
    patch_count: int
    hidden_size: int
    network_hidden_size: int = 128
    network_layers: int = 2

    @property
    def ambient_size(self) -> int:
        return self.history_frames * self.patch_count * self.hidden_size

    def __post_init__(self) -> None:
        positive = (
            self.latent_size,
            self.port_size,
            self.history_frames,
            self.patch_count,
            self.hidden_size,
            self.network_hidden_size,
            self.network_layers,
        )
        if any(value < 1 for value in positive):
            raise ValueError("all write-field dimensions must be positive")
        if self.port_size > self.ambient_size:
            raise ValueError("port_size cannot exceed the activation dimension")


class StateConditionedActivationWriteField(nn.Module):
    r"""Learn ``U(x)`` directly in the complete residual stream.

    Columns of the returned tensor are orthonormal after flattening its time,
    patch, and channel axes.  QR fixes scale but intentionally leaves an
    orthogonal port-coordinate gauge; downstream comparisons therefore use
    basis-invariant losses.  ``x`` is a learned latent coordinate only.  The
    module has no simulator-label interface.
    """

    def __init__(self, config: ActivationWriteFieldConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        current_size = config.latent_size
        for _ in range(config.network_layers):
            layers.extend(
                (
                    nn.Linear(current_size, config.network_hidden_size),
                    nn.Tanh(),
                )
            )
            current_size = config.network_hidden_size
        final = nn.Linear(
            current_size,
            config.ambient_size * config.port_size,
        )
        nn.init.normal_(final.weight, std=0.02)
        nn.init.normal_(final.bias, std=0.02)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def forward(self, latent_coordinates: torch.Tensor) -> torch.Tensor:
        if latent_coordinates.shape[-1] != self.config.latent_size:
            raise ValueError(
                f"expected latent dimension {self.config.latent_size}, "
                f"got {latent_coordinates.shape[-1]}"
            )
        raw = self.network(latent_coordinates).reshape(
            *latent_coordinates.shape[:-1],
            self.config.ambient_size,
            self.config.port_size,
        )
        basis, _ = torch.linalg.qr(raw, mode="reduced")
        return basis.reshape(
            *latent_coordinates.shape[:-1],
            self.config.history_frames,
            self.config.patch_count,
            self.config.hidden_size,
            self.config.port_size,
        )


@dataclass(frozen=True)
class MultiHorizonResponse:
    """Unperturbed responses and pulse Jacobians indexed by horizon."""

    baseline: dict[int, torch.Tensor]
    jacobians: dict[int, torch.Tensor]


@dataclass(frozen=True)
class PixelInterventionProxies:
    """Pixels-only local-chart diagnostics.

    These quantities are regularizers and audits, not evidence that the
    learned chart is a physical actuator.  In particular, the cycle term can
    also contain autonomous model drift and soft-rollout error.
    """

    odd_symmetry: torch.Tensor
    current_frame_leakage: torch.Tensor
    manifold_cycle: torch.Tensor
    first_order_signal: torch.Tensor


class FrozenSoftPixelActivationLens(nn.Module):
    r"""Differentiable residual-stream lens through a frozen video model.

    Autoregressive predictions feed back the expected class embedding
    ``sum_c p(c) embedding(c)``.  Thus horizons beyond one remain
    differentiable and use the frozen transformer's actual layers; no
    argmax, learned rollout surrogate, entity mask, or simulator variable is
    involved.  The soft feedback is a relaxation of categorical generation,
    so long-horizon results must still be checked against hard video rollout.
    """

    def __init__(
        self,
        backbone: DirectPixelTransformer,
        *,
        intervention_block: int,
        horizons: Iterable[int] = (1, 2, 4),
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0 <= intervention_block < len(backbone.blocks):
            raise ValueError("intervention_block is outside the transformer")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.backbone = backbone.eval().requires_grad_(False)
        self.intervention_block = intervention_block
        self.horizons = _validated_horizons(horizons)
        self.temperature = float(temperature)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @property
    def activation_shape(self) -> tuple[int, int, int]:
        config = self.backbone.config
        return (
            config.history_frames,
            config.grid_size**2,
            config.hidden_size,
        )

    def pixel_probabilities(self, pixel_context: torch.Tensor) -> torch.Tensor:
        """Convert hard palette indices or validate soft pixel distributions."""

        config = self.backbone.config
        hard_tail = (
            config.history_frames,
            config.image_size,
            config.image_size,
        )
        soft_tail = (
            config.history_frames,
            config.palette_size,
            config.image_size,
            config.image_size,
        )
        if pixel_context.ndim == 4 and tuple(pixel_context.shape[1:]) == hard_tail:
            return torch.nn.functional.one_hot(
                pixel_context.long(),
                num_classes=config.palette_size,
            ).permute(0, 1, 4, 2, 3).to(self.backbone.pixel_embedding.weight.dtype)
        if pixel_context.ndim == 5 and tuple(pixel_context.shape[1:]) == soft_tail:
            if not pixel_context.is_floating_point():
                raise TypeError("soft pixel distributions must be floating point")
            return pixel_context
        raise ValueError(
            f"expected hard pixel contexts [batch, {hard_tail}] or soft contexts "
            f"[batch, {soft_tail}]"
        )

    def _expected_patch_tokens(self, probabilities: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        config = model.config
        embedded = torch.einsum(
            "btchw,ce->bthwe",
            probabilities,
            model.pixel_embedding.weight,
        )
        patches = embedded.reshape(
            probabilities.shape[0],
            config.history_frames,
            config.grid_size,
            config.patch_size,
            config.grid_size,
            config.patch_size,
            config.pixel_embedding_size,
        ).permute(0, 1, 2, 4, 3, 5, 6)
        return patches.reshape(
            probabilities.shape[0],
            config.history_frames,
            config.grid_size**2,
            -1,
        )

    def soft_prefix_activation(self, pixel_context: torch.Tensor) -> torch.Tensor:
        """Return the exact soft residual stream at the intervention block.

        This is the soft-input counterpart of
        :meth:`WholeStreamFrozenEncoder.prefix_activation`.  It is exposed so
        a post-freeze, action-free planner can both condition a registered
        write field and reuse the same prefix for its intervened prediction.
        No learned surrogate or physical channel is introduced.
        """

        probabilities = self.pixel_probabilities(pixel_context)
        model = self.backbone
        tokens = (
            model.patch_projection(self._expected_patch_tokens(probabilities))
            + model.spatial_position
            + model.temporal_position
        )
        with differentiable_attention_backend(tokens):
            for block_index, block in enumerate(model.blocks):
                tokens = block(tokens)
                if block_index == self.intervention_block:
                    break
        return tokens

    def soft_logits_from_prefix(
        self,
        prefix_activation: torch.Tensor,
        *,
        residual_write: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Continue the exact frozen transformer from its lens activation."""

        expected = (prefix_activation.shape[0], *self.activation_shape)
        if tuple(prefix_activation.shape) != expected:
            raise ValueError(f"expected prefix_activation shape {expected}")
        tokens = prefix_activation
        if residual_write is not None:
            if tuple(residual_write.shape) != expected:
                raise ValueError(f"expected residual_write shape {expected}")
            tokens = tokens + residual_write
        with differentiable_attention_backend(tokens):
            for block_index in range(self.intervention_block + 1, len(self.backbone.blocks)):
                tokens = self.backbone.blocks[block_index](tokens)
            return self.backbone.unpatch_logits(tokens)

    def soft_forward(
        self,
        pixel_context: torch.Tensor,
        *,
        residual_write: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Run the exact frozen transformer layers on expected embeddings."""

        _, tokens = self._soft_suffix_tokens(
            pixel_context, residual_write=residual_write
        )
        logits = self.backbone.unpatch_logits(tokens)
        if return_logits:
            return logits
        return torch.softmax(logits / self.temperature, dim=2)

    def _soft_suffix_tokens(
        self,
        pixel_context: torch.Tensor,
        *,
        residual_write: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return soft input probabilities and exact post-backbone tokens."""

        probabilities = self.pixel_probabilities(pixel_context)
        model = self.backbone
        tokens = (
            model.patch_projection(self._expected_patch_tokens(probabilities))
            + model.spatial_position
            + model.temporal_position
        )
        if residual_write is not None:
            expected = (probabilities.shape[0], *self.activation_shape)
            if tuple(residual_write.shape) != expected:
                raise ValueError(f"expected residual_write shape {expected}")
        with differentiable_attention_backend(tokens):
            for block_index, block in enumerate(model.blocks):
                tokens = block(tokens)
                if residual_write is not None and block_index == self.intervention_block:
                    tokens = tokens + residual_write
        return probabilities, tokens

    def state_rollout(
        self,
        pixel_context: torch.Tensor,
        write_basis: torch.Tensor,
        latent_pulse: torch.Tensor,
        state_readout: TensorResponse,
        *,
        horizons: Iterable[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        r"""Re-encode soft frozen-rollout contexts at selected horizons.

        At transition ``h+1`` the frozen transformer consumes exactly the
        context produced after transition ``h``.  Its post-backbone tokens are
        therefore the same tokens that the registered encoder would obtain by
        re-encoding that context.  Reading them before producing the next
        frame gives ``E(frozen_rollout_h(A))`` and needs only one final extra
        transformer evaluation, irrespective of the number of horizons.
        """

        selected = self.horizons if horizons is None else _validated_horizons(horizons)
        current = self.pixel_probabilities(pixel_context)
        write = self.residual_write(write_basis, latent_pulse)
        responses: dict[int, torch.Tensor] = {}
        for transition in range(1, selected[-1] + 2):
            _, tokens = self._soft_suffix_tokens(
                current,
                residual_write=write if transition == 1 else None,
            )
            completed_horizon = transition - 1
            if completed_horizon in selected:
                responses[completed_horizon] = state_readout(tokens)
            if transition <= selected[-1]:
                probabilities = torch.softmax(
                    self.backbone.unpatch_logits(tokens) / self.temperature,
                    dim=2,
                )
                next_pixels = probabilities[:, -1]
                current = torch.cat((current[:, 1:], next_pixels[:, None]), dim=1)
        return responses

    def state_response_jacobians(
        self,
        pixel_context: torch.Tensor,
        write_basis: torch.Tensor,
        state_readout: TensorResponse,
        *,
        horizons: Iterable[int] | None = None,
        create_graph: bool = True,
    ) -> MultiHorizonResponse:
        r"""Compute the registered ``d E(frozen_rollout_h) / d pulse``."""

        selected = self.horizons if horizons is None else _validated_horizons(horizons)
        batch, port_size = write_basis.shape[0], write_basis.shape[-1]
        expanded_context = pixel_context[None].expand(
            port_size, *pixel_context.shape
        ).reshape(port_size * batch, *pixel_context.shape[1:])
        expanded_basis = write_basis[None].expand(
            port_size, *write_basis.shape
        ).reshape(port_size * batch, *write_basis.shape[1:])
        zero = write_basis.new_zeros(port_size * batch, port_size)
        tangent = torch.eye(
            port_size, dtype=write_basis.dtype, device=write_basis.device
        )[:, None, :].expand(port_size, batch, port_size).reshape_as(zero)

        def response_tuple(pulse: torch.Tensor) -> tuple[torch.Tensor, ...]:
            rollout = self.state_rollout(
                expanded_context,
                expanded_basis,
                pulse,
                state_readout,
                horizons=selected,
            )
            return tuple(_response_vector(rollout[horizon], None) for horizon in selected)

        expanded_baseline, expanded_directional = torch.autograd.functional.jvp(
            response_tuple,
            zero,
            tangent,
            create_graph=create_graph,
            strict=False,
        )
        baselines = tuple(
            value.reshape(port_size, batch, -1)[0]
            for value in expanded_baseline
        )
        jacobians = tuple(
            value.reshape(port_size, batch, -1).permute(1, 2, 0)
            for value in expanded_directional
        )
        return MultiHorizonResponse(
            baseline=dict(zip(selected, baselines, strict=True)),
            jacobians=dict(zip(selected, jacobians, strict=True)),
        )

    @staticmethod
    def residual_write(
        write_basis: torch.Tensor,
        latent_pulse: torch.Tensor,
    ) -> torch.Tensor:
        if write_basis.ndim != 5:
            raise ValueError("write_basis must have shape [batch, time, patch, hidden, port]")
        if latent_pulse.shape != (write_basis.shape[0], write_basis.shape[-1]):
            raise ValueError("latent_pulse must match the write-basis batch and port dimensions")
        return torch.einsum("btphm,bm->btph", write_basis, latent_pulse)

    def rollout(
        self,
        pixel_context: torch.Tensor,
        write_basis: torch.Tensor,
        latent_pulse: torch.Tensor,
        *,
        horizons: Iterable[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Apply one residual pulse, then unroll the soft frozen predictor."""

        selected = self.horizons if horizons is None else _validated_horizons(horizons)
        current = self.pixel_probabilities(pixel_context)
        write = self.residual_write(write_basis, latent_pulse)
        predictions: dict[int, torch.Tensor] = {}
        for step in range(1, selected[-1] + 1):
            all_times = self.soft_forward(
                current,
                residual_write=write if step == 1 else None,
            )
            next_pixels = all_times[:, -1]
            current = torch.cat((current[:, 1:], next_pixels[:, None]), dim=1)
            if step in selected:
                predictions[step] = next_pixels
        return predictions

    def response_jacobians(
        self,
        pixel_context: torch.Tensor,
        write_basis: torch.Tensor,
        *,
        horizons: Iterable[int] | None = None,
        response: TensorResponse | None = None,
        create_graph: bool = True,
    ) -> MultiHorizonResponse:
        r"""Compute ``K_h U = d response_h / d pulse`` by autodiff JVPs."""

        selected = self.horizons if horizons is None else _validated_horizons(horizons)
        batch, port_size = write_basis.shape[0], write_basis.shape[-1]

        # Every example is independent along the leading batch dimension.
        # Stack the m port tangents there and evaluate one batched JVP instead
        # of m sequential JVPs (each of which used to recompute the primal
        # rollout).  This is the exact same block-diagonal Jacobian product;
        # it merely exposes the independent columns to the GPU together.
        expanded_context = pixel_context[None].expand(
            port_size, *pixel_context.shape
        ).reshape(port_size * batch, *pixel_context.shape[1:])
        expanded_basis = write_basis[None].expand(
            port_size, *write_basis.shape
        ).reshape(port_size * batch, *write_basis.shape[1:])
        zero = write_basis.new_zeros(port_size * batch, port_size)
        tangent = torch.eye(
            port_size, dtype=write_basis.dtype, device=write_basis.device
        )[:, None, :].expand(port_size, batch, port_size).reshape_as(zero)

        def response_tuple(pulse: torch.Tensor) -> tuple[torch.Tensor, ...]:
            rollout = self.rollout(
                expanded_context,
                expanded_basis,
                pulse,
                horizons=selected,
            )
            return tuple(_response_vector(rollout[horizon], response) for horizon in selected)

        expanded_baseline, expanded_directional = torch.autograd.functional.jvp(
            response_tuple,
            zero,
            tangent,
            create_graph=create_graph,
            strict=False,
        )
        baselines = tuple(
            value.reshape(port_size, batch, -1)[0]
            for value in expanded_baseline
        )
        jacobians = tuple(
            value.reshape(port_size, batch, -1).permute(1, 2, 0)
            for value in expanded_directional
        )
        return MultiHorizonResponse(
            baseline=dict(zip(selected, baselines, strict=True)),
            jacobians=dict(zip(selected, jacobians, strict=True)),
        )

    def _scheduled_rollout(
        self,
        pixel_context: torch.Tensor,
        write_basis: torch.Tensor,
        pulse_schedule: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        current = self.pixel_probabilities(pixel_context)
        next_pixels = current[:, -1]
        for pulse in pulse_schedule:
            all_times = self.soft_forward(
                current,
                residual_write=self.residual_write(write_basis, pulse),
            )
            next_pixels = all_times[:, -1]
            current = torch.cat((current[:, 1:], next_pixels[:, None]), dim=1)
        return next_pixels

    def intervention_proxies(
        self,
        pixel_context: torch.Tensor,
        write_basis: torch.Tensor,
        *,
        amplitude: float = 0.05,
        eps: float = 1e-8,
    ) -> PixelInterventionProxies:
        """Measure local oddness, history leakage, and a soft pixel cycle."""

        if amplitude <= 0.0:
            raise ValueError("amplitude must be positive")
        probabilities = self.pixel_probabilities(pixel_context)
        batch, port_size = probabilities.shape[0], write_basis.shape[-1]
        if write_basis.shape[0] != batch:
            raise ValueError("pixel_context and write_basis batches must match")

        expanded_context = probabilities[:, None].expand(
            batch,
            port_size,
            *probabilities.shape[1:],
        ).reshape(batch * port_size, *probabilities.shape[1:])
        expanded_basis = write_basis[:, None].expand(
            batch,
            port_size,
            *write_basis.shape[1:],
        ).reshape(batch * port_size, *write_basis.shape[1:])
        directions = amplitude * torch.eye(
            port_size,
            dtype=write_basis.dtype,
            device=write_basis.device,
        )[None].expand(batch, port_size, port_size).reshape(batch * port_size, port_size)

        # The zero-write prediction is identical for every port, so compute it
        # once per source example and broadcast the result.  Positive and
        # negative interventions are independent and can share one larger
        # transformer call.  No approximation or graph detachment is involved.
        baseline_unique = self.soft_forward(probabilities)
        signed_context = torch.cat((expanded_context, expanded_context), dim=0)
        signed_basis = torch.cat((expanded_basis, expanded_basis), dim=0)
        signed_directions = torch.cat((directions, -directions), dim=0)
        signed_all = self.soft_forward(
            signed_context,
            residual_write=self.residual_write(signed_basis, signed_directions),
        )
        baseline_all = baseline_unique[:, None].expand(
            batch, port_size, *baseline_unique.shape[1:]
        ).reshape(batch * port_size, *baseline_unique.shape[1:])
        positive_all, negative_all = signed_all.chunk(2, dim=0)
        baseline_terminal = baseline_all[:, -1]
        positive_terminal = positive_all[:, -1]
        negative_terminal = negative_all[:, -1]

        odd = odd_symmetry_loss(
            positive_terminal,
            negative_terminal,
            baseline_terminal,
            eps=eps,
        )
        signal = (positive_terminal - negative_terminal).square().mean()
        if baseline_all.shape[1] > 1:
            history_change = 0.5 * (
                (positive_all[:, :-1] - baseline_all[:, :-1]).square().mean()
                + (negative_all[:, :-1] - baseline_all[:, :-1]).square().mean()
            )
            terminal_change = 0.5 * (
                (positive_terminal - baseline_terminal).square().mean()
                + (negative_terminal - baseline_terminal).square().mean()
            )
            leakage = history_change / terminal_change.clamp_min(eps)
        else:  # pragma: no cover - present configs have history greater than one
            leakage = signal.new_zeros(())

        # Reuse the already computed first positive/zero steps.  The previous
        # implementation repeated both before evaluating the second step.
        positive_context = torch.cat(
            (
                expanded_context[:, 1:],
                positive_terminal[:, None],
            ),
            dim=1,
        )
        cycled = self.soft_forward(
            positive_context,
            residual_write=self.residual_write(expanded_basis, -directions),
        )[:, -1]
        baseline_context = torch.cat(
            (probabilities[:, 1:], baseline_unique[:, -1, None]), dim=1
        )
        baseline_two_unique = self.soft_forward(baseline_context)[:, -1]
        baseline_two = baseline_two_unique[:, None].expand(
            batch, port_size, *baseline_two_unique.shape[1:]
        ).reshape(batch * port_size, *baseline_two_unique.shape[1:])
        cycle = (cycled - baseline_two).square().mean() / signal.clamp_min(eps)
        return PixelInterventionProxies(
            odd_symmetry=odd,
            current_frame_leakage=leakage,
            manifold_cycle=cycle,
            first_order_signal=signal,
        )


def direct_dynamics_pulse_responses(
    stepper: LatentStepper,
    latent_coordinates: torch.Tensor,
    port_size: int,
    *,
    horizons: Iterable[int] = (1, 2, 4),
    response: TensorResponse | None = None,
    create_graph: bool = True,
) -> MultiHorizonResponse:
    r"""Differentiate a direct latent dynamics rollout after one effort pulse.

    The supplied callable is evaluated as ``stepper(x, effort)``.  A pulse is
    used only on the first transition and zero latent effort thereafter.  No
    simulator quantity is part of this interface.
    """

    if latent_coordinates.ndim != 2:
        raise ValueError("latent_coordinates must have shape [batch, latent]")
    if port_size < 1:
        raise ValueError("port_size must be positive")
    selected = _validated_horizons(horizons)
    batch = latent_coordinates.shape[0]
    expanded_coordinates = latent_coordinates[None].expand(
        port_size, *latent_coordinates.shape
    ).reshape(port_size * batch, latent_coordinates.shape[-1])
    zero = latent_coordinates.new_zeros(port_size * batch, port_size)
    tangent = torch.eye(
        port_size,
        dtype=latent_coordinates.dtype,
        device=latent_coordinates.device,
    )[:, None, :].expand(port_size, batch, port_size).reshape_as(zero)

    def response_tuple(pulse: torch.Tensor) -> tuple[torch.Tensor, ...]:
        current = expanded_coordinates
        outputs: list[torch.Tensor] = []
        for step in range(1, selected[-1] + 1):
            effort = pulse if step == 1 else torch.zeros_like(pulse)
            current = stepper(current, effort)
            if step in selected:
                outputs.append(_response_vector(current, response))
        return tuple(outputs)

    expanded_baseline, expanded_directional = torch.autograd.functional.jvp(
        response_tuple,
        zero,
        tangent,
        create_graph=create_graph,
        strict=False,
    )
    baselines = tuple(
        value.reshape(port_size, batch, -1)[0] for value in expanded_baseline
    )
    jacobians = tuple(
        value.reshape(port_size, batch, -1).permute(1, 2, 0)
        for value in expanded_directional
    )
    return MultiHorizonResponse(
        baseline=dict(zip(selected, baselines, strict=True)),
        jacobians=dict(zip(selected, jacobians, strict=True)),
    )


def grassmann_response_loss(
    first_response: torch.Tensor,
    second_response: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Basis-invariant chordal distance between response column spaces."""

    return grassmannian_loss(
        first_response,
        second_response,
        reduction=reduction,  # type: ignore[arg-type]
    )


def basis_invariant_response_loss(
    first_response: torch.Tensor,
    second_response: torch.Tensor,
    *,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Scale-normalized orthogonal-Procrustes response discrepancy.

    Each response is normalized by its Frobenius norm, then the best
    orthogonal port-basis alignment is removed analytically.  The result is
    invariant to independent orthogonal rotations of both port bases and to
    global positive response scale.  It intentionally retains anisotropy
    information that a pure Grassmann distance discards.
    """

    if first_response.shape != second_response.shape or first_response.ndim < 2:
        raise ValueError("responses must have the same shape [..., observable, port]")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    first_norm = torch.linalg.matrix_norm(first_response, ord="fro", dim=(-2, -1))
    second_norm = torch.linalg.matrix_norm(second_response, ord="fro", dim=(-2, -1))
    first = first_response / first_norm.clamp_min(eps)[..., None, None]
    second = second_response / second_norm.clamp_min(eps)[..., None, None]
    cross = first.transpose(-1, -2) @ second
    nuclear = torch.linalg.svdvals(cross).sum(dim=-1)
    loss = (2.0 - 2.0 * nuclear).clamp_min(0.0)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def odd_symmetry_loss(
    positive_response: torch.Tensor,
    negative_response: torch.Tensor,
    baseline_response: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalized even-order remainder of symmetric residual writes."""

    if not (
        positive_response.shape == negative_response.shape == baseline_response.shape
    ):
        raise ValueError("positive, negative, and baseline responses must have equal shapes")
    even_remainder = positive_response + negative_response - 2.0 * baseline_response
    first_order = positive_response - negative_response
    return even_remainder.square().mean() / first_order.square().mean().clamp_min(eps)
