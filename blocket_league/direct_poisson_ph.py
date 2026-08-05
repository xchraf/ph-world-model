"""Direct latent port-Hamiltonian dynamics with a genuine Poisson tensor.

This module deliberately contains no physical-action interface.  Its only
exogenous variable is ``latent_effort``.  Experiment F infers that variable
from consecutive visual latent states with
``action_free_latent_effort.LatentEffortInference``; the optional Gaussian
:class:`LatentEffortEncoder` below is a standalone utility and is not wired
into the registered experiment.

The interconnection tensor is not obtained by merely antisymmetrising a neural
matrix.  A learned, exactly invertible affine-coupling diffeomorphism pushes a
constant canonical Poisson tensor forward.  The canonical tensor may be
degenerate, so odd-dimensional states and learned Casimir coordinates are
supported.  Consequently the resulting state-dependent tensor is Poisson
(skew and Jacobi) by construction.

Time stepping uses a Gonzalez discrete gradient and an unrolled implicit
fixed-point solve.  The discrete chain rule is algebraically exact away from
the explicitly reported tiny-step fallback.  A finite number of fixed-point
iterations is still a numerical approximation, so ``audited_step`` exposes
both the implicit residual and the resulting energy-balance defect.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DirectPoissonPHConfig:
    """Configuration for a direct Poisson port-Hamiltonian latent core."""

    state_size: int
    port_size: int
    poisson_rank: int | None = None
    hidden_size: int = 64
    hidden_layers: int = 2
    coupling_layers: int = 4
    coupling_scale_limit: float = 0.35
    resistance_floor: float = 0.0
    dt: float = 0.05
    implicit_iterations: int = 32
    implicit_relaxation: float = 0.8
    implicit_tolerance: float = 1e-10
    discrete_gradient_epsilon: float = 1e-14

    def __post_init__(self) -> None:
        if self.state_size < 2:
            raise ValueError("state_size must be at least two")
        resolved_rank = (
            self.state_size - self.state_size % 2
            if self.poisson_rank is None
            else self.poisson_rank
        )
        if (
            type(resolved_rank) is not int
            or resolved_rank < 0
            or resolved_rank > self.state_size
            or resolved_rank % 2
        ):
            raise ValueError(
                "poisson_rank must be an even integer between zero and state_size"
            )
        object.__setattr__(self, "poisson_rank", resolved_rank)
        if self.port_size < 1:
            raise ValueError("port_size must be positive")
        if self.port_size > self.state_size:
            raise ValueError("port_size cannot exceed state_size for a full-rank port")
        if self.hidden_size < 1 or self.hidden_layers < 1:
            raise ValueError("the neural functions need positive hidden dimensions")
        if self.coupling_layers < 2:
            raise ValueError("at least two alternating coupling layers are required")
        if self.coupling_scale_limit <= 0.0:
            raise ValueError("coupling_scale_limit must be positive")
        if self.resistance_floor < 0.0:
            raise ValueError("resistance_floor must be non-negative")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.implicit_iterations < 1:
            raise ValueError("implicit_iterations must be positive")
        if not 0.0 < self.implicit_relaxation <= 1.0:
            raise ValueError("implicit_relaxation must lie in (0, 1]")
        if self.implicit_tolerance < 0.0:
            raise ValueError("implicit_tolerance must be non-negative")
        if self.discrete_gradient_epsilon <= 0.0:
            raise ValueError("discrete_gradient_epsilon must be positive")


@dataclass(frozen=True)
class LatentEffortEncoderConfig:
    """Feature-only posterior used to infer the unobserved latent effort."""

    feature_size: int
    port_size: int
    hidden_size: int = 128
    hidden_layers: int = 2
    minimum_log_scale: float = -7.0
    maximum_log_scale: float = 2.0

    def __post_init__(self) -> None:
        if self.feature_size < 1 or self.port_size < 1:
            raise ValueError("feature_size and port_size must be positive")
        if self.hidden_size < 1 or self.hidden_layers < 1:
            raise ValueError("the encoder needs positive hidden dimensions")
        if self.minimum_log_scale >= self.maximum_log_scale:
            raise ValueError("minimum_log_scale must be below maximum_log_scale")


@dataclass(frozen=True)
class LatentEffortPosterior:
    """Diagonal Gaussian posterior over a transition's latent effort."""

    mean: torch.Tensor
    log_scale: torch.Tensor

    def rsample(self, sample_noise: torch.Tensor | None = None) -> torch.Tensor:
        if sample_noise is None:
            sample_noise = torch.randn_like(self.mean)
        if sample_noise.shape != self.mean.shape:
            raise ValueError("sample_noise must have the posterior mean's shape")
        return self.mean + self.log_scale.exp() * sample_noise

    def standard_normal_kl(self) -> torch.Tensor:
        """KL per example, leaving all leading batch dimensions intact."""

        variance = (2.0 * self.log_scale).exp()
        return 0.5 * (self.mean.square() + variance - 1.0 - 2.0 * self.log_scale).sum(-1)


@dataclass(frozen=True)
class DiscreteStepResult:
    """A latent step together with every term in its discrete power audit."""

    next_state: torch.Tensor
    energy_before: torch.Tensor
    energy_after: torch.Tensor
    energy_delta: torch.Tensor
    dissipated_energy: torch.Tensor
    supplied_energy: torch.Tensor
    balance_defect: torch.Tensor
    chain_rule_defect: torch.Tensor
    implicit_residual: torch.Tensor
    implicit_residual_norm: torch.Tensor
    latent_output: torch.Tensor


class _SmoothMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        hidden_layers: int,
        *,
        final_scale: float,
        final_bias: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_size = input_size
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(current_size, hidden_size), nn.Tanh()))
            current_size = hidden_size
        final = nn.Linear(current_size, output_size)
        nn.init.normal_(final.weight, mean=0.0, std=final_scale)
        nn.init.constant_(final.bias, final_bias)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)

    def value_and_jacobian(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the MLP and its exact input Jacobian in one pass.

        The networks used by the pH core contain only affine maps and
        elementwise ``tanh`` nonlinearities.  Propagating their Jacobian by
        the chain rule is algebraically identical to ``jacrev(self.forward)``
        but avoids constructing a nested autodiff transform at every implicit
        solver iteration.  All operations below remain in the ordinary
        autograd graph, so gradients with respect to both inputs and network
        parameters are still exact.
        """

        input_size = values.shape[-1]
        identity = torch.eye(
            input_size, dtype=values.dtype, device=values.device
        )
        jacobian = identity.expand(*values.shape[:-1], input_size, input_size)
        output = values
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                output = layer(output)
                jacobian = torch.einsum(
                    "oi,...ij->...oj", layer.weight, jacobian
                )
            elif isinstance(layer, nn.Tanh):
                output = torch.tanh(output)
                jacobian = (1.0 - output.square())[..., :, None] * jacobian
            else:  # pragma: no cover - guarded by this module's constructor
                raise TypeError(
                    "exact MLP Jacobian only supports Linear and Tanh layers"
                )
        return output, jacobian


class _AffineCouplingLayer(nn.Module):
    """A smooth RealNVP-style coupling layer with an exact inverse."""

    def __init__(
        self,
        state_size: int,
        condition_indices: torch.Tensor,
        hidden_size: int,
        hidden_layers: int,
        scale_limit: float,
    ) -> None:
        super().__init__()
        condition_indices = condition_indices.to(dtype=torch.long)
        all_indices = torch.arange(state_size)
        condition_mask = torch.zeros(state_size, dtype=torch.bool)
        condition_mask[condition_indices] = True
        transformed_indices = all_indices[~condition_mask]
        if condition_indices.numel() == 0 or transformed_indices.numel() == 0:
            raise ValueError("a coupling layer needs non-empty condition and transform sets")

        ordered_indices = torch.cat((condition_indices, transformed_indices))
        self.register_buffer("condition_indices", condition_indices)
        self.register_buffer("transformed_indices", transformed_indices)
        self.register_buffer("inverse_permutation", torch.argsort(ordered_indices))
        self.scale_limit = float(scale_limit)
        self.conditioner = _SmoothMLP(
            int(condition_indices.numel()),
            2 * int(transformed_indices.numel()),
            hidden_size,
            hidden_layers,
            final_scale=0.02,
        )

    def _coupling_parameters(
        self, conditioned: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_scale, shift = self.conditioner(conditioned).chunk(2, dim=-1)
        log_scale = self.scale_limit * torch.tanh(raw_scale)
        return log_scale, shift

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        conditioned = base[..., self.condition_indices]
        transformed = base[..., self.transformed_indices]
        log_scale, shift = self._coupling_parameters(conditioned)
        transformed = transformed * log_scale.exp() + shift
        ordered = torch.cat((conditioned, transformed), dim=-1)
        return ordered[..., self.inverse_permutation]

    def value_and_jacobian(
        self, base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the coupling output and its exact local Jacobian.

        This is the closed-form derivative of
        ``t -> t * exp(s(c)) + shift(c)``.  It deliberately uses regular
        differentiable tensor operations rather than detaching an analytic
        value, preserving the same higher-order gradients as ``jacrev``.
        """

        conditioned = base[..., self.condition_indices]
        transformed = base[..., self.transformed_indices]
        raw_parameters, parameter_jacobian = (
            self.conditioner.value_and_jacobian(conditioned)
        )
        transformed_size = int(self.transformed_indices.numel())
        raw_scale = raw_parameters[..., :transformed_size]
        shift = raw_parameters[..., transformed_size:]
        raw_scale_jacobian = parameter_jacobian[..., :transformed_size, :]
        shift_jacobian = parameter_jacobian[..., transformed_size:, :]

        tanh_scale = torch.tanh(raw_scale)
        log_scale = self.scale_limit * tanh_scale
        scale = log_scale.exp()
        log_scale_jacobian = (
            self.scale_limit * (1.0 - tanh_scale.square())
        )[..., :, None] * raw_scale_jacobian
        transformed_output = transformed * scale + shift
        cross_jacobian = (
            transformed * scale
        )[..., :, None] * log_scale_jacobian + shift_jacobian

        state_size = base.shape[-1]
        local_jacobian = base.new_zeros(
            *base.shape[:-1], state_size, state_size
        )
        condition_rows = self.condition_indices[:, None]
        condition_columns = self.condition_indices[None, :]
        transformed_rows = self.transformed_indices[:, None]
        transformed_columns = self.transformed_indices[None, :]
        local_jacobian[..., condition_rows, condition_columns] = torch.eye(
            int(self.condition_indices.numel()),
            dtype=base.dtype,
            device=base.device,
        )
        local_jacobian[..., transformed_rows, condition_columns] = cross_jacobian
        local_jacobian[..., transformed_rows, transformed_columns] = torch.diag_embed(
            scale
        )

        output = base.clone()
        output[..., self.transformed_indices] = transformed_output
        return output, local_jacobian

    def inverse(self, image: torch.Tensor) -> torch.Tensor:
        conditioned = image[..., self.condition_indices]
        transformed = image[..., self.transformed_indices]
        log_scale, shift = self._coupling_parameters(conditioned)
        transformed = (transformed - shift) * (-log_scale).exp()
        ordered = torch.cat((conditioned, transformed), dim=-1)
        return ordered[..., self.inverse_permutation]

    def inverse_and_forward_jacobian(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert this layer and return its forward Jacobian in one MLP pass."""

        conditioned = image[..., self.condition_indices]
        transformed_image = image[..., self.transformed_indices]
        raw_parameters, parameter_jacobian = (
            self.conditioner.value_and_jacobian(conditioned)
        )
        transformed_size = int(self.transformed_indices.numel())
        raw_scale = raw_parameters[..., :transformed_size]
        shift = raw_parameters[..., transformed_size:]
        raw_scale_jacobian = parameter_jacobian[..., :transformed_size, :]
        shift_jacobian = parameter_jacobian[..., transformed_size:, :]
        tanh_scale = torch.tanh(raw_scale)
        log_scale = self.scale_limit * tanh_scale
        scale = log_scale.exp()
        log_scale_jacobian = (
            self.scale_limit * (1.0 - tanh_scale.square())
        )[..., :, None] * raw_scale_jacobian
        transformed_base = (transformed_image - shift) * (-log_scale).exp()
        cross_jacobian = (
            transformed_base * scale
        )[..., :, None] * log_scale_jacobian + shift_jacobian

        state_size = image.shape[-1]
        local_jacobian = image.new_zeros(
            *image.shape[:-1], state_size, state_size
        )
        condition_rows = self.condition_indices[:, None]
        condition_columns = self.condition_indices[None, :]
        transformed_rows = self.transformed_indices[:, None]
        transformed_columns = self.transformed_indices[None, :]
        local_jacobian[..., condition_rows, condition_columns] = torch.eye(
            int(self.condition_indices.numel()),
            dtype=image.dtype,
            device=image.device,
        )
        local_jacobian[..., transformed_rows, condition_columns] = cross_jacobian
        local_jacobian[..., transformed_rows, transformed_columns] = torch.diag_embed(
            scale
        )
        base = image.clone()
        base[..., self.transformed_indices] = transformed_base
        return base, local_jacobian


class AffineCouplingDiffeomorphism(nn.Module):
    """A globally invertible learned chart used to transport ``J_canonical``."""

    def __init__(
        self,
        state_size: int,
        hidden_size: int,
        hidden_layers: int,
        coupling_layers: int,
        scale_limit: float,
    ) -> None:
        super().__init__()
        even = torch.arange(0, state_size, 2)
        odd = torch.arange(1, state_size, 2)
        layers = []
        for layer_index in range(coupling_layers):
            condition = even if layer_index % 2 == 0 else odd
            layers.append(
                _AffineCouplingLayer(
                    state_size,
                    condition,
                    hidden_size,
                    hidden_layers,
                    scale_limit,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.state_size = state_size

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        image = base
        for layer in self.layers:
            image = layer(image)
        return image

    def inverse(self, image: torch.Tensor) -> torch.Tensor:
        base = image
        for layer in reversed(self.layers):
            base = layer.inverse(base)
        return base

    def inverse_and_jacobian(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``phi^-1(image)`` and ``D phi`` without a second forward."""

        base = image
        identity = torch.eye(
            self.state_size, dtype=image.dtype, device=image.device
        )
        jacobian = identity.expand(
            *image.shape[:-1], self.state_size, self.state_size
        )
        for layer in reversed(self.layers):
            base, local_jacobian = layer.inverse_and_forward_jacobian(base)
            jacobian = jacobian @ local_jacobian
        return base, jacobian

    def jacobian(self, base: torch.Tensor) -> torch.Tensor:
        """Return ``D phi(base)`` and preserve higher-order autograd graphs."""

        _, jacobian = self.value_and_jacobian(base)
        return jacobian

    def value_and_jacobian(
        self, base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate ``phi`` and ``D phi`` by exact layerwise composition."""

        image = base
        identity = torch.eye(
            self.state_size, dtype=base.dtype, device=base.device
        )
        jacobian = identity.expand(
            *base.shape[:-1], self.state_size, self.state_size
        )
        for layer in self.layers:
            image, local_jacobian = layer.value_and_jacobian(image)
            jacobian = local_jacobian @ jacobian
        return image, jacobian


class LatentEffortEncoder(nn.Module):
    """Infer latent effort from visual features only.

    Neither simulator state nor physical command values are part of this API.
    The encoder observes a pair of consecutive frozen-backbone features.
    """

    def __init__(self, config: LatentEffortEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.network = _SmoothMLP(
            3 * config.feature_size,
            2 * config.port_size,
            config.hidden_size,
            config.hidden_layers,
            final_scale=0.02,
        )

    def forward(
        self,
        previous_features: torch.Tensor,
        next_features: torch.Tensor,
    ) -> LatentEffortPosterior:
        if previous_features.shape != next_features.shape:
            raise ValueError("consecutive feature tensors must have matching shapes")
        if previous_features.shape[-1] != self.config.feature_size:
            raise ValueError("the final feature dimension does not match feature_size")
        transition = torch.cat(
            (previous_features, next_features, next_features - previous_features), dim=-1
        )
        mean, raw_log_scale = self.network(transition).chunk(2, dim=-1)
        centre = 0.5 * (
            self.config.minimum_log_scale + self.config.maximum_log_scale
        )
        radius = 0.5 * (
            self.config.maximum_log_scale - self.config.minimum_log_scale
        )
        log_scale = centre + radius * torch.tanh(raw_log_scale)
        return LatentEffortPosterior(mean=mean, log_scale=log_scale)


class DirectPoissonPortHamiltonian(nn.Module):
    r"""Direct neural latent dynamics

    .. math::

        \dot x = (J(x)-R(x))\nabla H(x) + B(x)z,

    where ``z`` is an inferred latent effort.  ``H``, ``R`` and ``B`` are
    state-dependent neural functions.  ``J`` is the pushforward of a constant
    canonical Poisson tensor through a learned diffeomorphism.  Its configured
    even rank is preserved exactly; unused canonical coordinates become
    learned Casimirs after the coordinate change.
    """

    def __init__(self, config: DirectPoissonPHConfig) -> None:
        super().__init__()
        self.config = config
        n = config.state_size
        m = config.port_size

        self.energy_network = _SmoothMLP(
            n, 1, config.hidden_size, config.hidden_layers, final_scale=0.03
        )
        self.energy_curvature = nn.Parameter(torch.zeros(n))
        triangular_size = n * (n + 1) // 2
        self.resistance_network = _SmoothMLP(
            n,
            triangular_size,
            config.hidden_size,
            config.hidden_layers,
            final_scale=0.008,
        )
        self.port_network = _SmoothMLP(
            n,
            n * m,
            config.hidden_size,
            config.hidden_layers,
            final_scale=0.02,
        )
        self.coordinate_map = AffineCouplingDiffeomorphism(
            n,
            config.hidden_size,
            config.hidden_layers,
            config.coupling_layers,
            config.coupling_scale_limit,
        )

        lower_rows, lower_columns = torch.tril_indices(n, n)
        self.register_buffer("lower_rows", lower_rows)
        self.register_buffer("lower_columns", lower_columns)
        canonical = torch.zeros(n, n)
        poisson_half = int(config.poisson_rank) // 2
        canonical[:poisson_half, poisson_half : 2 * poisson_half] = torch.eye(
            poisson_half
        )
        canonical[poisson_half : 2 * poisson_half, :poisson_half] = -torch.eye(
            poisson_half
        )
        self.register_buffer("canonical_interconnection", canonical)

        resistance_bias = self.resistance_network.network[-1].bias
        with torch.no_grad():
            # L is unconstrained, so L L^T is genuinely PSD and can contain
            # exact zero modes.  This small non-zero initialization retains
            # the former experiment's initial damping scale without imposing
            # strict positive definiteness on the model class.
            resistance_bias[lower_rows == lower_columns] = 0.03

    def hamiltonian(self, state: torch.Tensor) -> torch.Tensor:
        curvature = F.softplus(self.energy_curvature) + 1e-3
        quadratic = 0.5 * (curvature * state.square()).sum(dim=-1)
        neural_residual = self.energy_network(state).squeeze(-1)
        return quadratic + neural_residual

    def _exact_energy_value_and_gradient(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate ``H`` and ``grad H`` by the exact MLP chain rule.

        The energy network contains only affine maps and elementwise tanh.
        Its already-tested analytic Jacobian therefore yields exactly the
        same derivative as a nested ``autograd.grad`` traversal, while keeping
        all operations in the ordinary graph for parameter and state
        derivatives of the training loss.
        """

        curvature = F.softplus(self.energy_curvature) + 1e-3
        neural_residual, neural_jacobian = self.energy_network.value_and_jacobian(
            state
        )
        quadratic = 0.5 * (curvature * state.square()).sum(dim=-1)
        energy = quadratic + neural_residual.squeeze(-1)
        gradient = curvature * state + neural_jacobian.squeeze(-2)
        return energy, gradient

    def interconnection(self, state: torch.Tensor) -> torch.Tensor:
        _, chart_jacobian = self.coordinate_map.inverse_and_jacobian(state)
        canonical = self.canonical_interconnection.to(dtype=state.dtype)
        return chart_jacobian @ canonical @ chart_jacobian.transpose(-1, -2)

    def resistance(self, state: torch.Tensor) -> torch.Tensor:
        n = self.config.state_size
        packed = self.resistance_network(state)
        factor = packed.new_zeros((*packed.shape[:-1], n, n))
        factor[..., self.lower_rows, self.lower_columns] = packed
        resistance = factor @ factor.transpose(-1, -2)
        if self.config.resistance_floor:
            identity = torch.eye(n, dtype=state.dtype, device=state.device)
            resistance = resistance + self.config.resistance_floor * identity
        return resistance

    def port(self, state: torch.Tensor) -> torch.Tensor:
        values = self.port_network(state)
        return values.reshape(
            *state.shape[:-1], self.config.state_size, self.config.port_size
        )

    def _energy_gradient(
        self,
        state: torch.Tensor,
        *,
        create_graph: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            differentiable_state = state
            if not differentiable_state.requires_grad:
                differentiable_state = state.detach().requires_grad_(True)
            energy, gradient = self._exact_energy_value_and_gradient(
                differentiable_state
            )
            if not create_graph:
                gradient = gradient.detach()
        return energy, gradient

    def components(
        self,
        state: torch.Tensor,
        *,
        create_graph: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``H, grad(H), J, R, B`` for the direct latent state."""

        if create_graph is None:
            create_graph = torch.is_grad_enabled()
        energy, gradient = self._energy_gradient(state, create_graph=create_graph)
        return (
            energy,
            gradient,
            self.interconnection(state),
            self.resistance(state),
            self.port(state),
        )

    def vector_field(
        self,
        state: torch.Tensor,
        latent_effort: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the continuous pH field for an inferred latent effort."""

        _, gradient, interconnection, resistance, port = self.components(state)
        drift = torch.einsum(
            "...ij,...j->...i", interconnection - resistance, gradient
        )
        supplied = torch.einsum("...im,...m->...i", port, latent_effort)
        return drift + supplied

    def discrete_gradient(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        *,
        create_graph: bool | None = None,
        state_energy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Gonzalez discrete gradient satisfying ``g_bar @ delta = delta H``.

        For a displacement whose squared norm is below
        ``discrete_gradient_epsilon`` the midpoint gradient is used.  This is
        the only chain-rule approximation and its defect is returned by
        :meth:`audited_step`.
        """

        if create_graph is None:
            create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            midpoint = 0.5 * (state + next_state)
            if not midpoint.requires_grad:
                midpoint = midpoint.detach().requires_grad_(True)
            _, midpoint_gradient = self._energy_gradient(
                midpoint, create_graph=create_graph
            )
            displacement = next_state - state
            if state_energy is None:
                state_energy = self.hamiltonian(state)
            energy_delta = self.hamiltonian(next_state) - state_energy
            linear_delta = torch.einsum(
                "...i,...i->...", midpoint_gradient, displacement
            )
            squared_norm = displacement.square().sum(dim=-1)
            safe_norm = squared_norm.clamp_min(self.config.discrete_gradient_epsilon)
            correction = (energy_delta - linear_delta) / safe_norm
            correction = torch.where(
                squared_norm > self.config.discrete_gradient_epsilon,
                correction,
                torch.zeros_like(correction),
            )
            return midpoint_gradient + correction[..., None] * displacement

    def _implicit_endpoint(
        self,
        state: torch.Tensor,
        latent_effort: torch.Tensor,
        step_size: float,
    ) -> torch.Tensor:
        # A continuous-field prediction avoids the zero-displacement branch of
        # the Gonzalez gradient on the first fixed-point iteration.  Keep the
        # state energy from this evaluation: it is invariant across all fixed-
        # point iterations and reusing it preserves the exact same graph while
        # avoiding 32 identical H(state) evaluations in the registered setup.
        (
            state_energy,
            state_gradient,
            state_interconnection,
            state_resistance,
            state_port,
        ) = self.components(state)
        state_field = torch.einsum(
            "...ij,...j->...i",
            state_interconnection - state_resistance,
            state_gradient,
        ) + torch.einsum("...im,...m->...i", state_port, latent_effort)
        next_state = state + step_size * state_field
        relaxation = self.config.implicit_relaxation
        # Reading a device scalar on every iteration would synchronize the GPU
        # and make training unnecessarily slow.  Training therefore always
        # executes the configured, differentiable number of iterations;
        # tolerance-based early stopping is an inference-only optimization.
        check_tolerance = (
            self.config.implicit_tolerance > 0.0 and not torch.is_grad_enabled()
        )
        for _ in range(self.config.implicit_iterations):
            midpoint = 0.5 * (state + next_state)
            discrete_gradient = self.discrete_gradient(
                state, next_state, state_energy=state_energy
            )
            interconnection = self.interconnection(midpoint)
            resistance = self.resistance(midpoint)
            port = self.port(midpoint)
            field = torch.einsum(
                "...ij,...j->...i",
                interconnection - resistance,
                discrete_gradient,
            ) + torch.einsum("...im,...m->...i", port, latent_effort)
            proposed = state + step_size * field
            updated = (1.0 - relaxation) * next_state + relaxation * proposed
            if check_tolerance:
                residual = (updated - next_state).detach().abs().amax()
                next_state = updated
                if float(residual) <= self.config.implicit_tolerance:
                    break
            else:
                next_state = updated
        return next_state

    def audited_step(
        self,
        state: torch.Tensor,
        latent_effort: torch.Tensor,
        *,
        dt: float | None = None,
    ) -> DiscreteStepResult:
        """Take one implicit step and expose its exact discrete power audit."""

        step_size = self.config.dt if dt is None else dt
        if step_size <= 0.0:
            raise ValueError("dt must be positive")
        next_state = self._implicit_endpoint(state, latent_effort, step_size)
        midpoint = 0.5 * (state + next_state)
        discrete_gradient = self.discrete_gradient(state, next_state)
        interconnection = self.interconnection(midpoint)
        resistance = self.resistance(midpoint)
        port = self.port(midpoint)

        internal_field = torch.einsum(
            "...ij,...j->...i",
            interconnection - resistance,
            discrete_gradient,
        )
        supplied_field = torch.einsum("...im,...m->...i", port, latent_effort)
        implicit_residual = next_state - state - step_size * (
            internal_field + supplied_field
        )

        energy_before = self.hamiltonian(state)
        energy_after = self.hamiltonian(next_state)
        energy_delta = energy_after - energy_before
        dissipated_energy = step_size * torch.einsum(
            "...i,...ij,...j->...",
            discrete_gradient,
            resistance,
            discrete_gradient,
        )
        latent_output = torch.einsum(
            "...im,...i->...m", port, discrete_gradient
        )
        supplied_energy = step_size * torch.einsum(
            "...m,...m->...", latent_effort, latent_output
        )
        chain_rule_defect = energy_delta - torch.einsum(
            "...i,...i->...", discrete_gradient, next_state - state
        )
        balance_defect = energy_delta + dissipated_energy - supplied_energy
        implicit_residual_norm = torch.linalg.vector_norm(implicit_residual, dim=-1)
        return DiscreteStepResult(
            next_state=next_state,
            energy_before=energy_before,
            energy_after=energy_after,
            energy_delta=energy_delta,
            dissipated_energy=dissipated_energy,
            supplied_energy=supplied_energy,
            balance_defect=balance_defect,
            chain_rule_defect=chain_rule_defect,
            implicit_residual=implicit_residual,
            implicit_residual_norm=implicit_residual_norm,
            latent_output=latent_output,
        )

    def step(
        self,
        state: torch.Tensor,
        latent_effort: torch.Tensor,
        *,
        dt: float | None = None,
    ) -> torch.Tensor:
        step_size = self.config.dt if dt is None else dt
        if step_size <= 0.0:
            raise ValueError("dt must be positive")
        # ``audited_step`` recomputes J, R, B and the discrete gradient at the
        # converged endpoint solely to report diagnostics.  Callers that only
        # need the state (notably Jacobian-lens JVPs) return the identical
        # implicit endpoint without that redundant final audit.
        return self._implicit_endpoint(state, latent_effort, step_size)

    def jacobi_tensor(
        self,
        state: torch.Tensor,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        r"""Numerically audit ``J_il d_l J_jk + cyclic permutations``."""

        with torch.enable_grad():
            differentiable_state = state
            if not differentiable_state.requires_grad:
                differentiable_state = state.detach().requires_grad_(True)
            interconnection = self.interconnection(differentiable_state)
            n = self.config.state_size
            derivatives = []
            for row in range(n):
                row_derivatives = []
                for column in range(n):
                    derivative = torch.autograd.grad(
                        interconnection[..., row, column].sum(),
                        differentiable_state,
                        create_graph=create_graph,
                        retain_graph=True,
                    )[0]
                    row_derivatives.append(derivative)
                derivatives.append(torch.stack(row_derivatives, dim=-2))
            derivative_tensor = torch.stack(derivatives, dim=-3)
            return (
                torch.einsum(
                    "...il,...jkl->...ijk", interconnection, derivative_tensor
                )
                + torch.einsum(
                    "...jl,...kil->...ijk", interconnection, derivative_tensor
                )
                + torch.einsum(
                    "...kl,...ijl->...ijk", interconnection, derivative_tensor
                )
            )

    def forward(
        self,
        state: torch.Tensor,
        latent_effort: torch.Tensor,
    ) -> torch.Tensor:
        return self.step(state, latent_effort)


__all__ = [
    "AffineCouplingDiffeomorphism",
    "DirectPoissonPHConfig",
    "DirectPoissonPortHamiltonian",
    "DiscreteStepResult",
    "LatentEffortEncoder",
    "LatentEffortEncoderConfig",
    "LatentEffortPosterior",
]
