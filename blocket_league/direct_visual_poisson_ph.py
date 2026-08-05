"""Direct pixels-to-Poisson-port-Hamiltonian latent model.

This is the main trainable architecture of Experiment F.  A frozen video
transformer supplies a complete residual stream; a generic trainable readout
maps that stream directly to the pH state.  There is no intermediate dynamics
teacher and no post-hoc projection into a structured coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import nn

from .action_free_latent_effort import (
    LatentEffortConfig,
    LatentEffortInference,
    latent_effort_statistics,
)
from .direct_action_free_data import weighted_pixel_cross_entropy
from .direct_poisson_ph import DirectPoissonPortHamiltonian
from .latent_patch_renderer import LatentPatchTransformerRenderer
from .tensor_provenance import module_tensor_hash
from .pixel_direct_model import DirectPixelTransformer


@dataclass(frozen=True)
class WholeStreamEncoderConfig:
    state_size: int
    readout_hidden_size: int = 192
    lens_block: int = 4

    def __post_init__(self) -> None:
        if self.state_size < 1 or self.readout_hidden_size < 1:
            raise ValueError("encoder dimensions must be positive")


class WholeStreamFrozenEncoder(nn.Module):
    """Read all time/patch tokens from a permanently frozen transformer.

    Interventions are full residual-stream tensors with shape ``[B,T,P,D]``.
    The public API deliberately has no mask argument: neither object identity
    nor a hand-selected entity location can enter the direct experiment.
    """

    def __init__(
        self,
        backbone: DirectPixelTransformer,
        config: WholeStreamEncoderConfig,
    ) -> None:
        super().__init__()
        if not 0 <= config.lens_block < len(backbone.blocks):
            raise ValueError("lens_block is outside the transformer")
        self.backbone = backbone.eval().requires_grad_(False)
        self.config = config
        hidden = backbone.config.hidden_size
        self.pool_score = nn.Linear(hidden, 1)
        self.readout = nn.Sequential(
            nn.LayerNorm(3 * hidden),
            nn.Linear(3 * hidden, config.readout_hidden_size),
            nn.Tanh(),
            nn.Linear(config.readout_hidden_size, config.readout_hidden_size),
            nn.Tanh(),
            nn.Linear(config.readout_hidden_size, config.state_size),
        )
        nn.init.normal_(self.readout[-1].weight, std=0.03)
        nn.init.zeros_(self.readout[-1].bias)
        self._sealed_backbone_hash = module_tensor_hash(self.backbone)

    @property
    def state_size(self) -> int:
        return self.config.state_size

    @property
    def activation_shape(self) -> tuple[int, int, int]:
        backbone = self.backbone.config
        return (
            backbone.history_frames,
            backbone.grid_size**2,
            backbone.hidden_size,
        )

    @property
    def sealed_backbone_hash(self) -> str:
        return self._sealed_backbone_hash

    def assert_backbone_frozen(self) -> None:
        if any(parameter.requires_grad for parameter in self.backbone.parameters()):
            raise AssertionError("the video backbone acquired a trainable parameter")
        if module_tensor_hash(self.backbone) != self._sealed_backbone_hash:
            raise AssertionError("the sealed video-backbone tensor hash changed")
        if self.backbone.training:
            raise AssertionError("the sealed video backbone left evaluation mode")

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _validate_contexts(self, contexts: torch.Tensor) -> None:
        expected = (
            self.backbone.config.history_frames,
            self.backbone.config.image_size,
            self.backbone.config.image_size,
        )
        if contexts.shape[-3:] != expected:
            raise ValueError(f"expected context tail {expected}, got {contexts.shape[-3:]}")

    def prefix_activation(self, contexts: torch.Tensor) -> torch.Tensor:
        """Return the sealed block activation, detached from backbone weights."""

        self._validate_contexts(contexts)
        model = self.backbone
        with torch.no_grad():
            tokens = (
                model.patch_projection(model.patch_tokens(contexts.long()))
                + model.spatial_position
                + model.temporal_position[:, : contexts.shape[1]]
            )
            for index, block in enumerate(model.blocks):
                tokens = block(tokens)
                if index == self.config.lens_block:
                    break
        return tokens.detach()

    def _suffix(self, activation: torch.Tensor) -> torch.Tensor:
        tokens = activation
        for index in range(self.config.lens_block + 1, len(self.backbone.blocks)):
            tokens = self.backbone.blocks[index](tokens)
        return tokens

    def _read(self, tokens: torch.Tensor) -> torch.Tensor:
        flat = tokens.flatten(1, 2)
        attention = self.pool_score(flat).squeeze(-1).softmax(dim=-1)
        attended = torch.einsum("bs,bsh->bh", attention, flat)
        features = torch.cat(
            (
                flat.mean(dim=1),
                flat.std(dim=1, unbiased=False),
                attended,
            ),
            dim=-1,
        )
        return self.readout(features.float())

    def read_suffix_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Read a latent state from the exact post-backbone token stream.

        The soft frozen rollout already computes the complete transformer
        token stream while predicting its next frame.  Reusing those tokens
        is algebraically identical to running ``soft_prefix_activation`` and
        the remaining frozen suffix a second time on the same soft context,
        while avoiding that redundant transformer evaluation.  This public
        entry point deliberately exposes only the learned state readout; it
        accepts no simulator or action quantity.
        """

        expected = (
            self.backbone.config.history_frames,
            self.backbone.config.grid_size**2,
            self.backbone.config.hidden_size,
        )
        if tokens.ndim != 4 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(f"expected suffix-token tail {expected}")
        return self._read(tokens)

    def from_activation(
        self,
        activation: torch.Tensor,
        *,
        intervention: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intervention is not None:
            if intervention.shape != activation.shape:
                raise ValueError("a write must match the complete residual-stream shape")
            activation = activation + intervention
        return self._read(self._suffix(activation))

    def forward(
        self,
        contexts: torch.Tensor,
        *,
        intervention: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_contexts(contexts)
        leading = contexts.shape[:-3]
        flat_contexts = contexts.reshape(-1, *contexts.shape[-3:])
        activation = self.prefix_activation(flat_contexts)
        if intervention is not None:
            expected = (*leading, *activation.shape[1:])
            if intervention.shape != expected:
                raise ValueError(f"expected intervention shape {expected}")
            intervention = intervention.reshape_as(activation)
        state = self.from_activation(activation, intervention=intervention)
        return state.reshape(*leading, self.state_size)

    def state_jacobian_from_activation(
        self,
        activation: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Return ``D_h E`` flattened as ``[B,n,T*P*D]``."""

        _, jacobian = self.state_and_jacobian_from_activation(
            activation, create_graph=create_graph
        )
        return jacobian

    def state_and_jacobian_from_activation(
        self,
        activation: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``E(h)`` and ``D_h E`` from one shared suffix forward."""

        if activation.ndim != 4:
            raise ValueError("activation must have shape [batch,time,patch,channel]")
        differentiable = activation
        if not differentiable.requires_grad:
            differentiable = activation.detach().requires_grad_(True)
        state = self.from_activation(differentiable)
        # The readout is batch-separable.  Evaluate all coordinate VJPs in one
        # batched autograd call rather than traversing the suffix transformer
        # ``state_size`` times.  Every seed is the same coordinate-wise batch
        # sum as before, so values and higher-order gradients are unchanged.
        seeds = torch.eye(
            self.state_size, dtype=state.dtype, device=state.device
        )[:, None, :].expand(self.state_size, state.shape[0], self.state_size)
        derivatives = torch.autograd.grad(
            state,
            differentiable,
            grad_outputs=seeds,
            create_graph=create_graph,
            retain_graph=create_graph,
            is_grads_batched=True,
        )[0]
        jacobian = derivatives.permute(
            1, 0, *range(2, derivatives.ndim)
        ).flatten(2)
        return state, jacobian


class DirectVisualPoissonPH(nn.Module):
    """Strict visual bottleneck followed immediately by direct pH dynamics."""

    def __init__(
        self,
        encoder: WholeStreamFrozenEncoder,
        renderer: LatentPatchTransformerRenderer,
        core: DirectPoissonPortHamiltonian,
        effort_inference: LatentEffortInference | None = None,
    ) -> None:
        super().__init__()
        if encoder.state_size != core.config.state_size:
            raise ValueError("encoder and pH state dimensions differ")
        if renderer.state_size != core.config.state_size:
            raise ValueError("renderer and pH state dimensions differ")
        self.encoder = encoder
        self.renderer = renderer
        self.core = core
        self.effort_inference = effort_inference or LatentEffortInference(
            LatentEffortConfig(
                state_size=core.config.state_size,
                effort_size=core.config.port_size,
            )
        )

    def encode(self, contexts: torch.Tensor) -> torch.Tensor:
        return self.encoder(contexts)

    def infer_latent_effort(
        self,
        present: torch.Tensor,
        successor: torch.Tensor,
    ) -> torch.Tensor:
        return self.effort_inference(present, successor)

    def step(self, state: torch.Tensor, latent_effort: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=state.device.type, enabled=False):
            return self.core.step(state.float(), latent_effort.float())

    def render(self, state: torch.Tensor) -> torch.Tensor:
        return self.renderer(state)


class PersistentOrthogonalPortFrame(nn.Module):
    """One experiment-wide SO(m) basis transform shared across all batches.

    A Cayley transform is smooth at repeated singular values and, unlike a
    per-batch Procrustes solve, cannot absorb a different ``Q(x)`` at every
    state.  For the scalar port the only continuous component is identity;
    any global sign is learned directly by the adjacent port networks.
    """

    def __init__(self, port_size: int) -> None:
        super().__init__()
        if port_size < 1:
            raise ValueError("port_size must be positive")
        self.raw = nn.Parameter(torch.zeros(port_size, port_size))

    def matrix(self) -> torch.Tensor:
        skew = self.raw - self.raw.T
        identity = torch.eye(
            skew.shape[0], dtype=skew.dtype, device=skew.device
        )
        return torch.linalg.solve(identity + 0.5 * skew, identity - 0.5 * skew)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        if vectors.shape[-1] != self.raw.shape[0]:
            raise ValueError("port-frame dimension mismatch")
        return vectors @ self.matrix()


@dataclass(frozen=True)
class DirectVideoLossConfig:
    reconstruction_weight: float = 1.0
    rollout_pixel_weight: float = 1.0
    rollout_latent_weight: float = 1.0
    innovation_weight: float = 0.10
    jacobian_bridge_weight: float = 1.0
    oddness_weight: float = 0.25
    manifold_cycle_weight: float = 0.25
    chart_conditioning_weight: float = 0.001
    whitening_weight: float = 0.05
    energy_gauge_weight: float = 0.01
    port_frame_weight: float = 0.10
    port_holonomy_weight: float = 0.05
    implicit_residual_weight: float = 0.01
    chain_rule_weight: float = 0.01
    implicit_residual_tolerance: float = 1e-8
    chain_rule_tolerance: float = 1e-7
    innovation_target_variance: float = 0.25
    rollout_horizons: tuple[int, ...] = (1, 2, 4, 8)

    def __post_init__(self) -> None:
        scalar_items = {
            name: value
            for name, value in self.__dict__.items()
            if name != "rollout_horizons"
        }
        if any(
            type(value) not in (int, float) or not math.isfinite(float(value))
            for value in scalar_items.values()
        ):
            raise ValueError("all direct-video scalar losses/tolerances must be finite numeric")
        if self.implicit_residual_tolerance <= 0.0:
            raise ValueError("implicit_residual_tolerance must be positive")
        if self.chain_rule_tolerance <= 0.0:
            raise ValueError("chain_rule_tolerance must be positive")
        if self.innovation_target_variance <= 0.0:
            raise ValueError("innovation_target_variance must be positive")
        if (
            type(self.rollout_horizons) is not tuple
            or not self.rollout_horizons
            or any(
                type(value) is not int or value < 1
                for value in self.rollout_horizons
            )
            or tuple(sorted(set(self.rollout_horizons))) != self.rollout_horizons
        ):
            raise ValueError(
                "rollout_horizons must be a sorted unique tuple of positive integers"
            )
        weights = tuple(
            value
            for name, value in scalar_items.items()
            if name.endswith("_weight")
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("loss weights must be non-negative")


def state_whitening_loss(state: torch.Tensor) -> torch.Tensor:
    flat = state.reshape(-1, state.shape[-1])
    centered = flat - flat.mean(dim=0, keepdim=True)
    scale = centered.std(dim=0, unbiased=False).clamp_min(1e-4)
    standardized = centered / scale
    covariance = standardized.T @ standardized / max(flat.shape[0], 1)
    identity = torch.eye(flat.shape[-1], device=flat.device, dtype=flat.dtype)
    return (covariance - identity).square().mean() + 0.02 * scale.log().square().mean()


def state_effort_independence_loss(
    state: torch.Tensor,
    latent_effort: torch.Tensor,
) -> torch.Tensor:
    flat_state = state.reshape(-1, state.shape[-1])
    flat_effort = latent_effort.reshape(-1, latent_effort.shape[-1])
    if flat_state.shape[0] != flat_effort.shape[0]:
        raise ValueError("state and effort sample axes differ")
    state_centered = flat_state - flat_state.mean(dim=0, keepdim=True)
    effort_centered = flat_effort - flat_effort.mean(dim=0, keepdim=True)
    state_scale = state_centered.std(dim=0, unbiased=False).clamp_min(1e-4)
    effort_scale = effort_centered.std(dim=0, unbiased=False).clamp_min(1e-4)
    cross = (
        (state_centered / state_scale).T
        @ (effort_centered / effort_scale)
        / max(flat_state.shape[0], 1)
    )
    return cross.square().mean()


def state_effort_second_moment_independence_loss(
    state: torch.Tensor,
    latent_effort: torch.Tensor,
) -> torch.Tensor:
    """RBF-HSIC between state and ``vec(zz^T)`` to fix local port scale.

    Global effort covariance alone cannot reject the gauge
    ``B(x)->c(x)B(x), z->z/c(x)`` when the effort has zero conditional mean.
    The conditional second moment changes under that gauge.  A characteristic
    kernel detects nonlinear dependence without a physical state/action label.
    """

    flat_state = state.reshape(-1, state.shape[-1]).float()
    flat_effort = latent_effort.reshape(-1, latent_effort.shape[-1]).float()
    if flat_state.shape[0] != flat_effort.shape[0]:
        raise ValueError("state and effort sample axes differ")
    if flat_state.shape[0] < 3:
        return flat_state.new_zeros(())
    second_moment = torch.einsum("bi,bj->bij", flat_effort, flat_effort).flatten(1)

    def centered_rbf(values: torch.Tensor) -> torch.Tensor:
        centered = values - values.mean(dim=0, keepdim=True)
        scale = centered.std(dim=0, unbiased=False).clamp_min(1e-4)
        standardized = centered / scale
        distance = torch.cdist(standardized, standardized).square()
        count = distance.shape[0]
        off_diagonal_mean = (
            distance.sum() / max(count * (count - 1), 1)
        ).detach().clamp_min(1e-4)
        kernel = torch.exp(-0.5 * distance / off_diagonal_mean)
        return (
            kernel
            - kernel.mean(dim=0, keepdim=True)
            - kernel.mean(dim=1, keepdim=True)
            + kernel.mean()
        )

    state_kernel = centered_rbf(flat_state)
    moment_kernel = centered_rbf(second_moment)
    return (state_kernel * moment_kernel).mean().clamp_min(0.0)


def chart_conditioning_loss(
    core: DirectPoissonPortHamiltonian,
    state: torch.Tensor,
) -> torch.Tensor:
    flat = state.reshape(-1, state.shape[-1])
    sample = flat[: min(flat.shape[0], 16)]
    base = core.coordinate_map.inverse(sample)
    jacobian = core.coordinate_map.jacobian(base)
    singular = torch.linalg.svdvals(jacobian).clamp_min(1e-6)
    return singular.log().square().mean()


def port_frame_regularizers(
    core: DirectPoissonPortHamiltonian,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fix the local O(m) gauge by global polar parallel transport.

    ``B(x)`` may change its physical column space with state.  We preserve that
    normal motion while penalizing the orthogonal factor of the transport
    between its column frames.  Penalizing only the skew part is insufficient:
    a symmetric column swap has zero skew yet represents a different local
    gauge.  The all-pairs graph also joins different videos, so disjoint
    trajectories cannot silently choose unrelated permutations/signs.
    """

    def polar_transport(overlap: torch.Tensor) -> torch.Tensor:
        """Differentiable polar factor without SVD/eigenvector gradients."""

        dimension = overlap.shape[-1]
        identity = torch.eye(
            dimension, dtype=overlap.dtype, device=overlap.device
        )
        gram = overlap.transpose(-1, -2) @ overlap + 1e-6 * identity
        scale = torch.linalg.matrix_norm(
            gram, ord="fro", dim=(-2, -1), keepdim=True
        ).clamp_min(1e-8)
        y = gram / scale
        z = identity.expand_as(gram)
        # Newton--Schulz inverse square root.  Every operation has a unique,
        # finite derivative at repeated singular values such as B=I.
        for _ in range(10):
            update = 0.5 * (3.0 * identity - z @ y)
            y = y @ update
            z = update @ z
        inverse_sqrt = z / scale.sqrt()
        return overlap @ inverse_sqrt

    if state.ndim != 3:
        raise ValueError("state must have shape [batch,time,coordinate]")
    port = core.port(state.float())
    gram = port.transpose(-1, -2) @ port
    port_identity = torch.eye(
        gram.shape[-1], dtype=gram.dtype, device=gram.device
    )
    mean_square = torch.diagonal(gram, dim1=-2, dim2=-1).mean(dim=-1)
    normalized_gram = gram / mean_square[..., None, None].clamp_min(1e-12)
    _, normalized_logdet = torch.linalg.slogdet(
        normalized_gram + 1e-8 * port_identity
    )
    # This condition barrier is per sample and scale-free: unlike a raw
    # determinant threshold, one large singular value cannot hide a zero one.
    minimum_normalized_logdet = gram.shape[-1] * gram.new_tensor(math.log(0.05))
    condition_barrier = torch.relu(
        minimum_normalized_logdet - normalized_logdet
    ).square()
    magnitude_barrier = torch.relu(1.0 - mean_square / 1e-6).square()
    rank_barrier = (condition_barrier + magnitude_barrier).mean()

    # The rectangular polar factor is an orthonormal frame whenever B is full
    # rank, yet remains finite at a deliberately rank-deficient adversarial
    # test point.  This avoids QR/SVD's undefined backward there.
    basis = polar_transport(port)
    if state.shape[1] < 2:
        zero = state.new_zeros(())
        return zero, zero, rank_barrier
    overlap = basis[:, :-1].transpose(-1, -2) @ basis[:, 1:]
    transport = polar_transport(overlap)
    identity = torch.eye(
        transport.shape[-1], dtype=transport.dtype, device=transport.device
    )
    frame = (transport - identity).square().mean()

    # A dense deterministic graph over states from all trajectories closes the
    # disconnected-video loophole.  Cap only the number of graph vertices, not
    # the trajectories represented by them; direct training batches are much
    # smaller than this cap in the registered experiment.
    graph_basis = basis.reshape(-1, basis.shape[-2], basis.shape[-1])
    if graph_basis.shape[0] > 32:
        indices = torch.linspace(
            0,
            graph_basis.shape[0] - 1,
            32,
            device=graph_basis.device,
        ).round().long()
        graph_basis = graph_basis[indices]
    if graph_basis.shape[0] >= 2:
        graph_overlap = torch.einsum(
            "vam,wan->vwmn", graph_basis, graph_basis
        )
        graph_transport = polar_transport(graph_overlap)
        upper = torch.triu(
            torch.ones(
                graph_basis.shape[0],
                graph_basis.shape[0],
                dtype=torch.bool,
                device=graph_basis.device,
            ),
            diagonal=1,
        )
        frame = frame + (graph_transport[upper] - identity).square().mean()

    if state.shape[1] >= 3:
        composed = transport[:, 0]
        for index in range(1, transport.shape[1]):
            composed = composed @ transport[:, index]
        endpoint_transport = polar_transport(
            basis[:, 0].transpose(-1, -2) @ basis[:, -1]
        )
        holonomy = (composed - endpoint_transport).square().mean()
    else:
        holonomy = frame.new_zeros(())

    return frame, holonomy, rank_barrier


def direct_video_objective(
    model: DirectVisualPoissonPH,
    pixel_contexts: torch.Tensor,
    frames: torch.Tensor,
    class_weights: torch.Tensor,
    config: DirectVideoLossConfig,
    *,
    lens_terms: Mapping[str, torch.Tensor] | None = None,
    require_lens_terms: bool = True,
    encoded_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One joint pixels-only objective; no target vector field is detached."""

    if pixel_contexts.ndim != 5:
        raise ValueError("pixel_contexts must be [batch,time,history,height,width]")
    if frames.shape[:2] != pixel_contexts.shape[:2]:
        raise ValueError("contexts and frame targets must share batch/time axes")
    batch, points = pixel_contexts.shape[:2]
    if points < 2:
        raise ValueError("at least one transition is required")
    if encoded_states is None:
        states = model.encode(pixel_contexts)
    else:
        expected_state_shape = (
            pixel_contexts.shape[0],
            pixel_contexts.shape[1],
            model.core.config.state_size,
        )
        if tuple(encoded_states.shape) != expected_state_shape:
            raise ValueError(
                f"expected encoded_states shape {expected_state_shape}, "
                f"got {tuple(encoded_states.shape)}"
            )
        states = encoded_states
    efforts = model.infer_latent_effort(states[:, :-1], states[:, 1:])

    reconstruction = weighted_pixel_cross_entropy(
        model.render(states), frames, class_weights
    )
    current = states[:, 0]
    rolled = []
    balance_defects = []
    chain_rule_defects = []
    implicit_residuals = []
    for index in range(points - 1):
        with torch.autocast(device_type=current.device.type, enabled=False):
            audited = model.core.audited_step(current.float(), efforts[:, index].float())
        current = audited.next_state
        rolled.append(current)
        balance_defects.append(audited.balance_defect)
        chain_rule_defects.append(audited.chain_rule_defect)
        implicit_residuals.append(audited.implicit_residual_norm)
    rollout = torch.stack(rolled, dim=1)
    targets = states[:, 1:]
    scale = states.detach().reshape(-1, states.shape[-1]).std(
        dim=0, unbiased=False
    ).clamp_min(0.05)
    rollout_latent = ((rollout - targets) / scale).square().mean()

    horizons = tuple(
        horizon for horizon in config.rollout_horizons if horizon <= points - 1
    )
    if not horizons:
        raise ValueError("no registered rollout horizon fits this batch")
    indices = torch.tensor(
        tuple(horizon - 1 for horizon in horizons), device=frames.device
    )
    rollout_pixel = weighted_pixel_cross_entropy(
        model.render(rollout[:, indices]),
        frames[:, indices + 1],
        class_weights,
    )

    effort_terms = latent_effort_statistics(
        efforts,
        target_variance=config.innovation_target_variance,
    )
    linear_independence = state_effort_independence_loss(states[:, :-1], efforts)
    second_moment_independence = state_effort_second_moment_independence_loss(
        states[:, :-1], efforts
    )
    independence = linear_independence + second_moment_independence
    innovation = effort_terms["total"] + 0.25 * effort_terms["temporal"] + independence
    whitening = state_whitening_loss(states)
    energy = model.core.hamiltonian(states.float())
    energy_gauge = energy.mean().square() + (energy.std(unbiased=False) - 1.0).square()
    chart = (
        chart_conditioning_loss(model.core, states.float())
        if config.chart_conditioning_weight > 0.0
        else states.new_zeros(())
    )

    frame_transport, frame_holonomy, port_rank = port_frame_regularizers(
        model.core, states
    )
    implicit_values = torch.stack(implicit_residuals, dim=1)
    chain_values = torch.stack(chain_rule_defects, dim=1)
    implicit_penalty = torch.log1p(
        (implicit_values / config.implicit_residual_tolerance).square()
    ).mean()
    chain_penalty = torch.log1p(
        (chain_values / config.chain_rule_tolerance).square()
    ).mean()

    lens_weights_active = any(
        weight > 0.0
        for weight in (
            config.jacobian_bridge_weight,
            config.oddness_weight,
            config.manifold_cycle_weight,
        )
    )
    if require_lens_terms and lens_weights_active and lens_terms is None:
        raise RuntimeError(
            "Jacobian-lens terms are mandatory when their registered weights are non-zero"
        )
    lens_terms = {} if lens_terms is None else dict(lens_terms)
    required_lens_keys = {"bridge", "oddness", "manifoldCycle"}
    if require_lens_terms and lens_weights_active and not required_lens_keys.issubset(lens_terms):
        raise RuntimeError(
            f"missing mandatory Jacobian-lens terms: {sorted(required_lens_keys - set(lens_terms))}"
        )
    zero = states.new_zeros(())
    bridge = lens_terms.get("bridge", zero)
    oddness = lens_terms.get("oddness", zero)
    manifold_cycle = lens_terms.get("manifoldCycle", zero)
    total = (
        config.reconstruction_weight * reconstruction
        + config.rollout_pixel_weight * rollout_pixel
        + config.rollout_latent_weight * rollout_latent
        + config.innovation_weight * innovation
        + config.jacobian_bridge_weight * bridge
        + config.oddness_weight * oddness
        + config.manifold_cycle_weight * manifold_cycle
        + config.chart_conditioning_weight * chart
        + config.whitening_weight * whitening
        + config.energy_gauge_weight * energy_gauge
        + config.port_frame_weight * (frame_transport + port_rank)
        + config.port_holonomy_weight * frame_holonomy
        + config.implicit_residual_weight * implicit_penalty
        + config.chain_rule_weight * chain_penalty
    )
    metrics = {
        "total": total,
        "reconstruction": reconstruction,
        "rolloutPixel": rollout_pixel,
        "rolloutLatent": rollout_latent,
        "innovation": innovation,
        "effortMean": effort_terms["mean"],
        "effortVariance": effort_terms["variance"],
        "effortDecorrelation": effort_terms["decorrelation"],
        "effortTemporal": effort_terms["temporal"],
        "stateEffortIndependence": independence,
        "stateEffortLinearIndependence": linear_independence,
        "stateEffortSecondMomentIndependence": second_moment_independence,
        "whitening": whitening,
        "energyGauge": energy_gauge,
        "chartConditioning": chart,
        "portFrameTransport": frame_transport,
        "portFrameHolonomy": frame_holonomy,
        "portRankOrientation": port_rank,
        "implicitResidualPenalty": implicit_penalty,
        "chainRulePenalty": chain_penalty,
        "jacobianBridge": bridge,
        "writeOddness": oddness,
        "manifoldCycle": manifold_cycle,
        "balanceDefectMax": torch.stack(balance_defects, dim=1).abs().max(),
        "implicitResidualMax": implicit_values.max(),
        "chainRuleDefectMax": chain_values.abs().max(),
    }
    return total, metrics


def trainable_parameters_without_backbone(model: DirectVisualPoissonPH) -> list[nn.Parameter]:
    model.encoder.assert_backbone_frozen()
    backbone_ids = {id(parameter) for parameter in model.encoder.backbone.parameters()}
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    if not parameters:
        raise AssertionError("the direct model has no trainable parameter")
    return parameters


__all__ = [
    "DirectVideoLossConfig",
    "DirectVisualPoissonPH",
    "PersistentOrthogonalPortFrame",
    "WholeStreamEncoderConfig",
    "WholeStreamFrozenEncoder",
    "chart_conditioning_loss",
    "direct_video_objective",
    "port_frame_regularizers",
    "state_effort_independence_loss",
    "state_effort_second_moment_independence_loss",
    "state_whitening_loss",
    "trainable_parameters_without_backbone",
]
