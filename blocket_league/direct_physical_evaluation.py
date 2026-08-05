"""Locked post-freeze physical evaluation for the direct visual pH experiment.

Nothing in this module is a training API.  The learned encoder, renderer and
dynamics must already be frozen and sealed before any function in this file is
called.  A physical environment is kept behind :class:`PixelPlant`; learned
modules receive pixel histories only.

The calibration protocol is deliberately small and non-adaptive.  Candidate
states are ranked from pixels and the learned ``B(x)`` *before* a physical
response is queried.  Exactly four paired ``+/-`` probes are then made per
physical interface axis and one constant latent-from-interface matrix is fit by
the closed-form ridge solution.  There is no optimizer or backpropagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .action_free_excitation import experiment_f_blocket_world_config
from .direct_activation_lens import (
    FrozenSoftPixelActivationLens,
)
from .direct_cotangent_bridge import (
    PixelChangeProbeBank,
    activation_observable_covectors,
)
from .direct_jacobian_port_extractor import FrozenEmpiricalJacobianActivationPort
from .direct_visual_poisson_ph import (
    PersistentOrthogonalPortFrame,
    WholeStreamFrozenEncoder,
)
from .env import BlocketLeagueEnv, PALETTE, WorldConfig, WorldState
from .passive_control_systems import (
    PendulumConfig,
    PendulumEnv,
    PendulumState,
    pendulum_target_frames,
    wrap_angle,
)


PAIRED_CALIBRATION_STATES_PER_AXIS = 4
RIDGE_COEFFICIENT = 1e-6


# Registered contact-mediated Blocket target.  This description is deliberately
# canonical and hashed into every episode identifier.  The hidden source-side
# oracle is used only to construct physically reachable categorical targets; it
# is never returned by ``make_builtin_control_episodes`` and never reaches a
# planner.
BLOCKET_CONTACT_TARGET_TASK_NAME = "blocket_contact_mediated_puck_displacement_v1"
BLOCKET_CONTACT_MIN_PIXEL_DISPLACEMENT = 0.18
BLOCKET_CONTACT_MAX_PIXEL_DISPLACEMENT = 0.28
BLOCKET_CONTACT_ORACLE_NATIVE_THRUST = 0.40
_BLOCKET_CONTACT_SURFACE_GAP = (0.012, 0.024)
_BLOCKET_CONTACT_SOURCE_THRESHOLD = (0.195, 0.215)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


BLOCKET_CONTACT_TARGET_TASK_SHA256 = _canonical_sha256(
    {
        "name": BLOCKET_CONTACT_TARGET_TASK_NAME,
        "targetSource": "categorical_pixels_only",
        "controlledPixelValues": (7, 8),
        "initialCondition": "stationary_player_behind_puck_small_positive_gap",
        "targetConstruction": "source_only_admissible_constant_thrust_simulator_rollout",
        "minimumPixelPuckDisplacement": BLOCKET_CONTACT_MIN_PIXEL_DISPLACEMENT,
        "maximumPixelPuckDisplacement": BLOCKET_CONTACT_MAX_PIXEL_DISPLACEMENT,
        "sourceCrossingThresholdInterval": _BLOCKET_CONTACT_SOURCE_THRESHOLD,
        "surfaceGapInterval": _BLOCKET_CONTACT_SURFACE_GAP,
        "oracleNativeThrustMagnitude": BLOCKET_CONTACT_ORACLE_NATIVE_THRUST,
        "maximumSourceRolloutSteps": 48,
        "goalsEnabled": False,
    }
)
_BLOCKET_CONTACT_IDENTIFIER_PREFIX = (
    f"{BLOCKET_CONTACT_TARGET_TASK_NAME}:"
    f"{BLOCKET_CONTACT_TARGET_TASK_SHA256}:"
)
_GENERIC_TARGET_TASK = {
    "name": "generic_categorical_pixel_target",
    "sha256": _canonical_sha256(
        {
            "name": "generic_categorical_pixel_target",
            "targetSource": "categorical_pixels_only",
        }
    ),
}


def _episode_target_task_identity(
    episode_identifiers: Sequence[str],
) -> dict[str, str]:
    """Derive a sealed task identity without adding mutable caller evidence."""

    contact = tuple(
        identifier.startswith(_BLOCKET_CONTACT_IDENTIFIER_PREFIX)
        for identifier in episode_identifiers
    )
    if any(contact) and not all(contact):
        raise ValueError("control evidence mixes contact-task and foreign episodes")
    if contact and all(contact):
        return {
            "name": BLOCKET_CONTACT_TARGET_TASK_NAME,
            "sha256": BLOCKET_CONTACT_TARGET_TASK_SHA256,
        }
    return dict(_GENERIC_TARGET_TASK)


@dataclass(frozen=True)
class EvaluationSystem:
    """Pixel-defined deployment protocol for one physical system."""

    name: str
    physical_action_size: int
    dt: float
    control_steps: int
    planning_horizon: int
    probe_amplitude: float = 0.25
    controlled_pixel_values: tuple[int, ...] = ()
    pixel_observable: str = "centroid"

    def __post_init__(self) -> None:
        if self.physical_action_size < 1:
            raise ValueError("physical_action_size must be positive")
        if self.dt <= 0.0 or self.control_steps < 1 or self.planning_horizon < 1:
            raise ValueError("time and control horizons must be positive")
        if not 0.0 < self.probe_amplitude <= 1.0:
            raise ValueError("probe_amplitude must lie in (0, 1]")


SYSTEMS: Mapping[str, EvaluationSystem] = {
    "pendulum": EvaluationSystem(
        name="pendulum",
        physical_action_size=1,
        dt=PendulumConfig().dt,
        control_steps=80,
        planning_horizon=24,
        controlled_pixel_values=(7, 8),
        pixel_observable="pendulum_angle",
    ),
    "blocket": EvaluationSystem(
        name="blocket",
        physical_action_size=2,
        dt=WorldConfig().dt,
        control_steps=48,
        planning_horizon=12,
        # The player is the actuator, but success is measured on the puck.  A
        # controller must therefore discover and exploit player-puck contact.
        controlled_pixel_values=(7, 8),
        pixel_observable="centroid",
    ),
}


@dataclass(frozen=True)
class PhysicalInterface:
    """A fixed deployment wrapper ``u_native = matrix @ u_interface``."""

    name: str
    native_from_interface: tuple[tuple[float, ...], ...]

    def matrix(self, *, dtype: np.dtype = np.float64) -> np.ndarray:
        value = np.asarray(self.native_from_interface, dtype=dtype)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError("a physical interface must be a square matrix")
        if not bool(np.isfinite(value).all()) or abs(float(np.linalg.det(value))) < 1e-10:
            raise ValueError("the registered physical interface must be finite and invertible")
        return value


def fixed_interfaces(system: EvaluationSystem | str) -> Mapping[str, PhysicalInterface]:
    """Return the native and preregistered never-seen deployment interfaces."""

    name = system if isinstance(system, str) else system.name
    if name == "pendulum":
        return {
            "native": PhysicalInterface("native", ((1.0,),)),
            "unseen": PhysicalInterface("unseen", ((-1.6,),)),
        }
    if name == "blocket":
        angle = math.radians(37.0)
        rotation = np.asarray(
            ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
            dtype=np.float64,
        )
        exchange_scale = np.asarray(((0.0, 0.65), (-1.40, 0.0)), dtype=np.float64)
        unseen = rotation @ exchange_scale
        return {
            "native": PhysicalInterface("native", ((1.0, 0.0), (0.0, 1.0))),
            "unseen": PhysicalInterface(
                "unseen", tuple(tuple(float(item) for item in row) for row in unseen)
            ),
        }
    raise KeyError(f"no registered physical interfaces for system {name!r}")


LINEAR_INTERFACE_BOUND_FORMULA = (
    "1/max_{M in scope,s in {-1,+1}^m} ||M s||_2"
)


def _box_to_native_l2_gain(interface: PhysicalInterface) -> float:
    matrix = interface.matrix()
    dimension = matrix.shape[1]
    maximum = 0.0
    for corner_index in np.ndindex(*(2,) * dimension):
        corner = np.asarray(
            tuple(-1.0 if value == 0 else 1.0 for value in corner_index),
            dtype=np.float64,
        )
        maximum = max(maximum, float(np.linalg.norm(matrix @ corner, ord=2)))
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("physical interface has an invalid box-to-native gain")
    return maximum


def registered_linear_interface_command_bound(
    system: EvaluationSystem | str,
) -> float:
    """Largest common symmetric box that never triggers plant saturation."""

    definition = SYSTEMS[system] if isinstance(system, str) else system
    if definition.name not in SYSTEMS:
        raise KeyError("a common native/unseen bound exists only for registered systems")
    registered = SYSTEMS[definition.name]
    if (
        definition.physical_action_size != registered.physical_action_size
        or not math.isclose(definition.dt, registered.dt, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("evaluation system differs from the registered linear domain")
    maximum_gain = max(
        _box_to_native_l2_gain(interface)
        for interface in fixed_interfaces(registered).values()
    )
    return 1.0 / maximum_gain


@dataclass(frozen=True)
class LinearInterfaceProtocol:
    """Canonical physical protocol bound into every response-bank digest."""

    system_name: str
    dt: float
    probe_amplitude: float
    physical_action_size: int
    interface_name: str
    native_from_interface: tuple[tuple[float, ...], ...]
    common_interface_command_bound: float
    maximum_box_to_native_l2_gain: float
    bound_scope: str
    bound_formula: str = LINEAR_INTERFACE_BOUND_FORMULA
    native_l2_limit: float = 1.0

    def __post_init__(self) -> None:
        matrix = np.asarray(self.native_from_interface, dtype=np.float64)
        numeric = (
            self.dt,
            self.probe_amplitude,
            self.common_interface_command_bound,
            self.maximum_box_to_native_l2_gain,
            self.native_l2_limit,
        )
        if (
            type(self.system_name) is not str
            or not self.system_name
            or type(self.interface_name) is not str
            or not self.interface_name
            or type(self.physical_action_size) is not int
            or self.physical_action_size < 1
            or matrix.shape
            != (self.physical_action_size, self.physical_action_size)
            or not bool(np.isfinite(matrix).all())
            or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in numeric)
            or self.bound_formula != LINEAR_INTERFACE_BOUND_FORMULA
            or self.bound_scope not in {
                "registered_native_and_unseen",
                "provided_interface_only",
            }
            or not math.isclose(self.native_l2_limit, 1.0, rel_tol=0.0, abs_tol=0.0)
        ):
            raise ValueError("linear physical-interface protocol is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system_name,
            "dt": self.dt,
            "probeAmplitude": self.probe_amplitude,
            "physicalActionSize": self.physical_action_size,
            "interface": self.interface_name,
            "nativeFromInterface": self.native_from_interface,
            "commonInterfaceCommandBound": self.common_interface_command_bound,
            "maximumBoxToNativeL2Gain": self.maximum_box_to_native_l2_gain,
            "boundScope": self.bound_scope,
            "boundFormula": self.bound_formula,
            "nativeL2Limit": self.native_l2_limit,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def linear_interface_protocol(
    system: EvaluationSystem,
    interface: PhysicalInterface,
) -> LinearInterfaceProtocol:
    matrix = interface.matrix()
    if matrix.shape != (system.physical_action_size, system.physical_action_size):
        raise ValueError("interface rank differs from the physical action rank")
    if system.name in SYSTEMS:
        interfaces = fixed_interfaces(system)
        if interface.name not in interfaces or not np.array_equal(
            interface.matrix(), interfaces[interface.name].matrix()
        ):
            raise ValueError("physical interface differs from the registered matrix")
        maximum_gain = max(
            _box_to_native_l2_gain(value) for value in interfaces.values()
        )
        scope = "registered_native_and_unseen"
    else:
        maximum_gain = _box_to_native_l2_gain(interface)
        scope = "provided_interface_only"
    return LinearInterfaceProtocol(
        system_name=system.name,
        dt=system.dt,
        probe_amplitude=system.probe_amplitude,
        physical_action_size=system.physical_action_size,
        interface_name=interface.name,
        native_from_interface=tuple(
            tuple(float(value) for value in row) for row in matrix
        ),
        common_interface_command_bound=1.0 / maximum_gain,
        maximum_box_to_native_l2_gain=maximum_gain,
        bound_scope=scope,
    )


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().contiguous().cpu().reshape(-1)
    return cpu.view(torch.uint8).numpy().tobytes()


def _module_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    digest.update(f"{type(module).__module__}.{type(module).__qualname__}".encode())
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


@dataclass
class FrozenEvaluationSeal:
    """References and hashes proving that evaluation cannot update a network."""

    modules: Mapping[str, nn.Module] = field(repr=False)
    hashes: Mapping[str, str]
    object_ids: Mapping[str, int]

    @classmethod
    def capture(cls, modules: Mapping[str, nn.Module]) -> "FrozenEvaluationSeal":
        if not modules:
            raise ValueError("at least one neural module must be sealed")
        copied = dict(modules)
        for name, module in copied.items():
            if not isinstance(module, nn.Module):
                raise TypeError(f"sealed object {name!r} is not a torch module")
            if module.training:
                raise RuntimeError(f"module {name!r} must be in eval mode before sealing")
            if any(parameter.requires_grad for parameter in module.parameters()):
                raise RuntimeError(f"module {name!r} still has trainable parameters")
        return cls(
            modules=copied,
            hashes={name: _module_hash(module) for name, module in copied.items()},
            object_ids={name: id(module) for name, module in copied.items()},
        )

    def assert_unchanged(self) -> None:
        for name, module in self.modules.items():
            if id(module) != self.object_ids[name]:
                raise AssertionError(f"sealed module {name!r} was replaced")
            if module.training:
                raise AssertionError(f"sealed module {name!r} left eval mode")
            if any(parameter.requires_grad for parameter in module.parameters()):
                raise AssertionError(f"sealed module {name!r} became trainable")
            if _module_hash(module) != self.hashes[name]:
                raise AssertionError(f"sealed module {name!r} changed during evaluation")


class FrozenActivationWriteWorldModel(nn.Module):
    r"""Strong generic planner baseline using only the frozen video model.

    At every predicted step this module reads the current categorical pixel
    context, recomputes the frozen observable Jacobian covectors, and extracts
    ``U`` with the sealed fit-only empirical tangent.  It applies ``U z`` at
    the same frozen-transformer block, then feeds back the predicted category.
    The pH core, latent renderer, inverse-effort head, simulator state, and
    physical command never enter the module.

    ``z`` is in the activation-write basis.  Conversion from an interface
    command to that basis is deliberately external and is fitted only by the
    locked post-freeze closed-form calibration protocol.
    """

    def __init__(
        self,
        encoder: WholeStreamFrozenEncoder,
        write_field: FrozenEmpiricalJacobianActivationPort,
        lens: FrozenSoftPixelActivationLens,
        probes: PixelChangeProbeBank,
    ) -> None:
        super().__init__()
        if not isinstance(write_field, FrozenEmpiricalJacobianActivationPort):
            raise TypeError(
                "activation evaluation requires FrozenEmpiricalJacobianActivationPort"
            )
        if not isinstance(probes, PixelChangeProbeBank):
            raise TypeError("activation evaluation requires a frozen pixel probe bank")
        if lens.backbone is not encoder.backbone:
            raise ValueError("activation baseline components must share one backbone")
        if lens.intervention_block != encoder.config.lens_block:
            raise ValueError("activation baseline lens blocks do not match")
        if (
            write_field.history_frames,
            write_field.patch_count,
            write_field.hidden_size,
        ) != encoder.activation_shape:
            raise ValueError("exact activation port has the wrong residual-stream shape")
        if write_field.port_size != probes.probe_size:
            raise ValueError("exact activation port and visual probe ranks differ")
        expected_probe_shape = (
            encoder.backbone.config.palette_size,
            encoder.backbone.config.image_size,
            encoder.backbone.config.image_size,
        )
        if tuple(probes.basis.shape[1:]) != expected_probe_shape:
            raise ValueError("activation probe bank has the wrong categorical image shape")
        write_field.assert_frozen_parameter_free()
        components = (encoder, write_field, lens, probes)
        if any(module.training for module in components):
            raise ValueError("activation baseline components must already be in eval mode")
        if any(
            parameter.requires_grad
            for module in components
            for parameter in module.parameters()
        ):
            raise ValueError("activation baseline components must already be frozen")
        self.encoder = encoder
        self.write_field = write_field
        self.lens = lens
        self.probes = probes
        self.horizons = tuple(lens.horizons)
        # Hash the unique conceptual components.  ``lens`` and ``encoder``
        # intentionally share the same backbone object.
        self._sealed_hashes = {
            "encoder": _module_hash(encoder),
            "writeField": _module_hash(write_field),
            "lens": _module_hash(lens),
            "probes": _module_hash(probes),
        }
        super().train(False)

    @property
    def port_size(self) -> int:
        return self.write_field.port_size

    def train(self, mode: bool = True):
        if mode:
            raise RuntimeError("the post-freeze activation world model cannot train")
        return super().train(False)

    def _assert_evaluation_flags(self) -> None:
        components = (self.encoder, self.write_field, self.lens, self.probes)
        if self.training or any(module.training for module in components):
            raise AssertionError("the activation world model left evaluation mode")
        if any(
            parameter.requires_grad
            for module in components
            for parameter in module.parameters()
        ):
            raise AssertionError("the activation world model became trainable")

    def assert_frozen_and_unchanged(self) -> None:
        components = {
            "encoder": self.encoder,
            "writeField": self.write_field,
            "lens": self.lens,
            "probes": self.probes,
        }
        self._assert_evaluation_flags()
        for name, module in components.items():
            if _module_hash(module) != self._sealed_hashes[name]:
                raise AssertionError(f"activation world-model component {name!r} changed")
        self.encoder.assert_backbone_frozen()

    def _extract_exact_port(
        self,
        pixel_contexts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the exact current-context prefix and empirical Jacobian port."""

        if pixel_contexts.ndim != 4 or pixel_contexts.is_floating_point():
            raise ValueError(
                "exact activation-port extraction requires categorical pixel contexts"
            )
        with torch.enable_grad():
            covectors = {
                horizon: value.detach()
                for horizon, value in activation_observable_covectors(
                    self.lens,
                    pixel_contexts,
                    self.probes,
                    horizons=self.horizons,
                    create_graph=False,
                ).items()
            }
        source_activation = self.encoder.prefix_activation(pixel_contexts).detach()
        extraction = self.write_field(covectors, source_activation)
        return source_activation, extraction.jacobian.write_basis

    def activation_state_rate_port(
        self,
        pixel_contexts: torch.Tensor,
        *,
        dt: float,
    ) -> torch.Tensor:
        r"""Return ``D_h E(h) U_J(h) / dt`` without a learned dynamics model.

        This Jacobian is used only to choose physical-blind calibration states
        and to solve the post-freeze constant ridge map.  Parameters remain
        frozen and no optimizer is involved.
        """

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self._assert_evaluation_flags()
        source_activation, write_basis = self._extract_exact_port(pixel_contexts)
        with torch.enable_grad():
            prefix = source_activation.detach()
            prefix.requires_grad_(True)
            state_jacobian = self.encoder.state_jacobian_from_activation(
                prefix, create_graph=False
            )
            basis = write_basis.detach().flatten(1, 3)
            result = torch.einsum("bna,bam->bnm", state_jacobian, basis) / dt
        return result.detach().float()

    @torch.no_grad()
    def forward(
        self,
        pixel_contexts: torch.Tensor,
        latent_effort_sequences: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw autoregressive logits ``[B,H,C,Y,X]``.

        The input sequence contains activation-write coefficients, not
        physical actions.  Each predicted categorical frame is fed back to the
        exact frozen backbone before the current-context Jacobian is extracted
        again for the next coefficient.
        """

        self._assert_evaluation_flags()
        if latent_effort_sequences.ndim != 3:
            raise ValueError("latent effort sequences must have shape [batch,horizon,port]")
        if latent_effort_sequences.shape[0] != pixel_contexts.shape[0]:
            raise ValueError("pixel contexts and effort sequences need the same batch")
        if latent_effort_sequences.shape[-1] != self.port_size:
            raise ValueError("latent effort sequence has the wrong port rank")
        if not latent_effort_sequences.is_floating_point() or not bool(
            torch.isfinite(latent_effort_sequences).all()
        ):
            raise ValueError("latent effort sequences must be finite floating tensors")
        if pixel_contexts.ndim != 4 or pixel_contexts.is_floating_point():
            raise ValueError("activation rollout requires categorical pixel contexts")
        current = pixel_contexts
        efforts = latent_effort_sequences.to(
            device=pixel_contexts.device,
            dtype=self.encoder.backbone.pixel_embedding.weight.dtype,
        )
        predicted_logits: list[torch.Tensor] = []
        for step in range(efforts.shape[1]):
            prefix, write_basis = self._extract_exact_port(current)
            residual_write = self.lens.residual_write(write_basis, efforts[:, step])
            logits = self.lens.soft_logits_from_prefix(
                prefix, residual_write=residual_write
            )[:, -1]
            predicted_logits.append(logits)
            next_pixels = logits.argmax(dim=1).to(dtype=current.dtype)
            current = torch.cat((current[:, 1:], next_pixels[:, None]), dim=1)
        if not predicted_logits:
            raise ValueError("latent effort horizon must be positive")
        result = torch.stack(predicted_logits, dim=1)
        return result


def activation_calibration_from_response_frame(
    structured_calibration: CalibrationResult | torch.Tensor,
    response_frame: PersistentOrthogonalPortFrame | torch.Tensor,
) -> torch.Tensor:
    r"""Analytically express a pH calibration in the activation-write basis.

    Training aligns ``D_lens`` with ``D_pH Q`` where ``Q`` is the persistent
    response frame.  Consequently ``z_lens = Q^T z_pH`` and the corresponding
    constant interface map is ``T_lens = Q^T T_pH``.  The registered runner
    uses a separate, equally budgeted activation-Jacobian calibration because
    it also absorbs finite lens/pH mismatch; this exact conversion remains an
    auditable no-query fallback and guards the basis convention.
    """

    transform = (
        structured_calibration.latent_from_interface
        if isinstance(structured_calibration, CalibrationResult)
        else structured_calibration
    )
    frame = response_frame.matrix() if hasattr(response_frame, "matrix") else response_frame
    if not isinstance(transform, torch.Tensor) or transform.ndim != 2:
        raise ValueError("structured calibration must be a matrix")
    if not isinstance(frame, torch.Tensor) or frame.ndim != 2:
        raise ValueError("response frame must be a matrix")
    if frame.shape[0] != frame.shape[1] or frame.shape[0] != transform.shape[0]:
        raise ValueError("response frame and calibration ranks differ")
    frame = frame.to(device=transform.device, dtype=transform.dtype)
    identity = torch.eye(frame.shape[0], device=frame.device, dtype=frame.dtype)
    if not torch.allclose(frame.T @ frame, identity, atol=2e-5, rtol=2e-5):
        raise ValueError("response frame is not orthogonal")
    return frame.T @ transform


class DynamicsEvaluationAdapter(nn.Module):
    """Give an already-frozen baseline the registered ``port/step`` API.

    The direct Poisson core already exposes both methods.  The matched
    unstructured baseline historically names its step ``integrate``; wrapping
    it here avoids changing or retraining that baseline.
    """

    def __init__(self, dynamics: nn.Module) -> None:
        super().__init__()
        self.dynamics = dynamics

    def port(self, state: torch.Tensor) -> torch.Tensor:
        return self.dynamics.port(state)  # type: ignore[attr-defined]

    def step(self, state: torch.Tensor, latent_effort: torch.Tensor) -> torch.Tensor:
        if hasattr(self.dynamics, "step"):
            return self.dynamics.step(state, latent_effort)  # type: ignore[attr-defined]
        if hasattr(self.dynamics, "integrate"):
            return self.dynamics.integrate(state, latent_effort)  # type: ignore[attr-defined]
        return self.dynamics(state, latent_effort)


def adapt_dynamics_for_evaluation(dynamics: nn.Module) -> nn.Module:
    """Return a zero-parameter API adapter when ``step`` is named differently."""

    if hasattr(dynamics, "port") and hasattr(dynamics, "step"):
        return dynamics
    if not hasattr(dynamics, "port"):
        raise TypeError("evaluation dynamics must expose a learned port(state)")
    return DynamicsEvaluationAdapter(dynamics)


def evaluation_system_from_direct_spec(specification: Any) -> EvaluationSystem:
    """Losslessly adapt ``direct_experiment_training.DirectSystemSpec``.

    Kept duck-typed to avoid a circular import in the training CLI.
    """

    name = str(specification.name)
    if name not in SYSTEMS:
        raise KeyError(f"no registered evaluation protocol for {name!r}")
    registered = SYSTEMS[name]
    port_size = int(specification.port_size)
    if port_size != registered.physical_action_size:
        raise ValueError("DirectSystemSpec port rank differs from the physical interface")
    dt = float(specification.dt)
    if not math.isclose(dt, registered.dt, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("DirectSystemSpec time step differs from the registered simulator")
    return registered


def capture_experiment_f_evaluation_seal(
    encoder: nn.Module,
    renderer: nn.Module,
    structured_dynamics: nn.Module,
    unstructured_dynamics: nn.Module,
    *,
    activation_world_model: nn.Module | None = None,
) -> FrozenEvaluationSeal:
    """Convenience constructor after callers permanently freeze all networks."""

    modules = {
        "visualEncoder": encoder,
        "pixelRenderer": renderer,
        "structuredDynamics": structured_dynamics,
        "unstructuredDynamics": unstructured_dynamics,
    }
    if activation_world_model is not None:
        modules["activationWorldModel"] = activation_world_model
    return FrozenEvaluationSeal.capture(modules)


@dataclass(frozen=True)
class PixelPlant:
    """Opaque physical deployment hooks; learned models see only ``context``."""

    clone_environment: Callable[[Any], Any] = field(repr=False)
    step_interface: Callable[[Any, PhysicalInterface, np.ndarray], None] = field(repr=False)
    append_observation: Callable[[torch.Tensor, Any], torch.Tensor] = field(repr=False)
    current_pixels: Callable[[Any], torch.Tensor] = field(repr=False)


@dataclass(frozen=True)
class ProbeCandidate:
    """An opaque environment paired with its pixel history, never a state label."""

    identifier: str
    context: torch.Tensor
    environment: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class DOptimalSelection:
    indices_by_axis: tuple[tuple[int, ...], ...]
    identifiers_by_axis: tuple[tuple[str, ...], ...]
    log_determinants_by_axis: tuple[tuple[float, ...], ...]
    paired_states_per_axis: int = PAIRED_CALIBRATION_STATES_PER_AXIS
    observed_response_count: int = 0
    selection_method: str = "single_model_d_optimal"
    selection_model_names: tuple[str, ...] = ()
    normalization_scales: tuple[float, ...] = ()
    candidate_pool_sha256: str = ""


def _update_selection_digest(
    digest: "hashlib._Hash",
    selection: DOptimalSelection,
) -> None:
    """Bind every pre-response selection fact into a physical-bank hash."""

    digest.update(str(selection.indices_by_axis).encode())
    digest.update(str(selection.identifiers_by_axis).encode())
    digest.update(str(selection.log_determinants_by_axis).encode())
    digest.update(str(selection.paired_states_per_axis).encode())
    digest.update(str(selection.observed_response_count).encode())
    digest.update(selection.selection_method.encode())
    digest.update(str(selection.selection_model_names).encode())
    digest.update(str(selection.normalization_scales).encode())
    digest.update(selection.candidate_pool_sha256.encode())


def _candidate_pool_sha256(candidates: Sequence[ProbeCandidate]) -> str:
    """Hash every full pre-probe pixel context, never an opaque environment."""

    digest = hashlib.sha256()
    for candidate in candidates:
        if type(candidate.identifier) is not str or not candidate.identifier:
            raise ValueError("calibration candidate identifier is invalid")
        context = candidate.context
        if (
            not isinstance(context, torch.Tensor)
            or context.requires_grad
            or context.grad_fn is not None
            or (context.is_floating_point() and not bool(torch.isfinite(context).all()))
        ):
            raise ValueError("calibration candidate context is invalid or attached")
        digest.update(candidate.identifier.encode("utf-8"))
        digest.update(str(context.dtype).encode("ascii"))
        digest.update(str(tuple(context.shape)).encode("ascii"))
        digest.update(_tensor_bytes(context))
    return digest.hexdigest()


def _calibration_response_evidence_sha256(
    selection: DOptimalSelection,
    responses: torch.Tensor,
    interface_name: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(interface_name.encode())
    _update_selection_digest(digest, selection)
    digest.update(str(responses.dtype).encode())
    digest.update(str(tuple(responses.shape)).encode())
    digest.update(_tensor_bytes(responses))
    return digest.hexdigest()


def _paired_response_bank_sha256(
    selection: DOptimalSelection,
    plus_contexts: torch.Tensor,
    minus_contexts: torch.Tensor,
    protocol: LinearInterfaceProtocol,
) -> str:
    digest = hashlib.sha256()
    digest.update(protocol.sha256.encode("ascii"))
    _update_selection_digest(digest, selection)
    for label, tensor in (("plus", plus_contexts), ("minus", minus_contexts)):
        digest.update(label.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


@dataclass(frozen=True)
class PairedCalibrationResponseBank:
    """The sole raw-pixel physical calibration evidence for one interface."""

    selection: DOptimalSelection
    interface_name: str
    plus_contexts: torch.Tensor = field(repr=False, compare=False)
    minus_contexts: torch.Tensor = field(repr=False, compare=False)
    physical_action_size: int
    paired_states_per_axis: int
    environment_steps: int
    protocol: LinearInterfaceProtocol
    evidence_sha256: str

    def __post_init__(self) -> None:
        expected_prefix = (
            self.physical_action_size,
            self.paired_states_per_axis,
        )
        if (
            not isinstance(self.plus_contexts, torch.Tensor)
            or not isinstance(self.minus_contexts, torch.Tensor)
            or self.plus_contexts.shape != self.minus_contexts.shape
            or self.plus_contexts.shape[:2] != expected_prefix
            or self.plus_contexts.ndim < 4
        ):
            raise ValueError("paired calibration response-bank tensor schema is invalid")
        if (
            self.plus_contexts.requires_grad
            or self.minus_contexts.requires_grad
            or self.plus_contexts.grad_fn is not None
            or self.minus_contexts.grad_fn is not None
        ):
            raise ValueError("paired calibration response bank must be detached")
        if (
            (self.plus_contexts.is_floating_point() and not bool(torch.isfinite(self.plus_contexts).all()))
            or (
                self.minus_contexts.is_floating_point()
                and not bool(torch.isfinite(self.minus_contexts).all())
            )
        ):
            raise ValueError("paired calibration response bank contains non-finite pixels")
        if (
            type(self.interface_name) is not str
            or not self.interface_name
            or type(self.physical_action_size) is not int
            or self.physical_action_size < 1
            or type(self.paired_states_per_axis) is not int
            or self.paired_states_per_axis != PAIRED_CALIBRATION_STATES_PER_AXIS
        ):
            raise ValueError("paired calibration response-bank metadata is invalid")
        expected_steps = 2 * self.paired_states_per_axis * self.physical_action_size
        if self.environment_steps != expected_steps:
            raise ValueError("paired calibration response-bank step count is invalid")
        observed = _paired_response_bank_sha256(
            self.selection,
            self.plus_contexts,
            self.minus_contexts,
            self.protocol,
        )
        if observed != self.evidence_sha256:
            raise ValueError("paired calibration response-bank hash mismatch")
        if (
            self.protocol.interface_name != self.interface_name
            or self.protocol.physical_action_size != self.physical_action_size
        ):
            raise ValueError("paired calibration response-bank protocol mismatch")


def _paired_heldout_bank_sha256(
    candidate_identifiers: tuple[str, ...],
    candidate_contexts_sha256: str,
    plus_contexts: torch.Tensor,
    minus_contexts: torch.Tensor,
    protocol: LinearInterfaceProtocol,
) -> str:
    digest = hashlib.sha256()
    digest.update(protocol.sha256.encode("ascii"))
    digest.update(str(candidate_identifiers).encode())
    digest.update(candidate_contexts_sha256.encode("ascii"))
    for label, tensor in (("plus", plus_contexts), ("minus", minus_contexts)):
        digest.update(label.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


@dataclass(frozen=True)
class PairedHeldoutResponseBank:
    """Unique raw-pixel held-out physical evidence for one interface."""

    candidate_identifiers: tuple[str, ...]
    candidate_contexts_sha256: str
    interface_name: str
    plus_contexts: torch.Tensor = field(repr=False, compare=False)
    minus_contexts: torch.Tensor = field(repr=False, compare=False)
    physical_action_size: int
    states_per_axis: int
    environment_steps: int
    protocol: LinearInterfaceProtocol
    evidence_sha256: str

    def __post_init__(self) -> None:
        expected_prefix = (self.physical_action_size, self.states_per_axis)
        if (
            len(self.candidate_identifiers) != self.states_per_axis
            or len(set(self.candidate_identifiers)) != self.states_per_axis
            or not isinstance(self.plus_contexts, torch.Tensor)
            or not isinstance(self.minus_contexts, torch.Tensor)
            or self.plus_contexts.shape != self.minus_contexts.shape
            or self.plus_contexts.shape[:2] != expected_prefix
            or self.plus_contexts.ndim < 4
        ):
            raise ValueError("paired held-out response-bank tensor schema is invalid")
        if (
            self.plus_contexts.requires_grad
            or self.minus_contexts.requires_grad
            or self.plus_contexts.grad_fn is not None
            or self.minus_contexts.grad_fn is not None
        ):
            raise ValueError("paired held-out response bank must be detached")
        if (
            (self.plus_contexts.is_floating_point() and not bool(torch.isfinite(self.plus_contexts).all()))
            or (
                self.minus_contexts.is_floating_point()
                and not bool(torch.isfinite(self.minus_contexts).all())
            )
        ):
            raise ValueError("paired held-out response bank contains non-finite pixels")
        if (
            type(self.interface_name) is not str
            or not self.interface_name
            or type(self.physical_action_size) is not int
            or self.physical_action_size < 1
            or type(self.states_per_axis) is not int
            or self.states_per_axis < 1
        ):
            raise ValueError("paired held-out response-bank metadata is invalid")
        if self.environment_steps != 2 * self.states_per_axis * self.physical_action_size:
            raise ValueError("paired held-out response-bank step count is invalid")
        observed = _paired_heldout_bank_sha256(
            self.candidate_identifiers,
            self.candidate_contexts_sha256,
            self.plus_contexts,
            self.minus_contexts,
            self.protocol,
        )
        if observed != self.evidence_sha256:
            raise ValueError("paired held-out response-bank hash mismatch")
        if re.fullmatch(r"[0-9a-f]{64}", self.candidate_contexts_sha256) is None:
            raise ValueError("paired held-out candidate-context hash is invalid")
        if (
            self.protocol.interface_name != self.interface_name
            or self.protocol.physical_action_size != self.physical_action_size
        ):
            raise ValueError("paired held-out response-bank protocol mismatch")


@dataclass(frozen=True)
class CalibrationResult:
    latent_from_interface: torch.Tensor
    selection: DOptimalSelection
    interface_name: str
    physical_action_size: int
    port_size: int
    paired_states_per_axis: int
    environment_steps: int
    gradient_updates: int
    ridge: float
    fit_relative_residual: float
    observed_state_responses: torch.Tensor = field(repr=False, compare=False)
    response_evidence_sha256: str
    encoded_response_sha256: str
    physical_protocol: LinearInterfaceProtocol
    additional_environment_steps: int
    responses_reused_from: str | None
    neural_hashes_before: Mapping[str, str]
    neural_hashes_after: Mapping[str, str]

    def __post_init__(self) -> None:
        responses = self.observed_state_responses
        if (
            not isinstance(responses, torch.Tensor)
            or responses.ndim != 3
            or responses.shape[0] != self.physical_action_size
            or responses.shape[1] != self.paired_states_per_axis
            or not responses.is_floating_point()
            or not bool(torch.isfinite(responses).all())
        ):
            raise ValueError("calibration response evidence has an invalid tensor schema")
        if not 0 <= self.additional_environment_steps <= self.environment_steps:
            raise ValueError("calibration additional-step accounting is invalid")
        if self.gradient_updates != 0:
            raise ValueError("analytic calibration cannot contain gradient updates")
        observed_hash = _calibration_response_evidence_sha256(
            self.selection,
            responses,
            self.interface_name,
        )
        if self.encoded_response_sha256 != observed_hash:
            raise ValueError("calibration encoded-response hash mismatch")
        if re.fullmatch(r"[0-9a-f]{64}", self.response_evidence_sha256) is None:
            raise ValueError("calibration physical response-bank hash is invalid")
        if (
            type(self.physical_protocol) is not LinearInterfaceProtocol
            or self.physical_protocol.interface_name != self.interface_name
            or self.physical_protocol.physical_action_size != self.physical_action_size
        ):
            raise ValueError("calibration physical protocol is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface_name,
            "matrixLatentFromInterface": self.latent_from_interface.detach().cpu().tolist(),
            "physicalActionSize": self.physical_action_size,
            "portSize": self.port_size,
            "pairedStatesPerAxis": self.paired_states_per_axis,
            "environmentSteps": self.environment_steps,
            "additionalEnvironmentSteps": self.additional_environment_steps,
            "gradientUpdates": self.gradient_updates,
            "ridge": self.ridge,
            "fitRelativeResidual": self.fit_relative_residual,
            "selectionObservedResponses": self.selection.observed_response_count,
            "selectionIdentifiersByAxis": self.selection.identifiers_by_axis,
            "responseEvidenceSha256": self.response_evidence_sha256,
            "encodedResponseSha256": self.encoded_response_sha256,
            "physicalProtocol": self.physical_protocol.to_dict(),
            "physicalProtocolSha256": self.physical_protocol.sha256,
            "responsesReusedFrom": self.responses_reused_from,
            "neuralHashesBefore": dict(self.neural_hashes_before),
            "neuralHashesAfter": dict(self.neural_hashes_after),
            "method": (
                "locked_shared_physical_responses_then_closed_form_constant_ridge"
                if self.responses_reused_from is not None
                else "locked_D_optimal_then_closed_form_constant_ridge"
            ),
        }


@dataclass(frozen=True)
class RealizabilityMetrics:
    mean_cosine: float
    axis_mean_cosines: tuple[float, ...]
    sign_agreement: float
    magnitude_r2: float
    axis_magnitude_r2: tuple[float, ...]
    response_cosines: tuple[float, ...]
    response_signs: tuple[bool, ...]
    actual_magnitudes: tuple[float, ...]
    predicted_magnitudes: tuple[float, ...]
    samples_per_axis: int
    environment_steps: int
    gradient_updates: int = 0
    response_evidence_sha256: str | None = None
    additional_environment_steps: int = 0
    responses_reused_from: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.additional_environment_steps <= self.environment_steps:
            raise ValueError("realizability additional-step accounting is invalid")
        if self.gradient_updates != 0:
            raise ValueError("realizability evaluation cannot contain gradient updates")
        if (
            self.response_evidence_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", self.response_evidence_sha256) is None
        ):
            raise ValueError("realizability response-bank hash is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "meanCalibratedResponseCosine": self.mean_cosine,
            "axisMeanResponseCosines": self.axis_mean_cosines,
            "commandedSignAgreement": self.sign_agreement,
            "centeredResponseMagnitudeR2": self.magnitude_r2,
            "axisMagnitudeR2": self.axis_magnitude_r2,
            "samplesPerPhysicalAxis": self.samples_per_axis,
            "environmentSteps": self.environment_steps,
            "additionalEnvironmentSteps": self.additional_environment_steps,
            "gradientUpdates": self.gradient_updates,
            "responseEvidenceSha256": self.response_evidence_sha256,
            "responsesReusedFrom": self.responses_reused_from,
        }


def _module_device(*objects: Any) -> torch.device:
    for value in objects:
        if isinstance(value, nn.Module):
            parameter = next(value.parameters(), None)
            if parameter is not None:
                return parameter.device
            buffer = next(value.buffers(), None)
            if buffer is not None:
                return buffer.device
    return torch.device("cpu")


def _module_storage_identities(*objects: Any) -> set[tuple[str, int | None, int]]:
    """Return storage identities, detecting aliases as well as tensor objects."""

    identities: set[tuple[str, int | None, int]] = set()
    for value in objects:
        if not isinstance(value, nn.Module):
            continue
        for tensor in (*tuple(value.parameters()), *tuple(value.buffers())):
            if tensor.numel() == 0 or tensor.device.type == "meta":
                continue
            identities.add(
                (
                    tensor.device.type,
                    tensor.device.index,
                    tensor.untyped_storage().data_ptr(),
                )
            )
    return identities


def _validate_primary_planner_isolation(
    encoder: nn.Module,
    renderer: nn.Module,
    structured_dynamics: nn.Module,
    unstructured_encoder: nn.Module,
    unstructured_renderer: nn.Module,
    unstructured_dynamics: nn.Module,
) -> None:
    """Allow only the single immutable backbone to cross planner boundaries."""

    structured_components = (encoder, renderer, structured_dynamics)
    independent_components = (
        unstructured_encoder,
        unstructured_renderer,
        unstructured_dynamics,
    )
    if {id(value) for value in structured_components} & {
        id(value) for value in independent_components
    }:
        raise ValueError("structured and unstructured planners share a module object")
    structured_backbone = getattr(encoder, "backbone", None)
    independent_backbone = getattr(unstructured_encoder, "backbone", None)
    allowed_shared_storage: set[tuple[str, int | None, int]] = set()
    if structured_backbone is not None or independent_backbone is not None:
        if (
            not isinstance(structured_backbone, nn.Module)
            or structured_backbone is not independent_backbone
            or structured_backbone.training
            or any(
                parameter.requires_grad
                for parameter in structured_backbone.parameters()
            )
        ):
            raise ValueError(
                "structured and unstructured encoders must share exactly one "
                "frozen authenticated backbone"
            )
        allowed_shared_storage = _module_storage_identities(structured_backbone)

        def private_encoder_modules(value: nn.Module) -> tuple[nn.Module, ...]:
            return tuple(
                child
                for name, child in value.named_children()
                if name != "backbone"
            )

        structured_downstream = (
            *private_encoder_modules(encoder),
            renderer,
            structured_dynamics,
        )
        independent_downstream = (
            *private_encoder_modules(unstructured_encoder),
            unstructured_renderer,
            unstructured_dynamics,
        )
        if (
            _module_storage_identities(*structured_downstream)
            | _module_storage_identities(*independent_downstream)
        ) & allowed_shared_storage:
            raise ValueError("a downstream planner tensor aliases the frozen backbone")
        if _module_storage_identities(
            *structured_downstream
        ) & _module_storage_identities(*independent_downstream):
            raise ValueError("structured and unstructured downstream tensors are shared")
    shared_storage = _module_storage_identities(
        *structured_components
    ) & _module_storage_identities(*independent_components)
    if not shared_storage.issubset(allowed_shared_storage):
        raise ValueError("structured and unstructured planners share tensor storage")


def _encode_pixels(encoder: Any, contexts: torch.Tensor) -> torch.Tensor:
    if hasattr(encoder, "encode") and callable(encoder.encode):
        encoded = encoder.encode(contexts)
    else:
        encoded = encoder(contexts)
    if not isinstance(encoded, torch.Tensor) or encoded.ndim != 2:
        raise ValueError("the frozen visual encoder must return [batch,state]")
    return encoded.float()


def _render_logits(renderer: Any, states: torch.Tensor) -> torch.Tensor:
    if hasattr(renderer, "render") and callable(renderer.render):
        logits = renderer.render(states)
    else:
        logits = renderer(states)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
        raise ValueError("the frozen renderer must return [batch,class,height,width]")
    return logits.float()


def _port_matrix(dynamics: Any, states: torch.Tensor) -> torch.Tensor:
    matrix = dynamics.port(states)
    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 3:
        raise ValueError("dynamics.port(state) must return [batch,state,port]")
    if matrix.shape[0] != states.shape[0] or matrix.shape[1] != states.shape[1]:
        raise ValueError("the learned port matrix has incompatible state axes")
    return matrix.float()


def _dynamics_step(dynamics: Any, state: torch.Tensor, latent_effort: torch.Tensor) -> torch.Tensor:
    next_state = dynamics.step(state, latent_effort)
    if not isinstance(next_state, torch.Tensor) or next_state.shape != state.shape:
        raise ValueError("dynamics.step must preserve the [batch,state] shape")
    return next_state.float()


@torch.no_grad()
def select_d_optimal_probe_states(
    encoder: Any,
    dynamics: Any,
    candidates: Sequence[ProbeCandidate],
    system: EvaluationSystem,
    *,
    seal: FrozenEvaluationSeal,
    ridge: float = RIDGE_COEFFICIENT,
    batch_size: int = 64,
) -> DOptimalSelection:
    """Choose all probe states before observing any physical response.

    The only candidate attribute read here is ``context``.  Greedy D-optimal
    selection maximizes the log determinant of ``sum B(x)^T B(x)``.  The same
    rule is run independently for each interface axis and always returns four
    distinct candidates per axis.
    """

    seal.assert_unchanged()
    if len(candidates) < PAIRED_CALIBRATION_STATES_PER_AXIS:
        raise ValueError("the pixel-only candidate pool needs at least four states")
    if ridge <= 0.0 or batch_size < 1:
        raise ValueError("ridge and batch_size must be positive")
    contexts = torch.stack([candidate.context for candidate in candidates])
    device = _module_device(encoder, dynamics)
    states = []
    for start in range(0, len(candidates), batch_size):
        states.append(_encode_pixels(encoder, contexts[start : start + batch_size].to(device)))
    encoded = torch.cat(states)
    ports = _port_matrix(dynamics, encoded)
    port_size = ports.shape[-1]
    if port_size != system.physical_action_size:
        raise ValueError(
            "Experiment F requires equal latent-port and physical-interface ranks"
        )

    gram = torch.einsum("bnp,bnq->bpq", ports.double(), ports.double())
    selections: list[tuple[int, ...]] = []
    identifiers: list[tuple[str, ...]] = []
    trajectories: list[tuple[float, ...]] = []
    for _axis in range(system.physical_action_size):
        information = ridge * torch.eye(port_size, device=ports.device, dtype=torch.float64)
        available = list(range(len(candidates)))
        chosen: list[int] = []
        log_determinants: list[float] = []
        for _ in range(PAIRED_CALIBRATION_STATES_PER_AXIS):
            scores = []
            for index in available:
                sign, value = torch.linalg.slogdet(information + gram[index])
                score = value if bool(sign > 0) else value.new_tensor(float("-inf"))
                scores.append(float(score))
            best_position = max(range(len(scores)), key=lambda item: (scores[item], -available[item]))
            best_index = available.pop(best_position)
            chosen.append(best_index)
            information = information + gram[best_index]
            log_determinants.append(scores[best_position])
        selections.append(tuple(chosen))
        identifiers.append(tuple(candidates[index].identifier for index in chosen))
        trajectories.append(tuple(log_determinants))

    seal.assert_unchanged()
    return DOptimalSelection(
        indices_by_axis=tuple(selections),
        identifiers_by_axis=tuple(identifiers),
        log_determinants_by_axis=tuple(trajectories),
        candidate_pool_sha256=_candidate_pool_sha256(candidates),
    )


@torch.no_grad()
def select_d_optimal_activation_probe_states(
    activation_world_model: FrozenActivationWriteWorldModel,
    candidates: Sequence[ProbeCandidate],
    system: EvaluationSystem,
    *,
    seal: FrozenEvaluationSeal,
    ridge: float = RIDGE_COEFFICIENT,
    batch_size: int = 4,
) -> DOptimalSelection:
    """Pixel-only D-optimal selection for the generic activation baseline.

    Candidate ranking uses only ``D_h E U / dt`` from frozen modules.  Every
    state is selected before any physical response is opened.
    """

    seal.assert_unchanged()
    activation_world_model.assert_frozen_and_unchanged()
    if len(candidates) < PAIRED_CALIBRATION_STATES_PER_AXIS:
        raise ValueError("the pixel-only candidate pool needs at least four states")
    if ridge <= 0.0 or batch_size < 1:
        raise ValueError("ridge and batch_size must be positive")
    contexts = torch.stack([candidate.context for candidate in candidates])
    device = _module_device(activation_world_model)
    ports = []
    for start in range(0, len(candidates), batch_size):
        ports.append(
            activation_world_model.activation_state_rate_port(
                contexts[start : start + batch_size].to(device), dt=system.dt
            )
        )
    port = torch.cat(ports)
    if port.shape[-1] != system.physical_action_size:
        raise ValueError(
            "Experiment F requires equal activation-write and physical-interface ranks"
        )
    gram = torch.einsum("bnp,bnq->bpq", port.double(), port.double())
    selections: list[tuple[int, ...]] = []
    identifiers: list[tuple[str, ...]] = []
    trajectories: list[tuple[float, ...]] = []
    for _axis in range(system.physical_action_size):
        information = ridge * torch.eye(
            port.shape[-1], device=port.device, dtype=torch.float64
        )
        available = list(range(len(candidates)))
        chosen: list[int] = []
        log_determinants: list[float] = []
        for _ in range(PAIRED_CALIBRATION_STATES_PER_AXIS):
            scores = []
            for index in available:
                sign, value = torch.linalg.slogdet(information + gram[index])
                score = value if bool(sign > 0) else value.new_tensor(float("-inf"))
                scores.append(float(score))
            best_position = max(
                range(len(scores)), key=lambda item: (scores[item], -available[item])
            )
            best_index = available.pop(best_position)
            chosen.append(best_index)
            information = information + gram[best_index]
            log_determinants.append(scores[best_position])
        selections.append(tuple(chosen))
        identifiers.append(tuple(candidates[index].identifier for index in chosen))
        trajectories.append(tuple(log_determinants))
    seal.assert_unchanged()
    activation_world_model.assert_frozen_and_unchanged()
    return DOptimalSelection(
        indices_by_axis=tuple(selections),
        identifiers_by_axis=tuple(identifiers),
        log_determinants_by_axis=tuple(trajectories),
        selection_method="activation_d_optimal",
        candidate_pool_sha256=_candidate_pool_sha256(candidates),
    )


@torch.no_grad()
def select_shared_maximin_probe_states(
    model_pairs: Mapping[str, tuple[Any, Any]],
    activation_world_model: FrozenActivationWriteWorldModel,
    candidates: Sequence[ProbeCandidate],
    system: EvaluationSystem,
    *,
    seal: FrozenEvaluationSeal,
    ridge: float = RIDGE_COEFFICIENT,
    batch_size: int = 64,
    activation_batch_size: int = 4,
) -> DOptimalSelection:
    r"""Choose one response-blind calibration design fairly for every model.

    For each frozen dynamics model the candidate Gram is ``B(x)^T B(x)``; for
    the generic activation planner it is ``A_U(x)^T A_U(x)`` with
    ``A_U = D_h E U_J(context) / dt``. Each model's Grams are divided by its
    own mean trace, removing arbitrary latent units. Greedy selection then maximizes
    the *worst* cumulative log determinant across all registered models.

    The algorithm reads only candidate pixels and frozen Jacobians.  It chooses
    exactly four common indices before a physical response is collected, and
    reuses those exact indices for every physical interface axis.
    """

    seal.assert_unchanged()
    activation_world_model.assert_frozen_and_unchanged()
    if type(model_pairs) is not dict or not model_pairs:
        raise ValueError("shared max-min selection requires a plain nonempty model map")
    if "activation" in model_pairs:
        raise ValueError("the reserved activation model name is duplicated")
    if any(type(name) is not str or not name for name in model_pairs):
        raise ValueError("shared max-min selection model names are invalid")
    if len(candidates) < PAIRED_CALIBRATION_STATES_PER_AXIS:
        raise ValueError("the pixel-only candidate pool needs at least four states")
    if len({candidate.identifier for candidate in candidates}) != len(candidates):
        raise ValueError("calibration candidate identifiers must be globally unique")
    if (
        not math.isfinite(ridge)
        or ridge <= 0.0
        or type(batch_size) is not int
        or batch_size < 1
        or type(activation_batch_size) is not int
        or activation_batch_size < 1
    ):
        raise ValueError("shared max-min ridge and batch sizes must be positive")

    contexts = torch.stack([candidate.context for candidate in candidates])
    if contexts.requires_grad or contexts.grad_fn is not None:
        raise ValueError("calibration candidate pixels must be detached")
    grams: dict[str, torch.Tensor] = {}
    for name in sorted(model_pairs):
        pair = model_pairs[name]
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError(f"model pair {name!r} must contain encoder and dynamics")
        encoder, dynamics = pair
        device = _module_device(encoder, dynamics)
        encoded_chunks = []
        for start in range(0, len(candidates), batch_size):
            encoded_chunks.append(
                _encode_pixels(
                    encoder, contexts[start : start + batch_size].to(device)
                )
            )
        ports = _port_matrix(dynamics, torch.cat(encoded_chunks))
        if ports.shape[-1] != system.physical_action_size:
            raise ValueError(
                f"model {name!r} port rank differs from the physical interface"
            )
        grams[name] = torch.einsum(
            "bnp,bnq->bpq", ports.double(), ports.double()
        ).detach().cpu()

    activation_device = _module_device(activation_world_model)
    activation_ports = []
    for start in range(0, len(candidates), activation_batch_size):
        activation_ports.append(
            activation_world_model.activation_state_rate_port(
                contexts[start : start + activation_batch_size].to(activation_device),
                dt=system.dt,
            )
        )
    activation_port = torch.cat(activation_ports)
    if activation_port.shape[-1] != system.physical_action_size:
        raise ValueError(
            "activation-write rank differs from the physical interface"
        )
    grams["activation"] = torch.einsum(
        "bnp,bnq->bpq", activation_port.double(), activation_port.double()
    ).detach().cpu()

    model_names = tuple(sorted(grams))
    normalized_grams: dict[str, torch.Tensor] = {}
    scales: list[float] = []
    for name in model_names:
        gram = grams[name]
        if gram.shape != (
            len(candidates),
            system.physical_action_size,
            system.physical_action_size,
        ) or not bool(torch.isfinite(gram).all()):
            raise ValueError(f"model {name!r} produced an invalid calibration Gram bank")
        scale = float(
            gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1).mean()
            / system.physical_action_size
        )
        if not math.isfinite(scale) or scale <= 1e-12:
            raise ValueError(f"model {name!r} has a degenerate calibration response field")
        normalized_grams[name] = gram / scale
        scales.append(scale)

    identity = torch.eye(system.physical_action_size, dtype=torch.float64)
    information = {
        name: ridge * identity.clone()
        for name in model_names
    }
    available = list(range(len(candidates)))
    chosen: list[int] = []
    maximin_log_determinants: list[float] = []
    for _ in range(PAIRED_CALIBRATION_STATES_PER_AXIS):
        candidate_scores: list[float] = []
        for index in available:
            per_model: list[float] = []
            for name in model_names:
                sign, value = torch.linalg.slogdet(
                    information[name] + normalized_grams[name][index]
                )
                per_model.append(float(value) if bool(sign > 0) else float("-inf"))
            candidate_scores.append(min(per_model))
        best_position = max(
            range(len(candidate_scores)),
            key=lambda position: (candidate_scores[position], -available[position]),
        )
        best_index = available.pop(best_position)
        if not math.isfinite(candidate_scores[best_position]):
            raise ValueError("shared max-min calibration design is singular")
        chosen.append(best_index)
        maximin_log_determinants.append(candidate_scores[best_position])
        for name in model_names:
            information[name] = information[name] + normalized_grams[name][best_index]

    common_indices = tuple(chosen)
    common_identifiers = tuple(candidates[index].identifier for index in chosen)
    common_trajectory = tuple(maximin_log_determinants)
    seal.assert_unchanged()
    activation_world_model.assert_frozen_and_unchanged()
    return DOptimalSelection(
        indices_by_axis=tuple(
            common_indices for _ in range(system.physical_action_size)
        ),
        identifiers_by_axis=tuple(
            common_identifiers for _ in range(system.physical_action_size)
        ),
        log_determinants_by_axis=tuple(
            common_trajectory for _ in range(system.physical_action_size)
        ),
        selection_method="shared_maximin_normalized_d_optimal",
        selection_model_names=model_names,
        normalization_scales=tuple(scales),
        candidate_pool_sha256=_candidate_pool_sha256(candidates),
    )


def _validate_locked_selection(
    selection: DOptimalSelection,
    candidates: Sequence[ProbeCandidate],
    action_size: int,
) -> None:
    if selection.observed_response_count != 0:
        raise ValueError("D-optimal selection must precede every response query")
    if selection.paired_states_per_axis != PAIRED_CALIBRATION_STATES_PER_AXIS:
        raise ValueError("calibration is locked to exactly four paired states per axis")
    if (
        len(selection.indices_by_axis) != action_size
        or len(selection.identifiers_by_axis) != action_size
        or len(selection.log_determinants_by_axis) != action_size
    ):
        raise ValueError("selection does not cover every physical interface axis")
    if len({candidate.identifier for candidate in candidates}) != len(candidates):
        raise ValueError("calibration candidate identifiers must be globally unique")
    if (
        re.fullmatch(r"[0-9a-f]{64}", selection.candidate_pool_sha256) is None
        or selection.candidate_pool_sha256 != _candidate_pool_sha256(candidates)
    ):
        raise ValueError("calibration candidate pixel-pool hash changed")
    allowed_methods = {
        "single_model_d_optimal",
        "activation_d_optimal",
        "shared_maximin_normalized_d_optimal",
    }
    if selection.selection_method not in allowed_methods:
        raise ValueError("calibration selection method is not registered")
    for axis, axis_indices in enumerate(selection.indices_by_axis):
        if len(axis_indices) != PAIRED_CALIBRATION_STATES_PER_AXIS:
            raise ValueError("every axis must contain exactly four paired states")
        if len(set(axis_indices)) != len(axis_indices):
            raise ValueError("probe states must be distinct within an axis")
        if any(index < 0 or index >= len(candidates) for index in axis_indices):
            raise IndexError("a selected candidate is outside the candidate pool")
        expected_identifiers = tuple(candidates[index].identifier for index in axis_indices)
        if selection.identifiers_by_axis[axis] != expected_identifiers:
            raise ValueError("selected candidate identifiers differ from their indices")
        log_determinants = selection.log_determinants_by_axis[axis]
        if (
            len(log_determinants) != PAIRED_CALIBRATION_STATES_PER_AXIS
            or any(not math.isfinite(float(value)) for value in log_determinants)
        ):
            raise ValueError("selection log-determinant evidence is invalid")
    if selection.selection_method == "shared_maximin_normalized_d_optimal":
        if (
            any(indices != selection.indices_by_axis[0] for indices in selection.indices_by_axis)
            or any(
                identifiers != selection.identifiers_by_axis[0]
                for identifiers in selection.identifiers_by_axis
            )
            or len(selection.selection_model_names) < 3
            or tuple(sorted(set(selection.selection_model_names)))
            != selection.selection_model_names
            or len(selection.normalization_scales)
            != len(selection.selection_model_names)
            or any(
                not math.isfinite(float(value)) or float(value) <= 0.0
                for value in selection.normalization_scales
            )
        ):
            raise ValueError("shared max-min selection provenance is invalid")
    elif selection.selection_model_names or selection.normalization_scales:
        raise ValueError("single-model selection contains forged joint provenance")


@torch.no_grad()
def _paired_response(
    encoder: Any,
    plant: PixelPlant,
    candidate: ProbeCandidate,
    interface: PhysicalInterface,
    command: np.ndarray,
    *,
    dt: float,
    amplitude: float,
    device: torch.device,
) -> torch.Tensor:
    plus_environment = plant.clone_environment(candidate.environment)
    minus_environment = plant.clone_environment(candidate.environment)
    plant.step_interface(plus_environment, interface, command)
    plant.step_interface(minus_environment, interface, -command)
    plus_context = plant.append_observation(candidate.context, plus_environment)
    minus_context = plant.append_observation(candidate.context, minus_environment)
    pair = torch.stack((plus_context, minus_context)).to(device)
    encoded = _encode_pixels(encoder, pair)
    return (encoded[0] - encoded[1]) / (2.0 * amplitude * dt)


@torch.no_grad()
def collect_paired_calibration_response_bank(
    plant: PixelPlant,
    candidates: Sequence[ProbeCandidate],
    selection: DOptimalSelection,
    system: EvaluationSystem,
    interface: PhysicalInterface,
    *,
    seal: FrozenEvaluationSeal,
) -> PairedCalibrationResponseBank:
    """Execute the only ``4 x axes`` physical calibration pairs.

    The bank stores raw before/after pixel histories, not an encoder-specific
    latent response.  Every registered model must re-encode this same sealed
    bank; no model-specific environment query is permitted.
    """

    seal.assert_unchanged()
    _validate_locked_selection(selection, candidates, system.physical_action_size)
    protocol = linear_interface_protocol(system, interface)
    if system.probe_amplitude > protocol.common_interface_command_bound + 1e-12:
        raise ValueError("calibration probe exceeds the registered linear command box")
    plus_by_axis: list[list[torch.Tensor]] = [
        [] for _ in range(system.physical_action_size)
    ]
    minus_by_axis: list[list[torch.Tensor]] = [
        [] for _ in range(system.physical_action_size)
    ]
    for axis, selected_indices in enumerate(selection.indices_by_axis):
        command = np.zeros(system.physical_action_size, dtype=np.float64)
        command[axis] = system.probe_amplitude
        for index in selected_indices:
            candidate = candidates[index]
            plus_environment = plant.clone_environment(candidate.environment)
            minus_environment = plant.clone_environment(candidate.environment)
            plant.step_interface(plus_environment, interface, command)
            plant.step_interface(minus_environment, interface, -command)
            plus_by_axis[axis].append(
                plant.append_observation(candidate.context, plus_environment)
                .detach()
                .clone()
            )
            minus_by_axis[axis].append(
                plant.append_observation(candidate.context, minus_environment)
                .detach()
                .clone()
            )
    plus = torch.stack([torch.stack(values) for values in plus_by_axis])
    minus = torch.stack([torch.stack(values) for values in minus_by_axis])
    evidence_hash = _paired_response_bank_sha256(
        selection, plus, minus, protocol
    )
    seal.assert_unchanged()
    return PairedCalibrationResponseBank(
        selection=selection,
        interface_name=interface.name,
        plus_contexts=plus,
        minus_contexts=minus,
        physical_action_size=system.physical_action_size,
        paired_states_per_axis=PAIRED_CALIBRATION_STATES_PER_AXIS,
        environment_steps=(
            2 * PAIRED_CALIBRATION_STATES_PER_AXIS * system.physical_action_size
        ),
        protocol=protocol,
        evidence_sha256=evidence_hash,
    )


@torch.no_grad()
def collect_paired_heldout_response_bank(
    plant: PixelPlant,
    candidates: Sequence[ProbeCandidate],
    system: EvaluationSystem,
    interface: PhysicalInterface,
    *,
    seal: FrozenEvaluationSeal,
    states_per_axis: int = 128,
) -> PairedHeldoutResponseBank:
    """Collect one raw-pixel held-out bank shared by every model."""

    seal.assert_unchanged()
    if states_per_axis < 1 or len(candidates) < states_per_axis:
        raise ValueError("insufficient held-out pixel candidates")
    protocol = linear_interface_protocol(system, interface)
    if system.probe_amplitude > protocol.common_interface_command_bound + 1e-12:
        raise ValueError("held-out probe exceeds the registered linear command box")
    selected = tuple(candidates[:states_per_axis])
    identifiers = tuple(candidate.identifier for candidate in selected)
    candidate_contexts_sha256 = _candidate_pool_sha256(selected)
    if len(set(identifiers)) != states_per_axis:
        raise ValueError("held-out response candidates must have unique identifiers")
    plus_by_axis: list[list[torch.Tensor]] = [
        [] for _ in range(system.physical_action_size)
    ]
    minus_by_axis: list[list[torch.Tensor]] = [
        [] for _ in range(system.physical_action_size)
    ]
    for axis in range(system.physical_action_size):
        command = np.zeros(system.physical_action_size, dtype=np.float64)
        command[axis] = system.probe_amplitude
        for candidate in selected:
            plus_environment = plant.clone_environment(candidate.environment)
            minus_environment = plant.clone_environment(candidate.environment)
            plant.step_interface(plus_environment, interface, command)
            plant.step_interface(minus_environment, interface, -command)
            plus_by_axis[axis].append(
                plant.append_observation(candidate.context, plus_environment)
                .detach()
                .clone()
            )
            minus_by_axis[axis].append(
                plant.append_observation(candidate.context, minus_environment)
                .detach()
                .clone()
            )
    plus = torch.stack([torch.stack(values) for values in plus_by_axis])
    minus = torch.stack([torch.stack(values) for values in minus_by_axis])
    evidence_hash = _paired_heldout_bank_sha256(
        identifiers, candidate_contexts_sha256, plus, minus, protocol
    )
    seal.assert_unchanged()
    return PairedHeldoutResponseBank(
        candidate_identifiers=identifiers,
        candidate_contexts_sha256=candidate_contexts_sha256,
        interface_name=interface.name,
        plus_contexts=plus,
        minus_contexts=minus,
        physical_action_size=system.physical_action_size,
        states_per_axis=states_per_axis,
        environment_steps=2 * states_per_axis * system.physical_action_size,
        protocol=protocol,
        evidence_sha256=evidence_hash,
    )


@torch.no_grad()
def fit_interface_calibration_from_response_bank(
    encoder: Any,
    dynamics: Any,
    candidates: Sequence[ProbeCandidate],
    response_bank: PairedCalibrationResponseBank,
    system: EvaluationSystem,
    interface: PhysicalInterface,
    *,
    seal: FrozenEvaluationSeal,
    ridge: float = RIDGE_COEFFICIENT,
    model_name: str,
) -> CalibrationResult:
    """Fit one model from the shared raw-pixel bank with zero physical steps."""

    seal.assert_unchanged()
    selection = response_bank.selection
    _validate_locked_selection(selection, candidates, system.physical_action_size)
    protocol = linear_interface_protocol(system, interface)
    if ridge <= 0.0 or not model_name:
        raise ValueError("ridge and model_name must be valid")
    if (
        response_bank.interface_name != interface.name
        or response_bank.physical_action_size != system.physical_action_size
        or response_bank.paired_states_per_axis
        != PAIRED_CALIBRATION_STATES_PER_AXIS
        or response_bank.protocol != protocol
    ):
        raise ValueError("shared response bank belongs to another protocol")
    if _paired_response_bank_sha256(
        selection,
        response_bank.plus_contexts,
        response_bank.minus_contexts,
        protocol,
    ) != response_bank.evidence_sha256:
        raise ValueError("shared physical response bank was modified")
    device = _module_device(encoder, dynamics)
    design_by_axis: list[list[torch.Tensor]] = [
        [] for _ in range(system.physical_action_size)
    ]
    response_by_axis: list[list[torch.Tensor]] = [
        [] for _ in range(system.physical_action_size)
    ]
    for axis, selected_indices in enumerate(selection.indices_by_axis):
        for position, index in enumerate(selected_indices):
            candidate = candidates[index]
            state = _encode_pixels(encoder, candidate.context[None].to(device))
            design_by_axis[axis].append(_port_matrix(dynamics, state)[0].double())
            pair = torch.stack(
                (
                    response_bank.plus_contexts[axis, position],
                    response_bank.minus_contexts[axis, position],
                )
            ).to(device)
            encoded = _encode_pixels(encoder, pair).double()
            response_by_axis[axis].append(
                (encoded[0] - encoded[1])
                / (2.0 * system.probe_amplitude * system.dt)
            )
    port_size = design_by_axis[0][0].shape[-1]
    if port_size != system.physical_action_size:
        raise ValueError("a square constant port-basis calibration is required")
    identity = torch.eye(port_size, device=device, dtype=torch.float64)
    columns = []
    predicted_chunks = []
    response_chunks = []
    for axis in range(system.physical_action_size):
        design = torch.cat(design_by_axis[axis], dim=0)
        response = torch.stack(response_by_axis[axis]).reshape(-1)
        column = torch.linalg.solve(
            design.T @ design + ridge * identity,
            design.T @ response,
        )
        columns.append(column)
        predicted_chunks.append(design @ column)
        response_chunks.append(response)
    calibration = torch.stack(columns, dim=1).float()
    predicted = torch.cat(predicted_chunks)
    observed = torch.cat(response_chunks)
    observed_response_tensor = torch.stack(
        [torch.stack(axis_responses) for axis_responses in response_by_axis]
    ).detach()
    relative_residual = float(
        torch.linalg.vector_norm(predicted - observed)
        / torch.linalg.vector_norm(observed).clamp_min(1e-12)
    )
    encoded_hash = _calibration_response_evidence_sha256(
        selection, observed_response_tensor, interface.name
    )
    seal.assert_unchanged()
    return CalibrationResult(
        latent_from_interface=calibration.detach(),
        selection=selection,
        interface_name=interface.name,
        physical_action_size=system.physical_action_size,
        port_size=port_size,
        paired_states_per_axis=PAIRED_CALIBRATION_STATES_PER_AXIS,
        environment_steps=response_bank.environment_steps,
        gradient_updates=0,
        ridge=ridge,
        fit_relative_residual=relative_residual,
        observed_state_responses=observed_response_tensor,
        response_evidence_sha256=response_bank.evidence_sha256,
        encoded_response_sha256=encoded_hash,
        physical_protocol=protocol,
        additional_environment_steps=0,
        responses_reused_from=f"shared-interface-bank:{model_name}",
        neural_hashes_before=dict(seal.hashes),
        neural_hashes_after={
            name: _module_hash(module) for name, module in seal.modules.items()
        },
    )


@torch.no_grad()
def calibrate_interface_after_freeze(
    encoder: Any,
    dynamics: Any,
    plant: PixelPlant,
    candidates: Sequence[ProbeCandidate],
    selection: DOptimalSelection,
    system: EvaluationSystem,
    interface: PhysicalInterface,
    *,
    seal: FrozenEvaluationSeal,
    ridge: float = RIDGE_COEFFICIENT,
) -> CalibrationResult:
    """Fit the sole constant matrix ``T`` with exactly ``4 x axes`` pairs."""

    seal.assert_unchanged()
    _validate_locked_selection(selection, candidates, system.physical_action_size)
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")
    matrix = interface.matrix()
    if matrix.shape != (system.physical_action_size, system.physical_action_size):
        raise ValueError("interface rank differs from the physical action rank")
    protocol = linear_interface_protocol(system, interface)
    if system.probe_amplitude > protocol.common_interface_command_bound + 1e-12:
        raise ValueError("calibration probe exceeds the registered linear command box")
    device = _module_device(encoder, dynamics)
    design_by_axis: list[list[torch.Tensor]] = [
        [] for _ in range(system.physical_action_size)
    ]
    response_by_axis: list[list[torch.Tensor]] = [
        [] for _ in range(system.physical_action_size)
    ]
    amplitude = system.probe_amplitude
    for axis, selected_indices in enumerate(selection.indices_by_axis):
        for index in selected_indices:
            candidate = candidates[index]
            state = _encode_pixels(encoder, candidate.context[None].to(device))
            design_by_axis[axis].append(_port_matrix(dynamics, state)[0].double())
            command = np.zeros(system.physical_action_size, dtype=np.float64)
            command[axis] = amplitude
            response_by_axis[axis].append(
                _paired_response(
                    encoder,
                    plant,
                    candidate,
                    interface,
                    command,
                    dt=system.dt,
                    amplitude=amplitude,
                    device=device,
                ).double()
            )

    port_size = design_by_axis[0][0].shape[-1]
    if port_size != system.physical_action_size:
        raise ValueError("a square constant port-basis calibration is required")
    identity = torch.eye(port_size, device=device, dtype=torch.float64)
    columns = []
    predicted_chunks = []
    response_chunks = []
    for axis in range(system.physical_action_size):
        design = torch.cat(design_by_axis[axis], dim=0)
        response = torch.stack(response_by_axis[axis]).reshape(-1)
        column = torch.linalg.solve(
            design.T @ design + ridge * identity,
            design.T @ response,
        )
        columns.append(column)
        predicted_chunks.append(design @ column)
        response_chunks.append(response)
    calibration = torch.stack(columns, dim=1).float()
    predicted = torch.cat(predicted_chunks)
    observed = torch.cat(response_chunks)
    observed_response_tensor = torch.stack(
        [torch.stack(axis_responses) for axis_responses in response_by_axis]
    ).detach()
    relative_residual = float(
        torch.linalg.vector_norm(predicted - observed)
        / torch.linalg.vector_norm(observed).clamp_min(1e-12)
    )
    seal.assert_unchanged()
    return CalibrationResult(
        latent_from_interface=calibration.detach(),
        selection=selection,
        interface_name=interface.name,
        physical_action_size=system.physical_action_size,
        port_size=port_size,
        paired_states_per_axis=PAIRED_CALIBRATION_STATES_PER_AXIS,
        environment_steps=2
        * PAIRED_CALIBRATION_STATES_PER_AXIS
        * system.physical_action_size,
        gradient_updates=0,
        ridge=ridge,
        fit_relative_residual=relative_residual,
        observed_state_responses=observed_response_tensor,
        response_evidence_sha256=_calibration_response_evidence_sha256(
            selection, observed_response_tensor, interface.name
        ),
        encoded_response_sha256=_calibration_response_evidence_sha256(
            selection, observed_response_tensor, interface.name
        ),
        physical_protocol=protocol,
        additional_environment_steps=(
            2 * PAIRED_CALIBRATION_STATES_PER_AXIS * system.physical_action_size
        ),
        responses_reused_from=None,
        neural_hashes_before=dict(seal.hashes),
        neural_hashes_after={name: _module_hash(module) for name, module in seal.modules.items()},
    )


@torch.no_grad()
def calibrate_activation_interface_after_freeze(
    activation_world_model: FrozenActivationWriteWorldModel,
    candidates: Sequence[ProbeCandidate],
    response_bank: PairedCalibrationResponseBank,
    system: EvaluationSystem,
    interface: PhysicalInterface,
    *,
    seal: FrozenEvaluationSeal,
    ridge: float = RIDGE_COEFFICIENT,
) -> CalibrationResult:
    r"""Refit the write map from the unique shared physical response bank.

    The design matrix is ``D_h E U / dt``.  Targets are obtained by re-encoding
    the bank's exact same raw ``+/-`` pixel histories.  No environment is
    accepted by this API, making an extra probe structurally impossible.
    """

    seal.assert_unchanged()
    activation_world_model.assert_frozen_and_unchanged()
    selection = response_bank.selection
    _validate_locked_selection(selection, candidates, system.physical_action_size)
    protocol = linear_interface_protocol(system, interface)
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")
    if (
        response_bank.interface_name != interface.name
        or response_bank.physical_action_size != system.physical_action_size
        or response_bank.paired_states_per_axis
        != PAIRED_CALIBRATION_STATES_PER_AXIS
        or response_bank.protocol != protocol
    ):
        raise ValueError("shared activation response bank is invalid")
    matrix = interface.matrix()
    if matrix.shape != (system.physical_action_size, system.physical_action_size):
        raise ValueError("interface rank differs from the physical action rank")
    device = _module_device(activation_world_model)
    designs: list[list[torch.Tensor]] = [
        [] for _ in range(system.physical_action_size)
    ]
    responses: list[list[torch.Tensor]] = [
        [] for _ in range(system.physical_action_size)
    ]
    for axis, selected_indices in enumerate(selection.indices_by_axis):
        for position, index in enumerate(selected_indices):
            candidate = candidates[index]
            designs[axis].append(
                activation_world_model.activation_state_rate_port(
                    candidate.context[None].to(device), dt=system.dt
                )[0].double()
            )
            pair = torch.stack(
                (
                    response_bank.plus_contexts[axis, position],
                    response_bank.minus_contexts[axis, position],
                )
            ).to(device)
            encoded = _encode_pixels(activation_world_model.encoder, pair).double()
            responses[axis].append(
                (encoded[0] - encoded[1])
                / (2.0 * system.probe_amplitude * system.dt)
            )
    port_size = designs[0][0].shape[-1]
    if port_size != system.physical_action_size:
        raise ValueError("a square constant activation calibration is required")
    if _paired_response_bank_sha256(
        selection,
        response_bank.plus_contexts,
        response_bank.minus_contexts,
        protocol,
    ) != response_bank.evidence_sha256:
        raise ValueError("shared activation response bank was modified")
    identity = torch.eye(port_size, device=device, dtype=torch.float64)
    columns = []
    predicted_chunks = []
    response_chunks = []
    for axis in range(system.physical_action_size):
        design = torch.cat(designs[axis], dim=0)
        response = torch.stack(responses[axis]).reshape(-1)
        column = torch.linalg.solve(
            design.T @ design + ridge * identity,
            design.T @ response,
        )
        columns.append(column)
        predicted_chunks.append(design @ column)
        response_chunks.append(response)
    calibration = torch.stack(columns, dim=1).float()
    predicted = torch.cat(predicted_chunks)
    observed = torch.cat(response_chunks)
    observed_response_tensor = torch.stack(
        [torch.stack(axis_responses) for axis_responses in responses]
    ).detach()
    relative_residual = float(
        torch.linalg.vector_norm(predicted - observed)
        / torch.linalg.vector_norm(observed).clamp_min(1e-12)
    )
    seal.assert_unchanged()
    activation_world_model.assert_frozen_and_unchanged()
    return CalibrationResult(
        latent_from_interface=calibration.detach(),
        selection=selection,
        interface_name=interface.name,
        physical_action_size=system.physical_action_size,
        port_size=port_size,
        paired_states_per_axis=PAIRED_CALIBRATION_STATES_PER_AXIS,
        environment_steps=response_bank.environment_steps,
        gradient_updates=0,
        ridge=ridge,
        fit_relative_residual=relative_residual,
        observed_state_responses=observed_response_tensor.detach(),
        response_evidence_sha256=response_bank.evidence_sha256,
        encoded_response_sha256=_calibration_response_evidence_sha256(
            selection, observed_response_tensor, interface.name
        ),
        physical_protocol=protocol,
        additional_environment_steps=0,
        responses_reused_from="shared-interface-bank:activation",
        neural_hashes_before=dict(seal.hashes),
        neural_hashes_after={
            name: _module_hash(module) for name, module in seal.modules.items()
        },
    )


def _centered_r2(actual: torch.Tensor, predicted: torch.Tensor) -> float:
    residual = (actual - predicted).square().sum()
    total = (actual - actual.mean()).square().sum()
    if float(total) <= 1e-12:
        return 1.0 if float(residual) <= 1e-12 else 0.0
    return float(1.0 - residual / total)


def _batched_encode_pixels(
    encoder: Any,
    contexts: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int = 32,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("encoding batch_size must be positive")
    return torch.cat(
        [
            _encode_pixels(encoder, contexts[start : start + batch_size].to(device))
            for start in range(0, contexts.shape[0], batch_size)
        ]
    )


def _realizability_metrics_from_response_tensors(
    actual: torch.Tensor,
    predicted: torch.Tensor,
    response_bank: PairedHeldoutResponseBank,
) -> RealizabilityMetrics:
    if actual.shape != predicted.shape or actual.ndim != 3:
        raise ValueError("realizability responses must be paired [axis,state,latent]")
    if actual.shape[:2] != (
        response_bank.physical_action_size,
        response_bank.states_per_axis,
    ):
        raise ValueError("realizability response axes differ from the shared bank")
    cosines = F.cosine_similarity(actual, predicted, dim=-1, eps=1e-12)
    signs = (actual * predicted).sum(dim=-1) > 0.0
    actual_norms = torch.linalg.vector_norm(actual, dim=-1)
    predicted_norms = torch.linalg.vector_norm(predicted, dim=-1)
    axis_cosines = tuple(float(values.mean()) for values in cosines)
    axis_r2 = tuple(
        _centered_r2(actual_norms[axis], predicted_norms[axis])
        for axis in range(actual.shape[0])
    )
    flat_cosines = cosines.flatten()
    flat_signs = signs.flatten()
    flat_actual = actual_norms.flatten()
    flat_predicted = predicted_norms.flatten()
    return RealizabilityMetrics(
        mean_cosine=float(flat_cosines.mean()),
        axis_mean_cosines=axis_cosines,
        sign_agreement=float(flat_signs.float().mean()),
        magnitude_r2=_centered_r2(flat_actual, flat_predicted),
        axis_magnitude_r2=axis_r2,
        response_cosines=tuple(float(value) for value in flat_cosines),
        response_signs=tuple(bool(value) for value in flat_signs),
        actual_magnitudes=tuple(float(value) for value in flat_actual),
        predicted_magnitudes=tuple(float(value) for value in flat_predicted),
        samples_per_axis=response_bank.states_per_axis,
        environment_steps=response_bank.environment_steps,
        response_evidence_sha256=response_bank.evidence_sha256,
        additional_environment_steps=0,
        responses_reused_from="shared-heldout-interface-bank",
    )


@torch.no_grad()
def evaluate_heldout_realizability_from_response_bank(
    encoder: Any,
    dynamics: Any,
    candidates: Sequence[ProbeCandidate],
    system: EvaluationSystem,
    interface: PhysicalInterface,
    calibration: CalibrationResult | torch.Tensor,
    response_bank: PairedHeldoutResponseBank,
    *,
    seal: FrozenEvaluationSeal,
) -> RealizabilityMetrics:
    """Evaluate one dynamics model without any model-specific plant step."""

    seal.assert_unchanged()
    protocol = linear_interface_protocol(system, interface)
    if (
        response_bank.interface_name != interface.name
        or response_bank.protocol != protocol
    ):
        raise ValueError("held-out response bank belongs to another interface")
    selected = tuple(candidates[: response_bank.states_per_axis])
    if tuple(candidate.identifier for candidate in selected) != response_bank.candidate_identifiers:
        raise ValueError("held-out response candidate identities changed")
    if _candidate_pool_sha256(selected) != response_bank.candidate_contexts_sha256:
        raise ValueError("held-out response candidate pixel contexts changed")
    if _paired_heldout_bank_sha256(
        response_bank.candidate_identifiers,
        response_bank.candidate_contexts_sha256,
        response_bank.plus_contexts,
        response_bank.minus_contexts,
        protocol,
    ) != response_bank.evidence_sha256:
        raise ValueError("shared held-out response bank was modified")
    transform = (
        calibration.latent_from_interface
        if isinstance(calibration, CalibrationResult)
        else calibration
    )
    if (
        isinstance(calibration, CalibrationResult)
        and calibration.physical_protocol != protocol
    ):
        raise ValueError("calibration belongs to another physical protocol")
    if transform.shape != (system.physical_action_size, system.physical_action_size):
        raise ValueError("calibration matrix has the wrong interface rank")
    device = _module_device(encoder, dynamics)
    transform = transform.to(device=device, dtype=torch.float32)
    contexts = torch.stack([candidate.context for candidate in selected])
    states = _batched_encode_pixels(encoder, contexts, device)
    ports = _port_matrix(dynamics, states)
    predicted = torch.stack(
        [ports @ transform[:, axis] for axis in range(system.physical_action_size)]
    )
    pairs = torch.stack(
        (response_bank.plus_contexts, response_bank.minus_contexts), dim=2
    )
    flat_pairs = pairs.flatten(0, 2)
    encoded = _batched_encode_pixels(encoder, flat_pairs, device).reshape(
        system.physical_action_size,
        response_bank.states_per_axis,
        2,
        -1,
    )
    actual = (encoded[:, :, 0] - encoded[:, :, 1]) / (
        2.0 * system.probe_amplitude * system.dt
    )
    result = _realizability_metrics_from_response_tensors(
        actual.double(), predicted.double(), response_bank
    )
    seal.assert_unchanged()
    return result


@torch.no_grad()
def evaluate_heldout_activation_from_response_bank(
    activation_world_model: FrozenActivationWriteWorldModel,
    candidates: Sequence[ProbeCandidate],
    system: EvaluationSystem,
    interface: PhysicalInterface,
    calibration: CalibrationResult | torch.Tensor,
    response_bank: PairedHeldoutResponseBank,
    *,
    seal: FrozenEvaluationSeal,
) -> RealizabilityMetrics:
    """Evaluate activation-Jacobian responses on the same held-out bank."""

    seal.assert_unchanged()
    activation_world_model.assert_frozen_and_unchanged()
    protocol = linear_interface_protocol(system, interface)
    if (
        response_bank.interface_name != interface.name
        or response_bank.protocol != protocol
    ):
        raise ValueError("held-out response bank belongs to another interface")
    selected = tuple(candidates[: response_bank.states_per_axis])
    if tuple(candidate.identifier for candidate in selected) != response_bank.candidate_identifiers:
        raise ValueError("held-out response candidate identities changed")
    if _candidate_pool_sha256(selected) != response_bank.candidate_contexts_sha256:
        raise ValueError("held-out response candidate pixel contexts changed")
    if _paired_heldout_bank_sha256(
        response_bank.candidate_identifiers,
        response_bank.candidate_contexts_sha256,
        response_bank.plus_contexts,
        response_bank.minus_contexts,
        protocol,
    ) != response_bank.evidence_sha256:
        raise ValueError("shared held-out response bank was modified")
    transform = (
        calibration.latent_from_interface
        if isinstance(calibration, CalibrationResult)
        else calibration
    )
    if (
        isinstance(calibration, CalibrationResult)
        and calibration.physical_protocol != protocol
    ):
        raise ValueError("activation calibration belongs to another physical protocol")
    if transform.shape != (system.physical_action_size, system.physical_action_size):
        raise ValueError("calibration matrix has the wrong interface rank")
    device = _module_device(activation_world_model)
    transform = transform.to(device=device, dtype=torch.float32)
    contexts = torch.stack([candidate.context for candidate in selected])
    designs = torch.cat(
        [
            activation_world_model.activation_state_rate_port(
                contexts[start : start + 4].to(device), dt=system.dt
            )
            for start in range(0, response_bank.states_per_axis, 4)
        ]
    )
    predicted = torch.stack(
        [designs @ transform[:, axis] for axis in range(system.physical_action_size)]
    )
    pairs = torch.stack(
        (response_bank.plus_contexts, response_bank.minus_contexts), dim=2
    )
    encoded = _batched_encode_pixels(
        activation_world_model.encoder,
        pairs.flatten(0, 2),
        device,
    ).reshape(
        system.physical_action_size,
        response_bank.states_per_axis,
        2,
        -1,
    )
    actual = (encoded[:, :, 0] - encoded[:, :, 1]) / (
        2.0 * system.probe_amplitude * system.dt
    )
    result = _realizability_metrics_from_response_tensors(
        actual.double(), predicted.double(), response_bank
    )
    seal.assert_unchanged()
    activation_world_model.assert_frozen_and_unchanged()
    return result


@torch.no_grad()
def evaluate_heldout_realizability(
    encoder: Any,
    dynamics: Any,
    plant: PixelPlant,
    candidates: Sequence[ProbeCandidate],
    system: EvaluationSystem,
    interface: PhysicalInterface,
    calibration: CalibrationResult | torch.Tensor,
    *,
    seal: FrozenEvaluationSeal,
    states_per_axis: int = 128,
) -> RealizabilityMetrics:
    """Evaluate paired physical responses not used by D-optimal calibration."""

    seal.assert_unchanged()
    if states_per_axis < 1 or len(candidates) < states_per_axis:
        raise ValueError("insufficient held-out pixel candidates")
    transform = (
        calibration.latent_from_interface
        if isinstance(calibration, CalibrationResult)
        else calibration
    )
    if transform.shape != (system.physical_action_size, system.physical_action_size):
        raise ValueError("calibration matrix has the wrong interface rank")
    device = _module_device(encoder, dynamics)
    transform = transform.to(device=device, dtype=torch.float32)
    cosines_by_axis: list[list[torch.Tensor]] = [
        [] for _ in range(system.physical_action_size)
    ]
    signs_by_axis: list[list[torch.Tensor]] = [
        [] for _ in range(system.physical_action_size)
    ]
    actual_norms_by_axis: list[list[torch.Tensor]] = [
        [] for _ in range(system.physical_action_size)
    ]
    predicted_norms_by_axis: list[list[torch.Tensor]] = [
        [] for _ in range(system.physical_action_size)
    ]
    for axis in range(system.physical_action_size):
        command = np.zeros(system.physical_action_size, dtype=np.float64)
        command[axis] = system.probe_amplitude
        for candidate in candidates[:states_per_axis]:
            state = _encode_pixels(encoder, candidate.context[None].to(device))
            actual = _paired_response(
                encoder,
                plant,
                candidate,
                interface,
                command,
                dt=system.dt,
                amplitude=system.probe_amplitude,
                device=device,
            )
            predicted = _port_matrix(dynamics, state)[0] @ transform[:, axis]
            cosine = F.cosine_similarity(actual[None], predicted[None], dim=-1, eps=1e-12)[0]
            cosines_by_axis[axis].append(cosine)
            signs_by_axis[axis].append(torch.dot(actual, predicted) > 0.0)
            actual_norms_by_axis[axis].append(torch.linalg.vector_norm(actual))
            predicted_norms_by_axis[axis].append(torch.linalg.vector_norm(predicted))

    axis_cosines = tuple(float(torch.stack(values).mean()) for values in cosines_by_axis)
    axis_r2 = tuple(
        _centered_r2(torch.stack(actual), torch.stack(predicted))
        for actual, predicted in zip(actual_norms_by_axis, predicted_norms_by_axis)
    )
    all_cosines = torch.cat([torch.stack(values) for values in cosines_by_axis])
    all_signs = torch.cat([torch.stack(values) for values in signs_by_axis])
    all_actual = torch.cat([torch.stack(values) for values in actual_norms_by_axis])
    all_predicted = torch.cat([torch.stack(values) for values in predicted_norms_by_axis])
    seal.assert_unchanged()
    return RealizabilityMetrics(
        mean_cosine=float(all_cosines.mean()),
        axis_mean_cosines=axis_cosines,
        sign_agreement=float(all_signs.float().mean()),
        magnitude_r2=_centered_r2(all_actual, all_predicted),
        axis_magnitude_r2=axis_r2,
        response_cosines=tuple(float(value) for value in all_cosines),
        response_signs=tuple(bool(value) for value in all_signs),
        actual_magnitudes=tuple(float(value) for value in all_actual),
        predicted_magnitudes=tuple(float(value) for value in all_predicted),
        samples_per_axis=states_per_axis,
        environment_steps=2 * states_per_axis * system.physical_action_size,
        additional_environment_steps=(
            2 * states_per_axis * system.physical_action_size
        ),
    )


@torch.no_grad()
def evaluate_heldout_activation_realizability(
    activation_world_model: FrozenActivationWriteWorldModel,
    plant: PixelPlant,
    candidates: Sequence[ProbeCandidate],
    system: EvaluationSystem,
    interface: PhysicalInterface,
    calibration: CalibrationResult | torch.Tensor,
    *,
    seal: FrozenEvaluationSeal,
    states_per_axis: int = 128,
) -> RealizabilityMetrics:
    """Held-out realizability of the activation-Jacobian write directions."""

    seal.assert_unchanged()
    activation_world_model.assert_frozen_and_unchanged()
    if states_per_axis < 1 or len(candidates) < states_per_axis:
        raise ValueError("insufficient held-out pixel candidates")
    transform = (
        calibration.latent_from_interface
        if isinstance(calibration, CalibrationResult)
        else calibration
    )
    if transform.shape != (system.physical_action_size, system.physical_action_size):
        raise ValueError("calibration matrix has the wrong interface rank")
    device = _module_device(activation_world_model)
    transform = transform.to(device=device, dtype=torch.float32)
    cosines_by_axis: list[list[torch.Tensor]] = [
        [] for _ in range(system.physical_action_size)
    ]
    signs_by_axis: list[list[torch.Tensor]] = [
        [] for _ in range(system.physical_action_size)
    ]
    actual_norms_by_axis: list[list[torch.Tensor]] = [
        [] for _ in range(system.physical_action_size)
    ]
    predicted_norms_by_axis: list[list[torch.Tensor]] = [
        [] for _ in range(system.physical_action_size)
    ]
    heldout_contexts = torch.stack(
        [candidate.context for candidate in candidates[:states_per_axis]]
    )
    heldout_designs = torch.cat(
        [
            activation_world_model.activation_state_rate_port(
                heldout_contexts[start : start + 4].to(device), dt=system.dt
            )
            for start in range(0, states_per_axis, 4)
        ]
    )
    for axis in range(system.physical_action_size):
        command = np.zeros(system.physical_action_size, dtype=np.float64)
        command[axis] = system.probe_amplitude
        for candidate_index, candidate in enumerate(candidates[:states_per_axis]):
            actual = _paired_response(
                activation_world_model.encoder,
                plant,
                candidate,
                interface,
                command,
                dt=system.dt,
                amplitude=system.probe_amplitude,
                device=device,
            )
            design = heldout_designs[candidate_index]
            predicted = design @ transform[:, axis]
            cosine = F.cosine_similarity(
                actual[None], predicted[None], dim=-1, eps=1e-12
            )[0]
            cosines_by_axis[axis].append(cosine)
            signs_by_axis[axis].append(torch.dot(actual, predicted) > 0.0)
            actual_norms_by_axis[axis].append(torch.linalg.vector_norm(actual))
            predicted_norms_by_axis[axis].append(torch.linalg.vector_norm(predicted))
    axis_cosines = tuple(float(torch.stack(values).mean()) for values in cosines_by_axis)
    axis_r2 = tuple(
        _centered_r2(torch.stack(actual), torch.stack(predicted))
        for actual, predicted in zip(actual_norms_by_axis, predicted_norms_by_axis)
    )
    all_cosines = torch.cat([torch.stack(values) for values in cosines_by_axis])
    all_signs = torch.cat([torch.stack(values) for values in signs_by_axis])
    all_actual = torch.cat([torch.stack(values) for values in actual_norms_by_axis])
    all_predicted = torch.cat([torch.stack(values) for values in predicted_norms_by_axis])
    seal.assert_unchanged()
    activation_world_model.assert_frozen_and_unchanged()
    return RealizabilityMetrics(
        mean_cosine=float(all_cosines.mean()),
        axis_mean_cosines=axis_cosines,
        sign_agreement=float(all_signs.float().mean()),
        magnitude_r2=_centered_r2(all_actual, all_predicted),
        axis_magnitude_r2=axis_r2,
        response_cosines=tuple(float(value) for value in all_cosines),
        response_signs=tuple(bool(value) for value in all_signs),
        actual_magnitudes=tuple(float(value) for value in all_actual),
        predicted_magnitudes=tuple(float(value) for value in all_predicted),
        samples_per_axis=states_per_axis,
        environment_steps=2 * states_per_axis * system.physical_action_size,
        additional_environment_steps=(
            2 * states_per_axis * system.physical_action_size
        ),
    )


@dataclass(frozen=True)
class CEMConfig:
    horizon: int
    candidates: int = 512
    iterations: int = 4
    elites: int = 64
    action_low: float = -1.0
    action_high: float = 1.0
    action_penalty: float = 0.01
    minimum_std: float = 0.05
    elite_momentum: float = 0.10
    activation_rollout_batch_size: int = 32

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.candidates < 2 or self.iterations < 1:
            raise ValueError("CEM horizons, candidates and iterations must be positive")
        if not 1 <= self.elites < self.candidates:
            raise ValueError("elites must be in [1, candidates)")
        numeric = (
            self.action_low,
            self.action_high,
            self.action_penalty,
            self.minimum_std,
            self.elite_momentum,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("CEM numeric configuration must be finite")
        if self.action_low >= self.action_high or self.minimum_std <= 0.0:
            raise ValueError("invalid CEM action bounds or minimum_std")
        if not 0.0 <= self.elite_momentum < 1.0:
            raise ValueError("elite_momentum must lie in [0, 1)")
        if (
            type(self.activation_rollout_batch_size) is not int
            or self.activation_rollout_batch_size < 1
        ):
            raise ValueError("activation_rollout_batch_size must be a positive integer")


def registered_cem_config(system: EvaluationSystem | str) -> CEMConfig:
    definition = SYSTEMS[system] if isinstance(system, str) else system
    bound = registered_linear_interface_command_bound(definition)
    return CEMConfig(
        horizon=definition.planning_horizon,
        action_low=-bound,
        action_high=bound,
    )


@dataclass(frozen=True)
class CEMPlan:
    first_interface_command: torch.Tensor
    best_interface_sequence: torch.Tensor
    elite_mean_sequence: torch.Tensor
    best_cost: float
    candidate_evaluations: int
    candidates_per_iteration: int
    iterations: int
    elites: int


def _assert_cem_plan_budget(plan: CEMPlan, config: CEMConfig) -> None:
    expected_evaluations = config.candidates * config.iterations
    if (
        plan.candidate_evaluations != expected_evaluations
        or plan.candidates_per_iteration != config.candidates
        or plan.iterations != config.iterations
        or plan.elites != config.elites
        or tuple(plan.best_interface_sequence.shape[:1]) != (config.horizon,)
    ):
        raise AssertionError("a learned planner did not receive the locked CEM budget")


@dataclass(frozen=True)
class PuckOnlyPixelObjective:
    """Pixel-derived puck supports used by every Blocket planner.

    The supports contain only membership in the registered categorical puck
    classes.  In particular, they retain no player class, background class,
    coordinate, environment value, or externally supplied entity mask.
    """

    source_support: torch.Tensor = field(repr=False)
    target_support: torch.Tensor = field(repr=False)
    puck_pixel_values: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_support, torch.Tensor)
            or not isinstance(self.target_support, torch.Tensor)
            or self.source_support.ndim != 2
            or self.target_support.shape != self.source_support.shape
            or self.source_support.dtype != torch.bool
            or self.target_support.dtype != torch.bool
            or self.source_support.requires_grad
            or self.target_support.requires_grad
        ):
            raise ValueError("puck supports must be matching gradient-free boolean images")
        if (
            not self.puck_pixel_values
            or len(set(self.puck_pixel_values)) != len(self.puck_pixel_values)
            or any(type(value) is not int or value < 0 for value in self.puck_pixel_values)
        ):
            raise ValueError("puck pixel values must be unique non-negative integers")
        if not bool(self.source_support.any()) or not bool(self.target_support.any()):
            raise ValueError("source and target categorical images must both contain the puck")
        # A frozen dataclass does not by itself protect tensor storage.  Own
        # compact copies so later mutations of the caller's categorical images
        # cannot alter a CEM objective after it has been constructed.
        object.__setattr__(
            self, "source_support", self.source_support.detach().clone().contiguous()
        )
        object.__setattr__(
            self, "target_support", self.target_support.detach().clone().contiguous()
        )


def _categorical_support(
    pixels: torch.Tensor,
    pixel_values: Sequence[int],
) -> torch.Tensor:
    if pixels.ndim != 2 or pixels.is_floating_point() or pixels.requires_grad:
        raise ValueError("categorical objective images must be gradient-free integer images")
    support = torch.zeros_like(pixels, dtype=torch.bool)
    for value in pixel_values:
        support |= pixels.eq(value)
    return support


def make_puck_only_pixel_objective(
    source_pixels: torch.Tensor,
    target_pixels: torch.Tensor,
    puck_pixel_values: Sequence[int],
) -> PuckOnlyPixelObjective:
    """Build the Blocket planning target from categorical pixels alone."""

    values = tuple(int(value) for value in puck_pixel_values)
    if source_pixels.shape != target_pixels.shape:
        raise ValueError("source and target categorical images must have matching shapes")
    return PuckOnlyPixelObjective(
        source_support=_categorical_support(source_pixels, values),
        target_support=_categorical_support(target_pixels, values),
        puck_pixel_values=values,
    )


def _assert_puck_objective_matches_target(
    objective: PuckOnlyPixelObjective,
    target_pixels: torch.Tensor,
) -> None:
    observed = _categorical_support(target_pixels, objective.puck_pixel_values)
    expected = objective.target_support.to(observed.device)
    if not torch.equal(observed, expected):
        raise ValueError("the categorical target puck support differs from its objective")


def puck_only_pixel_cost(
    logits: torch.Tensor,
    objective: PuckOnlyPixelObjective,
) -> torch.Tensor:
    r"""Balanced binary NLL for puck occupancy and source-to-target motion.

    The categorical model is reduced to exactly two semantic events at each
    pixel: ``puck`` (the registered puck classes) and ``not puck`` (the
    log-sum-exp of every other class).  Which non-puck class represents a
    player, wall, or background is therefore absent from the objective.
    """

    if logits.ndim != 4 or not logits.is_floating_point():
        raise ValueError("puck scoring expects [candidate,class,height,width] logits")
    if tuple(logits.shape[-2:]) != tuple(objective.target_support.shape):
        raise ValueError("puck objective and rendered logits have different image shapes")
    if max(objective.puck_pixel_values) >= logits.shape[1]:
        raise ValueError("a puck pixel value is outside the rendered categorical palette")
    puck_indices = torch.tensor(
        objective.puck_pixel_values,
        dtype=torch.long,
        device=logits.device,
    )
    non_puck_selector = torch.ones(
        logits.shape[1], dtype=torch.bool, device=logits.device
    )
    non_puck_selector[puck_indices] = False
    if not bool(non_puck_selector.any()):
        raise ValueError("puck scoring requires at least one non-puck category")
    log_probabilities = F.log_softmax(logits.float(), dim=1)
    log_puck = torch.logsumexp(
        log_probabilities.index_select(1, puck_indices), dim=1
    )
    log_not_puck = torch.logsumexp(
        log_probabilities[:, non_puck_selector], dim=1
    )
    source = objective.source_support.to(device=logits.device)
    target = objective.target_support.to(device=logits.device)
    target_background = ~target
    departed_source = source & target_background

    def masked_mean(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        weights = support.to(dtype=values.dtype)
        return (values * weights).sum(dim=(-2, -1)) / weights.sum().clamp_min(1.0)

    target_presence = masked_mean(-log_puck, target)
    target_exclusion = masked_mean(-log_not_puck, target_background)
    source_departure = masked_mean(-log_not_puck, departed_source)
    return (target_presence + target_exclusion + source_departure) / 3.0


def _categorical_pixel_cost(
    logits: torch.Tensor,
    target_pixels: torch.Tensor,
    class_weights: torch.Tensor | None,
    pixel_objective: PuckOnlyPixelObjective | None = None,
) -> torch.Tensor:
    if pixel_objective is not None:
        return puck_only_pixel_cost(logits, pixel_objective)
    if target_pixels.ndim != 2:
        raise ValueError("the controller target must be one categorical pixel image")
    target = target_pixels.to(device=logits.device, dtype=torch.long)
    target = target[None].expand(logits.shape[0], -1, -1)
    return F.cross_entropy(
        logits,
        target,
        weight=None if class_weights is None else class_weights.to(logits.device).float(),
        reduction="none",
    ).mean(dim=(-2, -1))


@torch.no_grad()
def _score_dynamics_sequences(
    dynamics: Any,
    renderer: Any,
    initial_state: torch.Tensor,
    target_pixels: torch.Tensor,
    interface_sequences: torch.Tensor,
    latent_from_interface: torch.Tensor,
    config: CEMConfig,
    class_weights: torch.Tensor | None,
    pixel_objective: PuckOnlyPixelObjective | None,
) -> torch.Tensor:
    candidate_count, horizon, _ = interface_sequences.shape
    state = initial_state.reshape(1, -1).expand(candidate_count, -1).clone()
    latent = torch.einsum(
        "chp,mp->chm", interface_sequences, latent_from_interface
    )
    pixel_cost = initial_state.new_zeros(candidate_count)
    scored_steps = 0
    for step in range(horizon):
        state = _dynamics_step(dynamics, state, latent[:, step])
        if step >= horizon // 2:
            pixel_cost = pixel_cost + _categorical_pixel_cost(
                _render_logits(renderer, state),
                target_pixels,
                class_weights,
                pixel_objective,
            )
            scored_steps += 1
    return pixel_cost / max(scored_steps, 1) + config.action_penalty * (
        interface_sequences.square().mean(dim=(1, 2))
    )


def _cem_samples(
    mean: torch.Tensor,
    std: torch.Tensor,
    config: CEMConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    noise = torch.randn(
        config.candidates,
        *mean.shape,
        device=mean.device,
        dtype=mean.dtype,
        generator=generator,
    )
    return (mean[None] + std[None] * noise).clamp(config.action_low, config.action_high)


@torch.no_grad()
def cem_pixel_target_mpc(
    dynamics: Any,
    renderer: Any,
    initial_state: torch.Tensor,
    target_pixels: torch.Tensor,
    latent_from_interface: CalibrationResult | torch.Tensor,
    config: CEMConfig,
    *,
    seal: FrozenEvaluationSeal,
    seed: int,
    class_weights: torch.Tensor | None = None,
    pixel_objective: PuckOnlyPixelObjective | None = None,
    _seal_already_verified: bool = False,
) -> CEMPlan:
    """CEM MPC shared unchanged by structured and unstructured dynamics."""

    if not _seal_already_verified:
        seal.assert_unchanged()
    transform = (
        latent_from_interface.latent_from_interface
        if isinstance(latent_from_interface, CalibrationResult)
        else latent_from_interface
    )
    if transform.ndim != 2:
        raise ValueError("latent_from_interface must be a matrix")
    if pixel_objective is not None:
        _assert_puck_objective_matches_target(pixel_objective, target_pixels)
    device = initial_state.device
    transform = transform.to(device=device, dtype=torch.float32)
    action_size = transform.shape[1]
    mean = torch.zeros(config.horizon, action_size, device=device)
    std = torch.full_like(mean, 0.5 * (config.action_high - config.action_low))
    generator = torch.Generator(device=device).manual_seed(seed)
    best_sequence = mean.clone()
    best_cost = float("inf")
    for _ in range(config.iterations):
        sequences = _cem_samples(mean, std, config, generator)
        costs = _score_dynamics_sequences(
            dynamics,
            renderer,
            initial_state,
            target_pixels,
            sequences,
            transform,
            config,
            class_weights,
            pixel_objective,
        )
        elite_indices = torch.topk(costs, config.elites, largest=False).indices
        elites = sequences[elite_indices]
        elite_mean = elites.mean(dim=0)
        elite_std = elites.std(dim=0, unbiased=False).clamp_min(config.minimum_std)
        mean = config.elite_momentum * mean + (1.0 - config.elite_momentum) * elite_mean
        std = config.elite_momentum * std + (1.0 - config.elite_momentum) * elite_std
        iteration_best = int(torch.argmin(costs))
        if float(costs[iteration_best]) < best_cost:
            best_cost = float(costs[iteration_best])
            best_sequence = sequences[iteration_best].clone()
    if not _seal_already_verified:
        seal.assert_unchanged()
    return CEMPlan(
        first_interface_command=best_sequence[0].detach(),
        best_interface_sequence=best_sequence.detach(),
        elite_mean_sequence=mean.detach(),
        best_cost=best_cost,
        candidate_evaluations=config.candidates * config.iterations,
        candidates_per_iteration=config.candidates,
        iterations=config.iterations,
        elites=config.elites,
    )


@torch.no_grad()
def cem_frozen_world_model_mpc(
    activation_rollout: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pixel_context: torch.Tensor,
    target_pixels: torch.Tensor,
    latent_from_interface: CalibrationResult | torch.Tensor,
    config: CEMConfig,
    *,
    seal: FrozenEvaluationSeal,
    seed: int,
    class_weights: torch.Tensor | None = None,
    pixel_objective: PuckOnlyPixelObjective | None = None,
    _seal_already_verified: bool = False,
) -> CEMPlan:
    """Generic frozen-WM baseline planning calibrated activation coefficients.

    ``activation_rollout(context_batch, latent_sequence)`` must return either
    ``[candidate,horizon,class,height,width]`` or terminal logits
    ``[candidate,class,height,width]``.  It may not consume physical actions.
    """

    if not _seal_already_verified:
        seal.assert_unchanged()
    transform = (
        latent_from_interface.latent_from_interface
        if isinstance(latent_from_interface, CalibrationResult)
        else latent_from_interface
    )
    if pixel_objective is not None:
        _assert_puck_objective_matches_target(pixel_objective, target_pixels)
    device = pixel_context.device
    transform = transform.to(device=device, dtype=torch.float32)
    action_size = transform.shape[1]
    mean = torch.zeros(config.horizon, action_size, device=device)
    std = torch.full_like(mean, 0.5 * (config.action_high - config.action_low))
    generator = torch.Generator(device=device).manual_seed(seed)
    best_sequence = mean.clone()
    best_cost = float("inf")
    for _ in range(config.iterations):
        commands = _cem_samples(mean, std, config, generator)
        latent = torch.einsum("chp,mp->chm", commands, transform)
        contexts = pixel_context[None].expand(config.candidates, *pixel_context.shape)
        # A full [512,horizon,class,height,width] tensor is needlessly large.
        # Frozen eval-mode transformers are batch-separable, so exact candidate
        # costs can be accumulated in fixed micro-batches without changing a
        # sample, CEM update, or candidate-evaluation count.
        pixel_cost_chunks: list[torch.Tensor] = []
        micro_batch = config.activation_rollout_batch_size
        for start in range(0, config.candidates, micro_batch):
            stop = min(start + micro_batch, config.candidates)
            logits = activation_rollout(contexts[start:stop], latent[start:stop])
            if not isinstance(logits, torch.Tensor) or logits.ndim not in (4, 5):
                raise ValueError(
                    "activation rollout must return terminal or temporal pixel logits"
                )
            if logits.shape[0] != stop - start:
                raise ValueError("activation rollout changed the registered candidate count")
            if logits.ndim == 5 and logits.shape[1] != config.horizon:
                raise ValueError("activation rollout changed the registered planning horizon")
            if logits.ndim == 4:
                chunk_cost = _categorical_pixel_cost(
                    logits, target_pixels, class_weights, pixel_objective
                )
            else:
                costs = [
                    _categorical_pixel_cost(
                        logits[:, step],
                        target_pixels,
                        class_weights,
                        pixel_objective,
                    )
                    for step in range(logits.shape[1] // 2, logits.shape[1])
                ]
                chunk_cost = torch.stack(costs).mean(dim=0)
            pixel_cost_chunks.append(chunk_cost)
        pixel_cost = torch.cat(pixel_cost_chunks)
        total = pixel_cost + config.action_penalty * commands.square().mean(dim=(1, 2))
        elite_indices = torch.topk(total, config.elites, largest=False).indices
        elites = commands[elite_indices]
        elite_mean = elites.mean(dim=0)
        elite_std = elites.std(dim=0, unbiased=False).clamp_min(config.minimum_std)
        mean = config.elite_momentum * mean + (1.0 - config.elite_momentum) * elite_mean
        std = config.elite_momentum * std + (1.0 - config.elite_momentum) * elite_std
        iteration_best = int(torch.argmin(total))
        if float(total[iteration_best]) < best_cost:
            best_cost = float(total[iteration_best])
            best_sequence = commands[iteration_best].clone()
    if not _seal_already_verified:
        seal.assert_unchanged()
    return CEMPlan(
        first_interface_command=best_sequence[0].detach(),
        best_interface_sequence=best_sequence.detach(),
        elite_mean_sequence=mean.detach(),
        best_cost=best_cost,
        candidate_evaluations=config.candidates * config.iterations,
        candidates_per_iteration=config.candidates,
        iterations=config.iterations,
        elites=config.elites,
    )


@dataclass(frozen=True)
class PixelControlEpisode:
    identifier: str
    environment: Any = field(repr=False, compare=False)
    context: torch.Tensor = field(repr=False)
    target_pixels: torch.Tensor = field(repr=False)


@dataclass(frozen=True)
class ControlResult:
    errors: Mapping[str, tuple[float, ...]]
    interface_name: str
    episodes: int
    control_steps: int
    planner_budget: Mapping[str, float | int | str]
    target_source: str = "categorical_pixels_only"
    episode_identifiers: tuple[str, ...] = ()
    interface_command_traces: Mapping[
        str, tuple[tuple[tuple[float, ...], ...], ...]
    ] = field(default_factory=dict, repr=False, compare=False)
    planner_seed_schedule_sha256: str | None = None
    physical_protocol: LinearInterfaceProtocol | None = None

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.control_steps < 1:
            raise ValueError("control result episode/horizon counts are invalid")
        if not self.errors or any(
            len(values) != self.episodes
            or not bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())
            for values in self.errors.values()
        ):
            raise ValueError("control result errors are incomplete or non-finite")
        if self.episode_identifiers or self.interface_command_traces:
            if (
                len(self.episode_identifiers) != self.episodes
                or len(set(self.episode_identifiers)) != self.episodes
                or set(self.interface_command_traces) != set(self.errors)
            ):
                raise ValueError("control trace episode/controller schema is invalid")
            action_size: int | None = None
            for controller, episode_traces in self.interface_command_traces.items():
                if len(episode_traces) != self.episodes:
                    raise ValueError(f"control trace {controller} episode count changed")
                for trace in episode_traces:
                    if len(trace) != self.control_steps:
                        raise ValueError(
                            f"control trace {controller} decision count changed"
                        )
                    for command in trace:
                        if action_size is None:
                            action_size = len(command)
                        if (
                            not command
                            or len(command) != action_size
                            or not bool(np.isfinite(np.asarray(command)).all())
                            or bool((np.abs(np.asarray(command)) > 1.000001).any())
                        ):
                            raise ValueError("control trace command is invalid")
                        if self.physical_protocol is not None:
                            interface_command = np.asarray(command, dtype=np.float64)
                            if bool(
                                (
                                    np.abs(interface_command)
                                    > self.physical_protocol.common_interface_command_bound
                                    + 1e-6
                                ).any()
                            ) or float(
                                np.linalg.norm(
                                    np.asarray(
                                        self.physical_protocol.native_from_interface,
                                        dtype=np.float64,
                                    )
                                    @ interface_command
                                )
                            ) > self.physical_protocol.native_l2_limit + 1e-6:
                                raise ValueError(
                                    "control trace leaves the sealed linear interface domain"
                                )
            if action_size is None:  # pragma: no cover - nonempty above
                raise ValueError("control trace contains no command")
            if (
                self.planner_seed_schedule_sha256 is None
                or re.fullmatch(
                    r"[0-9a-f]{64}", self.planner_seed_schedule_sha256
                )
                is None
            ):
                raise ValueError("control trace planner-seed seal is invalid")
        elif self.planner_seed_schedule_sha256 is not None:
            raise ValueError("control result has a planner-seed seal but no traces")
        # The task identity is derived from authenticated episode identifiers,
        # never supplied as a caller-controlled pass claim.
        _episode_target_task_identity(self.episode_identifiers)

    def means(self) -> dict[str, float]:
        return {name: float(np.mean(values)) for name, values in self.errors.items()}

    @property
    def target_task(self) -> dict[str, str]:
        return _episode_target_task_identity(self.episode_identifiers)

    def as_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface_name,
            "episodes": self.episodes,
            "controlSteps": self.control_steps,
            "meanFinalPixelTargetError": self.means(),
            "plannerBudget": dict(self.planner_budget),
            "targetSource": self.target_source,
            "targetTask": self.target_task,
            "episodeIdentifiers": self.episode_identifiers,
            "interfaceCommandTraceSha256": (
                None if not self.interface_command_traces else control_trace_sha256(self)
            ),
            "plannerSeedScheduleSha256": self.planner_seed_schedule_sha256,
            "physicalProtocol": (
                None
                if self.physical_protocol is None
                else self.physical_protocol.to_dict()
            ),
            "physicalProtocolSha256": (
                None
                if self.physical_protocol is None
                else self.physical_protocol.sha256
            ),
        }


def _planner_seed_schedule_sha256(
    *, seed: int, episode_offset: int, episodes: int, control_steps: int
) -> str:
    schedule = [
        seed + global_episode * 100_003 + decision
        for global_episode in range(episode_offset, episode_offset + episodes)
        for decision in range(control_steps)
    ]
    digest = hashlib.sha256()
    digest.update(str((seed, episode_offset, episodes, control_steps)).encode("ascii"))
    digest.update(np.asarray(schedule, dtype=np.int64).tobytes())
    return digest.hexdigest()


def control_trace_sha256(result: ControlResult) -> str:
    """Hash raw replayable actions and paired errors, not a supplied pass bit."""

    digest = hashlib.sha256()
    digest.update(result.interface_name.encode("utf-8"))
    digest.update(str(result.episode_identifiers).encode("utf-8"))
    for name in sorted(result.errors):
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(result.errors[name], dtype=np.float64).tobytes())
        commands = np.asarray(
            result.interface_command_traces[name], dtype=np.float32
        )
        digest.update(str(tuple(commands.shape)).encode("ascii"))
        digest.update(commands.tobytes())
    digest.update(str(dict(sorted(result.planner_budget.items()))).encode("ascii"))
    digest.update(str(result.planner_seed_schedule_sha256).encode("ascii"))
    digest.update(
        json.dumps(
            result.target_task,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(
        b"no-physical-protocol"
        if result.physical_protocol is None
        else result.physical_protocol.sha256.encode("ascii")
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenLatentPlannerSpec:
    """One additional frozen latent planner in a paired control comparison."""

    encoder: Any = field(repr=False, compare=False)
    renderer: Any = field(repr=False, compare=False)
    dynamics: Any = field(repr=False, compare=False)
    calibration: CalibrationResult | torch.Tensor = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("encoder", self.encoder),
            ("renderer", self.renderer),
            ("dynamics", self.dynamics),
        ):
            if isinstance(value, nn.Module) and (
                value.training
                or any(parameter.requires_grad for parameter in value.parameters())
            ):
                raise ValueError(f"additional planner {name} is not frozen")
        if isinstance(self.calibration, CalibrationResult):
            if self.calibration.gradient_updates != 0:
                raise ValueError("additional planner calibration used gradients")
        elif (
            not isinstance(self.calibration, torch.Tensor)
            or self.calibration.ndim != 2
            or self.calibration.requires_grad
            or not self.calibration.is_floating_point()
            or not bool(torch.isfinite(self.calibration).all())
        ):
            raise ValueError("additional planner calibration is invalid")


def _hard_centroid(classes: torch.Tensor, values: Sequence[int]) -> torch.Tensor:
    if classes.ndim == 2:
        classes = classes[None]
    mask = torch.zeros_like(classes, dtype=torch.float32)
    for value in values:
        mask = mask + classes.eq(value).float()
    height, width = classes.shape[-2:]
    x = torch.arange(width, device=classes.device, dtype=torch.float32) + 0.5
    y = torch.arange(height, device=classes.device, dtype=torch.float32) + 0.5
    mass = mask.sum(dim=(-2, -1)).clamp_min(1e-7)
    return torch.stack(
        ((mask * x).sum(dim=(-2, -1)) / mass, (mask * y[:, None]).sum(dim=(-2, -1)) / mass),
        dim=-1,
    )


def pixel_target_error(
    system: EvaluationSystem,
    current_pixels: torch.Tensor,
    target_pixels: torch.Tensor,
) -> float:
    """The locked target error is measured only from rendered categories."""

    current = _hard_centroid(current_pixels, system.controlled_pixel_values)[0]
    target = _hard_centroid(target_pixels, system.controlled_pixel_values)[0]
    if system.pixel_observable == "pendulum_angle":
        width = current_pixels.shape[-1]
        pivot = current.new_tensor((0.5 * width, 0.43 * width))
        current_angle = torch.atan2(current[0] - pivot[0], current[1] - pivot[1])
        target_angle = torch.atan2(target[0] - pivot[0], target[1] - pivot[1])
        delta = torch.atan2(torch.sin(current_angle - target_angle), torch.cos(current_angle - target_angle))
        return float(delta.abs())
    return float(torch.linalg.vector_norm(current - target) / current_pixels.shape[-1])


@torch.no_grad()
def evaluate_closed_loop_controllers(
    episodes: Sequence[PixelControlEpisode],
    system: EvaluationSystem,
    plant: PixelPlant,
    interface: PhysicalInterface,
    encoder: Any,
    renderer: Any,
    structured_dynamics: Any,
    unstructured_dynamics: Any,
    structured_calibration: CalibrationResult | torch.Tensor,
    unstructured_calibration: CalibrationResult | torch.Tensor,
    *,
    unstructured_encoder: Any,
    unstructured_renderer: Any,
    seal: FrozenEvaluationSeal,
    cem_config: CEMConfig | None = None,
    activation_rollout: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    activation_calibration: CalibrationResult | torch.Tensor | None = None,
    additional_latent_planners: Mapping[str, FrozenLatentPlannerSpec] | None = None,
    seed: int,
    episode_offset: int = 0,
    class_weights: torch.Tensor | None = None,
) -> ControlResult:
    """Paired receding-horizon evaluation with model-specific re-encoding.

    The independent ``unstructured`` world model is deliberately routed
    through its own encoder and renderer.  The one authenticated immutable
    backbone may be shared; every downstream module object and tensor storage
    must be disjoint before an episode is opened.
    """

    seal.assert_unchanged()
    if not episodes:
        raise ValueError("at least one paired control episode is required")
    if type(episode_offset) is not int or episode_offset < 0:
        raise ValueError("control episode_offset must be a non-negative integer")
    config = cem_config or registered_cem_config(system)
    protocol = linear_interface_protocol(system, interface)
    if (
        config.action_low < -protocol.common_interface_command_bound - 1e-12
        or config.action_high > protocol.common_interface_command_bound + 1e-12
    ):
        raise ValueError("CEM bounds leave the sealed linear interface domain")
    if config.horizon != system.planning_horizon and cem_config is None:
        raise AssertionError("registered CEM horizon drifted")
    if (activation_rollout is None) != (activation_calibration is None):
        raise ValueError("activation rollout and its calibration must be supplied together")
    structured_components = (encoder, renderer, structured_dynamics)
    independent_components = (
        unstructured_encoder,
        unstructured_renderer,
        unstructured_dynamics,
    )
    if any(not isinstance(value, nn.Module) for value in (*structured_components, *independent_components)):
        raise TypeError("structured and unstructured planners must be neural modules")
    _validate_primary_planner_isolation(
        encoder,
        renderer,
        structured_dynamics,
        unstructured_encoder,
        unstructured_renderer,
        unstructured_dynamics,
    )
    additional = dict(additional_latent_planners or {})
    reserved_names = {"structured", "unstructured", "activation", "coast", "random"}
    if (
        any(type(name) is not str or not name for name in additional)
        or set(additional) & reserved_names
        or any(type(specification) is not FrozenLatentPlannerSpec for specification in additional.values())
    ):
        raise ValueError("additional latent planner names/schema are invalid")
    sealed_module_ids = {
        id(module)
        for root in seal.modules.values()
        for module in root.modules()
    }
    sealed_storage_ids = _module_storage_identities(*seal.modules.values())
    for planner_name, components in (
        ("structured", structured_components),
        ("unstructured", independent_components),
    ):
        for component_name, component in zip(
            ("encoder", "renderer", "dynamics"), components, strict=True
        ):
            component_storage_ids = _module_storage_identities(component)
            if (
                component_storage_ids
                and not component_storage_ids.issubset(sealed_storage_ids)
            ) or (
                not component_storage_ids and id(component) not in sealed_module_ids
            ):
                raise ValueError(
                    f"{planner_name}.{component_name} is outside the frozen seal"
                )
    for name, specification in additional.items():
        for component_name, component in (
            ("encoder", specification.encoder),
            ("renderer", specification.renderer),
            ("dynamics", specification.dynamics),
        ):
            if isinstance(component, nn.Module) and id(component) not in sealed_module_ids:
                raise ValueError(
                    f"additional planner {name}.{component_name} is outside the frozen seal"
                )
    activation_device = _module_device(activation_rollout)
    controller_names = ["structured", "unstructured"]
    if activation_rollout is not None:
        controller_names.append("activation")
    controller_names.extend(sorted(additional))
    controller_names.extend(("coast", "random"))
    errors: dict[str, list[float]] = {name: [] for name in controller_names}
    command_traces: dict[str, list[tuple[tuple[float, ...], ...]]] = {
        name: [] for name in controller_names
    }

    for episode_index, episode in enumerate(episodes):
        global_episode_index = episode_offset + episode_index
        for controller_index, name in enumerate(controller_names):
            environment = plant.clone_environment(episode.environment)
            context = episode.context.clone()
            rng = np.random.default_rng(
                seed + global_episode_index * 1_000_003 + controller_index * 10_007
            )
            episode_commands: list[tuple[float, ...]] = []
            for decision in range(system.control_steps):
                # Learned planners receive the same random-normal draws.  The
                # first CEM population is therefore exactly paired; later
                # populations differ only through each model's elite update.
                paired_planner_seed = (
                    seed + global_episode_index * 100_003 + decision
                )
                pixel_objective = (
                    make_puck_only_pixel_objective(
                        context[-1],
                        episode.target_pixels,
                        system.controlled_pixel_values,
                    )
                    if system.name == "blocket" and name not in {"coast", "random"}
                    else None
                )
                if name == "coast":
                    command = np.zeros(system.physical_action_size, dtype=np.float32)
                elif name == "random":
                    command = rng.uniform(
                        config.action_low,
                        config.action_high,
                        size=system.physical_action_size,
                    ).astype(np.float32)
                elif name == "activation":
                    assert activation_rollout is not None and activation_calibration is not None
                    plan = cem_frozen_world_model_mpc(
                        activation_rollout,
                        context.to(activation_device),
                        episode.target_pixels.to(activation_device),
                        activation_calibration,
                        config,
                        seal=seal,
                        seed=paired_planner_seed,
                        class_weights=class_weights,
                        pixel_objective=pixel_objective,
                        _seal_already_verified=True,
                    )
                    _assert_cem_plan_budget(plan, config)
                    command = plan.first_interface_command.cpu().numpy()
                else:
                    if name in additional:
                        specification = additional[name]
                        planner_encoder = specification.encoder
                        planner_renderer = specification.renderer
                        dynamics = specification.dynamics
                        transform = specification.calibration
                    else:
                        if name == "structured":
                            planner_encoder = encoder
                            planner_renderer = renderer
                            dynamics = structured_dynamics
                            transform = structured_calibration
                        else:
                            planner_encoder = unstructured_encoder
                            planner_renderer = unstructured_renderer
                            dynamics = unstructured_dynamics
                            transform = unstructured_calibration
                    planner_device = _module_device(
                        planner_encoder, planner_renderer, dynamics
                    )
                    state = _encode_pixels(
                        planner_encoder, context[None].to(planner_device)
                    )[0]
                    plan = cem_pixel_target_mpc(
                        dynamics,
                        planner_renderer,
                        state,
                        episode.target_pixels.to(planner_device),
                        transform,
                        config,
                        seal=seal,
                        seed=paired_planner_seed,
                        class_weights=class_weights,
                        pixel_objective=pixel_objective,
                        _seal_already_verified=True,
                    )
                    _assert_cem_plan_budget(plan, config)
                    command = plan.first_interface_command.cpu().numpy()
                command = np.asarray(command, dtype=np.float32)
                if command.shape != (system.physical_action_size,) or not bool(
                    np.isfinite(command).all()
                ):
                    raise ValueError("controller produced an invalid interface command")
                if bool(
                    (
                        np.abs(command)
                        > protocol.common_interface_command_bound + 1e-6
                    ).any()
                ) or float(np.linalg.norm(interface.matrix() @ command)) > 1.0 + 1e-6:
                    raise ValueError(
                        "controller command leaves the sealed linear interface domain"
                    )
                episode_commands.append(tuple(float(value) for value in command))
                plant.step_interface(environment, interface, command)
                context = plant.append_observation(context, environment)
            command_traces[name].append(tuple(episode_commands))
            errors[name].append(
                pixel_target_error(system, plant.current_pixels(environment), episode.target_pixels)
            )
    seal.assert_unchanged()
    return ControlResult(
        errors={name: tuple(values) for name, values in errors.items()},
        interface_name=interface.name,
        episodes=len(episodes),
        control_steps=system.control_steps,
        planner_budget={
            "candidatesPerDecision": config.candidates,
            "iterationsPerDecision": config.iterations,
            "elitesPerIteration": config.elites,
            "horizon": config.horizon,
            "candidateEvaluationsPerDecision": config.candidates * config.iterations,
            "pairedCandidateNoiseAcrossLearnedPlanners": 1,
            "activationRolloutMicroBatch": config.activation_rollout_batch_size,
            "commonLinearInterfaceCommandBound": (
                protocol.common_interface_command_bound
            ),
            "linearInterfaceBoundFormula": protocol.bound_formula,
        },
        episode_identifiers=tuple(episode.identifier for episode in episodes),
        interface_command_traces={
            name: tuple(values) for name, values in command_traces.items()
        },
        planner_seed_schedule_sha256=_planner_seed_schedule_sha256(
            seed=seed,
            episode_offset=episode_offset,
            episodes=len(episodes),
            control_steps=system.control_steps,
        ),
        physical_protocol=protocol,
    )


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    low: float
    high: float
    confidence: float
    resamples: int
    seed: int


def paired_bootstrap_ci(
    candidate_errors: Sequence[float],
    reference_errors: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
    relative: bool = True,
) -> BootstrapCI:
    """Paired percentile interval for improvement (positive is better)."""

    candidate = np.asarray(candidate_errors, dtype=np.float64)
    reference = np.asarray(reference_errors, dtype=np.float64)
    if candidate.ndim != 1 or candidate.shape != reference.shape or candidate.size < 2:
        raise ValueError("paired bootstrap inputs must be equal nontrivial vectors")
    if not bool(np.isfinite(candidate).all() and np.isfinite(reference).all()):
        raise ValueError("bootstrap errors must be finite")
    if not 0.0 < confidence < 1.0 or resamples < 100:
        raise ValueError("invalid bootstrap confidence or resample count")

    def statistic(candidates: np.ndarray, references: np.ndarray) -> np.ndarray:
        difference = references.mean(axis=-1) - candidates.mean(axis=-1)
        if not relative:
            return difference
        return difference / np.maximum(references.mean(axis=-1), 1e-12)

    point = float(statistic(candidate[None], reference[None])[0])
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    chunk = 1_024
    for start in range(0, resamples, chunk):
        stop = min(start + chunk, resamples)
        indices = rng.integers(0, candidate.size, size=(stop - start, candidate.size))
        values[start:stop] = statistic(candidate[indices], reference[indices])
    tail = 0.5 * (1.0 - confidence)
    low, high = np.quantile(values, (tail, 1.0 - tail))
    return BootstrapCI(point, float(low), float(high), confidence, resamples, seed)


def realizability_gate_metrics(
    metrics: RealizabilityMetrics,
    *,
    single_horizon_mean_cosine: float | None = None,
    shuffled_lens_mean_cosine: float | None = None,
) -> dict[str, Any]:
    finite = all(
        math.isfinite(value)
        for value in (
            metrics.mean_cosine,
            metrics.sign_agreement,
            metrics.magnitude_r2,
            *metrics.axis_mean_cosines,
        )
    )
    checks: dict[str, bool] = {
        "finite": finite,
        "meanCosineAtLeast0.85": metrics.mean_cosine >= 0.85,
        "everyAxisCosineAtLeast0.75": min(metrics.axis_mean_cosines) >= 0.75,
        "signAgreementAtLeast0.85": metrics.sign_agreement >= 0.85,
        "magnitudeR2AtLeast0.60": metrics.magnitude_r2 >= 0.60,
    }
    if single_horizon_mean_cosine is not None:
        checks["beatsSingleHorizonBy0.10"] = (
            metrics.mean_cosine - single_horizon_mean_cosine >= 0.10
        )
    if shuffled_lens_mean_cosine is not None:
        checks["beatsShuffledLensBy0.10"] = (
            metrics.mean_cosine - shuffled_lens_mean_cosine >= 0.10
        )
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics.as_dict()}


def control_gate_metrics(
    result: ControlResult,
    *,
    no_jacobian_errors: Sequence[float] | None = None,
    shuffled_lens_errors: Sequence[float] | None = None,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    required = ("structured", "unstructured", "activation", "coast")
    missing = tuple(name for name in required if name not in result.errors)
    if missing:
        return {
            "passed": False,
            "checks": {"allRegisteredControllersPresent": False},
            "missingControllers": missing,
            "metrics": result.as_dict(),
        }
    errors = {name: np.asarray(values, dtype=np.float64) for name, values in result.errors.items()}
    means = {name: float(values.mean()) for name, values in errors.items()}
    structured = errors["structured"]
    coast = errors["coast"]
    improvement_coast = float((coast.mean() - structured.mean()) / max(coast.mean(), 1e-12))
    # Every non-ablation learned planner is audited separately.  Selecting the
    # lowest empirical mean first and then bootstrapping only that selected
    # vector understates uncertainty.  The name discovery deliberately admits
    # future independently trained WM baselines without weakening this gate.
    non_baseline_controllers = {
        "structured",
        "no_jacobian",
        "shuffled_lens",
        "coast",
        "random",
    }
    learned_baseline_names = tuple(
        sorted(name for name in errors if name not in non_baseline_controllers)
    )
    if not {"unstructured", "activation"}.issubset(learned_baseline_names):
        raise AssertionError("registered learned-baseline discovery drifted")
    comparisons: dict[str, dict[str, Any]] = {}
    for baseline_name in learned_baseline_names:
        baseline = errors[baseline_name]
        improvement = float(
            (baseline.mean() - structured.mean()) / max(baseline.mean(), 1e-12)
        )
        wins = float((structured < baseline).mean())
        interval = paired_bootstrap_ci(
            structured,
            baseline,
            resamples=bootstrap_resamples,
            # Reuse one preregistered episode-resampling schedule for every
            # comparator.  Each CI is still computed separately, while adding
            # a future baseline cannot perturb the existing comparisons.
            seed=bootstrap_seed,
        )
        comparisons[baseline_name] = {
            "improvement": improvement,
            "winsFraction": wins,
            "pairedBootstrap": interval.__dict__,
            "checks": {
                "improvementAtLeast0.15": improvement >= 0.15,
                "winsAtLeast0.65": wins >= 0.65,
                "paired95BootstrapExcludesZero": interval.low > 0.0,
            },
        }
    better_name = min(learned_baseline_names, key=lambda name: means[name])
    better_comparison = comparisons[better_name]
    checks: dict[str, bool] = {
        "allRegisteredControllersPresent": True,
        "improvementVsCoastAtLeast0.25": improvement_coast >= 0.25,
        "improvementVsEveryLearnedBaselineAtLeast0.15": all(
            value["checks"]["improvementAtLeast0.15"]
            for value in comparisons.values()
        ),
        "winsAgainstEveryLearnedBaselineAtLeast0.65": all(
            value["checks"]["winsAtLeast0.65"] for value in comparisons.values()
        ),
        "paired95BootstrapExcludesZeroForEveryLearnedBaseline": all(
            value["checks"]["paired95BootstrapExcludesZero"]
            for value in comparisons.values()
        ),
    }
    if no_jacobian_errors is not None:
        no_jacobian = np.asarray(no_jacobian_errors, dtype=np.float64)
        if no_jacobian.shape != structured.shape:
            raise ValueError("no-Jacobian errors must be episode-paired")
        checks["noJacobianAtLeast10PercentWorse"] = (
            no_jacobian.mean() >= 1.10 * structured.mean()
        )
    if shuffled_lens_errors is not None:
        shuffled = np.asarray(shuffled_lens_errors, dtype=np.float64)
        if shuffled.shape != structured.shape:
            raise ValueError("shuffled-lens errors must be episode-paired")
        checks["shuffledLensAtLeast10PercentWorse"] = (
            shuffled.mean() >= 1.10 * structured.mean()
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "betterLearnedBaseline": better_name,
        "improvementVsCoast": improvement_coast,
        "improvementVsBetterPlanner": better_comparison["improvement"],
        "winsFractionVsBetterPlanner": better_comparison["winsFraction"],
        "pairedBootstrapVsBetterPlanner": better_comparison["pairedBootstrap"],
        "learnedBaselineComparisons": comparisons,
        "metrics": result.as_dict(),
    }


@dataclass(frozen=True)
class InterfaceExecutionEvidence:
    """Typed static execution fingerprint for one deployment interface.

    Command traces and outcomes are intentionally absent: they are expected to
    differ.  Every graph/configuration input is present, while the two allowed
    interface-specific leaves are explicit: ``interface_protocol`` and the
    constant calibration matrices ``T`` represented by their SHA-256 values.
    """

    interface_protocol: LinearInterfaceProtocol
    training_lineage_sha256: str
    physical_sha256: str
    module_hashes_before: Mapping[str, str]
    module_hashes_after: Mapping[str, str]
    controller_graph_sha256: str
    cem_config: Mapping[str, Any]
    episode_seed: int
    planner_seed: int
    episodes: int
    control_steps: int
    controller_names: tuple[str, ...]
    target_source: str
    episode_identifiers: tuple[str, ...]
    episode_set_sha256: str
    target_set_sha256: str
    planner_seed_schedule_sha256: str
    calibration_matrix_sha256: Mapping[str, str]
    calibration_matrix_schema: Mapping[str, tuple[str, tuple[int, int]]]

    def __post_init__(self) -> None:
        try:
            json.dumps(dict(self.cem_config), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("CEM execution evidence is not canonical JSON") from error
        digest_values = (
            self.training_lineage_sha256,
            self.physical_sha256,
            self.controller_graph_sha256,
            self.episode_set_sha256,
            self.target_set_sha256,
            self.planner_seed_schedule_sha256,
            *self.module_hashes_before.values(),
            *self.module_hashes_after.values(),
            *self.calibration_matrix_sha256.values(),
        )
        if any(
            type(value) is not str
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in digest_values
        ):
            raise ValueError("interface execution evidence contains a malformed SHA-256")
        if (
            self.interface_protocol.interface_name not in {"native", "unseen"}
            or type(self.episode_seed) is not int
            or type(self.planner_seed) is not int
            or self.episodes != 64
            or self.control_steps < 1
            or len(self.episode_identifiers) != self.episodes
            or len(set(self.episode_identifiers)) != self.episodes
            or not self.controller_names
            or len(set(self.controller_names)) != len(self.controller_names)
            or self.target_source != "categorical_pixels_only"
            or not self.module_hashes_before
            or not self.calibration_matrix_sha256
            or set(self.calibration_matrix_sha256) != set(self.calibration_matrix_schema)
            or set(self.calibration_matrix_sha256)
            != set(self.controller_names).difference({"coast", "random"})
        ):
            raise ValueError("interface execution evidence schema is invalid")
        for dtype, shape in self.calibration_matrix_schema.values():
            if (
                type(dtype) is not str
                or not dtype
                or type(shape) is not tuple
                or len(shape) != 2
                or any(type(value) is not int or value < 1 for value in shape)
            ):
                raise ValueError("constant calibration T schema is invalid")

        # Fail closed if a contact-task episode set is mixed with any foreign
        # identifier.  The semantic task hash is therefore transitively bound
        # into the already authenticated episode set.
        _episode_target_task_identity(self.episode_identifiers)

    @property
    def target_task(self) -> dict[str, str]:
        return _episode_target_task_identity(self.episode_identifiers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interfaceProtocol": self.interface_protocol.to_dict(),
            "trainingLineageSha256": self.training_lineage_sha256,
            "physicalSha256": self.physical_sha256,
            "moduleHashesBefore": dict(self.module_hashes_before),
            "moduleHashesAfter": dict(self.module_hashes_after),
            "controllerGraphSha256": self.controller_graph_sha256,
            "cemConfig": dict(self.cem_config),
            "episodeSeed": self.episode_seed,
            "plannerSeed": self.planner_seed,
            "episodes": self.episodes,
            "controlSteps": self.control_steps,
            "controllerNames": self.controller_names,
            "targetSource": self.target_source,
            "targetTask": self.target_task,
            "episodeIdentifiers": self.episode_identifiers,
            "episodeSetSha256": self.episode_set_sha256,
            "targetSetSha256": self.target_set_sha256,
            "plannerSeedScheduleSha256": self.planner_seed_schedule_sha256,
            "calibrationMatrixSha256": dict(self.calibration_matrix_sha256),
            "calibrationMatrixSchema": dict(self.calibration_matrix_schema),
        }


@dataclass(frozen=True)
class InterfaceTransferEvidence:
    native: InterfaceExecutionEvidence
    unseen: InterfaceExecutionEvidence

    def __post_init__(self) -> None:
        if (
            self.native.interface_protocol.interface_name != "native"
            or self.unseen.interface_protocol.interface_name != "unseen"
        ):
            raise ValueError("transfer evidence must contain native then unseen execution")

    def to_dict(self) -> dict[str, Any]:
        return {"native": self.native.to_dict(), "unseen": self.unseen.to_dict()}


def _same_protocol_except_interface(
    native: LinearInterfaceProtocol,
    unseen: LinearInterfaceProtocol,
) -> bool:
    first = native.to_dict()
    second = unseen.to_dict()
    for key in ("interface", "nativeFromInterface"):
        first.pop(key)
        second.pop(key)
    return first == second


def _registered_interface_pair(evidence: InterfaceTransferEvidence) -> bool:
    system_name = evidence.native.interface_protocol.system_name
    if evidence.unseen.interface_protocol.system_name != system_name:
        return False
    try:
        system = SYSTEMS[system_name]
        registered = fixed_interfaces(system)
        native = linear_interface_protocol(system, registered["native"])
        unseen = linear_interface_protocol(system, registered["unseen"])
    except (KeyError, ValueError):
        return False
    return (
        evidence.native.interface_protocol == native
        and evidence.unseen.interface_protocol == unseen
        and not np.array_equal(
            evidence.native.interface_protocol.native_from_interface,
            evidence.unseen.interface_protocol.native_from_interface,
        )
    )


def interface_transfer_gate_metrics(
    native_control_gate: Mapping[str, Any],
    unseen_control_gate: Mapping[str, Any],
    unseen_realizability_gate: Mapping[str, Any],
    *,
    evidence: InterfaceTransferEvidence,
) -> dict[str, Any]:
    native = float(native_control_gate.get("improvementVsCoast", float("-inf")))
    unseen = float(unseen_control_gate.get("improvementVsCoast", float("-inf")))
    retention = unseen / native if native > 0.0 else float("-inf")
    native_evidence = evidence.native
    unseen_evidence = evidence.unseen
    native_modules_unchanged = (
        dict(native_evidence.module_hashes_before)
        == dict(native_evidence.module_hashes_after)
    )
    unseen_modules_unchanged = (
        dict(unseen_evidence.module_hashes_before)
        == dict(unseen_evidence.module_hashes_after)
    )
    checks = {
        "retentionAtLeast0.80": retention >= 0.80,
        "unseenRealizabilityPasses": bool(unseen_realizability_gate.get("passed", False)),
        "registeredDistinctPhysicalInterfaces": _registered_interface_pair(evidence),
        "interfaceProtocolOtherwiseExact": _same_protocol_except_interface(
            native_evidence.interface_protocol,
            unseen_evidence.interface_protocol,
        ),
        "trainingLineageExact": (
            native_evidence.training_lineage_sha256
            == unseen_evidence.training_lineage_sha256
        ),
        "sharedPhysicalEvidenceExact": (
            native_evidence.physical_sha256 == unseen_evidence.physical_sha256
        ),
        "nativeModuleHashesUnchanged": native_modules_unchanged,
        "unseenModuleHashesUnchanged": unseen_modules_unchanged,
        "moduleHashesAcrossInterfacesExact": (
            dict(native_evidence.module_hashes_before)
            == dict(unseen_evidence.module_hashes_before)
            and dict(native_evidence.module_hashes_after)
            == dict(unseen_evidence.module_hashes_after)
        ),
        "controllerGraphExact": (
            native_evidence.controller_graph_sha256
            == unseen_evidence.controller_graph_sha256
        ),
        "cemConfigExact": dict(native_evidence.cem_config) == dict(unseen_evidence.cem_config),
        "episodeSeedExact": native_evidence.episode_seed == unseen_evidence.episode_seed,
        "plannerSeedExact": native_evidence.planner_seed == unseen_evidence.planner_seed,
        "episodeCountExact": native_evidence.episodes == unseen_evidence.episodes,
        "controlStepsExact": native_evidence.control_steps == unseen_evidence.control_steps,
        "controllerSetAndOrderExact": (
            native_evidence.controller_names == unseen_evidence.controller_names
        ),
        "targetSourceExact": native_evidence.target_source == unseen_evidence.target_source,
        "targetTaskExact": native_evidence.target_task == unseen_evidence.target_task,
        "episodeIdentifiersExact": (
            native_evidence.episode_identifiers == unseen_evidence.episode_identifiers
        ),
        "episodeSetExact": (
            native_evidence.episode_set_sha256 == unseen_evidence.episode_set_sha256
        ),
        "targetSetExact": (
            native_evidence.target_set_sha256 == unseen_evidence.target_set_sha256
        ),
        "plannerSeedScheduleExact": (
            native_evidence.planner_seed_schedule_sha256
            == unseen_evidence.planner_seed_schedule_sha256
        ),
        # T values may differ, but their controller slots, dtype and fixed
        # matrix shapes must be identical.  No other mutable graph leaf exists
        # in this typed execution fingerprint.
        "constantCalibrationTSlotsExact": (
            set(native_evidence.calibration_matrix_sha256)
            == set(unseen_evidence.calibration_matrix_sha256)
            and dict(native_evidence.calibration_matrix_schema)
            == dict(unseen_evidence.calibration_matrix_schema)
        ),
    }
    static_equivalence_checks = tuple(
        name
        for name in checks
        if name not in {"retentionAtLeast0.80", "unseenRealizabilityPasses"}
    )
    checks["onlyPhysicalInterfaceAndConstantTMayDiffer"] = all(
        checks[name] for name in static_equivalence_checks
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "retention": retention,
        "typedTransferEvidence": evidence.to_dict(),
    }


def experiment_f_gate_metrics(
    native_realizability: RealizabilityMetrics,
    unseen_realizability: RealizabilityMetrics,
    native_control: ControlResult,
    unseen_control: ControlResult,
    *,
    transfer_evidence: InterfaceTransferEvidence,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Compose the locked physical Gates 6--8 for one system."""

    native_r = realizability_gate_metrics(native_realizability)
    unseen_r = realizability_gate_metrics(unseen_realizability)
    native_c = control_gate_metrics(
        native_control,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    unseen_c = control_gate_metrics(
        unseen_control,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed + 1,
    )
    transfer = interface_transfer_gate_metrics(
        native_c,
        unseen_c,
        unseen_r,
        evidence=transfer_evidence,
    )
    gates = {
        "nativeRealizability": native_r,
        "unseenRealizability": unseen_r,
        "nativeControl": native_c,
        "unseenControl": unseen_c,
        "interfaceTransfer": transfer,
    }
    return {"passed": all(item["passed"] for item in gates.values()), "gates": gates}


def _rgb_to_classes(frame: np.ndarray) -> torch.Tensor:
    palette = np.stack(tuple(PALETTE.values())).astype(np.int32)
    rgb = np.asarray(frame, dtype=np.int32)
    distance = ((rgb[..., None, :] - palette) ** 2).sum(axis=-1)
    return torch.from_numpy(distance.argmin(axis=-1).astype(np.uint8))


def _copy_world_state(state: WorldState) -> WorldState:
    return WorldState(
        player_position=state.player_position.copy(),
        player_velocity=state.player_velocity.copy(),
        puck_position=state.puck_position.copy(),
        puck_velocity=state.puck_velocity.copy(),
        score=state.score,
        tick=state.tick,
        reset_timer=state.reset_timer,
        last_event=state.last_event,
    )


def builtin_pixel_plant(system: EvaluationSystem | str) -> PixelPlant:
    """Deployment wrapper for the registered pendulum and Blocket simulators."""

    definition = SYSTEMS[system] if isinstance(system, str) else system

    def clone(environment: Any) -> Any:
        if definition.name == "blocket":
            result = BlocketLeagueEnv(seed=0, config=environment.config)
            result.state = _copy_world_state(environment.state)
            return result
        if definition.name == "pendulum":
            result = PendulumEnv(seed=0, config=environment.config)
            result.set_state(environment.state)
            return result
        return copy.deepcopy(environment)

    def step(environment: Any, interface: PhysicalInterface, command: np.ndarray) -> None:
        command = np.asarray(command, dtype=np.float64)
        protocol = linear_interface_protocol(definition, interface)
        if (
            command.shape != (definition.physical_action_size,)
            or not bool(np.isfinite(command).all())
        ):
            raise ValueError("interface command has the wrong shape")
        if bool(
            (
                np.abs(command)
                > protocol.common_interface_command_bound + 1e-7
            ).any()
        ):
            raise ValueError("interface command exceeds the common linear-domain box")
        native = interface.matrix() @ command
        if float(np.linalg.norm(native, ord=2)) > protocol.native_l2_limit + 1e-7:
            raise ValueError("transformed command would trigger plant saturation")
        if definition.name == "blocket":
            environment.step_vector(np.asarray(native, dtype=np.float32))
        elif definition.name == "pendulum":
            environment.step(
                float(native[0]) * environment.config.max_torque
            )
        else:
            raise KeyError(f"no built-in step for {definition.name!r}")

    def append(context: torch.Tensor, environment: Any) -> torch.Tensor:
        frame = _rgb_to_classes(environment.render())
        return torch.cat((context[1:], frame[None]), dim=0)

    def current(environment: Any) -> torch.Tensor:
        return _rgb_to_classes(environment.render())

    return PixelPlant(clone, step, append, current)


def make_builtin_probe_candidates(
    system: EvaluationSystem | str,
    *,
    history_frames: int,
    count: int,
    seed: int,
    image_size: int = 64,
) -> list[ProbeCandidate]:
    """Create a fixed pre-response candidate pool and export pixels only."""

    definition = SYSTEMS[system] if isinstance(system, str) else system
    if history_frames < 1 or count < 1 or image_size < 8:
        raise ValueError("history_frames/count/image_size are invalid")
    candidates = []
    for index in range(count):
        item_seed = seed + index * 104_729
        rng = np.random.default_rng(item_seed)
        if definition.name == "blocket":
            environment = BlocketLeagueEnv(
                seed=item_seed,
                config=experiment_f_blocket_world_config(image_size=image_size),
            )
            for _ in range(100):
                player = rng.uniform((0.16, 0.16), (0.58, 0.84)).astype(np.float32)
                puck = rng.uniform((0.42, 0.16), (0.84, 0.84)).astype(np.float32)
                if np.linalg.norm(player - puck) > 0.19:
                    break
            environment.state.player_position = player
            environment.state.puck_position = puck
            environment.state.player_velocity = rng.uniform(-0.45, 0.45, size=2).astype(np.float32)
            environment.state.puck_velocity = rng.uniform(-0.38, 0.38, size=2).astype(np.float32)
            environment.state.reset_timer = 0
        elif definition.name == "pendulum":
            environment = PendulumEnv(
                seed=item_seed,
                config=PendulumConfig(image_size=image_size),
            )
        else:
            raise KeyError(f"no built-in candidate generator for {definition.name!r}")
        frames = [_rgb_to_classes(environment.render())]
        for _ in range(history_frames - 1):
            if definition.name == "blocket":
                environment.step_vector(np.zeros(2, dtype=np.float32))
            else:
                environment.step(0.0)
            frames.append(_rgb_to_classes(environment.render()))
        candidates.append(
            ProbeCandidate(
                identifier=f"{definition.name}-{item_seed}",
                context=torch.stack(frames),
                environment=environment,
            )
        )
    return candidates


def make_builtin_control_episodes(
    system: EvaluationSystem | str,
    *,
    history_frames: int,
    count: int = 64,
    seed: int,
    image_size: int = 64,
) -> list[PixelControlEpisode]:
    """Create paired episodes whose controller-visible target is one image."""

    definition = SYSTEMS[system] if isinstance(system, str) else system
    if history_frames < 1 or count < 1 or image_size < 8:
        raise ValueError("history_frames/count/image_size are invalid")

    if definition.name == "blocket":
        return _make_builtin_blocket_contact_episodes(
            definition,
            history_frames=history_frames,
            count=count,
            seed=seed,
            image_size=image_size,
        )

    candidates = make_builtin_probe_candidates(
        definition,
        history_frames=history_frames,
        count=count,
        seed=seed,
        image_size=image_size,
    )
    episodes = []
    for index, candidate in enumerate(candidates):
        rng = np.random.default_rng(seed + index * 104_729 + 41)
        target_environment = builtin_pixel_plant(definition).clone_environment(
            candidate.environment
        )
        if definition.name == "pendulum":
            offset = float(rng.choice((-1.0, 1.0)) * rng.uniform(0.75, 1.35))
            angle = float(wrap_angle(target_environment.state.angle + offset))
            target = _rgb_to_classes(
                pendulum_target_frames(
                    angle,
                    frames=1,
                    image_size=target_environment.config.image_size,
                )[0]
            )
        else:
            raise KeyError(f"no built-in control episode for {definition.name!r}")
        episodes.append(
            PixelControlEpisode(
                identifier=candidate.identifier,
                environment=candidate.environment,
                context=candidate.context,
                target_pixels=target,
            )
        )
    return episodes


def _make_builtin_blocket_contact_episodes(
    definition: EvaluationSystem,
    *,
    history_frames: int,
    count: int,
    seed: int,
    image_size: int,
) -> list[PixelControlEpisode]:
    """Build contact-required targets from private, admissible simulator rollouts.

    Initial frames and targets cross the environment boundary only as categorical
    pixels.  The source-side direction and constant thrust below are discarded
    after target rendering and are never attached to :class:`PixelControlEpisode`.
    """

    if definition != SYSTEMS["blocket"]:
        raise ValueError("the contact task is defined only for registered Blocket")
    config = experiment_f_blocket_world_config(image_size=image_size)
    minimum_separation = config.player_radius + config.puck_radius
    player_low = config.wall + config.player_radius + 0.01
    player_high = 1.0 - config.wall - config.player_radius - 0.01
    puck_low = config.wall + config.puck_radius + 0.01
    puck_high = 1.0 - config.wall - config.puck_radius - 0.01
    golden_fraction = (math.sqrt(5.0) - 1.0) / 2.0
    zero = np.zeros(2, dtype=np.float32)
    episodes: list[PixelControlEpisode] = []

    for index in range(count):
        item_seed = seed + index * 104_729
        rng = np.random.default_rng(item_seed ^ 0x0B10C0E7)
        # Episode zero is an auditable straight-right source oracle.  The
        # remaining low-discrepancy directions cover the full actuator plane.
        angle = (
            0.0
            if index == 0
            else 2.0 * math.pi * ((index * golden_fraction) % 1.0)
        )
        direction = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float32)
        surface_gap = float(rng.uniform(*_BLOCKET_CONTACT_SURFACE_GAP))
        separation = minimum_separation + surface_gap

        initial_puck: np.ndarray | None = None
        initial_player: np.ndarray | None = None
        for _attempt in range(10_000):
            puck = rng.uniform((0.28, 0.28), (0.72, 0.72)).astype(np.float32)
            player = puck - direction * separation
            # Reserve enough straight-line space for the complete registered
            # displacement interval, including the trailing player disc.
            projected_puck = puck + direction * BLOCKET_CONTACT_MAX_PIXEL_DISPLACEMENT
            projected_player = projected_puck - direction * minimum_separation
            if (
                bool(((player >= player_low) & (player <= player_high)).all())
                and bool(((projected_player >= player_low) & (projected_player <= player_high)).all())
                and bool(((puck >= puck_low) & (puck <= puck_high)).all())
                and bool(((projected_puck >= puck_low) & (projected_puck <= puck_high)).all())
            ):
                initial_puck = puck
                initial_player = player
                break
        if initial_puck is None or initial_player is None:  # pragma: no cover
            raise RuntimeError("failed to place a registered Blocket contact episode")

        environment = BlocketLeagueEnv(seed=item_seed, config=config)
        environment.state.player_position = initial_player.copy()
        environment.state.puck_position = initial_puck.copy()
        environment.state.player_velocity[:] = 0.0
        environment.state.puck_velocity[:] = 0.0
        environment.state.score = 0
        environment.state.tick = 0
        environment.state.reset_timer = 0
        environment.state.last_event = "coast"

        # A real zero-input history, rather than synthetic frame repetition,
        # establishes that context[-1] is exactly the plant's current render.
        frames: list[torch.Tensor] = []
        for history_index in range(history_frames):
            frames.append(_rgb_to_classes(environment.render()))
            if history_index + 1 < history_frames:
                environment.step_vector(zero)
        context = torch.stack(frames)
        if not torch.equal(context[-1], _rgb_to_classes(environment.render())):
            raise AssertionError("Blocket context does not end at the plant state")

        target_environment = BlocketLeagueEnv(seed=0, config=config)
        target_environment.state = _copy_world_state(environment.state)
        target_threshold = float(rng.uniform(*_BLOCKET_CONTACT_SOURCE_THRESHOLD))
        source_touched_puck = False
        source_displacement = 0.0
        oracle_native = BLOCKET_CONTACT_ORACLE_NATIVE_THRUST * direction
        for _source_step in range(SYSTEMS["blocket"].control_steps):
            target_environment.step_vector(oracle_native)
            source_touched_puck = source_touched_puck or (
                target_environment.state.last_event == "impact"
            )
            source_displacement = float(
                np.linalg.norm(
                    target_environment.state.puck_position - initial_puck
                )
            )
            if source_displacement >= target_threshold:
                break
        if not source_touched_puck or source_displacement < target_threshold:
            raise RuntimeError("source-only oracle did not realize the contact target")
        if target_environment.state.reset_timer != 0 or target_environment.config.goals_enabled:
            raise AssertionError("contact target left the registered no-reset arena")

        target = _rgb_to_classes(target_environment.render())
        pixel_displacement = pixel_target_error(definition, context[-1], target)
        if not (
            BLOCKET_CONTACT_MIN_PIXEL_DISPLACEMENT
            <= pixel_displacement
            <= BLOCKET_CONTACT_MAX_PIXEL_DISPLACEMENT
        ):
            raise AssertionError(
                "rendered Blocket contact target left its displacement interval"
            )

        episode_digest = hashlib.sha256()
        episode_digest.update(BLOCKET_CONTACT_TARGET_TASK_SHA256.encode("ascii"))
        episode_digest.update(str((index, item_seed)).encode("ascii"))
        for tensor in (context, target):
            value = tensor.detach().cpu().contiguous()
            episode_digest.update(str((tuple(value.shape), value.dtype)).encode("ascii"))
            episode_digest.update(value.numpy().tobytes())
        identifier = (
            f"{_BLOCKET_CONTACT_IDENTIFIER_PREFIX}{index:03d}:"
            f"{episode_digest.hexdigest()}"
        )
        episodes.append(
            PixelControlEpisode(
                identifier=identifier,
                environment=environment,
                context=context,
                target_pixels=target,
            )
        )
    return episodes


__all__ = [
    "BLOCKET_CONTACT_MAX_PIXEL_DISPLACEMENT",
    "BLOCKET_CONTACT_MIN_PIXEL_DISPLACEMENT",
    "BLOCKET_CONTACT_ORACLE_NATIVE_THRUST",
    "BLOCKET_CONTACT_TARGET_TASK_NAME",
    "BLOCKET_CONTACT_TARGET_TASK_SHA256",
    "BootstrapCI",
    "CEMConfig",
    "CEMPlan",
    "CalibrationResult",
    "ControlResult",
    "DOptimalSelection",
    "DynamicsEvaluationAdapter",
    "EvaluationSystem",
    "FrozenActivationWriteWorldModel",
    "FrozenEvaluationSeal",
    "FrozenLatentPlannerSpec",
    "InterfaceExecutionEvidence",
    "InterfaceTransferEvidence",
    "LinearInterfaceProtocol",
    "PAIRED_CALIBRATION_STATES_PER_AXIS",
    "PairedCalibrationResponseBank",
    "PairedHeldoutResponseBank",
    "PhysicalInterface",
    "PixelControlEpisode",
    "PixelPlant",
    "PuckOnlyPixelObjective",
    "ProbeCandidate",
    "RIDGE_COEFFICIENT",
    "RealizabilityMetrics",
    "SYSTEMS",
    "adapt_dynamics_for_evaluation",
    "activation_calibration_from_response_frame",
    "builtin_pixel_plant",
    "calibrate_activation_interface_after_freeze",
    "calibrate_interface_after_freeze",
    "collect_paired_calibration_response_bank",
    "collect_paired_heldout_response_bank",
    "cem_frozen_world_model_mpc",
    "cem_pixel_target_mpc",
    "capture_experiment_f_evaluation_seal",
    "control_gate_metrics",
    "control_trace_sha256",
    "evaluate_closed_loop_controllers",
    "evaluate_heldout_activation_from_response_bank",
    "evaluate_heldout_activation_realizability",
    "evaluate_heldout_realizability",
    "evaluate_heldout_realizability_from_response_bank",
    "evaluation_system_from_direct_spec",
    "experiment_f_gate_metrics",
    "fixed_interfaces",
    "fit_interface_calibration_from_response_bank",
    "interface_transfer_gate_metrics",
    "make_builtin_control_episodes",
    "make_builtin_probe_candidates",
    "make_puck_only_pixel_objective",
    "paired_bootstrap_ci",
    "pixel_target_error",
    "puck_only_pixel_cost",
    "realizability_gate_metrics",
    "registered_cem_config",
    "registered_linear_interface_command_bound",
    "select_d_optimal_probe_states",
    "select_d_optimal_activation_probe_states",
    "select_shared_maximin_probe_states",
]
