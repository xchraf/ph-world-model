# Experiment E — action-free Jacobian ports for port-Hamiltonian control

## Locked claim

A frozen video transformer pretrained without actions can expose causal control
ports through activation Jacobians.  A passive port-Hamiltonian latent model can
use those ports, after only a small analytic one-step calibration, to control a
real environment and transfer to an unseen actuator interface better than an
otherwise matched unstructured latent world-model planner.

This is a single-seed falsification run.  It is not a reproducibility claim.
No failed gate may be replaced, removed, or re-thresholded after results exist.

## Systems

The identical experimental pipeline is run on two systems.

1. **Blocket League:** two freely moving and colliding discs; the deployment
   actuator applies a two-dimensional thrust to the player disc.
2. **Damped pendulum:** a passive nonlinear oscillator rendered as video; the
   deployment actuator applies a scalar pivot torque.

The Blocket transformer is the existing passive-video checkpoint.  The
pendulum transformer is pretrained from passive zero-torque videos in the same
job and is frozen before any downstream fitting.

## Information firewall

Pretraining and all downstream gradient-based optimization may contain only
passive pixel histories.  They contain no action tensor, action label, applied
force, torque, control-conditioned trajectory, simulator state, coordinate, or
event label.

Simulator state is collected in disjoint audit and deployment suites only.  It
may be used to initialize paired one-step actuator queries and to score final
control, never as an optimization target.  Target images supplied to a
controller are observations, not coordinates.

The transformer weights are frozen byte-for-byte before the passive latent
adapter, dynamics, or port fitting begins.  Their pre/post SHA-256 hashes must
match.

## Architecture

For each system, a trainable visual adapter reads internal activations from the
frozen transformer and produces a low-dimensional state.  A visual decoder and
the passive dynamics are trained only on passive pixels.  The structured branch
uses

```text
dx/dt = (J(x) - R(x)) grad H(x) + B(x) v,
J(x) = -J(x)^T, R(x) >= 0.
```

Autonomous discrete rollout uses a differentiable energy-level projection after
the midpoint proposal.  This is part of the registered numerical architecture:
it preserves the learned continuous vector field and removes only positive
energy introduced by time discretization.  Controlled steps remain governed by
the unprojected pH supply term.  The forward projection uses a bounded
energy-level search; passive fitting uses its straight-through derivative, so
the numerical guarantee is evaluated on exactly the same forward map used at
training time.

During passive fitting `v` is identically absent; the port network is not
optimized by trajectory prediction.

After the passive state and dynamics are fixed, the port targets are extracted
as follows at every sampled state:

1. insert an infinitesimal write into the controlled object's token at a frozen
   transformer block;
2. differentiate the next predicted visual observable with respect to that
   write;
3. map the resulting causal activation direction through the exact Jacobian of
   the frozen-backbone state adapter;
4. fit `B(x)` to these state-dependent Jacobian targets.

No actuator observation is involved in this stage.  A parameter-matched
unstructured autonomous neural ODE receives the same frozen visual state and
the same Jacobian-derived port fitting protocol.

## Causal and realizability tests

Paired positive/negative activation writes are evaluated on unseen passive
contexts.  They must alter the predicted controlled observable in the expected
direction and exceed norm-matched random activation directions.

Physical grounding uses exactly four independent paired one-step probe states
per actuator dimension (`+e_i` and `-e_i`): 16 environment steps for Blocket
League and 8 for the pendulum.  A constant interface map is solved analytically
by ridge least squares.  There is no gradient update and no multi-step
action-conditioned trajectory.

Grounding is scored on disjoint paired one-step trials.  Realizability means
that the predicted port displacement aligns with the actual re-encoded pixel
displacement.

## Control and transfer

Both the structured and parameter-matched unstructured planners use the same
encoder, decoder, Jacobian port targets, calibration budget, horizon, optimizer
budget, observations, and action bounds.  Each repeatedly plans in its own
latent world model, executes only the first command in the real simulator, and
replans from pixels.

Each controller is evaluated through:

- a native deployment interface;
- an interface absent from all prior stages, with a hidden sign/scale change
  for the pendulum and a hidden rotated, permuted, anisotropically scaled map
  for Blocket League.

Only the small analytic interface calibration is repeated.  No network may be
updated for transfer.

## Preregistered gates

Every gate must pass independently on both systems unless stated otherwise.

1. **Action firewall:** zero action/control/state tensors used in any gradient
   update; transformer parameter hash unchanged.
2. **Passive prediction:** pH latent rollout error is no worse than 1.15 times
   the parameter-matched neural ODE at the registered planning horizon.
3. **Causal lens:** expected-sign fraction at least 0.75 and paired causal
   effect at least twice the norm-matched random-direction effect.
4. **Held-out realizability:** mean cosine at least 0.70 and target sign
   agreement at least 0.75 after the fixed few-shot calibration.
5. **Real control:** pH final pixel-defined target error at least 10% below the
   generic unstructured planner and pH wins at least 60% of paired episodes.
   This must hold separately for the native and unseen interfaces.
6. **Transfer:** unseen-interface pH improvement over coast retains at least
   80% of its native-interface improvement, with no neural parameter update.
7. **Power accounting:** maximum continuous balance defect at most `1e-5`, and
   zero-input discrete energy increase fraction at most `0.01` over the tested
   passive step size.
8. **Two-system conjunction:** gates 1--7 pass for both Blocket League and the
   damped pendulum in the same locked run.

The only positive single-seed outcome string is
`breakthrough_supported_single_seed_two_systems`.  Any failed gate produces
`breakthrough_not_supported_single_seed` with all failures reported.
