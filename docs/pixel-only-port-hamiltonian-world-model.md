# Experiment D2 — pixel/action-only port-Hamiltonian world model

## Breakthrough claim under test

Can a visual world model discover a causally controllable port-Hamiltonian
latent from pixels and actions alone, without receiving simulator positions,
velocities, momenta, energies, events, object masks, or entity-token locations
in any optimization loss?

The learned controlled dynamics are

```text
dx/dt = (J_theta(x) - R_theta(x)) grad H_theta(x) + B_theta(x) u.
```

The first run uses one pH training seed, `131610731`. It is a falsifiable
single-system prototype, not a reproducibility claim.

## Inputs and forbidden supervision

The training path receives only:

- categorical frames produced by the existing pixel renderer;
- the two-dimensional action applied between consecutive frames;
- generic frozen block-5 features from the pixel-only pretrained transformer.

The generic block-5 readout concatenates spatial mean, spatial standard
deviation, and one fixed global token. It does not locate the player or puck.
Training tensors are runtime-checked to contain only visual features, frames,
action indices, and action vectors. Adding `worldStates` or any other tensor to
the fit dictionary raises an error.

The transformer backbone is frozen in this first decisive run. This makes the
claim precise: the pH state, renderer, `H`, `J`, `R`, and `B` are learned
without physical labels on top of a representation that was itself pretrained
from pixels only. A positive result would justify subsequent joint backbone
fine-tuning; a negative result cannot be blamed on state leakage.

## Architecture

An object-agnostic MLP maps generic visual features into an eight-dimensional
latent. A generic spatial-broadcast neural renderer receives only this latent
and Fourier pixel coordinates. It contains no assumption that any latent slot
is an x/y position and has no visual skip connection.

The visual encoder/renderer are first pretrained as an eight-dimensional
pixel autoencoder, then cloned exactly into two paired branches:

1. a neural port-Hamiltonian core learning complete state-dependent
   `H(x)`, `J(x)`, `R(x)`, and `B(x)`;
2. a capacity-matched unconstrained Neural ODE.

Both branches are then fine-tuned end to end from visual reconstruction,
one-step latent consistency, recursive latent prediction, rendered rollout
prediction, action contrast, and label-free latent whitening. The pH branch
also receives only a gradient-scale gauge; it never receives physical energy.

No hybrid jump network is allowed because it could bypass the pH core. Impacts,
walls, goals, and resets must be absorbed by the learned state-dependent
structured vector field or appear as measurable errors.

## Data and holdouts

The registered full run uses:

- 3,072 policy plus 3,072 cardinal-excitation fit trajectories;
- 512 independent policy test trajectories;
- 256 diagonal-action holdouts absent from excitation training;
- 256 rapid action-reversal holdouts;
- eight recursive transitions per trajectory;
- 1,024 additional trajectories used only after training to fit and test
  diagnostic linear maps from the learned latent to physical coordinates.

Physical states attached to test and audit suites are collected only after all
optimization has finished.

## Post-training audits

After freezing the complete branches, simulator states may be used to ask:

1. Are canonical positions and momenta linearly readable from the discovered
   eight-dimensional latent?
2. Does pushing `B(x)` through that post-hoc affine map recover the true player
   force incidence?
3. Do `+/-x` and `+/-y` latent action counterfactuals match simulator
   counterfactuals from the same physical states?
4. Does the pH power identity hold and does zero-input integration decrease
   learned energy?
5. Does the model use actions, as measured by shuffled and zero-action
   degradation?
6. Can gradient-based planning through the learned latent and pixel renderer
   find an eight-step action sequence that moves the real simulator player to
   a requested pixel target better than coasting and random actions?

The last audit optimizes continuous actions only through the learned model,
quantizes them to the environment's nine actions, and executes them in the real
simulator. Simulator states are used to score the result, never to plan it.

## Preregistered single-seed decision

A provisional breakthrough requires every gate below:

- current-frame player IoU at least `0.70` and puck IoU at least `0.50`;
- post-hoc position `q` R² at least `0.80` and momentum `p` R² at least `0.50`;
- physically aligned `B(x)` cosine at least `0.80`;
- one-step counterfactual player-momentum cosine at least `0.80` and rendered
  action sign agreement at least `0.80`;
- shuffled actions worsen horizon-8 weighted pixel cross-entropy by at least
  `5%`;
- learned closed-loop actions improve real-simulator target error by at least
  `20%` versus coasting and beat coasting on at least `65%` of starts;
- pH horizon-4 player-centroid error no worse than `1.15x` the matched Neural
  ODE;
- continuous power defect at most `1e-5` and zero-input learned energy
  increase fraction at most `1e-3`.

Failure of any gate rejects the complete breakthrough claim for this run while
leaving individual discoveries reportable. Sample-level test intervals cannot
replace future training-seed replication.

## Initial runtime estimate

Before the first GPU launch:

- implementation and validation: 3–5 hours;
- A100 pilot: 5–15 minutes;
- full single-seed training: 30–90 minutes;
- post-training audits and interpretation: 1–2 hours;
- first complete conclusion: 5–8 hours.

The full-run estimate will be revised from measured pilot throughput before
submission.
