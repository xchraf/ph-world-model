# Frozen port-Hamiltonian audit

## Scientific question

At which transformer stage does a canonical two-disc state become linearly
readable, and is the decoded state's one-frame evolution well explained by a
low-capacity port-Hamiltonian free-flow map?

This experiment is diagnostic. It does not retrain or modify the world-model
checkpoint. It also does not assume that transformer depth is physical time.
Every transition compares the representation at the same architecture stage
across two consecutive simulator times.

## Canonical state

The supervised diagnostic coordinate system is

```text
z = [q_player, q_puck, p_player, p_puck]
```

where each entry is two-dimensional and `p = mass * velocity`. Simulator state
is used to anchor and evaluate the diagnostic decoder. Entity-token locations
are selected from categorical rendered pixels, not from privileged simulator
coordinates.

Because canonical coordinates are supplied to the diagnostic probe, this audit
tests whether the frozen representation supports a port-Hamiltonian
description. It does not claim that canonical coordinates are uniquely
identifiable from pixels without this alignment.

## Data and leakage controls

- Each deterministic simulator trajectory contributes eight overlapping
  physical transitions.
- Complete trajectories, rather than individual windows, are assigned to the
  fit or test split.
- The full audit is repeated over five independent simulator seeds. Reported
  conclusions must be stable across seeds rather than selected from one run.
- Ridge state decoders are fit jointly on the `t` and `t+1` endpoints from fit
  trajectories only. A second, still-linear decoder adds normalized endpoint
  differences to test whether high state R² can be made temporally coherent at
  the much smaller one-frame physical scale.
- Port-Hamiltonian, Hamiltonian, affine, and MLP transition models are fit only
  on event-free transitions with no active reset timer.
- Disc impacts, walls, goal entry, goal pause, and kickoff are reported as
  separate out-of-assumption regimes.

## Readouts and stages

The audit includes the patch embedding and every transformer block. It tests:

1. the concatenated player and puck entity tokens;
2. the spatial mean of the latest-frame tokens;
3. one fixed bottom-right token.

Both the state-only and state-plus-delta linear decoders report separate
position and momentum R² and RMSE. Keeping both prevents a dynamics-aligned
probe from hiding a large loss of ordinary state readability.

## Dynamics models

The strict structured free-flow map has two coefficients per entity:

```text
q_next = q + position_gain * p
p_next = momentum_decay * p
```

The coefficients are tied across the x and y axes. With positive position gain
and momentum decay in `[0, 1]`, the map is the sampled flow of a free mechanical
port-Hamiltonian system with isotropic linear damping. The audit derives an
effective mass and drag from these discrete coefficients.

Controls are:

- persistence: zero parameters;
- Hamiltonian free flow: two position gains and no momentum decay;
- port-Hamiltonian free flow: two gains and two decays;
- unconstrained affine 8D transition: 72 parameters;
- two-layer tanh MLP delta predictor.

The low-capacity structured map is deliberately narrower than its controls. A
small structured gap is therefore stronger evidence than a fit obtained with a
fully flexible learned Hamiltonian, interconnection, and dissipation matrix.

## Primary metrics

- linear readout R² for `q` and `p` at every stage;
- next-state RMSE and delta-normalized RMSE;
- gap between the four-parameter pH map and the affine/MLP controls;
- the same free-flow comparison over 1, 2, 4, and 8 frames, so probe noise at
  one frame is distinguishable from a genuinely absent dynamical structure;
- discrete work-free energy-balance residual;
- passivity-violation rate;
- conformal symplectic defect of the fitted affine transition;
- the same transition errors reported separately for impacts, walls, goal
  entry, pause, and kickoff.

Two targets are kept distinct:

1. `decodedEndpointDynamics`: whether the decoded representation evolves
   according to the fitted structure;
2. `worldStateDynamics`: whether a transition initiated from the decoded state
   reaches the true next simulator state.

## Interpretation boundary

A favorable result means that a low-capacity pH explanation is sufficient for
the linearly decoded free dynamics at a given layer. It is not proof that the
world model internally computes an explicit Hamiltonian. A later causal or
hard-bottleneck experiment is required for that stronger claim.

The smooth pH map is expected to fail at instantaneous impacts and resets. This
is diagnostic evidence for a later hybrid pH transition, not evidence against
the pH description of collision-free motion.
