# Experiment B — causal port-Hamiltonian bottleneck

## Question

Does forcing the block-5 entity representation through a canonical hybrid
port-Hamiltonian state improve causal rollout, passivity, stability, and
intervention consistency relative to an equal-capacity unconstrained core?

Experiment A showed that block 5 is the best entity-wise compromise between
position readability, momentum readability, and structured multi-frame
dynamics. Experiment B changes the causal graph instead of fitting another
post-hoc probe.

## Hard bottleneck

The recovery checkpoint is frozen. Player and puck tokens are extracted after
block 5 using only their rendered categorical pixels. A learned linear encoder
maps the concatenated tokens to

```text
[q_player, q_puck, p_player, p_puck, score_phase, reset_phase].
```

The first eight coordinates are canonical; the last two are the finite-state
mode needed to distinguish normal flow, goal pause, and kickoff. During an
evaluated rollout, only the first block-5 feature is encoded. Every future
state is produced recursively by the bottleneck transition. Later ground-truth
features are used for supervision but never consumed by the rollout.

This is a hard state-dynamics bottleneck. It does not yet replace the pixel
renderer. Pixel reinjection is deliberately deferred until the canonical
dynamics itself passes the causal comparison.

## Smooth core and ports

For each entity, the structured branch uses

```text
q_next = q + positive_gain * p
p_next = decay * p + u_external,  0 <= decay <= 1.
```

This is the exact sampled flow of a free quadratic port-Hamiltonian system with
isotropic linear damping. The explicit external port acts on canonical
momentum. Directional interventions test each of its four coordinates.

Impacts, walls, goal entry, pause, and kickoff are not forced into the smooth
flow. A hybrid jump port predicts both the event mode and an instantaneous
state jump. It is trained to vanish on event-free transitions. This experiment
therefore distinguishes:

- smooth passive flow;
- contact/environment ports;
- discrete reset modes;
- an explicit external control port.

The autonomous checkpoint contains no action channel, so the external port is
tested causally by interventions rather than fitted to agent actions. An
action-conditioned successor experiment would be required to identify a
learned input matrix from policy data.

## Equal-capacity control

The control branch has the same encoder, hybrid-port architecture, losses, and
number of trainable parameters. Its smooth core has the same four scalar
coefficients and the same algebraic map, but gain and decay are unconstrained.
Both branches start from the same physical coefficients and see exactly the
same minibatches.

This isolates the effect of the pH domain constraints. It does not give the
structured branch a smaller transition or a favorable initialization.

## Data and training

- 8,192 deterministic trajectories per full run;
- eight transitions per trajectory;
- complete-trajectory 75/25 fit/test split;
- 20% goal-centered trajectories;
- collision, wall, goal entry, pause, and kickoff labels kept separate;
- six thousand paired optimization steps;
- five independent seeds planned;
- frozen step-12,000 recovery checkpoint;
- one Mesohelios A100, with jobs serialized by the site GPU quota.

The encoder is initialized by a fit-split ridge solution. Both branches are
then optimized end to end through one-step and eight-step state rollouts. The
loss contains normalized state reconstruction, teacher-forced dynamics,
recursive rollout, event classification, and free-port leakage penalties.

## Preregistered metrics and decision rule

Primary metrics are reported on held-out complete trajectories:

1. canonical `q` and `p` readout R²;
2. delta NRMSE at horizons 1, 2, 4, and 8;
3. the paired pH-versus-control gap at horizon 8;
4. free-flow energy growth and passivity violations;
5. 64-step unforced stability;
6. event-balanced accuracy and port magnitude by regime;
7. external-port direction, gain, and cross-talk;
8. a shuffled-initial-state negative control.

The pH branch passes the first causal gate if it:

- remains within 0.02 delta NRMSE of the equal-capacity control at horizon 8;
- has zero unforced core passivity violations and no 64-step energy growth;
- retains `q R² > 0.90` and `p R² > 0.80`;
- produces unit-direction external-port responses with negligible cross-talk;
- loses the rollout signal when the initial states are shuffled.

A prediction advantage is not required for success: the purpose of the pH
constraint is to obtain physical guarantees without materially sacrificing
accuracy. Pixel reinjection is justified only if this state-level gate passes.

## Runtime protocol

A short A100 calibration job uses the final code path with fewer trajectories
and steps. The full jobs are submitted only after that job passes. Total time is
estimated from separately measured collection, optimization, and evaluation
rates; queued time is reported apart from compute time.
