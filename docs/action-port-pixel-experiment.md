# Experiment B2/B3 — learned action port and pixel reinjection

## Questions

This successor to the causal port-Hamiltonian bottleneck separates two claims:

1. can real policy actions be identified as an explicit force port acting on
   the learned player coordinates, without direct puck cross-talk?
2. does the resulting canonical rollout retain its information when it is
   decoded back to pixels without a visual bypass?

The recovery checkpoint and its block-5 representation remain frozen. The
experiment changes neither the source checkpoint nor its training data.

## Active canonical transition

The state is the same ten-dimensional bottleneck used in Experiment B:

```text
[q_player, q_puck, p_player, p_puck, score_phase, reset_phase].
```

The structured smooth core applies the observed two-dimensional policy action
only to the player:

```text
q_player_next = q_player + positive_gain * p_player + positive_Bq * action
p_player_next = decay * p_player + positive_Bp * action
q_puck_next   = q_puck   + positive_gain * p_puck
p_puck_next   = decay * p_puck
```

with `0 <= decay <= 1`. Its initial coefficients reproduce the simulator's
four-substep damped integrator exactly away from collisions, walls, resets, and
the player speed cap. The action vectors are normalized exactly as they are in
the simulator. Hybrid contacts and resets remain a separate jump port.

## Tangent-matched control

The control has the same architecture, number of parameters, starting map,
minibatches, and losses. Its coefficient parameterization has the exact same
Jacobian as the structured parameterization at initialization, but its gains,
decays, and action gains can subsequently become negative or leave the passive
domain. This removes the local optimization-geometry confound seen in the
first Experiment B comparison. Unit tests verify both value and Jacobian
matching before training.

## Pixel reinjection without bypass

A shared implicit renderer receives only the ten-dimensional canonical state
and fixed pixel coordinates. It never receives an input frame, a transformer
token, or a skip connection. It is trained on fit-split ground-truth states and
frames, then frozen and used identically for both dynamics branches.

Its training objective combines palette-weighted cross-entropy with player and
puck soft-Dice losses. The first complete-path calibration is allowed to set
the renderer step count using oracle-state IoU only, before any full paired run;
branch-to-branch results are not used for this choice.

This design tests whether the state bottleneck contains enough information to
reconstruct and roll out the scene. It deliberately isolates dynamics from
renderer error: the renderer is not trained end to end through either branch,
so a difference between branches cannot be caused by giving one of them a
better decoder.

## Data, controls, and metrics

- 8,192 complete action-conditioned trajectories per full run;
- eight recursive transitions and a 75/25 trajectory split;
- one ridge-initialized block-5 encoder per paired comparison;
- 6,000 paired dynamics steps and 5,000 shared-renderer steps, selected from
  the preregistered oracle-only calibration;
- state delta NRMSE and `q`/`p` R² at horizons 1, 2, 4, and 8;
- pixel accuracy plus player and puck IoU at each horizon;
- oracle-state renderer quality, reported as a ceiling;
- shuffled initial states, shuffled action sequences, and zero-action controls;
- action-direction cosine and direct puck cross-talk;
- smooth-core 64-step zero-action stability and learned physical coefficients;
- jump-port magnitude by free, impact, wall, goal, pause, and kickoff regime.

The action-port claim requires directional momentum response with negligible
direct puck cross-talk and a measurable degradation under shuffled or removed
actions. The structural claim requires accuracy comparable to the
tangent-matched control while retaining the passive domain and zero-action
stability. Pixel results are interpreted relative to the oracle-renderer
ceiling, rather than raw background-dominated accuracy alone.

## Runtime protocol

A short job exercises the complete collection, paired training, rendering,
evaluation, and serialization path on one A100. Its measured stage rates are
used to revise the initial 1 h 15–1 h 45 end-to-end estimate before the full
independent repetitions are submitted. Queue time is reported separately from
compute time.

## Completed run

The five-seed aggregate, interpretation, checkpoints, logs, calibration record,
and measured runtime are stored in
[`results/mesohelios-action-port-pixel`](../results/mesohelios-action-port-pixel/README.md).
