# Experiment B — causal port-Hamiltonian bottleneck

## Conclusion

Experiment B passes its preregistered causal gate.

Forcing block-5 entity features through a canonical hybrid pH state preserves
the predictive accuracy of an equal-capacity sign-free control while producing
a substantially more physical free-flow model. The pH bottleneck is not more
accurate in a statistically meaningful sense; its benefit is physical fidelity
and guaranteed admissibility at essentially no prediction cost.

The strongest supported conclusion is:

> A canonical port-Hamiltonian bottleneck is a viable causal state for this
> world model. It preserves the useful block-5 information, supports stable
> directional ports, and prevents the transition from solving rollout error by
> collapsing momentum through excessive damping.

This is a state-dynamics result. Pixel reinjection and learned action ports have
not yet been tested.

## Causal change relative to Experiment A

Experiment A decoded independently observed endpoints and fitted dynamics
afterward. Experiment B changes the causal graph:

```text
first block-5 entity feature
          ↓
linear canonical encoder
          ↓
[q_player, q_puck, p_player, p_puck, score mode, reset mode]
          ↓
smooth four-parameter core + hybrid jump port
          ↓
recursive future states at h=1…8
```

Only the first feature is consumed during evaluation. Every later state is
generated recursively through the 10D bottleneck. Future backbone features are
supervision targets, not inputs.

Two 5,598-parameter branches see the same minibatches and start from the same
physical map:

- `portHamiltonian`: positive position gains and momentum decays constrained to
  `[0, 1]`;
- `signFreeControl`: the same four coefficients and algebraic map without those
  constraints.

The backbone through block 5 remains frozen in both branches. Impacts, walls,
goals, pauses, and kickoffs are handled by a separately learned hybrid jump
port rather than being forced into smooth free flow.

## Reproducibility

- Training code commit: `c749bbe`
- Launch manifest commit: `e3be823`
- Frozen checkpoint: `main-recovery-12000/checkpoint.pt`, step 12,000
- Five seeds: `91410731`, `101410733`, `111410737`, `121410739`, `131410741`
- 8,192 trajectories × 8 transitions per seed
- Complete-trajectory split: 6,144 fit / 2,048 test
- 6,000 paired optimization steps per seed
- One Mesohelios NVIDIA A100-PCIE-40GB
- Slurm jobs: `229292`, `229293`, `229294`, `229295`, `229296`
- All jobs completed with exit code `0:0`
- Forty project tests passed on the training commit

Each seed directory contains its 59 KB bottleneck checkpoint, complete training
log, and full evaluation summary. `aggregate.json` and `metrics.md` are
deterministic summaries of these five raw results.

## Runtime

| Measurement | Result |
| --- | ---: |
| Block-5 feature collection per seed | 99.1 ± 0.4 s |
| Paired training per seed | 193.4 ± 7.0 s |
| Internal total per seed | 292.6 ± 7.0 s |
| Sum of five Slurm allocations | 25 min 36 s |
| First start to final completion | 26 min 29 s |

After the first full job reached stable throughput, the announced estimate was
26–27 minutes for all five serialized seeds, with a conservative interval of
25–32 minutes. The observed wall time was 26:29.

## Representation level

| Branch | Position q R² | Momentum p R² | Hybrid-mode R² |
| --- | ---: | ---: | ---: |
| pH | 0.952 ± 0.001 | 0.837 ± 0.002 | 0.840 ± 0.008 |
| Sign-free | 0.952 ± 0.001 | 0.833 ± 0.003 | 0.841 ± 0.007 |

The hard bottleneck retains the Experiment A block-5 state information. The pH
constraint does not require a loss of ordinary state readability and slightly
improves momentum readability, although the difference is small.

## Causal rollout level

Delta NRMSE is normalized so persistence equals 1.0.

| Branch | h=1 | h=2 | h=4 | h=8 | Shuffled initial state, h=8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pH | 1.342 ± 0.082 | 0.933 ± 0.031 | 0.676 ± 0.009 | 0.648 ± 0.006 | 1.017 ± 0.010 |
| Sign-free | 1.340 ± 0.080 | 0.934 ± 0.030 | 0.680 ± 0.009 | 0.651 ± 0.006 | 1.009 ± 0.009 |

The paired h=8 pH-minus-control difference is
`-0.0025 ± 0.0035`. Four seeds favor pH and one favors the control, but the
effect is too small to claim a meaningful prediction advantage. The correct
result is non-inferiority: pH remains well within the preregistered `0.02`
tolerance.

The one-frame score is worse than persistence because the physical increment
is smaller than the block-5 decoding error. The recursive signal becomes clear
at two to eight frames. Shuffling the initial states destroys the h=8 benefit,
showing that the rollout is causally tied to its initial latent state rather
than only to dataset averages.

At h=8, the pH rollout has `q R² = 0.874 ± 0.004` but only
`p R² = 0.294 ± 0.015`. The bottleneck preserves positions much better than
recursive momenta. This is the main state-dynamics limitation remaining after
Experiment B.

## Physical-parameter level

| Branch | Effective mass, player / puck | Effective drag, player / puck |
| --- | ---: | ---: |
| True simulator | 1.800 / 1.000 | 0.120 / 0.120 |
| pH | 1.722 ± 0.031 / 0.965 ± 0.029 | 0.195 ± 0.015 / 0.151 ± 0.010 |
| Sign-free | 1.652 ± 0.049 / 0.932 ± 0.033 | 1.206 ± 0.281 / 0.775 ± 0.243 |

Both branches obtain nearly the same rollout error. The control achieves it by
destroying momentum much too quickly: roughly ten times the true player drag
and six times the true puck drag. The pH branch remains mildly overdamped but
is far closer to the simulator.

The control's learned coefficients happened to remain inside the passive
domain in all five runs, so this difference cannot be attributed only to the
hard feasibility boundary. The pH softplus/sigmoid parameterization also
changes local optimization sensitivity and acts as an implicit physical
regularizer. A tangent-sensitivity-matched control would be required to
separate parameter-domain and optimization-geometry effects completely.

## Energy and stability level

| Branch | Core passivity violations | 64-step energy-growth fraction | Energy after 64 steps |
| --- | ---: | ---: | ---: |
| pH | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.343 ± 0.021 |
| Sign-free | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.009 ± 0.009 |

Both fitted controls remained passive on these runs, but their qualitative
stability differs. With the simulator drag, the expected free kinetic-energy
fraction after 64 frames is approximately `0.464`. The pH branch retains
`0.343`; the sign-free branch almost freezes the system at `0.009`. Thus the pH
structure prevents a numerically stable but physically useless attractor.

## Hybrid-port level

| Regime | pH delta NRMSE | Control delta NRMSE | pH mean port norm |
| --- | ---: | ---: | ---: |
| Free | 1.019 ± 0.005 | 1.008 ± 0.005 | 0.053 ± 0.003 |
| Disc impact | 0.849 ± 0.020 | 0.873 ± 0.011 | 0.188 ± 0.034 |
| Wall | 0.909 ± 0.004 | 0.903 ± 0.007 | 0.148 ± 0.009 |
| Goal entry | 1.068 ± 0.033 | 1.051 ± 0.034 | 0.139 ± 0.013 |
| Goal pause | 1.010 ± 0.008 | 1.011 ± 0.008 | 0.137 ± 0.008 |
| Kickoff | 0.812 ± 0.009 | 0.820 ± 0.013 | 0.389 ± 0.020 |

Event-balanced accuracy is `0.875 ± 0.025` for pH and `0.882 ± 0.024` for the
control. The learned port magnitude is ordered sensibly: smallest in free flow,
larger for contacts, and largest for kickoff. Impacts and kickoffs benefit from
the hybrid port. Goal entry and pause remain at or worse than persistence.

The free-flow port norm is not zero despite its leakage penalty. Therefore the
full hybrid transition is not strictly passive on every free sample; only its
smooth pH core carries the hard guarantee. A harder free/event gate is needed
before claiming passivity of the complete hybrid model.

## External-control level

For all four canonical momentum coordinates and all five seeds:

- direction cosine: `1.000 ± 0.000`;
- port gain: `1.000 ± 0.000`;
- cross-talk RMSE: below `1e-8`.

This establishes exact directional control of player and puck momenta through
the explicit port. The input matrix is fixed by the architecture, so this is a
structural controllability guarantee, not an input matrix identified from
agent-action data. The autonomous checkpoint has no action channel.

## Preregistered decision

| Criterion | Required | Observed | Decision |
| --- | ---: | ---: | --- |
| h=8 pH gap versus control | ≤ +0.020 | -0.0025 ± 0.0035 | Pass |
| Position readout | q R² > 0.90 | 0.952 ± 0.001 | Pass |
| Momentum readout | p R² > 0.80 | 0.837 ± 0.002 | Pass |
| Smooth-core passivity violations | 0 | 0.000 | Pass |
| 64-step energy-growth fraction | 0 | 0.000 | Pass |
| External-port direction and gain | 1 | 1.000 | Pass |
| Shuffled-state negative control | destroys h=8 signal | 0.648 → 1.017 | Pass |

Experiment B therefore justifies the next architectural stage: reinjecting the
canonical state into a pixel decoder. That next stage should also add a
sensitivity-matched unconstrained control, a harder free-port gate, and a true
action-conditioned input port.
