# Experiment A — frozen port-Hamiltonian audit

## Conclusion

The frozen world model contains a robust, linearly readable approximation of
canonical position and momentum. Its late-layer, collision-free decoded motion
is compatible with a very low-capacity port-Hamiltonian (pH) map over several
frames. It does **not**, however, constitute an autonomous passive pH state at
the one-frame scale.

The appropriate conclusion is therefore:

> The current latent supports a coarse pH-compatible geometry, but there is no
> evidence that the frozen transformer internally evolves an explicit
> port-Hamiltonian state.

This experiment is a diagnostic baseline. No pH constraint was added to the
checkpoint and no model parameter was retrained.

## Reproducibility record

- Audit commit: `3de78e3`
- Frozen checkpoint: `outputs/main-recovery-12000/checkpoint.pt`, step 12,000
- Hardware: one Mesohelios NVIDIA A100 per run
- Runs: five independent simulator seeds
- Data per run: 1,024 trajectories × 8 transitions = 8,192 transitions
- Split: 768 complete trajectories for fitting, 256 for testing
- Slurm jobs: `229282`, `229283`, `229284`, `229285`, `229286`
- Runtime: 57 seconds per run; all exit codes `0:0`
- Seeds: `81240731`, `91240733`, `101240737`, `111240739`, `121240741`

The five raw JSON files contain the complete per-seed measurements.
`aggregate.json` and `metrics.md` are deterministic summaries produced by
`scripts/mesohelios/summarize_port_hamiltonian_audits.py`.

## What was tested

At the embedding and after each of the six transformer blocks, a linear probe
decodes the canonical state

```text
z = [q_player, q_puck, p_player, p_puck],  p = mass × velocity.
```

The same architectural stage is compared at physical times `t` and `t+h`;
transformer depth is never treated as physical time. Three representation
readouts are audited:

1. the player and puck entity tokens;
2. the spatial mean of the latest-frame tokens;
3. one fixed bottom-right token.

On event-free transitions, the four-parameter structured map is

```text
q_next = q + position_gain × p
p_next = momentum_decay × p,  with 0 ≤ momentum_decay ≤ 1.
```

It is compared with persistence, a Hamiltonian map without dissipation, a
72-parameter affine map, and, where applicable, a small MLP. The main error is
delta NRMSE: persistence is exactly 1.0 and lower is better.

## Calibration on privileged simulator state

The audit first fits the same pH map directly to true simulator state. This
recovers the known mechanics almost exactly:

| Quantity | Player | Puck |
| --- | ---: | ---: |
| True mass | 1.8000 | 1.0000 |
| Recovered mass | 1.8014 ± 0.0000 | 1.0008 ± 0.0000 |
| True drag | 0.1200 | 0.1200 |
| Recovered drag | 0.1200 ± 0.0000 | 0.1200 ± 0.0000 |

The normalized energy-balance residual is
`1.78e-7 ± 6.07e-8`. At eight frames, forward pH delta NRMSE is
`7.84e-7 ± 3.16e-8`; it becomes `0.9994 ± 0.0039` after endpoint shuffling.
In reverse time, constrained pH is `1.000003 ± 0.000000`, while the
unconstrained affine inverse reaches `0.00268 ± 0.00042`. The protocol can
therefore detect both the exact free mechanics and their dissipative time
direction.

## Architectural anatomy

### Position and momentum readability

| Readout | Stage | Position q R² | Momentum p R² |
| --- | --- | ---: | ---: |
| Entity pair | Embedding | 0.992 ± 0.003 | 0.088 ± 0.033 |
| Entity pair | Block 3 | 0.986 ± 0.001 | 0.802 ± 0.008 |
| Entity pair | Block 5 | 0.960 ± 0.002 | 0.864 ± 0.007 |
| Entity pair | Block 6 | 0.935 ± 0.001 | 0.869 ± 0.008 |
| Spatial mean | Embedding | 0.133 ± 0.009 | 0.081 ± 0.005 |
| Spatial mean | Block 1 | 0.911 ± 0.003 | 0.217 ± 0.034 |
| Spatial mean | Block 6 | 0.969 ± 0.002 | 0.850 ± 0.007 |
| Fixed token | Embedding | -0.004 ± 0.002 | -0.006 ± 0.003 |
| Fixed token | Block 1 | 0.826 ± 0.003 | 0.181 ± 0.025 |
| Fixed token | Block 6 | 0.947 ± 0.004 | 0.823 ± 0.007 |

Position is local and nearly explicit in the patch embedding. Momentum is not:
it emerges progressively through the transformer. After one block, position
is already broadcast to the spatial mean and even to a fixed remote token;
momentum is distributed mainly in blocks 3–6. This is stable across all seeds.

The entity-token position score peaks before the final block while momentum
continues improving. For a future entity-structured bottleneck, block 5 is the
best compromise in this checkpoint; block 6 is strongest for a global pooled
readout.

### Geometry of the decoded transition

The affine transition's conformal symplectic defect decreases strongly with
depth:

| Readout | Embedding | Block 3 | Block 5 | Block 6 |
| --- | ---: | ---: | ---: | ---: |
| Entity pair | 0.152 ± 0.012 | 0.066 ± 0.005 | 0.041 ± 0.004 | 0.049 ± 0.002 |
| Spatial mean | 0.792 ± 0.014 | 0.112 ± 0.006 | 0.044 ± 0.002 | 0.037 ± 0.004 |
| Fixed token | degenerate | 0.114 ± 0.006 | 0.051 ± 0.003 | 0.052 ± 0.008 |

Late effective masses are also close to the simulator values. At block 6 they
are `1.833/1.052` for entity tokens, `1.871/0.985` for the spatial mean, and
`1.826/1.081` for the fixed token. In contrast, the corresponding effective
drags are `0.815/0.749`, `0.742/0.597`, and `0.840/0.719`, far above the true
`0.120/0.120`. Probe noise and non-Markovian latent effects are being absorbed
as excessive damping. The apparent masses are encouraging; the drags prevent
a literal physical interpretation.

## Structured dynamics result

### Collision-free motion

| Readout, state-only decoder | h=1 pH / affine | h=8 pH / affine |
| --- | ---: | ---: |
| Entity pair, block 5 | 0.982 ± 0.001 / 0.976 ± 0.002 | 0.821 ± 0.008 / 0.831 ± 0.014 |
| Entity pair, block 6 | 0.983 ± 0.001 / 0.977 ± 0.001 | 0.825 ± 0.010 / 0.827 ± 0.010 |
| Spatial mean, block 6 | 0.983 ± 0.001 / 0.978 ± 0.002 | 0.815 ± 0.007 / 0.819 ± 0.012 |
| Fixed token, block 6 | 0.985 ± 0.001 / 0.980 ± 0.002 | 0.850 ± 0.008 / 0.856 ± 0.010 |

At one frame, every model is close to persistence: the useful physical
increment is much smaller than the probe error. At eight frames, the pH map
removes roughly 15–19% of persistence error. Despite having only four
parameters, it matches the 72-parameter affine map within about one percentage
point. The seed-wise pH–affine difference changes sign in several cases, so the
data support equivalence at this resolution, not a statistically meaningful
pH advantage.

The decoder trained additionally to align one-frame differences does not solve
the problem. At block 6 and h=8, its pH errors are `0.887`, `0.836`, and `0.884`
for the three readouts, all worse than the ordinary state-only decoder
(`0.825`, `0.815`, `0.850`). It also reduces position readability, especially
for entity tokens (`q R²: 0.935 → 0.811`). High state readability and a clean
one-frame dynamical coordinate are therefore distinct properties.

### Temporal negative controls

At block 6 and h=8:

| Readout | Forward pH | Shuffled endpoints | Reverse pH | Reverse affine |
| --- | ---: | ---: | ---: | ---: |
| Entity pair | 0.825 ± 0.010 | 0.979 ± 0.006 | 0.996 ± 0.002 | 0.842 ± 0.014 |
| Spatial mean | 0.815 ± 0.007 | 0.981 ± 0.007 | 0.990 ± 0.003 | 0.824 ± 0.031 |
| Fixed token | 0.850 ± 0.008 | 0.968 ± 0.009 | 0.985 ± 0.012 | 0.852 ± 0.034 |

Breaking endpoint pairing removes almost all structured benefit. Reverse pH is
also essentially persistence, while an unconstrained affine map can recover a
substantial part of the reverse transition. The late-layer forward structure
is therefore not explained merely by state autocorrelation or a map close to
the identity; it contains a reproducible dissipative time direction.

### Why this is still not a pH latent state

For event-free one-frame decoded transitions at block 6:

| Readout | Energy-balance residual | Passivity-violation rate |
| --- | ---: | ---: |
| Entity pair | 0.286 ± 0.011 | 0.501 ± 0.007 |
| Spatial mean | 0.299 ± 0.005 | 0.508 ± 0.004 |
| Fixed token | 0.341 ± 0.015 | 0.510 ± 0.011 |

Roughly half of the observed decoded steps increase kinetic energy beyond the
fitted passive tolerance. The pH predictor itself is passive by construction,
but the decoded endpoint sequence is not. A low symplectic defect and a good
multi-frame fit are consequently insufficient to claim that the latent state
obeys a pH law.

Impacts and resets also require a hybrid model. At block 6, pH has no clear
one-frame advantage over affine dynamics; for walls the entity-pair scores are
`0.976` versus `0.946`, and during goal pause pH is worse than persistence at
`1.025`. Goal-entry estimates use only `8.6 ± 2.2` test samples per run and must
not be overinterpreted.

## Decision for the next experiment

Experiment A validates the premise for imposing structure, but not the
stronger hypothesis that the existing state already follows it. A causal
Experiment B should therefore:

1. extract an entity-wise canonical bottleneck around block 5;
2. evolve it with an explicitly passive pH core with ports for player action;
3. separate smooth free flow from collision, wall, goal, and reset modes;
4. compare against an unconstrained transition with matched parameter count;
5. measure prediction, controllability, energy/work balance, passivity,
   long-horizon stability, and intervention consistency.

The exhaustive layer, decoder, horizon, event, and control tables are in
`metrics.md`.
