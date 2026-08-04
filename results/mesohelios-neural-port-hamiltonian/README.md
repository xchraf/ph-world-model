# Experiment C — learned state-dependent port-Hamiltonian functions

## Conclusion

The generic architecture successfully learns all four state-dependent
networks `H(x)`, `J(x)`, `R(x)`, and `B(x)` end to end, and it enforces the
continuous port-Hamiltonian power structure exactly by construction.  However,
the five-seed experiment rejects the stronger claim that the autonomous
factorization becomes physically unique from trajectory prediction alone.

The controlled incidence `B(x)` is the clear positive result.  It is recovered
without being told which object is actuated: its cosine with the simulator's
true player-momentum incidence is `0.9940 ± 0.0016`, and its cosine across the
ten seed pairs is `0.9889 ± 0.0018`.  Its spurious position-row fraction is
only `0.0303 ± 0.0015`, and its direct puck-momentum fraction is
`0.0695 ± 0.0076`.

The autonomous components are not similarly identified.  Learned `H(x)` has
only `0.0227 ± 0.0182` affine R² with true kinetic energy and `0.118 ± 0.231`
affine R² across seed pairs.  Learned `J(x)` has `-0.0468 ± 0.0863` cosine with
the canonical interconnection and `-0.054 ± 0.250` cosine across seeds.
`R(x)` is partially aligned with physical drag (`0.597 ± 0.057`) but remains
only moderately reproducible (`0.559 ± 0.074`).

This is not caused by an unreadable state or failed optimization.  The pH
branch retains `q` R² `0.9468 ± 0.0011` and `p` R² `0.8155 ± 0.0033`, every one
of the four networks moves substantially from initialization, and training is
stable in every seed.  Rather, data constrain the autonomous vector field

```text
f_free(x) = (J(x) - R(x)) grad H(x)
```

much more directly than its three factors.  Expanding the product does not
remove all equivalent decompositions in eight dimensions.  In contrast,
persistently varying `u` exposes `B(x)` directly through the controlled part of
the vector field, which explains the sharp identification difference.

## Architecture

The dimension-independent smooth core is

```text
dx/dt = (J_theta(x) - R_theta(x)) grad H_theta(x) + B_theta(x) u.
```

Four independent smooth MLPs learn the four functions.

- `H_theta(x)` is scalar and differentiated by autograd.
- The `J` network emits a strict matrix triangle, expanded so
  `J(x) = -J(x)^T` exactly.
- The `R` network emits a lower-triangular factor and uses
  `R(x) = L(x)L(x)^T`, hence `R(x)` is positive semidefinite.
- The `B` network emits the complete state-by-input matrix; no incidence
  topology is supplied.

The continuous power balance is therefore an algebraic identity.  Across the
trained float32 models its maximum absolute numerical residual is
`2.48e-6 ± 0.40e-6`; the exact-skew defect is zero.  All 64-step, zero-input
audits have monotonically decreasing learned energy.  The state-dependent
`J(x)` is not forced to be Poisson: its measured Jacobi RMS is
`0.0460 ± 0.0108`, so the experiment does not claim that stronger property.

The generic core is wrapped in the same Blocket-specific causal architecture
as Experiment B: frozen block-5 features, a ten-dimensional canonical readout,
an eight-dimensional smooth state, a separate hybrid jump port for impacts and
resets, and a shared frozen state-only pixel renderer.  The comparison Neural
ODE has 24,244 smooth-core parameters versus 24,209 for the pH core, a relative
gap of 0.145%.

## Protocol

Each of five independent seeds uses:

- 3,072 policy and 3,072 cardinal-excitation fit trajectories;
- 1,024 independent policy test trajectories;
- 512 diagonal-action holdouts excluded from excitation training;
- 512 rapid action-reversal holdouts;
- eight recursive transitions per trajectory;
- 4,000 paired optimization steps at batch size 128;
- a frozen shared renderer trained in Experiment B2/B3.

The complete run takes `467.8 ± 7.4` seconds per seed on one A100, including
`111.5 ± 0.5` seconds of feature collection and `354.4 ± 7.7` seconds of paired
dynamics training.

## Predictive results

| Suite | Neural pH H8 delta NRMSE | Neural ODE | Paired pH minus control |
| --- | ---: | ---: | ---: |
| Policy | 0.704 ± 0.036 | 0.576 ± 0.013 | +0.1279 ± 0.0473 |
| Diagonal OOD | 0.587 ± 0.067 | 0.491 ± 0.021 | +0.0961 ± 0.0525 |
| Reversal OOD | 4.070 ± 0.769 | 2.268 ± 0.073 | +1.8017 ± 0.7734 |

The structured branch is worse in all five paired policy comparisons and is
especially brittle to rapid reversals.  Shuffled and zeroed action controls
nevertheless worsen its predictions, confirming that it uses the learned
port.  The shared renderer ceiling remains high (policy player IoU `0.921`,
puck IoU `0.803`), so lower rollout IoU reflects dynamics error rather than a
failed decoder.

## Functional reproducibility

All five checkpoints were evaluated on the same 256 physical states and
controls.  Ten pairwise comparisons give:

| Learned quantity | Across-seed agreement |
| --- | ---: |
| `H`, after affine alignment, R² | 0.118 ± 0.231 |
| aligned `grad H` cosine | 0.253 ± 0.361 |
| `J` cosine | -0.054 ± 0.250 |
| `R` cosine | 0.559 ± 0.074 |
| `B` cosine | 0.989 ± 0.002 |
| complete vector-field cosine | 0.896 ± 0.020 |

Thus even the combined vector field does not meet the preregistered 0.95
agreement gate, while the separately learned autonomous functions disagree
much more strongly.  The failure is robust rather than a single bad seed.

## Interpretation and next design

The next useful experiment should retain full learned state dependence and
joint training, but make the autonomous decomposition better posed without
inserting Blocket-specific masses or incidence:

1. train the smooth core explicitly on free-transition derivatives or
   one-step maps evaluated at the target latent state, preventing the hybrid
   jump port from sharing smooth dynamics;
2. parameterize `H` as a normalized, lower-bounded energy model rather than an
   arbitrary scalar MLP plus a gradient-scale penalty;
3. constrain state-dependent `J` to a valid Poisson family (for example a
   learned Darboux-coordinate pullback) or explicitly regularize its Jacobi
   tensor;
4. fix the remaining coordinate/scale convention for `J` and `H`, then repeat
   the same cross-seed functional test;
5. keep the successful full `B(x)` network and persistent excitation protocol.

The key lesson is that port-Hamiltonian constraints guarantee passivity and a
power decomposition, but they do not by themselves guarantee recovery of the
simulator's particular energy, interconnection, and dissipation functions.

## Artifacts

- [`aggregate/aggregate.json`](aggregate/aggregate.json) — full five-seed
  aggregate and functional agreement metrics;
- [`aggregate/metrics.md`](aggregate/metrics.md) — generated compact tables;
- `artifacts/neural-ph-seed-*/checkpoint.pt` — all learned branches;
- `artifacts/neural-ph-seed-*/dynamics.jsonl` — optimization traces;
- `artifacts/neural-ph-seed-*/summary.json` — per-seed evaluations;
- [`launch.json`](launch.json) — Mesohelios jobs and provenance.
