# Experiment B2/B3 — learned action port and pixel reinjection

## Verdict

An action-conditioned port-Hamiltonian bottleneck is feasible here. Across five
paired seeds, it preserves the same held-out state and pixel accuracy as an
equal-capacity tangent-matched unconstrained control while guaranteeing the
zero-action passive domain. Real action sequences carry a large causal signal,
and the canonical state can be rendered back to pixels without any visual
bypass.

The experiment does **not** show a predictive advantage from the pH constraint.
The paired horizon-8 delta-NRMSE difference (pH minus control) is
`-0.000036 ± 0.000348`; a two-sided t interval with four degrees of freedom is
approximately `[-0.000468, +0.000396]`. The control also remained inside the
passive domain on all five runs. In this regime, the constraint supplies a hard
guarantee at essentially zero accuracy cost rather than a measurable accuracy
gain.

## Primary results

| Metric | Port-Hamiltonian | Tangent-matched control |
| --- | ---: | ---: |
| State `q` R² | 0.953 ± 0.001 | 0.953 ± 0.001 |
| State `p` R² | 0.819 ± 0.005 | 0.819 ± 0.005 |
| H8 delta NRMSE | 0.624 ± 0.012 | 0.624 ± 0.012 |
| H8 `q` R² | 0.720 ± 0.012 | 0.720 ± 0.012 |
| H8 `p` R² | 0.363 ± 0.030 | 0.363 ± 0.031 |
| Event-balanced accuracy | 0.747 ± 0.020 | 0.747 ± 0.020 |
| 64-step zero-action energy growth | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Runs inside the pH coefficient domain | 5/5 | 5/5 |

All preregistered decision gates passed. Full machine-readable results are in
[`aggregate.json`](aggregate.json), and the compact generated tables are in
[`metrics.md`](metrics.md).

## What changed at each architectural level

### 1. Frozen visual representation and canonical encoder

The block-5 entity features still support a strong canonical readout under
active control: `q R² = 0.953` and `p R² = 0.819`. The pH and control encoders
are numerically indistinguishable at aggregate scale. The structural constraint
therefore did not improve or damage the frozen transformer's state geometry.

The momentum gate is passed, but only narrowly. By horizon 8, recursive
momentum R² falls to `0.363`, while position R² falls to `0.720`. Long-rollout
information loss is therefore primarily a state-dynamics problem, not a
failure to read the initial canonical state.

### 2. Smooth mechanical core

The structured core learned physically plausible values:

| Parameter | Simulator | Learned pH | Tangent control |
| --- | ---: | ---: | ---: |
| Player mass | 1.800 | 1.761 ± 0.030 | 1.757 ± 0.031 |
| Puck mass | 1.000 | 1.016 ± 0.026 | 1.016 ± 0.026 |
| Player drag | 1.550 | 1.692 ± 0.068 | 1.689 ± 0.064 |
| Puck drag | 0.120 | 0.125 ± 0.006 | 0.125 ± 0.006 |
| Action-to-position gain | 0.00504 | 0.00338 ± 0.00027 | 0.00322 ± 0.00034 |
| Action-to-momentum gain | 0.2873 | 0.2617 ± 0.0048 | 0.2618 ± 0.0047 |

The learned action gains are attenuated relative to the exact uncapped smooth
integrator. This is compatible with the simulator's speed cap, encoder
attenuation, and compensation through the learned state-dependent jump branch.
It also means that individual coefficients are not fully identifiable from
rollout loss alone even when the final predictions are nearly identical.

The tangent control starts with the exact same map and parameter Jacobian as
the pH branch. Its remaining in the passive domain is therefore evidence that
the data and objective themselves favor a passive local solution, not an
artifact of a more favorable initial optimization slope for pH.

### 3. Real action port

The action sequence is causally important:

| H8 condition, pH branch | Delta NRMSE | Change from nominal |
| --- | ---: | ---: |
| Correct initial state and actions | 0.624 ± 0.012 | — |
| Shuffled action sequence | 0.938 ± 0.012 | +0.314 / +50.3% |
| Zero action sequence | 0.763 ± 0.004 | +0.139 / +22.2% |
| Shuffled initial state | 0.794 ± 0.003 | +0.170 / +27.3% |

This establishes that the learned rollout uses the real action channel rather
than treating active dynamics as autonomous noise.

The unit action direction cosine of 1.0 and zero direct puck cross-talk are
architectural guarantees, not discoveries: the experiment fixes the port
incidence to the player canonical axes and learns two scalar gains. It
identifies the strength and causal necessity of this port, but it does not yet
identify a free input matrix `G(z)` or discover which latent entity should be
actuated.

### 4. Hybrid contact and reset port

The state-dependent jump branch learned clearly different magnitudes by
regime:

| Regime | Samples per test split | Delta NRMSE | Mean jump norm |
| --- | ---: | ---: | ---: |
| Smooth/free (including thrust) | 13,345 ± 60 | 0.883 ± 0.005 | 0.048 ± 0.008 |
| Disc impact | 592 ± 26 | 0.994 ± 0.005 | 0.130 ± 0.068 |
| Wall | 1,427 ± 41 | 0.896 ± 0.004 | 0.375 ± 0.015 |
| Goal entry | 118 ± 11 | 1.053 ± 0.032 | 0.276 ± 0.009 |
| Goal pause | 771 ± 63 | 0.992 ± 0.008 | 0.195 ± 0.011 |
| Kickoff | 130 ± 11 | 0.678 ± 0.029 | 0.414 ± 0.058 |

This decomposition is useful but not clean. The smooth/free jump norm is not
zero, so part of the ordinary dynamics still leaks through the residual port.
Goal entry and disc impact remain especially difficult. The action itself
cannot leak through this branch because it receives state but not action, which
strengthens the action-channel negative control, but the smooth-versus-hybrid
separation is still imperfect.

### 5. Pixel reinjection without bypass

The shared state-only renderer has a high oracle ceiling:

| State source | Pixel accuracy | Player IoU | Puck IoU |
| --- | ---: | ---: | ---: |
| True canonical state | 0.916 ± 0.005 | 0.917 ± 0.005 | 0.861 ± 0.036 |
| pH rollout, horizon 1 | 0.899 ± 0.006 | 0.410 ± 0.005 | 0.216 ± 0.007 |
| pH rollout, horizon 4 | 0.894 ± 0.006 | 0.303 ± 0.002 | 0.155 ± 0.006 |
| pH rollout, horizon 8 | 0.884 ± 0.007 | 0.174 ± 0.007 | 0.081 ± 0.004 |

The oracle result shows that the ten-dimensional canonical state contains
enough information to reconstruct both objects and the scene. The sharp IoU
drop under predicted states localizes the failure upstream: small state errors
move a two-to-four-pixel-radius object enough to destroy overlap. At horizon 8,
the rollout retains only about 18.9% of the oracle player IoU and 9.4% of the
oracle puck IoU. Raw pixel accuracy hides this because the static background
dominates it.

The renderer was trained on true fit-split states and then shared and frozen.
Thus this is a clean bottleneck-sufficiency test, not an end-to-end pixel-loss
world model.

## Guarantees versus empirical findings

Guaranteed by construction:

- positive mechanical gains and `0 <= decay <= 1` for the pH branch;
- zero-action non-increasing core energy;
- action incidence on player coordinates only;
- no direct action-to-puck term;
- no image, token, or skip-connection path into the renderer.

Learned or empirically established:

- the block-5-to-canonical encoder;
- masses, damping, and scalar action gains;
- the need for the observed action sequence under held-out rollout;
- state-dependent event modes and jump corrections;
- state-to-pixel rendering;
- parity with a locally optimization-matched unconstrained control.

## Limitations and next discriminating experiment

1. The action incidence is imposed. A stronger test should learn a structured
   `G(q)` or full input matrix under the pH power balance, then test whether it
   independently selects player momentum with negligible puck coupling.
2. All held-out actions come from the same policy family. Novel pulse, reversal,
   and composition sequences are needed to test control extrapolation.
3. The hybrid residual leaks on smooth transitions and performs poorly at rare
   goal entries and impacts. A contact-aware complementarity or impulse layer
   would make the architectural attribution sharper.
4. The active experiment starts from the passively trained frozen transformer.
   End-to-end training through the canonical bottleneck and renderer may reduce
   long-horizon pixel loss, but it must preserve the no-bypass graph.
5. Because the unconstrained control stayed physical on every run, this dataset
   does not test whether the pH constraint helps under low data, distribution
   shift, or adversarial optimization pressure. Those are the conditions where
   the guarantee should become operationally valuable.

## Reproducibility and runtime

- Protocol code: commit `647a090`.
- Aggregator and launch record: commit `cfa7cba`.
- Base checkpoint: passive recovery checkpoint at step 12,000.
- Five seeds: `101510731`, `101520731`, `101530731`, `101540731`, `101550731`.
- Full jobs: `229311`–`229315`, all completed with exit code 0.
- Mean internal time: `391.0 ± 3.7 s` per seed.
- Submission-to-final-completion wall time: `34 min 47 s`.

Every run's checkpoint, dynamics log, renderer log, and raw summary is stored in
[`artifacts`](artifacts). The original runtime calibration and job mapping are
recorded in [`launch.json`](launch.json).
