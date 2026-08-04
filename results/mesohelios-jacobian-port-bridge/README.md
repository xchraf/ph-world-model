# Experiment D1 — Jacobian lens to learned pH action port

## Conclusion

On neural-pH training seed `111610731`, the independently derived block-5
Jacobian lens and the learned action port `B(x)` identify the same controlled
player-momentum directions in the local and moderate-write regimes. This is the
first experiment in this project that connects the transformer's native causal
activation geometry directly to the port-Hamiltonian bottleneck.

The strongest result is infinitesimal. The global Jacobian-lens writes are
computed only from gradients of the frozen transformer's next rendered player
centroid. After applying the separately trained readout differential `dE`, their
cosine with `dt B(x)e_axis` is `0.964` in the player-momentum coordinates and
`0.958` over the complete canonical state. The player-momentum sample-level 95%
interval is `[0.961, 0.966]`, while 16 pairs of random latent directions
orthogonal to the lens give `-0.244`. All mapped lens effects have the expected
controlled-coordinate sign.

This agreement is not caused by the lens simply writing rendered position.
Player momentum accounts for `99.5%` of the complete mapped-effect norm. The
puck-momentum norm is only `4.3%` of the player-momentum norm, and its direction
does not align systematically with the port. Thus the pixel-defined handle
maps specifically onto the object and state sector that `B(x)` independently
learned to actuate.

The finite causal test succeeds only in an operating window. A real write is
made after transformer block 5, the edited hard next frame is appended to the
history, block-5 features are extracted again, and `E` reads the resulting
state. At activation strength 4, the centered state effect has player-momentum
cosine `0.676` with the centered pH control step, versus `-0.027` for a
magnitude-matched random write. Axis specificity is `0.222`, and `88.7%` of
effects have the expected sign. All registered finite criteria pass at this
moderate strength.

The preregistered primary strength 8 does **not** pass the complete finite gate.
Its player-momentum cosine remains positive at `0.580`, but axis specificity
falls to `0.105`, below the registered `0.15` threshold. The rendered target
effect already plateaus between strengths 4 and 8 (`0.223` versus `0.224`
pixels), while the orthogonal-to-target effect grows from `0.058` to `0.092`
pixels. This identifies renderer/readout saturation and cross-axis leakage,
not disappearance of the controlled subspace.

The defensible single-seed conclusion is therefore:

> A causal Jacobian-lens direction in the transformer's block-5 latent maps to
> the independently learned port-Hamiltonian action port, infinitesimally and
> under moderate finite interventions. The bridge is not yet robust to large
> off-manifold writes, and training-seed reproducibility remains untested.

## Main measurements

### Differential bridge

| Readout view | Global lens cosine | Local Jacobian cosine | Random control |
| --- | ---: | ---: | ---: |
| Full canonical state | 0.958 | 0.616 | -0.226 |
| Standardized canonical state | 0.898 | 0.511 | -0.156 |
| Player momentum | **0.964** | 0.637 | -0.244 |
| Player position | 0.513 | 0.181 | 0.019 |
| Puck momentum | 0.198 | 0.046 | 0.025 |

The state-specific downstream gradients are noisier than their global average.
This suggests that `B(x)` corresponds to a stable controllable direction that
the global Jacobian average isolates from state-dependent renderer sensitivity,
rather than to every local pixel-output gradient.

### Finite render-and-reencode bridge

| Write strength | Player-momentum cosine | Axis specificity | Correct sign | Lens/random control |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.443 | -0.107 | 75.6% | not primary-controlled |
| 4 | **0.676** | **0.222** | **88.7%** | -0.027 random |
| 8 | 0.580 | 0.105 | 83.1% | -0.030 random |

Strength 1 is dominated by hard-frame quantization; strength 4 is the clearest
causal regime; strength 8 shows saturation and cross-axis leakage. The strength
4 random control was run after observing the preregistered sweep, on the same
training seed, evaluation seed, examples and random-direction construction. It
is a dose-matched diagnostic, not an independent replication. The original
strength-8 decision is retained in the primary artifact.

## Protocol and scope

- pH checkpoint: training seed `111610731` from Experiment C;
- evaluation seed: `121610731`;
- global-lens fit: 512 independent policy contexts;
- evaluation: 256 unseen policy and 256 unseen cardinal contexts;
- transformer and pH branch are frozen throughout;
- block-5 player and puck tokens feed the same affine `E` used in Experiment C;
- global lens directions have cosine `0.129` with one another;
- no evaluated state placed player and puck in the same selected patch;
- full run: 23.6 seconds of measured computation, 36 seconds wall time on A100.

The reported confidence intervals describe variation across evaluation
contexts conditional on one trained model. They do not measure uncertainty
over training seeds. Multi-seed confirmation is deliberately deferred.

## Artifacts

- [`artifacts/summary.json`](artifacts/summary.json) — preregistered strengths
  1/4/8 run; its automatic primary outcome is `infinitesimal_only_single_seed`
  because strength 8 misses the axis-specificity gate;
- [`artifacts/strength4-control-summary.json`](artifacts/strength4-control-summary.json)
  — same-seed, same-example diagnostic with the random finite control matched
  at strength 4;
- [`launch.json`](launch.json) — cluster jobs, checkpoint paths, commit and
  artifact hashes;
- [`../../docs/jacobian-port-bridge-experiment.md`](../../docs/jacobian-port-bridge-experiment.md)
  — protocol and decision thresholds written before the full run.
