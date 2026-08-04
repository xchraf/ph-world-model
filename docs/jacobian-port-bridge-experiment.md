# Experiment D1 — Jacobian-lens to port-Hamiltonian bridge

## Question

Does the frozen transformer's causal block-5 Jacobian lens correspond to the
action port learned independently by the state-dependent neural
port-Hamiltonian branch?

The registered infinitesimal comparison is

```text
dE(h) delta_h_axis   versus   dt B(E(h)) e_axis.
```

The lens direction is computed only from the downstream gradient of the frozen
pixel transformer's next rendered player centroid. It is not obtained by
inverting `E` or by inspecting `B`, so agreement is not true by construction.

## Single-seed first pass

The first D1 run deliberately uses only neural-pH seed `111610731`. It fits the
global lens on 512 independent policy contexts and evaluates on 256 unseen
policy plus 256 unseen cardinal-excitation contexts. This can support or reject
a within-seed bridge, but it cannot establish reproducibility across training
seeds. Sample-level confidence intervals will be labelled accordingly.

## Two levels of evidence

1. **Differential bridge.** The exact Jacobian of the trained affine readout
   `E` maps global and per-state player-token Jacobian directions into the
   canonical eight-dimensional smooth state. These effects are compared with
   `dt B(x)e_x` and `dt B(x)e_y`.
2. **Finite causal bridge.** At strengths 1, 4 and 8, the global lens is written
   with both signs into the actual block-5 player token. Each edited next frame
   is rendered, discretized, appended to the history, re-encoded through the
   frozen transformer and read by `E`. The centered state difference is
   compared with the centered trained pH step under `+u` and `-u`.

Full physical-state, standardized-state, player-position, player-momentum and
puck-momentum alignments are all reported. The player-momentum view is primary
because the learned `B(x)` from Experiment C places the controlled incidence
there.

## Controls and decision gates

- correct action axes are compared with exchanged axes;
- 16 pairs of latent directions orthogonal to the fitted lens are evaluated;
- finite effects use paired plus/minus writes of identical strength;
- player and puck sharing one visual patch is handled exactly in the readout
  differential.

The differential bridge passes only if player-momentum cosine is at least
`0.50`, correct-minus-wrong-axis cosine is at least `0.25`, and the correct
target-coordinate sign rate is at least `0.75`.

The finite bridge at the registered primary strength 8 passes only if
player-momentum cosine is at least `0.30`, axis specificity is at least `0.15`,
target-coordinate sign rate is at least `0.65`, and cosine exceeds the random
finite-write control by at least `0.20`.

Both levels must pass to claim a provisional single-seed bridge. Passing only
the differential level means the relationship does not survive the transformer's
render-and-reencode path. Multi-seed confirmation is explicitly deferred.
