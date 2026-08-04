# End-to-end port-Hamiltonian pixel world model — single seed

## Run identity

- Code commit: `d7b04d9`
- Slurm job: `229335`
- Training seed: `131610731`
- GPU: NVIDIA A100
- Remote artifacts: `/Work/Users/aelmessaoudi/world-model/outputs/end-to-end-ph-full-131610731`
- Wall clock: 26 min 18 s
- Measured pipeline time: 1565.32 s
- Result: `provisional_end_to_end_breakthrough_not_supported_single_seed`
- Gates passed: 10 / 13

## What was trained

The pretrained direct pixel transformer was used only as initialization. Its pixel
embeddings, positional representations and all transformer blocks remained trainable.
The complete trainable path was:

`pixel histories -> visual transformer -> 8-D latent -> dynamics -> patch-transformer decoder -> future pixels`

The structured dynamics learned state-dependent `H(z)`, `J(z)`, `R(z)` and `B(z)` by
construction. The control was a parameter-matched Neural ODE with an independent copy
of the identical initialized visual encoder and decoder. Complete capacity was
5,704,130 versus 5,704,165 parameters; the core gap was 0.145%.

Only raw pixel histories and actions entered optimization. Simulator states were
collected after training and used solely for the audits below.

## Main result

The strict preregistered breakthrough claim is not fully supported because three gates
failed: player reconstruction IoU (0.531 < 0.70), puck reconstruction IoU
(0.347 < 0.50), and linear momentum readout (p R2 = 0.312 < 0.50).

The remaining ten gates passed. In particular, the structured model:

- discovered position without labels (q R2 = 0.865);
- aligned its learned port with the physical actuation direction (cosine = 0.8004);
- predicted the player-momentum counterfactual direction (cosine = 0.9558);
- rendered the counterfactual direction correctly (cosine = 0.9781; sign = 100%);
- causally used actions (shuffling degraded H8 cross-entropy by 52.86%);
- reduced real simulator target error by 42.69% versus coast and beat coast in 75% of trials;
- preserved exact continuous power balance (maximum defect = 2.83e-7);
- had a PSD resistance matrix up to numerical precision;
- satisfied the discrete zero-input monotonicity gate;
- retained predictive parity with the parameter-matched Neural ODE.

The pH model also improved over the Neural ODE on several non-preregistered comparative
metrics: policy H8 player-centroid error fell by 41.3%, policy H8 cross-entropy by 6.6%,
diagonal-OOD H8 cross-entropy by 11.2%, and real closed-loop error by 11.1%. Its final
500-step mean rollout-pixel training loss was 0.1061 versus 0.1138 for the ODE.

## Important limitations

- The learned Hamiltonian is not identified with physical kinetic energy
  (affine R2 = 0.091). It is presently a learned storage function.
- `J(z)` is exactly skew-symmetric, but its Jacobi RMS is 0.0128. Therefore this run
  does not establish that `J` is a Poisson tensor.
- Momentum is only weakly linearly accessible, especially under OOD distribution
  shifts. This is the main representational failure.
- The learned counterfactual action effect has the correct direction but overestimates
  its magnitude (0.848 predicted pixels versus 0.316 true pixels).
- This is one seed. The comparative gains are provisional until replicated.

See `metrics.json` for the compact exact record and the remote `summary.json` for every
policy and OOD metric.
