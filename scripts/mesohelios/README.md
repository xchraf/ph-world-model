# Mesohelios A100 workflow

This directory runs the direct-pixel Blocket League model on the Mesohelios
Slurm cluster without Modal. Code stays in `$HOME`; containers, checkpoints,
logs, and generated artifacts stay under `$WORK/world-model`.

The main checkpoint lineage matches `docs/retraining-provenance.md`:

1. `main-base-30000.json`: the historical 30,000-step defaults.
2. `main-recovery-12000.json`: the documented 12,000-step recovery recipe.

Run `benchmark-a100.sbatch` first. After validation, submit the base job and
submit the recovery job with `--dependency=afterok:<base-job-id>`.
Submit `core-analyses.sbatch` with an `afterok` dependency on the recovery job.
It runs the frozen-checkpoint position, direction-ring, Jacobian-lens, causal
write, collision-anticipation, and random-weight-control analyses.

`port-hamiltonian-audit.sbatch` runs the separate Experiment A audit against the
same frozen recovery checkpoint. It extracts the embedding and every block at
consecutive physical times, fits canonical `q,p` readouts, and compares a
four-parameter dissipative port-Hamiltonian free-flow map with Hamiltonian,
affine, and MLP controls. Its output is written to
`$WORK/world-model/outputs/port-hamiltonian-audit/audit.json`.

`port-hamiltonian-bottleneck.sbatch` runs Experiment B. It freezes the recovery
checkpoint through block 5 and trains paired equal-capacity causal bottlenecks:
a constrained hybrid pH branch and a sign-free control. Runtime and scientific
criteria are preregistered in `docs/port-hamiltonian-bottleneck-experiment.md`.

`neural-port-hamiltonian.sbatch` runs Experiment C. Its dimension-independent
smooth core learns the complete state-dependent functions `H(x)`, `J(x)`,
`R(x)`, and `B(x)` while enforcing skew interconnection and positive
semidefinite resistance exactly. A capacity-matched Neural ODE, cardinal
excitation, diagonal and reversal holdouts, and a shared frozen pixel renderer
are evaluated in the same job. See `docs/neural-port-hamiltonian-experiment.md`.

`jacobian-port-bridge.sbatch` runs the single-seed first pass of Experiment D1.
It tests whether independently derived causal Jacobian-lens writes at block 5
map through the trained readout to the learned action port `B(x)`, both
infinitesimally and after a real render-and-reencode intervention. See
`docs/jacobian-port-bridge-experiment.md`.
