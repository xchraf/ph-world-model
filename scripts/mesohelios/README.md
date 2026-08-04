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
