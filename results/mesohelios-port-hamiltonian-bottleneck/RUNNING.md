# Experiment B — Mesohelios run manifest

Status: completed successfully.

## Frozen implementation

- Training commit: `c749bbe`
- Base checkpoint: `/Work/Users/aelmessaoudi/world-model/outputs/main-recovery-12000/checkpoint.pt`
- Base checkpoint step: 12,000
- Hardware: one NVIDIA A100-PCIE-40GB
- Full protocol: 8,192 trajectories, eight transitions, 6,000 paired steps
- Branch capacity: 5,598 parameters each

## Validation jobs

- `229288`: first end-to-end pilot; stopped after feature collection because a
  sequence-shaped regime label was passed to the 2D Experiment A helper. No
  training result from this job is used.
- `229291`: corrected end-to-end pilot; completed in 23 seconds with exit code
  `0:0`. Internal time was 3.85 seconds for collection, 3.78 seconds for 100
  paired steps, and 0.12 seconds for evaluation.

The missing sequence-axis regression test was added before the corrected pilot
and full submissions. The project suite then contained 40 passing tests.

## Full jobs

| Seed | Slurm job | Output directory |
| ---: | ---: | --- |
| 91410731 | 229296 | `port-hamiltonian-bottleneck-seed-91410731` |
| 101410733 | 229295 | `port-hamiltonian-bottleneck-seed-101410733` |
| 111410737 | 229293 | `port-hamiltonian-bottleneck-seed-111410737` |
| 121410739 | 229292 | `port-hamiltonian-bottleneck-seed-121410739` |
| 131410741 | 229294 | `port-hamiltonian-bottleneck-seed-131410741` |

Mesohelios serializes these jobs because the user quota permits one A100 at a
time. Job IDs therefore do not encode scientific priority; all five use the
same preregistered configuration.

## Runtime estimate after launch

The first full job measured:

- feature collection: 98.99 seconds for 8,192 trajectories;
- stable optimization throughput at step 700: 29.59 paired steps/second;
- projected 6,000-step optimization: 202.76 seconds.

The internal compute estimate per seed is therefore

```text
98.99 + 202.76 + evaluation/save ≈ 303–307 seconds.
```

The corrected pilot showed approximately 15 seconds of Slurm/container startup
overhead. The resulting wall-clock estimate is about 315 seconds per seed and
26–27 minutes for all five serialized seeds. A conservative interval is 25–32
minutes, excluding any unexpected scheduler outage but including ordinary
between-job startup overhead.

This estimate is based on the final batch size of 256 and the actual full data
volume, not on the smaller pilot throughput.

## Actual completion time

All five jobs completed with exit code `0:0`. Their individual Slurm times were
5:13, 5:10, 4:57, 5:14, and 5:02. Cumulative allocated time was 25:36. From the
first job start at 16:46:45 to the last job end at 17:13:14, serialized wall
time was 26:29, inside the preregistered 26–27 minute estimate.
