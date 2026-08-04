# Experiment B2/B3 — action port and pixel reinjection

Independent runs: 5. Values are mean ± sample standard deviation.

## Runtime

| Phase | Seconds per seed |
| --- | ---: |
| Feature collection | 119.6 ± 0.4 |
| Paired dynamics | 248.4 ± 2.8 |
| Shared renderer | 22.3 ± 0.9 |
| Evaluation | 0.8 ± 0.0 |
| Internal total | 391.0 ± 3.7 |

## State and action results

| Metric | Port-Hamiltonian | Tangent-matched control |
| --- | ---: | ---: |
| State q R² | 0.953 ± 0.001 | 0.953 ± 0.001 |
| State p R² | 0.819 ± 0.005 | 0.819 ± 0.005 |
| H8 delta NRMSE | 0.624 ± 0.012 | 0.624 ± 0.012 |
| H8 shuffled actions | 0.938 ± 0.012 | 0.939 ± 0.012 |
| H8 zero actions | 0.763 ± 0.004 | 0.763 ± 0.004 |
| 64-step energy growth | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Action momentum cosine | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Direct puck cross-talk | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |

Paired H8 pH-minus-control delta NRMSE: -0.0000 ± 0.0003.

## Pixel reinjection

| Metric | Oracle-state renderer | pH H8 | Control H8 |
| --- | ---: | ---: | ---: |
| Accuracy | 0.916 ± 0.005 | 0.884 ± 0.007 | 0.884 ± 0.007 |
| Player IoU | 0.917 ± 0.005 | 0.174 ± 0.007 | 0.173 ± 0.006 |
| Puck IoU | 0.861 ± 0.036 | 0.081 ± 0.004 | 0.081 ± 0.004 |

## Decision gates

- accuracyParity: PASS
- readoutGate: PASS
- passiveDomain: PASS
- zeroActionEnergyGrowth: PASS
- actionCausalSignal: PASS
- directionalPort: PASS
- rendererCeilingAdequate: PASS
