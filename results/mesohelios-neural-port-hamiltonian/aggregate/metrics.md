# Experiment C — state-dependent neural port-Hamiltonian dynamics

Independent runs: 5. Values are mean ± sample standard deviation.

## Runtime

| Phase | Seconds per seed |
| --- | ---: |
| Feature collection | 111.5 ± 0.5 |
| Paired dynamics | 354.4 ± 7.7 |
| Evaluation | 1.8 ± 0.0 |
| Total | 467.8 ± 7.4 |

## Predictive comparison

| Suite | pH H8 delta NRMSE | Neural ODE | Paired gap |
| --- | ---: | ---: | ---: |
| policy | 0.704 ± 0.036 | 0.576 ± 0.013 | 0.1279 ± 0.0473 |
| diagonalOod | 0.587 ± 0.067 | 0.491 ± 0.021 | 0.0961 ± 0.0525 |
| reversalOod | 4.070 ± 0.769 | 2.268 ± 0.073 | 1.8017 ± 0.7734 |

## Learned structure on policy states

| Audit | Value |
| --- | ---: |
| Max power defect | 0.00000248 ± 0.00000040 |
| Kinetic-energy affine R² | 0.023 ± 0.018 |
| Canonical J cosine | -0.047 ± 0.086 |
| Jacobi RMS | 0.045964 ± 0.010842 |
| Physical R cosine | 0.597 ± 0.057 |
| Physical B cosine | 0.994 ± 0.002 |

## Across-seed functional agreement

| Function | Agreement |
| --- | ---: |
| H affine R² | 0.118 ± 0.231 |
| aligned grad H cosine | 0.253 ± 0.361 |
| J cosine | -0.054 ± 0.250 |
| R cosine | 0.559 ± 0.074 |
| B cosine | 0.989 ± 0.002 |
| vector-field cosine | 0.896 ± 0.020 |

## Decision gates

- capacityMatched: PASS
- predictiveParity: FAIL
- readoutGate: PASS
- continuousPowerIdentity: PASS
- skewSymmetry: PASS
- positiveResistance: PASS
- actionCausalSignal: PASS
- functionalVectorFieldAgreement: FAIL
- separateFunctionAgreement: FAIL
