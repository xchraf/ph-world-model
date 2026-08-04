# Experiment B — causal pH bottleneck results

Independent runs: 5. Values are mean ± sample standard deviation across seeds.

## Runtime

| Phase | Seconds per seed |
| --- | ---: |
| Block-5 collection | 99.1 ± 0.4 |
| Paired training | 193.4 ± 7.0 |
| Evaluation | 0.12 ± 0.00 |
| Internal total | 292.6 ± 7.0 |

## State and causal rollout

| Branch | q R² | p R² | h=1 delta NRMSE | h=2 | h=4 | h=8 | Shuffled h=8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pH | 0.952 ± 0.001 | 0.837 ± 0.002 | 1.342 ± 0.082 | 0.933 ± 0.031 | 0.676 ± 0.009 | 0.648 ± 0.006 | 1.017 ± 0.010 |
| Sign-free | 0.952 ± 0.001 | 0.833 ± 0.003 | 1.340 ± 0.080 | 0.934 ± 0.030 | 0.680 ± 0.009 | 0.651 ± 0.006 | 1.009 ± 0.009 |

Paired h=8 pH minus control: -0.0025 ± 0.0035.

## Learned free core

| Branch | Mass, player / puck | Drag, player / puck | Passivity violations | 64-step energy growth | Final / initial energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| pH | 1.722 ± 0.031 / 0.965 ± 0.029 | 0.195 ± 0.015 / 0.151 ± 0.010 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.343 ± 0.021 |
| Sign-free | 1.652 ± 0.049 / 0.932 ± 0.033 | 1.206 ± 0.281 / 0.775 ± 0.243 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.009 ± 0.009 |

## Hybrid events

| Regime | Samples per seed | pH delta NRMSE | Control delta NRMSE | pH port norm | Control port norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| free | 13777.2 ± 24.1 | 1.019 ± 0.005 | 1.008 ± 0.005 | 0.053 ± 0.003 | 0.042 ± 0.003 |
| disc_impact | 446.0 ± 17.4 | 0.849 ± 0.020 | 0.873 ± 0.011 | 0.188 ± 0.034 | 0.164 ± 0.028 |
| wall | 574.0 ± 27.7 | 0.909 ± 0.004 | 0.903 ± 0.007 | 0.148 ± 0.009 | 0.142 ± 0.004 |
| goal_entry | 60.0 ± 7.1 | 1.068 ± 0.033 | 1.051 ± 0.034 | 0.139 ± 0.013 | 0.122 ± 0.013 |
| goal_pause | 1089.8 ± 14.2 | 1.010 ± 0.008 | 1.011 ± 0.008 | 0.137 ± 0.008 | 0.132 ± 0.008 |
| kickoff | 437.0 ± 3.7 | 0.812 ± 0.009 | 0.820 ± 0.013 | 0.389 ± 0.020 | 0.367 ± 0.026 |

## Event and external-port controls

| Branch | Event balanced accuracy | Port direction cosine | Port gain | Cross-talk RMSE |
| --- | ---: | ---: | ---: | ---: |
| pH | 0.875 ± 0.025 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.0000000 ± 0.0000000 |
| Sign-free | 0.882 ± 0.024 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.0000000 ± 0.0000000 |
