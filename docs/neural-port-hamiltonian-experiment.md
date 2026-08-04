# Experiment C — state-dependent neural port-Hamiltonian dynamics

## Question

This experiment asks whether the causal mechanical bottleneck can learn a
reusable nonlinear port-Hamiltonian vector field rather than simulator-specific
constant coefficients.  All four functions are learned from data:

```text
dx/dt = (J_theta(x) - R_theta(x)) grad H_theta(x) + B_theta(x) u.
```

No canonical sparsity pattern, player/puck incidence graph, mass, damping
coefficient, or constant matrix is supplied to these networks.  The current
Blocket League state has dimension eight and its control has dimension two,
but the core implementation accepts arbitrary state and input dimensions.

## Exact structure

Four independent smooth MLPs represent the four unknown functions.

- `H_theta(x)` is a scalar network and its gradient is obtained by automatic
  differentiation.
- The `J` network emits the strict upper triangle of a matrix; reflecting it
  with the opposite sign makes `J_theta(x) = -J_theta(x)^T` exactly.
- The `R` network emits a lower-triangular factor whose diagonal is positive;
  `R_theta(x) = L_theta(x)L_theta(x)^T` is therefore positive semidefinite.
- The `B` network emits the complete state-by-input matrix.  It is not told
  which object is controlled.

Consequently, the continuous power identity

```text
dH/dt = -grad(H)^T R grad(H) + u^T B^T grad(H)
```

holds algebraically at every state.  A midpoint integrator maps this vector
field between observed frames.  Numerical energy drift after discretization is
measured separately from the exact continuous identity.

Skew symmetry alone does not imply that a state-dependent `J(x)` is a Poisson
tensor.  The Jacobi tensor is therefore evaluated explicitly.  It is an audit
in this first run, not a hidden assumption or training label.

## Shared hybrid visual architecture

The frozen recovery world model supplies block-5 entity features.  A
ridge-initialized linear encoder reads the ten-dimensional causal state

```text
[q_player, q_puck, p_player, p_puck, score_phase, reset_phase].
```

The neural pH core evolves only the eight continuous mechanical coordinates.
The same learned hybrid jump port as Experiment B handles impacts, walls,
goals, pauses, and kickoffs.  The same already-trained implicit renderer is
frozen for every branch and seed; it receives only the predicted causal state,
with no visual bypass.

## Capacity-matched control

The control is an unconstrained Neural ODE with the same state, action,
integration scheme, losses, minibatches, encoder, hybrid port, and renderer.
Its hidden width is selected automatically so that its number of smooth-core
parameters differs from the four-network pH core by less than one percent.
For the preregistered width 64, the counts are 24,209 and 24,244 parameters.

## Excitation and holdouts

Each full repetition uses:

- 3,072 policy trajectories and 3,072 persistently excited cardinal-control
  trajectories for fitting;
- 1,024 independent policy trajectories for in-distribution evaluation;
- 512 diagonal-control trajectories, whose action directions were excluded
  from the excitation training set;
- 512 rapid reversal trajectories to test response to high-frequency control
  changes.

Every trajectory contains eight recursive transitions.  Splits are made by
complete trajectory, so no rollout crosses a split boundary.

## Losses and audits

Both branches use state readout, one-step dynamics, eight-step rollout, event
classification, and free-regime jump penalties.  A small pH-only normalization
penalty fixes the practical energy scale by keeping the RMS of the Hamiltonian
gradient in normalized coordinates near one; it does not prescribe the shape
of the energy.

Reported audits include:

- state delta NRMSE and position/momentum R² at horizons 1, 2, 4, and 8;
- player and puck pixel IoU relative to the shared oracle-renderer ceiling;
- shuffled initial states, shuffled action sequences, and zero actions;
- continuous power-balance defect and discrete zero-input energy drift;
- state variation of every learned function;
- affine agreement of `H(x)` and its gradient with true kinetic energy;
- agreement of learned `J`, `R`, and `B` with the simulator matrices, used only
  after training as ground-truth diagnostics;
- skew defect, minimum resistance eigenvalue, and Jacobi defect;
- parameter displacement for each of `H`, `J`, `R`, and `B`;
- across-seed predictive and functional reproducibility.

The decomposition is not judged from a single seed.  Multiple independent
initializations will reveal whether the four learned functions converge to the
same functions on a fixed reference-state set, rather than merely producing a
similar combined vector field.

## Runtime protocol

Two A100 pilots exercise the complete pipeline and measure the final batch
size before full repetitions are submitted.  Queue time is kept separate from
compute time.  The launch script is
`scripts/mesohelios/neural-port-hamiltonian.sbatch`.
