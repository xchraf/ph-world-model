# Experiment F — direct Jacobian–Poisson port-Hamiltonian control

## Status and locked question

This document is the preregistration for one decisive single-seed run. It is
frozen before training, calibration, physical realizability testing, or control
evaluation begins. Thresholds may not be relaxed, failed gates may not be
removed, and an ablation may not replace the registered model after any held-out
result is visible.

The question is whether a video transformer that never receives an action can
support a directly learned, physically executable port-Hamiltonian latent:

```text
pixels -> frozen transformer -> direct latent state x
dx/dt = (J(x) - R(x)) grad H(x) + B(x) v
physical command u -> analytically calibrated v = T u
```

There is no dynamics teacher, detached target vector field, student, projection
phase, or post-hoc pH fitting. After the frozen backbone is selected, a
zero-parameter empirical activation tangent is computed once in closed form
from fit-video innovations, before any pH module exists. At every later
context, the activation port itself is extracted from the current frozen
Jacobian; it is never a learned field. The visual state, latent efforts,
Hamiltonian, Poisson structure, dissipation, and state-dependent `B(x)` are
then optimized jointly in one direct pH model on action-erased videos while
that exact port operator remains frozen.

The neural-training and evaluation seed is `151910737`. The producer excitation
uses a separate 128-bit entropy value generated once at submission, stored in a
mode-`0600` `producer-private` file, and derived into a system seed inside the
producer container. Its value is never placed in an argument, environment
variable, learner mount, archive, or Slurm record. The private file is retained
so it can be revealed only after the locked result for exact re-rendering. This
is one neural-training seed, not a multi-seed reproducibility claim.

### Precise novelty boundary relative to 2026 work

The claim is not merely “a port-Hamiltonian world model” or “action-free video
control.” PH-Dreamer trains an action-controlled RSSM, applies a projected pH
phase-space mechanism, and uses proprioceptive energy information. *Identify
Then Project* reports that a staged identify/project strategy can be more
reliable than single-stage structure learning. VERA keeps its video planner
action-free but trains an embodiment-specific Jacobian IDM on action-labelled
self-play. Experiment F instead tests the narrower conjunction: a **jointly
trained direct pH latent**, exact parameter-free activation and cotangent
Jacobian ports, zero
physical-action gradient updates, a frozen video backbone, and only four paired
physical probes per axis after freezing. A positive run supports only that
conjunction on these two systems; it does not establish the first pH world
model or the first action-free video controller.

## Claim boundary and identifiability

The registered claim is deliberately gauge-aware. The experiment may support
all of the following:

- a predictive visual state and autonomous vector field;
- a Poisson tensor by construction in the learned coordinates;
- a passive controlled discrete-time model;
- a state-dependent actuator **distribution** `col(B(x))`;
- a power-conjugate port basis after a small analytic interface calibration;
- executable closed-loop control and transfer to an unseen actuator interface.

It may not claim that the physical functions `H`, `J`, `R`, or `B` have been
identified individually. For an invertible coordinate change `x' = phi(x)`,
with `P = D phi(x)`, the transformed objects

```text
H' = H ∘ phi^-1
J' = P J P^T
R' = P R P^T
B' = P B
```

represent the same dynamics. The port basis also has the unavoidable gauge
`B -> B S`, `v -> S^-1 v` for any invertible `S`, and the learned energy retains
a positive scale gauge. Calibration chooses one executable basis; it does not
make the decomposition unique.

Consequently, raw entry-wise agreement of learned and simulator `H`, `J`, `R`,
or `B` is neither a decision metric nor permitted evidence. Gauge-invariant
objects—rollouts, Poisson and power identities, principal angles between port
distributions, calibrated counterfactual responses, and control—are primary.

## Systems and disjoint suites

The same pipeline is run on both systems.

1. **Damped pendulum:** video observations of a nonlinear oscillator. Hidden
   training excitation consists of symmetric zero-mean torque pulses. The
   deployment interface is a scalar pivot torque.
2. **Blocket League:** video observations of two freely moving and colliding
   discs in one continuous arena. Goal scoring, pauses, and kickoff resets are
   disabled, so every clip and deployment episode follows one uninterrupted
   physical state. Video generation, calibration, realizability, and control
   use the same sealed default drags (`player_drag=1.55`, `puck_drag=0.12`).
   Hidden training excitation is a persistently exciting two-dimensional thrust
   applied to one disc. In a sealed 60% producer-only branch, an isotropically
   oriented near-contact initial condition and four-to-seven-frame approach
   pulse make contact transitions common; the remaining branch uses the generic
   antithetic excitation. Branch identity, geometry, contact, effort, and object
   identity never cross the pixel boundary. The model is not given which visual
   entity is actuated.

For each system, complete trajectories are assigned before rendering to 4,096
fit, 512 validation, and 512 action-erased test trajectories. Splits never share
an initial condition or a rollout. Additional suites are generated separately:

- analytic calibration probes;
- 128 paired held-out physical counterfactuals per physical action axis;
- 64 native-interface control episodes;
- 64 unseen-interface control episodes.

The additional suites are inaccessible until every neural parameter and model
selection decision has been frozen.

The public producer seal stores the exact `HiddenExcitationConfig`: 24 frames,
image size 64, generic hold interval `[1,4]`, coast probability `0.20`, Blocket
contact-rich probability `0.60`, pre-contact gap `[0.025,0.085]`, approach hold
interval `[4,7]`, and approach magnitude `[0.65,0.95]`. It stores both the
canonical serialization and SHA-256, but never the 128-bit producer seed.

## Pixels-only information firewall

### Excited videos with physically erased actions

The rendered training distribution is excited; it is not a zero-input passive
distribution. This distinction is part of the claim. A separate rendering
process applies hidden controls drawn independently of the state from a
registered symmetric, full-rank excitation distribution. It streams only
rendered frames into the sanitized dataset. It never serializes actions,
simulator states, forces, torques, object masks, coordinates, velocities,
momenta, energies, contacts, events, or random-generator states.

The raw producer payload has exactly one key, `frames`. Each sanitized archive
has exactly two serialized keys, `pixels` and `manifest`; the manifest contains
hashes and shape/schema metadata, not per-trajectory identifiers or frame
indices. No action or state sidecar is copied. The learner container mounts
only `fit-pixels.pt` and `validation-pixels.pt` for its system, one disjoint
code-free runtime cache, and a generated read-only learner source bundle. That
bundle is the reviewed AST import closure of the training entry point; it does
not contain `env.py`, `action_free_excitation.py`, `passive_control_systems.py`,
the producer entry point, or any legacy simulator-bearing experiment module.
The complete repository is therefore not physically visible in a learner
container. Neither `heldout`, `producer-private`, producer seals, the simulator,
nor the producer seed is mounted.
Re-rendering is possible only from the separately preserved private seed after
the result is locked.

This protocol supports the phrase **no action labels or action inputs during
pretraining**. It does not support the false claim that no physical excitation
occurred in the videos.

### Forbidden gradient information

The registered phases have two exact gradient schemas:

- backbone pretraining: `{pixels}`;
- direct structured training: `{pixelContexts, frames}`;
- every direct ablation: `{pixelContexts, frames}`;
- the independently initialized unstructured pixel world model:
  `{pixelContexts, frames}`.

Any additional key or mounted forbidden file aborts the run and fails the
firewall gate. In particular, no gradient update may depend on:

- a commanded or inferred physical action label;
- simulator state or a post-hoc physical coordinate;
- object identity, segmentation mask, centroid, entity-token location, event,
  contact, reward, or score label;
- physical energy, force, torque, work, or power;
- a simulator transition paired with its physical control or state metadata.

Latent actions inferred by the direct inverse head from two pixel-derived
states are allowed because they are model variables, not physical action
records. Their basis has no physical name before calibration.

Simulator state may be inspected only after neural training, checkpoint
selection, the calibration rule, and every neural hash are frozen. Its sole
registered use before the outcome is the affine relative-degree/locality audit
in Gate 5; it cannot update a parameter, choose a checkpoint, construct a port,
or change a threshold. Calibration, realizability, target costs, and control
scores use pixels and re-encoded pixel differences. Other state-based analyses
are secondary and are run only after the locked outcome is computed.

Every phase appends a hash-chained runtime firewall trace. It records real
batch tensor keys/shapes/dtypes, optimizer parameter names and object/storage
identities, protected-backbone overlap, same-descriptor (`O_NOFOLLOW`, `fstat`)
file hashes and inodes, Slurm identifiers, selected backbone hashes, explicit
trainer directory listings, and the canonical full `/proc/self/mountinfo`
inventory. It additionally records every path, byte length, and SHA-256 visible
under the learner source mount, plus a recursive inventory of the disjoint
runtime cache. Python source/bytecode/`.pth` files, symlinks, or special files
in that cache abort before training. Gate 1 replays these events; it does not manufacture “observed”
schemas or zero forbidden-read counts from constants. Critical JSON seals and
all checkpoints are written to a same-filesystem temporary file and published
with `os.replace`.

## Frozen-backbone contract

Each system uses one six-block causal pixel transformer pretrained by weighted
next-frame cross-entropy on the sanitized archive. There is no masked objective.
Its exact configuration is image size 64, patch size 4, palette size 9,
eight-frame history, pixel embedding width 8, hidden width 192, six blocks, six
heads, and MLP ratio 4. Backbone creation is completed before the direct model
is initialized. It is not a teacher: it supplies frozen representations and a
frozen autoregressive pixel predictor only.

The direct experiment uses the complete block-5 residual stream. It does not
select a hand-located player patch or an entity token. The following are sealed
in the run manifest:

- source checkpoint path and file SHA-256;
- a canonical SHA-256 over every named parameter and buffer tensor;
- a canonical Git-independent source-tree manifest containing every registered
  source/test/script/dependency-lock path, byte count, file SHA-256, and tree
  SHA-256, plus the sanitized-dataset SHA-256;
- a second canonical manifest over the exact learner AST closure, anchored to
  the full source-tree SHA-256 and verified byte-for-byte in every learner job;
- architecture and configuration serialization.

The transformer is in evaluation mode, has `requires_grad=False`, is absent
from every optimizer, and has no mutable running statistics. Canonical tensor
hashes are recomputed after direct training, after calibration, and after
control. Any mismatch is an automatic negative outcome.

Backbone pretraining uses batch size 16 for 30,000 AdamW steps, learning rate
`3e-4`, weight decay `1e-2`, 1,000 warm-up steps, and EMA decay `0.9995`.

## Direct architecture

### Visual state

Eight-frame histories are passed through the frozen transformer. A generic
attention-pooling adapter reads all block-5 tokens and produces canonical
coordinates `xi`. It has no object-specific query. An invertible six-coupling
flow `Psi` produces the pH coordinate `x = Psi(xi)`.

The latent dimensions are fixed to two for the pendulum and eight for Blocket
League. The port ranks are fixed to one and two, respectively. The whole-stream
readout width is 192. A `LatentPatchTransformerRenderer` reconstructs pixels
from `x` with hidden width 192, depth 3, and six heads; it is not a
spatial-broadcast decoder. There is no visual skip connection and no access to
the transformer input after the bottleneck.

The registered intervention is zero-based transformer block index 4 (the fifth
block in one-based prose). The pH core uses width 128, three hidden layers, six
invertible coupling layers, `dt=0.05`, 32 implicit iterations, and relaxation
`0.8`. The inverse effort head uses width 128 and two hidden layers. The
activation port has zero trainable parameters. The legacy
`write_hidden_size=16` and `write_hidden_layers=2` fields are retained only as
explicit checkpoint-schema compatibility fields and construct no module in
either the structured model or the independent baseline.

### True Poisson parameterization

The state-dependent interconnection tensor is not an arbitrary skew matrix. It
is the push-forward of the constant canonical tensor `J0`:

```text
J(x) = D Psi(xi) J0 D Psi(xi)^T,    xi = Psi^-1(x).
```

Because `Psi` is invertible, this construction is skew and satisfies the Jacobi
identity wherever the chart is regular. The chart is jointly learned from
pixels; it is not a projection learned after another dynamics model.

`H(x)` is a flexible non-convex scalar network with a coercive quadratic base
and bounded residual. `R(x) = L(x)L(x)^T` is positive semidefinite exactly.
`B(x)` is a complete state-by-port matrix with no supplied incidence pattern.
`H`, `R`, and `B` use separate three-layer width-128 MLPs. Their separate
parameterizations are architectural factors, not claims of separate physical
identification.

### Action-free latent excitation head

A two-state inverse head infers a rank-`m` latent innovation `a_t` from
`(x_t, x_{t+1})`. The direct transition loss is

```text
x_(t+1) ~= Phi_dt(x_t, a_t),
dx/dt = (J(x) - R(x)) grad H(x) + B(x) a_t.
```

The generic hidden excitation is antithetic and the complete Blocket corpus is
marginally isotropic; the contact-rich branch intentionally couples its private
approach direction to its private initial geometry. The inferred latent
innovations are constrained to zero batch mean, fixed covariance,
decorrelation, temporal regularity, and low first- and second-moment dependence
on `x_t`. These label-free penalties choose a practical gauge and discourage
the inverse head from storing state; they neither assert conditional
independence of the producer input nor remove the `GL(m)` port gauge.

### Exact ports from activation Jacobians

The activation interface is not learned. Before pH construction, exactly 4,096
response-blind fit transitions are selected from the sanitized pixels. For
each transition, the frozen transformer compares the observed successor
activation with the successor activation predicted under a zero write. A
streaming, closed-form channel covariance (rank 16), frozen activation-feature
locations, and spatiotemporal innovation support are sealed. This precompute
uses no optimizer, no physical action, no state label, and no pH tensor.

For a new context, fixed pixels-only change probes are differentiated through
the actual frozen autoregressive suffix at horizons `h in {1,2,4}`. The
current-context covectors are projected into the sealed empirical channel
tangent, weighted by support averaged over the 32 nearest fit activations, and
converted by the Euclidean Riesz map plus thin polar factor into
`U_J(context)`. Its columns are oriented and orthonormal, span the full
residual stream without a manual spatial mask, and contain no learned or
interpolated port parameters. The true Jacobian is recomputed for every new
categorical context, including every autoregressive planning step.

For horizons `h in {1, 2, 4}`, let

```text
K_h(x) = d [E(frozen_rollout_h(A_t))] / d A_t
```

be the exact Jacobian from the block-5 activation `A_t` to the re-encoded future
state. Let

```text
G_h^pH(x) = d Phi_h(x; epsilon e_j, 0, ..., 0) / d epsilon at epsilon = 0
```

be the controlled variational response of the direct pH model when a unit port
input is applied for the first frame only. The registered bridge loss matches

```text
K_h(x) U_J(x) ~= G_h^pH(x)
```

jointly across all three horizons, up to one shared persistent port-basis
transform. The comparison uses principal-angle and normalized response losses
rather than entry-wise vector MSE, while the persistent frame prevents a
different unconstrained basis from being chosen at every state.

The same bridge also has a cotangent branch. A fixed pixels-only PCA bank of
future-frame change observables is differentiated with respect to the full
activation stream. These visual covectors are pulled back through `D_h E` into
state covectors, then mapped by the Poisson sharp map `-J(x)` before their
distribution is compared with `B(x)`. Independently, the exact pushed-forward
tangent `D_h E U_J / dt` is compared with `B(x)`. Pullback compatibility,
cross-horizon consistency, isotropy, and one persistent global port frame are
included. The
tangent branch always differentiates the latent state
`E(frozen_rollout_h(context))`; it never substitutes renderer pixels or the
learned pH renderer for that state response.

The model is additionally penalized when an activation write behaves like an
instantaneous image rewrite rather than a dynamical port. The training proxy is
pixels-only: it penalizes change to the decoded current frame while requiring a
consistent change to future temporal differences under the autonomous flow. It
does not group latent coordinates as position or momentum. Paired positive and
negative writes enforce local odd symmetry. Decode–re-encode consistency
penalizes activation writes that leave the learned video manifold. The physical
relative-degree interpretation is tested only after freezing, in Gate 5.

### Structure-preserving discrete dynamics

Every trained and evaluated transition uses the same second-order discrete
gradient pH step:

```text
(x_(k+1) - x_k) / dt
  = (J_d - R_d) grad_bar H + B_d v_k,
y_k = B_d^T grad_bar H.
```

It must satisfy the following balance after the registered numerical solve:

```text
H(x_(k+1)) - H(x_k)
  = dt * (-grad_bar(H)^T R_d grad_bar(H) + y_k^T v_k).
```

Training always executes exactly 32 relaxed fixed-point iterations; its solver
early-stop tolerance is deliberately `0`, so `1e-8` is **not** a training-time
stopping rule. The exposed final implicit residual is audited separately and a
transition is admissible only when that residual is at most `1e-8`. Failed
solves are reported and cannot be silently replaced by Euler, midpoint, RK, or
an energy projection. Training and evaluation use this identical forward map.

### One joint optimization

From its first step, the direct objective contains 14 weighted terms. In exact
configuration-field order they are:

| term | weight |
|---|---:|
| current-frame reconstruction | `1.0` |
| rendered rollout, horizons `(1,2,4,8)` | `1.0` |
| normalized latent rollout | `1.0` |
| latent-innovation prior/independence | `0.10` |
| tangent plus cotangent Jacobian bridge | `1.0` |
| write oddness | `0.25` |
| write manifold cycle/current-frame leakage/signal floor | `0.25` |
| Poisson-chart conditioning | `0.001` |
| state whitening | `0.05` |
| Hamiltonian mean/scale gauge | `0.01` |
| port-frame transport plus rank orientation | `0.10` |
| port-frame holonomy | `0.05` |
| normalized implicit residual penalty | `0.01` |
| discrete chain-rule penalty | `0.01` |

Backbone pretraining and the closed-form empirical-tangent seal necessarily
precede pH construction, but there is no separately identified dynamics model,
no pH projection stage, and no detached dynamics target. Within the direct
model, `E`, the inverse latent-effort head, `H`, `J`, `R`, and `B` are optimized
jointly from the first step.
The implicit-residual and chain-rule normalization tolerances are `1e-8` and
`1e-7`; the innovation target variance is `0.25`.

The checkpoint score is deliberately narrower than the training loss. It is
the mean validation reconstruction, rendered-rollout, latent-rollout, and
`0.10`-weighted innovation loss, plus `1.0` bridge, `0.25` oddness, and `0.25`
manifold-cycle losses for lens-enabled variants. It excludes whitening,
energy-gauge, port-frame, chart, implicit-penalty, and chain-penalty terms.
Checkpoint candidates are ranked first by a fail-closed `structureEligible`
flag and then by that score. Eligibility requires audited implicit residual
`<=1e-8`, chain-rule defect `<=1e-7`, balance defect `<=1e-7`, minimum singular
value of `B(x)` `>=1e-5`, and, when the lens is enabled, first-order signal
`>=1e-7`, minimum frozen and pH response singular values `>=1e-6`, extracted
port singular value `>=1e-8`, polar orthonormality defect `<=1e-4`, and
projected frozen-Jacobian signal ratio `>=1e-6`.

Ordinary validation losses are averaged over eight fixed batches. Lens
validation separately uses eight disjoint groups of four trajectories: 32
distinct lens contexts. Its ordinary losses are means, while first-order
signal and both minimum response singular values use the worst group. No eight
graphs are retained together.

All direct components are optimized for 30,000 AdamW steps with microbatch 16,
gradient accumulation 1, and a four-example Jacobian lens on every step.
Learning rate is `2e-4`, weight decay `1e-5`, warm-up is 1,000 steps, minimum
cosine-decay ratio is `0.05`, gradient clipping is 1.0, and validation plus
atomic checkpointing occur every 500 steps. Test, calibration, realizability,
control, and simulator-state diagnostics are not queried during selection.

## Analytic physical grounding with no gradient

After model selection, every neural parameter is frozen permanently. Physical
grounding may use only known probe commands and before/after pixels.

At a chosen state and physical axis `e_j`, paired commands `+alpha e_j` and
`-alpha e_j` are applied for one environment step. Their centered re-encoded
effect is

```text
g_ij = (E(o_ij^+) - E(o_ij^-)) / (2 alpha dt).
```

Exactly four paired states per physical action axis are allowed: eight total
environment steps for the pendulum and sixteen for Blocket League. States are
chosen before any response is observed from a fixed pixel-only candidate pool.
For every calibrated model—including all ablations, the independent
unstructured world model, and the activation planner—the candidate Gram is
computed by re-encoding the raw candidate pixels with that model's own frozen
encoder and evaluating its own port field. Each model's Grams are normalized by their own mean trace, and a
deterministic greedy max-min D-optimal rule selects the four indices that
maximize the worst cumulative log determinant across models. The exact same
four indices are used for every physical axis. Thus neither the primary model
nor a comparator can select a private, better-conditioned calibration set. No
simulator state, physical response, or adaptive extra query enters selection.

Only one constant interface matrix `T` is fitted:

```text
T* = argmin_T sum_ij ||g_ij - B(x_i) T e_j||^2 + 1e-6 ||T||_F^2.
```

The unique ridge solution is computed by linear algebra. There is no optimizer,
backpropagation, finite-tuning step, state-dependent calibrator, or network
update. For the scalar pendulum, this reduces to one signed scale. For Blocket
League, it is one full `2 x 2` matrix. A calibration failure cannot be repaired
with more probes.

One single bank of four paired raw-pixel responses per physical axis and
interface is collected after that shared max-min selection. That exact sealed
`+/-` pixel bank is re-encoded separately by every registered model with its
own encoder; no baseline or
ablation may execute an additional calibration probe. The bank hash binds the
selection method, complete model list, normalization scales, indices,
identifiers, log-determinant trajectory, the complete pre-probe pixel-context
pool, the physical system/time step/probe amplitude, the exact interface
matrix, and raw tensors. For the generic
activation planner the fitted response field is not `B(x)` but the action-free
activation Jacobian

```text
A_U(x) = D_h E(h) U_J(context) / dt.
```

Its sole fitted object is therefore the constant ridge map in
`g_ij = A_U(x_i) T_U e_j`, using its re-encoding of those already collected
pixels. This is stronger and fairer than blindly reusing the pH calibration:
training aligns lens responses with `D_pH Q`, so a pure basis conversion would
be `T_U = Q^T T_pH`, but that conversion would not absorb finite lens/pH
mismatch. Both constructions are analytic and require no gradient; the shared-
bank refit is the registered result and adds exactly zero environment steps.

The same rule applies to realizability evaluation: one raw held-out bank of
128 paired responses per physical axis and interface is collected once, then
re-encoded by every model. Thus the physical budget per interface is exactly
`4 + 128` pairs per axis, independent of the number of baselines or ablations.
Both banks carry hashes, candidate identifiers, and exact environment-step
counts; a model-specific physical read fails the audit.

Calibration is repeated with the same budget for an actuator interface absent
from video generation and all previous stages. The native interface is the
identity. The unseen pendulum interface applies `u_native = -1.6 u`. The unseen
Blocket interface applies

```text
u_native = Rot(37 degrees) [[0, 0.65], [-1.40, 0]] u.
```

All calibration, held-out probes, CEM candidates, random controls, and executed
commands remain inside one symmetric interface-coordinate box shared by the
native and unseen wrappers. Its half-width is computed analytically as

```text
a = 1 / max_(M in {native,unseen}, s in {-1,+1}^m) ||M s||_2.
```

This gives `a = 0.625` for the pendulum and
`a = 0.647863548391737` for Blocket. The latter uses the Euclidean norm that
`BlocketLeagueEnv.step_vector` actually saturates. CEM and random shooting use
`[-a,a]^m`; the plant wrapper performs no clipping and fails if either an
interface command leaves this box or its transformed native command has norm
above one. Each artifact records the formula, bound, exact matrix, and protocol
hash.

This is a fixed invertible rotation, axis exchange, sign change, and
anisotropic scale. These maps are implemented in the environment wrapper and
are never passed to the controller. Only `T` is recomputed; `E`, `Psi`, `H`,
`J`, `R`, `B`, the sealed empirical tangent, the Jacobian extractor, the
decoder, and all planners remain unchanged.

The interface family is therefore explicitly restricted to a constant
invertible change of port basis. This experiment does not claim that a few
probes can identify an arbitrary state-dependent actuator interface.

## Registered baselines

All learned baselines use the same sanitized raw-pixel fit/validation splits,
the same immutable action-free backbone tensor values, latent dimension,
inferred-action rank, optimization-step count, checkpoint schedule,
calibration queries, planning horizon, physical action bounds, and control
episodes. Sharing the frozen backbone does not mean sharing a representation:
the primary unstructured baseline owns every trainable tensor downstream of
that backbone.

1. **Independent unstructured Jacobian-lens world model:** owns its own
   whole-stream encoder readout `E_u`, renderer, inverse latent-effort head,
   state-dependent drift `f_u(x)`, state-dependent port `B_u(x)`, and
   persistent response frame. It uses the exact same zero-parameter empirical
   Jacobian-port construction and pixels-only probes as the structured model;
   those frozen buffers are copied into a distinct module object. The
   trainable modules are
   optimized jointly end to end from `pixelContexts` and `frames` for exactly
   the same number of steps as the pH model. Its multi-horizon tangent bridge
   matches `D_p E_u(frozen_rollout_h)` with `D_v Phi_h^u`; it receives neither
   the cotangent Poisson bridge nor Hamiltonian-, Poisson-, passivity-, chart-,
   discrete-gradient-, or energy-specific losses. Every other applicable loss
   is identical: current reconstruction, latent and pixel rollout, inferred-
   effort gauge/independence, state whitening, port rank/frame/holonomy,
   write oddness, manifold cycle, and multi-horizon tangent alignment.

   Its encoder readout, renderer, inverse head, and response frame
   start from tensor-identical homologous values, but they are distinct tensor
   objects and are subsequently trained independently. Those values are
   reconstructed only from a fresh untrained full architecture under the
   sealed initialization seed; no selected or trained pH checkpoint is an
   input to baseline construction. The unstructured hidden width is selected
   so that the **complete downstream trainable count**—own encoder, renderer,
   inverse, drift/port, and response frame—differs from the
   complete pH downstream trainable count by at most one percent. Counts,
   width, relative gap, and homologous initialization hashes are recomputed
   from instantiated modules. The baseline checkpoint contains those modules
   separately and contains no backbone state, full-model hash, structured
   latent, or cached encoded-state tensor. Its archive and every optimization
   batch retain the exact raw-pixel-derived `{pixelContexts, frames}` schema.
2. **Generic frozen-world-model planner:** plans coefficients of the calibrated
   exact activation port directly through frozen autoregressive pixel rollouts.
   At every predicted step it recomputes the frozen observable covectors on the
   current categorical context, extracts `U_J(context)` with the sealed
   empirical tangent, writes `U_J(context) z` into the same frozen residual
   stream, and feeds back the hard argmax category. The pH core, latent
   renderer, and inverse head are inaccessible to this planner.
3. **Coast and random shooting:** execute zero input or the registered random
   action sampler under the same action bounds.

No baseline receives physical action labels during a gradient update. The
independent unstructured model is calibrated by fitting only its own constant
interface matrix against its own `E_u`/`B_u` responses from the same raw four-
pair bank; it never reuses a pH encoding, rendering, port, or calibration.
An
action-conditioned oracle may be reported only as a clearly separated upper
bound and can never participate in the breakthrough decision.

The generic activation planner intentionally shares the full model's
action-free encoder, pixels-only probes, and zero-parameter exact extractor. It
is therefore a strong *planning-mechanism*
baseline—direct pixel rollout versus pH latent rollout—not evidence that the
pretrained transformer alone identified an actuator. Jacobians are computed
through the differentiable soft rollout at each categorical context, while the
autoregressive feedback itself is hard argmax; any off-manifold compounding is
reported rather than repaired with another learned model.

## Mandatory ablations

Each ablation is trained from the same seed lineage and pixel splits. Ablations
are explanatory; none can replace the registered model or rescue a failed gate.

1. **No Jacobian bridge:** `B(x)` is learned only from inferred latent
   innovations.
2. **Single-horizon lens:** uses only `h = 1`, removing flow-transported
   multi-horizon consistency.
3. **Random/write-shuffled lens:** preserves norms and rank but permutes
   already-computed Jacobian correspondence targets across trajectories.
4. **Skew-only interconnection:** replaces the Poisson push-forward with an
   unconstrained state-dependent skew matrix, preserving continuous power
   orthogonality but not Jacobi.
5. **Constant port:** replaces `B(x)` with a learned constant matrix. Its exact
   *total end-to-end* trainable count is recomputed and must differ from the full
   model by at most one percent. The dynamics-submodule count and its (larger)
   gap are reported separately; this ablation is not claimed to be
   dynamics-capacity matched.
6. **Non-structure-preserving integrator audit:** evaluates, without retraining,
   matched-step RK2 to quantify discrete power error. It is never used by the
   registered controller.

## Locked evaluation

All primary control targets and errors are defined from pixels. All three
learned planners use the same receding-horizon candidate budget and execute only the
first command before re-encoding fresh pixels. Results are paired by initial
condition. Point estimates and paired 95% bootstrap intervals are reported,
but many episodes from one trained model do not substitute for multiple
training seeds.

Pendulum episodes last 80 environment steps and use a 24-step planning horizon.
Blocket episodes last 48 steps and use a 12-step horizon. Every planner receives
512 candidate sequences, four cross-entropy-method updates, and 64 elites per
decision. Costs compare predicted pixels with the supplied target image and add
the same action-magnitude penalty. No controller receives a target coordinate
or simulator state. The learned planners also receive identical CEM random
normal draws: their first candidate population is exactly paired, and later
populations differ only through model-dependent elite updates. Activation
rollouts are evaluated in deterministic micro-batches of 32 solely to cap GPU
memory; all 512 candidate costs are retained before each elite update.

Every gate below must pass independently on **both** systems. Interface gates
must pass separately on the native and unseen interfaces.

### Gate 1 — firewall and frozen backbone

- zero forbidden tensors or files are observed by any gradient-based phase;
- sanitized archive schema and SHA-256 match the sealed manifest;
- the canonical source-tree hash and every runtime-trace source boundary match;
- the sealed learner-bundle hash, every recursively observed learner source
  file, and one code-free cache inventory per gradient phase match exactly;
- zero unregistered learner source file, cache Python file, symlink, or special
  path is visible;
- the public canonical `HiddenExcitationConfig` serialization/digest matches,
  while the 128-bit producer seed is absent from every learner mount and
  serialized payload;
- canonical pre/post backbone parameter-and-buffer hashes match exactly;
- analytic calibration is the only stage that reads a physical command.

This gate is binary. One violation fails the complete outcome.

### Gate 2 — direct visual and dynamical quality

This is a posterior-conditioned reconstruction audit, not unconditional
action-free forecasting: each model's own frozen encoder and inverse head infer
a latent state/innovation sequence from consecutive held-out test frames; its
own dynamics rolls forward from its own first state and its own renderer
produces the evaluated pixels. It therefore tests
whether the learned port carries transition information without physical action
labels. It must not be interpreted as evidence for long-horizon autonomous
prediction.

- current-frame pendulum-bob pixel IoU is at least `0.80` and
  at least `0.70` for each Blocket disc;
- horizon-8 centroid error is no worse than `1.10x` the matched unstructured
  Neural ODE **separately for the pendulum bob, Blocket player disc, and Blocket
  puck disc**; object errors are never averaged before the pass/fail decision;
- horizon-8 weighted pixel cross-entropy is no worse than `1.10x` that baseline;
- shuffling inferred innovations worsens horizon-8 pixel error by at least
  `10%`, showing that the direct model uses its latent port.

### Gate 3 — Poisson, passivity, and numerical power

- maximum normalized skew defect is at most `1e-7`;
- maximum normalized Jacobi defect is at most `1e-5` on 4,096 held-out states;
- minimum eigenvalue of `R(x)` is at least `-1e-7`;
- minimum singular value of the complete state-dependent `B(x)` is at least
  `1e-5`;
- maximum relative continuous power-identity defect is at most `1e-6`;
- maximum relative controlled discrete power-identity defect is at most
  `1e-5`;
- zero-input energy-increase fraction is at most `0.001`;
- implicit-solver failure fraction is at most `0.001`.

These structural identities are recomputed on a float64 clone. In addition,
the literal deployed `model.step` path is audited in float32 for 16 autonomous
steps and must match an independent float32 core to `1e-6` relative error. Its
discrete balance must satisfy both relative defect `<=5e-3` and absolute defect
`<=5e-6`; its implicit residual is bounded by `5e-5`, with zero-effort energy
tolerance `5e-6`. Passing only the float64 clone is insufficient.

Energy is reported in learned units. Passing this gate does not identify a
physical joule scale.

### Gate 4 — internal Jacobian port

- paired activation writes have at least `0.90` odd-symmetry cosine;
- their effect norm is at least `3.0x` the median norm-matched random-write
  effect;
- the minimum principal cosine between flow-pulled-back response subspaces at
  horizons 1, 2, and 4 is at least `0.80`;
- normalized multi-horizon Jacobian-to-pH response error is at most `0.25`;
- on the exact activation path `U_J(context) -> frozen transformer suffix -> predicted
  soft frame -> shifted context -> E`, at least `90%` of horizon-1 Jacobian
  directions retain the corresponding direct pH-step direction after applying
  the same single orthogonal Procrustes frame fitted jointly over every held-out
  state and all three horizons. No state- or horizon-dependent frame is fit,
  and the learned pH renderer is not part of this pass/fail check.
- the numerical JVP/VJP adjoint defect of that real path is at most `2e-4`,
  and explicit `D_hE v` agrees with an independent JVP to `2e-4`;
- the extracted port has polar-orthonormality defect at most `2e-5`, relative
  minimum singular value at least `1e-6`, and projected frozen-Jacobian signal
  ratio in `[1e-6, 1 + 2e-5]`;
- every held-out extraction reports exactly 32 unique, in-range fit neighbors;
- the live source files, backbone tensors, complete empirical-extractor
  buffers, source tree, retention path, and horizons reproduce the sealed
  Gate-4 fingerprint exactly.

### Gate 5 — force-port signature

After one affine post-hoc audit alignment, used only to group standardized
configuration and velocity/momentum coordinates:

- normalized affine-chart error is at most `0.35` both on the 256-state
  alignment split and on a disjoint 128-state evaluation split;
- a **positive affine** map from the frozen learned Hamiltonian `H(x)` to the
  simulator's physical energy, fitted on those same 256 alignment states and
  evaluated on the disjoint 128 states, has held-out normalized RMSE at most
  `0.35`, `R^2` at least `0.85`, and Pearson correlation at least `0.90`;
- immediate configuration-effect norm is at most `0.35x` the immediate
  velocity/momentum-effect norm;
- the configuration-effect norm at horizon 4 is at least `1.50x` its horizon-1
  value;
- before first contact, the non-actuated Blocket disc receives at most `0.25x`
  the actuated disc's immediate momentum effect, evaluated on at least 64
  conservatively certified one-step pre-contact states.

This audit tests relative degree and locality. The affine alignment does not
make the learned coordinates or matrices unique. Simulator coordinates are
opened only in this post-freeze audit; the artifact is accepted only after both
candidate pools, coordinates, encodings, Jacobian responses, masks, and hashes
are regenerated from the sealed checkpoint. A standalone typed evidence object
cannot authenticate itself. The energy-semantic fit is audit-only: it supplies
no gradient, command, checkpoint choice, architecture choice, or threshold
choice. Its evidence is cryptographically bound to the system, Gate-5 evidence,
frozen model and checkpoint, and the alignment/evaluation context hashes.

### Gate 6 — held-out physical realizability

After the fixed analytic calibration and with no update on 128 new paired
counterfactuals per physical axis:

- mean calibrated response cosine is at least `0.85`;
- every physical axis has response cosine at least `0.75`;
- commanded-sign agreement is at least `0.85`;
- centered response-magnitude `R^2` is at least `0.60`;
- the full model's mean cosine exceeds both the single-horizon and shuffled-lens
  ablations by at least `0.10`.

All comparisons are made after optimal use of the already fitted constant `T`;
no raw column of `B` is assigned a physical name.

### Gate 7 — real closed-loop control

On 64 paired episodes per interface:

- final pixel-defined target error is at least `25%` below coast;
- it is at least `15%` below **each** learned baseline (including every later
  independently trained WM baseline admitted to the registered controller
  set);
- the registered model wins at least `65%` of paired episodes against each
  learned baseline;
- a separate paired 95% bootstrap interval excludes zero for every learned
  baseline; no baseline is selected from its observed mean before inference;
- the no-Jacobian and shuffled-lens ablations are each at least `10%` worse in
  final target error than the full model.

These conditions must hold separately for native and unseen interfaces.

### Gate 8 — unseen-interface transfer

- the unseen-interface improvement over coast retains at least `80%` of the
  native-interface improvement;
- held-out realizability under the unseen interface still satisfies Gate 6;
- a typed fingerprint reconstructed from authenticated/replayed native and
  unseen shards proves exact equality of the controller graph, every neural
  module hash, CEM configuration, episode/planner seeds, episode identities
  and hidden initial states, pixel targets, target source, controller order,
  horizon, and planner-seed schedule;
- the two registered `PhysicalInterface` matrices and the values of the frozen
  constant analytic `T` matrices are the only permitted differences.

### Gate 9 — two-system conjunction

Gates 1–8 pass for both the damped pendulum and Blocket League in this one
locked run. A result on only one system is informative but is not a positive
registered outcome.

## Single-seed outcome strings

There are exactly two allowed top-level outcomes:

```text
direct_jacobian_poisson_ph_breakthrough_supported_single_seed_two_systems
direct_jacobian_poisson_ph_breakthrough_not_supported_single_seed
```

The positive string is emitted only if every gate passes conjunctively. Any
failed or unauditable gate emits the negative string, followed by the complete
gate table and all ablation results. “Nearly passed,” post-hoc threshold changes,
or success on one system cannot produce an intermediate breakthrough label.

Even the positive string means that the registered breakthrough hypothesis is
supported for one training seed. It does not establish reproducibility and it
does not authorize a claim that `H`, `J`, `R`, or `B` were individually
recovered in physical coordinates.
