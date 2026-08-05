from __future__ import annotations

import copy
from dataclasses import replace
import inspect
import math
import unittest

import torch

from blocket_league.direct_ph_structure_audits import (
    ACTIVATION_SUFFIX_RETENTION_PATH,
    FirewallAuditEvidence,
    ForcePortThresholds,
    Gate3Thresholds,
    Gate4Thresholds,
    LensAuditEvidence,
    audit_force_port_signature,
    audit_matched_rk2_power_error,
    audit_gate_1,
    audit_gate_3,
    audit_gate_4,
    fit_postfreeze_affine_audit_alignment,
    gate4_path_fingerprint_sha256,
    seal_gate3_transition_samples,
)
from blocket_league.end_to_end_ph_experiment import LatentPatchTransformerRenderer
from blocket_league.passive_jacobian_ph_model import module_tensor_hash
from blocket_league.pixel_direct_model import DirectPixelTransformer, PixelDirectConfig
from blocket_league.direct_poisson_ph import (
    DirectPoissonPHConfig,
    DirectPoissonPortHamiltonian,
)
from blocket_league.direct_visual_poisson_ph import (
    DirectVisualPoissonPH,
    WholeStreamEncoderConfig,
    WholeStreamFrozenEncoder,
)


GATE3_SOURCE_SHA256 = "a" * 64


def _rk2_audit(core, states, efforts, thresholds, *, seal=None, **kwargs):
    transition_seal = seal or seal_gate3_transition_samples(
        core,
        states.detach(),
        efforts.detach(),
        source_manifest_sha256=GATE3_SOURCE_SHA256,
    )
    return audit_matched_rk2_power_error(
        core,
        states,
        efforts,
        thresholds,
        transition_seal=transition_seal,
        expected_source_manifest_sha256=GATE3_SOURCE_SHA256,
        expected_core_sha256=module_tensor_hash(core),
        **kwargs,
    )


class DirectPHStructureAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(907_113)

    @staticmethod
    def _visual_model() -> DirectVisualPoissonPH:
        backbone = DirectPixelTransformer(
            PixelDirectConfig(
                image_size=8,
                patch_size=4,
                palette_size=9,
                history_frames=2,
                pixel_embedding_size=3,
                hidden_size=8,
                depth=2,
                heads=2,
                mlp_ratio=2.0,
            )
        )
        encoder = WholeStreamFrozenEncoder(
            backbone,
            WholeStreamEncoderConfig(
                state_size=2,
                readout_hidden_size=8,
                lens_block=0,
            ),
        )
        renderer = LatentPatchTransformerRenderer(
            2,
            image_size=8,
            patch_size=4,
            palette_size=9,
            hidden_size=8,
            depth=1,
            heads=2,
        )
        core = DirectPoissonPortHamiltonian(
            DirectPoissonPHConfig(
                state_size=2,
                port_size=1,
                hidden_size=8,
                hidden_layers=1,
                coupling_layers=2,
                dt=0.01,
                implicit_iterations=32,
                implicit_relaxation=0.9,
                implicit_tolerance=1e-12,
                discrete_gradient_epsilon=1e-18,
            )
        )
        return DirectVisualPoissonPH(encoder, renderer, core)

    @staticmethod
    def _firewall_evidence(model: DirectVisualPoissonPH) -> FirewallAuditEvidence:
        backbone_hash = module_tensor_hash(model.encoder.backbone)
        return FirewallAuditEvidence(
            sealed_archive_sha256="a" * 64,
            observed_archive_sha256="a" * 64,
            sealed_source_tree_sha256="b" * 64,
            observed_source_tree_sha256="b" * 64,
            sealed_source_schema=("frames",),
            observed_source_schema=("frames",),
            sealed_gradient_schemas=(
                ("pixels",),
                ("pixelContexts", "frames"),
            ),
            observed_gradient_schemas=(
                ("pixels",),
                ("frames", "pixelContexts"),
            ),
            sealed_backbone_hash=backbone_hash,
            observed_backbone_hashes=(backbone_hash, backbone_hash, backbone_hash),
            forbidden_gradient_read_count=0,
            backbone_in_optimizer=False,
            nonanalytic_command_read_count=0,
            runtime_trace_event_count=20,
            runtime_gradient_batch_count=3,
            runtime_file_read_count=4,
            runtime_stage_boundary_count=3,
            runtime_mount_inventory_count=3,
            forbidden_training_mount_count=0,
            sealed_learner_bundle_sha256="c" * 64,
            observed_learner_bundle_sha256="c" * 64,
            expected_learner_manifest_count=3,
            observed_learner_manifest_count=3,
            expected_learner_source_inventory_count=3,
            observed_learner_source_inventory_count=3,
            forbidden_learner_source_file_count=0,
            learner_source_file_mismatch_count=0,
            expected_learner_cache_inventory_count=3,
            observed_learner_cache_inventory_count=3,
            learner_cache_python_code_file_count=0,
            learner_cache_unsafe_path_count=0,
        )

    def test_gate_1_passes_only_exact_firewall_and_backbone_evidence(self) -> None:
        model = self._visual_model()
        result = audit_gate_1(model, self._firewall_evidence(model))
        self.assertTrue(result.auditable, result.failures)
        self.assertTrue(result.passed, result.to_dict())
        self.assertTrue(all(result.checks.values()))

        bad = FirewallAuditEvidence(
            **{
                **self._firewall_evidence(model).__dict__,
                "observed_gradient_schemas": (("pixels", "action"),),
                "forbidden_gradient_read_count": 1,
            }
        )
        failed = audit_gate_1(model, bad)
        self.assertTrue(failed.auditable)
        self.assertFalse(failed.passed)
        self.assertFalse(failed.checks["gradient_schema_has_no_forbidden_key"])
        self.assertFalse(failed.checks["zero_forbidden_gradient_reads"])

        for field, check, value in (
            ("observed_archive_sha256", "archive_hash_exact", "f" * 64),
            (
                "observed_source_tree_sha256",
                "source_tree_hash_exact",
                "f" * 64,
            ),
            ("observed_source_schema", "source_schema_exact", ("states",)),
            (
                "observed_gradient_schemas",
                "gradient_schemas_exact",
                (("pixels",),),
            ),
            (
                "observed_backbone_hashes",
                "all_stage_backbone_hashes_exact",
                ("f" * 64,),
            ),
            (
                "forbidden_gradient_read_count",
                "zero_forbidden_gradient_reads",
                1,
            ),
            ("backbone_in_optimizer", "backbone_absent_from_optimizer", True),
            (
                "nonanalytic_command_read_count",
                "commands_read_only_by_analytic_grounding",
                1,
            ),
        ):
            tampered = FirewallAuditEvidence(
                **{**self._firewall_evidence(model).__dict__, field: value}
            )
            result = audit_gate_1(model, tampered)
            self.assertFalse(result.passed)
            self.assertFalse(result.checks[check])

        for field, check, value in (
            ("runtime_trace_event_count", "runtime_trace_is_nonempty", 0),
            (
                "runtime_gradient_batch_count",
                "runtime_gradient_batches_observed",
                0,
            ),
            ("runtime_file_read_count", "runtime_file_reads_observed", 0),
            (
                "runtime_stage_boundary_count",
                "runtime_stage_boundaries_observed",
                0,
            ),
            (
                "runtime_mount_inventory_count",
                "runtime_mount_inventory_observed",
                0,
            ),
            (
                "forbidden_training_mount_count",
                "no_forbidden_training_mount",
                1,
            ),
            (
                "observed_learner_bundle_sha256",
                "learner_bundle_hash_exact",
                "f" * 64,
            ),
            (
                "observed_learner_manifest_count",
                "all_learner_manifest_reads_observed",
                2,
            ),
            (
                "observed_learner_source_inventory_count",
                "all_learner_source_inventories_observed",
                2,
            ),
            (
                "forbidden_learner_source_file_count",
                "zero_forbidden_learner_source_files",
                1,
            ),
            (
                "learner_source_file_mismatch_count",
                "learner_source_files_match_manifest",
                1,
            ),
            (
                "observed_learner_cache_inventory_count",
                "all_learner_cache_inventories_observed",
                2,
            ),
            (
                "learner_cache_python_code_file_count",
                "learner_caches_contain_no_python_code",
                1,
            ),
            (
                "learner_cache_unsafe_path_count",
                "learner_caches_have_no_unsafe_paths",
                1,
            ),
        ):
            tampered = FirewallAuditEvidence(
                **{**self._firewall_evidence(model).__dict__, field: value}
            )
            result = audit_gate_1(model, tampered)
            self.assertFalse(result.passed)
            self.assertFalse(result.checks[check])

    def test_gate_1_missing_evidence_is_an_unauditable_failure(self) -> None:
        result = audit_gate_1(self._visual_model(), FirewallAuditEvidence())
        self.assertFalse(result.auditable)
        self.assertFalse(result.passed)
        self.assertTrue(any("missing Gate 1 evidence" in item for item in result.failures))

    def test_gate_3_direct_poisson_core_passes_all_structural_identities(self) -> None:
        core = DirectPoissonPortHamiltonian(
            DirectPoissonPHConfig(
                state_size=4,
                port_size=2,
                hidden_size=12,
                hidden_layers=1,
                coupling_layers=4,
                dt=0.012,
                implicit_iterations=56,
                implicit_relaxation=0.85,
                implicit_tolerance=1e-13,
                discrete_gradient_epsilon=1e-18,
            )
        ).double()
        states = 0.2 * torch.randn(4, 4, dtype=torch.float64)
        efforts = 0.1 * torch.randn(4, 2, dtype=torch.float64)
        result = audit_gate_3(
            core,
            states,
            efforts,
            Gate3Thresholds(minimum_states=4),
            production_step=copy.deepcopy(core).float().step,
            chunk_size=2,
        )
        self.assertTrue(result.auditable, result.failures)
        self.assertTrue(result.passed, result.to_dict())
        self.assertLessEqual(
            float(result.metrics["maximum_normalized_jacobi_defect"]), 1e-5
        )
        self.assertGreaterEqual(
            float(result.metrics["minimum_resistance_eigenvalue"]), -1e-7
        )

    def test_gate_3_never_skips_an_insufficient_heldout_suite(self) -> None:
        core = DirectPoissonPortHamiltonian(
            DirectPoissonPHConfig(state_size=2, port_size=1, coupling_layers=2)
        )
        result = audit_gate_3(
            core,
            torch.zeros(3, 2),
            torch.zeros(3, 1),
            Gate3Thresholds(minimum_states=4),
            production_step=copy.deepcopy(core).float().step,
        )
        self.assertFalse(result.auditable)
        self.assertFalse(result.passed)
        self.assertIn("requires at least 4", " ".join(result.failures))

    def test_matched_rk2_ablation_uses_the_same_frozen_gate3_samples(self) -> None:
        core = DirectPoissonPortHamiltonian(
            DirectPoissonPHConfig(
                state_size=4,
                port_size=2,
                hidden_size=10,
                hidden_layers=1,
                coupling_layers=4,
                dt=0.01,
                implicit_iterations=48,
                implicit_relaxation=0.85,
                discrete_gradient_epsilon=1e-18,
            )
        ).eval().requires_grad_(False)
        states = 0.2 * torch.randn(8, 4)
        efforts = 0.1 * torch.randn(8, 2)
        result = _rk2_audit(
            core,
            states,
            efforts,
            Gate3Thresholds(
                minimum_states=8,
                maximum_relative_discrete_power_defect=1e-4,
            ),
            chunk_size=4,
        )
        self.assertTrue(result.auditable, result.failures)
        self.assertTrue(result.passed, result.to_dict())
        self.assertEqual(result.sample_count, 8)
        self.assertEqual(result.metrics["numeric_dtype"], "float64")
        self.assertIn("rk2_maximum_relative_power_defect", result.metrics)
        self.assertIn(
            "matched_structure_preserving_maximum_relative_power_defect",
            result.metrics,
        )

    def test_matched_rk2_ablation_rejects_trainable_or_attached_evidence(self) -> None:
        core = DirectPoissonPortHamiltonian(
            DirectPoissonPHConfig(state_size=2, port_size=1, coupling_layers=2)
        )
        states = torch.zeros(4, 2)
        efforts = torch.zeros(4, 1)
        trainable = _rk2_audit(
            core,
            states,
            efforts,
            Gate3Thresholds(minimum_states=4),
        )
        self.assertFalse(trainable.auditable)
        core.eval().requires_grad_(False)
        attached = _rk2_audit(
            core,
            states.clone().requires_grad_(True),
            efforts,
            Gate3Thresholds(minimum_states=4),
        )
        self.assertFalse(attached.auditable)

    def test_matched_rk2_rejects_tampered_samples_manifest_and_core_hash(self) -> None:
        core = DirectPoissonPortHamiltonian(
            DirectPoissonPHConfig(state_size=2, port_size=1, coupling_layers=2)
        ).eval().requires_grad_(False)
        states = torch.randn(4, 2)
        efforts = torch.randn(4, 1)
        thresholds = Gate3Thresholds(minimum_states=4)
        seal = seal_gate3_transition_samples(
            core,
            states,
            efforts,
            source_manifest_sha256=GATE3_SOURCE_SHA256,
        )

        changed_states = states.clone()
        changed_states[0, 0] += 1e-3
        self.assertFalse(
            _rk2_audit(
                core, changed_states, efforts, thresholds, seal=seal
            ).auditable
        )
        wrong_manifest = replace(seal, source_manifest_sha256="b" * 64)
        self.assertFalse(
            _rk2_audit(
                core, states, efforts, thresholds, seal=wrong_manifest
            ).auditable
        )
        wrong_core = replace(seal, core_sha256="c" * 64)
        self.assertFalse(
            _rk2_audit(core, states, efforts, thresholds, seal=wrong_core).auditable
        )
        nonfinite = states.clone()
        nonfinite[0, 0] = float("nan")
        self.assertFalse(
            _rk2_audit(core, nonfinite, efforts, thresholds, seal=seal).auditable
        )

    @staticmethod
    def _passing_lens_evidence(
        *, sample_count: int = 4, port_size: int = 2
    ) -> LensAuditEvidence:
        observable_size = 5
        base = torch.randn(sample_count, observable_size, port_size)
        # Make every response comfortably full rank.
        base[..., :port_size, :] += 2.0 * torch.eye(port_size)
        angle = 0.37
        shared = torch.tensor(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ]
        )
        lens = {horizon: (1.0 + 0.1 * horizon) * base for horizon in (1, 2, 4)}
        ph = {horizon: response @ shared for horizon, response in lens.items()}

        delta = torch.randn(sample_count, port_size, 7)
        delta = delta / torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        baseline = torch.randn(sample_count, port_size, 7)
        positive = baseline + delta
        negative = baseline - delta
        random_norms = torch.full((sample_count, port_size, 4), 0.20)
        adjoint = torch.randn(sample_count, 3)
        explicit = torch.randn(sample_count, observable_size)
        code_sha256 = "a" * 64
        backbone_sha256 = "b" * 64
        extractor_sha256 = "e" * 64
        source_sha256 = "c" * 64
        fingerprint = gate4_path_fingerprint_sha256(
            code_sha256=code_sha256,
            backbone_sha256=backbone_sha256,
            extractor_sha256=extractor_sha256,
            source_tree_sha256=source_sha256,
            retention_path_kind=ACTIVATION_SUFFIX_RETENTION_PATH,
            horizons=(1, 2, 4),
        )
        return LensAuditEvidence(
            lens_responses=lens,
            ph_responses=ph,
            positive_effects=positive,
            negative_effects=negative,
            baseline_effects=baseline,
            random_write_effect_norms=random_norms,
            adjoint_jvp_inner_products=adjoint,
            adjoint_vjp_inner_products=adjoint.clone(),
            adjoint_jvp_norm_bounds=torch.ones_like(adjoint),
            adjoint_vjp_norm_bounds=torch.ones_like(adjoint),
            explicit_state_jacobian_products=explicit,
            independent_state_jvp_products=explicit.clone(),
            extracted_port_gram_matrices=torch.eye(port_size).expand(
                sample_count, port_size, port_size
            ).clone(),
            extracted_port_singular_values=torch.ones(sample_count, port_size),
            extracted_port_reported_orthonormality_defects=torch.zeros(
                sample_count
            ),
            extracted_projected_signal_ratios=torch.full(
                (sample_count,), 0.5
            ),
            extracted_neighbor_indices=torch.arange(32).expand(
                sample_count, 32
            ).clone(),
            extracted_neighbor_fit_population=64,
            path_code_sha256=code_sha256,
            sealed_path_code_sha256=code_sha256,
            path_backbone_sha256=backbone_sha256,
            sealed_backbone_sha256=backbone_sha256,
            path_extractor_sha256=extractor_sha256,
            sealed_extractor_sha256=extractor_sha256,
            path_source_tree_sha256=source_sha256,
            sealed_source_tree_sha256=source_sha256,
            path_fingerprint_sha256=fingerprint,
            random_writes_norm_matched=True,
            retention_path_kind=ACTIVATION_SUFFIX_RETENTION_PATH,
        )

    def test_gate_4_passes_basis_changed_multi_horizon_responses(self) -> None:
        result = audit_gate_4(
            self._passing_lens_evidence(),
            Gate4Thresholds(minimum_samples=4, minimum_random_draws=4),
        )
        self.assertTrue(result.auditable, result.failures)
        self.assertTrue(result.passed, result.to_dict())
        self.assertGreaterEqual(float(result.metrics["mean_odd_symmetry_cosine"]), 0.99)
        self.assertGreaterEqual(
            float(result.metrics["median_norm_matched_random_write_ratio"]), 4.9
        )
        self.assertLessEqual(
            float(result.metrics["normalized_multi_horizon_response_error"]), 1e-5
        )

    def test_gate_4_rejects_horizon_dependent_basis_and_graph_evidence(self) -> None:
        passing = self._passing_lens_evidence()
        ph = dict(passing.ph_responses or {})
        ph[4] = ph[4] @ torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        inconsistent = LensAuditEvidence(**{**passing.__dict__, "ph_responses": ph})
        result = audit_gate_4(
            inconsistent,
            Gate4Thresholds(
                minimum_samples=4,
                minimum_random_draws=4,
                maximum_normalized_response_error=0.05,
            ),
        )
        self.assertTrue(result.auditable)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["multi_horizon_response_error"])

        attached = LensAuditEvidence(
            **{
                **passing.__dict__,
                "positive_effects": passing.positive_effects.clone().requires_grad_(True),
            }
        )
        graph_result = audit_gate_4(
            attached,
            Gate4Thresholds(minimum_samples=4, minimum_random_draws=4),
        )
        self.assertFalse(graph_result.auditable)
        self.assertFalse(graph_result.passed)
        self.assertIn("autograd graph", " ".join(graph_result.failures))

    def test_gate_4_falsifies_numeric_provenance_and_extracted_port_hooks(self) -> None:
        passing = self._passing_lens_evidence()
        thresholds = Gate4Thresholds(minimum_samples=4, minimum_random_draws=4)
        corruptions = {
            "frozen_rollout_adjoint_identity": {
                "adjoint_vjp_inner_products": (
                    passing.adjoint_vjp_inner_products + 1.0
                )
            },
            "explicit_state_jacobian_matches_independent_jvp": {
                "independent_state_jvp_products": (
                    passing.independent_state_jvp_products + 1.0
                )
            },
            "extracted_port_orthonormality": {
                "extracted_port_gram_matrices": torch.zeros_like(
                    passing.extracted_port_gram_matrices
                )
            },
            "extracted_port_full_rank": {
                "extracted_port_singular_values": torch.cat(
                    (
                        passing.extracted_port_singular_values[..., :-1],
                        torch.zeros_like(
                            passing.extracted_port_singular_values[..., -1:]
                        ),
                    ),
                    dim=-1,
                )
            },
            "extracted_port_inside_empirical_tangent": {
                "extracted_projected_signal_ratios": torch.zeros_like(
                    passing.extracted_projected_signal_ratios
                )
            },
            "gate4_path_fingerprint": {"path_fingerprint_sha256": "d" * 64},
            "gate4_code_matches_seal": {"sealed_path_code_sha256": "d" * 64},
            "gate4_extractor_matches_seal": {
                "sealed_extractor_sha256": "d" * 64
            },
        }
        for failed_check, replacement in corruptions.items():
            with self.subTest(failed_check=failed_check):
                evidence = LensAuditEvidence(
                    **{**passing.__dict__, **replacement}
                )
                result = audit_gate_4(evidence, thresholds)
                self.assertTrue(result.auditable, result.failures)
                self.assertFalse(result.passed)
                self.assertFalse(result.checks[failed_check])

        invalid_neighbors = LensAuditEvidence(
            **{
                **passing.__dict__,
                "extracted_neighbor_indices": torch.full_like(
                    passing.extracted_neighbor_indices, 64
                ),
            }
        )
        neighbor_result = audit_gate_4(invalid_neighbors, thresholds)
        self.assertFalse(neighbor_result.auditable)
        self.assertIn("sealed fit population", " ".join(neighbor_result.failures))

    def test_gate_4_rejects_state_dependent_response_gain(self) -> None:
        passing = self._passing_lens_evidence(sample_count=8)
        ph = {
            horizon: response
            * torch.linspace(0.2, 2.0, response.shape[0])[:, None, None]
            for horizon, response in (passing.ph_responses or {}).items()
        }
        gauged = LensAuditEvidence(**{**passing.__dict__, "ph_responses": ph})
        result = audit_gate_4(
            gauged,
            Gate4Thresholds(
                minimum_samples=8,
                minimum_random_draws=4,
                maximum_normalized_response_error=0.05,
            ),
        )
        self.assertTrue(result.auditable, result.failures)
        self.assertFalse(result.passed)
        self.assertFalse(result.checks["multi_horizon_response_error"])

    def test_gate_4_retention_uses_one_global_port_frame(self) -> None:
        passing = self._passing_lens_evidence()
        quarter_turn = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        globally_gauged = LensAuditEvidence(
            **{
                **passing.__dict__,
                "ph_responses": {
                    horizon: response @ quarter_turn
                    for horizon, response in (passing.lens_responses or {}).items()
                },
            }
        )
        result = audit_gate_4(
            globally_gauged,
            Gate4Thresholds(minimum_samples=4, minimum_random_draws=4),
        )
        self.assertTrue(result.auditable, result.failures)
        self.assertTrue(result.passed, result.to_dict())
        self.assertGreaterEqual(
            float(result.metrics["activation_suffix_retained_direction_fraction"]),
            0.90,
        )

    def test_gate_4_renderer_cycle_cannot_satisfy_retention_provenance(self) -> None:
        passing = self._passing_lens_evidence()
        renderer_path = LensAuditEvidence(
            **{
                **passing.__dict__,
                "retention_path_kind": (
                    "port_hamiltonian_to_renderer_argmax_context_to_E"
                ),
            }
        )
        result = audit_gate_4(
            renderer_path,
            Gate4Thresholds(minimum_samples=4, minimum_random_draws=4),
        )
        self.assertFalse(result.auditable)
        self.assertFalse(result.passed)
        self.assertIn("exact activation U_J", " ".join(result.failures))

    def test_postfreeze_affine_force_port_signature_is_audit_only(self) -> None:
        model = self._visual_model().eval().requires_grad_(False)
        latent = torch.randn(16, 2)
        target = latent + torch.tensor([0.3, -0.2])
        alignment = fit_postfreeze_affine_audit_alignment(model, latent, target)
        self.assertLess(alignment.normalized_fit_error, 1e-5)
        self.assertFalse(alignment.weight.requires_grad)

        immediate = torch.tensor([0.10, 1.00]).reshape(1, 2, 1).expand(16, 2, 1).clone()
        delayed = torch.tensor([0.20, 0.70]).reshape(1, 2, 1).expand(16, 2, 1).clone()
        result = audit_force_port_signature(
            model,
            alignment,
            {1: immediate, 4: delayed},
            configuration_indices=(0,),
            momentum_indices=(1,),
            thresholds=ForcePortThresholds(minimum_samples=16),
        )
        self.assertTrue(result.auditable, result.failures)
        self.assertTrue(result.passed, result.to_dict())
        self.assertLessEqual(
            float(result.metrics["immediate_configuration_to_momentum_ratio"]), 0.35
        )
        self.assertGreaterEqual(
            float(result.metrics["configuration_horizon_4_to_1_ratio"]), 1.5
        )

    def test_force_port_locality_uses_only_detached_pre_event_mask(self) -> None:
        model = self._visual_model().eval().requires_grad_(False)
        latent = torch.randn(16, 2)
        alignment = fit_postfreeze_affine_audit_alignment(model, latent, latent)
        immediate = torch.tensor([1.0, 0.1]).reshape(1, 2, 1).expand(16, 2, 1).clone()
        delayed = torch.tensor([0.7, 0.2]).reshape(1, 2, 1).expand(16, 2, 1).clone()
        result = audit_force_port_signature(
            model,
            alignment,
            {1: immediate, 4: delayed},
            configuration_indices=(1,),
            momentum_indices=(0,),
            actuated_momentum_indices=(0,),
            nonactuated_momentum_indices=(1,),
            locality_sample_mask=torch.ones(16, dtype=torch.bool),
            require_locality=True,
            thresholds=ForcePortThresholds(minimum_samples=16),
        )
        # The locality grouping above is invalid by construction because the
        # nonactuated group is not a momentum subset.  It must be reported as
        # unauditable rather than silently omitted.
        self.assertFalse(result.auditable)
        self.assertFalse(result.passed)
        self.assertIn("subsets of momentum_indices", " ".join(result.failures))

    def test_gradient_phase_audit_api_has_no_external_command_parameter(self) -> None:
        for function in (audit_gate_3, audit_gate_4):
            names = tuple(inspect.signature(function).parameters)
            for forbidden in ("action", "control", "force", "torque"):
                self.assertTrue(
                    all(forbidden not in name.lower() for name in names),
                    (function.__name__, names),
                )


if __name__ == "__main__":
    unittest.main()
