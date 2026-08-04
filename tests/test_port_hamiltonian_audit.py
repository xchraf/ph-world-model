from __future__ import annotations

import unittest

import torch

from blocket_league.port_hamiltonian_audit import (
    _conformal_symplectic_defect,
    _energy_metrics,
    _fit_temporally_aligned_ridge,
    _free_horizon_analysis,
    _horizon_endpoints,
    _horizon_masks,
    _ridge_predict,
    canonical_state,
    fit_structured_free_map,
    predict_structured_free_map,
    transition_regimes,
)


class PortHamiltonianAuditTests(unittest.TestCase):
    def test_canonical_state_uses_momenta_not_raw_velocities(self) -> None:
        state = torch.tensor(
            [[0.2, 0.3, 0.4, -0.5, 0.7, 0.8, -0.2, 0.1, 0.0, 0.0]]
        )
        result = canonical_state(state, player_mass=2.0, puck_mass=0.5)
        expected = torch.tensor([[0.2, 0.3, 0.7, 0.8, 0.8, -1.0, -0.1, 0.05]])
        self.assertTrue(torch.allclose(result, expected))

    def test_transition_regimes_separate_goal_entry_pause_and_kickoff(self) -> None:
        states_t = torch.zeros(6, 10)
        states_tp1 = torch.zeros(6, 10)
        events = torch.tensor([0, 2, 3, 4, 4, 5])
        states_tp1[3, 8] = 1
        states_t[4, 9] = 3
        states_tp1[4, 9] = 2
        regimes = transition_regimes(states_t, states_tp1, events)
        self.assertEqual(regimes["free"].tolist(), [True, False, False, False, False, False])
        self.assertEqual(regimes["disc_impact"].tolist(), [False, True, False, False, False, False])
        self.assertEqual(regimes["wall"].tolist(), [False, False, True, False, False, False])
        self.assertEqual(regimes["goal_entry"].tolist(), [False, False, False, True, False, False])
        self.assertEqual(regimes["goal_pause"].tolist(), [False, False, False, False, True, False])
        self.assertEqual(regimes["kickoff"].tolist(), [False, False, False, False, False, True])

    def test_structured_free_map_recovers_tied_particle_coefficients(self) -> None:
        generator = torch.Generator().manual_seed(19)
        z_t = torch.randn(256, 8, generator=generator)
        parameters = {
            "positionGain": [0.027, 0.049],
            "momentumDecay": [0.991, 0.994],
        }
        z_tp1 = predict_structured_free_map(z_t, parameters)
        recovered = fit_structured_free_map(z_t, z_tp1, dissipative=True)
        self.assertAlmostEqual(recovered["positionGain"][0], 0.027, places=6)
        self.assertAlmostEqual(recovered["positionGain"][1], 0.049, places=6)
        self.assertAlmostEqual(recovered["momentumDecay"][0], 0.991, places=6)
        self.assertAlmostEqual(recovered["momentumDecay"][1], 0.994, places=6)

    def test_port_hamiltonian_fit_cannot_create_unforced_energy(self) -> None:
        z_t = torch.randn(64, 8, generator=torch.Generator().manual_seed(21))
        z_tp1 = z_t.clone()
        z_tp1[:, 4:] *= 1.2
        recovered = fit_structured_free_map(z_t, z_tp1, dissipative=True)
        self.assertEqual(recovered["momentumDecay"], [1.0, 1.0])

    def test_structured_map_has_zero_expected_conformal_symplectic_defect(self) -> None:
        parameters = {
            "positionGain": [0.03, 0.05],
            "momentumDecay": [0.98, 0.99],
        }
        weight = torch.zeros(9, 8)
        weight[:8] = torch.eye(8)
        for entity in range(2):
            q_slice = slice(entity * 2, entity * 2 + 2)
            p_slice = slice(4 + entity * 2, 4 + entity * 2 + 2)
            weight[p_slice, q_slice] = torch.eye(2) * parameters["positionGain"][entity]
            weight[p_slice, p_slice] = torch.eye(2) * parameters["momentumDecay"][entity]
        self.assertLess(_conformal_symplectic_defect(weight, parameters), 1e-6)

    def test_exact_structured_transition_closes_discrete_energy_balance(self) -> None:
        generator = torch.Generator().manual_seed(23)
        z_t = torch.randn(128, 8, generator=generator)
        parameters = {
            "positionGain": [0.027, 0.049],
            "momentumDecay": [0.991, 0.994],
        }
        z_tp1 = predict_structured_free_map(z_t, parameters)
        metrics = _energy_metrics(z_t, z_tp1, parameters, dt=0.05)
        self.assertLess(metrics["normalizedBalanceRmse"], 1e-5)
        self.assertEqual(metrics["passivityViolationRate"], 0.0)

    def test_temporally_aligned_decoder_is_one_linear_map_for_both_endpoints(self) -> None:
        generator = torch.Generator().manual_seed(31)
        z_t = torch.randn(128, 8, generator=generator)
        parameters = {
            "positionGain": [0.027, 0.049],
            "momentumDecay": [0.991, 0.994],
        }
        z_tp1 = predict_structured_free_map(z_t, parameters)
        mixing = torch.randn(8, 12, generator=generator)
        features_t = z_t @ mixing
        features_tp1 = z_tp1 @ mixing
        fit = _fit_temporally_aligned_ridge(
            features_t,
            features_tp1,
            z_t,
            z_tp1,
            ridge=1e-4,
            delta_weight=1.0,
        )
        predicted_t = _ridge_predict(fit, features_t)
        predicted_tp1 = _ridge_predict(fit, features_tp1)
        self.assertLess((predicted_t - z_t).square().mean().sqrt(), 1e-3)
        self.assertLess((predicted_tp1 - z_tp1).square().mean().sqrt(), 1e-3)

    def test_horizon_pairing_never_crosses_trajectory_boundaries(self) -> None:
        values_t = torch.tensor([0, 1, 2, 10, 11, 12], dtype=torch.float32)[:, None]
        values_tp1 = torch.tensor([1, 2, 3, 11, 12, 13], dtype=torch.float32)[:, None]
        start, end = _horizon_endpoints(
            values_t,
            values_tp1,
            trajectories=2,
            transitions_per_trajectory=3,
            horizon=2,
        )
        self.assertEqual(start[:, 0].tolist(), [0.0, 1.0, 10.0, 11.0])
        self.assertEqual(end[:, 0].tolist(), [2.0, 3.0, 12.0, 13.0])
        free = torch.tensor([True, True, False, True, False, True])
        fit, test = _horizon_masks(
            free,
            trajectories=2,
            transitions_per_trajectory=3,
            fit_trajectories=1,
            horizon=2,
        )
        self.assertEqual(fit.tolist(), [True, False, False, False])
        self.assertEqual(test.tolist(), [False, False, False, False])

    def test_negative_controls_distinguish_paired_forward_dissipation(self) -> None:
        generator = torch.Generator().manual_seed(41)
        z_t = torch.randn(256, 8, generator=generator)
        parameters = {
            "positionGain": [0.027, 0.049],
            "momentumDecay": [0.97, 0.98],
        }
        z_tp1 = predict_structured_free_map(z_t, parameters)
        fit = torch.arange(256) < 192
        test = ~fit
        result = _free_horizon_analysis(
            z_t,
            z_tp1,
            fit,
            test,
            dt=0.05,
            ridge=1e-5,
            device=torch.device("cpu"),
            control_seed=43,
        )
        self.assertLess(result["models"]["portHamiltonian"]["deltaNrmse"], 1e-5)
        self.assertGreater(result["pairingControl"]["portHamiltonian"]["deltaNrmse"], 0.9)
        reverse = result["reverseTimeControl"]["models"]
        self.assertGreater(
            reverse["portHamiltonian"]["deltaNrmse"],
            reverse["affine"]["deltaNrmse"],
        )


if __name__ == "__main__":
    unittest.main()
