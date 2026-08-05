from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import unittest

import numpy as np
import torch

from blocket_league.direct_physical_evaluation import (
    BLOCKET_CONTACT_MAX_PIXEL_DISPLACEMENT,
    BLOCKET_CONTACT_MIN_PIXEL_DISPLACEMENT,
    BLOCKET_CONTACT_ORACLE_NATIVE_THRUST,
    BLOCKET_CONTACT_TARGET_TASK_NAME,
    BLOCKET_CONTACT_TARGET_TASK_SHA256,
    ControlResult,
    SYSTEMS,
    builtin_pixel_plant,
    evaluate_closed_loop_controllers,
    fixed_interfaces,
    make_puck_only_pixel_objective,
    make_builtin_control_episodes,
    pixel_target_error,
    puck_only_pixel_cost,
    registered_linear_interface_command_bound,
)


EPISODE_SEED = 151_910_737


def _pixel_centroid(classes: torch.Tensor, values: tuple[int, ...]) -> np.ndarray:
    mask = torch.zeros_like(classes, dtype=torch.bool)
    for value in values:
        mask |= classes.eq(value)
    indices = torch.nonzero(mask, as_tuple=False).float()
    if indices.numel() == 0:
        raise AssertionError(f"missing categorical values {values}")
    yx = indices.mean(dim=0).cpu().numpy() + 0.5
    return np.asarray((yx[1], yx[0]), dtype=np.float64)


def _episode_set_sha256(episodes: list[object]) -> str:
    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(episode.identifier.encode("utf-8"))
        for tensor in (episode.context, episode.target_pixels):
            value = tensor.detach().cpu().contiguous()
            digest.update(str((tuple(value.shape), value.dtype)).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _categorical_logits(classes: torch.Tensor, palette_size: int = 9) -> torch.Tensor:
    logits = torch.full(
        (1, palette_size, *classes.shape),
        -8.0,
        dtype=torch.float32,
    )
    return logits.scatter_(1, classes.long()[None, None], 8.0)


def _relabel_every_non_puck_pixel(classes: torch.Tensor) -> torch.Tensor:
    changed = classes.clone()
    puck = classes.eq(7) | classes.eq(8)
    # Permute all seven non-puck categorical identities.  This moves/removes
    # player, wall, and background labels without touching puck support.
    changed[~puck] = (changed[~puck].long() + 3).remainder(7).to(changed.dtype)
    return changed


class BlocketContactControlTaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.system = SYSTEMS["blocket"]
        cls.episodes = make_builtin_control_episodes(
            cls.system,
            history_frames=8,
            count=64,
            seed=EPISODE_SEED,
            image_size=64,
        )

    def test_all_64_targets_are_puck_targets_with_registered_displacement(self) -> None:
        self.assertEqual(self.system.controlled_pixel_values, (7, 8))
        self.assertEqual(len(self.episodes), 64)
        for episode in self.episodes:
            with self.subTest(identifier=episode.identifier):
                self.assertEqual(episode.context.dtype, torch.uint8)
                self.assertEqual(episode.target_pixels.dtype, torch.uint8)
                self.assertTrue(
                    bool(
                        episode.context[-1].eq(7).any()
                        or episode.context[-1].eq(8).any()
                    )
                )
                self.assertTrue(
                    bool(
                        episode.target_pixels.eq(7).any()
                        or episode.target_pixels.eq(8).any()
                    )
                )
                displacement = pixel_target_error(
                    self.system, episode.context[-1], episode.target_pixels
                )
                self.assertGreaterEqual(
                    displacement, BLOCKET_CONTACT_MIN_PIXEL_DISPLACEMENT
                )
                self.assertLessEqual(
                    displacement, BLOCKET_CONTACT_MAX_PIXEL_DISPLACEMENT
                )

    def test_planning_cost_is_exactly_invariant_to_player_and_background(self) -> None:
        episode = self.episodes[0]
        source = episode.context[-1]
        target = episode.target_pixels
        objective = make_puck_only_pixel_objective(source, target, (7, 8))
        changed_source = _relabel_every_non_puck_pixel(source)
        changed_target = _relabel_every_non_puck_pixel(target)
        changed_objective = make_puck_only_pixel_objective(
            changed_source, changed_target, (7, 8)
        )
        self.assertTrue(
            torch.equal(objective.source_support, changed_objective.source_support)
        )
        self.assertTrue(
            torch.equal(objective.target_support, changed_objective.target_support)
        )

        original_prediction = _categorical_logits(target)
        changed_non_puck_prediction = _categorical_logits(changed_target)
        original_cost = puck_only_pixel_cost(original_prediction, objective)
        changed_target_cost = puck_only_pixel_cost(
            original_prediction, changed_objective
        )
        changed_prediction_cost = puck_only_pixel_cost(
            changed_non_puck_prediction, objective
        )
        torch.testing.assert_close(
            original_cost, changed_target_cost, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            original_cost, changed_prediction_cost, atol=0.0, rtol=0.0
        )

    def test_planning_cost_detects_missing_and_misplaced_puck(self) -> None:
        episode = self.episodes[0]
        source = episode.context[-1]
        target = episode.target_pixels
        objective = make_puck_only_pixel_objective(source, target, (7, 8))
        correct = target.clone()
        missing = target.clone()
        missing[objective.target_support] = 0
        misplaced = missing.clone()
        shifted = torch.roll(objective.target_support, shifts=(21, 17), dims=(0, 1))
        misplaced[shifted] = 7
        correct_cost = puck_only_pixel_cost(_categorical_logits(correct), objective)
        missing_cost = puck_only_pixel_cost(_categorical_logits(missing), objective)
        misplaced_cost = puck_only_pixel_cost(
            _categorical_logits(misplaced), objective
        )
        self.assertGreater(float(missing_cost), float(correct_cost))
        self.assertGreater(float(misplaced_cost), float(correct_cost))

    def test_puck_objective_api_has_no_physical_or_simulator_channel(self) -> None:
        forbidden = {
            "action",
            "command",
            "coordinate",
            "environment",
            "physical_state",
            "player_position",
            "puck_position",
            "simulator_state",
            "entity_mask",
        }
        for function in (make_puck_only_pixel_objective, puck_only_pixel_cost):
            with self.subTest(function=function.__name__):
                self.assertTrue(
                    set(inspect.signature(function).parameters).isdisjoint(forbidden)
                )
        controller_source = inspect.getsource(evaluate_closed_loop_controllers)
        self.assertIn("make_puck_only_pixel_objective", controller_source)
        self.assertEqual(
            controller_source.count("pixel_objective=pixel_objective"), 2
        )

    def test_initial_context_is_exact_render_and_player_is_staged_behind_puck(self) -> None:
        plant = builtin_pixel_plant(self.system)
        minimum = (
            self.episodes[0].environment.config.player_radius
            + self.episodes[0].environment.config.puck_radius
        )
        for episode in self.episodes:
            state = episode.environment.state
            with self.subTest(identifier=episode.identifier):
                self.assertTrue(
                    torch.equal(
                        episode.context[-1],
                        plant.current_pixels(episode.environment),
                    )
                )
                np.testing.assert_array_equal(state.player_velocity, np.zeros(2))
                np.testing.assert_array_equal(state.puck_velocity, np.zeros(2))
                start_puck = _pixel_centroid(episode.context[-1], (7, 8))
                target_puck = _pixel_centroid(episode.target_pixels, (7, 8))
                direction = target_puck - start_puck
                direction /= np.linalg.norm(direction)
                puck_from_player = state.puck_position - state.player_position
                surface_gap = float(np.linalg.norm(puck_from_player) - minimum)
                self.assertGreater(float(np.dot(puck_from_player, direction)), 0.0)
                lateral_offset = abs(
                    float(
                        direction[0] * puck_from_player[1]
                        - direction[1] * puck_from_player[0]
                    )
                )
                self.assertLess(lateral_offset, 0.01)
                self.assertGreaterEqual(surface_gap, 0.0119)
                self.assertLessEqual(surface_gap, 0.0241)

                # Both target discs are present and the rendered target player
                # remains behind the puck along the realized motion direction.
                target_player = _pixel_centroid(episode.target_pixels, (5, 6))
                self.assertGreater(float(np.dot(target_puck - target_player, direction)), 0.0)

    def test_episode_generation_and_task_identity_are_deterministic_and_sealed(self) -> None:
        repeated = make_builtin_control_episodes(
            self.system,
            history_frames=8,
            count=64,
            seed=EPISODE_SEED,
            image_size=64,
        )
        self.assertEqual(
            tuple(episode.identifier for episode in self.episodes),
            tuple(episode.identifier for episode in repeated),
        )
        self.assertEqual(_episode_set_sha256(self.episodes), _episode_set_sha256(repeated))
        for first, second in zip(self.episodes, repeated, strict=True):
            self.assertTrue(torch.equal(first.context, second.context))
            self.assertTrue(torch.equal(first.target_pixels, second.target_pixels))

        traces = {
            "coast": tuple(((0.0, 0.0),) for _ in range(64)),
        }
        result = ControlResult(
            errors={"coast": tuple(1.0 for _ in range(64))},
            interface_name="native",
            episodes=64,
            control_steps=1,
            planner_budget={"kind": "identity-test"},
            episode_identifiers=tuple(
                episode.identifier for episode in self.episodes
            ),
            interface_command_traces=traces,
            planner_seed_schedule_sha256="a" * 64,
        )
        self.assertEqual(
            result.target_task,
            {
                "name": BLOCKET_CONTACT_TARGET_TASK_NAME,
                "sha256": BLOCKET_CONTACT_TARGET_TASK_SHA256,
            },
        )
        self.assertEqual(result.as_dict()["targetTask"], result.target_task)
        mixed_identifiers = (
            "foreign-task",
            *result.episode_identifiers[1:],
        )
        with self.assertRaisesRegex(ValueError, "mixes contact-task"):
            replace(result, episode_identifiers=mixed_identifiers)

    def test_coast_cannot_move_the_stationary_puck(self) -> None:
        plant = builtin_pixel_plant(self.system)
        interface = fixed_interfaces(self.system)["native"]
        zero = np.zeros(2, dtype=np.float32)
        for episode in self.episodes:
            environment = plant.clone_environment(episode.environment)
            initial = environment.state.puck_position.copy()
            for _ in range(self.system.control_steps):
                plant.step_interface(environment, interface, zero)
            with self.subTest(identifier=episode.identifier):
                np.testing.assert_allclose(
                    environment.state.puck_position, initial, atol=0.0, rtol=0.0
                )
                self.assertLess(
                    pixel_target_error(
                        self.system,
                        episode.context[-1],
                        plant.current_pixels(environment),
                    ),
                    1e-8,
                )

    def test_source_only_right_thrust_oracle_contacts_and_realizes_episode_zero(self) -> None:
        episode = self.episodes[0]
        plant = builtin_pixel_plant(self.system)
        interface = fixed_interfaces(self.system)["native"]
        environment = plant.clone_environment(episode.environment)
        initial_puck = environment.state.puck_position.copy()
        command = np.asarray(
            (BLOCKET_CONTACT_ORACLE_NATIVE_THRUST, 0.0), dtype=np.float32
        )
        touched = False
        best_target_error = float("inf")
        for _ in range(self.system.control_steps):
            plant.step_interface(environment, interface, command)
            touched = touched or environment.state.last_event == "impact"
            best_target_error = min(
                best_target_error,
                pixel_target_error(
                    self.system,
                    plant.current_pixels(environment),
                    episode.target_pixels,
                ),
            )
        self.assertTrue(touched)
        self.assertGreater(
            float(np.linalg.norm(environment.state.puck_position - initial_puck)),
            BLOCKET_CONTACT_MIN_PIXEL_DISPLACEMENT,
        )
        self.assertLess(best_target_error, 1e-8)

    def test_same_contact_task_is_admissible_through_native_and_unseen_interfaces(self) -> None:
        interfaces = fixed_interfaces(self.system)
        bound = registered_linear_interface_command_bound(self.system)
        for episode in self.episodes:
            direction = (
                episode.environment.state.puck_position
                - episode.environment.state.player_position
            )
            direction /= np.linalg.norm(direction)
            native_effort = BLOCKET_CONTACT_ORACLE_NATIVE_THRUST * direction
            for name, interface in interfaces.items():
                command = np.linalg.solve(interface.matrix(), native_effort)
                with self.subTest(identifier=episode.identifier, interface=name):
                    self.assertLessEqual(float(np.abs(command).max()), bound + 1e-12)
                    np.testing.assert_allclose(
                        interface.matrix() @ command,
                        native_effort,
                        atol=1e-12,
                        rtol=1e-12,
                    )

    def test_controller_source_never_reads_simulator_state(self) -> None:
        source = inspect.getsource(evaluate_closed_loop_controllers)
        self.assertNotIn("episode.environment.state", source)
        self.assertNotIn("environment.state", source)


if __name__ == "__main__":
    unittest.main()
