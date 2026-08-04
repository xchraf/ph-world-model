from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from blocket_league.neural_port_hamiltonian import (
    NeuralPortHamiltonian,
    NeuralPortHamiltonianConfig,
)


SUITES = ("policy", "diagonalOod", "reversalOod")
BRANCHES = ("neuralPortHamiltonian", "neuralOdeControl")


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def _collect(
    runs: list[dict[str, Any]],
    getter: Callable[[dict[str, Any]], float],
) -> dict[str, float]:
    return _mean_std([float(getter(run)) for run in runs])


def _branch_aggregate(
    runs: list[dict[str, Any]], suite: str, branch: str
) -> dict[str, Any]:
    return {
        "stateReadout": {
            metric: _collect(
                runs,
                lambda run, m=metric: run["evaluation"][suite][branch][
                    "stateReadout"
                ][m],
            )
            for metric in ("qR2", "pR2", "modeR2")
        },
        "h8": {
            metric: _collect(
                runs,
                lambda run, m=metric: run["evaluation"][suite][branch][
                    "rolloutByHorizon"
                ]["8"][m],
            )
            for metric in ("deltaNrmse", "qR2", "pR2")
        }
        | {
            "pixels": {
                metric: _collect(
                    runs,
                    lambda run, m=metric: run["evaluation"][suite][branch][
                        "rolloutByHorizon"
                    ]["8"]["pixels"][m],
                )
                for metric in ("accuracy", "playerIou", "puckIou")
            }
        },
        "negativeControls": {
            metric: _collect(
                runs,
                lambda run, m=metric: run["evaluation"][suite][branch][
                    "negativeControls"
                ][m],
            )
            for metric in (
                "shuffledInitialStateH8",
                "shuffledActionsH8",
                "zeroActionsH8",
            )
        },
        "eventBalancedAccuracy": _collect(
            runs,
            lambda run: run["evaluation"][suite][branch][
                "eventBalancedAccuracy"
            ],
        ),
    }


def _structured_aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    path_groups = {
        "powerBalance": ("maxAbsDefect", "rmsDefect", "minimumDissipation"),
        "hamiltonian": (
            "kineticEnergyAffineR2",
            "normalizedGradientRms",
            "gradientScaleToKinetic",
            "kineticGradientCosine",
            "stateVariation",
        ),
        "interconnection": (
            "canonicalCosine",
            "stateVariation",
            "skewDefect",
            "jacobiRms",
            "jacobiMaxAbs",
        ),
        "resistance": (
            "physicalDragCosine",
            "stateVariation",
            "minimumEigenvalue",
        ),
        "port": (
            "physicalIncidenceCosine",
            "stateVariation",
            "positionRowFraction",
            "puckMomentumFraction",
        ),
        "zeroInputDiscreteEnergy": (
            "increaseFraction",
            "meanNetChange",
            "maximumStepIncrease",
        ),
    }
    return {
        group: {
            metric: _collect(
                runs,
                lambda run, g=group, m=metric: run["evaluation"]["policy"][
                    "neuralPortHamiltonian"
                ]["structure"][g][m],
            )
            for metric in metrics
        }
        for group, metrics in path_groups.items()
    }


def _load_core(checkpoint_path: Path) -> NeuralPortHamiltonian:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    branch_state = payload["branches"]["neuralPortHamiltonian"]
    core_config = NeuralPortHamiltonianConfig(
        state_size=8,
        input_size=2,
        hidden_size=int(config["hidden_size"]),
        hidden_layers=int(config["hidden_layers"]),
        dt=0.05,
        integration_method=config["integration_method"],
        integration_substeps=int(config["integration_substeps"]),
        resistance_floor=float(config["resistance_floor"]),
    )
    core = NeuralPortHamiltonian(
        core_config,
        state_mean=branch_state["state_mean"][:8],
        state_scale=branch_state["state_scale"][:8],
    )
    prefix = "core."
    core.load_state_dict(
        {
            name.removeprefix(prefix): value
            for name, value in branch_state.items()
            if name.startswith(prefix)
        }
    )
    return core.eval()


def _reference_probe() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(711_931)
    states = torch.empty(256, 8)
    states[:, :4] = 0.12 + 0.76 * torch.rand(256, 4, generator=generator)
    states[:, 4:6] = -2.1 + 4.2 * torch.rand(256, 2, generator=generator)
    states[:, 6:8] = -0.9 + 1.8 * torch.rand(256, 2, generator=generator)
    controls = torch.randn(256, 2, generator=generator)
    controls = controls / controls.norm(dim=-1, keepdim=True).clamp_min(1.0)
    return states, controls


def _probe_core(
    core: NeuralPortHamiltonian,
    states: torch.Tensor,
    controls: torch.Tensor,
) -> dict[str, torch.Tensor]:
    energy, gradient, interconnection, resistance, port = core.components(
        states, create_graph=False
    )
    vector_field = core.vector_field(states, controls)
    return {
        "H": energy.detach(),
        "gradH": gradient.detach(),
        "J": interconnection.detach(),
        "R": resistance.detach(),
        "B": port.detach(),
        "vectorField": vector_field.detach(),
    }


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(first.flatten()[None], second.flatten()[None]).item()
    )


def _relative_rmse(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(
        (first - second).square().mean().sqrt()
        / first.square().mean().sqrt().clamp_min(1e-12)
    )


def _functional_agreement(checkpoints: list[Path]) -> dict[str, Any]:
    if len(checkpoints) < 2:
        return {"pairs": 0}
    states, controls = _reference_probe()
    probes = [_probe_core(_load_core(path), states, controls) for path in checkpoints]
    pair_metrics: dict[str, list[float]] = {
        "hamiltonianAffineR2": [],
        "alignedGradientCosine": [],
        "interconnectionCosine": [],
        "interconnectionRelativeRmse": [],
        "resistanceCosine": [],
        "resistanceRelativeRmse": [],
        "portCosine": [],
        "portRelativeRmse": [],
        "vectorFieldCosine": [],
        "vectorFieldRelativeRmse": [],
    }
    for first, second in itertools.combinations(probes, 2):
        design = torch.stack((second["H"], torch.ones_like(second["H"])), dim=-1)
        coefficients = torch.linalg.lstsq(design, first["H"][:, None]).solution[:, 0]
        aligned_h = design @ coefficients
        residual = (aligned_h - first["H"]).square().sum()
        total = (first["H"] - first["H"].mean()).square().sum().clamp_min(1e-12)
        pair_metrics["hamiltonianAffineR2"].append(float(1.0 - residual / total))
        pair_metrics["alignedGradientCosine"].append(
            _cosine(first["gradH"], coefficients[0] * second["gradH"])
        )
        for key, prefix in (
            ("J", "interconnection"),
            ("R", "resistance"),
            ("B", "port"),
            ("vectorField", "vectorField"),
        ):
            pair_metrics[f"{prefix}Cosine"].append(_cosine(first[key], second[key]))
            pair_metrics[f"{prefix}RelativeRmse"].append(
                _relative_rmse(first[key], second[key])
            )
    return {
        "pairs": len(next(iter(pair_metrics.values()))),
        "referenceStates": states.shape[0],
        **{name: _mean_std(values) for name, values in pair_metrics.items()},
    }


def aggregate(
    runs: list[dict[str, Any]],
    summaries: list[Path],
    checkpoints: list[Path],
) -> dict[str, Any]:
    suites = {
        suite: {
            "rendererOracle": {
                metric: _collect(
                    runs,
                    lambda run, s=suite, m=metric: run["evaluation"][s][
                        "rendererOracle"
                    ][m],
                )
                for metric in ("accuracy", "playerIou", "puckIou")
            },
            "branches": {
                branch: _branch_aggregate(runs, suite, branch)
                for branch in BRANCHES
            },
            "pairedH8DeltaNrmsePhMinusControl": _collect(
                runs,
                lambda run, s=suite: run["evaluation"][s][
                    "neuralPortHamiltonian"
                ]["rolloutByHorizon"]["8"]["deltaNrmse"]
                - run["evaluation"][s]["neuralOdeControl"][
                    "rolloutByHorizon"
                ]["8"]["deltaNrmse"],
            ),
        }
        for suite in SUITES
    }
    structure = _structured_aggregate(runs)
    functional = _functional_agreement(checkpoints)
    timing = {
        metric: _collect(runs, lambda run, m=metric: run["timing"][m])
        for metric in (
            "collectionSeconds",
            "dynamicsSeconds",
            "evaluationSeconds",
            "totalSeconds",
        )
    }
    parameter_changes = {
        name: _collect(runs, lambda run, n=name: run["parameterChangeNorm"][n])
        for name in ("H", "J", "R", "B")
    }
    ph_policy = suites["policy"]["branches"]["neuralPortHamiltonian"]
    decisions = {
        "capacityMatched": runs[0]["capacity"]["relativeCoreGap"] < 0.01,
        "predictiveParity": abs(
            suites["policy"]["pairedH8DeltaNrmsePhMinusControl"]["mean"]
        ) <= 0.02,
        "readoutGate": (
            ph_policy["stateReadout"]["qR2"]["mean"] > 0.90
            and ph_policy["stateReadout"]["pR2"]["mean"] > 0.80
        ),
        "continuousPowerIdentity": structure["powerBalance"]["maxAbsDefect"][
            "maximum"
        ] < 1e-5,
        "skewSymmetry": structure["interconnection"]["skewDefect"]["maximum"] < 1e-7,
        "positiveResistance": structure["resistance"]["minimumEigenvalue"][
            "minimum"
        ] > -1e-6,
        "actionCausalSignal": all(
            suites[suite]["branches"]["neuralPortHamiltonian"][
                "negativeControls"
            ]["shuffledActionsH8"]["mean"]
            > suites[suite]["branches"]["neuralPortHamiltonian"]["h8"][
                "deltaNrmse"
            ]["mean"]
            for suite in SUITES
        ),
        "functionalVectorFieldAgreement": (
            functional.get("pairs", 0) > 0
            and functional["vectorFieldCosine"]["mean"] > 0.95
        ),
        "separateFunctionAgreement": (
            functional.get("pairs", 0) > 0
            and functional["hamiltonianAffineR2"]["mean"] > 0.90
            and functional["interconnectionCosine"]["mean"] > 0.90
            and functional["resistanceCosine"]["mean"] > 0.90
            and functional["portCosine"]["mean"] > 0.90
        ),
    }
    return {
        "kind": "state_dependent_neural_port_hamiltonian_aggregate",
        "version": 1,
        "runs": len(runs),
        "summarySources": [str(path) for path in summaries],
        "checkpointSources": [str(path) for path in checkpoints],
        "baseCheckpointStep": runs[0]["baseCheckpointStep"],
        "config": runs[0]["config"],
        "capacity": runs[0]["capacity"],
        "timing": timing,
        "parameterChangeNorm": parameter_changes,
        "suites": suites,
        "structure": structure,
        "functionalAgreement": functional,
        "decisions": decisions,
    }


def _fmt(metric: dict[str, float], digits: int = 3) -> str:
    return f"{metric['mean']:.{digits}f} ± {metric['std']:.{digits}f}"


def markdown_summary(result: dict[str, Any]) -> str:
    lines = [
        "# Experiment C — state-dependent neural port-Hamiltonian dynamics",
        "",
        f"Independent runs: {result['runs']}. Values are mean ± sample standard deviation.",
        "",
        "## Runtime",
        "",
        "| Phase | Seconds per seed |",
        "| --- | ---: |",
        f"| Feature collection | {_fmt(result['timing']['collectionSeconds'], 1)} |",
        f"| Paired dynamics | {_fmt(result['timing']['dynamicsSeconds'], 1)} |",
        f"| Evaluation | {_fmt(result['timing']['evaluationSeconds'], 1)} |",
        f"| Total | {_fmt(result['timing']['totalSeconds'], 1)} |",
        "",
        "## Predictive comparison",
        "",
        "| Suite | pH H8 delta NRMSE | Neural ODE | Paired gap |",
        "| --- | ---: | ---: | ---: |",
    ]
    for suite in SUITES:
        value = result["suites"][suite]
        lines.append(
            f"| {suite} | "
            f"{_fmt(value['branches']['neuralPortHamiltonian']['h8']['deltaNrmse'])} | "
            f"{_fmt(value['branches']['neuralOdeControl']['h8']['deltaNrmse'])} | "
            f"{_fmt(value['pairedH8DeltaNrmsePhMinusControl'], 4)} |"
        )
    structure = result["structure"]
    lines.extend(
        [
            "",
            "## Learned structure on policy states",
            "",
            "| Audit | Value |",
            "| --- | ---: |",
            f"| Max power defect | {_fmt(structure['powerBalance']['maxAbsDefect'], 8)} |",
            f"| Kinetic-energy affine R² | {_fmt(structure['hamiltonian']['kineticEnergyAffineR2'])} |",
            f"| Canonical J cosine | {_fmt(structure['interconnection']['canonicalCosine'])} |",
            f"| Jacobi RMS | {_fmt(structure['interconnection']['jacobiRms'], 6)} |",
            f"| Physical R cosine | {_fmt(structure['resistance']['physicalDragCosine'])} |",
            f"| Physical B cosine | {_fmt(structure['port']['physicalIncidenceCosine'])} |",
            "",
            "## Across-seed functional agreement",
            "",
        ]
    )
    functional = result["functionalAgreement"]
    if functional.get("pairs", 0):
        lines.extend(
            [
                "| Function | Agreement |",
                "| --- | ---: |",
                f"| H affine R² | {_fmt(functional['hamiltonianAffineR2'])} |",
                f"| aligned grad H cosine | {_fmt(functional['alignedGradientCosine'])} |",
                f"| J cosine | {_fmt(functional['interconnectionCosine'])} |",
                f"| R cosine | {_fmt(functional['resistanceCosine'])} |",
                f"| B cosine | {_fmt(functional['portCosine'])} |",
                f"| vector-field cosine | {_fmt(functional['vectorFieldCosine'])} |",
            ]
        )
    else:
        lines.append("At least two checkpoints are required.")
    lines.extend(["", "## Decision gates", ""])
    for name, passed in result["decisions"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--checkpoints", nargs="*", type=Path, default=[])
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    if any(
        run.get("kind") != "state_dependent_neural_port_hamiltonian"
        for run in runs
    ):
        raise ValueError("all inputs must be neural pH summaries")
    if len({run["config"]["seed"] for run in runs}) != len(runs):
        raise ValueError("summary seeds must be independent")
    reference = {key: value for key, value in runs[0]["config"].items() if key != "seed"}
    if any(
        {key: value for key, value in run["config"].items() if key != "seed"}
        != reference
        for run in runs[1:]
    ):
        raise ValueError("summary configurations must match apart from seed")
    if args.checkpoints and len(args.checkpoints) != len(args.summaries):
        raise ValueError("provide one checkpoint per summary, or none")
    result = aggregate(runs, args.summaries, args.checkpoints)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.markdown.write_text(markdown_summary(result), encoding="utf-8")


if __name__ == "__main__":
    main()
