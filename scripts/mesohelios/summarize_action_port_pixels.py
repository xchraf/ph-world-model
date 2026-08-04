from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable


BRANCHES = ("portHamiltonian", "tangentMatchedControl")
HORIZONS = ("1", "2", "4", "8")
REGIMES = (
    "free",
    "disc_impact",
    "wall",
    "goal_entry",
    "goal_pause",
    "kickoff",
)


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _collect(
    runs: list[dict[str, Any]],
    getter: Callable[[dict[str, Any]], float],
) -> dict[str, float]:
    return _mean_std([float(getter(run)) for run in runs])


def aggregate(runs: list[dict[str, Any]], sources: list[Path]) -> dict[str, Any]:
    branches: dict[str, Any] = {}
    for branch in BRANCHES:
        branches[branch] = {
            "stateReadout": {
                metric: _collect(
                    runs,
                    lambda run, b=branch, m=metric: run["evaluation"][b][
                        "stateReadout"
                    ][m],
                )
                for metric in ("qR2", "pR2", "modeR2")
            },
            "horizons": {
                horizon: {
                    **{
                        metric: _collect(
                            runs,
                            lambda run, b=branch, h=horizon, m=metric: run[
                                "evaluation"
                            ][b]["rolloutByHorizon"][h][m],
                        )
                        for metric in ("deltaNrmse", "qR2", "pR2")
                    },
                    "pixels": {
                        metric: _collect(
                            runs,
                            lambda run, b=branch, h=horizon, m=metric: run[
                                "evaluation"
                            ][b]["rolloutByHorizon"][h]["pixels"][m],
                        )
                        for metric in ("accuracy", "playerIou", "puckIou")
                    },
                }
                for horizon in HORIZONS
            },
            "negativeControls": {
                metric: _collect(
                    runs,
                    lambda run, b=branch, m=metric: run["evaluation"][b][
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
                lambda run, b=branch: run["evaluation"][b][
                    "eventBalancedAccuracy"
                ],
            ),
            "core": {
                "positionGain": [
                    _collect(
                        runs,
                        lambda run, b=branch, i=index: run["evaluation"][b][
                            "core"
                        ]["parameters"]["positionGain"][i],
                    )
                    for index in range(2)
                ],
                "momentumDecay": [
                    _collect(
                        runs,
                        lambda run, b=branch, i=index: run["evaluation"][b][
                            "core"
                        ]["parameters"]["momentumDecay"][i],
                    )
                    for index in range(2)
                ],
                "effectiveMass": [
                    _collect(
                        runs,
                        lambda run, b=branch, i=index: run["evaluation"][b][
                            "core"
                        ]["parameters"]["effectiveMass"][i],
                    )
                    for index in range(2)
                ],
                "effectiveDrag": [
                    _collect(
                        runs,
                        lambda run, b=branch, i=index: run["evaluation"][b][
                            "core"
                        ]["parameters"]["effectiveDrag"][i],
                    )
                    for index in range(2)
                ],
                "actionPositionGain": _collect(
                    runs,
                    lambda run, b=branch: run["evaluation"][b]["core"][
                        "parameters"
                    ]["actionPositionGain"],
                ),
                "actionMomentumGain": _collect(
                    runs,
                    lambda run, b=branch: run["evaluation"][b]["core"][
                        "parameters"
                    ]["actionMomentumGain"],
                ),
                "withinDomainFraction": statistics.fmean(
                    float(
                        run["evaluation"][branch]["core"]["parameters"][
                            "withinPortHamiltonianDomain"
                        ]
                    )
                    for run in runs
                ),
                "stability": {
                    metric: _collect(
                        runs,
                        lambda run, b=branch, m=metric: run["evaluation"][b][
                            "core"
                        ]["stability"][m],
                    )
                    for metric in (
                        "energyGrowthFraction",
                        "finalToInitialEnergy",
                    )
                },
            },
            "actionInterventions": {
                metric: _collect(
                    runs,
                    lambda run, b=branch, m=metric: statistics.fmean(
                        item[m]
                        for item in run["evaluation"][b]["actionInterventions"]
                    ),
                )
                for metric in ("momentumCosine", "puckCrossTalkRmse")
            },
            "regimes": {
                regime: {
                    metric: _collect(
                        runs,
                        lambda run, b=branch, g=regime, m=metric: run[
                            "evaluation"
                        ][b]["regimes"][g][m],
                    )
                    for metric in ("samples", "deltaNrmse", "meanJumpNorm")
                }
                for regime in REGIMES
            },
        }

    oracle = {
        metric: _collect(
            runs,
            lambda run, m=metric: run["evaluation"]["rendererOracle"][m],
        )
        for metric in ("accuracy", "playerIou", "puckIou")
    }
    paired = {
        "h8DeltaNrmsePhMinusControl": _collect(
            runs,
            lambda run: run["evaluation"]["portHamiltonian"][
                "rolloutByHorizon"
            ]["8"]["deltaNrmse"]
            - run["evaluation"]["tangentMatchedControl"]["rolloutByHorizon"][
                "8"
            ]["deltaNrmse"],
        ),
        "h8PlayerIouPhMinusControl": _collect(
            runs,
            lambda run: run["evaluation"]["portHamiltonian"][
                "rolloutByHorizon"
            ]["8"]["pixels"]["playerIou"]
            - run["evaluation"]["tangentMatchedControl"]["rolloutByHorizon"][
                "8"
            ]["pixels"]["playerIou"],
        ),
        "h8PuckIouPhMinusControl": _collect(
            runs,
            lambda run: run["evaluation"]["portHamiltonian"][
                "rolloutByHorizon"
            ]["8"]["pixels"]["puckIou"]
            - run["evaluation"]["tangentMatchedControl"]["rolloutByHorizon"][
                "8"
            ]["pixels"]["puckIou"],
        ),
    }
    ph = branches["portHamiltonian"]
    true_h8 = ph["horizons"]["8"]["deltaNrmse"]["mean"]
    decisions = {
        "accuracyParity": abs(paired["h8DeltaNrmsePhMinusControl"]["mean"]) <= 0.02,
        "readoutGate": (
            ph["stateReadout"]["qR2"]["mean"] > 0.90
            and ph["stateReadout"]["pR2"]["mean"] > 0.80
        ),
        "passiveDomain": ph["core"]["withinDomainFraction"] == 1.0,
        "zeroActionEnergyGrowth": (
            ph["core"]["stability"]["energyGrowthFraction"]["mean"] == 0.0
        ),
        "actionCausalSignal": (
            ph["negativeControls"]["shuffledActionsH8"]["mean"] > true_h8
            and ph["negativeControls"]["zeroActionsH8"]["mean"] > true_h8
        ),
        "directionalPort": (
            ph["actionInterventions"]["momentumCosine"]["mean"] > 0.99
            and ph["actionInterventions"]["puckCrossTalkRmse"]["mean"] < 1e-6
        ),
        "rendererCeilingAdequate": (
            oracle["playerIou"]["mean"] > 0.75
            and oracle["puckIou"]["mean"] > 0.75
        ),
    }
    timing = {
        metric: _collect(runs, lambda run, m=metric: run["timing"][m])
        for metric in (
            "collectionSeconds",
            "dynamicsSeconds",
            "rendererSeconds",
            "evaluationSeconds",
            "totalSeconds",
        )
    }
    return {
        "kind": "action_port_pixel_reinjection_aggregate",
        "version": 1,
        "runs": len(runs),
        "sources": [str(path) for path in sources],
        "baseCheckpointStep": runs[0]["baseCheckpointStep"],
        "config": runs[0]["config"],
        "capacity": runs[0]["capacity"],
        "timing": timing,
        "rendererOracle": oracle,
        "branches": branches,
        "paired": paired,
        "decisions": decisions,
    }


def _fmt(metric: dict[str, float], digits: int = 3) -> str:
    return f"{metric['mean']:.{digits}f} ± {metric['std']:.{digits}f}"


def markdown_summary(result: dict[str, Any]) -> str:
    ph = result["branches"]["portHamiltonian"]
    control = result["branches"]["tangentMatchedControl"]
    oracle = result["rendererOracle"]
    lines = [
        "# Experiment B2/B3 — action port and pixel reinjection",
        "",
        f"Independent runs: {result['runs']}. Values are mean ± sample standard deviation.",
        "",
        "## Runtime",
        "",
        "| Phase | Seconds per seed |",
        "| --- | ---: |",
        f"| Feature collection | {_fmt(result['timing']['collectionSeconds'], 1)} |",
        f"| Paired dynamics | {_fmt(result['timing']['dynamicsSeconds'], 1)} |",
        f"| Shared renderer | {_fmt(result['timing']['rendererSeconds'], 1)} |",
        f"| Evaluation | {_fmt(result['timing']['evaluationSeconds'], 1)} |",
        f"| Internal total | {_fmt(result['timing']['totalSeconds'], 1)} |",
        "",
        "## State and action results",
        "",
        "| Metric | Port-Hamiltonian | Tangent-matched control |",
        "| --- | ---: | ---: |",
        "| State q R² | "
        f"{_fmt(ph['stateReadout']['qR2'])} | "
        f"{_fmt(control['stateReadout']['qR2'])} |",
        "| State p R² | "
        f"{_fmt(ph['stateReadout']['pR2'])} | "
        f"{_fmt(control['stateReadout']['pR2'])} |",
        "| H8 delta NRMSE | "
        f"{_fmt(ph['horizons']['8']['deltaNrmse'])} | "
        f"{_fmt(control['horizons']['8']['deltaNrmse'])} |",
        "| H8 shuffled actions | "
        f"{_fmt(ph['negativeControls']['shuffledActionsH8'])} | "
        f"{_fmt(control['negativeControls']['shuffledActionsH8'])} |",
        "| H8 zero actions | "
        f"{_fmt(ph['negativeControls']['zeroActionsH8'])} | "
        f"{_fmt(control['negativeControls']['zeroActionsH8'])} |",
        "| 64-step energy growth | "
        f"{_fmt(ph['core']['stability']['energyGrowthFraction'])} | "
        f"{_fmt(control['core']['stability']['energyGrowthFraction'])} |",
        "| Action momentum cosine | "
        f"{_fmt(ph['actionInterventions']['momentumCosine'])} | "
        f"{_fmt(control['actionInterventions']['momentumCosine'])} |",
        "| Direct puck cross-talk | "
        f"{_fmt(ph['actionInterventions']['puckCrossTalkRmse'], 6)} | "
        f"{_fmt(control['actionInterventions']['puckCrossTalkRmse'], 6)} |",
        "",
        "Paired H8 pH-minus-control delta NRMSE: "
        f"{_fmt(result['paired']['h8DeltaNrmsePhMinusControl'], 4)}.",
        "",
        "## Pixel reinjection",
        "",
        "| Metric | Oracle-state renderer | pH H8 | Control H8 |",
        "| --- | ---: | ---: | ---: |",
        "| Accuracy | "
        f"{_fmt(oracle['accuracy'])} | "
        f"{_fmt(ph['horizons']['8']['pixels']['accuracy'])} | "
        f"{_fmt(control['horizons']['8']['pixels']['accuracy'])} |",
        "| Player IoU | "
        f"{_fmt(oracle['playerIou'])} | "
        f"{_fmt(ph['horizons']['8']['pixels']['playerIou'])} | "
        f"{_fmt(control['horizons']['8']['pixels']['playerIou'])} |",
        "| Puck IoU | "
        f"{_fmt(oracle['puckIou'])} | "
        f"{_fmt(ph['horizons']['8']['pixels']['puckIou'])} | "
        f"{_fmt(control['horizons']['8']['pixels']['puckIou'])} |",
        "",
        "## Decision gates",
        "",
    ]
    for name, passed in result["decisions"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    if any(run.get("kind") != "action_port_pixel_reinjection" for run in runs):
        raise ValueError("all inputs must be action-port pixel summaries")
    if len({run["config"]["seed"] for run in runs}) != len(runs):
        raise ValueError("summary seeds must be independent")
    reference_config = {
        key: value for key, value in runs[0]["config"].items() if key != "seed"
    }
    if any(
        {key: value for key, value in run["config"].items() if key != "seed"}
        != reference_config
        for run in runs[1:]
    ):
        raise ValueError("summary configurations must match apart from seed")
    result = aggregate(runs, args.summaries)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.markdown.write_text(markdown_summary(result), encoding="utf-8")


if __name__ == "__main__":
    main()
