from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable


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


def _fmt(metric: dict[str, float], digits: int = 3) -> str:
    return f"{metric['mean']:.{digits}f} ± {metric['std']:.{digits}f}"


def aggregate(runs: list[dict[str, Any]], sources: list[Path]) -> dict[str, Any]:
    branches: dict[str, Any] = {}
    for branch in ("portHamiltonian", "signFreeControl"):
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
                    metric: _collect(
                        runs,
                        lambda run, b=branch, h=horizon, m=metric: run[
                            "evaluation"
                        ][b]["rolloutByHorizon"][h][m],
                    )
                    for metric in ("deltaNrmse", "qR2", "pR2")
                }
                for horizon in ("1", "2", "4", "8")
            },
            "shuffledInitialState": _collect(
                runs,
                lambda run, b=branch: run["evaluation"][b][
                    "shuffledInitialStateControl"
                ]["deltaNrmse"],
            ),
            "eventPrediction": {
                metric: _collect(
                    runs,
                    lambda run, b=branch, m=metric: run["evaluation"][b][
                        "eventPrediction"
                    ][m],
                )
                for metric in ("accuracy", "balancedAccuracy")
            },
            "freeCore": {
                "effectiveMass": [
                    _collect(
                        runs,
                        lambda run, b=branch, i=index: run["evaluation"][b][
                            "freeCore"
                        ]["parameters"]["effectiveMass"][i],
                    )
                    for index in range(2)
                ],
                "effectiveDrag": [
                    _collect(
                        runs,
                        lambda run, b=branch, i=index: run["evaluation"][b][
                            "freeCore"
                        ]["parameters"]["effectiveDrag"][i],
                    )
                    for index in range(2)
                ],
                "violationRate": _collect(
                    runs,
                    lambda run, b=branch: run["evaluation"][b]["freeCore"][
                        "passivity"
                    ]["coreViolationRate"],
                ),
                "energyRatio": _collect(
                    runs,
                    lambda run, b=branch: run["evaluation"][b]["freeCore"][
                        "passivity"
                    ]["meanEnergyRatio"],
                ),
            },
            "stability": {
                metric: _collect(
                    runs,
                    lambda run, b=branch, m=metric: run["evaluation"][b][
                        "stability"
                    ][m],
                )
                for metric in ("energyGrowthFraction", "finalToInitialEnergy")
            },
            "externalPort": {
                metric: _collect(
                    runs,
                    lambda run, b=branch, m=metric: statistics.fmean(
                        item[m]
                        for item in run["evaluation"][b]["externalPortInterventions"]
                    ),
                )
                for metric in ("cosine", "gain", "crossTalkRmse")
            },
            "regimes": {
                regime: {
                    "samples": _collect(
                        runs,
                        lambda run, b=branch, g=regime: run["evaluation"][b][
                            "regimes"
                        ][g]["samples"],
                    ),
                    "deltaNrmse": _collect(
                        runs,
                        lambda run, b=branch, g=regime: run["evaluation"][b][
                            "regimes"
                        ][g]["transition"]["deltaNrmse"],
                    ),
                    "meanPortNorm": _collect(
                        runs,
                        lambda run, b=branch, g=regime: run["evaluation"][b][
                            "regimes"
                        ][g]["meanPortNorm"],
                    ),
                }
                for regime in (
                    "free",
                    "disc_impact",
                    "wall",
                    "goal_entry",
                    "goal_pause",
                    "kickoff",
                )
            },
        }

    paired = {
        "h8DeltaNrmsePhMinusControl": _collect(
            runs,
            lambda run: run["evaluation"]["portHamiltonian"]["rolloutByHorizon"][
                "8"
            ]["deltaNrmse"]
            - run["evaluation"]["signFreeControl"]["rolloutByHorizon"]["8"][
                "deltaNrmse"
            ],
        ),
        "playerDragPhMinusControl": _collect(
            runs,
            lambda run: run["evaluation"]["portHamiltonian"]["freeCore"][
                "parameters"
            ]["effectiveDrag"][0]
            - run["evaluation"]["signFreeControl"]["freeCore"]["parameters"][
                "effectiveDrag"
            ][0],
        ),
        "puckDragPhMinusControl": _collect(
            runs,
            lambda run: run["evaluation"]["portHamiltonian"]["freeCore"][
                "parameters"
            ]["effectiveDrag"][1]
            - run["evaluation"]["signFreeControl"]["freeCore"]["parameters"][
                "effectiveDrag"
            ][1],
        ),
    }
    timing = {
        metric: _collect(
            runs,
            lambda run, m=metric: run["timing"][m],
        )
        for metric in (
            "collectionSeconds",
            "trainingSeconds",
            "evaluationSeconds",
            "totalSeconds",
        )
    }
    return {
        "version": 1,
        "runs": len(runs),
        "sources": [str(path) for path in sources],
        "baseCheckpointStep": runs[0]["baseCheckpointStep"],
        "config": runs[0]["config"],
        "capacity": runs[0]["capacity"],
        "timing": timing,
        "branches": branches,
        "paired": paired,
    }


def markdown_summary(result: dict[str, Any]) -> str:
    ph = result["branches"]["portHamiltonian"]
    control = result["branches"]["signFreeControl"]
    lines = [
        "# Experiment B — causal pH bottleneck results",
        "",
        f"Independent runs: {result['runs']}. Values are mean ± sample standard deviation across seeds.",
        "",
        "## Runtime",
        "",
        "| Phase | Seconds per seed |",
        "| --- | ---: |",
        f"| Block-5 collection | {_fmt(result['timing']['collectionSeconds'], 1)} |",
        f"| Paired training | {_fmt(result['timing']['trainingSeconds'], 1)} |",
        f"| Evaluation | {_fmt(result['timing']['evaluationSeconds'], 2)} |",
        f"| Internal total | {_fmt(result['timing']['totalSeconds'], 1)} |",
        "",
        "## State and causal rollout",
        "",
        "| Branch | q R² | p R² | h=1 delta NRMSE | h=2 | h=4 | h=8 | Shuffled h=8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, branch in (("pH", ph), ("Sign-free", control)):
        lines.append(
            f"| {name} | {_fmt(branch['stateReadout']['qR2'])} "
            f"| {_fmt(branch['stateReadout']['pR2'])} "
            f"| {_fmt(branch['horizons']['1']['deltaNrmse'])} "
            f"| {_fmt(branch['horizons']['2']['deltaNrmse'])} "
            f"| {_fmt(branch['horizons']['4']['deltaNrmse'])} "
            f"| {_fmt(branch['horizons']['8']['deltaNrmse'])} "
            f"| {_fmt(branch['shuffledInitialState'])} |"
        )
    lines.extend(
        [
            "",
            f"Paired h=8 pH minus control: {_fmt(result['paired']['h8DeltaNrmsePhMinusControl'], 4)}.",
            "",
            "## Learned free core",
            "",
            "| Branch | Mass, player / puck | Drag, player / puck | Passivity violations | 64-step energy growth | Final / initial energy |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, branch in (("pH", ph), ("Sign-free", control)):
        free = branch["freeCore"]
        stability = branch["stability"]
        lines.append(
            f"| {name} | {_fmt(free['effectiveMass'][0])} / {_fmt(free['effectiveMass'][1])} "
            f"| {_fmt(free['effectiveDrag'][0])} / {_fmt(free['effectiveDrag'][1])} "
            f"| {_fmt(free['violationRate'])} "
            f"| {_fmt(stability['energyGrowthFraction'])} "
            f"| {_fmt(stability['finalToInitialEnergy'])} |"
        )
    lines.extend(
        [
            "",
            "## Hybrid events",
            "",
            "| Regime | Samples per seed | pH delta NRMSE | Control delta NRMSE | pH port norm | Control port norm |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for regime, values in ph["regimes"].items():
        other = control["regimes"][regime]
        lines.append(
            f"| {regime} | {_fmt(values['samples'], 1)} "
            f"| {_fmt(values['deltaNrmse'])} | {_fmt(other['deltaNrmse'])} "
            f"| {_fmt(values['meanPortNorm'])} | {_fmt(other['meanPortNorm'])} |"
        )
    lines.extend(
        [
            "",
            "## Event and external-port controls",
            "",
            "| Branch | Event balanced accuracy | Port direction cosine | Port gain | Cross-talk RMSE |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, branch in (("pH", ph), ("Sign-free", control)):
        port = branch["externalPort"]
        lines.append(
            f"| {name} | {_fmt(branch['eventPrediction']['balancedAccuracy'])} "
            f"| {_fmt(port['cosine'])} | {_fmt(port['gain'])} "
            f"| {_fmt(port['crossTalkRmse'], 7)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    result = aggregate(runs, args.inputs)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.markdown_output.write_text(markdown_summary(result), encoding="utf-8")


if __name__ == "__main__":
    main()
