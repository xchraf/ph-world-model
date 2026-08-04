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
    first = runs[0]
    readouts: dict[str, Any] = {}
    for readout, stages in first["readouts"].items():
        readout_stages = []
        for stage_index, stage in enumerate(stages):
            decoders: dict[str, Any] = {}
            for decoder in stage["decoders"]:
                decoders[decoder] = {
                    "qR2": _collect(
                        runs,
                        lambda run, r=readout, i=stage_index, d=decoder: run[
                            "readouts"
                        ][r][i]["decoders"][d]["stateReadout"]["qR2"],
                    ),
                    "pR2": _collect(
                        runs,
                        lambda run, r=readout, i=stage_index, d=decoder: run[
                            "readouts"
                        ][r][i]["decoders"][d]["stateReadout"]["pR2"],
                    ),
                    "energyBalance": _collect(
                        runs,
                        lambda run, r=readout, i=stage_index, d=decoder: run[
                            "readouts"
                        ][r][i]["decoders"][d]["freeEnergyBalance"][
                            "normalizedBalanceRmse"
                        ],
                    ),
                    "passivityViolationRate": _collect(
                        runs,
                        lambda run, r=readout, i=stage_index, d=decoder: run[
                            "readouts"
                        ][r][i]["decoders"][d]["freeEnergyBalance"][
                            "passivityViolationRate"
                        ],
                    ),
                    "conformalSymplecticDefect": _collect(
                        runs,
                        lambda run, r=readout, i=stage_index, d=decoder: run[
                            "readouts"
                        ][r][i]["decoders"][d][
                            "freeAffineConformalSymplecticDefect"
                        ],
                    ),
                    "horizons": {
                        horizon: {
                            "models": {
                                model: _collect(
                                    runs,
                                    lambda run,
                                    r=readout,
                                    i=stage_index,
                                    d=decoder,
                                    h=horizon,
                                    m=model: run["readouts"][r][i]["decoders"][d][
                                        "freeDynamicsByHorizon"
                                    ][h]["models"][m]["deltaNrmse"],
                                )
                                for model in ("portHamiltonian", "affine")
                            },
                            "pairingControl": {
                                model: _collect(
                                    runs,
                                    lambda run,
                                    r=readout,
                                    i=stage_index,
                                    d=decoder,
                                    h=horizon,
                                    m=model: run["readouts"][r][i]["decoders"][d][
                                        "freeDynamicsByHorizon"
                                    ][h]["pairingControl"][m]["deltaNrmse"],
                                )
                                for model in ("portHamiltonian", "affine")
                            },
                            "reverseTimeControl": {
                                model: _collect(
                                    runs,
                                    lambda run,
                                    r=readout,
                                    i=stage_index,
                                    d=decoder,
                                    h=horizon,
                                    m=model: run["readouts"][r][i]["decoders"][d][
                                        "freeDynamicsByHorizon"
                                    ][h]["reverseTimeControl"]["models"][m][
                                        "deltaNrmse"
                                    ],
                                )
                                for model in ("portHamiltonian", "affine")
                            },
                        }
                        for horizon in first["config"]["horizons"]
                        for horizon in (str(horizon),)
                    },
                }
            readout_stages.append({"stage": stage["stage"], "decoders": decoders})
        readouts[readout] = readout_stages

    world = {
        "effectiveMass": [
            _collect(
                runs,
                lambda run, i=index: run["worldStateReference"][
                    "portHamiltonianParameters"
                ]["effectiveMass"][i],
            )
            for index in range(2)
        ],
        "effectiveDrag": [
            _collect(
                runs,
                lambda run, i=index: run["worldStateReference"][
                    "portHamiltonianParameters"
                ]["effectiveDrag"][i],
            )
            for index in range(2)
        ],
        "energyBalance": _collect(
            runs,
            lambda run: run["worldStateReference"]["freeEnergyBalance"][
                "normalizedBalanceRmse"
            ],
        ),
        "horizons": {
            horizon: {
                "models": {
                    model: _collect(
                        runs,
                        lambda run, h=horizon, m=model: run[
                            "worldStateReference"
                        ]["freeDynamicsByHorizon"][h]["models"][m]["deltaNrmse"],
                    )
                    for model in ("portHamiltonian", "affine")
                },
                "pairingControl": {
                    model: _collect(
                        runs,
                        lambda run, h=horizon, m=model: run[
                            "worldStateReference"
                        ]["freeDynamicsByHorizon"][h]["pairingControl"][m][
                            "deltaNrmse"
                        ],
                    )
                    for model in ("portHamiltonian", "affine")
                },
                "reverseTimeControl": {
                    model: _collect(
                        runs,
                        lambda run, h=horizon, m=model: run[
                            "worldStateReference"
                        ]["freeDynamicsByHorizon"][h]["reverseTimeControl"]["models"][
                            m
                        ]["deltaNrmse"],
                    )
                    for model in ("portHamiltonian", "affine")
                },
            }
            for horizon in first["config"]["horizons"]
            for horizon in (str(horizon),)
        },
    }
    return {
        "version": 1,
        "runs": len(runs),
        "sources": [str(path) for path in sources],
        "checkpointStep": first["checkpointStep"],
        "splitPerRun": first["split"],
        "worldStateReference": world,
        "readouts": readouts,
    }


def markdown_summary(result: dict[str, Any]) -> str:
    lines = [
        "# Mesohelios frozen port-Hamiltonian audit",
        "",
        f"Independent runs: {result['runs']}. Values are mean ± sample standard deviation across seeds.",
        "",
        "## Simulator-state reference",
        "",
        "| Quantity | Player | Puck |",
        "| --- | ---: | ---: |",
        (
            f"| Effective mass | {_fmt(result['worldStateReference']['effectiveMass'][0], 4)} "
            f"| {_fmt(result['worldStateReference']['effectiveMass'][1], 4)} |"
        ),
        (
            f"| Effective drag | {_fmt(result['worldStateReference']['effectiveDrag'][0], 4)} "
            f"| {_fmt(result['worldStateReference']['effectiveDrag'][1], 4)} |"
        ),
        "",
        "The reference fit uses privileged simulator state and is a calibration floor, not a latent result.",
        "",
        "## Layer-wise state readability",
        "",
    ]
    for readout, stages in result["readouts"].items():
        lines.extend(
            [
                f"### {readout}",
                "",
                "| Stage | q R² | p R² | q R², delta-aligned | p R², delta-aligned |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for stage in stages:
            state = stage["decoders"]["stateOnly"]
            aligned = stage["decoders"]["statePlusDelta"]
            lines.append(
                f"| {stage['stage']} | {_fmt(state['qR2'])} | {_fmt(state['pR2'])} "
                f"| {_fmt(aligned['qR2'])} | {_fmt(aligned['pR2'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## State-only decoder: pH versus affine dynamics",
            "",
            "Delta NRMSE is normalized so persistence equals 1.0. Lower is better.",
            "",
        ]
    )
    for readout in ("entity_pair", "spatial_mean", "fixed_bottom_right"):
        lines.extend(
            [
                f"### {readout}",
                "",
                "| Stage | h=1 pH / affine | h=2 pH / affine | h=4 pH / affine | h=8 pH / affine | Energy residual | Passivity violations |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for stage in result["readouts"][readout]:
            decoder = stage["decoders"]["stateOnly"]
            horizon_cells = []
            for horizon in ("1", "2", "4", "8"):
                values = decoder["horizons"][horizon]["models"]
                horizon_cells.append(
                    f"{_fmt(values['portHamiltonian'])} / {_fmt(values['affine'])}"
                )
            lines.append(
                f"| {stage['stage']} | {' | '.join(horizon_cells)} "
                f"| {_fmt(decoder['energyBalance'])} "
                f"| {_fmt(decoder['passivityViolationRate'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Temporal negative controls at h=8",
            "",
            "The shuffled column breaks the true endpoint pairing. The reverse-time columns fit the same model classes from t+h back to t.",
            "",
        ]
    )
    for readout in ("entity_pair", "spatial_mean", "fixed_bottom_right"):
        lines.extend(
            [
                f"### {readout}",
                "",
                "| Stage | Forward pH | Shuffled pH | Reverse pH | Reverse affine |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for stage in result["readouts"][readout]:
            horizon = stage["decoders"]["stateOnly"]["horizons"]["8"]
            lines.append(
                f"| {stage['stage']} | {_fmt(horizon['models']['portHamiltonian'])} "
                f"| {_fmt(horizon['pairingControl']['portHamiltonian'])} "
                f"| {_fmt(horizon['reverseTimeControl']['portHamiltonian'])} "
                f"| {_fmt(horizon['reverseTimeControl']['affine'])} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    result = aggregate(runs, args.inputs)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.markdown_output.write_text(markdown_summary(result), encoding="utf-8")


if __name__ == "__main__":
    main()
