from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch

from blocket_league.collision_anticipation_probe import (
    run_collision_anticipation_probes,
    run_random_weight_control,
)
from blocket_league.pixel_probe import run_pixel_interpretability
from blocket_league.position_geometry_probe import run_position_geometry_probe
from blocket_league.position_write_probe import run_position_write_probe
from blocket_league.ring_probe import run_ring_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the central frozen-checkpoint analyses.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "analysis-index.json"
    index: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "stages": [],
    }

    def record(name: str, operation: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        stage: dict[str, Any] = {"name": name, "status": "running"}
        index["stages"].append(stage)
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        print(json.dumps({"analysis": name, "status": "running"}), flush=True)
        try:
            result = operation()
        except Exception:
            stage["status"] = "failed"
            stage["seconds"] = time.perf_counter() - started
            index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
            raise
        stage["status"] = "completed"
        stage["seconds"] = time.perf_counter() - started
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        print(json.dumps({"analysis": name, **stage}), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    pixel_path = args.output_dir / "pixel-interpretability.json"
    record(
        "pixel_interpretability_and_jacobian_lens",
        lambda: run_pixel_interpretability(
            args.checkpoint,
            pixel_path,
            fit_samples=512,
            test_samples=256,
            batch_size=32,
            rollout_frames=12,
            strength=8.0,
            write_frames=4,
        ),
    )
    record(
        "direction_ring",
        lambda: run_ring_probe(
            args.checkpoint,
            args.output_dir / "ring-probe.json",
            fit_samples=2048,
            test_samples=1024,
            batch_size=64,
            device_name="cuda",
            causal_manifest_path=pixel_path,
            causal_samples=256,
        ),
    )
    record(
        "position_geometry",
        lambda: run_position_geometry_probe(
            args.checkpoint,
            args.output_dir / "position-geometry.json",
            samples=4096,
            fit_samples=2048,
            quadrant_fit_samples=3072,
            batch_size=32,
            device_name="cuda",
        ),
    )
    record(
        "position_writes",
        lambda: run_position_write_probe(
            args.checkpoint,
            args.output_dir / "position-write.json",
            fit_samples=1024,
            test_samples=128,
            batch_size=32,
            block=5,
            rollout_frames=12,
            rollout_strength=8.0,
            device_name="cuda",
        ),
    )
    record(
        "collision_anticipation",
        lambda: run_collision_anticipation_probes(
            args.checkpoint,
            args.output_dir / "collision-anticipation.json",
            horizons=(1, 2, 4, 6, 8),
            fit_pairs=1024,
            test_pairs=512,
            batch_size=64,
            device_name="cuda",
        ),
    )
    record(
        "collision_random_weight_control",
        lambda: run_random_weight_control(
            args.checkpoint,
            args.output_dir / "collision-random-control.json",
            horizons=(1, 4, 8),
            fit_pairs=1024,
            test_pairs=512,
            batch_size=64,
            device_name="cuda",
        ),
    )


if __name__ == "__main__":
    main()
