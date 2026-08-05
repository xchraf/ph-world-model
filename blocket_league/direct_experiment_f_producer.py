"""Producer-only half of Experiment F.

This is the only Experiment F entry point that imports simulators or reads the
private excitation seed.  Its output boundary is the exact two-key archive
``{"pixels", "manifest"}``; this module is never copied into the learner
source bundle.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from .action_free_excitation import (
    HiddenExcitationConfig,
    action_free_environment_config_sha256,
    assert_pixels_only_payload,
    hidden_excitation_config_sha256,
    make_action_free_video,
    pixels_only_sha256,
    private_producer_seed_from_file,
)
from .direct_action_free_data import (
    PixelsOnlyManifest,
    classes_from_rgb,
    sanitized_pixel_tensor_sha256,
)
from .direct_pixels_io import load_sanitized_split
from .experiment_f_contract import (
    ExperimentFConfig,
    REGISTERED_VARIANTS,
    REGISTERED_SYSTEMS,
)
from .runtime_firewall_trace import RuntimeFirewallTrace, verify_runtime_trace
from .source_provenance import (
    build_source_manifest,
    load_source_manifest,
    verify_source_manifest,
)


def _atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _atomic_json_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _plain_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path} must contain a plain JSON object")
    return value


def _aggregate_pixel_hash(hashes: list[str], shape: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(str(shape).encode("ascii"))
    for item in hashes:
        digest.update(item.encode("ascii"))
    return digest.hexdigest()


@torch.no_grad()
def collect_action_free_video_cache(
    system: str,
    *,
    trajectories: int,
    frames_per_trajectory: int,
    image_size: int,
    seed: int,
    log_every: int = 256,
    runtime_trace: RuntimeFirewallTrace | None = None,
    trace_phase: str = "producer",
) -> tuple[torch.Tensor, PixelsOnlyManifest]:
    """Render hidden-excitation videos and immediately erase producer data."""

    if system not in REGISTERED_SYSTEMS:
        raise ValueError(f"unknown system {system!r}")
    if type(trajectories) is not int or trajectories < 1:
        raise ValueError("trajectories must be positive")
    excitation = HiddenExcitationConfig(
        frames=frames_per_trajectory,
        image_size=image_size,
    )
    videos: list[torch.Tensor] = []
    hashes: list[str] = []
    started = time.perf_counter()
    for index in range(trajectories):
        payload = make_action_free_video(
            system,
            seed + index * 104_729,
            config=excitation,
        )
        assert_pixels_only_payload(payload)
        if runtime_trace is not None:
            runtime_trace.record_tensor_payload(
                phase=trace_phase,
                role="raw_action_erased_video",
                tensors=payload,
            )
        hashes.append(pixels_only_sha256(payload))
        videos.append(classes_from_rgb(payload["frames"]))
        del payload
        if log_every > 0 and (
            (index + 1) % log_every == 0 or index + 1 == trajectories
        ):
            print(
                json.dumps(
                    {
                        "stage": "collect_action_free_pixels",
                        "system": system,
                        "trajectories": index + 1,
                        "total": trajectories,
                        "seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    cache = torch.stack(videos)
    return cache, PixelsOnlyManifest(
        system=system,
        trajectories=trajectories,
        frames_per_trajectory=frames_per_trajectory,
        image_size=image_size,
        aggregate_sha256=_aggregate_pixel_hash(hashes, tuple(cache.shape)),
        sanitized_tensor_sha256=sanitized_pixel_tensor_sha256(cache),
    )


def _producer_split_root(root: Path, split: str) -> Path:
    if split in {"fit", "validation"}:
        return root / "trainer-mount"
    if split == "test":
        return root / "heldout"
    raise ValueError(f"unknown sanitized split {split!r}")


def generate_sanitized_splits(
    system: str,
    sanitized_root: Path,
    config: ExperimentFConfig,
    *,
    producer_seed: int,
    runtime_trace: RuntimeFirewallTrace | None = None,
    source_tree_sha256: str | None = None,
) -> dict[str, Any]:
    """Generate all splits without ever serializing the hidden excitation."""

    if system not in REGISTERED_SYSTEMS:
        raise ValueError(f"unknown system {system!r}")
    if source_tree_sha256 is None:
        source_tree_sha256 = str(build_source_manifest()["treeSha256"])
    owns_runtime_trace = runtime_trace is None
    if runtime_trace is None:
        runtime_trace = RuntimeFirewallTrace(
            sanitized_root / "seals" / system / "firewall-trace.jsonl",
            stage=f"producer:{system}",
            source_tree_sha256=source_tree_sha256,
        )
    split_counts = {
        "fit": config.fit_trajectories,
        "validation": config.validation_trajectories,
        "test": config.test_trajectories,
    }
    manifests: dict[str, Any] = {}
    for split_index, (split, trajectories) in enumerate(split_counts.items()):
        path = (
            _producer_split_root(sanitized_root, split)
            / system
            / f"{split}-pixels.pt"
        )
        pixels, manifest = collect_action_free_video_cache(
            system,
            trajectories=trajectories,
            frames_per_trajectory=config.cache_frames,
            image_size=config.image_size,
            seed=producer_seed + split_index * 1_000_003,
            runtime_trace=runtime_trace,
            trace_phase="producer",
        )
        payload = {"pixels": pixels, "manifest": asdict(manifest)}
        _atomic_torch_save(payload, path)
        runtime_trace.record_file_read(
            path,
            role=f"sanitized_archive_published:{split}",
            serialized_keys=("manifest", "pixels"),
            semantic_sha256=manifest.sanitized_tensor_sha256,
        )
        manifests[split] = asdict(manifest)
    runtime_trace_seal = runtime_trace.snapshot().to_dict()
    if owns_runtime_trace:
        runtime_trace.close()
    excitation_config = HiddenExcitationConfig(
        frames=config.cache_frames, image_size=config.image_size
    )
    summary = {
        "system": system,
        "splits": manifests,
        "generationEnvironmentSha256": action_free_environment_config_sha256(
            system, image_size=config.image_size
        ),
        "producerSeedSerialized": False,
        "physicalCommandsSerialized": False,
        "simulatorStatesSerialized": False,
        "sourceTreeSha256": source_tree_sha256,
        "runtimeTrace": runtime_trace_seal,
        "hiddenExcitationConfig": asdict(excitation_config),
        "hiddenExcitationConfigSha256": hidden_excitation_config_sha256(
            excitation_config
        ),
    }
    _atomic_json_save(
        summary, sanitized_root / "seals" / system / "manifest.json"
    )
    return summary


def validate_completed_producer(
    sanitized_root: Path,
    system: str,
    experiment: ExperimentFConfig,
    source_tree_sha256: str,
) -> dict[str, Any] | None:
    paths = {
        "fit": sanitized_root / "trainer-mount" / system / "fit-pixels.pt",
        "validation": sanitized_root
        / "trainer-mount"
        / system
        / "validation-pixels.pt",
        "test": sanitized_root / "heldout" / system / "test-pixels.pt",
    }
    seal_path = sanitized_root / "seals" / system / "manifest.json"
    if not seal_path.exists() or not all(path.exists() for path in paths.values()):
        return None
    seal = _plain_json(seal_path)
    if set(seal) != {
        "system",
        "splits",
        "generationEnvironmentSha256",
        "producerSeedSerialized",
        "physicalCommandsSerialized",
        "simulatorStatesSerialized",
        "sourceTreeSha256",
        "runtimeTrace",
        "hiddenExcitationConfig",
        "hiddenExcitationConfigSha256",
    } or (
        seal["system"] != system
        or seal["generationEnvironmentSha256"]
        != action_free_environment_config_sha256(
            system, image_size=experiment.image_size
        )
        or seal["producerSeedSerialized"] is not False
        or seal["physicalCommandsSerialized"] is not False
        or seal["simulatorStatesSerialized"] is not False
        or seal["sourceTreeSha256"] != source_tree_sha256
    ):
        raise ValueError("producer completion seal is invalid")
    excitation_config = HiddenExcitationConfig(
        frames=experiment.cache_frames, image_size=experiment.image_size
    )
    if (
        seal["hiddenExcitationConfig"] != asdict(excitation_config)
        or seal["hiddenExcitationConfigSha256"]
        != hidden_excitation_config_sha256(excitation_config)
    ):
        raise ValueError("producer hidden-excitation configuration seal is invalid")
    expected_counts = {
        "fit": experiment.fit_trajectories,
        "validation": experiment.validation_trajectories,
        "test": experiment.test_trajectories,
    }
    if type(seal["splits"]) is not dict or set(seal["splits"]) != set(paths):
        raise ValueError("producer completion split table is invalid")
    for split, path in paths.items():
        _, manifest = load_sanitized_split(path, expected_system=system)
        raw_sealed = dict(seal["splits"][split])
        raw_sealed["source_schema"] = tuple(raw_sealed["source_schema"])
        raw_sealed["optimization_schema"] = tuple(
            raw_sealed["optimization_schema"]
        )
        if asdict(manifest) != raw_sealed:
            raise ValueError(f"producer {split} seal differs from archive")
        if (
            manifest.trajectories != expected_counts[split]
            or manifest.frames_per_trajectory != experiment.cache_frames
            or manifest.image_size != experiment.image_size
        ):
            raise ValueError(f"producer {split} archive differs from configuration")
    records = verify_runtime_trace(
        sanitized_root / "seals" / system / "firewall-trace.jsonl",
        seal["runtimeTrace"],
    )
    latest_attempt_sequence = max(
        record["sequence"]
        for record in records
        if record["event"] == "stage_boundary"
    )
    source_events = [
        record
        for record in records
        if record["event"] == "tensor_payload"
        and record["sequence"] > latest_attempt_sequence
    ]
    if (
        len(source_events) != sum(expected_counts.values())
        or any(
            record["payload"].get("phase") != "producer"
            or set(record["payload"].get("tensors", {})) != {"frames"}
            for record in source_events
        )
        or any(
            record["payload"].get("sourceTreeSha256") != source_tree_sha256
            for record in records
            if record["event"] == "stage_boundary"
        )
    ):
        raise ValueError("producer runtime firewall trace is incomplete")
    return seal


def run_producer_stage(
    sanitized_root: Path,
    *,
    system: str,
    producer_seed: int,
    experiment: ExperimentFConfig,
    source_manifest: Mapping[str, Any] | None = None,
    producer_seed_path: Path | None = None,
    source_manifest_path: Path | None = None,
) -> dict[str, Any]:
    effective_source_manifest = (
        build_source_manifest() if source_manifest is None else dict(source_manifest)
    )
    source_tree_sha256 = verify_source_manifest(effective_source_manifest)
    completed = validate_completed_producer(
        sanitized_root, system, experiment, source_tree_sha256
    )
    if completed is not None:
        return completed
    runtime_trace = RuntimeFirewallTrace(
        sanitized_root / "seals" / system / "firewall-trace.jsonl",
        stage=f"producer:{system}",
        source_tree_sha256=source_tree_sha256,
    )
    if producer_seed_path is not None:
        runtime_trace.record_file_read(
            producer_seed_path, role="producer_private_seed_file"
        )
    if source_manifest_path is not None:
        runtime_trace.record_file_read(
            source_manifest_path,
            role="sealed_source_manifest",
            serialized_keys=tuple(sorted(effective_source_manifest)),
            semantic_sha256=source_tree_sha256,
        )
    generate_sanitized_splits(
        system,
        sanitized_root,
        experiment,
        producer_seed=producer_seed,
        runtime_trace=runtime_trace,
        source_tree_sha256=source_tree_sha256,
    )
    runtime_trace.close()
    completed = validate_completed_producer(
        sanitized_root, system, experiment, source_tree_sha256
    )
    if completed is None:  # pragma: no cover
        raise AssertionError("producer returned without complete sanitized artifacts")
    return completed


def _experiment_from_args(args: argparse.Namespace) -> ExperimentFConfig:
    return ExperimentFConfig(
        seed=151_910_737,
        fit_trajectories=args.fit_trajectories,
        validation_trajectories=args.validation_trajectories,
        test_trajectories=args.test_trajectories,
        history_frames=8,
        transitions=args.transitions,
        cache_frames=args.cache_frames,
        image_size=args.image_size,
        patch_size=args.patch_size,
        backbone_preset=args.backbone_preset,
        variants=REGISTERED_VARIANTS,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sanitized_root", type=Path)
    parser.add_argument("--system", choices=REGISTERED_SYSTEMS, required=True)
    parser.add_argument("--producer-seed-file", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--fit-trajectories", type=int, default=4_096)
    parser.add_argument("--validation-trajectories", type=int, default=512)
    parser.add_argument("--test-trajectories", type=int, default=512)
    parser.add_argument("--transitions", type=int, default=8)
    parser.add_argument("--cache-frames", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--backbone-preset", default="tiny")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_manifest = load_source_manifest(args.source_manifest)
    result = run_producer_stage(
        args.sanitized_root,
        system=args.system,
        producer_seed=private_producer_seed_from_file(
            args.producer_seed_file, system=args.system
        ),
        experiment=_experiment_from_args(args),
        source_manifest=source_manifest,
        producer_seed_path=args.producer_seed_file,
        source_manifest_path=args.source_manifest,
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "collect_action_free_video_cache",
    "generate_sanitized_splits",
    "run_producer_stage",
    "validate_completed_producer",
]
