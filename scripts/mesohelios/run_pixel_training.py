from __future__ import annotations

import argparse
import json
from pathlib import Path

from blocket_league.train_pixel_direct import PixelDirectTrainConfig, train_pixel_direct


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Blocket League pixel training without Modal.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.config.read_text(encoding="utf-8"))
    payload["output_dir"] = str(args.output_dir)
    if args.init_checkpoint is not None:
        payload["init_checkpoint_path"] = str(args.init_checkpoint)

    summary = train_pixel_direct(PixelDirectTrainConfig(**payload))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
