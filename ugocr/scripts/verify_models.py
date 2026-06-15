from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ugocr.config import Settings
from ugocr.model_check import check_models, models_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify local offline model directories.")
    parser.add_argument("--config", type=Path, default=Path("configs/ugocr.example.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["UGOCR_CONFIG"] = str(args.config)
    settings = Settings.from_env()
    checks = check_models(settings)
    width = max(len(item.label) for item in checks)
    for item in checks:
        mark = "OK" if item.exists else "MISS"
        print(f"{mark:4} {item.label:<{width}} {item.path} ({item.detail})")
    if not models_ok(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
