from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm up UGOCR endpoints with local images.")
    parser.add_argument("--api-url", default="https://172.25.144.4:8090")
    parser.add_argument("--app-key", default="test_key")
    parser.add_argument("--chinese-image", type=Path)
    parser.add_argument("--table-image", type=Path)
    parser.add_argument("--use-vlm", default="false", choices=["true", "false"])
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--verify-tls", action="store_true", help="Verify HTTPS certificates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures = 0
    for endpoint, image in [
        ("/api/v1/ocr/recognize", args.chinese_image),
        ("/api/v1/ocr/table", args.table_image),
    ]:
        if image is None:
            continue
        try:
            elapsed = _post(
                args.api_url.rstrip("/") + endpoint,
                image,
                args.use_vlm == "true",
                args.app_key,
                args.timeout,
                args.verify_tls,
            )
            print(f"OK {endpoint} {image} elapsed={elapsed:.3f}s")
        except Exception as exc:
            failures += 1
            print(f"FAIL {endpoint} {image}: {exc}")
    if failures:
        sys.exit(1)


def _post(url: str, image: Path, use_vlm: bool, app_key: str, timeout: int, verify_tls: bool) -> float:
    import time

    started_at = time.perf_counter()
    with image.open("rb") as handle:
        response = requests.post(
            url,
            headers={"appKey": app_key},
            files={"file": (image.name, handle, "image/png")},
            data={"source_lang": "ch", "use_vlm": str(use_vlm).lower()},
            timeout=timeout,
            verify=verify_tls,
        )
    response.raise_for_status()
    return time.perf_counter() - started_at


if __name__ == "__main__":
    main()
