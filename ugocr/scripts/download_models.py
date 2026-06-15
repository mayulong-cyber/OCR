#!/usr/bin/env python3
"""
Download all required pre-trained models for UGOCR.

Downloads:
  1. PP-OCRv5 detection model   -> models/ppocrv5_det/
  2. PP-OCRv5 Chinese rec model -> models/ppocrv5_chinese_svtrv2_rec/
  3. PP-OCRv5 table rec model   -> models/ppocrv5_table_rec/
  4. SLANet table structure     -> models/slanet/

Usage:
  python scripts/download_models.py
  python scripts/download_models.py --dry-run            # print actions only
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _mini_image() -> Path:
    """Create a 100x100 white test image for triggering model downloads."""
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    path = PROJECT_ROOT / "data" / "outputs" / "_dl_test.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    return path


def download_detection(target: Path) -> bool:
    """Download PP-OCRv5 detection model."""
    from paddleocr import PaddleOCR

    _check_mark(target, "detection", "inference.pdmodel", "model.pdmodel", "inference.json")
    target.mkdir(parents=True, exist_ok=True)
    test_img = _mini_image()
    print(f"  Downloading detection model to {target} ...")
    try:
        ocr = PaddleOCR(
            lang="ch",
            det_model_dir=str(target),
            rec=False,
            use_angle_cls=False,
            show_log=False,
            use_gpu=False,
        )
        ocr.ocr(str(test_img))
        print(f"  Detection model ready: {target}")
        return True
    except Exception as exc:
        print(f"  WARNING: detection model download may have failed: {exc}")
        return _check_files_exist(target, "inference.pdmodel", "model.pdmodel", "inference.json")


def download_chinese_recognition(target: Path, det_dir: Path) -> bool:
    """Download PP-OCRv5 Chinese recognition model."""
    from paddleocr import PaddleOCR

    _check_mark(target, "chinese_rec", "inference.pdmodel", "model.pdmodel", "inference.json")
    target.mkdir(parents=True, exist_ok=True)
    test_img = _mini_image()
    print(f"  Downloading Chinese recognition model to {target} ...")
    try:
        ocr = PaddleOCR(
            lang="ch",
            det_model_dir=str(det_dir),
            rec_model_dir=str(target),
            use_angle_cls=False,
            show_log=False,
            use_gpu=False,
        )
        ocr.ocr(str(test_img))
        print(f"  Chinese recognition model ready: {target}")
        return True
    except Exception as exc:
        print(f"  WARNING: Chinese rec model download may have failed: {exc}")
        return _check_files_exist(target, "inference.pdmodel", "model.pdmodel", "inference.json")


def download_table_models(table_rec_dir: Path, slanet_dir: Path, det_dir: Path) -> bool:
    """Download PP-StructureV3 table models (table rec + SLANet structure)."""
    from paddleocr import PPStructure

    already_table = _check_mark(table_rec_dir, "table_rec", "inference.pdmodel", "model.pdmodel", "inference.json")
    already_slanet = _check_mark(slanet_dir, "slanet", "inference.pdmodel", "model.pdmodel", "inference.json")
    if already_table and already_slanet:
        return True

    table_rec_dir.mkdir(parents=True, exist_ok=True)
    slanet_dir.mkdir(parents=True, exist_ok=True)
    test_img = _mini_image()
    print(f"  Downloading table recognition model to {table_rec_dir} ...")
    print(f"  Downloading SLANet table structure model to {slanet_dir} ...")
    try:
        engine = PPStructure(
            lang="ch",
            det_model_dir=str(det_dir),
            table_model_dir=str(slanet_dir),
            table_char_dict_path=None,
            layout=False,
            table=True,
            image_orientation=False,
            show_log=False,
            use_gpu=False,
        )
        engine(str(test_img))
        ok1 = _check_files_exist(table_rec_dir, "inference.pdmodel", "model.pdmodel", "inference.json")
        ok2 = _check_files_exist(slanet_dir, "inference.pdmodel", "model.pdmodel", "inference.json")
        if ok1:
            print(f"  Table rec model ready: {table_rec_dir}")
        else:
            print(f"  WARNING: table rec model files not found in {table_rec_dir}")
        if ok2:
            print(f"  SLANet model ready: {slanet_dir}")
        else:
            print(f"  WARNING: SLANet model files not found in {slanet_dir}")
        return ok1 and ok2
    except Exception as exc:
        print(f"  WARNING: table models download may have failed: {exc}")
        return _check_files_exist(
            table_rec_dir, "inference.pdmodel", "model.pdmodel", "inference.json"
        ) and _check_files_exist(slanet_dir, "inference.pdmodel", "model.pdmodel", "inference.json")


def download_char_dict(target: Path) -> bool:
    """Copy the PaddleOCR Chinese character dictionary to the project."""
    _check_mark(target, "char_dict", target.name)
    if target.exists():
        return True
    try:
        import paddleocr.ppocr.utils as utils_module
        import os as _os

        src = Path(utils_module.__file__).parent / "ppocr_keys_v1.txt"
        if src.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(target))
            print(f"  Character dictionary copied: {target}")
            return True
    except Exception:
        pass
    print(f"  WARNING: Could not locate ppocr_keys_v1.txt, skipping char dict")
    return False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _check_files_exist(directory: Path, *candidates: str) -> bool:
    if not directory.exists():
        return False
    for name in candidates:
        if (directory / name).exists():
            return True
    return False


def _check_mark(directory: Path, label: str, *candidates: str) -> bool:
    """Return True if model files exist and print skip message."""
    if _check_files_exist(directory, *candidates):
        print(f"  [{label}] already exists, skipping: {directory}")
        return True
    return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download all required UGOCR models.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ugocr.example.yaml",
        help="Path to UGOCR YAML config (used to read target directories)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import yaml
    config_path = args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}, using default paths")
        det_dir = PROJECT_ROOT / "models" / "ppocrv5_det"
        chinese_rec_dir = PROJECT_ROOT / "models" / "ppocrv5_chinese_svtrv2_rec"
        table_rec_dir = PROJECT_ROOT / "models" / "ppocrv5_table_rec"
        slanet_dir = PROJECT_ROOT / "models" / "slanet"
        dict_path = PROJECT_ROOT / "configs" / "char_dicts" / "chinese.txt"
    else:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        paddle = data.get("paddle", {})
        det_dir = PROJECT_ROOT / paddle.get("det_model_dir", "models/ppocrv5_det")
        chinese_rec_dir = PROJECT_ROOT / paddle.get("chinese_rec_model_dir", "models/ppocrv5_chinese_svtrv2_rec")
        table_rec_dir = PROJECT_ROOT / paddle.get("table_rec_model_dir", "models/ppocrv5_table_rec")
        slanet_dir = PROJECT_ROOT / paddle.get("table_model_dir", "models/slanet")
        dict_path = PROJECT_ROOT / (paddle.get("chinese_char_dict_path") or "configs/char_dicts/chinese.txt")

    if args.dry_run:
        print("DRY RUN — would download to:")
        print(f"  Detection:      {det_dir}")
        print(f"  Chinese rec:    {chinese_rec_dir}")
        print(f"  Table rec:      {table_rec_dir}")
        print(f"  SLANet:         {slanet_dir}")
        print(f"  Char dict:      {dict_path}")
        return

    errors: list[str] = []

    print("=== PaddleOCR Models ===")
    if not download_detection(det_dir):
        errors.append("detection model")

    if not download_chinese_recognition(chinese_rec_dir, det_dir):
        errors.append("Chinese recognition model")

    if not download_table_models(table_rec_dir, slanet_dir, det_dir):
        errors.append("table models")

    if not download_char_dict(dict_path):
        errors.append("character dictionary")

    print()
    if errors:
        print(f"Completed with {len(errors)} issue(s): {', '.join(errors)}")
        print("Run   python scripts/verify_models.py   to check model status.")
        sys.exit(1)
    else:
        print("All models downloaded successfully.")
        print("Run   python scripts/verify_models.py   to confirm.")
        sys.exit(0)


if __name__ == "__main__":
    main()
