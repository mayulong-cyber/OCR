"""
Offline evaluation script for OCR vs OCR+VLM handwriting recognition.

Usage:
    python scripts/eval_handwriting_vlm.py --image-dir data/eval/handwriting/images --gt data/eval/handwriting/groundtruth.json

Groundtruth JSON format:
    {
        "image1.png": ["line1 text", "line2 text"],
        "image2.png": ["line1 text"]
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ugocr.config import Settings
from ugocr.ocr.factory import PipelineFactory


def char_error_rate(reference: str, hypothesis: str) -> float:
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    n = len(ref_chars)
    m = len(hyp_chars)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[n][m] / max(n, m)


def evaluate_image(pipeline, image_path: Path, groundtruth: list[str]) -> dict:
    image_bytes = image_path.read_bytes()

    started = time.perf_counter()
    result_no_vlm = pipeline.chinese.recognize(image_bytes, use_vlm=False)
    latency_no_vlm = time.perf_counter() - started

    started = time.perf_counter()
    result_with_vlm = pipeline.chinese.recognize(image_bytes, use_vlm=True)
    latency_with_vlm = time.perf_counter() - started

    ocr_lines = [line.text for line in result_no_vlm.lines if line.text.strip()]
    vlm_lines = [line.text for line in result_with_vlm.lines if line.text.strip()]

    def avg_cer(pred_lines: list[str], gt_lines: list[str]) -> float:
        pairs = min(len(pred_lines), len(gt_lines))
        if pairs == 0:
            return 1.0
        total = sum(char_error_rate(gt_lines[i], pred_lines[i]) for i in range(pairs))
        return total / pairs

    cer_ocr = avg_cer(ocr_lines, groundtruth)
    cer_vlm = avg_cer(vlm_lines, groundtruth)

    return {
        "ocr_lines": ocr_lines,
        "vlm_lines": vlm_lines,
        "groundtruth": groundtruth,
        "cer_ocr": round(cer_ocr, 4),
        "cer_vlm": round(cer_vlm, 4),
        "latency_ocr": round(latency_no_vlm, 3),
        "latency_vlm": round(latency_with_vlm, 3),
        "vlm_warnings": result_with_vlm.warnings,
        "vlm_corrected": result_with_vlm.corrected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR vs OCR+VLM on handwriting images.")
    parser.add_argument("--image-dir", required=True, help="Directory containing handwriting images.")
    parser.add_argument("--gt", required=True, help="Path to groundtruth JSON file.")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    gt_path = Path(args.gt)

    if not image_dir.is_dir():
        print(f"ERROR: Image directory not found: {image_dir}")
        sys.exit(1)
    if not gt_path.is_file():
        print(f"ERROR: Groundtruth file not found: {gt_path}")
        sys.exit(1)

    with gt_path.open("r", encoding="utf-8") as f:
        groundtruth: dict[str, list[str]] = json.load(f)

    settings = Settings.from_env()
    factory = PipelineFactory(settings)
    factory.warmup(include_table=False)

    results: dict[str, dict] = {}
    total_cer_ocr = 0.0
    total_cer_vlm = 0.0
    total_latency_ocr = 0.0
    total_latency_vlm = 0.0
    count = 0

    print(f"{'Image':<30} {'CER(OCR)':<12} {'CER(VLM)':<12} {'Lat(OCR)':<12} {'Lat(VLM)':<12} {'Improved'}")
    print("-" * 90)

    for image_name, gt_lines in sorted(groundtruth.items()):
        image_path = image_dir / image_name
        if not image_path.is_file():
            print(f"SKIP: {image_name} not found in {image_dir}")
            continue

        result = evaluate_image(factory, image_path, gt_lines)
        results[image_name] = result

        total_cer_ocr += result["cer_ocr"]
        total_cer_vlm += result["cer_vlm"]
        total_latency_ocr += result["latency_ocr"]
        total_latency_vlm += result["latency_vlm"]
        count += 1

        improved = "YES" if result["cer_vlm"] < result["cer_ocr"] else "NO"
        print(
            f"{image_name:<30} {result['cer_ocr']:<12.4f} {result['cer_vlm']:<12.4f} "
            f"{result['latency_ocr']:<12.3f} {result['latency_vlm']:<12.3f} {improved}"
        )

    print("-" * 90)
    if count > 0:
        avg_cer_ocr = total_cer_ocr / count
        avg_cer_vlm = total_cer_vlm / count
        avg_lat_ocr = total_latency_ocr / count
        avg_lat_vlm = total_latency_vlm / count
        latency_increase = avg_lat_vlm - avg_lat_ocr

        print(f"\nSummary ({count} images):")
        print(f"  OCR-only  avg CER: {avg_cer_ocr:.4f}")
        print(f"  OCR+VLM   avg CER: {avg_cer_vlm:.4f}")
        if avg_cer_vlm < avg_cer_ocr:
            improvement = (avg_cer_ocr - avg_cer_vlm) / avg_cer_ocr * 100
            print(f"  VLM improved accuracy by {improvement:.1f}%")
        else:
            print("  VLM did not improve accuracy on this eval set.")
        print(f"  OCR-only  avg latency: {avg_lat_ocr:.3f}s")
        print(f"  OCR+VLM   avg latency: {avg_lat_vlm:.3f}s")
        print(f"  Avg latency increase: {latency_increase:.3f}s")
    else:
        print("\nNo images evaluated.")

    output_path = image_dir.parent / "eval_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
