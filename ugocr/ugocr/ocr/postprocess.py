from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from ugocr.config import VLMSettings
from ugocr.dependencies import import_required
from ugocr.ocr.types import InternalTableCell, OCRLine
from ugocr.utils.images import encode_image_data_url, prepare_vlm_image_bytes
from ugocr.utils.text import extract_json


@dataclass(frozen=True)
class VLMLineResult:
    lines: list[str]
    corrected: bool
    warnings: list[str]


@dataclass(frozen=True)
class VLMTableResult:
    cells: list[InternalTableCell]
    corrected: bool
    warnings: list[str]


class VLMPostProcessor:
    def __init__(self, settings: VLMSettings) -> None:
        self.settings = settings

    def correct_text_lines(self, language: str, image_bytes: bytes, lines: list[OCRLine]) -> VLMLineResult:
        original = [line.text for line in lines]
        if not self.settings.enabled:
            return VLMLineResult(lines=original, corrected=False, warnings=["VLM postprocess is disabled."])
        if len(lines) == 0:
            return VLMLineResult(lines=[], corrected=False, warnings=[])

        image_bytes = prepare_vlm_image_bytes(image_bytes, max_side=640, jpeg_quality=85)
        payload = {
            "language": language,
            "line_count": len(lines),
            "lines": [{"index": idx, "text": line.text, "score": line.score} for idx, line in enumerate(lines)],
        }
        prompt = (
            "You are an offline VLM OCR correction engine. Use the image as the primary evidence, "
            "and use language context only to resolve visually ambiguous OCR characters. "
            "Preserve the exact line count and line order. Return every line, not only changed lines. "
            "Do not translate, summarize, add invisible content, or delete visible characters. "
            "If a character is uncertain, keep the original OCR character. "
            "For every returned line include confidence from 0 to 1. Use confidence >= 0.70 only when the changed "
            "text is clearly supported by the visible handwriting; otherwise use a lower confidence. "
            "Output compact JSON only, without markdown: {\"lines\":[{\"index\":0,\"text\":\"...\",\"confidence\":0.0}]}.\n"
            f"OCR JSON:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
        )
        started_at = time.perf_counter()
        try:
            response = self._chat(image_bytes, prompt, max_tokens=min(self.settings.max_tokens, 160))
            elapsed = time.perf_counter() - started_at
            data = json.loads(extract_json(response))
            corrected, confidences = _parse_vlm_lines(data, original)
            if len(corrected) != len(original):
                return VLMLineResult(
                    lines=original,
                    corrected=False,
                    warnings=[
                        f"VLM text correction returned {len(corrected)} lines, expected {len(original)}; kept OCR result.",
                        f"VLM text correction elapsed {elapsed:.3f}s.",
                    ],
                )
            gated, rejected_count = _apply_line_confidence_gate(
                original,
                corrected,
                confidences,
                min_confidence=0.70,
            )
            changed = gated != original
            min_confidence = min(confidences) if confidences else 0.0
            return VLMLineResult(
                lines=gated,
                corrected=changed,
                warnings=[
                    f"VLM text correction elapsed {elapsed:.3f}s; changed={str(changed).lower()}; "
                    f"min_confidence={min_confidence:.2f}; rejected_low_confidence={rejected_count}."
                ],
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            return VLMLineResult(
                lines=original,
                corrected=False,
                warnings=[f"VLM text correction failed after {elapsed:.3f}s: {exc}"],
            )

    def enhance_table(self, image_bytes: bytes, cells: list[InternalTableCell]) -> VLMTableResult:
        if not self.settings.enabled:
            return VLMTableResult(cells=cells, corrected=False, warnings=["VLM postprocess is disabled."])
        if len(cells) == 0:
            return VLMTableResult(cells=cells, corrected=False, warnings=[])

        image_bytes = prepare_vlm_image_bytes(image_bytes, max_side=1024, jpeg_quality=88)
        payload = {
            "cell_count": len(cells),
            "cells": [
                {
                    "row": cell.row,
                    "col": cell.col,
                    "rowspan": cell.rowspan,
                    "colspan": cell.colspan,
                    "text": cell.text,
                    "score": cell.score,
                }
                for cell in cells
            ],
        }
        prompt = (
            "You are an offline VLM table OCR postprocessor. Use the table image as primary evidence "
            "and the OCR JSON as the initial structure. Return the complete one-based cell list, not only changed cells. "
            "Correct row, column, rowspan, colspan, and text only when the image clearly supports it. "
            "Do not translate text or invent invisible content. If uncertain, keep the input structure and text. "
            "Output compact JSON only, without markdown: "
            "{\"cells\":[{\"row\":1,\"col\":1,\"rowspan\":1,\"colspan\":1,\"text\":\"...\"}]}.\n"
            f"Table OCR JSON:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
        )
        started_at = time.perf_counter()
        try:
            response = self._chat(image_bytes, prompt, max_tokens=min(self.settings.max_tokens, 384))
            elapsed = time.perf_counter() - started_at
            data = json.loads(extract_json(response))
            updated = _parse_vlm_cells(data, cells)
            if len(updated) < max(1, len(cells) // 2):
                return VLMTableResult(
                    cells=cells,
                    corrected=False,
                    warnings=[
                        "VLM table enhancement returned too few cells; kept PaddleX/OCR result.",
                        f"VLM table enhancement elapsed {elapsed:.3f}s.",
                    ],
                )
            changed = _cells_changed(cells, updated)
            return VLMTableResult(
                cells=updated,
                corrected=changed,
                warnings=[f"VLM table enhancement elapsed {elapsed:.3f}s; changed={str(changed).lower()}."],
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            return VLMTableResult(
                cells=cells,
                corrected=False,
                warnings=[f"VLM table enhancement failed after {elapsed:.3f}s: {exc}"],
            )

    def _chat(self, image_bytes: bytes, prompt: str, max_tokens: int | None = None) -> str:
        if self.settings.provider == "ollama":
            return self._chat_ollama(image_bytes, prompt, max_tokens=max_tokens)
        if self.settings.provider in {"openai", "vllm"}:
            return self._chat_openai_compatible(image_bytes, prompt, max_tokens=max_tokens)
        raise ValueError(f"Unsupported VLM provider: {self.settings.provider}")

    def _chat_openai_compatible(self, image_bytes: bytes, prompt: str, max_tokens: int | None = None) -> str:
        requests = import_required("requests", "requirements-base.txt")
        url = f"{self.settings.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json"}
        body: dict[str, Any] = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": encode_image_data_url(image_bytes)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        response = requests.post(url, headers=headers, json=body, timeout=(5, self.settings.request_timeout_sec))
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])

    def _chat_ollama(self, image_bytes: bytes, prompt: str, max_tokens: int | None = None) -> str:
        import base64

        requests = import_required("requests", "requirements-base.txt")
        base_url = self.settings.base_url.rstrip("/")
        url = f"{base_url}/api/chat" if not base_url.endswith("/api/chat") else base_url
        body: dict[str, Any] = {
            "model": self.settings.model,
            "stream": False,
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": self.settings.max_tokens if max_tokens is None else max_tokens,
            },
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                }
            ],
        }
        response = requests.post(url, json=body, timeout=(5, self.settings.request_timeout_sec))
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        if isinstance(message, dict) and "content" in message:
            return str(message["content"])
        if "response" in data:
            return str(data["response"])
        raise ValueError(f"Unexpected Ollama response: {data}")


def _parse_vlm_lines(data: Any, original: list[str]) -> tuple[list[str], list[float]]:
    if isinstance(data, list):
        raw_lines = data
    elif isinstance(data, dict):
        raw_lines = data.get("lines", [])
    else:
        return [], []
    if not isinstance(raw_lines, list):
        return [], []

    parsed = list(original)
    confidences = [1.0 for _ in original]
    seen: set[int] = set()
    for fallback_idx, item in enumerate(raw_lines):
        if isinstance(item, dict):
            idx = _positive_int(item.get("index"), fallback_idx)
            text_value = item.get("text", original[idx] if 0 <= idx < len(original) else "")
            confidence_value = item.get("confidence", 0.0)
        else:
            idx = fallback_idx
            text_value = item
            confidence_value = 0.0
        if 0 <= idx < len(parsed):
            text = str(text_value).strip()
            parsed[idx] = text if text else original[idx]
            confidences[idx] = _bounded_float(confidence_value, 0.0)
            seen.add(idx)
    if len(seen) < len(original):
        return [], []
    return parsed, confidences


def _apply_line_confidence_gate(
    original: list[str],
    corrected: list[str],
    confidences: list[float],
    min_confidence: float,
) -> tuple[list[str], int]:
    gated = list(corrected)
    rejected_count = 0
    for idx, (before, after) in enumerate(zip(original, corrected)):
        if before == after:
            continue
        confidence = confidences[idx] if idx < len(confidences) else 0.0
        if confidence < min_confidence:
            gated[idx] = before
            rejected_count += 1
    return gated, rejected_count


def _parse_vlm_cells(data: dict[str, Any], original: list[InternalTableCell]) -> list[InternalTableCell]:
    raw_cells = data.get("cells", [])
    if not isinstance(raw_cells, list):
        return []
    original_by_key = {(cell.row, cell.col): cell for cell in original}
    parsed: dict[tuple[int, int], InternalTableCell] = {}
    for item in raw_cells:
        if not isinstance(item, dict):
            continue
        row = _positive_int(item.get("row"), 0)
        col = _positive_int(item.get("col"), 0)
        if row < 1 or col < 1:
            continue
        base = original_by_key.get((row, col))
        text = str(item.get("text", base.text if base is not None else "")).strip()
        parsed[(row, col)] = InternalTableCell(
            row=row,
            col=col,
            text=text,
            score=_bounded_float(item.get("score"), base.score if base is not None else 1.0),
            rowspan=max(1, _positive_int(item.get("rowspan"), base.rowspan if base is not None else 1)),
            colspan=max(1, _positive_int(item.get("colspan"), base.colspan if base is not None else 1)),
            box=base.box if base is not None else [],
        )
    return [parsed[key] for key in sorted(parsed)]


def _cells_changed(original: list[InternalTableCell], updated: list[InternalTableCell]) -> bool:
    original_keyed = {(cell.row, cell.col): cell for cell in original}
    updated_keyed = {(cell.row, cell.col): cell for cell in updated}
    if set(original_keyed) != set(updated_keyed):
        return True
    for key, updated_cell in updated_keyed.items():
        original_cell = original_keyed[key]
        if (
            original_cell.text != updated_cell.text
            or original_cell.rowspan != updated_cell.rowspan
            or original_cell.colspan != updated_cell.colspan
        ):
            return True
    return False


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(0, parsed)


def _bounded_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return min(1.0, max(0.0, parsed))
