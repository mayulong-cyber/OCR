from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from PIL import Image

from ugocr.config import VLMSettings
from ugocr.dependencies import import_required
from ugocr.ocr.types import InternalTableCell, OCRLine, VLMDebugMetadata
from ugocr.utils.images import (
    create_contact_sheet,
    crop_line_image,
    encode_image_data_url,
    load_image,
    prepare_vlm_image_bytes,
)
from ugocr.utils.text import extract_json


@dataclass(frozen=True)
class VLMLineResult:
    lines: list[str]
    corrected: bool
    warnings: list[str]
    vlm_debug: VLMDebugMetadata | None = None


@dataclass(frozen=True)
class VLMTableResult:
    cells: list[InternalTableCell]
    corrected: bool
    warnings: list[str]


class _LRUCache:
    def __init__(self, capacity: int) -> None:
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._capacity = max(1, capacity)

    def get(self, key: str) -> Any | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        return key in self._cache


class VLMPostProcessor:
    def __init__(self, settings: VLMSettings) -> None:
        self.settings = settings
        self._cache = _LRUCache(settings.cache_size)
        self._line_score_threshold = settings.line_score_threshold
        self._max_lines = settings.max_lines_per_image
        self._min_text_len = settings.min_text_len

    def correct_text_lines(self, language: str, image_bytes: bytes, lines: list[OCRLine]) -> VLMLineResult:
        original = [line.text for line in lines]
        if not self.settings.enabled:
            return VLMLineResult(
                lines=original,
                corrected=False,
                warnings=["VLM postprocess is disabled."],
                vlm_debug=VLMDebugMetadata(vlm_skipped_reason="VLM disabled"),
            )
        if len(lines) == 0:
            return VLMLineResult(
                lines=[],
                corrected=False,
                warnings=[],
                vlm_debug=VLMDebugMetadata(),
            )

        candidates = self._select_candidates(lines)
        if not candidates:
            return VLMLineResult(
                lines=original,
                corrected=False,
                warnings=["VLM skipped: no low-confidence lines."],
                vlm_debug=VLMDebugMetadata(vlm_skipped_reason="no low-confidence lines"),
            )

        return self._correct_with_contact_sheet(language, image_bytes, lines, candidates)

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
            '{"cells":[{"row":1,"col":1,"rowspan":1,"colspan":1,"text":"..."}]}.\n'
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

    def _select_candidates(self, lines: list[OCRLine]) -> list[tuple[int, OCRLine]]:
        scored = [
            (idx, line)
            for idx, line in enumerate(lines)
            if line.score < self._line_score_threshold and len(line.text.strip()) >= self._min_text_len
        ]
        scored.sort(key=lambda x: x[1].score)
        return scored[: self._max_lines]

    def _correct_with_contact_sheet(
        self,
        language: str,
        image_bytes: bytes,
        lines: list[OCRLine],
        candidates: list[tuple[int, OCRLine]],
    ) -> VLMLineResult:
        original = [line.text for line in lines]
        candidate_indices = [idx for idx, _ in candidates]
        candidate_texts = [line.text for _, line in candidates]

        cache_key = self._make_cache_key(image_bytes, candidate_texts, candidate_indices)
        cached = self._cache.get(cache_key)
        if cached is not None:
            result_texts, result_corrected, result_warnings = cached
            final = list(original)
            for i, idx in enumerate(candidate_indices):
                if i < len(result_texts):
                    final[idx] = result_texts[i]
            return VLMLineResult(
                lines=final,
                corrected=result_corrected,
                warnings=["VLM cache hit."] + result_warnings,
                vlm_debug=VLMDebugMetadata(
                    vlm_candidate_lines=candidate_indices,
                    vlm_called=False,
                    vlm_cache_hit=True,
                    vlm_skipped_reason="cache hit",
                ),
            )

        image_payload = load_image(image_bytes)
        pil_image = image_payload.image
        line_images: list[Image.Image] = []
        valid_candidates: list[tuple[int, OCRLine]] = []
        for idx, line in candidates:
            crop = crop_line_image(pil_image, line.box)
            if crop is not None:
                line_images.append(crop)
                valid_candidates.append((idx, line))

        if not valid_candidates:
            return VLMLineResult(
                lines=original,
                corrected=False,
                warnings=["VLM skipped: no valid line crops."],
                vlm_debug=VLMDebugMetadata(
                    vlm_candidate_lines=candidate_indices,
                    vlm_skipped_reason="no valid line crops",
                ),
            )

        contact_sheet_bytes = create_contact_sheet(line_images)
        if contact_sheet_bytes is None:
            return VLMLineResult(
                lines=original,
                corrected=False,
                warnings=["VLM skipped: contact sheet creation failed."],
                vlm_debug=VLMDebugMetadata(
                    vlm_candidate_lines=candidate_indices,
                    vlm_skipped_reason="contact sheet creation failed",
                ),
            )

        valid_texts = [line.text for _, line in valid_candidates]
        valid_indices = [idx for idx, _ in valid_candidates]
        payload = {
            "language": language,
            "candidates": [
                {"index": i, "text": line.text, "ocr_score": round(line.score, 4)}
                for i, (_, line) in enumerate(valid_candidates)
            ],
        }
        prompt = (
            "Handwriting OCR corrector. Image shows cropped lines labeled [0],[1],... "
            "Compare each image with OCR text. Only fix clearly wrong characters. "
            "Do not add/remove/translate. If unsure, keep original and set changed=false. "
            "Return JSON array, one per candidate, same order. No markdown.\n"
            '[{"text":"...","confidence":0.0,"changed":false,"reason":"..."}]\n'
            f"OCR: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
        )

        contact_pil = load_image(contact_sheet_bytes).image
        input_size = (contact_pil.width, contact_pil.height)
        started_at = time.perf_counter()
        timeout_sec = self.settings.correction_timeout_sec
        try:
            response = self._chat(
                contact_sheet_bytes,
                prompt,
                max_tokens=self.settings.correction_max_tokens,
                timeout_override=timeout_sec,
            )
            elapsed = time.perf_counter() - started_at
            data = json.loads(extract_json(response))
            corrected_texts, accepted, rejected = self._apply_acceptance_rules(
                valid_texts, data
            )

            final = list(original)
            for i, idx in enumerate(valid_indices):
                if i < len(corrected_texts):
                    final[idx] = corrected_texts[i]

            any_changed = any(final[i] != original[i] for i in valid_indices)
            warnings = []
            for i, idx in enumerate(valid_indices):
                if i < len(accepted) and idx in accepted:
                    warnings.append(f"line {idx} accepted by VLM, confidence={accepted[idx]:.2f}")
                elif idx in rejected:
                    warnings.append(f"line {idx} rejected: {rejected[idx]}")
                else:
                    warnings.append(f"line {idx} skipped")
            warnings.append(f"VLM correction elapsed {elapsed:.3f}s; changed={str(any_changed).lower()}.")

            self._cache.put(cache_key, (corrected_texts, any_changed, warnings))
            return VLMLineResult(
                lines=final,
                corrected=any_changed,
                warnings=warnings,
                vlm_debug=VLMDebugMetadata(
                    vlm_candidate_lines=candidate_indices,
                    vlm_called=True,
                    vlm_cache_hit=False,
                    vlm_elapsed_sec=round(elapsed, 3),
                    vlm_timeout_sec=timeout_sec,
                    vlm_input_image_size=input_size,
                    vlm_accepted_lines=[idx for idx in valid_indices if idx in accepted],
                    vlm_rejected_lines=[idx for idx in valid_indices if idx in rejected],
                ),
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            return VLMLineResult(
                lines=original,
                corrected=False,
                warnings=[f"VLM correction failed after {elapsed:.3f}s: {exc}"],
                vlm_debug=VLMDebugMetadata(
                    vlm_candidate_lines=candidate_indices,
                    vlm_called=True,
                    vlm_cache_hit=False,
                    vlm_elapsed_sec=round(elapsed, 3),
                    vlm_timeout_sec=timeout_sec,
                    vlm_input_image_size=input_size,
                    vlm_skipped_reason=f"error: {exc}",
                ),
            )

    def _apply_acceptance_rules(
        self,
        original_texts: list[str],
        vlm_data: Any,
    ) -> tuple[list[str], dict[int, float], dict[int, str]]:
        if not isinstance(vlm_data, list):
            return list(original_texts), {}, {i: "invalid response format" for i in range(len(original_texts))}

        corrected = list(original_texts)
        accepted: dict[int, float] = {}
        rejected: dict[int, str] = {}

        for i, item in enumerate(vlm_data):
            if i >= len(original_texts):
                break
            if not isinstance(item, dict):
                rejected[i] = "invalid item format"
                continue

            text = str(item.get("text", "")).strip()
            confidence = _bounded_float(item.get("confidence"), 0.0)
            changed = bool(item.get("changed", False))

            if not text:
                rejected[i] = "empty text"
                continue
            if not changed:
                continue
            if confidence < 0.75:
                rejected[i] = f"low confidence ({confidence:.2f})"
                continue
            if abs(len(text) - len(original_texts[i])) > len(original_texts[i]) * 0.3:
                rejected[i] = "length difference > 30%"
                continue
            if _has_non_chinese_anomaly(text):
                rejected[i] = "non-Chinese characters detected"
                continue

            corrected[i] = text
            accepted[i] = confidence

        return corrected, accepted, rejected

    def _make_cache_key(self, image_bytes: bytes, texts: list[str], indices: list[int]) -> str:
        image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        content = json.dumps({"t": texts, "i": indices, "m": self.settings.model}, ensure_ascii=False, sort_keys=True)
        text_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{image_hash}:{text_hash}"

    def _chat(self, image_bytes: bytes, prompt: str, max_tokens: int | None = None, timeout_override: int | None = None) -> str:
        if self.settings.provider == "ollama":
            return self._chat_ollama(image_bytes, prompt, max_tokens=max_tokens, timeout_override=timeout_override)
        if self.settings.provider in {"openai", "vllm"}:
            return self._chat_openai_compatible(image_bytes, prompt, max_tokens=max_tokens, timeout_override=timeout_override)
        raise ValueError(f"Unsupported VLM provider: {self.settings.provider}")

    def _chat_openai_compatible(
        self, image_bytes: bytes, prompt: str, max_tokens: int | None = None, timeout_override: int | None = None
    ) -> str:
        requests = import_required("requests", "requirements-base.txt")
        url = f"{self.settings.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json"}
        timeout = timeout_override or self.settings.request_timeout_sec
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
        response = requests.post(url, headers=headers, json=body, timeout=(5, timeout))
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])

    def _chat_ollama(
        self, image_bytes: bytes, prompt: str, max_tokens: int | None = None, timeout_override: int | None = None
    ) -> str:
        import base64

        requests = import_required("requests", "requirements-base.txt")
        base_url = self.settings.base_url.rstrip("/")
        url = f"{base_url}/api/chat" if not base_url.endswith("/api/chat") else base_url
        timeout = timeout_override or self.settings.request_timeout_sec
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
        response = requests.post(url, json=body, timeout=(5, timeout))
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        if isinstance(message, dict) and "content" in message:
            return str(message["content"])
        if "response" in data:
            return str(data["response"])
        raise ValueError(f"Unexpected Ollama response: {data}")


def _has_non_chinese_anomaly(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:
            continue
        if 0x3400 <= cp <= 0x4DBF:
            continue
        if 0xF900 <= cp <= 0xFAFF:
            continue
        if ch in " \t\n\r":
            continue
        if 0x20 <= cp <= 0x7E:
            return True
        if 0x3000 <= cp <= 0x303F:
            continue
        if 0xFF00 <= cp <= 0xFFEF:
            continue
        if cp in (0x3001, 0x3002, 0xFF0C, 0xFF1A, 0xFF1B, 0xFF01, 0xFF1F):
            continue
        if 0x2E80 <= cp <= 0x2EFF:
            continue
        if 0x31C0 <= cp <= 0x31EF:
            continue
        return True
    return False


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
