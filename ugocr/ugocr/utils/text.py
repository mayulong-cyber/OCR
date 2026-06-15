from __future__ import annotations

import re
import unicodedata


def normalize_cjk(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def join_lines(lines: list[str], rtl: bool = False) -> str:
    cleaned = [line.strip() for line in lines if line.strip()]
    if rtl:
        return "\n".join(cleaned)
    return "\n".join(cleaned)


def extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    first_obj = stripped.find("{")
    first_arr = stripped.find("[")
    candidates = [idx for idx in [first_obj, first_arr] if idx >= 0]
    if not candidates:
        raise ValueError("Model response does not contain JSON.")
    start = min(candidates)
    opening = stripped[start]
    closing = "}" if opening == "{" else "]"
    end = stripped.rfind(closing)
    if end < start:
        raise ValueError("Model response JSON is incomplete.")
    return stripped[start : end + 1]
