from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ugocr.config import VLMSettings
from ugocr.dependencies import import_required
from ugocr.utils.text import extract_json


SUPPORTED_LANGS = {"uy", "ch", "kz"}


@dataclass(frozen=True)
class TranslationResult:
    source_lang: str
    target_lang: str
    texts: list[str]
    translate_results: list[str]


class TextTranslator:
    def __init__(self, settings: VLMSettings) -> None:
        self.settings = settings

    def translate(
        self,
        texts: list[str],
        *,
        target_lang: str,
        source_lang: str | None = None,
    ) -> TranslationResult:
        cleaned = [text.strip() for text in texts]
        if len(cleaned) == 0:
            return TranslationResult(source_lang=source_lang or target_lang, target_lang=target_lang, texts=[], translate_results=[])

        inferred_source = source_lang or detect_batch_language(cleaned)
        if inferred_source == target_lang:
            return TranslationResult(
                source_lang=inferred_source,
                target_lang=target_lang,
                texts=cleaned,
                translate_results=cleaned,
            )
        if not self.settings.enabled:
            raise RuntimeError("VLM translation backend is disabled.")

        prompt = (
            "You are an offline translation engine. "
            "Translate each input item into the requested target language. "
            "Supported language codes are only uy, ch, kz. "
            "Do not explain. Do not add numbering. Preserve order and item count. "
            "Return JSON only: "
            "{\"source_lang\":\"uy\",\"translate_results\":[{\"index\":0,\"text\":\"...\"}]}.\n"
            f"target_lang={target_lang}\n"
            f"source_lang_hint={inferred_source}\n"
            f"texts={json.dumps([{'index': idx, 'text': text} for idx, text in enumerate(cleaned)], ensure_ascii=False)}"
        )
        response = self._chat_text(prompt, max_tokens=min(self.settings.max_tokens, 512))
        data = json.loads(extract_json(response))
        returned_source = normalize_lang_code(str(data.get("source_lang", inferred_source)))
        translated = cleaned[:]
        for item in data.get("translate_results", []):
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index"))
            except Exception:
                continue
            if 0 <= idx < len(translated):
                translated[idx] = str(item.get("text", translated[idx])).strip()
        return TranslationResult(
            source_lang=returned_source,
            target_lang=target_lang,
            texts=cleaned,
            translate_results=translated,
        )

    def _chat_text(self, prompt: str, max_tokens: int | None = None) -> str:
        if self.settings.provider == "ollama":
            return self._chat_text_ollama(prompt, max_tokens=max_tokens)
        if self.settings.provider in {"openai", "vllm"}:
            return self._chat_text_openai_compatible(prompt, max_tokens=max_tokens)
        raise ValueError(f"Unsupported VLM provider: {self.settings.provider}")

    def _chat_text_openai_compatible(self, prompt: str, max_tokens: int | None = None) -> str:
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
                    "content": prompt,
                }
            ],
        }
        response = requests.post(url, headers=headers, json=body, timeout=(5, self.settings.request_timeout_sec))
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])

    def _chat_text_ollama(self, prompt: str, max_tokens: int | None = None) -> str:
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


def normalize_lang_code(value: str) -> str:
    raw = value.strip().lower()
    aliases = {
        "zh": "ch",
        "ch": "ch",
        "uy": "uy",
        "ug": "uy",
        "kk": "kz",
        "kz": "kz",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported language code: {value}")
    return normalized


def detect_batch_language(texts: list[str]) -> str:
    counts = {"ch": 0, "uy": 0, "kz": 0}
    for text in texts:
        counts[detect_text_language(text)] += 1
    return max(counts, key=counts.get)


def detect_text_language(text: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", text):
        return "ch"
    if re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", text):
        return "uy"
    if re.search(r"[\u0400-\u04FF]", text):
        return "kz"
    return "ch"
