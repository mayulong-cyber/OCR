from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from ugocr.config import PROJECT_ROOT, Settings
from ugocr.schemas import ModelCheck


PADDLE_INFERENCE_REQUIRED = ("inference.pdiparams", "inference.yml")
PADDLE_INFERENCE_PRIMARY = ("inference.pdmodel", "inference.json")


def check_models(settings: Settings) -> list[ModelCheck]:
    checks = [
        _check_paddle_model_dir("PP-OCRv5 Det", settings.paddle.det_model_dir),
        _check_paddle_model_dir("Chinese SVTRv2 Rec", settings.paddle.chinese_rec_model_dir),
        _check_paddle_model_dir("Table Rec", settings.paddle.table_rec_model_dir),
        _check_paddle_model_dir("SLANet Table", settings.paddle.table_model_dir),
    ]
    if settings.paddle.chinese_char_dict_path is not None:
        checks.append(_check_file("Chinese character dictionary", settings.paddle.chinese_char_dict_path))
    if settings.paddle.table_char_dict_path is not None:
        checks.append(_check_file("Table character dictionary", settings.paddle.table_char_dict_path))
    if settings.vlm.enabled and settings.vlm.provider in {"openai", "vllm"}:
        checks.append(_check_dir("Qwen2.5-VL AWQ", PROJECT_ROOT / "models" / settings.vlm.model, ("config.json",)))
    if settings.vlm.enabled and settings.vlm.provider == "ollama":
        checks.append(_check_ollama(settings.vlm.base_url, settings.vlm.model))
    return checks


def models_ok(checks: list[ModelCheck]) -> bool:
    return all(item.exists for item in checks)


def _check_file(label: str, path: Path) -> ModelCheck:
    return ModelCheck(label=label, path=str(path), exists=path.is_file(), detail="file found" if path.is_file() else "file missing")


def _check_dir(label: str, path: Path, candidate_files: tuple[str, ...], *, require_all: bool = True) -> ModelCheck:
    if not path.exists():
        return ModelCheck(label=label, path=str(path), exists=False, detail="directory missing")
    if not path.is_dir():
        return ModelCheck(label=label, path=str(path), exists=False, detail="path is not a directory")
    found = [name for name in candidate_files if (path / name).exists()]
    if require_all:
        if len(found) == len(candidate_files):
            return ModelCheck(label=label, path=str(path), exists=True, detail=f"found {', '.join(found)}")
        missing = [name for name in candidate_files if name not in found]
        return ModelCheck(label=label, path=str(path), exists=False, detail=f"missing: {', '.join(missing)}")
    if found:
        return ModelCheck(label=label, path=str(path), exists=True, detail=f"found {found[0]}")
    joined = ", ".join(candidate_files)
    return ModelCheck(label=label, path=str(path), exists=False, detail=f"missing one of: {joined}")


def _check_paddle_model_dir(label: str, path: Path) -> ModelCheck:
    if not path.exists():
        return ModelCheck(label=label, path=str(path), exists=False, detail="directory missing")
    if not path.is_dir():
        return ModelCheck(label=label, path=str(path), exists=False, detail="path is not a directory")
    missing_required = [name for name in PADDLE_INFERENCE_REQUIRED if not (path / name).exists()]
    if missing_required:
        return ModelCheck(label=label, path=str(path), exists=False, detail=f"missing: {', '.join(missing_required)}")
    if any((path / name).exists() for name in PADDLE_INFERENCE_PRIMARY):
        present = [name for name in PADDLE_INFERENCE_REQUIRED if (path / name).exists()]
        primary = next(name for name in PADDLE_INFERENCE_PRIMARY if (path / name).exists())
        return ModelCheck(label=label, path=str(path), exists=True, detail=f"found {primary}, {', '.join(present)}")
    return ModelCheck(label=label, path=str(path), exists=False, detail=f"missing one of: {', '.join(PADDLE_INFERENCE_PRIMARY)}")


def _check_ollama(base_url: str, model: str) -> ModelCheck:
    root = base_url.rstrip("/")
    try:
        with urlopen(f"{root}/api/tags", timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return ModelCheck(label="Ollama VLM endpoint", path=root, exists=False, detail=f"HTTP {exc.code}: {exc.reason}")
    except URLError as exc:
        return ModelCheck(label="Ollama VLM endpoint", path=root, exists=False, detail=f"unreachable: {exc.reason}")
    except Exception as exc:
        return ModelCheck(label="Ollama VLM endpoint", path=root, exists=False, detail=f"invalid response: {exc}")

    models = [str(item.get("name", "")) for item in payload.get("models", []) if isinstance(item, dict)]
    if model in models:
        return ModelCheck(label="Ollama VLM endpoint", path=root, exists=True, detail=f"model ready: {model}")
    if models:
        return ModelCheck(label="Ollama VLM endpoint", path=root, exists=False, detail=f"model missing: {model}; available: {', '.join(models[:8])}")
    return ModelCheck(label="Ollama VLM endpoint", path=root, exists=False, detail=f"model missing: {model}; no local models reported")
