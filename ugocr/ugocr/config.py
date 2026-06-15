from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    node: Any = data
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required when UGOCR_CONFIG points to a YAML file.") from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


@dataclass(frozen=True)
class RuntimeSettings:
    use_gpu: bool
    device: str
    lazy_load_models: bool
    upload_dir: Path
    output_dir: Path
    max_upload_mb: int


@dataclass(frozen=True)
class PaddleSettings:
    det_model_dir: Path
    chinese_rec_model_dir: Path
    table_rec_model_dir: Path
    table_model_dir: Path
    chinese_char_dict_path: Path | None
    table_char_dict_path: Path | None
    use_angle_cls: bool
    det_limit_side_len: int


@dataclass(frozen=True)
class VLMSettings:
    enabled: bool
    provider: str
    base_url: str
    model: str
    api_key: str
    max_tokens: int
    temperature: float
    request_timeout_sec: int
    line_score_threshold: float
    max_lines_per_image: int
    min_text_len: int
    correction_max_tokens: int
    correction_timeout_sec: int
    cache_size: int


@dataclass(frozen=True)
class AuthSettings:
    enabled: bool
    app_keys: list[str]


@dataclass(frozen=True)
class Settings:
    runtime: RuntimeSettings
    paddle: PaddleSettings
    vlm: VLMSettings
    auth: AuthSettings

    @classmethod
    def from_env(cls) -> "Settings":
        config_path = Path(os.getenv("UGOCR_CONFIG", "configs/ugocr.example.yaml"))
        data = _load_yaml(_project_path(config_path))

        runtime = RuntimeSettings(
            use_gpu=_env_bool("UGOCR_USE_GPU", bool(_nested(data, "runtime", "use_gpu", default=False))),
            device=os.getenv("UGOCR_DEVICE", str(_nested(data, "runtime", "device", default="cpu"))),
            lazy_load_models=_env_bool(
                "UGOCR_LAZY_LOAD_MODELS",
                bool(_nested(data, "runtime", "lazy_load_models", default=False)),
            ),
            upload_dir=_project_path(os.getenv("UGOCR_UPLOAD_DIR", _nested(data, "runtime", "upload_dir", default="data/uploads"))),
            output_dir=_project_path(os.getenv("UGOCR_OUTPUT_DIR", _nested(data, "runtime", "output_dir", default="data/outputs"))),
            max_upload_mb=_env_int("UGOCR_MAX_UPLOAD_MB", int(_nested(data, "runtime", "max_upload_mb", default=10))),
        )

        paddle = PaddleSettings(
            det_model_dir=_project_path(os.getenv("UGOCR_DET_MODEL_DIR", _nested(data, "paddle", "det_model_dir", default="models/ppocrv5_det"))),
            chinese_rec_model_dir=_project_path(
                os.getenv("UGOCR_CHINESE_REC_MODEL_DIR", _nested(data, "paddle", "chinese_rec_model_dir", default="models/ppocrv5_chinese_svtrv2_rec"))
            ),
            table_rec_model_dir=_project_path(
                os.getenv("UGOCR_TABLE_REC_MODEL_DIR", _nested(data, "paddle", "table_rec_model_dir", default="models/ppocrv5_table_rec"))
            ),
            table_model_dir=_project_path(os.getenv("UGOCR_TABLE_MODEL_DIR", _nested(data, "paddle", "table_model_dir", default="models/slanet"))),
            chinese_char_dict_path=_optional_path(os.getenv("UGOCR_CHINESE_DICT", _nested(data, "paddle", "chinese_char_dict_path", default=""))),
            table_char_dict_path=_optional_path(os.getenv("UGOCR_TABLE_DICT", _nested(data, "paddle", "table_char_dict_path", default=""))),
            use_angle_cls=_env_bool("UGOCR_USE_ANGLE_CLS", bool(_nested(data, "paddle", "use_angle_cls", default=True))),
            det_limit_side_len=_env_int("UGOCR_DET_LIMIT_SIDE_LEN", int(_nested(data, "paddle", "det_limit_side_len", default=1920))),
        )

        vlm = VLMSettings(
            enabled=_env_bool("UGOCR_VLM_ENABLED", bool(_nested(data, "vlm", "enabled", default=False))),
            provider=os.getenv("UGOCR_VLM_PROVIDER", str(_nested(data, "vlm", "provider", default="openai"))).strip().lower(),
            base_url=os.getenv("UGOCR_VLM_BASE_URL", str(_nested(data, "vlm", "base_url", default="http://localhost:8001/v1"))).rstrip("/"),
            model=os.getenv("UGOCR_VLM_MODEL", str(_nested(data, "vlm", "model", default="qwen2.5-vl-7b-instruct-awq"))),
            api_key=os.getenv("UGOCR_VLM_API_KEY", str(_nested(data, "vlm", "api_key", default="local-offline-key"))),
            max_tokens=_env_int("UGOCR_VLM_MAX_TOKENS", int(_nested(data, "vlm", "max_tokens", default=512))),
            temperature=float(os.getenv("UGOCR_VLM_TEMPERATURE", str(_nested(data, "vlm", "temperature", default=0.0)))),
            request_timeout_sec=_env_int("UGOCR_VLM_TIMEOUT_SEC", int(_nested(data, "vlm", "request_timeout_sec", default=120))),
            line_score_threshold=float(os.getenv("UGOCR_VLM_LINE_SCORE_THRESHOLD", str(_nested(data, "vlm", "line_score_threshold", default=0.92)))),
            max_lines_per_image=_env_int("UGOCR_VLM_MAX_LINES_PER_IMAGE", int(_nested(data, "vlm", "max_lines_per_image", default=3))),
            min_text_len=_env_int("UGOCR_VLM_MIN_TEXT_LEN", int(_nested(data, "vlm", "min_text_len", default=2))),
            correction_max_tokens=_env_int("UGOCR_VLM_CORRECTION_MAX_TOKENS", int(_nested(data, "vlm", "correction_max_tokens", default=128))),
            correction_timeout_sec=_env_int("UGOCR_VLM_CORRECTION_TIMEOUT_SEC", int(_nested(data, "vlm", "correction_timeout_sec", default=35))),
            cache_size=_env_int("UGOCR_VLM_CACHE_SIZE", int(_nested(data, "vlm", "cache_size", default=128))),
        )

        auth_enabled = _env_bool("UGOCR_AUTH_ENABLED", bool(_nested(data, "auth", "enabled", default=True)))
        raw_keys = _nested(data, "auth", "app_keys", default=[])
        app_keys = [str(k).strip() for k in raw_keys if str(k).strip()] if isinstance(raw_keys, list) else []
        auth = AuthSettings(
            enabled=auth_enabled,
            app_keys=app_keys,
        )

        runtime.upload_dir.mkdir(parents=True, exist_ok=True)
        runtime.output_dir.mkdir(parents=True, exist_ok=True)
        return cls(runtime=runtime, paddle=paddle, vlm=vlm, auth=auth)


def _optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return _project_path(text)
