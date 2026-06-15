from __future__ import annotations

import importlib
import os
import site
import sys
from types import ModuleType
from pathlib import Path


_DLL_DIRECTORY_HANDLES: list[object] = []
_CUDA_DLL_DIRS_ADDED: list[Path] = []
_CUDA_DLL_DIRS_CONFIGURED = False


def import_required(module_name: str, install_hint: str) -> ModuleType:
    if module_name.split(".", 1)[0] in {"torch", "paddle", "paddleocr", "paddlex"}:
        configure_windows_cuda_dll_dirs()
    if module_name.split(".", 1)[0] in {"paddleocr", "paddlex"}:
        install_offline_modelscope_stub()
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Missing runtime dependency '{module_name}'. Install offline wheel set: {install_hint}. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def install_offline_modelscope_stub() -> None:
    if os.getenv("UGOCR_USE_REAL_MODELSCOPE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    if "modelscope" in sys.modules:
        return

    class _ModelScopeNotExistError(Exception):
        """Local stand-in for modelscope.hub.errors.NotExistError."""

    class _ModelScopeHTTPError(Exception):
        """Local stand-in for modelscope.hub.errors.HTTPError."""

    def _offline_snapshot_download(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(
            "ModelScope downloads are disabled in offline UGOCR mode. "
            "Place model files under models/ and point configs/ugocr.ollama.yaml to those directories."
        )

    modelscope_module = ModuleType("modelscope")
    hub_module = ModuleType("modelscope.hub")
    errors_module = ModuleType("modelscope.hub.errors")
    errors_module.NotExistError = _ModelScopeNotExistError
    errors_module.HTTPError = _ModelScopeHTTPError
    modelscope_module.snapshot_download = _offline_snapshot_download
    modelscope_module.hub = hub_module
    hub_module.errors = errors_module
    sys.modules["modelscope"] = modelscope_module
    sys.modules["modelscope.hub"] = hub_module
    sys.modules["modelscope.hub.errors"] = errors_module


def configure_windows_cuda_dll_dirs() -> list[Path]:
    global _CUDA_DLL_DIRS_CONFIGURED
    if _CUDA_DLL_DIRS_CONFIGURED or os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return list(_CUDA_DLL_DIRS_ADDED)

    added: list[Path] = []
    for root in _candidate_site_package_roots():
        for directory in _candidate_cuda_dll_dirs(root):
            if not directory.exists():
                continue
            try:
                handle = os.add_dll_directory(str(directory))
            except OSError:
                continue
            _DLL_DIRECTORY_HANDLES.append(handle)
            _CUDA_DLL_DIRS_ADDED.append(directory)
            added.append(directory)
    _CUDA_DLL_DIRS_CONFIGURED = True
    return added


def _candidate_site_package_roots() -> list[Path]:
    roots: list[Path] = []
    raw_values: list[str] = []
    try:
        raw_values.extend(site.getsitepackages())
    except Exception:
        raw_values.extend([])
    try:
        raw_values.append(site.getusersitepackages())
    except Exception:
        raw_values.extend([])
    raw_values.extend(sys.path)
    for raw in raw_values:
        if not raw:
            continue
        path = Path(raw)
        if path.exists() and path.name == "site-packages" and path not in roots:
            roots.append(path)
    return roots


def _candidate_cuda_dll_dirs(site_packages: Path) -> list[Path]:
    dirs = [site_packages / "torch" / "lib"]
    nvidia_root = site_packages / "nvidia"
    if nvidia_root.exists():
        dirs.extend(sorted(path for path in nvidia_root.glob("*/bin") if path.is_dir()))
    return dirs


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def require_any_file(directory: Path, candidates: tuple[str, ...], label: str) -> Path:
    require_path(directory, label)
    for name in candidates:
        candidate = directory / name
        if candidate.exists():
            return candidate
    joined = ", ".join(candidates)
    raise FileNotFoundError(f"{label} must contain one of: {joined}. Directory: {directory}")
