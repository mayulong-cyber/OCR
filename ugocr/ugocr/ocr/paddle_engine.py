from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image

from ugocr.config import PaddleSettings, RuntimeSettings
from ugocr.dependencies import import_required, require_path
from ugocr.ocr.types import OCRLine
from ugocr.utils.images import image_to_numpy


def _to_box(value: Any) -> list[tuple[float, float]]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    points: list[tuple[float, float]] = []
    for point in value:
        if hasattr(point, "tolist"):
            point = point.tolist()
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append((float(point[0]), float(point[1])))
    return points


def _looks_like_box(value: Any) -> bool:
    box = _to_box(value)
    return len(box) >= 4


def _score(value: Any, default: float = 1.0) -> float:
    try:
        number = float(value)
    except Exception:
        number = default
    return min(1.0, max(0.0, number))


def parse_paddle_ocr_result(result: Any, engine_name: str) -> list[OCRLine]:
    lines: list[OCRLine] = []

    def parse_node(node: Any) -> None:
        if node is None:
            return
        if hasattr(node, "json") and isinstance(getattr(node, "json"), dict):
            parse_node(getattr(node, "json"))
            return
        if isinstance(node, dict):
            texts = node.get("rec_texts") or node.get("texts")
            scores = node.get("rec_scores") or node.get("scores") or []
            boxes = node.get("dt_polys") or node.get("rec_polys") or node.get("boxes") or node.get("polys") or []
            if texts is not None:
                for idx, text in enumerate(texts):
                    box = _to_box(boxes[idx] if idx < len(boxes) else [])
                    score = _score(scores[idx] if idx < len(scores) else 1.0)
                    lines.append(OCRLine(text=str(text), score=score, box=box, engine=engine_name))
                return
            for key in ("data", "res", "result", "ocr"):
                if key in node:
                    parse_node(node[key])
            return
        if isinstance(node, (list, tuple)):
            if len(node) == 2 and _looks_like_box(node[0]):
                text_part = node[1]
                text = ""
                score = 1.0
                if isinstance(text_part, (list, tuple)) and len(text_part) >= 1:
                    text = str(text_part[0])
                    if len(text_part) >= 2:
                        score = _score(text_part[1])
                elif isinstance(text_part, str):
                    text = text_part
                if text:
                    lines.append(OCRLine(text=text, score=score, box=_to_box(node[0]), engine=engine_name))
                    return
            for child in node:
                parse_node(child)

    parse_node(result)
    return lines


def parse_paddle_detection_result(result: Any) -> list[list[tuple[float, float]]]:
    boxes: list[list[tuple[float, float]]] = []

    def parse_node(node: Any) -> None:
        if node is None:
            return
        if hasattr(node, "json") and isinstance(getattr(node, "json"), dict):
            parse_node(getattr(node, "json"))
            return
        if isinstance(node, dict):
            for key in ("dt_polys", "boxes", "polys"):
                if key in node and isinstance(node[key], (list, tuple)):
                    for value in node[key]:
                        box = _to_box(value)
                        if len(box) >= 4:
                            boxes.append(box)
                    return
            for key in ("data", "res", "result"):
                if key in node:
                    parse_node(node[key])
            return
        if isinstance(node, (list, tuple)):
            if _looks_like_box(node):
                box = _to_box(node)
                if len(box) >= 4:
                    boxes.append(box)
                    return
            for child in node:
                parse_node(child)

    parse_node(result)
    return boxes


class PaddleOCRTextEngine:
    def __init__(
        self,
        runtime: RuntimeSettings,
        paddle: PaddleSettings,
        rec_model_dir: Path,
        char_dict_path: Path | None,
        engine_name: str,
    ) -> None:
        self.runtime = runtime
        self.paddle = paddle
        self.rec_model_dir = rec_model_dir
        self.char_dict_path = char_dict_path
        self.engine_name = engine_name
        self._engine: Any | None = None

    def _load(self) -> Any:
        if self._engine is not None:
            return self._engine
        require_path(self.paddle.det_model_dir, "PP-OCRv5 detection model directory")
        require_path(self.rec_model_dir, f"{self.engine_name} recognition model directory")
        if self.char_dict_path is not None:
            require_path(self.char_dict_path, f"{self.engine_name} character dictionary")
        self._engine = create_paddlex_ocr_pipeline(
            runtime=self.runtime,
            paddle=self.paddle,
            rec_model_dir=self.rec_model_dir,
        )
        return self._engine

    def warmup(self) -> None:
        self._load()

    def recognize(self, image: Image.Image) -> list[OCRLine]:
        engine = self._load()
        np_image = image_to_numpy(image)
        result = list(
            engine.predict(
                np_image,
                use_textline_orientation=False,
                text_det_limit_side_len=self.paddle.det_limit_side_len,
                text_det_limit_type="max",
            )
        )
        return parse_paddle_ocr_result(result, self.engine_name)


class PaddleDetector:
    def __init__(self, runtime: RuntimeSettings, paddle: PaddleSettings) -> None:
        self.runtime = runtime
        self.paddle = paddle
        self._engine: Any | None = None

    def _load(self) -> Any:
        if self._engine is not None:
            return self._engine
        require_path(self.paddle.det_model_dir, "PP-OCRv5 detection model directory")
        configure_paddle_runtime_env(self.runtime)
        paddleocr = import_required("paddleocr", "requirements-gpu.txt or requirements-cpu.txt")
        detector_cls = getattr(paddleocr, "TextDetection")
        init_kwargs: dict[str, Any] = {}
        device = _paddlex_device(self.runtime)
        if device != "cpu":
            init_kwargs["device"] = device
        self._engine = detector_cls(
            model_name="PP-OCRv5_server_det",
            model_dir=str(self.paddle.det_model_dir),
            limit_side_len=self.paddle.det_limit_side_len,
            limit_type="max",
            thresh=0.3,
            box_thresh=0.6,
            unclip_ratio=1.5,
            **init_kwargs,
        )
        return self._engine

    def detect(self, image: Image.Image) -> list[list[tuple[float, float]]]:
        engine = self._load()
        np_image = image_to_numpy(image)[:, :, ::-1].copy()
        result = list(engine.predict(np_image))
        boxes = parse_paddle_detection_result(result)
        return boxes


def create_paddlex_ocr_pipeline(
    runtime: RuntimeSettings,
    paddle: PaddleSettings,
    rec_model_dir: Path,
) -> Any:
    configure_paddle_runtime_env(runtime)
    paddlex_inference = import_required("paddlex.inference", "requirements-gpu.txt or requirements-cpu.txt")
    create_pipeline = getattr(paddlex_inference, "create_pipeline")
    pp_option = _paddlex_predictor_option(paddlex_inference, runtime)
    try:
        return create_pipeline(
            config=paddlex_ocr_config(paddle, rec_model_dir),
            device=None if _paddlex_device(runtime) == "cpu" else _paddlex_device(runtime),
            pp_option=pp_option,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not initialize PaddleX OCR pipeline: {exc}") from exc


def paddlex_ocr_config(paddle: PaddleSettings, rec_model_dir: Path) -> dict[str, Any]:
    return {
        "pipeline_name": "OCR",
        "text_type": "general",
        "use_doc_preprocessor": False,
        "use_textline_orientation": False,
        "SubModules": {
            "TextDetection": {
                "module_name": "text_detection",
                "model_name": "PP-OCRv5_server_det",
                "model_dir": str(paddle.det_model_dir),
                "limit_side_len": paddle.det_limit_side_len,
                "limit_type": "max",
                "max_side_limit": max(4000, paddle.det_limit_side_len),
                "thresh": 0.3,
                "box_thresh": 0.6,
                "unclip_ratio": 1.5,
            },
            "TextRecognition": {
                "module_name": "text_recognition",
                "model_name": "PP-OCRv5_server_rec",
                "model_dir": str(rec_model_dir),
                "batch_size": 6,
                "score_thresh": 0.0,
            },
        },
    }


def _paddlex_device(runtime: RuntimeSettings) -> str:
    requested = (runtime.device or "").strip().lower()
    if requested.startswith("cuda"):
        return requested.replace("cuda", "gpu", 1)
    if requested.startswith("gpu"):
        return requested
    if requested in {"cpu", "xpu", "npu", "mlu"}:
        return requested
    return "gpu" if runtime.use_gpu else "cpu"


def configure_paddle_runtime_env(runtime: RuntimeSettings) -> None:
    device = _paddlex_device(runtime)
    if device == "cpu":
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        os.environ.setdefault("FLAGS_use_onednn", "0")
        os.environ["FLAGS_enable_pir_api"] = "0"
        return
    if device.startswith("gpu"):
        paddle = import_required("paddle", "requirements-gpu.txt")
        paddle_version = getattr(paddle, "__version__", "unknown")
        if not paddle.device.is_compiled_with_cuda():
            raise RuntimeError(
                "GPU was requested, but the installed PaddlePaddle build is CPU-only "
                f"(paddle={paddle_version}, compiled_with_cuda=False). "
                "Install paddlepaddle-gpu in this Python environment, then rerun the smoke test."
            )
        device_count = None
        try:
            device_count = int(paddle.device.cuda.device_count())
        except Exception:
            device_count = None
        if device_count == 0:
            raise RuntimeError(
                "GPU was requested and PaddlePaddle has CUDA support, but Paddle found no CUDA devices. "
                "Check the NVIDIA driver and CUDA runtime visibility, then rerun the smoke test."
            )


def _paddlex_predictor_option(paddle_models: Any, runtime: RuntimeSettings) -> Any:
    PaddlePredictorOption = getattr(paddle_models, "PaddlePredictorOption")
    device = _paddlex_device(runtime)
    if device == "cpu":
        return PaddlePredictorOption(
            run_mode="paddle",
            enable_new_ir=False,
            enable_cinn=False,
            cpu_threads=1,
        )
    return PaddlePredictorOption(device_type=device)
