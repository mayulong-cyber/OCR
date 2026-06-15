from __future__ import annotations

from ugocr.config import Settings
from ugocr.ocr.paddle_engine import PaddleOCRTextEngine
from ugocr.ocr.postprocess import VLMPostProcessor
from ugocr.ocr.types import OCRLine, PipelineResult, VLMDebugMetadata
from ugocr.utils.images import load_image, sort_text_boxes
from ugocr.utils.text import join_lines, normalize_cjk


class ChineseHandwritingPipeline:
    def __init__(self, settings: Settings, postprocessor: VLMPostProcessor) -> None:
        self.settings = settings
        self.postprocessor = postprocessor
        self.engine = PaddleOCRTextEngine(
            runtime=settings.runtime,
            paddle=settings.paddle,
            rec_model_dir=settings.paddle.chinese_rec_model_dir,
            char_dict_path=settings.paddle.chinese_char_dict_path,
            engine_name="ppocrv5-svtrv2-chinese",
        )

    def warmup(self) -> None:
        self.engine.warmup()

    def recognize(self, image_bytes: bytes, use_vlm: bool | None = None) -> PipelineResult:
        payload = load_image(image_bytes)
        raw_lines = self.engine.recognize(payload.image)
        sorted_records = sort_text_boxes(
            [{"text": line.text, "score": line.score, "box": line.box, "engine": line.engine} for line in raw_lines],
        )
        lines = [
            OCRLine(
                text=normalize_cjk(str(item["text"])),
                score=float(item["score"]),
                box=item["box"],
                engine=str(item["engine"]),
            )
            for item in sorted_records
            if str(item["text"]).strip()
        ]
        if use_vlm is not True:
            text = join_lines([line.text for line in lines])
            return PipelineResult(
                text=text,
                lines=lines,
                corrected=False,
                warnings=["VLM postprocess skipped by request."],
                vlm_debug=VLMDebugMetadata(vlm_skipped_reason="use_vlm=false"),
            )
        vlm_result = self.postprocessor.correct_text_lines("zh", image_bytes, lines)
        final_lines = [
            OCRLine(text=normalize_cjk(vlm_result.lines[idx]), score=line.score, box=line.box, engine=line.engine)
            for idx, line in enumerate(lines)
        ]
        return PipelineResult(
            text=join_lines([line.text for line in final_lines]),
            lines=final_lines,
            corrected=vlm_result.corrected,
            warnings=vlm_result.warnings,
            vlm_debug=vlm_result.vlm_debug,
        )
