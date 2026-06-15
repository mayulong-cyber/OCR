from __future__ import annotations

from threading import Lock

from ugocr.config import Settings
from ugocr.ocr.chinese import ChineseHandwritingPipeline
from ugocr.ocr.postprocess import VLMPostProcessor
from ugocr.ocr.table import TablePipeline


class PipelineFactory:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.postprocessor = VLMPostProcessor(settings.vlm)
        self._lock = Lock()
        self._chinese: ChineseHandwritingPipeline | None = None
        self._table: TablePipeline | None = None

    @property
    def chinese(self) -> ChineseHandwritingPipeline:
        with self._lock:
            if self._chinese is None:
                self._chinese = ChineseHandwritingPipeline(self.settings, self.postprocessor)
            return self._chinese

    @property
    def table(self) -> TablePipeline:
        with self._lock:
            if self._table is None:
                self._table = TablePipeline(self.settings, self.postprocessor)
            return self._table

    def warmup(self, *, include_table: bool = False) -> None:
        self.chinese.warmup()
        if include_table:
            self.table.warmup()
