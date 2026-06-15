from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Point = tuple[float, float]


class OCRBox(BaseModel):
    text: str
    score: float = Field(ge=0.0, le=1.0)
    box: list[Point]
    language: Literal["zh", "table"]
    engine: str


class OCRResponse(BaseModel):
    request_id: str
    language: Literal["zh"]
    text: str
    boxes: list[OCRBox]
    corrected: bool
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class TableCell(BaseModel):
    row: int
    col: int
    text: str
    score: float = Field(ge=0.0, le=1.0)
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    box: list[Point] = Field(default_factory=list)


class TableResponse(BaseModel):
    request_id: str
    xlsx_path: str
    rows: int
    cols: int
    cells: list[TableCell]
    corrected: bool
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    lazy_load_models: bool
    vlm_enabled: bool
    device: str
    log_level: str
    models_ok: bool


class ModelCheck(BaseModel):
    label: str
    path: str
    exists: bool
    detail: str


class ModelCheckResponse(BaseModel):
    ok: bool
    checks: list[ModelCheck]


class OCRRecognizeResponse(BaseModel):
    target_lang: str
    result: list[str]


class TranslateTextRequest(BaseModel):
    texts: list[str] | None = None
    text: str | None = None
    source_lang: str | None = None
    target_lang: str


class TranslateTextResponse(BaseModel):
    source_lang: str
    target_lang: str
    texts: list[str]
    translate_results: list[str]


class TableRecognizeCell(BaseModel):
    row: int
    col: int
    text: str
    rowspan: int = 1
    colspan: int = 1


class TableRecognizeResponse(BaseModel):
    target_lang: str
    result: list[TableRecognizeCell]


class ErrorResponse(BaseModel):
    code: str
    message: str
