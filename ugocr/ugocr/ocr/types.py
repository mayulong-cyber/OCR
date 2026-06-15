from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OCRLine:
    text: str
    score: float
    box: list[tuple[float, float]]
    engine: str


@dataclass(frozen=True)
class PipelineResult:
    text: str
    lines: list[OCRLine]
    corrected: bool
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InternalTableCell:
    row: int
    col: int
    text: str
    score: float = 1.0
    rowspan: int = 1
    colspan: int = 1
    box: list[tuple[float, float]] = field(default_factory=list)

