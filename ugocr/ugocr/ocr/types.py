from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OCRLine:
    text: str
    score: float
    box: list[tuple[float, float]]
    engine: str


@dataclass(frozen=True)
class VLMDebugMetadata:
    vlm_candidate_lines: list[int] = field(default_factory=list)
    vlm_called: bool = False
    vlm_cache_hit: bool = False
    vlm_elapsed_sec: float = 0.0
    vlm_timeout_sec: float = 0.0
    vlm_input_image_size: tuple[int, int] = (0, 0)
    vlm_accepted_lines: list[int] = field(default_factory=list)
    vlm_rejected_lines: list[int] = field(default_factory=list)
    vlm_skipped_reason: str = ""


@dataclass(frozen=True)
class PipelineResult:
    text: str
    lines: list[OCRLine]
    corrected: bool
    warnings: list[str] = field(default_factory=list)
    vlm_debug: VLMDebugMetadata | None = None


@dataclass(frozen=True)
class InternalTableCell:
    row: int
    col: int
    text: str
    score: float = 1.0
    rowspan: int = 1
    colspan: int = 1
    box: list[tuple[float, float]] = field(default_factory=list)

