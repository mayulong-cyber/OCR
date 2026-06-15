from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

from ugocr.dependencies import import_required


@dataclass(frozen=True)
class ImagePayload:
    image: Image.Image
    sha256: str
    width: int
    height: int


def load_image(image_bytes: bytes) -> ImagePayload:
    if len(image_bytes) == 0:
        raise ValueError("Uploaded image is empty.")
    digest = hashlib.sha256(image_bytes).hexdigest()
    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    return ImagePayload(image=rgb, sha256=digest, width=rgb.width, height=rgb.height)


def image_to_numpy(image: Image.Image):
    np = import_required("numpy", "requirements-base.txt")
    return np.array(image.convert("RGB"))


def encode_image_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    import base64

    payload = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{payload}"


def prepare_vlm_image_bytes(image_bytes: bytes, max_side: int = 960, jpeg_quality: int = 88) -> bytes:
    if max_side <= 0:
        return image_bytes

    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        if max(rgb.width, rgb.height) <= max_side:
            return image_bytes

        resized = rgb.copy()
        resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True, progressive=True)
        return buffer.getvalue()


def save_upload(image_bytes: bytes, upload_dir: Path, suffix: str) -> Path:
    payload = load_image(image_bytes)
    clean_suffix = suffix.lower().lstrip(".") or "png"
    target = upload_dir / f"{payload.sha256}.{clean_suffix}"
    if not target.exists():
        target.write_bytes(image_bytes)
    return target


def polygon_bounds(points: Iterable[tuple[float, float]], width: int, height: int) -> tuple[int, int, int, int]:
    pts = list(points)
    if len(pts) == 0:
        raise ValueError("Cannot compute bounds for an empty polygon.")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    left = max(0, math.floor(min(xs)))
    top = max(0, math.floor(min(ys)))
    right = min(width, math.ceil(max(xs)))
    bottom = min(height, math.ceil(max(ys)))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid polygon bounds: {(left, top, right, bottom)}")
    return left, top, right, bottom


def crop_polygon(image: Image.Image, polygon: list[tuple[float, float]]) -> Image.Image:
    if len(polygon) < 4:
        return image.crop(polygon_bounds(polygon, image.width, image.height))
    try:
        cv2 = import_required("cv2", "requirements-base.txt")
        np = import_required("numpy", "requirements-base.txt")
        pts = np.array(polygon[:4], dtype="float32")
        width_top = np.linalg.norm(pts[1] - pts[0])
        width_bottom = np.linalg.norm(pts[2] - pts[3])
        height_left = np.linalg.norm(pts[3] - pts[0])
        height_right = np.linalg.norm(pts[2] - pts[1])
        target_width = max(1, int(max(width_top, width_bottom)))
        target_height = max(1, int(max(height_left, height_right)))
        dst = np.array(
            [
                [0, 0],
                [target_width - 1, 0],
                [target_width - 1, target_height - 1],
                [0, target_height - 1],
            ],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(image_to_numpy(image), matrix, (target_width, target_height))
        return Image.fromarray(warped).convert("RGB")
    except Exception:
        return image.crop(polygon_bounds(polygon, image.width, image.height)).convert("RGB")


def sort_text_boxes(boxes: list[dict]) -> list[dict]:
    if len(boxes) <= 1:
        return boxes
    enriched: list[tuple[float, float, float, dict]] = []
    for item in boxes:
        pts = item.get("box") or []
        if len(pts) == 0:
            enriched.append((0.0, 0.0, 0.0, item))
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        min_x = min(xs)
        max_x = max(xs)
        center_y = sum(ys) / len(ys)
        height = max(1.0, max(ys) - min(ys))
        enriched.append((center_y, min_x, max_x, item | {"_sort_height": height}))

    median_height = sorted(v[3]["_sort_height"] for v in enriched)[len(enriched) // 2]
    row_tolerance = max(8.0, median_height * 0.55)
    rows: list[list[tuple[float, float, float, dict]]] = []
    for record in sorted(enriched, key=lambda value: value[0]):
        placed = False
        for row in rows:
            row_y = sum(item[0] for item in row) / len(row)
            if abs(record[0] - row_y) <= row_tolerance:
                row.append(record)
                placed = True
                break
        if not placed:
            rows.append([record])

    output: list[dict] = []
    for row in rows:
        row.sort(key=lambda value: value[1])
        for _, _, _, item in row:
            clean = dict(item)
            clean.pop("_sort_height", None)
            output.append(clean)
    return output


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
