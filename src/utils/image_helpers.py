"""Shared image helpers: dimensions, palette extraction, perceptual dedupe."""

from __future__ import annotations

import io
import logging
import re
from io import BytesIO
from typing import Iterable, List, Optional, Tuple

import filetype
import imagehash
from PIL import Image

from src.models.brand_dna import ColorSwatch
from src.models.image_outputs import DiscoveredImage

logger = logging.getLogger("ImageHelpers")

ImageItem = Tuple[DiscoveredImage, bytes]
DEFAULT_PHASH_MAX_HAMMING = 8

_THUMB_SIZE = (96, 96)
_MAX_IMAGES_SAMPLED = 30
_DEFAULT_PALETTE_SIZE = 8
# Drop the outer ring of each thumbnail when sampling pixels. Product-page
# backgrounds (off-white, light grey) dominate the border and would otherwise
# crowd the palette with neutrals.
_FOREGROUND_CROP_PCT = 0.15
# Filter centroids that are effectively pure black/white/grey before they
# make it into the brand palette. These thresholds are intentionally tight —
# real garment colors that *happen* to be dark navy or off-white still survive.
_MIN_LIGHTNESS = 0.04
_MAX_LIGHTNESS = 0.93
_MAX_NEUTRAL_CHROMA = 8  # max(R,G,B) - min(R,G,B) below this == achromatic
# Quantize to more colors than we need, then prune neutrals / near-duplicates
# down to n_colors. Headroom prevents the palette from collapsing to greys
# when the brand has only a few distinctive hues.
_QUANTIZE_HEADROOM = 6
_DUP_DISTANCE = 40  # Manhattan distance below which two centroids collapse

_RESOLUTION_PATTERN = re.compile(r"^(\d+)x(\d+)$")


def parse_resolution(value: str) -> tuple[int, int]:
    match = _RESOLUTION_PATTERN.match(value.strip())
    if not match:
        raise ValueError(
            f"Invalid resolution format: {value!r}. Expected format like '512x512'."
        )
    return int(match.group(1)), int(match.group(2))


def meets_minimum(width: int, height: int, min_width: int, min_height: int) -> bool:
    return width >= min_width and height >= min_height


def get_image_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """Return (width, height) for raster image bytes; None for non-images,
    SVGs, or undecodable payloads.
    """
    kind = filetype.guess(data)
    if not kind or not kind.mime.startswith("image/"):
        return None
    if kind.mime == "image/svg+xml":
        return None

    try:
        with Image.open(BytesIO(data)) as image:
            return image.size
    except Exception:
        return None


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b))


def _load_foreground_thumbnail(image_bytes: bytes) -> Image.Image | None:
    """Decode -> RGB -> thumbnail -> center-crop. Returns None on decode failure.

    Cropping the outer border removes the bulk of PDP background pixels
    (white/off-white sweep) so brand colors can dominate the quantizer's
    centroids instead of being out-voted by neutrals.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - PIL throws many distinct types
        logger.debug("Skipping image during palette extraction: %s", exc)
        return None
    img.thumbnail(_THUMB_SIZE, Image.LANCZOS)
    w, h = img.size
    crop_x = int(w * _FOREGROUND_CROP_PCT)
    crop_y = int(h * _FOREGROUND_CROP_PCT)
    if w - 2 * crop_x < 8 or h - 2 * crop_y < 8:
        return img
    return img.crop((crop_x, crop_y, w - crop_x, h - crop_y))


def _is_neutral(r: int, g: int, b: int) -> bool:
    chroma = max(r, g, b) - min(r, g, b)
    if chroma <= _MAX_NEUTRAL_CHROMA:
        return True
    lightness = (max(r, g, b) + min(r, g, b)) / 2 / 255
    return lightness < _MIN_LIGHTNESS or lightness > _MAX_LIGHTNESS


def _too_similar(r: int, g: int, b: int, palette: List[ColorSwatch]) -> bool:
    return any(
        abs(r - sw.r) + abs(g - sw.g) + abs(b - sw.b) < _DUP_DISTANCE
        for sw in palette
    )


def extract_palette(
    image_byte_sources: Iterable[bytes],
    n_colors: int = _DEFAULT_PALETTE_SIZE,
) -> List[ColorSwatch]:
    """Return up to `n_colors` dominant brand colors across the provided images.

    Replaces the previous fixed-bucket histogram (which produced grey-dominated
    palettes that looked the same across brands) with Pillow's median-cut
    quantizer applied to a foreground-cropped pixel pool. Returns a shorter
    palette when fewer real colors survive — the grey-padding fallback is gone.
    """
    samples = list(image_byte_sources)[:_MAX_IMAGES_SAMPLED]
    pixel_pool: List[Image.Image] = []
    for raw in samples:
        cropped = _load_foreground_thumbnail(raw)
        if cropped is not None:
            pixel_pool.append(cropped)

    if not pixel_pool:
        return []

    total_w = sum(img.size[0] for img in pixel_pool)
    max_h = max(img.size[1] for img in pixel_pool)
    canvas = Image.new("RGB", (total_w, max_h))
    x_cursor = 0
    for img in pixel_pool:
        canvas.paste(img, (x_cursor, 0))
        x_cursor += img.size[0]

    request = max(n_colors + _QUANTIZE_HEADROOM, n_colors + 1)
    quantized = canvas.quantize(colors=request, method=Image.Quantize.MEDIANCUT)
    color_counts = quantized.getcolors(maxcolors=request * 2) or []
    raw_palette = quantized.getpalette() or []

    total_pixels = sum(count for count, _ in color_counts) or 1
    palette: List[ColorSwatch] = []
    for count, index in sorted(color_counts, key=lambda x: -x[0]):
        base = index * 3
        if base + 2 >= len(raw_palette):
            continue
        r, g, b = raw_palette[base], raw_palette[base + 1], raw_palette[base + 2]
        if _is_neutral(r, g, b):
            continue
        if _too_similar(r, g, b, palette):
            continue
        palette.append(
            ColorSwatch(
                hex=_rgb_to_hex(r, g, b),
                r=int(r),
                g=int(g),
                b=int(b),
                frequency_pct=round(count / total_pixels * 100, 1),
            )
        )
        if len(palette) >= n_colors:
            break

    # If the chromatic pass returned very few colors (e.g. truly monochrome
    # brand imagery), let some neutrals back in so the report still shows a
    # palette — but never the synthetic #CCCCCC pad of the old code.
    if len(palette) < min(n_colors, 3):
        for count, index in sorted(color_counts, key=lambda x: -x[0]):
            base = index * 3
            if base + 2 >= len(raw_palette):
                continue
            r, g, b = raw_palette[base], raw_palette[base + 1], raw_palette[base + 2]
            if _too_similar(r, g, b, palette):
                continue
            palette.append(
                ColorSwatch(
                    hex=_rgb_to_hex(r, g, b),
                    r=int(r),
                    g=int(g),
                    b=int(b),
                    frequency_pct=round(count / total_pixels * 100, 1),
                )
            )
            if len(palette) >= n_colors:
                break

    return palette[:n_colors]


def phash_from_bytes(data: bytes) -> Optional[imagehash.ImageHash]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            return imagehash.phash(image)
    except Exception as exc:  # noqa: BLE001 - PIL throws many distinct types
        logger.debug("phash: could not hash image bytes: %s", exc)
        return None


def _pixel_area(image: DiscoveredImage) -> int:
    return image.width * image.height


def _match_index(
    digest: imagehash.ImageHash,
    representatives: List[imagehash.ImageHash],
    max_hamming: int,
) -> Optional[int]:
    for index, representative in enumerate(representatives):
        if digest - representative <= max_hamming:
            return index
    return None


def dedupe_by_phash(
    items: List[ImageItem],
    *,
    max_hamming: int = DEFAULT_PHASH_MAX_HAMMING,
) -> List[ImageItem]:
    """Collapse near-duplicate images; keep the highest-resolution variant."""
    if not items:
        return []

    kept: List[ImageItem] = []
    rep_hashes: List[imagehash.ImageHash] = []
    dropped = 0

    for image, data in items:
        digest = phash_from_bytes(data)
        if digest is None:
            kept.append((image, data))
            continue

        match_index = _match_index(digest, rep_hashes, max_hamming)
        if match_index is None:
            kept.append((image, data))
            rep_hashes.append(digest)
            continue

        existing_image, _ = kept[match_index]
        if _pixel_area(image) > _pixel_area(existing_image):
            kept[match_index] = (image, data)
            rep_hashes[match_index] = digest
        dropped += 1

    if dropped:
        logger.info(
            "phash dedupe: removed %s/%s near-duplicates (hamming<=%s)",
            dropped,
            len(items),
            max_hamming,
        )

    return kept
