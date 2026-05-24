import pytest
from io import BytesIO

from PIL import Image, ImageDraw

from src.models.image_outputs import DiscoveredImage
from src.utils.image_helpers import (
    dedupe_by_phash,
    extract_palette,
    meets_minimum,
    parse_resolution,
    phash_from_bytes,
)


def test_parse_resolution_valid():
    assert parse_resolution("512x512") == (512, 512)


@pytest.mark.parametrize("value", ["512", "abc", ""])
def test_parse_resolution_invalid(value):
    with pytest.raises(ValueError):
        parse_resolution(value)


def test_meets_minimum_exact_match():
    assert meets_minimum(512, 512, 512, 512) is True


def test_meets_minimum_width_too_small():
    assert meets_minimum(511, 512, 512, 512) is False


def test_meets_minimum_larger_dimensions():
    assert meets_minimum(800, 600, 512, 512) is True


def _image_bytes(width: int, height: int, color: str = "red") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _pattern_image_bytes(width: int, height: int, pattern: str) -> bytes:
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    if pattern == "circle":
        draw.ellipse((32, 32, width - 32, height - 32), fill="red")
    elif pattern == "stripes":
        for x in range(0, width, 32):
            draw.rectangle((x, 0, x + 16, height), fill="blue")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _discovered(url: str, width: int, height: int) -> DiscoveredImage:
    return DiscoveredImage(
        url=url,
        width=width,
        height=height,
        source_page="https://example.com/product",
    )


def test_phash_from_bytes_returns_hash_for_valid_image():
    digest = phash_from_bytes(_image_bytes(512, 512))
    assert digest is not None


def test_phash_from_bytes_returns_none_for_invalid_payload():
    assert phash_from_bytes(b"not-an-image") is None


def test_dedupe_collapses_identical_bytes_with_different_urls():
    payload = _image_bytes(512, 512)
    items = [
        (_discovered("https://cdn.example.com/t_default/hero.jpg", 512, 512), payload),
        (
            _discovered("https://cdn.example.com/t_PDP_1280_v1/hero.jpg", 512, 512),
            payload,
        ),
    ]

    deduped = dedupe_by_phash(items)

    assert len(deduped) == 1
    assert str(deduped[0][0].url) == "https://cdn.example.com/t_default/hero.jpg"


def test_dedupe_keeps_highest_resolution_variant():
    base = _image_bytes(512, 512)
    resized = _image_bytes(1280, 1280)
    items = [
        (_discovered("https://cdn.example.com/t_default/hero.jpg", 512, 512), base),
        (
            _discovered("https://cdn.example.com/t_PDP_1280_v1/hero.jpg", 1280, 1280),
            resized,
        ),
    ]

    deduped = dedupe_by_phash(items)

    assert len(deduped) == 1
    assert str(deduped[0][0].url) == "https://cdn.example.com/t_PDP_1280_v1/hero.jpg"


def test_dedupe_keeps_distinct_images():
    items = [
        (
            _discovered("https://cdn.example.com/a.jpg", 512, 512),
            _pattern_image_bytes(512, 512, "circle"),
        ),
        (
            _discovered("https://cdn.example.com/b.jpg", 512, 512),
            _pattern_image_bytes(512, 512, "stripes"),
        ),
    ]

    deduped = dedupe_by_phash(items)

    assert len(deduped) == 2


def test_dedupe_is_deterministic():
    payload = _image_bytes(640, 640)
    items = [
        (_discovered("https://cdn.example.com/one.jpg", 640, 640), payload),
        (_discovered("https://cdn.example.com/two.jpg", 640, 640), payload),
        (_discovered("https://cdn.example.com/three.jpg", 640, 640), payload),
    ]

    first = dedupe_by_phash(items)
    second = dedupe_by_phash(items)

    assert [str(image.url) for image, _ in first] == [
        str(image.url) for image, _ in second
    ]


def test_dedupe_preserves_unhashable_items():
    valid = _image_bytes(512, 512)
    items = [
        (_discovered("https://cdn.example.com/valid.jpg", 512, 512), valid),
        (_discovered("https://cdn.example.com/invalid.jpg", 512, 512), b"not-an-image"),
    ]

    deduped = dedupe_by_phash(items)

    assert len(deduped) == 2


def _foreground_image_bytes(
    width: int,
    height: int,
    fg_color: str,
    bg_color: str = "white",
) -> bytes:
    """Off-white border + colored center, mimicking a PDP product shot."""
    image = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)
    inset = max(width, height) // 5
    draw.rectangle(
        (inset, inset, width - inset, height - inset), fill=fg_color
    )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_extract_palette_returns_empty_for_no_decodable_images():
    assert extract_palette([b"not-an-image"], n_colors=8) == []


def test_extract_palette_ignores_white_background_and_returns_brand_color():
    # Big white border with a vivid red center — the old grey-padding code
    # would either pick the white (now filtered) or pad with #CCCCCC. The
    # new extractor should surface the red instead.
    payload = _foreground_image_bytes(256, 256, fg_color="red", bg_color="white")
    palette = extract_palette([payload], n_colors=4)

    assert len(palette) >= 1
    r, g, b = palette[0].r, palette[0].g, palette[0].b
    assert r > 180 and g < 80 and b < 80, (
        f"expected red-dominant centroid, got #{r:02X}{g:02X}{b:02X}"
    )
    # Crucially, no synthetic #CCCCCC grey-pad swatches.
    assert all(sw.hex != "#CCCCCC" for sw in palette)


def test_extract_palette_caps_at_n_colors_and_never_grey_pads():
    payloads = [
        _foreground_image_bytes(256, 256, fg_color=c, bg_color="white")
        for c in ("red", "blue", "green", "purple")
    ]
    palette = extract_palette(payloads, n_colors=8)

    assert 0 < len(palette) <= 8
    # No grey padding regardless of how few real colors exist.
    assert all(sw.hex != "#CCCCCC" for sw in palette)
