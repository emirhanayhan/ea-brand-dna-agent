from src.models.image_outputs import DiscoveredImage
from src.utils.fashion_filter import (
    ClipFashionFilter,
    HeuristicFashionFilter,
    NullFashionFilter,
    make_fashion_filter,
)


def _img(width: int, height: int) -> DiscoveredImage:
    return DiscoveredImage(
        url="https://cdn.example.com/x.jpg",
        width=width,
        height=height,
        source_page="https://example.com/p",
    )


def test_null_filter_is_identity():
    items = [(_img(800, 800), b"a"), (_img(100, 1000), b"b")]
    assert NullFashionFilter().filter(items) == items


def test_heuristic_drops_extreme_aspect_ratios():
    items = [
        (_img(800, 800), b"square"),
        (_img(1600, 200), b"wide-banner"),       # 8.0 aspect
        (_img(200, 1200), b"tall-promo-strip"),  # 0.17 aspect
        (_img(900, 1200), b"portrait-product"),  # 0.75 aspect
    ]
    kept = HeuristicFashionFilter().filter(items)
    kept_bytes = [data for _, data in kept]
    assert b"square" in kept_bytes
    assert b"portrait-product" in kept_bytes
    assert b"wide-banner" not in kept_bytes
    assert b"tall-promo-strip" not in kept_bytes


def test_heuristic_keeps_everything_when_aspects_are_normal():
    items = [(_img(800, 1000), b"a"), (_img(900, 900), b"b")]
    assert HeuristicFashionFilter().filter(items) == items


def test_factory_returns_null_when_configured():
    f = make_fashion_filter({"brand_dna_fashion_filter_strategy": "null"})
    assert isinstance(f, NullFashionFilter)


def test_factory_returns_heuristic_by_default():
    assert isinstance(make_fashion_filter({}), HeuristicFashionFilter)


def test_factory_returns_clip_when_configured():
    f = make_fashion_filter({"brand_dna_fashion_filter_strategy": "clip"})
    assert isinstance(f, ClipFashionFilter)


def test_clip_filter_is_passthrough_when_dependencies_missing():
    """When open_clip / torch isn't installed, ClipFashionFilter must NOT
    crash the run — it logs once and returns input unchanged. This is the
    'graceful failure' the case-study brief calls for.
    """
    f = ClipFashionFilter()
    f._available = False  # simulate failed import
    items = [(_img(800, 800), b"a"), (_img(900, 900), b"b")]
    assert f.filter(items) == items
