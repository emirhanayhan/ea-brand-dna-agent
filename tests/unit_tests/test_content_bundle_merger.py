from datetime import datetime, timezone

import pytest

from src.models.brand_content import BrandContentBundle, BrandTextSnippet
from src.models.image_outputs import BotProtectionFailure, DiscoveredImage
from src.utils.content_bundle_merger import merge_content_bundles


def _image(url: str, source_page: str | None = None) -> DiscoveredImage:
    return DiscoveredImage(
        url=url,
        width=1024,
        height=1024,
        source_page=source_page or url,
    )


def _snippet(text: str, weight: float = 1.0, kind="editorial") -> BrandTextSnippet:
    return BrandTextSnippet(
        source_url="https://example.com/",
        kind=kind,
        text=text,
        weight=weight,
    )


def _bundle(
    source: str,
    *,
    brand_url: str = "https://example.com",
    images=(),
    texts=(),
    pages_crawled: int = 0,
    candidates_checked: int = 0,
    bot_protection: BotProtectionFailure | None = None,
) -> BrandContentBundle:
    return BrandContentBundle(
        source=source,
        brand_url=brand_url,
        min_resolution="512x512",
        images=list(images),
        texts=list(texts),
        pages_crawled=pages_crawled,
        candidates_checked=candidates_checked,
        created_at=datetime.now(timezone.utc),
        bot_protection=bot_protection,
    )


def test_merge_combines_three_bundles_and_dedupes_images_and_texts():
    web = _bundle(
        "website",
        images=[
            _image("https://example.com/a.jpg"),
            _image("https://example.com/b.jpg"),
        ],
        texts=[_snippet("Brand A", weight=2.0), _snippet("shared", weight=1.0)],
        pages_crawled=10,
        candidates_checked=20,
    )
    ig = _bundle(
        "instagram",
        images=[
            _image("https://example.com/a.jpg"),
            _image("https://scontent.cdninstagram.com/c.jpg"),
        ],
        texts=[_snippet("ig caption", weight=1.4), _snippet("shared", weight=2.5)],
        pages_crawled=1,
        candidates_checked=2,
    )
    tw = _bundle(
        "twitter",
        images=[_image("https://pbs.twimg.com/media/d.jpg?name=large")],
        texts=[_snippet("tweet text", weight=1.3)],
        pages_crawled=1,
        candidates_checked=1,
    )

    merged = merge_content_bundles([web, ig, tw])

    assert merged.source == "combined"
    assert str(merged.brand_url) == "https://example.com/"

    image_urls = {str(img.url) for img in merged.images}
    assert image_urls == {
        "https://example.com/a.jpg",
        "https://example.com/b.jpg",
        "https://scontent.cdninstagram.com/c.jpg",
        "https://pbs.twimg.com/media/d.jpg?name=large",
    }

    by_text = {t.text: t for t in merged.texts}
    assert by_text["shared"].weight == 2.5
    assert {"Brand A", "shared", "ig caption", "tweet text"} <= set(by_text)

    assert merged.pages_crawled == 12
    assert merged.candidates_checked == 23
    assert merged.bot_protection is None


def test_merge_propagates_first_bot_protection():
    detection = BotProtectionFailure(
        url="https://example.com",
        provider="cloudflare",
        reason="403",
    )
    web = _bundle("website", images=[_image("https://example.com/a.jpg")])
    ig = _bundle("instagram", bot_protection=detection)
    tw = _bundle("twitter")

    merged = merge_content_bundles([web, ig, tw])

    assert merged.bot_protection is not None
    assert merged.bot_protection.provider == "cloudflare"


def test_merge_inherits_brand_url_from_first_website_bundle():
    ig = _bundle(
        "instagram",
        brand_url="https://www.instagram.com/nike/",
        images=[_image("https://scontent.cdninstagram.com/x.jpg")],
    )
    web = _bundle(
        "website",
        brand_url="https://www.nike.com",
        images=[_image("https://www.nike.com/a.jpg")],
    )

    merged = merge_content_bundles([ig, web])

    assert str(merged.brand_url) == "https://www.nike.com/"


def test_merge_falls_back_to_first_bundle_when_no_website():
    ig = _bundle(
        "instagram",
        brand_url="https://www.instagram.com/nike/",
    )
    tw = _bundle("twitter", brand_url="https://x.com/nike")

    merged = merge_content_bundles([ig, tw])

    assert "instagram" in str(merged.brand_url)


def test_merge_empty_input_raises():
    with pytest.raises(ValueError):
        merge_content_bundles([])
