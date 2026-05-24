"""Combine multiple `BrandContentBundle`s into one for the brand DNA pipeline.

Today the website crawler, instagram scraper, and twitter scraper each
produce their own bundle. `BrandDnaService.run` accepts a single bundle,
so we merge here: concat + dedup images and texts, sum counters, and
surface the first bot-protection failure (if any) so callers can still
log it.

The merged bundle inherits `brand_url` and `min_resolution` from the
first website-source bundle when present, otherwise from the first input.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from src.models.brand_content import BrandContentBundle
from src.models.image_outputs import BotProtectionFailure, DiscoveredImage
from src.utils.text_snippet_extractor import dedupe_snippets

logger = logging.getLogger("ContentBundleMerger")


def merge_content_bundles(
    bundles: List[BrandContentBundle],
) -> BrandContentBundle:
    """Concat + dedup multiple bundles into a single `combined` bundle.

    Raises ValueError on an empty input list.
    """
    if not bundles:
        raise ValueError("merge_content_bundles requires at least one bundle")

    primary = _pick_primary(bundles)

    seen_image_urls: set[str] = set()
    images: List[DiscoveredImage] = []
    pages_crawled = 0
    candidates_checked = 0
    bot_protection: Optional[BotProtectionFailure] = None
    all_texts = []

    for bundle in bundles:
        pages_crawled += bundle.pages_crawled
        candidates_checked += bundle.candidates_checked
        if bot_protection is None and bundle.bot_protection is not None:
            bot_protection = bundle.bot_protection
        for image in bundle.images:
            url_key = str(image.url)
            if url_key in seen_image_urls:
                continue
            seen_image_urls.add(url_key)
            images.append(image)
        all_texts.extend(bundle.texts)

    deduped_texts = dedupe_snippets(all_texts, limit=max(40, len(all_texts)))

    logger.info(
        "merged %s bundles: %s images, %s texts (sources=%s)",
        len(bundles),
        len(images),
        len(deduped_texts),
        [b.source for b in bundles],
    )

    return BrandContentBundle(
        source="combined",
        brand_url=primary.brand_url,
        min_resolution=primary.min_resolution,
        images=images,
        texts=deduped_texts,
        pages_crawled=pages_crawled,
        candidates_checked=candidates_checked,
        created_at=datetime.now(timezone.utc),
        bot_protection=bot_protection,
    )


def _pick_primary(bundles: List[BrandContentBundle]) -> BrandContentBundle:
    """Prefer the first website bundle for shared metadata; fall back to
    the first bundle in the input order.
    """
    for bundle in bundles:
        if bundle.source == "website":
            return bundle
    return bundles[0]
