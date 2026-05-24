"""Instagram profile scraper.

Reuses the unauthenticated `web_profile_info` JSON endpoint that
`social_network_verifier.py` already calls. The endpoint returns the
profile bio plus the most recent ~12 posts (caption + media URLs), which
is enough signal for the brand DNA pipeline.

Output is a `BrandContentBundle` so it can be merged with other sources
(website, twitter) before being handed to `BrandDnaService`.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from src.models.brand_configs import BrandConfig
from src.models.brand_content import BrandContentBundle, BrandTextSnippet
from src.models.image_outputs import BotProtectionFailure, DiscoveredImage
from src.services.http_service import HttpClient
from src.services.social_network_verifier import (
    INSTAGRAM_API_HEADERS,
    normalize_social_handle,
)
from src.utils.image_helpers import get_image_dimensions, meets_minimum, parse_resolution
from src.utils.text_snippet_extractor import is_promotional_social_post

logger = logging.getLogger("InstagramScraper")

_DEFAULT_MAX_POSTS = 12
_PROFILE_API_TEMPLATE = (
    "https://www.instagram.com/api/v1/users/web_profile_info/?username={handle}"
)


class InstagramScraper:
    """Pulls bio + recent post imagery & captions for a brand handle."""

    def __init__(self, http_client: HttpClient, settings: Dict[str, Any]):
        self.http_client = http_client
        self.settings = settings
        self.max_posts = int(settings.get("instagram_max_posts", _DEFAULT_MAX_POSTS))
        self.min_resolution_str = settings.get(
            "lowest_acceptable_resolution", "512x512"
        )
        self.min_width, self.min_height = parse_resolution(self.min_resolution_str)

    def scrape(
        self,
        brand_config: BrandConfig,
        profile_url: str,
    ) -> BrandContentBundle:
        handle = _handle_from_profile_url(profile_url)
        if not handle:
            logger.warning("instagram: could not derive handle from %s", profile_url)
            return self._empty_bundle(brand_config)

        api_url = _PROFILE_API_TEMPLATE.format(handle=handle)
        payload = self._fetch_profile_payload(api_url)
        if payload is None:
            return self._empty_bundle(
                brand_config,
                bot_protection=self._bot_protection_for(api_url),
            )

        user = payload.get("data", {}).get("user")
        if not isinstance(user, dict):
            logger.warning("instagram: profile payload missing user object")
            return self._empty_bundle(brand_config)

        texts = list(_extract_text_snippets(user, profile_url, self.max_posts))
        candidate_images = list(_extract_image_candidates(user, self.max_posts))

        accepted_images: List[DiscoveredImage] = []
        seen_urls: set[str] = set()
        for image_url, post_url in candidate_images:
            if image_url in seen_urls:
                continue
            seen_urls.add(image_url)
            discovered = self._probe_image(image_url, post_url)
            if discovered:
                accepted_images.append(discovered)

        logger.info(
            "instagram: %s — %s captions, %s/%s images accepted",
            handle,
            len(texts),
            len(accepted_images),
            len(candidate_images),
        )

        return BrandContentBundle(
            source="instagram",
            brand_url=brand_config.url or profile_url,
            min_resolution=self.min_resolution_str,
            images=accepted_images,
            texts=texts,
            pages_crawled=1,
            candidates_checked=len(candidate_images),
            created_at=datetime.now(timezone.utc),
        )

    def _fetch_profile_payload(self, api_url: str) -> Optional[Dict[str, Any]]:
        try:
            response = self.http_client.client.get(
                api_url, headers=INSTAGRAM_API_HEADERS
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("instagram: GET %s failed: %s", api_url, exc)
            return None
        if response.status_code != 200:
            logger.warning(
                "instagram: GET %s returned %s", api_url, response.status_code
            )
            return None
        try:
            return response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("instagram: invalid JSON from %s: %s", api_url, exc)
            return None

    def _probe_image(
        self, image_url: str, source_page: str
    ) -> Optional[DiscoveredImage]:
        data = self.http_client.get_stream(image_url, max_bytes=4_000_000)
        if not data:
            return None
        dimensions = get_image_dimensions(data)
        if not dimensions:
            return None
        width, height = dimensions
        if not meets_minimum(width, height, self.min_width, self.min_height):
            logger.debug(
                "instagram: rejected image below resolution threshold: %s (%sx%s)",
                image_url,
                width,
                height,
            )
            return None
        return DiscoveredImage(
            url=image_url,
            width=width,
            height=height,
            source_page=source_page,
            source_kind="social",
        )

    def _bot_protection_for(self, url: str) -> Optional[BotProtectionFailure]:
        detection = self.http_client.last_bot_protection
        if not detection:
            return None
        return BotProtectionFailure(
            url=url, provider=detection.provider, reason=detection.reason
        )

    def _empty_bundle(
        self,
        brand_config: BrandConfig,
        bot_protection: Optional[BotProtectionFailure] = None,
    ) -> BrandContentBundle:
        return BrandContentBundle(
            source="instagram",
            brand_url=brand_config.url or "https://www.instagram.com/",
            min_resolution=self.min_resolution_str,
            images=[],
            texts=[],
            pages_crawled=0,
            candidates_checked=0,
            created_at=datetime.now(timezone.utc),
            bot_protection=bot_protection,
        )


_HANDLE_PATH_RE = re.compile(r"^/+([A-Za-z0-9._]+)/?$")


def _handle_from_profile_url(profile_url: str) -> Optional[str]:
    parsed = urlparse(profile_url)
    match = _HANDLE_PATH_RE.match(parsed.path or "")
    if not match:
        return None
    return normalize_social_handle(match.group(1))


def _extract_text_snippets(
    user: Dict[str, Any], profile_url: str, max_posts: int
) -> Iterable[BrandTextSnippet]:
    bio = (user.get("biography") or "").strip()
    if bio:
        yield BrandTextSnippet(
            source_url=profile_url,
            kind="tagline",
            text=bio,
            weight=2.5,
        )

    timeline = (
        user.get("edge_owner_to_timeline_media", {}) or {}
    ).get("edges", [])
    posts_seen = 0
    for edge in timeline:
        if posts_seen >= max_posts:
            break
        node = edge.get("node") if isinstance(edge, dict) else None
        if not isinstance(node, dict):
            continue
        posts_seen += 1
        caption = _extract_caption(node)
        if not caption:
            continue
        if is_promotional_social_post(caption):
            logger.debug(
                "instagram: skipping promotional post %s",
                _post_permalink(node) or profile_url,
            )
            continue
        post_url = _post_permalink(node)
        yield BrandTextSnippet(
            source_url=post_url or profile_url,
            kind="editorial",
            text=caption,
            weight=1.4,
        )


def _extract_caption(node: Dict[str, Any]) -> Optional[str]:
    edges = (node.get("edge_media_to_caption", {}) or {}).get("edges", [])
    if not edges:
        return None
    text = ((edges[0] or {}).get("node", {}) or {}).get("text", "")
    text = text.strip()
    return text or None


def _post_permalink(node: Dict[str, Any]) -> Optional[str]:
    shortcode = node.get("shortcode")
    if not shortcode:
        return None
    return f"https://www.instagram.com/p/{shortcode}/"


def _extract_image_candidates(
    user: Dict[str, Any], max_posts: int
) -> Iterable[Tuple[str, str]]:
    """Yield (image_url, post_permalink) tuples, capped at `max_posts`."""
    profile_url = f"https://www.instagram.com/{user.get('username', '')}/"
    timeline = (
        user.get("edge_owner_to_timeline_media", {}) or {}
    ).get("edges", [])
    posts_seen = 0
    for edge in timeline:
        if posts_seen >= max_posts:
            break
        node = edge.get("node") if isinstance(edge, dict) else None
        if not isinstance(node, dict):
            continue
        posts_seen += 1
        caption = _extract_caption(node)
        if is_promotional_social_post(caption):
            logger.debug(
                "instagram: skipping promotional post images %s",
                _post_permalink(node) or profile_url,
            )
            continue
        post_url = _post_permalink(node) or profile_url
        for image_url in _node_image_urls(node):
            if image_url:
                yield image_url, post_url


def _node_image_urls(node: Dict[str, Any]) -> Iterable[str]:
    """Pull display URLs out of a single timeline node, including carousel
    children. Skips video-only posts (no usable image).
    """
    typename = node.get("__typename") or ""
    if typename == "GraphSidecar":
        children = (
            node.get("edge_sidecar_to_children", {}) or {}
        ).get("edges", [])
        for child in children:
            child_node = child.get("node") if isinstance(child, dict) else None
            if isinstance(child_node, dict) and not child_node.get("is_video", False):
                url = child_node.get("display_url")
                if url:
                    yield url
        return

    if node.get("is_video"):
        # Use the still thumbnail of the video, if usable.
        thumb = node.get("display_url") or node.get("thumbnail_src")
        if thumb:
            yield thumb
        return

    url = node.get("display_url")
    if url:
        yield url
