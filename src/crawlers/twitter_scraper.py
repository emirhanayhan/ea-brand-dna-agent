"""Twitter/X profile scraper.

Uses the public syndication HTML endpoint that X embed widgets call:
`https://syndication.twitter.com/srv/timeline-profile/screen-name/<handle>`.
The page embeds a `<script id="__NEXT_DATA__" type="application/json">…</script>`
blob with the profile bio and the most recent ~20 tweets. No auth required.

The JSON shape changes from time to time, so the extractor walks the tree
defensively — any object that looks like a tweet (`full_text` + `entities`)
or like the matching user (`screen_name` + `description`) is harvested.

Output is a `BrandContentBundle` so it can be merged with other sources
(website, instagram) before being handed to `BrandDnaService`.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from lxml import html

from src.models.brand_configs import BrandConfig
from src.models.brand_content import BrandContentBundle, BrandTextSnippet
from src.models.image_outputs import BotProtectionFailure, DiscoveredImage
from src.services.http_service import HttpClient
from src.services.social_network_verifier import normalize_social_handle
from src.utils.image_helpers import get_image_dimensions, meets_minimum, parse_resolution
from src.utils.text_snippet_extractor import is_promotional_social_post

logger = logging.getLogger("TwitterScraper")

_DEFAULT_MAX_TWEETS = 20
_TIMELINE_TEMPLATE = (
    "https://syndication.twitter.com/srv/timeline-profile/screen-name/{handle}"
)
_SYNDICATION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://platform.twitter.com/",
    "Accept": "text/html,application/xhtml+xml",
}


class TwitterScraper:
    """Pulls bio + recent tweets/media for a brand handle via syndication."""

    def __init__(self, http_client: HttpClient, settings: Dict[str, Any]):
        self.http_client = http_client
        self.settings = settings
        self.max_tweets = int(settings.get("twitter_max_tweets", _DEFAULT_MAX_TWEETS))
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
            logger.warning("twitter: could not derive handle from %s", profile_url)
            return self._empty_bundle(brand_config)

        timeline_url = _TIMELINE_TEMPLATE.format(handle=handle)
        page_html = self._fetch_timeline(timeline_url)
        if page_html is None:
            return self._empty_bundle(
                brand_config,
                bot_protection=self._bot_protection_for(timeline_url),
            )

        next_data = _extract_next_data(page_html)
        if next_data is None:
            logger.warning("twitter: __NEXT_DATA__ missing from %s", timeline_url)
            return self._empty_bundle(brand_config)

        texts: List[BrandTextSnippet] = []
        candidate_images: List[Tuple[str, str]] = []
        seen_caption: set[str] = set()

        bio = _extract_user_bio(next_data, handle)
        if bio:
            texts.append(
                BrandTextSnippet(
                    source_url=profile_url,
                    kind="tagline",
                    text=bio,
                    weight=2.5,
                )
            )

        for tweet in _walk_tweets(next_data, max_tweets=self.max_tweets):
            text = (tweet.get("full_text") or tweet.get("text") or "").strip()
            tweet_url = _tweet_permalink(handle, tweet) or profile_url
            if is_promotional_social_post(text):
                logger.debug("twitter: skipping promotional tweet %s", tweet_url)
                continue
            if text and text.lower() not in seen_caption:
                seen_caption.add(text.lower())
                texts.append(
                    BrandTextSnippet(
                        source_url=tweet_url,
                        kind="editorial",
                        text=text,
                        weight=1.3,
                    )
                )
            for media_url in _tweet_photo_urls(tweet):
                candidate_images.append((media_url, tweet_url))

        accepted_images: List[DiscoveredImage] = []
        seen_images: set[str] = set()
        for image_url, post_url in candidate_images:
            if image_url in seen_images:
                continue
            seen_images.add(image_url)
            discovered = self._probe_image(image_url, post_url)
            if discovered:
                accepted_images.append(discovered)

        logger.info(
            "twitter: %s — %s text snippets, %s/%s images accepted",
            handle,
            len(texts),
            len(accepted_images),
            len(candidate_images),
        )

        return BrandContentBundle(
            source="twitter",
            brand_url=brand_config.url or profile_url,
            min_resolution=self.min_resolution_str,
            images=accepted_images,
            texts=texts,
            pages_crawled=1,
            candidates_checked=len(candidate_images),
            created_at=datetime.now(timezone.utc),
        )

    def _fetch_timeline(self, timeline_url: str) -> Optional[str]:
        try:
            response = self.http_client.client.get(
                timeline_url, headers=_SYNDICATION_HEADERS
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("twitter: GET %s failed: %s", timeline_url, exc)
            return None
        if response.status_code != 200:
            logger.warning(
                "twitter: GET %s returned %s", timeline_url, response.status_code
            )
            return None
        return response.text

    def _probe_image(
        self, image_url: str, source_page: str
    ) -> Optional[DiscoveredImage]:
        # Twitter media URLs accept ?name=large to upgrade resolution.
        upgraded = _upgrade_media_url(image_url)
        data = self.http_client.get_stream(upgraded, max_bytes=4_000_000)
        if not data:
            return None
        dimensions = get_image_dimensions(data)
        if not dimensions:
            return None
        width, height = dimensions
        if not meets_minimum(width, height, self.min_width, self.min_height):
            logger.debug(
                "twitter: rejected image below resolution threshold: %s (%sx%s)",
                upgraded,
                width,
                height,
            )
            return None
        return DiscoveredImage(
            url=upgraded,
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
            source="twitter",
            brand_url=brand_config.url or "https://x.com/",
            min_resolution=self.min_resolution_str,
            images=[],
            texts=[],
            pages_crawled=0,
            candidates_checked=0,
            created_at=datetime.now(timezone.utc),
            bot_protection=bot_protection,
        )


_HANDLE_PATH_RE = re.compile(r"^/+([A-Za-z0-9_]+)/?$")


def _handle_from_profile_url(profile_url: str) -> Optional[str]:
    parsed = urlparse(profile_url)
    match = _HANDLE_PATH_RE.match(parsed.path or "")
    if not match:
        return None
    return normalize_social_handle(match.group(1))


def _extract_next_data(page_html: str) -> Optional[Any]:
    try:
        tree = html.fromstring(page_html)
    except Exception:
        return None
    nodes = tree.xpath('//script[@id="__NEXT_DATA__"]/text()')
    if not nodes:
        return None
    raw = nodes[0]
    try:
        return json.loads(raw)
    except Exception:
        return None


def _walk(obj: Any) -> Iterable[Dict[str, Any]]:
    """Yield every dict node in an arbitrarily nested structure."""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item)


def _looks_like_tweet(node: Dict[str, Any]) -> bool:
    if not isinstance(node, dict):
        return False
    if "full_text" not in node and "text" not in node:
        return False
    return "entities" in node or "id_str" in node


def _walk_tweets(
    next_data: Any, max_tweets: int
) -> Iterable[Dict[str, Any]]:
    seen_ids: set[str] = set()
    yielded = 0
    for node in _walk(next_data):
        if yielded >= max_tweets:
            return
        if not _looks_like_tweet(node):
            continue
        tweet_id = str(node.get("id_str") or node.get("id") or "")
        if tweet_id and tweet_id in seen_ids:
            continue
        if tweet_id:
            seen_ids.add(tweet_id)
        yield node
        yielded += 1


def _extract_user_bio(next_data: Any, handle: str) -> Optional[str]:
    """Find the description on the user object whose screen_name matches."""
    handle_lower = handle.lower()
    fallback: Optional[str] = None
    for node in _walk(next_data):
        screen_name = str(node.get("screen_name") or "").lower()
        description = node.get("description")
        if not isinstance(description, str):
            continue
        description = description.strip()
        if not description:
            continue
        if screen_name == handle_lower:
            return description
        if fallback is None:
            fallback = description
    return fallback


def _tweet_permalink(handle: str, tweet: Dict[str, Any]) -> Optional[str]:
    tweet_id = tweet.get("id_str") or tweet.get("id")
    if not tweet_id:
        return None
    return f"https://x.com/{handle}/status/{tweet_id}"


def _tweet_photo_urls(tweet: Dict[str, Any]) -> Iterable[str]:
    entities = tweet.get("entities") or {}
    extended = tweet.get("extended_entities") or {}
    for source in (extended, entities):
        media_list = source.get("media") if isinstance(source, dict) else None
        if not isinstance(media_list, list):
            continue
        for media in media_list:
            if not isinstance(media, dict):
                continue
            if media.get("type") not in (None, "photo"):
                continue
            url = media.get("media_url_https") or media.get("media_url")
            if url:
                yield url


def _upgrade_media_url(url: str) -> str:
    """Twitter image CDN supports `?name=large` to fetch the largest size."""
    if "pbs.twimg.com" not in url:
        return url
    if "?" in url:
        return url
    return f"{url}?name=large"
