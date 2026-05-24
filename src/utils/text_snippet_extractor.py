"""Pure functions that harvest brand-voice text from already-fetched HTML.

Designed to plug into the website crawler without an extra HTTP round-trip:
for every page the crawler visits, we hand the parsed HTML to
`extract_text_snippets` and accumulate the results into the
`BrandContentBundle.texts` list. A separate small probe step inside the
crawler optionally walks editorial paths (/about, /story, ...) when those
weren't visited organically.
"""

from __future__ import annotations

import re
from typing import List, Optional
from urllib.parse import urlparse

from lxml import html

from src.models.brand_content import BrandTextSnippet, TextSnippetKind


_WS_RE = re.compile(r"\s+")
_PERCENT_OFF = re.compile(
    r"(?:\d+\s*%\s*(?:off|indirim(?:li|de)?)|%\s*\d+\s*(?:off|indirim(?:li|de)?)?)",
    re.IGNORECASE,
)
_PROMO_KEYWORDS = re.compile(
    r"\b(?:sale|discount|promo(?:tion|tional)?|clearance|indirim(?:li|de)?)\b",
    re.IGNORECASE,
)
_EVENT_REGISTER = re.compile(
    r"\b(?:register|registration|rsvp|sign\s*up)\b",
    re.IGNORECASE,
)
_EVENT_STORE = re.compile(
    r"\b(?:store|opening|flagship|boutique)\b",
    re.IGNORECASE,
)
_EVENT_INVITE = re.compile(
    r"\b(?:invite|invitation)\b",
    re.IGNORECASE,
)
_MIN_HERO_CHARS = 12
_MIN_PARAGRAPH_CHARS = 40
_MAX_SNIPPET_CHARS = 800
_MAX_PARAGRAPHS_PER_PAGE = 6
_MAX_PDP_PARAGRAPHS = 3

_EDITORIAL_PATH_HINTS = (
    "/about",
    "/story",
    "/stories",
    "/journal",
    "/manifesto",
    "/editorial",
    "/sustainability",
    "/values",
    "/heritage",
    "/our-",
)


def _is_event_announcement_post(text: str) -> bool:
    """Skip store-opening / RSVP flyers when at least two of register, store,
    invite keyword groups match — avoids dropping casual copy that mentions
    only one term (e.g. "visit our store").
    """
    hits = sum(
        (
            bool(_EVENT_REGISTER.search(text)),
            bool(_EVENT_STORE.search(text)),
            bool(_EVENT_INVITE.search(text)),
        )
    )
    return hits >= 2


def is_promotional_social_post(text: str | None) -> bool:
    """Return True when social caption/tweet text should be skipped.

    Matches sale/discount promos outright, or event announcements when two
    of three keyword groups hit (register, store, invite).
    """
    if not text or not text.strip():
        return False
    if _PERCENT_OFF.search(text) or _PROMO_KEYWORDS.search(text):
        return True
    return _is_event_announcement_post(text)


def _clean(text: Optional[str]) -> str:
    if not text:
        return ""
    return _WS_RE.sub(" ", text).strip()


def _truncate(text: str, limit: int = _MAX_SNIPPET_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _classify_page(url: str) -> str:
    path = urlparse(url).path.lower()
    if path in ("", "/"):
        return "homepage"
    if any(token in path for token in ("/p/", "/t/", "/product/", "/products/")):
        return "pdp"
    if any(token in path for token in _EDITORIAL_PATH_HINTS):
        return "editorial"
    return "other"


def is_editorial_path(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(token in path for token in _EDITORIAL_PATH_HINTS)


def editorial_candidate_paths() -> tuple[str, ...]:
    """Slugs we'll proactively probe when no editorial page was visited."""
    return (
        "/about",
        "/about-us",
        "/our-story",
        "/story",
        "/journal",
        "/manifesto",
        "/sustainability",
    )


def _snippet(
    source_url: str,
    kind: TextSnippetKind,
    text: str,
    weight: float,
) -> Optional[BrandTextSnippet]:
    cleaned = _clean(text)
    if not cleaned:
        return None
    return BrandTextSnippet(
        source_url=source_url,
        kind=kind,
        text=_truncate(cleaned),
        weight=weight,
    )


def extract_text_snippets(page_html: str, page_url: str) -> List[BrandTextSnippet]:
    """Pull brand-voice signals from a page's HTML.

    Weights are calibrated so that editorial/about pages outrank product copy,
    which outranks meta tags — without dropping any signal entirely.
    """
    if not page_html:
        return []

    try:
        tree = html.fromstring(page_html)
    except (ValueError, html.etree.ParserError):
        return []

    page_type = _classify_page(page_url)
    snippets: List[BrandTextSnippet] = []

    title_el = tree.find(".//title")
    if title_el is not None:
        snippets.append(_snippet(page_url, "title", title_el.text_content(), 0.4))

    for selector, kind, weight in (
        (".//meta[@name='description']", "meta_description", 0.6),
        (".//meta[@property='og:title']", "og_title", 0.5),
        (".//meta[@property='og:description']", "og_description", 0.7),
        (".//meta[@name='twitter:description']", "og_description", 0.6),
    ):
        for el in tree.xpath(selector):
            snippets.append(_snippet(page_url, kind, el.get("content"), weight))

    if page_type == "homepage":
        for el in tree.xpath(".//h1") + tree.xpath(".//h2")[:3]:
            txt = _clean(el.text_content())
            if len(txt) >= _MIN_HERO_CHARS:
                snippets.append(_snippet(page_url, "hero", txt, 1.1))

    if page_type in ("editorial", "homepage") or is_editorial_path(page_url):
        kept = 0
        for p in tree.xpath(".//p"):
            txt = _clean(p.text_content())
            if len(txt) < _MIN_PARAGRAPH_CHARS:
                continue
            kind: TextSnippetKind = (
                "about" if "/about" in page_url.lower() else "editorial"
            )
            snippets.append(_snippet(page_url, kind, txt, 1.5))
            kept += 1
            if kept >= _MAX_PARAGRAPHS_PER_PAGE:
                break

    if page_type == "pdp":
        kept = 0
        candidate_nodes = (
            tree.xpath(".//*[contains(@class,'description')]//p")
            + tree.xpath(".//*[contains(@class,'product-detail')]//p")
            + tree.xpath(".//*[@itemprop='description']")
        )
        for node in candidate_nodes:
            txt = _clean(node.text_content())
            if len(txt) < _MIN_PARAGRAPH_CHARS:
                continue
            snippets.append(
                _snippet(page_url, "product_description", txt, 0.9)
            )
            kept += 1
            if kept >= _MAX_PDP_PARAGRAPHS:
                break

    return [s for s in snippets if s is not None]


def dedupe_snippets(
    snippets: List[BrandTextSnippet],
    limit: int = 40,
) -> List[BrandTextSnippet]:
    """Drop near-duplicates (same text), keep the highest-weight copy, and
    return the top `limit` snippets ordered by weight desc.
    """
    by_text: dict[str, BrandTextSnippet] = {}
    for s in snippets:
        key = s.text.strip().lower()
        existing = by_text.get(key)
        if existing is None or s.weight > existing.weight:
            by_text[key] = s
    ordered = sorted(by_text.values(), key=lambda x: x.weight, reverse=True)
    return ordered[:limit]
