from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, HttpUrl

from src.models.image_outputs import BotProtectionFailure, DiscoveredImage


TextSnippetKind = Literal[
    "meta_description",
    "og_title",
    "og_description",
    "title",
    "hero",
    "about",
    "editorial",
    "product_description",
    "tagline",
]

ContentSource = Literal[
    "website",
    "instagram",
    "twitter",
    "tiktok",
    "pinterest",
    "combined",
    "other",
]


class BrandTextSnippet(BaseModel):
    """A short piece of brand-authored copy harvested during a crawl.

    Snippets are intentionally short and deduplicated; downstream the LLM
    receives a flat list ordered by `weight` (higher = more representative
    of brand voice).
    """

    source_url: HttpUrl
    kind: TextSnippetKind
    text: str
    weight: float = 1.0


class BrandContentBundle(BaseModel):
    """Generic crawl output that any content-producing crawler emits.

    The website crawler emits one of these today; future crawlers (social,
    editorial RSS, etc.) will emit the same shape so the downstream brand
    DNA pipeline doesn't care about provenance.
    """

    source: ContentSource = "website"
    brand_url: HttpUrl
    min_resolution: str
    images: List[DiscoveredImage]
    texts: List[BrandTextSnippet] = []
    pages_crawled: int
    candidates_checked: int
    created_at: datetime
    bot_protection: Optional[BotProtectionFailure] = None

    @property
    def stopped_due_to_bot_protection(self) -> bool:
        return self.bot_protection is not None
