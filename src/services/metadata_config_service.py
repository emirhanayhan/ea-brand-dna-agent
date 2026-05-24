import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from lxml import html

from src.models.metadata_extraction_config import MetadataExtractionConfig
from src.models.page_sample import PageSample
from src.repositories.extraction_config_repository import MongoExtractionConfigRepository
from src.services.http_service import HttpClient
from src.services.llm_service import LlmService
from src.utils.metadata_extractor import ConfigDrivenMetadataExtractor

logger = logging.getLogger("MetadataConfigService")

_NEXT_DATA_PATTERN = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)
_LD_JSON_PATTERN = re.compile(
    r'<script type="application/ld\+json">(.*?)</script>',
    re.DOTALL,
)
_PRODUCT_CARD_PATTERN = re.compile(
    r'(<[^>]+data-testid="product-card"[^>]*>.*?</div>)',
    re.DOTALL | re.IGNORECASE,
)
_MAX_SNIPPET_CHARS = 10_000


class MetadataConfigService:
    def __init__(
        self,
        settings: Dict[str, Any],
        http_client: HttpClient,
        llm_service: LlmService,
        repository: MongoExtractionConfigRepository,
    ):
        self.http_client = http_client
        self.settings = settings
        self.llm_service = llm_service
        self.repository = repository

    @staticmethod
    def normalize_domain(url: str) -> str:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            return netloc[4:]
        return netloc

    def get_by_domain(self, domain: str) -> Optional[MetadataExtractionConfig]:
        return self.repository.get_by_domain(domain)

    def update(self, config: MetadataExtractionConfig) -> None:
        self.repository.update(config)

    def build_page_sample(self, page_html: str, url: str) -> PageSample:
        snippets: List[str] = []

        for match in _LD_JSON_PATTERN.finditer(page_html):
            snippets.append(match.group(0)[:2000])

        next_match = _NEXT_DATA_PATTERN.search(page_html)
        if next_match:
            snippets.append(
                f'<script id="__NEXT_DATA__" type="application/json">{next_match.group(1)[:4000]}</script>'
            )

        for match in _PRODUCT_CARD_PATTERN.finditer(page_html):
            snippets.append(match.group(1)[:2000])

        if not snippets:
            snippets.append(page_html[:_MAX_SNIPPET_CHARS])
        else:
            combined = "\n".join(snippets)
            if len(combined) > _MAX_SNIPPET_CHARS:
                combined = combined[:_MAX_SNIPPET_CHARS]

        page_type = "other"
        lower_url = url.lower()
        if any(token in lower_url for token in ("/w/", "/shop/", "/collection")):
            page_type = "listing"
        elif any(token in lower_url for token in ("/t/", "/product/", "/p/")):
            page_type = "pdp"
        elif urlparse(url).path in ("", "/"):
            page_type = "homepage"

        return PageSample(
            url=url,
            page_type=page_type,
            html_snippet=snippets[0] if len(snippets) == 1 else "\n".join(snippets),
        )

    def fetch_sample_pages_from_url(self, seed_url: str, limit: int = 4) -> List[tuple[str, str]]:
        samples: List[tuple[str, str]] = []
        visited: set[str] = set()
        queue: List[str] = [seed_url]

        while queue and len(samples) < limit:
            page_url = queue.pop(0)
            if page_url in visited:
                continue
            visited.add(page_url)

            response = self.http_client.get(page_url)
            if not response or "html" not in response.headers.get("content-type", "").lower():
                continue

            samples.append((str(response.url), response.text))

            if len(samples) >= limit:
                break

            tree = html.fromstring(response.text)
            prioritized: List[str] = []
            other: List[str] = []
            base = str(response.url)
            for element in tree.xpath("//a[@href]"):
                href = element.get("href")
                if not href:
                    continue
                absolute = urljoin(base, href)
                parsed = urlparse(absolute)
                if parsed.scheme not in ("http", "https"):
                    continue
                if self.normalize_domain(absolute) != self.normalize_domain(seed_url):
                    continue
                if absolute in visited:
                    continue
                if any(token in absolute for token in ("/w/", "/t/", "/products/", "/product/", "/p/")):
                    prioritized.append(absolute)
                else:
                    other.append(absolute)

            for link in prioritized[:2]:
                if link not in visited:
                    queue.append(link)
            if len(samples) < 2:
                for link in other[:1]:
                    if link not in visited:
                        queue.append(link)

        return samples

    def validate_config(
        self,
        config: MetadataExtractionConfig,
        samples: List[PageSample],
    ) -> bool:
        extractor = ConfigDrivenMetadataExtractor(config)
        for sample in samples:
            products = extractor.parse_page_products(sample.html_snippet, str(sample.url))
            if any(product.name or product.sku_id for product in products):
                return True
            candidates = extractor.extract_image_candidates(
                sample.html_snippet,
                str(sample.url),
                str(sample.url),
            )
            if any(candidate.product_name or candidate.sku_id for candidate in candidates):
                return True
        return False

    def load_or_infer(
        self,
        brand_url: str,
        refresh: bool = False,
    ) -> MetadataExtractionConfig:
        domain = self.normalize_domain(brand_url)

        if not refresh:
            cached = self.get_by_domain(domain)
            if cached and _is_fresh_today(cached.updated_at):
                logger.info("Loaded cached extraction config for %s", domain)
                return cached
            if cached:
                logger.info(
                    "Cached extraction config for %s is stale (updated_at=%s); re-inferring",
                    domain,
                    cached.updated_at,
                )

        page_pairs = self.fetch_sample_pages_from_url(brand_url)
        page_samples = [self.build_page_sample(html, url) for url, html in page_pairs]

        config = self.llm_service.infer_extraction_config(page_samples, domain).normalized()
        if not self.validate_config(config, page_samples):
            logger.warning(
                "LLM extraction config failed validation for %s; using ecommerce defaults",
                domain,
            )
            product_card = config.product_card
            config = MetadataExtractionConfig.ecommerce_defaults(domain)
            if product_card:
                config = config.model_copy(update={"product_card": product_card}).normalized()

        if config.domain != domain:
            config = config.model_copy(update={"domain": domain})

        self.update(config)
        return config


def _is_fresh_today(updated_at: Optional[datetime]) -> bool:
    """Cached configs are considered fresh only on the same UTC calendar
    day they were written; anything older triggers an LLM re-inference.
    """
    if updated_at is None:
        return False
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return updated_at.astimezone(timezone.utc).date() == datetime.now(timezone.utc).date()
