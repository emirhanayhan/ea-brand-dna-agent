import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse

from lxml import html

from src.models.brand_configs import BrandConfig
from src.models.brand_content import BrandContentBundle, BrandTextSnippet
from src.models.image_outputs import BotProtectionFailure, DiscoveredImage
from src.services.http_service import HttpClient
from src.services.metadata_config_service import MetadataConfigService
from src.utils.bot_protection import BotProtectionDetection
from src.utils.image_helpers import get_image_dimensions, meets_minimum, parse_resolution
from src.utils.metadata_extractor import (
    ImageCandidate,
    extract_image_candidates,
    is_plausible_image_url,
    merge_candidate_metadata,
)
from src.utils.text_snippet_extractor import (
    dedupe_snippets,
    editorial_candidate_paths,
    extract_text_snippets,
    is_editorial_path,
)

logger = logging.getLogger("WebsiteCrawler")


def normalize_netloc(netloc: str) -> str:
    netloc = netloc.lower()
    if netloc.startswith("www."):
        return netloc[4:]
    return netloc


def brand_token_from_url(seed_url: str) -> str:
    netloc = normalize_netloc(urlparse(seed_url).netloc)
    parts = netloc.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return parts[0]


def is_same_domain(url: str, seed_url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    seed = urlparse(seed_url)
    return normalize_netloc(parsed.netloc) == normalize_netloc(seed.netloc)


def is_allowed_image_url(url: str, seed_url: str) -> bool:
    if not is_plausible_image_url(url):
        return False
    if is_same_domain(url, seed_url):
        return True

    brand_token = brand_token_from_url(seed_url)
    host = urlparse(url).netloc.lower().replace(".", "").replace("-", "")
    return brand_token in host


def normalize_url(url: str) -> str:
    normalized, _ = urldefrag(url)
    return normalized


def extract_page_links(page_html: str, base_url: str, seed_url: str) -> Set[str]:
    links: Set[str] = set()
    tree = html.fromstring(page_html)

    for element in tree.xpath("//a[@href]"):
        href = element.get("href")
        if not href:
            continue
        absolute = normalize_url(urljoin(base_url, href))
        if is_same_domain(absolute, seed_url):
            links.add(absolute)

    return links


def _has_complete_metadata(candidate: ImageCandidate) -> bool:
    return bool(candidate.product_name and candidate.sku_id)


class WebsiteCrawler:
    def __init__(
        self,
        http_client: HttpClient,
        settings: Dict[str, Any],
        metadata_config_service: MetadataConfigService,
        refresh_extraction_config: bool = False,
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        self.http_client = http_client
        self.settings = settings
        self.metadata_config_service = metadata_config_service
        self.refresh_extraction_config = refresh_extraction_config
        self.executor = executor
        self.max_pages = settings["max_crawl_pages"]
        self.max_depth = settings["max_crawl_depth"]
        self.target_product_count = settings["lowest_processeable_product_count"]
        self.min_resolution_str = settings["lowest_acceptable_resolution"]
        self.worker_count = settings["worker_count"]
        self.min_width, self.min_height = parse_resolution(self.min_resolution_str)
        self.editorial_probe_enabled = bool(
            settings.get("editorial_probe_enabled", False)
        )

    def crawl(self, brand_config: BrandConfig) -> BrandContentBundle:
        if not brand_config.url:
            raise ValueError("brand_config.url is required for image crawling")

        seed_url = str(brand_config.url)
        extraction_config = self.metadata_config_service.load_or_infer(
            seed_url,
            refresh=self.refresh_extraction_config,
        )

        priority_paths = extraction_config.priority_link_path_contains or ["/w/", "/t/"]
        visited_pages: Set[str] = set()
        candidate_index: Dict[str, ImageCandidate] = {}
        pending_candidates: deque[ImageCandidate] = deque()
        probed_urls: Set[str] = set()
        accepted_sku_ids: Set[str] = set()
        pending_sku_ids: Set[str] = set()
        accepted_images: list[DiscoveredImage] = []
        text_snippets: list[BrandTextSnippet] = []
        editorial_pages_seen = False
        pages_crawled = 0
        candidates_checked = 0

        queue: deque[Tuple[str, int]] = deque([(seed_url, 0)])

        while queue and pages_crawled < self.max_pages:
            if len(accepted_sku_ids) >= self.target_product_count:
                break

            page_url, depth = queue.popleft()
            if page_url in visited_pages:
                continue
            if depth > self.max_depth:
                continue

            visited_pages.add(page_url)
            response = self.http_client.get(page_url)
            if not response:
                if self.http_client.last_bot_protection:
                    return self._result_with_bot_protection(
                        brand_config,
                        accepted_images=accepted_images,
                        text_snippets=text_snippets,
                        pages_crawled=pages_crawled,
                        candidates_checked=candidates_checked,
                        page_url=page_url,
                        detection=self.http_client.last_bot_protection,
                    )
                continue

            content_type = response.headers.get("content-type", "")
            if "html" not in content_type.lower():
                continue

            pages_crawled += 1
            page_html = response.text
            base_url = str(response.url)

            harvested = extract_text_snippets(page_html, base_url)
            if harvested:
                text_snippets.extend(harvested)
                if is_editorial_path(base_url):
                    editorial_pages_seen = True

            for candidate in extract_image_candidates(
                page_html,
                base_url,
                page_url,
                extraction_config,
            ):
                if not is_allowed_image_url(candidate.url, seed_url):
                    continue
                existing = candidate_index.get(candidate.url)
                if existing is None:
                    candidate_index[candidate.url] = candidate
                    self._schedule_probe(
                        candidate,
                        pending_candidates,
                        probed_urls,
                        accepted_sku_ids,
                        pending_sku_ids,
                    )
                else:
                    merged = merge_candidate_metadata(existing, candidate)
                    candidate_index[candidate.url] = merged
                    if _has_complete_metadata(merged):
                        self._schedule_probe(
                            merged,
                            pending_candidates,
                            probed_urls,
                            accepted_sku_ids,
                            pending_sku_ids,
                        )

            if depth < self.max_depth:
                product_links = []
                other_links = []
                for link in extract_page_links(page_html, base_url, seed_url):
                    if link in visited_pages:
                        continue
                    if any(token in link for token in priority_paths):
                        product_links.append(link)
                    else:
                        other_links.append(link)
                for link in reversed(other_links):
                    queue.append((link, depth + 1))
                for link in reversed(product_links):
                    queue.appendleft((link, depth + 1))

            while pending_candidates and len(accepted_sku_ids) < self.target_product_count:
                batch: list[ImageCandidate] = []
                while pending_candidates and len(batch) < self.worker_count:
                    candidate = pending_candidates.popleft()
                    if candidate.sku_id:
                        pending_sku_ids.discard(candidate.sku_id)
                    if candidate.url in probed_urls:
                        continue
                    probed_urls.add(candidate.url)
                    batch.append(candidate_index[candidate.url])
                if not batch:
                    break
                accepted, checked = self._probe_candidates(batch)
                candidates_checked += checked
                for image in accepted:
                    sku_id = image.sku_id
                    if not sku_id or sku_id in accepted_sku_ids:
                        continue
                    accepted_sku_ids.add(sku_id)
                    accepted_images.append(image)
                    if len(accepted_sku_ids) >= self.target_product_count:
                        break

        if self.editorial_probe_enabled and not editorial_pages_seen:
            extra = self._probe_editorial_paths(seed_url, visited_pages)
            if extra:
                text_snippets.extend(extra)

        return self._build_result(
            brand_config,
            accepted_images=accepted_images,
            text_snippets=text_snippets,
            pages_crawled=pages_crawled,
            candidates_checked=candidates_checked,
        )

    def _probe_editorial_paths(
        self,
        seed_url: str,
        visited_pages: Set[str],
    ) -> list[BrandTextSnippet]:
        """Best-effort fetch of /about, /story, /journal, ... — bounded and
        silently skipped on any bot protection or HTTP error.
        """
        snippets: list[BrandTextSnippet] = []
        parsed = urlparse(seed_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        probed = 0
        for path in editorial_candidate_paths():
            if probed >= 4:
                break
            candidate_url = urljoin(origin, path)
            if candidate_url in visited_pages:
                continue
            response = self.http_client.get(candidate_url)
            probed += 1
            if not response:
                if self.http_client.last_bot_protection:
                    logger.info(
                        "editorial probe halted (bot protection) at %s",
                        candidate_url,
                    )
                    break
                continue
            if "html" not in response.headers.get("content-type", "").lower():
                continue
            snippets.extend(extract_text_snippets(response.text, str(response.url)))
        return snippets

    def _build_result(
        self,
        brand_config: BrandConfig,
        *,
        accepted_images: list[DiscoveredImage],
        pages_crawled: int,
        candidates_checked: int,
        text_snippets: Optional[list[BrandTextSnippet]] = None,
        bot_protection: Optional[BotProtectionFailure] = None,
    ) -> BrandContentBundle:
        return BrandContentBundle(
            source="website",
            brand_url=brand_config.url,
            min_resolution=self.min_resolution_str,
            images=accepted_images,
            texts=dedupe_snippets(text_snippets or []),
            pages_crawled=pages_crawled,
            candidates_checked=candidates_checked,
            created_at=datetime.now(timezone.utc),
            bot_protection=bot_protection,
        )

    def _result_with_bot_protection(
        self,
        brand_config: BrandConfig,
        *,
        accepted_images: list[DiscoveredImage],
        pages_crawled: int,
        candidates_checked: int,
        page_url: str,
        detection: BotProtectionDetection,
        text_snippets: Optional[list[BrandTextSnippet]] = None,
    ) -> BrandContentBundle:
        logger.error(
            "website_crawler stopped due to bot protection at %s: %s",
            page_url,
            detection,
        )
        return self._build_result(
            brand_config,
            accepted_images=accepted_images,
            text_snippets=text_snippets,
            pages_crawled=pages_crawled,
            candidates_checked=candidates_checked,
            bot_protection=BotProtectionFailure(
                url=page_url,
                provider=detection.provider,
                reason=detection.reason,
            ),
        )

    @staticmethod
    def _schedule_probe(
        candidate: ImageCandidate,
        pending_candidates: deque[ImageCandidate],
        probed_urls: Set[str],
        accepted_sku_ids: Set[str],
        pending_sku_ids: Set[str],
    ) -> None:
        if candidate.url in probed_urls:
            return
        if not _has_complete_metadata(candidate):
            return
        sku_id = candidate.sku_id
        if not sku_id or sku_id in accepted_sku_ids or sku_id in pending_sku_ids:
            return
        pending_sku_ids.add(sku_id)
        pending_candidates.appendleft(candidate)

    def _probe_candidates(
        self,
        candidates: list[ImageCandidate],
    ) -> Tuple[list[DiscoveredImage], int]:
        accepted: list[DiscoveredImage] = []
        checked = 0

        # Reuse the injected shared executor when one is provided
        # (production path); otherwise fall back to a per-call pool so
        # tests and ad-hoc runs still work.
        executor_cm = (
            nullcontext(self.executor)
            if self.executor is not None
            else ThreadPoolExecutor(max_workers=self.worker_count)
        )
        with executor_cm as executor:
            futures = {
                executor.submit(self._probe_single, candidate): candidate
                for candidate in candidates
            }
            for future in as_completed(futures):
                checked += 1
                result = future.result()
                if result:
                    accepted.append(result)

        return accepted, checked

    def _probe_single(self, candidate: ImageCandidate) -> Optional[DiscoveredImage]:
        if not _has_complete_metadata(candidate):
            return None

        image_url = candidate.url
        if not image_url.startswith(("http://", "https://")):
            return None

        data = self.http_client.get_stream(image_url)
        if not data:
            return None

        dimensions = get_image_dimensions(data)
        if not dimensions:
            return None

        width, height = dimensions
        if not meets_minimum(width, height, self.min_width, self.min_height):
            logger.debug("Rejected image below resolution threshold: %s (%sx%s)", image_url, width, height)
            return None

        return DiscoveredImage(
            url=image_url,
            width=width,
            height=height,
            source_page=candidate.source_page,
            product_name=candidate.product_name,
            sku_id=candidate.sku_id,
            source_kind=(
                "editorial"
                if is_editorial_path(str(candidate.source_page))
                else "product"
            ),
        )
