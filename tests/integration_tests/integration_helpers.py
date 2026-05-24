import os
import re
from dataclasses import dataclass
from typing import List

import pytest

from src.models.metadata_extraction_config import MetadataExtractionConfig
from src.services.metadata_config_service import MetadataConfigService
from src.utils.metadata_extractor import ConfigDrivenMetadataExtractor, ImageCandidate

_UUID_SKU_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BrandIntegrationCase:
    name: str
    brand_url: str
    min_rich_candidates: int = 3
    require_style_code_sku: bool = True
    min_crawl_products: int = 10
    expect_bot_protection: bool = False


BRAND_CASES = (
    BrandIntegrationCase(
        name="nike",
        brand_url="https://www.nike.com/",
    ),
    BrandIntegrationCase(
        name="prada",
        brand_url="https://www.prada.com/",
        min_rich_candidates=2,
        require_style_code_sku=False,
    ),
    BrandIntegrationCase(
        name="allbirds",
        brand_url="https://www.allbirds.com/",
        min_rich_candidates=2,
        require_style_code_sku=False,
    ),

    # bot prevented ones
    BrandIntegrationCase(
        name="adidas",
        brand_url="https://www.adidas.com/",
        expect_bot_protection=True,
    ),
    BrandIntegrationCase(
        name="zara",
        brand_url="https://www.zara.com/",
        expect_bot_protection=True,
    ),
)


def load_brand_extraction_config(
    service: MetadataConfigService,
    brand_url: str,
    *,
    refresh: bool = False,
) -> MetadataExtractionConfig:
    domain = MetadataConfigService.normalize_domain(brand_url)
    if refresh and not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is required to refresh extraction config via load_or_infer")
    if not refresh and service.repository.get_by_domain(domain) is None:
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip(
                f"No cached extraction config for {domain}; "
                "GEMINI_API_KEY is required to infer via load_or_infer"
            )
    return service.load_or_infer(brand_url, refresh=refresh)


def fetch_brand_sample_pages(
    service: MetadataConfigService,
    brand_url: str,
    limit: int = 4,
) -> list[tuple[str, str]]:
    page_pairs = service.fetch_sample_pages_from_url(brand_url, limit=limit)
    if not page_pairs:
        pytest.skip(f"Could not fetch sample pages from brand URL: {brand_url}")
    return page_pairs


def metadata_rich_candidates(candidates: List[ImageCandidate]) -> List[ImageCandidate]:
    return [
        candidate
        for candidate in candidates
        if candidate.product_name and candidate.sku_id
    ]


def assert_no_uuid_skus(
    candidates: List[ImageCandidate],
    extractor: ConfigDrivenMetadataExtractor,
    *,
    require_style_code: bool = True,
) -> None:
    for candidate in candidates:
        sku_id = candidate.sku_id or ""
        assert not _UUID_SKU_PATTERN.match(sku_id), f"UUID-like sku_id rejected: {sku_id}"
        if require_style_code:
            assert extractor.is_style_code_sku(sku_id), f"sku_id failed style check: {sku_id}"
