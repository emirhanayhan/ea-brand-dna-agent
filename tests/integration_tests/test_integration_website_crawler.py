import pytest

from src.models.brand_configs import BrandConfig
from src.crawlers.website_crawler import WebsiteCrawler
from tests.integration_tests.integration_helpers import BRAND_CASES, load_brand_extraction_config


@pytest.fixture
def website_crawler(integration_http_client, integration_settings, metadata_config_service):
    crawl_settings = {
        **integration_settings,
        "lowest_processeable_product_count": 10,
    }
    return WebsiteCrawler(
        integration_http_client,
        crawl_settings,
        metadata_config_service=metadata_config_service,
    )


@pytest.mark.integration
@pytest.mark.parametrize("brand_case", BRAND_CASES, ids=lambda case: case.name)
def test_live_brand_crawl(website_crawler, metadata_config_service, brand_case):
    load_brand_extraction_config(metadata_config_service, brand_case.brand_url)
    result = website_crawler.crawl(BrandConfig(url=brand_case.brand_url))

    if brand_case.expect_bot_protection:
        assert result.stopped_due_to_bot_protection, (
            f"expected bot protection for {brand_case.brand_url}, "
            f"got {len(result.images)} products from {result.pages_crawled} pages"
        )
        return

    assert not result.stopped_due_to_bot_protection, (
        f"unexpected bot protection for {brand_case.brand_url}: "
        f"{result.bot_protection.provider}/{result.bot_protection.reason}"
    )
    assert len(result.images) >= brand_case.min_crawl_products, (
        f"expected at least {brand_case.min_crawl_products} products from "
        f"{brand_case.brand_url}, got {len(result.images)} "
        f"after {result.pages_crawled} pages"
    )
