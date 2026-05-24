import pytest

from src.utils.metadata_extractor import ConfigDrivenMetadataExtractor
from tests.integration_tests.integration_helpers import (
    BRAND_CASES,
    assert_no_uuid_skus,
    fetch_brand_sample_pages,
    load_brand_extraction_config,
    metadata_rich_candidates,
)


@pytest.mark.integration
@pytest.mark.parametrize("brand_case", BRAND_CASES, ids=lambda case: case.name)
def test_live_brand_extracts_product_metadata(metadata_config_service, brand_case):
    if brand_case.expect_bot_protection:
        pytest.skip(f"{brand_case.name} is blocked by bot protection")

    config = load_brand_extraction_config(metadata_config_service, brand_case.brand_url)
    page_pairs = fetch_brand_sample_pages(metadata_config_service, brand_case.brand_url)
    extractor = ConfigDrivenMetadataExtractor(config)

    rich: list = []
    for page_url, page_html in page_pairs:
        candidates = extractor.extract_image_candidates(page_html, page_url, page_url)
        rich.extend(metadata_rich_candidates(candidates))

    assert len(rich) >= brand_case.min_rich_candidates, (
        f"expected at least {brand_case.min_rich_candidates} metadata-rich candidates "
        f"from pages discovered from {brand_case.brand_url}, got {len(rich)}"
    )
    assert_no_uuid_skus(
        rich,
        extractor,
        require_style_code=brand_case.require_style_code_sku,
    )

    for candidate in rich:
        assert candidate.product_name.strip()
        assert candidate.sku_id.strip()
        assert candidate.url.startswith("http")


@pytest.mark.integration
@pytest.mark.parametrize("brand_case", BRAND_CASES, ids=lambda case: case.name)
def test_live_brand_parses_page_products(metadata_config_service, brand_case):
    if brand_case.expect_bot_protection:
        pytest.skip(f"{brand_case.name} is blocked by bot protection")

    config = load_brand_extraction_config(metadata_config_service, brand_case.brand_url)
    page_pairs = fetch_brand_sample_pages(metadata_config_service, brand_case.brand_url)
    extractor = ConfigDrivenMetadataExtractor(config)

    max_rich_products = 0
    parsed_products = []
    for page_url, page_html in page_pairs:
        products = extractor.parse_page_products(page_html, page_url)
        rich_products = [product for product in products if product.name and product.sku_id]
        max_rich_products = max(max_rich_products, len(rich_products))
        parsed_products.extend(rich_products)

    unique_products = {
        product.sku_id: product
        for product in parsed_products
        if product.sku_id
    }

    assert len(unique_products) >= brand_case.min_rich_candidates, (
        f"expected at least {brand_case.min_rich_candidates} parsed products "
        f"from pages discovered from {brand_case.brand_url}, got {len(unique_products)} "
        f"(best single page had {max_rich_products})"
    )
