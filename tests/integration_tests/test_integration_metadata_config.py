import pytest

from src.utils.metadata_extractor import ConfigDrivenMetadataExtractor
from tests.integration_tests.integration_helpers import (
    load_brand_extraction_config,
    metadata_rich_candidates,
)


@pytest.mark.integration
@pytest.mark.llm_integration
def test_load_or_infer_nike_refresh(metadata_config_service, gemini_settings):
    brand_url = "https://www.nike.com/"
    config = load_brand_extraction_config(
        metadata_config_service,
        brand_url,
        refresh=True,
    )

    page_pairs = metadata_config_service.fetch_sample_pages_from_url(brand_url)
    assert page_pairs, "expected at least one sample page from Nike"

    page_samples = [
        metadata_config_service.build_page_sample(html, url) for url, html in page_pairs
    ]
    assert metadata_config_service.validate_config(config, page_samples)

    extractor = ConfigDrivenMetadataExtractor(config)
    rich_total = 0
    for url, html in page_pairs:
        candidates = extractor.extract_image_candidates(html, url, url)
        rich_total += len(metadata_rich_candidates(candidates))

    assert rich_total >= 3, f"expected metadata-rich candidates from inferred config, got {rich_total}"
