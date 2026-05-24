from datetime import timezone

import pytest
import respx

from src.models.brand_configs import BrandConfig
from src.models.metadata_extraction_config import MetadataExtractionConfig
from src.services.http_service import HttpClient
from src.crawlers.website_crawler import (
    WebsiteCrawler,
    brand_token_from_url,
    is_allowed_image_url,
    is_same_domain,
)
from src.utils.metadata_extractor import extract_embedded_image_urls, extract_image_candidates, is_plausible_image_url


BASE_URL = "https://example.com"


def _html(content: str) -> str:
    return f"<!doctype html><html><body>{content}</body></html>"


def _product_page_html(name: str, sku: str, image_path: str, extra: str = "") -> str:
    return _html(
        f"""
        <script type="application/ld+json">
        {{
          "@type": "Product",
          "name": "{name}",
          "sku": "{sku}",
          "image": "{image_path}"
        }}
        </script>
        <img src="{image_path}">
        {extra}
        """
    )


@pytest.fixture
def generic_metadata_config_service():
    class _StubMetadataConfigService:
        def load_or_infer(self, brand_url, refresh=False):
            return MetadataExtractionConfig.generic()

    return _StubMetadataConfigService()


def test_extract_image_urls_from_html():
    page_html = _html(
        '<img src="/a.jpg" data-src="/b.jpg" srcset="/c.jpg 1x, /d.jpg 2x">'
        '<meta property="og:image" content="/og.jpg">'
    )
    urls = {
        candidate.url
        for candidate in extract_image_candidates(page_html, BASE_URL, BASE_URL)
    }
    assert urls == {
        f"{BASE_URL}/a.jpg",
        f"{BASE_URL}/b.jpg",
        f"{BASE_URL}/c.jpg",
        f"{BASE_URL}/d.jpg",
        f"{BASE_URL}/og.jpg",
    }


def test_is_same_domain_normalizes_www():
    assert is_same_domain("https://www.example.com/page", "https://example.com") is True
    assert is_same_domain("https://other.com/page", "https://example.com") is False


def test_is_allowed_image_url_allows_brand_cdn():
    seed = "https://www.nike.com/"
    assert is_allowed_image_url(
        "https://static.nike.com/a/images/t_default/product.png",
        seed,
    )
    assert is_allowed_image_url(
        "https://c.static-nike.com/a/images/w_1920,c_limit/image.jpg",
        seed,
    )
    assert not is_allowed_image_url("https://other.com/external.jpg", seed)


def test_is_plausible_image_url_rejects_broken_transform_paths():
    assert not is_plausible_image_url(
        "https://www.nike.com/tr/w/fl_layer_apply/740ff5df-8988-4eeb-a11d-3bcda3d3f298/product.png"
    )
    assert is_plausible_image_url("https://static.nike.com/a/images/product.png")


def test_extract_embedded_image_urls_from_json_fields():
    page_html = _html(
        '"portraitURL":"https://static.nike.com/a/images/t_default/product.png",'
        '"note":"https://other.com/tracker.png"'
    )
    urls = extract_embedded_image_urls(page_html)
    assert "https://static.nike.com/a/images/t_default/product.png" in urls
    assert "https://other.com/tracker.png" in urls


def test_brand_token_from_url():
    assert brand_token_from_url("https://www.nike.com/") == "nike"


def test_extract_candidates_from_json_ld_product():
    page_html = _html(
        """
        <script type="application/ld+json">
        {
          "@type": "Product",
          "name": "Red Jacket",
          "sku": "SKU-123",
          "image": "/product.jpg"
        }
        </script>
        <img src="/product.jpg">
        """
    )
    candidates = extract_image_candidates(page_html, BASE_URL, BASE_URL)
    assert len(candidates) == 1
    assert candidates[0].product_name == "Red Jacket"
    assert candidates[0].sku_id == "SKU-123"


def test_extract_candidates_from_nike_inline_product_block(mock_extraction_config):
    page_html = _html(
        """
        {
          "copy": {"title": "Nike Air Force 1 '07"},
          "globalProductId": "12196903-727e-3fb6-836b-c612f86cec7f",
          "colorwayImages": {
            "portraitURL": "https://static.nike.com/a/images/t_default/dd3c1e9a-7dcd-45ee-adb5-b089df15e741/product.png",
            "squarishURL": "https://static.nike.com/a/images/t_default/740ff5df-8988-4eeb-a11d-3bcda3d3f298/product.png"
          },
          "pdpUrl": {"url": "https://www.nike.com/tr/t/air-force-1-07/IU7557-100"}
        }
        """
    )
    candidates = extract_image_candidates(
        page_html,
        "https://www.nike.com",
        "https://www.nike.com/tr/w/shoes",
        mock_extraction_config,
    )
    with_meta = [candidate for candidate in candidates if candidate.product_name and candidate.sku_id]
    assert with_meta
    assert with_meta[0].product_name == "Nike Air Force 1 '07"
    assert with_meta[0].sku_id == "IU7557-100"


def test_extract_candidates_from_nike_product_card_dom(mock_extraction_config):
    page_html = _html(
        """
        <div data-testid="product-card">
          <a class="product-card__link-overlay" href="https://www.nike.com/tr/t/free-2025/HF1078-006">Nike Free 2025</a>
          <img src="https://static.nike.com/a/images/t_web_pw_592_v2/f_auto/u_9ddf04c7/example/e2014d87-32b1-4889-b960-4e6b016de55a/NIKE+FREE+2025.png">
        </div>
        """
    )
    candidates = extract_image_candidates(
        page_html,
        "https://www.nike.com",
        "https://www.nike.com/tr/w/kadin-antrenman-ve-spor-salonu-ayakkabilar-58jtoz5e1x6zy7ok",
        mock_extraction_config,
    )
    assert len(candidates) == 1
    assert candidates[0].product_name == "Nike Free 2025"
    assert candidates[0].sku_id == "HF1078-006"


def test_extract_candidates_prefers_pdp_url_style_code_over_product_code(mock_extraction_config):
    page_html = _html(
        """
        {
          "copy": {"title": "Nike Free 2025"},
          "productCode": "HF1078-102",
          "globalProductId": "5c359410-95f5-4dcd-bc80-b9a5cfbbd0e2",
          "colorwayImages": {
            "portraitURL": "https://static.nike.com/a/images/t_default/u_9ddf04c7/example/e2014d87-32b1-4889-b960-4e6b016de55a/NIKE+FREE+2025.png",
            "squarishURL": "https://static.nike.com/a/images/t_default/u_9ddf04c7/example/e2014d87-32b1-4889-b960-4e6b016de55a/NIKE+FREE+2025.png"
          },
          "pdpUrl": {"url": "https://www.nike.com/tr/t/free-2025-antrenman-ayakkabısı-HjdEXceo/HF1078-006"}
        }
        """
    )
    candidates = extract_image_candidates(
        page_html,
        "https://www.nike.com",
        "https://www.nike.com/tr/w/kadin-antrenman-ve-spor-salonu-ayakkabilar-58jtoz5e1x6zy7ok",
        mock_extraction_config,
    )
    with_meta = [candidate for candidate in candidates if candidate.sku_id]
    assert with_meta
    assert all(candidate.sku_id == "HF1078-006" for candidate in with_meta)
    assert all("5c359410" not in (candidate.sku_id or "") for candidate in with_meta)


def test_extract_candidates_from_dom_data_attributes():
    page_html = _html(
        '<div data-sku="ABC" data-product-name="Shoe"><img src="/shoe.jpg"></div>'
    )
    candidates = extract_image_candidates(page_html, BASE_URL, BASE_URL)
    assert len(candidates) == 1
    assert candidates[0].product_name == "Shoe"
    assert candidates[0].sku_id == "ABC"


def test_extract_candidates_without_product_context():
    page_html = _html('<img src="/hero.jpg">')
    candidates = extract_image_candidates(page_html, BASE_URL, BASE_URL)
    assert len(candidates) == 1
    assert candidates[0].product_name is None
    assert candidates[0].sku_id is None


@respx.mock
def test_crawl_accepts_images_meeting_resolution(
    sample_png_bytes, sample_small_png_bytes, settings, brand_config, generic_metadata_config_service
):
    respx.get(f"{BASE_URL}/").respond(
        200,
        text=_product_page_html(
            "Big Product",
            "SKU-BIG",
            "/big.jpg",
            extra='<a href="/page2">next</a>',
        ),
        headers={"content-type": "text/html"},
    )
    respx.get(f"{BASE_URL}/page2").respond(
        200,
        text=_product_page_html(
            "OG Product",
            "SKU-OG",
            "/og.jpg",
            extra='<img srcset="/big.jpg 2x, /small.jpg 1x">',
        ),
        headers={"content-type": "text/html"},
    )
    respx.get(f"{BASE_URL}/big.jpg").respond(200, content=sample_png_bytes)
    respx.get(f"{BASE_URL}/og.jpg").respond(200, content=sample_png_bytes)
    respx.get(f"{BASE_URL}/small.jpg").respond(200, content=sample_small_png_bytes)

    client = HttpClient()
    try:
        crawler = WebsiteCrawler(
            client,
            settings,
            metadata_config_service=generic_metadata_config_service,
        )
        result = crawler.crawl(brand_config)
    finally:
        client.close()

    image_urls = {str(image.url) for image in result.images}
    assert f"{BASE_URL}/big.jpg" in image_urls
    assert f"{BASE_URL}/og.jpg" in image_urls
    assert f"{BASE_URL}/small.jpg" not in image_urls


@respx.mock
def test_crawl_stops_at_product_count_limit(
    sample_png_bytes, settings, generic_metadata_config_service
):
    respx.get(f"{BASE_URL}/").respond(
        200,
        text=_product_page_html(
            "Big Product",
            "SKU-BIG",
            "/big.jpg",
            extra='<img src="/og.jpg">',
        ),
        headers={"content-type": "text/html"},
    )
    respx.get(f"{BASE_URL}/big.jpg").respond(200, content=sample_png_bytes)
    respx.get(f"{BASE_URL}/og.jpg").respond(200, content=sample_png_bytes)

    client = HttpClient()
    try:
        crawler = WebsiteCrawler(
            client,
            {
                **settings,
                "lowest_processeable_product_count": 1,
            },
            metadata_config_service=generic_metadata_config_service,
        )
        result = crawler.crawl(BrandConfig(url=BASE_URL))
    finally:
        client.close()

    assert len(result.images) == 1
    assert len({image.sku_id for image in result.images}) == 1


@respx.mock
def test_crawl_deduplicates_by_sku_id(sample_png_bytes, settings, generic_metadata_config_service):
    respx.get(f"{BASE_URL}/").respond(
        200,
        text=_html(
            """
            <script type="application/ld+json">
            {
              "@type": "Product",
              "name": "Red Jacket",
              "sku": "SKU-123",
              "image": ["/product-a.jpg", "/product-b.jpg"]
            }
            </script>
            <img src="/product-a.jpg">
            <img src="/product-b.jpg">
            """
        ),
        headers={"content-type": "text/html"},
    )
    respx.get(f"{BASE_URL}/product-a.jpg").respond(200, content=sample_png_bytes)
    respx.get(f"{BASE_URL}/product-b.jpg").respond(200, content=sample_png_bytes)

    client = HttpClient()
    try:
        crawler = WebsiteCrawler(
            client,
            {
                **settings,
                "lowest_processeable_product_count": 5,
            },
            metadata_config_service=generic_metadata_config_service,
        )
        result = crawler.crawl(BrandConfig(url=BASE_URL))
    finally:
        client.close()

    assert len(result.images) == 1
    assert result.images[0].sku_id == "SKU-123"


@respx.mock
def test_crawl_same_domain_only(sample_png_bytes, settings, generic_metadata_config_service):
    respx.get(f"{BASE_URL}/").respond(
        200,
        text=_product_page_html(
            "Big Product",
            "SKU-BIG",
            "/big.jpg",
            extra=(
                '<img src="https://other.com/external.jpg">'
                '<a href="https://other.com/page">external</a>'
            ),
        ),
        headers={"content-type": "text/html"},
    )
    respx.get(f"{BASE_URL}/big.jpg").respond(200, content=sample_png_bytes)

    client = HttpClient()
    try:
        crawler = WebsiteCrawler(
            client,
            settings,
            metadata_config_service=generic_metadata_config_service,
        )
        result = crawler.crawl(BrandConfig(url=BASE_URL))
    finally:
        client.close()

    image_urls = {str(image.url) for image in result.images}
    assert image_urls == {f"{BASE_URL}/big.jpg"}
    assert result.pages_crawled == 1


@respx.mock
def test_crawl_deduplicates_images(sample_png_bytes, settings, generic_metadata_config_service):
    respx.get(f"{BASE_URL}/").respond(
        200,
        text=_product_page_html(
            "Big Product",
            "SKU-BIG",
            "/big.jpg",
            extra='<a href="/page2">next</a>',
        ),
        headers={"content-type": "text/html"},
    )
    respx.get(f"{BASE_URL}/page2").respond(
        200,
        text=_product_page_html("Big Product", "SKU-BIG", "/big.jpg"),
        headers={"content-type": "text/html"},
    )
    respx.get(f"{BASE_URL}/big.jpg").respond(200, content=sample_png_bytes)

    client = HttpClient()
    try:
        crawler = WebsiteCrawler(
            client,
            settings,
            metadata_config_service=generic_metadata_config_service,
        )
        result = crawler.crawl(BrandConfig(url=BASE_URL))
    finally:
        client.close()

    assert len(result.images) == 1
    assert str(result.images[0].url) == f"{BASE_URL}/big.jpg"


@respx.mock
def test_crawl_respects_max_depth(sample_png_bytes, settings, generic_metadata_config_service):
    pages = {
        f"{BASE_URL}/": _html('<a href="/level1">next</a>'),
        f"{BASE_URL}/level1": _html('<a href="/level2">next</a>'),
        f"{BASE_URL}/level2": _html('<a href="/level3">next</a>'),
        f"{BASE_URL}/level3": _html('<a href="/level4">next</a>'),
        f"{BASE_URL}/level4": _html('<a href="/level5">next</a>'),
        f"{BASE_URL}/level5": _html('<a href="/level6">next</a>'),
        f"{BASE_URL}/level6": _product_page_html("Deep Product", "SKU-DEEP", "/deep.jpg"),
    }

    for url, body in pages.items():
        respx.get(url).respond(
            200,
            text=body,
            headers={"content-type": "text/html"},
        )
    respx.get(f"{BASE_URL}/deep.jpg").respond(200, content=sample_png_bytes)

    client = HttpClient()
    try:
        crawler = WebsiteCrawler(
            client,
            settings,
            metadata_config_service=generic_metadata_config_service,
        )
        result = crawler.crawl(BrandConfig(url=BASE_URL))
    finally:
        client.close()

    image_urls = {str(image.url) for image in result.images}
    assert f"{BASE_URL}/deep.jpg" not in image_urls
    assert result.pages_crawled <= 6


@respx.mock
def test_get_stream_truncates_large_response():
    respx.get(f"{BASE_URL}/large.bin").respond(200, content=b"x" * 1024)

    client = HttpClient()
    try:
        data = client.get_stream(f"{BASE_URL}/large.bin", max_bytes=128)
    finally:
        client.close()

    assert data is not None
    assert len(data) == 128


@respx.mock
def test_crawl_propagates_product_metadata(sample_png_bytes, settings, generic_metadata_config_service):
    respx.get(f"{BASE_URL}/").respond(
        200,
        text=_html(
            """
            <script type="application/ld+json">
            {
              "@type": "Product",
              "name": "Red Jacket",
              "sku": "SKU-123",
              "image": "/product.jpg"
            }
            </script>
            <img src="/product.jpg">
            """
        ),
        headers={"content-type": "text/html"},
    )
    respx.get(f"{BASE_URL}/product.jpg").respond(200, content=sample_png_bytes)

    client = HttpClient()
    try:
        crawler = WebsiteCrawler(
            client,
            settings,
            metadata_config_service=generic_metadata_config_service,
        )
        result = crawler.crawl(BrandConfig(url=BASE_URL))
    finally:
        client.close()

    assert len(result.images) == 1
    assert result.images[0].product_name == "Red Jacket"
    assert result.images[0].sku_id == "SKU-123"


@respx.mock
def test_crawl_skips_images_without_complete_metadata(
    sample_png_bytes, settings, generic_metadata_config_service
):
    respx.get(f"{BASE_URL}/").respond(
        200,
        text=_html(
            """
            <script type="application/ld+json">
            {
              "@type": "Product",
              "name": "Name Only",
              "image": "/name-only.jpg"
            }
            </script>
            <img src="/name-only.jpg">
            <img src="/bare.jpg">
            """
        ),
        headers={"content-type": "text/html"},
    )
    respx.get(f"{BASE_URL}/name-only.jpg").respond(200, content=sample_png_bytes)
    respx.get(f"{BASE_URL}/bare.jpg").respond(200, content=sample_png_bytes)

    client = HttpClient()
    try:
        crawler = WebsiteCrawler(
            client,
            settings,
            metadata_config_service=generic_metadata_config_service,
        )
        result = crawler.crawl(BrandConfig(url=BASE_URL))
    finally:
        client.close()

    assert result.images == []


@respx.mock
def test_crawl_result_has_created_at(sample_png_bytes, settings, generic_metadata_config_service):
    respx.get(f"{BASE_URL}/").respond(
        200,
        text=_html('<img src="/big.jpg">'),
        headers={"content-type": "text/html"},
    )
    respx.get(f"{BASE_URL}/big.jpg").respond(200, content=sample_png_bytes)

    client = HttpClient()
    try:
        crawler = WebsiteCrawler(
            client,
            settings,
            metadata_config_service=generic_metadata_config_service,
        )
        result = crawler.crawl(BrandConfig(url=BASE_URL))
    finally:
        client.close()

    assert result.images == []
    assert result.created_at.tzinfo is not None
    assert result.created_at.tzinfo == timezone.utc
