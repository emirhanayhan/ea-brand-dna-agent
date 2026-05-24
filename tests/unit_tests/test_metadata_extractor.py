from src.models.metadata_extraction_config import (
    JsonSourceRule,
    MetadataExtractionConfig,
    ProductCardRule,
)
from src.utils.metadata_extractor import ConfigDrivenMetadataExtractor, extract_image_candidates


def _html(content: str) -> str:
    return f"<!doctype html><html><body>{content}</body></html>"


def test_generic_config_extracts_json_ld_product():
    page_html = _html(
        """
        <script type="application/ld+json">
        {
          "@type": "Product",
          "name": "Silk Scarf",
          "sku": "GUC-123",
          "image": "https://cdn.example.com/scarf.jpg"
        }
        </script>
        <img src="https://cdn.example.com/scarf.jpg">
        """
    )
    candidates = extract_image_candidates(
        page_html,
        "https://www.gucci.com",
        "https://www.gucci.com/en/product/scarf",
    )
    assert len(candidates) == 1
    assert candidates[0].product_name == "Silk Scarf"
    assert candidates[0].sku_id == "GUC-123"


def test_custom_config_extracts_product_card():
    config = MetadataExtractionConfig(
        domain="gucci.com",
        product_card=ProductCardRule(
            card_selector='//*[@class="product-tile"]',
            link_selector='.//a[@class="product-tile-link"]',
        ),
        dom_context={
            "card_selector": '//*[@class="product-tile"]',
            "card_link_selector": './/a[@class="product-tile-link"]',
            "sku_attrs": ["data-sku"],
            "name_attrs": ["data-product-name"],
        },
    )
    page_html = _html(
        """
        <div class="product-tile" data-sku="8056649" data-product-name="Horsebit Loafer">
          <a class="product-tile-link" href="https://www.gucci.com/en/product/8056649">Horsebit Loafer</a>
          <img src="https://cdn.example.com/loafer.jpg">
        </div>
        """
    )
    extractor = ConfigDrivenMetadataExtractor(config)
    candidates = extractor.extract_image_candidates(
        page_html,
        "https://www.gucci.com",
        "https://www.gucci.com/en/shop",
    )
    assert len(candidates) == 1
    assert candidates[0].product_name == "Horsebit Loafer"
    assert candidates[0].sku_id == "8056649"


def test_custom_json_source_rule():
    config = MetadataExtractionConfig(
        domain="gucci.com",
        json_sources=[
            JsonSourceRule(
                source="inline_json",
                name_keys=["copy.title"],
                sku_keys=["productCode"],
                sku_prefer_pdp_url=True,
                pdp_url_keys=["pdpUrl.url"],
                image_keys=["portraitURL", "squarishURL"],
            )
        ],
    )
    page_html = _html(
        """
        {
          "copy": {"title": "Gucci Bag"},
          "productCode": "123456",
          "portraitURL": "https://cdn.example.com/bag.png",
          "squarishURL": "https://cdn.example.com/bag.png",
          "pdpUrl": {"url": "https://www.gucci.com/en/product/HF1078-006"}
        }
        """
    )
    products = ConfigDrivenMetadataExtractor(config).parse_page_products(
        page_html,
        "https://www.gucci.com",
    )
    assert products
    assert products[0].name == "Gucci Bag"
    assert products[0].sku_id == "HF1078-006"


def test_algolia_state_source_extracts_products():
    config = MetadataExtractionConfig(
        domain="prada.com",
        sku_style_pattern=r"^[0-9]{6}_[0-9A-Z_]+$",
        json_sources=[
            JsonSourceRule(
                source="algolia_state",
                name_keys=["ProductName.en_GB"],
                sku_keys=["ParentVariant"],
                sku_prefer_pdp_url=True,
                pdp_url_keys=["UrlReconstructed.en_GB"],
                image_keys=["Images.PLPBKG"],
            )
        ],
    )
    page_html = _html(
        """
        <script>
        window.__ALGOLIA_STATE__ = {
          "PLP_COLOR_PRADA_Online_US": {
            "results": [{
              "hits": [{
                "ProductName": {"en_GB": "Re-Nylon piqué leggings"},
                "ParentVariant": "122048_1210_F0002_S_AAO",
                "UrlReconstructed": {"en_GB": "/p/re-nylon-pique-leggings/122048_1210_F0002_S_AAO"},
                "Images": {
                  "PLPBKG": "https://www.prada.com/content/dam/pradabkg_products/122048_1210_F0002_S_AAO_SLF.jpg"
                }
              }]
            }]
          }
        };
        </script>
        <img src="https://www.prada.com/content/dam/pradabkg_products/122048_1210_F0002_S_AAO_SLF.jpg">
        """
    )
    extractor = ConfigDrivenMetadataExtractor(config)
    products = extractor.parse_page_products(page_html, "https://www.prada.com")
    assert len(products) == 1
    assert products[0].name == "Re-Nylon piqué leggings"
    assert products[0].sku_id == "122048_1210_F0002_S_AAO"
