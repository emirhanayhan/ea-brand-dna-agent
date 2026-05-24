from src.models.metadata_extraction_config import (
    DomContextRule,
    JsonSourceRule,
    MetadataExtractionConfig,
    ProductCardRule,
)


def build_mock_extraction_config() -> MetadataExtractionConfig:
    return MetadataExtractionConfig(
        domain="nike.com",
        product_card=ProductCardRule(
            card_selector='//*[@data-testid="product-card"]',
            link_selector='.//a[contains(@class, "product-card__link-overlay")]',
            name_from_link_text=True,
            sku_from_link_url=True,
            image_within_card=True,
        ),
        json_sources=[
            JsonSourceRule(
                source="json_ld",
                name_keys=["name"],
                sku_keys=["sku", "productID", "mpn"],
                sku_prefer_pdp_url=False,
                pdp_url_keys=["pdpUrl.url"],
                image_keys=["image"],
            ),
            JsonSourceRule(
                source="next_data",
                name_keys=["copy.title", "title", "fullTitle", "productTitle", "name"],
                sku_keys=["styleColor", "productCode", "sku", "productID", "productId", "mpn"],
                sku_prefer_pdp_url=True,
                pdp_url_keys=["pdpUrl.url"],
                image_keys=[
                    "colorwayImages.portraitURL",
                    "colorwayImages.squarishURL",
                    "portraitURL",
                    "squarishURL",
                    "portraitImg",
                    "squarishImg",
                    "image",
                    "imageUrl",
                ],
            ),
            JsonSourceRule(
                source="inline_json",
                name_keys=["copy.title"],
                sku_keys=["productCode", "styleColor"],
                sku_prefer_pdp_url=True,
                pdp_url_keys=["pdpUrl.url"],
                image_keys=["portraitURL", "squarishURL"],
            ),
        ],
        dom_context=DomContextRule(
            ancestor_limit=5,
            card_selector='//*[@data-testid="product-card"]',
            card_link_selector='.//a[contains(@class, "product-card__link-overlay")]',
            sku_attrs=["data-sku", "data-product-sku", "data-product-id", "data-item-id"],
            name_attrs=["data-product-name", "data-product-title"],
        ),
        embedded_image_json_fields=[
            "portraitURL",
            "squarishURL",
            "portraitImg",
            "squarishImg",
            "imageUrl",
            "image_url",
            "thumbnailUrl",
            "src",
        ],
        sku_style_pattern=r"^[A-Z0-9]*\d[A-Z0-9]*-\d+$",
        single_product_page_path_contains="/t/",
        priority_link_path_contains=["/w/", "/t/"],
    )
