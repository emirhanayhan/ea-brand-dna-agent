from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ProductCardRule(BaseModel):
    card_selector: str
    link_selector: str
    name_from_link_text: bool = True
    sku_from_link_url: bool = True
    image_within_card: bool = True


class JsonSourceRule(BaseModel):
    source: Literal["next_data", "inline_json", "json_ld", "algolia_state"]
    name_keys: list[str] = Field(default_factory=list)
    sku_keys: list[str] = Field(default_factory=list)
    sku_prefer_pdp_url: bool = True
    pdp_url_keys: list[str] = Field(default_factory=lambda: ["pdpUrl.url"])
    image_keys: list[str] = Field(default_factory=list)


class DomContextRule(BaseModel):
    ancestor_limit: int = 5
    card_selector: Optional[str] = None
    card_link_selector: Optional[str] = None
    sku_attrs: list[str] = Field(
        default_factory=lambda: ["data-sku", "data-product-sku", "data-product-id", "data-item-id"]
    )
    name_attrs: list[str] = Field(
        default_factory=lambda: ["data-product-name", "data-product-title"]
    )


class MetadataExtractionConfig(BaseModel):
    domain: str = ""
    product_card: Optional[ProductCardRule] = None
    json_sources: list[JsonSourceRule] = Field(default_factory=list)
    dom_context: DomContextRule = Field(default_factory=DomContextRule)
    embedded_image_json_fields: list[str] = Field(
        default_factory=lambda: [
            "portraitURL",
            "squarishURL",
            "imageUrl",
            "image_url",
            "thumbnailUrl",
            "src",
        ]
    )
    sku_style_pattern: Optional[str] = r"^[A-Z0-9]*\d[A-Z0-9]*-\d+$"
    single_product_page_path_contains: Optional[str] = None
    priority_link_path_contains: list[str] = Field(default_factory=list)
    updated_at: Optional[datetime] = None

    @classmethod
    def generic(cls) -> "MetadataExtractionConfig":
        return cls(
            domain="",
            json_sources=[
                JsonSourceRule(
                    source="json_ld",
                    name_keys=["name"],
                    sku_keys=["sku", "productID", "mpn"],
                    sku_prefer_pdp_url=False,
                    image_keys=["image"],
                )
            ],
        )

    @classmethod
    def ecommerce_defaults(cls, domain: str = "") -> "MetadataExtractionConfig":
        return cls(
            domain=domain,
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
            priority_link_path_contains=["/w/", "/t/", "/products/", "/product/", "/p/", "/shop/"],
        )

    def normalized(self) -> "MetadataExtractionConfig":
        defaults = self.ecommerce_defaults(self.domain)
        dom = self.dom_context
        product_card = self.product_card

        dom_updates: dict = {}
        if product_card:
            if not dom.card_selector:
                dom_updates["card_selector"] = product_card.card_selector
            if not dom.card_link_selector:
                dom_updates["card_link_selector"] = product_card.link_selector
        if not dom.sku_attrs:
            dom_updates["sku_attrs"] = list(defaults.dom_context.sku_attrs)
        if not dom.name_attrs:
            dom_updates["name_attrs"] = list(defaults.dom_context.name_attrs)
        if dom.ancestor_limit < 3:
            dom_updates["ancestor_limit"] = defaults.dom_context.ancestor_limit

        priority_paths = list(self.priority_link_path_contains)
        for path in defaults.priority_link_path_contains:
            if path not in priority_paths:
                priority_paths.insert(0, path)
        if not priority_paths:
            priority_paths = list(defaults.priority_link_path_contains)

        json_sources = self.json_sources or list(defaults.json_sources)
        embedded_fields = self.embedded_image_json_fields or list(defaults.embedded_image_json_fields)
        sku_pattern = self.sku_style_pattern or defaults.sku_style_pattern
        single_product_path = self.single_product_page_path_contains or defaults.single_product_page_path_contains

        normalized_dom = dom.model_copy(update=dom_updates) if dom_updates else dom
        return self.model_copy(
            update={
                "json_sources": json_sources,
                "dom_context": normalized_dom,
                "embedded_image_json_fields": embedded_fields,
                "sku_style_pattern": sku_pattern,
                "single_product_page_path_contains": single_product_path,
                "priority_link_path_contains": priority_paths,
            }
        )
