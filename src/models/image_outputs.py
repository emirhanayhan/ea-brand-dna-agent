from typing import Literal, Optional

from pydantic import BaseModel, HttpUrl

# What kind of page this image came from. Drives cluster-rep ranking:
# product shots beat editorial campaign hero images (which often carry
# watermarks and brand overlays) when both compete for a representative slot.
SourceKind = Literal["product", "editorial", "social", "other"]


class DiscoveredImage(BaseModel):
    url: HttpUrl
    width: int
    height: int
    source_page: HttpUrl
    product_name: Optional[str] = None
    sku_id: Optional[str] = None
    source_kind: SourceKind = "product"


class BotProtectionFailure(BaseModel):
    url: HttpUrl
    provider: str
    reason: str
