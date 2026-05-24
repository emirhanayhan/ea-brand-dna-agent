from typing import Literal, Optional

from pydantic import BaseModel, HttpUrl


class PageSample(BaseModel):
    url: HttpUrl
    page_type: Literal["homepage", "listing", "pdp", "other"] = "other"
    html_snippet: str
    notes: Optional[str] = None
