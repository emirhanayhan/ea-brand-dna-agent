from typing import Dict, Optional

from pydantic import BaseModel, HttpUrl


class BrandConfig(BaseModel):
    """The input configuration for the brand."""
    name: Optional[str] = None
    url: Optional[HttpUrl] = None
    social_handles: Optional[Dict[str, str]] = None
