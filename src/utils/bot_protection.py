import re
from dataclasses import dataclass
from typing import Mapping, Optional

import httpx

_AKAMAI_BODY_MARKERS = (
    "akamai",
    "page_owner",
    "access denied",
    "reference #",
    "edgesuite.net",
)
_AKAMAI_ERROR_CODES = (
    "GE401001",
)
_INTERSTITIAL_MARKERS = (
    "bm-verify",
    "/interstitial/ic.html",
    "triggerinterstitialchallenge",
)
_MIN_INTERSTITIAL_BODY_LENGTH = 8_000


@dataclass(frozen=True)
class BotProtectionDetection:
    provider: str
    reason: str

    def __str__(self) -> str:
        return f"{self.provider}: {self.reason}"


def detect_bot_protection(
    *,
    status_code: int,
    headers: Mapping[str, str],
    body: str,
) -> Optional[BotProtectionDetection]:
    normalized_body = body.lower()
    header_values = " ".join(str(value).lower() for value in headers.values())

    if status_code == 429:
        return BotProtectionDetection(provider="unknown", reason="http_429_rate_limited")

    if status_code == 403:
        if any(marker in normalized_body for marker in _AKAMAI_BODY_MARKERS):
            return BotProtectionDetection(provider="akamai", reason="http_403_forbidden")
        if "ak_bmsc" in header_values or "akamai" in header_values:
            return BotProtectionDetection(provider="akamai", reason="http_403_forbidden")
        return BotProtectionDetection(provider="unknown", reason="http_403_forbidden")

    if status_code == 400:
        if "akamaighost" in header_values or any(
            code in body for code in _AKAMAI_ERROR_CODES
        ):
            return BotProtectionDetection(provider="akamai", reason="http_400_bad_request")
        return None

    if status_code != 200 or not body:
        return None

    if any(marker in normalized_body for marker in _INTERSTITIAL_MARKERS):
        return BotProtectionDetection(provider="akamai", reason="interstitial_challenge")

    if len(body) < _MIN_INTERSTITIAL_BODY_LENGTH and re.search(
        r"<title>\s*&nbsp;\s*</title>",
        body,
        re.IGNORECASE,
    ):
        return BotProtectionDetection(provider="akamai", reason="interstitial_challenge")

    return None


def detect_bot_protection_from_response(
    response: httpx.Response,
) -> Optional[BotProtectionDetection]:
    return detect_bot_protection(
        status_code=response.status_code,
        headers=response.headers,
        body=response.text,
    )
