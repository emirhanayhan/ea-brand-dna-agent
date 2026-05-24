import httpx
import pytest

from src.models.brand_configs import BrandConfig
from src.services.http_service import HttpClient
from src.crawlers.website_crawler import WebsiteCrawler
from src.utils.bot_protection import (
    BotProtectionDetection,
    detect_bot_protection,
    detect_bot_protection_from_response,
)


ZARA_INTERSTITIAL_HTML = """<!DOCTYPE html><html><head>
<meta http-equiv="refresh" content="5; URL='/?bm-verify=AAQAAAAN_____token'" />
<title>&nbsp;</title>
<script>function triggerInterstitialChallenge() {}</script>
</head><body>
<iframe src="/interstitial/ic.html"></iframe>
</body></html>"""

ADIDAS_403_HTML = """<!DOCTYPE html>
<html lang="en"><head><title>Access Denied</title></head>
<body>Reference #18.page_owner: AKAMAI</body></html>"""


def test_detect_bot_protection_zara_interstitial():
    detection = detect_bot_protection(
        status_code=200,
        headers={"content-type": "text/html"},
        body=ZARA_INTERSTITIAL_HTML,
    )
    assert detection == BotProtectionDetection(
        provider="akamai",
        reason="interstitial_challenge",
    )


def test_detect_bot_protection_adidas_403():
    detection = detect_bot_protection(
        status_code=403,
        headers={"content-type": "text/html"},
        body=ADIDAS_403_HTML,
    )
    assert detection == BotProtectionDetection(
        provider="akamai",
        reason="http_403_forbidden",
    )


def test_detect_bot_protection_lululemon_400():
    detection = detect_bot_protection(
        status_code=400,
        headers={"server": "AkamaiGHost", "content-type": "application/json"},
        body='{"message": "Bad Request.", "errorCode": "GE401001"}',
    )
    assert detection == BotProtectionDetection(
        provider="akamai",
        reason="http_400_bad_request",
    )


def test_detect_bot_protection_lululemon_403_edgesuite():
    detection = detect_bot_protection(
        status_code=403,
        headers={"server": "AkamaiGHost", "content-type": "text/html"},
        body=(
            "<HTML><HEAD><TITLE>Access Denied</TITLE></HEAD><BODY>"
            "<H1>Access Denied</H1>"
            "https://errors.edgesuite.net/18.example</BODY></HTML>"
        ),
    )
    assert detection == BotProtectionDetection(
        provider="akamai",
        reason="http_403_forbidden",
    )


def test_detect_bot_protection_allows_normal_homepage():
    detection = detect_bot_protection(
        status_code=200,
        headers={"content-type": "text/html"},
        body="<!doctype html><html><body><a href='/products'>Shop</a></body></html>" * 100,
    )
    assert detection is None


def test_detect_bot_protection_from_response():
    response = httpx.Response(
        429,
        text="rate limited",
        request=httpx.Request("GET", "https://example.com/"),
    )
    detection = detect_bot_protection_from_response(response)
    assert detection == BotProtectionDetection(
        provider="unknown",
        reason="http_429_rate_limited",
    )


@pytest.fixture
def generic_metadata_config_service():
    class _StubMetadataConfigService:
        def load_or_infer(self, brand_url, refresh=False):
            from src.models.metadata_extraction_config import MetadataExtractionConfig

            return MetadataExtractionConfig.generic()

    return _StubMetadataConfigService()


def test_http_client_records_bot_protection(respx_mock):
    respx_mock.get("https://blocked.example/").respond(
        200,
        text=ZARA_INTERSTITIAL_HTML,
        headers={"content-type": "text/html"},
    )

    client = HttpClient()
    try:
        response = client.get("https://blocked.example/")
        assert response is None
        assert client.last_bot_protection == BotProtectionDetection(
            provider="akamai",
            reason="interstitial_challenge",
        )
    finally:
        client.close()


def test_crawler_stops_on_seed_bot_protection(respx_mock, settings, generic_metadata_config_service):
    respx_mock.get("https://blocked.example/").respond(
        200,
        text=ZARA_INTERSTITIAL_HTML,
        headers={"content-type": "text/html"},
    )

    client = HttpClient()
    try:
        crawler = WebsiteCrawler(
            client,
            settings,
            metadata_config_service=generic_metadata_config_service,
        )
        result = crawler.crawl(BrandConfig(url="https://blocked.example/"))
    finally:
        client.close()

    assert result.stopped_due_to_bot_protection is True
    assert str(result.bot_protection.url) == "https://blocked.example/"
    assert result.bot_protection.provider == "akamai"
    assert result.bot_protection.reason == "interstitial_challenge"
    assert result.pages_crawled == 0
    assert result.images == []


def test_crawler_stops_when_bot_protection_hits_later_page(
    respx_mock,
    settings,
    generic_metadata_config_service,
):
    base_url = "https://example.com"
    respx_mock.get(f"{base_url}/").respond(
        200,
        text="<!doctype html><html><body><a href='/page2'>next</a></body></html>",
        headers={"content-type": "text/html"},
    )
    respx_mock.get(f"{base_url}/page2").respond(
        200,
        text=ZARA_INTERSTITIAL_HTML,
        headers={"content-type": "text/html"},
    )

    client = HttpClient()
    try:
        crawler = WebsiteCrawler(
            client,
            settings,
            metadata_config_service=generic_metadata_config_service,
        )
        result = crawler.crawl(BrandConfig(url=base_url))
    finally:
        client.close()

    assert result.stopped_due_to_bot_protection is True
    assert str(result.bot_protection.url) == f"{base_url}/page2"
    assert result.bot_protection.reason == "interstitial_challenge"
    assert result.pages_crawled == 1


def test_crawler_continues_on_non_bot_request_failures(
    respx_mock,
    settings,
    generic_metadata_config_service,
):
    base_url = "https://example.com"
    respx_mock.get(f"{base_url}/").respond(
        200,
        text="<!doctype html><html><body><a href='/missing'>next</a></body></html>",
        headers={"content-type": "text/html"},
    )
    respx_mock.get(f"{base_url}/missing").respond(404)

    client = HttpClient()
    try:
        crawler = WebsiteCrawler(
            client,
            settings,
            metadata_config_service=generic_metadata_config_service,
        )
        result = crawler.crawl(BrandConfig(url=base_url))
    finally:
        client.close()

    assert result.stopped_due_to_bot_protection is False
    assert result.pages_crawled == 1
    assert result.images == []
