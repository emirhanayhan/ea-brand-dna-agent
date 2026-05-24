import pytest
import respx

from src.services.http_service import HttpClient
from src.services.social_network_verifier import (
    check_social_media_exists,
    validate_html_for_instagram,
    validate_html_for_twitter,
)

VALID_TWITTER_HTML = '<html><body>{"profile":"nike"}</body></html>'
INVALID_TWITTER_HTML = '<html><body>"url":"userbyrestid","status":401</body></html>'

VALID_INSTAGRAM_HTML = '<html><body>{"username":"nike","edge_owner_to_timeline_media":{}}</body></html>'
INVALID_INSTAGRAM_HTML = '<html><body>"error_type":"user_not_found"</body></html>'


def test_validate_html_for_twitter_valid():
    assert validate_html_for_twitter(VALID_TWITTER_HTML.lower()) is True


def test_validate_html_for_twitter_invalid():
    assert validate_html_for_twitter(INVALID_TWITTER_HTML.lower()) is False


def test_validate_html_for_instagram_valid():
    assert validate_html_for_instagram(VALID_INSTAGRAM_HTML.lower()) is True


def test_validate_html_for_instagram_invalid():
    assert validate_html_for_instagram(INVALID_INSTAGRAM_HTML.lower()) is False


@respx.mock
def test_check_social_media_exists_valid_nike_mocked():
    respx.get("https://x.com/nike").respond(200, text=VALID_TWITTER_HTML)
    respx.get("https://www.instagram.com/nike/").respond(200, text=VALID_INSTAGRAM_HTML)

    client = HttpClient()
    try:
        result = check_social_media_exists(client, "@nike")
    finally:
        client.close()

    assert result == {
        "twitter": "https://x.com/nike",
        "instagram": "https://www.instagram.com/nike/",
    }


@respx.mock
def test_check_social_media_exists_invalid_nikee_mocked():
    respx.get("https://x.com/nikee").respond(200, text=INVALID_TWITTER_HTML)
    respx.get("https://www.instagram.com/nikee/").respond(200, text=INVALID_INSTAGRAM_HTML)

    client = HttpClient()
    try:
        with pytest.raises(Exception, match="Social network validation failed"):
            check_social_media_exists(client, "@nikee")
    finally:
        client.close()
