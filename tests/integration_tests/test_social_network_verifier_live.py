import pytest

from src.services.http_service import HttpClient
from src.services.social_network_verifier import check_social_media_exists


@pytest.mark.integration
def test_check_social_media_exists_valid_nike_live():
    client = HttpClient()
    try:
        result = check_social_media_exists(client, "@nike")
    finally:
        client.close()

    assert result["twitter"] == "https://x.com/nike"
    assert result["instagram"] == "https://www.instagram.com/nike/"


@pytest.mark.integration
def test_check_social_media_exists_invalid_nikee_live():
    client = HttpClient()
    try:
        with pytest.raises(Exception, match="Social network validation failed"):
            check_social_media_exists(client, "@nikee")
    finally:
        client.close()
