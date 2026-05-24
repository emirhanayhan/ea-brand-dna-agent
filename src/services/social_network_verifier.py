import logging
import re
from typing import Callable, Dict, Optional

logger = logging.getLogger("BrandAgent.NetworkVerifier")

INSTAGRAM_INVALID_MARKERS = (
    '"error_type":"user_not_found"',
    '"page_not_found"',
    "sorry, this page isn't available",
)

URL_TEMPLATES = {
    "twitter": "https://x.com/{}",
    "instagram": "https://www.instagram.com/{}/",
}

INSTAGRAM_API_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "X-IG-App-ID": "936619743392459",
}


def normalize_social_handle(handle: str) -> str:
    handle = handle.strip().lower()

    # remove leading @ and repeated symbols
    handle = re.sub(r"^@+", "", handle)

    # remove invalid characters (optional but safer)
    handle = re.sub(r"[^a-z0-9._]", "", handle)

    return handle


def validate_html_for_twitter(html: str) -> bool:
    twitter_error_signature = '"url":"userbyrestid","status":401'
    if twitter_error_signature in html:
        return False
    return True


def validate_html_for_instagram(html: str) -> bool:
    for marker in INSTAGRAM_INVALID_MARKERS:
        if marker in html:
            return False
    return True


def _confirm_instagram_user_via_api(http_service, handle: str) -> bool:
    url = f"https://www.instagram.com/api/v1/users/web_profile_info/?username={handle}"
    try:
        response = http_service.client.get(url, headers=INSTAGRAM_API_HEADERS)
        if not response or response.status_code != 200:
            return False
        payload = response.json()
        user = payload.get("data", {}).get("user")
        return isinstance(user, dict) and user.get("username", "").lower() == handle
    except Exception as e:
        logger.warning(f"instagram api check failed: {e}")
        return False


def instagram_profile_exists(http_service, handle: str, html: str) -> bool:
    if not validate_html_for_instagram(html):
        return False
    if f'"username":"{handle}"' in html:
        return True
    return _confirm_instagram_user_via_api(http_service, handle)


HTML_VALIDATORS: Dict[str, Callable[[str], bool]] = {
    "twitter": validate_html_for_twitter,
}


def check_social_media_exists(http_service, social_media_handle: str) -> Optional[Dict[str, str]]:
    """
    Returns:
        - Dict of existing social URLs if any exist
        - None if no social media profiles are found
    """
    social_media_handle = normalize_social_handle(social_media_handle)

    results = {}

    for network, template in URL_TEMPLATES.items():
        url = template.format(social_media_handle)

        try:
            response = http_service.get(url)
            if not response or response.status_code != 200:
                continue

            html_content = response.text.lower()
            if network == "instagram":
                is_valid = instagram_profile_exists(
                    http_service,
                    social_media_handle,
                    html_content,
                )
            else:
                validator = HTML_VALIDATORS[network]
                is_valid = validator(html_content)

            if is_valid:
                results[network] = url

        except Exception as e:
            logger.warning(f"{network} check failed: {e}")

    if not results:
        raise Exception("Social network validation failed")

    return results
