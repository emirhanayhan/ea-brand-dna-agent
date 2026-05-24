import pytest
import respx

from src.models.brand_configs import BrandConfig
from src.services.http_service import HttpClient
from src.crawlers.instagram_scraper import InstagramScraper
from src.utils.text_snippet_extractor import is_promotional_social_post


PROFILE_URL = "https://www.instagram.com/nike/"
API_URL = (
    "https://www.instagram.com/api/v1/users/web_profile_info/?username=nike"
)


def _payload(
    biography: str = "Just do it.",
    posts=None,
):
    posts = posts if posts is not None else []
    return {
        "data": {
            "user": {
                "username": "nike",
                "biography": biography,
                "edge_owner_to_timeline_media": {"edges": posts},
            }
        }
    }


def _post_edge(
    shortcode: str,
    display_url: str,
    caption: str,
    is_video: bool = False,
    typename: str = "GraphImage",
    children=None,
):
    node = {
        "__typename": typename,
        "shortcode": shortcode,
        "display_url": display_url,
        "is_video": is_video,
        "edge_media_to_caption": {
            "edges": [{"node": {"text": caption}}] if caption else []
        },
    }
    if children is not None:
        node["edge_sidecar_to_children"] = {"edges": children}
    return {"node": node}


@respx.mock
def test_instagram_scraper_extracts_bio_caption_and_image(
    settings, sample_png_bytes
):
    settings = dict(settings)
    settings["instagram_max_posts"] = 12

    posts = [
        _post_edge(
            shortcode="abc",
            display_url="https://scontent.cdninstagram.com/post-1.jpg",
            caption="Hello world",
        )
    ]
    respx.get(API_URL).respond(200, json=_payload("Just do it.", posts))
    respx.get("https://scontent.cdninstagram.com/post-1.jpg").respond(
        200, content=sample_png_bytes
    )

    client = HttpClient()
    try:
        bundle = InstagramScraper(client, settings).scrape(
            BrandConfig(url=PROFILE_URL), PROFILE_URL
        )
    finally:
        client.close()

    assert bundle.source == "instagram"
    assert bundle.candidates_checked == 1
    assert len(bundle.images) == 1
    image = bundle.images[0]
    assert str(image.url) == "https://scontent.cdninstagram.com/post-1.jpg"
    assert image.width == 512 and image.height == 512

    kinds = {(t.kind, t.text) for t in bundle.texts}
    assert ("tagline", "Just do it.") in kinds
    assert ("editorial", "Hello world") in kinds


@respx.mock
def test_instagram_scraper_caps_posts(settings, sample_png_bytes):
    settings = dict(settings)
    settings["instagram_max_posts"] = 2

    posts = [
        _post_edge(
            shortcode=f"sc{i}",
            display_url=f"https://scontent.cdninstagram.com/p{i}.jpg",
            caption=f"caption {i}",
        )
        for i in range(5)
    ]
    respx.get(API_URL).respond(200, json=_payload(posts=posts))
    for i in range(5):
        respx.get(f"https://scontent.cdninstagram.com/p{i}.jpg").respond(
            200, content=sample_png_bytes
        )

    client = HttpClient()
    try:
        bundle = InstagramScraper(client, settings).scrape(
            BrandConfig(url=PROFILE_URL), PROFILE_URL
        )
    finally:
        client.close()

    assert bundle.candidates_checked == 2
    assert len(bundle.images) == 2
    captions = [t.text for t in bundle.texts if t.kind == "editorial"]
    assert sorted(captions) == ["caption 0", "caption 1"]


@respx.mock
def test_instagram_scraper_skips_below_resolution(
    settings, sample_small_png_bytes
):
    settings = dict(settings)

    posts = [
        _post_edge(
            shortcode="abc",
            display_url="https://scontent.cdninstagram.com/tiny.jpg",
            caption="too small",
        )
    ]
    respx.get(API_URL).respond(200, json=_payload(posts=posts))
    respx.get("https://scontent.cdninstagram.com/tiny.jpg").respond(
        200, content=sample_small_png_bytes
    )

    client = HttpClient()
    try:
        bundle = InstagramScraper(client, settings).scrape(
            BrandConfig(url=PROFILE_URL), PROFILE_URL
        )
    finally:
        client.close()

    assert bundle.candidates_checked == 1
    assert bundle.images == []


@respx.mock
def test_instagram_scraper_extracts_carousel_children(
    settings, sample_png_bytes
):
    settings = dict(settings)

    children = [
        {
            "node": {
                "display_url": "https://scontent.cdninstagram.com/c1.jpg",
                "is_video": False,
            }
        },
        {
            "node": {
                "display_url": "https://scontent.cdninstagram.com/c2.jpg",
                "is_video": True,
            }
        },
    ]
    posts = [
        _post_edge(
            shortcode="abc",
            display_url="https://scontent.cdninstagram.com/cover.jpg",
            caption="carousel",
            typename="GraphSidecar",
            children=children,
        )
    ]
    respx.get(API_URL).respond(200, json=_payload(posts=posts))
    respx.get("https://scontent.cdninstagram.com/c1.jpg").respond(
        200, content=sample_png_bytes
    )

    client = HttpClient()
    try:
        bundle = InstagramScraper(client, settings).scrape(
            BrandConfig(url=PROFILE_URL), PROFILE_URL
        )
    finally:
        client.close()

    image_urls = [str(img.url) for img in bundle.images]
    assert image_urls == ["https://scontent.cdninstagram.com/c1.jpg"]


@respx.mock
def test_instagram_scraper_returns_empty_bundle_on_404(settings):
    respx.get(API_URL).respond(404)

    client = HttpClient()
    try:
        bundle = InstagramScraper(client, settings).scrape(
            BrandConfig(url=PROFILE_URL), PROFILE_URL
        )
    finally:
        client.close()

    assert bundle.images == []
    assert bundle.texts == []
    assert bundle.pages_crawled == 0


@respx.mock
def test_instagram_scraper_skips_promotional_post(settings, sample_png_bytes):
    settings = dict(settings)

    posts = [
        _post_edge(
            shortcode="promo",
            display_url="https://scontent.cdninstagram.com/promo.jpg",
            caption="The second item is 50% off on designs picked together.",
        ),
        _post_edge(
            shortcode="regular",
            display_url="https://scontent.cdninstagram.com/regular.jpg",
            caption="New tee designs are live.",
        ),
    ]
    respx.get(API_URL).respond(200, json=_payload(posts=posts))
    respx.get("https://scontent.cdninstagram.com/regular.jpg").respond(
        200, content=sample_png_bytes
    )

    client = HttpClient()
    try:
        bundle = InstagramScraper(client, settings).scrape(
            BrandConfig(url=PROFILE_URL), PROFILE_URL
        )
    finally:
        client.close()

    assert bundle.candidates_checked == 1
    assert len(bundle.images) == 1
    assert str(bundle.images[0].url) == "https://scontent.cdninstagram.com/regular.jpg"
    editorial = [t.text for t in bundle.texts if t.kind == "editorial"]
    assert editorial == ["New tee designs are live."]


@pytest.mark.parametrize(
    "text",
    [
        "The second item is 50% off on designs picked together.",
        "Yan yana gelen tasarımlarda ikinci ürün %50 indirimli.",
        "Flash sale this weekend only",
        "Extra 20% off with code SAVE",
        "%30 indirim",
        "Clearance event starts now",
        "Promo code inside",
        "Register now for our Milan store opening.",
        "RSVP for the invite to our flagship store launch.",
        "You're invited to register for the new boutique opening.",
    ],
)
def test_is_promotional_social_post_detects_sale_copy(text):
    assert is_promotional_social_post(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        None,
        "New men's T-shirt designs are now on our website.",
        "Lines turned into notes of my story.",
        "Just dropped our spring collection.",
        "Visit our store for the new collection.",
        "Register for updates on new arrivals.",
        "Private invitation to the runway show.",
    ],
)
def test_is_promotional_social_post_allows_regular_copy(text):
    assert is_promotional_social_post(text) is False


@respx.mock
def test_instagram_scraper_skips_event_announcement_post(settings, sample_png_bytes):
    settings = dict(settings)

    posts = [
        _post_edge(
            shortcode="opening",
            display_url="https://scontent.cdninstagram.com/opening.jpg",
            caption="Register now for our Paris store opening — save the date.",
        ),
        _post_edge(
            shortcode="lookbook",
            display_url="https://scontent.cdninstagram.com/lookbook.jpg",
            caption="Spring tailoring, captured in Milan.",
        ),
    ]
    respx.get(API_URL).respond(200, json=_payload(posts=posts))
    respx.get("https://scontent.cdninstagram.com/lookbook.jpg").respond(
        200, content=sample_png_bytes
    )

    client = HttpClient()
    try:
        bundle = InstagramScraper(client, settings).scrape(
            BrandConfig(url=PROFILE_URL), PROFILE_URL
        )
    finally:
        client.close()

    assert bundle.candidates_checked == 1
    assert len(bundle.images) == 1
    assert str(bundle.images[0].url) == "https://scontent.cdninstagram.com/lookbook.jpg"
    editorial = [t.text for t in bundle.texts if t.kind == "editorial"]
    assert editorial == ["Spring tailoring, captured in Milan."]
