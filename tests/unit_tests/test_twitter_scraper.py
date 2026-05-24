import json

import respx

from src.models.brand_configs import BrandConfig
from src.services.http_service import HttpClient
from src.crawlers.twitter_scraper import TwitterScraper


PROFILE_URL = "https://x.com/nike"
TIMELINE_URL = (
    "https://syndication.twitter.com/srv/timeline-profile/screen-name/nike"
)


def _next_data(bio: str, tweets):
    payload = {
        "props": {
            "pageProps": {
                "contextProvider": {
                    "user": {
                        "screen_name": "nike",
                        "description": bio,
                    }
                },
                "timeline": {"entries": []},
            }
        }
    }
    entries = []
    for tweet in tweets:
        entries.append({"type": "tweet", "content": {"tweet": tweet}})
    payload["props"]["pageProps"]["timeline"]["entries"] = entries
    return payload


def _html_with_next_data(data) -> str:
    return (
        "<!doctype html><html><body>"
        '<script id="__NEXT_DATA__" type="application/json">'
        f"{json.dumps(data)}"
        "</script>"
        "</body></html>"
    )


@respx.mock
def test_twitter_scraper_extracts_bio_tweets_and_photos(
    settings, sample_png_bytes
):
    settings = dict(settings)
    settings["twitter_max_tweets"] = 20

    tweets = [
        {
            "id_str": "1",
            "full_text": "Hello world",
            "entities": {
                "media": [
                    {
                        "type": "photo",
                        "media_url_https": "https://pbs.twimg.com/media/abc.jpg",
                    }
                ]
            },
        },
        {
            "id_str": "2",
            "full_text": "Plain tweet",
            "entities": {},
        },
    ]
    page = _html_with_next_data(_next_data("Just do it.", tweets))
    respx.get(TIMELINE_URL).respond(200, text=page)
    respx.get("https://pbs.twimg.com/media/abc.jpg?name=large").respond(
        200, content=sample_png_bytes
    )

    client = HttpClient()
    try:
        bundle = TwitterScraper(client, settings).scrape(
            BrandConfig(url=PROFILE_URL), PROFILE_URL
        )
    finally:
        client.close()

    assert bundle.source == "twitter"
    text_pairs = {(t.kind, t.text) for t in bundle.texts}
    assert ("tagline", "Just do it.") in text_pairs
    assert ("editorial", "Hello world") in text_pairs
    assert ("editorial", "Plain tweet") in text_pairs

    assert len(bundle.images) == 1
    image_url = str(bundle.images[0].url)
    assert image_url == "https://pbs.twimg.com/media/abc.jpg?name=large"


@respx.mock
def test_twitter_scraper_caps_tweets(settings, sample_png_bytes):
    settings = dict(settings)
    settings["twitter_max_tweets"] = 2

    tweets = [
        {
            "id_str": str(i),
            "full_text": f"tweet {i}",
            "entities": {},
        }
        for i in range(5)
    ]
    page = _html_with_next_data(_next_data("bio", tweets))
    respx.get(TIMELINE_URL).respond(200, text=page)

    client = HttpClient()
    try:
        bundle = TwitterScraper(client, settings).scrape(
            BrandConfig(url=PROFILE_URL), PROFILE_URL
        )
    finally:
        client.close()

    editorial = [t.text for t in bundle.texts if t.kind == "editorial"]
    assert editorial == ["tweet 0", "tweet 1"]


@respx.mock
def test_twitter_scraper_skips_video_media(settings, sample_png_bytes):
    settings = dict(settings)

    tweets = [
        {
            "id_str": "1",
            "full_text": "video tweet",
            "extended_entities": {
                "media": [
                    {
                        "type": "video",
                        "media_url_https": "https://pbs.twimg.com/ext_tw_video_thumb/v.jpg",
                    }
                ]
            },
            "entities": {},
        }
    ]
    page = _html_with_next_data(_next_data("bio", tweets))
    respx.get(TIMELINE_URL).respond(200, text=page)

    client = HttpClient()
    try:
        bundle = TwitterScraper(client, settings).scrape(
            BrandConfig(url=PROFILE_URL), PROFILE_URL
        )
    finally:
        client.close()

    assert bundle.images == []
    assert any(t.text == "video tweet" for t in bundle.texts)


@respx.mock
def test_twitter_scraper_returns_empty_bundle_when_next_data_missing(settings):
    page = "<!doctype html><html><body><h1>nothing here</h1></body></html>"
    respx.get(TIMELINE_URL).respond(200, text=page)

    client = HttpClient()
    try:
        bundle = TwitterScraper(client, settings).scrape(
            BrandConfig(url=PROFILE_URL), PROFILE_URL
        )
    finally:
        client.close()

    assert bundle.images == []
    assert bundle.texts == []
    assert bundle.pages_crawled == 0


@respx.mock
def test_twitter_scraper_skips_promotional_tweet(settings, sample_png_bytes):
    settings = dict(settings)

    tweets = [
        {
            "id_str": "1",
            "full_text": "Everything is 50% off today only",
            "entities": {
                "media": [
                    {
                        "type": "photo",
                        "media_url_https": "https://pbs.twimg.com/media/promo.jpg",
                    }
                ]
            },
        },
        {
            "id_str": "2",
            "full_text": "New collection is live",
            "entities": {
                "media": [
                    {
                        "type": "photo",
                        "media_url_https": "https://pbs.twimg.com/media/regular.jpg",
                    }
                ]
            },
        },
    ]
    page = _html_with_next_data(_next_data("bio", tweets))
    respx.get(TIMELINE_URL).respond(200, text=page)
    respx.get("https://pbs.twimg.com/media/regular.jpg?name=large").respond(
        200, content=sample_png_bytes
    )

    client = HttpClient()
    try:
        bundle = TwitterScraper(client, settings).scrape(
            BrandConfig(url=PROFILE_URL), PROFILE_URL
        )
    finally:
        client.close()

    assert bundle.candidates_checked == 1
    assert len(bundle.images) == 1
    assert str(bundle.images[0].url) == "https://pbs.twimg.com/media/regular.jpg?name=large"
    editorial = [t.text for t in bundle.texts if t.kind == "editorial"]
    assert editorial == ["New collection is live"]



@respx.mock
def test_twitter_scraper_returns_empty_bundle_on_500(settings):
    respx.get(TIMELINE_URL).respond(500)

    client = HttpClient()
    try:
        bundle = TwitterScraper(client, settings).scrape(
            BrandConfig(url=PROFILE_URL), PROFILE_URL
        )
    finally:
        client.close()

    assert bundle.images == []
    assert bundle.texts == []
