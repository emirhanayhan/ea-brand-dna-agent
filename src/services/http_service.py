import logging
from typing import Any, Dict, Optional

import httpx

from src.utils.bot_protection import BotProtectionDetection, detect_bot_protection_from_response

logger = logging.getLogger("HttpClient")

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


class HttpClient:
    def __init__(self, timeout: float = 15.0, max_connections: int = 100):
        """
        Initializes a new instance of the HttpClient with its own connection pool.
        """
        limits = httpx.Limits(max_keepalive_connections=50, max_connections=max_connections)

        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            limits=limits,
            follow_redirects=True,
            headers=_DEFAULT_HEADERS,
        )
        self.last_bot_protection: Optional[BotProtectionDetection] = None
        logger.info("Initialized HTTP client instance and connection pool.")

    def get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[httpx.Response]:
        """Instance-level GET request."""
        self.last_bot_protection = None
        try:
            response = self.client.get(url, params=params)
        except httpx.RequestError as e:
            logger.warning(f"GET {url} failed: {e}")
            return None

        detection = detect_bot_protection_from_response(response)
        if detection:
            self.last_bot_protection = detection
            logger.error("GET %s blocked by bot protection: %s", url, detection)
            return None

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.warning(f"GET {url} failed: {e}")
            return None

        return response

    def get_stream(self, url: str, max_bytes: int = 512_000) -> Optional[bytes]:
        """Stream a GET response and return up to max_bytes."""
        try:
            with self.client.stream("GET", url) as response:
                response.raise_for_status()
                chunks = []
                remaining = max_bytes
                for chunk in response.iter_bytes():
                    if len(chunk) >= remaining:
                        chunks.append(chunk[:remaining])
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                return b"".join(chunks)
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.warning(f"GET stream {url} failed: {e}")
            return None

    def post(self, url: str, json_data: Optional[Dict[str, Any]] = None) -> Optional[httpx.Response]:
        """Instance-level POST request."""
        try:
            response = self.client.post(url, json=json_data)
            response.raise_for_status()
            return response
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.warning(f"POST {url} failed: {e}")
            return None

    def put(self, url: str, json_data: Optional[Dict[str, Any]] = None) -> Optional[httpx.Response]:
        """Instance-level PUT request."""
        try:
            response = self.client.put(url, json=json_data)
            response.raise_for_status()
            return response
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.warning(f"PUT {url} failed: {e}")
            return None

    def close(self) -> None:
        """Explicitly close the connection pool."""
        logger.info("Closing HTTP client connection pool...")
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()