from pymongo.database import Database

from src.services.http_service import HttpClient


def healthcheck_website_url(http_service: HttpClient, url: str) -> str:
    """Validate that the website URL responds with HTTP 200."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    resp = http_service.get(url)
    if not resp:
        if http_service.last_bot_protection:
            raise Exception(
                f"Website healthcheck failed due to bot protection at {url}: "
                f"{http_service.last_bot_protection}"
            )
        raise Exception(f"Failed to healthcheck website {url}")

    if resp.status_code != 200:
        raise Exception(
            f"Website healthcheck failed for {url}: expected HTTP 200, got {resp.status_code}"
        )

    return str(resp.url)


def db_healthcheck(db: Database) -> None:
    """Validates MongoDB connectivity by issuing an admin ping."""
    try:
        db.client.admin.command("ping")
    except Exception as exc:
        raise Exception("Failed to healthcheck database") from exc
