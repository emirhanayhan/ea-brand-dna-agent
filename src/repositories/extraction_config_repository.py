from datetime import datetime, timezone
from typing import Optional

from pymongo.collection import Collection

from src.models.metadata_extraction_config import MetadataExtractionConfig


class MongoExtractionConfigRepository:
    def __init__(
        self,
        db
    ):
        self._db = db
        self._collection: Collection = db["extraction_configs"]

    def get_by_domain(self, domain: str) -> Optional[MetadataExtractionConfig]:
        document = self._collection.find_one(
            {"domain": domain}, {"_id": 0, "config": 1, "updated_at": 1}
        )
        if not document or "config" not in document:
            return None
        config = MetadataExtractionConfig.model_validate(document["config"])
        updates: dict = {}
        if config.domain != domain:
            updates["domain"] = domain
        if document.get("updated_at") is not None:
            updates["updated_at"] = document["updated_at"]
        if updates:
            config = config.model_copy(update=updates)
        return config.normalized()

    def update(self, config: MetadataExtractionConfig) -> None:
        now = datetime.now(timezone.utc)
        # `updated_at` is stored top-level on the doc; strip it from the inner
        # config payload so it never goes stale relative to the Mongo column.
        config_payload = config.model_dump(mode="json", exclude={"updated_at"})
        payload = {
            "domain": config.domain,
            "config": config_payload,
            "updated_at": now,
        }
        self._collection.update_one(
            {"domain": config.domain},
            {"$set": payload},
            upsert=True,
        )
