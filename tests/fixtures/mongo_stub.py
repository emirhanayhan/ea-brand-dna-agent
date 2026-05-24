"""In-process pymongo replacement for local development / one-off runs.

Use this when you want to run main.py without a real MongoDB instance:
all writes are no-ops, every read returns None, healthcheck pings succeed.
The agent therefore re-infers per-domain extraction configs on every run
(no caching), which is fine for one-off PDF generation jobs.

Usage:
    import pymongo
    from tests.fixtures.mongo_stub import StubMongoClient
    pymongo.MongoClient = StubMongoClient  # patch BEFORE importing main

The stub deliberately implements only the surface area the agent uses
(find_one, update_one, admin.command('ping'), close). It is NOT a general
pymongo replacement — for richer scenarios use the `mongomock` package.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class _StubCollection:
    def find_one(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return None

    def update_one(self, *args: Any, **kwargs: Any) -> None:
        return None

    def delete_many(self, *args: Any, **kwargs: Any) -> None:
        return None


class _StubAdmin:
    @staticmethod
    def command(*args: Any, **kwargs: Any) -> Dict[str, float]:
        return {"ok": 1.0}


class _StubDatabase:
    def __init__(self, client: "StubMongoClient") -> None:
        self.client = client

    def __getitem__(self, _name: str) -> _StubCollection:
        return _StubCollection()


class StubMongoClient:
    """Drop-in for pymongo.MongoClient. Ignores connection args."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.admin = _StubAdmin()

    def __getitem__(self, _name: str) -> _StubDatabase:
        return _StubDatabase(self)

    def close(self) -> None:
        return None
