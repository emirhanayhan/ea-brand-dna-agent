import os
from datetime import datetime, timezone
from io import BytesIO

import pytest
from PIL import Image
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from configs.test import test_config
from src.models.brand_configs import BrandConfig
from src.models.metadata_extraction_config import MetadataExtractionConfig
from src.repositories.extraction_config_repository import MongoExtractionConfigRepository
from src.services.http_service import HttpClient
from src.services.llm_service import LlmService
from src.services.metadata_config_service import MetadataConfigService
from tests.fixtures.mock_extraction_config import build_mock_extraction_config
from tests.integration_tests.integration_helpers import BRAND_CASES


class _StubLlmService:
    def infer_extraction_config(self, samples, domain):
        raise ValueError("GEMINI_API_KEY is required to infer extraction config")


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run live integration tests that call real network endpoints",
    )
    parser.addoption(
        "--run-llm-integration",
        action="store_true",
        default=False,
        help="Run live LLM integration tests (requires GEMINI_API_KEY)",
    )
    parser.addoption(
        "--run-benchmark",
        action="store_true",
        default=False,
        help="Run Brand DNA LLM model benchmark tests (requires GEMINI_API_KEY)",
    )
    parser.addoption(
        "--benchmark-brand",
        action="store",
        default=None,
        help="Run benchmark for a single BRAND_CASES name (e.g. nike)",
    )
    parser.addoption(
        "--benchmark-live-crawl",
        action="store_true",
        default=False,
        help="Regenerate benchmark bundle fixtures via live crawl before running",
    )


def pytest_collection_modifyitems(config, items):
    run_integration = config.getoption("--run-integration")
    run_llm_integration = config.getoption("--run-llm-integration")
    run_benchmark = config.getoption("--run-benchmark")

    if not run_benchmark:
        skip_benchmark = pytest.mark.skip(
            reason="pass --run-benchmark to run Brand DNA LLM model benchmarks"
        )
        for item in items:
            if "benchmark" in item.keywords:
                item.add_marker(skip_benchmark)

    if not run_integration:
        skip_integration = pytest.mark.skip(
            reason="pass --run-integration to run live integration tests"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    if not run_llm_integration:
        skip_llm = pytest.mark.skip(
            reason="pass --run-llm-integration to run live LLM integration tests"
        )
        for item in items:
            if "llm_integration" in item.keywords:
                item.add_marker(skip_llm)


@pytest.fixture(scope="session")
def mongo_client():
    client = MongoClient(
        test_config["mongo_connection_string"],
        serverSelectionTimeoutMS=2000,
    )
    try:
        client.admin.command("ping")
    except PyMongoError as exc:
        client.close()
        pytest.skip(f"MongoDB is not available for tests: {exc}")
    yield client
    client.close()


@pytest.fixture(scope="session")
def db(mongo_client):
    return mongo_client[test_config["db_name"]]


@pytest.fixture(scope="session", autouse=True)
def seed_brand_extraction_configs(request, db):
    if not request.config.getoption("--run-integration") and not (
        request.config.getoption("--run-benchmark")
        and request.config.getoption("--benchmark-live-crawl")
    ):
        return

    repository = MongoExtractionConfigRepository(db=db)
    for brand_case in BRAND_CASES:
        if brand_case.expect_bot_protection:
            continue
        domain = MetadataConfigService.normalize_domain(brand_case.brand_url)
        if repository.get_by_domain(domain) is not None:
            continue
        repository.update(MetadataExtractionConfig.ecommerce_defaults(domain))


@pytest.fixture
def test_settings(db):
    settings = dict(test_config)
    settings["db"] = db
    return settings


@pytest.fixture
def clean_extraction_configs(db):
    collection = db["extraction_configs"]
    collection.delete_many({})
    yield collection
    collection.delete_many({})


@pytest.fixture
def integration_http_client():
    client = HttpClient()
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def settings():
    return dict(test_config)


@pytest.fixture
def integration_settings(test_settings):
    return dict(test_settings)


@pytest.fixture
def metadata_config_service(integration_http_client, integration_settings):
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        llm_settings = dict(integration_settings)
        llm_settings["gemini_api_key"] = api_key
        llm_settings["llm_model"] = os.getenv("LLM_MODEL", "gemini-3.5-flash")
        llm_service = LlmService(llm_settings)
    else:
        llm_service = _StubLlmService()

    repository = MongoExtractionConfigRepository(db=integration_settings["db"])
    return MetadataConfigService(
        integration_settings,
        integration_http_client,
        llm_service,
        repository,
    )


@pytest.fixture(scope="session")
def benchmark_run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@pytest.fixture
def gemini_settings():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY is required for LLM integration tests")
    return {
        "gemini_api_key": api_key,
        "llm_model": os.getenv("LLM_MODEL", "gemini-3.5-flash"),
    }


@pytest.fixture
def mock_extraction_config():
    return build_mock_extraction_config()


@pytest.fixture
def brand_config():
    return BrandConfig(url="https://example.com")


def _make_image_bytes(width: int, height: int, fmt: str = "PNG") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color="red").save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def sample_png_bytes():
    return _make_image_bytes(512, 512, "PNG")


@pytest.fixture
def sample_small_png_bytes():
    return _make_image_bytes(256, 256, "PNG")
