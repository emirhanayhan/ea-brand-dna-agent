import pytest

from src.models.metadata_extraction_config import MetadataExtractionConfig, ProductCardRule
from src.repositories.extraction_config_repository import MongoExtractionConfigRepository


@pytest.fixture
def sample_config():
    return MetadataExtractionConfig.ecommerce_defaults("nike.com")


@pytest.fixture
def mongo_repository(db, clean_extraction_configs):
    repository = MongoExtractionConfigRepository(db=db)
    yield repository


def test_mongo_repository_round_trip(mongo_repository, sample_config):
    assert mongo_repository.get_by_domain("nike.com") is None

    mongo_repository.update(sample_config)
    loaded = mongo_repository.get_by_domain("nike.com")

    assert loaded is not None
    assert loaded.domain == "nike.com"
    assert len(loaded.json_sources) == len(sample_config.normalized().json_sources)


def test_mongo_repository_round_trip_with_product_card(mongo_repository):
    config = MetadataExtractionConfig.ecommerce_defaults("gucci.com")
    config = config.model_copy(
        update={
            "product_card": ProductCardRule(
                card_selector='//*[@class="product-tile"]',
                link_selector='.//a[@class="product-link"]',
            )
        }
    ).normalized()

    assert mongo_repository.get_by_domain("gucci.com") is None
    mongo_repository.update(config)
    loaded = mongo_repository.get_by_domain("gucci.com")

    assert loaded is not None
    assert loaded.domain == "gucci.com"
    assert loaded.product_card is not None
