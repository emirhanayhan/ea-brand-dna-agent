import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

from pymongo import MongoClient

from configs.local import local_config
from configs.stage import stage_config
from configs.prod import prod_config
from src.models.brand_configs import BrandConfig
from src.models.brand_content import BrandContentBundle
from src.repositories.extraction_config_repository import MongoExtractionConfigRepository
from src.services.brand_dna_service import BrandDnaService
from src.utils.content_bundle_merger import merge_content_bundles
from src.utils.healthchecks import db_healthcheck, healthcheck_website_url
from src.services.http_service import HttpClient
from src.crawlers.instagram_scraper import InstagramScraper
from src.services.llm_service import LlmService
from src.services.metadata_config_service import MetadataConfigService
from src.services.social_network_verifier import check_social_media_exists
from src.crawlers.twitter_scraper import TwitterScraper
from src.crawlers.website_crawler import WebsiteCrawler

logger = logging.getLogger("BrandAgent")

CONFIG_LOOKUP = {
    "prod": prod_config,
    "stage": stage_config,
    "local": local_config
}

parser = argparse.ArgumentParser(description="Autonomously produce a Brand DNA dossier.")
parser.add_argument(
    "--config",
    type=str,
    choices=["local", "stage", "prod"],
    default="local",
    help="app config"
)

parser.add_argument("--name", type=str, help="Brand display name.", required=True)
parser.add_argument("--url", type=str, help="Explicitly set the primary website URL.", required=True)
parser.add_argument("--social_handle", type=str)
parser.add_argument(
    "--refresh-extraction-config",
    action="store_true",
    help="Re-infer and overwrite cached metadata extraction config for this domain",
)

args = parser.parse_args()
settings = CONFIG_LOOKUP[args.config]
logger.info("initialized application settings env: {}".format(args.config))


def main():
    db = MongoClient(
        settings["mongo_connection_string"],
        retryWrites=True,
    )[settings["db_name"]]
    http_client = HttpClient()
    executor = ThreadPoolExecutor(
        max_workers=settings["worker_count"],
        thread_name_prefix="brand-agent",
    )
    try:
        brand_official_site = args.url
        brands_social_tag = args.social_handle

        # init repositories
        extraction_config_repository = MongoExtractionConfigRepository(db=db)

        # init services
        llm_service = LlmService(settings)
        metadata_config_service = MetadataConfigService(
            settings,
            http_client,
            llm_service,
            extraction_config_repository,
        )
        brand_dna_service = BrandDnaService(
            settings, http_client, llm_service, executor=executor
        )

        # health checks and validations
        db_healthcheck(db)
        healthcheck_website_url(http_client, brand_official_site)
        brand_socials: dict[str, str] | None = None
        if brands_social_tag:
            brand_socials = check_social_media_exists(
                http_client, brands_social_tag
            ) or None

        brand_config = BrandConfig(
            name=args.name,
            url=brand_official_site,
            social_handles=brand_socials,
        )
        logger.info("--- initialized brand config---")
        logger.info(brand_config.model_dump_json(indent=2))

        crawler = WebsiteCrawler(
            http_client,
            settings,
            metadata_config_service=metadata_config_service,
            refresh_extraction_config=args.refresh_extraction_config,
            executor=executor,
        )
        website_bundle = crawler.crawl(brand_config)
        if website_bundle.stopped_due_to_bot_protection:
            logger.error(
                "website_crawler stopped due to bot protection at %s: %s/%s",
                website_bundle.bot_protection.url,
                website_bundle.bot_protection.provider,
                website_bundle.bot_protection.reason,
            )
        logger.info("--- website bundle ---")
        logger.info(website_bundle.model_dump_json(indent=2))

        bundles: list[BrandContentBundle] = [website_bundle]

        instagram_url = (brand_config.social_handles or {}).get("instagram")
        if instagram_url:
            try:
                ig_bundle = InstagramScraper(http_client, settings).scrape(
                    brand_config, instagram_url
                )
                logger.info(
                    "--- instagram bundle (%s images, %s texts) ---",
                    len(ig_bundle.images),
                    len(ig_bundle.texts),
                )
                bundles.append(ig_bundle)
            except Exception as exc:
                logger.warning("instagram scraper failed: %s", exc)

        twitter_url = (brand_config.social_handles or {}).get("twitter")
        if twitter_url:
            try:
                tw_bundle = TwitterScraper(http_client, settings).scrape(
                    brand_config, twitter_url
                )
                logger.info(
                    "--- twitter bundle (%s images, %s texts) ---",
                    len(tw_bundle.images),
                    len(tw_bundle.texts),
                )
                bundles.append(tw_bundle)
            except Exception as exc:
                logger.warning("twitter scraper failed: %s", exc)

        combined_bundle = merge_content_bundles(bundles)
        logger.info(
            "--- combined bundle: %s images, %s texts ---",
            len(combined_bundle.images),
            len(combined_bundle.texts),
        )

        if not combined_bundle.images:
            logger.warning(
                "No images collected — skipping brand DNA synthesis."
            )
            return

        report, pdf_path = brand_dna_service.run(brand_config, combined_bundle)
        logger.info("--- brand DNA report ---")
        logger.info(
            "PDF: %s  ·  images analyzed: %s  ·  text snippets: %s",
            pdf_path,
            report.image_count,
            report.text_snippet_count,
        )

    except Exception as e:
        logger.error("Error while running agent: %s", e)
        exit(1)
    finally:
        executor.shutdown(wait=True)
        http_client.close()
        db.client.close()

if __name__ == "__main__":
    main()
