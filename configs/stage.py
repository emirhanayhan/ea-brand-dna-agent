import os

stage_config = {
    "lowest_acceptable_resolution": os.getenv("LOWEST_ACCEPTABLE_RESOLUTION", "512x512"),
    "lowest_processeable_product_count": int(
        os.getenv("LOWEST_PROCESSEABLE_PRODUCT_COUNT", "100")
    ),
    "max_crawl_pages": int(os.getenv("MAX_CRAWL_PAGES", "100")),
    "max_crawl_depth": int(os.getenv("MAX_CRAWL_DEPTH", "5")),
    "worker_count": int(os.getenv("WORKER_COUNT", str(os.cpu_count() or 8))),
    "editorial_probe_enabled": os.getenv("EDITORIAL_PROBE_ENABLED", "true").lower()
    in ("1", "true", "yes"),
    "instagram_max_posts": int(os.getenv("INSTAGRAM_MAX_POSTS", "12")),
    "twitter_max_tweets": int(os.getenv("TWITTER_MAX_TWEETS", "20")),
    "brand_dna_max_images": int(os.getenv("BRAND_DNA_MAX_IMAGES", "100")),
    "brand_dna_cluster_target": int(os.getenv("BRAND_DNA_CLUSTER_TARGET", "5")),
    "brand_dna_cluster_discovery_images": int(
        os.getenv("BRAND_DNA_CLUSTER_DISCOVERY_IMAGES", "20")
    ),
    "brand_dna_cluster_assignment_batch_size": int(
        os.getenv("BRAND_DNA_CLUSTER_ASSIGNMENT_BATCH_SIZE", "12")
    ),
    "brand_dna_palette_size": int(os.getenv("BRAND_DNA_PALETTE_SIZE", "8")),
    "brand_dna_min_images_per_cluster": int(
        os.getenv("BRAND_DNA_MIN_IMAGES_PER_CLUSTER", "2")
    ),
    "brand_dna_max_images_per_cluster": int(
        os.getenv("BRAND_DNA_MAX_IMAGES_PER_CLUSTER", "3")
    ),
    "brand_dna_min_cluster_count": int(
        os.getenv("BRAND_DNA_MIN_CLUSTER_COUNT", "3")
    ),
    "brand_dna_fashion_filter_strategy": os.getenv(
        "BRAND_DNA_FASHION_FILTER_STRATEGY", "clip"
    ),
    "brand_dna_fashion_filter_model": os.getenv(
        "BRAND_DNA_FASHION_FILTER_MODEL", "ViT-B-32"
    ),
    "brand_dna_fashion_filter_pretrained": os.getenv(
        "BRAND_DNA_FASHION_FILTER_PRETRAINED", "openai"
    ),
    "brand_dna_fashion_filter_threshold": float(
        os.getenv("BRAND_DNA_FASHION_FILTER_THRESHOLD", "0.55")
    ),
    "brand_dna_output_dir": os.getenv("BRAND_DNA_OUTPUT_DIR", "outputs"),
    "mongo_connection_string": os.getenv("MONGO_CONNECTION_STRING"),
    "db_name": os.getenv("DB_NAME", "brand_intelligence"),
    "gemini_api_key": os.getenv("GEMINI_API_KEY"),
    "llm_model": os.getenv("LLM_MODEL", "gemini-3.5-flash"),
}
