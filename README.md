# EA Brand Intelligence Agent

Autonomous pipeline that crawls a brand's website and social profiles, merges the collected imagery and copy into a unified content bundle, and synthesizes a **Brand DNA** dossier (structured JSON + PDF) using Gemini vision and text models.

```
Brand URL + optional social handle
        │
        ▼
┌───────────────────┐     ┌────────────────────┐     ┌─────────────────────┐
│ WebsiteCrawler    │     │ InstagramScraper   │     │ TwitterScraper      │
│ (product imagery, │     │ (bio + posts)      │     │ (bio + tweets)      │
│  editorial copy)  │     └─────────┬──────────┘     └──────────┬──────────┘
└─────────┬─────────┘               │                           │
          │                         └───────────┬───────────────┘
          │                                     ▼
          │                          merge_content_bundles()
          │                                     │
          └─────────────────────────────────────┘
                                                ▼
                                     BrandDnaService.run()
                                                │
                                                ▼
                              outputs/<slug>_brand_dna.{json,pdf}
```

---

## Environments

The app is configured through environment-specific Python dicts in `configs/`. Pass the desired profile with `--config` when running `main.py`.

| Profile | Config file | Typical use |
|---------|-------------|-------------|
| `local` | `configs/local.py` | Local development (default) |
| `stage` | `configs/stage.py` | Staging / pre-production |
| `prod` | `configs/prod.py` | Production |

Tests use a separate profile in `configs/test.py` (lower product thresholds, dedicated test database).

### Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes (for Brand DNA) | — | Google Gemini API key for LLM vision + synthesis |
| `MONGO_CONNECTION_STRING` | Yes | — | MongoDB connection URI |
| `DB_NAME` | No | `brand_intelligence` | MongoDB database name |
| `LLM_MODEL` | No | `gemini-3.5-flash` | Gemini model identifier (see [Benchmark results](#benchmark-results)) |
| `WORKER_COUNT` | No | CPU count or `8` | Thread pool size for crawlers and image downloads |

Docker Compose additionally supports `MONGO_USER` and `MONGO_PASSWORD` (defaults: `ea_brand` / `ea_brand_dev`).

### Tunable settings (all environments)

These keys live in each config dict. They are grouped below by what they control.

#### Crawl & scrape

| Key | Default | Purpose |
|-----|---------|---------|
| `lowest_acceptable_resolution` | `512x512` | Minimum image dimensions to keep |
| `lowest_processeable_product_count` | `100` (3 in test) | Stop crawling once this many products are found |
| `max_crawl_pages` | `100` | Max pages the website crawler visits |
| `max_crawl_depth` | `5` | Max link depth from seed URL |
| `worker_count` | CPU count or `8` | Shared thread pool for crawlers and Brand DNA image downloads |
| `editorial_probe_enabled` | `true` | Probe editorial/about pages when no editorial copy was found during crawl |
| `instagram_max_posts` | `12` | Recent Instagram posts to scrape |
| `twitter_max_tweets` | `20` | Recent tweets to scrape |

Crawl limits indirectly affect LLM cost by capping how many images can reach Brand DNA.

#### LLM cost & vision

These settings control **Gemini vision** usage (image tokens and API call count). The text synthesis pass (`synthesize_brand_dna`) has no separate cap — it runs once per report.

| Key | Default | Purpose |
|-----|---------|---------|
| `LLM_MODEL` (env) | `gemini-3.5-flash` | Model used for vision + synthesis (see [Benchmark results](#benchmark-results)) |
| `brand_dna_max_images` | `100` | **Main cost lever** — max images downloaded and sent through the vision pipeline (matches `lowest_processeable_product_count`) |
| `brand_dna_cluster_discovery_images` | `20` | Images shown to Gemini in Phase 1 theme discovery (even when `brand_dna_max_images` is higher) |
| `brand_dna_cluster_assignment_batch_size` | `12` | Images per Phase 2 assignment batch — larger batches mean fewer API calls |
| `brand_dna_cluster_target` | `5` | Target number of aesthetic clusters in the vision output |

Rough call pattern when image count exceeds cluster target: **1 discovery call** + **ceil(N / assignment_batch_size) assignment calls** + **1 synthesis call**, where N ≤ `brand_dna_max_images` (after pHash dedupe).

#### Brand DNA output & local processing

These `brand_dna_*` keys are **not** LLM cost controls.

| Key | Default | Purpose |
|-----|---------|---------|
| `brand_dna_palette_size` | `8` | Dominant colors extracted locally (`extract_palette` — no Gemini call) |
| `brand_dna_output_dir` | `outputs` | Directory for JSON + PDF output |
| `brand_dna_download_workers` | `8` | Parallel HTTP downloads before the LLM (ignored when `main.py` injects the shared executor; then `worker_count` applies) |
| `brand_dna_min_images_per_cluster` | `2` | Each rendered cluster is backfilled from `unassigned_image_indices` until it has at least this many reps |
| `brand_dna_max_images_per_cluster` | `3` | Cap on images shown per cluster strip in the PDF |
| `brand_dna_min_cluster_count` | `3` | Hard floor on rendered clusters — empty clusters are kept rather than dropped if removing them would breach this floor |

#### Fashion / clothing image gate

`BrandDnaService` runs the downloaded image set through a `FashionFilter` (`src/utils/fashion_filter.py`) **before** pHash dedupe and the vision LLM. This addresses the case-study requirement that analyzed images be confidently fashion-related, and keeps editorial illustrations or store banners from biasing the LLM's read of the brand.

| Strategy | Behavior | Extra deps |
|----------|----------|------------|
| `null` | Identity — keeps every image | None |
| `heuristic` | Drops images with extreme aspect ratios (banners, hero strips, icons); keeps the long tail | None |
| `clip` (default) | Zero-shot OpenCLIP classification against positive (clothing/apparel/accessory) and negative (illustration, interior, landscape, logo) prompts; drops anything below the configured threshold | `pip install open_clip_torch torch` — lazy-imported; if missing, the filter logs once and falls back to a no-op so the run still completes |

| Key | Default | Purpose |
|-----|---------|---------|
| `brand_dna_fashion_filter_strategy` | `clip` | One of `null` / `heuristic` / `clip` |
| `brand_dna_fashion_filter_model` | `ViT-B-32` | OpenCLIP model (clip strategy only) |
| `brand_dna_fashion_filter_pretrained` | `openai` | OpenCLIP pretrained weights tag (clip strategy only) |
| `brand_dna_fashion_filter_threshold` | `0.55` | Minimum P(fashion) softmax score to keep an image (clip strategy only) |

If the filter removes **every** image, `BrandDnaService` raises rather than calling Gemini on an empty set — lower the threshold or set strategy to `null` to bypass.

Create a `.env` file in the project root for local runs (gitignored), then export the vars in your shell before running. Example:

```bash
GEMINI_API_KEY=your-key-here
MONGO_CONNECTION_STRING=mongodb://ea_brand:ea_brand_dev@localhost:27017/?authSource=admin
DB_NAME=brand_intelligence
```

---

## Healthchecks and validation

Before any crawling begins, `main.py` runs a series of checks via `src/utils/healthchecks.py`. Failures exit with code `1`.

### Database healthcheck

`db_healthcheck()` issues a MongoDB admin `ping` against the configured database. This confirms the connection string is valid and the server is reachable.

### Website URL healthcheck

`healthcheck_website_url()` validates the `--url` argument:

- Must start with `http://` or `https://`
- Must respond with HTTP **200**
- Bot-protection interstitials (Akamai, etc.) are treated as failures and surface the detected provider/reason — see [Bot prevention](#bot-prevention) for full detection and handling behavior

### Social handle validation

When `--social_handle` is provided, `check_social_media_exists()` probes Instagram and Twitter/X for the normalized handle. Only platforms that resolve to a live profile are scraped. Invalid handles are ignored rather than failing the run.

### Runtime guards

- Website crawler records bot-protection events in the bundle (`stopped_due_to_bot_protection`) but may still return partial results
- Instagram and Twitter scrapers fail gracefully — warnings are logged and the pipeline continues with remaining sources
- Brand DNA synthesis is skipped when the merged bundle contains **no images**

---

## Bot prevention

Many retail sites sit behind Akamai and similar WAF/CDN layers. The agent detects common bot-blocking responses and stops crawling rather than treating interstitial pages as real content.

### Detection

`detect_bot_protection()` in `src/utils/bot_protection.py` inspects every HTTP GET response (via `HttpClient.get()`). It returns a `BotProtectionDetection` with `provider` and `reason` when any of these signals match:

| Signal | Provider | Reason |
|--------|----------|--------|
| HTTP **429** | `unknown` | `http_429_rate_limited` |
| HTTP **403** with Akamai body/header markers | `akamai` | `http_403_forbidden` |
| HTTP **403** (generic) | `unknown` | `http_403_forbidden` |
| Akamai interstitial HTML (`bm-verify`, `/interstitial/ic.html`, `triggerInterstitialChallenge`, empty-title challenge pages) | `akamai` | `interstitial_challenge` |

When bot protection is detected, `HttpClient` sets `last_bot_protection`, logs the block, and returns `None` instead of the response body.

### Behavior by stage

| Stage | On bot protection |
|-------|-------------------|
| **Startup healthcheck** (`healthcheck_website_url`) | Hard failure — the run exits with code `1` |
| **Website crawler** | Stops immediately; returns a `BrandContentBundle` with any products/text collected so far plus `bot_protection` populated |
| **Instagram / Twitter scrapers** | Returns an empty or partial bundle with `bot_protection` when the API/timeline request is blocked |
| **Bundle merge** | Preserves the first `bot_protection` event across sources so callers can log it |
| **Brand DNA** | Continues if the merged bundle still has images from other sources; skipped entirely when no images remain |

Check `bundle.stopped_due_to_bot_protection` (or `bundle.bot_protection`) on any `BrandContentBundle` to see whether a source was blocked. The CLI logs provider and reason when the website crawl is stopped.

Real-world examples exercised in tests: **Adidas** (Akamai 403) and **Zara** (Akamai interstitial challenge). **Nike**, **Prada**, and **Allbirds** typically crawl successfully without triggering detection.

---

## Setup

### Prerequisites

- Python **3.13**
- MongoDB **7** (local install or via Docker Compose)
- A Gemini API key

### Direct Python run

```bash
# Clone and enter the project
cd ea-brand-intelligence-agent

# Create and activate a virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start MongoDB (if not already running)
docker compose up -d ea-brand-mongodb

# Export required env vars (or source a .env file in your shell)
export GEMINI_API_KEY=your-key-here
export MONGO_CONNECTION_STRING=mongodb://ea_brand:ea_brand_dev@localhost:27017/?authSource=admin

# Run the agent
python main.py \
  --config local \
  --url https://www.prada.com \
  --name prada \
  --social_handle @prada
```

#### CLI arguments

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | No | `local`, `stage`, or `prod` (default: `local`) |
| `--url` | **Yes** | Primary brand website URL |
| `--name` | **Yes** | Brand display name |
| `--social_handle` | No | Instagram/Twitter handle (with or without `@`) |
| `--refresh-extraction-config` | No | Re-infer and overwrite cached per-domain metadata extraction config |

Output is written to `outputs/<slug>_brand_dna.json` and `outputs/<slug>_brand_dna.pdf`.

### Docker

Build the image:

```bash
docker build -t ea-brand-agent .
```

Run a one-off job (MongoDB must be reachable at the connection string you pass):

```bash
docker run --rm \
  -e GEMINI_API_KEY=your-key-here \
  -e MONGO_CONNECTION_STRING=mongodb://ea_brand:ea_brand_dev@host.docker.internal:27017/?authSource=admin \
  -e DB_NAME=brand_intelligence \
  -v "$(pwd)/outputs:/app/outputs" \
  ea-brand-agent \
  --config local \
  --url https://www.prada.com \
  --name prada \
  --social_handle @prada
```

The container entrypoint is `python main.py`. Default `CMD` is `--help`.

### Docker Compose

Compose defines two services:

| Service | Purpose |
|---------|---------|
| `ea-brand-mongodb` | MongoDB 7 with healthcheck (always available) |
| `ea-brand-agent` | Application container (profile: `app`) |

**Start MongoDB only** (typical for local Python development):

```bash
docker compose up -d ea-brand-mongodb
```

**Run the agent via Compose** (builds the image, waits for MongoDB to be healthy):

```bash
export GEMINI_API_KEY=your-key-here

docker compose --profile app run --rm \
  -v "$(pwd)/outputs:/app/outputs" \
  ea-brand-agent \
  --config local \
  --url https://www.prada.com \
  --name prada \
  --social_handle @prada
```

MongoDB data persists in the `ea-brand-mongodb-data` volume.

---

## Scrapers and crawlers (bundles)

Every content source emits the same Pydantic model: `BrandContentBundle` (`src/models/brand_content.py`). This keeps downstream services (especially Brand DNA) agnostic of where images and text came from.

```python
BrandContentBundle(
    source="website" | "instagram" | "twitter" | "combined" | ...,
    brand_url=...,
    min_resolution="512x512",
    images=[DiscoveredImage, ...],
    texts=[BrandTextSnippet, ...],
    pages_crawled=int,
    candidates_checked=int,
    created_at=datetime,
    bot_protection=BotProtectionFailure | None,
)
```

### Website crawler

`WebsiteCrawler` (`src/crawlers/website_crawler.py`) is the primary data source.

- Breadth-first crawl of the brand domain up to `max_crawl_pages` / `max_crawl_depth`
- Extracts product imagery via config-driven metadata rules (JSON-LD, DOM selectors, embedded state)
- Collects editorial and on-page copy (meta tags, hero text, about pages)
- Caches per-domain extraction configs in MongoDB via `MetadataConfigService`
- Stops early on bot protection or when `lowest_processeable_product_count` is reached

### Instagram scraper

`InstagramScraper` (`src/crawlers/instagram_scraper.py`) uses the unauthenticated `web_profile_info` API.

- Profile bio + up to `instagram_max_posts` recent posts
- Carousel children, captions, and image URLs above minimum resolution
- Returns an empty bundle (does not raise) when the profile is missing

### Twitter / X scraper

`TwitterScraper` (`src/crawlers/twitter_scraper.py`) reads the public syndication timeline endpoint.

- Profile bio + up to `twitter_max_tweets` recent tweets
- Photo media only (video skipped)
- Defensive JSON tree walk — resilient to syndication payload shape changes

### Merging bundles

`merge_content_bundles()` (`src/utils/content_bundle_merger.py`) combines website, Instagram, and Twitter output:

- Concatenates and **deduplicates images by exact URL** (CDN transform variants like `/t_default/` vs `/t_PDP_1280_v1/` remain separate at this stage — perceptual dedup happens later in Brand DNA)
- Dedupes text snippets by content
- Sums crawl counters across sources
- Surfaces the first bot-protection failure (if any)
- Sets `source="combined"`; inherits `brand_url` from the first website bundle

In `main.py` the flow is: crawl website → optionally scrape social → merge → pass to Brand DNA.

### Brand-voice text harvesting

Alongside images, the website crawler harvests short brand-voice snippets from every page it visits via `src/utils/text_snippet_extractor.py`. No extra HTTP round-trips — the same parsed HTML is reused.

Snippet kinds and their weights (higher = stronger brand-voice signal, used to order the top 40 snippets passed to Gemini synthesis):

| Kind | Weight | Source |
|------|--------|--------|
| `editorial` / `about` | 1.5 | `<p>` blocks on editorial pages or pages whose path matches `/about`, `/story`, `/journal`, `/manifesto`, `/sustainability`, `/heritage`, `/values`, `/our-` |
| `hero` | 1.1 | `<h1>` and first three `<h2>` on the homepage |
| `product_description` | 0.9 | Description blocks on PDPs (class contains `description` / `product-detail`, or `itemprop="description"`) |
| `og_description` | 0.6–0.7 | `<meta property="og:description">` / `twitter:description` |
| `meta_description` | 0.6 | `<meta name="description">` |
| `og_title` | 0.5 | `<meta property="og:title">` |
| `title` | 0.4 | `<title>` |

When the crawl never visits an editorial page and `editorial_probe_enabled=true`, the crawler best-effort fetches up to four of `editorial_candidate_paths()` (`/about`, `/about-us`, `/our-story`, `/story`, `/journal`, `/manifesto`, `/sustainability`) — bounded, silent on errors, halts on bot protection.

Social captions and tweets are filtered through `is_promotional_social_post()` to skip percentage-off promos, sale/discount copy, and store-opening/RSVP flyers (when at least two of the *register* / *store* / *invite* keyword groups hit).

### Image deduplication

Images are deduplicated in two stages:

| Stage | Where | Method | What it catches |
|-------|--------|--------|-----------------|
| **URL dedup** | Website crawler (`candidate_index`), bundle merge (`merge_content_bundles`) | Exact URL match | Same image URL harvested twice from different pages or sources |
| **Perceptual dedup** | `BrandDnaService` (`src/utils/image_helpers.py`) | pHash (Hamming distance ≤ 8) | Same product shot at different CDN transform URLs — e.g. Nike `/t_default/hero.jpg`, `/t_PDP_1280_v1/hero.jpg`, `/t_product_v1/hero.jpg` |

Perceptual dedup runs **after image download** and **before** palette extraction or the vision LLM. When near-duplicates are found, the highest-resolution variant is kept and the rest are dropped from the analysis set.

---

## Brand DNA service

`BrandDnaService` (`src/services/brand_dna_service.py`) is **crawler-agnostic**. It accepts any `BrandContentBundle` — today produced by the scrapers above, tomorrow by additional sources — and returns a structured report plus rendered PDF.

### Pipeline steps

1. **Sample images** — diversity-aware selection (one image per source page first, then top-up) capped at `brand_dna_max_images`
2. **Download** — parallel fetch of image bytes (reuses the shared thread pool in production)
3. **Fashion gate** — `FashionFilter.filter()` drops non-apparel imagery; see [Fashion / clothing image gate](#fashion--clothing-image-gate). Raises if nothing survives.
4. **Perceptual dedupe** — pHash near-duplicate collapse (`image_helpers.py`); keeps highest-resolution variant per duplicate group
5. **Color palette** — dominant swatches extracted via `extract_palette()` (`image_helpers.py`)
6. **Visual analysis** — Gemini vision pass (`LlmService.analyze_brand_visuals`) → clusters, silhouettes, mood
7. **Synthesis** — Gemini text pass (`LlmService.synthesize_brand_dna`) combining visuals, palette, text snippets, and product names
8. **Cluster rendering** — `_resolve_clusters()` maps LLM cluster assignments back to image URLs. Product shots beat editorial/social images when competing for a representative slot (editorial images often carry brand overlays that bias what reads as the canonical garment). Thin clusters are backfilled from `unassigned_image_indices` up to `brand_dna_min_images_per_cluster`; empty clusters are dropped unless that would breach `brand_dna_min_cluster_count`.
9. **Render** — PDF via ReportLab (`src/utils/brand_dna_pdf_renderer.py`) and JSON snapshot on disk

### Outputs

| File | Contents |
|------|----------|
| `outputs/<slug>_brand_dna.json` | Full `BrandDnaReport` (palette, visual analysis, brand voice, positioning, clusters) |
| `outputs/<slug>_brand_dna.pdf` | Human-readable dossier with color swatches and cluster image strips |

---

## Tests

Tests are organized under `tests/` and run with **pytest**. Configuration is in `pytest.ini`.

```
tests/
├── conftest.py              # Shared fixtures and pytest options
├── fixtures/                # Shared test fixtures (e.g. mock extraction config)
├── unit_tests/              # Fast offline tests (default `pytest` run)
├── integration_tests/       # Live network / MongoDB tests (opt-in)
└── benchmarks/              # Brand DNA LLM model benchmarks (opt-in)
```

### Unit tests

Location: `tests/unit_tests/`

Fast, offline tests with mocked HTTP responses. No MongoDB, network, or API keys required.

| File | Focus |
|------|-------|
| `test_bot_protection.py` | Bot-protection detection and crawler stop behavior |
| `test_content_bundle_merger.py` | Bundle merge and deduplication |
| `test_extraction_config_repository.py` | MongoDB config persistence (uses test DB if available) |
| `test_instagram_scraper.py` | Instagram parsing (mocked API payloads) |
| `test_twitter_scraper.py` | Twitter syndication parsing (mocked HTML) |
| `test_website_crawler.py` | URL rules, metadata extraction, crawl logic |
| `test_metadata_extractor.py` | Config-driven product metadata extraction |
| `test_llm_service.py` | Vision batching and cluster assignment helpers |
| `test_image_helpers.py` | Image dimensions, resolution parsing, palette extraction, and pHash dedupe |
| `test_brand_dna_service.py` | Brand DNA pipeline wiring (including phash dedupe before LLM) |
| `test_fashion_filter.py` | Null/heuristic/CLIP fashion-filter strategies and factory wiring (CLIP graceful-failure path covered) |
| `test_social_network_verifier.py` | Handle normalization and HTML validation (mocked) |

**Run unit tests:**

```bash
source .venv/bin/activate
python -m pytest tests/unit_tests
```

Run a single file or test:

```bash
python -m pytest tests/unit_tests/test_website_crawler.py -v
python -m pytest tests/unit_tests/test_content_bundle_merger.py::test_merge_combines_three_bundles_and_dedupes_images_and_texts -v
```

By default, tests marked `@pytest.mark.integration`, `@pytest.mark.llm_integration`, and `@pytest.mark.benchmark` are **skipped**.

### Integration tests

Location: `tests/integration_tests/`

Live network tests against real brand sites. Require MongoDB and explicit opt-in flags.

| File | What it exercises |
|------|-------------------|
| `test_integration_website_crawler.py` | End-to-end website crawl per brand |
| `test_integration_metadata_extractor.py` | Live HTML → product metadata extraction |
| `test_integration_metadata_config.py` | LLM-driven extraction config inference (`@pytest.mark.llm_integration`) |
| `test_social_network_verifier_live.py` | Live Instagram/Twitter handle checks |

#### Brand fixtures and expected outcomes

Integration brand cases are defined in `tests/integration_tests/integration_helpers.py` (`BRAND_CASES`). Each brand is parametrized across the live crawler and metadata tests.

| Brand | URL | Expected outcome | Website crawl test | Metadata extractor tests |
|-------|-----|------------------|--------------------|--------------------------|
| **Nike** | `https://www.nike.com/` | Success | Crawl completes; ≥ 10 products | Extracts ≥ 3 metadata-rich candidates; style-code SKUs required |
| **Prada** | `https://www.prada.com/` | Success | Crawl completes; ≥ 10 products | Extracts ≥ 2 rich candidates; style-code check relaxed |
| **Allbirds** | `https://www.allbirds.com/` | Success | Crawl completes; ≥ 10 products | Extracts ≥ 2 rich candidates; style-code check relaxed |
| **Adidas** | `https://www.adidas.com/` | Bot protection | Asserts `stopped_due_to_bot_protection` (Akamai 403) | Skipped — site blocked |
| **Zara** | `https://www.zara.com/` | Bot protection | Asserts `stopped_due_to_bot_protection` (Akamai interstitial) | Skipped — site blocked |

Notes:

- Bot-protected brands are **not** seeded with extraction configs in MongoDB (see `conftest.py`).
- `test_integration_metadata_config.py` runs against **Nike only** and requires `--run-llm-integration` plus `GEMINI_API_KEY` to infer a fresh extraction config.
- Unit tests in `tests/unit_tests/test_bot_protection.py` cover Adidas/Zara blocking patterns with mocked HTML — no network required.

**Prerequisites:**

```bash
# MongoDB running (matches configs/test.py defaults)
docker compose up -d ea-brand-mongodb

# Optional: for LLM config inference tests
export GEMINI_API_KEY=your-key-here
```

**Run integration tests:**

```bash
# Live network tests (crawlers, metadata, social verification)
python -m pytest tests/integration_tests --run-integration -m integration

# Include LLM config inference (requires GEMINI_API_KEY)
python -m pytest tests/integration_tests --run-integration --run-llm-integration
```

**Run everything (unit + integration):**

```bash
python -m pytest tests/unit_tests tests/integration_tests --run-integration --run-llm-integration
```

Integration tests seed default extraction configs into the test database (`ea_brand_agent_test`) on first run. Tests skip automatically when MongoDB is unreachable.

### Benchmark tests

Location: `tests/benchmarks/`

Compare Brand DNA **PDF dossier quality** across Gemini models for each crawlable brand in `BRAND_CASES` (nike, prada, allbirds). Adidas and zara are excluded (`expect_bot_protection=True`).

| Role | Model |
|------|-------|
| **Judge** | `gemini-3.1-pro-preview` (reads each candidate PDF and scores quality) |
| **Candidates** | `gemini-2.5-flash`, `gemini-3.5-flash`, `gemini-3.1-pro-preview` |

Scoring is **PDF output quality only** — no speed or latency metrics. Each candidate runs the full `BrandDnaService` pipeline; the judge (`gemini-3.1-pro-preview`) evaluates the rendered PDF using a six-dimension scorecard:

| Dimension | Weight |
|-----------|--------|
| Visual Identity | 25% |
| Brand Voice & Textual Identity | 20% |
| Audience Signals | 15% |
| Aesthetic Clusters | 20% |
| Completeness & Structure | 10% |
| Actionability | 10% |

Results include per-dimension scores, justifications, top strengths/gaps, a one-line verdict, and a weighted score used for ranking.

#### Recommended model

**Use `gemini-3.5-flash` for Brand DNA synthesis** (`LLM_MODEL=gemini-3.5-flash`).

It ranked first on all three benchmark brands (Nike, Prada, Allbirds), with the highest weighted PDF quality scores across visual identity, brand voice, and actionability. `gemini-2.5-flash` consistently scored lowest; `gemini-3.1-pro-preview` was a close second but did not win any brand.

```bash
export LLM_MODEL=gemini-3.5-flash
```

#### Benchmark results

Latest run (May 2026). Judge: `gemini-3.1-pro-preview`. Scores are weighted PDF quality (1–10).

**Per brand**

| Brand | gemini-3.5-flash | gemini-3.1-pro-preview | gemini-2.5-flash | Winner |
|-------|------------------|------------------------|------------------|--------|
| **Nike** | **8.25** | 7.75 | 6.35 | gemini-3.5-flash |
| **Prada** | **8.85** | 8.40 | 6.30 | gemini-3.5-flash |
| **Allbirds** | **8.40** | 7.45 | 7.15 | gemini-3.5-flash |

**Aggregate ranking** (average weighted score across brands)

| Rank | Model | Avg score |
|------|-------|-----------|
| 1 | **gemini-3.5-flash** | **8.50** |
| 2 | gemini-3.1-pro-preview | 7.87 |
| 3 | gemini-2.5-flash | 6.60 |

**Highlights**

- **gemini-3.5-flash** — strongest on visual identity and actionability; top one-line verdicts on Prada ("exceptionally strong, highly actionable") and Allbirds ("ideal foundation for AI-driven design and copy generation").
- **gemini-3.1-pro-preview** — solid second place; competitive on Prada (8.40) but trailed on Nike and Allbirds.
- **gemini-2.5-flash** — weakest overall; recurring gaps in brand voice accuracy and generic audience signals (e.g. Nike brand voice scored 4/10).

**Prerequisites:**

- `GEMINI_API_KEY`
- Network access (image downloads from fixture URLs)
- Per-brand fixtures at `tests/benchmarks/fixtures/benchmark/{name}_brand_content_bundle.json`

Regenerate fixtures from a live crawl when stale:

```bash
python -m pytest tests/benchmarks/benchmark_llm_models.py \
  --run-benchmark --run-llm-integration --run-integration \
  --benchmark-live-crawl --benchmark-brand nike -v
```

**Run benchmark:**

```bash
# All crawlable brands (~27 LLM calls)
python -m pytest tests/benchmarks --run-benchmark --run-llm-integration -v

# Single brand
python -m pytest tests/benchmarks/benchmark_llm_models.py \
  --run-benchmark --run-llm-integration --benchmark-brand nike -v
```

**Output:** `outputs/benchmarks/<run_id>/<brand>/` — one PDF per model subdirectory + `scores.json` with judge scores and ranking.

---

## Project layout

```
├── configs/           # Environment profiles (local, stage, prod, test)
├── main.py            # CLI entrypoint and orchestration
├── src/
│   ├── crawlers/        # Website, Instagram, Twitter scrapers
│   ├── models/        # Pydantic models (BrandConfig, BrandContentBundle, BrandDnaReport, …)
│   ├── repositories/  # MongoDB persistence
│   ├── services/      # Brand DNA, LLM, HTTP, metadata config, social verification
│   └── utils/         # Healthchecks, bundle merge, metadata extraction, image helpers, PDF renderer, bot protection, fashion filter, text snippet extractor
├── tests/
│   ├── conftest.py
│   ├── fixtures/              # Shared test fixtures
│   ├── unit_tests/            # Offline unit tests
│   ├── integration_tests/     # Live network / MongoDB tests
│   └── benchmarks/              # Brand DNA LLM benchmarks
├── outputs/           # Generated Brand DNA output (gitignored in practice)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## TODO

- [ ] **Upload outputs to S3** — persist generated Brand DNA artifacts (`outputs/<slug>_brand_dna.{json,pdf}`) to S3 after each run
- [ ] **Enable multiprocessing for crawlers** — run `WebsiteCrawler`, `InstagramScraper`, and `TwitterScraper` in parallel instead of sequentially before bundle merge
- [ ] **`--skip-website` (social-only mode)** — optional CLI flag to bypass the website probe and crawl entirely and continue with Instagram/Twitter only (no HTTP 200 requirement on `--url`; useful when the site is bot-blocked or you only want social-sourced Brand DNA)
