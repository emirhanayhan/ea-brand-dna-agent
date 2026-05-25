"""End-to-end Brand DNA synthesis pipeline.

Inputs: a BrandConfig + a BrandContentBundle (today produced by the website
crawler; tomorrow by social-media crawlers — same shape).

Outputs: a BrandDnaReport (Pydantic) plus a rendered PDF on disk.

The service is intentionally crawler-agnostic. It only knows about the
generic bundle and the LlmService; nothing about HTML, SKUs, or selectors.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from src.models.brand_configs import BrandConfig
from src.models.brand_content import BrandContentBundle
from src.models.brand_dna import (
    BrandDnaReport,
    BrandVisualAnalysis,
    ClusterRender,
)
from src.models.image_outputs import DiscoveredImage
from src.utils.brand_dna_pdf_renderer import render_brand_dna_pdf
from src.services.http_service import HttpClient
from src.services.llm_service import LlmService
from src.utils.fashion_filter import FashionFilter, make_fashion_filter
from src.utils.image_helpers import dedupe_by_phash, extract_palette

logger = logging.getLogger("BrandDnaService")

_IMAGE_FETCH_MAX_BYTES = 4_000_000


class BrandDnaService:
    def __init__(
        self,
        settings: Dict[str, Any],
        http_client: HttpClient,
        llm_service: LlmService,
        executor: Optional[ThreadPoolExecutor] = None,
        fashion_filter: Optional[FashionFilter] = None,
    ):
        self.settings = settings
        self.http_client = http_client
        self.llm_service = llm_service
        self.executor = executor
        self.fashion_filter: FashionFilter = (
            fashion_filter if fashion_filter is not None else make_fashion_filter(settings)
        )
        self.max_analysis_images = int(settings["brand_dna_max_images"])
        self.cluster_target = int(settings["brand_dna_cluster_target"])
        self.palette_size = int(settings["brand_dna_palette_size"])
        self.output_dir = Path(settings["brand_dna_output_dir"])
        # Cluster sizing knobs. Defaults match case-study expectations: each
        # cluster should ship at least 2 representative images, the PDF caps
        # at 3 images per cluster, and we never drop below 3 clusters total
        # (the lower bound the brief specifies for the dossier).
        self.min_images_per_cluster = int(
            settings.get("brand_dna_min_images_per_cluster", 2)
        )
        self.max_images_per_cluster = int(
            settings.get("brand_dna_max_images_per_cluster", 3)
        )
        self.min_cluster_count = int(
            settings.get("brand_dna_min_cluster_count", 3)
        )

    def run(
        self,
        brand_config: BrandConfig,
        bundle: BrandContentBundle,
        output_dir: Optional[Path | str] = None,
    ) -> Tuple[BrandDnaReport, Path]:
        if not bundle.images:
            raise ValueError(
                "BrandContentBundle has no images; cannot synthesize brand DNA."
            )

        brand_name = (
            brand_config.name
            or _brand_name_from_url(str(brand_config.url) if brand_config.url else "")
            or "Untitled Brand"
        )
        brand_url = str(brand_config.url) if brand_config.url else None
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        sampled_images = self._sample_images(bundle.images)
        logger.info(
            "brand_dna: analyzing %s of %s images",
            len(sampled_images),
            len(bundle.images),
        )

        image_bytes_by_url = self._download_images(sampled_images)
        usable = [
            (img, image_bytes_by_url[str(img.url)])
            for img in sampled_images
            if str(img.url) in image_bytes_by_url
        ]
        if not usable:
            raise RuntimeError(
                "Could not download any analysis images; aborting brand DNA run."
            )

        before_filter = len(usable)
        usable = self.fashion_filter.filter(usable)
        if len(usable) < before_filter:
            logger.info(
                "brand_dna: fashion_filter[%s] kept %s of %s images",
                self.fashion_filter.name,
                len(usable),
                before_filter,
            )
        if not usable:
            raise RuntimeError(
                "fashion filter removed every image; check filter threshold or "
                "disable via brand_dna_fashion_filter_strategy=null."
            )

        before = len(usable)
        usable = dedupe_by_phash(usable)
        if len(usable) < before:
            logger.info(
                "brand_dna: phash dedupe kept %s of %s downloaded images",
                len(usable),
                before,
            )
        image_bytes_by_url = {str(image.url): data for image, data in usable}

        palette = extract_palette(
            (raw for _, raw in usable), n_colors=self.palette_size
        )

        visual = self.llm_service.analyze_brand_visuals(
            brand_name=brand_name,
            brand_url=brand_url,
            images=usable,
            cluster_target=self.cluster_target,
        )

        synthesis = self.llm_service.synthesize_brand_dna(
            brand_name=brand_name,
            brand_url=brand_url,
            visual=visual,
            palette=palette,
            text_snippets=bundle.texts,
            product_names=[img.product_name for img in bundle.images if img.product_name],
        )

        clusters = _resolve_clusters(
            visual,
            [img for img, _ in usable],
            min_images_per_cluster=self.min_images_per_cluster,
            max_images_per_cluster=self.max_images_per_cluster,
            min_cluster_count=self.min_cluster_count,
        )

        report = BrandDnaReport(
            brand_name=brand_name,
            brand_url=brand_url,
            generated_at=datetime.now(timezone.utc),
            palette=palette,
            visual=visual,
            synthesis=synthesis,
            clusters=clusters,
            image_count=len(bundle.images),
            text_snippet_count=len(bundle.texts),
        )

        slug = _slugify(brand_name or brand_url or "brand")
        pdf_path = render_brand_dna_pdf(
            report,
            image_bytes_by_url=image_bytes_by_url,
            output_path=out_dir / f"{slug}_brand_dna.pdf",
        )
        json_path = out_dir / f"{slug}_brand_dna.json"
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        logger.info("brand_dna: report JSON written to %s", json_path)

        return report, pdf_path

    # --- helpers ------------------------------------------------------------

    def _sample_images(
        self, images: List[DiscoveredImage]
    ) -> List[DiscoveredImage]:
        """Deterministic, diversity-aware sampling.

        When over the cap, first pick one image per source page, then top up
        with the remainder in the original order. This protects us from a
        single PDP dominating the analysis.
        """
        if len(images) <= self.max_analysis_images:
            return list(images)

        by_page: Dict[str, List[DiscoveredImage]] = {}
        for img in images:
            by_page.setdefault(str(img.source_page), []).append(img)

        diverse: List[DiscoveredImage] = []
        leftovers: List[DiscoveredImage] = []
        for page, page_images in by_page.items():
            diverse.append(page_images[0])
            leftovers.extend(page_images[1:])

        if len(diverse) >= self.max_analysis_images:
            return diverse[: self.max_analysis_images]
        topup_needed = self.max_analysis_images - len(diverse)
        return diverse + leftovers[:topup_needed]

    def _download_images(
        self, images: List[DiscoveredImage]
    ) -> Dict[str, bytes]:
        results: Dict[str, bytes] = {}

        def _fetch(image: DiscoveredImage) -> Tuple[str, Optional[bytes]]:
            url = str(image.url)
            data = self.http_client.get_stream(url, max_bytes=_IMAGE_FETCH_MAX_BYTES)
            return url, data

        with self.executor as pool:
            for url, data in pool.map(_fetch, images):
                if data:
                    results[url] = data
                else:
                    logger.debug("brand_dna: failed to download %s", url)

        logger.info(
            "brand_dna: downloaded %s/%s analysis images",
            len(results),
            len(images),
        )
        return results


def _resolve_clusters(
    visual: BrandVisualAnalysis,
    images: List[DiscoveredImage],
    *,
    min_images_per_cluster: int = 2,
    max_images_per_cluster: int = 3,
    min_cluster_count: int = 3,
) -> List[ClusterRender]:
    """Map LLM cluster assignments to image URLs, backfilling thin clusters.

    Behavior beyond a plain 1:1 map:
    - Each cluster is filled up to max_images_per_cluster from its own
      assigned indices.
    - Clusters that still have fewer than min_images_per_cluster reps after
      that pass are topped up from visual.unassigned_image_indices (the pool
      produced by issue #2's fix). This keeps strategist-facing reports
      visually balanced when the LLM left a cluster under-populated.
    - Clusters that end up with zero usable reps are dropped — unless that
      would push the rendered cluster count below min_cluster_count (the
      case-study floor of 3), in which case the empty cluster is kept and
      will render without an image strip.
    """
    used_urls: set[str] = set()
    primary_picks: List[List[str]] = []
    for cluster in visual.cluster_observations:
        # Rank candidates so product shots are picked before editorial /
        # campaign hero images. Editorial images often carry brand overlays
        # ("LES BENJAMINS El Gringo") and bias what a strategist reads as
        # the canonical garment representation. We keep ordering stable
        # within each tier so the LLM's own ranking still matters.
        ordered = sorted(
            range(len(cluster.image_indices)),
            key=lambda pos: _source_kind_rank(
                images, cluster.image_indices[pos] - 1
            ),
        )
        urls: List[str] = []
        for pos in ordered:
            idx = cluster.image_indices[pos]
            i = idx - 1
            if not (0 <= i < len(images)):
                continue
            url = str(images[i].url)
            if url in used_urls:
                continue
            urls.append(url)
            used_urls.add(url)
            if len(urls) >= max_images_per_cluster:
                break
        primary_picks.append(urls)

    backfill_pool: deque[int] = deque(
        idx for idx in getattr(visual, "unassigned_image_indices", []) or []
    )

    def _next_backfill() -> Optional[str]:
        while backfill_pool:
            idx = backfill_pool.popleft()
            i = idx - 1
            if not (0 <= i < len(images)):
                continue
            url = str(images[i].url)
            if url in used_urls:
                continue
            used_urls.add(url)
            return url
        return None

    for urls in primary_picks:
        while len(urls) < min_images_per_cluster:
            picked = _next_backfill()
            if picked is None:
                break
            urls.append(picked)

    rendered: List[ClusterRender] = []
    skipped = 0
    for cluster, urls in zip(visual.cluster_observations, primary_picks):
        rendered.append(
            ClusterRender(
                label=cluster.cluster_label,
                description=cluster.description,
                key_elements=cluster.key_elements,
                image_urls=urls,
            )
        )

    if len(rendered) > min_cluster_count:
        kept: List[ClusterRender] = [c for c in rendered if c.image_urls]
        skipped = len(rendered) - len(kept)
        if skipped and len(kept) >= min_cluster_count:
            logger.warning(
                "brand_dna: dropped %s empty cluster(s) from PDF "
                "(kept %s, min required %s)",
                skipped,
                len(kept),
                min_cluster_count,
            )
            rendered = kept

    return rendered


# Lower rank wins. Product shots first, then editorial, social, anything else.
_SOURCE_KIND_PRIORITY: Dict[str, int] = {
    "product": 0,
    "editorial": 1,
    "social": 2,
    "other": 3,
}


def _source_kind_rank(images: List[DiscoveredImage], i: int) -> int:
    if not (0 <= i < len(images)):
        return 99
    return _SOURCE_KIND_PRIORITY.get(images[i].source_kind, 4)


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.lower()).strip("-")
    return slug or "brand"


def _brand_name_from_url(url: str) -> Optional[str]:
    if not url:
        return None
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    parts = netloc.split(".")
    if len(parts) >= 2:
        return parts[-2].capitalize()
    return netloc or None
