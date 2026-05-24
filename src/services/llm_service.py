import base64
import io
import json
import logging
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from PIL import Image

from src.models.brand_content import BrandTextSnippet
from src.models.brand_dna import (
    BatchClusterAssignment,
    BrandDnaSynthesis,
    BrandVisualAnalysis,
    ClusterObservation,
    ClusterTheme,
    ClusterThemeDiscovery,
    ColorSwatch,
)
from src.models.image_outputs import DiscoveredImage
from src.models.metadata_extraction_config import MetadataExtractionConfig
from src.models.page_sample import PageSample

logger = logging.getLogger("LlmService")

_EXTRACTION_SYSTEM_PROMPT = """You are a web scraping architect. Given HTML snippets from a brand website,
produce a MetadataExtractionConfig that describes how to extract product_name, sku_id, and
product image URLs during crawling.

Rules:
- sku_id must be a merchant style code when visible (e.g. HF1078-006), never an internal UUID.
- Prefer PDP link URL last path segment for sku_id when product cards or pdpUrl fields exist.
- product_name comes from visible product titles in cards or JSON copy.title / name fields.
- image_keys should list JSON fields containing product hero images.
- card_selector and link_selector must be valid XPath expressions for lxml.
- priority_link_path_contains must include listing and PDP paths such as /w/, /t/, /product/, /shop/.
- Always populate json_sources with next_data and/or json_ld rules when those blocks appear in samples.
- Always populate embedded_image_json_fields with common commerce image JSON keys.
- Copy dom_context card_selector/card_link_selector from product_card when product cards exist.
- sku_style_pattern should match alphanumeric style codes like HF1078-006: ^[A-Z0-9]*\\d[A-Z0-9]*-\\d+$
- Use json_ld source when application/ld+json Product nodes exist.
- Use next_data source when __NEXT_DATA__ script exists.
- Use inline_json when product JSON blocks are embedded in HTML outside script tags.
- Never return empty json_sources, embedded_image_json_fields, or dom_context sku/name attrs when products are visible in samples.
"""

_DISCOVERY_SYSTEM_PROMPT = """You are a senior fashion brand strategist planning aesthetic
territories for a Brand DNA report.

You will see a representative sample of product images from one brand. Your job is
to define a fixed set of mutually exclusive aesthetic territories that span the
brand's visual range.

Rules:
- Each territory label must be specific and editorial — not generic ("casual",
  "athletic performance", "modern essentials").
- Territories must differ on at least one concrete axis: sport/category, audience,
  channel (PDP vs editorial), color story, silhouette, or product type.
- No two labels may share a primary phrase (e.g. do not produce both "Elite Athletic
  Performance" and "Elite Athleticism").
- seed_image_indices must reference images from the sample you are shown (1-based).
"""

_ASSIGNMENT_SYSTEM_PROMPT = """You are a senior fashion brand strategist assigning product
images to predefined aesthetic territories.

You will be given a fixed list of theme labels and descriptions. Your job is ONLY
to assign each image in the batch to exactly one theme via image_indices.

Rules:
- cluster_label in each assignment MUST match a provided theme label exactly.
- Do not rename, merge, or invent themes.
- Each image may appear in at most one theme's image_indices.
- Leave a theme's image_indices empty if no images in this batch match it.
- image_indices are 1-based within the current batch.
"""

_SYNTHESIS_SYSTEM_PROMPT = """You are a senior brand strategist writing a confidential Brand DNA
report. You will be given a visual analysis, the dominant color palette, and any
brand-authored text snippets that were available.

Write like a seasoned consultant: direct, specific, editorial, no clichés.
Ground every claim in the supplied evidence. If textual evidence is thin, infer
voice and audience from the visuals + product naming conventions, but stay
defensible — do not fabricate quoted copy or invented values.
"""

_MAX_TEXT_SNIPPETS_TO_INCLUDE = 30
_DEFAULT_DISCOVERY_SAMPLE_SIZE = 20
_DEFAULT_ASSIGNMENT_BATCH_SIZE = 12
_IMAGE_MAX_DIM = 768
_IMAGE_JPEG_QUALITY = 75


def _sample_for_discovery(
    images: Sequence[Tuple[DiscoveredImage, bytes]], sample_size: int
) -> Tuple[List[Tuple[DiscoveredImage, bytes]], List[int]]:
    """Pick evenly spaced images across the full list for Phase 1 discovery."""
    n = len(images)
    if n == 0:
        return [], []

    count = min(max(sample_size, 1), n)
    if count == n:
        return list(images), list(range(n))

    step = (n - 1) / (count - 1) if count > 1 else 0
    global_indices = [round(i * step) for i in range(count)]
    return [images[i] for i in global_indices], global_indices


def _chunk_for_assignment(
    images: Sequence[Tuple[DiscoveredImage, bytes]], batch_size: int
) -> List[List[Tuple[DiscoveredImage, bytes]]]:
    """Split all images into fixed-size batches for Phase 2 assignment."""
    size = max(batch_size, 1)
    return [list(images[i : i + size]) for i in range(0, len(images), size)]


def _init_assignment_accumulator(
    themes: Sequence[ClusterTheme],
) -> Dict[str, List[int]]:
    return {theme.cluster_label: [] for theme in themes}


def _merge_assignments(
    accumulator: Dict[str, List[int]],
    themes: Sequence[ClusterTheme],
    batch_assignments: Sequence[ClusterObservation],
    batch_start: int,
    batch_len: int,
    assigned_globally: set[int],
) -> None:
    """Merge one batch's local assignments into the global accumulator."""
    theme_labels = {theme.cluster_label for theme in themes}
    for assignment in batch_assignments:
        if assignment.cluster_label not in theme_labels:
            logger.warning(
                "vision assignment: unknown theme %r; skipping",
                assignment.cluster_label,
            )
            continue
        for local_idx in assignment.image_indices:
            if not (1 <= local_idx <= batch_len):
                continue
            global_idx = batch_start + local_idx
            if global_idx in assigned_globally:
                continue
            assigned_globally.add(global_idx)
            accumulator[assignment.cluster_label].append(global_idx)


def _themes_to_observations(
    themes: Sequence[ClusterTheme],
    accumulator: Dict[str, List[int]],
) -> List[ClusterObservation]:
    observations: List[ClusterObservation] = []
    for theme in themes:
        observations.append(
            ClusterObservation(
                cluster_label=theme.cluster_label,
                description=theme.description,
                key_elements=list(theme.key_elements),
                image_indices=accumulator.get(theme.cluster_label, []),
            )
        )
    return observations


def _log_unassigned_images(unassigned: List[int]) -> None:
    """Log unassigned image indices instead of force-merging into clusters.

    A previous implementation appended each unassigned image to whichever
    cluster currently had the fewest images. That silently corrupted the
    clustering whenever the LLM returned partial assignments or an entire
    batch failed — the unrelated images polluted whichever cluster happened
    to be smallest. We now surface them as a separate pool that callers can
    use as a backfill source (see BrandVisualAnalysis.unassigned_image_indices).
    """
    if not unassigned:
        return
    logger.warning(
        "vision assignment: %s images unassigned; preserved as backfill pool",
        len(unassigned),
    )


class LlmService:
    def __init__(self, settings):
        # Brand-analysis prompts contain product copy, campaign names, and
        # cultural references that the default Gemini safety profile
        # occasionally flags (observed: PROHIBITED_CONTENT on Les Benjamins'
        # "El Gringo" / Eastern-streetwear copy). Setting BLOCK_ONLY_HIGH on
        # all categories keeps protection against genuinely unsafe content
        # while letting legitimate fashion analysis through.
        self.llm = ChatGoogleGenerativeAI(
            model=settings["llm_model"],
            google_api_key=settings["gemini_api_key"],
            temperature=0,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
        )
        self._discovery_sample_size = int(
            settings.get(
                "brand_dna_cluster_discovery_images", _DEFAULT_DISCOVERY_SAMPLE_SIZE
            )
        )
        self._assignment_batch_size = int(
            settings.get(
                "brand_dna_cluster_assignment_batch_size",
                _DEFAULT_ASSIGNMENT_BATCH_SIZE,
            )
        )

    # --- existing metadata-extraction inference -----------------------------

    def infer_extraction_config(
        self,
        samples: list[PageSample],
        domain: str,
    ) -> MetadataExtractionConfig:
        structured_llm = self.llm.with_structured_output(MetadataExtractionConfig)

        sample_blocks = []
        for index, sample in enumerate(samples, start=1):
            sample_blocks.append(
                f"Sample {index} ({sample.page_type})\n"
                f"URL: {sample.url}\n"
                f"HTML snippet:\n{sample.html_snippet}\n"
            )

        human_prompt = (
            f"Target domain: {domain}\n"
            f"Samples: {len(samples)} pages\n\n"
            "Produce a MetadataExtractionConfig for crawling this site.\n"
            "For each sample, infer rules for:\n"
            "- product_name\n"
            "- sku_id (merchant style code, not internal UUID)\n"
            "- product image URLs\n\n"
            "Use page_type as a hint:\n"
            "- listing/homepage: product_card selectors, card/link XPath, listing path prefixes\n"
            "- pdp: single-product path pattern, PDP JSON fields\n"
            "- other: only add rules if product signals are visible\n\n"
            "Prefer json_ld / __NEXT_DATA__ / inline product JSON when present.\n"
            "Merge evidence across all samples; prefer selectors/keys that work on multiple pages.\n"
            f"Set domain to {domain!r}.\n\n"
            + "\n---\n".join(sample_blocks)
        )

        result = structured_llm.invoke(
            [
                SystemMessage(content=_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=human_prompt),
            ]
        )
        if isinstance(result, MetadataExtractionConfig):
            if not result.domain:
                return result.model_copy(update={"domain": domain})
            return result
        return MetadataExtractionConfig.model_validate(result)

    # --- brand DNA: vision pass ---------------------------------------------

    def analyze_brand_visuals(
        self,
        brand_name: str,
        brand_url: Optional[str],
        images: Sequence[Tuple[DiscoveredImage, bytes]],
        cluster_target: int = 5,
    ) -> BrandVisualAnalysis:
        """Two-phase vision pass: discover themes globally, then assign all images."""
        if not images:
            return BrandVisualAnalysis()

        effective_target = min(max(cluster_target, 1), len(images))

        if len(images) <= effective_target:
            return self._analyze_small_image_set(
                brand_name=brand_name,
                brand_url=brand_url,
                images=images,
                cluster_target=effective_target,
            )

        discovery = self._discover_cluster_themes(
            brand_name=brand_name,
            brand_url=brand_url,
            images=images,
            cluster_target=effective_target,
        )
        themes = discovery.themes[:effective_target]
        if not themes:
            raise RuntimeError(
                "Theme discovery returned no clusters; cannot produce brand DNA report."
            )

        cluster_observations, unassigned = self._assign_images_to_themes(
            brand_name=brand_name,
            brand_url=brand_url,
            images=images,
            themes=themes,
        )

        return BrandVisualAnalysis(
            garment_categories=discovery.garment_categories,
            silhouettes=discovery.silhouettes,
            styling_notes=discovery.styling_notes,
            formality_spectrum=discovery.formality_spectrum,
            recurring_motifs=discovery.recurring_motifs,
            mood_words=discovery.mood_words,
            cluster_observations=cluster_observations,
            unassigned_image_indices=unassigned,
        )

    def _analyze_small_image_set(
        self,
        brand_name: str,
        brand_url: Optional[str],
        images: Sequence[Tuple[DiscoveredImage, bytes]],
        cluster_target: int,
    ) -> BrandVisualAnalysis:
        discovery = self._discover_cluster_themes(
            brand_name=brand_name,
            brand_url=brand_url,
            images=images,
            cluster_target=cluster_target,
            sample_size=len(images),
        )
        themes = discovery.themes[: len(images)]
        observations: List[ClusterObservation] = []
        for i, theme in enumerate(themes):
            observations.append(
                ClusterObservation(
                    cluster_label=theme.cluster_label,
                    description=theme.description,
                    key_elements=list(theme.key_elements),
                    image_indices=[i + 1],
                )
            )
        return BrandVisualAnalysis(
            garment_categories=discovery.garment_categories,
            silhouettes=discovery.silhouettes,
            styling_notes=discovery.styling_notes,
            formality_spectrum=discovery.formality_spectrum,
            recurring_motifs=discovery.recurring_motifs,
            mood_words=discovery.mood_words,
            cluster_observations=observations,
        )

    def _discover_cluster_themes(
        self,
        brand_name: str,
        brand_url: Optional[str],
        images: Sequence[Tuple[DiscoveredImage, bytes]],
        cluster_target: int,
        sample_size: Optional[int] = None,
    ) -> ClusterThemeDiscovery:
        sample, _global_indices = _sample_for_discovery(
            images, sample_size or self._discovery_sample_size
        )
        content = self._build_discovery_content(
            brand_name=brand_name,
            brand_url=brand_url,
            sample=sample,
            cluster_target=cluster_target,
        )
        structured_llm = self.llm.with_structured_output(ClusterThemeDiscovery)
        try:
            result = structured_llm.invoke(
                [
                    SystemMessage(content=_DISCOVERY_SYSTEM_PROMPT),
                    HumanMessage(content=content),
                ]
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("theme discovery failed: %s", exc)
            raise RuntimeError(
                "Theme discovery failed; cannot produce brand DNA report."
            ) from exc

        if isinstance(result, ClusterThemeDiscovery):
            return result
        return ClusterThemeDiscovery.model_validate(result)

    def _assign_images_to_themes(
        self,
        brand_name: str,
        brand_url: Optional[str],
        images: Sequence[Tuple[DiscoveredImage, bytes]],
        themes: Sequence[ClusterTheme],
    ) -> Tuple[List[ClusterObservation], List[int]]:
        batches = _chunk_for_assignment(images, self._assignment_batch_size)
        structured_llm = self.llm.with_structured_output(BatchClusterAssignment)
        accumulator = _init_assignment_accumulator(themes)
        assigned_globally: set[int] = set()
        batch_start = 0

        for batch_index, batch in enumerate(batches):
            content = self._build_assignment_content(
                brand_name=brand_name,
                brand_url=brand_url,
                batch=batch,
                themes=themes,
            )
            result = self._invoke_assignment_batch(
                structured_llm=structured_llm,
                content=content,
                batch_index=batch_index,
                batch_count=len(batches),
            )
            if result is None:
                batch_start += len(batch)
                continue

            _merge_assignments(
                accumulator=accumulator,
                themes=themes,
                batch_assignments=result.assignments,
                batch_start=batch_start,
                batch_len=len(batch),
                assigned_globally=assigned_globally,
            )
            batch_start += len(batch)

        observations = _themes_to_observations(themes, accumulator)
        unassigned = [
            idx
            for idx in range(1, len(images) + 1)
            if idx not in assigned_globally
        ]
        _log_unassigned_images(unassigned)
        return observations, unassigned

    def _invoke_assignment_batch(
        self,
        structured_llm,
        content: list,
        batch_index: int,
        batch_count: int,
    ) -> Optional[BatchClusterAssignment]:
        """Run one assignment batch with a single retry on failure.

        Returns None when both attempts fail — caller treats the whole batch
        as unassigned rather than dropping its images into a random cluster.
        """
        messages = [
            SystemMessage(content=_ASSIGNMENT_SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]
        for attempt in (1, 2):
            try:
                result = structured_llm.invoke(messages)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "assignment batch %s/%s attempt %s failed: %s",
                    batch_index + 1,
                    batch_count,
                    attempt,
                    exc,
                )
                continue
            if not isinstance(result, BatchClusterAssignment):
                result = BatchClusterAssignment.model_validate(result)
            return result
        logger.error(
            "assignment batch %s/%s gave up after 2 attempts; "
            "batch images will be unassigned",
            batch_index + 1,
            batch_count,
        )
        return None

    # --- brand DNA: synthesis pass ------------------------------------------

    def synthesize_brand_dna(
        self,
        brand_name: str,
        brand_url: Optional[str],
        visual: BrandVisualAnalysis,
        palette: List[ColorSwatch],
        text_snippets: List[BrandTextSnippet],
        product_names: Optional[List[str]] = None,
    ) -> BrandDnaSynthesis:
        structured_llm = self.llm.with_structured_output(BrandDnaSynthesis)

        snippets_block = self._format_text_snippets(text_snippets)
        palette_block = ", ".join(
            f"{sw.hex} ({sw.frequency_pct}%)" for sw in palette if sw.hex
        )
        product_block = ""
        if product_names:
            unique = list(dict.fromkeys(p for p in product_names if p))[:40]
            if unique:
                product_block = (
                    "Representative product names (observed during crawl):\n"
                    + "\n".join(f"- {name}" for name in unique)
                )

        visual_block = json.dumps(visual.model_dump(), indent=2)

        prompt = (
            f'Brand: "{brand_name}"\n'
            f"Website: {brand_url or 'unknown'}\n\n"
            f"VISUAL ANALYSIS:\n{visual_block}\n\n"
            f"DOMINANT PALETTE: {palette_block or 'unavailable'}\n\n"
            f"{product_block}\n\n"
            f"BRAND TEXT SNIPPETS (harvested from the live site, ordered by relevance):\n"
            f"{snippets_block}\n\n"
            "Produce the Brand DNA synthesis. Where textual evidence is thin, infer "
            "from visuals and naming, but flag nothing as a stated value unless the "
            "text snippets support it."
        )

        result = structured_llm.invoke(
            [
                SystemMessage(content=_SYNTHESIS_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        if isinstance(result, BrandDnaSynthesis):
            return result
        return BrandDnaSynthesis.model_validate(result)

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _format_text_snippets(snippets: List[BrandTextSnippet]) -> str:
        if not snippets:
            return "(no text snippets were available)"
        ordered = snippets[:_MAX_TEXT_SNIPPETS_TO_INCLUDE]
        lines = []
        for s in ordered:
            lines.append(
                f"- [{s.kind} | weight={s.weight:.1f} | {s.source_url}]\n  {s.text}"
            )
        return "\n".join(lines)

    def _build_discovery_content(
        self,
        brand_name: str,
        brand_url: Optional[str],
        sample: Sequence[Tuple[DiscoveredImage, bytes]],
        cluster_target: int,
    ) -> list[dict]:
        intro = (
            f'You are analyzing a representative sample of product imagery from the brand '
            f'"{brand_name}" ({brand_url or "unknown site"}). I am showing you '
            f"{len(sample)} images sampled evenly across the brand's full image set.\n\n"
            f"Identify exactly {cluster_target} mutually exclusive aesthetic territories "
            "that together span this brand's visual range.\n\n"
            "Return them in themes[]. For each theme provide cluster_label, description, "
            "key_elements, and seed_image_indices (1-based indices into the sample below)."
        )
        return self._build_image_content(intro, sample)

    def _build_assignment_content(
        self,
        brand_name: str,
        brand_url: Optional[str],
        batch: Sequence[Tuple[DiscoveredImage, bytes]],
        themes: Sequence[ClusterTheme],
    ) -> list[dict]:
        theme_lines = []
        for i, theme in enumerate(themes, start=1):
            theme_lines.append(
                f"{i}. {theme.cluster_label}: {theme.description}"
            )
        intro = (
            f'Assign images from "{brand_name}" ({brand_url or "unknown site"}) to the '
            f"fixed aesthetic territories below. I am showing you {len(batch)} images "
            "from this batch.\n\n"
            "FIXED THEMES (do not rename):\n"
            + "\n".join(theme_lines)
            + "\n\n"
            "Return assignments[] with one entry per theme listed above. "
            "cluster_label must match exactly. Assign each image to at most one theme "
            "via 1-based image_indices within this batch."
        )
        return self._build_image_content(intro, batch)

    def _build_image_content(
        self,
        intro: str,
        batch: Sequence[Tuple[DiscoveredImage, bytes]],
    ) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": intro}]
        for i, (image_meta, raw) in enumerate(batch, start=1):
            data_uri = _encode_image_to_data_uri(raw)
            if data_uri is None:
                continue
            caption_parts = [f"Image {i}"]
            if image_meta.product_name:
                caption_parts.append(f"product: {image_meta.product_name}")
            if image_meta.sku_id:
                caption_parts.append(f"sku: {image_meta.sku_id}")
            content.append(
                {"type": "text", "text": "[" + " · ".join(caption_parts) + "]"}
            )
            content.append({"type": "image_url", "image_url": data_uri})
        return content


def _encode_image_to_data_uri(raw: bytes) -> Optional[str]:
    """Resize down + JPEG-encode + return a data URI Gemini can ingest.

    Returning None signals the caller to skip this image gracefully.
    """
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        logger.debug("vision: could not decode image: %s", exc)
        return None

    w, h = img.size
    longest = max(w, h)
    if longest > _IMAGE_MAX_DIM:
        scale = _IMAGE_MAX_DIM / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_IMAGE_JPEG_QUALITY, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
