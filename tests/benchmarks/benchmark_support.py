"""Helpers for the Brand DNA LLM model benchmark suite."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, model_validator

from src.models.brand_configs import BrandConfig
from src.models.brand_content import BrandContentBundle
from src.services.llm_service import LlmService
from tests.integration_tests.integration_helpers import BRAND_CASES, BrandIntegrationCase

logger = logging.getLogger("BenchmarkSupport")

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "benchmark"
BENCHMARK_OUTPUT_ROOT = Path("outputs") / "benchmarks"

JUDGE_MODEL = "gemini-3.1-pro-preview"
CANDIDATE_MODELS = [
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
]
BENCHMARK_MAX_IMAGES = 20

BENCHMARK_BRAND_CASES = tuple(
    case for case in BRAND_CASES if not case.expect_bot_protection
)

_JUDGE_SYSTEM_PROMPT = """You are an expert brand strategist evaluating an AI-generated Brand DNA report. A Brand DNA report is a structured dossier that captures a fashion brand's visual identity, tone of voice, audience profile, and aesthetic signature — intended to guide AI design generation for that brand.

You have been given a Brand DNA PDF. Your task is to critically evaluate it across six dimensions and produce a structured scorecard.

---

## EVALUATION DIMENSIONS

Score each dimension from 1–10. Be critical. A score of 7+ requires concrete justification.

### 1. Visual Identity (weight: 25%)
Evaluate whether the report accurately captures:
- Dominant color palette with specific values or descriptors (not vague generalities)
- Garment category breakdown (e.g. % outerwear, knitwear, accessories)
- Recurring silhouettes with precise language (e.g. "oversized boxy shoulders", not just "relaxed fit")
- Styling cues: layering logic, fabric texture signals, formality register

Penalty: vague descriptors ("neutral tones", "minimalist aesthetic") with no supporting specifics.

### 2. Brand Voice & Textual Identity (weight: 20%)
Evaluate whether the report accurately captures:
- Tone of voice with 3+ concrete adjectives grounded in actual copy evidence
- Recurring vocabulary or phrase patterns pulled from the brand's own language
- Stated values vs. implied values — does the report distinguish between what the brand claims and what the content actually signals?
- A positioning statement that is brand-specific (not interchangeable with competitors)

Penalty: generic descriptions that could apply to any fashion brand.

### 3. Audience Signals (weight: 15%)
Evaluate whether the report accurately captures:
- Demographic cues (age range, gender, geographic signals if present)
- Psychographic cues (lifestyle, aspiration, cultural references)
- Price-point signals (premium, accessible luxury, mass market)
- Community or subculture indicators if present

Penalty: unsupported claims not evidenced by the brand's actual content.

### 4. Aesthetic Clusters (weight: 20%)
Evaluate whether the report:
- Identifies 3–6 meaningful and distinct aesthetic clusters (not redundant groupings)
- Labels each cluster with a specific, evocative name (not generic like "casual" or "formal")
- Provides a short, precise description per cluster grounded in visual evidence
- Includes representative images or clear visual references per cluster

Penalty: clusters that overlap significantly, or descriptions so generic they apply to any brand.

### 5. Completeness & Structure (weight: 10%)
Evaluate:
- Are all four core sections present (visual, textual, audience, clusters)?
- Is the document navigable for a non-technical brand strategist?
- Are data sources referenced where claims are made?
- Is there a clear summary or positioning synthesis at the end?

### 6. Actionability (weight: 10%)
Evaluate:
- Could an AI design system use this dossier to generate on-brand outputs without additional input?
- Are the insights specific enough to constrain design decisions?
- What would a designer be unable to determine from this report alone?

---

## CALIBRATION NOTES

- A score of 8+ means the section is genuinely exceptional — specific, evidence-grounded, and non-generic.
- A score of 5–7 means the section is present but surface-level or partially generic.
- A score below 5 means the section is missing, misleading, or so vague it provides no signal.
- Do not inflate scores for effort or structure alone — evaluate the quality of insight, not the presence of formatting.

Evaluate only what appears in the PDF. Use any supplementary source-material summary in the user message only to assess grounding and faithfulness — not to invent content that is missing from the PDF.
"""

_SLUG_RE = re.compile(r"[^a-z0-9]+")

_SCORE_WEIGHTS = {
    "visual_identity": 0.25,
    "brand_voice": 0.20,
    "audience_signals": 0.15,
    "aesthetic_clusters": 0.20,
    "completeness": 0.10,
    "actionability": 0.10,
}


class BrandDnaPdfBenchmarkScore(BaseModel):
    visual_identity: float = Field(ge=1, le=10)
    visual_identity_justification: str = Field(
        description="Two-sentence justification for Visual Identity score."
    )
    brand_voice: float = Field(ge=1, le=10)
    brand_voice_justification: str = Field(
        description="Two-sentence justification for Brand Voice score."
    )
    audience_signals: float = Field(ge=1, le=10)
    audience_signals_justification: str = Field(
        description="Two-sentence justification for Audience Signals score."
    )
    aesthetic_clusters: float = Field(ge=1, le=10)
    aesthetic_clusters_justification: str = Field(
        description="Two-sentence justification for Aesthetic Clusters score."
    )
    completeness: float = Field(ge=1, le=10)
    completeness_justification: str = Field(
        description="Two-sentence justification for Completeness score."
    )
    actionability: float = Field(ge=1, le=10)
    actionability_justification: str = Field(
        description="Two-sentence justification for Actionability score."
    )
    top_strengths: List[str] = Field(
        min_length=3,
        max_length=3,
        description="Top 3 strengths with evidence from the PDF.",
    )
    top_gaps: List[str] = Field(
        min_length=3,
        max_length=3,
        description="Top 3 gaps — what is missing or too vague.",
    )
    one_line_verdict: str = Field(
        description=(
            "Single sentence: would this Brand DNA report reliably guide AI design "
            "generation for this brand? Why or why not?"
        )
    )
    weighted_score: float = Field(default=0.0, ge=0, le=10)

    @model_validator(mode="after")
    def compute_weighted_score(self) -> "BrandDnaPdfBenchmarkScore":
        self.weighted_score = round(
            self.visual_identity * _SCORE_WEIGHTS["visual_identity"]
            + self.brand_voice * _SCORE_WEIGHTS["brand_voice"]
            + self.audience_signals * _SCORE_WEIGHTS["audience_signals"]
            + self.aesthetic_clusters * _SCORE_WEIGHTS["aesthetic_clusters"]
            + self.completeness * _SCORE_WEIGHTS["completeness"]
            + self.actionability * _SCORE_WEIGHTS["actionability"],
            2,
        )
        return self


class BenchmarkRunResult(BaseModel):
    brand_name: str
    model: str
    pdf_path: Optional[str] = None
    json_path: Optional[str] = None
    score: Optional[BrandDnaPdfBenchmarkScore] = None
    error: Optional[str] = None


def model_slug(model: str) -> str:
    return _SLUG_RE.sub("-", model.lower()).strip("-") or "model"


def fixture_path_for(brand_case: BrandIntegrationCase) -> Path:
    return FIXTURES_DIR / f"{brand_case.name}_brand_content_bundle.json"


def brand_config_from_case(brand_case: BrandIntegrationCase) -> BrandConfig:
    return BrandConfig(name=brand_case.name.title(), url=brand_case.brand_url)


def load_benchmark_bundle(brand_case: BrandIntegrationCase) -> BrandContentBundle:
    path = fixture_path_for(brand_case)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing benchmark fixture {path}. "
            f"Run capture_benchmark_fixture() or pytest with --benchmark-live-crawl."
        )
    return BrandContentBundle.model_validate_json(path.read_text(encoding="utf-8"))


def save_benchmark_bundle(
    brand_case: BrandIntegrationCase, bundle: BrandContentBundle
) -> Path:
    path = fixture_path_for(brand_case)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    return path


def trim_bundle(bundle: BrandContentBundle, max_images: int = BENCHMARK_MAX_IMAGES) -> BrandContentBundle:
    return bundle.model_copy(
        update={
            "images": bundle.images[:max_images],
            "texts": bundle.texts[: max(20, len(bundle.texts))],
        }
    )


def capture_benchmark_fixture(
    brand_case: BrandIntegrationCase,
    *,
    website_crawler,
    metadata_config_service,
    max_images: int = BENCHMARK_MAX_IMAGES,
) -> BrandContentBundle:
    """Live-crawl a brand and persist a trimmed bundle fixture."""
    from tests.integration_tests.integration_helpers import load_brand_extraction_config

    load_brand_extraction_config(metadata_config_service, brand_case.brand_url)
    bundle = website_crawler.crawl(brand_config_from_case(brand_case))
    if bundle.stopped_due_to_bot_protection:
        raise RuntimeError(
            f"Cannot capture fixture for {brand_case.name}: bot protection "
            f"{bundle.bot_protection.provider}/{bundle.bot_protection.reason}"
        )
    if not bundle.images:
        raise RuntimeError(f"Cannot capture fixture for {brand_case.name}: no images crawled")

    trimmed = trim_bundle(bundle, max_images=max_images)
    path = save_benchmark_bundle(brand_case, trimmed)
    logger.info("Wrote benchmark fixture %s (%s images)", path, len(trimmed.images))
    return trimmed


def bundle_summary_for_judge(bundle: BrandContentBundle) -> str:
    product_names = [
        img.product_name for img in bundle.images if img.product_name
    ][:30]
    snippet_lines = [
        f"- [{s.kind}] {s.text[:200]}"
        for s in bundle.texts[:15]
    ]
    return (
        f"Brand URL: {bundle.brand_url}\n"
        f"Images in source bundle: {len(bundle.images)}\n"
        f"Text snippets in source bundle: {len(bundle.texts)}\n"
        f"Sample product names:\n"
        + ("\n".join(f"  · {name}" for name in product_names) or "  (none)")
        + "\n\nSample text snippets:\n"
        + ("\n".join(snippet_lines) or "(none)")
    )


def build_judge_llm(gemini_api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=JUDGE_MODEL,
        google_api_key=gemini_api_key,
        temperature=0,
    )


def judge_brand_dna_pdf(
    judge_llm: ChatGoogleGenerativeAI,
    gemini_api_key: str,
    bundle: BrandContentBundle,
    pdf_path: Path,
    brand_case: BrandIntegrationCase,
) -> BrandDnaPdfBenchmarkScore:
    pdf_bytes = pdf_path.read_bytes()
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError(f"Not a valid PDF: {pdf_path}")

    structured_llm = judge_llm.with_structured_output(BrandDnaPdfBenchmarkScore)
    summary = bundle_summary_for_judge(bundle)
    brand_name = brand_case.name.title()
    evaluation_date = datetime.now(timezone.utc).date().isoformat()

    prompt_text = (
        f"Brand name: {brand_name}\n"
        f"PDF attached: {pdf_path.name}\n"
        f"Evaluation date: {evaluation_date}\n\n"
        "Critically evaluate the attached Brand DNA PDF across all six dimensions. "
        "Return the structured scorecard with per-dimension scores (1–10), two-sentence "
        "justifications, top 3 strengths, top 3 gaps, and a one-line verdict.\n\n"
        "SUPPLEMENTARY SOURCE MATERIAL SUMMARY (for grounding checks only — "
        "evaluate insight quality from the PDF itself):\n"
        f"{summary}"
    )

    uploaded = _upload_pdf_for_judge(gemini_api_key, pdf_path, pdf_bytes)
    try:
        content: List[dict] = [
            {"type": "text", "text": prompt_text},
            {
                "type": "file",
                "file_id": uploaded.uri,
                "mime_type": "application/pdf",
            },
        ]
        result = structured_llm.invoke(
            [
                SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=content),
            ]
        )
    finally:
        _delete_uploaded_file(gemini_api_key, uploaded.name)

    if isinstance(result, BrandDnaPdfBenchmarkScore):
        return result
    return BrandDnaPdfBenchmarkScore.model_validate(result)


def _upload_pdf_for_judge(api_key: str, pdf_path: Path, pdf_bytes: bytes):
    import io

    client = genai.Client(api_key=api_key)
    buffer = io.BytesIO(pdf_bytes)
    buffer.seek(0)
    uploaded = client.files.upload(
        file=buffer,
        config={"mime_type": "application/pdf", "display_name": pdf_path.name},
    )
    deadline = time.time() + 120
    while uploaded.state.name == "PROCESSING":
        if time.time() > deadline:
            raise TimeoutError(f"PDF upload still processing after 120s: {pdf_path}")
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"PDF upload failed with state {uploaded.state.name}")
    return uploaded


def _delete_uploaded_file(api_key: str, name: str) -> None:
    try:
        genai.Client(api_key=api_key).files.delete(name=name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not delete uploaded judge file %s: %s", name, exc)


def build_candidate_settings(base_settings: Dict[str, Any], model: str) -> Dict[str, Any]:
    settings = dict(base_settings)
    settings["llm_model"] = model
    settings["brand_dna_max_images"] = BENCHMARK_MAX_IMAGES
    return settings


def brand_output_dir(run_id: str, brand_case: BrandIntegrationCase) -> Path:
    return BENCHMARK_OUTPUT_ROOT / run_id / brand_case.name


def write_benchmark_results(
    run_dir: Path,
    brand_case: BrandIntegrationCase,
    results: List[BenchmarkRunResult],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        [r for r in results if r.score is not None],
        key=lambda r: r.score.weighted_score if r.score else 0,
        reverse=True,
    )
    payload = {
        "brand": brand_case.name,
        "brand_url": brand_case.brand_url,
        "judge_model": JUDGE_MODEL,
        "candidate_models": CANDIDATE_MODELS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": [r.model_dump() for r in results],
        "ranking_by_weighted_score": [
            {
                "model": r.model,
                "weighted_score": r.score.weighted_score if r.score else None,
            }
            for r in ranked
        ],
    }
    out_path = run_dir / "scores.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def build_llm_service(settings: Dict[str, Any]) -> LlmService:
    return LlmService(settings)
