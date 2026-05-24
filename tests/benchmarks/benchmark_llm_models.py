"""Benchmark Brand DNA PDF quality across Gemini models.

Opt-in only — requires --run-benchmark and --run-llm-integration.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from configs.test import test_config
from src.services.brand_dna_service import BrandDnaService
from src.services.http_service import HttpClient
from tests.benchmarks.benchmark_support import (
    BENCHMARK_BRAND_CASES,
    CANDIDATE_MODELS,
    BenchmarkRunResult,
    brand_config_from_case,
    brand_output_dir,
    build_candidate_settings,
    build_judge_llm,
    build_llm_service,
    capture_benchmark_fixture,
    judge_brand_dna_pdf,
    load_benchmark_bundle,
    model_slug,
    write_benchmark_results,
)
from tests.integration_tests.integration_helpers import BrandIntegrationCase


def pytest_generate_tests(metafunc):
    if "brand_case" not in metafunc.fixturenames:
        return

    selected = metafunc.config.getoption("--benchmark-brand")
    cases = BENCHMARK_BRAND_CASES
    if selected:
        cases = tuple(c for c in cases if c.name == selected)
        if not cases:
            raise pytest.UsageError(
                f"--benchmark-brand {selected!r} is not a crawlable BRAND_CASES entry "
                f"(choices: {[c.name for c in BENCHMARK_BRAND_CASES]})"
            )
    metafunc.parametrize("brand_case", cases, ids=lambda c: c.name)


@pytest.fixture(scope="session")
def benchmark_website_crawler(integration_http_client, integration_settings, metadata_config_service):
    from src.crawlers.website_crawler import WebsiteCrawler

    crawl_settings = {
        **integration_settings,
        "lowest_processeable_product_count": 10,
    }
    return WebsiteCrawler(
        integration_http_client,
        crawl_settings,
        metadata_config_service=metadata_config_service,
    )


@pytest.fixture
def benchmark_bundle(
    brand_case: BrandIntegrationCase,
    request,
):
    if request.config.getoption("--benchmark-live-crawl"):
        benchmark_website_crawler = request.getfixturevalue("benchmark_website_crawler")
        metadata_config_service = request.getfixturevalue("metadata_config_service")
        return capture_benchmark_fixture(
            brand_case,
            website_crawler=benchmark_website_crawler,
            metadata_config_service=metadata_config_service,
        )
    try:
        return load_benchmark_bundle(brand_case)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.mark.benchmark
@pytest.mark.llm_integration
def test_benchmark_brand_dna_models(
    brand_case: BrandIntegrationCase,
    benchmark_bundle,
    benchmark_run_id: str,
    gemini_settings,
):
    if brand_case.expect_bot_protection:
        pytest.skip(f"{brand_case.name} is blocked by bot protection")

    brand_config = brand_config_from_case(brand_case)
    run_dir = brand_output_dir(benchmark_run_id, brand_case)
    base_settings = {**test_config, **gemini_settings}

    judge_llm = build_judge_llm(gemini_settings["gemini_api_key"])
    results: list[BenchmarkRunResult] = []

    http_client = HttpClient()
    executor = ThreadPoolExecutor(max_workers=test_config["worker_count"])
    try:
        for model in CANDIDATE_MODELS:
            model_dir = run_dir / model_slug(model)
            model_dir.mkdir(parents=True, exist_ok=True)
            try:
                settings = build_candidate_settings(base_settings, model)
                llm_service = build_llm_service(settings)
                brand_dna_service = BrandDnaService(
                    settings,
                    http_client,
                    llm_service,
                    executor=executor,
                )
                _report, pdf_path = brand_dna_service.run(
                    brand_config,
                    benchmark_bundle,
                    output_dir=model_dir,
                )
                json_candidates = list(model_dir.glob("*_brand_dna.json"))
                json_path = str(json_candidates[0]) if json_candidates else None
                score = judge_brand_dna_pdf(
                    judge_llm,
                    gemini_settings["gemini_api_key"],
                    benchmark_bundle,
                    pdf_path,
                    brand_case,
                )
                results.append(
                    BenchmarkRunResult(
                        brand_name=brand_case.name,
                        model=model,
                        pdf_path=str(pdf_path),
                        json_path=json_path,
                        score=score,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    BenchmarkRunResult(
                        brand_name=brand_case.name,
                        model=model,
                        error=str(exc),
                    )
                )
    finally:
        executor.shutdown(wait=True)
        http_client.close()

    scores_path = write_benchmark_results(run_dir, brand_case, results)
    failures = [r for r in results if r.error or r.score is None]
    assert not failures, (
        f"Benchmark failures for {brand_case.name}; see {scores_path}: "
        + "; ".join(f"{r.model}: {r.error}" for r in failures)
    )

    for result in results:
        assert result.score is not None
        assert Path(result.pdf_path).is_file()
