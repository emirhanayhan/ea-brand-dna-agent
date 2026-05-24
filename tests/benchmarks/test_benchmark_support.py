from tests.benchmarks.benchmark_support import (
    BENCHMARK_BRAND_CASES,
    BrandDnaPdfBenchmarkScore,
    brand_config_from_case,
    load_benchmark_bundle,
    model_slug,
    trim_bundle,
)
from tests.integration_tests.integration_helpers import BRAND_CASES


def test_benchmark_brand_cases_exclude_bot_protection():
    blocked = {case.name for case in BRAND_CASES if case.expect_bot_protection}
    benchmark_names = {case.name for case in BENCHMARK_BRAND_CASES}
    assert blocked.isdisjoint(benchmark_names)
    assert benchmark_names == {"nike", "prada", "allbirds"}


def test_model_slug():
    assert model_slug("gemini-2.5-flash") == "gemini-2-5-flash"


def test_load_nike_benchmark_fixture():
    nike = next(case for case in BENCHMARK_BRAND_CASES if case.name == "nike")
    bundle = load_benchmark_bundle(nike)
    assert bundle.source == "website"
    assert len(bundle.images) > 0
    assert str(bundle.brand_url).startswith("https://www.nike.com")


def test_trim_bundle_caps_images():
    nike = next(case for case in BENCHMARK_BRAND_CASES if case.name == "nike")
    bundle = load_benchmark_bundle(nike)
    trimmed = trim_bundle(bundle, max_images=5)
    assert len(trimmed.images) == 5


def test_brand_config_from_case():
    nike = next(case for case in BENCHMARK_BRAND_CASES if case.name == "nike")
    config = brand_config_from_case(nike)
    assert config.name == "Nike"
    assert str(config.url) == nike.brand_url


def test_weighted_score_calculation():
    score = BrandDnaPdfBenchmarkScore(
        visual_identity=8.0,
        visual_identity_justification="Strong palette specificity.",
        brand_voice=7.0,
        brand_voice_justification="Voice is grounded in copy.",
        audience_signals=6.0,
        audience_signals_justification="Audience cues are present but thin.",
        aesthetic_clusters=8.0,
        aesthetic_clusters_justification="Clusters are distinct and named well.",
        completeness=7.0,
        completeness_justification="All core sections are present.",
        actionability=6.0,
        actionability_justification="Useful but missing some design constraints.",
        top_strengths=["a", "b", "c"],
        top_gaps=["x", "y", "z"],
        one_line_verdict="Mostly actionable with some gaps.",
    )
    assert score.weighted_score == 7.2
