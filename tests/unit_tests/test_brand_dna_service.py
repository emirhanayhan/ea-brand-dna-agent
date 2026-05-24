from datetime import datetime, timezone
from io import BytesIO
from tests.fixtures.mocks import MagicMock, patch

import respx
from PIL import Image

from configs.test import test_config
from src.models.brand_configs import BrandConfig
from src.models.brand_content import BrandContentBundle
from src.models.brand_dna import (
    BrandDnaSynthesis,
    BrandVisualAnalysis,
    BrandVoice,
    ClusterObservation,
)
from src.models.image_outputs import DiscoveredImage
from src.services.brand_dna_service import BrandDnaService, _resolve_clusters
from src.services.http_service import HttpClient


def _image_bytes(width: int, height: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


def _discovered(url: str, width: int, height: int) -> DiscoveredImage:
    return DiscoveredImage(
        url=url,
        width=width,
        height=height,
        source_page="https://example.com/product",
    )


def _bundle(images: list[DiscoveredImage]) -> BrandContentBundle:
    return BrandContentBundle(
        brand_url="https://example.com",
        min_resolution="512x512",
        images=images,
        pages_crawled=1,
        candidates_checked=len(images),
        created_at=datetime.now(timezone.utc),
    )


def _minimal_synthesis() -> BrandDnaSynthesis:
    return BrandDnaSynthesis(
        positioning_statement="Example is the category for people who test.",
        brand_essence="Example brand essence.",
        brand_voice=BrandVoice(tone="Direct"),
        visual_identity_summary="Example visual summary.",
        audience={},
    )


@patch("src.services.brand_dna_service.render_brand_dna_pdf")
@patch("src.services.brand_dna_service.extract_palette", return_value=[])
@respx.mock
def test_run_dedupes_near_duplicate_images_before_llm(
    mock_palette,
    mock_render,
    tmp_path,
):
    payload = _image_bytes(512, 512)
    images = [
        _discovered("https://cdn.example.com/t_default/hero.jpg", 512, 512),
        _discovered("https://cdn.example.com/t_PDP_1280_v1/hero.jpg", 1280, 1280),
        _discovered("https://cdn.example.com/t_product_v1/hero.jpg", 1024, 1024),
    ]
    for image in images:
        respx.get(str(image.url)).respond(200, content=payload)

    llm_service = MagicMock()
    llm_service.analyze_brand_visuals.return_value = BrandVisualAnalysis()
    llm_service.synthesize_brand_dna.return_value = _minimal_synthesis()
    mock_render.return_value = tmp_path / "report.pdf"

    settings = {**test_config, "brand_dna_output_dir": str(tmp_path)}
    client = HttpClient()
    try:
        service = BrandDnaService(settings, client, llm_service)
        service.run(
            BrandConfig(url="https://example.com", name="Example"),
            _bundle(images),
            output_dir=tmp_path,
        )
    finally:
        client.close()

    images_passed = llm_service.analyze_brand_visuals.call_args.kwargs["images"]
    assert len(images_passed) == 1
    assert (
        str(images_passed[0][0].url)
        == "https://cdn.example.com/t_PDP_1280_v1/hero.jpg"
    )


def _img_at(n: int, source_kind: str = "product") -> DiscoveredImage:
    return DiscoveredImage(
        url=f"https://cdn.example.com/img-{n}.jpg",
        width=800,
        height=800,
        source_page="https://example.com/product",
        source_kind=source_kind,
    )


def _obs(label: str, indices: list[int]) -> ClusterObservation:
    return ClusterObservation(
        cluster_label=label,
        description=f"{label} description",
        key_elements=["x"],
        image_indices=indices,
    )


def test_resolve_clusters_backfills_thin_clusters_from_unassigned_pool():
    images = [_img_at(i) for i in range(1, 11)]  # 10 images
    visual = BrandVisualAnalysis(
        cluster_observations=[
            _obs("Full", [1, 2, 3]),     # already has 3 reps
            _obs("Thin", [4]),           # has 1 — should be backfilled to 2
            _obs("Empty", []),           # has 0 — should be backfilled to 2
        ],
        unassigned_image_indices=[5, 6, 7, 8, 9, 10],
    )

    clusters = _resolve_clusters(
        visual,
        images,
        min_images_per_cluster=2,
        max_images_per_cluster=3,
        min_cluster_count=3,
    )

    assert [c.label for c in clusters] == ["Full", "Thin", "Empty"]
    assert len(clusters[0].image_urls) == 3
    assert len(clusters[1].image_urls) >= 2
    assert len(clusters[2].image_urls) >= 2
    # No image URL appears in more than one cluster.
    all_urls = [u for c in clusters for u in c.image_urls]
    assert len(all_urls) == len(set(all_urls))


def test_resolve_clusters_drops_empty_cluster_when_above_min_count():
    images = [_img_at(i) for i in range(1, 8)]  # 7 images
    visual = BrandVisualAnalysis(
        cluster_observations=[
            _obs("A", [1, 2, 3]),
            _obs("B", [4, 5]),
            _obs("C", [6, 7]),
            _obs("Empty", []),   # nothing assigned, no backfill pool either
        ],
        unassigned_image_indices=[],
    )

    clusters = _resolve_clusters(
        visual,
        images,
        min_images_per_cluster=2,
        max_images_per_cluster=3,
        min_cluster_count=3,
    )

    # 4 -> 3 because Empty was dropped (we still satisfy min_cluster_count=3).
    assert [c.label for c in clusters] == ["A", "B", "C"]


def test_resolve_clusters_prefers_product_over_editorial_images():
    # Cluster cites 4 image indices: editorial(1), product(2), social(3),
    # product(4). With max_images_per_cluster=2 we want both products to win.
    images = [
        _img_at(1, source_kind="editorial"),
        _img_at(2, source_kind="product"),
        _img_at(3, source_kind="social"),
        _img_at(4, source_kind="product"),
    ]
    visual = BrandVisualAnalysis(
        cluster_observations=[
            _obs("A", [1, 2, 3, 4]),
            _obs("B", [1, 2, 3, 4]),  # second cluster, also from same pool
            _obs("C", [1, 2, 3, 4]),
        ],
        unassigned_image_indices=[],
    )

    clusters = _resolve_clusters(
        visual,
        images,
        min_images_per_cluster=1,
        max_images_per_cluster=2,
        min_cluster_count=3,
    )

    # Cluster A picks both product images first (img-2 then img-4).
    assert clusters[0].image_urls == [
        "https://cdn.example.com/img-2.jpg",
        "https://cdn.example.com/img-4.jpg",
    ]
    # Cluster B then takes the next-best non-claimed: editorial (img-1) then social (img-3).
    assert clusters[1].image_urls == [
        "https://cdn.example.com/img-1.jpg",
        "https://cdn.example.com/img-3.jpg",
    ]
    # Cluster C has nothing left and stays empty (but is kept since dropping
    # would push us below min_cluster_count=3).
    assert clusters[2].image_urls == []


def test_resolve_clusters_keeps_empty_cluster_when_dropping_would_violate_min():
    images = [_img_at(i) for i in range(1, 5)]
    visual = BrandVisualAnalysis(
        cluster_observations=[
            _obs("A", [1, 2]),
            _obs("B", [3, 4]),
            _obs("Empty", []),  # dropping would leave only 2 clusters
        ],
        unassigned_image_indices=[],
    )

    clusters = _resolve_clusters(
        visual,
        images,
        min_images_per_cluster=2,
        max_images_per_cluster=3,
        min_cluster_count=3,
    )

    # All three kept — empty cluster renders without an image strip rather
    # than violating the case-study floor of 3 clusters.
    assert [c.label for c in clusters] == ["A", "B", "Empty"]
    assert clusters[2].image_urls == []
