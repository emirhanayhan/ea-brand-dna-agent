from tests.fixtures.mocks import MagicMock

from src.models.brand_dna import (
    BatchClusterAssignment,
    ClusterObservation,
    ClusterTheme,
)
from src.services.llm_service import (
    LlmService,
    _chunk_for_assignment,
    _init_assignment_accumulator,
    _log_unassigned_images,
    _merge_assignments,
    _sample_for_discovery,
    _themes_to_observations,
)


def _img(i: int):
    return (f"meta-{i}", b"bytes")


def test_sample_for_discovery_even_spacing():
    images = [_img(i) for i in range(60)]
    sample, global_indices = _sample_for_discovery(images, sample_size=20)

    assert len(sample) == 20
    assert global_indices[0] == 0
    assert global_indices[-1] == 59
    assert global_indices == sorted(set(global_indices))
    assert len(global_indices) == 20


def test_sample_for_discovery_fewer_than_sample_size():
    images = [_img(i) for i in range(8)]
    sample, global_indices = _sample_for_discovery(images, sample_size=20)

    assert len(sample) == 8
    assert global_indices == list(range(8))


def test_sample_for_discovery_single_image():
    images = [_img(0)]
    sample, global_indices = _sample_for_discovery(images, sample_size=20)

    assert len(sample) == 1
    assert global_indices == [0]


def test_chunk_for_assignment():
    images = [_img(i) for i in range(25)]
    batches = _chunk_for_assignment(images, batch_size=12)

    assert len(batches) == 3
    assert [len(batch) for batch in batches] == [12, 12, 1]


def test_merge_assignments_offsets_indices_and_dedupes():
    themes = [
        ClusterTheme(
            cluster_label="Theme A",
            description="A",
            key_elements=["a"],
        ),
        ClusterTheme(
            cluster_label="Theme B",
            description="B",
            key_elements=["b"],
        ),
    ]
    accumulator = _init_assignment_accumulator(themes)
    assigned: set[int] = set()

    _merge_assignments(
        accumulator=accumulator,
        themes=themes,
        batch_assignments=[
            ClusterObservation(
                cluster_label="Theme A",
                description="ignored",
                image_indices=[1, 2],
            ),
            ClusterObservation(
                cluster_label="Theme B",
                description="ignored",
                image_indices=[2, 3],
            ),
        ],
        batch_start=12,
        batch_len=12,
        assigned_globally=assigned,
    )

    assert accumulator["Theme A"] == [13, 14]
    assert accumulator["Theme B"] == [15]
    assert assigned == {13, 14, 15}


def test_themes_to_observations_preserves_theme_metadata():
    themes = [
        ClusterTheme(
            cluster_label="Youth Football Kits",
            description="Youth kits",
            key_elements=["jerseys"],
        )
    ]
    observations = _themes_to_observations(
        themes,
        {"Youth Football Kits": [1, 4, 9]},
    )

    assert len(observations) == 1
    assert observations[0].cluster_label == "Youth Football Kits"
    assert observations[0].description == "Youth kits"
    assert observations[0].key_elements == ["jerseys"]
    assert observations[0].image_indices == [1, 4, 9]


def test_log_unassigned_images_does_not_mutate_clusters():
    """Replaces the old _assign_unassigned_images behavior. Unassigned images
    must NOT be silently force-merged into the smallest cluster — they are
    surfaced as a separate pool the caller can use as backfill.
    """
    observations = [
        ClusterObservation(
            cluster_label="A",
            description="a",
            image_indices=[1, 2, 3],
        ),
        ClusterObservation(
            cluster_label="B",
            description="b",
            image_indices=[4],
        ),
    ]
    _log_unassigned_images([5, 6])

    assert observations[0].image_indices == [1, 2, 3]
    assert observations[1].image_indices == [4]


def _make_llm_service() -> LlmService:
    service = LlmService.__new__(LlmService)
    service.llm = MagicMock()
    service._discovery_sample_size = 20
    service._assignment_batch_size = 12
    return service


def test_invoke_assignment_batch_retries_once_on_failure():
    service = _make_llm_service()
    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = [
        RuntimeError("transient"),
        BatchClusterAssignment(
            assignments=[
                ClusterObservation(
                    cluster_label="Theme A",
                    description="ok",
                    image_indices=[1],
                )
            ]
        ),
    ]

    result = service._invoke_assignment_batch(
        structured_llm=structured_llm,
        content=[{"type": "text", "text": "..."}],
        batch_index=0,
        batch_count=1,
    )

    assert result is not None
    assert structured_llm.invoke.call_count == 2
    assert result.assignments[0].image_indices == [1]


def test_invoke_assignment_batch_returns_none_after_two_failures():
    service = _make_llm_service()
    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = [
        RuntimeError("transient 1"),
        RuntimeError("transient 2"),
    ]

    result = service._invoke_assignment_batch(
        structured_llm=structured_llm,
        content=[{"type": "text", "text": "..."}],
        batch_index=2,
        batch_count=3,
    )

    assert result is None
    assert structured_llm.invoke.call_count == 2
