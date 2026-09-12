# std-lib imports

# 3 party imports
import numpy as np
import pytest

# package imports
from trajectoryflow.experiment.evaluation import (
    VelocityReference,
    load_velocity_reference,
    project_velocity_to_reference,
    save_velocity_reference,
    velocity_alignment_metrics,
)


def reference():
    return VelocityReference(
        name="pts",
        cell_ids=np.array(["a", "b", "c"]),
        vectors=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        pca_components=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        pca_mean=np.zeros(3),
        positions=np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        ),
        groups=np.array(["A_to_B", "A_to_B", "B_to_C"]),
        genes=np.array(["g1", "g2", "g3"]),
    )


def test_reference_roundtrip(tmp_path):
    path = tmp_path / "pts.npz"
    save_velocity_reference(path, reference())
    loaded = load_velocity_reference(path)

    assert loaded.name == "pts"
    assert loaded.n_components == 2
    assert loaded.n_genes == 3
    np.testing.assert_array_equal(
        loaded.cell_ids,
        ["a", "b", "c"],
    )
    np.testing.assert_allclose(
        loaded.vectors,
        reference().vectors,
    )


def test_reference_cell_matching():
    query, ref = reference().indices_for(
        np.array(["c", "missing", "a"])
    )

    np.testing.assert_array_equal(query, [0, 2])
    np.testing.assert_array_equal(ref, [2, 0])


def test_linear_projection():
    expression = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ]
    )
    velocity = np.array(
        [
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
        ]
    )

    positions, projected = project_velocity_to_reference(
        expression,
        velocity,
        reference(),
    )

    np.testing.assert_allclose(
        positions,
        expression[:, :2],
    )
    np.testing.assert_allclose(
        projected,
        [[1.0, 0.0], [0.0, 1.0]],
    )


def test_velocity_alignment_metrics_perfect_alignment():
    ref = reference()
    metrics, cells = velocity_alignment_metrics(
        predicted=ref.vectors,
        reference=ref.vectors,
        cell_ids=ref.cell_ids,
        groups=ref.groups,
    )

    overall = metrics[
        (metrics["metric"] == "cosine_similarity")
        & metrics["group"].isna()
    ].iloc[0]

    assert overall["value"] == 1.0
    assert cells["valid"].all()
    np.testing.assert_allclose(
        cells["cosine_similarity"],
        1.0,
    )

    group_names = set(
        metrics["group"].dropna()
    )
    assert group_names == {"A_to_B", "B_to_C"}


def test_zero_velocity_is_invalid_not_fake_alignment():
    ref = reference()
    metrics, cells = velocity_alignment_metrics(
        predicted=np.zeros_like(ref.vectors),
        reference=ref.vectors,
        cell_ids=ref.cell_ids,
    )

    overall = metrics[
        (metrics["metric"] == "valid_fraction")
        & metrics["group"].isna()
    ].iloc[0]

    assert overall["value"] == 0.0
    assert cells["cosine_similarity"].isna().all()


def test_cbd_mean_unit_direction_matches_mean_neighbor_cosine():
    predicted = np.array(
        [
            [2.0, 0.0],
            [0.0, 3.0],
        ]
    )

    # Row 1 represents mean([1, 0], [0, 1]) = [0.5, 0.5].
    # Its CBD score for velocity [1, 0] is therefore 0.5.
    # Row 2 represents mean([0, 1], [1, 0]) = [0.5, 0.5].
    # Its CBD score for velocity [0, 1] is also 0.5.
    cbd_reference = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )

    metrics, cells = velocity_alignment_metrics(
        predicted=predicted,
        reference=cbd_reference,
        cell_ids=np.array(["a", "b"]),
        groups=np.array(["A->B", "A->B"]),
        alignment_mode="mean_unit_direction",
    )

    np.testing.assert_allclose(
        cells["cosine_similarity"],
        [0.5, 0.5],
    )

    overall = metrics[
        (metrics["metric"] == "cosine_similarity")
        & metrics["group"].isna()
    ].iloc[0]

    assert overall["value"] == 0.5


def test_reference_matching_keeps_duplicate_cbd_rows():
    ref = VelocityReference(
        name="cbd",
        cell_ids=np.array(["a", "a", "b"]),
        vectors=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        pca_components=np.eye(2),
        pca_mean=np.zeros(2),
        groups=np.array(["A->B", "A->C", "B->C"]),
        alignment_mode="mean_unit_direction",
    )

    query, reference_indices = ref.indices_for(np.array(["a", "b"]))

    np.testing.assert_array_equal(query, [0, 0, 1])
    np.testing.assert_array_equal(reference_indices, [0, 1, 2])


def test_reference_rejects_empty_pca_or_gene_names():
    with pytest.raises(ValueError, match="non-zero shape"):
        VelocityReference(name="bad", cell_ids=np.array(["a"]), vectors=np.empty((1, 0)), pca_components=np.empty((0, 2)), pca_mean=np.zeros(2))

    with pytest.raises(ValueError, match="non-empty strings"):
        VelocityReference(name="bad", cell_ids=np.array(["a"]), vectors=np.ones((1, 1)), pca_components=np.ones((1, 2)), pca_mean=np.zeros(2), genes=np.array(["g1", ""]))


def test_velocity_alignment_rejects_nonpositive_epsilon():
    values = np.ones((1, 2))
    with pytest.raises(ValueError, match="eps must be > 0"):
        velocity_alignment_metrics(values, values, np.array(["a"]), eps=0.0)
