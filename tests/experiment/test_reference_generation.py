# std-lib imports
from pathlib import Path

# 3 party imports
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ad = pytest.importorskip("anndata")

# package imports
from trajectoryflow.experiment.evaluation import load_velocity_reference
from trajectoryflow.experiment.reference_generation import (
    TIMEPOINT_COLUMN,
    build_cbd_reference,
    choose_scfates_root,
    library_log1p_sparse,
    principal_tree_reference_vectors,
    transition_matrix_reference_vectors,
)


def graph_adata() -> ad.AnnData:
    adata = ad.AnnData(
        X=np.ones((4, 3), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "cell_id": ["a", "b", "c", "d"],
                TIMEPOINT_COLUMN: ["5h", "5h", "10h", "10h"],
            },
            index=["a", "b", "c", "d"],
        ),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    adata.obsm["X_pca"] = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [1.1, 0.0],
            [1.8, 0.0],
        ]
    )
    adata.varm["PCs"] = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    adata.uns["pca_mean"] = np.zeros(3)
    adata.uns["trajectoryflow_pca_library_size"] = 10_000.0
    adata.uns["graph"] = {
        "F": np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ]
        ),
        "B": np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ]
        ),
    }
    adata.obsm["X_R"] = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return adata


def test_library_log1p_sparse_normalizes_rows():
    x = sparse.csr_matrix(
        np.array(
            [
                [1.0, 1.0],
                [2.0, 0.0],
            ]
        )
    )

    transformed = library_log1p_sparse(x, library_size=10.0).toarray()

    np.testing.assert_allclose(
        transformed,
        np.log1p(
            np.array(
                [
                    [5.0, 5.0],
                    [10.0, 0.0],
                ]
            )
        ),
    )


def test_root_is_principal_point_nearest_earliest_population():
    adata = graph_adata()

    assert choose_scfates_root(adata) == 0


def test_principal_tree_vectors_point_away_from_root():
    adata = graph_adata()

    vectors = principal_tree_reference_vectors(adata, root=0)

    np.testing.assert_allclose(vectors[0], [1.0, 0.0])
    np.testing.assert_allclose(vectors[1], [1.0, 0.0])
    np.testing.assert_allclose(vectors[2], [1.0, 0.0])
    np.testing.assert_allclose(vectors[3], [0.0, 0.0])


def test_cellrank_reference_is_expected_future_displacement():
    transition = sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 1.0],
        ]
    )

    vectors = transition_matrix_reference_vectors(transition, positions)

    np.testing.assert_allclose(
        vectors,
        np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ]
        ),
    )


def test_cbd_builds_mean_unit_target_neighbor_direction(tmp_path: Path):
    adata = graph_adata()
    adata.obs["cell_type"] = ["A", "A", "B", "B"]
    output = tmp_path / "cbd.npz"

    build_cbd_reference(
        adata,
        output=output,
        cluster_column="cell_type",
        transitions=[("A", "B")],
        n_neighbors=3,
        min_target_neighbors=1,
    )

    reference = load_velocity_reference(output)

    assert reference.alignment_mode == "mean_unit_direction"
    assert set(reference.groups.tolist()) == {"A->B"}
    assert len(reference.cell_ids) >= 1
    assert np.isfinite(reference.vectors).all()
    assert (reference.vectors[:, 0] > 0).all()
