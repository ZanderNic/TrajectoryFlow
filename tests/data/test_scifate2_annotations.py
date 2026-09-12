# std-lib imports

# 3 party imports
import numpy as np
import pandas as pd
import pytest

# package imports
from trajectoryflow.data import scifate2_annotations as annotations


def test_authors_qc_mask_matches_published_percentile_rules():
    obs = pd.DataFrame({
        "total_counts": np.arange(10, dtype=float),
        "n_genes_by_counts": np.arange(10, dtype=float),
        "label_rate": np.arange(10, dtype=float),
        "pct_counts_mt": np.arange(10, dtype=float),
    })
    mask = annotations.authors_qc_mask(obs)
    expected = (
        (obs.total_counts > np.percentile(obs.total_counts, 10))
        & (obs.total_counts < np.percentile(obs.total_counts, 90))
        & (obs.n_genes_by_counts > np.percentile(obs.n_genes_by_counts, 10))
        & (obs.n_genes_by_counts < np.percentile(obs.n_genes_by_counts, 90))
        & (obs.label_rate > np.percentile(obs.label_rate, 10))
        & (obs.pct_counts_mt < np.percentile(obs.pct_counts_mt, 90))
    )
    assert mask.tolist() == expected.tolist()


def test_authors_knowledge_matrix_shape_and_key_templates():
    assert len(annotations.AUTHORS_KNOWLEDGE_GENES) == 28
    assert len(annotations.AUTHORS_KNOWLEDGE_TEMPLATES) == 22
    assert annotations.AUTHORS_KNOWLEDGE_TEMPLATES[0] == ("NMP", "NMP", ("Sox2", "T", "Fgf8", "Lef1"))
    assert annotations.AUTHORS_KNOWLEDGE_TEMPLATES[1] == ("Neural", "Early_Neural", ("Sox2", "Nkx1-2"))
    assert annotations.AUTHORS_KNOWLEDGE_TEMPLATES[2] == ("Neural", "Neural", ("Sox2", "Pax6", "Irx3"))


def _marker_row(*positive):
    return np.asarray([int(gene in positive) for gene in annotations.AUTHORS_KNOWLEDGE_GENES], dtype=np.float32)


def test_marker_classification_reproduces_authors_patterns_and_refines_neural_state():
    matrix = np.vstack([
        _marker_row("Sox2", "T", "Fgf8", "Lef1"),
        _marker_row("Sox2", "Nkx1-2"),
        _marker_row("Sox2", "Pax6", "Irx3"),
        _marker_row(),
    ])
    result = annotations.classify_authors_markers(matrix)
    assert result["marker_based_classification"].tolist() == ["NMP", "Neural", "Neural", "null_cell"]
    assert result["marker_based_cell_type"].tolist() == ["NMP", "Early_Neural", "Neural", "other"]
    assert result["marker_distance"].tolist() == [0, 0, 0, 0]


def test_cluster_consensus_uses_current_clusters_and_rejects_ambiguous_groups():
    leiden = pd.Series(["8", "8", "8", "8", "2", "2", "2", "2", "11", "11", "11", "11"])
    labels = pd.Series(["NMP", "NMP", "NMP", "other", "pMN", "pMN", "p3", "p3", "Early_Neural", "Early_Neural", "Early_Neural", "Neural"])
    result = annotations.cluster_consensus_cell_types(leiden, labels)
    assert result["cell_type"].tolist() == ["NMP"] * 4 + ["other"] * 4 + ["Early_Neural"] * 4
    assert result.loc[0, "cell_type_support"] == pytest.approx(0.75)
    assert result.loc[4, "cell_type_margin"] == pytest.approx(0.0)


def _write_csr_layer(h5, name, matrix):
    matrix = matrix.tocsr()
    group = h5.require_group("layers").create_group(name)
    group.attrs["encoding-type"] = "csr_matrix"
    group.attrs["shape"] = matrix.shape
    group.create_dataset("data", data=matrix.data)
    group.create_dataset("indices", data=matrix.indices)
    group.create_dataset("indptr", data=matrix.indptr)


def test_backed_reader_collects_only_requested_rows_and_columns(tmp_path):
    import h5py
    from scipy import sparse

    matrix = sparse.csr_matrix(np.arange(30, dtype=np.float32).reshape(6, 5))
    path = tmp_path / "layers.h5"
    with h5py.File(path, "w") as h5:
        _write_csr_layer(h5, "total", matrix)
    with h5py.File(path, "r") as h5:
        reader = annotations._BackedLayerReader(h5, "total")
        keep = np.array([False, True, False, True, True, False])
        selected = annotations._collect_rows(reader, keep, chunk_size=2, columns=np.array([1, 4]))
    np.testing.assert_array_equal(selected.toarray(), matrix[keep][:, [1, 4]].toarray())


def test_stream_qc_metrics_matches_dense_calculation(tmp_path):
    import h5py
    from scipy import sparse

    total = sparse.csr_matrix(np.array([[1, 0, 2], [0, 3, 1], [4, 0, 0]], dtype=np.float32))
    new = sparse.csr_matrix(np.array([[1, 0, 1], [0, 1, 0], [2, 0, 0]], dtype=np.float32))
    path = tmp_path / "layers.h5"
    with h5py.File(path, "w") as h5:
        _write_csr_layer(h5, "total", total)
        _write_csr_layer(h5, "new", new)
    with h5py.File(path, "r") as h5:
        metrics = annotations._stream_qc_metrics(
            annotations._BackedLayerReader(h5, "total"), annotations._BackedLayerReader(h5, "new"),
            np.array([False, False, True]), chunk_size=2,
        )
    total_counts, n_genes, pct_mt, label_rate = metrics
    np.testing.assert_allclose(total_counts, [3, 4, 4])
    np.testing.assert_array_equal(n_genes, [2, 2, 1])
    np.testing.assert_allclose(pct_mt, [200 / 3, 25, 0])
    np.testing.assert_allclose(label_rate, [2 / 3, 1 / 4, 1 / 2])


def test_selected_gene_normalization_uses_full_library_size():
    from scipy import sparse

    selected_counts = sparse.csr_matrix(np.array([[2, 3], [1, 0]], dtype=np.float32))
    normalized = annotations._normalize_log_selected(selected_counts, np.array([10.0, 4.0]))
    expected = np.log1p(np.array([[2000.0, 3000.0], [2500.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(normalized.toarray(), expected, rtol=1e-6)


def test_annotation_chunk_size_validation():
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="chunk_size"):
            annotations._validate_chunk_size(bad)
