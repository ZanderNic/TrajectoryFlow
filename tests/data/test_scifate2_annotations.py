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


def test_authors_cluster_mapping_is_complete_and_matches_key_labels():
    assert set(annotations.AUTHORS_CLUSTER_ANNOTATIONS) == {str(i) for i in range(33)}
    assert annotations.AUTHORS_CLUSTER_ANNOTATIONS["1"] == "NMP"
    assert annotations.AUTHORS_CLUSTER_ANNOTATIONS["4"] == "Early_Neural"
    assert annotations.AUTHORS_CLUSTER_ANNOTATIONS["14"] == "FP"
    assert annotations.AUTHORS_CLUSTER_ANNOTATIONS["25"] == "MN"


def test_map_authors_leiden_rejects_unknown_cluster():
    with pytest.raises(RuntimeError, match="does not match"):
        annotations.map_authors_leiden(pd.Series(["0", "33"]))


def test_map_authors_leiden_maps_published_clusters():
    mapped = annotations.map_authors_leiden(pd.Series(["0", "1", "13", "17"]))
    assert mapped.tolist() == ["Mesoderm", "NMP", "V3", "MN"]
