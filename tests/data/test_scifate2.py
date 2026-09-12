# std-lib imports

# 3 party imports
import numpy as np
import pandas as pd
import pytest

# package imports
from trajectoryflow.data import scifate2


def test_timepoint_normalization():
    assert scifate2.normalize_timepoint_label("05h") == "5h"
    assert scifate2.normalize_timepoint_label("D7") == "168h"
    assert scifate2.normalize_timepoint_label("1.5d") == "36h"
    with pytest.raises(ValueError):
        scifate2.normalize_timepoint_label("week2")


def test_merge_cell_annotations_fills_missing_without_overwrite(tmp_path):
    obs = pd.DataFrame({"cell_id": ["a", "b"], "cell_type": ["existing", None]})
    path = tmp_path / "annotations.csv"
    pd.DataFrame({"cell_id": ["a", "b"], "cell_type": ["existing", "new"], "batch": [1, 2]}).to_csv(path, index=False)
    merged = scifate2.merge_cell_annotations(obs, path)
    assert merged["cell_type"].tolist() == ["existing", "new"]
    assert merged["batch"].tolist() == [1, 2]


def test_merge_cell_annotations_rejects_conflicts_and_nonmatching_ids(tmp_path):
    obs = pd.DataFrame({"cell_id": ["a"], "cell_type": ["x"]})
    conflict = tmp_path / "conflict.csv"
    pd.DataFrame({"cell_id": ["a"], "cell_type": ["y"]}).to_csv(conflict, index=False)
    with pytest.raises(ValueError, match="conflict"):
        scifate2.merge_cell_annotations(obs, conflict)

    missing = tmp_path / "missing.csv"
    pd.DataFrame({"cell_id": ["other"], "cell_type": ["y"]}).to_csv(missing, index=False)
    with pytest.raises(ValueError, match="No annotation cell IDs"):
        scifate2.merge_cell_annotations(obs, missing)


def test_merge_cell_annotations_promotes_authors_cell_annotation(tmp_path):
    obs = pd.DataFrame({"cell_id": ["a", "b", "c"]})
    path = tmp_path / "authors.csv"
    pd.DataFrame({
        "cell_id": ["a", "b"],
        "cell_annotation": ["NMP", "Mesoderm"],
        "marker_based_classification": ["NMP", "Mesoderm"],
    }).to_csv(path, index=False)

    merged = scifate2.merge_cell_annotations(obs, path)
    assert merged["cell_annotation"].iloc[:2].tolist() == ["NMP", "Mesoderm"]
    assert merged["cell_type"].iloc[:2].tolist() == ["NMP", "Mesoderm"]
    assert pd.isna(merged.loc[2, "cell_type"])
    assert merged["marker_based_classification"].iloc[:2].tolist() == ["NMP", "Mesoderm"]


def test_merge_cell_annotations_can_select_cell_type_column(tmp_path):
    obs = pd.DataFrame({"cell_id": ["a", "b"]})
    path = tmp_path / "authors.csv"
    pd.DataFrame({"cell_id": ["a", "b"], "final_label": ["pMN", "MN"]}).to_csv(path, index=False)
    merged = scifate2.merge_cell_annotations(obs, path, cell_type_column="final_label")
    assert merged["cell_type"].tolist() == ["pMN", "MN"]
    assert merged["final_label"].tolist() == ["pMN", "MN"]


def test_read_h5ad_annotations_uses_obs_names_when_cell_id_missing(tmp_path, monkeypatch):
    class FakeFile:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeAdata:
        def __init__(self):
            self.obs = pd.DataFrame({"cell_annotation": ["NMP", "Neural"]})
            self.obs_names = pd.Index(["cell-a", "cell-b"])
            self.file = FakeFile()

    fake = FakeAdata()

    class FakeAnndata:
        @staticmethod
        def read_h5ad(path, backed=None):
            assert backed == "r"
            return fake

    import sys
    monkeypatch.setitem(sys.modules, "anndata", FakeAnndata)
    table = scifate2._read_annotation_table(tmp_path / "published.h5ad", "cell_id")
    assert table["cell_id"].tolist() == ["cell-a", "cell-b"]
    assert table["cell_annotation"].tolist() == ["NMP", "Neural"]
    assert fake.file.closed


def test_force_process_failure_keeps_previous_dataset(tmp_path, monkeypatch):
    output = tmp_path / "processed"
    output.mkdir()
    old_manifest = output / "manifest.json"
    old_manifest.write_text('{"old": true}', encoding="utf-8")
    h5ad = tmp_path / "source.h5ad"
    h5ad.touch()

    def fail(**kwargs):
        raise RuntimeError("processing failed")

    monkeypatch.setattr(scifate2, "write_processed_dataset", fail)
    with pytest.raises(RuntimeError, match="processing failed"):
        scifate2.prepare_scifate2(
            dataset="estimate", work_dir=tmp_path / "raw", output_dir=output, h5ad_path=h5ad,
            activation_layer="total", new_layer="new", min_cells=0, min_gene_nonzero_fraction=0.0,
            top_genes_by_detection=0, force_download=False, force_process=True, delete_h5ad=False,
            clip_ratio=True, timepoint_column="time", gene_id_column=None, compressed_npz=True,
        )
    assert old_manifest.read_text(encoding="utf-8") == '{"old": true}'
    assert not (tmp_path / ".processed.staging").exists()


class _Reader:
    def __init__(self, rows):
        self.rows = rows

    def get_row(self, row):
        return self.rows[row]


def test_prepare_row_outputs_keeps_selected_genes_and_computes_ntr():
    total = _Reader({0: (np.array([0, 2, 3]), np.array([10.0, 4.0, 2.0], dtype=np.float32))})
    new = _Reader({0: (np.array([0, 2, 3]), np.array([2.0, 8.0, 1.0], dtype=np.float32))})
    gene_map = np.array([1, -1, 0, 2])
    expression_cols, expression, new_cols, new_values, ntr_cols, ntr = scifate2.prepare_row_outputs(0, total, new, gene_map, True)
    assert expression_cols.tolist() == [0, 1, 2]
    assert expression.tolist() == pytest.approx([4.0, 10.0, 2.0])
    assert new_cols.tolist() == [0, 1, 2]
    assert new_values.tolist() == pytest.approx([8.0, 2.0, 1.0])
    assert ntr_cols.tolist() == [0, 1, 2]
    assert ntr.tolist() == pytest.approx([1.0, 0.2, 0.5])


def test_prepare_row_outputs_rejects_negative_rna():
    total = _Reader({0: (np.array([0]), np.array([-1.0], dtype=np.float32))})
    new = _Reader({0: (np.array([0]), np.array([0.0], dtype=np.float32))})
    with pytest.raises(ValueError, match="Negative RNA"):
        scifate2.prepare_row_outputs(0, total, new, np.array([0]), True)


def test_force_download_failure_keeps_existing_h5ad(tmp_path, monkeypatch):
    output = tmp_path / "source.h5ad"
    output.write_bytes(b"old-data")

    def fail(*args, **kwargs):
        raise RuntimeError("network failed")

    monkeypatch.setattr(scifate2.requests, "get", fail)
    with pytest.raises(RuntimeError, match="network failed"):
        scifate2.download_and_decompress_gzip("https://example.invalid/data.gz", output, force=True)
    assert output.read_bytes() == b"old-data"
    assert not (tmp_path / "source.h5ad.tmp").exists()


def test_prepare_row_outputs_rejects_nonfinite_rna():
    total = _Reader({0: (np.array([0]), np.array([np.nan], dtype=np.float32))})
    new = _Reader({0: (np.array([0]), np.array([0.0], dtype=np.float32))})
    with pytest.raises(ValueError, match="Non-finite RNA"):
        scifate2.prepare_row_outputs(0, total, new, np.array([0]), True)


def test_merge_authors_annotations_marks_unmatched_cells_as_not_qc_pass(tmp_path):
    obs = pd.DataFrame({"cell_id": ["a", "b", "c"]})
    path = tmp_path / "authors.csv"
    pd.DataFrame({
        "cell_id": ["a", "b"], "cell_annotation": ["NMP", "Neural"], "authors_qc_pass": [True, True]
    }).to_csv(path, index=False)
    merged = scifate2.merge_cell_annotations(obs, path)
    assert merged["authors_qc_pass"].tolist() == [True, True, False]


def test_prepare_scifate2_reconstructs_authors_annotations_from_counting(tmp_path, monkeypatch):
    estimate = tmp_path / "estimate.h5ad"
    counting = tmp_path / "counting.h5ad"
    estimate.touch()
    counting.touch()
    cache = tmp_path / "raw" / "GSE236512_authors_cell_annotations.csv"
    captured = {}

    def fake_annotations(counting_h5ad, cache_csv, force=False, strict=True):
        assert counting_h5ad == counting
        assert cache_csv == cache
        assert force is True
        assert strict is True
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        cache_csv.write_text("cell_id,cell_annotation\na,NMP\n", encoding="utf-8")
        return cache_csv

    def fake_write(**kwargs):
        captured.update(kwargs)
        kwargs["output_dir"].mkdir(parents=True)
        (kwargs["output_dir"] / "manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(scifate2, "load_or_reconstruct_authors_cell_annotations", fake_annotations)
    monkeypatch.setattr(scifate2, "write_processed_dataset", fake_write)
    scifate2.prepare_scifate2(
        dataset="estimate", work_dir=tmp_path / "raw", output_dir=tmp_path / "processed", h5ad_path=estimate,
        activation_layer="total", new_layer="new_estimated", min_cells=0, min_gene_nonzero_fraction=0.0,
        top_genes_by_detection=0, force_download=False, force_process=True, delete_h5ad=False, clip_ratio=True,
        timepoint_column="timepoint", gene_id_column=None, compressed_npz=True, authors_cell_types=True,
        authors_counting_h5ad=counting, force_authors_cell_types=True,
    )
    assert captured["cell_annotations"] == cache
    assert captured["annotation_cell_id_column"] == "cell_id"
    assert captured["annotation_cell_type_column"] == "cell_annotation"


def test_prepare_scifate2_rejects_two_annotation_sources(tmp_path):
    with pytest.raises(ValueError, match="either --authors-cell-types or --cell-annotations"):
        scifate2.prepare_scifate2(
            dataset="estimate", work_dir=tmp_path / "raw", output_dir=tmp_path / "processed", h5ad_path=tmp_path / "x.h5ad",
            activation_layer="total", new_layer="new_estimated", min_cells=0, min_gene_nonzero_fraction=0.0,
            top_genes_by_detection=0, force_download=False, force_process=True, delete_h5ad=False, clip_ratio=True,
            timepoint_column="timepoint", gene_id_column=None, compressed_npz=True, cell_annotations=tmp_path / "a.csv",
            authors_cell_types=True,
        )
