# Velvet for TrajectoryFlow

Copy the files under `src/` into the matching locations in your TrajectoryFlow
repository.

Required data layout per timepoint:

```text
expression.npz   # total RNA
new.npz          # new/labeled RNA
ntr.npz          # optional for Velvet, used by TrajectoryFlow
obs.parquet
```

`TimepointData` must expose:

```python
timepoint: str
expression: sparse.csr_matrix
new: sparse.csr_matrix
ntr: sparse.csr_matrix
obs: pd.DataFrame
```

and `ScifateStore.load()` must load the `new` path from `manifest.json`.

## Dependencies

The implementation uses your existing PyTorch/NumPy/SciPy stack plus
scikit-learn for KNN/SVD helpers.

Add `scikit-learn` as a direct dependency if it is not already declared.

There is no `scvi-tools`, Lightning, velvetvae or torchsde dependency.

## Smoke test

After copying the files:

```bash
python scripts/run_velvet_smoke.py --timepoints 5h 10h --n-cells 500
```

The smoke test intentionally trains only one epoch per stage.

## Full training

Use the default configurations:

```python
vae_config = VelvetVAEConfig()
sde_config = VelvetSDEConfig()
```

which retain the main paper-level training defaults. For practical GPU memory,
you may set `VelvetVAEConfig(batch_size=...)`; the paper itself used the entire
dataset as one batch.

Read `REFERENCES.md` before treating benchmark numbers as an exact reproduction
of the original software.
