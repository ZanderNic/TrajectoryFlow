<h1 align="center">TrajectoryFlow</h1>
<p align="center"><strong>Kinetically guided stochastic population forecasting from metabolic-labeling single-cell RNA</strong></p>
<p align="center">A research framework for learning future cell-state distributions from time-resolved SCI-FATE2 snapshots without assuming cell-to-cell correspondences across time.</p>
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-models-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20%2F%20experimental-6f42c1">
  <img alt="Data" src="https://img.shields.io/badge/data-SCI--FATE2-0A7E8C">
</p>

---

## Overview

**TrajectoryFlow** studies how cellular populations change over time using time-resolved single-cell RNA measurements.

The main model is a **dual-scale stochastic residual model**. It combines:

- a **state representation** from total RNA and NTR,
- a **local kinetic representation** learned from total RNA and supervised with newly synthesized RNA,
- a **one-step stochastic residual transition** in latent space,
- and **population-level Sliced-Wasserstein training** between generated and observed future cell populations.

The current implementation does **not** use the earlier latent-diffusion transition. The generative transition is now a direct stochastic residual MLP.

The central setting is difficult because single-cell measurements are destructive: cells measured at time $t$ are not the same cells measured at $t+\Delta t$. TrajectoryFlow therefore treats timepoints as **unpaired populations**, not as known cell lineages.

---

## Core idea

For a source cell at time $t$,

$$
z_t = E_{\mathrm{state}}(x_t, r_t)
$$

represents the current state, while

$$
k_t = E_{\mathrm{kin}}(x_t)
$$

represents local transcriptional information.

The future latent state is generated through a stochastic residual transition,

$$
\epsilon \sim \mathcal{N}(0,I),
$$

$$
\Delta z =
\Delta t\,
G_\theta(z_t,k_t,\Delta t,\epsilon),
$$

$$
z_{t+\Delta t}=z_t+\Delta z.
$$

The future total-RNA state is then decoded as

$$
\hat{x}_{t+\Delta t}=D(z_{t+\Delta t}).
$$

Training combines three objectives:

- **Reconstruction loss** preserves the current cell state.
- **Kinetic loss** makes the kinetic representation predictive of newly synthesized RNA.
- **Sliced-Wasserstein loss** matches generated and observed future populations without introducing artificial cell pairs.

```mermaid
flowchart LR
   X["Total RNA x_t"] --> S["State encoder"]
   R["NTR r_t"] --> S
   X --> K["Kinetic encoder"]
   S --> Z["State latent z_t"]
   K --> KT["Kinetic latent k_t"]
   KT --> AG["alpha / gamma"]
   AG --> N["Predicted new RNA"]
   N --> LK["Local kinetic loss"]
   Z --> G["Stochastic residual transition"]
   KT --> G
   DT["Delta t"] --> G
   E["epsilon ~ N(0,I)"] --> G
   G --> DZ["Delta z"]
   Z --> ADD["z_t + Delta z"]
   DZ --> ADD
   ADD --> D["State decoder"]
   D --> XF["Generated future population"]
   TARGET["Observed future population"] --> SW["Sliced-Wasserstein loss"]
   XF --> SW
```

---

## What is included

TrajectoryFlow currently provides:

- SCI-FATE2 download and preprocessing
- sparse timepoint-wise dataset storage
- total RNA, newly synthesized RNA and NTR handling
- a dual-scale stochastic residual model
- local kinetic supervision through effective production/degradation parameters
- population-level unpaired training
- a no-change baseline
- a TrajectoryFlow implementation of VelvetVAE + VelvetSDE
- reproducible TOML-based benchmark definitions
- temporal holdout and extrapolation experiments
- multi-seed benchmarking
- population-level evaluation metrics
- benchmark aggregation and ranking
- paper-ready benchmark plots
- dual-scale training/checkpoint diagnostics
- optional velocity/direction-reference utilities
- tests for data, models, training, evaluation and experiment infrastructure

The default population metrics include:

- Sliced-Wasserstein distance
- Maximum Mean Discrepancy (MMD)
- Chamfer distance
- centroid distance
- mean-expression error
- variance error
- mean and variance correlations
- dispersion error

---

## What the project does **not** assume

TrajectoryFlow intentionally does **not** assume that a cell measured at one timepoint has a known matching cell at a later timepoint.

It also does not claim to:

- reconstruct ground-truth individual cell lineages,
- uniquely identify biological production and degradation rates from one labeling interval,
- use an explicitly integrated velocity field in the proposed dual-scale model,
- use latent diffusion in the current proposed model,
- treat generated trajectories as experimentally observed trajectories.

Generated states should be interpreted as **plausible stochastic futures that are consistent with observed population dynamics**.

---

# Quick start

## 1. Clone and create an environment

```bash
git clone <YOUR-REPOSITORY-URL>
cd TrajectoryFlow
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

For development and tests, if the repository defines the development extra:

```bash
pip install -e ".[dev]"
```

TrajectoryFlow has been developed with Python 3.12.

A CUDA-capable GPU is recommended for the larger benchmark runs, but the framework also supports CPU execution through the benchmark configuration.

---

# SCI-FATE2 dataset

TrajectoryFlow is built around the SCI-FATE2 dataset from Maizels et al.

- **GEO accession:** [GSE236512](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE236512)
- **Paper:** [Reconstructing developmental trajectories using latent dynamical systems and time-resolved transcriptomics](https://doi.org/10.1016/j.cels.2024.04.004)

The repository downloads the data directly from GEO. The preprocessing script supports the published `estimate`, `counting`, and `splicing` H5AD files.

## Download and preprocess

The default setup uses the `estimate` dataset and keeps the 10,000 most frequently detected genes:

```bash
python scripts/download_scifate2.py
```

This is equivalent to:

```bash
python scripts/download_scifate2.py \
   --dataset estimate \
   --work-dir data/raw \
   --output-dir data/processed/scifate2 \
   --top-genes-by-detection 10000
```

To additionally reconstruct and merge the authors' published cell annotations:

```bash
python scripts/download_scifate2.py \
   --dataset estimate \
   --authors-cell-types
```

If disk space is limited, the downloaded H5AD can be removed after preprocessing:

```bash
python scripts/download_scifate2.py \
   --dataset estimate \
   --delete-h5ad
```

To preprocess an already downloaded H5AD instead:

```bash
python scripts/download_scifate2.py \
   --h5ad-path /path/to/file.h5ad
```

Useful options include:

```text
--min-cells
--min-gene-nonzero-fraction
--top-genes-by-detection
--clip-ratio
--authors-cell-types
--cell-annotations
--uncompressed-npz
--force-download
--force-process
```

Run

```bash
python scripts/download_scifate2.py --help
```

for the full list.

## Processed data layout

The preprocessing step stores every timepoint independently and keeps the matrices sparse:

```text
data/processed/scifate2/
├── manifest.json
├── preprocessing.json
├── selected_gene_indices.npy
├── genes.*
├── 5h/
│   ├── expression.npz
│   ├── new.npz
│   ├── ntr.npz
│   └── obs.*
├── 10h/
│   └── ...
├── 15h/
│   └── ...
└── ...
```

The same selected genes and gene ordering are used across all timepoints.

`expression.npz`, `new.npz`, and `ntr.npz` are stored as SciPy CSR matrices and only sampled rows are materialized densely during training. This is important for keeping the full dataset manageable in memory.

---

# Running the benchmarks

Benchmarks are defined in TOML files and executed with:

```bash
python scripts/run_benchmark.py path/to/config.toml
```

The benchmark runner records the config, dataset fingerprints, code hashes, runtime information, metrics and per-run artifacts.

Useful flags:

```bash
# Start from a clean output directory
python scripts/run_benchmark.py path/to/config.toml --overwrite
# Resume completed experiments when config/data/code still match
python scripts/run_benchmark.py path/to/config.toml --resume
```

---

## Main benchmark

The main benchmark compares:

1. **No Change** — predicts the source population as the future population
2. **Velvet** — VelvetVAE + VelvetSDE baseline
3. **Dual Scale Full** — the proposed stochastic residual model

The main strict extrapolation experiment trains on:

```text
5h, 10h, 15h
```

and evaluates:

```text
15h -> 20h
```

with the 20h population completely hidden during training.

Recommended config location:

```text
configs/benchmark_main_stochastic_residual.toml
```

Run:

```bash
python scripts/run_benchmark.py \
   configs/benchmark_main_stochastic_residual.toml \
   --overwrite
```

---

## Kinetic ablation study

The ablation study uses a $2\times2$ design:

| Model | Kinetic encoder | Local kinetic loss |
|---|---:|---:|
| `dual_scale_no_kin_no_local` | no | no |
| `dual_scale_kin_no_local` | yes | no |
| `dual_scale_no_kin_local` | no | yes |
| `dual_scale_full` | yes | yes |

This separates the effect of the cell-specific kinetic representation from the effect of direct local supervision with newly synthesized RNA.

Recommended config location:

```text
configs/benchmark_dual_scale_kinetic_ablation_stochastic.toml
```

Run:

```bash
python scripts/run_benchmark.py \
   configs/benchmark_dual_scale_kinetic_ablation_stochastic.toml \
   --overwrite
```

> **Important:** NTR is already an input to the state encoder. The ablation therefore measures the additional contribution of the separately supervised kinetic branch, not the total contribution of metabolic-labeling information.

---

# Benchmark outputs

A benchmark run produces an output directory such as:

```text
runs/main_benchmark_15h_to_20h/
├── benchmark_config.toml
├── benchmark_manifest.json
├── metrics.csv
├── runtimes.csv
├── experiments.csv
├── no_change/
├── velvet/
└── dual_scale_full/
```

Individual runs are organized by model, split and seed:

```text
runs/<benchmark>/
└── <model>/
   └── <split>/
       └── seed_<N>/
           ├── result.json
           ├── metrics.csv
           ├── runtime.csv
           ├── predictions/
           └── ...
```

Checkpoints and predictions are controlled through the benchmark config:

```toml
save_checkpoints = true
save_predictions = true
```

---

# Analyze a completed benchmark

After a benchmark has finished:

```bash
python scripts/analyze_benchmark.py \
   runs/main_benchmark_15h_to_20h
```

The analysis is written to:

```text
runs/main_benchmark_15h_to_20h/analysis/
├── metric_summary.csv
├── runtime_summary.csv
├── experiment_summary.csv
├── rankings.csv
└── paired_comparisons.csv
```

The analysis aggregates seeds, computes mean/std/median values, creates metric-wise rankings, and performs paired seed comparisons between models.

---

# Create figures

## Paper-ready benchmark figures

```bash
python scripts/plot_benchmark.py \
   runs/main_benchmark_15h_to_20h
```

Optional arguments:

```bash
python scripts/plot_benchmark.py \
   runs/main_benchmark_15h_to_20h \
   --phase test \
   --seed 0 \
   --prediction-sample 0 \
   --max-cells 3000
```

The plotting pipeline can create metric comparisons, forecast visualizations, efficiency summaries, cell-type composition plots and velocity/direction figures when the required data are available.

## Dual-scale diagnostics

For training curves and checkpoint diagnostics:

```bash
python scripts/plot_dual_scale.py \
   runs/dual_scale_kinetic_ablation
```

To inspect one split/seed:

```bash
python scripts/plot_dual_scale.py \
   runs/dual_scale_kinetic_ablation \
   --split extrapolation_15h_to_20h \
   --seed 0
```

Training-log plots only:

```bash
python scripts/plot_dual_scale.py \
   runs/dual_scale_kinetic_ablation \
   --training-only
```

---

# Optional direction references

TrajectoryFlow also contains utilities for velocity/direction-based evaluation.

These are primarily relevant for methods that expose a velocity prediction and are not required for the main population-forecasting experiment.

Generate direction references:

```bash
python scripts/prepare_direction_references.py --help
```

Generate a CBD reference:

```bash
python scripts/prepare_cbd_reference.py --help
```

The SCI-FATE2 authors' cell-type transitions can be included through the corresponding command-line options.

---

# Velvet baseline

TrajectoryFlow contains its own PyTorch implementation/adaptation of the Velvet components used for benchmarking.

The benchmark implementation is kept inside:

```text
src/trajectoryflow/models/baselines/velvet/
```

It does not require the official `velvetvae` package for normal TrajectoryFlow benchmark execution.

The official Velvet repository is:

[github.com/rorymaizels/velvetVAE](https://github.com/rorymaizels/velvetVAE)

The included validator can compare core numerical components against a pinned official reference environment:

```bash
python scripts/validate_velvet_equivalence.py
```

For TrajectoryFlow-only regression checks:

```bash
python scripts/validate_velvet_equivalence.py --self-only
```

This validation does not imply that independently trained models, complete training trajectories, or reported paper scores are bitwise-identical to the original implementation.

---

# Project structure

```text
TrajectoryFlow/
├── src/
│   └── trajectoryflow/
│       ├── data/
│       │   ├── scifate2.py
│       │   ├── scifate2_annotations.py
│       │   ├── store.py
│       │   └── ...
│       ├── models/
│       │   ├── dual_scale/
│       │   │   ├── components.py
│       │   │   ├── config.py
│       │   │   ├── losses.py
│       │   │   └── model.py
│       │   └── baselines/
│       │       ├── no_change.py
│       │       └── velvet/
│       ├── training/
│       │   ├── dual_scale.py
│       │   ├── dual_scale_data.py
│       │   ├── velvet.py
│       │   └── progress.py
│       ├── evaluation/
│       │   ├── evaluator.py
│       │   └── metrics/
│       ├── experiment/
│       │   ├── config.py
│       │   ├── data.py
│       │   ├── models.py
│       │   ├── runner.py
│       │   ├── runtime.py
│       │   └── split.py
│       └── plotting/
│           ├── paper.py
│           ├── dual_scale.py
│           └── ...
├── scripts/
│   ├── download_scifate2.py
│   ├── run_benchmark.py
│   ├── analyze_benchmark.py
│   ├── plot_benchmark.py
│   ├── plot_dual_scale.py
│   ├── prepare_direction_references.py
│   ├── prepare_cbd_reference.py
│   └── validate_velvet_equivalence.py
├── configs/
│   ├── benchmark_main_stochastic_residual.toml
│   └── benchmark_dual_scale_kinetic_ablation_stochastic.toml
├── tests/
├── data/
└── runs/
```

---

# Reproducibility

The benchmark framework is designed to make model comparisons reproducible.

Each benchmark records:

- the benchmark configuration,
- enabled models,
- seeds,
- data manifest and preprocessing hashes,
- selected-gene indices,
- code hashes,
- system information,
- split fingerprints,
- runtimes,
- evaluation metrics,
- saved predictions/checkpoints when enabled.

The default paper experiments use:

```toml
seeds = [0, 1, 2]
[runtime]
device = "auto"
deterministic = true
precision = "float32"
```

Use `--resume` only when you want completed runs to be reused if their configuration, data, references, split and code still match.

---

# Testing

Run the full test suite with:

```bash
pytest -q
```

Individual groups can also be run separately:

```bash
pytest tests/models -q
pytest tests/training -q
pytest tests/evaluation -q
pytest tests/experiment -q
pytest tests/data -q
pytest tests/scripts -q
```

Some optional plotting or reference-generation functionality may require additional scientific Python packages beyond the minimal core environment.

---

# Research scope and limitations

This repository is research code and the proposed model is experimental.

The current study focuses on SCI-FATE2 and population-level forecasting. In particular:

- cross-time supervision is distributional rather than cell-paired,
- the model does not recover experimentally observed individual trajectories,
- stochastic predictions are not lineage assignments,
- the learned $\alpha$ and $\gamma$ values are effective kinetic parameters rather than uniquely identified biological rates,
- the current main evaluation is a strict early-time extrapolation experiment,
- generalization to other biological systems requires additional datasets and experiments.

---

# References

**SCI-FATE2 / Velvet**

R. Maizels et al., **Reconstructing developmental trajectories using latent dynamical systems and time-resolved transcriptomics**, Cell Systems, 2024.

https://doi.org/10.1016/j.cels.2024.04.004

Dataset:

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE236512

**Official Velvet implementation**

https://github.com/rorymaizels/velvetVAE

---

# AI assistance

OpenAI ChatGPT (GPT-5.6 Sol) was used as an AI-assisted tool during this project. It supported parts of the software implementation, including code generation, refactoring, debugging, and documentation. ChatGPT was also used for language editing and for rewriting or reformulating parts of the accompanying manuscript to improve clarity and scientific presentation.

All methodological decisions, experimental design, interpretation of results, verification of sources, and final project content were reviewed and determined by the author.
