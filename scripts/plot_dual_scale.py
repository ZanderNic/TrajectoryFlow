#!/usr/bin/env python3

# std-lib imports
import argparse
import json
from pathlib import Path

# 3 party imports
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

# package imports
from trajectoryflow.data.store import ScifateStore
from trajectoryflow.experiment.config import load_benchmark_config
from trajectoryflow.experiment.split import timepoint_hours
from trajectoryflow.models.dual_scale import DualScaleModelConfig, build_default_dual_scale_model
from trajectoryflow.plotting.dual_scale import (
    GLOBAL_METRICS,
    LOCAL_METRICS,
    plot_alignment_cosine,
    plot_cell_kinetics,
    plot_gene_baselines,
    plot_gene_mean_agreement,
    plot_latent_transition,
    plot_loss_group,
    plot_past_reconstruction,
    plot_total_loss_history,
    plot_transition_uncertainty,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training and checkpoint diagnostics for dual_scale benchmark runs."
    )
    parser.add_argument("benchmark_dir", type=Path, help="Benchmark output directory containing benchmark_config.toml.")
    parser.add_argument("--split", type=str, default=None, help="Only plot one split.")
    parser.add_argument("--seed", type=int, default=None, help="Only plot one seed.")
    parser.add_argument("--max-cells", type=int, default=512, help="Maximum cells used for checkpoint diagnostics.")
    parser.add_argument("--smooth-window", type=int, default=20, help="Rolling window for live-step loss curves.")
    parser.add_argument("--device", type=str, default="cpu", help="Device used for checkpoint diagnostics.")
    parser.add_argument("--training-only", action="store_true", help="Only generate plots from saved training logs.")
    return parser.parse_args()


def _close(fig) -> None:
    if fig is not None:
        import matplotlib.pyplot as plt
        plt.close(fig)


def _model_config(config):
    matches = [model for model in config.models if model.name == "dual_scale" and model.enabled]
    if not matches:
        raise ValueError("benchmark config does not contain an enabled dual_scale model.")
    return matches[0]


def _run_dirs(root: Path, split: str | None, seed: int | None) -> list[Path]:
    base = root / "dual_scale"
    if not base.exists():
        raise FileNotFoundError(f"No dual_scale benchmark output found at {base}.")
    paths = sorted(base.glob("*/seed_*"))
    if split is not None:
        paths = [path for path in paths if path.parent.name == split]
    if seed is not None:
        paths = [path for path in paths if path.name == f"seed_{seed}"]
    return [path for path in paths if path.is_dir()]


def _training_plots(run_dir: Path, output: Path, smooth_window: int) -> None:
    path = run_dir / "training" / "training_metrics.csv"
    if not path.exists():
        return
    metrics = pd.read_csv(path)
    if metrics.empty:
        return
    _close(plot_total_loss_history(metrics, output / "training_total_loss.png", smooth_window))
    _close(plot_loss_group(metrics, LOCAL_METRICS, "local", "Local kinetic losses", output / "training_local_losses.png", smooth_window))
    _close(plot_loss_group(metrics, GLOBAL_METRICS, "global", "Global population losses", output / "training_global_losses.png", smooth_window))


def _split_train_indices(run_dir: Path, timepoint: str) -> np.ndarray:
    safe = timepoint.replace(".", "_")
    path = run_dir / "split_indices.npz"
    with np.load(path) as values:
        key = f"{safe}__train"
        if key not in values:
            raise KeyError(f"Missing {key!r} in {path}.")
        return np.asarray(values[key], dtype=np.int64)


def _sample_snapshot(store: ScifateStore, run_dir: Path, timepoint: str, n: int, seed: int):
    snapshot = store.load(timepoint)
    indices = _split_train_indices(run_dir, timepoint)
    if len(indices) > n:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=n, replace=False))
    total = snapshot.expression[indices].toarray().astype("float32", copy=False)
    new = snapshot.new[indices].toarray().astype("float32", copy=False)
    ntr = snapshot.ntr[indices].toarray().astype("float32", copy=False)
    obs = snapshot.obs.iloc[indices].reset_index(drop=True)
    return indices, total, new, ntr, obs


def _load_model(root: Path, run_dir: Path, device: torch.device):
    config = load_benchmark_config(root / "benchmark_config.toml")
    model_config = _model_config(config)
    store = ScifateStore(config.data_root, cache_size=0)
    params = dict(model_config.model_params)
    from dataclasses import replace
    dual_config = replace(DualScaleModelConfig(), **params)
    model = build_default_dual_scale_model(store.n_genes, dual_config).to(device)
    try:
        state = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(run_dir / "checkpoint.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    trainer_params = dict(model_config.trainer_params)
    labeling_time = float(trainer_params.get("labeling_time", 2.0))
    return config, store, model, labeling_time


def _base_tables(model, gene_names: list[str]) -> pd.DataFrame:
    device = next(model.parameters()).device
    kinetic_dim = model.kinetic_encoder.kinetic_dim
    zero = torch.zeros((1, kinetic_dim), device=device)
    with torch.inference_mode():
        _, alpha_base = model.production_head(zero)
        _, gamma_base = model.degradation_head(zero)
    return pd.DataFrame(
        {
            "gene": gene_names,
            "production_base": alpha_base.detach().cpu().numpy(),
            "degradation_base": gamma_base.detach().cpu().numpy(),
        }
    )


def _checkpoint_plots(root: Path, run_dir: Path, output: Path, max_cells: int, device: torch.device) -> None:
    if not (run_dir / "checkpoint.pt").exists():
        return
    config, store, model, labeling_time = _load_model(root, run_dir, device)
    split_info = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    fit = sorted(split_info["fit_timepoints"], key=timepoint_hours)
    if not fit:
        return
    seed = int(run_dir.name.removeprefix("seed_"))

    source_time = fit[-2] if len(fit) >= 2 else fit[-1]
    target_time = fit[-1]
    earlier = [tp for tp in fit if timepoint_hours(tp) < timepoint_hours(source_time)]
    past_time = earlier[-1] if earlier else None
    _, total_np, new_np, ntr_np, obs = _sample_snapshot(store, run_dir, source_time, max_cells, seed)
    _, target_np, _, _, _ = _sample_snapshot(store, run_dir, target_time, max_cells, seed + 1)

    total = torch.from_numpy(total_np).to(device)
    new = torch.from_numpy(new_np).to(device)
    ntr = torch.from_numpy(ntr_np).to(device)
    old = (total - new).clamp_min(0)
    delta_time = timepoint_hours(target_time) - timepoint_hours(source_time)

    with torch.inference_mode():
        local = model.forward_local(total, old, new, ntr, labeling_time)
        global_out = model.forward_global(
            total=total,
            old=old,
            new=new,
            ntr=ntr,
            delta_time=delta_time,
            labeling_time=labeling_time,
            n_samples=8,
            deterministic=False,
        )
        target_latent = model.encode_state(torch.from_numpy(target_np).to(device))

    genes = _base_tables(model, store.gene_names)
    genes.to_csv(output / "gene_kinetics.csv", index=False)
    _close(plot_gene_baselines(genes, output / "kinetic_gene_baselines.png"))
    _close(plot_gene_mean_agreement(total_np, local.reconstruction.cpu().numpy(), "Total-RNA reconstruction", "Observed log1p gene mean", "Reconstructed log1p gene mean", output / "reconstruction_gene_means.png"))
    _close(plot_gene_mean_agreement(new_np, local.predicted_new.cpu().numpy(), "New-RNA kinetic prediction", "Observed log1p gene mean", "Predicted log1p gene mean", output / "new_rna_gene_means.png"))

    mean_norm = global_out.transition_mean.norm(dim=-1)
    std_norm = global_out.transition_std.norm(dim=-1)
    kinetic_norm = global_out.kinetic_direction_latent.norm(dim=-1)
    transition_sample_norm = global_out.sampled_delta.mean(dim=0).norm(dim=-1)
    alignment = F.cosine_similarity(
        global_out.kinetic_direction_latent,
        global_out.transition_mean,
        dim=-1,
    )
    cells = pd.DataFrame(
        {
            "source_timepoint": source_time,
            "cell_index": np.arange(len(total_np)),
            "alpha_mean": local.alpha.mean(dim=-1).cpu().numpy(),
            "gamma_mean": local.gamma.mean(dim=-1).cpu().numpy(),
            "transition_mean_norm": mean_norm.cpu().numpy(),
            "transition_std_norm": std_norm.cpu().numpy(),
            "kinetic_direction_norm": kinetic_norm.cpu().numpy(),
            "transition_sample_mean_norm": transition_sample_norm.cpu().numpy(),
            "alignment_cosine": alignment.cpu().numpy(),
        }
    )
    if "cell_id" in obs.columns:
        cells.insert(1, "cell_id", obs["cell_id"].astype(str))
    cells.to_csv(output / "cell_diagnostics.csv", index=False)
    _close(plot_cell_kinetics(cells, output / "cell_kinetics.png"))
    _close(plot_transition_uncertainty(cells, output / "transition_uncertainty.png"))
    _close(plot_alignment_cosine(cells, output / "alignment_cosine.png"))
    _close(
        plot_latent_transition(
            global_out.latent.cpu().numpy(),
            target_latent.cpu().numpy(),
            global_out.future_latent[0].cpu().numpy(),
            output / "latent_future_transition.png",
            title=f"Latent transition {source_time} → {target_time}",
        )
    )

    if past_time is not None:
        _, past_np, _, _, _ = _sample_snapshot(store, run_dir, past_time, max_cells, seed + 2)
        with torch.inference_mode():
            real_past_latent = model.encode_state(torch.from_numpy(past_np).to(device))
            reconstructed_past_latent = model.encode_state(global_out.reconstructed_past)
        _close(
            plot_past_reconstruction(
                real_past_latent.cpu().numpy(),
                reconstructed_past_latent.cpu().numpy(),
                output / "latent_past_reconstruction.png",
                title=f"Past reconstruction {source_time} → {past_time}",
            )
        )

    summary = {
        "source_timepoint": source_time,
        "target_timepoint": target_time,
        "past_timepoint": past_time,
        "n_cells": len(total_np),
        "labeling_time": labeling_time,
        "delta_time": delta_time,
        "device": str(device),
    }
    (output / "diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_cells < 2:
        raise ValueError("--max-cells must be >= 2.")
    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be >= 1.")

    root = args.benchmark_dir.resolve()
    runs = _run_dirs(root, args.split, args.seed)
    if not runs:
        raise FileNotFoundError("No matching dual_scale runs were found.")
    device = torch.device(args.device)

    for run_dir in runs:
        output = run_dir / "model_figures"
        output.mkdir(parents=True, exist_ok=True)
        _training_plots(run_dir, output, args.smooth_window)
        if not args.training_only:
            _checkpoint_plots(root, run_dir, output, args.max_cells, device)
        print(f"Dual-scale figures saved to {output}")


if __name__ == "__main__":
    main()
