# std-lib imports
import argparse
import json
from pathlib import Path

# 3 party imports
import matplotlib.pyplot as plt
import numpy as np
import torch

# package imports
from trajectoryflow.data.loader import make_timepoint_loader
from trajectoryflow.data.store import ScifateStore, TimepointData
from trajectoryflow.evaluation.default import make_default_evaluator
from trajectoryflow.experiments.runner import ExperimentRunner
from trajectoryflow.models.baselines.no_change import NoChangeBaseline
from trajectoryflow.plotting.embedding import UmapProjector
from trajectoryflow.plotting.umap import (
    plot_joint_prediction_overlay,
    plot_prediction_comparison,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def sample_population(
    data: TimepointData,
    n_cells: int,
    normalize_expression: bool = True,
) -> torch.Tensor:
    batch_size = min(n_cells, len(data))

    loader = make_timepoint_loader(
        data=data,
        batch_size=batch_size,
        shuffle=True,
        normalize_expression=normalize_expression,
    )

    batch = next(iter(loader))
    return batch["expression"]


def save_results(report, output_dir: Path, config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    report.to_frame().to_csv(output_dir / "metrics.csv", index=False)

    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def save_plots(
    source: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    source_timepoint: str,
    target_timepoint: str,
    output_dir: Path,
    seed: int,
) -> None:
    source_np = source.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    prediction_np = prediction.detach().cpu().numpy()

    # Fit the embedding only on observed data.
    real_population = np.vstack([source_np, target_np])

    projector = UmapProjector(
        n_pca_components=50,
        n_neighbors=30,
        min_dist=0.3,
        random_state=seed,
    )

    projector.fit(real_population)

    source_umap = projector.transform(source_np)
    target_umap = projector.transform(target_np)
    prediction_umap = projector.transform(prediction_np)

    # ------------------------------------------------------------------
    # Source / target / prediction
    # ------------------------------------------------------------------

    fig, _ = plot_prediction_comparison(
        source_coordinates=source_umap,
        target_coordinates=target_umap,
        prediction_coordinates=prediction_umap,
        source_title=f"Source ({source_timepoint})",
        target_title=f"Real Target ({target_timepoint})",
        prediction_title=f"Prediction ({target_timepoint})",
    )

    fig.savefig(
        output_dir / "umap_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ------------------------------------------------------------------
    # Target / prediction overlay
    # ------------------------------------------------------------------

    fig, _ = plot_joint_prediction_overlay(
        target_coordinates=target_umap,
        prediction_coordinates=prediction_umap,
        title=f"Target vs Prediction ({target_timepoint})",
    )

    fig.savefig(
        output_dir / "umap_target_prediction.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ------------------------------------------------------------------
# Experiment
# ------------------------------------------------------------------


def run_experiment(
    data_root: Path,
    source_timepoint: str,
    target_timepoint: str,
    n_cells: int,
    seed: int,
    output_root: Path,
) -> None:
    torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    store = ScifateStore(data_root)

    source_data = store.load(source_timepoint)
    target_data = store.load(target_timepoint)

    if target_data.time_hours <= source_data.time_hours:
        raise ValueError(
            f"Target time ({target_data.timepoint}) must be later than "
            f"source time ({source_data.timepoint})."
        )

    source = sample_population(source_data, n_cells=n_cells)
    target = sample_population(target_data, n_cells=n_cells)

    print()
    print("Data")
    print("----")
    print(f"Source: {source_data.timepoint} -> {source.shape}")
    print(f"Target: {target_data.timepoint} -> {target.shape}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model = NoChangeBaseline()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    evaluator = make_default_evaluator()

    runner = ExperimentRunner(
        model=model,
        evaluator=evaluator,
        trainer=None,
    )

    result = runner.run(
        source=source,
        target=target,
        source_time=source_data.time_hours,
        target_time=target_data.time_hours,
    )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------

    experiment_name = (
        f"no_change_{source_data.timepoint}_to_{target_data.timepoint}"
    )

    output_dir = output_root / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    report = result.report.to_frame()

    print()
    print("Evaluation")
    print("----------")
    print(report.to_string(index=False))

    config = {
        "model": "no_change",
        "source_timepoint": source_data.timepoint,
        "target_timepoint": target_data.timepoint,
        "source_time_hours": source_data.time_hours,
        "target_time_hours": target_data.time_hours,
        "n_cells": n_cells,
        "seed": seed,
        "normalize_expression": True,
    }

    save_results(
        report=result.report,
        output_dir=output_dir,
        config=config,
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    print()
    print("Creating UMAP plots...")

    save_plots(
        source=source,
        target=target,
        prediction=result.prediction.states[0],
        source_timepoint=source_data.timepoint,
        target_timepoint=target_data.timepoint,
        output_dir=output_dir,
        seed=seed,
    )

    print()
    print(f"Results saved to: {output_dir}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the NoChange baseline on SCI-FATE2."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed/scifate2"),
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source timepoint, for example 5h or 168h.",
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target timepoint, for example 24h or 192h.",
    )

    parser.add_argument(
        "--n-cells",
        type=int,
        default=512,
        help="Number of cells sampled from each population.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_experiment(
        data_root=args.data_root,
        source_timepoint=args.source,
        target_timepoint=args.target,
        n_cells=args.n_cells,
        seed=args.seed,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()