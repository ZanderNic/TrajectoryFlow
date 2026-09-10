#!/usr/bin/env python3

# std-lib imports
import argparse
from pathlib import Path

# 3 party imports
import numpy as np
import pandas as pd

# package imports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and summarize a completed TrajectoryFlow benchmark."
    )
    parser.add_argument(
        "benchmark_dir",
        type=Path,
        help="Directory produced by scripts/run_benchmark.py.",
    )
    return parser.parse_args()


def load_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing benchmark file: {path}")
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()

    group = [
        column
        for column in (
            "model",
            "split",
            "phase",
            "task_type",
            "task",
            "task_kind",
            "reference",
            "group",
            "timepoints",
            "source",
            "target",
            "delta_hours",
            "metric",
            "higher_is_better",
        )
        if column in metrics.columns
    ]

    summary = (
        metrics.groupby(group, dropna=False)["value"]
        .agg(mean="mean", std="std", median="median", n="count")
        .reset_index()
    )

    summary["std"] = summary["std"].fillna(0.0)
    return summary


def runtime_summary(runtimes: pd.DataFrame) -> pd.DataFrame:
    if runtimes.empty:
        return pd.DataFrame()

    group = ["model", "split", "phase", "task"]
    columns = [
        "wall_seconds",
        "cpu_seconds",
        "rss_peak_mb",
        "gpu_peak_allocated_mb",
        "gpu_peak_reserved_mb",
    ]
    available = [column for column in columns if column in runtimes.columns]

    if not available:
        return pd.DataFrame()

    summary = runtimes.groupby(group, dropna=False)[available].agg(["mean", "std"])
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    summary = summary.reset_index()

    for column in summary.columns:
        if column.endswith("_std"):
            summary[column] = summary[column].fillna(0.0)

    return summary


def experiment_summary(experiments: pd.DataFrame) -> pd.DataFrame:
    if experiments.empty:
        return pd.DataFrame()

    rows = []

    for (model, split), frame in experiments.groupby(["model", "split"], dropna=False):
        training = pd.to_numeric(frame.get("training_seconds"), errors="coerce")
        total = pd.to_numeric(frame.get("total_seconds"), errors="coerce")

        rows.append(
            {
                "model": model,
                "split": split,
                "n_runs": len(frame),
                "n_completed": int((frame["status"] == "completed").sum()),
                "n_failed": int((frame["status"] == "failed").sum()),
                "training_seconds_mean": float(training.mean()) if training.notna().any() else np.nan,
                "training_seconds_std": float(training.std(ddof=1)) if training.notna().sum() > 1 else 0.0,
                "total_seconds_mean": float(total.mean()) if total.notna().any() else np.nan,
                "total_seconds_std": float(total.std(ddof=1)) if total.notna().sum() > 1 else 0.0,
            }
        )

    return pd.DataFrame(rows)


def as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}

    return bool(value)


def ranking(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    rows = []
    group = [
        column
        for column in (
            "split",
            "phase",
            "task_type",
            "task",
            "reference",
            "group",
            "metric",
        )
        if column in summary.columns
    ]

    for keys, frame in summary.groupby(group, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group, keys))

        higher = as_bool(frame["higher_is_better"].iloc[0])
        ordered = frame.sort_values("mean", ascending=not higher).reset_index(drop=True)

        for rank, row in ordered.iterrows():
            rows.append(
                {
                    **key_values,
                    "rank": rank + 1,
                    "model": row["model"],
                    "mean": row["mean"],
                    "std": row["std"],
                    "n": row["n"],
                    "higher_is_better": higher,
                }
            )

    return pd.DataFrame(rows)


def paired_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or metrics["model"].nunique() < 2:
        return pd.DataFrame()

    rows = []
    group = [
        column
        for column in (
            "split",
            "phase",
            "task_type",
            "task",
            "reference",
            "group",
            "metric",
        )
        if column in metrics.columns
    ]

    for keys, frame in metrics.groupby(group, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group, keys))
        models = sorted(frame["model"].dropna().unique())
        higher = as_bool(frame["higher_is_better"].dropna().iloc[0])

        for i, model_a in enumerate(models):
            for model_b in models[i + 1 :]:
                a = frame.loc[frame["model"] == model_a, ["seed", "value"]].rename(
                    columns={"value": "a"}
                )
                b = frame.loc[frame["model"] == model_b, ["seed", "value"]].rename(
                    columns={"value": "b"}
                )
                paired = a.merge(b, on="seed", how="inner").dropna()

                if paired.empty:
                    continue

                delta = paired["a"] - paired["b"]
                better = delta > 0 if higher else delta < 0

                rows.append(
                    {
                        **key_values,
                        "model_a": model_a,
                        "model_b": model_b,
                        "n_paired_seeds": len(paired),
                        "mean_a_minus_b": float(delta.mean()),
                        "std_a_minus_b": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                        "a_win_fraction": float(better.mean()),
                        "higher_is_better": higher,
                    }
                )

    return pd.DataFrame(rows)


def print_overview(
    experiments: pd.DataFrame,
    metric_table: pd.DataFrame,
    ranking_table: pd.DataFrame,
) -> None:
    print("Benchmark analysis")

    if experiments.empty:
        print("  experiments: 0")
    else:
        print(f"  experiments: {len(experiments)}")
        print(f"  completed  : {(experiments['status'] == 'completed').sum()}")
        print(f"  failed     : {(experiments['status'] == 'failed').sum()}")

    if not metric_table.empty:
        print(f"  metric rows: {len(metric_table)}")

    if ranking_table.empty:
        return

    print()
    print("Best model per split/task/metric")

    winners = ranking_table.loc[ranking_table["rank"] == 1]

    for _, row in winners.iterrows():
        prefix = f"{row['split']} | {row['task']}"
        if "reference" in row.index and pd.notna(row["reference"]):
            prefix += f" | {row['reference']}"

        print(
            f"  {prefix} | {row['metric']}: "
            f"{row['model']} ({row['mean']:.6g} ± {row['std']:.3g})"
        )


def main() -> None:
    args = parse_args()
    root = args.benchmark_dir.resolve()

    metrics = load_csv(root / "metrics.csv")
    runtimes = load_csv(root / "runtimes.csv", required=False)
    experiments = load_csv(root / "experiments.csv", required=False)

    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    metrics_summary = metric_summary(metrics)
    runtimes_summary = runtime_summary(runtimes)
    experiments_summary = experiment_summary(experiments)
    rankings = ranking(metrics_summary)
    comparisons = paired_comparisons(metrics)

    metrics_summary.to_csv(analysis_dir / "metric_summary.csv", index=False)
    runtimes_summary.to_csv(analysis_dir / "runtime_summary.csv", index=False)
    experiments_summary.to_csv(analysis_dir / "experiment_summary.csv", index=False)
    rankings.to_csv(analysis_dir / "rankings.csv", index=False)
    comparisons.to_csv(analysis_dir / "paired_comparisons.csv", index=False)

    print_overview(experiments, metrics_summary, rankings)
    print()
    print(f"Analysis saved to {analysis_dir}")


if __name__ == "__main__":
    main()
