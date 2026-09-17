#!/usr/bin/env python3

# std-lib imports
import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

# 3 party imports
import pandas as pd
from tqdm.auto import tqdm

# package imports
from trajectoryflow.data.store import ScifateStore
from trajectoryflow.evaluation.default import make_default_evaluator
from trajectoryflow.experiment.config import BenchmarkConfig, load_benchmark_config
from trajectoryflow.experiment.evaluation import CallableEvaluationAdapter
from trajectoryflow.experiment.models import DualScaleExperimentAdapter, ExperimentRegistry, NoChangeExperimentAdapter, VelvetExperimentAdapter
from trajectoryflow.experiment.runner import ExperimentRunner
from trajectoryflow.experiment.runtime import system_info
from trajectoryflow.experiment.split import resolve_split
from trajectoryflow.models.base import TrajectoryPrediction


_DEFAULT_EVALUATOR = make_default_evaluator()
METRICS = tuple(_DEFAULT_EVALUATOR.available_metrics())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible TrajectoryFlow benchmark from a TOML config."
    )
    parser.add_argument("config", type=Path, help="Benchmark TOML config.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed experiments only when split, config, data, references, and code still match.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the configured output directory before running.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def evaluate_prediction(prediction, target) -> pd.DataFrame:
    if not isinstance(prediction, TrajectoryPrediction):
        prediction = TrajectoryPrediction(states=prediction.states, source_time=float(getattr(prediction, "source_time", 0.0)), target_time=float(getattr(prediction, "target_time", 1.0)), metadata=dict(getattr(prediction, "metadata", {})))
    frame = _DEFAULT_EVALUATOR.evaluate(prediction, target).to_frame()
    return frame.rename(columns={"mean": "value"})


def make_registry(config: BenchmarkConfig) -> ExperimentRegistry:
    registry = ExperimentRegistry()
    enabled = {model.name for model in config.models if model.enabled}

    if "no_change" in enabled:
        registry.register_model("no_change", NoChangeExperimentAdapter())

    if "velvet" in enabled:
        registry.register_model("velvet", VelvetExperimentAdapter())

    dual_scale_names = sorted(
        name for name in enabled if name == "dual_scale" or name.startswith("dual_scale_")
    )
    for name in dual_scale_names:
        registry.register_model(name, DualScaleExperimentAdapter())

    known = {"no_change", "velvet", *dual_scale_names}
    unknown = sorted(enabled - known)

    if unknown:
        raise ValueError(
            f"No experiment adapter is registered for: {unknown}. "
            "Add the adapter in make_registry()."
        )

    registry.register_evaluator(CallableEvaluationAdapter(evaluate_prediction))
    return registry


def experiment_dir(config: BenchmarkConfig, model: str, split: str, seed: int) -> Path:
    return config.output_dir / model / split / f"seed_{seed}"


def completed_result_matches(
    path: Path,
    split_fingerprint: str,
    resume_signature: str | None = None,
) -> bool:
    try:
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    matches = (
        result.get("status") == "completed"
        and result.get("split_fingerprint") == split_fingerprint
    )
    if not matches or resume_signature is None:
        return matches

    try:
        resume = json.loads((path / "resume.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return resume.get("signature") == resume_signature


def _json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_tree_sha256(root: Path) -> str | None:
    if not root.exists():
        return None
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def dataset_file_state(store: ScifateStore) -> list[dict]:
    manifest = store.manifest
    relative_paths = {manifest["genes"], manifest["preprocessing"]}
    for snapshot in manifest["snapshots"]:
        relative_paths.update(snapshot[key] for key in ("expression", "new", "ntr", "obs"))
    states = []
    for relative in sorted(relative_paths):
        path = store.root / relative
        stat = path.stat()
        states.append({"path": str(relative), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return states


def resume_signature(config: BenchmarkConfig, model, split, store: ScifateStore) -> str:
    root = Path(__file__).resolve().parents[1]
    payload = {
        "schema": 1,
        "model": asdict(model),
        "training": asdict(config.training),
        "evaluation": asdict(config.evaluation),
        "runtime": asdict(config.runtime),
        "save_checkpoints": config.save_checkpoints,
        "save_predictions": config.save_predictions,
        "split_fingerprint": split.fingerprint,
        "dataset": dataset_manifest(store, config),
        "dataset_files": dataset_file_state(store),
        "velocity_references": velocity_reference_manifest(config),
        "code": {
            "trajectoryflow": source_tree_sha256(root / "src" / "trajectoryflow"),
            "run_benchmark": sha256_file(Path(__file__)),
        },
    }
    return _json_hash(payload)


def save_resume_signature(path: Path, signature: str) -> None:
    (path / "resume.json").write_text(json.dumps({"signature": signature}, indent=2), encoding="utf-8")


def dataset_manifest(store: ScifateStore, config: BenchmarkConfig) -> dict:
    root = config.data_root

    return {
        "root": str(root.resolve()),
        "n_genes": int(store.n_genes),
        "timepoints": list(store.timepoints),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "preprocessing_sha256": sha256_file(root / "preprocessing.json"),
        "selected_gene_indices_sha256": sha256_file(root / "selected_gene_indices.npy"),
    }


def velocity_reference_manifest(config: BenchmarkConfig) -> dict[str, dict]:
    references = {
        task.reference
        for split in config.splits
        for task in (
            split.validation_velocity_tasks
            + split.test_velocity_tasks
        )
    }
    output = {}

    for reference in sorted(references):
        path = Path(reference)

        if not path.suffix:
            path = Path(f"{reference}.npz")

        if not path.is_absolute():
            path = config.evaluation.velocity.reference_dir / path

        output[reference] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }

    return output


def save_benchmark_manifest(
    config_path: Path,
    config: BenchmarkConfig,
    store: ScifateStore,
) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "dataset": dataset_manifest(store, config),
        "velocity_references": velocity_reference_manifest(config),
        "system": system_info(),
        "models": [model.name for model in config.models if model.enabled],
        "splits": [split.name for split in config.splits],
        "seeds": list(config.seeds),
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    shutil.copy2(config_path, config.output_dir / "benchmark_config.toml")


def read_csvs(root: Path, name: str) -> pd.DataFrame:
    paths = sorted(root.glob(f"*/*/seed_*/{name}"))
    frames = []

    for path in paths:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue

        if not frame.empty:
            frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def experiment_rows(root: Path) -> pd.DataFrame:
    rows = []

    for path in sorted(root.glob("*/*/seed_*/result.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        training = result.get("training") or {}
        total = result.get("total") or {}
        parameters = result.get("parameters") or {}
        training_summary = result.get("training_summary") or {}

        rows.append(
            {
                "experiment_id": result.get("experiment_id"),
                "model": result.get("model"),
                "split": result.get("split"),
                "seed": result.get("seed"),
                "status": result.get("status"),
                "split_fingerprint": result.get("split_fingerprint"),
                "training_seconds": training.get("wall_seconds"),
                "total_seconds": total.get("wall_seconds"),
                "parameters": parameters.get("total"),
                "trainable_parameters": parameters.get("trainable"),
                "stop_reason": training_summary.get("stop_reason"),
                "error": result.get("error"),
                "result_path": str(path),
            }
        )

    return pd.DataFrame(rows)


def aggregate_outputs(root: Path) -> None:
    metrics = read_csvs(root, "metrics.csv")
    runtimes = read_csvs(root, "runtime.csv")
    experiments = experiment_rows(root)

    metrics.to_csv(root / "metrics.csv", index=False)
    runtimes.to_csv(root / "runtimes.csv", index=False)
    experiments.to_csv(root / "experiments.csv", index=False)


def print_plan(config: BenchmarkConfig) -> None:
    models = [model.name for model in config.models if model.enabled]
    n_experiments = len(models) * len(config.splits) * len(config.seeds)

    print("Benchmark")
    print(f"  data       : {config.data_root}")
    print(f"  output     : {config.output_dir}")
    print(f"  models     : {', '.join(models)}")
    print(f"  splits     : {', '.join(split.name for split in config.splits)}")
    print(f"  seeds      : {config.seeds}")
    print(f"  experiments: {n_experiments}")
    if config.evaluation.velocity.enabled:
        print(
            f"  velocity   : enabled "
            f"({config.evaluation.velocity.reference_dir})"
        )
    print()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_benchmark_config(config_path)

    # Paths in the TOML are interpreted relative to the repository/current shell,
    # then stored as absolute paths in the actual run configuration.
    config = replace(
        config,
        data_root=config.data_root.resolve(),
        output_dir=config.output_dir.resolve(),
        evaluation=replace(
            config.evaluation,
            velocity=replace(
                config.evaluation.velocity,
                reference_dir=(
                    config.evaluation.velocity.reference_dir.resolve()
                ),
            ),
        ),
    )

    if args.overwrite and config.output_dir.exists():
        shutil.rmtree(config.output_dir)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    store = ScifateStore(config.data_root, cache_size=config.runtime.store_cache_size)
    registry = make_registry(config)
    runner = ExperimentRunner(config=config, registry=registry, store=store)

    save_benchmark_manifest(config_path, config, store)
    print_plan(config)

    completed = 0
    skipped = 0
    failed = 0
    enabled_models = [model for model in config.models if model.enabled]
    total_experiments = len(enabled_models) * len(config.splits) * len(config.seeds)
    benchmark_progress = tqdm(
        total=total_experiments,
        desc="Benchmark",
        unit="experiment",
        dynamic_ncols=True,
        position=0,
    )

    def emit(message: str) -> None:
        benchmark_progress.write(message)

    try:
        for split_spec in config.splits:
            for seed in config.seeds:
                split = resolve_split(store=store, spec=split_spec, seed=seed)

                for model in enabled_models:
                    run_dir = experiment_dir(config, model.name, split.name, seed)
                    label = f"{model.name} | {split.name} | seed={seed}"
                    benchmark_progress.set_description_str(f"Benchmark | {label}")

                    signature = resume_signature(config, model, split, store)
                    if args.resume and completed_result_matches(
                        run_dir, split.fingerprint, signature
                    ):
                        emit(f"[skip] {label}")
                        skipped += 1
                        benchmark_progress.update(1)
                        continue

                    emit(f"[run ] {label}")

                    try:
                        result = runner.run(
                            model_config=model,
                            split=split,
                            seed=seed,
                        )
                    except Exception as error:
                        failed += 1
                        emit(f"[fail] {label}: {type(error).__name__}: {error}")
                        benchmark_progress.update(1)

                        if config.fail_fast:
                            aggregate_outputs(config.output_dir)
                            raise

                        continue

                    if result.status == "completed":
                        save_resume_signature(run_dir, signature)
                        completed += 1
                        emit(f"[done] {label}")
                    else:
                        failed += 1
                        emit(f"[fail] {label}: {result.error}")
                    benchmark_progress.update(1)
    finally:
        benchmark_progress.close()

    aggregate_outputs(config.output_dir)

    print()
    print("Benchmark finished")
    print(f"  completed : {completed}")
    print(f"  skipped   : {skipped}")
    print(f"  failed    : {failed}")
    print(f"  results   : {config.output_dir}")
    print(f"  metrics   : {config.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
