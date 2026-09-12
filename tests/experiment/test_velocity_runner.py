# std-lib imports

# 3 party imports
import numpy as np
import torch

# package imports
from trajectoryflow.experiment.config import (
    BenchmarkConfig,
    EvaluationConfig,
    ModelConfig,
    RuntimeConfig,
    VelocityEvaluationConfig,
)
from trajectoryflow.experiment.evaluation import (
    VelocityReference,
    save_velocity_reference,
)
from trajectoryflow.experiment.models import (
    BuiltExperimentModel,
    ExperimentModelAdapter,
    ExperimentRegistry,
)
from trajectoryflow.experiment.runner import ExperimentRunner
from trajectoryflow.experiment.split import (
    CellSplitSpec,
    SplitSpec,
    VelocityTask,
    resolve_split,
)


class VelocityModel(torch.nn.Module):

    def predict_velocity(self, source):
        return source


class VelocityAdapter(ExperimentModelAdapter):

    def build(self, config, data, device, seed):
        return BuiltExperimentModel(
            model=VelocityModel().to(device),
            trainer=None,
            device=device,
        )

    def fit(self, built, config):
        return {"trained": False}


def test_velocity_runner_end_to_end(tmp_path, fake_store):
    reference_dir = tmp_path / "refs"
    reference_dir.mkdir()

    snapshot = fake_store.load("5h")
    expression = snapshot.expression.toarray().astype(np.float64)

    reference = VelocityReference(
        name="pts",
        cell_ids=snapshot.obs["cell_id"].astype(str).to_numpy(),
        vectors=expression,
        pca_components=np.eye(fake_store.n_genes),
        pca_mean=np.zeros(fake_store.n_genes),
        positions=expression,
        genes=fake_store.genes["gene"].astype(str).to_numpy(),
    )
    save_velocity_reference(reference_dir / "pts.npz", reference)

    split_spec = SplitSpec(
        name="velocity_test",
        fit_timepoints=("5h",),
        cell_split=CellSplitSpec(
            train_fraction=0.5,
            validation_fraction=0.0,
            test_fraction=0.5,
            stratify_by=("rep",),
        ),
        test_velocity_tasks=(
            VelocityTask(
                reference="pts",
                timepoints=("5h",),
                partition="test",
            ),
        ),
    )

    config = BenchmarkConfig(
        data_root=tmp_path / "data",
        output_dir=tmp_path / "runs",
        seeds=(0,),
        models=(ModelConfig(name="velocity"),),
        splits=(split_spec,),
        evaluation=EvaluationConfig(
            velocity=VelocityEvaluationConfig(
                enabled=True,
                reference_dir=reference_dir,
                plots=False,
            )
        ),
        runtime=RuntimeConfig(device="cpu"),
        save_checkpoints=False,
        save_predictions=False,
    )

    registry = ExperimentRegistry()
    registry.register_model("velocity", VelocityAdapter())

    split = resolve_split(fake_store, split_spec, seed=0)
    result = ExperimentRunner(
        config=config,
        registry=registry,
        store=fake_store,
    ).run(
        model_config=config.models[0],
        split=split,
        seed=0,
    )

    assert result.status == "completed"
    assert len(result.velocity_tasks) == 1

    velocity_result = result.velocity_tasks[0]
    assert velocity_result.status == "completed"
    assert velocity_result.n_cells == 10

    score = next(
        metric.value
        for metric in velocity_result.metrics
        if metric.name == "cosine_similarity" and metric.group is None
    )
    assert score == 1.0

    folder = (
        config.output_dir
        / "velocity"
        / split.name
        / "seed_0"
        / "velocity"
        / "test__pts__5h"
    )

    assert (folder / "metrics.csv").exists()
    assert (folder / "cells.csv").exists()
