# std-lib imports
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from typing import Any

# 3 party imports
import pandas as pd
import torch
from scipy import sparse

# package imports
from trajectoryflow.experiment.config import ModelConfig, TrainingConfig
from trajectoryflow.experiment.data import ExperimentData


@dataclass
class BuiltExperimentModel:
    model: Any
    trainer: Any | None
    device: torch.device


class ExperimentModelAdapter(ABC):
    """Small model-specific bridge used by the generic benchmark runner."""

    @abstractmethod
    def build(
        self,
        config: ModelConfig,
        data: ExperimentData,
        device: torch.device,
        seed: int,
    ) -> BuiltExperimentModel:
        pass

    @abstractmethod
    def fit(
        self,
        built: BuiltExperimentModel,
        config: TrainingConfig,
    ) -> dict[str, Any]:
        pass

    def predict(
        self,
        built: BuiltExperimentModel,
        source: torch.Tensor,
        source_time: float,
        target_time: float,
        n_samples: int,
    ):
        return built.model.predict(
            source=source,
            source_time=source_time,
            target_time=target_time,
            n_samples=n_samples,
        )

    def supports_velocity(self, built: BuiltExperimentModel) -> bool:
        return callable(getattr(built.model, "predict_velocity", None))

    def predict_velocity(
        self,
        built: BuiltExperimentModel,
        source: torch.Tensor,
    ) -> torch.Tensor:
        method = getattr(built.model, "predict_velocity", None)

        if not callable(method):
            raise NotImplementedError(
                f"{type(built.model).__name__} does not expose predict_velocity()."
            )

        velocity = method(source)

        if hasattr(velocity, "velocity"):
            velocity = velocity.velocity

        if not isinstance(velocity, torch.Tensor):
            velocity = torch.as_tensor(velocity, device=source.device)

        if velocity.shape != source.shape:
            raise ValueError(
                "predict_velocity() must return shape [n_cells, n_genes]; "
                f"got {tuple(velocity.shape)} for source {tuple(source.shape)}."
            )

        return velocity

    def checkpoint_state(self, built: BuiltExperimentModel):
        if isinstance(built.model, torch.nn.Module):
            return built.model.state_dict()

        return None


class EvaluationAdapter(ABC):

    @abstractmethod
    def evaluate(self, prediction, target: torch.Tensor) -> pd.DataFrame:
        pass


class ExperimentRegistry:

    def __init__(self):
        self._models: dict[str, ExperimentModelAdapter] = {}
        self._evaluator: EvaluationAdapter | None = None

    def register_model(self, name: str, adapter: ExperimentModelAdapter) -> None:
        if name in self._models:
            raise KeyError(f"Model {name!r} is already registered.")

        self._models[name] = adapter

    def register_evaluator(self, evaluator: EvaluationAdapter) -> None:
        self._evaluator = evaluator

    def model(self, name: str) -> ExperimentModelAdapter:
        try:
            return self._models[name]
        except KeyError as error:
            raise KeyError(
                f"Unknown model {name!r}. Registered models: {sorted(self._models)}."
            ) from error

    @property
    def evaluator(self) -> EvaluationAdapter:
        if self._evaluator is None:
            raise RuntimeError("No evaluator has been registered.")

        return self._evaluator


class NoChangeExperimentAdapter(ExperimentModelAdapter):

    def build(
        self,
        config: ModelConfig,
        data: ExperimentData,
        device: torch.device,
        seed: int,
    ) -> BuiltExperimentModel:
        from trajectoryflow.models.baselines.no_change import NoChangeBaseline

        if config.trainer_params:
            raise ValueError("no_change does not use trainer_params.")
        model = NoChangeBaseline(**config.model_params)

        return BuiltExperimentModel(
            model=model,
            trainer=None,
            device=device,
        )

    def fit(
        self,
        built: BuiltExperimentModel,
        config: TrainingConfig,
    ) -> dict[str, Any]:
        return {
            "trained": False,
            "epochs": 0,
            "steps": 0,
            "stop_reason": "no_training_required",
        }


def build_velvet_training_data(data: ExperimentData):
    from trajectoryflow.models.baselines.velvet.data import VelvetData

    snapshots = data.training_snapshots()
    total = sparse.vstack([snapshot.expression for snapshot in snapshots], format="csr")
    new = sparse.vstack([snapshot.new for snapshot in snapshots], format="csr")
    obs_parts = []
    for snapshot in snapshots:
        obs = snapshot.obs.copy()
        obs["timepoint"], obs["time_hours"] = snapshot.timepoint, snapshot.time_hours
        obs_parts.append(obs)
    return VelvetData(total=total, new=new, obs=pd.concat(obs_parts, ignore_index=True), timepoints=data.training_timepoints)


class VelvetExperimentAdapter(ExperimentModelAdapter):

    def build(
        self,
        config: ModelConfig,
        data: ExperimentData,
        device: torch.device,
        seed: int,
    ) -> BuiltExperimentModel:
        from trajectoryflow.models.baselines.velvet import (
            VelvetBaseline,
            VelvetSDEConfig,
            VelvetVAEConfig,
        )
        from trajectoryflow.training.velvet import VelvetTrainer

        model_params = dict(config.model_params)
        vae_params = dict(model_params.pop("vae", {}))
        sde_params = dict(model_params.pop("sde", {}))

        vae_params.setdefault("seed", seed)
        sde_params.setdefault("seed", seed)

        vae_config = replace(VelvetVAEConfig(), **vae_params)
        sde_config = replace(VelvetSDEConfig(), **sde_params)

        velvet_data = build_velvet_training_data(data)

        model = VelvetBaseline(
            n_genes=data.n_genes,
            vae_config=vae_config,
            sde_config=sde_config,
            **model_params,
        )

        trainer = VelvetTrainer(
            data=velvet_data,
            device=device,
            **config.trainer_params,
        )

        return BuiltExperimentModel(
            model=model,
            trainer=trainer,
            device=device,
        )

    def fit(
        self,
        built: BuiltExperimentModel,
        config: TrainingConfig,
    ) -> dict[str, Any]:
        if config.budget.limited:
            raise NotImplementedError("VelvetTrainer is not budget-aware yet; fixed-compute budgets cannot be reported safely.")
        if config.early_stopping is not None:
            raise NotImplementedError("VelvetTrainer does not implement the generic early-stopping configuration yet.")

        built.trainer.fit(built.model)

        return {
            "stage1_epochs": len(built.trainer.history.stage1),
            "stage2_epochs": len(built.trainer.history.stage2),
            "sde_epochs": len(built.trainer.history.sde),
            "stop_reason": "configured_training_complete",
        }

    def checkpoint_state(self, built: BuiltExperimentModel):
        model = built.model
        return {
            "velvet": model.velvet.state_dict(),
            "vae_config": asdict(model.velvet.config),
            "sde_config": asdict(model.sde.config),
            "hours_per_sde_unit": model.hours_per_sde_unit,
        }
