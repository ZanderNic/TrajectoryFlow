# std-lib imports
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from typing import Any

# 3 party imports
import pandas as pd
import torch
from scipy import sparse

# package imports
from trajectoryflow.experiment.config import ModelConfig, TrainingConfig
from trajectoryflow.experiment.data import ExperimentData, SnapshotSelection
from trajectoryflow.experiment.runtime import stable_seed
from trajectoryflow.experiment.split import timepoint_hours




@dataclass
class PredictionContext:
    """Source-cell modalities available to a model during forecast inference."""

    total: torch.Tensor
    new: torch.Tensor | None = None
    ntr: torch.Tensor | None = None
    old: torch.Tensor | None = None


@dataclass
class DualScaleTrainingBundle:
    trainer: Any
    local_loader: Any
    global_loader: Any

@dataclass
class BuiltExperimentModel:
    model: Any
    trainer: Any | None
    device: torch.device
    metadata: dict[str, Any] = field(default_factory=dict)


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

    def prepare_prediction_context(
        self,
        selection: SnapshotSelection,
        device: torch.device,
    ) -> PredictionContext:
        total = torch.from_numpy(selection.expression.toarray()).float().to(device)
        return PredictionContext(total=total)

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

    def predict_with_context(
        self,
        built: BuiltExperimentModel,
        context: PredictionContext,
        source_time: float,
        target_time: float,
        n_samples: int,
    ):
        return self.predict(
            built=built,
            source=context.total,
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

    def set_training_progress(self, built: BuiltExperimentModel, progress) -> None:
        """Attach a generic progress reporter to the model-specific trainer."""
        trainer = built.trainer
        nested = getattr(trainer, "trainer", None)
        target = nested if nested is not None else trainer
        if target is not None and hasattr(target, "progress"):
            target.progress = progress

    def release_training_resources(self, built: BuiltExperimentModel) -> None:
        """Release adapter-owned training data that is not needed for inference."""


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

    total_parts, new_parts, obs_parts = [], [], []
    for timepoint in data.training_timepoints:
        snapshot = data.training_snapshot(timepoint)
        total_parts.append(snapshot.expression.astype("float32", copy=False))
        new_parts.append(snapshot.new.astype("float32", copy=False))
        obs = snapshot.obs.copy()
        obs["timepoint"], obs["time_hours"] = snapshot.timepoint, snapshot.time_hours
        obs_parts.append(obs)
        data.unload(timepoint)
        del snapshot

    total = sparse.vstack(total_parts, format="csr")
    total_parts.clear()
    new = sparse.vstack(new_parts, format="csr")
    new_parts.clear()
    obs = pd.concat(obs_parts, ignore_index=True)
    return VelvetData(total=total, new=new, obs=obs, timepoints=data.training_timepoints)


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

    def release_training_resources(self, built: BuiltExperimentModel) -> None:
        if built.trainer is not None:
            built.trainer.data = None
            built.trainer.neighbor_indices = None


class DualScaleExperimentAdapter(ExperimentModelAdapter):
    """Benchmark bridge for the dual-scale kinetic transition model."""

    @staticmethod
    def _trainer_configuration(params: dict[str, Any]):
        from trajectoryflow.training.dual_scale import (
            DualScaleLossWeights,
            DualScaleTrainerConfig,
            DualScaleTrainingSchedule,
        )

        params = dict(params)
        loader = {
            "local_batch_size": int(params.pop("local_batch_size", 128)),
            "global_batch_size": int(params.pop("global_batch_size", 128)),
            "labeling_time": float(params.pop("labeling_time", 2.0)),
            "global_transitions": params.pop("global_transitions", None),
        }
        if loader["local_batch_size"] < 1 or loader["global_batch_size"] < 1:
            raise ValueError("dual_scale batch sizes must be >= 1.")
        if loader["labeling_time"] <= 0:
            raise ValueError("dual_scale labeling_time must be positive.")

        schedule = replace(DualScaleTrainingSchedule(), **dict(params.pop("schedule", {})))
        loss_weights = replace(DualScaleLossWeights(), **dict(params.pop("loss_weights", {})))
        trainer_config = replace(
            DualScaleTrainerConfig(),
            schedule=schedule,
            loss_weights=loss_weights,
            **params,
        )
        return trainer_config, loader

    @staticmethod
    def _fit_timepoint_pairs(
        data: ExperimentData,
        configured,
    ) -> tuple[tuple[str, str], ...]:
        allowed = set(data.training_timepoints)
        if configured is None:
            ordered = sorted(data.training_timepoints, key=timepoint_hours)
            return tuple(zip(ordered[:-1], ordered[1:]))

        pairs = []
        for pair in configured:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("global_transitions must contain [source, target] pairs.")
            source, target = str(pair[0]), str(pair[1])
            if source not in allowed or target not in allowed:
                raise ValueError(
                    "dual_scale global transitions may only use fit_timepoints; "
                    f"got {source!r}->{target!r}, allowed={sorted(allowed)}."
                )
            if timepoint_hours(target) <= timepoint_hours(source):
                raise ValueError("dual_scale global transition targets must be later than sources.")
            pairs.append((source, target))
        return tuple(pairs)

    @staticmethod
    def _steps_per_loader(schedule, kind: str) -> int:
        if kind == "local":
            return max(
                1,
                schedule.local_steps_per_pretrain_epoch,
                schedule.local_steps_per_joint_cycle * schedule.joint_cycles_per_epoch,
            )
        return max(
            1,
            schedule.global_steps_per_pretrain_epoch,
            schedule.global_steps_per_joint_cycle * schedule.joint_cycles_per_epoch,
        )

    @staticmethod
    def _requires_global(schedule) -> bool:
        return bool(
            (schedule.global_pretrain_epochs and schedule.global_steps_per_pretrain_epoch)
            or (
                schedule.joint_epochs
                and schedule.global_steps_per_joint_cycle
                and schedule.joint_cycles_per_epoch
            )
        )

    def build(
        self,
        config: ModelConfig,
        data: ExperimentData,
        device: torch.device,
        seed: int,
    ) -> BuiltExperimentModel:
        from trajectoryflow.models.dual_scale import DualScaleModelConfig, build_default_dual_scale_model
        from trajectoryflow.training.dual_scale import DualScaleTrainer
        from trajectoryflow.training.dual_scale_data import (
            LocalPopulationLoader,
            MixedBatchLoader,
            SparseKineticPopulation,
            UnpairedGlobalPopulationLoader,
        )

        model_config = replace(DualScaleModelConfig(), **dict(config.model_params))
        model = build_default_dual_scale_model(data.n_genes, model_config)
        trainer_config, loader_config = self._trainer_configuration(config.trainer_params)
        labeling_time = loader_config["labeling_time"]

        populations = {}
        for timepoint in data.training_timepoints:
            snapshot = data.training_snapshot(timepoint)
            populations[timepoint] = SparseKineticPopulation(
                total=snapshot.expression.astype("float32", copy=False),
                new=snapshot.new.astype("float32", copy=False),
                ntr=snapshot.ntr.astype("float32", copy=False),
                labeling_time=labeling_time,
            )
            data.unload(timepoint)

        local_steps = self._steps_per_loader(trainer_config.schedule, "local")
        local_loaders = tuple(
            LocalPopulationLoader(
                population=population,
                batch_size=loader_config["local_batch_size"],
                seed=stable_seed(seed, "dual_scale", "local", timepoint),
            )
            for timepoint, population in populations.items()
        )
        local_loader = MixedBatchLoader(
            loaders=local_loaders,
            steps_per_epoch=local_steps,
            weights=tuple(float(len(populations[timepoint])) for timepoint in populations),
            seed=stable_seed(seed, "dual_scale", "local_mix"),
        )

        pairs = self._fit_timepoint_pairs(data, loader_config["global_transitions"])
        global_loaders = []
        past_constraints = []
        global_steps = self._steps_per_loader(trainer_config.schedule, "global")
        ordered_training = sorted(populations, key=timepoint_hours)
        for source_timepoint, target_timepoint in pairs:
            source_hours = timepoint_hours(source_timepoint)
            earlier = [tp for tp in ordered_training if timepoint_hours(tp) < source_hours]
            past_timepoint = earlier[-1] if earlier else None
            past_delta_time = (
                None
                if past_timepoint is None
                else source_hours - timepoint_hours(past_timepoint)
            )
            if past_timepoint is not None:
                past_constraints.append((past_timepoint, source_timepoint))
            global_loaders.append(
                UnpairedGlobalPopulationLoader(
                    source=populations[source_timepoint],
                    future_target_total=populations[target_timepoint].total,
                    past_source=None if past_timepoint is None else populations[past_timepoint],
                    past_delta_time=past_delta_time,
                    delta_time=timepoint_hours(target_timepoint) - source_hours,
                    batch_size=loader_config["global_batch_size"],
                    steps_per_epoch=global_steps,
                    seed=stable_seed(seed, "dual_scale", "global", source_timepoint, target_timepoint),
                )
            )

        if not global_loaders and self._requires_global(trainer_config.schedule):
            raise ValueError(
                "dual_scale training schedule requests global steps, but there are no valid "
                "fit-timepoint transitions. Provide at least two fit_timepoints or set all global steps to 0."
            )
        global_loader = (
            MixedBatchLoader(
                loaders=tuple(global_loaders),
                steps_per_epoch=global_steps,
                seed=stable_seed(seed, "dual_scale", "global_mix"),
            )
            if global_loaders
            else ()
        )

        trainer = DualScaleTrainer(model=model, device=device, config=trainer_config)
        bundle = DualScaleTrainingBundle(trainer=trainer, local_loader=local_loader, global_loader=global_loader)
        return BuiltExperimentModel(
            model=model,
            trainer=bundle,
            device=device,
            metadata={
                "model_name": config.name,
                "labeling_time": labeling_time,
                "training_transitions": pairs,
                "past_training_constraints": tuple(past_constraints),
            },
        )

    def fit(
        self,
        built: BuiltExperimentModel,
        config: TrainingConfig,
    ) -> dict[str, Any]:
        if config.budget.limited:
            raise NotImplementedError(
                "DualScaleTrainer uses its explicit local/global schedule and is not yet wired to the generic training budget."
            )
        if config.early_stopping is not None:
            raise NotImplementedError("DualScaleTrainer does not implement generic early stopping yet.")
        if not isinstance(built.trainer, DualScaleTrainingBundle):
            raise TypeError("DualScaleExperimentAdapter expected a DualScaleTrainingBundle.")

        history = built.trainer.trainer.fit(
            local_loader=built.trainer.local_loader,
            global_loader=built.trainer.global_loader,
        )
        return {
            "epochs": len(history.epochs),
            "local_steps": len(history.local),
            "global_steps": len(history.global_),
            "stop_reason": "configured_training_complete",
        }

    def prepare_prediction_context(
        self,
        selection: SnapshotSelection,
        device: torch.device,
    ) -> PredictionContext:
        total = torch.from_numpy(selection.expression.toarray()).float().to(device)
        new = torch.from_numpy(selection.new.toarray()).float().to(device)
        ntr = torch.from_numpy(selection.ntr.toarray()).float().to(device)
        return PredictionContext(total=total, new=new, ntr=ntr)

    def predict_with_context(
        self,
        built: BuiltExperimentModel,
        context: PredictionContext,
        source_time: float,
        target_time: float,
        n_samples: int,
    ):
        from trajectoryflow.models.base import TrajectoryPrediction

        if context.new is None or context.ntr is None:
            raise ValueError("dual_scale prediction requires source total, new RNA and NTR.")
        if target_time <= source_time:
            raise ValueError("target_time must be later than source_time.")
        old = (context.total - context.new).clamp_min(0) if context.old is None else context.old
        output = built.model.forward_global(
            total=context.total,
            old=old,
            new=context.new,
            ntr=context.ntr,
            delta_time=target_time - source_time,
            labeling_time=float(built.metadata.get("labeling_time", built.model.time_scale_hours)),
            n_samples=n_samples,
        )
        return TrajectoryPrediction(
            states=output.future_total,
            source_time=source_time,
            target_time=target_time,
            metadata={
                "model": str(built.metadata.get("model_name", "dual_scale")),
                "stochastic": True,
                "uses_kinetic_context": True,
                "transition": built.model.transition_name,
            },
        )

    def release_training_resources(self, built: BuiltExperimentModel) -> None:
        built.trainer = None
