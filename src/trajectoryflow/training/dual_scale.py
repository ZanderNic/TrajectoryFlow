# std-lib imports
from dataclasses import dataclass, field
from typing import Iterable, Iterator

# 3 party imports
import torch
from torch import nn

# package imports
from trajectoryflow.models.dual_scale import DualScaleKineticTransitionModel
from trajectoryflow.models.dual_scale.losses import (
    kinetic_loss,
    population_sliced_wasserstein_loss,
    reconstruction_loss,
)


@dataclass
class LocalBatch:
    total: torch.Tensor
    new: torch.Tensor
    ntr: torch.Tensor | None = None
    labeling_time: float = 2.0

    def resolved(self, device: torch.device) -> "LocalBatch":
        total = self.total.to(device=device, dtype=torch.float32)
        new = self.new.to(device=device, dtype=torch.float32)
        ntr = (
            new / total.clamp_min(1e-8)
            if self.ntr is None
            else self.ntr.to(device=device, dtype=torch.float32)
        )
        return LocalBatch(
            total=total,
            new=new,
            ntr=ntr.clamp(0, 1),
            labeling_time=self.labeling_time,
        )


@dataclass
class GlobalBatch:
    total: torch.Tensor
    new: torch.Tensor
    target_total: torch.Tensor
    delta_time: float
    ntr: torch.Tensor | None = None
    labeling_time: float = 2.0

    def resolved(self, device: torch.device) -> "GlobalBatch":
        source = LocalBatch(
            total=self.total,
            new=self.new,
            ntr=self.ntr,
            labeling_time=self.labeling_time,
        ).resolved(device)
        return GlobalBatch(
            total=source.total,
            new=source.new,
            ntr=source.ntr,
            target_total=self.target_total.to(device=device, dtype=torch.float32),
            delta_time=self.delta_time,
            labeling_time=self.labeling_time,
        )


@dataclass(frozen=True)
class DualScaleLossWeights:
    """Weights for the three objectives used by the model."""

    reconstruction: float = 1.0
    local_kinetic: float = 1.0
    population_sliced_wasserstein: float = 1.0


@dataclass(frozen=True)
class DualScaleTrainingSchedule:
    local_pretrain_epochs: int = 20
    global_pretrain_epochs: int = 0
    joint_epochs: int = 100
    local_steps_per_pretrain_epoch: int = 100
    global_steps_per_pretrain_epoch: int = 100
    local_steps_per_joint_cycle: int = 2
    global_steps_per_joint_cycle: int = 1
    joint_cycles_per_epoch: int = 100

    def __post_init__(self) -> None:
        values = (
            self.local_pretrain_epochs,
            self.global_pretrain_epochs,
            self.joint_epochs,
            self.local_steps_per_pretrain_epoch,
            self.global_steps_per_pretrain_epoch,
            self.local_steps_per_joint_cycle,
            self.global_steps_per_joint_cycle,
            self.joint_cycles_per_epoch,
        )
        if any(value < 0 for value in values):
            raise ValueError("Training schedule values must be non-negative.")
        if self.joint_epochs and not self.joint_cycles_per_epoch:
            raise ValueError("joint_cycles_per_epoch must be > 0 when joint_epochs > 0.")


@dataclass(frozen=True)
class DualScaleTrainerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float | None = 5.0
    global_samples: int = 1
    sw_projections: int = 16
    schedule: DualScaleTrainingSchedule = field(default_factory=DualScaleTrainingSchedule)
    loss_weights: DualScaleLossWeights = field(default_factory=DualScaleLossWeights)

    def __post_init__(self) -> None:
        if self.global_samples < 1:
            raise ValueError("global_samples must be >= 1.")
        if self.sw_projections < 1:
            raise ValueError("sw_projections must be >= 1.")


@dataclass
class DualScaleTrainingHistory:
    local: list[dict[str, float]] = field(default_factory=list)
    global_: list[dict[str, float]] = field(default_factory=list)
    epochs: list[dict[str, float]] = field(default_factory=list)


class _CyclingIterator:
    def __init__(self, loader: Iterable):
        self.loader = loader
        self.iterator: Iterator | None = None

    def next(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


class DualScaleTrainer:
    """Train the three-objective stochastic residual transition model.

    1. reconstruction:
       current total RNA + NTR -> state latent -> reconstructed total RNA

    2. local_kinetic:
       total RNA -> kinetic latent -> alpha/gamma -> predicted new RNA

    3. population_sliced_wasserstein:
       source population -> stochastic residual transition -> generated future
       population, compared with the independently sampled observed target
       population.

    No source-target cell pairing is introduced across timepoints.
    """

    def __init__(
        self,
        model: DualScaleKineticTransitionModel,
        device: torch.device | str,
        config: DualScaleTrainerConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        progress=None,
    ):
        self.model = model
        self.device = torch.device(device)
        self.config = config or DualScaleTrainerConfig()
        self.model.to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.history = DualScaleTrainingHistory()
        self._epoch = 0
        self.progress = progress

    def _optimize(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

    def local_losses(self, batch: LocalBatch) -> dict[str, torch.Tensor]:
        batch = batch.resolved(self.device)
        output = self.model.forward_local(
            total=batch.total,
            ntr=batch.ntr,
            labeling_time=batch.labeling_time,
        )

        reconstruction = reconstruction_loss(output.reconstruction, batch.total)
        local_kinetic = kinetic_loss(output.predicted_new, batch.new)

        weights = self.config.loss_weights
        total = (
            weights.reconstruction * reconstruction
            + weights.local_kinetic * local_kinetic
        )
        return {
            "reconstruction": reconstruction,
            "local_kinetic": local_kinetic,
            "total": total,
        }

    def _population_sw(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                population_sliced_wasserstein_loss(
                    sample,
                    target,
                    self.config.sw_projections,
                )
                for sample in predicted
            ]
        ).mean()

    def global_losses(self, batch: GlobalBatch) -> dict[str, torch.Tensor]:
        batch = batch.resolved(self.device)
        future = self.model.forward_global(
            total=batch.total,
            ntr=batch.ntr,
            delta_time=batch.delta_time,
            n_samples=self.config.global_samples,
        )
        population_sw = self._population_sw(
            future.future_total,
            batch.target_total,
        )
        total = (
            self.config.loss_weights.population_sliced_wasserstein
            * population_sw
        )
        return {
            "population_sliced_wasserstein": population_sw,
            "total": total,
        }

    @staticmethod
    def _numbers(losses: dict[str, torch.Tensor]) -> dict[str, float]:
        return {
            name: float(value.detach().cpu())
            for name, value in losses.items()
        }

    def local_step(self, batch: LocalBatch, stage: str = "local") -> dict[str, float]:
        self.model.train()
        losses = self.local_losses(batch)
        self._optimize(losses["total"])
        values = self._numbers(losses)
        self.history.local.append(values)
        if self.progress is not None:
            self.progress.update(stage=stage, epoch=self._epoch, metrics=values)
        return values

    def global_step(self, batch: GlobalBatch, stage: str = "global") -> dict[str, float]:
        self.model.train()
        losses = self.global_losses(batch)
        self._optimize(losses["total"])
        values = self._numbers(losses)
        self.history.global_.append(values)
        if self.progress is not None:
            self.progress.update(stage=stage, epoch=self._epoch, metrics=values)
        return values

    def _run_phase(
        self,
        loader: Iterable,
        steps: int,
        kind: str,
        stage: str,
    ) -> dict[str, float]:
        if steps == 0:
            return {}
        batches = _CyclingIterator(loader)
        records = []
        step = self.local_step if kind == "local" else self.global_step
        for _ in range(steps):
            records.append(step(batches.next(), stage=stage))
        keys = records[0]
        return {
            f"{kind}_{key}": sum(record[key] for record in records) / len(records)
            for key in keys
        }

    def fit(
        self,
        local_loader: Iterable,
        global_loader: Iterable,
    ) -> DualScaleTrainingHistory:
        schedule = self.config.schedule
        local_batches = _CyclingIterator(local_loader)
        global_batches = _CyclingIterator(global_loader)

        total_steps = (
            schedule.local_pretrain_epochs * schedule.local_steps_per_pretrain_epoch
            + schedule.global_pretrain_epochs * schedule.global_steps_per_pretrain_epoch
            + schedule.joint_epochs
            * schedule.joint_cycles_per_epoch
            * (schedule.local_steps_per_joint_cycle + schedule.global_steps_per_joint_cycle)
        )

        if self.progress is not None:
            self.progress.start(total=total_steps, unit="step")

        for _ in range(schedule.local_pretrain_epochs):
            self._epoch += 1
            stage = "local_pretrain"
            summary = self._run_phase(
                local_loader,
                schedule.local_steps_per_pretrain_epoch,
                "local",
                stage,
            )
            summary["epoch"] = float(self._epoch)
            self.history.epochs.append(summary)
            if self.progress is not None:
                self.progress.epoch(stage=stage, epoch=self._epoch, metrics=summary)

        for _ in range(schedule.global_pretrain_epochs):
            self._epoch += 1
            stage = "global_pretrain"
            summary = self._run_phase(
                global_loader,
                schedule.global_steps_per_pretrain_epoch,
                "global",
                stage,
            )
            summary["epoch"] = float(self._epoch)
            self.history.epochs.append(summary)
            if self.progress is not None:
                self.progress.epoch(stage=stage, epoch=self._epoch, metrics=summary)

        for _ in range(schedule.joint_epochs):
            self._epoch += 1
            records: list[tuple[str, dict[str, float]]] = []

            for _ in range(schedule.joint_cycles_per_epoch):
                for _ in range(schedule.local_steps_per_joint_cycle):
                    records.append(
                        (
                            "local",
                            self.local_step(local_batches.next(), stage="joint/local"),
                        )
                    )
                for _ in range(schedule.global_steps_per_joint_cycle):
                    records.append(
                        (
                            "global",
                            self.global_step(global_batches.next(), stage="joint/global"),
                        )
                    )

            summary: dict[str, float] = {"epoch": float(self._epoch)}
            for kind in ("local", "global"):
                kind_records = [
                    record
                    for record_kind, record in records
                    if record_kind == kind
                ]
                if kind_records:
                    for key in kind_records[0]:
                        summary[f"{kind}_{key}"] = (
                            sum(record[key] for record in kind_records)
                            / len(kind_records)
                        )

            self.history.epochs.append(summary)
            if self.progress is not None:
                self.progress.epoch(
                    stage="joint",
                    epoch=self._epoch,
                    metrics=summary,
                )

        return self.history
