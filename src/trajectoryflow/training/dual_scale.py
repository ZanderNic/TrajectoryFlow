# std-lib imports
from dataclasses import dataclass, field
from typing import Iterable, Iterator

# 3 party imports
import torch
from torch import nn
from torch.nn import functional as F

# package imports
from trajectoryflow.models.dual_scale import DualScaleKineticTransitionModel


def _resolve_sampled_residual(output) -> torch.Tensor:
    """Return the diffusion residual across current and legacy model outputs.

    New mechanistic-residual models expose ``sampled_residual`` directly.
    A partially upgraded model may only expose ``sampled_delta`` and
    ``kinetic_delta``; in that case recover the residual exactly.
    Very old models only expose ``sampled_delta``. We intentionally reject
    those here because treating the full transition as a residual would change
    the scientific objective silently.
    """
    residual = getattr(output, "sampled_residual", None)
    if residual is not None:
        return residual

    sampled_delta = getattr(output, "sampled_delta", None)
    kinetic_delta = getattr(output, "kinetic_delta", None)
    if sampled_delta is not None and kinetic_delta is not None:
        return sampled_delta - kinetic_delta.unsqueeze(0)

    raise RuntimeError(
        "Dual-scale model/trainer interface mismatch: GlobalOutputs has no "
        "sampled_residual (and it cannot be reconstructed from sampled_delta "
        "and kinetic_delta). Your trainer is from the mechanistic-residual "
        "version but models/dual_scale/model.py is still from an older patch. "
        "Update model.py before training."
    )


@dataclass
class LocalBatch:
    total: torch.Tensor
    new: torch.Tensor
    old: torch.Tensor | None = None
    ntr: torch.Tensor | None = None
    labeling_time: float = 2.0

    def resolved(self, device: torch.device) -> "LocalBatch":
        total = self.total.to(device=device, dtype=torch.float32)
        new = self.new.to(device=device, dtype=torch.float32)
        old = (total - new).clamp_min(0) if self.old is None else self.old.to(device=device, dtype=torch.float32)
        ntr = new / total.clamp_min(1e-8) if self.ntr is None else self.ntr.to(device=device, dtype=torch.float32)
        return LocalBatch(total=total, new=new, old=old, ntr=ntr.clamp(0, 1), labeling_time=self.labeling_time)


@dataclass
class GlobalBatch:
    total: torch.Tensor
    new: torch.Tensor
    target_total: torch.Tensor
    delta_time: float
    old: torch.Tensor | None = None
    ntr: torch.Tensor | None = None
    past_target_total: torch.Tensor | None = None
    past_source: LocalBatch | None = None
    past_delta_time: float | None = None
    labeling_time: float = 2.0

    def resolved(self, device: torch.device) -> "GlobalBatch":
        local = LocalBatch(self.total, self.new, self.old, self.ntr, self.labeling_time).resolved(device)
        past_source = None if self.past_source is None else self.past_source.resolved(device)
        return GlobalBatch(
            total=local.total,
            new=local.new,
            old=local.old,
            ntr=local.ntr,
            target_total=self.target_total.to(device=device, dtype=torch.float32),
            past_target_total=None if self.past_target_total is None else self.past_target_total.to(device=device, dtype=torch.float32),
            past_source=past_source,
            past_delta_time=self.past_delta_time,
            delta_time=self.delta_time,
            labeling_time=self.labeling_time,
        )


@dataclass(frozen=True)
class DualScaleLossWeights:
    reconstruction: float = 1.0
    new_rna: float = 1.0
    gamma_regularization: float = 1e-3
    future_sliced_wasserstein: float = 1.0
    future_mmd: float = 0.1
    future_gene_sliced_wasserstein: float = 0.0
    past_direction_alignment: float = 0.0
    kinetic_direction_alignment: float = 0.0
    transition_regularization: float = 0.0

    # Legacy fields accepted so old TOML files still parse. They no longer drive
    # the scientific objective in the direction-prior model.
    past_sliced_wasserstein: float = 0.0
    alignment: float = 0.0
    residual_magnitude: float = 0.0
    transition_kl: float = 0.0


@dataclass(frozen=True)
class DualScaleTrainingSchedule:
    local_pretrain_epochs: int = 10
    global_pretrain_epochs: int = 0
    joint_epochs: int = 100
    local_steps_per_pretrain_epoch: int = 100
    global_steps_per_pretrain_epoch: int = 100
    local_steps_per_joint_cycle: int = 1
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
    distribution_space: str = "latent"
    sw_projections: int = 64
    gene_sw_projections: int = 16
    mmd_bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    kinetic_direction_step_hours: float = 0.25
    alignment_start_epoch: int = 0
    detach_alignment_teacher: bool = True
    schedule: DualScaleTrainingSchedule = field(default_factory=DualScaleTrainingSchedule)
    loss_weights: DualScaleLossWeights = field(default_factory=DualScaleLossWeights)

    def __post_init__(self) -> None:
        if self.distribution_space not in {"latent", "gene"}:
            raise ValueError("distribution_space must be 'latent' or 'gene'.")
        if self.global_samples < 1:
            raise ValueError("global_samples must be >= 1.")
        if self.gene_sw_projections < 1:
            raise ValueError("gene_sw_projections must be >= 1.")
        if self.kinetic_direction_step_hours <= 0:
            raise ValueError("kinetic_direction_step_hours must be positive.")


@dataclass
class DualScaleTrainingHistory:
    local: list[dict[str, float]] = field(default_factory=list)
    global_: list[dict[str, float]] = field(default_factory=list)
    epochs: list[dict[str, float]] = field(default_factory=list)


def _log_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(torch.log1p(prediction.clamp_min(0)), torch.log1p(target.clamp_min(0)))


def sliced_wasserstein_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    n_projections: int = 64,
) -> torch.Tensor:
    if prediction.ndim != 2 or target.ndim != 2 or prediction.shape[1] != target.shape[1]:
        raise ValueError("prediction and target must be [n_cells, n_features] with matching features.")
    directions = torch.randn(
        prediction.shape[1], n_projections, device=prediction.device, dtype=prediction.dtype
    )
    directions = F.normalize(directions, dim=0)
    pred_proj, target_proj = prediction @ directions, target @ directions
    n_quantiles = min(prediction.shape[0], target.shape[0])
    quantiles = torch.linspace(0, 1, n_quantiles, device=prediction.device, dtype=prediction.dtype)
    pred_q = torch.quantile(pred_proj, quantiles, dim=0)
    target_q = torch.quantile(target_proj, quantiles, dim=0)
    return F.mse_loss(pred_q, target_q)


def mmd_rbf_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    xx = torch.cdist(prediction, prediction).square()
    yy = torch.cdist(target, target).square()
    xy = torch.cdist(prediction, target).square()
    loss = prediction.new_zeros(())
    for bandwidth in bandwidths:
        scale = 2 * bandwidth * bandwidth
        loss = loss + torch.exp(-xx / scale).mean() + torch.exp(-yy / scale).mean() - 2 * torch.exp(-xy / scale).mean()
    return loss / len(bandwidths)


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

    def _validate_model_contract(self) -> None:
        required = (
            "kinetic_latent_direction",
            "transition_regularization",
            "forward_global",
        )
        missing = [name for name in required if not hasattr(self.model, name)]
        if missing:
            raise RuntimeError(
                "Dual-scale model/trainer interface mismatch before training: "
                f"model is missing {missing}. The kinetic-direction trainer "
                "requires the matching models/dual_scale/model.py."
            )

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
            old=batch.old,
            new=batch.new,
            ntr=batch.ntr,
            labeling_time=batch.labeling_time,
        )
        gamma_base = output.gamma_base.unsqueeze(0).expand_as(output.gamma)
        losses = {
            "reconstruction": _log_mse(output.reconstruction, batch.total),
            "new_rna": _log_mse(output.predicted_new, batch.new),
            "gamma_regularization": F.mse_loss(torch.log(output.gamma), torch.log(gamma_base)),
        }
        weights = self.config.loss_weights
        losses["total"] = (
            weights.reconstruction * losses["reconstruction"]
            + weights.new_rna * losses["new_rna"]
            + weights.gamma_regularization * losses["gamma_regularization"]
        )
        return losses

    @staticmethod
    def _direction_cosine_loss(
        prediction: torch.Tensor,
        teacher: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if prediction.shape != teacher.shape:
            raise ValueError("prediction and teacher directions must have matching shape.")
        valid = (prediction.norm(dim=-1) > eps) & (teacher.norm(dim=-1) > eps)
        if not bool(valid.any()):
            return prediction.new_zeros(())
        return (1 - F.cosine_similarity(prediction[valid], teacher[valid], dim=-1)).mean()

    def global_losses(self, batch: GlobalBatch) -> dict[str, torch.Tensor]:
        batch = batch.resolved(self.device)
        output = self.model.forward_global(
            total=batch.total,
            old=batch.old,
            new=batch.new,
            ntr=batch.ntr,
            delta_time=batch.delta_time,
            labeling_time=batch.labeling_time,
            n_samples=self.config.global_samples,
            kinetic_direction_step_hours=self.config.kinetic_direction_step_hours,
        )
        if self.config.distribution_space == "latent":
            with torch.no_grad():
                target_distribution = self.model.encode_state(batch.target_total)
            predicted_distributions = output.future_latent
        else:
            target_distribution = batch.target_total
            predicted_distributions = output.future_total

        future_sw = torch.stack(
            [sliced_wasserstein_loss(sample, target_distribution, self.config.sw_projections) for sample in predicted_distributions]
        ).mean()
        future_mmd = torch.stack(
            [mmd_rbf_loss(sample, target_distribution, self.config.mmd_bandwidths) for sample in predicted_distributions]
        ).mean()

        # Use the same raw expression representation as the default benchmark
        # when evaluation.space.transform = "none". This prevents the latent
        # encoder from hiding badly calibrated decoded expression values.
        future_gene_sw = torch.stack(
            [
                sliced_wasserstein_loss(
                    sample,
                    batch.target_total,
                    self.config.gene_sw_projections,
                )
                for sample in output.future_total
            ]
        ).mean()

        # Past supervision is directional only: previous-snapshot kinetics should
        # point in the same coarse direction as the observed population shift to
        # the current snapshot. No cell pairing or long-horizon kinetic magnitude
        # is imposed.
        past_direction = batch.total.new_zeros(())
        if batch.past_source is not None:
            past_output = self.model.forward_local(
                total=batch.past_source.total,
                old=batch.past_source.old,
                new=batch.past_source.new,
                ntr=batch.past_source.ntr,
                labeling_time=batch.past_source.labeling_time,
            )
            _, _, past_kinetic_direction = self.model.kinetic_latent_direction(
                total=batch.past_source.total,
                latent=past_output.latent,
                alpha=past_output.alpha,
                gamma=past_output.gamma,
                direction_step_hours=self.config.kinetic_direction_step_hours,
            )
            with torch.no_grad():
                current_latent = self.model.encode_state(batch.total)
                observed_population_direction = (
                    current_latent.mean(dim=0) - past_output.latent.mean(dim=0)
                )
            past_direction = self._direction_cosine_loss(
                past_kinetic_direction.mean(dim=0, keepdim=True),
                observed_population_direction.unsqueeze(0),
            )

        kinetic_direction_alignment = batch.total.new_zeros(())
        if self._epoch >= self.config.alignment_start_epoch:
            teacher = output.kinetic_direction_latent
            if self.config.detach_alignment_teacher:
                teacher = teacher.detach()
            kinetic_direction_alignment = self._direction_cosine_loss(
                output.transition_mean,
                teacher,
            )

        transition_regularization = self.model.transition_regularization(
            output.transition, output.sampled_delta
        )
        gamma_base = output.gamma_base.unsqueeze(0).expand_as(output.gamma)
        gamma_reg = F.mse_loss(torch.log(output.gamma), torch.log(gamma_base))
        losses = {
            "past_direction_alignment": past_direction,
            "future_sliced_wasserstein": future_sw,
            "future_mmd": future_mmd,
            "future_gene_sliced_wasserstein": future_gene_sw,
            "kinetic_direction_alignment": kinetic_direction_alignment,
            "transition_regularization": transition_regularization,
            "gamma_regularization": gamma_reg,
        }
        weights = self.config.loss_weights
        # ``alignment`` remains a legacy alias for kinetic_direction_alignment.
        direction_weight = weights.kinetic_direction_alignment + weights.alignment
        losses["total"] = (
            weights.past_direction_alignment * past_direction
            + weights.future_sliced_wasserstein * future_sw
            + weights.future_mmd * future_mmd
            + weights.future_gene_sliced_wasserstein * future_gene_sw
            + direction_weight * kinetic_direction_alignment
            + (weights.transition_regularization + weights.transition_kl) * transition_regularization
            + weights.gamma_regularization * gamma_reg
        )
        return losses

    @staticmethod
    def _numbers(losses: dict[str, torch.Tensor]) -> dict[str, float]:
        return {name: float(value.detach().cpu()) for name, value in losses.items()}

    def local_step(
        self,
        batch: LocalBatch,
        stage: str = "local",
    ) -> dict[str, float]:
        self.model.train()
        losses = self.local_losses(batch)
        self._optimize(losses["total"])
        values = self._numbers(losses)
        self.history.local.append(values)
        if self.progress is not None:
            self.progress.update(stage=stage, epoch=self._epoch, metrics=values)
        return values

    def global_step(
        self,
        batch: GlobalBatch,
        stage: str = "global",
    ) -> dict[str, float]:
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
        return {f"{kind}_{key}": sum(record[key] for record in records) / len(records) for key in keys}

    def fit(self, local_loader: Iterable, global_loader: Iterable) -> DualScaleTrainingHistory:
        self._validate_model_contract()
        schedule = self.config.schedule
        local_batches, global_batches = _CyclingIterator(local_loader), _CyclingIterator(global_loader)
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
                        ("local", self.local_step(local_batches.next(), stage="joint/local"))
                    )
                for _ in range(schedule.global_steps_per_joint_cycle):
                    records.append(
                        ("global", self.global_step(global_batches.next(), stage="joint/global"))
                    )
            summary: dict[str, float] = {"epoch": float(self._epoch)}
            for kind in ("local", "global"):
                kind_records = [record for record_kind, record in records if record_kind == kind]
                if kind_records:
                    for key in kind_records[0]:
                        summary[f"{kind}_{key}"] = sum(record[key] for record in kind_records) / len(kind_records)
            self.history.epochs.append(summary)
            if self.progress is not None:
                self.progress.epoch(stage="joint", epoch=self._epoch, metrics=summary)

        return self.history
