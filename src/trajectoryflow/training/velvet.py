# std-lib imports
from dataclasses import dataclass

# 3 party imports
import numpy as np
import torch
from scipy import sparse

# package imports
from trajectoryflow.models.baselines.velvet.baseline import VelvetBaseline
from trajectoryflow.models.baselines.velvet.data import VelvetData, build_velvet_neighbors
from trajectoryflow.models.baselines.velvet.dynamics import (
    estimate_gamma_extreme_regression,
)
from trajectoryflow.models.baselines.velvet.neighborhood import (
    build_knn_indices,
    transition_probabilities,
)
from trajectoryflow.training.base import BaseTrainer


@dataclass
class VelvetTrainingHistory:
    stage1: list[dict[str, float]]
    stage2: list[dict[str, float]]
    sde: list[float]


class VelvetTrainer(BaseTrainer):

    def __init__(
        self,
        data: VelvetData,
        neighbor_indices: np.ndarray | None = None,
        device: torch.device | str | None = None,
    ):
        self.data = data
        self.neighbor_indices = neighbor_indices

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.history = VelvetTrainingHistory(stage1=[], stage2=[], sde=[])

    @staticmethod
    def _require_finite_loss(result, stage: str) -> None:
        metrics = result.detached()
        bad = {
            name: value
            for name, value in metrics.items()
            if not np.isfinite(value)
        }

        if bad:
            raise FloatingPointError(
                f"Non-finite Velvet {stage} loss before optimizer step: {bad}"
            )

    def _batch_indices(
        self,
        n_cells: int,
        batch_size: int | None,
        shuffle: bool = True,
    ):
        if batch_size is None or batch_size >= n_cells:
            yield np.arange(n_cells, dtype=np.int64)
            return

        indices = np.arange(n_cells, dtype=np.int64)

        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n_cells, batch_size):
            yield indices[start : start + batch_size]

    def _dense(
        self,
        matrix: sparse.csr_matrix,
        indices: np.ndarray,
    ) -> torch.Tensor:
        values = matrix[indices].toarray()
        return torch.from_numpy(values).float().to(self.device)

    def _initialize_gamma(self, baseline: VelvetBaseline) -> None:
        config = baseline.velvet.config

        if not config.initialize_gamma:
            return

        n_cells = min(config.gamma_init_cells, self.data.n_cells)
        rng = np.random.default_rng(config.seed)
        indices = rng.choice(self.data.n_cells, size=n_cells, replace=False)

        total = self._dense(self.data.total, indices)
        new = self._dense(self.data.new, indices)

        gamma = estimate_gamma_extreme_regression(
            total=total,
            new=new,
            labelling_time=config.labelling_time,
            quantile=config.gamma_extreme_quantile,
            ratio_eps=config.gamma_ratio_eps,
            default_gamma=config.gamma_default,
        )

        print(
            "[velvet gamma] "
            f"min={gamma.min().item():.6g} "
            f"median={gamma.median().item():.6g} "
            f"max={gamma.max().item():.6g}"
        )

        baseline.velvet.biophysics.set_gamma(gamma)

    def _latent_all(self, baseline: VelvetBaseline) -> torch.Tensor:
        config = baseline.velvet.config
        chunks = []

        baseline.velvet.eval()

        with torch.no_grad():
            for indices in self._batch_indices(
                n_cells=self.data.n_cells,
                batch_size=config.latent_batch_size,
                shuffle=False,
            ):
                total = self._dense(self.data.total, indices)
                z = baseline.velvet.latent_representation(total)
                chunks.append(z.cpu())

        return torch.cat(chunks, dim=0)

    def _train_stage1(self, baseline: VelvetBaseline) -> None:
        model = baseline.velvet
        config = model.config

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        model.train()

        for epoch in range(config.stage1_epochs):
            metrics = []

            for indices in self._batch_indices(
                n_cells=self.data.n_cells,
                batch_size=config.batch_size,
            ):
                total = self._dense(self.data.total, indices)
                new = self._dense(self.data.new, indices)

                optimizer.zero_grad(set_to_none=True)

                result = model.stage1_loss(total=total, new=new)
                self._require_finite_loss(result, stage="stage 1")

                result.loss.backward()
                optimizer.step()
                metrics.append(result.detached())

            mean = {
                key: float(np.mean([item[key] for item in metrics]))
                for key in metrics[0]
            }

            self.history.stage1.append(mean)

            print(
                f"[velvet stage 1] {epoch + 1:4d}/{config.stage1_epochs} "
                f"loss={mean['loss']:.4f} "
                f"vae={(mean['reconstruction'] + mean['kl']):.4f} "
                f"velocity={mean['velocity']:.4f}"
            )

    def _train_stage2(
        self,
        baseline: VelvetBaseline,
        all_z_cpu: torch.Tensor,
        neighbor_indices: np.ndarray,
    ) -> None:
        model = baseline.velvet
        config = model.config

        model.freeze_vae()

        optimizer = torch.optim.AdamW(
            list(model.stage2_parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        model.train()

        for epoch in range(config.stage2_epochs):
            metrics = []

            for indices in self._batch_indices(
                n_cells=self.data.n_cells,
                batch_size=config.batch_size,
            ):
                total = self._dense(self.data.total, indices)
                new = self._dense(self.data.new, indices)

                z = all_z_cpu[indices].to(self.device)
                neighbor_ids = neighbor_indices[indices]
                neighbor_z = all_z_cpu[neighbor_ids].to(self.device)

                optimizer.zero_grad(set_to_none=True)

                result = model.stage2_loss(
                    total=total,
                    new=new,
                    z=z,
                    neighbor_z=neighbor_z,
                )

                self._require_finite_loss(result, stage="stage 2")

                result.loss.backward()
                optimizer.step()

                metrics.append(result.detached())

            mean = {
                key: float(np.mean([item[key] for item in metrics]))
                for key in metrics[0]
            }

            self.history.stage2.append(mean)

            print(
                f"[velvet stage 2] {epoch + 1:4d}/{config.stage2_epochs} "
                f"loss={mean['loss']:.4f} "
                f"velocity={mean['velocity']:.4f} "
                f"neighbor={mean['neighborhood']:.4f}"
            )

    def _train_sde(
        self,
        baseline: VelvetBaseline,
        all_z_cpu: torch.Tensor,
    ) -> None:
        sde = baseline.sde
        config = sde.config

        all_z = all_z_cpu.to(self.device)

        markov_neighbors = build_knn_indices(
            embedding=all_z_cpu.numpy(),
            n_neighbors=config.markov_neighbors,
        )

        markov_neighbors = torch.from_numpy(markov_neighbors).long().to(self.device)

        optimizer = torch.optim.AdamW(
            baseline.velvet.vector_field.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        rng = np.random.default_rng(config.seed)

        for epoch in range(config.epochs):
            optimizer.zero_grad(set_to_none=True)

            # Recompute probabilities because the drift/vector field is updated.
            velocity = baseline.velvet.vector_field(all_z)

            transition_matrix = transition_probabilities(
                z=all_z,
                velocity=velocity,
                neighbor_indices=markov_neighbors,
                sigma=config.transition_sigma,
            )

            n_start = min(config.cells_per_epoch, self.data.n_cells)
            start = rng.choice(self.data.n_cells, size=n_start, replace=False)
            start = torch.from_numpy(start).long().to(self.device)

            loss = sde.training_loss(
                all_z=all_z,
                transition_matrix=transition_matrix,
                neighbor_indices=markov_neighbors,
                start_indices=start,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite Velvet SDE loss before optimizer step: {float(loss.detach())}"
                )

            loss.backward()
            optimizer.step()

            value = float(loss.detach())
            self.history.sde.append(value)

            print(
                f"[velvet SDE]     {epoch + 1:4d}/{config.epochs} "
                f"loss={value:.4f}"
            )

    def fit(self, model: VelvetBaseline) -> None:
        if not isinstance(model, VelvetBaseline):
            raise TypeError("VelvetTrainer requires a VelvetBaseline.")

        torch.manual_seed(model.velvet.config.seed)
        np.random.seed(model.velvet.config.seed)

        model.velvet.to(self.device)

        self._initialize_gamma(model)
        self._train_stage1(model)

        all_z_cpu = self._latent_all(model)

        if self.neighbor_indices is None:
            self.neighbor_indices = build_velvet_neighbors(
                data=self.data,
                n_neighbors=model.velvet.config.n_neighbors,
                seed=model.velvet.config.seed,
            )

        if self.neighbor_indices.shape != (
            self.data.n_cells,
            model.velvet.config.n_neighbors,
        ):
            raise ValueError(
                "neighbor_indices has unexpected shape: "
                f"{self.neighbor_indices.shape}."
            )

        self._train_stage2(
            baseline=model,
            all_z_cpu=all_z_cpu,
            neighbor_indices=self.neighbor_indices,
        )

        self._train_sde(
            baseline=model,
            all_z_cpu=all_z_cpu,
        )

        model._is_fitted = True
