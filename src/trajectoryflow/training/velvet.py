# std-lib imports
from dataclasses import dataclass, field

# 3 party imports
import numpy as np
import torch
from scipy import sparse

# package imports
from trajectoryflow.models.baselines.velvet.baseline import VelvetBaseline
from trajectoryflow.models.baselines.velvet.data import VelvetData, build_velvet_neighbors
from trajectoryflow.models.baselines.velvet.dynamics import estimate_gamma_extreme_regression
from trajectoryflow.models.baselines.velvet.neighborhood import build_knn_indices, transition_probabilities
from trajectoryflow.training.base import BaseTrainer


@dataclass
class VelvetTrainingHistory:
    stage1: list[dict[str, float]] = field(default_factory=list)
    stage2: list[dict[str, float]] = field(default_factory=list)
    sde: list[float] = field(default_factory=list)


class VelvetTrainer(BaseTrainer):
    def __init__(self, data: VelvetData, neighbor_indices: np.ndarray | None = None, device: torch.device | str | None = None, train_sde: bool = True, verbose: bool = False):
        if not isinstance(train_sde, bool) or not isinstance(verbose, bool):
            raise ValueError("train_sde and verbose must be boolean.")
        self.data = data
        self.neighbor_indices = neighbor_indices
        self.train_sde = train_sde
        self.verbose = verbose
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.history = VelvetTrainingHistory()
        self._rng = np.random.default_rng()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    @staticmethod
    def _require_finite_loss(result, stage: str) -> None:
        bad = {name: value for name, value in result.detached().items() if not np.isfinite(value)}
        if bad:
            raise FloatingPointError(f"Non-finite Velvet {stage} loss before optimizer step: {bad}")

    def _batch_indices(self, n_cells: int, batch_size: int | None, shuffle: bool = True):
        if batch_size is None or batch_size >= n_cells:
            yield np.arange(n_cells, dtype=np.int64)
            return
        indices = self._rng.permutation(n_cells) if shuffle else np.arange(n_cells, dtype=np.int64)
        for start in range(0, n_cells, batch_size):
            yield indices[start : start + batch_size]

    def _dense(self, matrix: sparse.csr_matrix, indices: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(matrix[indices].toarray()).float().to(self.device)

    def _initialize_gamma(self, baseline: VelvetBaseline) -> None:
        config = baseline.velvet.config
        if not config.initialize_gamma:
            return
        n_cells = min(config.gamma_init_cells, self.data.n_cells)
        indices = self._rng.choice(self.data.n_cells, size=n_cells, replace=False)
        gamma = estimate_gamma_extreme_regression(total=self._dense(self.data.total, indices), new=self._dense(self.data.new, indices), labelling_time=config.labelling_time, quantile=config.gamma_extreme_quantile, ratio_eps=config.gamma_ratio_eps, default_gamma=config.gamma_default)
        self._log(f"[velvet gamma] min={gamma.min().item():.6g} median={gamma.median().item():.6g} max={gamma.max().item():.6g}")
        baseline.velvet.biophysics.set_gamma(gamma)

    def _latent_all(self, baseline: VelvetBaseline) -> torch.Tensor:
        chunks = []
        baseline.velvet.eval()
        with torch.no_grad():
            for indices in self._batch_indices(self.data.n_cells, baseline.velvet.config.latent_batch_size, shuffle=False):
                chunks.append(baseline.velvet.latent_representation(self._dense(self.data.total, indices)).cpu())
        return torch.cat(chunks, dim=0)

    def _train_stage1(self, baseline: VelvetBaseline) -> None:
        model, config = baseline.velvet, baseline.velvet.config
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        model.train()
        for epoch in range(config.stage1_epochs):
            metrics = []
            for indices in self._batch_indices(self.data.n_cells, config.batch_size):
                optimizer.zero_grad(set_to_none=True)
                result = model.stage1_loss(total=self._dense(self.data.total, indices), new=self._dense(self.data.new, indices))
                self._require_finite_loss(result, "stage 1")
                result.loss.backward()
                optimizer.step()
                metrics.append(result.detached())
            mean = {key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]}
            self.history.stage1.append(mean)
            self._log(f"[velvet stage 1] {epoch + 1:4d}/{config.stage1_epochs} loss={mean['loss']:.4f} vae={(mean['reconstruction'] + mean['kl']):.4f} velocity={mean['velocity']:.4f}")

    def _train_stage2(self, baseline: VelvetBaseline, all_z_cpu: torch.Tensor, neighbor_indices: np.ndarray) -> None:
        model, config = baseline.velvet, baseline.velvet.config
        model.freeze_vae()
        optimizer = torch.optim.AdamW(list(model.stage2_parameters()), lr=config.lr, weight_decay=config.weight_decay)
        model.train()
        for epoch in range(config.stage2_epochs):
            metrics = []
            for indices in self._batch_indices(self.data.n_cells, config.batch_size):
                total, new = self._dense(self.data.total, indices), self._dense(self.data.new, indices)
                z = all_z_cpu[indices].to(self.device)
                neighbor_z = all_z_cpu[neighbor_indices[indices]].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                result = model.stage2_loss(total=total, new=new, z=z, neighbor_z=neighbor_z)
                self._require_finite_loss(result, "stage 2")
                result.loss.backward()
                optimizer.step()
                metrics.append(result.detached())
            mean = {key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]}
            self.history.stage2.append(mean)
            self._log(f"[velvet stage 2] {epoch + 1:4d}/{config.stage2_epochs} loss={mean['loss']:.4f} velocity={mean['velocity']:.4f} neighbor={mean['neighborhood']:.4f}")

    def _train_sde(self, baseline: VelvetBaseline, all_z_cpu: torch.Tensor) -> None:
        sde, config = baseline.sde, baseline.sde.config
        all_z = all_z_cpu.to(self.device)
        markov_neighbors = torch.from_numpy(build_knn_indices(all_z_cpu.numpy(), config.markov_neighbors)).long().to(self.device)
        optimizer = torch.optim.AdamW(baseline.velvet.vector_field.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        for epoch in range(config.epochs):
            optimizer.zero_grad(set_to_none=True)
            velocity = baseline.velvet.vector_field(all_z)
            transition_matrix = transition_probabilities(all_z, velocity, markov_neighbors, sigma=config.transition_sigma)
            n_start = min(config.cells_per_epoch, self.data.n_cells)
            start = torch.from_numpy(self._rng.choice(self.data.n_cells, size=n_start, replace=False)).long().to(self.device)
            loss = sde.training_loss(all_z=all_z, transition_matrix=transition_matrix, neighbor_indices=markov_neighbors, start_indices=start)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite Velvet SDE loss before optimizer step: {float(loss.detach())}")
            loss.backward()
            optimizer.step()
            value = float(loss.detach())
            self.history.sde.append(value)
            self._log(f"[velvet SDE] {epoch + 1:4d}/{config.epochs} loss={value:.4f}")

    def _fit_seeded(self, model: VelvetBaseline) -> None:
        self._initialize_gamma(model)
        self._train_stage1(model)
        all_z_cpu = self._latent_all(model)
        if self.neighbor_indices is None:
            self.neighbor_indices = build_velvet_neighbors(data=self.data, n_neighbors=model.velvet.config.n_neighbors, seed=model.velvet.config.seed)
        expected_shape = (self.data.n_cells, model.velvet.config.n_neighbors)
        if self.neighbor_indices.shape != expected_shape:
            raise ValueError(f"neighbor_indices has shape {self.neighbor_indices.shape}, expected {expected_shape}.")
        self._train_stage2(model, all_z_cpu, self.neighbor_indices)
        if self.train_sde:
            self._train_sde(model, all_z_cpu)

    def fit(self, model: VelvetBaseline) -> None:
        if not isinstance(model, VelvetBaseline):
            raise TypeError("VelvetTrainer requires a VelvetBaseline.")
        seed = model.velvet.config.seed
        self.history = VelvetTrainingHistory()
        self._rng = np.random.default_rng(seed)
        model.velvet.to(self.device)

        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            self._fit_seeded(model)
        model._is_fitted = True
