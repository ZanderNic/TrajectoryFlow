# std-lib imports

# 3 party imports
import numpy as np
import torch
from scipy import sparse

# package imports
from trajectoryflow.models.baselines.velvet.data import VelvetData
from trajectoryflow.training.velvet import VelvetTrainer


def test_trainer_can_skip_sde(monkeypatch):
    data = VelvetData(
        total=sparse.csr_matrix(np.ones((4, 3), dtype=np.float32)),
        new=sparse.csr_matrix(np.ones((4, 3), dtype=np.float32)),
        obs=None,
        timepoints=("5h",),
    )

    trainer = VelvetTrainer(
        data=data,
        device=torch.device("cpu"),
        train_sde=False,
    )

    assert trainer.train_sde is False


def test_microbatching_keeps_one_optimizer_step_per_effective_batch(monkeypatch):
    from trajectoryflow.models.baselines.velvet import VelvetBaseline, VelvetVAEConfig

    data = VelvetData(
        total=sparse.csr_matrix(np.ones((6, 3), dtype=np.float32)),
        new=sparse.csr_matrix(np.ones((6, 3), dtype=np.float32)),
        obs=None,
        timepoints=("5h",),
    )
    model = VelvetBaseline(
        n_genes=3,
        vae_config=VelvetVAEConfig(
            n_hidden=4,
            n_latent=2,
            vector_hidden=4,
            vector_layers=1,
            dropout_rate=0.0,
            stage1_epochs=2,
            stage2_epochs=0,
            initialize_gamma=False,
            batch_size=None,
            microbatch_size=2,
        ),
    )
    trainer = VelvetTrainer(data=data, device="cpu", train_sde=False)
    steps = []
    original_step = torch.optim.AdamW.step

    def recording_step(optimizer, *args, **kwargs):
        steps.append(1)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", recording_step)
    trainer._train_stage1(model)

    assert len(steps) == 2
