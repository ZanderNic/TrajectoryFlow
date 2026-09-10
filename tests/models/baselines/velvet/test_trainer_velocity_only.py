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
