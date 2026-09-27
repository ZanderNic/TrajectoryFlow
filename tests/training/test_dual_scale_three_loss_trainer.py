import torch

from trajectoryflow.models.dual_scale import DualScaleModelConfig, build_default_dual_scale_model
from trajectoryflow.training import (
    DualScaleTrainer,
    DualScaleTrainerConfig,
    KineticPopulation,
    UnpairedGlobalPopulationLoader,
)


def setup():
    n_genes = 10
    total = torch.rand(24, n_genes) * 4
    new = total * torch.rand(24, n_genes) * 0.3
    source = KineticPopulation(total=total, new=new)
    target = torch.rand(30, n_genes) * 4
    loader = UnpairedGlobalPopulationLoader(
        source=source,
        future_target_total=target,
        delta_time=5.0,
        batch_size=6,
        steps_per_epoch=2,
    )
    model = build_default_dual_scale_model(
        n_genes,
        DualScaleModelConfig(
            latent_dim=5,
            kinetic_dim=3,
            state_hidden_dims=(12,),
            decoder_hidden_dims=(12,),
            state_projection_dim=6,
            kinetic_hidden_dims=(8,),
            transition_noise_dim=4,
            transition_hidden_dim=10,
            transition_blocks=2,
            time_scale_hours=5.0,
        ),
    )
    return model, source, loader


def test_exactly_three_training_objectives():
    model, source, global_loader = setup()
    trainer = DualScaleTrainer(model, "cpu", DualScaleTrainerConfig(sw_projections=4))
    local_batch = source.take(torch.arange(6))

    local = trainer.local_losses(local_batch)
    global_ = trainer.global_losses(next(iter(global_loader)))

    assert set(local) == {"reconstruction", "local_kinetic", "total"}
    assert set(global_) == {
        "population_sliced_wasserstein",
        "total",
    }
    assert torch.isfinite(local["reconstruction"])
    assert torch.isfinite(local["local_kinetic"])
    assert torch.isfinite(global_["population_sliced_wasserstein"])


def test_local_kinetic_loss_does_not_update_state_encoder():
    model, source, _ = setup()
    trainer = DualScaleTrainer(model, "cpu", DualScaleTrainerConfig(sw_projections=4))
    batch = source.take(torch.arange(6)).resolved(torch.device("cpu"))
    output = model.forward_local(batch.total, batch.ntr, batch.labeling_time)
    loss = torch.nn.functional.mse_loss(torch.log1p(output.predicted_new), torch.log1p(batch.new))
    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()

    assert all(p.grad is None for p in model.state_encoder.parameters())
    assert any(p.grad is not None for p in model.kinetic_encoder.parameters())


def test_reconstruction_does_not_update_kinetic_encoder():
    model, source, _ = setup()
    trainer = DualScaleTrainer(model, "cpu", DualScaleTrainerConfig(sw_projections=4))
    batch = source.take(torch.arange(6)).resolved(torch.device("cpu"))
    output = model.forward_local(batch.total, batch.ntr, batch.labeling_time)
    loss = torch.nn.functional.mse_loss(torch.log1p(output.reconstruction), torch.log1p(batch.total))
    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()

    assert any(p.grad is not None for p in model.state_encoder.parameters())
    assert all(p.grad is None for p in model.kinetic_encoder.parameters())


def test_population_sw_updates_kinetic_encoder():
    model, _, global_loader = setup()
    trainer = DualScaleTrainer(model, "cpu", DualScaleTrainerConfig(sw_projections=4))
    loss = trainer.global_losses(next(iter(global_loader)))["population_sliced_wasserstein"]
    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()

    assert any(p.grad is not None for p in model.kinetic_encoder.parameters())
    assert any(p.grad is not None for p in model.state_encoder.parameters())
    assert any(p.grad is not None for p in model.transition_model.parameters())
    assert any(p.grad is not None for p in model.state_decoder.parameters())
