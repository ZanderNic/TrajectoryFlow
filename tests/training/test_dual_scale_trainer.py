# std-lib imports

# 3 party imports
import torch

# package imports
from trajectoryflow.models.dual_scale import DualScaleModelConfig, build_default_dual_scale_model
from trajectoryflow.training import (
    DualScaleTrainer,
    DualScaleTrainerConfig,
    DualScaleTrainingSchedule,
    KineticPopulation,
    LocalPopulationLoader,
    UnpairedGlobalPopulationLoader,
)


def setup_training():
    n_cells, n_genes = 24, 10
    total = torch.rand(n_cells, n_genes) * 4
    new = total * torch.rand(n_cells, n_genes) * 0.3
    population = KineticPopulation(total=total, new=new)
    local_loader = LocalPopulationLoader(population, batch_size=6, seed=1)
    global_loader = UnpairedGlobalPopulationLoader(
        source=population,
        future_target_total=torch.rand(30, n_genes) * 4,
        past_source=population,
        past_delta_time=2.0,
        delta_time=2.0,
        batch_size=6,
        steps_per_epoch=4,
        seed=2,
    )
    model = build_default_dual_scale_model(
        n_genes,
        DualScaleModelConfig(
            latent_dim=5,
            kinetic_dim=3,
            state_hidden_dims=(12,),
            decoder_hidden_dims=(12,),
            kinetic_projection_dim=4,
            kinetic_hidden_dims=(8,),
            transition_hidden_dims=(10,),
            alignment_hidden_dims=(6,),
        ),
    )
    return model, local_loader, global_loader


def test_trainer_schedule_runs_requested_local_and_global_steps():
    model, local_loader, global_loader = setup_training()
    schedule = DualScaleTrainingSchedule(
        local_pretrain_epochs=1,
        global_pretrain_epochs=1,
        joint_epochs=1,
        local_steps_per_pretrain_epoch=2,
        global_steps_per_pretrain_epoch=3,
        local_steps_per_joint_cycle=2,
        global_steps_per_joint_cycle=1,
        joint_cycles_per_epoch=2,
    )
    trainer = DualScaleTrainer(
        model,
        "cpu",
        DualScaleTrainerConfig(
            sw_projections=4,
            schedule=schedule,
        ),
    )
    history = trainer.fit(local_loader, global_loader)

    assert len(history.local) == 2 + 2 * 2
    assert len(history.global_) == 3 + 1 * 2
    assert len(history.epochs) == 3


def test_global_loader_samples_unpaired_populations_with_expected_shapes():
    _, _, global_loader = setup_training()
    batch = next(iter(global_loader))

    assert batch.total.shape == (6, 10)
    assert batch.target_total.shape == (6, 10)
    assert batch.past_source.total.shape == (6, 10)
    assert batch.past_delta_time == 2.0
    assert batch.delta_time == 2.0


def test_global_loss_is_finite_in_latent_and_gene_space():
    model, _, global_loader = setup_training()
    batch = next(iter(global_loader))

    for space in ("latent", "gene"):
        trainer = DualScaleTrainer(
            model,
            "cpu",
            DualScaleTrainerConfig(sw_projections=4, distribution_space=space),
        )
        losses = trainer.global_losses(batch)
        assert all(torch.isfinite(value) for value in losses.values())
        assert losses["past_direction_alignment"] >= 0
        assert losses["future_gene_sliced_wasserstein"] >= 0
        assert losses["kinetic_direction_alignment"] >= 0


def test_sparse_population_only_densifies_sampled_rows():
    from scipy import sparse

    from trajectoryflow.training import SparseKineticPopulation

    total = sparse.csr_matrix(torch.rand(20, 10).numpy())
    new = total.multiply(0.2).tocsr()
    population = SparseKineticPopulation(total=total, new=new)
    batch = population.take(torch.tensor([1, 3, 7]))

    assert batch.total.shape == (3, 10)
    assert batch.new.shape == (3, 10)
    assert isinstance(batch.total, torch.Tensor)


def test_mixed_loader_can_mix_multiple_transition_loaders():
    from trajectoryflow.training import MixedBatchLoader

    _, _, loader_a = setup_training()
    _, _, loader_b = setup_training()
    mixed = MixedBatchLoader((loader_a, loader_b), steps_per_epoch=5, seed=4)

    batches = list(mixed)
    assert len(batches) == 5
    assert all(batch.target_total.shape == (6, 10) for batch in batches)


def test_global_step_updates_diffusion_transition_parameters():
    model, _, global_loader = setup_training()
    trainer = DualScaleTrainer(
        model,
        "cpu",
        DualScaleTrainerConfig(sw_projections=4),
    )
    before = [parameter.detach().clone() for parameter in model.transition_model.parameters()]
    trainer.global_step(next(iter(global_loader)))
    after = list(model.transition_model.parameters())
    assert any(not torch.allclose(old, new.detach()) for old, new in zip(before, after))


def test_kinetic_direction_alignment_is_a_direction_only_teacher():
    model, _, global_loader = setup_training()
    trainer = DualScaleTrainer(
        model,
        "cpu",
        DualScaleTrainerConfig(
            sw_projections=4,
            kinetic_direction_step_hours=0.25,
        ),
    )
    batch = next(iter(global_loader)).resolved(torch.device("cpu"))
    output = model.forward_global(
        total=batch.total,
        old=batch.old,
        new=batch.new,
        ntr=batch.ntr,
        delta_time=5.0,
        labeling_time=batch.labeling_time,
        n_samples=1,
        kinetic_direction_step_hours=0.25,
    )
    assert output.kinetic_direction_latent.shape == output.transition_mean.shape
    loss = trainer._direction_cosine_loss(
        output.transition_mean,
        output.kinetic_direction_latent.detach(),
    )
    assert torch.isfinite(loss)
    assert 0 <= float(loss.detach()) <= 2
