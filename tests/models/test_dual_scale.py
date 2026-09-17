# std-lib imports

# 3 party imports
import torch

# package imports
from trajectoryflow.models.dual_scale import DualScaleModelConfig, build_default_dual_scale_model


def small_model(n_genes: int = 12):
    return build_default_dual_scale_model(
        n_genes,
        DualScaleModelConfig(
            latent_dim=6,
            kinetic_dim=4,
            state_hidden_dims=(16,),
            decoder_hidden_dims=(16,),
            kinetic_projection_dim=5,
            kinetic_hidden_dims=(10,),
            transition_hidden_dims=(12,),
            alignment_hidden_dims=(8,),
        ),
    )


def kinetic_inputs(n_cells: int = 7, n_genes: int = 12):
    total = torch.rand(n_cells, n_genes) * 5
    new = total * torch.rand(n_cells, n_genes) * 0.4
    old = total - new
    ntr = new / total.clamp_min(1e-8)
    return total, old, new, ntr


def test_local_forward_has_expected_shapes_and_positive_rates():
    model = small_model()
    total, old, new, ntr = kinetic_inputs()
    output = model.forward_local(total, old, new, ntr, labeling_time=2.0)

    assert output.latent.shape == (7, 6)
    assert output.kinetic.shape == (7, 4)
    assert output.reconstruction.shape == total.shape
    assert output.alpha.shape == total.shape
    assert output.gamma.shape == total.shape
    assert output.predicted_new.shape == total.shape
    assert torch.all(output.alpha > 0)
    assert torch.all(output.gamma > 0)
    assert torch.all(output.predicted_new >= 0)


def test_global_forward_is_stochastic_and_decodes_future_population():
    model = small_model()
    total, old, new, ntr = kinetic_inputs()
    output = model.forward_global(
        total,
        old,
        new,
        ntr,
        delta_time=2.0,
        labeling_time=2.0,
        n_samples=3,
    )

    assert output.sampled_delta.shape == (3, 7, 6)
    assert output.sampled_residual.shape == (3, 7, 6)
    assert output.kinetic_future_total.shape == total.shape
    assert output.kinetic_future_latent.shape == (7, 6)
    assert output.kinetic_delta.shape == (7, 6)
    assert output.future_latent.shape == (3, 7, 6)
    assert output.future_total.shape == (3, 7, 12)
    assert output.reconstructed_past.shape == total.shape
    assert output.alignment_delta.shape == (7, 6)


def test_new_rna_equation_matches_expected_limit():
    alpha = torch.tensor([[2.0]])
    gamma = torch.tensor([[0.5]])
    predicted = small_model().predict_new_rna(alpha, gamma, labeling_time=2.0)
    expected = alpha / gamma * (1 - torch.exp(-gamma * 2.0))
    assert torch.allclose(predicted, expected)


def test_kinetic_future_equation_matches_analytic_solution():
    total = torch.tensor([[3.0]])
    alpha = torch.tensor([[2.0]])
    gamma = torch.tensor([[0.5]])
    predicted = small_model().predict_kinetic_future_total(total, alpha, gamma, delta_time=2.0)
    decay = torch.exp(-gamma * 2.0)
    expected = total * decay + alpha / gamma * (1 - decay)
    assert torch.allclose(predicted, expected)


def test_future_is_free_diffusion_from_source_latent():
    model = small_model()
    total, old, new, ntr = kinetic_inputs(n_cells=4)
    output = model.forward_global(
        total,
        old,
        new,
        ntr,
        delta_time=2.0,
        labeling_time=2.0,
        n_samples=2,
    )
    expected = output.latent.unsqueeze(0) + output.sampled_delta
    assert torch.allclose(output.future_latent, expected)
    assert torch.allclose(output.sampled_residual, output.sampled_delta)


def test_kinetic_direction_is_short_horizon_not_forecast_horizon():
    model = small_model()
    total, old, new, ntr = kinetic_inputs(n_cells=4)
    output = model.forward_global(
        total, old, new, ntr, delta_time=5.0, labeling_time=2.0, n_samples=1,
        kinetic_direction_step_hours=0.25,
    )
    expected = model.predict_kinetic_future_total(total, output.alpha, output.gamma, 0.25)
    assert torch.allclose(output.kinetic_direction_total, expected)
    assert output.kinetic_direction_latent.shape == output.latent.shape


def test_diffusion_transition_is_default_and_differentiable():
    model = small_model()
    assert model.transition_name == "latent_diffusion_delta_with_kinetic_direction_prior"
    total, old, new, ntr = kinetic_inputs(n_cells=4)
    output = model.forward_global(
        total,
        old,
        new,
        ntr,
        delta_time=2.0,
        labeling_time=2.0,
        n_samples=2,
    )
    loss = output.future_latent.square().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.transition_model.parameters()
        if parameter.requires_grad
    ]
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_diffusion_deterministic_sampling_is_repeatable():
    model = small_model()
    total, old, new, ntr = kinetic_inputs(n_cells=4)
    first = model.forward_global(
        total,
        old,
        new,
        ntr,
        delta_time=2.0,
        labeling_time=2.0,
        n_samples=1,
        deterministic=True,
    ).sampled_delta
    second = model.forward_global(
        total,
        old,
        new,
        ntr,
        delta_time=2.0,
        labeling_time=2.0,
        n_samples=1,
        deterministic=True,
    ).sampled_delta
    assert torch.allclose(first, second)
