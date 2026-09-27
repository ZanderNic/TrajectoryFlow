import torch

from trajectoryflow.models.dual_scale import DualScaleModelConfig, build_default_dual_scale_model


def model():
    return build_default_dual_scale_model(
        12,
        DualScaleModelConfig(
            latent_dim=6,
            kinetic_dim=4,
            state_hidden_dims=(16,),
            decoder_hidden_dims=(16,),
            state_projection_dim=8,
            kinetic_hidden_dims=(10,),
            transition_noise_dim=4,
            transition_hidden_dim=12,
            transition_blocks=2,
            time_scale_hours=5.0,
        ),
    )


def test_state_uses_total_and_ntr_while_kinetics_use_total_only():
    m = model()
    total = torch.rand(5, 12) * 3
    ntr_a = torch.rand(5, 12)
    ntr_b = 1 - ntr_a

    z_a = m.encode_state(total, ntr_a)
    z_b = m.encode_state(total, ntr_b)
    k_a = m.encode_kinetics(total)
    k_b = m.encode_kinetics(total.clone())

    assert z_a.shape == (5, 6)
    assert not torch.allclose(z_a, z_b)
    assert torch.allclose(k_a, k_b)


def test_local_new_is_target_not_kinetic_input():
    m = model()
    total = torch.rand(5, 12) * 3
    ntr = torch.rand(5, 12)
    output = m.forward_local(total=total, ntr=ntr, labeling_time=2.0)

    assert output.predicted_new.shape == total.shape
    assert output.alpha.shape == total.shape
    assert output.gamma.shape == total.shape
    assert torch.all(output.predicted_new >= 0)


def test_global_path_decodes_future_population():
    m = model()
    total = torch.rand(5, 12) * 3
    ntr = torch.rand(5, 12)
    output = m.forward_global(total=total, ntr=ntr, delta_time=5.0, n_samples=2)

    assert output.future_latent.shape == (2, 5, 6)
    assert output.future_total.shape == (2, 5, 12)
