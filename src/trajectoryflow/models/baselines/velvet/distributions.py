# std-lib imports

# 3 party imports
import torch
import torch.nn.functional as F

# package imports


def negative_binomial_log_prob(
    x: torch.Tensor,
    mean: torch.Tensor,
    inverse_dispersion: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Element-wise NB log probability using mean/inverse-dispersion parameters."""
    mean = mean.clamp_min(eps)
    theta = inverse_dispersion.clamp_min(eps)

    log_theta_mu = torch.log(theta + mean)
    return (
        theta * (torch.log(theta) - log_theta_mu)
        + x * (torch.log(mean) - log_theta_mu)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
    )


def zinb_nll(
    x: torch.Tensor,
    mean: torch.Tensor,
    inverse_dispersion: torch.Tensor,
    dropout_logits: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Element-wise zero-inflated negative-binomial negative log likelihood."""
    nb_log_prob = negative_binomial_log_prob(
        x=x,
        mean=mean,
        inverse_dispersion=inverse_dispersion,
        eps=eps,
    )

    log_pi = -F.softplus(-dropout_logits)
    log_one_minus_pi = -F.softplus(dropout_logits)

    zero_log_prob = torch.logaddexp(
        log_pi,
        log_one_minus_pi
        + negative_binomial_log_prob(
            x=torch.zeros_like(x),
            mean=mean,
            inverse_dispersion=inverse_dispersion,
            eps=eps,
        ),
    )

    nonzero_log_prob = log_one_minus_pi + nb_log_prob
    log_prob = torch.where(x <= eps, zero_log_prob, nonzero_log_prob)

    return -log_prob


def standard_normal_kl(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """Element-wise KL(q(z|x) || N(0, I))."""
    return 0.5 * (mean.pow(2) + log_variance.exp() - 1.0 - log_variance)


def diagonal_gaussian_kl(
    p_samples: torch.Tensor,
    q_samples: torch.Tensor,
    jitter: float = 1e-4,
) -> torch.Tensor:
    """
    KL(N_p || N_q) estimated from simulation samples.

    Samples must have shape [n_simulations, n_items, n_features]. The returned
    tensor has shape [n_items].
    """
    if p_samples.shape != q_samples.shape:
        raise ValueError(
            f"Gaussian sample shapes differ: {p_samples.shape} vs {q_samples.shape}."
        )

    p_mean = p_samples.mean(dim=0)
    q_mean = q_samples.mean(dim=0)

    p_var = p_samples.var(dim=0, unbiased=False).clamp_min(jitter)
    q_var = q_samples.var(dim=0, unbiased=False).clamp_min(jitter)

    kl = 0.5 * (
        torch.log(q_var / p_var)
        + (p_var + (p_mean - q_mean).pow(2)) / q_var
        - 1.0
    )

    return kl.sum(dim=-1)
