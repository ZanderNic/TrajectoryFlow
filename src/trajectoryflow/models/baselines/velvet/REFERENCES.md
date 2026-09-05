# Velvet reimplementation notes

This directory contains a clean PyTorch reimplementation for TrajectoryFlow.
It is **not** a verbatim copy of the upstream VelvetVAE repository.

## Primary references

- Rory J. Maizels, Daniel M. Snell, James Briscoe.
  *Reconstructing developmental trajectories using latent dynamical systems
  and time-resolved transcriptomics*. Cell Systems 15 (2024), 411-424.e9.
  DOI: 10.1016/j.cels.2024.04.004

- Official implementation:
  https://github.com/rorymaizels/velvetVAE

- Official analysis code:
  https://github.com/rorymaizels/Maizels2023aa

## Paper-derived behavior implemented here

VelvetVAE:

- total RNA only is encoded by the VAE
- Gaussian latent representation
- default latent dimensionality 50
- linearly decoded count model
- ZINB reconstruction likelihood
- latent vector field
- gene velocity decoded as `psi(z + v_z) - psi(z)`
- metabolic-labeling equation:
  `N = (1 - exp(-gamma*t))/gamma * (dX/dt + gamma*X)`
- stage-1 loss: VAE loss + 10 * labeling/velocity loss
- stage 1: 200 epochs
- stage 2: freeze VAE; labeling loss + neighborhood constraint
- stage 2: 800 epochs
- AdamW, learning rate 1e-3, weight decay 1e-3
- default 100-neighbor constraint
- transition probabilities based on cosine alignment and a softmax

VelvetSDE:

- Velvet vector field is the SDE drift
- constant scalar Brownian diffusion
- Stratonovich/midpoint-style integration
- Markov random walks use the velocity-guided transition matrix
- SDE-vs-Markov timestep distributions are trained with KL divergence
- 250 epochs
- 200 starting cells per epoch
- 50 simulations per cell
- paper-reported training noise magnitude 0.2

## Explicit implementation choices / differences

These are intentionally visible rather than silently presented as upstream facts:

1. The code uses modern plain PyTorch instead of scvi-tools, Lightning and
   the old Velvet package infrastructure.

2. Encoder defaults (`128` hidden units, one hidden layer, `0.1` dropout)
   follow standard scVI defaults. The paper says Velvet builds on scvi-tools,
   but does not enumerate every encoder-layer default in STAR Methods.

3. The vector-field hidden architecture is configurable and defaults to a
   small two-layer SiLU MLP. Treat this as a TrajectoryFlow reimplementation
   choice unless independently verified against the exact upstream source.

4. The paper's velocity loss is written with `log`. This implementation uses
   `log1p` so zero-count genes remain numerically defined.

5. The paper states that neighborhood projection includes a correction for
   non-uniform sampling density, referring to prior methods. That correction
   is not guessed here. Transition probabilities and expected displacements
   follow the published equations, but no unverified density term is added.

6. The SDE uses direct PyTorch autograd through a stochastic midpoint solver
   rather than torchsde adjoint gradients. The forward model remains a
   constant-diffusion midpoint SDE; the gradient-computation strategy differs.

7. Equation 17 only states that timestep simulation distributions are
   Gaussian. This implementation uses diagonal Gaussian covariance for a
   stable, scalable KL objective.

8. `build_velvet_neighbors()` constructs fixed neighbors from a log-normalized
   SVD embedding. The trainer also accepts externally precomputed neighbor
   indices, which should be used if exact reproduction of the authors'
   preprocessing is required.

9. Biological time in hours is not automatically identical to VelvetSDE
   integration time. `VelvetBaseline` therefore requires an explicit
   `hours_per_sde_unit` instead of silently inventing a conversion.

## Attribution

If this implementation is used in a thesis, paper, benchmark or public
repository, cite the Cell Systems paper and link the official VelvetVAE
repository. Describe this code as a reimplementation/adaptation, not as the
authors' official implementation.
