# std-lib imports

# 3 party imports
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

# package imports


def build_knn_indices(
    embedding: np.ndarray,
    n_neighbors: int,
    metric: str = "euclidean",
) -> np.ndarray:
    """Build fixed nearest-neighbor indices and remove each cell's self-match."""
    if embedding.ndim != 2:
        raise ValueError("embedding must have shape [n_cells, n_features].")
    if not 1 <= n_neighbors < len(embedding):
        raise ValueError("n_neighbors must be >= 1 and smaller than n_cells.")

    model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    model.fit(embedding)

    indices = model.kneighbors(return_distance=False)

    # sklearn normally returns self as the first neighbor, but remove self
    # robustly instead of relying on that ordering.
    cleaned = np.empty((len(embedding), n_neighbors), dtype=np.int64)

    for row, neighbors in enumerate(indices):
        neighbors = neighbors[neighbors != row]
        if len(neighbors) < n_neighbors:
            raise RuntimeError(f"Could not find {n_neighbors} neighbors for row {row}.")
        cleaned[row] = neighbors[:n_neighbors]

    return cleaned


def transition_probabilities(
    z: torch.Tensor,
    velocity: torch.Tensor,
    neighbor_indices: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Velocity-guided transition probabilities from Velvet Equation 13.

    z:
        [n_cells, n_latent]
    velocity:
        [n_cells, n_latent]
    neighbor_indices:
        [n_cells, n_neighbors]
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    if len(z) != len(velocity) or len(z) != len(neighbor_indices):
        raise ValueError("z, velocity and neighbor_indices must share n_cells.")

    neighbor_z = z[neighbor_indices]
    displacements = neighbor_z - z[:, None, :]

    cosine = F.cosine_similarity(
        displacements,
        velocity[:, None, :],
        dim=-1,
        eps=1e-8,
    )

    return torch.softmax(cosine / (sigma**2), dim=-1)


def neighborhood_projection(
    z: torch.Tensor,
    velocity: torch.Tensor,
    neighbor_z: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Expected local displacement under the velocity-guided transition weights.

    The paper additionally references a correction for non-uniform sampling
    density. That correction is intentionally not guessed here.
    """
    if neighbor_z.ndim != 3:
        raise ValueError("neighbor_z must have shape [batch, n_neighbors, n_latent].")
    if neighbor_z.shape[0] != z.shape[0]:
        raise ValueError("neighbor_z and z must share the batch dimension.")

    displacements = neighbor_z - z[:, None, :]

    cosine = F.cosine_similarity(
        displacements,
        velocity[:, None, :],
        dim=-1,
        eps=1e-8,
    )

    probabilities = torch.softmax(cosine / (sigma**2), dim=-1)
    return (probabilities[..., None] * displacements).sum(dim=1)


def neighborhood_constraint_loss(
    z: torch.Tensor,
    velocity: torch.Tensor,
    neighbor_z: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Velvet Equation 14: 1 - cosine(velocity, neighborhood projection)."""
    projected = neighborhood_projection(
        z=z,
        velocity=velocity,
        neighbor_z=neighbor_z,
        sigma=sigma,
    )

    cosine = F.cosine_similarity(velocity, projected, dim=-1, eps=1e-8)
    return (1.0 - cosine).mean()


@torch.no_grad()
def sample_markov_paths(
    z: torch.Tensor,
    transition_matrix: torch.Tensor,
    neighbor_indices: torch.Tensor,
    start_indices: torch.Tensor,
    n_simulations: int,
    n_steps: int,
) -> torch.Tensor:
    """
    Simulate Velvet Equation 16.

    Returns latent paths with shape:
        [n_steps + 1, n_simulations, n_start_cells, n_latent]
    """
    if n_simulations < 1 or n_steps < 1:
        raise ValueError("n_simulations and n_steps must be >= 1.")

    n_start = len(start_indices)
    current = start_indices[None, :].expand(n_simulations, -1).clone()

    paths = [z[current]]

    for _ in range(n_steps):
        flat_current = current.reshape(-1)
        probabilities = transition_matrix[flat_current]
        candidates = neighbor_indices[flat_current]

        choice = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
        next_flat = candidates.gather(1, choice[:, None]).squeeze(1)

        current = next_flat.reshape(n_simulations, n_start)
        paths.append(z[current])

    return torch.stack(paths, dim=0)


def cubic_interpolate_paths(paths: torch.Tensor, n_steps: int) -> torch.Tensor:
    """
    Catmull-Rom cubic interpolation along path time.

    This provides a pure-PyTorch cubic interpolation of discrete Markov paths.
    Gradients are not required because Markov trajectories are training targets.
    """
    if paths.ndim != 4:
        raise ValueError(
            "paths must have shape [time, n_simulations, n_cells, n_latent]."
        )
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2.")

    source_steps = paths.shape[0]
    if source_steps == n_steps:
        return paths

    positions = torch.linspace(
        0.0,
        source_steps - 1,
        n_steps,
        device=paths.device,
        dtype=paths.dtype,
    )

    i1 = positions.floor().long().clamp(0, source_steps - 1)
    i2 = (i1 + 1).clamp(0, source_steps - 1)
    i0 = (i1 - 1).clamp(0, source_steps - 1)
    i3 = (i2 + 1).clamp(0, source_steps - 1)

    t = (positions - i1.to(positions.dtype)).view(-1, 1, 1, 1)

    p0 = paths[i0]
    p1 = paths[i1]
    p2 = paths[i2]
    p3 = paths[i3]

    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t.square()
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t.pow(3)
    )
