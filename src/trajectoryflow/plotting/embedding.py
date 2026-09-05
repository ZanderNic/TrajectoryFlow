# std-lib imports
from dataclasses import dataclass

# 3 party imports
import numpy as np
from sklearn.decomposition import PCA
import umap


@dataclass
class EmbeddingResult:
    coordinates: np.ndarray
    labels: np.ndarray | None = None
    title: str | None = None


class UmapProjector:

    def __init__(
        self,
        n_pca_components: int = 50,
        n_neighbors: int = 30,
        min_dist: float = 0.3,
        metric: str = "euclidean",
        random_state: int = 42,
    ):
        self.n_pca_components = n_pca_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state

        self.pca = None
        self.umap_model = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, x: np.ndarray) -> "UmapProjector":
        if x.ndim != 2:
            raise ValueError("x must have shape [n_cells, n_features].")

        n_components = min(self.n_pca_components, x.shape[1], x.shape[0])

        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        x_pca = self.pca.fit_transform(x)

        self.umap_model = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )

        self.umap_model.fit(x_pca)
        self._is_fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("UmapProjector must be fitted before transform().")

        if x.ndim != 2:
            raise ValueError("x must have shape [n_cells, n_features].")

        x_pca = self.pca.transform(x)
        return self.umap_model.transform(x_pca)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)