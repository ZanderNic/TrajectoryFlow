# std-lib imports

# 3 party imports
import numpy as np
import umap
from sklearn.decomposition import PCA

# package imports


class UmapProjector:
    def __init__(self, n_pca_components: int = 50, n_neighbors: int = 30, min_dist: float = 0.3, metric: str = "euclidean", random_state: int = 42):
        if n_pca_components < 1 or n_neighbors < 2:
            raise ValueError("n_pca_components must be >= 1 and n_neighbors must be >= 2.")
        if min_dist < 0 or not metric:
            raise ValueError("min_dist must be >= 0 and metric must not be empty.")
        self.n_pca_components, self.n_neighbors, self.min_dist = n_pca_components, n_neighbors, min_dist
        self.metric, self.random_state = metric, random_state
        self.pca: PCA | None = None
        self.umap_model = None

    @property
    def is_fitted(self) -> bool:
        return self.pca is not None and self.umap_model is not None

    @staticmethod
    def _validate(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim != 2 or 0 in x.shape:
            raise ValueError("x must have non-empty shape [n_cells, n_features].")
        if not np.isfinite(x).all():
            raise ValueError("x contains non-finite values.")
        return x

    def fit(self, x: np.ndarray) -> "UmapProjector":
        x = self._validate(x)
        if len(x) < 3:
            raise ValueError("UMAP requires at least three cells.")
        n_components = min(self.n_pca_components, x.shape[0], x.shape[1])
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        x_pca = self.pca.fit_transform(x)
        self.umap_model = umap.UMAP(n_neighbors=min(self.n_neighbors, len(x) - 1), min_dist=self.min_dist, metric=self.metric, random_state=self.random_state)
        self.umap_model.fit(x_pca)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("UmapProjector must be fitted before transform().")
        x = self._validate(x)
        return self.umap_model.transform(self.pca.transform(x))

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)
