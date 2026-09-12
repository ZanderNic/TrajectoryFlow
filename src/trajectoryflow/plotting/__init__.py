# std-lib imports

# 3 party imports

# package imports
from trajectoryflow.plotting.embedding import UmapProjector
from trajectoryflow.plotting.umap import plot_grouped_umap, plot_prediction_comparison, plot_umap
from trajectoryflow.plotting.paper import (
    plot_cbd_transitions,
    plot_cell_type_composition,
    plot_efficiency_summary,
    plot_forecast_comparison,
    plot_metric_comparison,
    plot_velocity_fields,
)


__all__ = [
    "UmapProjector",
    "plot_cbd_transitions",
    "plot_cell_type_composition",
    "plot_efficiency_summary",
    "plot_forecast_comparison",
    "plot_grouped_umap",
    "plot_metric_comparison",
    "plot_prediction_comparison",
    "plot_umap",
    "plot_velocity_fields",
]


def __getattr__(name: str):
    if name == "UmapProjector":
        from trajectoryflow.plotting.embedding import UmapProjector

        return UmapProjector
    raise AttributeError(name)