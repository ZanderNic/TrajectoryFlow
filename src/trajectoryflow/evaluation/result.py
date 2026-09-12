# std-lib imports
from dataclasses import dataclass

# 3 party imports
import pandas as pd

# package imports


@dataclass(frozen=True)
class MetricRange:
    lower: float | None
    upper: float | None
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    def __str__(self) -> str:
        left, right = ("[" if self.lower_inclusive else "("), ("]" if self.upper_inclusive else ")")
        lower, upper = ("-∞" if self.lower is None else f"{self.lower:g}"), ("∞" if self.upper is None else f"{self.upper:g}")
        return f"{left}{lower}, {upper}{right}"


@dataclass
class MetricResult:
    name: str
    mean: float
    std: float
    values: list[float]
    higher_is_better: bool
    value_range: MetricRange


@dataclass
class EvaluationReport:
    model_name: str
    source_time: float
    target_time: float
    metrics: dict[str, MetricResult]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"metric": result.name, "mean": result.mean, "std": result.std, "value_range": str(result.value_range), "higher_is_better": result.higher_is_better} for result in self.metrics.values()])
