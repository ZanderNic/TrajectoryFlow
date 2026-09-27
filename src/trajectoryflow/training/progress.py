# std-lib imports
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Mapping

# 3 party imports
from tqdm.auto import tqdm

# package imports


_METRIC_ALIASES = {
    "total": "loss",
    "loss": "loss",
    "reconstruction": "rec",
    "local_kinetic": "kin",
    "population_sliced_wasserstein": "popSW",
    "velocity": "vel",
    "neighborhood": "nbr",
}


class TrainingProgressReporter:
    """Live tqdm status plus persistent training logs for one benchmark run."""

    def __init__(
        self,
        output_dir: str | Path,
        label: str,
        enabled: bool | None = None,
        log_every_steps: int = 1,
    ):
        if log_every_steps < 1:
            raise ValueError("log_every_steps must be >= 1.")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label = str(label)
        self.enabled = sys.stderr.isatty() if enabled is None else bool(enabled)
        self.log_every_steps = int(log_every_steps)
        self.metrics_path = self.output_dir / "training_metrics.csv"
        self.events_path = self.output_dir / "training_events.jsonl"
        self.log_path = self.output_dir / "training.log"
        self.summary_path = self.output_dir / "training_summary.json"
        self._start = perf_counter()
        self._step = 0
        self._bar = None
        self._last_metrics: dict[str, float] = {}
        self._stages: list[str] = []
        self._closed = False
        self._metrics_file = self.metrics_path.open("w", newline="", encoding="utf-8")
        self._metrics_writer = csv.writer(self._metrics_file)
        self._events_file = self.events_path.open("w", encoding="utf-8")
        self._log_file = self.log_path.open("w", encoding="utf-8")
        self._write_metrics_header()

    @property
    def step(self) -> int:
        return self._step

    def _write_metrics_header(self) -> None:
        self._metrics_writer.writerow(
            [
                "timestamp_utc",
                "elapsed_seconds",
                "granularity",
                "stage",
                "epoch",
                "step",
                "metric",
                "value",
            ]
        )
        self._metrics_file.flush()

    @staticmethod
    def _numbers(metrics: Mapping[str, float] | None) -> dict[str, float]:
        if not metrics:
            return {}
        values = {}
        for key, value in metrics.items():
            try:
                values[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return values

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _elapsed(self) -> float:
        return perf_counter() - self._start

    def _event(self, kind: str, **payload) -> None:
        event = {
            "timestamp_utc": self._timestamp(),
            "elapsed_seconds": self._elapsed(),
            "kind": kind,
            **payload,
        }
        self._events_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._events_file.flush()

    def _text(self, message: str) -> None:
        elapsed = self._elapsed()
        line = f"[{elapsed:10.1f}s] {message}"
        self._log_file.write(line + "\n")
        self._log_file.flush()

    def _metrics(
        self,
        granularity: str,
        stage: str,
        epoch: int | None,
        step: int,
        metrics: Mapping[str, float],
    ) -> None:
        timestamp = self._timestamp()
        elapsed = self._elapsed()
        for name, value in metrics.items():
            self._metrics_writer.writerow(
                [timestamp, elapsed, granularity, stage, epoch, step, name, value]
            )
        self._metrics_file.flush()

    def start(self, total: int, unit: str = "step") -> None:
        if total < 0:
            raise ValueError("total must be non-negative.")
        if self._bar is not None:
            self._bar.total = total
            self._bar.refresh()
            return
        self._bar = tqdm(
            total=total,
            desc=self.label,
            unit=unit,
            dynamic_ncols=True,
            leave=False,
            position=1,
            disable=not self.enabled,
        )
        self._event("start", label=self.label, total=total, unit=unit)
        self._text(f"START {self.label} total={total} {unit}s")

    def status(self, stage: str, message: str | None = None) -> None:
        if stage not in self._stages:
            self._stages.append(stage)
        if self._bar is not None:
            self._bar.set_description_str(f"{self.label} | {stage}")
        text = stage if message is None else f"{stage}: {message}"
        self._event("status", stage=stage, message=message)
        self._text(text)

    def update(
        self,
        stage: str,
        epoch: int | None,
        metrics: Mapping[str, float] | None = None,
        advance: int = 1,
    ) -> None:
        if advance < 0:
            raise ValueError("advance must be non-negative.")
        values = self._numbers(metrics)
        self._step += advance
        self._last_metrics = values
        if stage not in self._stages:
            self._stages.append(stage)

        if self._bar is not None:
            epoch_text = "" if epoch is None else f" e{epoch}"
            self._bar.set_description_str(f"{self.label} | {stage}{epoch_text}")
            postfix = {}
            for name, value in values.items():
                if name in _METRIC_ALIASES:
                    postfix[_METRIC_ALIASES[name]] = f"{value:.4g}"
            if not postfix:
                postfix = {name: f"{value:.4g}" for name, value in list(values.items())[:4]}
            if postfix:
                self._bar.set_postfix(postfix, refresh=False)
            self._bar.update(advance)

        if self._step % self.log_every_steps == 0 or advance == 0:
            self._metrics("step", stage, epoch, self._step, values)
            self._event("step", stage=stage, epoch=epoch, step=self._step, metrics=values)
            rendered = " ".join(f"{name}={value:.6g}" for name, value in values.items())
            self._text(f"STEP stage={stage} epoch={epoch} step={self._step} {rendered}".rstrip())

    def epoch(
        self,
        stage: str,
        epoch: int,
        metrics: Mapping[str, float] | None = None,
    ) -> None:
        values = self._numbers(metrics)
        self._metrics("epoch", stage, epoch, self._step, values)
        self._event("epoch", stage=stage, epoch=epoch, step=self._step, metrics=values)
        rendered = " ".join(f"{name}={value:.6g}" for name, value in values.items())
        self._text(f"EPOCH stage={stage} epoch={epoch} step={self._step} {rendered}".rstrip())

    def close(self, status: str = "completed", error: str | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        summary = {
            "status": status,
            "error": error,
            "label": self.label,
            "elapsed_seconds": self._elapsed(),
            "steps": self._step,
            "stages": self._stages,
            "last_metrics": self._last_metrics,
            "files": {
                "metrics": str(self.metrics_path.name),
                "events": str(self.events_path.name),
                "log": str(self.log_path.name),
            },
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._event("close", status=status, error=error, step=self._step)
        self._text(f"END status={status} steps={self._step} error={error}")
        self._metrics_file.close()
        self._events_file.close()
        self._log_file.close()

    def manifest(self) -> dict[str, str]:
        return {
            "directory": str(self.output_dir),
            "metrics": str(self.metrics_path),
            "events": str(self.events_path),
            "log": str(self.log_path),
            "summary": str(self.summary_path),
        }
