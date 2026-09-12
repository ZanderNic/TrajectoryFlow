# std-lib imports
import hashlib
import os
import platform
import random
import subprocess
import sys
import threading
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from time import perf_counter, process_time

# 3 party imports
import numpy as np
import torch
from torch import nn

# package imports


def stable_seed(seed: int, *parts: object) -> int:
    text = "::".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)


_MIB = 1024**2


def _current_rss_bytes() -> int | None:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as file:
            resident_pages = int(file.read().split()[1])

        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


class _MemorySampler:

    def __init__(self, interval_seconds: float = 0.02):
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes: int | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Memory sampler is already running.")
        self._stop.clear()
        self.peak_bytes = _current_rss_bytes()

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                value = _current_rss_bytes()

                if value is None:
                    continue

                if self.peak_bytes is None or value > self.peak_bytes:
                    self.peak_bytes = value

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

        if self._thread is not None:
            self._thread.join()
            self._thread = None


@dataclass(frozen=True)
class PhaseStats:
    wall_seconds: float
    cpu_seconds: float
    rss_start_mb: float | None
    rss_end_mb: float | None
    rss_peak_mb: float | None
    gpu_start_allocated_mb: float | None
    gpu_peak_allocated_mb: float | None
    gpu_peak_delta_mb: float | None
    gpu_peak_reserved_mb: float | None

    def to_dict(self) -> dict:
        return asdict(self)


class PhaseProfiler(AbstractContextManager):

    def __init__(self, device: torch.device | str | None = None, profile_cuda_memory: bool = True):
        self.device = torch.device(device) if device is not None else None
        self.profile_cuda_memory = profile_cuda_memory
        self.stats: PhaseStats | None = None

    def _cuda_enabled(self) -> bool:
        return self.profile_cuda_memory and self.device is not None and self.device.type == "cuda" and torch.cuda.is_available()

    def __enter__(self):
        if self._cuda_enabled():
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.gpu_start = torch.cuda.memory_allocated(self.device)
        else:
            self.gpu_start = None

        self.rss_start = _current_rss_bytes()
        self.memory_sampler = _MemorySampler()
        self.memory_sampler.start()
        self.wall_start = perf_counter()
        self.cpu_start = process_time()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._cuda_enabled():
            torch.cuda.synchronize(self.device)

        wall_seconds = perf_counter() - self.wall_start
        cpu_seconds = process_time() - self.cpu_start

        self.memory_sampler.stop()
        rss_end = _current_rss_bytes()
        rss_peak = self.memory_sampler.peak_bytes

        if self._cuda_enabled():
            peak_allocated = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)
            peak_delta = max(peak_allocated - (self.gpu_start or 0), 0)
        else:
            peak_allocated = None
            peak_reserved = None
            peak_delta = None

        self.stats = PhaseStats(
            wall_seconds=wall_seconds,
            cpu_seconds=cpu_seconds,
            rss_start_mb=(self.rss_start / _MIB if self.rss_start is not None else None),
            rss_end_mb=(rss_end / _MIB if rss_end is not None else None),
            rss_peak_mb=(rss_peak / _MIB if rss_peak is not None else None),
            gpu_start_allocated_mb=(
                self.gpu_start / _MIB if self.gpu_start is not None else None
            ),
            gpu_peak_allocated_mb=(
                peak_allocated / _MIB if peak_allocated is not None else None
            ),
            gpu_peak_delta_mb=(peak_delta / _MIB if peak_delta is not None else None),
            gpu_peak_reserved_mb=(
                peak_reserved / _MIB if peak_reserved is not None else None
            ),
        )

        return False


@dataclass(frozen=True)
class ParameterStats:
    total: int
    trainable: int


def parameter_stats(model) -> ParameterStats:
    modules = []

    if model is None:
        return ParameterStats(total=0, trainable=0)

    if isinstance(model, nn.Module):
        modules.append(model)
    elif hasattr(model, "__dict__"):
        modules.extend(
            value
            for value in vars(model).values()
            if isinstance(value, nn.Module)
        )

    seen = set()
    total = 0
    trainable = 0

    for module in modules:
        for parameter in module.parameters():
            key = id(parameter)

            if key in seen:
                continue

            seen.add(key)
            total += parameter.numel()

            if parameter.requires_grad:
                trainable += parameter.numel()

    return ParameterStats(total=total, trainable=trainable)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def system_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "git_commit": git_commit(),
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()

    return info
