"""Small opt-in progress logger for long Dataflow compile and launch phases."""

from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any

from .compile_config import capture_compile_environment


_PROGRESS_ENV_NAMES = ("DATAFLOW_PROGRESS", "DATAFLOW_COMPILE_PROGRESS", "DATAFLOW_COMPILE_LOG")


class DataflowProgressLogger:
    def __init__(self, enabled: bool, *, prefix: str) -> None:
        self.enabled = enabled
        self.prefix = prefix

    def start(self, label: str) -> float:
        started_at = time.perf_counter()
        if self.enabled:
            print(f"{self.prefix} {label} started", flush=True)
        return started_at

    def finish(self, label: str, started_at: float, detail: str | None = None) -> None:
        if not self.enabled:
            return
        suffix = "" if not detail else f"; {detail}"
        print(f"{self.prefix} {label} finished in {elapsed_s(started_at):.2f}s{suffix}", flush=True)

    def message(self, message: str) -> None:
        if self.enabled:
            print(f"{self.prefix} {message}", flush=True)


def dataflow_progress_enabled(
    options: dict[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
) -> bool:
    value = options.get("progress", options.get("log_progress"))
    if value is None:
        environment = capture_compile_environment(environment)
        for env_name in _PROGRESS_ENV_NAMES:
            if env_name in environment:
                value = environment[env_name]
                break
    return as_bool(value, default=False)


def as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    raise TypeError(f"Dataflow progress option must be a bool-like value, got {value!r}")


def elapsed_s(started_at: float) -> float:
    return time.perf_counter() - started_at
