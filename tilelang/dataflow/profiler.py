"""Profiling helpers for Dataflow compiled programs."""

from __future__ import annotations

from dataclasses import dataclass
import statistics
from typing import Any, Literal
from collections.abc import Callable


DataflowReturnMode = Literal["min", "max", "mean", "median"]
DataflowEventScope = Literal["kernel", "launch"]
DEFAULT_DATAFLOW_PROFILE_METRIC = "pure_kernel_event_ms"


@dataclass(frozen=True)
class DataflowProfileResult:
    """Collected Dataflow launch timing samples."""

    metric: str
    samples_ms: tuple[float, ...]
    warmup_iterations: int
    repeat_iterations: int
    setup_timings_ms: dict[str, float]
    last_timings_ms: dict[str, float]
    execution: Any = None
    target_fingerprint: str | None = None
    artifact_fingerprint: str | None = None

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def max_ms(self) -> float:
        return max(self.samples_ms)

    def summary(self, return_mode: DataflowReturnMode = "mean") -> float:
        if return_mode == "mean":
            return self.mean_ms
        if return_mode == "median":
            return self.median_ms
        if return_mode == "min":
            return self.min_ms
        if return_mode == "max":
            return self.max_ms
        raise ValueError(f"Invalid Dataflow profiler return_mode: {return_mode!r}")

    def quantiles(self, quantiles: list[float]) -> list[float]:
        return [compute_quantile(self.samples_ms, quantile) for quantile in quantiles]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "samples_ms": list(self.samples_ms),
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "warmup_iterations": self.warmup_iterations,
            "repeat_iterations": self.repeat_iterations,
            "setup_timings_ms": dict(self.setup_timings_ms),
            "last_timings_ms": dict(self.last_timings_ms),
            "target_fingerprint": self.target_fingerprint,
            "artifact_fingerprint": self.artifact_fingerprint,
        }


@dataclass
class DataflowProfiler:
    """Benchmark a Dataflow compiled program with user-provided tensors."""

    compiled_program: Any

    def profile(
        self,
        *args: Any,
        warmup: float = 25,
        rep: float = 100,
        n_warmup: int = 0,
        n_repeat: int = 0,
        metric: str = DEFAULT_DATAFLOW_PROFILE_METRIC,
        event_scope: DataflowEventScope | None = None,
        stream: Any | None = None,
        copy_global_staging: bool = False,
        flush_l2_cache: bool = True,
        cache_flush_bytes: int = 256_000_000,
        fast_flush: bool = True,
        **kwargs: Any,
    ) -> DataflowProfileResult:
        event_scope = event_scope_for_metric(metric) if event_scope is None else event_scope
        runtime_tensor_args = self.compiled_program.pack_tensor_args(*args, **kwargs)
        executable = self.compiled_program.persistent_executable(*args, **kwargs)
        flush_l2 = make_l2_cache_flush(cache_flush_bytes, fast_flush) if flush_l2_cache else noop_cache_flush
        samples: list[float] = []
        last_timings: dict[str, float] = {}
        execution = None
        with executable.open():
            if n_warmup <= 0 or n_repeat <= 0:
                estimate_ms = self.estimate_ms(
                    executable,
                    runtime_tensor_args.tensor_args_bytes,
                    metric=metric,
                    event_scope=event_scope,
                    stream=stream,
                    copy_global_staging=copy_global_staging,
                    flush_l2=flush_l2,
                )
                if n_warmup <= 0:
                    n_warmup = max(1, int(float(warmup) / estimate_ms))
                if n_repeat <= 0:
                    n_repeat = max(1, int(float(rep) / estimate_ms))

            for _ in range(n_warmup):
                executable.profile_launch(
                    tensor_args_bytes=runtime_tensor_args.tensor_args_bytes,
                    stream=stream,
                    copy_global_staging=copy_global_staging,
                    use_cuda_events=False,
                    event_scope=event_scope,
                )

            for _ in range(n_repeat):
                flush_l2()
                launch_result = executable.profile_launch(
                    tensor_args_bytes=runtime_tensor_args.tensor_args_bytes,
                    stream=stream,
                    copy_global_staging=copy_global_staging,
                    use_cuda_events=True,
                    event_scope=event_scope,
                )
                last_timings = dict(launch_result.timings_ms)
                if metric not in last_timings:
                    raise KeyError(f"Dataflow profiler metric {metric!r} was not recorded; available metrics: {sorted(last_timings)}")
                samples.append(float(last_timings[metric]))
                execution = launch_result.execution

        return DataflowProfileResult(
            metric=metric,
            samples_ms=tuple(samples),
            warmup_iterations=n_warmup,
            repeat_iterations=n_repeat,
            setup_timings_ms=dict(getattr(executable, "setup_timings_ms", {})),
            last_timings_ms=last_timings,
            execution=execution,
            target_fingerprint=getattr(self.compiled_program, "target_fingerprint", None),
            artifact_fingerprint=getattr(self.compiled_program, "artifact_fingerprint", None),
        )

    def do_bench(
        self,
        *args: Any,
        warmup: float = 25,
        rep: float = 100,
        n_warmup: int = 0,
        n_repeat: int = 0,
        metric: str = DEFAULT_DATAFLOW_PROFILE_METRIC,
        event_scope: DataflowEventScope | None = None,
        quantiles: list[float] | None = None,
        return_mode: DataflowReturnMode = "mean",
        stream: Any | None = None,
        copy_global_staging: bool = False,
        flush_l2_cache: bool = True,
        cache_flush_bytes: int = 256_000_000,
        fast_flush: bool = True,
        **kwargs: Any,
    ) -> float | list[float]:
        result = self.profile(
            *args,
            warmup=warmup,
            rep=rep,
            n_warmup=n_warmup,
            n_repeat=n_repeat,
            metric=metric,
            event_scope=event_scope,
            stream=stream,
            copy_global_staging=copy_global_staging,
            flush_l2_cache=flush_l2_cache,
            cache_flush_bytes=cache_flush_bytes,
            fast_flush=fast_flush,
            **kwargs,
        )
        if quantiles is not None:
            values = result.quantiles(quantiles)
            return values[0] if len(values) == 1 else values
        return result.summary(return_mode)

    def estimate_ms(
        self,
        executable: Any,
        tensor_args_bytes: bytes,
        *,
        metric: str,
        event_scope: DataflowEventScope,
        stream: Any | None,
        copy_global_staging: bool,
        flush_l2: Callable[[], Any],
    ) -> float:
        samples = []
        for _ in range(5):
            flush_l2()
            launch_result = executable.profile_launch(
                tensor_args_bytes=tensor_args_bytes,
                stream=stream,
                copy_global_staging=copy_global_staging,
                use_cuda_events=True,
                event_scope=event_scope,
            )
            if metric not in launch_result.timings_ms:
                raise KeyError(
                    f"Dataflow profiler metric {metric!r} was not recorded; available metrics: {sorted(launch_result.timings_ms)}"
                )
            samples.append(float(launch_result.timings_ms[metric]))
        return max(statistics.fmean(samples), 1.0e-6)


def event_scope_for_metric(metric: str) -> DataflowEventScope:
    if metric == "launch_event_ms":
        return "launch"
    return "kernel"


def make_l2_cache_flush(cache_flush_bytes: int, fast_flush: bool) -> Callable[[], Any]:
    if cache_flush_bytes <= 0:
        return noop_cache_flush

    import torch

    element_bytes = 4 if fast_flush else 1
    cache_elements = max(1, int(cache_flush_bytes) // element_bytes)
    cache_dtype = torch.int if fast_flush else torch.int8
    cache = torch.empty(cache_elements, dtype=cache_dtype, device="cuda")
    return cache.zero_


def noop_cache_flush() -> None:
    return None


def compute_quantile(samples: tuple[float, ...], quantile: float) -> float:
    if not 0 <= quantile <= 1:
        raise ValueError(f"quantile must be between 0 and 1, got {quantile}")
    values = sorted(samples)
    if len(values) == 1:
        return values[0]
    position = quantile * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight
