"""Immutable Dataflow compile inputs and compile-environment snapshots."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from typing import Any

from tilelang.utils.target_capabilities import TargetCapabilitySnapshot

from .scheduler_config import DataflowSchedulerConfig
from .semantic_config import DataflowSemanticConfig
from .operation_contracts import DataflowOperationRequest
from .topology import GPUTopology


DATAFLOW_COMPILE_MODE_EXECUTABLE = "executable"
DATAFLOW_COMPILE_MODE_INSPECT = "inspect"
DATAFLOW_COMPILE_MODE_DEBUG = "debug"
DATAFLOW_COMPILE_MODES = (
    DATAFLOW_COMPILE_MODE_EXECUTABLE,
    DATAFLOW_COMPILE_MODE_INSPECT,
)
_DATAFLOW_INTERNAL_COMPILE_MODES = (
    *DATAFLOW_COMPILE_MODES,
    DATAFLOW_COMPILE_MODE_DEBUG,
)

_COMPILE_ENV_NAMES = frozenset(
    (
        "DATAFLOW_COMPILE_LOG",
        "DATAFLOW_COMPILE_PROGRESS",
        "DATAFLOW_PROGRESS",
        "TILELANG_DATAFLOW_WRAPPER_COMPILE_ARCH",
    )
)
_ACTIVE_TARGET_CAPABILITIES: ContextVar[TargetCapabilitySnapshot | None] = ContextVar(
    "tilelang_dataflow_target_capabilities",
    default=None,
)


@dataclass(frozen=True)
class FrozenMapping(Mapping[Any, Any]):
    items: tuple[tuple[Any, Any], ...] = ()

    def __getitem__(self, key: Any) -> Any:
        for item_key, value in self.items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[Any]:
        return (key for key, _ in self.items)

    def __len__(self) -> int:
        return len(self.items)


@dataclass(frozen=True)
class FrozenSequence:
    kind: str
    items: tuple[Any, ...]


@dataclass(frozen=True)
class DataflowCompileConfig:
    """A canonical, immutable snapshot of every Dataflow compile input.

    Container values are recursively frozen when the snapshot is constructed.
    ``canonical_json`` and ``fingerprint`` are therefore stable even if the
    caller later mutates the dictionaries used to create the kernel spec.
    """

    mode: str
    handler_lowering: str
    lower_primfunc_handlers: bool
    link_primfunc_handlers: bool
    topology: GPUTopology
    range_lengths: FrozenMapping
    range_offsets: FrozenMapping | None
    block_size: int
    task_extents: tuple[int, ...] | None
    include_exit: bool
    target: str
    arch: str
    target_capabilities: TargetCapabilitySnapshot
    scheduler_policy: str
    reduce_strategy: str
    scheduler_config: DataflowSchedulerConfig
    semantic_config: DataflowSemanticConfig
    options: FrozenMapping
    environment: FrozenMapping
    provenance: FrozenMapping
    factory_arguments: FrozenMapping
    program_fingerprint: str
    schema_version: int = 20
    canonical_json: str = ""
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.mode not in _DATAFLOW_INTERNAL_COMPILE_MODES:
            expected = ", ".join(_DATAFLOW_INTERNAL_COMPILE_MODES)
            raise ValueError(f"Unsupported Dataflow compile mode {self.mode!r}; expected one of: {expected}")
        payload = self.canonical_payload()
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "canonical_json", canonical_json)
        object.__setattr__(
            self,
            "fingerprint",
            hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
        )

    @classmethod
    def create(
        cls,
        *,
        mode: str,
        handler_lowering: str,
        lower_primfunc_handlers: bool,
        link_primfunc_handlers: bool,
        topology: GPUTopology,
        range_lengths: Mapping[Any, Any],
        range_offsets: Mapping[Any, Any] | None,
        block_size: int,
        task_extents: Any,
        include_exit: bool,
        target_capabilities: TargetCapabilitySnapshot,
        scheduler_policy: str,
        reduce_strategy: str,
        scheduler_config: DataflowSchedulerConfig,
        semantic_config: DataflowSemanticConfig,
        options: Mapping[str, Any],
        environment: Mapping[str, str],
        provenance: Mapping[str, str],
        factory_arguments: Mapping[str, Any] | None = None,
        program_fingerprint: str,
    ) -> DataflowCompileConfig:
        return cls(
            mode=str(mode),
            handler_lowering=str(handler_lowering),
            lower_primfunc_handlers=bool(lower_primfunc_handlers),
            link_primfunc_handlers=bool(link_primfunc_handlers),
            topology=topology,
            range_lengths=freeze_mapping(range_lengths),
            range_offsets=(None if range_offsets is None else freeze_mapping(range_offsets)),
            block_size=int(block_size),
            task_extents=(None if task_extents is None else tuple(int(item) for item in task_extents)),
            include_exit=bool(include_exit),
            target=target_capabilities.target,
            arch=target_capabilities.arch,
            target_capabilities=target_capabilities,
            scheduler_policy=str(scheduler_policy),
            reduce_strategy=str(reduce_strategy),
            scheduler_config=scheduler_config,
            semantic_config=semantic_config,
            options=freeze_mapping(options),
            environment=freeze_mapping(environment),
            provenance=freeze_mapping(provenance),
            factory_arguments=freeze_mapping(factory_arguments or {}),
            program_fingerprint=str(program_fingerprint),
        )

    def options_dict(self) -> dict[str, Any]:
        return thaw_mapping(self.options)

    def environment_dict(self) -> dict[str, str]:
        return {str(key): str(value) for key, value in thaw_mapping(self.environment).items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "mode": self.mode,
            "handler": {
                "lowering": self.handler_lowering,
                "lower_primfunc_handlers": self.lower_primfunc_handlers,
                "link_primfunc_handlers": self.link_primfunc_handlers,
            },
            "topology": {
                "sm_count": self.topology.sm_count,
                "cluster_size": self.topology.cluster_size,
            },
            "range_lengths": thaw_mapping(self.range_lengths),
            "range_offsets": (None if self.range_offsets is None else thaw_mapping(self.range_offsets)),
            "block_size": self.block_size,
            "task_extents": self.task_extents,
            "include_exit": self.include_exit,
            "target": self.target,
            "arch": self.arch,
            "target_capabilities": self.target_capabilities.to_dict(),
            "scheduler_policy": self.scheduler_policy,
            "reduce_strategy": self.reduce_strategy,
            "scheduler_config": self.scheduler_config.to_dict(),
            "semantic_config": self.semantic_config.to_dict(),
            "options": self.options_dict(),
            "environment": self.environment_dict(),
            "provenance": thaw_mapping(self.provenance),
            "factory_arguments": thaw_mapping(self.factory_arguments),
            "program_fingerprint": self.program_fingerprint,
        }

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "handler_lowering": self.handler_lowering,
            "lower_primfunc_handlers": self.lower_primfunc_handlers,
            "link_primfunc_handlers": self.link_primfunc_handlers,
            "topology": canonical_value(self.topology),
            "range_lengths": canonical_value(self.range_lengths),
            "range_offsets": canonical_value(self.range_offsets),
            "block_size": self.block_size,
            "task_extents": canonical_value(self.task_extents),
            "include_exit": self.include_exit,
            "target": self.target,
            "arch": self.arch,
            "target_capabilities": self.target_capabilities.to_dict(),
            "scheduler_policy": self.scheduler_policy,
            "reduce_strategy": self.reduce_strategy,
            "scheduler_config": canonical_value(self.scheduler_config),
            "semantic_config": canonical_value(self.semantic_config),
            "options": canonical_value(self.options),
            "environment": canonical_value(self.environment),
            "provenance": canonical_value(self.provenance),
            "factory_arguments": canonical_value(self.factory_arguments),
            "program_fingerprint": self.program_fingerprint,
        }


def capture_compile_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Read the remaining non-policy compile controls exactly once."""

    source = os.environ if environment is None else environment
    return {str(name): str(value) for name, value in sorted(source.items()) if name in _COMPILE_ENV_NAMES}


@contextmanager
def use_target_capabilities(target_capabilities: TargetCapabilitySnapshot):
    """Expose the one compile-boundary target snapshot during program construction."""

    if not isinstance(target_capabilities, TargetCapabilitySnapshot):
        raise TypeError(f"use_target_capabilities expects TargetCapabilitySnapshot, got {type(target_capabilities)!r}")
    token = _ACTIVE_TARGET_CAPABILITIES.set(target_capabilities)
    try:
        yield
    finally:
        _ACTIVE_TARGET_CAPABILITIES.reset(token)


def current_target_capabilities() -> TargetCapabilitySnapshot | None:
    """Return the active compiler-owned target snapshot, if construction is in progress."""

    return _ACTIVE_TARGET_CAPABILITIES.get()


def canonical_fingerprint(value: Any) -> str:
    canonical_json = json.dumps(
        canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def freeze_mapping(value: Mapping[Any, Any]) -> FrozenMapping:
    return FrozenMapping(
        tuple((deep_freeze(key), deep_freeze(item)) for key, item in sorted(value.items(), key=lambda pair: repr(pair[0])))
    )


def deep_freeze(value: Any) -> Any:
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return freeze_mapping(value)
    if isinstance(value, (tuple, list)):
        return FrozenSequence(
            type(value).__name__,
            tuple(deep_freeze(item) for item in value),
        )
    if isinstance(value, (set, frozenset)):
        return frozenset(deep_freeze(item) for item in value)
    return value


def thaw_mapping(value: FrozenMapping) -> dict[Any, Any]:
    return {thaw(key): thaw(item) for key, item in value.items}


def thaw(value: Any) -> Any:
    if isinstance(value, FrozenMapping):
        return thaw_mapping(value)
    if isinstance(value, FrozenSequence):
        items = tuple(thaw(item) for item in value.items)
        return list(items) if value.kind == "list" else items
    if isinstance(value, tuple):
        return tuple(thaw(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(thaw(item) for item in value)
    return value


def canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "nan"}
        if math.isinf(value):
            return {"float": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, Enum):
        return {
            "enum": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "value": canonical_value(value.value),
        }
    if isinstance(value, DataflowOperationRequest):
        return {"operation_contract": value.canonical_payload()}
    if isinstance(value, FrozenMapping):
        items = [[canonical_value(key), canonical_value(item)] for key, item in value.items]
        return {"mapping": sorted(items, key=lambda item: json.dumps(item[0], sort_keys=True))}
    if isinstance(value, FrozenSequence):
        return {
            "sequence_type": value.kind,
            "items": [canonical_value(item) for item in value.items],
        }
    if isinstance(value, Mapping):
        return canonical_value(freeze_mapping(value))
    if isinstance(value, (tuple, list)):
        return canonical_value(deep_freeze(value))
    if isinstance(value, (set, frozenset)):
        items = [canonical_value(item) for item in value]
        return {"set": sorted(items, key=lambda item: json.dumps(item, sort_keys=True))}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "fields": [
                [item.name, canonical_value(getattr(value, item.name))]
                for item in fields(value)
                if item.name not in {"canonical_json", "fingerprint"}
            ],
        }
    if isinstance(value, os.PathLike):
        return {"path": os.fspath(value)}
    if callable(value):
        return {
            "callable": f"{getattr(value, '__module__', type(value).__module__)}."
            f"{getattr(value, '__qualname__', getattr(value, '__name__', type(value).__qualname__))}"
        }
    return {
        "object": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
        "value": str(value),
    }
