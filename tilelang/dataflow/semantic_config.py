"""Typed correctness-affecting compile options for Dataflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Mapping

from .implementation_registry import dataflow_implementation_registry
from .precision import DataflowPrecisionPolicy


DATAFLOW_SEMANTIC_CONFIG_SCHEMA_VERSION = 3
_DATAFLOW_SEMANTIC_CONFIG_RECORDED_SCHEMA_VERSION = 2


def bool_field(value: Mapping[str, Any], name: str) -> bool:
    raw = value.get(name)
    if raw is None or str(raw).strip() == "":
        return False
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a bool, got {raw!r}")


@dataclass(frozen=True)
class DataflowSemanticConfig:
    """Semantic options that must be fingerprinted and visible in plan dumps."""

    schema_version: int = DATAFLOW_SEMANTIC_CONFIG_SCHEMA_VERSION
    direct_slot_seed_reduce: bool = False
    skip_finalize_post_sync: bool = False
    fast_math: bool = False
    precision: DataflowPrecisionPolicy = field(default_factory=DataflowPrecisionPolicy)

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_SEMANTIC_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"Unsupported Dataflow semantic config schema version {self.schema_version}")
        for name in (
            "direct_slot_seed_reduce",
            "skip_finalize_post_sync",
            "fast_math",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"Dataflow semantic config {name} must be a bool")
        if not isinstance(self.precision, DataflowPrecisionPolicy):
            if not isinstance(self.precision, Mapping):
                raise TypeError("Dataflow semantic config precision must be DataflowPrecisionPolicy or a mapping")
            object.__setattr__(
                self,
                "precision",
                DataflowPrecisionPolicy.from_dict(self.precision),
            )
        registry = dataflow_implementation_registry()
        if self.direct_slot_seed_reduce:
            registry.require_selectable(
                "semantic.direct_slot_seed_reduce",
                selected_explicitly=True,
            )
        if self.skip_finalize_post_sync:
            registry.require_selectable(
                "semantic.skip_finalize_post_sync",
                selected_explicitly=True,
            )
        if self.fast_math:
            registry.require_selectable(
                "semantic.fast_math",
                selected_explicitly=True,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "direct_slot_seed_reduce": self.direct_slot_seed_reduce,
            "skip_finalize_post_sync": self.skip_finalize_post_sync,
            "fast_math": self.fast_math,
            "precision": self.precision.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowSemanticConfig:
        if not isinstance(value, Mapping):
            raise TypeError(f"Dataflow semantic config must be a mapping, got {type(value)!r}")
        schema_version = int(value.get("schema_version", DATAFLOW_SEMANTIC_CONFIG_SCHEMA_VERSION))
        if schema_version not in {
            _DATAFLOW_SEMANTIC_CONFIG_RECORDED_SCHEMA_VERSION,
            DATAFLOW_SEMANTIC_CONFIG_SCHEMA_VERSION,
        }:
            raise ValueError(f"Unsupported Dataflow semantic config schema version {schema_version}")
        if schema_version == _DATAFLOW_SEMANTIC_CONFIG_RECORDED_SCHEMA_VERSION and "fast_math" in value:
            raise ValueError("Dataflow semantic config schema 2 cannot encode fast_math")
        return cls(
            direct_slot_seed_reduce=bool_field(value, "direct_slot_seed_reduce"),
            skip_finalize_post_sync=bool_field(value, "skip_finalize_post_sync"),
            fast_math=bool_field(value, "fast_math"),
            precision=DataflowPrecisionPolicy.from_dict(value.get("precision", {})),
        )


def resolve_semantic_config(
    value: DataflowSemanticConfig | Mapping[str, Any] | None,
) -> DataflowSemanticConfig:
    if value is None:
        return DataflowSemanticConfig()
    if isinstance(value, DataflowSemanticConfig):
        return value
    if isinstance(value, Mapping):
        return DataflowSemanticConfig.from_dict(value)
    raise TypeError(f"Dataflow semantic_config must be DataflowSemanticConfig or a serialized mapping, got {type(value)!r}")
