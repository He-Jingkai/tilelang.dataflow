"""Canonical operator-closure specialization metadata."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import inspect
import json
import math
import os
from typing import Any
from collections.abc import Callable
from collections.abc import Mapping

from .dtype_registry import dataflow_dtype_info
from .ir import get_intermediate_type


DATAFLOW_SPECIALIZATION_CAPTURE_SCHEMA_VERSION = 1
DATAFLOW_SPECIALIZATION_SNAPSHOT_ATTR = "specialization_snapshot"
DATAFLOW_SPECIALIZATION_CAPTURE_IMPLEMENTATION = "dataflow.specialization.closure_capture.v1"


class DataflowSpecializationCaptureError(TypeError):
    """Raised when an operator uses a closure value that cannot be replayed."""


def serialize_canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(serialize_canonical_json(value).encode("utf-8")).hexdigest()


def type_name(value: Any) -> str:
    return f"{value.__class__.__module__}.{value.__class__.__qualname__}"


def reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str],
    context: str,
) -> None:
    unknown = set(value).difference(allowed)
    if unknown:
        raise ValueError(f"{context} has unknown fields {sorted(unknown)!r}")


def is_tilelang_macro(value: Any) -> bool:
    try:
        from tilelang.language.eager.builder import Macro
    except ImportError:
        return False
    return isinstance(value, Macro)


def is_tilelang_dtype(value: Any) -> bool:
    try:
        from tilelang import tvm
    except ImportError:
        return False
    return isinstance(value, tvm.DataType)


def canonical_tilelang_macro(value: Any, path: str) -> Any:
    source = getattr(value, "source", None)
    orig_func = getattr(value, "orig_func", None)
    if not isinstance(source, str) or not source or not inspect.isfunction(orig_func):
        raise DataflowSpecializationCaptureError(f"Dataflow specialization {path} has invalid TileLang macro metadata")
    try:
        closure = inspect.getclosurevars(orig_func)
    except TypeError as err:
        raise DataflowSpecializationCaptureError(f"cannot inspect TileLang macro closure {path}") from err
    return {
        "tilelang_macro": type_name(value),
        "source": source,
        "nonlocals": [
            [name, canonical_value(item, f"{path}.{name}")]
            for name, item in sorted(closure.nonlocals.items())
            if not is_structural_reference(item)
        ],
    }


def canonical_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataflowSpecializationCaptureError(f"Dataflow specialization {path} must be finite, got {value!r}")
        return value
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, Enum):
        return {
            "enum": type_name(value),
            "value": canonical_value(value.value, f"{path}.value"),
        }
    if isinstance(value, os.PathLike):
        return {"path": os.fspath(value)}
    if is_tilelang_dtype(value):
        dtype = dataflow_dtype_info(value)
        if dtype is None:
            raise DataflowSpecializationCaptureError(f"Dataflow specialization {path} uses unsupported TileLang dtype {value!r}")
        return dtype.name
    if is_tilelang_macro(value):
        return canonical_tilelang_macro(value, path)
    if isinstance(value, Mapping):
        items = [
            [
                canonical_value(key, f"{path}.key"),
                canonical_value(item, f"{path}[{key!r}]"),
            ]
            for key, item in value.items()
        ]
        items.sort(key=lambda item: serialize_canonical_json(item[0]))
        return {"mapping": items}
    if isinstance(value, tuple):
        return {"tuple": [canonical_value(item, f"{path}[{index}]") for index, item in enumerate(value)]}
    if isinstance(value, list):
        return {"list": [canonical_value(item, f"{path}[{index}]") for index, item in enumerate(value)]}
    if isinstance(value, (set, frozenset)):
        items = [canonical_value(item, f"{path}.item") for item in value]
        items.sort(key=serialize_canonical_json)
        return {"set": items}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": type_name(value),
            "fields": [
                [
                    item.name,
                    canonical_value(
                        getattr(value, item.name),
                        f"{path}.{item.name}",
                    ),
                ]
                for item in fields(value)
                if item.name not in {"canonical_json", "fingerprint"}
            ],
        }
    raise DataflowSpecializationCaptureError(
        f"Dataflow operator closure value {path} has unsupported type "
        f"{type_name(value)!r}; use a finite canonical value or a typed dataclass"
    )


def is_structural_reference(value: Any) -> bool:
    return get_intermediate_type(value) is not None


def operator_specialization_values(func: Callable[..., Any]) -> dict[str, Any]:
    """Return only non-structural values actually captured by ``func``."""

    if not inspect.isfunction(func):
        raise TypeError(f"operator_specialization_values expects a Python function, got {func!r}")
    try:
        closure = inspect.getclosurevars(func)
    except TypeError as err:
        raise DataflowSpecializationCaptureError(f"cannot inspect Dataflow operator closure {func.__qualname__!r}") from err
    result = {}
    for name, value in sorted(closure.nonlocals.items()):
        if is_structural_reference(value):
            continue
        canonical_value(value, name)
        result[name] = value
    return result


@dataclass(frozen=True)
class DataflowSpecializationEntry:
    name: str
    canonical_value: Any

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError(f"Dataflow specialization entry names must be identifiers, got {self.name!r}")
        serialize_canonical_json(self.canonical_value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "canonical_value": self.canonical_value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowSpecializationEntry:
        if not isinstance(value, Mapping):
            raise TypeError("Dataflow specialization entry must be a mapping")
        reject_unknown(
            value,
            {"name", "canonical_value"},
            "Dataflow specialization entry",
        )
        missing = {"name", "canonical_value"}.difference(value)
        if missing:
            raise ValueError(f"Dataflow specialization entry is missing fields {sorted(missing)!r}")
        return cls(
            name=value.get("name"),
            canonical_value=value.get("canonical_value"),
        )


@dataclass(frozen=True)
class DataflowSpecializationSnapshot:
    entries: tuple[DataflowSpecializationEntry, ...] = ()
    schema_version: int = DATAFLOW_SPECIALIZATION_CAPTURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_SPECIALIZATION_CAPTURE_SCHEMA_VERSION:
            raise ValueError(f"unsupported Dataflow specialization capture schema version {self.schema_version}")
        entries = tuple(self.entries)
        if any(not isinstance(item, DataflowSpecializationEntry) for item in entries):
            raise TypeError("Dataflow specialization entries must be typed")
        entries = tuple(sorted(entries, key=lambda item: item.name))
        names = tuple(item.name for item in entries)
        if len(names) != len(set(names)):
            raise ValueError("Dataflow specialization entries must have unique names")
        object.__setattr__(self, "entries", entries)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.canonical_payload())

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "implementation": DATAFLOW_SPECIALIZATION_CAPTURE_IMPLEMENTATION,
            "entries": [item.to_dict() for item in self.entries],
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.canonical_payload()
        result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowSpecializationSnapshot:
        if not isinstance(value, Mapping):
            raise TypeError("Dataflow specialization snapshot must be a mapping")
        reject_unknown(
            value,
            {"schema_version", "implementation", "entries", "fingerprint"},
            "Dataflow specialization snapshot",
        )
        snapshot = cls(
            entries=tuple(DataflowSpecializationEntry.from_dict(item) for item in value.get("entries", ())),
            schema_version=int(
                value.get(
                    "schema_version",
                    DATAFLOW_SPECIALIZATION_CAPTURE_SCHEMA_VERSION,
                )
            ),
        )
        if value.get("implementation") not in {
            None,
            DATAFLOW_SPECIALIZATION_CAPTURE_IMPLEMENTATION,
        }:
            raise ValueError("Dataflow specialization implementation is stale")
        if value.get("fingerprint") not in {None, snapshot.fingerprint}:
            raise ValueError("Dataflow specialization snapshot fingerprint is stale")
        return snapshot


def capture_operator_specialization(
    func: Callable[..., Any],
) -> DataflowSpecializationSnapshot:
    values = operator_specialization_values(func)
    return DataflowSpecializationSnapshot(
        entries=tuple(
            DataflowSpecializationEntry(
                name=name,
                canonical_value=canonical_value(value, name),
            )
            for name, value in values.items()
        )
    )


def validated_operator_specialization_values(
    func: Callable[..., Any],
    snapshot: DataflowSpecializationSnapshot,
) -> dict[str, Any]:
    """Return live closure values after verifying the captured cache contract."""

    if not isinstance(snapshot, DataflowSpecializationSnapshot):
        raise TypeError("Dataflow specialization validation requires a typed snapshot")
    values = operator_specialization_values(func)
    current = DataflowSpecializationSnapshot(
        entries=tuple(
            DataflowSpecializationEntry(
                name=name,
                canonical_value=canonical_value(value, name),
            )
            for name, value in values.items()
        )
    )
    if current != snapshot:
        raise DataflowSpecializationCaptureError(
            f"Dataflow operator closure {func.__qualname__!r} changed after its specialization snapshot was captured"
        )
    return values


def specialization_snapshot_from_attrs(
    attrs: Mapping[str, Any],
) -> DataflowSpecializationSnapshot:
    value = attrs.get(DATAFLOW_SPECIALIZATION_SNAPSHOT_ATTR)
    if value is None:
        return DataflowSpecializationSnapshot()
    if not isinstance(value, DataflowSpecializationSnapshot):
        raise TypeError("Dataflow specialization snapshot attr must be typed")
    return value


__all__ = [
    "DATAFLOW_SPECIALIZATION_CAPTURE_IMPLEMENTATION",
    "DATAFLOW_SPECIALIZATION_CAPTURE_SCHEMA_VERSION",
    "DATAFLOW_SPECIALIZATION_SNAPSHOT_ATTR",
    "DataflowSpecializationCaptureError",
    "DataflowSpecializationEntry",
    "DataflowSpecializationSnapshot",
    "capture_operator_specialization",
    "operator_specialization_values",
    "specialization_snapshot_from_attrs",
    "validated_operator_specialization_values",
]
