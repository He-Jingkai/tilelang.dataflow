"""Typed logical-to-physical layouts for external Dataflow tensor arguments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any
from collections.abc import Mapping, Sequence


DATAFLOW_TENSOR_ARGUMENT_LAYOUT_SCHEMA_VERSION = 1
DATAFLOW_TENSOR_ARGUMENT_LAYOUTS_ATTR = "tensor_argument_layouts"
DATAFLOW_TENSOR_ARGUMENT_LAYOUT_IMPLEMENTATION = "dataflow.layout.tensor_argument_interleaved.v1"
DATAFLOW_TENSOR_LAYOUT_FINGERPRINT_ATTR = "_tilelang_dataflow_layout_fingerprint"


def positive_shape(value: Sequence[Any], name: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(item) for item in value)
    except (TypeError, ValueError) as err:
        raise TypeError(f"{name} must be an integer sequence") from err
    if not shape or any(item <= 0 for item in shape):
        raise ValueError(f"{name} must contain positive extents, got {shape!r}")
    return shape


@dataclass(frozen=True)
class DataflowTensorArgumentLayout:
    parameter_name: str
    logical_shape: tuple[int, ...]
    physical_shape: tuple[int, ...]
    physical_axes: tuple[tuple[int, ...], ...]
    selector_axis: int
    interleave_axis: int
    interleave: int
    contiguous: bool = True
    schema_version: int = DATAFLOW_TENSOR_ARGUMENT_LAYOUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_TENSOR_ARGUMENT_LAYOUT_SCHEMA_VERSION:
            raise ValueError(f"unsupported Dataflow tensor argument layout schema version {self.schema_version}")
        if not isinstance(self.parameter_name, str) or not self.parameter_name.isidentifier():
            raise ValueError("Dataflow tensor layout parameter_name must be an identifier")
        logical_shape = positive_shape(self.logical_shape, "logical_shape")
        physical_shape = positive_shape(self.physical_shape, "physical_shape")
        object.__setattr__(self, "logical_shape", logical_shape)
        object.__setattr__(self, "physical_shape", physical_shape)
        if isinstance(self.selector_axis, bool) or not isinstance(self.selector_axis, int):
            raise TypeError("selector_axis must be an integer")
        if isinstance(self.interleave_axis, bool) or not isinstance(self.interleave_axis, int):
            raise TypeError("interleave_axis must be an integer")
        logical_rank = len(logical_shape)
        for name, axis in (
            ("selector_axis", self.selector_axis),
            ("interleave_axis", self.interleave_axis),
        ):
            if axis < 0 or axis >= logical_rank:
                raise ValueError(f"{name} must index logical_shape")
        if self.selector_axis == self.interleave_axis:
            raise ValueError("selector_axis and interleave_axis must differ")
        if isinstance(self.interleave, bool) or not isinstance(self.interleave, int) or self.interleave <= 1:
            raise ValueError("interleave must be an integer greater than one")
        if logical_shape[self.interleave_axis] % self.interleave:
            raise ValueError("logical interleave extent must be divisible by interleave")
        if not isinstance(self.contiguous, bool):
            raise TypeError("contiguous must be a bool")
        physical_axes = tuple(tuple(group) for group in self.physical_axes)
        if len(physical_axes) != len(physical_shape) or any(not group for group in physical_axes):
            raise ValueError("physical_axes must contain one non-empty logical-axis group per physical axis")
        flattened = tuple(axis for group in physical_axes for axis in group)
        if sorted(flattened) != list(range(logical_rank)):
            raise ValueError("physical_axes must map every logical axis exactly once")
        fused = tuple(index for index, group in enumerate(physical_axes) if self.selector_axis in group or self.interleave_axis in group)
        if len(fused) != 1 or set(physical_axes[fused[0]]) != {
            self.selector_axis,
            self.interleave_axis,
        }:
            raise ValueError("selector_axis and interleave_axis must form one physical axis")
        expected_shape = tuple(axis_product(logical_shape, group) for group in physical_axes)
        if expected_shape != physical_shape:
            raise ValueError(f"physical_shape does not match physical_axes: expected {expected_shape!r}, got {physical_shape!r}")
        object.__setattr__(self, "physical_axes", physical_axes)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(include_fingerprint=False),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "implementation": DATAFLOW_TENSOR_ARGUMENT_LAYOUT_IMPLEMENTATION,
            "parameter_name": self.parameter_name,
            "logical_shape": list(self.logical_shape),
            "physical_shape": list(self.physical_shape),
            "physical_axes": [list(group) for group in self.physical_axes],
            "selector_axis": self.selector_axis,
            "interleave_axis": self.interleave_axis,
            "interleave": self.interleave,
            "contiguous": self.contiguous,
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowTensorArgumentLayout:
        if not isinstance(value, Mapping):
            raise TypeError("Dataflow tensor argument layout must be a mapping")
        allowed = {
            "schema_version",
            "implementation",
            "parameter_name",
            "logical_shape",
            "physical_shape",
            "physical_axes",
            "selector_axis",
            "interleave_axis",
            "interleave",
            "contiguous",
            "fingerprint",
        }
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError(f"Dataflow tensor argument layout has unknown fields {sorted(unknown)!r}")
        layout = cls(
            parameter_name=value.get("parameter_name"),
            logical_shape=tuple(value.get("logical_shape", ())),
            physical_shape=tuple(value.get("physical_shape", ())),
            physical_axes=tuple(tuple(group) for group in value.get("physical_axes", ())),
            selector_axis=value.get("selector_axis"),
            interleave_axis=value.get("interleave_axis"),
            interleave=value.get("interleave"),
            contiguous=value.get("contiguous", True),
            schema_version=int(
                value.get(
                    "schema_version",
                    DATAFLOW_TENSOR_ARGUMENT_LAYOUT_SCHEMA_VERSION,
                )
            ),
        )
        if value.get("implementation") not in {
            None,
            DATAFLOW_TENSOR_ARGUMENT_LAYOUT_IMPLEMENTATION,
        }:
            raise ValueError("Dataflow tensor argument layout implementation is stale")
        if value.get("fingerprint") not in {None, layout.fingerprint}:
            raise ValueError("Dataflow tensor argument layout fingerprint is stale")
        return layout


def axis_product(shape: tuple[int, ...], axes: tuple[int, ...]) -> int:
    result = 1
    for axis in axes:
        result *= shape[axis]
    return result


def normalize_tensor_argument_layouts(
    value: Any,
) -> tuple[DataflowTensorArgumentLayout, ...]:
    if value is None:
        return ()
    if isinstance(value, (DataflowTensorArgumentLayout, Mapping)):
        values = (value,)
    elif isinstance(value, (tuple, list)):
        values = tuple(value)
    else:
        raise TypeError("tensor_argument_layouts must be a typed layout or sequence")
    layouts = tuple(
        item if isinstance(item, DataflowTensorArgumentLayout) else DataflowTensorArgumentLayout.from_dict(item) for item in values
    )
    names = tuple(item.parameter_name for item in layouts)
    if len(names) != len(set(names)):
        raise ValueError("tensor_argument_layouts cannot repeat a parameter")
    return tuple(sorted(layouts, key=lambda item: item.parameter_name))


def tensor_argument_layout(attrs: Mapping[str, Any], parameter_name: str) -> DataflowTensorArgumentLayout | None:
    layouts = attrs.get(DATAFLOW_TENSOR_ARGUMENT_LAYOUTS_ATTR, ())
    for layout in layouts:
        if not isinstance(layout, DataflowTensorArgumentLayout):
            raise TypeError("Dataflow tensor argument layout attr must be typed")
        if layout.parameter_name == parameter_name:
            return layout
    return None


def mark_tensor_layout(value: Any, layout: DataflowTensorArgumentLayout) -> Any:
    if not isinstance(layout, DataflowTensorArgumentLayout):
        raise TypeError("mark_tensor_layout expects DataflowTensorArgumentLayout")
    try:
        setattr(value, DATAFLOW_TENSOR_LAYOUT_FINGERPRINT_ATTR, layout.fingerprint)
    except Exception as err:
        raise TypeError("tensor value cannot carry Dataflow layout fingerprint metadata") from err
    return value


__all__ = [
    "DATAFLOW_TENSOR_ARGUMENT_LAYOUT_IMPLEMENTATION",
    "DATAFLOW_TENSOR_ARGUMENT_LAYOUT_SCHEMA_VERSION",
    "DATAFLOW_TENSOR_ARGUMENT_LAYOUTS_ATTR",
    "DATAFLOW_TENSOR_LAYOUT_FINGERPRINT_ATTR",
    "DataflowTensorArgumentLayout",
    "mark_tensor_layout",
    "normalize_tensor_argument_layouts",
    "tensor_argument_layout",
]
