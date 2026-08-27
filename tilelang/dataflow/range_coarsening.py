"""Generic logical-range to handler-range coarsening plans."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any
from collections.abc import Mapping

from .dtype_registry import dataflow_dtype_info
from .ir import IntermediateType
from .operation_contracts import (
    DATAFLOW_RANGE_INDEPENDENT,
    DATAFLOW_REMAINDER_EXACT,
    DATAFLOW_REMAINDER_POLICIES,
    DataflowRangeCoarseningRequest,
)


DATAFLOW_RANGE_COARSENING_PLAN_SCHEMA_VERSION = 1
DATAFLOW_RANGE_TILES_PER_HANDLER_SPECIALIZATION = "dataflow_range_tiles_per_handler"
_RANGE_FALLBACK_REASONS = frozenset(
    {
        "fusion_not_permitted",
        "dependence_ordered",
        "dependence_reduction",
        "resource_budget",
        "exact_divisibility",
    }
)


class DataflowRangeCoarseningError(ValueError):
    """Raised when no legal handler-range plan can satisfy a typed request."""


def positive_int(value: Any, name: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DataflowRangeCoarseningError(f"Dataflow range coarsening {name} must be a positive integer, got {value!r}")
    return int(value)


def nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataflowRangeCoarseningError(f"Dataflow range coarsening {name} must be a non-negative integer, got {value!r}")
    return int(value)


def fingerprint(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DataflowRangeOutputMapping:
    """Map one physical handler output to its contiguous logical tiles."""

    handler_index: int
    range_begin: int
    range_end: int
    padded_range_end: int
    logical_tile_begin: int
    logical_tile_end: int
    physical_output_indices: tuple[int, ...]
    padding_output_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "handler_index",
            "range_begin",
            "range_end",
            "padded_range_end",
            "logical_tile_begin",
            "logical_tile_end",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"range output mapping {name} must be non-negative")
        if self.range_begin >= self.range_end:
            raise ValueError("range output mapping must cover a non-empty logical range")
        if self.padded_range_end < self.range_end:
            raise ValueError("range output mapping padded end cannot precede range end")
        if self.logical_tile_begin >= self.logical_tile_end:
            raise ValueError("range output mapping must cover at least one logical tile")
        physical = tuple(self.physical_output_indices)
        padding = tuple(self.padding_output_indices)
        for name, indices in (
            ("physical_output_indices", physical),
            ("padding_output_indices", padding),
        ):
            if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices):
                raise ValueError(f"range output mapping {name} must contain non-negative integers")
        if physical != tuple(range(len(physical))):
            raise ValueError("range output physical indices must be contiguous from zero")
        if padding != tuple(range(len(physical), len(physical) + len(padding))):
            raise ValueError("range output padding indices must follow physical indices")
        object.__setattr__(self, "physical_output_indices", physical)
        object.__setattr__(self, "padding_output_indices", padding)

    @property
    def valid_range_extent(self) -> int:
        return self.range_end - self.range_begin

    @property
    def padded_range_extent(self) -> int:
        return self.padded_range_end - self.range_begin

    @property
    def logical_tile_count(self) -> int:
        return self.logical_tile_end - self.logical_tile_begin

    @property
    def has_remainder(self) -> bool:
        return self.range_end != self.padded_range_end

    def to_dict(self) -> dict[str, Any]:
        return {
            "handler_index": self.handler_index,
            "range_begin": self.range_begin,
            "range_end": self.range_end,
            "padded_range_end": self.padded_range_end,
            "valid_range_extent": self.valid_range_extent,
            "padded_range_extent": self.padded_range_extent,
            "logical_tile_begin": self.logical_tile_begin,
            "logical_tile_end": self.logical_tile_end,
            "logical_tile_count": self.logical_tile_count,
            "physical_output_indices": list(self.physical_output_indices),
            "padding_output_indices": list(self.padding_output_indices),
            "has_remainder": self.has_remainder,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowRangeOutputMapping:
        if not isinstance(value, Mapping):
            raise TypeError("range output mapping must be a mapping")
        expected_fields = {
            "handler_index",
            "range_begin",
            "range_end",
            "padded_range_end",
            "valid_range_extent",
            "padded_range_extent",
            "logical_tile_begin",
            "logical_tile_end",
            "logical_tile_count",
            "physical_output_indices",
            "padding_output_indices",
            "has_remainder",
        }
        if set(value) != expected_fields:
            raise ValueError(
                "range output mapping fields do not match schema: "
                f"missing={sorted(expected_fields - set(value))!r}, "
                f"unknown={sorted(set(value) - expected_fields)!r}"
            )
        for name in ("physical_output_indices", "padding_output_indices"):
            if not isinstance(value[name], (list, tuple)):
                raise TypeError(f"range output mapping {name} must be a sequence")
        mapping = cls(
            handler_index=value["handler_index"],
            range_begin=value["range_begin"],
            range_end=value["range_end"],
            padded_range_end=value["padded_range_end"],
            logical_tile_begin=value["logical_tile_begin"],
            logical_tile_end=value["logical_tile_end"],
            physical_output_indices=tuple(value["physical_output_indices"]),
            padding_output_indices=tuple(value["padding_output_indices"]),
        )
        if dict(value) != mapping.to_dict():
            raise ValueError("range output mapping derived fields do not match payload")
        return mapping


@dataclass(frozen=True)
class DataflowRangeCoarseningPlan:
    """Selected handler extent and output mapping for one logical range."""

    request_fingerprint: str
    logical_range_extent: int
    logical_tile_extent: int
    logical_tile_count: int
    requested_handler_range_extent: int
    requested_tiles_per_handler: int
    selected_handler_range_extent: int
    tiles_per_handler: int
    handler_count: int
    output_tile_arity: int
    remainder_policy: str
    resource_bytes_per_tile: int
    resource_budget_bytes: int | None
    requested_resource_bytes: int
    selected_resource_bytes: int
    fallback_reasons: tuple[str, ...]
    output_mappings: tuple[DataflowRangeOutputMapping, ...]
    schema_version: int = DATAFLOW_RANGE_COARSENING_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != DATAFLOW_RANGE_COARSENING_PLAN_SCHEMA_VERSION
        ):
            raise ValueError(f"Unsupported Dataflow range coarsening plan schema version {self.schema_version}")
        if (
            not isinstance(self.request_fingerprint, str)
            or len(self.request_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in self.request_fingerprint)
        ):
            raise ValueError("range coarsening plan requires a request fingerprint")
        for name in (
            "logical_range_extent",
            "logical_tile_extent",
            "logical_tile_count",
            "requested_handler_range_extent",
            "requested_tiles_per_handler",
            "selected_handler_range_extent",
            "tiles_per_handler",
            "handler_count",
            "output_tile_arity",
        ):
            positive_int(getattr(self, name), name)
        for name in (
            "resource_bytes_per_tile",
            "requested_resource_bytes",
            "selected_resource_bytes",
        ):
            nonnegative_int(getattr(self, name), name)
        if self.resource_budget_bytes is not None:
            positive_int(self.resource_budget_bytes, "resource_budget_bytes")
            if self.selected_resource_bytes > self.resource_budget_bytes:
                raise ValueError("selected range plan exceeds its resource budget")
        if self.remainder_policy not in DATAFLOW_REMAINDER_POLICIES:
            raise ValueError(f"unsupported range remainder policy {self.remainder_policy!r}")
        if self.logical_tile_count != math.ceil(self.logical_range_extent / self.logical_tile_extent):
            raise ValueError("range plan logical tile count is inconsistent")
        expected_requested_tiles = math.ceil(self.requested_handler_range_extent / self.logical_tile_extent)
        if self.requested_tiles_per_handler != expected_requested_tiles:
            raise ValueError("range plan requested tile count is inconsistent")
        if self.tiles_per_handler > self.requested_tiles_per_handler:
            raise ValueError("selected range plan cannot grow the requested handler")
        if self.tiles_per_handler > self.output_tile_arity:
            raise ValueError("selected range plan exceeds physical output tile arity")
        expected_selected_extent = (
            self.requested_handler_range_extent
            if self.tiles_per_handler == self.requested_tiles_per_handler
            else self.tiles_per_handler * self.logical_tile_extent
        )
        if self.selected_handler_range_extent != expected_selected_extent:
            raise ValueError("range plan selected handler extent is inconsistent")
        if self.handler_count != math.ceil(self.logical_range_extent / self.selected_handler_range_extent):
            raise ValueError("range plan handler count is inconsistent")
        if self.requested_resource_bytes != (self.requested_tiles_per_handler * self.resource_bytes_per_tile):
            raise ValueError("range plan requested resource estimate is inconsistent")
        if self.selected_resource_bytes != (self.tiles_per_handler * self.resource_bytes_per_tile):
            raise ValueError("range plan selected resource estimate is inconsistent")
        reasons = tuple(self.fallback_reasons)
        if any(not isinstance(reason, str) or reason not in _RANGE_FALLBACK_REASONS for reason in reasons):
            raise ValueError("range plan contains an unsupported fallback reason")
        if len(reasons) != len(set(reasons)):
            raise ValueError("range plan fallback reasons must be unique")
        if bool(reasons) != (self.tiles_per_handler < self.requested_tiles_per_handler):
            raise ValueError("range plan fallback reasons do not match the selected handler extent")
        if self.remainder_policy == DATAFLOW_REMAINDER_EXACT:
            if self.logical_range_extent % self.logical_tile_extent:
                raise ValueError("exact range plan contains a partial logical tile")
            if self.logical_tile_count % self.tiles_per_handler:
                raise ValueError("exact range plan contains a partial handler")
        mappings = tuple(self.output_mappings)
        if len(mappings) != self.handler_count:
            raise ValueError("range plan mapping count must match handler count")
        if tuple(mapping.handler_index for mapping in mappings) != tuple(range(self.handler_count)):
            raise ValueError("range plan mappings must use contiguous handler indices")
        if mappings[0].range_begin != 0 or mappings[-1].range_end != self.logical_range_extent:
            raise ValueError("range plan mappings must cover the complete logical range")
        if mappings[0].logical_tile_begin != 0 or mappings[-1].logical_tile_end != self.logical_tile_count:
            raise ValueError("range plan mappings must cover all logical tiles")
        for left, right in zip(mappings, mappings[1:]):
            if left.range_end != right.range_begin:
                raise ValueError("range plan mappings must be contiguous")
            if left.logical_tile_end != right.logical_tile_begin:
                raise ValueError("range plan logical tile mappings must be contiguous")
        expected_mappings = range_output_mappings(
            logical_range_extent=self.logical_range_extent,
            logical_tile_extent=self.logical_tile_extent,
            selected_handler_range_extent=self.selected_handler_range_extent,
            tiles_per_handler=self.tiles_per_handler,
            handler_count=self.handler_count,
            output_tile_arity=self.output_tile_arity,
        )
        if mappings != expected_mappings:
            raise ValueError("range plan output mappings are inconsistent")
        object.__setattr__(self, "fallback_reasons", reasons)
        object.__setattr__(self, "output_mappings", mappings)

    @property
    def used_fallback(self) -> bool:
        return bool(self.fallback_reasons)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_fingerprint": self.request_fingerprint,
            "logical_range_extent": self.logical_range_extent,
            "logical_tile_extent": self.logical_tile_extent,
            "logical_tile_count": self.logical_tile_count,
            "requested_handler_range_extent": self.requested_handler_range_extent,
            "requested_tiles_per_handler": self.requested_tiles_per_handler,
            "selected_handler_range_extent": self.selected_handler_range_extent,
            "tiles_per_handler": self.tiles_per_handler,
            "handler_count": self.handler_count,
            "output_tile_arity": self.output_tile_arity,
            "remainder_policy": self.remainder_policy,
            "resource_bytes_per_tile": self.resource_bytes_per_tile,
            "resource_budget_bytes": self.resource_budget_bytes,
            "requested_resource_bytes": self.requested_resource_bytes,
            "selected_resource_bytes": self.selected_resource_bytes,
            "used_fallback": self.used_fallback,
            "fallback_reasons": list(self.fallback_reasons),
            "output_mappings": [item.to_dict() for item in self.output_mappings],
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        result = self.canonical_payload()
        result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowRangeCoarseningPlan:
        if not isinstance(value, Mapping):
            raise TypeError("range coarsening plan must be a mapping")
        expected_fields = {
            "schema_version",
            "request_fingerprint",
            "logical_range_extent",
            "logical_tile_extent",
            "logical_tile_count",
            "requested_handler_range_extent",
            "requested_tiles_per_handler",
            "selected_handler_range_extent",
            "tiles_per_handler",
            "handler_count",
            "output_tile_arity",
            "remainder_policy",
            "resource_bytes_per_tile",
            "resource_budget_bytes",
            "requested_resource_bytes",
            "selected_resource_bytes",
            "used_fallback",
            "fallback_reasons",
            "output_mappings",
            "fingerprint",
        }
        if set(value) != expected_fields:
            raise ValueError(
                "range coarsening plan fields do not match schema: "
                f"missing={sorted(expected_fields - set(value))!r}, "
                f"unknown={sorted(set(value) - expected_fields)!r}"
            )
        if not isinstance(value["fallback_reasons"], (list, tuple)):
            raise TypeError("range coarsening fallback_reasons must be a sequence")
        if not isinstance(value["output_mappings"], (list, tuple)):
            raise TypeError("range coarsening output_mappings must be a sequence")
        plan = cls(
            request_fingerprint=value["request_fingerprint"],
            logical_range_extent=value["logical_range_extent"],
            logical_tile_extent=value["logical_tile_extent"],
            logical_tile_count=value["logical_tile_count"],
            requested_handler_range_extent=value["requested_handler_range_extent"],
            requested_tiles_per_handler=value["requested_tiles_per_handler"],
            selected_handler_range_extent=value["selected_handler_range_extent"],
            tiles_per_handler=value["tiles_per_handler"],
            handler_count=value["handler_count"],
            output_tile_arity=value["output_tile_arity"],
            remainder_policy=value["remainder_policy"],
            resource_bytes_per_tile=value["resource_bytes_per_tile"],
            resource_budget_bytes=value["resource_budget_bytes"],
            requested_resource_bytes=value["requested_resource_bytes"],
            selected_resource_bytes=value["selected_resource_bytes"],
            fallback_reasons=tuple(value["fallback_reasons"]),
            output_mappings=tuple(DataflowRangeOutputMapping.from_dict(item) for item in value["output_mappings"]),
            schema_version=value["schema_version"],
        )
        recorded = value["fingerprint"]
        if not isinstance(recorded, str) or recorded != plan.fingerprint:
            raise ValueError("range coarsening plan fingerprint does not match payload")
        recorded_fallback = value["used_fallback"]
        if not isinstance(recorded_fallback, bool):
            raise TypeError("range coarsening used_fallback must be a bool")
        if recorded_fallback != plan.used_fallback:
            raise ValueError("range coarsening plan fallback state does not match reasons")
        return plan


def largest_exact_tile_count(logical_tiles: int, maximum: int) -> int:
    for candidate in range(min(logical_tiles, maximum), 0, -1):
        if logical_tiles % candidate == 0:
            return candidate
    raise AssertionError("one logical tile always divides the logical tile count")


def range_output_mappings(
    *,
    logical_range_extent: int,
    logical_tile_extent: int,
    selected_handler_range_extent: int,
    tiles_per_handler: int,
    handler_count: int,
    output_tile_arity: int,
) -> tuple[DataflowRangeOutputMapping, ...]:
    mappings = []
    for handler_index in range(handler_count):
        begin = handler_index * selected_handler_range_extent
        end = min(begin + selected_handler_range_extent, logical_range_extent)
        tile_begin = begin // logical_tile_extent
        tile_end = math.ceil(end / logical_tile_extent)
        valid_tiles = tile_end - tile_begin
        mappings.append(
            DataflowRangeOutputMapping(
                handler_index=handler_index,
                range_begin=begin,
                range_end=end,
                padded_range_end=begin + tiles_per_handler * logical_tile_extent,
                logical_tile_begin=tile_begin,
                logical_tile_end=tile_end,
                physical_output_indices=tuple(range(valid_tiles)),
                padding_output_indices=tuple(range(valid_tiles, output_tile_arity)),
            )
        )
    return tuple(mappings)


def plan_range_coarsening(
    request: DataflowRangeCoarseningRequest,
    *,
    logical_range_extent: int | None = None,
    resource_bytes_per_tile: int = 0,
    resource_budget_bytes: int | None = None,
) -> DataflowRangeCoarseningPlan:
    """Select a name-independent contiguous handler mapping for one range."""

    if not isinstance(request, DataflowRangeCoarseningRequest):
        raise TypeError(f"plan_range_coarsening expects DataflowRangeCoarseningRequest, got {type(request)!r}")
    resolved_range = request.logical_range_extent if logical_range_extent is None else logical_range_extent
    resolved_range = positive_int(resolved_range, "logical_range_extent")
    assert resolved_range is not None
    if request.logical_range_extent is not None and resolved_range != request.logical_range_extent:
        raise DataflowRangeCoarseningError(
            f"runtime logical range does not match the typed range request: {resolved_range} != {request.logical_range_extent}"
        )
    resource_bytes_per_tile = nonnegative_int(
        resource_bytes_per_tile,
        "resource_bytes_per_tile",
    )
    resource_budget_bytes = positive_int(
        resource_budget_bytes,
        "resource_budget_bytes",
        allow_none=True,
    )

    tile_extent = request.logical_tile_extent
    logical_tiles = math.ceil(resolved_range / tile_extent)
    requested_extent = request.handler_range_extent or tile_extent
    requested_tiles = math.ceil(requested_extent / tile_extent)

    selected_tiles = requested_tiles
    fallback_reasons: list[str] = []
    if requested_tiles > 1 and not request.allow_fusion:
        selected_tiles = 1
        fallback_reasons.append("fusion_not_permitted")
    if requested_tiles > 1 and request.dependence != DATAFLOW_RANGE_INDEPENDENT:
        selected_tiles = 1
        fallback_reasons.append(f"dependence_{request.dependence}")

    requested_resource_bytes = requested_tiles * resource_bytes_per_tile
    if resource_budget_bytes is not None and resource_bytes_per_tile and selected_tiles * resource_bytes_per_tile > resource_budget_bytes:
        resource_tiles = resource_budget_bytes // resource_bytes_per_tile
        if resource_tiles <= 0:
            raise DataflowRangeCoarseningError(
                "one logical range tile exceeds the resource budget: "
                f"tile_bytes={resource_bytes_per_tile}, "
                f"budget_bytes={resource_budget_bytes}"
            )
        selected_tiles = min(selected_tiles, resource_tiles)
        fallback_reasons.append("resource_budget")

    if request.remainder_policy == DATAFLOW_REMAINDER_EXACT:
        if resolved_range % tile_extent:
            raise DataflowRangeCoarseningError("exact range coarsening requires logical_range_extent to contain whole logical tiles")
        exact_tiles = largest_exact_tile_count(logical_tiles, selected_tiles)
        if exact_tiles != selected_tiles:
            selected_tiles = exact_tiles
            fallback_reasons.append("exact_divisibility")

    if request.output_tile_arity < selected_tiles:
        raise DataflowRangeCoarseningError(
            "range output_tile_arity cannot represent the selected handler tiles: "
            f"output_tile_arity={request.output_tile_arity}, "
            f"selected_tiles_per_handler={selected_tiles}"
        )

    selected_extent = requested_extent if selected_tiles == requested_tiles else selected_tiles * tile_extent
    handler_count = math.ceil(resolved_range / selected_extent)
    if handler_count > 1 and selected_extent % tile_extent:
        raise DataflowRangeCoarseningError(
            "a non-final handler boundary must align to logical_tile_extent: "
            f"handler_range_extent={selected_extent}, "
            f"logical_tile_extent={tile_extent}"
        )
    mappings = range_output_mappings(
        logical_range_extent=resolved_range,
        logical_tile_extent=tile_extent,
        selected_handler_range_extent=selected_extent,
        tiles_per_handler=selected_tiles,
        handler_count=handler_count,
        output_tile_arity=request.output_tile_arity,
    )

    return DataflowRangeCoarseningPlan(
        request_fingerprint=request.fingerprint,
        logical_range_extent=resolved_range,
        logical_tile_extent=tile_extent,
        logical_tile_count=logical_tiles,
        requested_handler_range_extent=requested_extent,
        requested_tiles_per_handler=requested_tiles,
        selected_handler_range_extent=selected_extent,
        tiles_per_handler=selected_tiles,
        handler_count=handler_count,
        output_tile_arity=request.output_tile_arity,
        remainder_policy=request.remainder_policy,
        resource_bytes_per_tile=resource_bytes_per_tile,
        resource_budget_bytes=resource_budget_bytes,
        requested_resource_bytes=requested_resource_bytes,
        selected_resource_bytes=selected_tiles * resource_bytes_per_tile,
        fallback_reasons=tuple(dict.fromkeys(fallback_reasons)),
        output_mappings=mappings,
    )


def estimate_range_output_bytes_per_tile(
    intermediate: IntermediateType,
    *,
    output_tile_arity: int,
) -> int:
    """Estimate one logical output tile without consulting operator identity."""

    if not isinstance(intermediate, IntermediateType):
        raise TypeError(f"range output resource estimation expects an IntermediateType, got {type(intermediate)!r}")
    output_tile_arity = positive_int(output_tile_arity, "output_tile_arity")
    assert output_tile_arity is not None
    total_bytes = 0
    for field in intermediate.fields:
        dtype = dataflow_dtype_info(field.dtype)
        if dtype is None:
            return 0
        elements = 1
        if field.shape is not None:
            for extent in field.shape:
                try:
                    extent = int(extent)
                except (TypeError, ValueError):
                    return 0
                if extent <= 0:
                    return 0
                elements *= extent
        total_bytes += elements * dtype.element_bytes
    return math.ceil(total_bytes / output_tile_arity)


__all__ = [
    "DATAFLOW_RANGE_COARSENING_PLAN_SCHEMA_VERSION",
    "DATAFLOW_RANGE_TILES_PER_HANDLER_SPECIALIZATION",
    "DataflowRangeCoarseningError",
    "DataflowRangeCoarseningPlan",
    "DataflowRangeOutputMapping",
    "estimate_range_output_bytes_per_tile",
    "plan_range_coarsening",
]
