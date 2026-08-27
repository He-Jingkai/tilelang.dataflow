"""Versioned target-independent operation requests and decision provenance."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import re
from typing import Any, ClassVar
from collections.abc import Mapping, Sequence


DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION = 5
DATAFLOW_OPERATION_DECISION_SCHEMA_VERSION = 1

DATAFLOW_OPERATION_RANGE_COARSENING = "range_coarsening"
DATAFLOW_OPERATION_PIPELINE = "pipeline_dataflow"
DATAFLOW_OPERATION_RESHARED_TRANSPORT = "reshared_transport"
DATAFLOW_OPERATION_CROSS_HANDLER_HANDOFF = "cross_handler_handoff"
DATAFLOW_OPERATION_TENSOR_LAYOUT = "tensor_layout"
DATAFLOW_OPERATION_CONTRACT_KINDS = (
    DATAFLOW_OPERATION_RANGE_COARSENING,
    DATAFLOW_OPERATION_PIPELINE,
    DATAFLOW_OPERATION_RESHARED_TRANSPORT,
    DATAFLOW_OPERATION_CROSS_HANDLER_HANDOFF,
    DATAFLOW_OPERATION_TENSOR_LAYOUT,
)

DATAFLOW_RANGE_INDEPENDENT = "independent"
DATAFLOW_RANGE_ORDERED = "ordered"
DATAFLOW_RANGE_REDUCTION = "reduction"
DATAFLOW_RANGE_DEPENDENCES = (
    DATAFLOW_RANGE_INDEPENDENT,
    DATAFLOW_RANGE_ORDERED,
    DATAFLOW_RANGE_REDUCTION,
)
DATAFLOW_REMAINDER_EXACT = "exact"
DATAFLOW_REMAINDER_PREDICATE = "predicate"
DATAFLOW_REMAINDER_PAD = "pad"
DATAFLOW_REMAINDER_POLICIES = (
    DATAFLOW_REMAINDER_EXACT,
    DATAFLOW_REMAINDER_PREDICATE,
    DATAFLOW_REMAINDER_PAD,
)

DATAFLOW_PIPELINE_PRODUCER_VISIBLE = "producer_visible"
DATAFLOW_PIPELINE_CONSUMER_VISIBLE = "consumer_visible"
DATAFLOW_PIPELINE_OWNED_COMPLETION = "pipeline_owned"
DATAFLOW_PIPELINE_COMPLETION_SEMANTICS = (
    DATAFLOW_PIPELINE_PRODUCER_VISIBLE,
    DATAFLOW_PIPELINE_CONSUMER_VISIBLE,
    DATAFLOW_PIPELINE_OWNED_COMPLETION,
)
DATAFLOW_PIPELINE_CONSUMER_RELEASE = "consumer_release"
DATAFLOW_PIPELINE_OWNED_RELEASE = "pipeline_release"
DATAFLOW_PIPELINE_SCOPE_EXIT_RELEASE = "scope_exit"
DATAFLOW_PIPELINE_RELEASE_SEMANTICS = (
    DATAFLOW_PIPELINE_CONSUMER_RELEASE,
    DATAFLOW_PIPELINE_OWNED_RELEASE,
    DATAFLOW_PIPELINE_SCOPE_EXIT_RELEASE,
)
DATAFLOW_PIPELINE_SYNC_TRANSFER = "transfer"
DATAFLOW_PIPELINE_SYNC_PIPELINE = "pipeline"
DATAFLOW_PIPELINE_SYNC_CALLER = "caller"
DATAFLOW_PIPELINE_SYNCHRONIZATION_OWNERS = (
    DATAFLOW_PIPELINE_SYNC_TRANSFER,
    DATAFLOW_PIPELINE_SYNC_PIPELINE,
    DATAFLOW_PIPELINE_SYNC_CALLER,
)
DATAFLOW_PIPELINE_EVICT_NORMAL = "evict_normal"
DATAFLOW_PIPELINE_EVICT_FIRST = "evict_first"
DATAFLOW_PIPELINE_EVICT_LAST = "evict_last"
DATAFLOW_PIPELINE_EVICTION_HINTS = (
    DATAFLOW_PIPELINE_EVICT_NORMAL,
    DATAFLOW_PIPELINE_EVICT_FIRST,
    DATAFLOW_PIPELINE_EVICT_LAST,
)
DATAFLOW_PIPELINE_MATERIALIZE_COPY = "copy"
DATAFLOW_PIPELINE_MATERIALIZE_RESIDENT = "resident"
DATAFLOW_PIPELINE_MATERIALIZATIONS = (
    DATAFLOW_PIPELINE_MATERIALIZE_COPY,
    DATAFLOW_PIPELINE_MATERIALIZE_RESIDENT,
)
DATAFLOW_PIPELINE_RESIDENT_IMPLEMENTATION = "common.resident"

DATAFLOW_TRANSPORT_AUTO = "auto"
DATAFLOW_TRANSPORT_HBM = "hbm"
DATAFLOW_TRANSPORT_ALL_GATHER = "all_gather"
DATAFLOW_TRANSPORT_STREAMED = "streamed"
DATAFLOW_TRANSPORT_FAMILIES = (
    DATAFLOW_TRANSPORT_AUTO,
    DATAFLOW_TRANSPORT_HBM,
    DATAFLOW_TRANSPORT_ALL_GATHER,
    DATAFLOW_TRANSPORT_STREAMED,
)
DATAFLOW_TRANSPORT_LOGICAL_ORDER = "logical_order"
DATAFLOW_TRANSPORT_INDEPENDENT_ORDER = "independent"
DATAFLOW_TRANSPORT_ACCESS_ORDERS = (
    DATAFLOW_TRANSPORT_LOGICAL_ORDER,
    DATAFLOW_TRANSPORT_INDEPENDENT_ORDER,
)

DATAFLOW_HANDOFF_NEUTRAL_TAIL = "neutral"
DATAFLOW_HANDOFF_NOOP_TAIL = "noop"
DATAFLOW_HANDOFF_TAIL_POLICIES = (
    DATAFLOW_HANDOFF_NEUTRAL_TAIL,
    DATAFLOW_HANDOFF_NOOP_TAIL,
)

DATAFLOW_LAYOUT_LINEAR = "linear"
DATAFLOW_LAYOUT_MATRIX_SWIZZLE = "matrix_swizzle"
DATAFLOW_LAYOUT_INTERLEAVED = "interleaved"
DATAFLOW_LAYOUT_FAMILIES = (
    DATAFLOW_LAYOUT_LINEAR,
    DATAFLOW_LAYOUT_MATRIX_SWIZZLE,
    DATAFLOW_LAYOUT_INTERLEAVED,
)

DATAFLOW_RANGE_STAGE_GRAPH_IMPLEMENTATION = "dataflow.range.stage_graph.v1"
DATAFLOW_PIPELINE_PLAN_IMPLEMENTATION = "tilelang.pipeline.dataflow_plan.v1"
DATAFLOW_TRANSPORT_HBM_IMPLEMENTATION = "dataflow.transport.hbm.v1"
DATAFLOW_TRANSPORT_ALL_GATHER_IMPLEMENTATION = "dataflow.transport.cluster_all_gather.v1"
DATAFLOW_TRANSPORT_STREAMED_IMPLEMENTATION = "dataflow.transport.cluster_streamed.v1"
DATAFLOW_HANDOFF_QUEUE_IMPLEMENTATION = "dataflow.handoff.queue_lookahead.v1"
DATAFLOW_LAYOUT_LINEAR_IMPLEMENTATION = "dataflow.layout.linear.v1"
DATAFLOW_LAYOUT_MATRIX_SWIZZLE_IMPLEMENTATION = "dataflow.layout.matrix_swizzle.v1"

DATAFLOW_RANGE_CONTRACT_ATTR = "range_contract"
DATAFLOW_PIPELINE_CONTRACT_ATTR = "pipeline_contract"
DATAFLOW_TRANSPORT_CONTRACT_ATTR = "transport_contract"
DATAFLOW_HANDOFF_CONTRACT_ATTR = "handoff_contract"
DATAFLOW_LAYOUT_CONTRACTS_ATTR = "layout_contracts"


def serialize_canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as err:
        raise TypeError("Dataflow operation contracts must contain only finite JSON values") from err


def fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(serialize_canonical_json(value).encode("utf-8")).hexdigest()


def positive_int(value: Any, name: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def nonnegative_int(
    value: Any,
    name: str,
    *,
    allow_none: bool = False,
) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def validate_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {value!r}")
    return value


def choice(value: Any, name: str, choices: Sequence[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {value!r}")
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ValueError(f"unsupported {name} {value!r}; expected one of: " + ", ".join(choices))
    return normalized


def diagnostic_name(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("diagnostic_name must be a non-empty string or None")
    return value.strip()


def require_schema(value: int, expected: int, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"Unsupported {context} schema version {value!r}; expected {expected}")


def reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str],
    context: str,
) -> None:
    unknown = set(value).difference(allowed)
    if unknown:
        raise ValueError(f"{context} has unknown fields {sorted(unknown)!r}")


def require_fields(
    value: Mapping[str, Any],
    required: set[str],
    context: str,
) -> None:
    missing = required.difference(value)
    if missing:
        raise ValueError(f"{context} is missing fields {sorted(missing)!r}")


class DataflowOperationRequest:
    """Base protocol for canonical compile-boundary operation requests."""

    KIND: ClassVar[str]
    schema_version: int
    diagnostic_name: str | None

    def semantic_fields(self) -> dict[str, Any]:
        raise NotImplementedError

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.KIND,
            **self.semantic_fields(),
        }

    @property
    def canonical_json(self) -> str:
        return serialize_canonical_json(self.canonical_payload())

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def to_dict(
        self,
        *,
        include_fingerprint: bool = True,
        include_diagnostics: bool = False,
    ) -> dict[str, Any]:
        result = self.canonical_payload()
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        if include_diagnostics and self.diagnostic_name is not None:
            result["diagnostic_name"] = self.diagnostic_name
        return result


@dataclass(frozen=True)
class DataflowRangeCoarseningRequest(DataflowOperationRequest):
    """Target-independent permission to coarsen one logical range."""

    KIND: ClassVar[str] = DATAFLOW_OPERATION_RANGE_COARSENING

    logical_tile_extent: int
    logical_range_extent: int | None = None
    handler_range_extent: int | None = None
    output_tile_arity: int = 1
    dependence: str = DATAFLOW_RANGE_INDEPENDENT
    remainder_policy: str = DATAFLOW_REMAINDER_PREDICATE
    allow_fusion: bool = True
    diagnostic_name: str | None = None
    schema_version: int = DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_schema(
            self.schema_version,
            DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION,
            "Dataflow range coarsening request",
        )
        positive_int(self.logical_tile_extent, "logical_tile_extent")
        positive_int(
            self.logical_range_extent,
            "logical_range_extent",
            allow_none=True,
        )
        handler_extent = positive_int(
            self.handler_range_extent,
            "handler_range_extent",
            allow_none=True,
        )
        positive_int(self.output_tile_arity, "output_tile_arity")
        object.__setattr__(
            self,
            "dependence",
            choice(self.dependence, "range dependence", DATAFLOW_RANGE_DEPENDENCES),
        )
        object.__setattr__(
            self,
            "remainder_policy",
            choice(
                self.remainder_policy,
                "range remainder policy",
                DATAFLOW_REMAINDER_POLICIES,
            ),
        )
        validate_bool(self.allow_fusion, "allow_fusion")
        object.__setattr__(
            self,
            "diagnostic_name",
            diagnostic_name(self.diagnostic_name),
        )
        if handler_extent is not None:
            if handler_extent < self.logical_tile_extent:
                raise ValueError("handler_range_extent cannot be smaller than logical_tile_extent")
            if self.remainder_policy == DATAFLOW_REMAINDER_EXACT and handler_extent % self.logical_tile_extent:
                raise ValueError("exact range remainder policy requires handler_range_extent to contain whole logical tiles")
            if not self.allow_fusion and handler_extent != self.logical_tile_extent:
                raise ValueError("range fusion must be allowed when handler_range_extent coarsens multiple logical tiles")
            if (
                self.remainder_policy == DATAFLOW_REMAINDER_EXACT
                and self.logical_range_extent is not None
                and self.logical_range_extent % handler_extent
            ):
                raise ValueError("exact range remainder policy requires logical_range_extent to be divisible by handler_range_extent")

    @property
    def tiles_per_handler(self) -> int | None:
        if self.handler_range_extent is None:
            return None
        return (self.handler_range_extent + self.logical_tile_extent - 1) // self.logical_tile_extent

    def semantic_fields(self) -> dict[str, Any]:
        return {
            "logical_range_extent": self.logical_range_extent,
            "logical_tile_extent": self.logical_tile_extent,
            "handler_range_extent": self.handler_range_extent,
            "output_tile_arity": self.output_tile_arity,
            "dependence": self.dependence,
            "remainder_policy": self.remainder_policy,
            "allow_fusion": self.allow_fusion,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowRangeCoarseningRequest:
        return cls(
            logical_tile_extent=value.get("logical_tile_extent"),
            logical_range_extent=value.get("logical_range_extent"),
            handler_range_extent=value.get("handler_range_extent"),
            output_tile_arity=value.get("output_tile_arity", 1),
            dependence=value.get("dependence", DATAFLOW_RANGE_INDEPENDENT),
            remainder_policy=value.get("remainder_policy", DATAFLOW_REMAINDER_PREDICATE),
            allow_fusion=value.get("allow_fusion", True),
            diagnostic_name=value.get("diagnostic_name"),
            schema_version=value.get("schema_version", DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class DataflowPipelineTransfer:
    """One logical transfer producing a stage-local buffer."""

    destination_buffer_index: int
    logical_extent: tuple[int | None, ...]
    bytes_per_stage: int
    materialization: str = DATAFLOW_PIPELINE_MATERIALIZE_COPY
    producer_partition: int | None = None
    async_permitted: bool = True
    multicast_permitted: bool = False
    eviction_hint: str = DATAFLOW_PIPELINE_EVICT_NORMAL

    def __post_init__(self) -> None:
        nonnegative_int(
            self.destination_buffer_index,
            "pipeline transfer destination_buffer_index",
        )
        if not isinstance(self.logical_extent, (tuple, list)):
            raise TypeError("pipeline transfer logical_extent must be a sequence")
        extent = tuple(self.logical_extent)
        if not extent:
            raise ValueError("pipeline transfer logical_extent cannot be empty")
        for index, value in enumerate(extent):
            positive_int(
                value,
                f"pipeline transfer logical_extent[{index}]",
                allow_none=True,
            )
        object.__setattr__(self, "logical_extent", extent)
        positive_int(self.bytes_per_stage, "pipeline transfer bytes_per_stage")
        object.__setattr__(
            self,
            "materialization",
            choice(
                self.materialization,
                "pipeline transfer materialization",
                DATAFLOW_PIPELINE_MATERIALIZATIONS,
            ),
        )
        nonnegative_int(
            self.producer_partition,
            "pipeline transfer producer_partition",
            allow_none=True,
        )
        validate_bool(self.async_permitted, "pipeline transfer async_permitted")
        validate_bool(self.multicast_permitted, "pipeline transfer multicast_permitted")
        if self.materialization == DATAFLOW_PIPELINE_MATERIALIZE_RESIDENT and (
            self.async_permitted or self.multicast_permitted or self.producer_partition is not None
        ):
            raise ValueError("resident pipeline buffers cannot request async, multicast, or producer partition materialization")
        object.__setattr__(
            self,
            "eviction_hint",
            choice(
                self.eviction_hint,
                "pipeline transfer eviction_hint",
                DATAFLOW_PIPELINE_EVICTION_HINTS,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "destination_buffer_index": self.destination_buffer_index,
            "logical_extent": list(self.logical_extent),
            "bytes_per_stage": self.bytes_per_stage,
            "materialization": self.materialization,
            "producer_partition": self.producer_partition,
            "async_permitted": self.async_permitted,
            "multicast_permitted": self.multicast_permitted,
            "eviction_hint": self.eviction_hint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowPipelineTransfer:
        if not isinstance(value, Mapping):
            raise TypeError("pipeline transfer must be a mapping")
        expected = {item.name for item in fields(cls)}
        if set(value) != expected:
            raise ValueError(
                "pipeline transfer fields do not match schema: "
                f"missing={sorted(expected - set(value))!r}, "
                f"unknown={sorted(set(value) - expected)!r}"
            )
        if not isinstance(value["logical_extent"], (tuple, list)):
            raise TypeError("pipeline transfer logical_extent must be a sequence")
        return cls(
            destination_buffer_index=value["destination_buffer_index"],
            logical_extent=tuple(value["logical_extent"]),
            bytes_per_stage=value["bytes_per_stage"],
            materialization=value["materialization"],
            producer_partition=value["producer_partition"],
            async_permitted=value["async_permitted"],
            multicast_permitted=value["multicast_permitted"],
            eviction_hint=value["eviction_hint"],
        )


@dataclass(frozen=True)
class DataflowPipelineGemm:
    """One GEMM consumer and its accumulator dependence."""

    input_buffer_indices: tuple[int, int]
    accumulator_index: int
    accumulator_dependency: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.input_buffer_indices, (tuple, list)):
            raise TypeError("pipeline GEMM input_buffer_indices must be a sequence")
        inputs = tuple(self.input_buffer_indices)
        if len(inputs) != 2:
            raise ValueError("pipeline GEMM requires exactly two input buffers")
        for index, value in enumerate(inputs):
            nonnegative_int(value, f"pipeline GEMM input_buffer_indices[{index}]")
        object.__setattr__(self, "input_buffer_indices", inputs)
        nonnegative_int(self.accumulator_index, "pipeline GEMM accumulator_index")
        validate_bool(
            self.accumulator_dependency,
            "pipeline GEMM accumulator_dependency",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_buffer_indices": list(self.input_buffer_indices),
            "accumulator_index": self.accumulator_index,
            "accumulator_dependency": self.accumulator_dependency,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowPipelineGemm:
        if not isinstance(value, Mapping):
            raise TypeError("pipeline GEMM must be a mapping")
        expected = {item.name for item in fields(cls)}
        if set(value) != expected:
            raise ValueError(
                "pipeline GEMM fields do not match schema: "
                f"missing={sorted(expected - set(value))!r}, "
                f"unknown={sorted(set(value) - expected)!r}"
            )
        if not isinstance(value["input_buffer_indices"], (tuple, list)):
            raise TypeError("pipeline GEMM input_buffer_indices must be a sequence")
        return cls(
            input_buffer_indices=tuple(value["input_buffer_indices"]),
            accumulator_index=value["accumulator_index"],
            accumulator_dependency=value["accumulator_dependency"],
        )


@dataclass(frozen=True)
class DataflowPipelineBufferLifetime:
    """Producer, consumers, release dependence, and versioning permission."""

    buffer_index: int
    producer_transfer_index: int
    consumer_gemm_indices: tuple[int, ...]
    release_after_gemm_index: int
    allow_multiversion: bool = True

    def __post_init__(self) -> None:
        nonnegative_int(self.buffer_index, "pipeline lifetime buffer_index")
        nonnegative_int(
            self.producer_transfer_index,
            "pipeline lifetime producer_transfer_index",
        )
        if not isinstance(self.consumer_gemm_indices, (tuple, list)):
            raise TypeError("pipeline lifetime consumer_gemm_indices must be a sequence")
        consumers = tuple(self.consumer_gemm_indices)
        if not consumers:
            raise ValueError("pipeline lifetime requires at least one GEMM consumer")
        for index, value in enumerate(consumers):
            nonnegative_int(
                value,
                f"pipeline lifetime consumer_gemm_indices[{index}]",
            )
        if consumers != tuple(sorted(set(consumers))):
            raise ValueError("pipeline lifetime consumer GEMM indices must be sorted and unique")
        object.__setattr__(self, "consumer_gemm_indices", consumers)
        nonnegative_int(
            self.release_after_gemm_index,
            "pipeline lifetime release_after_gemm_index",
        )
        if self.release_after_gemm_index != consumers[-1]:
            raise ValueError("pipeline lifetime release must follow its final GEMM consumer")
        validate_bool(self.allow_multiversion, "pipeline lifetime allow_multiversion")

    def to_dict(self) -> dict[str, Any]:
        return {
            "buffer_index": self.buffer_index,
            "producer_transfer_index": self.producer_transfer_index,
            "consumer_gemm_indices": list(self.consumer_gemm_indices),
            "release_after_gemm_index": self.release_after_gemm_index,
            "allow_multiversion": self.allow_multiversion,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> DataflowPipelineBufferLifetime:
        if not isinstance(value, Mapping):
            raise TypeError("pipeline buffer lifetime must be a mapping")
        expected = {item.name for item in fields(cls)}
        if set(value) != expected:
            raise ValueError(
                "pipeline buffer lifetime fields do not match schema: "
                f"missing={sorted(expected - set(value))!r}, "
                f"unknown={sorted(set(value) - expected)!r}"
            )
        if not isinstance(value["consumer_gemm_indices"], (tuple, list)):
            raise TypeError("pipeline lifetime consumer_gemm_indices must be a sequence")
        return cls(
            buffer_index=value["buffer_index"],
            producer_transfer_index=value["producer_transfer_index"],
            consumer_gemm_indices=tuple(value["consumer_gemm_indices"]),
            release_after_gemm_index=value["release_after_gemm_index"],
            allow_multiversion=value["allow_multiversion"],
        )


@dataclass(frozen=True)
class DataflowPipelineRequest(DataflowOperationRequest):
    """Logical multi-transfer/GEMM dataflow before implementation selection."""

    KIND: ClassVar[str] = DATAFLOW_OPERATION_PIPELINE

    transfers: tuple[DataflowPipelineTransfer, ...]
    gemms: tuple[DataflowPipelineGemm, ...]
    buffer_lifetimes: tuple[DataflowPipelineBufferLifetime, ...]
    stage_budget: int | None = None
    max_outstanding: int | None = None
    producer_threads: int | None = None
    consumer_threads: int | None = None
    fixed_register_bytes: int = 0
    register_bytes_per_stage: int = 0
    max_shared_memory_bytes: int | None = None
    max_register_bytes: int | None = None
    max_barrier_count: int | None = None
    synchronization_owner: str = DATAFLOW_PIPELINE_SYNC_PIPELINE
    completion_semantics: str = DATAFLOW_PIPELINE_CONSUMER_VISIBLE
    release_semantics: str = DATAFLOW_PIPELINE_CONSUMER_RELEASE
    diagnostic_name: str | None = None
    schema_version: int = DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_schema(
            self.schema_version,
            DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION,
            "Dataflow pipeline dataflow request",
        )
        for name, expected_type in (
            ("transfers", DataflowPipelineTransfer),
            ("gemms", DataflowPipelineGemm),
            ("buffer_lifetimes", DataflowPipelineBufferLifetime),
        ):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or not values:
                raise ValueError(f"pipeline {name} must be a non-empty sequence")
            normalized = tuple(values)
            if any(not isinstance(item, expected_type) for item in normalized):
                raise TypeError(f"pipeline {name} must contain typed entries")
            object.__setattr__(self, name, normalized)

        buffer_count = len(self.buffer_lifetimes)
        expected_buffers = tuple(range(buffer_count))
        transfer_buffers = tuple(transfer.destination_buffer_index for transfer in self.transfers)
        lifetime_buffers = tuple(lifetime.buffer_index for lifetime in self.buffer_lifetimes)
        if transfer_buffers != expected_buffers or lifetime_buffers != expected_buffers:
            raise ValueError("pipeline buffers and transfers must use canonical contiguous indices")
        if len(self.transfers) != buffer_count:
            raise ValueError("pipeline requires one producer transfer per buffer")
        gemm_count = len(self.gemms)
        for gemm_index, gemm in enumerate(self.gemms):
            if any(index >= buffer_count for index in gemm.input_buffer_indices):
                raise ValueError(f"pipeline GEMM {gemm_index} references an unknown buffer")
        for lifetime in self.buffer_lifetimes:
            if lifetime.producer_transfer_index >= len(self.transfers):
                raise ValueError("pipeline lifetime references an unknown transfer")
            producer = self.transfers[lifetime.producer_transfer_index]
            if producer.destination_buffer_index != lifetime.buffer_index:
                raise ValueError("pipeline lifetime producer does not match its buffer")
            if lifetime.release_after_gemm_index >= gemm_count:
                raise ValueError("pipeline lifetime release references an unknown GEMM")
            expected_consumers = tuple(index for index, gemm in enumerate(self.gemms) if lifetime.buffer_index in gemm.input_buffer_indices)
            if lifetime.consumer_gemm_indices != expected_consumers:
                raise ValueError("pipeline lifetime consumers do not match GEMM dependencies")
        accumulator_groups: dict[int, list[tuple[int, DataflowPipelineGemm]]] = {}
        for gemm_index, gemm in enumerate(self.gemms):
            accumulator_groups.setdefault(gemm.accumulator_index, []).append((gemm_index, gemm))
        for group in accumulator_groups.values():
            if any(not gemm.accumulator_dependency for _, gemm in group[1:]):
                raise ValueError("later GEMMs in one accumulator group require an additive dependence")

        positive_int(self.stage_budget, "stage_budget", allow_none=True)
        positive_int(self.max_outstanding, "max_outstanding", allow_none=True)
        positive_int(self.producer_threads, "producer_threads", allow_none=True)
        positive_int(self.consumer_threads, "consumer_threads", allow_none=True)
        assigned_partitions = tuple(transfer.producer_partition for transfer in self.transfers if transfer.producer_partition is not None)
        if assigned_partitions:
            expected_partitions = tuple(range(max(assigned_partitions) + 1))
            if tuple(sorted(set(assigned_partitions))) != expected_partitions:
                raise ValueError("pipeline producer partitions must use canonical contiguous indices")
            if self.producer_threads is None:
                raise ValueError("pipeline producer partitions require a producer thread budget")
            if any(transfer.producer_partition is None for transfer in self.transfers if transfer.async_permitted):
                raise ValueError("pipeline async transfers must all declare a producer partition when partitioning is requested")
            if any(transfer.producer_partition is not None for transfer in self.transfers if not transfer.async_permitted):
                raise ValueError("pipeline synchronous transfers cannot declare a producer partition")
        nonnegative_int(self.fixed_register_bytes, "fixed_register_bytes")
        nonnegative_int(
            self.register_bytes_per_stage,
            "register_bytes_per_stage",
        )
        for name in (
            "max_shared_memory_bytes",
            "max_register_bytes",
            "max_barrier_count",
        ):
            nonnegative_int(getattr(self, name), name, allow_none=True)
        object.__setattr__(
            self,
            "synchronization_owner",
            choice(
                self.synchronization_owner,
                "pipeline synchronization_owner",
                DATAFLOW_PIPELINE_SYNCHRONIZATION_OWNERS,
            ),
        )
        object.__setattr__(
            self,
            "completion_semantics",
            choice(
                self.completion_semantics,
                "pipeline completion semantics",
                DATAFLOW_PIPELINE_COMPLETION_SEMANTICS,
            ),
        )
        object.__setattr__(
            self,
            "release_semantics",
            choice(
                self.release_semantics,
                "pipeline release semantics",
                DATAFLOW_PIPELINE_RELEASE_SEMANTICS,
            ),
        )
        object.__setattr__(
            self,
            "diagnostic_name",
            diagnostic_name(self.diagnostic_name),
        )

    def semantic_fields(self) -> dict[str, Any]:
        return {
            "transfers": [item.to_dict() for item in self.transfers],
            "gemms": [item.to_dict() for item in self.gemms],
            "buffer_lifetimes": [item.to_dict() for item in self.buffer_lifetimes],
            "stage_budget": self.stage_budget,
            "max_outstanding": self.max_outstanding,
            "producer_threads": self.producer_threads,
            "consumer_threads": self.consumer_threads,
            "fixed_register_bytes": self.fixed_register_bytes,
            "register_bytes_per_stage": self.register_bytes_per_stage,
            "max_shared_memory_bytes": self.max_shared_memory_bytes,
            "max_register_bytes": self.max_register_bytes,
            "max_barrier_count": self.max_barrier_count,
            "synchronization_owner": self.synchronization_owner,
            "completion_semantics": self.completion_semantics,
            "release_semantics": self.release_semantics,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowPipelineRequest:
        for name in ("transfers", "gemms", "buffer_lifetimes"):
            if not isinstance(value.get(name), (tuple, list)):
                raise TypeError(f"pipeline {name} must be a sequence")
        return cls(
            transfers=tuple(DataflowPipelineTransfer.from_dict(item) for item in value["transfers"]),
            gemms=tuple(DataflowPipelineGemm.from_dict(item) for item in value["gemms"]),
            buffer_lifetimes=tuple(DataflowPipelineBufferLifetime.from_dict(item) for item in value["buffer_lifetimes"]),
            stage_budget=value["stage_budget"],
            max_outstanding=value["max_outstanding"],
            producer_threads=value["producer_threads"],
            consumer_threads=value["consumer_threads"],
            fixed_register_bytes=value["fixed_register_bytes"],
            register_bytes_per_stage=value["register_bytes_per_stage"],
            max_shared_memory_bytes=value["max_shared_memory_bytes"],
            max_register_bytes=value["max_register_bytes"],
            max_barrier_count=value["max_barrier_count"],
            synchronization_owner=value["synchronization_owner"],
            completion_semantics=value["completion_semantics"],
            release_semantics=value["release_semantics"],
            diagnostic_name=value.get("diagnostic_name"),
            schema_version=value["schema_version"],
        )


@dataclass(frozen=True)
class DataflowResharedFieldMapping:
    """Map one physical tensor field into a logical reshared tensor axis."""

    field_index: int
    physical_value_axis: int
    physical_tile_axis: int | None = None

    def __post_init__(self) -> None:
        nonnegative_int(self.field_index, "field_index")
        nonnegative_int(self.physical_value_axis, "physical_value_axis")
        nonnegative_int(
            self.physical_tile_axis,
            "physical_tile_axis",
            allow_none=True,
        )
        if self.physical_tile_axis == self.physical_value_axis:
            raise ValueError("physical_tile_axis and physical_value_axis must differ")

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_index": self.field_index,
            "physical_value_axis": self.physical_value_axis,
            "physical_tile_axis": self.physical_tile_axis,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowResharedFieldMapping:
        if not isinstance(value, Mapping):
            raise TypeError("Dataflow reshared field mapping must be a mapping")
        reject_unknown(
            value,
            {"field_index", "physical_value_axis", "physical_tile_axis"},
            "Dataflow reshared field mapping",
        )
        return cls(
            field_index=value.get("field_index"),
            physical_value_axis=value.get("physical_value_axis"),
            physical_tile_axis=value.get("physical_tile_axis"),
        )


@dataclass(frozen=True)
class DataflowResharedTransportRequest(DataflowOperationRequest):
    """Logical cross-placement transport request with optional family constraint."""

    KIND: ClassVar[str] = DATAFLOW_OPERATION_RESHARED_TRANSPORT

    family: str = DATAFLOW_TRANSPORT_AUTO
    logical_output_arity: int = 1
    physical_output_arity: int = 1
    slot_bytes: int | None = None
    max_transaction_bytes: int | None = None
    max_temporary_bytes: int | None = None
    consumer_access_order: str = DATAFLOW_TRANSPORT_LOGICAL_ORDER
    allow_multicast: bool = True
    field_mappings: tuple[DataflowResharedFieldMapping, ...] = ()
    diagnostic_name: str | None = None
    schema_version: int = DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_schema(
            self.schema_version,
            DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION,
            "Dataflow reshared transport request",
        )
        object.__setattr__(
            self,
            "family",
            choice(self.family, "transport family", DATAFLOW_TRANSPORT_FAMILIES),
        )
        positive_int(self.logical_output_arity, "logical_output_arity")
        positive_int(self.physical_output_arity, "physical_output_arity")
        if self.physical_output_arity > self.logical_output_arity:
            raise ValueError("physical_output_arity cannot exceed logical_output_arity")
        positive_int(self.slot_bytes, "slot_bytes", allow_none=True)
        nonnegative_int(
            self.max_transaction_bytes,
            "max_transaction_bytes",
            allow_none=True,
        )
        nonnegative_int(
            self.max_temporary_bytes,
            "max_temporary_bytes",
            allow_none=True,
        )
        object.__setattr__(
            self,
            "consumer_access_order",
            choice(
                self.consumer_access_order,
                "consumer access order",
                DATAFLOW_TRANSPORT_ACCESS_ORDERS,
            ),
        )
        validate_bool(self.allow_multicast, "allow_multicast")
        mappings = tuple(
            item if isinstance(item, DataflowResharedFieldMapping) else DataflowResharedFieldMapping.from_dict(item)
            for item in self.field_mappings
        )
        mappings = tuple(sorted(mappings, key=lambda item: item.field_index))
        indices = tuple(item.field_index for item in mappings)
        if len(indices) != len(set(indices)):
            raise ValueError("reshared field mappings cannot repeat a field_index")
        object.__setattr__(self, "field_mappings", mappings)
        object.__setattr__(
            self,
            "diagnostic_name",
            diagnostic_name(self.diagnostic_name),
        )

    @property
    def explicit(self) -> bool:
        return self.family != DATAFLOW_TRANSPORT_AUTO

    def semantic_fields(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "logical_output_arity": self.logical_output_arity,
            "physical_output_arity": self.physical_output_arity,
            "slot_bytes": self.slot_bytes,
            "max_transaction_bytes": self.max_transaction_bytes,
            "max_temporary_bytes": self.max_temporary_bytes,
            "consumer_access_order": self.consumer_access_order,
            "allow_multicast": self.allow_multicast,
            "field_mappings": [item.to_dict() for item in self.field_mappings],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowResharedTransportRequest:
        return cls(
            family=value.get("family", DATAFLOW_TRANSPORT_AUTO),
            logical_output_arity=value.get("logical_output_arity", 1),
            physical_output_arity=value.get("physical_output_arity", 1),
            slot_bytes=value.get("slot_bytes"),
            max_transaction_bytes=value.get("max_transaction_bytes"),
            max_temporary_bytes=value.get("max_temporary_bytes"),
            consumer_access_order=value.get("consumer_access_order", DATAFLOW_TRANSPORT_LOGICAL_ORDER),
            allow_multicast=value.get("allow_multicast", True),
            field_mappings=tuple(DataflowResharedFieldMapping.from_dict(item) for item in value.get("field_mappings", ())),
            diagnostic_name=value.get("diagnostic_name"),
            schema_version=value.get("schema_version", DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION),
        )

    @classmethod
    def from_legacy_policy(
        cls,
        policy: str,
        **kwargs: Any,
    ) -> DataflowResharedTransportRequest:
        try:
            family = {
                "hbm_all_gather": DATAFLOW_TRANSPORT_HBM,
                "cluster_shared_all_gather": DATAFLOW_TRANSPORT_ALL_GATHER,
                "cluster_shared_pull_ring": DATAFLOW_TRANSPORT_STREAMED,
            }[str(policy)]
        except KeyError as err:
            raise ValueError(f"unsupported legacy reshared policy {policy!r}") from err
        return cls(family=family, **kwargs)


@dataclass(frozen=True)
class DataflowCrossHandlerHandoffRequest(DataflowOperationRequest):
    """Logical lookahead binding between two handler stages."""

    KIND: ClassVar[str] = DATAFLOW_OPERATION_CROSS_HANDLER_HANDOFF

    consumer_stage_id: int
    lookahead_distance: int = 1
    buffer_stages: int = 1
    value_bindings: tuple[int, ...] = ()
    tail_policy: str = DATAFLOW_HANDOFF_NEUTRAL_TAIL
    diagnostic_name: str | None = None
    schema_version: int = DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_schema(
            self.schema_version,
            DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION,
            "Dataflow cross-handler handoff request",
        )
        nonnegative_int(self.consumer_stage_id, "consumer_stage_id")
        positive_int(self.lookahead_distance, "lookahead_distance")
        positive_int(self.buffer_stages, "buffer_stages")
        if not isinstance(self.value_bindings, (tuple, list)):
            raise TypeError("value_bindings must be a sequence")
        bindings = tuple(self.value_bindings)
        for index, binding in enumerate(bindings):
            nonnegative_int(binding, f"value_bindings[{index}]")
        if len(bindings) != len(set(bindings)):
            raise ValueError("value_bindings must be unique")
        object.__setattr__(self, "value_bindings", bindings)
        object.__setattr__(
            self,
            "tail_policy",
            choice(
                self.tail_policy,
                "handoff tail policy",
                DATAFLOW_HANDOFF_TAIL_POLICIES,
            ),
        )
        object.__setattr__(
            self,
            "diagnostic_name",
            diagnostic_name(self.diagnostic_name),
        )

    def semantic_fields(self) -> dict[str, Any]:
        return {
            "consumer_stage_id": self.consumer_stage_id,
            "lookahead_distance": self.lookahead_distance,
            "buffer_stages": self.buffer_stages,
            "value_bindings": list(self.value_bindings),
            "tail_policy": self.tail_policy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowCrossHandlerHandoffRequest:
        return cls(
            consumer_stage_id=value.get("consumer_stage_id"),
            lookahead_distance=value.get("lookahead_distance", 1),
            buffer_stages=value.get("buffer_stages", 1),
            value_bindings=tuple(value.get("value_bindings", ())),
            tail_policy=value.get("tail_policy", DATAFLOW_HANDOFF_NEUTRAL_TAIL),
            diagnostic_name=value.get("diagnostic_name"),
            schema_version=value.get("schema_version", DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class DataflowTensorLayoutRequest(DataflowOperationRequest):
    """Tensor-field-ordinal logical-to-physical layout request."""

    KIND: ClassVar[str] = DATAFLOW_OPERATION_TENSOR_LAYOUT

    field_index: int
    logical_rank: int
    layout_family: str = DATAFLOW_LAYOUT_LINEAR
    major_axis: int | None = None
    alignment_bytes: int | None = None
    interleave: int | None = None
    allow_padding: bool = False
    diagnostic_name: str | None = None
    schema_version: int = DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_schema(
            self.schema_version,
            DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION,
            "Dataflow tensor layout request",
        )
        nonnegative_int(self.field_index, "field_index")
        positive_int(self.logical_rank, "logical_rank")
        object.__setattr__(
            self,
            "layout_family",
            choice(self.layout_family, "layout family", DATAFLOW_LAYOUT_FAMILIES),
        )
        if self.major_axis is not None:
            nonnegative_int(self.major_axis, "major_axis")
            if self.major_axis >= self.logical_rank:
                raise ValueError("major_axis must be smaller than logical_rank")
        if self.alignment_bytes is not None:
            alignment = positive_int(self.alignment_bytes, "alignment_bytes")
            assert alignment is not None
            if alignment & (alignment - 1):
                raise ValueError("alignment_bytes must be a power of two")
        positive_int(self.interleave, "interleave", allow_none=True)
        validate_bool(self.allow_padding, "allow_padding")
        object.__setattr__(
            self,
            "diagnostic_name",
            diagnostic_name(self.diagnostic_name),
        )
        if self.layout_family == DATAFLOW_LAYOUT_LINEAR:
            if self.major_axis is not None or self.interleave is not None:
                raise ValueError("linear layout cannot declare major_axis or interleave")
        elif self.layout_family == DATAFLOW_LAYOUT_MATRIX_SWIZZLE:
            if self.logical_rank < 2 or self.major_axis is None:
                raise ValueError("matrix_swizzle layout requires logical_rank >= 2 and major_axis")
            if self.interleave is not None:
                raise ValueError("matrix_swizzle layout cannot declare interleave")
        elif self.interleave is None or self.interleave <= 1:
            raise ValueError("interleaved layout requires interleave > 1")

    def semantic_fields(self) -> dict[str, Any]:
        return {
            "field_index": self.field_index,
            "logical_rank": self.logical_rank,
            "layout_family": self.layout_family,
            "major_axis": self.major_axis,
            "alignment_bytes": self.alignment_bytes,
            "interleave": self.interleave,
            "allow_padding": self.allow_padding,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowTensorLayoutRequest:
        return cls(
            field_index=value.get("field_index"),
            logical_rank=value.get("logical_rank"),
            layout_family=value.get("layout_family", DATAFLOW_LAYOUT_LINEAR),
            major_axis=value.get("major_axis"),
            alignment_bytes=value.get("alignment_bytes"),
            interleave=value.get("interleave"),
            allow_padding=value.get("allow_padding", False),
            diagnostic_name=value.get("diagnostic_name"),
            schema_version=value.get("schema_version", DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION),
        )


_REQUEST_TYPES: dict[str, type[DataflowOperationRequest]] = {
    request_type.KIND: request_type
    for request_type in (
        DataflowRangeCoarseningRequest,
        DataflowPipelineRequest,
        DataflowResharedTransportRequest,
        DataflowCrossHandlerHandoffRequest,
        DataflowTensorLayoutRequest,
    )
}


def dataflow_operation_request_from_dict(
    value: Mapping[str, Any],
) -> DataflowOperationRequest:
    if not isinstance(value, Mapping):
        raise TypeError(f"Dataflow operation request must be a mapping, got {type(value)!r}")
    missing = {"kind", "schema_version"}.difference(value)
    if missing:
        raise ValueError(f"serialized Dataflow operation request is missing versioned fields {sorted(missing)!r}")
    kind = value.get("kind")
    try:
        request_type = _REQUEST_TYPES[str(kind)]
    except KeyError as err:
        raise ValueError(f"unsupported Dataflow operation request kind {kind!r}") from err
    request_fields = {item.name for item in fields(request_type)}
    reject_unknown(
        value,
        request_fields | {"kind", "fingerprint"},
        f"Dataflow {kind} request",
    )
    require_fields(
        value,
        (request_fields - {"diagnostic_name"}) | {"kind"},
        f"Dataflow {kind} request",
    )
    request = request_type.from_dict(value)  # type: ignore[attr-defined]
    recorded_fingerprint = value.get("fingerprint")
    if recorded_fingerprint is not None and recorded_fingerprint != request.fingerprint:
        raise ValueError("Dataflow operation request fingerprint does not match payload")
    return request


def normalize_operation_request(
    value: DataflowOperationRequest | Mapping[str, Any],
    expected_type: type[DataflowOperationRequest],
) -> DataflowOperationRequest:
    request = (
        value
        if isinstance(value, DataflowOperationRequest)
        else dataflow_operation_request_from_dict(value)
        if isinstance(value, Mapping)
        else None
    )
    if request is None or not isinstance(request, expected_type):
        raise TypeError(f"expected {expected_type.__name__}, got {type(value).__name__}")
    return request


def normalize_layout_contracts(
    value: Any,
) -> tuple[DataflowTensorLayoutRequest, ...]:
    if value is None:
        return ()
    if isinstance(value, (DataflowTensorLayoutRequest, Mapping)):
        values = (value,)
    elif isinstance(value, (tuple, list)):
        values = tuple(value)
    else:
        raise TypeError("layout_contracts must be a layout request or a sequence of requests")
    requests = tuple(normalize_operation_request(item, DataflowTensorLayoutRequest) for item in values)
    field_indices = tuple(request.field_index for request in requests)
    if len(field_indices) != len(set(field_indices)):
        raise ValueError("layout_contracts cannot repeat a field_index")
    return tuple(sorted(requests, key=lambda request: request.field_index))


def legacy_policy_for_transport_request(
    request: DataflowResharedTransportRequest,
) -> str:
    if request.family == DATAFLOW_TRANSPORT_AUTO:
        raise ValueError("auto reshared transport requires compiler selection in a later lowering PR")
    return {
        DATAFLOW_TRANSPORT_HBM: "hbm_all_gather",
        DATAFLOW_TRANSPORT_ALL_GATHER: "cluster_shared_all_gather",
        DATAFLOW_TRANSPORT_STREAMED: "cluster_shared_pull_ring",
    }[request.family]


def transport_implementation_id(family: str) -> str:
    try:
        return {
            DATAFLOW_TRANSPORT_HBM: DATAFLOW_TRANSPORT_HBM_IMPLEMENTATION,
            DATAFLOW_TRANSPORT_ALL_GATHER: DATAFLOW_TRANSPORT_ALL_GATHER_IMPLEMENTATION,
            DATAFLOW_TRANSPORT_STREAMED: DATAFLOW_TRANSPORT_STREAMED_IMPLEMENTATION,
        }[family]
    except KeyError as err:
        raise ValueError(f"transport family {family!r} has no selected implementation") from err


def layout_implementation_id(request: DataflowTensorLayoutRequest) -> str:
    if request.layout_family == DATAFLOW_LAYOUT_LINEAR:
        return DATAFLOW_LAYOUT_LINEAR_IMPLEMENTATION
    if request.layout_family == DATAFLOW_LAYOUT_MATRIX_SWIZZLE:
        return DATAFLOW_LAYOUT_MATRIX_SWIZZLE_IMPLEMENTATION
    raise ValueError(f"layout family {request.layout_family!r} has no selected implementation")


@dataclass(frozen=True)
class DataflowOperationResourceEstimate:
    shared_memory_bytes: int | None = None
    register_bytes: int | None = None
    barrier_count: int | None = None
    slot_bytes: int | None = None
    transaction_bytes: int | None = None
    temporary_bytes: int | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            nonnegative_int(
                getattr(self, item.name),
                item.name,
                allow_none=True,
            )

    def to_dict(self) -> dict[str, int | None]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> DataflowOperationResourceEstimate:
        if not isinstance(value, Mapping):
            raise TypeError("operation resources must be a mapping")
        reject_unknown(
            value,
            {item.name for item in fields(cls)},
            "Dataflow operation resources",
        )
        return cls(**{item.name: value.get(item.name) for item in fields(cls)})


@dataclass(frozen=True)
class DataflowOperationCandidate:
    implementation_id: str
    legal: bool
    resources: DataflowOperationResourceEstimate = DataflowOperationResourceEstimate()
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.implementation_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.:-]*", self.implementation_id):
            raise ValueError(f"invalid operation implementation id {self.implementation_id!r}")
        validate_bool(self.legal, "candidate legal")
        if not isinstance(self.resources, DataflowOperationResourceEstimate):
            raise TypeError("candidate resources must be DataflowOperationResourceEstimate")
        reasons = tuple(str(reason) for reason in self.rejection_reasons if str(reason))
        object.__setattr__(self, "rejection_reasons", reasons)
        if self.legal == bool(reasons):
            raise ValueError("legal operation candidates cannot have rejection reasons and illegal candidates require at least one reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation_id": self.implementation_id,
            "legal": self.legal,
            "resources": self.resources.to_dict(),
            "rejection_reasons": list(self.rejection_reasons),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowOperationCandidate:
        if not isinstance(value, Mapping):
            raise TypeError("operation candidate must be a mapping")
        reject_unknown(
            value,
            {
                "implementation_id",
                "legal",
                "resources",
                "rejection_reasons",
            },
            "Dataflow operation candidate",
        )
        require_fields(
            value,
            {
                "implementation_id",
                "legal",
                "resources",
                "rejection_reasons",
            },
            "Dataflow operation candidate",
        )
        return cls(
            implementation_id=str(value.get("implementation_id", "")),
            legal=value.get("legal"),
            resources=DataflowOperationResourceEstimate.from_dict(value.get("resources", {})),
            rejection_reasons=tuple(value.get("rejection_reasons", ())),
        )


@dataclass(frozen=True)
class DataflowOperationDecision:
    request: DataflowOperationRequest
    candidates: tuple[DataflowOperationCandidate, ...]
    selected_implementation: str | None
    used_fallback: bool
    selection_reason: str
    schema_version: int = DATAFLOW_OPERATION_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_schema(
            self.schema_version,
            DATAFLOW_OPERATION_DECISION_SCHEMA_VERSION,
            "Dataflow operation decision",
        )
        if not isinstance(self.request, DataflowOperationRequest):
            raise TypeError("operation decision request must be typed")
        if not isinstance(self.candidates, (tuple, list)) or not self.candidates:
            raise ValueError("operation decision requires candidates")
        candidates = tuple(self.candidates)
        if any(not isinstance(item, DataflowOperationCandidate) for item in candidates):
            raise TypeError("operation decision candidates must be typed")
        ids = tuple(item.implementation_id for item in candidates)
        if len(ids) != len(set(ids)):
            raise ValueError("operation decision candidate ids must be unique")
        object.__setattr__(self, "candidates", candidates)
        validate_bool(self.used_fallback, "used_fallback")
        if not isinstance(self.selection_reason, str) or not self.selection_reason:
            raise ValueError("operation decision requires a selection_reason")
        if self.selected_implementation is None:
            if any(candidate.legal for candidate in candidates):
                raise ValueError("rejected operation decision cannot contain a legal candidate")
            if self.used_fallback:
                raise ValueError("rejected operation decision cannot use fallback")
        else:
            selected = tuple(candidate for candidate in candidates if candidate.implementation_id == self.selected_implementation)
            if len(selected) != 1 or not selected[0].legal:
                raise ValueError("operation decision must select exactly one legal candidate")

    @property
    def resources(self) -> DataflowOperationResourceEstimate | None:
        if self.selected_implementation is None:
            return None
        return next(candidate.resources for candidate in self.candidates if candidate.implementation_id == self.selected_implementation)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request": self.request.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected_implementation": self.selected_implementation,
            "used_fallback": self.used_fallback,
            "selection_reason": self.selection_reason,
            "resources": None if self.resources is None else self.resources.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        result = self.canonical_payload()
        result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowOperationDecision:
        if not isinstance(value, Mapping):
            raise TypeError("operation decision must be a mapping")
        reject_unknown(
            value,
            {
                "schema_version",
                "request",
                "candidates",
                "selected_implementation",
                "used_fallback",
                "selection_reason",
                "resources",
                "fingerprint",
            },
            "Dataflow operation decision",
        )
        require_fields(
            value,
            {
                "schema_version",
                "request",
                "candidates",
                "selected_implementation",
                "used_fallback",
                "selection_reason",
                "resources",
            },
            "Dataflow operation decision",
        )
        decision = cls(
            request=dataflow_operation_request_from_dict(value.get("request", {})),
            candidates=tuple(DataflowOperationCandidate.from_dict(item) for item in value.get("candidates", ())),
            selected_implementation=value.get("selected_implementation"),
            used_fallback=value.get("used_fallback", False),
            selection_reason=str(value.get("selection_reason", "")),
            schema_version=value.get("schema_version", DATAFLOW_OPERATION_DECISION_SCHEMA_VERSION),
        )
        recorded_fingerprint = value.get("fingerprint")
        if recorded_fingerprint is not None and recorded_fingerprint != decision.fingerprint:
            raise ValueError("Dataflow operation decision fingerprint does not match payload")
        recorded_resources = value["resources"]
        expected_resources = None if decision.resources is None else decision.resources.to_dict()
        if recorded_resources != expected_resources:
            raise ValueError("Dataflow operation decision resources do not match selection")
        return decision


def selected_operation_decision(
    request: DataflowOperationRequest,
    implementation_id: str,
    *,
    resources: DataflowOperationResourceEstimate | None = None,
    selection_reason: str,
    used_fallback: bool = False,
    rejected_candidates: Sequence[DataflowOperationCandidate] = (),
) -> DataflowOperationDecision:
    selected = DataflowOperationCandidate(
        implementation_id=implementation_id,
        legal=True,
        resources=resources or DataflowOperationResourceEstimate(),
    )
    return DataflowOperationDecision(
        request=request,
        candidates=(*tuple(rejected_candidates), selected),
        selected_implementation=implementation_id,
        used_fallback=used_fallback,
        selection_reason=selection_reason,
    )


def dataflow_operation_contract_schema_dict(
    *,
    include_fingerprint: bool = True,
) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "schema_version": DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION,
        "decision_schema_version": DATAFLOW_OPERATION_DECISION_SCHEMA_VERSION,
        "request_fields": {
            request_type.KIND: [item.name for item in fields(request_type) if item.name not in {"diagnostic_name", "schema_version"}]
            for request_type in _REQUEST_TYPES.values()
        },
        "request_domains": {
            "range_dependence": list(DATAFLOW_RANGE_DEPENDENCES),
            "remainder_policy": list(DATAFLOW_REMAINDER_POLICIES),
            "pipeline_completion": list(DATAFLOW_PIPELINE_COMPLETION_SEMANTICS),
            "pipeline_release": list(DATAFLOW_PIPELINE_RELEASE_SEMANTICS),
            "pipeline_synchronization_owner": list(DATAFLOW_PIPELINE_SYNCHRONIZATION_OWNERS),
            "pipeline_eviction_hint": list(DATAFLOW_PIPELINE_EVICTION_HINTS),
            "pipeline_materialization": list(DATAFLOW_PIPELINE_MATERIALIZATIONS),
            "transport_family": list(DATAFLOW_TRANSPORT_FAMILIES),
            "transport_access_order": list(DATAFLOW_TRANSPORT_ACCESS_ORDERS),
            "handoff_tail_policy": list(DATAFLOW_HANDOFF_TAIL_POLICIES),
            "layout_family": list(DATAFLOW_LAYOUT_FAMILIES),
        },
        "pipeline_component_fields": {
            "transfer": [item.name for item in fields(DataflowPipelineTransfer)],
            "gemm": [item.name for item in fields(DataflowPipelineGemm)],
            "buffer_lifetime": [item.name for item in fields(DataflowPipelineBufferLifetime)],
        },
        "reshared_component_fields": {
            "field_mapping": [item.name for item in fields(DataflowResharedFieldMapping)],
        },
        "candidate_fields": [item.name for item in fields(DataflowOperationCandidate)],
        "decision_fields": [item.name for item in fields(DataflowOperationDecision)],
        "resource_fields": [item.name for item in fields(DataflowOperationResourceEstimate)],
    }
    if include_fingerprint:
        descriptor["fingerprint"] = fingerprint(descriptor)
    return descriptor


DATAFLOW_OPERATION_CONTRACT_FINGERPRINT = dataflow_operation_contract_schema_dict()["fingerprint"]


__all__ = [
    "DATAFLOW_HANDOFF_NEUTRAL_TAIL",
    "DATAFLOW_HANDOFF_NOOP_TAIL",
    "DATAFLOW_HANDOFF_QUEUE_IMPLEMENTATION",
    "DATAFLOW_LAYOUT_CONTRACTS_ATTR",
    "DATAFLOW_LAYOUT_FAMILIES",
    "DATAFLOW_LAYOUT_INTERLEAVED",
    "DATAFLOW_LAYOUT_LINEAR",
    "DATAFLOW_LAYOUT_LINEAR_IMPLEMENTATION",
    "DATAFLOW_LAYOUT_MATRIX_SWIZZLE",
    "DATAFLOW_LAYOUT_MATRIX_SWIZZLE_IMPLEMENTATION",
    "DATAFLOW_OPERATION_CONTRACT_FINGERPRINT",
    "DATAFLOW_OPERATION_CONTRACT_KINDS",
    "DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION",
    "DATAFLOW_OPERATION_CROSS_HANDLER_HANDOFF",
    "DATAFLOW_OPERATION_DECISION_SCHEMA_VERSION",
    "DATAFLOW_HANDOFF_CONTRACT_ATTR",
    "DATAFLOW_OPERATION_PIPELINE",
    "DATAFLOW_OPERATION_RANGE_COARSENING",
    "DATAFLOW_OPERATION_RESHARED_TRANSPORT",
    "DATAFLOW_OPERATION_TENSOR_LAYOUT",
    "DATAFLOW_PIPELINE_PLAN_IMPLEMENTATION",
    "DATAFLOW_PIPELINE_CONTRACT_ATTR",
    "DATAFLOW_PIPELINE_CONSUMER_RELEASE",
    "DATAFLOW_PIPELINE_CONSUMER_VISIBLE",
    "DATAFLOW_PIPELINE_EVICTION_HINTS",
    "DATAFLOW_PIPELINE_EVICT_FIRST",
    "DATAFLOW_PIPELINE_EVICT_LAST",
    "DATAFLOW_PIPELINE_EVICT_NORMAL",
    "DATAFLOW_PIPELINE_MATERIALIZATIONS",
    "DATAFLOW_PIPELINE_MATERIALIZE_COPY",
    "DATAFLOW_PIPELINE_MATERIALIZE_RESIDENT",
    "DATAFLOW_PIPELINE_OWNED_COMPLETION",
    "DATAFLOW_PIPELINE_OWNED_RELEASE",
    "DATAFLOW_PIPELINE_PRODUCER_VISIBLE",
    "DATAFLOW_PIPELINE_RESIDENT_IMPLEMENTATION",
    "DATAFLOW_PIPELINE_SCOPE_EXIT_RELEASE",
    "DATAFLOW_PIPELINE_SYNCHRONIZATION_OWNERS",
    "DATAFLOW_PIPELINE_SYNC_CALLER",
    "DATAFLOW_PIPELINE_SYNC_PIPELINE",
    "DATAFLOW_PIPELINE_SYNC_TRANSFER",
    "DATAFLOW_RANGE_CONTRACT_ATTR",
    "DATAFLOW_RANGE_INDEPENDENT",
    "DATAFLOW_RANGE_ORDERED",
    "DATAFLOW_RANGE_REDUCTION",
    "DATAFLOW_RANGE_STAGE_GRAPH_IMPLEMENTATION",
    "DATAFLOW_REMAINDER_EXACT",
    "DATAFLOW_REMAINDER_PAD",
    "DATAFLOW_REMAINDER_PREDICATE",
    "DATAFLOW_TRANSPORT_ALL_GATHER",
    "DATAFLOW_TRANSPORT_ALL_GATHER_IMPLEMENTATION",
    "DATAFLOW_TRANSPORT_AUTO",
    "DATAFLOW_TRANSPORT_CONTRACT_ATTR",
    "DATAFLOW_TRANSPORT_HBM",
    "DATAFLOW_TRANSPORT_HBM_IMPLEMENTATION",
    "DATAFLOW_TRANSPORT_INDEPENDENT_ORDER",
    "DATAFLOW_TRANSPORT_LOGICAL_ORDER",
    "DATAFLOW_TRANSPORT_STREAMED",
    "DATAFLOW_TRANSPORT_STREAMED_IMPLEMENTATION",
    "DataflowCrossHandlerHandoffRequest",
    "DataflowOperationCandidate",
    "DataflowOperationDecision",
    "DataflowOperationRequest",
    "DataflowOperationResourceEstimate",
    "DataflowPipelineBufferLifetime",
    "DataflowPipelineRequest",
    "DataflowPipelineGemm",
    "DataflowPipelineTransfer",
    "DataflowRangeCoarseningRequest",
    "DataflowResharedFieldMapping",
    "DataflowResharedTransportRequest",
    "DataflowTensorLayoutRequest",
    "layout_implementation_id",
    "legacy_policy_for_transport_request",
    "dataflow_operation_contract_schema_dict",
    "dataflow_operation_request_from_dict",
    "normalize_layout_contracts",
    "normalize_operation_request",
    "selected_operation_decision",
    "transport_implementation_id",
]
