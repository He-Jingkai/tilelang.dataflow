"""Generic planning for cross-handler pipeline handoff."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any
from collections.abc import Mapping

from tilelang import _ffi_api
from tvm import ir, tir

from .operation_contracts import (
    DATAFLOW_HANDOFF_QUEUE_IMPLEMENTATION,
    DATAFLOW_PIPELINE_MATERIALIZE_COPY,
    DataflowCrossHandlerHandoffRequest,
    DataflowOperationCandidate,
    DataflowOperationDecision,
    DataflowOperationResourceEstimate,
    DataflowPipelineRequest,
)
from .pipeline_planning import DataflowPipelinePlan, plan_pipeline_dataflow


DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_SCHEMA_VERSION = 3
DATAFLOW_CROSS_HANDLER_HANDOFF_PLANNER_VERSION = "dataflow.handoff.pipeline.v3"
DATAFLOW_CROSS_HANDLER_HANDOFF_LOWERING_IMPLEMENTATION = "dataflow.handoff.pipeline_prefix.v1"
DATAFLOW_CROSS_HANDLER_HANDOFF_DISABLED_IMPLEMENTATION = "dataflow.handoff.resource_fallback.v1"
DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_FINGERPRINT_ATTR = "tl.cross_handler_handoff_plan_fingerprint"
DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_SCHEMA_ATTR = "tl.cross_handler_handoff_plan_schema_version"
DATAFLOW_CROSS_HANDLER_HANDOFF_ROLE_ATTR = "tl.cross_handler_handoff_role"
DATAFLOW_CROSS_HANDLER_HANDOFF_ENABLED_ATTR = "tl.cross_handler_handoff_enabled"
DATAFLOW_CROSS_HANDLER_HANDOFF_TRANSFER_INDEX_ATTR = "tl.cross_handler_handoff_transfer_index"
DATAFLOW_CROSS_HANDLER_HANDOFF_BUFFER_STAGES_ATTR = "tl.cross_handler_handoff_buffer_stages"
DATAFLOW_CROSS_HANDLER_HANDOFF_LOOKAHEAD_SLOTS_ATTR = "tl.cross_handler_handoff_lookahead_slots"
DATAFLOW_CROSS_HANDLER_HANDOFF_ARENA_OFFSET_ATTR = "tl.cross_handler_handoff_arena_offset"
DATAFLOW_CROSS_HANDLER_HANDOFF_ARENA_BYTES_ATTR = "tl.cross_handler_handoff_arena_bytes"
DATAFLOW_CROSS_HANDLER_HANDOFF_PRODUCER_TRANSFER_ATTR = "tl.cross_handler_handoff_producer_transfer"
DATAFLOW_CROSS_HANDLER_HANDOFF_PRODUCER_STAGE_ATTR = "tl.cross_handler_handoff_producer_stage"


class DataflowCrossHandlerHandoffPlanningError(ValueError):
    """Raised when a handoff request has no coherent graph-level plan."""


def fingerprint(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def nonnegative(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"handoff {name} must be a non-negative integer")


def positive(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"handoff {name} must be a positive integer")


def align_up(value: int, alignment: int) -> int:
    nonnegative(value, "byte extent")
    positive(alignment, "alignment")
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class DataflowHandoffTransferPlan:
    """One consumer pipeline transfer materialized by a future producer."""

    binding_index: int
    pipeline_transfer_index: int
    destination_buffer_index: int
    logical_extent: tuple[int, ...]
    bytes_per_stage: int
    buffer_stages: int
    physical_buffer_stages: int
    lookahead_slots: int
    arena_offset: int
    arena_bytes: int
    producer_partition: int | None
    async_permitted: bool
    multicast_permitted: bool
    eviction_hint: str

    def __post_init__(self) -> None:
        for name in (
            "binding_index",
            "pipeline_transfer_index",
            "destination_buffer_index",
            "arena_offset",
        ):
            nonnegative(getattr(self, name), name)
        for name in (
            "bytes_per_stage",
            "buffer_stages",
            "physical_buffer_stages",
            "lookahead_slots",
            "arena_bytes",
        ):
            positive(getattr(self, name), name)
        extent = tuple(self.logical_extent)
        if not extent or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in extent):
            raise ValueError("handoff transfer logical_extent must be concrete")
        object.__setattr__(self, "logical_extent", extent)
        if self.buffer_stages > self.physical_buffer_stages:
            raise ValueError("handoff prefix exceeds the physical consumer buffer")
        if self.arena_bytes != (self.bytes_per_stage * self.physical_buffer_stages * self.lookahead_slots):
            raise ValueError("handoff transfer arena does not cover every buffer version")
        if self.producer_partition is not None:
            nonnegative(self.producer_partition, "producer_partition")
        for name in ("async_permitted", "multicast_permitted"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"handoff transfer {name} must be a bool")
        if not isinstance(self.eviction_hint, str) or not self.eviction_hint:
            raise ValueError("handoff transfer requires an eviction hint")

    def to_dict(self) -> dict[str, Any]:
        return {
            "binding_index": self.binding_index,
            "pipeline_transfer_index": self.pipeline_transfer_index,
            "destination_buffer_index": self.destination_buffer_index,
            "logical_extent": list(self.logical_extent),
            "bytes_per_stage": self.bytes_per_stage,
            "buffer_stages": self.buffer_stages,
            "physical_buffer_stages": self.physical_buffer_stages,
            "lookahead_slots": self.lookahead_slots,
            "arena_offset": self.arena_offset,
            "arena_bytes": self.arena_bytes,
            "producer_partition": self.producer_partition,
            "async_permitted": self.async_permitted,
            "multicast_permitted": self.multicast_permitted,
            "eviction_hint": self.eviction_hint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowHandoffTransferPlan:
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ValueError("handoff transfer fields do not match the plan schema")
        return cls(
            binding_index=int(value["binding_index"]),
            pipeline_transfer_index=int(value["pipeline_transfer_index"]),
            destination_buffer_index=int(value["destination_buffer_index"]),
            logical_extent=tuple(int(item) for item in value["logical_extent"]),
            bytes_per_stage=int(value["bytes_per_stage"]),
            buffer_stages=int(value["buffer_stages"]),
            physical_buffer_stages=int(value["physical_buffer_stages"]),
            lookahead_slots=int(value["lookahead_slots"]),
            arena_offset=int(value["arena_offset"]),
            arena_bytes=int(value["arena_bytes"]),
            producer_partition=value["producer_partition"],
            async_permitted=bool(value["async_permitted"]),
            multicast_permitted=bool(value["multicast_permitted"]),
            eviction_hint=str(value["eviction_hint"]),
        )


@dataclass(frozen=True)
class DataflowHandoffBufferLifetime:
    """Compiler-owned live range for one transfer's staged values."""

    binding_index: int
    first_stage: int
    stage_count: int
    acquire_phase: str
    wait_phase: str
    release_phase: str

    def __post_init__(self) -> None:
        nonnegative(self.binding_index, "binding_index")
        nonnegative(self.first_stage, "first_stage")
        positive(self.stage_count, "stage_count")
        expected = (
            (self.acquire_phase, "producer_prefetch"),
            (self.wait_phase, "consumer_entry"),
            (self.release_phase, "consumer_release"),
        )
        if any(actual != required for actual, required in expected):
            raise ValueError("handoff lifetime phases do not match the physical protocol")

    def to_dict(self) -> dict[str, Any]:
        return {
            "binding_index": self.binding_index,
            "first_stage": self.first_stage,
            "stage_count": self.stage_count,
            "acquire_phase": self.acquire_phase,
            "wait_phase": self.wait_phase,
            "release_phase": self.release_phase,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowHandoffBufferLifetime:
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("handoff lifetime fields do not match the plan schema")
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class DataflowHandoffQueueBinding:
    """One concrete producer-to-future-consumer dependency in a queue."""

    binding_id: int
    plan_fingerprint: str
    sm_id: int
    producer_instruction_id: int
    consumer_instruction_id: int | None
    producer_task_id: int
    consumer_task_id: int | None
    consumer_task_coords: tuple[int, ...] | None
    consumer_range_begin: int | None
    consumer_range_end: int | None
    stage_count: int
    arena_slot: int | None
    state: str

    def __post_init__(self) -> None:
        for name in (
            "binding_id",
            "sm_id",
            "producer_instruction_id",
            "producer_task_id",
            "stage_count",
        ):
            nonnegative(getattr(self, name), name)
        if len(self.plan_fingerprint) != 64:
            raise ValueError("handoff queue binding requires a plan fingerprint")
        if self.state not in {"active", "tail", "disabled"}:
            raise ValueError(f"unsupported handoff queue state {self.state!r}")
        consumer_values = (
            self.consumer_instruction_id,
            self.consumer_task_id,
            self.consumer_task_coords,
            self.consumer_range_begin,
            self.consumer_range_end,
        )
        if self.state == "active":
            if any(item is None for item in consumer_values):
                raise ValueError("active handoff binding requires a consumer")
            assert self.consumer_task_coords is not None
            coords = tuple(self.consumer_task_coords)
            if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in coords):
                raise ValueError("handoff consumer task coordinates must be non-negative")
            object.__setattr__(self, "consumer_task_coords", coords)
            assert self.consumer_range_begin is not None
            assert self.consumer_range_end is not None
            if self.consumer_range_end < self.consumer_range_begin:
                raise ValueError("handoff consumer range cannot be reversed")
            positive(self.stage_count, "stage_count")
            if self.arena_slot is None:
                raise ValueError("active handoff binding requires an arena slot")
            nonnegative(self.arena_slot, "arena_slot")
        else:
            if any(item is not None for item in consumer_values):
                raise ValueError("inactive handoff binding cannot name a consumer")
            if self.stage_count != 0 or self.arena_slot is not None:
                raise ValueError("inactive handoff binding cannot own physical stages")

    def to_dict(self) -> dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "plan_fingerprint": self.plan_fingerprint,
            "sm_id": self.sm_id,
            "producer_instruction_id": self.producer_instruction_id,
            "consumer_instruction_id": self.consumer_instruction_id,
            "producer_task_id": self.producer_task_id,
            "consumer_task_id": self.consumer_task_id,
            "consumer_task_coords": (None if self.consumer_task_coords is None else list(self.consumer_task_coords)),
            "consumer_range_begin": self.consumer_range_begin,
            "consumer_range_end": self.consumer_range_end,
            "stage_count": self.stage_count,
            "arena_slot": self.arena_slot,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowHandoffQueueBinding:
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("handoff queue binding fields do not match the schema")
        return cls(
            **{
                **{name: value[name] for name in cls.__dataclass_fields__},
                "consumer_task_coords": (None if value["consumer_task_coords"] is None else tuple(value["consumer_task_coords"])),
            }
        )


@dataclass(frozen=True)
class DataflowCrossHandlerHandoffPlan:
    """Versioned graph-to-pipeline handoff plan."""

    request_fingerprint: str
    producer_stage_id: int
    consumer_stage_id: int
    task_coord_rank: int
    lookahead_distance: int
    requested_buffer_stages: int
    selected_buffer_stages: int
    value_bindings: tuple[int, ...]
    tail_policy: str
    transfer_plans: tuple[DataflowHandoffTransferPlan, ...]
    buffer_lifetimes: tuple[DataflowHandoffBufferLifetime, ...]
    arena_alignment: int
    resource_budget_bytes: int | None
    required_arena_bytes: int
    arena_bytes: int
    barrier_count: int
    wait_required: bool
    release_required: bool
    enabled: bool
    fallback_reason: str | None
    implementation_id: str
    lowering_implementation_id: str
    planner_version: str
    pipeline_plan_fingerprint: str | None
    decision: DataflowOperationDecision
    schema_version: int = DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported cross-handler handoff plan schema")
        if self.planner_version != DATAFLOW_CROSS_HANDLER_HANDOFF_PLANNER_VERSION:
            raise ValueError("unsupported cross-handler handoff planner")
        for name in (
            "producer_stage_id",
            "consumer_stage_id",
            "task_coord_rank",
            "selected_buffer_stages",
            "required_arena_bytes",
            "arena_bytes",
            "barrier_count",
        ):
            nonnegative(getattr(self, name), name)
        if self.resource_budget_bytes is not None:
            nonnegative(self.resource_budget_bytes, "resource_budget_bytes")
        positive(self.lookahead_distance, "lookahead_distance")
        positive(self.requested_buffer_stages, "requested_buffer_stages")
        positive(self.arena_alignment, "arena_alignment")
        if self.producer_stage_id == self.consumer_stage_id:
            raise ValueError("handoff producer and consumer stages must differ")
        bindings = tuple(self.value_bindings)
        if bindings != tuple(sorted(set(bindings))):
            raise ValueError("handoff value bindings must be sorted and unique")
        object.__setattr__(self, "value_bindings", bindings)
        transfers = tuple(self.transfer_plans)
        lifetimes = tuple(self.buffer_lifetimes)
        object.__setattr__(self, "transfer_plans", transfers)
        object.__setattr__(self, "buffer_lifetimes", lifetimes)
        if tuple(item.binding_index for item in transfers) != tuple(range(len(transfers))):
            raise ValueError("handoff transfer binding indices must be contiguous")
        if tuple(item.pipeline_transfer_index for item in transfers) != bindings:
            raise ValueError("handoff transfer plans differ from value bindings")
        if tuple(item.binding_index for item in lifetimes) != tuple(range(len(transfers))):
            raise ValueError("handoff lifetimes must cover every transfer")
        expected_end = 0
        for transfer in transfers:
            if transfer.arena_offset < expected_end:
                raise ValueError("handoff arena ranges overlap")
            expected_end = transfer.arena_offset + transfer.arena_bytes
        if transfers and self.arena_bytes != align_up(expected_end, self.arena_alignment):
            raise ValueError("handoff arena byte extent is inconsistent")
        if not transfers and self.arena_bytes:
            raise ValueError("disabled handoff cannot allocate an arena")
        for name in ("wait_required", "release_required", "enabled"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"handoff {name} must be a bool")
        if self.enabled:
            if self.selected_buffer_stages <= 0 or not transfers:
                raise ValueError("enabled handoff requires staged transfers")
            if self.required_arena_bytes != self.arena_bytes:
                raise ValueError("enabled handoff must allocate its required arena")
            if self.resource_budget_bytes is not None and self.required_arena_bytes > self.resource_budget_bytes:
                raise ValueError("enabled handoff exceeds its recorded resource budget")
            if self.fallback_reason is not None:
                raise ValueError("enabled handoff cannot carry a fallback reason")
            if not self.wait_required or not self.release_required:
                raise ValueError("enabled handoff must own wait and release")
            if self.lowering_implementation_id != DATAFLOW_CROSS_HANDLER_HANDOFF_LOWERING_IMPLEMENTATION:
                raise ValueError("enabled handoff selected a stale lowering")
            if self.pipeline_plan_fingerprint is None:
                raise ValueError("enabled handoff requires its consumer pipeline plan")
        else:
            if self.selected_buffer_stages or transfers or lifetimes:
                raise ValueError("disabled handoff cannot retain physical transfers")
            if not self.fallback_reason:
                raise ValueError("disabled handoff requires a fallback reason")
            if self.wait_required or self.release_required or self.barrier_count:
                raise ValueError("disabled handoff cannot own synchronization")
            if self.lowering_implementation_id != DATAFLOW_CROSS_HANDLER_HANDOFF_DISABLED_IMPLEMENTATION:
                raise ValueError("disabled handoff selected a stale fallback")
            if self.fallback_reason == "handoff_arena_exceeds_shared_memory_budget":
                if self.resource_budget_bytes is None:
                    raise ValueError("resource-fallback handoff requires a recorded budget")
                if self.required_arena_bytes <= self.resource_budget_bytes:
                    raise ValueError("resource-fallback handoff must exceed its budget")
        if self.decision.request.fingerprint != self.request_fingerprint:
            raise ValueError("handoff decision changed its typed request")
        if self.decision.selected_implementation != self.implementation_id:
            raise ValueError("handoff decision and plan implementation differ")

    @property
    def resources(self) -> DataflowOperationResourceEstimate:
        resources = self.decision.resources
        assert resources is not None
        return resources

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_fingerprint": self.request_fingerprint,
            "producer_stage_id": self.producer_stage_id,
            "consumer_stage_id": self.consumer_stage_id,
            "task_coord_rank": self.task_coord_rank,
            "lookahead_distance": self.lookahead_distance,
            "requested_buffer_stages": self.requested_buffer_stages,
            "selected_buffer_stages": self.selected_buffer_stages,
            "value_bindings": list(self.value_bindings),
            "tail_policy": self.tail_policy,
            "transfer_plans": [item.to_dict() for item in self.transfer_plans],
            "buffer_lifetimes": [item.to_dict() for item in self.buffer_lifetimes],
            "arena_alignment": self.arena_alignment,
            "resource_budget_bytes": self.resource_budget_bytes,
            "required_arena_bytes": self.required_arena_bytes,
            "arena_bytes": self.arena_bytes,
            "barrier_count": self.barrier_count,
            "wait_required": self.wait_required,
            "release_required": self.release_required,
            "enabled": self.enabled,
            "fallback_reason": self.fallback_reason,
            "implementation_id": self.implementation_id,
            "lowering_implementation_id": self.lowering_implementation_id,
            "planner_version": self.planner_version,
            "pipeline_plan_fingerprint": self.pipeline_plan_fingerprint,
            "decision": self.decision.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        result = self.canonical_payload()
        result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowCrossHandlerHandoffPlan:
        allowed = set(cls.__dataclass_fields__) | {"fingerprint"}
        if set(value).difference(allowed):
            raise ValueError("cross-handler handoff plan has unknown fields")
        plan = cls(
            request_fingerprint=str(value["request_fingerprint"]),
            producer_stage_id=int(value["producer_stage_id"]),
            consumer_stage_id=int(value["consumer_stage_id"]),
            task_coord_rank=int(value["task_coord_rank"]),
            lookahead_distance=int(value["lookahead_distance"]),
            requested_buffer_stages=int(value["requested_buffer_stages"]),
            selected_buffer_stages=int(value["selected_buffer_stages"]),
            value_bindings=tuple(int(item) for item in value["value_bindings"]),
            tail_policy=str(value["tail_policy"]),
            transfer_plans=tuple(DataflowHandoffTransferPlan.from_dict(item) for item in value["transfer_plans"]),
            buffer_lifetimes=tuple(DataflowHandoffBufferLifetime.from_dict(item) for item in value["buffer_lifetimes"]),
            arena_alignment=int(value["arena_alignment"]),
            resource_budget_bytes=(None if value["resource_budget_bytes"] is None else int(value["resource_budget_bytes"])),
            required_arena_bytes=int(value["required_arena_bytes"]),
            arena_bytes=int(value["arena_bytes"]),
            barrier_count=int(value["barrier_count"]),
            wait_required=bool(value["wait_required"]),
            release_required=bool(value["release_required"]),
            enabled=bool(value["enabled"]),
            fallback_reason=value["fallback_reason"],
            implementation_id=str(value["implementation_id"]),
            lowering_implementation_id=str(value["lowering_implementation_id"]),
            planner_version=str(value["planner_version"]),
            pipeline_plan_fingerprint=value["pipeline_plan_fingerprint"],
            decision=DataflowOperationDecision.from_dict(value["decision"]),
            schema_version=int(value.get("schema_version", 0)),
        )
        if value.get("fingerprint") not in (None, plan.fingerprint):
            raise ValueError("cross-handler handoff plan fingerprint mismatch")
        return plan


def plan_cross_handler_handoff(
    request: DataflowCrossHandlerHandoffRequest,
    *,
    producer_stage_id: int,
    consumer_pipeline_request: DataflowPipelineRequest | None,
    task_coord_rank: int,
    max_shared_memory_bytes: int | None = None,
    arena_alignment: int = 16,
) -> DataflowCrossHandlerHandoffPlan:
    """Derive physical prefix transfers solely from typed graph contracts."""

    if not isinstance(request, DataflowCrossHandlerHandoffRequest):
        raise TypeError("handoff planning requires a typed request")
    nonnegative(producer_stage_id, "producer_stage_id")
    nonnegative(task_coord_rank, "task_coord_rank")
    positive(arena_alignment, "arena_alignment")
    if max_shared_memory_bytes is not None:
        nonnegative(max_shared_memory_bytes, "max_shared_memory_bytes")

    pipeline_plan: DataflowPipelinePlan | None = None
    fallback_reason = None
    bindings: tuple[int, ...] = ()
    if consumer_pipeline_request is None:
        fallback_reason = "consumer_stage_has_no_pipeline_contract"
    else:
        pipeline_plan = plan_pipeline_dataflow(consumer_pipeline_request)
        copy_indices = tuple(
            index
            for index, transfer in enumerate(consumer_pipeline_request.transfers)
            if transfer.materialization == DATAFLOW_PIPELINE_MATERIALIZE_COPY
        )
        bindings = tuple(request.value_bindings) if request.value_bindings else copy_indices
        invalid = tuple(index for index in bindings if index >= len(consumer_pipeline_request.transfers))
        if invalid:
            raise DataflowCrossHandlerHandoffPlanningError(f"handoff value_bindings reference missing consumer transfers: {invalid!r}")
        resident = tuple(
            index for index in bindings if consumer_pipeline_request.transfers[index].materialization != DATAFLOW_PIPELINE_MATERIALIZE_COPY
        )
        if resident:
            raise DataflowCrossHandlerHandoffPlanningError(
                f"handoff value_bindings must reference copy-materialized transfers: {resident!r}"
            )
        if not bindings:
            fallback_reason = "consumer_pipeline_has_no_copy_transfers"
        elif pipeline_plan.used_fallback:
            fallback_reason = "consumer_pipeline_selected_synchronous_fallback"
        elif any(
            request.buffer_stages
            > pipeline_plan.implementation_requirements.buffer_versions[consumer_pipeline_request.transfers[index].destination_buffer_index]
            for index in bindings
        ):
            fallback_reason = "handoff_prefix_exceeds_consumer_buffer_versions"

    transfers: list[DataflowHandoffTransferPlan] = []
    offset = 0
    if fallback_reason is None:
        assert consumer_pipeline_request is not None
        assert pipeline_plan is not None
        for binding_index, transfer_index in enumerate(bindings):
            transfer = consumer_pipeline_request.transfers[transfer_index]
            physical_buffer_stages = pipeline_plan.implementation_requirements.buffer_versions[transfer.destination_buffer_index]
            offset = align_up(offset, arena_alignment)
            arena_bytes = transfer.bytes_per_stage * physical_buffer_stages
            transfers.append(
                DataflowHandoffTransferPlan(
                    binding_index=binding_index,
                    pipeline_transfer_index=transfer_index,
                    destination_buffer_index=transfer.destination_buffer_index,
                    logical_extent=transfer.logical_extent,
                    bytes_per_stage=transfer.bytes_per_stage,
                    buffer_stages=request.buffer_stages,
                    physical_buffer_stages=physical_buffer_stages,
                    lookahead_slots=request.lookahead_distance,
                    arena_offset=offset,
                    arena_bytes=arena_bytes * request.lookahead_distance,
                    producer_partition=transfer.producer_partition,
                    async_permitted=transfer.async_permitted,
                    multicast_permitted=transfer.multicast_permitted,
                    eviction_hint=transfer.eviction_hint,
                )
            )
            offset += arena_bytes * request.lookahead_distance
    required_arena_bytes = align_up(offset, arena_alignment) if transfers else 0
    arena_bytes = required_arena_bytes
    if fallback_reason is None and max_shared_memory_bytes is not None and arena_bytes > max_shared_memory_bytes:
        fallback_reason = "handoff_arena_exceeds_shared_memory_budget"
        transfers = []
        arena_bytes = 0

    enabled = fallback_reason is None
    selected_stages = request.buffer_stages if enabled else 0
    lifetimes = tuple(
        DataflowHandoffBufferLifetime(
            binding_index=item.binding_index,
            first_stage=0,
            stage_count=selected_stages,
            acquire_phase="producer_prefetch",
            wait_phase="consumer_entry",
            release_phase="consumer_release",
        )
        for item in transfers
    )
    resources = DataflowOperationResourceEstimate(
        shared_memory_bytes=arena_bytes,
        barrier_count=len(transfers) * selected_stages,
        temporary_bytes=arena_bytes,
        transaction_bytes=(None if not transfers else max(item.bytes_per_stage for item in transfers)),
    )
    rejected: tuple[DataflowOperationCandidate, ...] = ()
    lowering_id = DATAFLOW_CROSS_HANDLER_HANDOFF_LOWERING_IMPLEMENTATION
    if not enabled:
        lowering_id = DATAFLOW_CROSS_HANDLER_HANDOFF_DISABLED_IMPLEMENTATION
        rejected = (
            DataflowOperationCandidate(
                implementation_id=DATAFLOW_CROSS_HANDLER_HANDOFF_LOWERING_IMPLEMENTATION,
                legal=False,
                resources=DataflowOperationResourceEstimate(
                    shared_memory_bytes=required_arena_bytes,
                    temporary_bytes=required_arena_bytes,
                ),
                rejection_reasons=(str(fallback_reason),),
            ),
        )
    decision = DataflowOperationDecision(
        request=request,
        candidates=(
            *rejected,
            DataflowOperationCandidate(
                implementation_id=DATAFLOW_HANDOFF_QUEUE_IMPLEMENTATION,
                legal=True,
                resources=resources,
            ),
        ),
        selected_implementation=DATAFLOW_HANDOFF_QUEUE_IMPLEMENTATION,
        used_fallback=not enabled,
        selection_reason=("typed_pipeline_prefix_handoff" if enabled else str(fallback_reason)),
    )
    return DataflowCrossHandlerHandoffPlan(
        request_fingerprint=request.fingerprint,
        producer_stage_id=producer_stage_id,
        consumer_stage_id=request.consumer_stage_id,
        task_coord_rank=task_coord_rank,
        lookahead_distance=request.lookahead_distance,
        requested_buffer_stages=request.buffer_stages,
        selected_buffer_stages=selected_stages,
        value_bindings=bindings if enabled else (),
        tail_policy=request.tail_policy,
        transfer_plans=tuple(transfers),
        buffer_lifetimes=lifetimes,
        arena_alignment=arena_alignment,
        resource_budget_bytes=max_shared_memory_bytes,
        required_arena_bytes=required_arena_bytes,
        arena_bytes=arena_bytes,
        barrier_count=len(transfers) * selected_stages,
        wait_required=enabled,
        release_required=enabled,
        enabled=enabled,
        fallback_reason=fallback_reason,
        implementation_id=DATAFLOW_HANDOFF_QUEUE_IMPLEMENTATION,
        lowering_implementation_id=lowering_id,
        planner_version=DATAFLOW_CROSS_HANDLER_HANDOFF_PLANNER_VERSION,
        pipeline_plan_fingerprint=(None if pipeline_plan is None else pipeline_plan.fingerprint),
        decision=decision,
    )


def fallback_cross_handler_handoff_plan(
    plan: DataflowCrossHandlerHandoffPlan,
    *,
    reason: str,
) -> DataflowCrossHandlerHandoffPlan:
    """Disable an otherwise legal plan after physical queue materialization."""

    if not isinstance(plan, DataflowCrossHandlerHandoffPlan):
        raise TypeError("handoff fallback requires a typed plan")
    if not plan.enabled:
        raise ValueError("handoff post-planning fallback requires an enabled plan")
    if not isinstance(reason, str) or not reason:
        raise ValueError("handoff post-planning fallback requires a reason")
    rejected_resources = plan.resources
    fallback_resources = DataflowOperationResourceEstimate(
        shared_memory_bytes=0,
        barrier_count=0,
        temporary_bytes=0,
    )
    decision = DataflowOperationDecision(
        request=plan.decision.request,
        candidates=(
            DataflowOperationCandidate(
                implementation_id=(DATAFLOW_CROSS_HANDLER_HANDOFF_LOWERING_IMPLEMENTATION),
                legal=False,
                resources=rejected_resources,
                rejection_reasons=(reason,),
            ),
            DataflowOperationCandidate(
                implementation_id=DATAFLOW_HANDOFF_QUEUE_IMPLEMENTATION,
                legal=True,
                resources=fallback_resources,
            ),
        ),
        selected_implementation=DATAFLOW_HANDOFF_QUEUE_IMPLEMENTATION,
        used_fallback=True,
        selection_reason=reason,
    )
    return replace(
        plan,
        selected_buffer_stages=0,
        value_bindings=(),
        transfer_plans=(),
        buffer_lifetimes=(),
        arena_bytes=0,
        barrier_count=0,
        wait_required=False,
        release_required=False,
        enabled=False,
        fallback_reason=reason,
        lowering_implementation_id=(DATAFLOW_CROSS_HANDLER_HANDOFF_DISABLED_IMPLEMENTATION),
        decision=decision,
    )


def bind_cross_handler_handoff_plan(
    prim_func: tir.PrimFunc,
    plan: DataflowCrossHandlerHandoffPlan,
    *,
    role: str,
    pipeline_request: DataflowPipelineRequest | None = None,
) -> tir.PrimFunc:
    """Bind a graph handoff plan to producer or consumer PrimFunc IR."""

    if not isinstance(prim_func, tir.PrimFunc):
        raise TypeError("handoff physical binding requires a tir.PrimFunc")
    if not isinstance(plan, DataflowCrossHandlerHandoffPlan):
        raise TypeError("handoff physical binding requires a typed plan")
    if role not in {"producer", "consumer"}:
        raise ValueError("handoff PrimFunc role must be producer or consumer")
    if role == "producer" or not plan.enabled:
        return (
            prim_func.with_attr(
                DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_SCHEMA_ATTR,
                plan.schema_version,
            )
            .with_attr(
                DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_FINGERPRINT_ATTR,
                plan.fingerprint,
            )
            .with_attr(DATAFLOW_CROSS_HANDLER_HANDOFF_ROLE_ATTR, role)
            .with_attr(
                DATAFLOW_CROSS_HANDLER_HANDOFF_ENABLED_ATTR,
                int(plan.enabled),
            )
        )
    if not isinstance(pipeline_request, DataflowPipelineRequest):
        raise TypeError("enabled handoff consumer requires its pipeline request")

    loops: list[tir.For] = []

    def collect_loop(node: Any) -> None:
        if isinstance(node, tir.For) and "tl.pipeline_dataflow_plan_fingerprint" in node.annotations:
            loops.append(node)

    # The plan fingerprint is a PrimFunc attr, while the selected loop carries
    # the dataflow mode. Accept the sole typed pipeline loop after binding.
    tir.stmt_functor.post_order_visit(prim_func.body, collect_loop)
    if not loops:

        def collect_typed_loop(node: Any) -> None:
            if isinstance(node, tir.For) and "tl.pipeline_dataflow_mode" in node.annotations:
                loops.append(node)

        tir.stmt_functor.post_order_visit(prim_func.body, collect_typed_loop)
    if len(loops) != 1:
        raise DataflowCrossHandlerHandoffPlanningError("enabled handoff consumer requires exactly one typed pipeline loop")
    selected_loop = loops[0]
    copy_calls: list[tir.Call] = []

    def collect_copy(node: Any) -> None:
        if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) and node.op.name == "tl.tileop.copy":
            copy_calls.append(node)

    tir.stmt_functor.post_order_visit(selected_loop.body, collect_copy)
    groups: list[list[tir.Call]] = []
    destinations = []
    for call in copy_calls:
        parsed = _ffi_api.ParseOperator(call)
        group_index = next(
            (index for index, destination in enumerate(destinations) if parsed.dst.data.same_as(destination.data)),
            None,
        )
        if group_index is None:
            destinations.append(parsed.dst)
            groups.append([call])
        else:
            groups[group_index].append(call)
    if len(groups) != len(pipeline_request.transfers):
        raise DataflowCrossHandlerHandoffPlanningError("handoff consumer transfer graph differs from its pipeline request")
    transfer_by_pipeline_index = {item.pipeline_transfer_index: item for item in plan.transfer_plans}

    def annotate_transfer(node: Any) -> Any:
        if not isinstance(node, tir.Call):
            return node
        pipeline_index = next(
            (index for index, group in enumerate(groups) if any(node.same_as(call) for call in group)),
            None,
        )
        transfer = transfer_by_pipeline_index.get(pipeline_index)
        if transfer is None:
            return node
        compiler_owned = (
            DATAFLOW_CROSS_HANDLER_HANDOFF_TRANSFER_INDEX_ATTR,
            DATAFLOW_CROSS_HANDLER_HANDOFF_BUFFER_STAGES_ATTR,
            DATAFLOW_CROSS_HANDLER_HANDOFF_LOOKAHEAD_SLOTS_ATTR,
            DATAFLOW_CROSS_HANDLER_HANDOFF_ARENA_OFFSET_ATTR,
            DATAFLOW_CROSS_HANDLER_HANDOFF_ARENA_BYTES_ATTR,
            DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_FINGERPRINT_ATTR,
        )
        if any(name in node.annotations for name in compiler_owned):
            raise DataflowCrossHandlerHandoffPlanningError("handoff transfer annotations are compiler-owned")
        annotations = dict(node.annotations)
        annotations.update(
            {
                DATAFLOW_CROSS_HANDLER_HANDOFF_TRANSFER_INDEX_ATTR: tir.IntImm("int32", transfer.binding_index),
                DATAFLOW_CROSS_HANDLER_HANDOFF_BUFFER_STAGES_ATTR: tir.IntImm("int32", transfer.buffer_stages),
                DATAFLOW_CROSS_HANDLER_HANDOFF_LOOKAHEAD_SLOTS_ATTR: tir.IntImm("int32", transfer.lookahead_slots),
                DATAFLOW_CROSS_HANDLER_HANDOFF_ARENA_OFFSET_ATTR: tir.IntImm("int64", transfer.arena_offset),
                DATAFLOW_CROSS_HANDLER_HANDOFF_ARENA_BYTES_ATTR: tir.IntImm("int64", transfer.arena_bytes),
                DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_FINGERPRINT_ATTR: tir.StringImm(plan.fingerprint),
            }
        )
        return tir.Call(
            node.dtype,
            node.op,
            list(node.args),
            annotations=annotations,
            span=node.span,
        )

    def rewrite_loop(node: Any) -> Any:
        if not isinstance(node, tir.For) or not node.same_as(selected_loop):
            return node
        annotations = dict(node.annotations)
        annotations[DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_FINGERPRINT_ATTR] = tir.StringImm(plan.fingerprint)
        annotations[DATAFLOW_CROSS_HANDLER_HANDOFF_BUFFER_STAGES_ATTR] = tir.IntImm("int32", plan.selected_buffer_stages)
        return tir.For(
            node.loop_var,
            node.min,
            node.extent,
            node.kind,
            tir.stmt_functor.ir_transform(
                node.body,
                None,
                annotate_transfer,
                ["tir.Call"],
            ),
            node.thread_binding,
            annotations,
            node.step,
            getattr(node, "span", None),
        )

    bound = prim_func.with_body(
        tir.stmt_functor.ir_transform(
            prim_func.body,
            None,
            rewrite_loop,
            ["tir.For"],
        )
    )
    return (
        bound.with_attr(
            DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_SCHEMA_ATTR,
            plan.schema_version,
        )
        .with_attr(
            DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_FINGERPRINT_ATTR,
            plan.fingerprint,
        )
        .with_attr(DATAFLOW_CROSS_HANDLER_HANDOFF_ROLE_ATTR, role)
        .with_attr(DATAFLOW_CROSS_HANDLER_HANDOFF_ENABLED_ATTR, 1)
    )


__all__ = [
    "DATAFLOW_CROSS_HANDLER_HANDOFF_DISABLED_IMPLEMENTATION",
    "DATAFLOW_CROSS_HANDLER_HANDOFF_ARENA_BYTES_ATTR",
    "DATAFLOW_CROSS_HANDLER_HANDOFF_ARENA_OFFSET_ATTR",
    "DATAFLOW_CROSS_HANDLER_HANDOFF_BUFFER_STAGES_ATTR",
    "DATAFLOW_CROSS_HANDLER_HANDOFF_ENABLED_ATTR",
    "DATAFLOW_CROSS_HANDLER_HANDOFF_LOWERING_IMPLEMENTATION",
    "DATAFLOW_CROSS_HANDLER_HANDOFF_LOOKAHEAD_SLOTS_ATTR",
    "DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_FINGERPRINT_ATTR",
    "DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_SCHEMA_VERSION",
    "DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_SCHEMA_ATTR",
    "DATAFLOW_CROSS_HANDLER_HANDOFF_PLANNER_VERSION",
    "DATAFLOW_CROSS_HANDLER_HANDOFF_PRODUCER_STAGE_ATTR",
    "DATAFLOW_CROSS_HANDLER_HANDOFF_PRODUCER_TRANSFER_ATTR",
    "DATAFLOW_CROSS_HANDLER_HANDOFF_ROLE_ATTR",
    "DATAFLOW_CROSS_HANDLER_HANDOFF_TRANSFER_INDEX_ATTR",
    "DataflowCrossHandlerHandoffPlan",
    "DataflowCrossHandlerHandoffPlanningError",
    "DataflowHandoffBufferLifetime",
    "DataflowHandoffQueueBinding",
    "DataflowHandoffTransferPlan",
    "bind_cross_handler_handoff_plan",
    "fallback_cross_handler_handoff_plan",
    "plan_cross_handler_handoff",
]
