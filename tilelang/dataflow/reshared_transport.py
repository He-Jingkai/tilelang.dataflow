"""Generic planning for logical reshared values and physical transport slots."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any
from collections.abc import Mapping

from tilelang import _ffi_api, tvm
from tilelang.utils.target_capabilities import TargetCapabilitySnapshot
from tvm import ir, tir

from .operation_contracts import (
    DATAFLOW_TRANSPORT_ALL_GATHER,
    DATAFLOW_TRANSPORT_ALL_GATHER_IMPLEMENTATION,
    DATAFLOW_TRANSPORT_AUTO,
    DATAFLOW_TRANSPORT_HBM,
    DATAFLOW_TRANSPORT_HBM_IMPLEMENTATION,
    DATAFLOW_TRANSPORT_INDEPENDENT_ORDER,
    DATAFLOW_TRANSPORT_STREAMED,
    DATAFLOW_TRANSPORT_STREAMED_IMPLEMENTATION,
    DataflowOperationCandidate,
    DataflowOperationDecision,
    DataflowOperationResourceEstimate,
    DataflowResharedTransportRequest,
)


DATAFLOW_RESHARED_TRANSPORT_PLAN_SCHEMA_VERSION = 2
DATAFLOW_RESHARED_TRANSPORT_PLANNER_VERSION = "dataflow.reshared_transport.v2"
DATAFLOW_RESHARED_HBM_LOWERING_IMPLEMENTATION = "dataflow.transport.hbm.scheduler_comms.v1"
DATAFLOW_RESHARED_ALL_GATHER_LOWERING_IMPLEMENTATION = "dataflow.transport.cluster_all_gather.scheduler_comms.v1"
DATAFLOW_RESHARED_STREAMED_PULL_IMPLEMENTATION = "dataflow.transport.cluster_streamed.pull.v1"
DATAFLOW_RESHARED_STREAMED_PUSH_IMPLEMENTATION = "dataflow.transport.cluster_streamed.push.v2"
DATAFLOW_RESHARED_TRANSPORT_FAMILY_ATTR = "tl.reshared_transport_family"
DATAFLOW_RESHARED_TRANSPORT_PLAN_FINGERPRINT_ATTR = "tl.reshared_transport_plan_fingerprint"
DATAFLOW_RESHARED_TRANSPORT_PLAN_SCHEMA_ATTR = "tl.reshared_transport_plan_schema_version"
DATAFLOW_RESHARED_SOURCE_RANK_ATTR = "tl.reshared_source_rank"
DATAFLOW_RESHARED_PAYLOAD_PARTITION_BYTES_ATTR = "tl.reshared_payload_partition_bytes"
DATAFLOW_RESHARED_TRANSFER_THREADS_ATTR = "tl.reshared_transfer_threads"
DATAFLOW_RESHARED_CREDIT_TARGET_RANK_ATTR = "tl.reshared_credit_target_rank"
DATAFLOW_RESHARED_RECEIVE_STAGES_ATTR = "tl.reshared_receive_stages"


@dataclass(frozen=True)
class DataflowResharedTransportCapability:
    """ISA-level limits consumed by transport planning."""

    cluster_transport_supported: bool
    max_transaction_bytes: int | None
    transfer_threads: int | None
    default_receive_stages: int
    registry_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.cluster_transport_supported, bool):
            raise TypeError("cluster_transport_supported must be a bool")
        if self.cluster_transport_supported:
            if self.max_transaction_bytes is None or self.max_transaction_bytes <= 0:
                raise ValueError("cluster transport capability requires a positive transaction limit")
            if self.transfer_threads is None or self.transfer_threads <= 0:
                raise ValueError("cluster transport capability requires a positive transfer thread count")
        elif self.max_transaction_bytes is not None or self.transfer_threads is not None:
            raise ValueError("non-cluster capability cannot expose cluster transport resources")
        if self.default_receive_stages <= 0:
            raise ValueError("default_receive_stages must be positive")
        if not self.registry_key:
            raise ValueError("transport capability requires a registry_key")

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_key": self.registry_key,
            "cluster_transport_supported": self.cluster_transport_supported,
            "max_transaction_bytes": self.max_transaction_bytes,
            "transfer_threads": self.transfer_threads,
            "default_receive_stages": self.default_receive_stages,
        }


DATAFLOW_RESHARED_TRANSPORT_CAPABILITY_REGISTRY = (
    DataflowResharedTransportCapability(
        cluster_transport_supported=True,
        max_transaction_bytes=8 * 1024,
        transfer_threads=128,
        default_receive_stages=2,
        registry_key="cuda.cluster_transport.sm90.v1",
    ),
)


def resolve_reshared_transport_capability(
    target: TargetCapabilitySnapshot | None,
) -> DataflowResharedTransportCapability:
    """Resolve transport limits from target feature bits, never device identity."""

    cluster_supported = bool(target is not None and target.supports_cluster_launch and target.compute_capability >= (9, 0))
    if not cluster_supported:
        return DataflowResharedTransportCapability(
            cluster_transport_supported=False,
            max_transaction_bytes=None,
            transfer_threads=None,
            default_receive_stages=1,
            registry_key="cuda.cluster_transport.unavailable",
        )
    return DATAFLOW_RESHARED_TRANSPORT_CAPABILITY_REGISTRY[0]


@dataclass(frozen=True)
class DataflowResharedPayloadPartition:
    partition_index: int
    byte_offset: int
    byte_count: int

    def __post_init__(self) -> None:
        if self.partition_index < 0 or self.byte_offset < 0 or self.byte_count <= 0:
            raise ValueError("reshared payload partitions require non-negative offsets")

    def to_dict(self) -> dict[str, int]:
        return {
            "partition_index": self.partition_index,
            "byte_offset": self.byte_offset,
            "byte_count": self.byte_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowResharedPayloadPartition:
        return cls(
            partition_index=int(value["partition_index"]),
            byte_offset=int(value["byte_offset"]),
            byte_count=int(value["byte_count"]),
        )


@dataclass(frozen=True)
class DataflowResharedTransportStep:
    consumer_rank: int
    access_step: int
    logical_tile_index: int
    physical_slot_index: int
    source_rank: int
    source_local_slot_index: int
    source_tile_index: int
    locality: str
    transport_required: bool
    receive_stage: int | None
    credit_acquire_step: int | None
    credit_release_step: int | None

    def __post_init__(self) -> None:
        for name in (
            "consumer_rank",
            "access_step",
            "logical_tile_index",
            "physical_slot_index",
            "source_rank",
            "source_local_slot_index",
            "source_tile_index",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"reshared transport {name} must be non-negative")
        if self.locality not in {"local", "remote", "global"}:
            raise ValueError(f"unsupported reshared locality {self.locality!r}")
        if not isinstance(self.transport_required, bool):
            raise TypeError("transport_required must be a bool")
        lifetime = (
            self.receive_stage,
            self.credit_acquire_step,
            self.credit_release_step,
        )
        if any(item is None for item in lifetime) != all(item is None for item in lifetime):
            raise ValueError("receive stage and credit lifetime must be present together")
        if self.receive_stage is not None:
            if self.receive_stage < 0 or self.credit_acquire_step < 0:  # type: ignore[operator]
                raise ValueError("receive stage credit values must be non-negative")
            if self.credit_release_step < self.credit_acquire_step:  # type: ignore[operator]
                raise ValueError("credit release cannot precede acquire")

    def to_dict(self) -> dict[str, Any]:
        return {
            "consumer_rank": self.consumer_rank,
            "access_step": self.access_step,
            "logical_tile_index": self.logical_tile_index,
            "physical_slot_index": self.physical_slot_index,
            "source_rank": self.source_rank,
            "source_local_slot_index": self.source_local_slot_index,
            "source_tile_index": self.source_tile_index,
            "locality": self.locality,
            "transport_required": self.transport_required,
            "receive_stage": self.receive_stage,
            "credit_acquire_step": self.credit_acquire_step,
            "credit_release_step": self.credit_release_step,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowResharedTransportStep:
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class DataflowResharedCreditLifetime:
    consumer_rank: int
    receive_stage: int
    acquire_steps: tuple[int, ...]
    release_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.consumer_rank < 0 or self.receive_stage < 0:
            raise ValueError("credit lifetime rank and stage must be non-negative")
        object.__setattr__(self, "acquire_steps", tuple(self.acquire_steps))
        object.__setattr__(self, "release_steps", tuple(self.release_steps))
        if not self.acquire_steps or len(self.acquire_steps) != len(self.release_steps):
            raise ValueError("credit lifetime requires paired acquire/release steps")
        if any(acquire < 0 or release < acquire for acquire, release in zip(self.acquire_steps, self.release_steps)):
            raise ValueError("credit lifetime contains an invalid acquire/release pair")

    def to_dict(self) -> dict[str, Any]:
        return {
            "consumer_rank": self.consumer_rank,
            "receive_stage": self.receive_stage,
            "acquire_steps": list(self.acquire_steps),
            "release_steps": list(self.release_steps),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowResharedCreditLifetime:
        return cls(
            consumer_rank=int(value["consumer_rank"]),
            receive_stage=int(value["receive_stage"]),
            acquire_steps=tuple(int(item) for item in value["acquire_steps"]),
            release_steps=tuple(int(item) for item in value["release_steps"]),
        )


@dataclass(frozen=True)
class DataflowResharedTransportPlan:
    request_fingerprint: str
    family: str
    implementation_id: str
    lowering_implementation_id: str
    planner_version: str
    capability: DataflowResharedTransportCapability
    cluster_size: int
    logical_tile_count: int
    consumer_access_order: str
    physical_slot_count: int
    producer_slots_per_rank: int
    logical_tiles_per_physical_slot: int
    physical_slot_bytes: int
    logical_tile_bytes: int
    transaction_bytes: int
    transfer_threads: int
    payload_partitions: tuple[DataflowResharedPayloadPartition, ...]
    receive_stage_count: int
    credit_count: int
    remote_barrier_count: int
    publish_sync_required: bool
    release_sync_required: bool
    steps: tuple[DataflowResharedTransportStep, ...]
    credit_lifetimes: tuple[DataflowResharedCreditLifetime, ...]
    decision: DataflowOperationDecision
    schema_version: int = DATAFLOW_RESHARED_TRANSPORT_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_RESHARED_TRANSPORT_PLAN_SCHEMA_VERSION:
            raise ValueError(f"unsupported reshared transport plan schema {self.schema_version}")
        if self.planner_version != DATAFLOW_RESHARED_TRANSPORT_PLANNER_VERSION:
            raise ValueError(f"unsupported reshared planner {self.planner_version!r}")
        if self.decision.request.fingerprint != self.request_fingerprint:
            raise ValueError("reshared plan request and decision fingerprints differ")
        if self.decision.selected_implementation != self.implementation_id:
            raise ValueError("reshared plan implementation and decision differ")
        if self.lowering_implementation_id != resolve_lowering_implementation_id(
            self.family,
            self.consumer_access_order,
        ):
            raise ValueError("reshared plan physical lowering implementation is stale")
        request = self.decision.request
        if not isinstance(request, DataflowResharedTransportRequest):
            raise TypeError("reshared plan decision must contain a transport request")
        if request.family != DATAFLOW_TRANSPORT_AUTO and request.family != self.family:
            raise ValueError("explicit reshared family differs from the selected family")
        if self.cluster_size <= 0 or self.logical_tile_count <= 0:
            raise ValueError("reshared plan extents must be positive")
        if self.physical_slot_count <= 0 or self.producer_slots_per_rank <= 0:
            raise ValueError("reshared physical slot extents must be positive")
        if self.physical_slot_count != self.producer_slots_per_rank * self.cluster_size:
            raise ValueError("reshared physical slots must partition evenly by rank")
        if self.logical_tile_count != (self.physical_slot_count * self.logical_tiles_per_physical_slot):
            raise ValueError("reshared logical/physical tile arity is inconsistent")
        if (
            request.logical_output_arity != self.logical_tile_count
            or request.physical_output_arity != self.physical_slot_count
            or request.consumer_access_order != self.consumer_access_order
        ):
            raise ValueError("reshared plan topology differs from its typed request")
        if self.physical_slot_bytes != (self.logical_tile_bytes * self.logical_tiles_per_physical_slot):
            raise ValueError("reshared logical/physical slot bytes are inconsistent")
        if self.transaction_bytes <= 0:
            raise ValueError("reshared transaction_bytes must be positive")
        partitions = tuple(self.payload_partitions)
        object.__setattr__(self, "payload_partitions", partitions)
        if sum(part.byte_count for part in partitions) != self.logical_tile_bytes:
            raise ValueError("reshared payload partitions do not cover one logical tile")
        expected_offset = 0
        for index, part in enumerate(partitions):
            if part.partition_index != index or part.byte_offset != expected_offset:
                raise ValueError("reshared payload partitions must be contiguous and ordered")
            if part.byte_count > self.transaction_bytes:
                raise ValueError("reshared payload partition exceeds transaction_bytes")
            expected_offset += part.byte_count
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "credit_lifetimes", tuple(self.credit_lifetimes))
        if len(self.steps) != self.cluster_size * self.logical_tile_count:
            raise ValueError("reshared plan must record every consumer access step")
        expected_steps = transport_steps(
            request,
            family=self.family,
            cluster_size=self.cluster_size,
            receive_stage_count=self.receive_stage_count,
        )
        if self.steps != expected_steps:
            raise ValueError("reshared transport steps do not match the typed topology")
        expected_lifetimes = credit_lifetimes(
            expected_steps,
            self.cluster_size,
            self.receive_stage_count,
        )
        if self.credit_lifetimes != expected_lifetimes:
            raise ValueError("reshared credit lifetimes do not match transport steps")
        if self.family == DATAFLOW_TRANSPORT_STREAMED:
            if self.receive_stage_count <= 0 or self.credit_count != self.receive_stage_count:
                raise ValueError("streamed transport requires one credit per receive stage")
            push_lowering = self.lowering_implementation_id == DATAFLOW_RESHARED_STREAMED_PUSH_IMPLEMENTATION
            if self.publish_sync_required == push_lowering:
                raise ValueError("streamed transport publish sync must be owned by exactly one of the scheduler or producer-push lowering")
            if not self.release_sync_required:
                raise ValueError("streamed transport requires release sync")
            if not self.capability.cluster_transport_supported:
                raise ValueError("streamed transport requires cluster capability")
            if (
                self.transfer_threads <= 0
                or self.capability.transfer_threads is None
                or self.transfer_threads > self.capability.transfer_threads
            ):
                raise ValueError("streamed transport has an invalid transfer thread count")
        elif self.receive_stage_count or self.credit_count or self.credit_lifetimes:
            raise ValueError("non-streamed transport cannot own receive credits")
        elif self.transfer_threads != 0:
            raise ValueError("non-streamed transport cannot own transfer threads")
        expected_remote_barriers = (
            self.receive_stage_count if self.lowering_implementation_id == DATAFLOW_RESHARED_STREAMED_PUSH_IMPLEMENTATION else 0
        )
        if self.remote_barrier_count != expected_remote_barriers:
            raise ValueError("reshared remote barrier count does not match its physical lowering")

    @property
    def resources(self) -> DataflowOperationResourceEstimate:
        selected = self.decision.resources
        assert selected is not None
        return selected

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "planner_version": self.planner_version,
            "request_fingerprint": self.request_fingerprint,
            "family": self.family,
            "implementation_id": self.implementation_id,
            "lowering_implementation_id": self.lowering_implementation_id,
            "capability": self.capability.to_dict(),
            "cluster_size": self.cluster_size,
            "logical_tile_count": self.logical_tile_count,
            "consumer_access_order": self.consumer_access_order,
            "physical_slot_count": self.physical_slot_count,
            "producer_slots_per_rank": self.producer_slots_per_rank,
            "logical_tiles_per_physical_slot": self.logical_tiles_per_physical_slot,
            "physical_slot_bytes": self.physical_slot_bytes,
            "logical_tile_bytes": self.logical_tile_bytes,
            "transaction_bytes": self.transaction_bytes,
            "transfer_threads": self.transfer_threads,
            "payload_partitions": [item.to_dict() for item in self.payload_partitions],
            "receive_stage_count": self.receive_stage_count,
            "credit_count": self.credit_count,
            "remote_barrier_count": self.remote_barrier_count,
            "publish_sync_required": self.publish_sync_required,
            "release_sync_required": self.release_sync_required,
            "steps": [item.to_dict() for item in self.steps],
            "credit_lifetimes": [item.to_dict() for item in self.credit_lifetimes],
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
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowResharedTransportPlan:
        allowed = set(cls.__dataclass_fields__) | {"fingerprint", "capability"}
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError(f"reshared transport plan has unknown fields {sorted(unknown)!r}")
        capability_value = value["capability"]
        capability = DataflowResharedTransportCapability(
            cluster_transport_supported=capability_value["cluster_transport_supported"],
            max_transaction_bytes=capability_value["max_transaction_bytes"],
            transfer_threads=capability_value["transfer_threads"],
            default_receive_stages=capability_value["default_receive_stages"],
            registry_key=capability_value["registry_key"],
        )
        plan = cls(
            request_fingerprint=str(value["request_fingerprint"]),
            family=str(value["family"]),
            implementation_id=str(value["implementation_id"]),
            lowering_implementation_id=str(value["lowering_implementation_id"]),
            planner_version=str(value["planner_version"]),
            capability=capability,
            cluster_size=int(value["cluster_size"]),
            logical_tile_count=int(value["logical_tile_count"]),
            consumer_access_order=str(value["consumer_access_order"]),
            physical_slot_count=int(value["physical_slot_count"]),
            producer_slots_per_rank=int(value["producer_slots_per_rank"]),
            logical_tiles_per_physical_slot=int(value["logical_tiles_per_physical_slot"]),
            physical_slot_bytes=int(value["physical_slot_bytes"]),
            logical_tile_bytes=int(value["logical_tile_bytes"]),
            transaction_bytes=int(value["transaction_bytes"]),
            transfer_threads=int(value["transfer_threads"]),
            payload_partitions=tuple(DataflowResharedPayloadPartition.from_dict(item) for item in value["payload_partitions"]),
            receive_stage_count=int(value["receive_stage_count"]),
            credit_count=int(value["credit_count"]),
            remote_barrier_count=int(value["remote_barrier_count"]),
            publish_sync_required=value["publish_sync_required"],
            release_sync_required=value["release_sync_required"],
            steps=tuple(DataflowResharedTransportStep.from_dict(item) for item in value["steps"]),
            credit_lifetimes=tuple(DataflowResharedCreditLifetime.from_dict(item) for item in value["credit_lifetimes"]),
            decision=DataflowOperationDecision.from_dict(value["decision"]),
            schema_version=int(value["schema_version"]),
        )
        if value.get("fingerprint") not in {None, plan.fingerprint}:
            raise ValueError("reshared transport plan fingerprint does not match payload")
        return plan


class DataflowResharedTransportPlanningError(ValueError):
    """Structured rejection for an explicit or auto transport request."""

    def __init__(self, message: str, decision: DataflowOperationDecision):
        super().__init__(message)
        self.decision = decision


def plan_reshared_transport(
    request: DataflowResharedTransportRequest,
    *,
    cluster_size: int,
    physical_slot_bytes: int,
    target_capabilities: TargetCapabilitySnapshot | None = None,
    receive_stage_count: int | None = None,
    available_threads: int | None = None,
) -> DataflowResharedTransportPlan:
    """Select and fully describe one target-aware reshared transport."""

    if not isinstance(request, DataflowResharedTransportRequest):
        raise TypeError("reshared transport planning requires a typed request")
    if cluster_size <= 0 or physical_slot_bytes <= 0:
        raise ValueError("cluster_size and physical_slot_bytes must be positive")
    if request.physical_output_arity % cluster_size:
        raise ValueError("physical_output_arity must divide evenly across producer ranks")
    if request.logical_output_arity % request.physical_output_arity:
        raise ValueError("logical_output_arity must be divisible by physical_output_arity")
    logical_tiles_per_physical = request.logical_output_arity // request.physical_output_arity
    if physical_slot_bytes % logical_tiles_per_physical:
        raise ValueError("physical slot bytes must partition evenly into logical tiles")
    logical_tile_bytes = physical_slot_bytes // logical_tiles_per_physical
    if request.slot_bytes is not None and request.slot_bytes != logical_tile_bytes:
        raise ValueError(
            f"typed reshared slot_bytes does not match the physical intermediate layout: {request.slot_bytes} != {logical_tile_bytes}"
        )

    capability = resolve_reshared_transport_capability(target_capabilities)
    stages = capability.default_receive_stages if receive_stage_count is None else receive_stage_count
    if isinstance(stages, bool) or not isinstance(stages, int):
        raise TypeError("receive_stage_count must be an integer")
    stages = min(stages, request.logical_output_arity)
    if stages <= 0:
        raise ValueError("receive_stage_count must be positive")
    transaction_limit = capability.max_transaction_bytes or logical_tile_bytes
    if request.max_transaction_bytes is not None:
        transaction_limit = min(transaction_limit, request.max_transaction_bytes)
    if available_threads is not None and (
        isinstance(available_threads, bool) or not isinstance(available_threads, int) or available_threads <= 0
    ):
        raise ValueError("available_threads must be a positive integer")
    transfer_threads = 0
    if capability.transfer_threads is not None:
        transfer_threads = min(
            capability.transfer_threads,
            available_threads or capability.transfer_threads,
        )
        transfer_threads = (transfer_threads // 32) * 32

    candidates = transport_candidates(
        request,
        cluster_size=cluster_size,
        logical_tile_bytes=logical_tile_bytes,
        receive_stage_count=stages,
        capability=capability,
        transaction_bytes=transaction_limit,
        transfer_threads=transfer_threads,
    )
    selected_family = select_transport_family(
        request,
        candidates,
        stages,
        cluster_size=cluster_size,
        capability=capability,
    )
    implementation_id = family_implementation_id(selected_family)
    lowering_implementation_id = resolve_lowering_implementation_id(
        selected_family,
        request.consumer_access_order,
    )
    selected_candidate = next(item for item in candidates if item.implementation_id == implementation_id)
    if not selected_candidate.legal:
        decision = DataflowOperationDecision(
            request=request,
            candidates=(selected_candidate,),
            selected_implementation=None,
            used_fallback=False,
            selection_reason="explicit_transport_family_rejected",
        )
        raise DataflowResharedTransportPlanningError("; ".join(selected_candidate.rejection_reasons), decision)
    decision = DataflowOperationDecision(
        request=request,
        candidates=tuple(candidates if request.family == DATAFLOW_TRANSPORT_AUTO else (selected_candidate,)),
        selected_implementation=implementation_id,
        used_fallback=False,
        selection_reason=(
            "auto_transport_cost_and_resource_selection" if request.family == DATAFLOW_TRANSPORT_AUTO else "explicit_transport_family"
        ),
    )

    planned_transaction_bytes = transaction_limit if transaction_limit > 0 else logical_tile_bytes
    partitions = tuple(
        DataflowResharedPayloadPartition(index, offset, min(planned_transaction_bytes, logical_tile_bytes - offset))
        for index, offset in enumerate(range(0, logical_tile_bytes, planned_transaction_bytes))
    )
    streamed = selected_family == DATAFLOW_TRANSPORT_STREAMED
    plan_stages = stages if streamed else 0
    steps = transport_steps(
        request,
        family=selected_family,
        cluster_size=cluster_size,
        receive_stage_count=plan_stages,
    )
    lifetimes = credit_lifetimes(steps, cluster_size, plan_stages)
    return DataflowResharedTransportPlan(
        request_fingerprint=request.fingerprint,
        family=selected_family,
        implementation_id=implementation_id,
        lowering_implementation_id=lowering_implementation_id,
        planner_version=DATAFLOW_RESHARED_TRANSPORT_PLANNER_VERSION,
        capability=capability,
        cluster_size=cluster_size,
        logical_tile_count=request.logical_output_arity,
        consumer_access_order=request.consumer_access_order,
        physical_slot_count=request.physical_output_arity,
        producer_slots_per_rank=request.physical_output_arity // cluster_size,
        logical_tiles_per_physical_slot=logical_tiles_per_physical,
        physical_slot_bytes=physical_slot_bytes,
        logical_tile_bytes=logical_tile_bytes,
        transaction_bytes=planned_transaction_bytes,
        transfer_threads=transfer_threads if streamed else 0,
        payload_partitions=partitions,
        receive_stage_count=plan_stages,
        credit_count=plan_stages,
        remote_barrier_count=(plan_stages if lowering_implementation_id == DATAFLOW_RESHARED_STREAMED_PUSH_IMPLEMENTATION else 0),
        publish_sync_required=(streamed and lowering_implementation_id != DATAFLOW_RESHARED_STREAMED_PUSH_IMPLEMENTATION),
        release_sync_required=streamed,
        steps=steps,
        credit_lifetimes=lifetimes,
        decision=decision,
    )


def transport_candidates(
    request: DataflowResharedTransportRequest,
    *,
    cluster_size: int,
    logical_tile_bytes: int,
    receive_stage_count: int,
    capability: DataflowResharedTransportCapability,
    transaction_bytes: int,
    transfer_threads: int,
) -> tuple[DataflowOperationCandidate, ...]:
    full_bytes = request.logical_output_arity * logical_tile_bytes
    streamed_bytes = receive_stage_count * logical_tile_bytes
    temporary_limit = request.max_temporary_bytes

    def candidate(
        implementation_id: str,
        *,
        temporary_bytes: int,
        reasons: tuple[str, ...] = (),
        barrier_count: int = 0,
    ) -> DataflowOperationCandidate:
        return DataflowOperationCandidate(
            implementation_id=implementation_id,
            legal=not reasons,
            resources=DataflowOperationResourceEstimate(
                shared_memory_bytes=temporary_bytes,
                barrier_count=barrier_count,
                slot_bytes=logical_tile_bytes,
                transaction_bytes=max(0, transaction_bytes),
                temporary_bytes=temporary_bytes,
            ),
            rejection_reasons=reasons,
        )

    multi_rank_cluster_reasons = []
    if cluster_size > 1 and not capability.cluster_transport_supported:
        multi_rank_cluster_reasons.append("target lacks cluster DSM transport capability")
    if transaction_bytes <= 0:
        multi_rank_cluster_reasons.append("transport transaction budget is zero")

    all_gather_reasons = list(multi_rank_cluster_reasons)
    if temporary_limit is not None and full_bytes > temporary_limit:
        all_gather_reasons.append("all-gather temporary bytes exceed max_temporary_bytes")
    streamed_reasons = list(multi_rank_cluster_reasons)
    if cluster_size <= 1:
        streamed_reasons.append("streamed transport requires cluster_size > 1")
        if not capability.cluster_transport_supported:
            streamed_reasons.append("target lacks cluster DSM transport capability")
    if transfer_threads < 32:
        streamed_reasons.append("streamed transport requires at least one cooperative warp")
    if temporary_limit is not None and streamed_bytes > temporary_limit:
        streamed_reasons.append("streamed receive stages exceed max_temporary_bytes")
    return (
        candidate(
            DATAFLOW_TRANSPORT_HBM_IMPLEMENTATION,
            temporary_bytes=full_bytes,
        ),
        candidate(
            DATAFLOW_TRANSPORT_ALL_GATHER_IMPLEMENTATION,
            temporary_bytes=full_bytes,
            reasons=tuple(all_gather_reasons),
            barrier_count=request.logical_output_arity,
        ),
        candidate(
            DATAFLOW_TRANSPORT_STREAMED_IMPLEMENTATION,
            temporary_bytes=streamed_bytes,
            reasons=tuple(streamed_reasons),
            barrier_count=2 * receive_stage_count,
        ),
    )


def select_transport_family(
    request: DataflowResharedTransportRequest,
    candidates: tuple[DataflowOperationCandidate, ...],
    receive_stage_count: int,
    *,
    cluster_size: int,
    capability: DataflowResharedTransportCapability,
) -> str:
    if request.family != DATAFLOW_TRANSPORT_AUTO:
        return request.family
    if cluster_size <= 1 or not capability.cluster_transport_supported:
        return DATAFLOW_TRANSPORT_HBM
    legal = {item.implementation_id for item in candidates if item.legal}
    if DATAFLOW_TRANSPORT_STREAMED_IMPLEMENTATION in legal and request.logical_output_arity > receive_stage_count:
        return DATAFLOW_TRANSPORT_STREAMED
    if DATAFLOW_TRANSPORT_ALL_GATHER_IMPLEMENTATION in legal:
        return DATAFLOW_TRANSPORT_ALL_GATHER
    return DATAFLOW_TRANSPORT_HBM


def family_implementation_id(family: str) -> str:
    return {
        DATAFLOW_TRANSPORT_HBM: DATAFLOW_TRANSPORT_HBM_IMPLEMENTATION,
        DATAFLOW_TRANSPORT_ALL_GATHER: DATAFLOW_TRANSPORT_ALL_GATHER_IMPLEMENTATION,
        DATAFLOW_TRANSPORT_STREAMED: DATAFLOW_TRANSPORT_STREAMED_IMPLEMENTATION,
    }[family]


def resolve_lowering_implementation_id(
    family: str,
    consumer_access_order: str,
) -> str:
    return {
        DATAFLOW_TRANSPORT_HBM: DATAFLOW_RESHARED_HBM_LOWERING_IMPLEMENTATION,
        DATAFLOW_TRANSPORT_ALL_GATHER: (DATAFLOW_RESHARED_ALL_GATHER_LOWERING_IMPLEMENTATION),
        DATAFLOW_TRANSPORT_STREAMED: (
            DATAFLOW_RESHARED_STREAMED_PUSH_IMPLEMENTATION
            if consumer_access_order == DATAFLOW_TRANSPORT_INDEPENDENT_ORDER
            else DATAFLOW_RESHARED_STREAMED_PULL_IMPLEMENTATION
        ),
    }[family]


def transport_steps(
    request: DataflowResharedTransportRequest,
    *,
    family: str,
    cluster_size: int,
    receive_stage_count: int,
) -> tuple[DataflowResharedTransportStep, ...]:
    physical_per_rank = request.physical_output_arity // cluster_size
    logical_per_physical = request.logical_output_arity // request.physical_output_arity
    logical_per_rank = physical_per_rank * logical_per_physical
    steps = []
    for consumer_rank in range(cluster_size):
        for access_step in range(request.logical_output_arity):
            if request.consumer_access_order == DATAFLOW_TRANSPORT_INDEPENDENT_ORDER:
                rank_step, local_logical = divmod(access_step, logical_per_rank)
                source_rank = (consumer_rank + rank_step) % cluster_size
                logical_tile = source_rank * logical_per_rank + local_logical
            else:
                logical_tile = access_step
            physical_slot, source_tile = divmod(logical_tile, logical_per_physical)
            source_rank, source_local_slot = divmod(physical_slot, physical_per_rank)
            locality = "global" if family == DATAFLOW_TRANSPORT_HBM else "local" if source_rank == consumer_rank else "remote"
            streamed = family == DATAFLOW_TRANSPORT_STREAMED
            receive_stage = access_step % receive_stage_count if streamed else None
            steps.append(
                DataflowResharedTransportStep(
                    consumer_rank=consumer_rank,
                    access_step=access_step,
                    logical_tile_index=logical_tile,
                    physical_slot_index=physical_slot,
                    source_rank=source_rank,
                    source_local_slot_index=source_local_slot,
                    source_tile_index=source_tile,
                    locality=locality,
                    transport_required=(locality != "local"),
                    receive_stage=receive_stage,
                    credit_acquire_step=access_step if streamed else None,
                    credit_release_step=access_step if streamed else None,
                )
            )
    return tuple(steps)


def credit_lifetimes(
    steps: tuple[DataflowResharedTransportStep, ...],
    cluster_size: int,
    receive_stage_count: int,
) -> tuple[DataflowResharedCreditLifetime, ...]:
    if receive_stage_count == 0:
        return ()
    result = []
    for consumer_rank in range(cluster_size):
        rank_steps = [step for step in steps if step.consumer_rank == consumer_rank]
        for stage in range(receive_stage_count):
            stage_steps = [step for step in rank_steps if step.receive_stage == stage]
            result.append(
                DataflowResharedCreditLifetime(
                    consumer_rank=consumer_rank,
                    receive_stage=stage,
                    acquire_steps=tuple(int(step.credit_acquire_step) for step in stage_steps),
                    release_steps=tuple(int(step.credit_release_step) for step in stage_steps),
                )
            )
    return tuple(result)


def fingerprint(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def bind_reshared_transport_plan(
    prim_func: tir.PrimFunc,
    plan: DataflowResharedTransportPlan,
    *,
    source_buffer_names: tuple[str, ...],
    loop_index: int = 0,
) -> tir.PrimFunc:
    """Bind logical input tiles to the selected streamed DSM lowering."""

    if not isinstance(prim_func, tir.PrimFunc):
        raise TypeError("reshared transport binding requires a tir.PrimFunc")
    if not isinstance(plan, DataflowResharedTransportPlan):
        raise TypeError("reshared transport binding requires a typed plan")
    if plan.family != DATAFLOW_TRANSPORT_STREAMED:
        return prim_func
    if not source_buffer_names:
        raise ValueError("streamed transport binding requires input-slot buffers")

    source_buffers = {buffer.name: buffer for buffer in prim_func.buffer_map.values() if str(buffer.name) in source_buffer_names}
    missing = set(source_buffer_names).difference(str(name) for name in source_buffers)
    if missing:
        raise ValueError(f"streamed transport could not resolve input buffers {sorted(missing)!r}")
    source_data = {buffer.data for buffer in source_buffers.values()}
    logical_copy_vars: set[tir.Var] = set()
    let_dependencies: dict[tir.Var, set[tir.Var]] = {}

    def collect_expr_vars(expr: Any) -> set[tir.Var]:
        variables: set[tir.Var] = set()

        def collect_var(node: Any) -> None:
            if isinstance(node, tir.Var):
                variables.add(node)

        tir.stmt_functor.post_order_visit(expr, collect_var)
        return variables

    def collect_let_dependencies(node: Any) -> None:
        if isinstance(node, tir.LetStmt):
            let_dependencies[node.var] = collect_expr_vars(node.value)

    tir.stmt_functor.post_order_visit(prim_func.body, collect_let_dependencies)

    def collect_copy_vars(node: Any) -> None:
        if not (isinstance(node, tir.Call) and isinstance(node.op, ir.Op) and node.op.name == "tl.tileop.copy"):
            return
        parsed = _ffi_api.ParseOperator(node)
        if parsed.src.data not in source_data:
            return

        for region in parsed.src_range:
            logical_copy_vars.update(collect_expr_vars(region.min))

    tir.stmt_functor.post_order_visit(prim_func.body, collect_copy_vars)
    pending = list(logical_copy_vars)
    while pending:
        variable = pending.pop()
        for dependency in let_dependencies.get(variable, ()):
            if dependency not in logical_copy_vars:
                logical_copy_vars.add(dependency)
                pending.append(dependency)
    logical_loops: list[tir.For] = []

    def collect_loop(node: Any) -> None:
        if not isinstance(node, tir.For) or node.loop_var not in logical_copy_vars:
            return
        loop_min = getattr(node.min, "value", None)
        loop_extent = getattr(node.extent, "value", None)
        if loop_min == 0 and loop_extent == plan.logical_tile_count:
            logical_loops.append(node)

    tir.stmt_functor.post_order_visit(prim_func.body, collect_loop)
    if loop_index < 0 or loop_index >= len(logical_loops):
        raise ValueError(f"streamed transport expected logical access loop {loop_index}, found {len(logical_loops)}")
    selected_loop = logical_loops[loop_index]
    loop_min = getattr(selected_loop.min, "value", None)
    loop_extent = getattr(selected_loop.extent, "value", None)
    if loop_min != 0 or loop_extent != plan.logical_tile_count:
        raise ValueError(
            "streamed transport loop must enumerate each logical tile exactly once: "
            f"min={loop_min}, extent={loop_extent}, expected={plan.logical_tile_count}"
        )

    transport_body = prim_func.body
    if plan.consumer_access_order == DATAFLOW_TRANSPORT_INDEPENDENT_ORDER:
        index_dtype = selected_loop.loop_var.dtype
        access_step = selected_loop.loop_var
        logical_tiles_per_rank = plan.logical_tile_count // plan.cluster_size
        rank_step = tir.floordiv(
            access_step,
            tir.IntImm(index_dtype, logical_tiles_per_rank),
        )
        local_logical_tile = tir.floormod(
            access_step,
            tir.IntImm(index_dtype, logical_tiles_per_rank),
        )
        consumer_rank = tir.Call(
            index_dtype,
            ir.Op.get("tl.block_rank_in_cluster"),
            [],
        )
        semantic_source_rank = tir.floormod(
            consumer_rank + rank_step,
            tir.IntImm(index_dtype, plan.cluster_size),
        )
        semantic_logical_tile = semantic_source_rank * tir.IntImm(index_dtype, logical_tiles_per_rank) + local_logical_tile

        def rewrite_access_order(node: Any) -> Any:
            if not isinstance(node, tir.For) or not node.same_as(selected_loop):
                return node
            return tir.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                tir.stmt_functor.substitute(
                    node.body,
                    {node.loop_var: semantic_logical_tile},
                ),
                node.thread_binding,
                node.annotations,
                node.step,
                getattr(node, "span", None),
            )

        transport_body = tir.stmt_functor.ir_transform(
            transport_body,
            None,
            rewrite_access_order,
            ["tir.For"],
        )

    bound_copy_count = 0

    def rewrite_call(node: Any) -> Any:
        nonlocal bound_copy_count
        if not (isinstance(node, tir.Call) and isinstance(node.op, ir.Op) and node.op.name == "tl.tileop.copy"):
            return node
        parsed = _ffi_api.ParseOperator(node)
        if parsed.src.data not in source_data:
            return node
        if any(
            key in node.annotations
            for key in (
                "src_block",
                "dst_block",
                DATAFLOW_RESHARED_TRANSPORT_FAMILY_ATTR,
                DATAFLOW_RESHARED_SOURCE_RANK_ATTR,
            )
        ):
            raise ValueError("reshared physical transport annotations are compiler-owned")
        if len(parsed.src_range) < 1:
            raise ValueError("streamed transport source requires a slot dimension")
        leading_extent = getattr(parsed.src.shape[0], "value", None)
        if leading_extent != plan.producer_slots_per_rank:
            raise ValueError(
                "streamed transport input slot extent does not match producer-local plan: "
                f"{leading_extent} != {plan.producer_slots_per_rank}"
            )
        index_dtype = selected_loop.loop_var.dtype
        access_step = selected_loop.loop_var
        if plan.consumer_access_order == DATAFLOW_TRANSPORT_INDEPENDENT_ORDER:
            logical_tiles_per_rank = plan.logical_tile_count // plan.cluster_size
            rank_step = tir.floordiv(
                access_step,
                tir.IntImm(index_dtype, logical_tiles_per_rank),
            )
            local_logical_tile = tir.floormod(
                access_step,
                tir.IntImm(index_dtype, logical_tiles_per_rank),
            )
            consumer_rank = tir.Call(
                index_dtype,
                ir.Op.get("tl.block_rank_in_cluster"),
                [],
            )
            planned_source_rank = tir.floormod(
                consumer_rank + rank_step,
                tir.IntImm(index_dtype, plan.cluster_size),
            )
            logical_tile = planned_source_rank * tir.IntImm(index_dtype, logical_tiles_per_rank) + local_logical_tile
        else:
            logical_tile = access_step
        physical_slot = tir.floordiv(
            logical_tile,
            tir.IntImm(index_dtype, plan.logical_tiles_per_physical_slot),
        )
        local_slot = tir.floormod(
            physical_slot,
            tir.IntImm(index_dtype, plan.producer_slots_per_rank),
        )
        source_rank = tir.floordiv(
            physical_slot,
            tir.IntImm(index_dtype, plan.producer_slots_per_rank),
        )
        source_tile = tir.floormod(
            logical_tile,
            tir.IntImm(index_dtype, plan.logical_tiles_per_physical_slot),
        )

        def rewrite_region(expr: Any) -> Any:
            if not isinstance(expr, tir.Call):
                return expr
            if (
                isinstance(expr.op, ir.Op)
                and expr.op.name == "tl.tileop.region"
                and expr.args
                and isinstance(expr.args[0], tir.BufferLoad)
                and expr.args[0].buffer.data in source_data
            ):
                load = expr.args[0]
                indices = list(load.indices)
                indices[0] = local_slot
                if plan.logical_tiles_per_physical_slot > 1:
                    if len(indices) < 2:
                        raise ValueError("streamed transport source lacks its physical tile axis")
                    indices[1] = source_tile
                rewritten_load = tir.BufferLoad(
                    load.buffer,
                    indices,
                    load.predicate,
                    load.span,
                )
                return tir.Call(
                    expr.dtype,
                    expr.op,
                    [rewritten_load, *expr.args[1:]],
                    annotations=expr.annotations,
                    span=expr.span,
                )
            rewritten_args = [rewrite_region(arg) for arg in expr.args]
            if any(not new.same_as(old) for new, old in zip(rewritten_args, expr.args)):
                return tir.Call(
                    expr.dtype,
                    expr.op,
                    rewritten_args,
                    annotations=expr.annotations,
                    span=expr.span,
                )
            return expr

        rewritten_args = [rewrite_region(arg) for arg in node.args]
        annotations = dict(node.annotations)
        if plan.lowering_implementation_id == DATAFLOW_RESHARED_STREAMED_PUSH_IMPLEMENTATION:
            if plan.consumer_access_order != DATAFLOW_TRANSPORT_INDEPENDENT_ORDER:
                raise ValueError("producer-push streamed transport requires independent consumer access order")
            logical_tiles_per_rank = plan.logical_tile_count // plan.cluster_size
            rank_step = tir.floordiv(
                access_step,
                tir.IntImm(index_dtype, logical_tiles_per_rank),
            )
            local_rank = tir.Call(
                index_dtype,
                ir.Op.get("tl.block_rank_in_cluster"),
                [],
            )
            destination_rank = tir.floormod(
                local_rank - rank_step,
                tir.IntImm(index_dtype, plan.cluster_size),
            )
            future_rank_step = tir.floordiv(
                access_step + tir.IntImm(index_dtype, plan.receive_stage_count),
                tir.IntImm(index_dtype, logical_tiles_per_rank),
            )
            credit_target_rank = tir.floormod(
                local_rank + future_rank_step,
                tir.IntImm(index_dtype, plan.cluster_size),
            )
            annotations["dst_block"] = destination_rank
            annotations[DATAFLOW_RESHARED_CREDIT_TARGET_RANK_ATTR] = credit_target_rank
            annotations[DATAFLOW_RESHARED_RECEIVE_STAGES_ATTR] = tir.IntImm(
                "int32",
                plan.receive_stage_count,
            )
        else:
            annotations["src_block"] = source_rank
        annotations[DATAFLOW_RESHARED_TRANSPORT_FAMILY_ATTR] = tir.StringImm(DATAFLOW_TRANSPORT_STREAMED)
        annotations[DATAFLOW_RESHARED_SOURCE_RANK_ATTR] = source_rank
        annotations[DATAFLOW_RESHARED_PAYLOAD_PARTITION_BYTES_ATTR] = tvm.runtime.convert(
            [tir.IntImm("int64", item.byte_count) for item in plan.payload_partitions]
        )
        assert plan.capability.transfer_threads is not None
        annotations[DATAFLOW_RESHARED_TRANSFER_THREADS_ATTR] = tir.IntImm("int32", plan.transfer_threads)
        annotations[DATAFLOW_RESHARED_TRANSPORT_PLAN_FINGERPRINT_ATTR] = tir.StringImm(plan.fingerprint)
        bound_copy_count += 1
        return tir.Call(
            node.dtype,
            node.op,
            rewritten_args,
            annotations=annotations,
            span=node.span,
        )

    body = tir.stmt_functor.ir_transform(
        transport_body,
        None,
        rewrite_call,
        ["tir.Call"],
    )
    if bound_copy_count != 1:
        raise ValueError(f"streamed transport requires exactly one logical input copy per pipeline loop, found {bound_copy_count}")
    return (
        prim_func.with_body(body)
        .with_attr(
            DATAFLOW_RESHARED_TRANSPORT_PLAN_SCHEMA_ATTR,
            plan.schema_version,
        )
        .with_attr(
            DATAFLOW_RESHARED_TRANSPORT_PLAN_FINGERPRINT_ATTR,
            plan.fingerprint,
        )
    )


__all__ = [
    "DATAFLOW_RESHARED_ALL_GATHER_LOWERING_IMPLEMENTATION",
    "DATAFLOW_RESHARED_CREDIT_TARGET_RANK_ATTR",
    "DATAFLOW_RESHARED_HBM_LOWERING_IMPLEMENTATION",
    "DATAFLOW_RESHARED_RECEIVE_STAGES_ATTR",
    "DATAFLOW_RESHARED_STREAMED_PUSH_IMPLEMENTATION",
    "DATAFLOW_RESHARED_STREAMED_PULL_IMPLEMENTATION",
    "DATAFLOW_RESHARED_PAYLOAD_PARTITION_BYTES_ATTR",
    "DATAFLOW_RESHARED_SOURCE_RANK_ATTR",
    "DATAFLOW_RESHARED_TRANSFER_THREADS_ATTR",
    "DATAFLOW_RESHARED_TRANSPORT_FAMILY_ATTR",
    "DATAFLOW_RESHARED_TRANSPORT_PLAN_FINGERPRINT_ATTR",
    "DATAFLOW_RESHARED_TRANSPORT_PLAN_SCHEMA_ATTR",
    "DATAFLOW_RESHARED_TRANSPORT_PLAN_SCHEMA_VERSION",
    "DATAFLOW_RESHARED_TRANSPORT_PLANNER_VERSION",
    "DATAFLOW_RESHARED_TRANSPORT_CAPABILITY_REGISTRY",
    "DataflowResharedCreditLifetime",
    "DataflowResharedPayloadPartition",
    "DataflowResharedTransportCapability",
    "DataflowResharedTransportPlan",
    "DataflowResharedTransportPlanningError",
    "DataflowResharedTransportStep",
    "bind_reshared_transport_plan",
    "plan_reshared_transport",
    "resolve_reshared_transport_capability",
]
