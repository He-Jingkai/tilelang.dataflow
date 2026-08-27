"""Resource-constrained scheduling for Dataflow compute and communication events."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
from typing import Any
from collections.abc import Mapping, Sequence

from .topology import GPUTopology


DATAFLOW_JOINT_EXECUTION_PLAN_SCHEMA_VERSION = 3
DATAFLOW_JOINT_EXECUTION_PLAN_IMPLEMENTATION = "dataflow.joint_execution.v3"


class DataflowJointScheduleError(ValueError):
    """Raised when a joint compute/communication schedule is not realizable."""


class DataflowTransportKind(str, Enum):
    LOCAL = "local"
    CLUSTER_PUSH = "cluster_push"
    HBM_STAGED = "hbm_staged"


class DataflowScheduleEventKind(str, Enum):
    HANDLER = "handler"
    CLUSTER_PUSH_ISSUE = "cluster_push_issue"
    CLUSTER_PUSH_READY = "cluster_push_ready"
    HBM_STORE_ISSUE = "hbm_store_issue"
    HBM_STORE_READY = "hbm_store_ready"
    HBM_LOAD_ISSUE = "hbm_load_issue"
    HBM_LOAD_READY = "hbm_load_ready"
    INBOX_RETAIN_COPY = "inbox_retain_copy"
    TRANSFER_READY = "transfer_ready"
    SOURCE_RELEASE = "source_release"


class DataflowCommSlotMode(str, Enum):
    INBOX = "inbox"
    OUTBOX = "outbox"


class DataflowCommStorageKind(str, Enum):
    """Physical storage class occupied by one scheduled handoff interval."""

    PERMANENT = "permanent"
    TRANSIENT_PREFETCH = "transient_prefetch"


@dataclass(frozen=True)
class DataflowJointScheduleNode:
    """One fixed-CTA compute node in a logical value DAG.

    ``output_alias_input_index`` is a physical capability, not a scheduler
    preference.  It declares which input may share storage with the output.
    ``None`` means that no input/output alias may be assumed.
    """

    node_id: int
    cta_id: int
    duration_us: float
    input_values: tuple[int, ...] = ()
    output_value: int | None = None
    output_alias_input_index: int | None = None
    transient_prefetch_bytes: int = 0
    sort_key: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.node_id, bool) or not isinstance(self.node_id, int) or self.node_id < 0:
            raise DataflowJointScheduleError(f"joint schedule node id must be a non-negative integer, got {self.node_id!r}")
        if isinstance(self.cta_id, bool) or not isinstance(self.cta_id, int) or self.cta_id < 0:
            raise DataflowJointScheduleError(f"joint schedule CTA id must be a non-negative integer, got {self.cta_id!r}")
        duration = float(self.duration_us)
        if not math.isfinite(duration) or duration < 0.0:
            raise DataflowJointScheduleError(f"joint schedule node duration must be finite and non-negative, got {self.duration_us!r}")
        object.__setattr__(self, "duration_us", duration)
        if self.output_alias_input_index is not None and (
            self.output_alias_input_index < 0 or self.output_alias_input_index >= len(self.input_values)
        ):
            raise DataflowJointScheduleError(
                f"joint schedule node {self.node_id} has invalid alias input index {self.output_alias_input_index}"
            )
        if (
            isinstance(self.transient_prefetch_bytes, bool)
            or not isinstance(self.transient_prefetch_bytes, int)
            or self.transient_prefetch_bytes < 0
        ):
            raise DataflowJointScheduleError(
                f"joint schedule transient prefetch capacity must be a non-negative integer, got {self.transient_prefetch_bytes!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "cta_id": self.cta_id,
            "duration_us": self.duration_us,
            "input_values": list(self.input_values),
            "output_value": self.output_value,
            "output_alias_input_index": self.output_alias_input_index,
            "transient_prefetch_bytes": self.transient_prefetch_bytes,
            "sort_key": list(self.sort_key),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowJointScheduleNode:
        return cls(
            node_id=int(value["node_id"]),
            cta_id=int(value["cta_id"]),
            duration_us=float(value["duration_us"]),
            input_values=tuple(int(item) for item in value.get("input_values", ())),
            output_value=(None if value.get("output_value") is None else int(value["output_value"])),
            output_alias_input_index=(None if value.get("output_alias_input_index") is None else int(value["output_alias_input_index"])),
            transient_prefetch_bytes=int(value.get("transient_prefetch_bytes", 0)),
            sort_key=tuple(value.get("sort_key", ())),
        )


@dataclass(frozen=True)
class DataflowScheduleEvent:
    event_id: int
    kind: DataflowScheduleEventKind
    cta_id: int | None
    node_id: int | None
    transfer_id: int | None
    predecessor_event_ids: tuple[int, ...]
    start_us: float
    end_us: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind.value,
            "cta_id": self.cta_id,
            "node_id": self.node_id,
            "transfer_id": self.transfer_id,
            "predecessor_event_ids": list(self.predecessor_event_ids),
            "start_us": self.start_us,
            "end_us": self.end_us,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowScheduleEvent:
        return cls(
            event_id=int(value["event_id"]),
            kind=DataflowScheduleEventKind(value["kind"]),
            cta_id=None if value.get("cta_id") is None else int(value["cta_id"]),
            node_id=(None if value.get("node_id") is None else int(value["node_id"])),
            transfer_id=(None if value.get("transfer_id") is None else int(value["transfer_id"])),
            predecessor_event_ids=tuple(int(item) for item in value.get("predecessor_event_ids", ())),
            start_us=float(value["start_us"]),
            end_us=float(value["end_us"]),
        )


@dataclass(frozen=True)
class DataflowCommSlotInterval:
    cta_id: int
    epoch: int
    storage_kind: DataflowCommStorageKind
    storage_lane: int
    mode: DataflowCommSlotMode
    value_id: int
    transfer_id: int | None
    begin_event_id: int
    end_event_id: int
    begin_us: float
    end_us: float

    def __post_init__(self) -> None:
        if isinstance(self.storage_lane, bool) or not isinstance(self.storage_lane, int) or self.storage_lane < 0:
            raise DataflowJointScheduleError(f"joint communicate storage lane must be a non-negative integer, got {self.storage_lane!r}")
        if self.storage_kind is DataflowCommStorageKind.PERMANENT and self.storage_lane != 0:
            raise DataflowJointScheduleError("permanent communicate storage only has lane zero")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cta_id": self.cta_id,
            "epoch": self.epoch,
            "storage_kind": self.storage_kind.value,
            "storage_lane": self.storage_lane,
            "mode": self.mode.value,
            "value_id": self.value_id,
            "transfer_id": self.transfer_id,
            "begin_event_id": self.begin_event_id,
            "end_event_id": self.end_event_id,
            "begin_us": self.begin_us,
            "end_us": self.end_us,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowCommSlotInterval:
        return cls(
            cta_id=int(value["cta_id"]),
            epoch=int(value["epoch"]),
            storage_kind=DataflowCommStorageKind(value.get("storage_kind", DataflowCommStorageKind.PERMANENT.value)),
            storage_lane=int(value.get("storage_lane", 0)),
            mode=DataflowCommSlotMode(value["mode"]),
            value_id=int(value["value_id"]),
            transfer_id=(None if value.get("transfer_id") is None else int(value["transfer_id"])),
            begin_event_id=int(value["begin_event_id"]),
            end_event_id=int(value["end_event_id"]),
            begin_us=float(value["begin_us"]),
            end_us=float(value["end_us"]),
        )


@dataclass(frozen=True)
class DataflowTransferSegmentSchedule:
    segment_id: int
    byte_offset: int
    byte_count: int
    store_issue_event_id: int
    store_ready_event_id: int
    load_issue_event_id: int | None
    load_ready_event_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "byte_offset": self.byte_offset,
            "byte_count": self.byte_count,
            "store_issue_event_id": self.store_issue_event_id,
            "store_ready_event_id": self.store_ready_event_id,
            "load_issue_event_id": self.load_issue_event_id,
            "load_ready_event_id": self.load_ready_event_id,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> DataflowTransferSegmentSchedule:
        return cls(
            segment_id=int(value["segment_id"]),
            byte_offset=int(value["byte_offset"]),
            byte_count=int(value["byte_count"]),
            store_issue_event_id=int(value["store_issue_event_id"]),
            store_ready_event_id=int(value["store_ready_event_id"]),
            load_issue_event_id=(None if value.get("load_issue_event_id") is None else int(value["load_issue_event_id"])),
            load_ready_event_id=int(value["load_ready_event_id"]),
        )


@dataclass(frozen=True)
class DataflowTransferSchedule:
    transfer_id: int
    kind: DataflowTransportKind
    value_id: int
    producer_node_id: int
    consumer_node_id: int
    consumer_input_index: int
    producer_cta: int
    consumer_cta: int
    producer_issue_event_id: int
    producer_release_event_id: int
    consumer_issue_event_id: int | None
    destination_ready_event_id: int
    consume_event_id: int
    destination_release_event_id: int
    producer_slot_epoch: int | None
    consumer_slot_epoch: int
    consumer_storage_kind: DataflowCommStorageKind = DataflowCommStorageKind.PERMANENT
    consumer_storage_lane: int = 0
    segments: tuple[DataflowTransferSegmentSchedule, ...] = ()
    hbm_stage_id: int | None = None
    retained_input_copy_event_id: int | None = None
    modeled_producer_guard_wait_us: float = 0.0
    modeled_consumer_guard_wait_us: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.consumer_storage_lane, bool)
            or not isinstance(self.consumer_storage_lane, int)
            or self.consumer_storage_lane < 0
        ):
            raise DataflowJointScheduleError(
                f"joint transfer consumer storage lane must be a non-negative integer, got {self.consumer_storage_lane!r}"
            )
        if self.consumer_storage_kind is DataflowCommStorageKind.PERMANENT and self.consumer_storage_lane != 0:
            raise DataflowJointScheduleError("permanent communicate storage only has lane zero")

    def to_dict(self) -> dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "kind": self.kind.value,
            "value_id": self.value_id,
            "producer_node_id": self.producer_node_id,
            "consumer_node_id": self.consumer_node_id,
            "consumer_input_index": self.consumer_input_index,
            "producer_cta": self.producer_cta,
            "consumer_cta": self.consumer_cta,
            "producer_issue_event_id": self.producer_issue_event_id,
            "producer_release_event_id": self.producer_release_event_id,
            "consumer_issue_event_id": self.consumer_issue_event_id,
            "destination_ready_event_id": self.destination_ready_event_id,
            "consume_event_id": self.consume_event_id,
            "destination_release_event_id": self.destination_release_event_id,
            "producer_slot_epoch": self.producer_slot_epoch,
            "consumer_slot_epoch": self.consumer_slot_epoch,
            "consumer_storage_kind": self.consumer_storage_kind.value,
            "consumer_storage_lane": self.consumer_storage_lane,
            "segments": [segment.to_dict() for segment in self.segments],
            "hbm_stage_id": self.hbm_stage_id,
            "retained_input_copy_event_id": self.retained_input_copy_event_id,
            "modeled_producer_guard_wait_us": self.modeled_producer_guard_wait_us,
            "modeled_consumer_guard_wait_us": self.modeled_consumer_guard_wait_us,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowTransferSchedule:
        def optional_int(name: str) -> int | None:
            return None if value.get(name) is None else int(value[name])

        return cls(
            transfer_id=int(value["transfer_id"]),
            kind=DataflowTransportKind(value["kind"]),
            value_id=int(value["value_id"]),
            producer_node_id=int(value["producer_node_id"]),
            consumer_node_id=int(value["consumer_node_id"]),
            consumer_input_index=int(value["consumer_input_index"]),
            producer_cta=int(value["producer_cta"]),
            consumer_cta=int(value["consumer_cta"]),
            producer_issue_event_id=int(value["producer_issue_event_id"]),
            producer_release_event_id=int(value["producer_release_event_id"]),
            consumer_issue_event_id=optional_int("consumer_issue_event_id"),
            destination_ready_event_id=int(value["destination_ready_event_id"]),
            consume_event_id=int(value["consume_event_id"]),
            destination_release_event_id=int(value["destination_release_event_id"]),
            producer_slot_epoch=optional_int("producer_slot_epoch"),
            consumer_slot_epoch=int(value["consumer_slot_epoch"]),
            consumer_storage_kind=DataflowCommStorageKind(
                value.get(
                    "consumer_storage_kind",
                    DataflowCommStorageKind.PERMANENT.value,
                )
            ),
            consumer_storage_lane=int(value.get("consumer_storage_lane", 0)),
            segments=tuple(DataflowTransferSegmentSchedule.from_dict(item) for item in value.get("segments", ())),
            hbm_stage_id=optional_int("hbm_stage_id"),
            retained_input_copy_event_id=optional_int("retained_input_copy_event_id"),
            modeled_producer_guard_wait_us=float(value.get("modeled_producer_guard_wait_us", 0.0)),
            modeled_consumer_guard_wait_us=float(value.get("modeled_consumer_guard_wait_us", 0.0)),
        )


@dataclass(frozen=True)
class DataflowJointExecutionSchedule:
    events: tuple[DataflowScheduleEvent, ...]
    transfers: tuple[DataflowTransferSchedule, ...]
    comm_slot_intervals: tuple[DataflowCommSlotInterval, ...]
    queue_node_ids: tuple[tuple[int, tuple[int, ...]], ...]
    makespan_us: float
    critical_path_guard_wait_us: float
    total_guard_wait_us: float
    diagnostics: tuple[str, ...] = ()

    def queue(self, cta_id: int) -> tuple[int, ...]:
        for owner, node_ids in self.queue_node_ids:
            if owner == cta_id:
                return node_ids
        return ()

    def event(self, event_id: int) -> DataflowScheduleEvent:
        try:
            event = self.events[event_id]
        except IndexError as err:
            raise KeyError(event_id) from err
        if event.event_id != event_id:
            raise KeyError(event_id)
        return event

    def require_valid(
        self,
        *,
        topology: GPUTopology,
        nodes: Sequence[DataflowJointScheduleNode],
    ) -> DataflowJointExecutionSchedule:
        nodes_by_id = {node.node_id: node for node in nodes}
        if len(nodes_by_id) != len(nodes):
            raise DataflowJointScheduleError("joint schedule contains duplicate node ids")
        event_ids = tuple(event.event_id for event in self.events)
        if event_ids != tuple(range(len(self.events))):
            raise DataflowJointScheduleError("joint schedule event ids must be dense and ordered")
        for event in self.events:
            if event.cta_id is not None and not 0 <= event.cta_id < topology.sm_count:
                raise DataflowJointScheduleError(f"joint schedule event {event.event_id} uses invalid CTA {event.cta_id}")
            if event.start_us < 0.0 or event.end_us < event.start_us:
                raise DataflowJointScheduleError(f"joint schedule event {event.event_id} has invalid interval")
            for predecessor_id in event.predecessor_event_ids:
                if predecessor_id < 0 or predecessor_id >= len(self.events):
                    raise DataflowJointScheduleError(f"joint schedule event {event.event_id} has invalid predecessor {predecessor_id}")
                predecessor = self.events[predecessor_id]
                if predecessor.end_us > event.start_us + 1e-9:
                    raise DataflowJointScheduleError(
                        f"joint schedule event {event.event_id} starts before predecessor {predecessor_id} completes"
                    )

        visit_state = [0] * len(self.events)

        def visit_event(event_id: int) -> None:
            state = visit_state[event_id]
            if state == 2:
                return
            if state == 1:
                raise DataflowJointScheduleError("joint schedule event graph contains a cycle")
            visit_state[event_id] = 1
            for predecessor_id in self.events[event_id].predecessor_event_ids:
                visit_event(predecessor_id)
            visit_state[event_id] = 2

        for event in self.events:
            visit_event(event.event_id)

        dependency_cache: dict[tuple[int, int], bool] = {}

        def depends_on(event_id: int, predecessor_id: int) -> bool:
            key = (event_id, predecessor_id)
            cached = dependency_cache.get(key)
            if cached is not None:
                return cached
            event = self.events[event_id]
            result = predecessor_id in event.predecessor_event_ids or any(
                depends_on(parent_id, predecessor_id) for parent_id in event.predecessor_event_ids
            )
            dependency_cache[key] = result
            return result

        issue_intervals_by_cta: dict[int, list[DataflowScheduleEvent]] = {}
        for event in self.events:
            if event.cta_id is not None and event.end_us > event.start_us:
                issue_intervals_by_cta.setdefault(event.cta_id, []).append(event)
        for cta_id, issue_intervals in issue_intervals_by_cta.items():
            previous_end = 0.0
            for event in sorted(
                issue_intervals,
                key=lambda item: (item.start_us, item.end_us, item.event_id),
            ):
                if event.start_us + 1e-9 < previous_end:
                    raise DataflowJointScheduleError(f"CTA {cta_id} has overlapping issue events at event {event.event_id}")
                previous_end = max(previous_end, event.end_us)

        queued_nodes: list[int] = []
        for cta_id, node_ids in self.queue_node_ids:
            if not 0 <= cta_id < topology.sm_count:
                raise DataflowJointScheduleError(f"joint schedule has invalid queue CTA {cta_id}")
            previous_end = 0.0
            for node_id in node_ids:
                node = nodes_by_id.get(node_id)
                if node is None or node.cta_id != cta_id:
                    raise DataflowJointScheduleError(f"joint schedule queue CTA {cta_id} contains invalid node {node_id}")
                event = next(
                    (item for item in self.events if item.kind is DataflowScheduleEventKind.HANDLER and item.node_id == node_id),
                    None,
                )
                if event is None or event.start_us + 1e-9 < previous_end:
                    raise DataflowJointScheduleError(f"joint schedule queue CTA {cta_id} is not sequential at node {node_id}")
                previous_end = event.end_us
                queued_nodes.append(node_id)
        if sorted(queued_nodes) != sorted(nodes_by_id):
            raise DataflowJointScheduleError("joint schedule queues must contain every compute node exactly once")

        intervals_by_cta: dict[int, list[DataflowCommSlotInterval]] = {}
        intervals_by_resource: dict[tuple[int, DataflowCommStorageKind, int], list[DataflowCommSlotInterval]] = {}
        for interval in self.comm_slot_intervals:
            if not 0 <= interval.cta_id < topology.sm_count:
                raise DataflowJointScheduleError(f"communicate-slot interval uses invalid CTA {interval.cta_id}")
            if interval.begin_us < 0.0 or interval.end_us < interval.begin_us:
                raise DataflowJointScheduleError(
                    f"joint schedule communicate-slot epoch {interval.cta_id}/{interval.epoch} has an invalid interval"
                )
            if self.event(interval.begin_event_id).start_us > interval.begin_us + 1e-9:
                raise DataflowJointScheduleError("communicate-slot interval begins before its event")
            if self.event(interval.end_event_id).end_us + 1e-9 < interval.end_us:
                raise DataflowJointScheduleError("communicate-slot interval outlives its end event")
            if interval.storage_kind is DataflowCommStorageKind.TRANSIENT_PREFETCH and interval.mode is not DataflowCommSlotMode.INBOX:
                raise DataflowJointScheduleError("transient prefetch storage may only hold an inbox value")
            intervals_by_cta.setdefault(interval.cta_id, []).append(interval)
            intervals_by_resource.setdefault(
                (
                    interval.cta_id,
                    interval.storage_kind,
                    interval.storage_lane,
                ),
                [],
            ).append(interval)
        for (cta_id, storage_kind, storage_lane), intervals in intervals_by_resource.items():
            intervals.sort(key=lambda item: (item.begin_us, item.end_us, item.epoch))
            previous_epoch = 0
            previous_end = 0.0
            for interval in intervals:
                if interval.epoch <= previous_epoch:
                    raise DataflowJointScheduleError(f"CTA {cta_id} {storage_kind.value}/{storage_lane} storage epochs are not monotonic")
                if interval.begin_us + 1e-9 < previous_end:
                    raise DataflowJointScheduleError(f"CTA {cta_id} {storage_kind.value}/{storage_lane} storage capacity exceeds one")
                previous_epoch = interval.epoch
                previous_end = interval.end_us

        transfer_ids = tuple(transfer.transfer_id for transfer in self.transfers)
        if transfer_ids != tuple(range(len(self.transfers))):
            raise DataflowJointScheduleError("joint schedule transfer ids must be dense and ordered")

        handler_event_by_node: dict[int, DataflowScheduleEvent] = {}
        for event in self.events:
            if event.kind is not DataflowScheduleEventKind.HANDLER:
                continue
            if event.node_id is None or event.node_id in handler_event_by_node:
                raise DataflowJointScheduleError(f"joint schedule has an invalid handler event {event.event_id}")
            handler_event_by_node[event.node_id] = event

        producer_by_value: dict[int, int] = {}
        for node in nodes:
            if node.output_value is None:
                continue
            previous = producer_by_value.setdefault(node.output_value, node.node_id)
            if previous != node.node_id:
                raise DataflowJointScheduleError(f"joint schedule value {node.output_value} has multiple producers")
        expected_remote_edges: set[tuple[int, int, int]] = set()
        for consumer in nodes:
            for value_id in set(consumer.input_values):
                producer_id = producer_by_value.get(value_id)
                if producer_id is None:
                    continue
                producer = nodes_by_id[producer_id]
                if producer.cta_id != consumer.cta_id:
                    expected_remote_edges.add((producer_id, consumer.node_id, value_id))

        actual_remote_edges: set[tuple[int, int, int]] = set()
        for transfer in self.transfers:
            producer = nodes_by_id.get(transfer.producer_node_id)
            consumer = nodes_by_id.get(transfer.consumer_node_id)
            if producer is None or consumer is None:
                raise DataflowJointScheduleError(f"joint transfer {transfer.transfer_id} references an unknown node")
            if (
                transfer.producer_cta != producer.cta_id
                or transfer.consumer_cta != consumer.cta_id
                or producer.output_value != transfer.value_id
                or transfer.consumer_input_index < 0
                or transfer.consumer_input_index >= len(consumer.input_values)
                or consumer.input_values[transfer.consumer_input_index] != transfer.value_id
            ):
                raise DataflowJointScheduleError(f"joint transfer {transfer.transfer_id} does not match its data edge")
            edge = (producer.node_id, consumer.node_id, transfer.value_id)
            if edge in actual_remote_edges:
                raise DataflowJointScheduleError(f"joint schedule duplicates remote edge {edge!r}")
            actual_remote_edges.add(edge)

            producer_issue = self.event(transfer.producer_issue_event_id)
            producer_release = self.event(transfer.producer_release_event_id)
            ready = self.event(transfer.destination_ready_event_id)
            consume = self.event(transfer.consume_event_id)
            release = self.event(transfer.destination_release_event_id)
            producer_handler = handler_event_by_node.get(producer.node_id)
            if producer_handler is None:
                raise DataflowJointScheduleError(f"joint transfer {transfer.transfer_id} has no producer handler")
            if (
                producer_issue.transfer_id != transfer.transfer_id
                or ready.transfer_id != transfer.transfer_id
                or consume.kind is not DataflowScheduleEventKind.HANDLER
                or consume.node_id != consumer.node_id
                or producer_release.kind is not DataflowScheduleEventKind.SOURCE_RELEASE
                or producer_release.node_id != producer.node_id
            ):
                raise DataflowJointScheduleError(f"joint transfer {transfer.transfer_id} references mismatched events")
            if producer_issue.start_us + 1e-9 < producer_handler.end_us:
                raise DataflowJointScheduleError(f"joint transfer {transfer.transfer_id} issues before production")
            destination_dependency_event_ids = (ready.event_id,)
            if transfer.kind is DataflowTransportKind.CLUSTER_PUSH:
                if not topology.same_cluster(transfer.producer_cta, transfer.consumer_cta):
                    raise DataflowJointScheduleError(f"joint transfer {transfer.transfer_id} uses cluster push across clusters")
                if transfer.hbm_stage_id is not None or transfer.consumer_issue_event_id is not None:
                    raise DataflowJointScheduleError(f"cluster transfer {transfer.transfer_id} carries HBM metadata")
                if transfer.segments:
                    raise DataflowJointScheduleError(f"cluster transfer {transfer.transfer_id} carries HBM segments")
                if (
                    producer_issue.kind is not DataflowScheduleEventKind.CLUSTER_PUSH_ISSUE
                    or ready.kind is not DataflowScheduleEventKind.CLUSTER_PUSH_READY
                    or not depends_on(ready.event_id, producer_issue.event_id)
                    or producer_release.end_us + 1e-9 < ready.end_us
                    or not depends_on(producer_release.event_id, ready.event_id)
                ):
                    raise DataflowJointScheduleError(f"cluster transfer {transfer.transfer_id} has an unsafe source lifetime")
            elif transfer.kind is DataflowTransportKind.HBM_STAGED:
                if transfer.hbm_stage_id is None or transfer.consumer_issue_event_id is None or not transfer.segments:
                    raise DataflowJointScheduleError(f"HBM transfer {transfer.transfer_id} is missing staging metadata")
                if ready.kind is not DataflowScheduleEventKind.TRANSFER_READY:
                    raise DataflowJointScheduleError(f"HBM transfer {transfer.transfer_id} has no destination-ready join")
                destination_dependency_event_ids = tuple(segment.load_ready_event_id for segment in transfer.segments)
                previous_segment_id = -1
                previous_byte_end = 0
                for segment in sorted(
                    transfer.segments,
                    key=lambda item: (item.byte_offset, item.segment_id),
                ):
                    if (
                        segment.segment_id <= previous_segment_id
                        or segment.byte_offset < previous_byte_end
                        or segment.byte_offset < 0
                        or segment.byte_count < 0
                        or segment.load_issue_event_id is None
                    ):
                        raise DataflowJointScheduleError(f"HBM transfer {transfer.transfer_id} has invalid segments")
                    store_issue = self.event(segment.store_issue_event_id)
                    store_ready = self.event(segment.store_ready_event_id)
                    load_issue = self.event(segment.load_issue_event_id)
                    load_ready = self.event(segment.load_ready_event_id)
                    if (
                        store_issue.kind is not DataflowScheduleEventKind.HBM_STORE_ISSUE
                        or store_ready.kind is not DataflowScheduleEventKind.HBM_STORE_READY
                        or load_issue.kind is not DataflowScheduleEventKind.HBM_LOAD_ISSUE
                        or load_ready.kind is not DataflowScheduleEventKind.HBM_LOAD_READY
                        or not depends_on(store_ready.event_id, store_issue.event_id)
                        or not depends_on(load_issue.event_id, store_ready.event_id)
                        or not depends_on(load_ready.event_id, load_issue.event_id)
                        or not depends_on(ready.event_id, load_ready.event_id)
                        or not depends_on(producer_release.event_id, store_ready.event_id)
                    ):
                        raise DataflowJointScheduleError(
                            f"HBM transfer {transfer.transfer_id} segment {segment.segment_id} violates store/load ordering"
                        )
                    previous_segment_id = segment.segment_id
                    previous_byte_end = segment.byte_offset + segment.byte_count
            else:
                raise DataflowJointScheduleError(f"non-local transfer {transfer.transfer_id} has invalid kind {transfer.kind}")
            if producer_release.end_us + 1e-9 < producer_issue.end_us:
                raise DataflowJointScheduleError(f"joint transfer {transfer.transfer_id} releases source before issue")
            if consume.start_us + 1e-9 < ready.end_us or not all(
                depends_on(consume.event_id, dependency_event_id) for dependency_event_id in destination_dependency_event_ids
            ):
                raise DataflowJointScheduleError(f"joint transfer {transfer.transfer_id} consumes before destination ready")
            if transfer.retained_input_copy_event_id is None:
                if release.event_id != consume.event_id or release.end_us + 1e-9 < consume.end_us:
                    raise DataflowJointScheduleError(f"joint transfer {transfer.transfer_id} releases inbox before consumption")
            else:
                retained_copy = self.event(transfer.retained_input_copy_event_id)
                if (
                    retained_copy.kind is not DataflowScheduleEventKind.INBOX_RETAIN_COPY
                    or retained_copy.transfer_id != transfer.transfer_id
                    or release.event_id != retained_copy.event_id
                    or retained_copy.start_us + 1e-9 < ready.end_us
                    or not all(
                        depends_on(retained_copy.event_id, dependency_event_id) for dependency_event_id in destination_dependency_event_ids
                    )
                    or consume.start_us + 1e-9 < retained_copy.end_us
                    or not depends_on(consume.event_id, retained_copy.event_id)
                ):
                    raise DataflowJointScheduleError(f"joint transfer {transfer.transfer_id} has an unsafe retained-input copy")

            consumer_intervals = [
                interval
                for interval in intervals_by_cta.get(transfer.consumer_cta, ())
                if interval.epoch == transfer.consumer_slot_epoch
                and interval.storage_kind is transfer.consumer_storage_kind
                and interval.storage_lane == transfer.consumer_storage_lane
                and interval.mode is DataflowCommSlotMode.INBOX
                and interval.transfer_id == transfer.transfer_id
            ]
            if len(consumer_intervals) != 1:
                raise DataflowJointScheduleError(f"joint transfer {transfer.transfer_id} has no unique consumer inbox epoch")
            consumer_interval = consumer_intervals[0]
            if consumer_interval.begin_us > ready.end_us + 1e-9 or consumer_interval.end_us + 1e-9 < release.end_us:
                raise DataflowJointScheduleError(f"joint transfer {transfer.transfer_id} has an unsafe inbox lifetime")
            if transfer.consumer_storage_kind is DataflowCommStorageKind.TRANSIENT_PREFETCH:
                required_bytes = sum(segment.byte_count for segment in transfer.segments)
                for handler_event in handler_event_by_node.values():
                    if handler_event.cta_id != transfer.consumer_cta:
                        continue
                    if (
                        handler_event.end_us <= consumer_interval.begin_us + 1e-9
                        or handler_event.start_us + 1e-9 >= consumer_interval.end_us
                    ):
                        continue
                    handler_node = nodes_by_id[handler_event.node_id]
                    required_capacity = (transfer.consumer_storage_lane + 1) * required_bytes
                    if handler_node.transient_prefetch_bytes < required_capacity:
                        raise DataflowJointScheduleError(
                            f"joint transfer {transfer.transfer_id} keeps transient lane "
                            f"{transfer.consumer_storage_lane} ({required_capacity} total "
                            f"prefetch bytes) live across incompatible node "
                            f"{handler_node.node_id}"
                        )

            if transfer.producer_slot_epoch is not None:
                producer_intervals = [
                    interval
                    for interval in intervals_by_cta.get(transfer.producer_cta, ())
                    if interval.epoch == transfer.producer_slot_epoch
                    and interval.storage_kind is DataflowCommStorageKind.PERMANENT
                    and interval.storage_lane == 0
                    and interval.mode is DataflowCommSlotMode.OUTBOX
                    and interval.value_id == transfer.value_id
                ]
                if len(producer_intervals) != 1:
                    raise DataflowJointScheduleError(f"joint transfer {transfer.transfer_id} has no unique producer outbox epoch")
                if producer_intervals[0].end_us + 1e-9 < producer_release.end_us:
                    raise DataflowJointScheduleError(f"joint transfer {transfer.transfer_id} releases its outbox too early")

        if actual_remote_edges != expected_remote_edges:
            missing = sorted(expected_remote_edges - actual_remote_edges)
            unexpected = sorted(actual_remote_edges - expected_remote_edges)
            raise DataflowJointScheduleError(f"joint schedule remote-edge mismatch: missing={missing!r}, unexpected={unexpected!r}")

        expected_makespan = max((event.end_us for event in self.events), default=0.0)
        if abs(expected_makespan - self.makespan_us) > 1e-6:
            raise DataflowJointScheduleError(f"joint schedule makespan mismatch: {self.makespan_us} != {expected_makespan}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [event.to_dict() for event in self.events],
            "transfers": [transfer.to_dict() for transfer in self.transfers],
            "comm_slot_intervals": [interval.to_dict() for interval in self.comm_slot_intervals],
            "queue_node_ids": {str(cta_id): list(node_ids) for cta_id, node_ids in self.queue_node_ids},
            "makespan_us": self.makespan_us,
            "critical_path_guard_wait_us": self.critical_path_guard_wait_us,
            "total_guard_wait_us": self.total_guard_wait_us,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> DataflowJointExecutionSchedule:
        queue_node_ids = value.get("queue_node_ids", {})
        if not isinstance(queue_node_ids, Mapping):
            raise TypeError("joint execution queue_node_ids must be a mapping")
        return cls(
            events=tuple(DataflowScheduleEvent.from_dict(item) for item in value.get("events", ())),
            transfers=tuple(DataflowTransferSchedule.from_dict(item) for item in value.get("transfers", ())),
            comm_slot_intervals=tuple(DataflowCommSlotInterval.from_dict(item) for item in value.get("comm_slot_intervals", ())),
            queue_node_ids=tuple(
                (int(cta_id), tuple(int(item) for item in node_ids))
                for cta_id, node_ids in sorted(
                    queue_node_ids.items(),
                    key=lambda item: int(item[0]),
                )
            ),
            makespan_us=float(value["makespan_us"]),
            critical_path_guard_wait_us=float(value.get("critical_path_guard_wait_us", 0.0)),
            total_guard_wait_us=float(value.get("total_guard_wait_us", 0.0)),
            diagnostics=tuple(str(item) for item in value.get("diagnostics", ())),
        )


@dataclass(frozen=True)
class DataflowJointExecutionPlan:
    """Serializable schedule selected by both scoring and physical lowering."""

    nodes: tuple[DataflowJointScheduleNode, ...]
    schedule: DataflowJointExecutionSchedule
    hbm_segment_bytes: int | None = None
    schema_version: int = DATAFLOW_JOINT_EXECUTION_PLAN_SCHEMA_VERSION
    implementation: str = DATAFLOW_JOINT_EXECUTION_PLAN_IMPLEMENTATION

    def __post_init__(self) -> None:
        if self.schema_version != DATAFLOW_JOINT_EXECUTION_PLAN_SCHEMA_VERSION:
            raise ValueError(f"unsupported Dataflow joint execution schema {self.schema_version}")
        if self.implementation != DATAFLOW_JOINT_EXECUTION_PLAN_IMPLEMENTATION:
            raise ValueError(f"unsupported Dataflow joint execution implementation {self.implementation!r}")
        if self.hbm_segment_bytes is not None and (
            isinstance(self.hbm_segment_bytes, bool) or not isinstance(self.hbm_segment_bytes, int) or self.hbm_segment_bytes <= 0
        ):
            raise ValueError("joint execution hbm_segment_bytes must be a positive integer or None")

    def require_valid(self, *, topology: GPUTopology) -> DataflowJointExecutionPlan:
        self.schedule.require_valid(topology=topology, nodes=self.nodes)
        return self

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "implementation": self.implementation,
            "hbm_segment_bytes": self.hbm_segment_bytes,
            "nodes": [node.to_dict() for node in self.nodes],
            "schedule": self.schedule.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self.canonical_payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DataflowJointExecutionPlan:
        plan = cls(
            nodes=tuple(DataflowJointScheduleNode.from_dict(item) for item in value.get("nodes", ())),
            schedule=DataflowJointExecutionSchedule.from_dict(value["schedule"]),
            hbm_segment_bytes=(None if value.get("hbm_segment_bytes") is None else int(value["hbm_segment_bytes"])),
            schema_version=int(
                value.get(
                    "schema_version",
                    DATAFLOW_JOINT_EXECUTION_PLAN_SCHEMA_VERSION,
                )
            ),
            implementation=str(
                value.get(
                    "implementation",
                    DATAFLOW_JOINT_EXECUTION_PLAN_IMPLEMENTATION,
                )
            ),
        )
        if value.get("fingerprint") not in {None, plan.fingerprint}:
            raise ValueError("Dataflow joint execution plan fingerprint is stale")
        return plan


@dataclass(frozen=True)
class TransferSpec:
    transfer_id: int
    value_id: int
    producer_node_id: int
    consumer_node_id: int
    consumer_input_index: int
    producer_cta: int
    consumer_cta: int
    kind: DataflowTransportKind


@dataclass(frozen=True)
class Operation:
    key: tuple[str, int]
    cta_id: int
    cta_duration_us: float
    completion_delay_us: float
    predecessors: tuple[tuple[str, int], ...]
    node_id: int | None
    transfer_id: int | None
    segment_id: int | None
    sort_key: tuple[Any, ...]


@dataclass
class OpenSlotInterval:
    cta_id: int
    epoch: int
    storage_kind: DataflowCommStorageKind
    storage_lane: int
    mode: DataflowCommSlotMode
    value_id: int
    transfer_id: int | None
    begin_event_id: int
    begin_us: float


_StorageResource = tuple[DataflowCommStorageKind, int]
_CTAStorageResource = tuple[int, DataflowCommStorageKind, int]


def operation_kind(key: tuple[str, int]) -> DataflowScheduleEventKind:
    return {
        "handler": DataflowScheduleEventKind.HANDLER,
        "cluster_push": DataflowScheduleEventKind.CLUSTER_PUSH_ISSUE,
        "hbm_store": DataflowScheduleEventKind.HBM_STORE_ISSUE,
        "hbm_load": DataflowScheduleEventKind.HBM_LOAD_ISSUE,
        "retain_copy": DataflowScheduleEventKind.INBOX_RETAIN_COPY,
    }[key[0]]


def resolve_completion_kind(key: tuple[str, int]) -> DataflowScheduleEventKind | None:
    return {
        "handler": None,
        "cluster_push": DataflowScheduleEventKind.CLUSTER_PUSH_READY,
        "hbm_store": DataflowScheduleEventKind.HBM_STORE_READY,
        "hbm_load": DataflowScheduleEventKind.HBM_LOAD_READY,
        "retain_copy": None,
    }[key[0]]


def build_transfer_specs(
    topology: GPUTopology,
    nodes: Sequence[DataflowJointScheduleNode],
) -> tuple[
    tuple[TransferSpec, ...],
    dict[int, DataflowJointScheduleNode],
    dict[int, int],
]:
    nodes_by_id = {node.node_id: node for node in nodes}
    if len(nodes_by_id) != len(nodes):
        raise DataflowJointScheduleError("joint schedule nodes must have unique ids")
    producer_by_value: dict[int, int] = {}
    for node in nodes:
        if node.cta_id >= topology.sm_count:
            raise DataflowJointScheduleError(f"joint schedule node {node.node_id} uses CTA {node.cta_id} outside topology")
        if node.output_value is None:
            continue
        previous = producer_by_value.setdefault(node.output_value, node.node_id)
        if previous != node.node_id:
            raise DataflowJointScheduleError(f"joint schedule value {node.output_value} has multiple producers")

    transfers: list[TransferSpec] = []
    for consumer in nodes:
        seen_values: set[int] = set()
        for input_index, value_id in enumerate(consumer.input_values):
            if value_id in seen_values:
                continue
            seen_values.add(value_id)
            producer_id = producer_by_value.get(value_id)
            if producer_id is None:
                continue
            producer = nodes_by_id[producer_id]
            if producer.cta_id == consumer.cta_id:
                continue
            kind = (
                DataflowTransportKind.CLUSTER_PUSH
                if topology.same_cluster(producer.cta_id, consumer.cta_id)
                else DataflowTransportKind.HBM_STAGED
            )
            transfers.append(
                TransferSpec(
                    transfer_id=len(transfers),
                    value_id=value_id,
                    producer_node_id=producer.node_id,
                    consumer_node_id=consumer.node_id,
                    consumer_input_index=input_index,
                    producer_cta=producer.cta_id,
                    consumer_cta=consumer.cta_id,
                    kind=kind,
                )
            )
    return tuple(transfers), nodes_by_id, producer_by_value


def schedule_joint_compute_communication(
    topology: GPUTopology,
    nodes: Sequence[DataflowJointScheduleNode],
    *,
    cluster_transfer_us: float,
    hbm_transfer_us: float,
    transfer_issue_us: float = 0.0,
    retained_copy_us: float = 0.0,
    value_nbytes: Mapping[int, int] | None = None,
    hbm_segment_bytes: int | None = None,
    fixed_queue_node_ids: Mapping[int, Sequence[int]] | None = None,
    cluster_async_receive_pipeline: bool = False,
    critical_hbm_permanent_prefetch: bool = False,
    serialize_permanent_inbox_reuse: bool = False,
    hbm_spill_blocked_cluster_push: bool = False,
) -> DataflowJointExecutionSchedule:
    """Schedule fixed-placement compute with capacity-one asynchronous handoff.

    This is a deterministic resource-constrained list scheduler.  Transport
    operations are first-class nodes.  It never represents communication as a
    scalar delay attached to a consumer handler.
    """

    if not isinstance(topology, GPUTopology):
        raise TypeError(f"topology must be GPUTopology, got {topology!r}")
    if not isinstance(cluster_async_receive_pipeline, bool):
        raise TypeError(f"cluster_async_receive_pipeline must be a bool, got {cluster_async_receive_pipeline!r}")
    if not isinstance(critical_hbm_permanent_prefetch, bool):
        raise TypeError(f"critical_hbm_permanent_prefetch must be a bool, got {critical_hbm_permanent_prefetch!r}")
    if not isinstance(serialize_permanent_inbox_reuse, bool):
        raise TypeError(f"serialize_permanent_inbox_reuse must be a bool, got {serialize_permanent_inbox_reuse!r}")
    if not isinstance(hbm_spill_blocked_cluster_push, bool):
        raise TypeError(f"hbm_spill_blocked_cluster_push must be a bool, got {hbm_spill_blocked_cluster_push!r}")
    nodes = tuple(nodes)
    value_nbytes = {} if value_nbytes is None else dict(value_nbytes)
    if hbm_segment_bytes is not None and (
        isinstance(hbm_segment_bytes, bool) or not isinstance(hbm_segment_bytes, int) or hbm_segment_bytes <= 0
    ):
        raise DataflowJointScheduleError(f"hbm_segment_bytes must be a positive integer or None, got {hbm_segment_bytes!r}")
    for value_id, byte_count in value_nbytes.items():
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise DataflowJointScheduleError(f"value_nbytes[{value_id!r}] must be a positive integer, got {byte_count!r}")
    for name, value in (
        ("cluster_transfer_us", cluster_transfer_us),
        ("hbm_transfer_us", hbm_transfer_us),
        ("transfer_issue_us", transfer_issue_us),
        ("retained_copy_us", retained_copy_us),
    ):
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0.0:
            raise DataflowJointScheduleError(f"{name} must be finite and non-negative, got {value!r}")

    transfers, nodes_by_id, producer_by_value = build_transfer_specs(
        topology,
        nodes,
    )
    fixed_predecessor_by_node: dict[int, int] = {}
    fixed_position_by_node: dict[int, int] = {}
    if fixed_queue_node_ids is not None:
        queued_node_ids: list[int] = []
        for cta_id, node_ids in fixed_queue_node_ids.items():
            if isinstance(cta_id, bool) or not isinstance(cta_id, int) or not 0 <= cta_id < topology.sm_count:
                raise DataflowJointScheduleError(f"fixed joint queue has invalid CTA id {cta_id!r}")
            previous_node_id = None
            for queue_position, node_id in enumerate(node_ids):
                node = nodes_by_id.get(node_id)
                if node is None or node.cta_id != cta_id:
                    raise DataflowJointScheduleError(f"fixed joint queue CTA {cta_id} contains invalid node {node_id!r}")
                if previous_node_id is not None:
                    fixed_predecessor_by_node[node_id] = previous_node_id
                fixed_position_by_node[node_id] = queue_position
                previous_node_id = node_id
                queued_node_ids.append(node_id)
        if len(queued_node_ids) != len(set(queued_node_ids)):
            raise DataflowJointScheduleError("fixed joint queues contain one compute node more than once")
        if set(queued_node_ids) != set(nodes_by_id):
            missing = sorted(set(nodes_by_id) - set(queued_node_ids))
            unexpected = sorted(set(queued_node_ids) - set(nodes_by_id))
            raise DataflowJointScheduleError(
                f"fixed joint queues must contain every compute node exactly once: missing={missing!r}, unexpected={unexpected!r}"
            )
    if hbm_spill_blocked_cluster_push and fixed_queue_node_ids is not None:
        fixed_queues = {cta_id: tuple(node_ids) for cta_id, node_ids in fixed_queue_node_ids.items()}
        queue_start_us: dict[int, float] = {}
        queue_end_us: dict[int, float] = {}
        for node_ids in fixed_queues.values():
            cursor_us = 0.0
            for node_id in node_ids:
                queue_start_us[node_id] = cursor_us
                cursor_us += nodes_by_id[node_id].duration_us
                queue_end_us[node_id] = cursor_us

        promoted_transfer_ids: set[int] = set()
        for transfer in transfers:
            if transfer.kind is not DataflowTransportKind.CLUSTER_PUSH:
                continue
            producer_queue = fixed_queues.get(transfer.producer_cta, ())
            producer_position = fixed_position_by_node[transfer.producer_node_id]
            if producer_position + 1 >= len(producer_queue):
                continue
            next_node = nodes_by_id[producer_queue[producer_position + 1]]
            producer = nodes_by_id[transfer.producer_node_id]
            if producer.output_value is not None and producer.output_value in next_node.input_values:
                continue
            consumer_release_us = queue_start_us[transfer.consumer_node_id]
            producer_ready_us = queue_end_us[transfer.producer_node_id]
            blocked_us = consumer_release_us - producer_ready_us
            hbm_extra_us = max(
                0.0,
                float(hbm_transfer_us) - float(cluster_transfer_us),
            )
            useful_overlap_us = min(blocked_us, next_node.duration_us)
            if useful_overlap_us <= hbm_extra_us + 1e-9:
                continue
            promoted_transfer_ids.add(transfer.transfer_id)
        if promoted_transfer_ids:
            transfers = tuple(
                replace(transfer, kind=DataflowTransportKind.HBM_STAGED) if transfer.transfer_id in promoted_transfer_ids else transfer
                for transfer in transfers
            )
    diagnostics: list[str] = []
    if hbm_spill_blocked_cluster_push:
        promoted = tuple(
            transfer.transfer_id
            for transfer in transfers
            if transfer.kind is DataflowTransportKind.HBM_STAGED
            and topology.same_cluster(
                transfer.producer_cta,
                transfer.consumer_cta,
            )
        )
        if promoted:
            diagnostics.append(f"spill blocked same-cluster pushes through HBM for transfers {list(promoted)}")
    for node in nodes:
        duplicate_values = sorted(value_id for value_id in set(node.input_values) if node.input_values.count(value_id) > 1)
        if duplicate_values:
            diagnostics.append(f"node {node.node_id} reuses input values {duplicate_values}; each value shares one physical arrival")

    incoming_lists: dict[int, list[TransferSpec]] = {}
    outgoing_lists: dict[int, list[TransferSpec]] = {}
    for transfer in transfers:
        incoming_lists.setdefault(transfer.consumer_node_id, []).append(transfer)
        outgoing_lists.setdefault(transfer.producer_node_id, []).append(transfer)
    incoming_by_node = {
        node_id: tuple(sorted(items, key=lambda item: item.consumer_input_index)) for node_id, items in incoming_lists.items()
    }
    outgoing_by_node = {
        node_id: tuple(
            sorted(
                items,
                key=lambda item: (
                    item.consumer_node_id,
                    item.consumer_input_index,
                    item.transfer_id,
                ),
            )
        )
        for node_id, items in outgoing_lists.items()
    }
    for node_id, outgoing in outgoing_by_node.items():
        if len(outgoing) > 1:
            diagnostics.append(
                f"node {node_id} has {len(outgoing)} remote consumers; retain its source through every asynchronous transfer"
            )

    local_consumer_producers: set[int] = set()
    for consumer in nodes:
        for value_id in set(consumer.input_values):
            producer_id = producer_by_value.get(value_id)
            if producer_id is None:
                continue
            if nodes_by_id[producer_id].cta_id == consumer.cta_id:
                local_consumer_producers.add(producer_id)

    final_incoming_by_node: dict[int, TransferSpec] = {}
    alias_transition_by_node: dict[int, int] = {}
    for node_id, incoming_transfers in incoming_by_node.items():
        node = nodes_by_id[node_id]
        alias_transfer = None
        if node.output_alias_input_index is not None:
            alias_value = node.input_values[node.output_alias_input_index]
            alias_transfer = next(
                (transfer for transfer in incoming_transfers if transfer.value_id == alias_value),
                None,
            )
        final_transfer = alias_transfer if alias_transfer is not None and node_id in outgoing_by_node else incoming_transfers[-1]
        final_incoming_by_node[node_id] = final_transfer
        if alias_transfer is final_transfer:
            alias_transition_by_node[node_id] = final_transfer.transfer_id

    # Remote inputs may stay resident in one permanent communicate lane and any
    # number of explicitly provisioned transient lanes.  This is a physical
    # storage/lifetime contract, not an incidental consequence of list order.
    resident_storage_by_transfer: dict[int, _StorageResource] = {}
    resident_incoming_by_node: dict[int, tuple[TransferSpec, ...]] = {}
    for node_id, incoming_transfers in incoming_by_node.items():
        final_transfer = final_incoming_by_node[node_id]
        resident = [final_transfer]
        earlier_transfers = [transfer for transfer in incoming_transfers if transfer.transfer_id != final_transfer.transfer_id]
        transient_lane_bytes = max(
            (int(value_nbytes.get(transfer.value_id, 0)) for transfer in earlier_transfers),
            default=0,
        )
        transient_lane_count = 0 if transient_lane_bytes <= 0 else nodes_by_id[node_id].transient_prefetch_bytes // transient_lane_bytes
        for storage_lane, transient_transfer in enumerate(earlier_transfers[:transient_lane_count]):
            resident.append(transient_transfer)
            resident_storage_by_transfer[transient_transfer.transfer_id] = (
                DataflowCommStorageKind.TRANSIENT_PREFETCH,
                storage_lane,
            )
        if node_id in alias_transition_by_node:
            resident_storage_by_transfer[final_transfer.transfer_id] = (
                DataflowCommStorageKind.PERMANENT,
                0,
            )
        if len(resident) > 1:
            diagnostics.append(
                f"node {node_id} keeps {len(resident)} remote inputs resident in "
                f"one permanent and {len(resident) - 1} transient storage lanes"
            )
        resident_incoming_by_node[node_id] = tuple(sorted(resident, key=lambda item: item.consumer_input_index))
        if len(incoming_transfers) > len(resident):
            diagnostics.append(
                f"node {node_id} has {len(incoming_transfers)} remote inputs; "
                "serialize excess inbox epochs and retain them in ordinary storage"
            )

    source_slot_candidates: set[int] = set()
    for node_id, _outgoing in outgoing_by_node.items():
        node = nodes_by_id[node_id]
        incoming = incoming_by_node.get(node_id, ())
        if node_id in local_consumer_producers:
            diagnostics.append(f"node {node_id} also has local consumers; retain its outgoing value in ordinary storage")
            continue
        if fixed_position_by_node and any(
            nodes_by_id[later_node_id].cta_id == node.cta_id and fixed_position_by_node[later_node_id] > fixed_position_by_node[node_id]
            for later_node_id in incoming_by_node
        ):
            diagnostics.append(
                f"node {node_id} has a later fixed-queue inbox on CTA {node.cta_id}; retain ordinary source storage across its transfer"
            )
            continue
        if not incoming:
            source_slot_candidates.add(node_id)
            continue
        final_transfer = final_incoming_by_node[node_id]
        if alias_transition_by_node.get(node_id) == final_transfer.transfer_id:
            source_slot_candidates.add(node_id)
            continue
        diagnostics.append(
            f"node {node_id} cannot alias its final inbox into output value {node.output_value}; retain ordinary source storage"
        )

    transfer_by_id = {transfer.transfer_id: transfer for transfer in transfers}
    transfer_by_value_consumer = {(transfer.value_id, transfer.consumer_node_id): transfer for transfer in transfers}
    segments_by_transfer: dict[int, tuple[tuple[int, int, int], ...]] = {}
    cluster_key_by_transfer: dict[int, tuple[str, int]] = {}
    store_keys_by_transfer: dict[int, tuple[tuple[str, int], ...]] = {}
    load_keys_by_transfer: dict[int, tuple[tuple[str, int], ...]] = {}
    retain_copy_key_by_transfer: dict[int, tuple[str, int]] = {}
    next_transport_operation_id = 0
    for transfer in transfers:
        byte_count = int(value_nbytes.get(transfer.value_id, 0))
        if transfer.kind is DataflowTransportKind.HBM_STAGED and hbm_segment_bytes is not None and byte_count > 0:
            descriptors = tuple(
                (
                    segment_id,
                    byte_offset,
                    min(hbm_segment_bytes, byte_count - byte_offset),
                )
                for segment_id, byte_offset in enumerate(range(0, byte_count, hbm_segment_bytes))
            )
        else:
            descriptors = ((0, 0, byte_count),)
        segments_by_transfer[transfer.transfer_id] = descriptors
        if transfer.kind is DataflowTransportKind.CLUSTER_PUSH:
            cluster_key_by_transfer[transfer.transfer_id] = (
                "cluster_push",
                next_transport_operation_id,
            )
            next_transport_operation_id += 1
            continue
        store_keys: list[tuple[str, int]] = []
        load_keys: list[tuple[str, int]] = []
        for _segment_id, _offset, _count in descriptors:
            store_keys.append(("hbm_store", next_transport_operation_id))
            next_transport_operation_id += 1
            load_keys.append(("hbm_load", next_transport_operation_id))
            next_transport_operation_id += 1
        store_keys_by_transfer[transfer.transfer_id] = tuple(store_keys)
        load_keys_by_transfer[transfer.transfer_id] = tuple(load_keys)

    for node_id, incoming_transfers in incoming_by_node.items():
        resident_transfer_ids = {transfer.transfer_id for transfer in resident_incoming_by_node[node_id]}
        for transfer in incoming_transfers:
            if transfer.transfer_id in resident_transfer_ids:
                continue
            retain_copy_key_by_transfer[transfer.transfer_id] = (
                "retain_copy",
                next_transport_operation_id,
            )
            next_transport_operation_id += 1

    arrival_gate_by_transfer: dict[int, tuple[str, int]] = {}
    for node_id, incoming_transfers in incoming_by_node.items():
        final_transfer = final_incoming_by_node[node_id]
        receive_order = tuple(transfer for transfer in incoming_transfers if transfer.transfer_id != final_transfer.transfer_id) + (
            final_transfer,
        )
        previous_copy_key = None
        for transfer in receive_order:
            if previous_copy_key is not None:
                arrival_gate_by_transfer[transfer.transfer_id] = previous_copy_key
            previous_copy_key = retain_copy_key_by_transfer.get(transfer.transfer_id)

    def destination_operation_keys(
        transfer: TransferSpec,
    ) -> tuple[tuple[str, int], ...]:
        if transfer.kind is DataflowTransportKind.CLUSTER_PUSH:
            return (cluster_key_by_transfer[transfer.transfer_id],)
        return load_keys_by_transfer[transfer.transfer_id]

    def transfer_prefetch_capacity(transfer: TransferSpec) -> int:
        return sum(byte_count for _segment_id, _byte_offset, byte_count in segments_by_transfer[transfer.transfer_id])

    def fixed_interval_supports_transient_prefetch(
        transfer: TransferSpec,
        *,
        storage_lane: int,
        after_position: int,
    ) -> bool:
        required_bytes = transfer_prefetch_capacity(transfer)
        if required_bytes <= 0:
            return False
        required_capacity = (storage_lane + 1) * required_bytes
        target_position = fixed_position_by_node.get(transfer.consumer_node_id)
        if target_position is None:
            return nodes_by_id[transfer.consumer_node_id].transient_prefetch_bytes >= required_capacity
        return all(
            node.transient_prefetch_bytes >= required_capacity
            for node in nodes
            if node.cta_id == transfer.consumer_cta
            and after_position <= fixed_position_by_node.get(node.node_id, target_position + 1) <= target_position
        )

    def fixed_interval_transient_resources(
        transfer: TransferSpec,
        *,
        after_position: int,
    ) -> tuple[_StorageResource, ...]:
        required_bytes = transfer_prefetch_capacity(transfer)
        if required_bytes <= 0:
            return ()
        maximum_lanes = nodes_by_id[transfer.consumer_node_id].transient_prefetch_bytes // required_bytes
        return tuple(
            (DataflowCommStorageKind.TRANSIENT_PREFETCH, storage_lane)
            for storage_lane in range(maximum_lanes)
            if fixed_interval_supports_transient_prefetch(
                transfer,
                storage_lane=storage_lane,
                after_position=after_position,
            )
        )

    priority_hbm_transfer_ids: frozenset[int] = frozenset()
    if critical_hbm_permanent_prefetch and fixed_queue_node_ids is not None:
        hbm_transfers = tuple(transfer for transfer in transfers if transfer.kind is DataflowTransportKind.HBM_STAGED)

        def critical_completion_us(transfer: TransferSpec) -> float:
            consumer_queue = tuple(fixed_queue_node_ids.get(transfer.consumer_cta, ()))
            producer_queue = tuple(fixed_queue_node_ids.get(transfer.producer_cta, ()))
            consumer_index = consumer_queue.index(transfer.consumer_node_id)
            producer_index = producer_queue.index(transfer.producer_node_id)
            consumer_prefix = sum(nodes_by_id[node_id].duration_us for node_id in consumer_queue[:consumer_index])
            producer_ready = sum(nodes_by_id[node_id].duration_us for node_id in producer_queue[: producer_index + 1])
            consumer_suffix = sum(nodes_by_id[node_id].duration_us for node_id in consumer_queue[consumer_index:])
            return (
                max(
                    consumer_prefix,
                    producer_ready + float(hbm_transfer_us),
                )
                + consumer_suffix
            )

        if hbm_transfers:
            critical_transfer = max(
                hbm_transfers,
                key=lambda transfer: (
                    critical_completion_us(transfer),
                    -transfer.transfer_id,
                ),
            )
            priority_hbm_transfer_ids = frozenset({critical_transfer.transfer_id})
            diagnostics.append(f"reserve the permanent inbox for critical HBM transfer {critical_transfer.transfer_id}")

    prior_destination_gate_by_transfer: dict[int, tuple[str, int]] = {}
    if fixed_position_by_node:
        incoming_by_cta: dict[int, list[TransferSpec]] = {}
        for transfer in transfers:
            incoming_by_cta.setdefault(transfer.consumer_cta, []).append(transfer)
        for incoming_transfers in incoming_by_cta.values():
            ordered = sorted(
                incoming_transfers,
                key=lambda item: (
                    fixed_position_by_node[item.consumer_node_id],
                    item.consumer_input_index,
                    item.transfer_id,
                ),
            )
            previous = None
            previous_storage = None
            next_hbm_index = [None] * len(ordered)
            nearest_hbm = None
            for index in range(len(ordered) - 1, -1, -1):
                if ordered[index].transfer_id in priority_hbm_transfer_ids:
                    nearest_hbm = index
                next_hbm_index[index] = nearest_hbm
            for transfer_index, transfer in enumerate(ordered):
                assigned_storage = resident_storage_by_transfer.get(transfer.transfer_id)
                upcoming_hbm_index = next_hbm_index[transfer_index]
                reserve_permanent_for_hbm = (
                    critical_hbm_permanent_prefetch
                    and upcoming_hbm_index is not None
                    and transfer_index < upcoming_hbm_index
                    and previous is not None
                )
                if assigned_storage is None and cluster_async_receive_pipeline and transfer.kind is DataflowTransportKind.CLUSTER_PUSH:
                    assigned_storage = (DataflowCommStorageKind.PERMANENT, 0)
                    if (
                        previous is not None
                        and (previous_storage == (DataflowCommStorageKind.PERMANENT, 0) or reserve_permanent_for_hbm)
                        and alias_transition_by_node.get(transfer.consumer_node_id) != transfer.transfer_id
                    ):
                        transient_resources = fixed_interval_transient_resources(
                            transfer,
                            after_position=fixed_position_by_node[previous.consumer_node_id],
                        )
                        if transient_resources:
                            assigned_storage = transient_resources[0]
                    resident_storage_by_transfer[transfer.transfer_id] = assigned_storage
                if assigned_storage is None and critical_hbm_permanent_prefetch and transfer.transfer_id in priority_hbm_transfer_ids:
                    assigned_storage = (
                        DataflowCommStorageKind.PERMANENT,
                        0,
                    )
                    resident_storage_by_transfer[transfer.transfer_id] = assigned_storage
                if previous is not None:
                    previous_resource = resident_storage_by_transfer.get(previous.transfer_id)
                    current_resource = resident_storage_by_transfer.get(transfer.transfer_id)
                    # Fixed-queue arrivals may overlap only when planning has
                    # proved that they occupy distinct physical resources.  If
                    # either resource is still dynamic, retain arrival order;
                    # the feasibility check will then wait for an occupied
                    # resource to be released.  This protects an earlier
                    # consumer reservation without delaying the first arrival
                    # behind unrelated compute on the destination CTA.
                    if not (previous_resource is not None and current_resource is not None and previous_resource != current_resource):
                        same_permanent_resource = (
                            previous_resource == (DataflowCommStorageKind.PERMANENT, 0) and current_resource == previous_resource
                        )
                        prior_destination_gate_by_transfer[transfer.transfer_id] = (
                            ("handler", previous.consumer_node_id)
                            if serialize_permanent_inbox_reuse
                            and same_permanent_resource
                            and previous.consumer_node_id != transfer.consumer_node_id
                            else destination_operation_keys(previous)[-1]
                        )
                previous = transfer
                previous_storage = assigned_storage

    operations: dict[tuple[str, int], Operation] = {}

    for transfer in transfers:
        producer_key = ("handler", transfer.producer_node_id)
        arrival_gate = arrival_gate_by_transfer.get(transfer.transfer_id)
        if transfer.kind is DataflowTransportKind.CLUSTER_PUSH:
            key = cluster_key_by_transfer[transfer.transfer_id]
            predecessors = [producer_key]
            prior_destination_gate = prior_destination_gate_by_transfer.get(transfer.transfer_id)
            if prior_destination_gate is not None:
                # Once the preceding arrival owns the permanent inbox, a
                # following push may fill transient reduce scratch and overlap
                # the preceding consumer.  Physical lowering adds the
                # consumer-side readiness/reuse handshake for that scratch.
                predecessors.append(prior_destination_gate)
            if resident_storage_by_transfer.get(transfer.transfer_id, (None, 0))[0] is DataflowCommStorageKind.TRANSIENT_PREFETCH:
                consumer_queue_predecessor = fixed_predecessor_by_node.get(transfer.consumer_node_id)
                if consumer_queue_predecessor is not None:
                    predecessors.append(("handler", consumer_queue_predecessor))
            if arrival_gate is not None:
                predecessors.append(arrival_gate)
            operations[key] = Operation(
                key=key,
                cta_id=transfer.producer_cta,
                cta_duration_us=float(transfer_issue_us),
                completion_delay_us=float(transfer_issue_us) + float(cluster_transfer_us),
                predecessors=tuple(predecessors),
                node_id=None,
                transfer_id=transfer.transfer_id,
                segment_id=0,
                sort_key=(transfer.consumer_node_id, transfer.consumer_input_index, 1, 0),
            )
            continue
        descriptors = segments_by_transfer[transfer.transfer_id]
        total_bytes = sum(byte_count for _segment_id, _offset, byte_count in descriptors)
        previous_store_key = None
        previous_load_key = None
        for descriptor, store_key, load_key in zip(
            descriptors,
            store_keys_by_transfer[transfer.transfer_id],
            load_keys_by_transfer[transfer.transfer_id],
        ):
            segment_id, _offset, byte_count = descriptor
            byte_fraction = byte_count / total_bytes if total_bytes > 0 else 1.0 / len(descriptors)
            segment_half_us = float(hbm_transfer_us) * 0.5 * byte_fraction
            store_predecessors = [producer_key]
            if previous_store_key is not None:
                store_predecessors.append(previous_store_key)
            operations[store_key] = Operation(
                key=store_key,
                cta_id=transfer.producer_cta,
                cta_duration_us=float(transfer_issue_us),
                completion_delay_us=float(transfer_issue_us) + segment_half_us,
                predecessors=tuple(store_predecessors),
                node_id=None,
                transfer_id=transfer.transfer_id,
                segment_id=segment_id,
                sort_key=(
                    transfer.consumer_node_id,
                    transfer.consumer_input_index,
                    0,
                    segment_id,
                ),
            )
            load_predecessors = [store_key]
            if previous_load_key is not None:
                load_predecessors.append(previous_load_key)
            else:
                prior_destination_gate = prior_destination_gate_by_transfer.get(transfer.transfer_id)
                if prior_destination_gate is not None:
                    # The next HBM arrival may occupy transient scratch while
                    # the previous inbox is consumed.  Preserve arrival order
                    # without forcing it behind the previous handler.
                    load_predecessors.append(prior_destination_gate)
                if arrival_gate is not None:
                    load_predecessors.append(arrival_gate)
                if (
                    resident_storage_by_transfer.get(
                        transfer.transfer_id,
                        (None, 0),
                    )[0]
                    is DataflowCommStorageKind.TRANSIENT_PREFETCH
                ):
                    consumer_queue_predecessor = fixed_predecessor_by_node.get(transfer.consumer_node_id)
                    if consumer_queue_predecessor is not None:
                        load_predecessors.append(("handler", consumer_queue_predecessor))
            operations[load_key] = Operation(
                key=load_key,
                cta_id=transfer.consumer_cta,
                cta_duration_us=float(transfer_issue_us),
                completion_delay_us=float(transfer_issue_us) + segment_half_us,
                predecessors=tuple(load_predecessors),
                node_id=None,
                transfer_id=transfer.transfer_id,
                segment_id=segment_id,
                sort_key=(
                    transfer.consumer_node_id,
                    transfer.consumer_input_index,
                    1,
                    segment_id,
                ),
            )
            previous_store_key = store_key
            previous_load_key = load_key

    destination_keys_by_transfer: dict[int, tuple[tuple[str, int], ...]] = {}
    producer_keys_by_node: dict[int, list[tuple[str, int]]] = {}
    for transfer in transfers:
        if transfer.kind is DataflowTransportKind.CLUSTER_PUSH:
            producer_keys = (cluster_key_by_transfer[transfer.transfer_id],)
            destination_keys = producer_keys
        else:
            producer_keys = store_keys_by_transfer[transfer.transfer_id]
            destination_keys = load_keys_by_transfer[transfer.transfer_id]
        producer_keys_by_node.setdefault(transfer.producer_node_id, []).extend(producer_keys)
        destination_keys_by_transfer[transfer.transfer_id] = destination_keys

    for transfer_id, copy_key in retain_copy_key_by_transfer.items():
        transfer = transfer_by_id[transfer_id]
        operations[copy_key] = Operation(
            key=copy_key,
            cta_id=transfer.consumer_cta,
            cta_duration_us=float(retained_copy_us),
            completion_delay_us=float(retained_copy_us),
            predecessors=destination_keys_by_transfer[transfer_id],
            node_id=None,
            transfer_id=transfer_id,
            segment_id=None,
            sort_key=(transfer.consumer_node_id, transfer.consumer_input_index, 2, 0),
        )

    for node in nodes:
        predecessors: list[tuple[str, int]] = []
        fixed_predecessor = fixed_predecessor_by_node.get(node.node_id)
        if fixed_predecessor is not None:
            predecessors.append(("handler", fixed_predecessor))
            if fixed_predecessor not in source_slot_candidates:
                predecessors.extend(producer_keys_by_node.get(fixed_predecessor, ()))
        for value_id in dict.fromkeys(node.input_values):
            producer_id = producer_by_value.get(value_id)
            if producer_id is None:
                continue
            producer = nodes_by_id[producer_id]
            if producer.cta_id == node.cta_id:
                predecessors.append(("handler", producer_id))
                continue
            transfer = transfer_by_value_consumer[(value_id, node.node_id)]
            copy_key = retain_copy_key_by_transfer.get(transfer.transfer_id)
            if copy_key is not None:
                predecessors.append(copy_key)
            else:
                predecessors.extend(destination_keys_by_transfer[transfer.transfer_id])
        operations[("handler", node.node_id)] = Operation(
            key=("handler", node.node_id),
            cta_id=node.cta_id,
            cta_duration_us=node.duration_us,
            completion_delay_us=node.duration_us,
            predecessors=tuple(dict.fromkeys(predecessors)),
            node_id=node.node_id,
            transfer_id=None,
            segment_id=None,
            sort_key=node.sort_key + (node.node_id,),
        )

    successors: dict[tuple[str, int], list[tuple[str, int]]] = {key: [] for key in operations}
    for operation in operations.values():
        for predecessor in operation.predecessors:
            if predecessor not in operations:
                raise DataflowJointScheduleError(f"joint operation {operation.key!r} has unknown predecessor {predecessor!r}")
            successors[predecessor].append(operation.key)

    rank_cache: dict[tuple[str, int], float] = {}
    rank_stack: set[tuple[str, int]] = set()

    def upward_rank(key: tuple[str, int]) -> float:
        cached = rank_cache.get(key)
        if cached is not None:
            return cached
        if key in rank_stack:
            raise DataflowJointScheduleError("joint operation dependency graph contains a cycle")
        rank_stack.add(key)
        operation = operations[key]
        rank = operation.completion_delay_us + max(
            (upward_rank(successor) for successor in successors[key]),
            default=0.0,
        )
        rank_stack.remove(key)
        rank_cache[key] = rank
        return rank

    for key in operations:
        upward_rank(key)

    def source_slot_cta_cycle() -> tuple[int, ...] | None:
        adjacency: dict[int, set[int]] = {}
        for node_id in source_slot_candidates:
            producer_cta = nodes_by_id[node_id].cta_id
            for transfer in outgoing_by_node[node_id]:
                adjacency.setdefault(producer_cta, set()).add(transfer.consumer_cta)
        state: dict[int, int] = {}
        stack: list[int] = []

        def visit(cta_id: int) -> tuple[int, ...] | None:
            state[cta_id] = 1
            stack.append(cta_id)
            for consumer_cta in sorted(adjacency.get(cta_id, ())):
                if state.get(consumer_cta, 0) == 0:
                    cycle = visit(consumer_cta)
                    if cycle is not None:
                        return cycle
                elif state.get(consumer_cta) == 1:
                    begin = stack.index(consumer_cta)
                    return tuple(stack[begin:])
            stack.pop()
            state[cta_id] = 2
            return None

        for cta_id in sorted(adjacency):
            if state.get(cta_id, 0) == 0:
                cycle = visit(cta_id)
                if cycle is not None:
                    return cycle
        return None

    while True:
        cycle = source_slot_cta_cycle()
        if cycle is None:
            break
        cycle_ctas = set(cycle)
        conflicting_nodes = [
            node_id
            for node_id in source_slot_candidates
            if nodes_by_id[node_id].cta_id in cycle_ctas
            and any(transfer.consumer_cta in cycle_ctas for transfer in outgoing_by_node[node_id])
        ]
        if not conflicting_nodes:
            break
        retained_node_id = max(conflicting_nodes)
        source_slot_candidates.remove(retained_node_id)
        diagnostics.append(
            f"source-slot placement across CTAs {sorted(cycle_ctas)} forms a capacity "
            f"wait cycle; node {retained_node_id} retains ordinary source storage"
        )

    events: list[DataflowScheduleEvent] = []
    operation_event: dict[tuple[str, int], int] = {}
    operation_completion_event: dict[tuple[str, int], int] = {}
    operation_completion_us: dict[tuple[str, int], float] = {}
    cta_ready = {cta_id: 0.0 for cta_id in range(topology.sm_count)}
    queue_nodes: dict[int, list[int]] = {cta_id: [] for cta_id in range(topology.sm_count)}
    storage_resources: set[_CTAStorageResource] = {(cta_id, DataflowCommStorageKind.PERMANENT, 0) for cta_id in range(topology.sm_count)}
    for transfer in transfers:
        required_bytes = transfer_prefetch_capacity(transfer)
        if required_bytes <= 0:
            continue
        transient_lane_count = nodes_by_id[transfer.consumer_node_id].transient_prefetch_bytes // required_bytes
        storage_resources.update(
            (
                transfer.consumer_cta,
                DataflowCommStorageKind.TRANSIENT_PREFETCH,
                storage_lane,
            )
            for storage_lane in range(transient_lane_count)
        )
    storage_resources = tuple(
        sorted(
            storage_resources,
            key=lambda resource: (
                resource[0],
                resource[1].value,
                resource[2],
            ),
        )
    )
    slot_open: dict[_CTAStorageResource, OpenSlotInterval | None] = {resource: None for resource in storage_resources}
    slot_free_us = {resource: 0.0 for resource in storage_resources}
    slot_next_epoch = {resource: 1 for resource in storage_resources}
    slot_intervals: list[DataflowCommSlotInterval] = []
    producer_epoch_by_node: dict[int, int] = {}
    consumer_epoch_by_transfer: dict[int, int] = {}
    consumer_storage_by_transfer: dict[int, _StorageResource] = {}
    retained_copy_event_by_transfer: dict[int, int] = {}
    dynamic_handler_predecessor_events: dict[int, list[int]] = {}
    dynamic_handler_ready_us: dict[int, float] = {}
    destination_ready_event_by_transfer: dict[int, int] = {}
    source_release_event_by_node: dict[int, int] = {}
    source_pending_by_node = {node_id: set(keys) for node_id, keys in producer_keys_by_node.items()}
    producer_wait_by_transfer: dict[int, float] = {transfer.transfer_id: 0.0 for transfer in transfers}
    consumer_wait_by_transfer: dict[int, float] = {transfer.transfer_id: 0.0 for transfer in transfers}

    def append_event(
        *,
        kind: DataflowScheduleEventKind,
        cta_id: int | None,
        node_id: int | None,
        transfer_id: int | None,
        predecessor_event_ids: tuple[int, ...],
        start_us: float,
        end_us: float,
    ) -> int:
        event_id = len(events)
        events.append(
            DataflowScheduleEvent(
                event_id=event_id,
                kind=kind,
                cta_id=cta_id,
                node_id=node_id,
                transfer_id=transfer_id,
                predecessor_event_ids=predecessor_event_ids,
                start_us=start_us,
                end_us=end_us,
            )
        )
        return event_id

    def open_slot(
        *,
        cta_id: int,
        storage_kind: DataflowCommStorageKind = DataflowCommStorageKind.PERMANENT,
        storage_lane: int = 0,
        mode: DataflowCommSlotMode,
        value_id: int,
        transfer_id: int | None,
        begin_event_id: int,
        begin_us: float,
    ) -> int:
        resource = (cta_id, storage_kind, storage_lane)
        if slot_open[resource] is not None or begin_us + 1e-9 < slot_free_us[resource]:
            raise DataflowJointScheduleError(f"CTA {cta_id} {storage_kind.value}/{storage_lane} storage is not free at {begin_us:.6f} us")
        epoch = slot_next_epoch[resource]
        slot_next_epoch[resource] += 1
        slot_open[resource] = OpenSlotInterval(
            cta_id=cta_id,
            epoch=epoch,
            storage_kind=storage_kind,
            storage_lane=storage_lane,
            mode=mode,
            value_id=value_id,
            transfer_id=transfer_id,
            begin_event_id=begin_event_id,
            begin_us=begin_us,
        )
        return epoch

    def close_slot(
        *,
        cta_id: int,
        storage_kind: DataflowCommStorageKind = DataflowCommStorageKind.PERMANENT,
        storage_lane: int = 0,
        end_event_id: int,
        end_us: float,
    ) -> OpenSlotInterval:
        resource = (cta_id, storage_kind, storage_lane)
        opened = slot_open[resource]
        if opened is None:
            raise DataflowJointScheduleError(f"CTA {cta_id} {storage_kind.value}/{storage_lane} storage is not occupied")
        if end_us + 1e-9 < opened.begin_us:
            raise DataflowJointScheduleError(f"CTA {cta_id} communicate slot closes before it opens")
        slot_intervals.append(
            DataflowCommSlotInterval(
                cta_id=cta_id,
                epoch=opened.epoch,
                storage_kind=opened.storage_kind,
                storage_lane=opened.storage_lane,
                mode=opened.mode,
                value_id=opened.value_id,
                transfer_id=opened.transfer_id,
                begin_event_id=opened.begin_event_id,
                end_event_id=end_event_id,
                begin_us=opened.begin_us,
                end_us=end_us,
            )
        )
        slot_open[resource] = None
        slot_free_us[resource] = end_us
        return opened

    unscheduled = set(operations)

    def predecessor_ready(operation: Operation) -> bool:
        return all(key in operation_completion_us for key in operation.predecessors)

    def base_start(operation: Operation) -> float:
        return max(
            cta_ready[operation.cta_id],
            max(
                (operation_completion_us[key] for key in operation.predecessors),
                default=0.0,
            ),
            (dynamic_handler_ready_us.get(operation.node_id, 0.0) if operation.node_id is not None else 0.0),
        )

    def pending_earlier_reservations(
        transfer: TransferSpec,
        resource: _CTAStorageResource,
    ) -> int:
        """Count not-yet-arrived fixed-queue values preferring this resource."""

        current_position = fixed_position_by_node.get(
            transfer.consumer_node_id,
            math.inf,
        )
        storage_resource = (resource[1], resource[2])
        return sum(
            1
            for other_transfer_id, preferred in resident_storage_by_transfer.items()
            if preferred == storage_resource
            and other_transfer_id != transfer.transfer_id
            and (other_transfer := transfer_by_id[other_transfer_id]).consumer_cta == transfer.consumer_cta
            and fixed_position_by_node.get(
                other_transfer.consumer_node_id,
                math.inf,
            )
            <= current_position
            and not all(key in operation_completion_event for key in destination_keys_by_transfer[other_transfer_id])
        )

    def feasible_start(
        operation: Operation,
    ) -> tuple[float, _StorageResource | None] | None:
        start = base_start(operation)
        kind, item_id = operation.key
        if kind == "handler":
            node = nodes_by_id[item_id]
            for resource, transient_open in slot_open.items():
                resource_cta, storage_kind, storage_lane = resource
                if resource_cta != node.cta_id or storage_kind is not DataflowCommStorageKind.TRANSIENT_PREFETCH or transient_open is None:
                    continue
                assert transient_open.transfer_id is not None
                transient_transfer = transfer_by_id[transient_open.transfer_id]
                if node.transient_prefetch_bytes < (storage_lane + 1) * transfer_prefetch_capacity(transient_transfer):
                    return None
            resident_incoming = tuple(
                transfer
                for transfer in resident_incoming_by_node.get(item_id, ())
                if transfer.transfer_id not in retained_copy_event_by_transfer
            )
            for incoming_transfer in resident_incoming:
                storage_resource = consumer_storage_by_transfer.get(incoming_transfer.transfer_id)
                if storage_resource is None:
                    return None
                opened = slot_open[(node.cta_id, *storage_resource)]
                if opened is None or opened.mode is not DataflowCommSlotMode.INBOX or opened.transfer_id != incoming_transfer.transfer_id:
                    return None
            if resident_incoming:
                final_incoming = final_incoming_by_node[item_id]
                return start, consumer_storage_by_transfer.get(final_incoming.transfer_id)
            if item_id in source_slot_candidates:
                resource = (
                    node.cta_id,
                    DataflowCommStorageKind.PERMANENT,
                    0,
                )
                opened = slot_open[resource]
                if opened is not None:
                    return None
                return max(start, slot_free_us[resource]), None
            return start, None
        if operation.transfer_id is None:
            raise DataflowJointScheduleError(f"joint transport operation {operation.key!r} has no transfer")
        transfer = transfer_by_id[operation.transfer_id]
        if kind in {"cluster_push", "hbm_store"} and transfer.producer_node_id in source_slot_candidates:
            opened = slot_open[
                (
                    transfer.producer_cta,
                    DataflowCommStorageKind.PERMANENT,
                    0,
                )
            ]
            if opened is None or opened.mode is not DataflowCommSlotMode.OUTBOX or opened.value_id != transfer.value_id:
                return None
        if kind == "cluster_push":
            preferred_resource = resident_storage_by_transfer.get(transfer.transfer_id)
            permanent_resource = (
                transfer.consumer_cta,
                DataflowCommStorageKind.PERMANENT,
                0,
            )
            if (
                preferred_resource
                in {
                    None,
                    (DataflowCommStorageKind.PERMANENT, 0),
                }
                and slot_open[permanent_resource] is None
            ):
                return (
                    max(start, slot_free_us[permanent_resource]),
                    (DataflowCommStorageKind.PERMANENT, 0),
                )
            if preferred_resource == (DataflowCommStorageKind.PERMANENT, 0):
                return None
            if (preferred_resource is None and transfer.transfer_id not in prior_destination_gate_by_transfer) or (
                alias_transition_by_node.get(transfer.consumer_node_id) == transfer.transfer_id
            ):
                return None
            required_bytes = transfer_prefetch_capacity(transfer)
            transient_lane_count = (
                0 if required_bytes <= 0 else nodes_by_id[transfer.consumer_node_id].transient_prefetch_bytes // required_bytes
            )
            transient_resources = tuple(
                (
                    transfer.consumer_cta,
                    DataflowCommStorageKind.TRANSIENT_PREFETCH,
                    storage_lane,
                )
                for storage_lane in range(transient_lane_count)
                if preferred_resource
                in {
                    None,
                    (DataflowCommStorageKind.TRANSIENT_PREFETCH, storage_lane),
                }
            )
            available = tuple(resource for resource in transient_resources if slot_open[resource] is None)
            if not available:
                return None
            selected = min(
                available,
                key=lambda resource: (
                    pending_earlier_reservations(transfer, resource),
                    max(start, slot_free_us[resource]),
                    resource[2],
                ),
            )
            return max(start, slot_free_us[selected]), (selected[1], selected[2])
        if kind == "hbm_load":
            assigned_resource = consumer_storage_by_transfer.get(transfer.transfer_id)
            if assigned_resource is not None:
                opened = slot_open[(transfer.consumer_cta, *assigned_resource)]
                if opened is None or opened.transfer_id != transfer.transfer_id:
                    return None
                return start, assigned_resource

            preferred_resource = resident_storage_by_transfer.get(transfer.transfer_id)
            permanent_resource = (
                transfer.consumer_cta,
                DataflowCommStorageKind.PERMANENT,
                0,
            )
            if (
                preferred_resource
                in {
                    None,
                    (DataflowCommStorageKind.PERMANENT, 0),
                }
                and slot_open[permanent_resource] is None
            ):
                return (
                    max(start, slot_free_us[permanent_resource]),
                    (DataflowCommStorageKind.PERMANENT, 0),
                )
            if preferred_resource == (DataflowCommStorageKind.PERMANENT, 0):
                return None

            if (preferred_resource is None and transfer.transfer_id not in prior_destination_gate_by_transfer) or (
                alias_transition_by_node.get(transfer.consumer_node_id) == transfer.transfer_id
            ):
                return None
            required_bytes = transfer_prefetch_capacity(transfer)
            transient_lane_count = (
                0 if required_bytes <= 0 else nodes_by_id[transfer.consumer_node_id].transient_prefetch_bytes // required_bytes
            )
            transient_resources = tuple(
                (
                    transfer.consumer_cta,
                    DataflowCommStorageKind.TRANSIENT_PREFETCH,
                    storage_lane,
                )
                for storage_lane in range(transient_lane_count)
                if preferred_resource
                in {
                    None,
                    (DataflowCommStorageKind.TRANSIENT_PREFETCH, storage_lane),
                }
            )
            available = tuple(resource for resource in transient_resources if slot_open[resource] is None)
            if not available:
                return None
            selected = min(
                available,
                key=lambda resource: (
                    pending_earlier_reservations(transfer, resource),
                    max(start, slot_free_us[resource]),
                    resource[2],
                ),
            )
            return max(start, slot_free_us[selected]), (selected[1], selected[2])
        if kind == "retain_copy":
            storage_resource = consumer_storage_by_transfer.get(transfer.transfer_id)
            if storage_resource is None:
                return None
            opened = slot_open[(transfer.consumer_cta, *storage_resource)]
            if opened is None or opened.mode is not DataflowCommSlotMode.INBOX or opened.transfer_id != transfer.transfer_id:
                return None
            return start, storage_resource
        return start, None

    while unscheduled:
        candidates: list[tuple[float, Operation, _StorageResource | None]] = []
        for key in sorted(unscheduled):
            operation = operations[key]
            if not predecessor_ready(operation):
                continue
            feasible = feasible_start(operation)
            if feasible is not None:
                start, storage_resource = feasible
                candidates.append((start, operation, storage_resource))
        if not candidates:
            demoted_source_nodes: set[int] = set()
            for resource, opened in slot_open.items():
                cta_id, storage_kind, _storage_lane = resource
                if (
                    storage_kind is not DataflowCommStorageKind.PERMANENT
                    or opened is None
                    or opened.mode is not DataflowCommSlotMode.OUTBOX
                ):
                    continue
                producer_node_id = producer_by_value.get(opened.value_id)
                if producer_node_id not in source_slot_candidates:
                    continue
                source_slot_candidates.remove(producer_node_id)
                producer_epoch_by_node.pop(producer_node_id, None)
                slot_open[resource] = None
                slot_free_us[resource] = opened.begin_us
                demoted_source_nodes.add(producer_node_id)
            for operation_key in unscheduled:
                if operation_key[0] != "handler":
                    continue
                node_id = operation_key[1]
                if node_id not in source_slot_candidates:
                    continue
                node = nodes_by_id[node_id]
                opened = slot_open[(node.cta_id, DataflowCommStorageKind.PERMANENT, 0)]
                if opened is None or opened.mode is not DataflowCommSlotMode.INBOX:
                    continue
                source_slot_candidates.remove(node_id)
                demoted_source_nodes.add(node_id)
            if demoted_source_nodes:
                diagnostics.append(
                    "communicate-slot placement caused a resource wait cycle; "
                    f"nodes {sorted(demoted_source_nodes)} retain ordinary source storage"
                )
                continue
            retained_transfer_ids: list[int] = []
            for resource, opened in tuple(slot_open.items()):
                cta_id, storage_kind, storage_lane = resource
                if (
                    opened is None
                    or opened.mode is not DataflowCommSlotMode.INBOX
                    or opened.transfer_id is None
                    or opened.transfer_id in retained_copy_event_by_transfer
                ):
                    continue
                transfer = transfer_by_id[opened.transfer_id]
                destination_keys = destination_keys_by_transfer[transfer.transfer_id]
                if not all(key in operation_completion_event for key in destination_keys):
                    continue
                ready_us = max(operation_completion_us[key] for key in destination_keys)
                copy_cta_before = cta_ready[cta_id]
                copy_start = max(copy_cta_before, ready_us)
                copy_end = copy_start + float(retained_copy_us)
                copy_event_id = append_event(
                    kind=DataflowScheduleEventKind.INBOX_RETAIN_COPY,
                    cta_id=cta_id,
                    node_id=None,
                    transfer_id=transfer.transfer_id,
                    predecessor_event_ids=tuple(operation_completion_event[key] for key in destination_keys),
                    start_us=copy_start,
                    end_us=copy_end,
                )
                cta_ready[cta_id] = copy_end
                close_slot(
                    cta_id=cta_id,
                    storage_kind=storage_kind,
                    storage_lane=storage_lane,
                    end_event_id=copy_event_id,
                    end_us=copy_end,
                )
                retained_copy_event_by_transfer[transfer.transfer_id] = copy_event_id
                dynamic_handler_predecessor_events.setdefault(
                    transfer.consumer_node_id,
                    [],
                ).append(copy_event_id)
                dynamic_handler_ready_us[transfer.consumer_node_id] = max(
                    dynamic_handler_ready_us.get(transfer.consumer_node_id, 0.0),
                    copy_end,
                )
                source_slot_candidates.discard(transfer.consumer_node_id)
                consumer_wait_by_transfer[transfer.transfer_id] = max(
                    consumer_wait_by_transfer[transfer.transfer_id],
                    max(0.0, ready_us - copy_cta_before),
                )
                retained_transfer_ids.append(transfer.transfer_id)
            if retained_transfer_ids:
                diagnostics.append(
                    "inbox placement caused a resource wait cycle; retain transfers "
                    f"{sorted(retained_transfer_ids)} in ordinary input storage"
                )
                continue
            blocked = sorted(unscheduled)[:8]
            occupied = {
                f"{cta_id}/{storage_kind.value}/{storage_lane}": opened
                for (cta_id, storage_kind, storage_lane), opened in slot_open.items()
                if opened is not None
            }
            raise DataflowJointScheduleError(f"joint scheduler cannot make progress; blocked={blocked!r}, occupied_slots={occupied!r}")
        earliest = min(start for start, _, _ in candidates)
        eligible = [(start, operation, storage_resource) for start, operation, storage_resource in candidates if start <= earliest + 1e-9]
        start, operation, selected_storage_resource = min(
            eligible,
            key=lambda item: (
                0 if item[1].completion_delay_us > item[1].cta_duration_us + 1e-9 else 1,
                -(rank_cache[item[1].key] - item[1].completion_delay_us),
                -rank_cache[item[1].key],
                item[1].sort_key,
                item[1].key,
                ("" if item[2] is None else f"{item[2][0].value}/{item[2][1]}"),
            ),
        )
        cta_before = cta_ready[operation.cta_id]
        predecessor_events = tuple(operation_completion_event[key] for key in operation.predecessors) + tuple(
            dynamic_handler_predecessor_events.get(operation.node_id, ()) if operation.node_id is not None else ()
        )
        issue_end = start + operation.cta_duration_us
        completion_us = start + operation.completion_delay_us
        event_id = append_event(
            kind=operation_kind(operation.key),
            cta_id=operation.cta_id,
            node_id=operation.node_id,
            transfer_id=operation.transfer_id,
            predecessor_event_ids=predecessor_events,
            start_us=start,
            end_us=issue_end,
        )
        completion_kind = resolve_completion_kind(operation.key)
        completion_event_id = event_id
        if completion_kind is not None:
            completion_event_id = append_event(
                kind=completion_kind,
                cta_id=None,
                node_id=None,
                transfer_id=operation.transfer_id,
                predecessor_event_ids=(event_id,),
                start_us=completion_us,
                end_us=completion_us,
            )
        operation_event[operation.key] = event_id
        operation_completion_event[operation.key] = completion_event_id
        operation_completion_us[operation.key] = completion_us
        cta_ready[operation.cta_id] = issue_end
        unscheduled.remove(operation.key)

        kind, item_id = operation.key
        if kind == "handler":
            node = nodes_by_id[item_id]
            queue_nodes[node.cta_id].append(node.node_id)
            resident_incoming = tuple(
                transfer
                for transfer in resident_incoming_by_node.get(node.node_id, ())
                if transfer.transfer_id not in retained_copy_event_by_transfer
            )
            for incoming_transfer in resident_incoming:
                remote_ready = max(operation_completion_us[key] for key in destination_keys_by_transfer[incoming_transfer.transfer_id])
                consumer_wait_by_transfer[incoming_transfer.transfer_id] = max(
                    consumer_wait_by_transfer[incoming_transfer.transfer_id],
                    max(0.0, remote_ready - cta_before),
                )
                storage_resource = consumer_storage_by_transfer[incoming_transfer.transfer_id]
                storage_kind, storage_lane = storage_resource
                opened = slot_open[(node.cta_id, *storage_resource)]
                if opened is None or opened.transfer_id != incoming_transfer.transfer_id:
                    raise DataflowJointScheduleError(
                        f"joint consumer node {node.node_id} does not own transfer "
                        f"{incoming_transfer.transfer_id} in its "
                        f"{storage_kind.value}/{storage_lane} inbox"
                    )
                close_slot(
                    cta_id=node.cta_id,
                    storage_kind=storage_kind,
                    storage_lane=storage_lane,
                    end_event_id=event_id,
                    end_us=completion_us,
                )
            if resident_incoming and node.node_id in source_slot_candidates:
                final_incoming = final_incoming_by_node[node.node_id]
                storage_resource = consumer_storage_by_transfer[final_incoming.transfer_id]
                if storage_resource != (DataflowCommStorageKind.PERMANENT, 0):
                    raise DataflowJointScheduleError(
                        f"joint node {node.node_id} cannot transition transient prefetch storage into a permanent outbox"
                    )
                if alias_transition_by_node.get(node.node_id) != final_incoming.transfer_id:
                    raise DataflowJointScheduleError(f"joint node {node.node_id} uses an unsupported inbox/outbox alias")
                epoch = open_slot(
                    cta_id=node.cta_id,
                    storage_kind=DataflowCommStorageKind.PERMANENT,
                    storage_lane=0,
                    mode=DataflowCommSlotMode.OUTBOX,
                    value_id=node.output_value,
                    transfer_id=None,
                    begin_event_id=event_id,
                    begin_us=completion_us,
                )
                producer_epoch_by_node[node.node_id] = epoch
            elif not resident_incoming and node.node_id in source_slot_candidates:
                epoch = open_slot(
                    cta_id=node.cta_id,
                    storage_kind=DataflowCommStorageKind.PERMANENT,
                    storage_lane=0,
                    mode=DataflowCommSlotMode.OUTBOX,
                    value_id=node.output_value,
                    transfer_id=None,
                    begin_event_id=event_id,
                    begin_us=start,
                )
                producer_epoch_by_node[node.node_id] = epoch
            continue

        if operation.transfer_id is None:
            raise DataflowJointScheduleError(f"joint operation {operation.key!r} lost its transfer identity")
        transfer = transfer_by_id[operation.transfer_id]
        if kind in {"cluster_push", "hbm_store"}:
            if transfer.producer_node_id in source_slot_candidates:
                opened = slot_open[
                    (
                        transfer.producer_cta,
                        DataflowCommStorageKind.PERMANENT,
                        0,
                    )
                ]
                if opened is None or opened.value_id != transfer.value_id:
                    raise DataflowJointScheduleError(f"joint transfer {transfer.transfer_id} does not own its source outbox")
            producer_wait_by_transfer[transfer.transfer_id] = max(
                producer_wait_by_transfer[transfer.transfer_id],
                max(
                    0.0,
                    start
                    - max(
                        cta_before,
                        operation_completion_us[("handler", transfer.producer_node_id)],
                    ),
                ),
            )
            pending = source_pending_by_node[transfer.producer_node_id]
            pending.remove(operation.key)
            if not pending:
                source_keys = producer_keys_by_node[transfer.producer_node_id]
                release_us = max(operation_completion_us[key] for key in source_keys)
                release_predecessors = tuple(operation_completion_event[key] for key in source_keys)
                release_event_id = append_event(
                    kind=DataflowScheduleEventKind.SOURCE_RELEASE,
                    cta_id=None,
                    node_id=transfer.producer_node_id,
                    transfer_id=None,
                    predecessor_event_ids=release_predecessors,
                    start_us=release_us,
                    end_us=release_us,
                )
                source_release_event_by_node[transfer.producer_node_id] = release_event_id
                if transfer.producer_node_id in source_slot_candidates:
                    close_slot(
                        cta_id=transfer.producer_cta,
                        storage_kind=DataflowCommStorageKind.PERMANENT,
                        storage_lane=0,
                        end_event_id=release_event_id,
                        end_us=release_us,
                    )
        if kind == "cluster_push":
            if selected_storage_resource is None:
                raise DataflowJointScheduleError("cluster push selected no destination storage")
            selected_storage_kind, selected_storage_lane = selected_storage_resource
            epoch = open_slot(
                cta_id=transfer.consumer_cta,
                storage_kind=selected_storage_kind,
                storage_lane=selected_storage_lane,
                mode=DataflowCommSlotMode.INBOX,
                value_id=transfer.value_id,
                transfer_id=transfer.transfer_id,
                begin_event_id=event_id,
                begin_us=start,
            )
            consumer_epoch_by_transfer[transfer.transfer_id] = epoch
            consumer_storage_by_transfer[transfer.transfer_id] = selected_storage_resource
            destination_ready_event_by_transfer[transfer.transfer_id] = completion_event_id
        elif kind == "hbm_load":
            consumer_wait_by_transfer[transfer.transfer_id] = max(
                consumer_wait_by_transfer[transfer.transfer_id],
                max(
                    0.0,
                    max(operation_completion_us[key] for key in operation.predecessors) - cta_before,
                ),
            )
            storage_resource = consumer_storage_by_transfer.get(transfer.transfer_id)
            if storage_resource is None:
                if selected_storage_resource is None:
                    raise DataflowJointScheduleError(f"joint HBM transfer {transfer.transfer_id} has no destination storage")
                storage_resource = selected_storage_resource
                storage_kind, storage_lane = storage_resource
                epoch = open_slot(
                    cta_id=transfer.consumer_cta,
                    storage_kind=storage_kind,
                    storage_lane=storage_lane,
                    mode=DataflowCommSlotMode.INBOX,
                    value_id=transfer.value_id,
                    transfer_id=transfer.transfer_id,
                    begin_event_id=event_id,
                    begin_us=start,
                )
                consumer_epoch_by_transfer[transfer.transfer_id] = epoch
                consumer_storage_by_transfer[transfer.transfer_id] = storage_resource
            destination_keys = destination_keys_by_transfer[transfer.transfer_id]
            if all(key in operation_completion_event for key in destination_keys):
                ready_us = max(operation_completion_us[key] for key in destination_keys)
                ready_event_id = append_event(
                    kind=DataflowScheduleEventKind.TRANSFER_READY,
                    cta_id=None,
                    node_id=None,
                    transfer_id=transfer.transfer_id,
                    predecessor_event_ids=tuple(operation_completion_event[key] for key in destination_keys),
                    start_us=ready_us,
                    end_us=ready_us,
                )
                destination_ready_event_by_transfer[transfer.transfer_id] = ready_event_id
        elif kind == "retain_copy":
            ready_us = max(operation_completion_us[key] for key in destination_keys_by_transfer[transfer.transfer_id])
            consumer_wait_by_transfer[transfer.transfer_id] = max(
                consumer_wait_by_transfer[transfer.transfer_id],
                max(0.0, ready_us - cta_before),
            )
            storage_kind, storage_lane = consumer_storage_by_transfer[transfer.transfer_id]
            close_slot(
                cta_id=transfer.consumer_cta,
                storage_kind=storage_kind,
                storage_lane=storage_lane,
                end_event_id=event_id,
                end_us=completion_us,
            )
            retained_copy_event_by_transfer[transfer.transfer_id] = event_id

    if any(opened is not None for opened in slot_open.values()):
        raise DataflowJointScheduleError("joint schedule ended with a live communicate-slot interval")

    transfer_schedules: list[DataflowTransferSchedule] = []
    for transfer in transfers:
        if transfer.kind is DataflowTransportKind.CLUSTER_PUSH:
            producer_key = cluster_key_by_transfer[transfer.transfer_id]
            consumer_issue_event_id = None
            segments: tuple[DataflowTransferSegmentSchedule, ...] = ()
        else:
            store_keys = store_keys_by_transfer[transfer.transfer_id]
            load_keys = load_keys_by_transfer[transfer.transfer_id]
            producer_key = store_keys[0]
            consumer_issue_event_id = operation_event[load_keys[0]]
            segments = tuple(
                DataflowTransferSegmentSchedule(
                    segment_id=segment_id,
                    byte_offset=byte_offset,
                    byte_count=byte_count,
                    store_issue_event_id=operation_event[store_key],
                    store_ready_event_id=operation_completion_event[store_key],
                    load_issue_event_id=operation_event[load_key],
                    load_ready_event_id=operation_completion_event[load_key],
                )
                for (segment_id, byte_offset, byte_count), store_key, load_key in zip(
                    segments_by_transfer[transfer.transfer_id],
                    store_keys,
                    load_keys,
                )
            )
        consume_event_id = operation_event[("handler", transfer.consumer_node_id)]
        retained_copy_event_id = retained_copy_event_by_transfer.get(transfer.transfer_id)
        transfer_schedules.append(
            DataflowTransferSchedule(
                transfer_id=transfer.transfer_id,
                kind=transfer.kind,
                value_id=transfer.value_id,
                producer_node_id=transfer.producer_node_id,
                consumer_node_id=transfer.consumer_node_id,
                consumer_input_index=transfer.consumer_input_index,
                producer_cta=transfer.producer_cta,
                consumer_cta=transfer.consumer_cta,
                producer_issue_event_id=operation_event[producer_key],
                producer_release_event_id=source_release_event_by_node[transfer.producer_node_id],
                consumer_issue_event_id=consumer_issue_event_id,
                destination_ready_event_id=destination_ready_event_by_transfer[transfer.transfer_id],
                consume_event_id=consume_event_id,
                destination_release_event_id=(retained_copy_event_id if retained_copy_event_id is not None else consume_event_id),
                producer_slot_epoch=producer_epoch_by_node.get(transfer.producer_node_id),
                consumer_slot_epoch=consumer_epoch_by_transfer[transfer.transfer_id],
                consumer_storage_kind=consumer_storage_by_transfer[transfer.transfer_id][0],
                consumer_storage_lane=consumer_storage_by_transfer[transfer.transfer_id][1],
                segments=segments,
                hbm_stage_id=(None if transfer.kind is DataflowTransportKind.CLUSTER_PUSH else transfer.transfer_id),
                retained_input_copy_event_id=retained_copy_event_id,
                modeled_producer_guard_wait_us=producer_wait_by_transfer[transfer.transfer_id],
                modeled_consumer_guard_wait_us=consumer_wait_by_transfer[transfer.transfer_id],
            )
        )

    guard_waits = [
        wait
        for transfer in transfer_schedules
        for wait in (
            transfer.modeled_producer_guard_wait_us,
            transfer.modeled_consumer_guard_wait_us,
        )
    ]
    result = DataflowJointExecutionSchedule(
        events=tuple(events),
        transfers=tuple(transfer_schedules),
        comm_slot_intervals=tuple(
            sorted(
                slot_intervals,
                key=lambda item: (
                    item.cta_id,
                    item.storage_kind.value,
                    item.storage_lane,
                    item.epoch,
                ),
            )
        ),
        queue_node_ids=tuple((cta_id, tuple(node_ids)) for cta_id, node_ids in sorted(queue_nodes.items())),
        makespan_us=max((event.end_us for event in events), default=0.0),
        critical_path_guard_wait_us=max(guard_waits, default=0.0),
        total_guard_wait_us=sum(guard_waits),
        diagnostics=tuple(diagnostics),
    )
    return result.require_valid(topology=topology, nodes=nodes)


__all__ = [
    "DATAFLOW_JOINT_EXECUTION_PLAN_IMPLEMENTATION",
    "DATAFLOW_JOINT_EXECUTION_PLAN_SCHEMA_VERSION",
    "DataflowCommSlotInterval",
    "DataflowCommSlotMode",
    "DataflowCommStorageKind",
    "DataflowJointExecutionPlan",
    "DataflowJointExecutionSchedule",
    "DataflowJointScheduleError",
    "DataflowJointScheduleNode",
    "DataflowScheduleEvent",
    "DataflowScheduleEventKind",
    "DataflowTransferSchedule",
    "DataflowTransferSegmentSchedule",
    "DataflowTransportKind",
    "schedule_joint_compute_communication",
]
