"""Live-range planning for Dataflow asynchronous receive barriers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import heapq
from typing import Any

from .scheduler import CommPlan, InstructionPlan, DataflowCommKind, SlotPlan


class DataflowBarrierPlanningError(ValueError):
    """Raised when an instruction plan has no legal barrier allocation."""


@dataclass(frozen=True)
class DataflowBarrierUse:
    """One asynchronous receive interval from producer send to consumer wait."""

    transfer_source_instruction_id: int
    transfer_target_instruction_id: int
    send_instruction_id: int
    recv_instruction_id: int
    source_slot_id: int
    target_slot_id: int
    consumer_sm: int
    kind: DataflowCommKind
    original_phase: int
    segment_id: int = 0

    @property
    def transfer_key(self) -> tuple[int, int, int, int, int]:
        return (
            self.transfer_source_instruction_id,
            self.transfer_target_instruction_id,
            self.source_slot_id,
            self.target_slot_id,
            self.segment_id,
        )

    @property
    def transfer_group_key(self) -> tuple[int, int, int, int]:
        """Return the logical transfer shared by all of its byte segments."""

        return self.transfer_key[:4]

    def to_dict(self) -> dict[str, Any]:
        return {
            "transfer_key": list(self.transfer_key),
            "transfer_source_instruction_id": self.transfer_source_instruction_id,
            "transfer_target_instruction_id": self.transfer_target_instruction_id,
            "send_instruction_id": self.send_instruction_id,
            "recv_instruction_id": self.recv_instruction_id,
            "source_slot_id": self.source_slot_id,
            "target_slot_id": self.target_slot_id,
            "consumer_sm": self.consumer_sm,
            "kind": self.kind.value,
            "original_phase": self.original_phase,
            "segment_id": self.segment_id,
        }


@dataclass(frozen=True)
class DataflowBarrierLiveRange:
    """All asynchronous receives bound to one target slot barrier field."""

    live_range_id: int
    consumer_sm: int
    target_slot_id: int
    barrier_lane: int
    uses: tuple[DataflowBarrierUse, ...]

    @property
    def requires_exclusive_barrier(self) -> bool:
        # The ABI carries HBM readiness epochs and mbarrier phases in separate
        # fields.  A physical barrier can therefore be recolored whenever the
        # ordinary live-range proof says the receive intervals do not overlap.
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "live_range_id": self.live_range_id,
            "consumer_sm": self.consumer_sm,
            "target_slot_id": self.target_slot_id,
            "barrier_lane": self.barrier_lane,
            "requires_exclusive_barrier": self.requires_exclusive_barrier,
            "uses": [use.to_dict() for use in self.uses],
        }


@dataclass(frozen=True)
class DataflowBarrierInterferenceGraph:
    """Canonical undirected graph used by the barrier color allocator."""

    node_ids: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if len(self.node_ids) != len(set(self.node_ids)):
            raise ValueError(f"barrier interference graph has duplicate nodes: {self.node_ids!r}")
        nodes = tuple(sorted(self.node_ids))
        node_set = set(nodes)
        normalized_edges: set[tuple[int, int]] = set()
        for first, second in self.edges:
            if first == second:
                raise ValueError(f"barrier interference graph has a self edge for node {first}")
            if first not in node_set or second not in node_set:
                raise ValueError(f"barrier interference graph edge references an unknown node: {(first, second)!r}, nodes={nodes!r}")
            normalized_edges.add((min(first, second), max(first, second)))
        object.__setattr__(self, "node_ids", nodes)
        object.__setattr__(self, "edges", tuple(sorted(normalized_edges)))

    def color(self) -> dict[int, int]:
        """Return a deterministic legal first-fit coloring."""

        adjacency = {node_id: set() for node_id in self.node_ids}
        for first, second in self.edges:
            adjacency[first].add(second)
            adjacency[second].add(first)
        order = sorted(self.node_ids, key=lambda node_id: (-len(adjacency[node_id]), node_id))
        colors: dict[int, int] = {}
        for node_id in order:
            unavailable = {colors[neighbor] for neighbor in adjacency[node_id] if neighbor in colors}
            color = 0
            while color in unavailable:
                color += 1
            colors[node_id] = color
        self.require_legal_coloring(colors)
        return colors

    def require_legal_coloring(self, colors: Mapping[int, int]) -> None:
        if set(colors) != set(self.node_ids):
            raise ValueError(
                f"barrier coloring nodes do not match the interference graph: colors={sorted(colors)!r}, nodes={list(self.node_ids)!r}"
            )
        for node_id, color in colors.items():
            if not isinstance(color, int) or color < 0:
                raise ValueError(f"barrier color for node {node_id} must be a non-negative int, got {color!r}")
        for first, second in self.edges:
            if colors[first] == colors[second]:
                raise ValueError(f"interfering barrier nodes {first} and {second} share color {colors[first]}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_ids": list(self.node_ids),
            "edges": [list(edge) for edge in self.edges],
        }


@dataclass(frozen=True)
class DataflowBarrierUseAssignment:
    use: DataflowBarrierUse
    phase: int

    def to_dict(self) -> dict[str, Any]:
        return {**self.use.to_dict(), "phase": self.phase}


@dataclass(frozen=True)
class DataflowBarrierAssignment:
    live_range_id: int
    consumer_sm: int
    target_slot_id: int
    barrier_lane: int
    barrier_index: int
    uses: tuple[DataflowBarrierUseAssignment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "live_range_id": self.live_range_id,
            "consumer_sm": self.consumer_sm,
            "target_slot_id": self.target_slot_id,
            "barrier_lane": self.barrier_lane,
            "barrier_index": self.barrier_index,
            "uses": [use.to_dict() for use in self.uses],
        }


@dataclass(frozen=True)
class DataflowBarrierAllocation:
    """Structured live ranges, interference graph, colors, and phases."""

    live_ranges: tuple[DataflowBarrierLiveRange, ...]
    interference_graph: DataflowBarrierInterferenceGraph
    assignments: tuple[DataflowBarrierAssignment, ...]
    barrier_count: int

    @property
    def slot_barrier_indices(self) -> dict[int, int]:
        result: dict[int, int] = {}
        # Cluster receives use the slot-level fallback.  HBM segments carry
        # an explicit per-transfer barrier, but lane zero remains the slot
        # fallback so non-segmented plans preserve their compact ABI binding.
        for assignment in sorted(
            self.assignments,
            key=lambda item: (
                item.target_slot_id,
                item.barrier_lane != -1,
                item.barrier_lane,
            ),
        ):
            if assignment.barrier_lane in {-1, 0}:
                result.setdefault(
                    assignment.target_slot_id,
                    assignment.barrier_index,
                )
        return result

    @property
    def transfer_barrier_indices(self) -> dict[tuple[int, int, int, int, int], int]:
        return {
            use_assignment.use.transfer_key: assignment.barrier_index
            for assignment in self.assignments
            for use_assignment in assignment.uses
        }

    @property
    def transfer_phases(self) -> dict[tuple[int, int, int, int, int], int]:
        return {
            use_assignment.use.transfer_key: use_assignment.phase for assignment in self.assignments for use_assignment in assignment.uses
        }

    def assignment_for_slot(
        self,
        slot_id: int,
        *,
        barrier_lane: int | None = None,
    ) -> DataflowBarrierAssignment:
        candidates = tuple(
            assignment
            for assignment in self.assignments
            if assignment.target_slot_id == slot_id and (barrier_lane is None or assignment.barrier_lane == barrier_lane)
        )
        if candidates:
            return min(
                candidates,
                key=lambda item: (
                    item.barrier_lane != -1,
                    item.barrier_lane,
                ),
            )
        raise KeyError(f"Dataflow barrier allocation has no assignment for slot {slot_id}")

    def require_valid(self) -> DataflowBarrierAllocation:
        colors = {assignment.live_range_id: assignment.barrier_index for assignment in self.assignments}
        self.interference_graph.require_legal_coloring(colors)
        expected_count = max(colors.values(), default=-1) + 1
        if self.barrier_count != expected_count:
            raise DataflowBarrierPlanningError(f"barrier allocation count mismatch: {self.barrier_count} != {expected_count}")
        if len(self.assignments) != len(self.live_ranges):
            raise DataflowBarrierPlanningError(
                f"barrier allocation does not assign every live range: {len(self.assignments)} != {len(self.live_ranges)}"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "barrier_count": self.barrier_count,
            "live_ranges": [live_range.to_dict() for live_range in self.live_ranges],
            "interference_graph": self.interference_graph.to_dict(),
            "assignments": [assignment.to_dict() for assignment in self.assignments],
        }


def plan_dataflow_barriers(
    plan: InstructionPlan,
    *,
    hbm_direct_global_slots: frozenset[int] = frozenset(),
) -> DataflowBarrierAllocation:
    """Color asynchronous receive intervals using instruction happens-before."""

    if not isinstance(plan, InstructionPlan):
        raise TypeError(f"plan_dataflow_barriers expects InstructionPlan, got {plan!r}")
    instruction_ids = tuple(instruction.instruction_id for instruction in plan.instructions)
    if len(instruction_ids) != len(set(instruction_ids)):
        raise DataflowBarrierPlanningError("instruction plan contains duplicate instruction ids")
    instructions_by_id = {instruction.instruction_id: instruction for instruction in plan.instructions}
    slots_by_id = {slot.slot_id: slot for slot in plan.slots}
    if len(slots_by_id) != len(plan.slots):
        raise DataflowBarrierPlanningError("instruction plan contains duplicate slot ids")

    send_comms_by_key: dict[tuple[int, int, int, int, int], CommPlan] = {}
    for comm in plan.comms:
        if comm.kind not in (DataflowCommKind.CLUSTER_SEND, DataflowCommKind.HBM_SEND):
            continue
        transfer_key = comm_transfer_key(comm)
        if transfer_key in send_comms_by_key:
            raise DataflowBarrierPlanningError(f"duplicate async send transfer {transfer_key!r}")
        send_comms_by_key[transfer_key] = comm

    grouped_uses: dict[tuple[int, int, int], list[DataflowBarrierUse]] = {}
    seen_transfer_keys: set[tuple[int, int, int, int, int]] = set()
    async_pairs: list[tuple[CommPlan, CommPlan]] = []
    for comm in plan.comms:
        if comm.kind not in (
            DataflowCommKind.CLUSTER_RECV,
            DataflowCommKind.HBM_RECV,
            DataflowCommKind.HBM_RECV_WAIT,
        ):
            continue
        if direct_global_signal_only_comm(comm, slots_by_id, hbm_direct_global_slots):
            continue
        transfer_key = comm_transfer_key(comm)
        send_comm = send_comms_by_key.get(transfer_key)
        expected_send_kind = DataflowCommKind.CLUSTER_SEND if comm.kind is DataflowCommKind.CLUSTER_RECV else DataflowCommKind.HBM_SEND
        if send_comm is None or send_comm.kind is not expected_send_kind:
            raise DataflowBarrierPlanningError(f"{comm.kind.value} has no matching {expected_send_kind.value}: {transfer_key!r}")
        validate_async_pair(send_comm, comm, instructions_by_id, slots_by_id)
        async_pairs.append((send_comm, comm))
        use = DataflowBarrierUse(
            transfer_source_instruction_id=comm.source_instruction_id,
            transfer_target_instruction_id=comm.target_instruction_id,
            send_instruction_id=send_comm.resolved_dispatch_instruction_id,
            recv_instruction_id=comm.resolved_dispatch_instruction_id,
            source_slot_id=comm.source_slot_id,
            target_slot_id=comm.target_slot_id,
            consumer_sm=comm.consumer_sm,
            kind=comm.kind,
            original_phase=comm.barrier_phase,
            segment_id=comm.segment_id,
        )
        if use.transfer_key in seen_transfer_keys:
            raise DataflowBarrierPlanningError(f"duplicate async receive transfer {use.transfer_key!r}")
        seen_transfer_keys.add(use.transfer_key)
        # All byte segments of one HBM transfer contribute to one mbarrier
        # transaction group.  Keep HBM receives in a distinct slot lane from
        # cluster receives, but never create one lane per byte segment.
        barrier_lane = 0 if use.kind in {DataflowCommKind.HBM_RECV, DataflowCommKind.HBM_RECV_WAIT} else -1
        grouped_uses.setdefault(
            (use.consumer_sm, use.target_slot_id, barrier_lane),
            [],
        ).append(use)
    unmatched_cluster_sends = tuple(
        sorted(
            transfer_key
            for transfer_key, comm in send_comms_by_key.items()
            if comm.kind is DataflowCommKind.CLUSTER_SEND and transfer_key not in seen_transfer_keys
        )
    )
    if unmatched_cluster_sends:
        raise DataflowBarrierPlanningError(f"cluster sends have no matching receive intervals: {unmatched_cluster_sends!r}")

    reachable, topological_positions = instruction_reachability(
        plan,
        instructions_by_id,
        slots_by_id,
        async_pairs,
    )

    live_ranges = tuple(
        DataflowBarrierLiveRange(
            live_range_id=live_range_id,
            consumer_sm=consumer_sm,
            target_slot_id=target_slot_id,
            barrier_lane=barrier_lane,
            uses=tuple(
                sorted(
                    uses,
                    key=lambda use: (
                        topological_positions[use.send_instruction_id],
                        topological_positions[use.recv_instruction_id],
                        use.transfer_key,
                    ),
                )
            ),
        )
        for live_range_id, (
            (consumer_sm, target_slot_id, barrier_lane),
            uses,
        ) in enumerate(sorted(grouped_uses.items()))
    )
    for live_range in live_ranges:
        require_non_overlapping_uses(live_range, reachable)

    edges: list[tuple[int, int]] = []
    for index, first in enumerate(live_ranges):
        for second in live_ranges[index + 1 :]:
            if first.consumer_sm != second.consumer_sm:
                continue
            if first.requires_exclusive_barrier or second.requires_exclusive_barrier or live_ranges_overlap(first, second, reachable):
                edges.append((first.live_range_id, second.live_range_id))
    graph = DataflowBarrierInterferenceGraph(
        node_ids=tuple(live_range.live_range_id for live_range in live_ranges),
        edges=tuple(edges),
    )
    colors = graph.color()

    use_phases: dict[tuple[int, int, int, int, int], int] = {}
    uses_by_local_barrier: dict[tuple[int, int], list[DataflowBarrierUse]] = {}
    for live_range in live_ranges:
        uses_by_local_barrier.setdefault((live_range.consumer_sm, colors[live_range.live_range_id]), []).extend(live_range.uses)
    for uses in uses_by_local_barrier.values():
        transfer_groups: dict[
            tuple[int, int, int, int],
            list[DataflowBarrierUse],
        ] = {}
        for use in uses:
            transfer_groups.setdefault(use.transfer_group_key, []).append(use)
        ordered_groups = sorted(
            transfer_groups.values(),
            key=lambda group: (
                min(topological_positions[use.send_instruction_id] for use in group),
                max(topological_positions[use.recv_instruction_id] for use in group),
                group[0].transfer_group_key,
            ),
        )
        barrier_phase = 0
        for group in ordered_groups:
            for use in group:
                use_phases[use.transfer_key] = barrier_phase
            barrier_phase ^= 1

    assignments = tuple(
        DataflowBarrierAssignment(
            live_range_id=live_range.live_range_id,
            consumer_sm=live_range.consumer_sm,
            target_slot_id=live_range.target_slot_id,
            barrier_lane=live_range.barrier_lane,
            barrier_index=colors[live_range.live_range_id],
            uses=tuple(
                DataflowBarrierUseAssignment(
                    use=use,
                    phase=use_phases[use.transfer_key],
                )
                for use in live_range.uses
            ),
        )
        for live_range in live_ranges
    )
    allocation = DataflowBarrierAllocation(
        live_ranges=live_ranges,
        interference_graph=graph,
        assignments=assignments,
        barrier_count=max(colors.values(), default=-1) + 1,
    )
    return allocation.require_valid()


def instruction_reachability(
    plan: InstructionPlan,
    instructions_by_id: Mapping[int, Any],
    slots_by_id: Mapping[int, SlotPlan],
    async_pairs: list[tuple[CommPlan, CommPlan]],
) -> tuple[dict[int, frozenset[int]], dict[int, int]]:
    adjacency = {instruction_id: set() for instruction_id in instructions_by_id}
    for sm_id, queue in plan.queues.items():
        queue_ids = tuple(instruction.instruction_id for instruction in queue)
        for instruction in queue:
            expected = instructions_by_id.get(instruction.instruction_id)
            if expected is None:
                raise DataflowBarrierPlanningError(f"queue {sm_id} references unknown instruction {instruction.instruction_id}")
            if instruction.sm_id != sm_id:
                raise DataflowBarrierPlanningError(
                    f"queue {sm_id} contains instruction {instruction.instruction_id} assigned to SM {instruction.sm_id}"
                )
        for first, second in zip(queue_ids, queue_ids[1:]):
            if first != second:
                adjacency[first].add(second)
    for comm in plan.comms:
        if comm.source_instruction_id not in adjacency or comm.target_instruction_id not in adjacency:
            raise DataflowBarrierPlanningError(
                f"communication references an unknown instruction: source={comm.source_instruction_id}, target={comm.target_instruction_id}"
            )
        if comm.source_instruction_id != comm.target_instruction_id:
            adjacency[comm.source_instruction_id].add(comm.target_instruction_id)
    for send_comm, recv_comm in async_pairs:
        send_instruction_id = send_comm.resolved_dispatch_instruction_id
        recv_instruction_id = recv_comm.resolved_dispatch_instruction_id
        if send_instruction_id != recv_instruction_id:
            adjacency[send_instruction_id].add(recv_instruction_id)
    for instruction in plan.instructions:
        for slot_id in instruction.input_slots:
            slot = slots_by_id.get(slot_id)
            if slot is None:
                raise DataflowBarrierPlanningError(f"instruction {instruction.instruction_id} references unknown input slot {slot_id}")
            producer_id = slot.producer_instruction_id
            if producer_id is not None and producer_id != instruction.instruction_id:
                if producer_id not in adjacency:
                    raise DataflowBarrierPlanningError(f"slot {slot_id} references unknown producer instruction {producer_id}")
                adjacency[producer_id].add(instruction.instruction_id)

    indegree = {instruction_id: 0 for instruction_id in adjacency}
    for successors in adjacency.values():
        for successor in successors:
            indegree[successor] += 1
    ready = [instruction_id for instruction_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    topological_order: list[int] = []
    while ready:
        instruction_id = heapq.heappop(ready)
        topological_order.append(instruction_id)
        for successor in sorted(adjacency[instruction_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, successor)
    if len(topological_order) != len(adjacency):
        cyclic_ids = sorted(instruction_id for instruction_id, degree in indegree.items() if degree)
        raise DataflowBarrierPlanningError(f"instruction happens-before graph contains a cycle: {cyclic_ids!r}")

    reachable: dict[int, frozenset[int]] = {}
    for instruction_id in reversed(topological_order):
        descendants: set[int] = set()
        for successor in adjacency[instruction_id]:
            descendants.add(successor)
            descendants.update(reachable[successor])
        reachable[instruction_id] = frozenset(descendants)
    positions = {instruction_id: index for index, instruction_id in enumerate(topological_order)}
    return reachable, positions


def validate_async_pair(
    send_comm: CommPlan,
    recv_comm: CommPlan,
    instructions_by_id: Mapping[int, Any],
    slots_by_id: Mapping[int, SlotPlan],
) -> None:
    if recv_comm.source_slot_id not in slots_by_id or recv_comm.target_slot_id not in slots_by_id:
        raise DataflowBarrierPlanningError(
            f"async receive references an unknown slot: source={recv_comm.source_slot_id}, target={recv_comm.target_slot_id}"
        )
    send_instruction = instructions_by_id.get(send_comm.resolved_dispatch_instruction_id)
    recv_instruction = instructions_by_id.get(recv_comm.resolved_dispatch_instruction_id)
    if send_instruction is None or recv_instruction is None:
        raise DataflowBarrierPlanningError(
            "async communication dispatch references an unknown instruction: "
            f"send={send_comm.resolved_dispatch_instruction_id}, "
            f"recv={recv_comm.resolved_dispatch_instruction_id}"
        )
    if send_instruction.sm_id != send_comm.producer_sm:
        raise DataflowBarrierPlanningError(
            f"async send producer SM mismatch for instruction {send_instruction.instruction_id}: "
            f"instruction={send_instruction.sm_id}, comm={send_comm.producer_sm}"
        )
    if recv_instruction.sm_id != recv_comm.consumer_sm:
        raise DataflowBarrierPlanningError(
            f"async receive consumer SM mismatch for instruction {recv_instruction.instruction_id}: "
            f"instruction={recv_instruction.sm_id}, comm={recv_comm.consumer_sm}"
        )


def require_non_overlapping_uses(
    live_range: DataflowBarrierLiveRange,
    reachable: Mapping[int, frozenset[int]],
) -> None:
    for index, first in enumerate(live_range.uses):
        for second in live_range.uses[index + 1 :]:
            if uses_overlap(first, second, reachable):
                raise DataflowBarrierPlanningError(
                    f"target slot {live_range.target_slot_id} has overlapping async receive intervals "
                    f"{first.transfer_key!r} and {second.transfer_key!r}"
                )


def live_ranges_overlap(
    first: DataflowBarrierLiveRange,
    second: DataflowBarrierLiveRange,
    reachable: Mapping[int, frozenset[int]],
) -> bool:
    return any(uses_overlap(first_use, second_use, reachable) for first_use in first.uses for second_use in second.uses)


def uses_overlap(
    first: DataflowBarrierUse,
    second: DataflowBarrierUse,
    reachable: Mapping[int, frozenset[int]],
) -> bool:
    same_segmented_transfer = (
        first.transfer_source_instruction_id == second.transfer_source_instruction_id
        and first.transfer_target_instruction_id == second.transfer_target_instruction_id
        and first.source_slot_id == second.source_slot_id
        and first.target_slot_id == second.target_slot_id
        and first.segment_id != second.segment_id
    )
    if same_segmented_transfer:
        # One CTA dispatches all HBM receive segments as one mbarrier
        # transaction group.  Their shared interval is intentionally not an
        # interference edge with itself.
        return False
    return not (
        precedes_or_same(first.recv_instruction_id, second.send_instruction_id, reachable)
        or precedes_or_same(second.recv_instruction_id, first.send_instruction_id, reachable)
    )


def precedes_or_same(
    first_instruction_id: int,
    second_instruction_id: int,
    reachable: Mapping[int, frozenset[int]],
) -> bool:
    return first_instruction_id == second_instruction_id or second_instruction_id in reachable[first_instruction_id]


def slot_global_alias_key(slot: SlotPlan) -> int:
    return slot.global_storage_id if slot.global_storage_id is not None else slot.slot_id


def direct_global_signal_only_comm(
    comm: CommPlan,
    slots_by_id: Mapping[int, SlotPlan],
    hbm_direct_global_slots: frozenset[int],
) -> bool:
    if comm.kind not in {DataflowCommKind.HBM_RECV, DataflowCommKind.HBM_RECV_WAIT}:
        return False
    if comm.source_slot_id not in hbm_direct_global_slots or comm.target_slot_id not in hbm_direct_global_slots:
        return False
    source_slot = slots_by_id.get(comm.source_slot_id)
    target_slot = slots_by_id.get(comm.target_slot_id)
    if source_slot is None or target_slot is None:
        return False
    return slot_global_alias_key(source_slot) == slot_global_alias_key(target_slot)


def comm_transfer_key(comm: CommPlan) -> tuple[int, int, int, int, int]:
    return (
        comm.source_instruction_id,
        comm.target_instruction_id,
        comm.source_slot_id,
        comm.target_slot_id,
        comm.segment_id,
    )
