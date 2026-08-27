"""Structured validation and diagnostics for Dataflow runtime memory layouts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from tilelang.utils.target_capabilities import TargetCapabilitySnapshot

from .handler_codegen import DataflowHandlerCodegenArtifact
from .launch import (
    DATAFLOW_BARRIER_BYTES,
    DATAFLOW_SHARED_ALIGNMENT,
    UINT32_BYTES,
    DataflowLaunchPackage,
)
from .runtime import (
    DATAFLOW_SLOT_ALIGNMENT,
    DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL,
    DATAFLOW_SLOT_FLAG_SCRATCH_BACKED,
    PackedRuntimePlan,
    UINT32_SENTINEL,
)
from .scheduler import InstructionPlan
from .wrapper import DataflowWrapperSpec


class DataflowLayoutRegionKind(str, Enum):
    """Kind of one allocated interval in a Dataflow artifact."""

    BARRIER = "barrier"
    SCRATCH_ARENA = "scratch_arena"
    SHARED_SLOT = "shared_slot"
    SCRATCH_SLOT = "scratch_slot"
    GLOBAL_SLOT = "global_slot"
    HANDLER_SCRATCH = "handler_scratch"
    HANDOFF_ARENA = "handoff_arena"


@dataclass(frozen=True)
class DataflowSlotLiveRange:
    """Structured instruction ownership and uses for one logical slot."""

    producer_instruction_id: int | None
    consumer_instruction_ids: tuple[int, ...]
    comm_dispatch_instruction_ids: tuple[int, ...]

    @property
    def instruction_ids(self) -> tuple[int, ...]:
        ids = set(self.consumer_instruction_ids) | set(self.comm_dispatch_instruction_ids)
        if self.producer_instruction_id is not None:
            ids.add(self.producer_instruction_id)
        return tuple(sorted(ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer_instruction_id": self.producer_instruction_id,
            "consumer_instruction_ids": list(self.consumer_instruction_ids),
            "comm_dispatch_instruction_ids": list(self.comm_dispatch_instruction_ids),
        }


@dataclass(frozen=True)
class DataflowLayoutRegion:
    """One concrete byte interval in shared or global memory."""

    name: str
    kind: DataflowLayoutRegionKind
    memory_space: str
    offset: int
    bytes: int
    alignment: int
    placement_reason: str
    owner_cta: int | None = None
    slot_id: int | None = None
    handler_id: int | None = None
    handler_symbol: str | None = None
    storage_id: int | None = None
    live_range: DataflowSlotLiveRange | None = None

    @property
    def end(self) -> int:
        return self.offset + self.bytes

    def overlaps(self, other: DataflowLayoutRegion) -> bool:
        if self.memory_space != other.memory_space or not self.bytes or not other.bytes:
            return False
        return self.offset < other.end and other.offset < self.end

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "memory_space": self.memory_space,
            "offset": self.offset,
            "bytes": self.bytes,
            "end": self.end,
            "alignment": self.alignment,
            "placement_reason": self.placement_reason,
            "owner_cta": self.owner_cta,
            "slot_id": self.slot_id,
            "handler_id": self.handler_id,
            "handler_symbol": self.handler_symbol,
            "storage_id": self.storage_id,
            "live_range": (None if self.live_range is None else self.live_range.to_dict()),
        }


@dataclass(frozen=True)
class DataflowLayoutConflict:
    """A pair of intervals whose overlap is not permitted by their access binding."""

    kind: str
    first_region: str
    second_region: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "first_region": self.first_region,
            "second_region": self.second_region,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class DataflowLayoutValidationReport:
    """Complete structured result of validating one compiled Dataflow layout."""

    regions: tuple[DataflowLayoutRegion, ...]
    conflicts: tuple[DataflowLayoutConflict, ...]
    errors: tuple[str, ...]
    shared_memory_bytes: int
    global_memory_bytes: int
    target_shared_memory_limit: int | None
    scratch_offset: int
    scratch_bytes: int
    handoff_arena_offset: int
    handoff_arena_bytes: int

    @property
    def valid(self) -> bool:
        return not self.errors and not self.conflicts

    @property
    def shared_memory_within_target_limit(self) -> bool:
        return self.target_shared_memory_limit is None or self.shared_memory_bytes <= self.target_shared_memory_limit

    def regions_for_kind(
        self,
        kind: DataflowLayoutRegionKind,
    ) -> tuple[DataflowLayoutRegion, ...]:
        return tuple(region for region in self.regions if region.kind is kind)

    def slot_region(self, slot_id: int, *, memory_space: str = "shared") -> DataflowLayoutRegion:
        for region in self.regions:
            if region.slot_id == slot_id and region.memory_space == memory_space:
                return region
        raise KeyError(f"Dataflow layout report has no {memory_space} region for slot {slot_id}")

    def handler_region(self, handler_id: int) -> DataflowLayoutRegion:
        for region in self.regions:
            if region.kind is DataflowLayoutRegionKind.HANDLER_SCRATCH and region.handler_id == handler_id:
                return region
        raise KeyError(f"Dataflow layout report has no scratch region for handler {handler_id}")

    def require_valid(self) -> DataflowLayoutValidationReport:
        if self.valid:
            return self
        details = [*self.errors, *(conflict.detail for conflict in self.conflicts)]
        raise ValueError("invalid Dataflow memory layout: " + "; ".join(details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "shared_memory_bytes": self.shared_memory_bytes,
            "global_memory_bytes": self.global_memory_bytes,
            "target_shared_memory_limit": self.target_shared_memory_limit,
            "shared_memory_within_target_limit": self.shared_memory_within_target_limit,
            "scratch_offset": self.scratch_offset,
            "scratch_bytes": self.scratch_bytes,
            "handoff_arena_offset": self.handoff_arena_offset,
            "handoff_arena_bytes": self.handoff_arena_bytes,
            "regions": [region.to_dict() for region in self.regions],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "errors": list(self.errors),
        }


def validate_dataflow_memory_layout(
    packed_plan: PackedRuntimePlan,
    launch_package: DataflowLaunchPackage,
    wrapper_spec: DataflowWrapperSpec,
    *,
    plan: InstructionPlan | None = None,
    handler_artifacts: tuple[DataflowHandlerCodegenArtifact, ...] = (),
    target_capabilities: TargetCapabilitySnapshot | None = None,
) -> DataflowLayoutValidationReport:
    """Validate allocation invariants without depending on allocator offsets."""

    if not isinstance(packed_plan, PackedRuntimePlan):
        raise TypeError(f"validate_dataflow_memory_layout expects PackedRuntimePlan, got {packed_plan!r}")
    if not isinstance(launch_package, DataflowLaunchPackage):
        raise TypeError(f"validate_dataflow_memory_layout expects DataflowLaunchPackage, got {launch_package!r}")
    if not isinstance(wrapper_spec, DataflowWrapperSpec):
        raise TypeError(f"validate_dataflow_memory_layout expects DataflowWrapperSpec, got {wrapper_spec!r}")
    if plan is not None and not isinstance(plan, InstructionPlan):
        raise TypeError(f"validate_dataflow_memory_layout plan must be InstructionPlan, got {plan!r}")
    if target_capabilities is not None and not isinstance(target_capabilities, TargetCapabilitySnapshot):
        raise TypeError(
            f"validate_dataflow_memory_layout target_capabilities must be TargetCapabilitySnapshot, got {target_capabilities!r}"
        )

    errors: list[str] = []
    conflicts: list[DataflowLayoutConflict] = []
    regions: list[DataflowLayoutRegion] = []
    if len(packed_plan.slots) != launch_package.slot_count:
        errors.append(f"packed slot count does not match launch package: {len(packed_plan.slots)} != {launch_package.slot_count}")
    if launch_package.shared_memory_bytes != wrapper_spec.shared_memory_bytes:
        errors.append(
            f"launch/wrapper shared-memory size mismatch: {launch_package.shared_memory_bytes} != {wrapper_spec.shared_memory_bytes}"
        )
    if launch_package.shared_slot_base_offset != wrapper_spec.shared_slot_base_offset:
        errors.append(
            f"launch/wrapper shared-slot base mismatch: {launch_package.shared_slot_base_offset} != {wrapper_spec.shared_slot_base_offset}"
        )
    if launch_package.barrier_count != wrapper_spec.barrier_count:
        errors.append(f"launch/wrapper barrier-count mismatch: {launch_package.barrier_count} != {wrapper_spec.barrier_count}")
    if launch_package.cluster_ack_count != wrapper_spec.cluster_ack_count:
        errors.append(f"launch/wrapper cluster-ack-count mismatch: {launch_package.cluster_ack_count} != {wrapper_spec.cluster_ack_count}")
    if launch_package.cluster_inbox_offset != wrapper_spec.cluster_inbox_offset:
        errors.append(
            f"launch/wrapper cluster-inbox offset mismatch: {launch_package.cluster_inbox_offset} != {wrapper_spec.cluster_inbox_offset}"
        )
    if launch_package.cluster_inbox_bytes != wrapper_spec.cluster_inbox_bytes:
        errors.append(
            f"launch/wrapper cluster-inbox bytes mismatch: {launch_package.cluster_inbox_bytes} != {wrapper_spec.cluster_inbox_bytes}"
        )
    try:
        barrier_allocation = packed_plan.barrier_allocation.require_valid()
    except (TypeError, ValueError) as err:
        errors.append(f"invalid packed barrier allocation: {err}")
    else:
        if launch_package.barrier_count != barrier_allocation.barrier_count:
            errors.append(f"launch/barrier-allocation count mismatch: {launch_package.barrier_count} != {barrier_allocation.barrier_count}")
        for slot_id, barrier_index in barrier_allocation.slot_barrier_indices.items():
            if slot_id >= len(packed_plan.slots):
                errors.append(f"barrier allocation references unknown packed slot {slot_id}")
            elif packed_plan.slots[slot_id].barrier_index != barrier_index:
                errors.append(f"slot {slot_id} barrier binding mismatch: {packed_plan.slots[slot_id].barrier_index} != {barrier_index}")
    if launch_package.comm_count != wrapper_spec.comm_count:
        errors.append(f"launch/wrapper comm-count mismatch: {launch_package.comm_count} != {wrapper_spec.comm_count}")
    if plan is not None and len(plan.slots) != len(packed_plan.slots):
        errors.append(f"plan/packed slot-count mismatch: {len(plan.slots)} != {len(packed_plan.slots)}")
    if launch_package.barrier_bytes != launch_package.barrier_count * DATAFLOW_BARRIER_BYTES:
        errors.append(
            "barrier byte extent does not match the barrier count: "
            f"{launch_package.barrier_bytes} != "
            f"{launch_package.barrier_count} * {DATAFLOW_BARRIER_BYTES}"
        )
    if launch_package.cluster_ack_barrier_bytes != launch_package.cluster_ack_count * DATAFLOW_BARRIER_BYTES:
        errors.append(
            "cluster ack barrier byte extent does not match the ack count: "
            f"{launch_package.cluster_ack_barrier_bytes} != "
            f"{launch_package.cluster_ack_count} * {DATAFLOW_BARRIER_BYTES}"
        )
    expected_pending_bytes = ((launch_package.cluster_ack_count + 31) // 32) * UINT32_BYTES
    if launch_package.cluster_ack_pending_bytes != expected_pending_bytes:
        errors.append(
            "cluster ack pending bitset byte extent does not match the ack count: "
            f"{launch_package.cluster_ack_pending_bytes} != {expected_pending_bytes}"
        )
    expected_hbm_recv_state_bytes = ((launch_package.barrier_count + 31) // 32) * UINT32_BYTES
    if launch_package.hbm_recv_state_bytes != expected_hbm_recv_state_bytes:
        errors.append(
            "HBM receive state bitset byte extent does not match the barrier count: "
            f"{launch_package.hbm_recv_state_bytes} != "
            f"{expected_hbm_recv_state_bytes}"
        )
    expected_control_bytes = (
        launch_package.barrier_bytes
        + launch_package.cluster_ack_barrier_bytes
        + launch_package.cluster_ack_pending_bytes
        + launch_package.hbm_recv_state_bytes
    )
    if launch_package.shared_control_bytes != expected_control_bytes:
        errors.append(
            f"shared control byte extent does not match its regions: {launch_package.shared_control_bytes} != {expected_control_bytes}"
        )
    if launch_package.shared_slot_base_offset < launch_package.shared_control_bytes:
        errors.append(
            "shared-slot base overlaps the control region: "
            f"{launch_package.shared_slot_base_offset} < "
            f"{launch_package.shared_control_bytes}"
        )

    scratch_region = DataflowLayoutRegion(
        name="primfunc_scratch_arena",
        kind=DataflowLayoutRegionKind.SCRATCH_ARENA,
        memory_space="shared",
        offset=wrapper_spec.primfunc_scratch_offset,
        bytes=wrapper_spec.primfunc_scratch_bytes,
        alignment=DATAFLOW_SHARED_ALIGNMENT,
        placement_reason="shared arena reserved for lowered handlers and scratch-backed slots",
    )
    validate_region_alignment(scratch_region, errors)
    validate_region_bounds(
        scratch_region,
        0,
        launch_package.shared_memory_bytes,
        errors,
    )
    if scratch_region.bytes:
        regions.append(scratch_region)
    if wrapper_spec.cluster_inbox_bytes:
        if wrapper_spec.cluster_inbox_offset % DATAFLOW_SLOT_ALIGNMENT:
            errors.append(f"cluster inbox is not slot-aligned: offset={wrapper_spec.cluster_inbox_offset}")
        if wrapper_spec.cluster_inbox_offset + wrapper_spec.cluster_inbox_bytes > wrapper_spec.primfunc_scratch_bytes:
            errors.append(
                "cluster inbox escapes PrimFunc scratch: "
                f"end={wrapper_spec.cluster_inbox_offset + wrapper_spec.cluster_inbox_bytes}, "
                f"scratch_bytes={wrapper_spec.primfunc_scratch_bytes}"
            )
    elif wrapper_spec.cluster_inbox_offset:
        errors.append("zero-byte cluster inbox has a nonzero offset")

    arena_fingerprints: set[str] = set()
    for arena in wrapper_spec.handoff_plan_arenas:
        if arena.plan_fingerprint in arena_fingerprints:
            errors.append(f"wrapper handoff arena repeats plan {arena.plan_fingerprint!r}")
        arena_fingerprints.add(arena.plan_fingerprint)
        region = DataflowLayoutRegion(
            name=f"handoff[{arena.plan_fingerprint[:12]}].arena",
            kind=DataflowLayoutRegionKind.HANDOFF_ARENA,
            memory_space="shared",
            offset=arena.offset,
            bytes=arena.bytes,
            alignment=arena.alignment,
            placement_reason="compiler-owned cross-handler staged-value arena",
        )
        regions.append(region)
        validate_region_alignment(region, errors)
        validate_region_bounds(region, 0, launch_package.shared_memory_bytes, errors)
    if wrapper_spec.handoff_plan_arenas:
        arena_begin = min(arena.offset for arena in wrapper_spec.handoff_plan_arenas)
        arena_end = max(arena.end for arena in wrapper_spec.handoff_plan_arenas)
        if wrapper_spec.handoff_arena_offset != arena_begin:
            errors.append(
                f"wrapper handoff arena base differs from its plan intervals: {wrapper_spec.handoff_arena_offset} != {arena_begin}"
            )
        if wrapper_spec.handoff_arena_bytes != arena_end - arena_begin:
            errors.append(
                "wrapper handoff arena extent differs from its plan intervals: "
                f"{wrapper_spec.handoff_arena_bytes} != {arena_end - arena_begin}"
            )
    elif wrapper_spec.handoff_arena_offset or wrapper_spec.handoff_arena_bytes:
        errors.append("wrapper handoff arena aggregate lacks plan intervals")
    if plan is not None:
        enabled_handoffs = {item.fingerprint: item for item in plan.cross_handler_handoff_plans if item.enabled}
        if set(enabled_handoffs) != arena_fingerprints:
            errors.append(
                "wrapper handoff arenas differ from enabled instruction plans: "
                f"arenas={sorted(arena_fingerprints)!r}, "
                f"plans={sorted(enabled_handoffs)!r}"
            )
        for arena in wrapper_spec.handoff_plan_arenas:
            handoff = enabled_handoffs.get(arena.plan_fingerprint)
            if handoff is None:
                continue
            if arena.bytes != handoff.arena_bytes:
                errors.append(f"handoff plan {arena.plan_fingerprint!r} arena bytes changed: {arena.bytes} != {handoff.arena_bytes}")
            if arena.alignment != handoff.arena_alignment:
                errors.append(
                    f"handoff plan {arena.plan_fingerprint!r} arena alignment changed: {arena.alignment} != {handoff.arena_alignment}"
                )

    if launch_package.shared_control_bytes:
        barrier_region = DataflowLayoutRegion(
            name="cluster_control",
            kind=DataflowLayoutRegionKind.BARRIER,
            memory_space="shared",
            offset=0,
            bytes=launch_package.shared_control_bytes,
            alignment=8,
            placement_reason=("receive barriers, completion acknowledgements, and runtime state bitsets"),
        )
        regions.append(barrier_region)
        validate_region_alignment(barrier_region, errors)
        validate_region_bounds(
            barrier_region,
            0,
            launch_package.shared_memory_bytes,
            errors,
        )

    if plan is not None and len({slot.slot_id for slot in plan.slots}) != len(plan.slots):
        errors.append("instruction plan contains duplicate slot ids")
    plan_slots = None if plan is None else {slot.slot_id: slot for slot in plan.slots}
    live_ranges = {} if plan is None else slot_live_ranges(plan)
    for slot_id, packed_slot in enumerate(packed_plan.slots):
        plan_slot = None if plan_slots is None else plan_slots.get(slot_id)
        owner_cta = None if packed_slot.owner_cta == UINT32_SENTINEL else packed_slot.owner_cta
        storage_id = None if plan_slot is None else plan_slot.shared_storage_id
        live_range = live_ranges.get(slot_id)
        if plan_slot is not None and plan_slot.scratch_backed != packed_slot.is_scratch_backed:
            errors.append(
                f"slot {slot_id} scratch-backed plan/packed mismatch: {plan_slot.scratch_backed} != {packed_slot.is_scratch_backed}"
            )
        if packed_slot.is_scratch_backed:
            kind = DataflowLayoutRegionKind.SCRATCH_SLOT
            offset = wrapper_spec.primfunc_scratch_offset + packed_slot.shared_offset
            placement_reason = "slot explicitly assigned to PrimFunc scratch"
            if plan_slot is not None and plan_slot.scratch_offset is not None and plan_slot.scratch_offset != packed_slot.shared_offset:
                errors.append(
                    f"slot {slot_id} scratch offset does not match its plan binding: "
                    f"{packed_slot.shared_offset} != {plan_slot.scratch_offset}"
                )
        elif packed_slot.flags & DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL:
            kind = None
            offset = 0
            placement_reason = "slot uses direct global storage"
        else:
            kind = DataflowLayoutRegionKind.SHARED_SLOT
            offset = launch_package.shared_slot_base_offset + packed_slot.shared_offset
            placement_reason = "ordinary shared slot allocation"
        if kind is not None:
            region = DataflowLayoutRegion(
                name=f"slot[{slot_id}].shared",
                kind=kind,
                memory_space="shared",
                offset=offset,
                bytes=packed_slot.bytes,
                alignment=DATAFLOW_SLOT_ALIGNMENT,
                placement_reason=placement_reason,
                owner_cta=owner_cta,
                slot_id=slot_id,
                storage_id=storage_id,
                live_range=live_range,
            )
            regions.append(region)
            validate_region_alignment(region, errors)
            if kind is DataflowLayoutRegionKind.SCRATCH_SLOT:
                validate_region_bounds(
                    region,
                    wrapper_spec.primfunc_scratch_offset,
                    wrapper_spec.primfunc_scratch_offset + wrapper_spec.primfunc_scratch_bytes,
                    errors,
                )
            else:
                validate_region_bounds(
                    region,
                    launch_package.shared_slot_base_offset,
                    launch_package.shared_slot_base_offset + launch_package.shared_slot_bytes,
                    errors,
                )

        global_region = DataflowLayoutRegion(
            name=f"slot[{slot_id}].global",
            kind=DataflowLayoutRegionKind.GLOBAL_SLOT,
            memory_space="global",
            offset=packed_slot.global_offset,
            bytes=packed_slot.bytes,
            alignment=DATAFLOW_SLOT_ALIGNMENT,
            placement_reason="runtime global staging allocation",
            owner_cta=owner_cta,
            slot_id=slot_id,
            storage_id=(None if plan_slot is None else plan_slot.global_storage_id),
            live_range=live_range,
        )
        regions.append(global_region)
        validate_region_alignment(global_region, errors)
        validate_region_bounds(
            global_region,
            0,
            len(launch_package.global_staging_bytes),
            errors,
        )

    artifact_symbols = tuple(artifact.device_symbol for artifact in handler_artifacts)
    artifact_handler_ids = tuple(artifact.handler_id for artifact in handler_artifacts)
    if len(artifact_symbols) != len(set(artifact_symbols)):
        errors.append("handler codegen artifacts contain duplicate device symbols")
    if len(artifact_handler_ids) != len(set(artifact_handler_ids)):
        errors.append("handler codegen artifacts contain duplicate handler ids")
    artifact_by_symbol = {artifact.device_symbol: artifact for artifact in handler_artifacts}
    offset_entries = wrapper_spec.primfunc_handler_scratch_offsets
    if len(offset_entries) != len({symbol for symbol, _ in offset_entries}):
        errors.append("handler scratch offsets contain duplicate device symbols")
    handler_offset_by_symbol = dict(offset_entries)
    for symbol in handler_offset_by_symbol:
        if symbol not in artifact_by_symbol:
            errors.append(f"handler scratch offset references unknown artifact symbol {symbol!r}")
    for artifact in handler_artifacts:
        if artifact.dynamic_shared_bytes <= 0:
            continue
        relative_offset = handler_offset_by_symbol.get(artifact.device_symbol, 0)
        region = DataflowLayoutRegion(
            name=f"handler[{artifact.handler_id}].scratch",
            kind=DataflowLayoutRegionKind.HANDLER_SCRATCH,
            memory_space="shared",
            offset=wrapper_spec.primfunc_scratch_offset + relative_offset,
            bytes=artifact.dynamic_shared_bytes,
            alignment=DATAFLOW_SHARED_ALIGNMENT,
            placement_reason="lowered handler dynamic-shared requirement",
            handler_id=artifact.handler_id,
            handler_symbol=artifact.device_symbol,
        )
        regions.append(region)
        validate_region_alignment(region, errors)
        validate_region_bounds(
            region,
            wrapper_spec.primfunc_scratch_offset,
            wrapper_spec.primfunc_scratch_offset + wrapper_spec.primfunc_scratch_bytes,
            errors,
        )

    reusable_communicate_slot_ids = (
        frozenset()
        if plan is None
        else frozenset(
            slot.slot_id
            for slot in plan.slots
            if slot.role
            in {
                "cluster_inbox",
                "cluster_gated_inbox",
                "hbm_spill_inbox",
                "joint_comm_inbox",
                "joint_comm_outbox",
                "joint_prefetch_inbox",
            }
        )
    )
    physical_alias_slot_pairs = (
        frozenset()
        if plan is None
        else frozenset(frozenset((slot.slot_id, slot.alias_of_slot_id)) for slot in plan.slots if slot.alias_of_slot_id is not None)
    )
    validate_slot_region_overlaps(
        regions,
        conflicts,
        reusable_communicate_slot_ids=reusable_communicate_slot_ids,
        physical_alias_slot_pairs=physical_alias_slot_pairs,
    )
    validate_barrier_overlaps(regions, conflicts)
    validate_reserved_scratch_overlaps(regions, conflicts)
    if plan is not None:
        validate_handler_input_overlaps(plan, wrapper_spec, regions, conflicts)
        validate_value_forward_ownership(
            plan,
            packed_plan,
            wrapper_spec,
            errors,
        )

    target_limit = None if target_capabilities is None else target_capabilities.max_dynamic_shared_memory
    if target_limit is not None and launch_package.shared_memory_bytes > target_limit:
        errors.append(f"dynamic shared-memory requirement exceeds target limit: {launch_package.shared_memory_bytes} > {target_limit}")
    return DataflowLayoutValidationReport(
        regions=tuple(regions),
        conflicts=tuple(conflicts),
        errors=tuple(errors),
        shared_memory_bytes=launch_package.shared_memory_bytes,
        global_memory_bytes=len(launch_package.global_staging_bytes),
        target_shared_memory_limit=target_limit,
        scratch_offset=wrapper_spec.primfunc_scratch_offset,
        scratch_bytes=wrapper_spec.primfunc_scratch_bytes,
        handoff_arena_offset=wrapper_spec.handoff_arena_offset,
        handoff_arena_bytes=wrapper_spec.handoff_arena_bytes,
    )


def slot_live_ranges(plan: InstructionPlan) -> dict[int, DataflowSlotLiveRange]:
    consumers: dict[int, set[int]] = {slot.slot_id: set() for slot in plan.slots}
    comm_dispatches: dict[int, set[int]] = {slot.slot_id: set() for slot in plan.slots}
    for instruction in plan.instructions:
        for slot_id in instruction.input_slots:
            consumers.setdefault(slot_id, set()).add(instruction.instruction_id)
    for comm in plan.comms:
        dispatch_id = comm.resolved_dispatch_instruction_id
        comm_dispatches.setdefault(comm.source_slot_id, set()).add(dispatch_id)
        comm_dispatches.setdefault(comm.target_slot_id, set()).add(dispatch_id)
    return {
        slot.slot_id: DataflowSlotLiveRange(
            producer_instruction_id=slot.producer_instruction_id,
            consumer_instruction_ids=tuple(sorted(consumers.get(slot.slot_id, ()))),
            comm_dispatch_instruction_ids=tuple(sorted(comm_dispatches.get(slot.slot_id, ()))),
        )
        for slot in plan.slots
    }


def validate_region_alignment(
    region: DataflowLayoutRegion,
    errors: list[str],
) -> None:
    if region.offset < 0 or region.bytes < 0:
        errors.append(f"region {region.name!r} has negative offset/size: offset={region.offset}, bytes={region.bytes}")
    if region.alignment <= 0 or region.offset % region.alignment:
        errors.append(f"region {region.name!r} is not {region.alignment}-byte aligned: offset={region.offset}")


def validate_region_bounds(
    region: DataflowLayoutRegion,
    begin: int,
    end: int,
    errors: list[str],
) -> None:
    if region.offset < begin or region.end > end:
        errors.append(f"region {region.name!r} [{region.offset}, {region.end}) is outside [{begin}, {end})")


def validate_slot_region_overlaps(
    regions: list[DataflowLayoutRegion],
    conflicts: list[DataflowLayoutConflict],
    *,
    reusable_communicate_slot_ids: frozenset[int] = frozenset(),
    physical_alias_slot_pairs: frozenset[frozenset[int]] = frozenset(),
) -> None:
    slot_regions = [
        region
        for region in regions
        if region.kind
        in {
            DataflowLayoutRegionKind.SHARED_SLOT,
            DataflowLayoutRegionKind.SCRATCH_SLOT,
            DataflowLayoutRegionKind.GLOBAL_SLOT,
        }
    ]
    for index, first in enumerate(slot_regions):
        for second in slot_regions[index + 1 :]:
            if not first.overlaps(second):
                continue
            if (
                first.memory_space == "shared"
                and second.memory_space == "shared"
                and first.owner_cta is not None
                and second.owner_cta is not None
                and first.owner_cta != second.owner_cta
            ):
                continue
            if first.storage_id is not None and first.storage_id == second.storage_id:
                continue
            if (
                first.slot_id is not None
                and second.slot_id is not None
                and frozenset((first.slot_id, second.slot_id)) in physical_alias_slot_pairs
            ):
                continue
            if (
                first.memory_space == "shared"
                and first.slot_id in reusable_communicate_slot_ids
                and second.slot_id in reusable_communicate_slot_ids
            ):
                continue
            if not live_ranges_overlap(first.live_range, second.live_range):
                continue
            conflicts.append(
                DataflowLayoutConflict(
                    kind="slot_live_range_overlap",
                    first_region=first.name,
                    second_region=second.name,
                    detail=(f"shared slot regions {first.name!r} and {second.name!r} overlap with intersecting live ranges"),
                )
            )


def validate_barrier_overlaps(
    regions: list[DataflowLayoutRegion],
    conflicts: list[DataflowLayoutConflict],
) -> None:
    barriers = [region for region in regions if region.kind is DataflowLayoutRegionKind.BARRIER]
    shared_regions = [
        region for region in regions if region.memory_space == "shared" and region.kind is not DataflowLayoutRegionKind.BARRIER
    ]
    for barrier in barriers:
        for region in shared_regions:
            if not barrier.overlaps(region):
                continue
            conflicts.append(
                DataflowLayoutConflict(
                    kind="barrier_region_overlap",
                    first_region=barrier.name,
                    second_region=region.name,
                    detail=(f"barrier region {barrier.name!r} overlaps shared region {region.name!r}"),
                )
            )


def validate_reserved_scratch_overlaps(
    regions: list[DataflowLayoutRegion],
    conflicts: list[DataflowLayoutConflict],
) -> None:
    reserved_regions = [
        region
        for region in regions
        if region.kind
        in {
            DataflowLayoutRegionKind.HANDOFF_ARENA,
        }
    ]
    for reserved in reserved_regions:
        for region in regions:
            if region is reserved or region.kind is DataflowLayoutRegionKind.BARRIER:
                continue
            if not reserved.overlaps(region):
                continue
            conflicts.append(
                DataflowLayoutConflict(
                    kind="reserved_scratch_overlap",
                    first_region=reserved.name,
                    second_region=region.name,
                    detail=(f"reserved scratch region {reserved.name!r} overlaps region {region.name!r}"),
                )
            )


def validate_handler_input_overlaps(
    plan: InstructionPlan,
    wrapper_spec: DataflowWrapperSpec,
    regions: list[DataflowLayoutRegion],
    conflicts: list[DataflowLayoutConflict],
) -> None:
    handler_id_by_variant = {
        handler.handler_variant_key: handler.handler_id for handler in wrapper_spec.handlers if handler.handler_variant_key is not None
    }
    handler_regions = {region.handler_id: region for region in regions if region.kind is DataflowLayoutRegionKind.HANDLER_SCRATCH}
    shared_slot_regions = {region.slot_id: region for region in regions if region.slot_id is not None and region.memory_space == "shared"}
    for instruction in plan.instructions:
        handler_id = handler_id_by_variant.get(instruction.handler_variant_key)
        handler_region = handler_regions.get(handler_id)
        if handler_region is None:
            continue
        for slot_id in instruction.input_slots:
            slot_region = shared_slot_regions.get(slot_id)
            if slot_region is None or not handler_region.overlaps(slot_region):
                continue
            conflicts.append(
                DataflowLayoutConflict(
                    kind="handler_input_overlap",
                    first_region=handler_region.name,
                    second_region=slot_region.name,
                    detail=(f"handler {handler_id} scratch overlaps input slot {slot_id} for instruction {instruction.instruction_id}"),
                )
            )


def validate_value_forward_ownership(
    plan: InstructionPlan,
    packed_plan: PackedRuntimePlan,
    wrapper_spec: DataflowWrapperSpec,
    errors: list[str],
) -> None:
    slots_by_id = {slot.slot_id: slot for slot in plan.slots}
    consumers: dict[int, set[int]] = {slot.slot_id: set() for slot in plan.slots}
    for instruction in plan.instructions:
        for slot_id in instruction.input_slots:
            consumers.setdefault(slot_id, set()).add(instruction.instruction_id)

    def location(slot_id: int) -> tuple[str, int, int]:
        packed = packed_plan.slots[slot_id]
        if wrapper_spec.primfunc_use_global_slot_fields and packed.flags & DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL:
            return ("global", packed.global_offset, packed.bytes)
        if packed.flags & DATAFLOW_SLOT_FLAG_SCRATCH_BACKED:
            return ("scratch", packed.shared_offset, packed.bytes)
        return ("shared", packed.shared_offset, packed.bytes)

    for instruction in plan.instructions:
        forward = instruction.value_forward
        if forward is None:
            continue
        source = slots_by_id.get(forward.source_slot_id)
        output = slots_by_id.get(forward.output_slot_id)
        if source is None or output is None:
            errors.append(f"instruction {instruction.instruction_id} ValueForward references an unknown slot")
            continue
        if source.intermediate_type is not output.intermediate_type:
            errors.append(f"instruction {instruction.instruction_id} ValueForward source/output types differ")
        source_location = location(source.slot_id)
        output_location = location(output.slot_id)
        if source_location[2] != output_location[2]:
            errors.append(f"instruction {instruction.instruction_id} ValueForward source/output byte sizes differ")
        aliases = source_location == output_location
        if aliases and consumers.get(source.slot_id, set()) != {instruction.instruction_id}:
            errors.append(
                f"instruction {instruction.instruction_id} non-owning ValueForward alias requires "
                f"a unique source consumer, got {sorted(consumers.get(source.slot_id, set()))!r}"
            )
        if aliases:
            owner_slot_id = source.slot_id if source.allocation_owner_slot_id is None else source.allocation_owner_slot_id
            if forward.alias_owner_slot_id != owner_slot_id:
                errors.append(
                    f"instruction {instruction.instruction_id} ValueForward alias owner does not "
                    f"match source allocation owner {owner_slot_id}"
                )
            if output.allocation_owner_slot_id != owner_slot_id:
                errors.append(
                    f"instruction {instruction.instruction_id} ValueForward output does not retain allocation owner {owner_slot_id}"
                )
            if output.alias_of_slot_id != source.slot_id:
                errors.append(f"instruction {instruction.instruction_id} ValueForward output is not a non-owning alias of its source slot")
        if not aliases and forward.copy_owner_slot_id != output.slot_id:
            errors.append(f"instruction {instruction.instruction_id} ValueForward copy owner is not its output slot")


def live_ranges_overlap(
    first: DataflowSlotLiveRange | None,
    second: DataflowSlotLiveRange | None,
) -> bool:
    if first is None or second is None:
        return True
    first_ids = first.instruction_ids
    second_ids = second.instruction_ids
    if not first_ids or not second_ids:
        return True
    return min(first_ids) <= max(second_ids) and min(second_ids) <= max(first_ids)
