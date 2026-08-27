"""Runtime-facing ABI packing for Dataflow instruction plans."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from collections.abc import Hashable, Mapping

from .abi_schema import (
    ABI_VERSION,
    ARG_STRUCT,
    COMM_STRUCT,
    INSTRUCTION_STRUCT,
    DATAFLOW_COMM_KIND_ABI_VALUES,
    DATAFLOW_HANDOFF_FLAG_CONSUMER,
    DATAFLOW_HANDOFF_FLAG_DISABLED,
    DATAFLOW_HANDOFF_FLAG_PRODUCER,
    DATAFLOW_HANDOFF_FLAG_TAIL,
    DATAFLOW_INSTRUCTION_CLUSTER_COMM_COUNT_MASK,
    DATAFLOW_INSTRUCTION_CLUSTER_SEND_COUNT_SHIFT,
    DATAFLOW_OPCODE_ABI_VALUES,
    DATAFLOW_SLOT_ALIGNMENT,
    DATAFLOW_SLOT_FLAG_CLUSTER_GATED_PUSH,
    DATAFLOW_SLOT_FLAG_COMMUNICATE,
    DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL,
    DATAFLOW_SLOT_FLAG_SCRATCH_BACKED,
    SLOT_STRUCT,
    UINT32_SENTINEL,
    UINT32_STRUCT,
)
from .barrier_planning import DataflowBarrierAllocation, plan_dataflow_barriers
from .dtype_registry import dataflow_dtype_info
from .handler_identity import (
    DataflowHandlerIdentity,
    DataflowHandlerVariantKey,
    ReduceAritySpecialization,
)
from .joint_schedule import DataflowJointExecutionPlan
from .scheduler import CommPlan, Instruction, InstructionPlan, DataflowCommKind, DataflowOpcode, SlotPlan


DATAFLOW_OPCODE_TO_ABI = {opcode: DATAFLOW_OPCODE_ABI_VALUES[opcode.value] for opcode in DataflowOpcode}

DATAFLOW_COMM_KIND_TO_ABI = {kind: DATAFLOW_COMM_KIND_ABI_VALUES[kind.value] for kind in DataflowCommKind}


# PackedHandlerArgs.reserved0 carries a plan-derived source-lifetime action.
# Keeping the encoding zero-default preserves compatibility for hand-authored
# packed arguments and lets wrappers elide the check when the plan is not
# eligible for static cluster-source analysis.
DATAFLOW_CLUSTER_SOURCE_WAIT_NONE = 0
DATAFLOW_CLUSTER_SOURCE_WAIT_ALL = 1
DATAFLOW_CLUSTER_SOURCE_WAIT_PREVIOUS = 2


@dataclass(frozen=True)
class PackedInstruction:
    opcode: int
    handler_id: int
    task_id: int
    arg_offset: int
    comm_offset: int
    comm_count: int
    slot_id: int
    flags: int = 0

    def to_tuple(self) -> tuple[int, ...]:
        return (
            self.opcode,
            self.handler_id,
            self.task_id,
            self.arg_offset,
            self.comm_offset,
            self.comm_count,
            self.slot_id,
            self.flags,
        )


@dataclass(frozen=True)
class PackedSlot:
    shared_offset: int
    global_offset: int
    bytes: int
    flag_index: int
    barrier_index: int
    owner_cta: int
    flags: int = 0
    reserved: int = 0

    def to_tuple(self) -> tuple[int, ...]:
        return (
            self.shared_offset,
            self.global_offset,
            self.bytes,
            self.flag_index,
            self.barrier_index,
            self.owner_cta,
            self.flags,
            self.reserved,
        )

    @property
    def is_scratch_backed(self) -> bool:
        return bool(self.flags & DATAFLOW_SLOT_FLAG_SCRATCH_BACKED)


@dataclass(frozen=True)
class PackedComm:
    kind: int
    src_slot_id: int
    dst_slot_id: int
    peer_cta_rank: int
    barrier_phase: int = 0
    flag_index: int = UINT32_SENTINEL
    flag_epoch: int = 0
    barrier_index: int = UINT32_SENTINEL
    byte_offset: int = 0
    byte_count: int = 0
    segment_id: int = 0
    segment_count: int = 1

    def to_tuple(self) -> tuple[int, ...]:
        return (
            self.kind,
            self.src_slot_id,
            self.dst_slot_id,
            self.peer_cta_rank,
            self.barrier_phase,
            self.flag_index,
            self.flag_epoch,
            self.barrier_index,
            self.byte_offset,
            self.byte_count,
            self.segment_id,
            self.segment_count,
        )


@dataclass(frozen=True)
class PackedHandlerArgs:
    task_id: int
    task_coord_offset: int
    task_coord_count: int
    range_begin: int
    range_end: int
    input_slot_offset: int
    input_slot_count: int
    output_slot: int
    handoff_peer_arg_offset: int = UINT32_SENTINEL
    handoff_plan_index: int = UINT32_SENTINEL
    handoff_binding_index: int = UINT32_SENTINEL
    handoff_stage_count: int = 0
    handoff_arena_slot: int = UINT32_SENTINEL
    handoff_flags: int = 0
    reserved0: int = 0
    reserved1: int = 0

    def to_tuple(self) -> tuple[int, ...]:
        return (
            self.task_id,
            self.task_coord_offset,
            self.task_coord_count,
            self.range_begin,
            self.range_end,
            self.input_slot_offset,
            self.input_slot_count,
            self.output_slot,
            self.handoff_peer_arg_offset,
            self.handoff_plan_index,
            self.handoff_binding_index,
            self.handoff_stage_count,
            self.handoff_arena_slot,
            self.handoff_flags,
            self.reserved0,
            self.reserved1,
        )


@dataclass(frozen=True)
class PackedRuntimePlan:
    abi_version: int
    instruction_record_size: int
    slot_record_size: int
    comm_record_size: int
    arg_record_size: int
    operator_table: dict[str, int]
    operator_kinds: dict[str, str]
    handler_names: tuple[str, ...]
    handler_identities: tuple[DataflowHandlerIdentity | None, ...]
    handler_variant_keys: tuple[DataflowHandlerVariantKey | None, ...]
    intermediate_type_table: dict[str, int]
    queue_offsets: tuple[int, ...]
    queue_lengths: tuple[int, ...]
    input_slots: tuple[int, ...]
    task_coords: tuple[int, ...]
    instructions: tuple[PackedInstruction, ...]
    slots: tuple[PackedSlot, ...]
    comms: tuple[PackedComm, ...]
    args: tuple[PackedHandlerArgs, ...]
    barrier_allocation: DataflowBarrierAllocation
    joint_execution_plan: DataflowJointExecutionPlan | None
    instruction_bytes: bytes
    slot_bytes: bytes
    comm_bytes: bytes
    arg_bytes: bytes
    queue_offsets_bytes: bytes
    queue_lengths_bytes: bytes
    input_slots_bytes: bytes
    task_coords_bytes: bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "abi_version": self.abi_version,
            "instruction_record_size": self.instruction_record_size,
            "slot_record_size": self.slot_record_size,
            "comm_record_size": self.comm_record_size,
            "arg_record_size": self.arg_record_size,
            "operator_table": dict(self.operator_table),
            "operator_kinds": dict(self.operator_kinds),
            "handlers": [
                {
                    "handler_id": handler_id,
                    "operator_name": self.handler_names[handler_id],
                    "identity": None if identity is None else identity.to_dict(),
                    "variant_key": (
                        None if self.handler_variant_keys[handler_id] is None else self.handler_variant_keys[handler_id].to_dict()
                    ),
                }
                for handler_id, identity in enumerate(self.handler_identities)
            ],
            "intermediate_type_table": dict(self.intermediate_type_table),
            "queue_offsets": list(self.queue_offsets),
            "queue_lengths": list(self.queue_lengths),
            "input_slots": list(self.input_slots),
            "task_coords": list(self.task_coords),
            "instructions": [item.to_tuple() for item in self.instructions],
            "slots": [item.to_tuple() for item in self.slots],
            "comms": [item.to_tuple() for item in self.comms],
            "args": [item.to_tuple() for item in self.args],
            "barrier_allocation": self.barrier_allocation.to_dict(),
            "joint_execution_plan": (None if self.joint_execution_plan is None else self.joint_execution_plan.to_dict()),
        }

    def dump_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def u32_or_sentinel(value: int | None) -> int:
    if value is None:
        return UINT32_SENTINEL
    if value < 0 or value > UINT32_SENTINEL:
        raise ValueError(f"Dataflow ABI value must fit uint32, got {value}")
    return value


def pack_u32_array(values: tuple[int, ...]) -> bytes:
    return b"".join(UINT32_STRUCT.pack(value) for value in values)


def align_up(value: int, alignment: int = DATAFLOW_SLOT_ALIGNMENT) -> int:
    if value < 0:
        raise ValueError(f"Dataflow ABI byte size must be non-negative, got {value}")
    return ((value + alignment - 1) // alignment) * alignment


def get_dtype_nbytes(dtype: str | None) -> int:
    info = dataflow_dtype_info(dtype)
    return 0 if info is None else info.element_bytes


def shape_numel(shape: tuple[Any, ...] | None) -> int:
    if shape is None:
        return 1
    numel = 1
    for extent in shape:
        if not isinstance(extent, int):
            try:
                extent = int(extent)
            except (TypeError, ValueError):
                return 0
        if extent < 0:
            return 0
        numel *= extent
    return numel


def intermediate_nbytes(slot: SlotPlan) -> int:
    total = 0
    for field in slot.intermediate_type.fields:
        dtype_nbytes = get_dtype_nbytes(field.dtype)
        if dtype_nbytes == 0:
            return 0
        total = align_up(total, min(dtype_nbytes, DATAFLOW_SLOT_ALIGNMENT))
        field_nbytes = dtype_nbytes * shape_numel(field.shape)
        total += field_nbytes
    return align_up(total)


def operator_kind_for_instruction(instruction: Instruction) -> str | None:
    if instruction.opcode in (DataflowOpcode.EXIT, DataflowOpcode.CLUSTER_SYNC):
        return None
    if instruction.opcode is DataflowOpcode.RESHARED:
        return None
    if instruction.opcode is DataflowOpcode.REDUCE_UPDATE:
        return DataflowOpcode.REDUCE.value
    return instruction.opcode.value


def instruction_handler_key(instruction: Instruction) -> Hashable | None:
    operator_kind = operator_kind_for_instruction(instruction)
    if operator_kind is None:
        return None
    identity = instruction.handler_identity
    variant_key = instruction.handler_variant_key
    if identity is not None or variant_key is not None:
        if identity is None or variant_key is None:
            raise ValueError(
                "Dataflow structured handler dispatch requires both base identity "
                f"and variant key: instruction_id={instruction.instruction_id}"
            )
        if variant_key.base_identity != identity:
            raise ValueError(
                f"Dataflow instruction handler identity does not match variant base: instruction_id={instruction.instruction_id}"
            )
        if identity.operator_kind != operator_kind:
            raise ValueError(
                "Dataflow instruction handler kind does not match opcode: "
                f"instruction_id={instruction.instruction_id}, "
                f"opcode_kind={operator_kind!r}, identity_kind={identity.operator_kind!r}"
            )
        if operator_kind == "reduce":
            expected_arity = ReduceAritySpecialization.for_arity(len(instruction.input_slots))
            if variant_key.reduce_arity != expected_arity:
                raise ValueError(
                    "Dataflow reduce handler variant does not match instruction arity: "
                    f"instruction_id={instruction.instruction_id}, "
                    f"input_slots={len(instruction.input_slots)}, "
                    f"variant={variant_key.to_dict()!r}"
                )
        return ("variant", variant_key)
    return ("legacy", operator_kind, instruction.operator_name)


@dataclass(frozen=True)
class HandlerTable:
    names: tuple[str, ...]
    identities: tuple[DataflowHandlerIdentity | None, ...]
    variant_keys: tuple[DataflowHandlerVariantKey | None, ...]
    ids_by_key: Mapping[Hashable, int]
    operator_table: dict[str, int]
    operator_kinds: dict[str, str]


@dataclass(frozen=True)
class HandlerTableEntry:
    key: Hashable
    name: str
    identity: DataflowHandlerIdentity | None
    variant_key: DataflowHandlerVariantKey | None
    operator_kind: str


def build_handler_table(plan: InstructionPlan) -> HandlerTable:
    entries_by_key: dict[Hashable, HandlerTableEntry] = {}
    primary_entries: list[HandlerTableEntry] = []
    additional_variant_entries: list[HandlerTableEntry] = []
    seen_binding_keys: set[Hashable] = set()
    operator_table: dict[str, int] = {}
    operator_kinds: dict[str, str] = {}
    for instruction in plan.instructions:
        key = instruction_handler_key(instruction)
        if key is None or key in entries_by_key:
            continue
        identity = instruction.handler_identity
        operator_kind = identity.operator_kind if identity is not None else operator_kind_for_instruction(instruction)
        assert operator_kind is not None
        entry = HandlerTableEntry(
            key=key,
            name=instruction.operator_name,
            identity=identity,
            variant_key=instruction.handler_variant_key,
            operator_kind=operator_kind,
        )
        entries_by_key[key] = entry
        binding_key: Hashable = ("structured", identity.binding_key) if identity is not None else ("legacy", key)
        if binding_key in seen_binding_keys:
            additional_variant_entries.append(entry)
        else:
            seen_binding_keys.add(binding_key)
            primary_entries.append(entry)

    ordered_entries = (*primary_entries, *additional_variant_entries)
    ids_by_key = {entry.key: handler_id for handler_id, entry in enumerate(ordered_entries)}
    for handler_id, entry in enumerate(ordered_entries):
        operator_table.setdefault(entry.name, handler_id)
        operator_kinds.setdefault(entry.name, entry.operator_kind)
    return HandlerTable(
        names=tuple(entry.name for entry in ordered_entries),
        identities=tuple(entry.identity for entry in ordered_entries),
        variant_keys=tuple(entry.variant_key for entry in ordered_entries),
        ids_by_key=ids_by_key,
        operator_table=operator_table,
        operator_kinds=operator_kinds,
    )


def build_intermediate_type_table(slots: tuple[SlotPlan, ...]) -> dict[str, int]:
    names = []
    seen = set()
    for slot in slots:
        name = slot.intermediate_type.name
        if name not in seen:
            names.append(name)
            seen.add(name)
    return {name: index for index, name in enumerate(names)}


def order_comms_by_dispatch(
    comms: tuple[CommPlan, ...],
    ordered_instructions: tuple[Instruction, ...],
) -> tuple[tuple[CommPlan, ...], dict[int, tuple[int, int, int]]]:
    grouped: dict[int, list[CommPlan]] = {}
    for comm in comms:
        grouped.setdefault(comm.resolved_dispatch_instruction_id, []).append(comm)

    ordered: list[CommPlan] = []
    offsets: dict[int, tuple[int, int, int]] = {}
    for instruction in ordered_instructions:
        instruction_comms = grouped.pop(instruction.instruction_id, [])
        if not instruction_comms:
            continue
        instruction_comms.sort(
            key=lambda comm: (
                {
                    DataflowCommKind.CLUSTER_RECV: 0,
                    DataflowCommKind.CLUSTER_SEND: 1,
                    DataflowCommKind.CLUSTER_RELEASE: 1,
                    DataflowCommKind.HBM_RECV_ISSUE: 2,
                    DataflowCommKind.HBM_RECV: 3,
                    DataflowCommKind.HBM_RECV_WAIT: 3,
                }.get(comm.kind, 2),
                comm.source_instruction_id,
                comm.target_instruction_id,
                comm.source_slot_id,
                comm.target_slot_id,
                comm.segment_id,
            )
        )
        cluster_recv_count = sum(comm.kind is DataflowCommKind.CLUSTER_RECV for comm in instruction_comms)
        cluster_send_count = sum(
            comm.kind in (DataflowCommKind.CLUSTER_SEND, DataflowCommKind.CLUSTER_RELEASE) for comm in instruction_comms
        )
        if (
            cluster_recv_count > DATAFLOW_INSTRUCTION_CLUSTER_COMM_COUNT_MASK
            or cluster_send_count > DATAFLOW_INSTRUCTION_CLUSTER_COMM_COUNT_MASK
        ):
            raise ValueError(
                "Dataflow instruction cluster communication count exceeds the packed ABI limit: "
                f"instruction_id={instruction.instruction_id}, "
                f"cluster_recv_count={cluster_recv_count}, "
                f"cluster_send_count={cluster_send_count}"
            )
        comm_flags = cluster_recv_count | (cluster_send_count << DATAFLOW_INSTRUCTION_CLUSTER_SEND_COUNT_SHIFT)
        offsets[instruction.instruction_id] = (
            len(ordered),
            len(instruction_comms),
            comm_flags,
        )
        ordered.extend(instruction_comms)

    if grouped:
        missing = sorted(grouped)
        raise ValueError(f"Dataflow comm dispatch instruction ids are not in the instruction plan: {missing}")
    return tuple(ordered), offsets


def comm_transfer_key(comm: CommPlan) -> tuple[int, int, int, int, int]:
    return (
        comm.source_instruction_id,
        comm.target_instruction_id,
        comm.source_slot_id,
        comm.target_slot_id,
        comm.segment_id,
    )


def comm_cluster_gate_key(comm: CommPlan) -> tuple[str, Any]:
    if comm.cluster_gate_id is not None:
        return ("group", comm.cluster_gate_id)
    return ("transfer", comm_transfer_key(comm))


def pack_comms(
    comms: tuple[CommPlan, ...],
    *,
    cluster_size: int,
    reuse_hbm_flags: bool = False,
    cluster_barrier_phases: Mapping[tuple[int, int, int, int, int], int] | None = None,
    receive_barrier_indices: Mapping[tuple[int, int, int, int, int], int] | None = None,
) -> tuple[tuple[PackedComm, ...], bytes]:
    if cluster_size <= 0:
        raise ValueError(f"cluster_size must be positive, got {cluster_size}")
    hbm_transfer_epochs: dict[tuple[int, int, int, int, int], int] = {}
    cluster_barrier_phases = {} if cluster_barrier_phases is None else cluster_barrier_phases
    receive_barrier_indices = {} if receive_barrier_indices is None else receive_barrier_indices
    # A cluster ACK is a reverse-direction lifetime gate: the producer may
    # reuse a gated outbox only after the consumer emits CLUSTER_RELEASE.
    # Ordinary cluster push transfers already use their receive barrier for
    # arrival and TMA source-read completion for source lifetime, so assigning
    # an ACK barrier to every cluster send/receive only wastes shared memory.
    cluster_ack_transfer_keys = frozenset(comm_cluster_gate_key(comm) for comm in comms if comm.kind is DataflowCommKind.CLUSTER_RELEASE)
    cluster_ack_indices: dict[tuple[str, Any], int] = {}
    cluster_ack_counts_by_producer: dict[int, int] = {}
    cluster_send_producers = {comm_cluster_gate_key(comm): comm.producer_sm for comm in comms if comm.kind is DataflowCommKind.CLUSTER_SEND}

    # ACK barriers live in producer-local shared memory.  Indices therefore
    # only need to be unique among gates owned by the same producer CTA; the
    # reverse release targets that CTA and uses the matching local index.
    for comm in comms:
        key = comm_cluster_gate_key(comm)
        if key not in cluster_ack_transfer_keys or key in cluster_ack_indices:
            continue
        producer_sm = cluster_send_producers.get(key)
        if producer_sm is None:
            raise ValueError(f"Dataflow cluster release has no matching cluster send: transfer={key!r}")
        index = cluster_ack_counts_by_producer.get(producer_sm, 0)
        cluster_ack_indices[key] = index
        cluster_ack_counts_by_producer[producer_sm] = index + 1

    def hbm_epoch(comm: CommPlan) -> int:
        if not reuse_hbm_flags:
            return comm.flag_epoch
        key = comm_transfer_key(comm)
        epoch = hbm_transfer_epochs.get(key)
        if epoch is None:
            epoch = len(hbm_transfer_epochs) + 1
            hbm_transfer_epochs[key] = epoch
        return epoch

    packed = tuple(
        PackedComm(
            kind=DATAFLOW_COMM_KIND_TO_ABI[comm.kind],
            src_slot_id=comm.source_slot_id,
            dst_slot_id=comm.target_slot_id,
            peer_cta_rank=(
                u32_or_sentinel(
                    comm.peer_cta_rank
                    if comm.peer_cta_rank is not None
                    else (comm.consumer_sm if comm.kind is DataflowCommKind.CLUSTER_SEND else comm.producer_sm) % cluster_size
                )
                if comm.kind
                in (
                    DataflowCommKind.CLUSTER_SEND,
                    DataflowCommKind.CLUSTER_RECV,
                    DataflowCommKind.CLUSTER_RELEASE,
                )
                else UINT32_SENTINEL
            ),
            barrier_phase=cluster_barrier_phases.get(
                comm_transfer_key(comm),
                comm.barrier_phase,
            ),
            flag_index=(
                cluster_ack_indices.get(
                    comm_cluster_gate_key(comm),
                    UINT32_SENTINEL,
                )
                if comm.kind
                in (
                    DataflowCommKind.CLUSTER_SEND,
                    DataflowCommKind.CLUSTER_RECV,
                    DataflowCommKind.CLUSTER_RELEASE,
                )
                else (
                    0
                    if reuse_hbm_flags
                    and comm.kind
                    in (
                        DataflowCommKind.HBM_SEND,
                        DataflowCommKind.HBM_RECV,
                        DataflowCommKind.HBM_RECV_ISSUE,
                        DataflowCommKind.HBM_RECV_WAIT,
                    )
                    else UINT32_SENTINEL
                )
            ),
            flag_epoch=(
                hbm_epoch(comm)
                if comm.kind
                in (
                    DataflowCommKind.HBM_SEND,
                    DataflowCommKind.HBM_RECV,
                    DataflowCommKind.HBM_RECV_ISSUE,
                    DataflowCommKind.HBM_RECV_WAIT,
                )
                else cluster_barrier_phases.get(comm_transfer_key(comm), comm.barrier_phase)
            ),
            barrier_index=(
                receive_barrier_indices.get(
                    comm_transfer_key(comm),
                    UINT32_SENTINEL,
                )
                if comm.kind
                in (
                    DataflowCommKind.HBM_RECV,
                    DataflowCommKind.HBM_RECV_ISSUE,
                    DataflowCommKind.HBM_RECV_WAIT,
                )
                else UINT32_SENTINEL
            ),
            byte_offset=comm.byte_offset,
            byte_count=comm.byte_count,
            segment_id=comm.segment_id,
            segment_count=comm.segment_count,
        )
        for comm in comms
    )
    return packed, b"".join(COMM_STRUCT.pack(*item.to_tuple()) for item in packed)


def producer_ctas(plan: InstructionPlan) -> dict[int, int]:
    return {instruction.instruction_id: instruction.sm_id for instruction in plan.instructions if instruction.sm_id is not None}


def pack_slots(
    slots: tuple[SlotPlan, ...],
    producer_ctas: dict[int, int],
    *,
    cluster_recv_barriers: dict[int, int] | None = None,
    reuse_hbm_flags: bool = False,
    hbm_direct_global_slots: frozenset[int] = frozenset(),
) -> tuple[tuple[PackedSlot, ...], bytes]:
    packed = []
    cluster_recv_barriers = {} if cluster_recv_barriers is None else cluster_recv_barriers
    next_shared_offset = 0
    next_global_offset = 0
    shared_offsets: dict[tuple[str, int], int] = {}
    global_offsets: dict[tuple[str, int], int] = {}
    for slot in slots:
        slot_bytes = intermediate_nbytes(slot)
        slot_stride = max(slot_bytes, DATAFLOW_SLOT_ALIGNMENT)
        slot_uses_hbm_direct_global = slot.slot_id in hbm_direct_global_slots
        shared_key = ("shared", slot.shared_storage_id) if slot.shared_storage_id is not None else ("slot", slot.slot_id)
        global_key = ("global", slot.global_storage_id) if slot.global_storage_id is not None else ("slot", slot.slot_id)

        slot_flags = 0
        slot_reserved = 0
        if slot.scratch_backed:
            if slot.scratch_offset is None:
                raise ValueError(f"Dataflow scratch-backed slot {slot.slot_id} requires a scratch_offset")
            shared_offset = align_up(slot.scratch_offset, DATAFLOW_SLOT_ALIGNMENT)
            if shared_offset != slot.scratch_offset:
                raise ValueError(
                    f"Dataflow scratch-backed slot {slot.slot_id} scratch_offset must be "
                    f"{DATAFLOW_SLOT_ALIGNMENT}-byte aligned, got {slot.scratch_offset}"
                )
            slot_flags |= DATAFLOW_SLOT_FLAG_SCRATCH_BACKED
            slot_reserved = shared_offset
        elif slot_uses_hbm_direct_global:
            shared_offset = 0
        else:
            shared_offset = shared_offsets.get(shared_key)
            if shared_offset is None:
                shared_offset = next_shared_offset
                shared_offsets[shared_key] = shared_offset
                next_shared_offset += align_up(slot_stride)
        if slot_uses_hbm_direct_global:
            slot_flags |= DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL
        if slot.cluster_gated_push or slot.role == "cluster_gated_inbox":
            slot_flags |= DATAFLOW_SLOT_FLAG_CLUSTER_GATED_PUSH
        if slot.role in {
            "cluster_inbox",
            "cluster_gated_inbox",
            "hbm_spill_inbox",
            "joint_comm_inbox",
            "joint_comm_outbox",
            "joint_prefetch_inbox",
        }:
            if not slot.scratch_backed:
                raise ValueError(f"Dataflow communicate slot {slot.slot_id} must be scratch-backed")
            slot_flags |= DATAFLOW_SLOT_FLAG_COMMUNICATE

        global_offset = global_offsets.get(global_key)
        if global_offset is None:
            global_offset = next_global_offset
            global_offsets[global_key] = global_offset
            next_global_offset += align_up(slot_stride)

        packed.append(
            PackedSlot(
                shared_offset=shared_offset,
                global_offset=global_offset,
                bytes=slot_bytes,
                flag_index=0 if reuse_hbm_flags else slot.slot_id,
                barrier_index=cluster_recv_barriers.get(slot.slot_id, UINT32_SENTINEL),
                owner_cta=u32_or_sentinel(
                    slot.physical_owner_cta
                    if slot.physical_owner_cta is not None
                    else (None if slot.producer_instruction_id is None else producer_ctas.get(slot.producer_instruction_id))
                ),
                flags=slot_flags,
                reserved=slot_reserved,
            )
        )
    packed_tuple = tuple(packed)
    return packed_tuple, b"".join(SLOT_STRUCT.pack(*item.to_tuple()) for item in packed_tuple)


def collect_hbm_direct_global_slot_ids(
    slots: tuple[SlotPlan, ...],
    comms: tuple[CommPlan, ...],
) -> frozenset[int]:
    has_hbm_comm = any(
        comm.kind
        in (
            DataflowCommKind.HBM_SEND,
            DataflowCommKind.HBM_RECV,
            DataflowCommKind.HBM_RECV_ISSUE,
            DataflowCommKind.HBM_RECV_WAIT,
        )
        for comm in comms
    )
    has_cluster_comm = any(comm.kind in (DataflowCommKind.CLUSTER_SEND, DataflowCommKind.CLUSTER_RECV) for comm in comms)
    if has_hbm_comm and not has_cluster_comm:
        return frozenset(slot.slot_id for slot in slots)

    slot_comm_kinds: dict[int, set[DataflowCommKind]] = {}
    for comm in comms:
        if comm.kind not in (
            DataflowCommKind.CLUSTER_SEND,
            DataflowCommKind.CLUSTER_RECV,
            DataflowCommKind.HBM_SEND,
            DataflowCommKind.HBM_RECV,
            DataflowCommKind.HBM_RECV_ISSUE,
            DataflowCommKind.HBM_RECV_WAIT,
        ):
            continue
        slot_comm_kinds.setdefault(comm.source_slot_id, set()).add(comm.kind)
        slot_comm_kinds.setdefault(comm.target_slot_id, set()).add(comm.kind)

    direct_kinds = {
        DataflowCommKind.HBM_SEND,
        DataflowCommKind.HBM_RECV,
        DataflowCommKind.HBM_RECV_ISSUE,
        DataflowCommKind.HBM_RECV_WAIT,
    }
    direct_slot_ids = {slot_id for slot_id, kinds in slot_comm_kinds.items() if kinds and kinds.issubset(direct_kinds)}
    direct_shared_storage_ids = {
        slot.shared_storage_id for slot in slots if slot.slot_id in direct_slot_ids and slot.shared_storage_id is not None
    }
    direct_global_storage_ids = {
        slot.global_storage_id for slot in slots if slot.slot_id in direct_slot_ids and slot.global_storage_id is not None
    }
    direct_slot_ids.update(
        slot.slot_id
        for slot in slots
        if (
            (slot.shared_storage_id is not None and slot.shared_storage_id in direct_shared_storage_ids)
            or (slot.global_storage_id is not None and slot.global_storage_id in direct_global_storage_ids)
        )
    )
    return frozenset(direct_slot_ids)


def queue_offsets_and_lengths(plan: InstructionPlan) -> tuple[tuple[int, ...], tuple[int, ...]]:
    offsets = []
    lengths = []
    offset = 0
    for sm_id in range(plan.topology.sm_count):
        queue = plan.queue(sm_id)
        offsets.append(offset)
        lengths.append(len(queue))
        offset += len(queue)
    return tuple(offsets), tuple(lengths)


def instruction_pack_order(plan: InstructionPlan) -> tuple[Instruction, ...]:
    ordered: list[Instruction] = []
    seen: set[int] = set()
    for sm_id in range(plan.topology.sm_count):
        for instruction in plan.queue(sm_id):
            ordered.append(instruction)
            seen.add(instruction.instruction_id)

    for instruction in plan.instructions:
        if instruction.instruction_id not in seen:
            ordered.append(instruction)
    return tuple(ordered)


def pack_instruction_args(
    ordered_instructions: tuple[Instruction, ...],
    *,
    handler_ids: Mapping[Hashable, int],
    comm_offsets: dict[int, tuple[int, int, int]],
    handoff_plan_indices: Mapping[str, int],
    cluster_source_wait_actions: Mapping[int, int] | None = None,
) -> tuple[
    tuple[PackedInstruction, ...],
    tuple[PackedHandlerArgs, ...],
    tuple[int, ...],
    tuple[int, ...],
    bytes,
    bytes,
    bytes,
    bytes,
]:
    cluster_source_wait_actions = {} if cluster_source_wait_actions is None else cluster_source_wait_actions
    input_slots: list[int] = []
    task_coords: list[int] = []
    packed: list[PackedInstruction] = []
    args: list[PackedHandlerArgs] = []
    input_slot_offsets: dict[int, tuple[int, int]] = {}
    task_coord_offsets: dict[int, tuple[int, int]] = {}

    arg_offset_by_instruction_id: dict[int, int] = {}
    next_arg_index = 0
    for instruction in ordered_instructions:
        if instruction.opcode is DataflowOpcode.EXIT:
            continue
        arg_offset_by_instruction_id[instruction.instruction_id] = next_arg_index * ARG_STRUCT.size
        next_arg_index += 1

    for instruction in ordered_instructions:
        input_slot_offsets[instruction.instruction_id] = (len(input_slots), len(instruction.input_slots))
        input_slots.extend(instruction.input_slots)
        task_coord_offsets[instruction.instruction_id] = (len(task_coords), len(instruction.task_coords))
        task_coords.extend(instruction.task_coords)

    for instruction in ordered_instructions:
        arg_offset = UINT32_SENTINEL
        if instruction.opcode is not DataflowOpcode.EXIT:
            task_range = instruction.task_range
            input_slot_offset, input_slot_count = input_slot_offsets[instruction.instruction_id]
            task_coord_offset, task_coord_count = task_coord_offsets[instruction.instruction_id]
            arg_offset = len(args) * ARG_STRUCT.size
            attrs = instruction.attrs
            stage_count = int(attrs.get("cross_handler_handoff_stage_count", 0))
            if stage_count < 0 or stage_count >= UINT32_SENTINEL:
                raise ValueError(f"Dataflow cross-handler handoff stage count must fit uint32, got {stage_count}")
            role = attrs.get("cross_handler_handoff_role")
            state = attrs.get("cross_handler_handoff_state")
            handoff_flags = 0
            if role == "producer":
                handoff_flags |= DATAFLOW_HANDOFF_FLAG_PRODUCER
            elif role == "consumer":
                handoff_flags |= DATAFLOW_HANDOFF_FLAG_CONSUMER
            if state == "tail":
                handoff_flags |= DATAFLOW_HANDOFF_FLAG_TAIL
            elif state == "disabled":
                handoff_flags |= DATAFLOW_HANDOFF_FLAG_DISABLED
            peer_instruction_id = attrs.get("cross_handler_handoff_peer_instruction_id")
            peer_arg_offset = (
                UINT32_SENTINEL
                if peer_instruction_id is None
                else arg_offset_by_instruction_id.get(
                    int(peer_instruction_id),
                    UINT32_SENTINEL,
                )
            )
            plan_fingerprint = attrs.get("cross_handler_handoff_plan_fingerprint")
            plan_index = UINT32_SENTINEL if plan_fingerprint is None else handoff_plan_indices[str(plan_fingerprint)]
            args.append(
                PackedHandlerArgs(
                    task_id=u32_or_sentinel(instruction.task_id),
                    task_coord_offset=task_coord_offset,
                    task_coord_count=task_coord_count,
                    range_begin=0 if task_range is None else task_range.begin,
                    range_end=0 if task_range is None else task_range.end,
                    input_slot_offset=input_slot_offset,
                    input_slot_count=input_slot_count,
                    output_slot=u32_or_sentinel(instruction.output_slot),
                    handoff_peer_arg_offset=peer_arg_offset,
                    handoff_plan_index=plan_index,
                    handoff_binding_index=u32_or_sentinel(attrs.get("cross_handler_handoff_binding_id")),
                    handoff_stage_count=stage_count,
                    handoff_arena_slot=u32_or_sentinel(attrs.get("cross_handler_handoff_arena_slot")),
                    handoff_flags=handoff_flags,
                    reserved0=int(
                        cluster_source_wait_actions.get(
                            instruction.instruction_id,
                            DATAFLOW_CLUSTER_SOURCE_WAIT_NONE,
                        )
                    ),
                )
            )

        comm_offset, comm_count, comm_flags = comm_offsets.get(instruction.instruction_id, (0, 0, 0))
        handler_key = instruction_handler_key(instruction)
        packed.append(
            PackedInstruction(
                opcode=DATAFLOW_OPCODE_TO_ABI[instruction.opcode],
                handler_id=(UINT32_SENTINEL if handler_key is None else handler_ids[handler_key]),
                task_id=u32_or_sentinel(instruction.task_id),
                arg_offset=arg_offset,
                comm_offset=comm_offset,
                comm_count=comm_count,
                slot_id=u32_or_sentinel(
                    instruction.output_slot
                    if instruction.output_slot is not None
                    else (instruction.input_slots[0] if instruction.input_slots else None)
                ),
                flags=comm_flags,
            )
        )

    input_slot_tuple = tuple(input_slots)
    task_coord_tuple = tuple(task_coords)
    args_tuple = tuple(args)
    return (
        tuple(packed),
        args_tuple,
        input_slot_tuple,
        task_coord_tuple,
        b"".join(INSTRUCTION_STRUCT.pack(*item.to_tuple()) for item in packed),
        b"".join(ARG_STRUCT.pack(*item.to_tuple()) for item in args_tuple),
        pack_u32_array(input_slot_tuple),
        pack_u32_array(task_coord_tuple),
    )


def collect_cluster_source_wait_actions(
    plan: InstructionPlan,
    *,
    ordered_comms: tuple[CommPlan, ...],
    comm_offsets: Mapping[int, tuple[int, int, int]],
    packed_slots: tuple[PackedSlot, ...],
) -> dict[int, int]:
    """Precompute ordered bulk-group source lifetime fences per CTA queue.

    Each cluster-send instruction commits one ordered TMA bulk group.  A later
    shared-memory write can retain the newest non-overlapping group with
    ``wait_group.read 1``; if it aliases the newest group it must wait for all
    groups.  Storage spaces are kept distinct because PrimFunc scratch and
    ordinary slots are placed in disjoint launch-package arenas.
    """

    def slot_range(slot_id: int) -> tuple[bool, int, int]:
        slot = packed_slots[slot_id]
        return (slot.is_scratch_backed, slot.shared_offset, slot.shared_offset + slot.bytes)

    def overlaps(
        lhs: tuple[bool, int, int],
        rhs: tuple[bool, int, int],
    ) -> bool:
        return lhs[0] == rhs[0] and lhs[1] < rhs[2] and rhs[1] < lhs[2]

    def instruction_comms(instruction_id: int) -> tuple[CommPlan, ...]:
        offset, count, _ = comm_offsets.get(instruction_id, (0, 0, 0))
        return ordered_comms[offset : offset + count]

    actions: dict[int, int] = {}
    receive_kinds = {
        DataflowCommKind.CLUSTER_RECV,
        DataflowCommKind.HBM_RECV,
        DataflowCommKind.HBM_RECV_ISSUE,
        DataflowCommKind.HBM_RECV_WAIT,
    }

    def write_ranges(instruction: Instruction) -> list[tuple[bool, int, int]]:
        comms = instruction_comms(instruction.instruction_id)
        writes = [] if instruction.output_slot is None else [slot_range(instruction.output_slot)]
        writes.extend(slot_range(comm.target_slot_id) for comm in comms if comm.kind in receive_kinds)
        return writes

    for queue in plan.queues.values():
        previous_groups: list[tuple[bool, int, int]] = []
        latest_group: list[tuple[bool, int, int]] = []
        for position, instruction in enumerate(queue):
            comms = instruction_comms(instruction.instruction_id)
            writes = write_ranges(instruction)
            send_group = [slot_range(comm.source_slot_id) for comm in comms if comm.kind is DataflowCommKind.CLUSTER_SEND]
            aliases_latest = any(overlaps(write, source) for write in writes for source in latest_group)
            aliases_previous = any(overlaps(write, source) for write in writes for source in previous_groups)
            if aliases_latest:
                actions[instruction.instruction_id] = DATAFLOW_CLUSTER_SOURCE_WAIT_ALL
                previous_groups = []
                latest_group = []
            elif aliases_previous:
                next_writes = [] if position + 1 >= len(queue) else write_ranges(queue[position + 1])
                next_reuses_latest = any(overlaps(write, source) for write in next_writes for source in latest_group)
                next_reuses_new_group = any(overlaps(write, source) for write in next_writes for source in send_group)
                collapse_groups = next_reuses_latest and not next_reuses_new_group
                actions[instruction.instruction_id] = (
                    DATAFLOW_CLUSTER_SOURCE_WAIT_ALL if collapse_groups else DATAFLOW_CLUSTER_SOURCE_WAIT_PREVIOUS
                )
                previous_groups = []
                if collapse_groups:
                    latest_group = []

            if send_group:
                previous_groups.extend(latest_group)
                latest_group = send_group
    return actions


def pack_instruction_plan(
    plan: InstructionPlan,
    *,
    reuse_hbm_flags: bool = False,
    hbm_direct_global: bool = False,
    hbm_direct_global_slot_ids: frozenset[int] = frozenset(),
) -> PackedRuntimePlan:
    if not isinstance(plan, InstructionPlan):
        raise TypeError(f"pack_instruction_plan expects InstructionPlan, got {plan!r}")
    validate_joint_execution_plan(plan)

    handler_table = build_handler_table(plan)
    intermediate_type_table = build_intermediate_type_table(plan.slots)
    ordered_instructions = instruction_pack_order(plan)
    ordered_comms, comm_offsets = order_comms_by_dispatch(plan.comms, ordered_instructions)
    if hbm_direct_global and hbm_direct_global_slot_ids:
        raise ValueError("Dataflow runtime plan cannot combine full and selective HBM direct-global slots")
    known_slot_ids = {slot.slot_id for slot in plan.slots}
    unknown_direct_slots = set(hbm_direct_global_slot_ids) - known_slot_ids
    if unknown_direct_slots:
        raise ValueError(f"Dataflow selective HBM direct-global slots must exist in the plan; unknown={sorted(unknown_direct_slots)!r}")
    hbm_comm_slot_ids = {
        slot_id
        for comm in ordered_comms
        if comm.kind
        in (
            DataflowCommKind.HBM_SEND,
            DataflowCommKind.HBM_RECV,
            DataflowCommKind.HBM_RECV_ISSUE,
            DataflowCommKind.HBM_RECV_WAIT,
        )
        for slot_id in (comm.source_slot_id, comm.target_slot_id)
    }
    non_hbm_direct_slots = set(hbm_direct_global_slot_ids) - hbm_comm_slot_ids
    if non_hbm_direct_slots:
        raise ValueError(
            f"Dataflow selective HBM direct-global slots must participate in HBM communication; invalid={sorted(non_hbm_direct_slots)!r}"
        )
    hbm_direct_global_slots = (
        collect_hbm_direct_global_slot_ids(plan.slots, ordered_comms) if hbm_direct_global else frozenset(hbm_direct_global_slot_ids)
    )
    barrier_allocation = plan_dataflow_barriers(
        plan,
        hbm_direct_global_slots=hbm_direct_global_slots,
    )
    packed_comms, comm_bytes = pack_comms(
        ordered_comms,
        cluster_size=plan.topology.cluster_size,
        reuse_hbm_flags=reuse_hbm_flags,
        cluster_barrier_phases=barrier_allocation.transfer_phases,
        receive_barrier_indices=barrier_allocation.transfer_barrier_indices,
    )
    packed_slots, slot_bytes = pack_slots(
        plan.slots,
        producer_ctas(plan),
        cluster_recv_barriers=barrier_allocation.slot_barrier_indices,
        reuse_hbm_flags=reuse_hbm_flags,
        hbm_direct_global_slots=hbm_direct_global_slots,
    )
    cluster_source_wait_actions = collect_cluster_source_wait_actions(
        plan,
        ordered_comms=ordered_comms,
        comm_offsets=comm_offsets,
        packed_slots=packed_slots,
    )
    handoff_plan_indices = {handoff_plan.fingerprint: index for index, handoff_plan in enumerate(plan.cross_handler_handoff_plans)}
    (
        packed_instructions,
        packed_args,
        input_slots,
        task_coords,
        instruction_bytes,
        arg_bytes,
        input_slot_bytes,
        task_coord_bytes,
    ) = pack_instruction_args(
        ordered_instructions,
        handler_ids=handler_table.ids_by_key,
        comm_offsets=comm_offsets,
        handoff_plan_indices=handoff_plan_indices,
        cluster_source_wait_actions=cluster_source_wait_actions,
    )
    queue_offsets, queue_lengths = queue_offsets_and_lengths(plan)

    return PackedRuntimePlan(
        abi_version=ABI_VERSION,
        instruction_record_size=INSTRUCTION_STRUCT.size,
        slot_record_size=SLOT_STRUCT.size,
        comm_record_size=COMM_STRUCT.size,
        arg_record_size=ARG_STRUCT.size,
        operator_table=handler_table.operator_table,
        operator_kinds=handler_table.operator_kinds,
        handler_names=handler_table.names,
        handler_identities=handler_table.identities,
        handler_variant_keys=handler_table.variant_keys,
        intermediate_type_table=intermediate_type_table,
        queue_offsets=queue_offsets,
        queue_lengths=queue_lengths,
        input_slots=input_slots,
        task_coords=task_coords,
        instructions=packed_instructions,
        slots=packed_slots,
        comms=packed_comms,
        args=packed_args,
        barrier_allocation=barrier_allocation,
        joint_execution_plan=plan.joint_execution_plan,
        instruction_bytes=instruction_bytes,
        slot_bytes=slot_bytes,
        comm_bytes=comm_bytes,
        arg_bytes=arg_bytes,
        queue_offsets_bytes=pack_u32_array(queue_offsets),
        queue_lengths_bytes=pack_u32_array(queue_lengths),
        input_slots_bytes=input_slot_bytes,
        task_coords_bytes=task_coord_bytes,
    )


def validate_joint_execution_plan(plan: InstructionPlan) -> None:
    joint_plan = plan.joint_execution_plan
    if joint_plan is None:
        return
    joint_plan.require_valid(topology=plan.topology)
    nodes_by_id = {node.node_id: node for node in joint_plan.nodes}
    expected_instructions = {
        instruction.instruction_id: instruction
        for instruction in plan.instructions
        if instruction.opcode not in {DataflowOpcode.EXIT, DataflowOpcode.CLUSTER_SYNC}
    }
    if set(nodes_by_id) != set(expected_instructions):
        raise ValueError("Dataflow joint execution nodes must match executable instructions")
    for instruction_id, instruction in expected_instructions.items():
        node = nodes_by_id[instruction_id]
        if instruction.sm_id is None or node.cta_id != instruction.sm_id:
            raise ValueError(f"Dataflow joint execution node ownership differs from its instruction: instruction_id={instruction_id}")
    for cta_id in range(plan.topology.sm_count):
        instruction_ids = tuple(
            instruction.instruction_id
            for instruction in plan.queue(cta_id)
            if instruction.opcode not in {DataflowOpcode.EXIT, DataflowOpcode.CLUSTER_SYNC}
        )
        if joint_plan.schedule.queue(cta_id) != instruction_ids:
            raise ValueError(f"Dataflow joint execution queue differs from the emitted instruction queue: cta={cta_id}")
