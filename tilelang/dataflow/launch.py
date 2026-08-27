"""Host-side launch package construction for Dataflow wrapper skeletons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .abi_schema import (
    ABI_VERSION,
    DATAFLOW_BARRIER_BYTES,
    DATAFLOW_CLUSTER_LOAD_BALANCING_IMPLEMENTATION as DATAFLOW_CLUSTER_LOAD_BALANCING_IMPLEMENTATION,
    DATAFLOW_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING,
    DATAFLOW_COMM_KIND_ABI_VALUES,
    DATAFLOW_SHARED_ALIGNMENT,
)
from .runtime import (
    ARG_STRUCT,
    COMM_STRUCT,
    INSTRUCTION_STRUCT,
    DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL,
    DATAFLOW_SLOT_FLAG_SCRATCH_BACKED,
    SLOT_STRUCT,
    UINT32_STRUCT,
    UINT32_SENTINEL,
    PackedRuntimePlan,
)


UINT32_BYTES = 4


def pack_u32_array(values: tuple[int, ...]) -> bytes:
    return b"".join(UINT32_STRUCT.pack(int(value)) for value in values)


@dataclass(frozen=True)
class DataflowQueueLaunchBuffers:
    instructions_bytes: bytes
    offsets_bytes: bytes
    lengths_bytes: bytes
    barrier_init_offsets: tuple[int, ...]
    barrier_init_lengths: tuple[int, ...]
    barrier_init_indices: tuple[int, ...]
    barrier_init_offsets_bytes: bytes
    barrier_init_lengths_bytes: bytes
    barrier_init_indices_bytes: bytes
    instruction_count: int
    queue_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction_bytes": len(self.instructions_bytes),
            "offsets_bytes": len(self.offsets_bytes),
            "lengths_bytes": len(self.lengths_bytes),
            "barrier_init_offsets": list(self.barrier_init_offsets),
            "barrier_init_lengths": list(self.barrier_init_lengths),
            "barrier_init_indices": list(self.barrier_init_indices),
            "barrier_init_offsets_bytes": len(self.barrier_init_offsets_bytes),
            "barrier_init_lengths_bytes": len(self.barrier_init_lengths_bytes),
            "barrier_init_indices_bytes": len(self.barrier_init_indices_bytes),
            "instruction_count": self.instruction_count,
            "queue_count": self.queue_count,
        }


@dataclass(frozen=True)
class DataflowLaunchPackage:
    abi_version: int
    queue: DataflowQueueLaunchBuffers
    slots_bytes: bytes
    comms_bytes: bytes
    args_bytes: bytes
    input_slots_bytes: bytes
    task_coords_bytes: bytes
    global_staging_bytes: bytes
    flags_bytes: bytes
    shared_memory_bytes: int
    shared_slot_bytes: int
    scratch_backed_slot_count: int
    scratch_backed_slot_bytes: int
    shared_slot_base_offset: int
    barrier_count: int
    barrier_bytes: int
    cluster_ack_count: int
    cluster_ack_barrier_bytes: int
    cluster_ack_pending_bytes: int
    hbm_recv_state_bytes: int
    shared_control_bytes: int
    cluster_inbox_offset: int
    cluster_inbox_bytes: int
    flag_count: int
    slot_count: int
    comm_count: int
    arg_count: int
    cluster_scheduling_policy: str = DATAFLOW_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING

    def to_dict(self) -> dict[str, Any]:
        return {
            "abi_version": self.abi_version,
            "queue": self.queue.to_dict(),
            "slots_bytes": len(self.slots_bytes),
            "comms_bytes": len(self.comms_bytes),
            "args_bytes": len(self.args_bytes),
            "input_slots_bytes": len(self.input_slots_bytes),
            "task_coords_bytes": len(self.task_coords_bytes),
            "global_staging_bytes": len(self.global_staging_bytes),
            "flags_bytes": len(self.flags_bytes),
            "shared_memory_bytes": self.shared_memory_bytes,
            "shared_slot_bytes": self.shared_slot_bytes,
            "scratch_backed_slot_count": self.scratch_backed_slot_count,
            "scratch_backed_slot_bytes": self.scratch_backed_slot_bytes,
            "shared_slot_base_offset": self.shared_slot_base_offset,
            "barrier_count": self.barrier_count,
            "barrier_bytes": self.barrier_bytes,
            "cluster_ack_count": self.cluster_ack_count,
            "cluster_ack_barrier_bytes": self.cluster_ack_barrier_bytes,
            "cluster_ack_pending_bytes": self.cluster_ack_pending_bytes,
            "hbm_recv_state_bytes": self.hbm_recv_state_bytes,
            "shared_control_bytes": self.shared_control_bytes,
            "cluster_inbox_offset": self.cluster_inbox_offset,
            "cluster_inbox_bytes": self.cluster_inbox_bytes,
            "flag_count": self.flag_count,
            "slot_count": self.slot_count,
            "comm_count": self.comm_count,
            "arg_count": self.arg_count,
            "cluster_scheduling_policy": self.cluster_scheduling_policy,
        }


def build_launch_package(packed_plan: PackedRuntimePlan) -> DataflowLaunchPackage:
    if not isinstance(packed_plan, PackedRuntimePlan):
        raise TypeError(f"build_launch_package expects PackedRuntimePlan, got {packed_plan!r}")
    if packed_plan.abi_version != ABI_VERSION:
        raise ValueError(f"Dataflow packed plan ABI version {packed_plan.abi_version} does not match runtime ABI version {ABI_VERSION}")

    validate_record_layout(packed_plan)
    validate_queue(packed_plan)
    validate_instructions(packed_plan)
    validate_args(packed_plan)
    validate_comms(packed_plan)

    normal_shared_slots = [slot for slot in packed_plan.slots if not slot_uses_external_storage(slot)]
    scratch_backed_slots = [slot for slot in packed_plan.slots if slot_is_scratch_backed(slot)]
    shared_slot_bytes = aligned_byte_extent(slot.shared_offset + slot.bytes for slot in normal_shared_slots)
    scratch_backed_slot_bytes = sum(slot.bytes for slot in scratch_backed_slots)
    global_staging_size = aligned_byte_extent(slot.global_offset + slot.bytes for slot in packed_plan.slots)
    barrier_count = index_count([slot.barrier_index for slot in packed_plan.slots] + [comm.barrier_index for comm in packed_plan.comms])
    flag_count = index_count([slot.flag_index for slot in packed_plan.slots] + [comm.flag_index for comm in packed_plan.comms])
    barrier_bytes = barrier_count * DATAFLOW_BARRIER_BYTES
    cluster_ack_count = index_count(
        [
            comm.flag_index
            for comm in packed_plan.comms
            if comm.kind
            in {
                DATAFLOW_COMM_KIND_ABI_VALUES["cluster_send"],
                DATAFLOW_COMM_KIND_ABI_VALUES["cluster_recv"],
            }
        ]
    )
    cluster_ack_barrier_bytes = cluster_ack_count * DATAFLOW_BARRIER_BYTES
    cluster_ack_pending_bytes = ((cluster_ack_count + 31) // 32) * UINT32_BYTES
    hbm_recv_state_bytes = ((barrier_count + 31) // 32) * UINT32_BYTES
    shared_control_bytes = barrier_bytes + cluster_ack_barrier_bytes + cluster_ack_pending_bytes + hbm_recv_state_bytes
    shared_slot_base_offset = align_up(shared_control_bytes, DATAFLOW_SHARED_ALIGNMENT)
    shared_memory_bytes = shared_slot_base_offset + shared_slot_bytes
    (
        barrier_init_offsets,
        barrier_init_lengths,
        barrier_init_indices,
    ) = queue_barrier_init_plan(packed_plan)

    return DataflowLaunchPackage(
        abi_version=packed_plan.abi_version,
        queue=DataflowQueueLaunchBuffers(
            instructions_bytes=packed_plan.instruction_bytes,
            offsets_bytes=packed_plan.queue_offsets_bytes,
            lengths_bytes=packed_plan.queue_lengths_bytes,
            barrier_init_offsets=barrier_init_offsets,
            barrier_init_lengths=barrier_init_lengths,
            barrier_init_indices=barrier_init_indices,
            barrier_init_offsets_bytes=pack_u32_array(barrier_init_offsets),
            barrier_init_lengths_bytes=pack_u32_array(barrier_init_lengths),
            barrier_init_indices_bytes=pack_u32_array(barrier_init_indices),
            instruction_count=len(packed_plan.instructions),
            queue_count=len(packed_plan.queue_offsets),
        ),
        slots_bytes=packed_plan.slot_bytes,
        comms_bytes=packed_plan.comm_bytes,
        args_bytes=packed_plan.arg_bytes,
        input_slots_bytes=packed_plan.input_slots_bytes,
        task_coords_bytes=packed_plan.task_coords_bytes,
        global_staging_bytes=bytes(global_staging_size),
        flags_bytes=bytes(flag_count * UINT32_BYTES),
        shared_memory_bytes=shared_memory_bytes,
        shared_slot_bytes=shared_slot_bytes,
        scratch_backed_slot_count=len(scratch_backed_slots),
        scratch_backed_slot_bytes=scratch_backed_slot_bytes,
        shared_slot_base_offset=shared_slot_base_offset,
        barrier_count=barrier_count,
        barrier_bytes=barrier_bytes,
        cluster_ack_count=cluster_ack_count,
        cluster_ack_barrier_bytes=cluster_ack_barrier_bytes,
        cluster_ack_pending_bytes=cluster_ack_pending_bytes,
        hbm_recv_state_bytes=hbm_recv_state_bytes,
        shared_control_bytes=shared_control_bytes,
        cluster_inbox_offset=0,
        cluster_inbox_bytes=0,
        flag_count=flag_count,
        slot_count=len(packed_plan.slots),
        comm_count=len(packed_plan.comms),
        arg_count=len(packed_plan.args),
        cluster_scheduling_policy=(DATAFLOW_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING),
    )


def queue_barrier_init_plan(
    packed_plan: PackedRuntimePlan,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    offsets: list[int] = []
    lengths: list[int] = []
    indices: list[int] = []
    for offset, length in zip(packed_plan.queue_offsets, packed_plan.queue_lengths):
        queue_indices: list[int] = []
        seen: set[int] = set()
        for packed_instruction in packed_plan.instructions[offset : offset + length]:
            if packed_instruction.opcode == 0:
                break
            for comm in packed_plan.comms[packed_instruction.comm_offset : packed_instruction.comm_offset + packed_instruction.comm_count]:
                if comm.kind not in (
                    DATAFLOW_COMM_KIND_ABI_VALUES["cluster_recv"],
                    DATAFLOW_COMM_KIND_ABI_VALUES["hbm_recv"],
                    DATAFLOW_COMM_KIND_ABI_VALUES["hbm_recv_issue"],
                    DATAFLOW_COMM_KIND_ABI_VALUES["hbm_recv_wait"],
                ):
                    continue
                if comm.dst_slot_id >= len(packed_plan.slots):
                    continue
                barrier_index = comm.barrier_index
                if barrier_index == UINT32_SENTINEL:
                    barrier_index = packed_plan.slots[comm.dst_slot_id].barrier_index
                if barrier_index == UINT32_SENTINEL or barrier_index in seen:
                    continue
                seen.add(barrier_index)
                queue_indices.append(barrier_index)
        offsets.append(len(indices))
        lengths.append(len(queue_indices))
        indices.extend(queue_indices)
    return tuple(offsets), tuple(lengths), tuple(indices)


def validate_record_layout(packed_plan: PackedRuntimePlan) -> None:
    expected = (
        (
            "instruction",
            packed_plan.instruction_record_size,
            INSTRUCTION_STRUCT.size,
            len(packed_plan.instructions),
            packed_plan.instruction_bytes,
        ),
        ("slot", packed_plan.slot_record_size, SLOT_STRUCT.size, len(packed_plan.slots), packed_plan.slot_bytes),
        ("comm", packed_plan.comm_record_size, COMM_STRUCT.size, len(packed_plan.comms), packed_plan.comm_bytes),
        ("arg", packed_plan.arg_record_size, ARG_STRUCT.size, len(packed_plan.args), packed_plan.arg_bytes),
    )
    for name, actual_size, expected_size, count, data in expected:
        if actual_size != expected_size:
            raise ValueError(f"Dataflow {name} record size mismatch: got {actual_size}, expected {expected_size}")
        if len(data) != count * expected_size:
            raise ValueError(f"Dataflow {name} byte buffer size does not match record count")

    validate_u32_array("queue_offsets", packed_plan.queue_offsets, packed_plan.queue_offsets_bytes)
    validate_u32_array("queue_lengths", packed_plan.queue_lengths, packed_plan.queue_lengths_bytes)
    validate_u32_array("input_slots", packed_plan.input_slots, packed_plan.input_slots_bytes)
    validate_u32_array("task_coords", packed_plan.task_coords, packed_plan.task_coords_bytes)


def validate_u32_array(name: str, values: tuple[int, ...], data: bytes) -> None:
    if len(data) != len(values) * UINT32_BYTES:
        raise ValueError(f"Dataflow {name} byte buffer size does not match element count")


def validate_queue(packed_plan: PackedRuntimePlan) -> None:
    if len(packed_plan.queue_offsets) != len(packed_plan.queue_lengths):
        raise ValueError("Dataflow queue offsets and lengths must have the same length")

    expected_offset = 0
    instruction_count = len(packed_plan.instructions)
    for queue_index, (offset, length) in enumerate(zip(packed_plan.queue_offsets, packed_plan.queue_lengths)):
        if offset != expected_offset:
            raise ValueError(
                f"Dataflow queue offsets must describe a contiguous flattened queue; "
                f"queue {queue_index} starts at {offset}, expected {expected_offset}"
            )
        if offset + length > instruction_count:
            raise ValueError(f"Dataflow queue {queue_index} extends past the instruction buffer")
        expected_offset += length

    if expected_offset != instruction_count:
        raise ValueError("Dataflow queue lengths do not cover the packed instruction buffer")


def validate_instructions(packed_plan: PackedRuntimePlan) -> None:
    for index, instruction in enumerate(packed_plan.instructions):
        if instruction.comm_offset + instruction.comm_count > len(packed_plan.comms):
            raise ValueError(f"Dataflow instruction {index} references comm records out of range")
        if instruction.slot_id != UINT32_SENTINEL and instruction.slot_id >= len(packed_plan.slots):
            raise ValueError(f"Dataflow instruction {index} references slot_id out of range")

        if instruction.opcode == 0:
            if instruction.arg_offset != UINT32_SENTINEL:
                raise ValueError(f"Dataflow EXIT instruction {index} must use sentinel arg_offset")
            continue

        validate_record_offset(
            "arg_offset",
            instruction.arg_offset,
            ARG_STRUCT.size,
            len(packed_plan.arg_bytes),
            owner=f"instruction {index}",
        )


def validate_args(packed_plan: PackedRuntimePlan) -> None:
    for index, args in enumerate(packed_plan.args):
        if args.task_coord_offset + args.task_coord_count > len(packed_plan.task_coords):
            raise ValueError(f"Dataflow arg record {index} references task coordinates out of range")
        if args.input_slot_offset + args.input_slot_count > len(packed_plan.input_slots):
            raise ValueError(f"Dataflow arg record {index} references input slots out of range")
        if args.output_slot != UINT32_SENTINEL and args.output_slot >= len(packed_plan.slots):
            raise ValueError(f"Dataflow arg record {index} references output_slot out of range")

    for index, slot_id in enumerate(packed_plan.input_slots):
        if slot_id >= len(packed_plan.slots):
            raise ValueError(f"Dataflow input_slots[{index}] references slot_id out of range")


def validate_comms(packed_plan: PackedRuntimePlan) -> None:
    for index, comm in enumerate(packed_plan.comms):
        if comm.src_slot_id >= len(packed_plan.slots):
            raise ValueError(f"Dataflow comm record {index} references src_slot_id out of range")
        if comm.dst_slot_id >= len(packed_plan.slots):
            raise ValueError(f"Dataflow comm record {index} references dst_slot_id out of range")

        src_slot = packed_plan.slots[comm.src_slot_id]
        dst_slot = packed_plan.slots[comm.dst_slot_id]
        transfer_bytes = comm.byte_count if comm.byte_count != 0 else min(src_slot.bytes, dst_slot.bytes)
        if transfer_bytes > src_slot.bytes or transfer_bytes > dst_slot.bytes:
            raise ValueError(f"Dataflow comm record {index} transfer size exceeds slot capacity")


def validate_record_offset(name: str, offset: int, record_size: int, buffer_size: int, *, owner: str) -> None:
    if offset == UINT32_SENTINEL:
        raise ValueError(f"Dataflow {owner} uses sentinel {name} for a non-exit instruction")
    if offset % record_size != 0:
        raise ValueError(f"Dataflow {owner} {name} must be aligned to its record size")
    if offset + record_size > buffer_size:
        raise ValueError(f"Dataflow {owner} {name} points outside the argument buffer")


def aligned_byte_extent(values: Any) -> int:
    maximum = max(values, default=0)
    return align_up(maximum, DATAFLOW_SHARED_ALIGNMENT)


def slot_is_scratch_backed(slot: Any) -> bool:
    return bool(int(getattr(slot, "flags", 0)) & DATAFLOW_SLOT_FLAG_SCRATCH_BACKED)


def slot_uses_external_storage(slot: Any) -> bool:
    return bool(int(getattr(slot, "flags", 0)) & (DATAFLOW_SLOT_FLAG_SCRATCH_BACKED | DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL))


def index_count(values: list[int]) -> int:
    valid = [value for value in values if value != UINT32_SENTINEL]
    if not valid:
        return 0
    return max(valid) + 1


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment
