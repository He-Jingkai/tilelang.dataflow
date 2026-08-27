from __future__ import annotations

from dataclasses import replace

import pytest

import tilelang.language as T
import tilelang.dataflow as df
from tilelang.dataflow.experimental import debug_handlers as dataflow_debug
from tilelang.dataflow.launch import DATAFLOW_BARRIER_BYTES, DATAFLOW_SHARED_ALIGNMENT


@T.dataflow_intermediate
class AttnInter:
    lse: T.float32
    o: T.Tensor((2, 8), T.float16)


@T.dataflow.iter(range=("kv_begin", "kv_end"))
def split_kv(seq: T.int32, head: T.int32, Q, K, V) -> AttnInter:
    raise AssertionError("Dataflow iter body should not execute during launch packaging")


@T.dataflow.reduce
def combine(items: list[AttnInter]) -> AttnInter:
    raise AssertionError("Dataflow reduce body should not execute during launch packaging")


@T.dataflow.finalize
def finalize(inter: AttnInter, seq: T.int32, head: T.int32, Output) -> None:
    raise AssertionError("Dataflow finalize body should not execute during launch packaging")


def make_packed_plan():
    program = (
        T.dataflow_program(task_domain=("batch", "head"), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_kv(Q="Q", K="K", V="V"),
            task_args=("seq", "head"),
            range_axis="kv",
        )
        .reduce(combine())
        .finalize(finalize(Output="O"))
    )
    plan = df.schedule(
        program,
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [384, 256]},
        block_size=128,
        task_extents=(2,),
    )
    return df.pack_instruction_plan(plan)


def test_build_launch_package_collects_wrapper_buffers_and_sizes():
    packed = make_packed_plan()
    package = df.build_launch_package(packed)

    assert package.abi_version == df.ABI_VERSION
    assert package.queue.instruction_count == len(packed.instructions)
    assert package.queue.queue_count == 4
    assert package.queue.instructions_bytes == packed.instruction_bytes
    assert package.queue.offsets_bytes == packed.queue_offsets_bytes
    assert package.queue.lengths_bytes == packed.queue_lengths_bytes
    assert len(package.queue.barrier_init_offsets) == package.queue.queue_count
    assert len(package.queue.barrier_init_lengths) == package.queue.queue_count
    assert sum(package.queue.barrier_init_lengths) == len(package.queue.barrier_init_indices)
    for offset, length in zip(package.queue.barrier_init_offsets, package.queue.barrier_init_lengths):
        queue_indices = package.queue.barrier_init_indices[offset : offset + length]
        assert len(queue_indices) == len(set(queue_indices))
        assert all(0 <= barrier_index < package.barrier_count for barrier_index in queue_indices)
    assert len(package.queue.barrier_init_offsets_bytes) == 4 * 4
    assert len(package.queue.barrier_init_lengths_bytes) == 4 * 4
    assert len(package.queue.barrier_init_indices_bytes) == len(package.queue.barrier_init_indices) * 4

    assert package.slots_bytes == packed.slot_bytes
    assert package.comms_bytes == packed.comm_bytes
    assert package.args_bytes == packed.arg_bytes
    assert package.input_slots_bytes == packed.input_slots_bytes
    assert package.task_coords_bytes == packed.task_coords_bytes
    assert package.cluster_scheduling_policy == "load_balancing"

    assert package.slot_count == 7
    assert package.comm_count == 6
    assert package.arg_count == 9
    assert package.flag_count == 7
    assert len(package.flags_bytes) == 7 * 4
    assert package.barrier_count == packed.barrier_allocation.barrier_count
    assert package.barrier_bytes == package.barrier_count * DATAFLOW_BARRIER_BYTES
    assert package.cluster_ack_count == 0
    assert package.cluster_ack_barrier_bytes == 0
    assert package.cluster_ack_pending_bytes == 0
    assert package.hbm_recv_state_bytes == 4
    assert package.cluster_inbox_offset == 0
    assert package.cluster_inbox_bytes == 0
    assert package.shared_control_bytes == (
        package.barrier_bytes + package.cluster_ack_barrier_bytes + package.cluster_ack_pending_bytes + package.hbm_recv_state_bytes
    )
    assert package.shared_slot_base_offset % DATAFLOW_SHARED_ALIGNMENT == 0
    assert package.shared_slot_base_offset >= package.shared_control_bytes
    assert package.shared_slot_bytes == 336
    assert package.shared_memory_bytes == package.shared_slot_base_offset + package.shared_slot_bytes
    assert len(package.global_staging_bytes) == 336


def test_build_launch_package_rejects_mismatched_abi_version():
    packed = replace(make_packed_plan(), abi_version=df.ABI_VERSION + 1)

    with pytest.raises(ValueError, match="does not match runtime ABI version"):
        df.build_launch_package(packed)


def test_build_launch_package_excludes_scratch_backed_slots_from_shared_slot_bytes():
    packed = make_packed_plan()
    slots = list(packed.slots)
    slots[1] = replace(
        slots[1],
        shared_offset=4096,
        flags=slots[1].flags | df.DATAFLOW_SLOT_FLAG_SCRATCH_BACKED,
        reserved=4096,
    )
    packed = replace(
        packed,
        slots=tuple(slots),
        slot_bytes=b"".join(df.SLOT_STRUCT.pack(*slot.to_tuple()) for slot in slots),
    )

    package = df.build_launch_package(packed)

    assert package.shared_slot_bytes == 336
    assert package.scratch_backed_slot_count == 1
    assert package.scratch_backed_slot_bytes == 48
    assert package.shared_memory_bytes == package.shared_slot_base_offset + package.shared_slot_bytes
    dumped = package.to_dict()
    assert dumped["scratch_backed_slot_count"] == 1
    assert dumped["scratch_backed_slot_bytes"] == 48


def test_build_launch_package_excludes_hbm_direct_global_slots_from_shared_slot_bytes():
    packed = make_packed_plan()
    slots = list(packed.slots)
    slots[1] = replace(
        slots[1],
        shared_offset=4096,
        flags=slots[1].flags | df.DATAFLOW_SLOT_FLAG_HBM_DIRECT_GLOBAL,
    )
    packed = replace(
        packed,
        slots=tuple(slots),
        slot_bytes=b"".join(df.SLOT_STRUCT.pack(*slot.to_tuple()) for slot in slots),
    )

    package = df.build_launch_package(packed)

    assert package.shared_slot_bytes == 336
    assert package.scratch_backed_slot_count == 0
    assert package.shared_memory_bytes == package.shared_slot_base_offset + package.shared_slot_bytes


def test_build_launch_package_to_dict_is_stable_debug_metadata():
    package = df.build_launch_package(make_packed_plan())
    dumped = package.to_dict()

    assert dumped["queue"]["instruction_count"] == 13
    assert dumped["queue"]["queue_count"] == 4
    assert dumped["slots_bytes"] == 7 * df.SLOT_STRUCT.size
    assert dumped["comms_bytes"] == 6 * df.COMM_STRUCT.size
    assert dumped["args_bytes"] == 9 * df.ARG_STRUCT.size
    assert dumped["barrier_count"] == package.barrier_count
    assert dumped["barrier_bytes"] == package.barrier_bytes
    assert dumped["cluster_ack_count"] == package.cluster_ack_count
    assert dumped["cluster_ack_barrier_bytes"] == package.cluster_ack_barrier_bytes
    assert dumped["cluster_ack_pending_bytes"] == package.cluster_ack_pending_bytes
    assert dumped["hbm_recv_state_bytes"] == package.hbm_recv_state_bytes
    assert dumped["cluster_inbox_offset"] == 0
    assert dumped["cluster_inbox_bytes"] == 0
    assert dumped["shared_control_bytes"] == package.shared_control_bytes
    assert dumped["shared_slot_base_offset"] == package.shared_slot_base_offset
    assert dumped["shared_memory_bytes"] == package.shared_memory_bytes
    assert dumped["cluster_scheduling_policy"] == "load_balancing"


def test_compile_exposes_launch_package_and_updates_wrapper_shared_base():
    program = (
        T.dataflow_program(task_domain=("batch", "head"), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_kv(Q="Q", K="K", V="V"),
            task_args=("seq", "head"),
            range_axis="kv",
        )
        .reduce(combine())
        .finalize(finalize(Output="O"))
    )

    compiled = dataflow_debug.compile(
        program,
        handler=dataflow_debug.EMPTY_HANDLER,
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [384, 256]},
        block_size=128,
        task_extents=(2,),
    )

    assert compiled.wrapper_spec.shared_slot_base_offset == compiled.launch_package.shared_slot_base_offset
    assert compiled.wrapper_spec.barrier_count == compiled.packed_plan.barrier_allocation.barrier_count
    compiled.validate_memory_layout().require_valid()

    dumped = compiled.dump_plan()
    assert dumped["launch_package"]["shared_memory_bytes"] == compiled.launch_package.shared_memory_bytes
    assert dumped["wrapper"]["shared_slot_base_offset"] == compiled.launch_package.shared_slot_base_offset
    assert dumped["wrapper"]["barrier_count"] == compiled.launch_package.barrier_count


def test_build_launch_package_rejects_invalid_inputs():
    with pytest.raises(TypeError, match="PackedRuntimePlan"):
        df.build_launch_package(object())

    packed = make_packed_plan()
    with pytest.raises(ValueError, match="record size"):
        df.build_launch_package(replace(packed, instruction_record_size=16))

    with pytest.raises(ValueError, match="queue offsets"):
        df.build_launch_package(replace(packed, queue_offsets=(0, 6, 7, 9)))

    instructions = list(packed.instructions)
    instructions[0] = replace(instructions[0], arg_offset=len(packed.arg_bytes))
    with pytest.raises(ValueError, match="argument buffer"):
        df.build_launch_package(replace(packed, instructions=tuple(instructions)))

    comms = list(packed.comms)
    comms[0] = replace(comms[0], src_slot_id=999)
    with pytest.raises(ValueError, match="src_slot_id"):
        df.build_launch_package(replace(packed, comms=tuple(comms)))
