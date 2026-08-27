from __future__ import annotations

from dataclasses import replace
import os
import struct
import importlib
from types import SimpleNamespace

import pytest

import tilelang.language as T
import tilelang.dataflow as df
import tilelang.dataflow.compiler as dataflow_compiler
from tilelang.dataflow.executor import DataflowExecutableKernel
from tilelang.dataflow.experimental import debug_handlers as dataflow_debug

dataflow_profiler_module = importlib.import_module("tilelang.dataflow.profiler")
from tilelang.contrib import nvcc


@T.dataflow_intermediate
class AttnInter:
    lse: T.float32
    o: T.Tensor((2, 8), T.float16)


@T.dataflow.iter(range=("kv_begin", "kv_end"))
def split_kv(seq: T.int32, head: T.int32, Q, K, V) -> AttnInter:
    raise AssertionError("Dataflow iter body should not execute during compile skeleton")


@T.dataflow.reduce(associative=True)
def combine(left: AttnInter, right: AttnInter) -> AttnInter:
    raise AssertionError("Dataflow reduce body should not execute during compile skeleton")


@T.dataflow.finalize
def finalize(inter: AttnInter, seq: T.int32, head: T.int32, Output) -> None:
    raise AssertionError("Dataflow finalize body should not execute during compile skeleton")


@T.dataflow_intermediate
class UpShard:
    value: T.Tensor((2, 4), T.float16)


@T.dataflow_intermediate
class UpFull:
    value: T.Tensor((2, 8), T.float16)


@T.dataflow_intermediate
class HiddenShard:
    value: T.Tensor((2, 4), T.float16)


@T.dataflow_intermediate
class IntShard:
    value: T.int32


@T.dataflow_intermediate
class IntFull:
    value: T.int32


@T.dataflow_intermediate
class IntHidden:
    value: T.int32


@T.dataflow.map(range=("expert_begin", "expert_end"))
def moe_map1(expert: T.int32, token: T.int32, Input, W1) -> UpShard:
    raise AssertionError("Dataflow map body should not execute during compile skeleton")


@T.dataflow.map(range=("hidden_begin", "hidden_end"))
def moe_map2(parts: list[UpShard], expert: T.int32, token: T.int32, W2) -> HiddenShard:
    raise AssertionError("Dataflow map body should not execute during compile skeleton")


@T.dataflow.finalize
def finalize_hidden(hidden: HiddenShard, expert: T.int32, token: T.int32, Output) -> None:
    raise AssertionError("Dataflow finalize body should not execute during compile skeleton")


@T.dataflow.map(range=("expert_begin", "expert_end"))
def prim_map1(expert: T.int32, token: T.int32, Input: T.Tensor((64,), T.int32)) -> IntShard:
    value = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += Input[i]
    return IntShard(value=value)


@T.dataflow.map(range=("hidden_begin", "hidden_end"))
def prim_map2(parts: list[IntShard], expert: T.int32, token: T.int32, Bias: T.Tensor((64,), T.int32)) -> IntHidden:
    value = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += parts[0].value + parts[1].value + Bias[i]
    return IntHidden(value=value)


@T.dataflow.finalize
def prim_finalize(hidden: IntHidden, expert: T.int32, token: T.int32, Output: T.Tensor((64,), T.int32)) -> None:
    Output[T.dataflow_range_begin()] = hidden.value + expert + token


def require_nvcc() -> None:
    try:
        compiler = nvcc.get_nvcc_compiler()
    except Exception as err:
        pytest.skip(f"CUDA toolkit/NVCC unavailable: {err}")
    if not os.path.exists(compiler):
        pytest.skip(f"NVCC not found at {compiler}")


def require_executable_cuda() -> None:
    require_nvcc()

    try:
        from cuda.bindings import driver
    except Exception as err:
        pytest.skip(f"CUDA driver bindings unavailable: {err}")

    result = driver.cuInit(0)[0]
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA driver unavailable: {result}")
    result, count = driver.cuDeviceGetCount()
    if result != driver.CUresult.CUDA_SUCCESS or count == 0:
        pytest.skip(f"CUDA device unavailable: {result}, count={count}")

    result, device = driver.cuDeviceGet(0)
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA device 0 unavailable: {result}")
    result, context = driver.cuDevicePrimaryCtxRetain(device)
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA primary context unavailable: {result}")
    result = driver.cuCtxSetCurrent(context)[0]
    driver.cuDevicePrimaryCtxRelease(device)
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA context activation failed: {result}")


def make_program():
    return (
        T.dataflow_program(task_domain=("batch", "head"), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_kv(Q="Q", K="K", V="V"),
            task_args=("seq", "head"),
            range_axis="kv",
        )
        .reduce(combine())
        .finalize(finalize(Output="O"))
    )


def make_scratch_handoff_plan(
    kind: df.DataflowCommKind,
    *,
    local_input_offset: int = 64,
    later_output_offset: int = 64,
) -> df.InstructionPlan:
    intermediate_type = split_kv.output_type
    assert intermediate_type is not None
    producer = df.Instruction(
        instruction_id=0,
        opcode=df.DataflowOpcode.ITER,
        operator_name="producer",
        task_id=0,
        sm_id=0,
        output_slot=0,
    )
    consumer = df.Instruction(
        instruction_id=1,
        opcode=df.DataflowOpcode.REDUCE_UPDATE,
        operator_name="consumer",
        task_id=0,
        sm_id=1,
        input_slots=(0, 1),
        output_slot=2,
    )
    later = df.Instruction(
        instruction_id=2,
        opcode=df.DataflowOpcode.ITER,
        operator_name="later",
        task_id=1,
        sm_id=0,
        output_slot=3,
    )
    slots = (
        df.SlotPlan(
            slot_id=0,
            task_id=0,
            intermediate_type=intermediate_type,
            role="partial",
            producer_instruction_id=0,
            scratch_backed=True,
            scratch_offset=0,
        ),
        df.SlotPlan(
            slot_id=1,
            task_id=0,
            intermediate_type=intermediate_type,
            role="streaming_acc",
            scratch_backed=True,
            scratch_offset=local_input_offset,
        ),
        df.SlotPlan(
            slot_id=2,
            task_id=0,
            intermediate_type=intermediate_type,
            role="streaming_acc",
            producer_instruction_id=1,
            scratch_backed=True,
            scratch_offset=96,
        ),
        df.SlotPlan(
            slot_id=3,
            task_id=1,
            intermediate_type=intermediate_type,
            role="partial",
            producer_instruction_id=2,
            scratch_backed=True,
            scratch_offset=later_output_offset,
        ),
    )
    send_kind = df.DataflowCommKind.CLUSTER_SEND if kind is df.DataflowCommKind.CLUSTER_RECV else df.DataflowCommKind.HBM_SEND
    peer_rank = 1 if kind is df.DataflowCommKind.CLUSTER_RECV else None
    producer_rank = 0 if kind is df.DataflowCommKind.CLUSTER_RECV else None
    comms = (
        df.CommPlan(
            source_instruction_id=0,
            target_instruction_id=1,
            source_slot_id=0,
            target_slot_id=0,
            producer_sm=0,
            consumer_sm=1,
            kind=send_kind,
            dispatch_instruction_id=0,
            peer_cta_rank=peer_rank,
        ),
        df.CommPlan(
            source_instruction_id=0,
            target_instruction_id=1,
            source_slot_id=0,
            target_slot_id=0,
            producer_sm=0,
            consumer_sm=1,
            kind=kind,
            dispatch_instruction_id=1,
            peer_cta_rank=producer_rank,
        ),
    )
    return df.InstructionPlan(
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        block_size=64,
        range_axis="kv",
        scheduler_policy="test",
        reduce_strategy="streaming_tree",
        task_extents=(2,),
        task_range_lengths=(64, 64),
        instructions=(producer, consumer, later),
        queues={0: (producer, later), 1: (consumer,)},
        slots=slots,
        comms=comms,
    )


def test_scratch_hbm_alias_uses_byte_ranges_for_selective_direct_global():
    overlapping = make_scratch_handoff_plan(
        df.DataflowCommKind.HBM_RECV,
        local_input_offset=32,
    )
    disjoint = make_scratch_handoff_plan(
        df.DataflowCommKind.HBM_RECV,
        local_input_offset=64,
    )

    assert dataflow_compiler.scratch_hbm_direct_global_slot_ids(overlapping) == frozenset((0,))
    assert dataflow_compiler.scratch_hbm_direct_global_slot_ids(disjoint) == frozenset()


def test_queue_slot_lifetime_proof_allows_boundary_reuse_but_rejects_overlap():
    plan = make_scratch_handoff_plan(
        df.DataflowCommKind.CLUSTER_RECV,
        later_output_offset=0,
    )
    shared_slots = (plan.slots[0], plan.slots[3])

    def physical_group(slot, owner_cta):
        return owner_cta, slot.scratch_offset

    assert dataflow_compiler.queue_slot_lifetimes_are_linear(
        plan,
        shared_slots,
        physical_group=physical_group,
    )

    late_consumer = df.Instruction(
        instruction_id=3,
        opcode=df.DataflowOpcode.FINALIZE,
        operator_name="late_consumer",
        task_id=0,
        sm_id=0,
        input_slots=(0,),
    )
    overlapping = replace(
        plan,
        instructions=plan.instructions + (late_consumer,),
        queues={
            **plan.queues,
            0: plan.queue(0) + (late_consumer,),
        },
    )
    assert not dataflow_compiler.queue_slot_lifetimes_are_linear(
        overlapping,
        shared_slots,
        physical_group=physical_group,
    )


def test_scratch_candidate_preflight_rejects_only_overlapping_storage_class():
    plan = make_scratch_handoff_plan(df.DataflowCommKind.CLUSTER_RECV)
    slots = tuple(replace(slot, role="streaming_acc", shared_storage_id=0) if slot.slot_id in {0, 3} else slot for slot in plan.slots)
    plan = replace(plan, slots=slots)

    assert (
        dataflow_compiler.scratch_backed_candidate_rejection_reason(
            plan,
            partial_only=False,
        )
        is None
    )

    late_consumer = df.Instruction(
        instruction_id=3,
        opcode=df.DataflowOpcode.FINALIZE,
        operator_name="late_consumer",
        task_id=0,
        sm_id=0,
        input_slots=(0,),
    )
    overlapping = replace(
        plan,
        instructions=plan.instructions + (late_consumer,),
        queues={**plan.queues, 0: plan.queue(0) + (late_consumer,)},
    )
    rejection = dataflow_compiler.scratch_backed_candidate_rejection_reason(
        overlapping,
        partial_only=False,
    )
    assert rejection is not None
    assert "bounded linear lifetime" in rejection


def test_cluster_inbox_preserves_source_and_separates_consumer_inputs():
    plan = make_scratch_handoff_plan(
        df.DataflowCommKind.CLUSTER_RECV,
        local_input_offset=0,
    )

    staged = dataflow_compiler.materialize_cluster_inbox_slots(
        plan,
        cluster_inbox_offset=256,
    )

    assert len(staged.slots) == len(plan.slots) + 1
    staging_slot = staged.slots[-1]
    assert staging_slot.role == "cluster_inbox"
    assert staging_slot.scratch_backed is True
    assert staging_slot.scratch_offset == 256
    assert [comm.source_slot_id for comm in staged.comms] == [0, 0]
    assert [comm.target_slot_id for comm in staged.comms] == [
        staging_slot.slot_id,
        staging_slot.slot_id,
    ]
    assert staged.queue(1)[0].input_slots == (staging_slot.slot_id, 1)
    input_ranges = [dataflow_compiler.scratch_slot_range(staged.slots[slot_id]) for slot_id in staged.queue(1)[0].input_slots]
    assert all(scratch_range is not None for scratch_range in input_ranges)
    assert not dataflow_compiler.ranges_overlap(*input_ranges)


def test_joint_cluster_local_plan_materializes_independent_accumulator_destination():
    plan = df.schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": (512, 384)},
        block_size=64,
        task_extents=(2,),
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        scheduler_config=df.DataflowSchedulerConfig.from_options(
            direct_leaf_acc=True,
            hier_cross_cluster_split=True,
            joint_schedule=True,
            ready_time_tree=True,
        ),
    )
    assert plan.joint_execution_plan is not None

    handler_abi = SimpleNamespace(slot_bytes=48, field_layouts=())
    wrapper_spec = SimpleNamespace(
        handlers=(
            SimpleNamespace(handler_id=0, operator_kind="iter"),
            SimpleNamespace(handler_id=1, operator_kind="reduce"),
            SimpleNamespace(handler_id=2, operator_kind="finalize"),
        )
    )
    primfunc_lowering = SimpleNamespace(
        dynamic_shared_bytes=48,
        handlers=(
            SimpleNamespace(
                handler_id=0,
                dynamic_shared_bytes=48,
                device_symbol="iter",
            ),
            SimpleNamespace(
                handler_id=1,
                dynamic_shared_bytes=48,
                device_symbol="reduce",
            ),
            SimpleNamespace(
                handler_id=2,
                dynamic_shared_bytes=0,
                device_symbol="finalize",
            ),
        ),
    )

    scratch = dataflow_compiler.plan_scratch_backed_slots(
        plan,
        handler_abi,
        primfunc_lowering,
        wrapper_spec,
        async_safe_cluster_handoff=False,
    )
    staged = scratch.plan
    communicate_slots = tuple(slot for slot in staged.slots if slot.role in {"joint_comm_inbox", "joint_comm_outbox"})
    assert communicate_slots
    assert scratch.cluster_inbox_offset > 0
    assert scratch.required_scratch_bytes >= (scratch.cluster_inbox_offset + scratch.cluster_inbox_bytes)
    assert scratch.cluster_inbox_bytes > 0
    assert all(
        comm.kind
        in {
            df.DataflowCommKind.CLUSTER_SEND,
            df.DataflowCommKind.CLUSTER_RECV,
            df.DataflowCommKind.CLUSTER_RELEASE,
        }
        for comm in staged.comms
    )

    slots_by_id = {slot.slot_id: slot for slot in staged.slots}
    instructions_by_id = {instruction.instruction_id: instruction for instruction in staged.instructions}
    for recv in staged.comms:
        if recv.kind is not df.DataflowCommKind.CLUSTER_RECV:
            continue
        target = instructions_by_id[recv.target_instruction_id]
        remote_range = dataflow_compiler.scratch_slot_range(slots_by_id[recv.target_slot_id])
        local_ranges = tuple(
            dataflow_compiler.scratch_slot_range(slots_by_id[slot_id]) for slot_id in target.input_slots if slot_id != recv.target_slot_id
        )
        assert remote_range is not None
        assert all(local_range is None or not dataflow_compiler.ranges_overlap(remote_range, local_range) for local_range in local_ranges)

    packed = df.pack_instruction_plan(staged)
    assert any(slot.flags & df.DATAFLOW_SLOT_FLAG_COMMUNICATE for slot in packed.slots)
    launch = df.build_launch_package(packed)
    physical_wrapper = df.build_wrapper_spec(
        packed,
        launch_package=launch,
        cluster_size=staged.topology.cluster_size,
    )
    launch, physical_wrapper = dataflow_compiler.extend_launch_for_primfunc_dynamic_shared(
        launch,
        physical_wrapper,
        scratch.required_scratch_bytes,
        cluster_inbox_offset=scratch.cluster_inbox_offset,
        cluster_inbox_bytes=scratch.cluster_inbox_bytes,
    )
    df.validate_dataflow_memory_layout(
        packed,
        launch,
        physical_wrapper,
        plan=staged,
    ).require_valid()


def test_joint_multi_input_reduce_materializes_two_independent_inboxes():
    plan = df.schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": (512,)},
        block_size=64,
        task_extents=(1,),
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        scheduler_config=df.DataflowSchedulerConfig.from_options(
            direct_leaf_acc=True,
            joint_schedule=True,
            joint_hbm_async_receive_pipeline=True,
            ordered_interval_tree=True,
            ordered_tree_max_reduce_arity=3,
        ),
    )
    assert plan.joint_execution_plan is not None
    fused = next(instruction for instruction in plan.instructions if instruction.attrs.get("fused_reduce_arity") == 3)
    incoming = tuple(
        transfer for transfer in plan.joint_execution_plan.schedule.transfers if transfer.consumer_node_id == fused.instruction_id
    )
    assert len(incoming) == 2
    assert all(transfer.retained_input_copy_event_id is None for transfer in incoming)
    assert {transfer.consumer_storage_kind for transfer in incoming} == {
        df.DataflowCommStorageKind.PERMANENT,
        df.DataflowCommStorageKind.TRANSIENT_PREFETCH,
    }

    handler_abi = SimpleNamespace(slot_bytes=48, field_layouts=())
    wrapper_spec = SimpleNamespace(
        handlers=(
            SimpleNamespace(handler_id=0, operator_kind="iter"),
            SimpleNamespace(handler_id=1, operator_kind="reduce"),
            SimpleNamespace(handler_id=2, operator_kind="finalize"),
        )
    )
    primfunc_lowering = SimpleNamespace(
        dynamic_shared_bytes=48,
        handlers=(
            SimpleNamespace(
                handler_id=0,
                dynamic_shared_bytes=48,
                device_symbol="iter",
            ),
            SimpleNamespace(
                handler_id=1,
                dynamic_shared_bytes=48,
                device_symbol="reduce",
            ),
            SimpleNamespace(
                handler_id=2,
                dynamic_shared_bytes=0,
                device_symbol="finalize",
            ),
        ),
    )

    scratch = dataflow_compiler.plan_scratch_backed_slots(
        plan,
        handler_abi,
        primfunc_lowering,
        wrapper_spec,
        async_safe_cluster_handoff=False,
    )
    materialized = next(instruction for instruction in scratch.plan.instructions if instruction.instruction_id == fused.instruction_id)
    slots_by_id = {slot.slot_id: slot for slot in scratch.plan.slots}
    input_ranges = tuple(dataflow_compiler.scratch_slot_range(slots_by_id[slot_id]) for slot_id in materialized.input_slots)
    assert all(scratch_range is not None for scratch_range in input_ranges)
    assert all(
        not dataflow_compiler.ranges_overlap(left, right) for index, left in enumerate(input_ranges) for right in input_ranges[index + 1 :]
    )
    remote_roles = {
        slots_by_id[slot_id].role
        for slot_id in materialized.input_slots
        if slots_by_id[slot_id].physical_owner_cta == materialized.sm_id and slots_by_id[slot_id].producer_instruction_id is not None
    }
    assert remote_roles
    assert remote_roles <= {
        "joint_comm_inbox",
        "joint_prefetch_inbox",
        "cluster_gated_inbox",
    }


def test_cluster_inbox_does_not_rewrite_hbm_handoff():
    plan = make_scratch_handoff_plan(
        df.DataflowCommKind.HBM_RECV,
        local_input_offset=0,
    )
    assert (
        dataflow_compiler.materialize_cluster_inbox_slots(
            plan,
            cluster_inbox_offset=256,
        )
        is plan
    )


def test_cluster_inbox_rejects_multiple_remote_inputs_per_instruction():
    plan = make_scratch_handoff_plan(
        df.DataflowCommKind.CLUSTER_RECV,
    )
    duplicate_recv = next(comm for comm in plan.comms if comm.kind is df.DataflowCommKind.CLUSTER_RECV)

    with pytest.raises(df.DataflowMemoryPlanningError, match="at most one.*per target"):
        dataflow_compiler.materialize_cluster_inbox_slots(
            replace(plan, comms=plan.comms + (duplicate_recv,)),
            cluster_inbox_offset=256,
        )


def test_repeated_cluster_inbox_spills_later_push_without_blocking_producer():
    plan = make_scratch_handoff_plan(
        df.DataflowCommKind.CLUSTER_RECV,
    )
    first_send, first_recv = plan.comms
    later_producer = replace(
        plan.instructions[2],
        output_slot=3,
    )
    later_consumer = replace(
        plan.instructions[1],
        instruction_id=3,
        task_id=1,
        input_slots=(3,),
        output_slot=None,
    )
    later_send = replace(
        first_send,
        source_instruction_id=later_producer.instruction_id,
        target_instruction_id=later_consumer.instruction_id,
        source_slot_id=3,
        target_slot_id=3,
    )
    later_recv = replace(
        first_recv,
        source_instruction_id=later_producer.instruction_id,
        target_instruction_id=later_consumer.instruction_id,
        source_slot_id=3,
        target_slot_id=3,
    )
    repeated = replace(
        plan,
        instructions=plan.instructions + (later_consumer,),
        queues={
            0: plan.queues[0],
            1: plan.queues[1] + (later_consumer,),
        },
        comms=(first_send, first_recv, later_send, later_recv),
    )

    staged = dataflow_compiler.materialize_cluster_inbox_slots(
        repeated,
        cluster_inbox_offset=256,
    )

    assert [comm.kind for comm in staged.comms] == [
        df.DataflowCommKind.CLUSTER_SEND,
        df.DataflowCommKind.CLUSTER_RECV,
        df.DataflowCommKind.HBM_SEND,
        df.DataflowCommKind.HBM_RECV,
    ]
    inboxes = staged.slots[len(plan.slots) :]
    assert [slot.role for slot in inboxes] == [
        "cluster_inbox",
        "hbm_spill_inbox",
    ]
    assert [slot.scratch_offset for slot in inboxes] == [256, 256]
    assert staged.queue(1)[0].input_slots[0] == inboxes[0].slot_id
    assert staged.queue(1)[1].input_slots == (inboxes[1].slot_id,)
    packed = df.pack_instruction_plan(staged)
    assert packed.slots[inboxes[0].slot_id].flags & df.DATAFLOW_SLOT_FLAG_CLUSTER_GATED_PUSH == 0
    assert packed.slots[inboxes[1].slot_id].flags & df.DATAFLOW_SLOT_FLAG_CLUSTER_GATED_PUSH == 0
    assert all(packed.slots[inbox.slot_id].flags & df.DATAFLOW_SLOT_FLAG_COMMUNICATE for inbox in inboxes)


def make_reshared_program():
    return (
        T.dataflow_program(
            task_domain=("expert", "token"),
            dynamic_ranges={"expert_tile": "expert_tiles", "hidden_tile": "hidden_tiles"},
        )
        .map(
            moe_map1(Input="Input", W1="W1"),
            name="map1",
            task_args=("expert", "token"),
            range_axis="expert_tile",
        )
        .reshared(
            input="map1",
            name="gather_up",
            output_type=UpFull,
            physical_output_type=UpShard,
            output_arity=2,
            policy="hbm_all_gather",
        )
        .map(
            moe_map2(W2="W2"),
            name="map2",
            input="gather_up",
            task_args=("expert", "token"),
            range_axis="hidden_tile",
        )
        .finalize(finalize_hidden(Output="Output"), input="map2")
    )


def make_reshared_primfunc_program():
    return (
        T.dataflow_program(
            task_domain=("expert", "token"),
            dynamic_ranges={"expert_tile": "expert_tiles", "hidden_tile": "hidden_tiles"},
        )
        .map(
            prim_map1(Input="Input"),
            name="map1",
            task_args=("expert", "token"),
            range_axis="expert_tile",
        )
        .reshared(
            input="map1",
            name="gather_up",
            output_type=IntFull,
            physical_output_type=IntShard,
            output_arity=2,
            policy="hbm_all_gather",
        )
        .map(
            prim_map2(Bias="Bias"),
            name="map2",
            input="gather_up",
            task_args=("expert", "token"),
            range_axis="hidden_tile",
        )
        .finalize(prim_finalize(Output="Output"), input="map2")
    )


def test_dataflow_compile_returns_planning_and_wrapper_package():
    compiled = df.compile(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [256]},
        block_size=128,
        task_extents=(1,),
        debug_name="attention",
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    assert isinstance(compiled, df.DataflowCompiledProgram)
    assert compiled.plan.task_range_lengths == (256,)
    assert compiled.packed_plan.operator_table == {
        "split_kv": 0,
        "combine": 1,
        "finalize": 2,
    }
    assert compiled.options["topology"] == {"sm_count": 2, "cluster_size": 1}
    assert compiled.options["range_lengths"] == {"kv": [256]}
    assert compiled.options["block_size"] == 128
    assert compiled.options["task_extents"] == (1,)
    assert compiled.options["debug_name"] == "attention"
    assert compiled.tensor_arg_plan.names == ("Q", "K", "V", "O")
    assert compiled.wrapper_spec.tensor_arg_count == 4
    assert compiled.artifact_contract == "debug"
    assert compiled.executable is False
    assert "empty experimental handlers" in compiled.non_executable_reason

    dumped = compiled.dump_plan()
    assert dumped["artifact"] == {
        "contract": "debug",
        "executable": False,
        "non_executable_reason": "empty experimental handlers do not perform program semantics",
    }
    assert dumped["compile_config"]["mode"] == "debug"
    assert dumped["compile_config"]["fingerprint"] == compiled.compile_config.fingerprint
    assert dumped["compile_config"]["provenance"]["mode"] == "option"
    assert dumped["plan"]["instruction_count"] == len(compiled.plan.instructions)
    assert dumped["plan"]["slot_count"] == len(compiled.plan.slots)
    assert dumped["plan"]["comm_count"] == len(compiled.plan.comms)
    assert dumped["packed_plan"]["abi_version"] == df.ABI_VERSION
    assert dumped["tensor_args"]["names"] == ["Q", "K", "V", "O"]
    assert dumped["wrapper"]["tensor_arg_count"] == 4


def test_dataflow_artifact_state_rejects_inconsistent_contracts():
    with pytest.raises(ValueError, match="Unsupported Dataflow artifact contract"):
        df.DataflowArtifactState(contract="unknown", executable=True)
    with pytest.raises(ValueError, match="cannot have a non-executable reason"):
        df.DataflowArtifactState(
            contract="debug",
            executable=True,
            non_executable_reason="unexpected",
        )
    with pytest.raises(ValueError, match="require a reason"):
        df.DataflowArtifactState(
            contract=df.DATAFLOW_ARTIFACT_INSPECTION,
            executable=False,
        )
    with pytest.raises(ValueError, match="production Dataflow artifacts must be executable"):
        df.DataflowArtifactState(
            contract=df.DATAFLOW_ARTIFACT_PRODUCTION,
            executable=False,
            non_executable_reason="invalid production state",
        )


def test_dataflow_compile_handles_reshared_stage_graph_with_empty_handlers():
    compiled = df.compile(
        make_reshared_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"expert_tile": [256], "hidden_tile": [256]},
        block_size=128,
        task_extents=(1, 1),
        include_exit=False,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
        force_hbm_comms=True,
        wrapper_name="dataflow_moe_stage_graph",
    )

    assert compiled.plan.scheduler_policy == "stage_graph"
    assert compiled.artifact_contract == "debug"
    assert compiled.executable is False
    assert "empty experimental handlers" in compiled.non_executable_reason
    assert compiled.packed_plan.operator_table == {
        "moe_map1": 0,
        "moe_map2": 1,
        "finalize_hidden": 2,
    }
    assert compiled.tensor_arg_plan.names == ("Input", "W1", "W2", "Output")
    assert [
        (binding.operator_kind, binding.operator_name, binding.parameter_name, binding.tensor_name)
        for binding in compiled.tensor_arg_plan.bindings
    ] == [
        ("map", "moe_map1", "Input", "Input"),
        ("map", "moe_map1", "W1", "W1"),
        ("map", "moe_map2", "W2", "W2"),
        ("finalize", "finalize_hidden", "Output", "Output"),
    ]
    assert [(handler.operator_name, handler.operator_kind) for handler in compiled.wrapper_spec.handlers] == [
        ("moe_map1", "map"),
        ("moe_map2", "map"),
        ("finalize_hidden", "finalize"),
    ]


def test_dataflow_compile_lowers_reshared_stage_graph_primfunc_handlers_without_cuda():
    compiled = df.compile(
        make_reshared_primfunc_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"expert_tile": [64], "hidden_tile": [64]},
        block_size=32,
        task_extents=(1, 1),
        include_exit=False,
        mode="inspect",
        inspection_stage="ir",
        force_hbm_comms=True,
        wrapper_name="dataflow_moe_stage_graph_primfunc",
    )

    assert compiled.primfunc_lowering is not None
    assert compiled.artifact_contract == df.DATAFLOW_ARTIFACT_INSPECTION
    assert compiled.executable is False
    assert "not lowered to CUDA" in compiled.non_executable_reason
    assert compiled.dump_plan()["artifact"] == {
        "contract": "inspection",
        "executable": False,
        "non_executable_reason": ("PrimFunc handlers were retained as IR and were not lowered to CUDA"),
    }
    assert [
        (handler.operator_name, handler.operator_kind, handler.task_param_names) for handler in compiled.primfunc_lowering.handlers
    ] == [
        ("prim_map1", "map", ("expert", "token")),
        ("prim_map2", "map", ("expert", "token")),
        ("prim_finalize", "finalize", ("expert", "token")),
    ]
    map1_script = compiled.primfunc_lowering.handlers[0].prim_func.script()
    map2_script = compiled.primfunc_lowering.handlers[1].prim_func.script()
    finalize_script = compiled.primfunc_lowering.handlers[2].prim_func.script()
    assert "for i in range(range_begin" in map1_script
    assert "Input" in map1_script
    assert "for i in range(range_begin" in map2_script
    assert "Bias" in map2_script
    assert "expert" in finalize_script
    assert "token" in finalize_script
    assert "range_begin" in finalize_script


def test_dataflow_compile_rejects_lowered_but_unlinked_inspection_artifact_launch():
    compiled = df.compile(
        make_reshared_primfunc_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"expert_tile": [64], "hidden_tile": [64]},
        block_size=32,
        task_extents=(1, 1),
        include_exit=False,
        mode="inspect",
        inspection_stage="cuda",
        force_hbm_comms=True,
        wrapper_name="dataflow_moe_stage_graph_primfunc_cuda_inspection",
    )

    assert compiled.primfunc_lowering is not None
    assert compiled.primfunc_lowering.cuda_source
    assert compiled.compile_config.lower_primfunc_handlers is True
    assert compiled.compile_config.link_primfunc_handlers is False
    assert compiled.artifact_contract == df.DATAFLOW_ARTIFACT_INSPECTION
    assert compiled.executable is False
    assert compiled.wrapper_source == ""
    assert "no executable wrapper was generated" in compiled.non_executable_reason
    with pytest.raises(df.DataflowArtifactNotExecutableError, match="no executable wrapper was generated"):
        compiled()
    with pytest.raises(df.DataflowArtifactNotExecutableError, match="no executable wrapper was generated"):
        compiled.persistent_executable()


def test_dataflow_compile_links_reshared_stage_graph_primfunc_handlers():
    compiled = df.compile(
        make_reshared_primfunc_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"expert_tile": [64], "hidden_tile": [64]},
        block_size=32,
        task_extents=(1, 1),
        include_exit=False,
        mode="executable",
        force_hbm_comms=True,
        wrapper_name="dataflow_moe_stage_graph_primfunc_linked",
    )

    source = compiled.wrapper_source
    assert compiled.primfunc_lowering is not None
    assert compiled.artifact_contract == df.DATAFLOW_ARTIFACT_PRODUCTION
    assert compiled.executable is True
    assert compiled.non_executable_reason is None
    assert [(handler.operator_name, handler.operator_kind) for handler in compiled.wrapper_spec.handlers] == [
        ("prim_map1", "map"),
        ("prim_map2", "map"),
        ("prim_finalize", "finalize"),
    ]
    assert "dataflow_primfunc_prim_map1_device_kernel" in source
    assert "dataflow_primfunc_prim_map2_device_kernel" in source
    assert "dataflow_primfunc_prim_finalize_device_kernel" in source
    assert "Input_tensor" in source
    assert "Bias_tensor" in source
    assert "Output_tensor" in source
    assert "slots[input_slots[handler_args.input_slot_offset + 0u]]" in source
    assert "slots[input_slots[handler_args.input_slot_offset + 1u]]" in source
    assert "dataflow_primfunc_prim_map2_device_kernel(Bias_tensor, dataflow_primfunc_slot_field" in source
    assert "dataflow_primfunc_prim_finalize_device_kernel(inter_value, Output_tensor" in source
    assert "handler_args.range_begin" in source
    assert "task_coords[handler_args.task_coord_offset]" in source


def test_dataflow_compile_passes_range_offsets_to_scheduler():
    compiled = df.compile(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"kv": [128, 64]},
        range_offsets={"kv": [1024, 2048]},
        block_size=64,
        task_extents=(2,),
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    iter_instructions = [inst for inst in compiled.plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    assert [(inst.task_id, inst.task_range.begin, inst.task_range.end) for inst in iter_instructions] == [
        (0, 1024, 1152),
        (1, 2048, 2112),
    ]


def test_dataflow_compile_passes_partial_only_to_scheduler():
    compiled = df.compile(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"kv": [128]},
        block_size=64,
        task_extents=(1,),
        include_exit=False,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        partial_only=True,
    )

    assert compiled.options["partial_only"] is True
    assert [inst.opcode for inst in compiled.plan.instructions] == [
        df.DataflowOpcode.ITER,
        df.DataflowOpcode.ITER,
    ]


def test_dataflow_compile_progress_logging_is_opt_in(monkeypatch, capsys):
    monkeypatch.delenv("DATAFLOW_PROGRESS", raising=False)
    monkeypatch.delenv("DATAFLOW_COMPILE_PROGRESS", raising=False)
    monkeypatch.delenv("DATAFLOW_COMPILE_LOG", raising=False)

    df.compile(
        make_program(),
        topology=df.GPUTopology(sm_count=1),
        range_lengths={"kv": [128]},
        block_size=128,
        wrapper_name="dataflow_progress_silent",
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )
    assert capsys.readouterr().out == ""

    compiled = df.compile(
        make_program(),
        topology=df.GPUTopology(sm_count=1),
        range_lengths={"kv": [128]},
        block_size=128,
        wrapper_name="dataflow_progress_loud",
        progress=True,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )
    output = capsys.readouterr().out

    assert compiled.options["progress"] is True
    assert "[dataflow.compile] compile started" in output
    assert "[dataflow.compile] schedule finished" in output
    assert "[dataflow.compile] compile finished" in output


def test_dataflow_jit_compiles_kernel_spec_factory():
    @df.jit
    def make_kernel():
        return df.DataflowKernelSpec(
            program=make_program(),
            topology=(2, 1),
            range_lengths={"kv": [256]},
            block_size=128,
            task_extents=(1,),
            wrapper_name="dataflow_jit_kernel_spec",
            mode="debug",
            options={"_experimental_debug_handler": dataflow_debug.EMPTY_HANDLER.name},
        )

    kernel = make_kernel()

    assert isinstance(make_kernel, df.DataflowJITFunction)
    assert isinstance(kernel, df.DataflowCompiledProgram)
    assert kernel.wrapper_spec.kernel_name == "dataflow_jit_kernel_spec"
    assert kernel.plan.topology == df.GPUTopology(sm_count=2, cluster_size=1)
    assert kernel.plan.task_range_lengths == (256,)
    assert kernel.options["handler_lowering"] == dataflow_debug.EMPTY_HANDLER.name
    assert isinstance(kernel.get_profiler(), df.DataflowProfiler)
    assert callable(kernel.profile_walltime)


def test_dataflow_jit_caches_kernel_specs_by_factory_arguments():
    calls = 0

    @df.jit
    def make_kernel(seq_lens, *, wrapper_name="dataflow_jit_cached_kernel_spec"):
        nonlocal calls
        calls += 1
        return df.DataflowKernelSpec(
            program=make_program(),
            topology=(2, 1),
            range_lengths={"kv": list(seq_lens)},
            block_size=128,
            task_extents=(1,),
            wrapper_name=wrapper_name,
            mode="debug",
            options={"_experimental_debug_handler": dataflow_debug.EMPTY_HANDLER.name},
        )

    first = make_kernel([256])
    second = make_kernel([256])
    third = make_kernel((128,))

    assert first is second
    assert third is not first
    assert calls == 3
    assert make_kernel.cache_size == 2

    make_kernel.clear_cache()
    assert make_kernel.cache_size == 0
    assert make_kernel([256]) is not first
    assert calls == 4


def test_dataflow_compile_config_fingerprint_covers_target_scheduler_and_typed_config():
    def compile_debug(**options):
        target_override = options.pop(
            "target_override",
            df.TargetCapabilitySnapshot.for_cuda(
                (8, 0),
                compiler_version=(12, 8),
            ),
        )
        return df.compile(
            make_program(),
            topology=df.GPUTopology(sm_count=2, cluster_size=1),
            range_lengths={"kv": [256]},
            block_size=128,
            task_extents=(1,),
            mode="debug",
            _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
            target_override=target_override,
            **options,
        )

    baseline = compile_debug()
    repeated = compile_debug()
    changed_target = compile_debug(
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
        )
    )
    changed_scheduler = compile_debug(scheduler_policy="cluster_local")
    changed_scheduler_config = compile_debug(scheduler_config=df.DataflowSchedulerConfig.from_options(level0_queue_order="long_first"))
    changed_semantic_config = compile_debug(semantic_config=df.DataflowSemanticConfig(direct_slot_seed_reduce=True))

    assert baseline.compile_config.fingerprint == repeated.compile_config.fingerprint
    assert baseline.wrapper_source == repeated.wrapper_source
    assert baseline.plan == repeated.plan
    assert (
        len(
            {
                baseline.compile_config.fingerprint,
                changed_target.compile_config.fingerprint,
                changed_scheduler.compile_config.fingerprint,
                changed_scheduler_config.compile_config.fingerprint,
                changed_semantic_config.compile_config.fingerprint,
            }
        )
        == 5
    )
    dumped = changed_semantic_config.dump_plan()["compile_config"]
    assert dumped["semantic_config"]["direct_slot_seed_reduce"] is True
    assert dumped["environment"] == {}
    assert changed_target.compile_config.to_dict()["provenance"]["target"] == "target-override-snapshot"
    assert changed_target.target_capabilities.supports_wgmma is True
    assert changed_target.dump_plan()["target_capabilities"]["arch"] == "sm_90a"


def test_dataflow_jit_ignores_retired_compile_environment(monkeypatch):
    calls = 0
    target = df.TargetCapabilitySnapshot.for_cuda(
        (9, 0),
        compiler_version=(12, 8),
    )

    @df.jit
    def make_kernel(target_override=target):
        nonlocal calls
        calls += 1
        return df.DataflowKernelSpec(
            program=make_program(),
            topology=(2, 1),
            range_lengths={"kv": [256]},
            block_size=128,
            task_extents=(1,),
            target_override=target_override,
            mode="debug",
            options={"_experimental_debug_handler": dataflow_debug.EMPTY_HANDLER.name},
        )

    first = make_kernel()
    repeated = make_kernel()
    monkeypatch.setenv("DATAFLOW_MLA_DIRECT_SLOT_SEED_REDUCE", "1")
    unchanged = make_kernel()

    assert first is repeated
    assert unchanged is first
    assert calls == 3
    assert make_kernel.cache_size == 1
    assert "DATAFLOW_MLA_DIRECT_SLOT_SEED_REDUCE" not in (unchanged.compile_config.environment_dict())


def test_dataflow_jit_cache_key_includes_target_device_identity():
    calls = 0
    active_targets = []

    @df.jit
    def make_kernel(target_capabilities):
        nonlocal calls
        calls += 1
        active_targets.append(df.current_target_capabilities())
        return df.DataflowKernelSpec(
            program=make_program(),
            topology=(2, 1),
            range_lengths={"kv": [256]},
            block_size=128,
            task_extents=(1,),
            mode="debug",
            options={"_experimental_debug_handler": dataflow_debug.EMPTY_HANDLER.name},
        )

    device0 = df.TargetCapabilitySnapshot.for_cuda(
        (12, 0),
        device_ordinal=0,
        compiler_version=(13, 0),
    )
    device1 = df.TargetCapabilitySnapshot.for_cuda(
        (12, 0),
        device_ordinal=1,
        compiler_version=(13, 0),
    )

    first = make_kernel(device0)
    repeated = make_kernel(device0)
    changed_device = make_kernel(device1)

    assert first is repeated
    assert changed_device is not first
    assert calls == 3
    assert active_targets == [device0, device0, device1]
    assert make_kernel.cache_size == 2
    assert first.target_capabilities is device0
    assert changed_device.target_capabilities is device1
    assert first.target_fingerprint != changed_device.target_fingerprint


def test_make_kernel_spec_routes_declared_fields_and_extensible_options():
    spec = df.make_kernel_spec(
        make_program(),
        topology=(2, 1),
        range_lengths={"kv": [256]},
        block_size=128,
        progress=True,
        task_coord_overrides=((0,),),
        compile_flags=("--use_fast_math",),
    )

    assert spec.progress is True
    assert spec.options == {
        "task_coord_overrides": ((0,),),
        "compile_flags": ("--use_fast_math",),
    }


def test_dataflow_compile_rejects_cluster_topology_without_target_capability():
    target_capabilities = df.TargetCapabilitySnapshot.for_cuda(
        (8, 0),
        supports_cluster_launch=False,
        compiler_version=(12, 8),
    )

    with pytest.raises(df.DataflowUnsupportedTargetCapabilityError, match="cluster launch"):
        df.compile(
            make_program(),
            topology=df.GPUTopology(sm_count=2, cluster_size=2),
            range_lengths={"kv": [256]},
            block_size=128,
            task_extents=(1,),
            mode="debug",
            _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
            target_capabilities=target_capabilities,
        )


def test_dataflow_compile_passes_one_target_snapshot_through_primfunc_lowering():
    target_capabilities = df.TargetCapabilitySnapshot.for_cuda(
        (9, 0),
        compiler_version=(12, 8),
    )

    compiled = df.compile(
        make_reshared_primfunc_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"expert_tile": [64], "hidden_tile": [64]},
        block_size=32,
        task_extents=(1, 1),
        mode="inspect",
        inspection_stage="ir",
        target_capabilities=target_capabilities,
    )

    assert compiled.target_capabilities is target_capabilities
    assert compiled.primfunc_lowering is not None
    assert compiled.primfunc_lowering.target_fingerprint == compiled.target_fingerprint
    assert compiled.dump_plan()["primfunc_lowering"]["target_fingerprint"] == compiled.target_fingerprint


def test_dataflow_jit_typed_config_is_not_affected_by_factory_environment_mutation(monkeypatch):
    monkeypatch.setenv("DATAFLOW_MLA_LEVEL0_QUEUE_ORDER", "task")
    target = df.TargetCapabilitySnapshot.for_cuda(
        (9, 0),
        compiler_version=(12, 8),
    )

    @df.jit(cache=False)
    def make_kernel(target_override=target):
        monkeypatch.setenv("DATAFLOW_MLA_LEVEL0_QUEUE_ORDER", "invalid-after-snapshot")
        return df.DataflowKernelSpec(
            program=make_program(),
            topology=(2, 2),
            range_lengths={"kv": [128]},
            block_size=64,
            task_extents=(1,),
            scheduler_policy="cluster_local",
            reduce_strategy="streaming_tree",
            scheduler_config=df.DataflowSchedulerConfig.from_options(level0_queue_order="task"),
            target_override=target_override,
            mode="debug",
            options={"_experimental_debug_handler": dataflow_debug.EMPTY_HANDLER.name},
        )

    compiled = make_kernel()

    assert compiled.compile_config.scheduler_config.get("level0_queue_order") == "task"
    assert "DATAFLOW_MLA_LEVEL0_QUEUE_ORDER" not in (compiled.compile_config.environment_dict())
    assert os.environ["DATAFLOW_MLA_LEVEL0_QUEUE_ORDER"] == "invalid-after-snapshot"


class FakeDataflowExecutable:
    def __init__(self):
        self.setup_timings_ms = {"module_load_ms": 1.25}
        self.launches = []
        self._launch_event_samples = iter((1.0, 2.0, 3.0))
        self._kernel_event_samples = iter((0.5, 0.75, 1.0))

    def open(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def profile_launch(
        self,
        *,
        tensor_args_bytes=None,
        stream=None,
        copy_global_staging=False,
        use_cuda_events=True,
        event_scope="kernel",
    ):
        self.launches.append(
            {
                "tensor_args_bytes": tensor_args_bytes,
                "stream": stream,
                "copy_global_staging": copy_global_staging,
                "use_cuda_events": use_cuda_events,
                "event_scope": event_scope,
            }
        )
        timings = {
            "kernel_host_ms": 9.0,
            "launch_total_host_ms": 10.0,
        }
        if use_cuda_events:
            if event_scope == "launch":
                timings["launch_event_ms"] = next(self._launch_event_samples)
            else:
                timings["pure_kernel_event_ms"] = next(self._kernel_event_samples)
        return SimpleNamespace(timings_ms=timings, execution="fake_execution")


class FakeDataflowCompiledProgram:
    def __init__(self):
        self.executable = FakeDataflowExecutable()
        self.pack_calls = []
        self.persistent_calls = []

    def pack_tensor_args(self, *args, **kwargs):
        self.pack_calls.append((args, kwargs))
        return SimpleNamespace(tensor_args_bytes=b"packed_tensor_args")

    def persistent_executable(self, *args, **kwargs):
        self.persistent_calls.append((args, kwargs))
        return self.executable


def test_dataflow_profiler_defaults_to_pure_kernel_event_samples():
    compiled = FakeDataflowCompiledProgram()
    profiler = df.DataflowProfiler(compiled)

    result = profiler.profile(
        "Q",
        Output="O",
        n_warmup=2,
        n_repeat=3,
        copy_global_staging=True,
        flush_l2_cache=False,
    )

    assert compiled.pack_calls == [(("Q",), {"Output": "O"})]
    assert compiled.persistent_calls == [(("Q",), {"Output": "O"})]
    assert [launch["use_cuda_events"] for launch in compiled.executable.launches] == [
        False,
        False,
        True,
        True,
        True,
    ]
    assert [launch["event_scope"] for launch in compiled.executable.launches] == [
        "kernel",
        "kernel",
        "kernel",
        "kernel",
        "kernel",
    ]
    assert {launch["tensor_args_bytes"] for launch in compiled.executable.launches} == {b"packed_tensor_args"}
    assert all(launch["copy_global_staging"] is True for launch in compiled.executable.launches)
    assert result.metric == "pure_kernel_event_ms"
    assert result.samples_ms == (0.5, 0.75, 1.0)
    assert result.mean_ms == 0.75
    assert result.median_ms == 0.75
    assert result.min_ms == 0.5
    assert result.max_ms == 1.0
    assert result.summary("max") == 1.0
    assert result.quantiles([0.5, 1.0]) == [0.75, 1.0]
    assert result.setup_timings_ms == {"module_load_ms": 1.25}
    assert result.execution == "fake_execution"
    assert result.to_dict()["repeat_iterations"] == 3


def test_dataflow_profiler_can_measure_launch_event_metric():
    compiled = FakeDataflowCompiledProgram()
    profiler = df.DataflowProfiler(compiled)

    result = profiler.profile(
        "Q",
        Output="O",
        n_warmup=1,
        n_repeat=2,
        metric="launch_event_ms",
        flush_l2_cache=False,
    )

    assert [launch["event_scope"] for launch in compiled.executable.launches] == [
        "launch",
        "launch",
        "launch",
    ]
    assert result.metric == "launch_event_ms"
    assert result.samples_ms == (1.0, 2.0)


def test_dataflow_profiler_flushes_l2_before_event_samples(monkeypatch):
    compiled = FakeDataflowCompiledProgram()
    profiler = df.DataflowProfiler(compiled)
    flush_calls = []
    factory_calls = []

    def fake_make_l2_cache_flush(cache_flush_bytes, fast_flush):
        factory_calls.append((cache_flush_bytes, fast_flush))

        def flush():
            flush_calls.append("flush")

        return flush

    monkeypatch.setattr(dataflow_profiler_module, "make_l2_cache_flush", fake_make_l2_cache_flush)

    profiler.profile("Q", Output="O", n_warmup=2, n_repeat=3)

    assert factory_calls == [(256_000_000, True)]
    assert flush_calls == ["flush", "flush", "flush"]


def test_dataflow_compiled_program_profile_walltime_delegates_to_walltime_module(monkeypatch):
    import tilelang.dataflow.walltime as walltime

    compiled = df.compile(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [128]},
        block_size=64,
        task_extents=(1,),
        wrapper_name="dataflow_profile_walltime_entry",
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )
    calls = []

    def fake_profile(compiled_program, *args, **kwargs):
        calls.append((compiled_program, args, kwargs))
        return "walltime-result"

    monkeypatch.setattr(walltime, "profile_compiled_walltime", fake_profile)

    result = compiled.profile_walltime("Q", Output="O", repeat=2, warmup=1, top_k=4)

    assert result == "walltime-result"
    assert calls == [
        (
            compiled,
            ("Q",),
            {
                "Output": "O",
                "repeat": 2,
                "warmup": 1,
                "top_k": 4,
            },
        )
    ]


def test_dataflow_walltime_global_staging_instrumentation_preserves_kernel_abi():
    from tilelang.dataflow.walltime import build_walltime_instrumentation

    compiled = df.compile(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [128]},
        block_size=64,
        task_extents=(1,),
        wrapper_name="dataflow_walltime_staging_patch",
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
        ),
    )

    instrumentation = build_walltime_instrumentation(
        max_queue_len=4,
        timing_global_offset=128,
    )
    instrumented = df.generate_wrapper_source(
        compiled.wrapper_spec,
        instrumentation=instrumentation,
    )

    assert instrumentation.name == "instruction_walltime"
    assert instrumentation.extra_kernel_params == ()
    assert instrumentation.to_dict()["hook_bytes"]["kernel_prologue"] > 0
    assert instrumented != compiled.wrapper_source
    assert compiled.wrapper_source == df.generate_wrapper_source(compiled.wrapper_spec)


def test_dataflow_profile_instrumentation_contracts_compile_through_wrapper_codegen():
    from tilelang.dataflow.walltime import build_walltime_instrumentation

    require_nvcc()
    target = df.TargetCapabilitySnapshot.for_cuda(
        (9, 0),
        compiler_version=(12, 8),
    )
    compiled = df.compile(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [128]},
        block_size=64,
        task_extents=(1,),
        wrapper_name="dataflow_profile_instrumentation_compile",
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
        target_override=target,
    )
    instrumentations = (
        build_walltime_instrumentation(max_queue_len=4),
        build_walltime_instrumentation(max_queue_len=4, span_only=True),
        df.DataflowWrapperInstrumentation(
            name="wrapper_span",
            extra_kernel_params=(
                df.DataflowWrapperKernelParam(
                    name="wrapper_ticks",
                    c_type="uint64_t *",
                ),
            ),
            before_loop=("if (tl::dataflow_is_leader_thread()) {\n  wrapper_ticks[queue_rank * 2u] = clock64();\n}"),
            kernel_epilogue=("if (tl::dataflow_is_leader_thread()) {\n  wrapper_ticks[queue_rank * 2u + 1u] = clock64();\n}"),
        ),
    )

    for instrumentation in instrumentations:
        assert instrumentation.extra_kernel_params
        assert all(isinstance(param, df.DataflowWrapperKernelParam) for param in instrumentation.extra_kernel_params)
        source = df.generate_wrapper_source(
            compiled.wrapper_spec,
            instrumentation=instrumentation,
        )
        cubin, arch = DataflowExecutableKernel(
            kernel_name=compiled.wrapper_spec.kernel_name,
            source=source,
            launch_package=compiled.launch_package,
            topology=compiled.plan.topology,
            target_capabilities=compiled.target_capabilities,
            options=compiled.options,
        ).compile_cubin()
        assert cubin
        assert arch == target.arch


def test_dataflow_walltime_builds_aligned_persistent_staging_suffix():
    from tilelang.dataflow.walltime import (
        TIMING_RECORD_ALIGNMENT,
        TIMING_RECORD_WORDS,
        build_persistent_walltime_executable,
    )

    compiled = df.compile(
        make_reshared_primfunc_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"expert_tile": [64], "hidden_tile": [64]},
        block_size=64,
        task_extents=(1, 1),
        wrapper_name="dataflow_walltime_staging_executable",
    )

    timed = build_persistent_walltime_executable(compiled)
    expected_bytes = compiled.launch_package.queue.queue_count * timed.max_queue_len * TIMING_RECORD_WORDS * 8

    assert timed.timing_global_offset % TIMING_RECORD_ALIGNMENT == 0
    assert timed.timing_bytes == expected_bytes
    assert len(timed.executable.launch_package.global_staging_bytes) == (timed.timing_global_offset + expected_bytes)
    assert timed.executable.launch_package is not compiled.launch_package
    assert timed.executable.source != compiled.wrapper_source
    assert len(compiled.launch_package.global_staging_bytes) <= timed.timing_global_offset


def test_dataflow_memory_layout_validator_reports_structured_regions_and_live_ranges():
    target = df.TargetCapabilitySnapshot.for_cuda(
        (9, 0),
        compiler_version=(12, 8),
        max_dynamic_shared_memory=1 << 20,
    )
    compiled = df.compile(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [128]},
        block_size=64,
        task_extents=(1,),
        wrapper_name="dataflow_layout_validator_contract",
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
        target_override=target,
    )

    report = compiled.validate_memory_layout().require_valid()

    assert report.target_shared_memory_limit == target.max_dynamic_shared_memory
    assert report.shared_memory_within_target_limit
    slot_regions = tuple(region for region in report.regions if region.slot_id is not None)
    assert slot_regions
    assert all(region.offset % region.alignment == 0 for region in report.regions)
    assert all(region.live_range is not None for region in slot_regions)
    assert all(region.placement_reason for region in report.regions)
    assert report.to_dict()["valid"] is True


def test_dataflow_memory_layout_validator_fails_closed_on_resource_and_schema_mismatch():
    target = df.TargetCapabilitySnapshot.for_cuda(
        (9, 0),
        compiler_version=(12, 8),
        max_dynamic_shared_memory=1 << 20,
    )
    compiled = df.compile(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [128]},
        block_size=64,
        task_extents=(1,),
        wrapper_name="dataflow_layout_validator_fail_closed",
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
        target_override=target,
    )
    undersized_target = replace(
        target,
        max_dynamic_shared_memory=compiled.launch_package.shared_memory_bytes - 1,
    )
    mismatched_wrapper = replace(
        compiled.wrapper_spec,
        shared_memory_bytes=compiled.wrapper_spec.shared_memory_bytes + 16,
    )
    out_of_bounds_wrapper = replace(
        compiled.wrapper_spec,
        primfunc_scratch_offset=compiled.launch_package.shared_memory_bytes,
        primfunc_scratch_bytes=1,
    )

    resource_report = df.validate_dataflow_memory_layout(
        compiled.packed_plan,
        compiled.launch_package,
        compiled.wrapper_spec,
        plan=compiled.plan,
        target_capabilities=undersized_target,
    )
    schema_report = df.validate_dataflow_memory_layout(
        compiled.packed_plan,
        compiled.launch_package,
        mismatched_wrapper,
        plan=compiled.plan,
        target_capabilities=target,
    )
    bounds_report = df.validate_dataflow_memory_layout(
        compiled.packed_plan,
        compiled.launch_package,
        out_of_bounds_wrapper,
        plan=compiled.plan,
        target_capabilities=target,
    )

    assert not resource_report.valid
    assert not resource_report.shared_memory_within_target_limit
    assert any("exceeds target limit" in error for error in resource_report.errors)
    assert not schema_report.valid
    assert any("shared-memory size mismatch" in error for error in schema_report.errors)
    assert not bounds_report.valid
    assert any("outside" in error for error in bounds_report.errors)
    with pytest.raises(ValueError, match="invalid Dataflow memory layout"):
        resource_report.require_valid()


def test_dataflow_walltime_rows_include_scheduled_operator_name():
    from tilelang.dataflow.walltime import TIMING_RECORD_WORDS, timing_rows

    compiled = df.compile(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [128]},
        block_size=64,
        task_extents=(1,),
        wrapper_name="dataflow_walltime_operator_rows",
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )
    max_queue_len = max(len(queue) for queue in compiled.plan.queues.values())
    timing = bytearray(compiled.launch_package.queue.queue_count * max_queue_len * TIMING_RECORD_WORDS * 8)
    struct.pack_into("<8Q", timing, 0, 1000, 1100, 1300, 1400, 8, 1, 0, 0)

    rows = timing_rows(compiled, timing, clock_khz=1_000_000, sample=0)

    assert len(rows) == 1
    assert rows[0]["operator_name"] == compiled.plan.queue(0)[0].operator_name


def test_dataflow_walltime_balance_report_summarizes_finish_spread_and_top_sms():
    from tilelang.dataflow.walltime import format_walltime_balance_report

    detail_rows = [
        {
            "sample": 0,
            "logical_sm": 0,
            "real_smid": 8,
            "opcode": "iter",
            "recv_us": 1.0,
            "handler_us": 20.0,
            "send_us": 2.0,
            "total_us": 23.0,
            "start_time_ns": 100_000,
            "end_time_ns": 123_000,
        },
        {
            "sample": 0,
            "logical_sm": 1,
            "real_smid": 9,
            "opcode": "reduce_update",
            "recv_us": 12.0,
            "handler_us": 4.0,
            "send_us": 1.0,
            "total_us": 17.0,
            "start_time_ns": 100_000,
            "end_time_ns": 145_000,
        },
    ]
    summary_rows = [
        {
            "sample": 0,
            "logical_sm": 0,
            "real_smids": "8",
            "count": 1,
            "finish_time_us": 23.0,
            "active_span_us": 23.0,
            "recv_us_sum": 1.0,
            "handler_us_sum": 20.0,
            "send_us_sum": 2.0,
            "idle_gap_us": 0.0,
        },
        {
            "sample": 0,
            "logical_sm": 1,
            "real_smids": "9",
            "count": 1,
            "finish_time_us": 45.0,
            "active_span_us": 45.0,
            "recv_us_sum": 12.0,
            "handler_us_sum": 4.0,
            "send_us_sum": 1.0,
            "idle_gap_us": 28.0,
        },
    ]

    report = format_walltime_balance_report(detail_rows, summary_rows, top_k=2)

    assert "finish min/p50/p95/max/spread: 23.000 / 34.000 / 43.900 / 45.000 / 22.000 us" in report
    assert "top logical SMs by finish_time_us:" in report
    assert "logical_sm=  1 real_smids=[9]" in report
    assert "recv=12.000us handler=4.000us send=1.000us" in report
    assert "opcode total_us stats:" in report


def test_dataflow_compile_can_reuse_hbm_flags_in_launch_package():
    compiled = df.compile(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [384, 256]},
        block_size=128,
        task_extents=(2,),
        reuse_hbm_flags=True,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    hbm_comms = [comm for comm in compiled.packed_plan.comms if comm.kind in (3, 4)]

    assert hbm_comms
    assert compiled.options["reuse_hbm_flags"] is True
    assert compiled.launch_package.flag_count == 1
    assert {slot.flag_index for slot in compiled.packed_plan.slots} == {0}
    assert {comm.flag_index for comm in hbm_comms} == {0}
    assert sorted(set(comm.flag_epoch for comm in hbm_comms)) == [1, 2]


def test_dataflow_compile_can_dump_schedule_pic(tmp_path):
    df.compile(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [256]},
        block_size=128,
        task_extents=(1,),
        pic=True,
        pic_dir=tmp_path,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    txt_files = sorted(tmp_path.glob("*.txt"))
    md_files = sorted(tmp_path.glob("*.md"))
    assert len(txt_files) == 1
    assert len(md_files) == 1
    assert "Dataflow Schedule" in txt_files[0].read_text()


def test_dataflow_compile_packs_user_tensor_arguments():
    compiled = df.compile(
        make_program(),
        topology=df.GPUTopology(sm_count=1),
        range_lengths={"kv": [128]},
        block_size=128,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    packed = compiled.pack_tensor_args(0x1000, K=0x2000, V=0x3000, O=0x4000)

    assert len(packed.records) == 4
    assert packed.record_size == df.TENSOR_ARG_STRUCT.size
    assert len(packed.tensor_args_bytes) == 4 * df.TENSOR_ARG_STRUCT.size
    assert [record.data_ptr for record in packed.records] == [0x1000, 0x2000, 0x3000, 0x4000]
    assert [record.flags for record in packed.records] == [df.DATAFLOW_TENSOR_ARG_FLAG_RAW_POINTER] * 4
    assert struct.unpack_from("<Q", packed.tensor_args_bytes, 0)[0] == 0x1000


class FakeCudaArray:
    __cuda_array_interface__ = {
        "data": (0xABC0, False),
        "shape": (2, 8),
        "typestr": "<f4",
        "version": 3,
    }


class FakeTorchTensor:
    is_cuda = True
    shape = (4, 4)
    dtype = "torch.float16"

    @property
    def __cuda_array_interface__(self):
        raise KeyError("torch float8 does not expose a CUDA array typestr")

    def data_ptr(self):
        return 0xDEF0


def test_dataflow_compile_packs_cuda_tensor_like_arguments():
    compiled = df.compile(
        make_program(),
        topology=df.GPUTopology(sm_count=1),
        range_lengths={"kv": [128]},
        block_size=128,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    packed = compiled.pack_tensor_args(
        FakeCudaArray(),
        K=FakeTorchTensor(),
        V=0x3000,
        O=0x4000,
    )

    assert packed.records[0].data_ptr == 0xABC0
    assert packed.records[0].ndim == 2
    assert packed.records[0].dtype_code == df.DATAFLOW_TENSOR_DTYPE_FLOAT
    assert packed.records[0].dtype_bits == 32
    assert packed.records[0].flags == df.DATAFLOW_TENSOR_ARG_FLAG_CUDA_ARRAY_INTERFACE
    assert packed.records[1].data_ptr == 0xDEF0
    assert packed.records[1].ndim == 2
    assert packed.records[1].dtype_code == df.DATAFLOW_TENSOR_DTYPE_FLOAT
    assert packed.records[1].dtype_bits == 16
    assert packed.records[1].flags == df.DATAFLOW_TENSOR_ARG_FLAG_TORCH
    assert packed.metadata[0].shape == (2, 8)
    assert packed.metadata[0].strides_bytes == (32, 4)
    assert packed.metadata[0].device_type == "cuda"
    assert packed.metadata[1].shape == (4, 4)
    assert packed.metadata[1].strides_bytes == (8, 2)
    assert packed.metadata[1].dtype_code == df.DATAFLOW_TENSOR_DTYPE_FLOAT
    assert packed.metadata[1].dtype_bits == 16


def test_dataflow_compile_rejects_invalid_user_tensor_arguments_before_launch():
    compiled = df.compile(
        make_reshared_primfunc_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"expert_tile": [64], "hidden_tile": [64]},
        block_size=64,
    )

    with pytest.raises(TypeError, match="Missing Dataflow tensor argument"):
        compiled.pack_tensor_args(0x1000)
    with pytest.raises(TypeError, match="Unknown Dataflow tensor argument"):
        compiled.pack_tensor_args(0x1000, Bias=0x2000, Bad=0x4000)
    with pytest.raises(TypeError, match="CUDA tensor-like"):
        compiled("Input", "Bias", "Output")


def test_dataflow_compile_builds_persistent_executable_without_launch():
    compiled = df.compile(
        make_reshared_primfunc_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"expert_tile": [64], "hidden_tile": [64]},
        block_size=64,
        wrapper_name="dataflow_persistent_executable_smoke",
    )

    executable = compiled.persistent_executable(Input=0x1000, Bias=0x2000, Output=0x3000)

    assert isinstance(executable, df.DataflowPersistentExecutable)
    assert not executable.is_open
    assert executable.kernel_name == "dataflow_persistent_executable_smoke"
    assert executable.launch_package is compiled.launch_package
    assert executable.topology == compiled.plan.topology
    assert len(executable.tensor_args_bytes) == 3 * df.TENSOR_ARG_STRUCT.size
    assert compiled.compile_config.mode == df.DATAFLOW_COMPILE_MODE_EXECUTABLE
    assert compiled.artifact_contract == df.DATAFLOW_ARTIFACT_PRODUCTION


def test_dataflow_compile_rejects_empty_debug_wrapper_launch():
    compiled = df.compile(
        make_program(),
        topology=df.GPUTopology(sm_count=1),
        range_lengths={"kv": [128]},
        block_size=128,
        wrapper_name="dataflow_empty_executable",
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    with pytest.raises(df.DataflowArtifactNotExecutableError, match="empty experimental handlers"):
        compiled()
    with pytest.raises(df.DataflowArtifactNotExecutableError, match="empty experimental handlers"):
        compiled.persistent_executable()


def test_dataflow_compile_rejects_invalid_inputs():
    with pytest.raises(TypeError, match="DataflowProgram"):
        df.compile(
            object(),
            topology=df.GPUTopology(sm_count=1),
            range_lengths={"kv": [128]},
            block_size=128,
        )

    incomplete = T.dataflow_program(task_domain=("batch", "head")).partial(
        split_kv(Q="Q", K="K", V="V"),
        task_args=("seq", "head"),
        range_axis="kv",
    )
    with pytest.raises(ValueError, match="missing a reduce stage"):
        df.compile(
            incomplete,
            topology=df.GPUTopology(sm_count=1),
            range_lengths={"kv": [128]},
            block_size=128,
            mode="debug",
            _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
        )
