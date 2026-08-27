from __future__ import annotations

from dataclasses import replace

import pytest

import tilelang.language as T
import tilelang.dataflow as df
from tilelang.dataflow.experimental import debug_handlers as dataflow_debug
from tilelang.dataflow.walltime import generate_walltime_wrapper_source


@T.dataflow_intermediate
class AttnInter:
    lse: T.float32
    o: T.Tensor((2, 8), T.float16)


@T.dataflow.iter(range=("kv_begin", "kv_end"))
def split_kv(seq: T.int32, head: T.int32, Q, K, V) -> AttnInter:
    raise AssertionError("Dataflow iter body should not execute during wrapper generation")


@T.dataflow.reduce
def combine(items: list[AttnInter]) -> AttnInter:
    raise AssertionError("Dataflow reduce body should not execute during wrapper generation")


@T.dataflow.finalize
def finalize(inter: AttnInter, seq: T.int32, head: T.int32, Output) -> None:
    raise AssertionError("Dataflow finalize body should not execute during wrapper generation")


@T.dataflow_intermediate
class UpShard:
    value: T.Tensor((2, 4), T.float16)


@T.dataflow_intermediate
class UpFull:
    value: T.Tensor((2, 8), T.float16)


@T.dataflow_intermediate
class HiddenShard:
    value: T.Tensor((2, 4), T.float16)


@T.dataflow.map(range=("expert_begin", "expert_end"))
def moe_map1(expert: T.int32, token: T.int32, Input, W1) -> UpShard:
    raise AssertionError("Dataflow map body should not execute during wrapper generation")


@T.dataflow.map(range=("hidden_begin", "hidden_end"))
def moe_map2(parts: list[UpShard], expert: T.int32, token: T.int32, W2) -> HiddenShard:
    raise AssertionError("Dataflow map body should not execute during wrapper generation")


@T.dataflow.finalize
def finalize_hidden(hidden: HiddenShard, expert: T.int32, token: T.int32, Output) -> None:
    raise AssertionError("Dataflow finalize body should not execute during wrapper generation")


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


def make_packed_plan():
    plan = df.schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [384, 256]},
        block_size=128,
        task_extents=(2,),
    )
    return df.pack_instruction_plan(plan)


def make_reshared_packed_plan():
    plan = df.schedule(
        make_reshared_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"expert_tile": [256], "hidden_tile": [256]},
        block_size=128,
        task_extents=(1,),
        include_exit=False,
        force_hbm_comms=True,
    )
    return df.pack_instruction_plan(plan)


def make_cluster_only_reshared_packed_plan():
    plan = df.schedule(
        make_reshared_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"expert_tile": [256], "hidden_tile": [256]},
        block_size=128,
        task_extents=(1,),
        include_exit=False,
        force_hbm_comms=False,
    )
    return df.pack_instruction_plan(plan)


def make_debug_wrapper_spec(packed_plan, **kwargs):
    return dataflow_debug.populate_wrapper_spec(
        df.build_wrapper_spec(packed_plan, **kwargs),
        dataflow_debug.EMPTY_HANDLER,
    )


def test_build_wrapper_spec_from_packed_plan():
    packed = make_packed_plan()
    spec = df.build_wrapper_spec(packed, kernel_name="123 bad-kernel")

    assert spec.kernel_name == "dataflow_wrapper_123_bad_kernel"
    assert spec.queue_indexing == "cta_rank"
    assert [(handler.handler_id, handler.operator_name, handler.operator_kind, handler.symbol_name) for handler in spec.handlers] == [
        (0, "split_kv", "iter", "dataflow_handler_0_split_kv"),
        (1, "combine", "reduce", "dataflow_handler_1_combine"),
        (2, "finalize", "finalize", "dataflow_handler_2_finalize"),
        (3, "combine", "reduce", "dataflow_handler_3_combine"),
    ]
    assert [handler.handler_variant_key for handler in spec.handlers] == list(packed.handler_variant_keys)


def test_build_wrapper_spec_preserves_stage_graph_handler_kinds():
    spec = df.build_wrapper_spec(make_reshared_packed_plan(), kernel_name="dataflow_moe_graph")

    assert [(handler.handler_id, handler.operator_name, handler.operator_kind, handler.symbol_name) for handler in spec.handlers] == [
        (0, "moe_map1", "map", "dataflow_handler_0_moe_map1"),
        (1, "moe_map2", "map", "dataflow_handler_1_moe_map2"),
        (2, "finalize_hidden", "finalize", "dataflow_handler_2_finalize_hidden"),
    ]


def test_build_wrapper_spec_derives_transport_specialization_from_packed_plan():
    from tilelang.dataflow.abi_schema import (
        DATAFLOW_COMM_KIND_ABI_VALUES,
        DATAFLOW_SLOT_FLAG_CLUSTER_GATED_PUSH,
    )

    cluster_only = make_cluster_only_reshared_packed_plan()
    cluster_spec = df.build_wrapper_spec(cluster_only)
    assert not cluster_spec.has_hbm_comms
    assert cluster_spec.has_cluster_send_comms
    assert not cluster_spec.has_cluster_release_comms
    assert not cluster_spec.has_gated_cluster_push
    assert cluster_spec.precomputed_cluster_source_lifetime

    hbm_spec = df.build_wrapper_spec(make_reshared_packed_plan())
    assert hbm_spec.has_hbm_comms

    send = next(comm for comm in cluster_only.comms if comm.kind == DATAFLOW_COMM_KIND_ABI_VALUES["cluster_send"])
    slots = list(cluster_only.slots)
    slots[send.dst_slot_id] = replace(
        slots[send.dst_slot_id],
        flags=slots[send.dst_slot_id].flags | DATAFLOW_SLOT_FLAG_CLUSTER_GATED_PUSH,
    )
    gated = replace(
        cluster_only,
        slots=tuple(slots),
        comms=cluster_only.comms
        + (
            replace(
                send,
                kind=DATAFLOW_COMM_KIND_ABI_VALUES["cluster_release"],
            ),
        ),
    )
    gated_spec = df.build_wrapper_spec(gated)
    assert gated_spec.has_cluster_release_comms
    assert gated_spec.has_gated_cluster_push
    assert gated_spec.precomputed_cluster_source_lifetime


def test_pack_precomputes_cluster_source_reuse_wait_action():
    plan = df.schedule(
        make_reshared_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"expert_tile": [256], "hidden_tile": [256]},
        block_size=128,
        task_extents=(1,),
        include_exit=False,
        force_hbm_comms=False,
    )
    queue = plan.queue(0)
    send = next(
        comm
        for comm in plan.comms
        if comm.kind is df.DataflowCommKind.CLUSTER_SEND and comm.resolved_dispatch_instruction_id == queue[0].instruction_id
    )
    recv = next(
        comm
        for comm in plan.comms
        if comm.kind is df.DataflowCommKind.CLUSTER_RECV and comm.resolved_dispatch_instruction_id == queue[1].instruction_id
    )
    slots = list(plan.slots)
    slots[recv.target_slot_id] = replace(
        slots[recv.target_slot_id],
        shared_storage_id=slots[send.source_slot_id].shared_storage_id,
    )
    packed = df.pack_instruction_plan(replace(plan, slots=tuple(slots)))
    recv_instruction = packed.instructions[packed.queue_offsets[0] + 1]
    recv_args = packed.args[recv_instruction.arg_offset // packed.arg_record_size]

    assert recv_args.reserved0 == 1

    spec = make_debug_wrapper_spec(
        packed,
        kernel_name="dataflow_cluster_source_reuse_wrapper",
    )
    source = df.generate_wrapper_source(spec)

    assert spec.has_cluster_source_reuse_wait_actions
    assert "kDataflowHasClusterSourceReuseWaitActions = true" in source
    assert "const uint32_t cluster_source_wait_action" in source
    source_reuse = source.index("const uint32_t cluster_source_wait_action")
    source_fence = source.index("dataflow_wait_cluster_send_source_reads", source_reuse)
    cluster_recv = source.index("dataflow_dispatch_cluster_recv_comms", source_fence)
    assert source_reuse < source_fence < cluster_recv


def test_cluster_only_wrapper_emits_compile_time_transport_elision():
    spec = make_debug_wrapper_spec(make_cluster_only_reshared_packed_plan())
    source = df.generate_wrapper_source(spec)

    assert "kDataflowHasHBMComms = false" in source
    assert "kDataflowHasClusterReleaseComms = false" in source
    assert "kDataflowHasGatedClusterPush = false" in source
    assert "if (tl_dataflow_generated::kDataflowHasHBMComms)" in source
    assert "if (kDataflowHasHBMComms)" in source


def test_generate_wrapper_source_contains_queue_drain_and_pre_post_comm_dispatch():
    spec = replace(
        make_debug_wrapper_spec(make_packed_plan(), kernel_name="dataflow_test_wrapper"),
        cluster_size=2,
    )
    source = df.generate_wrapper_source(spec)

    assert "#include <tl_templates/cuda/copy.h>" in source
    assert "#include <tl_templates/cuda/dataflow_comm.h>" in source
    assert 'extern "C" __global__ void dataflow_test_wrapper(' in source
    assert "const tl::DataflowInstruction *instructions" in source
    assert "const uint32_t *queue_offsets" in source
    assert "const uint32_t *queue_lengths" in source
    assert "tl::DataflowQueue queue{instructions, queue_offsets, queue_lengths};" in source
    assert "uint32_t queue_rank = tl::dataflow_cta_rank_in_grid();" in source
    assert "tl::dataflow_queue_length(queue, queue_rank)" in source
    assert "tl::dataflow_queue_load(queue, queue_rank, pc)" in source
    assert "tl::dataflow_opcode_is_exit(inst)" in source
    assert "tl::dataflow_handler_args(arg_base, inst.arg_offset)" in source
    assert "const tl::DataflowHandlerArgs &handler_args" in source
    assert "const tl::DataflowTensorArg *tensor_args" in source
    assert "const uint32_t *input_slots" in source
    assert "const uint32_t *task_coords" in source
    assert "kDataflowTensorArgCount" in source
    assert "const uint32_t *barrier_init_offsets" in source
    assert "const uint32_t *barrier_init_lengths" in source
    assert "const uint32_t *barrier_init_indices" in source
    assert (
        "dataflow_init_queue_recv_barriers(\n"
        "      queue_rank, barrier_init_offsets, barrier_init_lengths, barrier_init_indices, barriers)" in source
    )
    assert "tl::mbarrier_init(barriers[barrier_index], 1u);" in source
    assert "tl::fence_barrier_init();" in source

    queue_init_helper = source[
        source.index("TL_DEVICE void dataflow_init_queue_recv_barriers(") : source.index("TL_DEVICE void dataflow_init_cluster_ack_state(")
    ]
    ack_init_helper = source[
        source.index("TL_DEVICE void dataflow_init_cluster_ack_state(") : source.index("TL_DEVICE void dataflow_init_hbm_recv_state(")
    ]
    hbm_init_helper = source[
        source.index("TL_DEVICE void dataflow_init_hbm_recv_state(") : source.index(
            "TL_DEVICE void dataflow_wait_cluster_send_source_reads("
        )
    ]
    assert "__syncthreads();" not in queue_init_helper
    assert "__syncthreads();" not in ack_init_helper
    assert "__syncthreads();" not in hbm_init_helper

    kernel_entry = source.index('extern "C" __global__ void dataflow_test_wrapper(')
    init_call = source.index("tl_dataflow_generated::dataflow_init_queue_recv_barriers(", kernel_entry)
    cluster_collective = source.index("tl::cluster_sync();", init_call)
    loop = source.index("for (uint32_t pc = 0; pc < length; ++pc)", cluster_collective)
    assert init_call < cluster_collective < loop
    assert "tl::dataflow_instruction_cluster_recv_count(inst)" in source
    assert "tl::dataflow_instruction_cluster_send_count(inst)" in source
    assert "tl::dataflow_complete_cluster_recv_batch();" in source
    assert "tl::dataflow_begin_cluster_send_batch();" in source
    assert (
        "dataflow_dispatch_cluster_recv_relaxed(\n        comm, slots, shared_base, scratch_base, barriers,\n        cluster_ack_barriers);"
    ) in source
    assert (
        "dataflow_dispatch_cluster_send_relaxed(\n"
        "        comm, slots, shared_base, scratch_base, barriers,\n"
        "        cluster_ack_barriers, cluster_ack_pending);"
    ) in source
    assert "tl::mbarrier_init(cluster_ack_barriers[comm.flag_index], 1u);" in source
    assert "tl::dataflow_slot_uses_cluster_gated_push(dst_slot)" in source
    assert "tl::DataflowCommKind::kClusterRelease" in source
    assert "tl::mbarrier_wait(cluster_ack_barriers[comm.flag_index], 0);" in source
    assert "dataflow_pull_cluster" not in source
    assert "cluster_ack_pending[comm.flag_index >> 5]" not in source
    assert "dataflow_wait_cluster_send_source_reads(" in source
    assert "tl::tma_store_arrive();" in source
    assert "tl::tma_store_wait<0>();" in source
    assert "tl::mbarrier_arrive(" in source
    assert "cluster_ack_barriers[comm.flag_index], comm.peer_cta_rank, 1u" in source
    assert "barriers, false);" not in source
    assert "cluster_recv_pending" not in source
    assert "cluster_send_started" not in source

    kernel_loop = source.index("for (uint32_t pc = 0; pc < length; ++pc)")
    recv = source.index("tl_dataflow_generated::dataflow_dispatch_cluster_recv_comms", kernel_loop)
    hbm_wait = source.index("tl_dataflow_generated::dataflow_dispatch_hbm_recv_wait_comms", recv)
    handler = source.index("tl_dataflow_generated::dataflow_dispatch_handler", kernel_loop)
    hbm_issue = source.index("tl_dataflow_generated::dataflow_dispatch_hbm_recv_issue_comms", handler)
    send = source.index("tl_dataflow_generated::dataflow_dispatch_send_comms", kernel_loop)
    assert recv < hbm_wait < handler < hbm_issue < send
    source_read_wait = source.index("tl_dataflow_generated::dataflow_finish_cluster_send_source_reads", send)
    assert send < source_read_wait

    assert "case 0u:" in source
    assert (
        "dataflow_handler_0_split_kv(inst, handler_args, arg_base, slots, tensor_args, input_slots, task_coords, shared_base, global_base);"
        in source
    )
    assert "case 1u:" in source
    assert (
        "dataflow_handler_1_combine(inst, handler_args, arg_base, slots, tensor_args, input_slots, task_coords, shared_base, global_base);"
        in source
    )
    assert "case 2u:" in source
    assert (
        "dataflow_handler_2_finalize(inst, handler_args, arg_base, slots, tensor_args, input_slots, task_coords, shared_base, global_base);"
        in source
    )


def test_hbm_only_production_wrapper_elides_startup_collective():
    spec = replace(
        make_debug_wrapper_spec(make_reshared_packed_plan()),
        cluster_size=16,
    )
    assert spec.has_hbm_comms
    assert not spec.has_cluster_send_comms

    production = df.generate_wrapper_source(spec)
    kernel_entry = production.index(f'extern "C" __global__ void {spec.kernel_name}(')
    init_call = production.index("tl_dataflow_generated::dataflow_init_queue_recv_barriers(", kernel_entry)
    loop = production.index("for (uint32_t pc = 0; pc < length; ++pc)", init_call)
    startup = production[init_call:loop]
    assert "tl::cluster_sync();" not in startup
    assert "__syncthreads();" not in startup

    instrumented = df.generate_wrapper_source(
        spec,
        instrumentation=df.DataflowWrapperInstrumentation(name="visibility"),
    )
    kernel_entry = instrumented.index(f'extern "C" __global__ void {spec.kernel_name}(')
    init_call = instrumented.index("tl_dataflow_generated::dataflow_init_queue_recv_barriers(", kernel_entry)
    loop = instrumented.index("for (uint32_t pc = 0; pc < length; ++pc)", init_call)
    startup = instrumented[init_call:loop]
    assert "tl::cluster_sync();" not in startup
    assert startup.count("__syncthreads();") == 1


def test_launch_pacing_is_preserved_in_production_and_walltime_wrappers():
    spec = replace(
        make_debug_wrapper_spec(make_packed_plan()),
        cluster_size=4,
        launch_pacing=df.DataflowWrapperLaunchPacing(
            delay_ns=115_000,
            cluster_stagger_ns=15_000,
            cluster_stagger_group_size=2,
        ),
    )

    production = df.generate_wrapper_source(spec)
    instrumented = generate_walltime_wrapper_source(spec, max_queue_len=8)
    pacing_source = "((blockIdx.x / 4u) /\n         2u) *\n        15000ull"

    for source in (production, instrumented):
        assert "115000ull" in source
        assert pacing_source in source
        assert "dataflow_delay_now - dataflow_delay_start" in source


@pytest.mark.parametrize(
    "values, message",
    (
        ({"delay_ns": -1}, "delay must be non-negative"),
        ({"cluster_stagger_ns": -1}, "stagger must be non-negative"),
        ({"cluster_stagger_group_size": 0}, "group size must be positive"),
    ),
)
def test_launch_pacing_rejects_invalid_typed_values(values, message):
    with pytest.raises(ValueError, match=message):
        df.DataflowWrapperLaunchPacing(**values)


def test_wrapper_skips_redundant_send_barrier_only_for_synchronized_handlers():
    spec = make_debug_wrapper_spec(
        make_packed_plan(),
        kernel_name="dataflow_handler_completion_sync",
    )
    synchronized = replace(
        spec.handler_sources[0],
        completion_synchronized=True,
    )
    spec = replace(
        spec,
        handler_sources=(synchronized, *spec.handler_sources[1:]),
    )

    source = df.generate_wrapper_source(spec)

    assert "cluster_send_count != 0u && !handler_completion_synchronized" in source
    assert "const bool handler_completion_synchronized =" in source
    first_case = source.index("case 0u:")
    second_case = source.index("case 1u:")
    assert "return true;" in source[first_case:second_case]
    third_case = source.index("case 2u:")
    assert "return false;" in source[second_case:third_case]


def test_generate_wrapper_source_uses_permanent_cluster_inbox_and_source_read_fence():
    spec = replace(
        make_debug_wrapper_spec(
            make_packed_plan(),
            kernel_name="dataflow_cluster_inbox_wrapper",
        ),
        cluster_inbox_offset=4096,
        cluster_inbox_bytes=48,
        cluster_size=2,
    )

    source = df.generate_wrapper_source(spec)

    assert "static constexpr uint32_t kDataflowClusterInboxOffset = 4096u;" in source
    assert "static constexpr uint32_t kDataflowClusterInboxBytes = 48u;" in source
    assert "kDataflowClusterCreditGated" not in source
    assert "TL_DEVICE bool dataflow_dispatch_send_comms(" in source
    assert spec.precomputed_cluster_source_lifetime
    assert not spec.has_cluster_source_reuse_wait_actions
    assert "kDataflowPrecomputedClusterSourceLifetime = true" in source
    assert "kDataflowHasClusterSourceReuseWaitActions = false" in source
    assert "bool cluster_send_source_pending = false;" not in source
    assert "const uint32_t cluster_source_wait_action" not in source
    assert "TL_DEVICE uint32_t dataflow_slot_shared_arena_offset(" in source
    assert "? kDataflowPrimFuncScratchOffset" in source
    assert ": kDataflowSharedSlotBaseOffset" in source
    assert "const uint32_t source_begin =\n          dataflow_slot_shared_arena_offset(source_slot);" in source
    overlap_helper = source[
        source.index("TL_DEVICE bool dataflow_slot_overlaps_pending_cluster_source(") : source.index(
            "TL_DEVICE bool dataflow_instruction_writes_pending_cluster_source("
        )
    ]
    assert "!tl::dataflow_slot_is_scratch_backed(slot)" not in overlap_helper
    assert "dataflow_instruction_releases_cluster_slot(" in source
    assert "tl::tma_store_wait<1>();" in source
    epilogue_helper = source[
        source.index("TL_DEVICE void dataflow_finish_cluster_send_source_reads()") : source.index(
            "TL_DEVICE uint32_t dataflow_slot_shared_arena_offset("
        )
    ]
    assert "tl::tma_store_wait<0>();" in epilogue_helper
    assert "__syncthreads();" not in epilogue_helper
    assert "hbm_recv_issued, nullptr, nullptr);" in source
    assert "dataflow_instruction_sends_scratch_cluster_slot" not in source
    assert "TL_DEVICE void dataflow_init_cluster_ack_state(" in source
    assert "initialized_gated_barrier" in source
    assert "tl::mbarrier_init(cluster_ack_barriers[comm.flag_index], 1u);" in source
    loop = source.index("for (uint32_t pc = 0; pc < length; ++pc)")
    ack_init = source.index(
        "tl_dataflow_generated::dataflow_init_cluster_ack_state(",
        source.index(f'extern "C" __global__ void {spec.kernel_name}'),
    )
    cluster_sync = source.index("tl::cluster_sync();", ack_init)
    cluster_recv = source.index("dataflow_dispatch_cluster_recv_comms", loop)
    hbm_wait = source.index("dataflow_dispatch_hbm_recv_wait_comms", cluster_recv)
    handler = source.index("dataflow_dispatch_handler", hbm_wait)
    hbm_issue = source.index("dataflow_dispatch_hbm_recv_issue_comms", handler)
    assert ack_init < cluster_sync < cluster_recv < hbm_wait < handler < hbm_issue
    assert "kDataflowHasClusterSendComms" in source
    assert "tl::tma_load(" in source
    assert "tl::tma_load<tl::CacheHintSm90::EVICT_FIRST>(" not in source
    assert "tl::tma_store<tl::CacheHintSm90::EVICT_LAST>(" in source
    assert spec.to_dict()["cluster_inbox_offset"] == 4096
    assert spec.to_dict()["cluster_inbox_bytes"] == 48


def test_generate_wrapper_source_resolves_scratch_backed_slots_through_unified_accessor():
    package = df.build_launch_package(make_packed_plan())
    spec = df.build_wrapper_spec(
        make_packed_plan(),
        kernel_name="dataflow_scratch_slot_wrapper",
        launch_package=package,
    )
    spec = replace(
        spec,
        primfunc_scratch_offset=2048,
        primfunc_scratch_bytes=8192,
    )
    spec = dataflow_debug.populate_wrapper_spec(spec, dataflow_debug.SCALAR_U32_HANDLER)

    source = df.generate_wrapper_source(spec)

    assert "kDataflowSlotFlagScratchBacked" in source
    assert "kDataflowPrimFuncScratchOffset = 2048u" in source
    assert "tl::dataflow_slot_shared_ptr(shared_base, slot)" not in source
    assert "tl::dataflow_slot_shared_ptr(\n      shared_base, dataflow_primfunc_scratch_base(shared_base), slot)" in source
    assert "tl_dataflow_generated::dataflow_slot_shared_ptr(shared_base, slot)" in source


def test_generate_wrapper_source_direct_global_hbm_recv_waits_on_target_slot_flag():
    spec = df.build_wrapper_spec(make_reshared_packed_plan(), kernel_name="dataflow_direct_global")
    spec = dataflow_debug.populate_wrapper_spec(
        replace(spec, primfunc_use_global_slot_fields=True),
        dataflow_debug.SCALAR_U32_HANDLER,
    )

    source = df.generate_wrapper_source(spec)

    assert "src_slot.global_offset == dst_slot.global_offset" in source
    issue_branch = source.index("tl::DataflowCommKind::kHBMRecvIssue")
    wait_branch = source.index("tl::DataflowCommKind::kHBMRecvWait", issue_branch)
    blocking_wait = source.index("tl::dataflow_recv_hbm_direct_global(", wait_branch)
    assert "tl::dataflow_recv_hbm_direct_global(" not in source[issue_branch:wait_branch]
    assert wait_branch < blocking_wait
    assert "kDataflowBarrierWordCount" in source
    assert "dataflow_init_hbm_recv_state(hbm_recv_issued);" in source
    assert "flags, barriers, hbm_recv_issued);" in source
    assert (
        "tl::dataflow_recv_hbm_direct_global(\n"
        "        flags,\n"
        "        tl::dataflow_select_index(comm.flag_index, dst_slot.flag_index),\n"
        "        comm.flag_epoch);"
    ) in source
    assert "tl::dataflow_select_index(comm.flag_index, src_slot.flag_index)" not in source


def test_compile_returns_wrapper_skeleton_source():
    compiled = dataflow_debug.compile(
        make_program(),
        handler=dataflow_debug.EMPTY_HANDLER,
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [256]},
        block_size=128,
        wrapper_name="attention_wrapper",
    )

    assert compiled.wrapper_spec.kernel_name == "attention_wrapper"
    assert compiled.wrapper_spec.barrier_count == compiled.launch_package.barrier_count
    assert compiled.wrapper_spec.cluster_size == 1
    assert compiled.wrapper_spec.tensor_arg_count == 4
    assert compiled.wrapper_source == df.generate_wrapper_source(compiled.wrapper_spec)
    assert (
        'extern "C" __global__ void '
        f"__launch_bounds__({compiled.wrapper_spec.launch_bound_threads}, 1) attention_wrapper(" in compiled.wrapper_source
    )
    assert "kDataflowClusterSize = 1u" in compiled.wrapper_source
    assert "kDataflowTensorArgCount = 4u" in compiled.wrapper_source

    dumped = compiled.dump_plan()
    assert dumped["wrapper"]["kernel_name"] == "attention_wrapper"
    assert dumped["wrapper"]["cluster_size"] == 1
    assert dumped["wrapper"]["tensor_arg_count"] == 4
    assert dumped["wrapper"]["queue_indexing"] == "cta_rank"
    assert [handler["operator_name"] for handler in dumped["wrapper"]["handlers"]] == [
        "split_kv",
        "combine",
        "finalize",
    ]


def test_wrapper_can_index_queue_by_smid_when_requested():
    compiled = dataflow_debug.compile(
        make_program(),
        handler=dataflow_debug.EMPTY_HANDLER,
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [256]},
        block_size=128,
        wrapper_name="attention_wrapper_smid",
        queue_indexing="smid",
    )

    assert compiled.wrapper_spec.queue_indexing == "smid"
    assert compiled.options["queue_indexing"] == "smid"
    assert "uint32_t queue_rank = tl::dataflow_smid();" in compiled.wrapper_source
    assert "tl::dataflow_cta_rank_in_grid()" not in compiled.wrapper_source


def test_wrapper_helpers_reject_wrong_input_type():
    with pytest.raises(TypeError, match="PackedRuntimePlan"):
        df.build_wrapper_spec(object())

    with pytest.raises(TypeError, match="DataflowWrapperSpec"):
        df.generate_wrapper_source(object())

    with pytest.raises(TypeError, match="DataflowWrapperInstrumentation"):
        df.generate_wrapper_source(
            make_debug_wrapper_spec(make_packed_plan()),
            instrumentation=object(),
        )

    with pytest.raises(ValueError, match="Dataflow queue_indexing must be 'cta_rank' or 'smid'"):
        df.build_wrapper_spec(make_packed_plan(), queue_indexing="warp")


def test_generate_wrapper_source_places_registered_instrumentation_at_semantic_hooks():
    spec = make_debug_wrapper_spec(
        make_packed_plan(),
        kernel_name="dataflow_instrumented_wrapper",
    )
    instrumentation = df.DataflowWrapperInstrumentation(
        name="semantic_hook_contract",
        extra_kernel_params=(df.DataflowWrapperKernelParam(name="records", c_type="uint64_t *"),),
        namespace_source="TL_DEVICE void instrumentation_namespace_hook() {}",
        kernel_prologue="instrumentation_kernel_prologue();",
        before_barrier_init="instrumentation_before_barrier_init();",
        after_barrier_init="instrumentation_after_barrier_init();",
        after_cluster_sync="instrumentation_after_cluster_sync();",
        before_loop="instrumentation_before_loop();",
        before_recv="instrumentation_before_recv();",
        after_recv="instrumentation_after_recv();",
        after_handler="instrumentation_after_handler();",
        after_send="instrumentation_after_send();",
        kernel_epilogue="instrumentation_kernel_epilogue();",
    )

    source = df.generate_wrapper_source(spec, instrumentation=instrumentation)

    expected_order = (
        "instrumentation_kernel_prologue();",
        "instrumentation_before_barrier_init();",
        "instrumentation_after_barrier_init();",
        "instrumentation_after_cluster_sync();",
        "instrumentation_before_loop();",
        "instrumentation_before_recv();",
        "instrumentation_after_recv();",
        "instrumentation_after_handler();",
        "instrumentation_after_send();",
        "instrumentation_kernel_epilogue();",
    )
    positions = tuple(source.index(marker) for marker in expected_order)
    assert positions == tuple(sorted(positions))
    assert source.count("uint64_t * records") == 1
    assert instrumentation.to_dict()["extra_kernel_params"] == [{"name": "records", "c_type": "uint64_t *"}]


def test_wrapper_instrumentation_rejects_invalid_kernel_params():
    with pytest.raises(ValueError, match="duplicate kernel parameters"):
        df.DataflowWrapperInstrumentation(
            name="duplicates",
            extra_kernel_params=(
                df.DataflowWrapperKernelParam(name="records", c_type="uint64_t *"),
                df.DataflowWrapperKernelParam(name="records", c_type="uint32_t *"),
            ),
        )

    with pytest.raises(TypeError, match="DataflowWrapperKernelParam instances"):
        df.DataflowWrapperInstrumentation(
            name="wrong_param_type",
            extra_kernel_params=(object(),),
        )

    with pytest.raises(ValueError, match="collide with the wrapper ABI"):
        df.generate_wrapper_source(
            make_debug_wrapper_spec(make_packed_plan()),
            instrumentation=df.DataflowWrapperInstrumentation(
                name="reserved_param",
                extra_kernel_params=(df.DataflowWrapperKernelParam(name="flags", c_type="uint64_t *"),),
            ),
        )
