from __future__ import annotations

import math
import os

import pytest

import tilelang.language as T
import tilelang.dataflow as df
import tilelang.dataflow.scheduler as scheduler


@T.dataflow_intermediate
class AttnInter:
    lse: T.float32
    o: T.Tensor((2, 8), T.float16)


@T.dataflow_intermediate
class UpShard:
    value: T.Tensor((2, 4), T.float16)


@T.dataflow_intermediate
class UpFull:
    value: T.Tensor((2, 8), T.float16)


@T.dataflow_intermediate
class HiddenShard:
    value: T.Tensor((2, 4), T.float16)


@T.dataflow.iter(range=("kv_begin", "kv_end"))
def split_kv(seq: T.int32, head: T.int32, Q, K, V) -> AttnInter:
    raise AssertionError("Dataflow iter body should not execute during scheduling")


@T.dataflow.iter(range=("kv_begin", "kv_end"))
def split_kv_no_stage_axis(seq: T.int32, head: T.int32, Q, K, V) -> AttnInter:
    raise AssertionError("Dataflow iter body should not execute during scheduling")


@T.dataflow.reduce(associative=True)
def combine(left: AttnInter, right: AttnInter) -> AttnInter:
    raise AssertionError("Dataflow reduce body should not execute during scheduling")


@T.dataflow.finalize
def finalize(inter: AttnInter, seq: T.int32, head: T.int32, Output) -> None:
    raise AssertionError("Dataflow finalize body should not execute during scheduling")


@T.dataflow.map(range=("expert_begin", "expert_end"))
def moe_map1(expert: T.int32, token: T.int32, Input, W1) -> UpShard:
    raise AssertionError("Dataflow map body should not execute during scheduling")


@T.dataflow.map(range=("hidden_begin", "hidden_end"))
def moe_map2(parts: list[UpShard], expert: T.int32, token: T.int32, W2) -> HiddenShard:
    raise AssertionError("Dataflow map body should not execute during scheduling")


@T.dataflow.finalize
def finalize_hidden(hidden: HiddenShard, expert: T.int32, token: T.int32, Output) -> None:
    raise AssertionError("Dataflow finalize body should not execute during scheduling")


def make_program(*, range_axis="kv"):
    program = T.dataflow_program(
        task_domain=("batch", "head"),
        dynamic_ranges={"kv": "seq_lens"},
    )
    return (
        program.partial(
            split_kv(Q="Q", K="K", V="V"),
            task_args=("seq", "head"),
            range_axis=range_axis,
        )
        .reduce(combine())
        .finalize(finalize(Output="O"))
    )


_NX_SCHEDULE = df.schedule
_TEST_SCHEDULER_OPTION_PREFIX = "TEST_DATAFLOW_SCHEDULER_"


def scheduler_test_options():
    return {
        spec.option_name: os.environ[f"{_TEST_SCHEDULER_OPTION_PREFIX}{spec.option_name.upper()}"]
        for spec in df.DATAFLOW_SCHEDULER_OPTION_SPECS
        if f"{_TEST_SCHEDULER_OPTION_PREFIX}{spec.option_name.upper()}" in os.environ
    }


def build_schedule(*args, **kwargs):
    """Bind per-test knobs through the production typed config boundary."""

    options = scheduler_test_options()
    config = df.resolve_scheduler_config(kwargs.get("scheduler_config"))
    if options:
        config = config.with_options(**options)
    kwargs["scheduler_config"] = config
    return _NX_SCHEDULE(*args, **kwargs)


def test_scheduler_streaming_comm_cost_can_be_calibrated():
    topology = df.GPUTopology(sm_count=4, cluster_size=2)

    assert scheduler.estimated_streaming_comm_cost_us(topology, 0, 1) == pytest.approx(1.0)
    assert scheduler.estimated_streaming_comm_cost_us(topology, 0, 2) == pytest.approx(8.0)

    config = df.DataflowSchedulerConfig.from_options(
        cluster_comm_cost_us=0.75,
        hbm_comm_cost_us=2.0,
    )
    with df.use_scheduler_config(config):
        assert scheduler.estimated_streaming_comm_cost_us(topology, 0, 1) == pytest.approx(0.75)
        assert scheduler.estimated_streaming_comm_cost_us(topology, 0, 2) == pytest.approx(2.0)
        assert scheduler.estimated_streaming_comm_cost_us(
            topology,
            0,
            1,
            force_hbm_comms=True,
        ) == pytest.approx(2.0)


def test_scheduler_iter_cost_can_penalize_large_tail_chunks():
    aligned = df.TaskRange(axis="kv", begin=0, end=640)
    tail = df.TaskRange(axis="kv", begin=0, end=598)

    assert scheduler.estimated_streaming_iter_cost_us(aligned, 64) == pytest.approx(32.0)
    assert scheduler.estimated_streaming_iter_cost_us(tail, 64) == pytest.approx(32.0)

    config = df.DataflowSchedulerConfig.from_options(
        iter_tail_penalty_us=20,
        iter_tail_penalty_min_blocks=8,
    )
    with df.use_scheduler_config(config):
        assert scheduler.estimated_streaming_iter_cost_us(aligned, 64) == pytest.approx(32.0)
        assert scheduler.estimated_streaming_iter_cost_us(tail, 64) == pytest.approx(52.0)


def test_scheduler_producer_ready_queue_fills_remote_wait_with_independent_work():
    order = scheduler.producer_ready_queue_order(
        df.GPUTopology(sm_count=2, cluster_size=1),
        (
            scheduler.ProducerReadyQueueNode(
                node_id=0,
                sm_id=0,
                input_slots=(),
                output_slots=(0,),
                duration_us=10.0,
                sort_key=(0,),
            ),
            scheduler.ProducerReadyQueueNode(
                node_id=1,
                sm_id=1,
                input_slots=(0,),
                output_slots=(1,),
                duration_us=1.0,
                sort_key=(1,),
            ),
            scheduler.ProducerReadyQueueNode(
                node_id=2,
                sm_id=1,
                input_slots=(),
                output_slots=(2,),
                duration_us=5.0,
                sort_key=(2,),
            ),
        ),
    )

    assert order == {0: (0,), 1: (2, 1)}


def test_scheduler_ordered_tree_hides_remote_latency_without_reordering_leaves():
    topology = df.GPUTopology(sm_count=5, cluster_size=1)
    config = df.DataflowSchedulerConfig()

    with df.use_scheduler_config(config):
        plan = scheduler.ordered_reduction_tree_plan(
            topology,
            tuple((sm_id, 10.0) for sm_id in range(5)),
            consumer_side="right",
        )

    def leaf_order(node):
        if node.is_leaf:
            return (node.begin,)
        assert node.left is not None and node.right is not None
        return leaf_order(node.left) + leaf_order(node.right)

    assert leaf_order(plan) == tuple(range(5))
    assert plan.consumer_sm == 4
    assert plan.ready_us < 10.0 + 3 * (8.0 + 3.2)
    assert plan.critical_guard_wait_us == pytest.approx(8.0)


def test_scheduler_ordered_tree_can_choose_the_ready_consumer_child():
    topology = df.GPUTopology(sm_count=4, cluster_size=4)

    with df.use_scheduler_config(df.DataflowSchedulerConfig()):
        fixed = scheduler.ordered_reduction_tree_plan(
            topology,
            ((0, 10.0), (1, 10.0), (2, 10.0), (3, 100.0)),
            consumer_side="left",
        )
        adaptive = scheduler.ordered_reduction_tree_plan(
            topology,
            ((0, 10.0), (1, 10.0), (2, 10.0), (3, 100.0)),
            consumer_side="left",
            adaptive_consumer=True,
        )

    assert fixed.consumer_sm == 0
    assert adaptive.consumer_sm != 0
    assert adaptive.ready_us < fixed.ready_us


def test_scheduler_ordered_tree_fusion_respects_materialized_remote_inboxes():
    topology = df.GPUTopology(sm_count=16, cluster_size=8)

    with df.use_scheduler_config(df.DataflowSchedulerConfig()):
        plan = scheduler.ordered_reduction_tree_plan(
            topology,
            tuple((sm_id, 10.0) for sm_id in range(16)),
            consumer_side="right",
        )

    frontier = scheduler.ordered_reduction_fused_frontier(
        plan,
        max_reduce_arity=16,
        max_resident_remote_inputs=len(df.DataflowCommStorageKind),
    )

    assert len(frontier) == 3
    assert sum(child.consumer_sm != plan.consumer_sm for child in frontier) == len(df.DataflowCommStorageKind)


def test_scheduler_producer_ready_replay_exposes_remote_producer_before_long_local_work():
    chunks = (
        (
            scheduler.StreamingChunk(
                task_id=0,
                sm_id=0,
                task_range=df.TaskRange(axis="kv", begin=0, end=64),
                part_index=0,
                part_count=2,
            ),
            scheduler.StreamingChunk(
                task_id=0,
                sm_id=1,
                task_range=df.TaskRange(axis="kv", begin=64, end=128),
                part_index=1,
                part_count=2,
            ),
        ),
        (
            scheduler.StreamingChunk(
                task_id=1,
                sm_id=0,
                task_range=df.TaskRange(axis="kv", begin=0, end=1024),
                part_index=0,
                part_count=1,
            ),
        ),
    )
    config = df.DataflowSchedulerConfig.from_options(
        direct_leaf_acc=True,
        level0_queue_order="producer_ready",
    )

    with df.use_scheduler_config(config):
        replay = scheduler.replay_streaming_tree_score(
            df.GPUTopology(sm_count=2, cluster_size=2),
            chunks,
            block_size=64,
            streaming_tree_consumer="right",
            hierarchical_cross_cluster=False,
            ready_time_tree=False,
        )

    assert replay.max_finish_us == pytest.approx(59.1)
    assert replay.max_recv_wait_us == pytest.approx(1.0)


def test_scheduler_global_copack_reuses_ctas_across_tasks_without_self_overlap():
    topology = df.GPUTopology(sm_count=8, cluster_size=4)
    lengths = (128, 192, 1024)
    offsets = (0, 0, 0)

    with df.use_scheduler_config(
        df.DataflowSchedulerConfig.from_options(
            global_capacity_chunks=True,
            global_capacity_minimax_chunks=True,
        )
    ):
        partitioned = scheduler.global_capacity_streaming_chunks(
            topology,
            lengths,
            block_size=64,
            axis="kv",
            task_range_offsets=offsets,
        )
    with df.use_scheduler_config(
        df.DataflowSchedulerConfig.from_options(
            global_capacity_chunks=True,
            global_capacity_copacked_chunks=True,
        )
    ):
        copacked = scheduler.global_capacity_streaming_chunks(
            topology,
            lengths,
            block_size=64,
            axis="kv",
            task_range_offsets=offsets,
        )

    for task_id, task_chunks in enumerate(copacked):
        assert task_chunks[0].task_range.begin == 0
        assert task_chunks[-1].task_range.end == lengths[task_id]
        assert all(left.task_range.end == right.task_range.begin for left, right in zip(task_chunks, task_chunks[1:]))
        assert len({chunk.sm_id for chunk in task_chunks}) == len(task_chunks)

    tasks_by_sm = {}
    for task_chunks in copacked:
        for chunk in task_chunks:
            tasks_by_sm.setdefault(chunk.sm_id, set()).add(chunk.task_id)
    assert any(len(task_ids) > 1 for task_ids in tasks_by_sm.values())
    assert max(chunk.task_range.length for task_chunks in copacked for chunk in task_chunks) < max(
        chunk.task_range.length for task_chunks in partitioned for chunk in task_chunks
    )


def test_scheduler_partitions_ranges_and_builds_instruction_plan():
    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [384, 256]},
        block_size=128,
        task_extents=(2,),
    )

    assert plan.range_axis == "kv"
    assert plan.block_size == 128
    assert plan.task_extents == (2,)
    assert plan.task_range_lengths == (384, 256)

    iter_instructions = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    assert [(inst.task_id, inst.sm_id, inst.task_range.begin, inst.task_range.end, inst.output_slot) for inst in iter_instructions] == [
        (0, 0, 0, 128, 0),
        (0, 1, 128, 256, 1),
        (0, 2, 256, 384, 2),
        (1, 3, 0, 128, 4),
        (1, 0, 128, 256, 5),
    ]

    reduce_instructions = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.REDUCE]
    assert [(inst.task_id, inst.sm_id, inst.input_slots, inst.output_slot) for inst in reduce_instructions] == [
        (0, 0, (0, 1, 2), 3),
        (1, 3, (4, 5), 6),
    ]

    finalize_instructions = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.FINALIZE]
    assert [(inst.task_id, inst.sm_id, inst.input_slots) for inst in finalize_instructions] == [
        (0, 0, (3,)),
        (1, 3, (6,)),
    ]

    assert [slot.slot_id for slot in plan.slots] == list(range(7))
    assert [(slot.task_id, slot.role, slot.producer_instruction_id) for slot in plan.slots] == [
        (0, "partial", 0),
        (0, "partial", 1),
        (0, "partial", 2),
        (0, "reduced", 3),
        (1, "partial", 5),
        (1, "partial", 6),
        (1, "reduced", 7),
    ]

    assert [
        (
            comm.source_slot_id,
            comm.target_slot_id,
            comm.producer_sm,
            comm.consumer_sm,
            comm.kind,
            comm.resolved_dispatch_instruction_id,
            comm.dispatch_phase,
        )
        for comm in plan.comms
    ] == [
        (1, 1, 1, 0, df.DataflowCommKind.CLUSTER_SEND, 1, "post"),
        (1, 1, 1, 0, df.DataflowCommKind.CLUSTER_RECV, 3, "pre"),
        (2, 2, 2, 0, df.DataflowCommKind.HBM_SEND, 2, "post"),
        (2, 2, 2, 0, df.DataflowCommKind.HBM_RECV, 3, "pre"),
        (5, 5, 0, 3, df.DataflowCommKind.HBM_SEND, 6, "post"),
        (5, 5, 0, 3, df.DataflowCommKind.HBM_RECV, 7, "pre"),
    ]

    assert [inst.opcode for inst in plan.queue(0)] == [
        df.DataflowOpcode.ITER,
        df.DataflowOpcode.REDUCE,
        df.DataflowOpcode.FINALIZE,
        df.DataflowOpcode.ITER,
        df.DataflowOpcode.EXIT,
    ]
    assert all(plan.queue(sm_id)[-1].opcode is df.DataflowOpcode.EXIT for sm_id in range(4))


def test_scheduler_preserves_task_coordinates_from_task_extents():
    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [128, 128, 128, 128]},
        block_size=128,
        task_extents=(2, 2),
        include_exit=False,
    )

    iter_instructions = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    assert [inst.task_coords for inst in iter_instructions] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert all(plan.queue(sm_id)[-1].opcode is not df.DataflowOpcode.EXIT for sm_id in range(2))


def test_scheduler_cluster_local_policy_keeps_split_ranges_inside_task_cluster():
    default_plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [64]},
        block_size=16,
        task_extents=(1,),
        include_exit=False,
    )
    default_iters = [inst for inst in default_plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    assert [inst.sm_id for inst in default_iters] == [0, 1, 2, 3]
    assert any(comm.kind is df.DataflowCommKind.HBM_SEND for comm in default_plan.comms)

    mla_plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [64, 64]},
        block_size=16,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
    )

    iter_instructions = [inst for inst in mla_plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    assert [inst.sm_id for inst in iter_instructions[:4]] == [0, 1, 0, 1]
    assert [inst.sm_id for inst in iter_instructions[4:8]] == [2, 3, 2, 3]
    reduce_instructions = [inst for inst in mla_plan.instructions if inst.opcode is df.DataflowOpcode.REDUCE]
    assert [inst.sm_id for inst in reduce_instructions] == [0, 2]
    assert any(comm.kind is df.DataflowCommKind.CLUSTER_SEND for comm in mla_plan.comms)
    assert all(comm.kind is not df.DataflowCommKind.HBM_SEND for comm in mla_plan.comms)


def test_scheduler_streaming_reduce_chains_ranges_inside_load_balanced_clusters():
    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [512, 128, 384]},
        block_size=64,
        task_extents=(3,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming",
    )

    assert plan.reduce_strategy == "streaming"
    iter_instructions = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    reduce_updates = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.REDUCE_UPDATE]
    finalizes = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.FINALIZE]

    assert len(reduce_updates) == len(iter_instructions)
    assert all(len(inst.input_slots) in (1, 2) for inst in reduce_updates)
    assert [len(inst.input_slots) for inst in reduce_updates if inst.task_id == 0] == [1, 2]
    assert [inst.sm_id for inst in iter_instructions if inst.task_id == 0] == [0, 1]
    assert [inst.sm_id for inst in iter_instructions if inst.task_id == 1] == [3]
    assert [inst.sm_id for inst in iter_instructions if inst.task_id == 2] == [2, 3]
    assert [(inst.task_id, inst.task_range.begin, inst.task_range.end) for inst in iter_instructions if inst.task_id == 0] == [
        (0, 0, 256),
        (0, 256, 512),
    ]
    assert [(inst.task_id, inst.sm_id) for inst in finalizes] == [(0, 1), (1, 3), (2, 3)]
    assert any(comm.kind is df.DataflowCommKind.CLUSTER_SEND for comm in plan.comms)
    assert all(comm.kind is not df.DataflowCommKind.HBM_SEND for comm in plan.comms)


def test_scheduler_streaming_tree_reduce_uses_pairwise_cluster_reduction():
    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [1024]},
        block_size=64,
        task_extents=(1,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    assert plan.reduce_strategy == "streaming_tree"
    iter_instructions = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    reduce_updates = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.REDUCE_UPDATE]
    finalizes = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.FINALIZE]

    assert [(inst.sm_id, inst.task_range.begin, inst.task_range.end) for inst in iter_instructions] == [
        (0, 0, 256),
        (1, 256, 512),
        (2, 512, 768),
        (3, 768, 1024),
    ]
    assert [len(inst.input_slots) for inst in reduce_updates] == [1, 1, 1, 1, 2, 2, 2]
    assert [inst.sm_id for inst in reduce_updates] == [0, 1, 2, 3, 0, 2, 0]
    assert [(inst.task_id, inst.sm_id) for inst in finalizes] == [(0, 0)]
    assert len(plan.comms) == 6
    assert any(comm.kind is df.DataflowCommKind.CLUSTER_SEND for comm in plan.comms)
    assert all(comm.kind is not df.DataflowCommKind.HBM_SEND for comm in plan.comms)


def test_scheduler_streaming_tree_can_reduce_on_right_ready_side(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_STREAMING_TREE_CONSUMER", "right")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [1024]},
        block_size=64,
        task_extents=(1,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    reduce_updates = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.REDUCE_UPDATE]
    finalizes = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.FINALIZE]

    assert [len(inst.input_slots) for inst in reduce_updates] == [1, 1, 1, 1, 2, 2, 2]
    assert [inst.sm_id for inst in reduce_updates] == [0, 1, 2, 3, 1, 3, 3]
    assert [(inst.task_id, inst.sm_id) for inst in finalizes] == [(0, 3)]
    assert any(comm.kind is df.DataflowCommKind.CLUSTER_SEND for comm in plan.comms)
    assert all(comm.kind is not df.DataflowCommKind.HBM_SEND for comm in plan.comms)


def test_scheduler_streaming_tree_consumer_option_overrides_typed_config(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_STREAMING_TREE_CONSUMER", "left")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [1024]},
        block_size=64,
        task_extents=(1,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        streaming_tree_consumer="right",
    )

    reduce_updates = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.REDUCE_UPDATE]
    finalizes = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.FINALIZE]

    assert [inst.sm_id for inst in reduce_updates] == [0, 1, 2, 3, 1, 3, 3]
    assert [(inst.task_id, inst.sm_id) for inst in finalizes] == [(0, 3)]


def test_scheduler_streaming_tree_heap_chunks_keep_task_root_off_previous_tail(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_HEAP_STREAMING_CHUNKS", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [160, 160]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    iter_instructions = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    assert [(inst.task_id, inst.sm_id, inst.task_range.begin, inst.task_range.end) for inst in iter_instructions] == [
        (0, 0, 0, 128),
        (0, 1, 128, 160),
        (1, 2, 0, 128),
        (1, 3, 128, 160),
    ]

    finalizes = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.FINALIZE]
    assert [(inst.task_id, inst.sm_id) for inst in finalizes] == [(0, 0), (1, 2)]


def test_scheduler_streaming_tree_can_skip_tiny_root_fragment(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_SKIP_TINY_ROOT_FRAGMENT", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [448, 384]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    iter_instructions = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    assert [(inst.task_id, inst.sm_id, inst.task_range.begin, inst.task_range.end) for inst in iter_instructions] == [
        (0, 0, 0, 256),
        (0, 1, 256, 448),
        (1, 2, 0, 256),
        (1, 3, 256, 384),
    ]

    finalizes = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.FINALIZE]
    assert [(inst.task_id, inst.sm_id) for inst in finalizes] == [(0, 0), (1, 2)]


def test_scheduler_streaming_tree_skip_tiny_root_option_overrides_typed_config(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_SKIP_TINY_ROOT_FRAGMENT", "0")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [448, 384]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        skip_tiny_root_fragment=True,
    )

    iter_instructions = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    assert [(inst.task_id, inst.sm_id, inst.task_range.begin, inst.task_range.end) for inst in iter_instructions] == [
        (0, 0, 0, 256),
        (0, 1, 256, 448),
        (1, 2, 0, 256),
        (1, 3, 256, 384),
    ]


def test_scheduler_streaming_tree_accepts_explicit_cluster_task_assignment():
    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=6, cluster_size=2),
        range_lengths={"kv": [64, 64, 256]},
        block_size=64,
        task_extents=(3,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        cluster_task_assignment=((2,), (0,), (1,)),
    )

    iter_instructions = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    assert [(inst.task_id, plan.topology.cluster_id(inst.sm_id)) for inst in iter_instructions] == [
        (0, 1),
        (1, 2),
        (2, 0),
        (2, 0),
    ]
    assert all(comm.kind is not df.DataflowCommKind.HBM_SEND for comm in plan.comms)


def test_scheduler_streaming_tree_parses_typed_cluster_task_assignment(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CLUSTER_TASK_ASSIGNMENT", "1|2|0")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=6, cluster_size=2),
        range_lengths={"kv": [64, 64, 256]},
        block_size=64,
        task_extents=(3,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    iter_instructions = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    assert [(inst.task_id, plan.topology.cluster_id(inst.sm_id)) for inst in iter_instructions] == [
        (0, 2),
        (1, 0),
        (2, 1),
        (2, 1),
    ]


def test_scheduler_streaming_tree_skips_tiny_root_even_when_next_task_fits_one_sm(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_SKIP_TINY_ROOT_FRAGMENT", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [128, 448]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    iter_instructions = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    assert [(inst.task_id, inst.sm_id, inst.task_range.begin, inst.task_range.end) for inst in iter_instructions] == [
        (0, 3, 0, 128),
        (1, 0, 0, 192),
        (1, 1, 192, 384),
        (1, 2, 384, 448),
    ]

    finalizes = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.FINALIZE]
    assert [(inst.task_id, inst.sm_id) for inst in finalizes] == [(0, 3), (1, 0)]


def test_scheduler_streaming_tree_can_merge_tiny_final_chunk(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_MERGE_TINY_FINAL_CHUNK", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_TINY_FINAL_CHUNK_BLOCKS", "4")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [1088]},
        block_size=64,
        task_extents=(1,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    iter_instructions = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    assert [(inst.sm_id, inst.task_range.begin, inst.task_range.end) for inst in iter_instructions] == [
        (0, 0, 320),
        (1, 320, 640),
        (2, 640, 1088),
    ]


def test_scheduler_streaming_tree_can_split_long_sequence_across_clusters(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CROSS_CLUSTER_LONG_SPLIT", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=8, cluster_size=4),
        range_lengths={"kv": [1024, 64]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    task0_iter_clusters = {
        plan.topology.cluster_id(inst.sm_id) for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER and inst.task_id == 0
    }
    task1_iter_clusters = {
        plan.topology.cluster_id(inst.sm_id) for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER and inst.task_id == 1
    }

    assert task0_iter_clusters == {0, 1}
    assert len(task1_iter_clusters) == 1
    assert any(comm.kind is df.DataflowCommKind.CLUSTER_SEND for comm in plan.comms)
    assert any(comm.kind is df.DataflowCommKind.HBM_SEND for comm in plan.comms)
    assert any(comm.kind is df.DataflowCommKind.HBM_RECV for comm in plan.comms)


def test_scheduler_cross_cluster_orders_long_segments_before_short_fillers(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CROSS_CLUSTER_LONG_SPLIT", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=8, cluster_size=4),
        range_lengths={"kv": [64, 128, 1024]},
        block_size=64,
        task_extents=(3,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    task2_iters = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER and inst.task_id == 2]
    task2_first = task2_iters[0]

    assert (task2_first.sm_id, task2_first.task_range.begin, task2_first.task_range.end) == (
        0,
        0,
        192,
    )
    assert [(inst.task_id, inst.sm_id) for inst in plan.instructions if inst.opcode is df.DataflowOpcode.FINALIZE] == [
        (0, 6),
        (1, 2),
        (2, 0),
    ]


def test_scheduler_manual_cluster_segments_override_streaming_chunks(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_MANUAL_CLUSTER_SEGMENTS", "0:0:0-6,1:6-16;1:1:0-2")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_HIER_CROSS_CLUSTER_SPLIT", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=8, cluster_size=4),
        range_lengths={"kv": [1024, 128]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    task0_iter_clusters = {
        plan.topology.cluster_id(inst.sm_id) for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER and inst.task_id == 0
    }
    task1_iter_clusters = {
        plan.topology.cluster_id(inst.sm_id) for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER and inst.task_id == 1
    }
    task0_ranges = [
        (plan.topology.cluster_id(inst.sm_id), inst.task_range.begin, inst.task_range.end)
        for inst in plan.instructions
        if inst.opcode is df.DataflowOpcode.ITER and inst.task_id == 0
    ]

    assert task0_iter_clusters == {0, 1}
    assert task1_iter_clusters == {1}
    assert min(begin for cluster_id, begin, _ in task0_ranges if cluster_id == 1) == 384
    assert any(
        comm.kind is df.DataflowCommKind.HBM_SEND for comm in plan.comms if plan.instructions[comm.source_instruction_id].task_id == 0
    )


def test_scheduler_manual_segments_can_avoid_tiny_tail_copack(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_MANUAL_CLUSTER_SEGMENTS", "0:0:0-9;1:0:0-5")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_AVOID_TINY_SEGMENT_COPACK", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_TINY_SEGMENT_COPACK_BLOCKS", "2")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [576, 320]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        partial_only=True,
    )

    iter_instructions = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    task0_tail = next(inst for inst in iter_instructions if inst.task_id == 0 and inst.task_range.begin == 512)
    task1_first = next(inst for inst in iter_instructions if inst.task_id == 1 and inst.task_range.begin == 0)

    assert task0_tail.task_range.end == 576
    assert task0_tail.sm_id != task1_first.sm_id


def test_scheduler_manual_segments_can_evenly_chunk_each_segment(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_MANUAL_CLUSTER_SEGMENTS", "0:0:0-9;1:0:0-5")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_EVEN_SEGMENT_CHUNKS", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [576, 320]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        partial_only=True,
    )

    task0_ranges = [
        (inst.sm_id, inst.task_range.begin, inst.task_range.end)
        for inst in plan.instructions
        if inst.opcode is df.DataflowOpcode.ITER and inst.task_id == 0
    ]

    assert task0_ranges == [
        (0, 0, 192),
        (1, 192, 384),
        (2, 384, 576),
    ]


def test_scheduler_manual_segments_merge_adjacent_same_cluster_ranges(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_MANUAL_CLUSTER_SEGMENTS", "0:0:0-8,0:8-12;1:0:0-7")

    split_plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [768, 448]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        partial_only=True,
    )

    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_MANUAL_CLUSTER_SEGMENTS", "0:0:0-12;1:0:0-7")
    merged_plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [768, 448]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        partial_only=True,
    )

    split_ranges = [
        (inst.sm_id, inst.task_range.begin, inst.task_range.end)
        for inst in split_plan.instructions
        if inst.opcode is df.DataflowOpcode.ITER
    ]
    merged_ranges = [
        (inst.sm_id, inst.task_range.begin, inst.task_range.end)
        for inst in merged_plan.instructions
        if inst.opcode is df.DataflowOpcode.ITER
    ]

    assert split_ranges == merged_ranges


def test_scheduler_hierarchical_cross_cluster_split_hbm_only_between_local_roots(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CROSS_CLUSTER_LONG_SPLIT", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_HIER_CROSS_CLUSTER_SPLIT", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=8, cluster_size=4),
        range_lengths={"kv": [1024, 64]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    instructions = {inst.instruction_id: inst for inst in plan.instructions}
    task0_hbm_sends = [
        comm for comm in plan.comms if comm.kind is df.DataflowCommKind.HBM_SEND and instructions[comm.target_instruction_id].task_id == 0
    ]
    task0_hbm_recvs = [
        comm for comm in plan.comms if comm.kind is df.DataflowCommKind.HBM_RECV and instructions[comm.target_instruction_id].task_id == 0
    ]

    assert len(task0_hbm_sends) == 1
    assert len(task0_hbm_recvs) == 1
    hbm_source = instructions[task0_hbm_sends[0].source_instruction_id]
    hbm_target = instructions[task0_hbm_sends[0].target_instruction_id]
    assert hbm_source.attrs["hierarchical_role"] == "local_root"
    assert hbm_target.attrs["hierarchical_scope"] == "global"
    assert hbm_source.attrs["cluster_id"] != hbm_target.attrs["cluster_id"]

    early_cross_cluster_edges = [comm for comm in task0_hbm_sends if instructions[comm.source_instruction_id].attrs.get("tree_level") == 0]
    assert early_cross_cluster_edges == []


def test_scheduler_hierarchical_tree_preserves_non_commutative_range_order(monkeypatch):
    monkeypatch.setenv(
        "TEST_DATAFLOW_SCHEDULER_MANUAL_CLUSTER_SEGMENTS",
        "0:0:0-2,1:2-4,0:4-6",
    )
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_HIER_CROSS_CLUSTER_SPLIT", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_DIRECT_LEAF_ACC", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [384]},
        block_size=64,
        task_extents=(1,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    producer_by_slot = {inst.output_slot: inst for inst in plan.instructions if inst.output_slot is not None}

    def leaf_ranges(slot_id):
        instruction = producer_by_slot[slot_id]
        if instruction.opcode is df.DataflowOpcode.ITER:
            return [(instruction.task_range.begin, instruction.task_range.end)]
        return [task_range for input_slot in instruction.input_slots for task_range in leaf_ranges(input_slot)]

    finalize_instruction = next(inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.FINALIZE)
    local_roots = sorted(
        (
            inst.attrs["hierarchical_run_index"],
            inst.attrs["cluster_id"],
        )
        for inst in plan.instructions
        if inst.attrs.get("hierarchical_role") == "local_root"
    )

    assert local_roots == [(0, 0), (1, 1), (2, 0)]
    assert leaf_ranges(finalize_instruction.input_slots[0]) == [
        (0, 128),
        (128, 192),
        (192, 256),
        (256, 384),
    ]


def test_scheduler_can_skip_cross_cluster_reduce_for_diagnostic(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CROSS_CLUSTER_LONG_SPLIT", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_HIER_CROSS_CLUSTER_SPLIT", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_SKIP_CROSS_CLUSTER_REDUCE", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=8, cluster_size=4),
        range_lengths={"kv": [1024, 64]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    instructions = {inst.instruction_id: inst for inst in plan.instructions}
    task0_hbm_comms = [
        comm
        for comm in plan.comms
        if instructions[comm.target_instruction_id].task_id == 0
        and comm.kind in (df.DataflowCommKind.HBM_SEND, df.DataflowCommKind.HBM_RECV)
    ]
    task0_finalizes = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.FINALIZE and inst.task_id == 0]
    task0_global_reduces = [
        inst
        for inst in plan.instructions
        if inst.opcode is df.DataflowOpcode.REDUCE_UPDATE and inst.task_id == 0 and inst.attrs.get("hierarchical_scope") == "global"
    ]

    assert task0_hbm_comms == []
    assert task0_global_reduces == []
    assert len(task0_finalizes) == 2
    assert all(inst.attrs["diagnostic_skip_cross_cluster_reduce"] for inst in task0_finalizes)
    assert {inst.attrs["local_root_cluster_id"] for inst in task0_finalizes} == {0, 1}


def test_scheduler_ready_time_tree_places_parent_on_less_loaded_child(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_STREAMING_TREE_CONSUMER", "left")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"kv": [256, 128, 256]},
        block_size=64,
        task_extents=(3,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    task2_parent_reduces = [
        inst
        for inst in plan.instructions
        if inst.opcode is df.DataflowOpcode.REDUCE_UPDATE and inst.task_id == 2 and len(inst.input_slots) == 2
    ]

    assert len(task2_parent_reduces) == 1
    assert task2_parent_reduces[0].sm_id == 1
    assert task2_parent_reduces[0].attrs["placement_policy"] == "ready_time"


def test_scheduler_balanced_dag_can_override_cluster_task_assignment_for_long_task(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_BALANCED_DAG_SCHED", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_BALANCED_DAG_SCORE", "load")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_BALANCED_DAG_SPLIT_GAIN_BLOCKS", "0")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_HIER_CROSS_CLUSTER_SPLIT", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=8, cluster_size=4),
        range_lengths={"kv": [1024, 64]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        cluster_task_assignment=((0, 1), ()),
    )

    task0_iter_clusters = {
        plan.topology.cluster_id(inst.sm_id) for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER and inst.task_id == 0
    }
    instructions = {inst.instruction_id: inst for inst in plan.instructions}
    task0_hbm_sends = [
        comm for comm in plan.comms if comm.kind is df.DataflowCommKind.HBM_SEND and instructions[comm.target_instruction_id].task_id == 0
    ]

    assert task0_iter_clusters == {0, 1}
    assert len(task0_hbm_sends) == 1
    assert instructions[task0_hbm_sends[0].source_instruction_id].attrs["hierarchical_role"] == "local_root"


def test_scheduler_balanced_dag_replay_score_rejects_hbm_tail_regression(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_BALANCED_DAG_SCHED", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_BALANCED_DAG_SCORE", "replay")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_BALANCED_DAG_SPLIT_GAIN_BLOCKS", "0")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_HIER_CROSS_CLUSTER_SPLIT", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=8, cluster_size=4),
        range_lengths={"kv": [1024, 64]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    instructions = {inst.instruction_id: inst for inst in plan.instructions}
    task0_iter_clusters = {
        plan.topology.cluster_id(inst.sm_id) for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER and inst.task_id == 0
    }
    task0_hbm_sends = [
        comm for comm in plan.comms if comm.kind is df.DataflowCommKind.HBM_SEND and instructions[comm.target_instruction_id].task_id == 0
    ]

    assert len(task0_iter_clusters) == 1
    assert task0_hbm_sends == []


def test_scheduler_replay_score_uses_level_ordered_queue_model():
    replay = scheduler.replay_streaming_tree_score(
        df.GPUTopology(sm_count=2, cluster_size=2),
        (
            (
                scheduler.StreamingChunk(
                    task_id=0,
                    sm_id=0,
                    task_range=df.TaskRange(axis="kv", begin=0, end=64),
                    part_index=0,
                    part_count=2,
                ),
                scheduler.StreamingChunk(
                    task_id=0,
                    sm_id=1,
                    task_range=df.TaskRange(axis="kv", begin=64, end=128),
                    part_index=1,
                    part_count=2,
                ),
            ),
            (
                scheduler.StreamingChunk(
                    task_id=1,
                    sm_id=0,
                    task_range=df.TaskRange(axis="kv", begin=0, end=1024),
                    part_index=0,
                    part_count=1,
                ),
            ),
        ),
        block_size=64,
        streaming_tree_consumer="left",
        hierarchical_cross_cluster=False,
        ready_time_tree=False,
    )

    assert replay.max_finish_us == pytest.approx(69.9)
    assert replay.max_recv_wait_us == pytest.approx(0.0)


def test_scheduler_replay_score_honors_long_first_level0_order():
    config = df.DataflowSchedulerConfig.from_options(
        direct_leaf_acc=True,
        level0_queue_order="long_first",
    )
    with df.use_scheduler_config(config):
        replay = scheduler.replay_streaming_tree_score(
            df.GPUTopology(sm_count=2, cluster_size=2),
            (
                (
                    scheduler.StreamingChunk(
                        task_id=0,
                        sm_id=0,
                        task_range=df.TaskRange(axis="kv", begin=0, end=64),
                        part_index=0,
                        part_count=2,
                    ),
                    scheduler.StreamingChunk(
                        task_id=0,
                        sm_id=1,
                        task_range=df.TaskRange(axis="kv", begin=64, end=128),
                        part_index=1,
                        part_count=2,
                    ),
                ),
                (
                    scheduler.StreamingChunk(
                        task_id=1,
                        sm_id=0,
                        task_range=df.TaskRange(axis="kv", begin=0, end=1024),
                        part_index=0,
                        part_count=1,
                    ),
                ),
            ),
            block_size=64,
            streaming_tree_consumer="right",
            hierarchical_cross_cluster=False,
            ready_time_tree=False,
        )

    assert replay.max_finish_us == pytest.approx(63.3)
    assert replay.max_recv_wait_us == pytest.approx(49.2)


def test_scheduler_replay_score_reports_cross_cluster_local_root_skew():
    config = df.DataflowSchedulerConfig.from_options(direct_leaf_acc=True)
    with df.use_scheduler_config(config):
        replay = scheduler.replay_streaming_tree_score(
            df.GPUTopology(sm_count=4, cluster_size=2),
            (
                (
                    scheduler.StreamingChunk(
                        task_id=0,
                        sm_id=0,
                        task_range=df.TaskRange(axis="kv", begin=0, end=64),
                        part_index=0,
                        part_count=2,
                    ),
                    scheduler.StreamingChunk(
                        task_id=0,
                        sm_id=2,
                        task_range=df.TaskRange(axis="kv", begin=64, end=1344),
                        part_index=1,
                        part_count=2,
                    ),
                ),
            ),
            block_size=64,
            streaming_tree_consumer="right",
            hierarchical_cross_cluster=True,
            ready_time_tree=False,
        )

    assert replay.split_task_count == 1
    assert replay.max_local_root_skew_us == pytest.approx(51.3)
    assert replay.total_local_root_skew_us == pytest.approx(51.3)


def test_scheduler_balanced_dag_objective_can_penalize_local_root_skew():
    replay = scheduler.StreamingDagReplayScore(
        max_finish_us=100.0,
        p95_finish_us=90.0,
        finish_spread_us=50.0,
        total_recv_wait_us=20.0,
        max_recv_wait_us=10.0,
        hbm_edges=2,
        instruction_count=40,
        slot_count=32,
        max_local_root_skew_us=25.0,
        total_local_root_skew_us=40.0,
        split_task_count=2,
    )

    base = scheduler.balanced_dag_replay_objective(replay, hbm_penalty_blocks=0)
    config = df.DataflowSchedulerConfig.from_options(balanced_dag_root_skew_weight=2.0)
    with df.use_scheduler_config(config):
        weighted = scheduler.balanced_dag_replay_objective(replay, hbm_penalty_blocks=0)

    assert weighted == pytest.approx(base + 50.0)


def test_scheduler_level_bucket_tree_accounts_for_earlier_finalize_before_later_reduce(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL_BUCKET_TREE", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [64, 1024]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    producer_sm_by_slot = {inst.output_slot: inst.sm_id for inst in plan.instructions if inst.output_slot is not None}
    task1_level1_reduces = [
        inst
        for inst in plan.instructions
        if inst.opcode is df.DataflowOpcode.REDUCE_UPDATE
        and inst.task_id == 1
        and inst.attrs.get("tree_level") == 1
        and len(inst.input_slots) == 2
    ]
    reduce_from_sm2_sm3 = [inst for inst in task1_level1_reduces if {producer_sm_by_slot[slot] for slot in inst.input_slots} == {2, 3}]

    assert len(reduce_from_sm2_sm3) == 1
    assert reduce_from_sm2_sm3[0].sm_id == 2


def test_scheduler_can_order_level0_leaf_groups_by_longest_chunk(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL_BUCKET_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL0_QUEUE_ORDER", "long_first")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [64, 512]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    sm2_iters = [inst for inst in plan.queue(2) if inst.opcode is df.DataflowOpcode.ITER]

    assert [(inst.task_id, inst.task_range.length) for inst in sm2_iters] == [
        (1, 128),
        (0, 64),
    ]


def test_scheduler_level0_priority_tasks_override_long_first_order(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL_BUCKET_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL0_QUEUE_ORDER", "long_first")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL0_PRIORITY_TASKS", "0")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [64, 512]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    sm2_iters = [inst for inst in plan.queue(2) if inst.opcode is df.DataflowOpcode.ITER]

    assert [(inst.task_id, inst.task_range.length) for inst in sm2_iters] == [
        (0, 64),
        (1, 128),
    ]


def test_scheduler_can_replay_optimize_level0_order_to_avoid_critical_short_leaf_delay(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL_BUCKET_TREE", "1")

    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL0_QUEUE_ORDER", "long_first")
    long_first_plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [128, 512]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )
    long_first_sm2_iters = [inst for inst in long_first_plan.queue(2) if inst.opcode is df.DataflowOpcode.ITER]
    assert [(inst.task_id, inst.task_range.length) for inst in long_first_sm2_iters] == [
        (1, 128),
        (0, 64),
    ]

    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL0_QUEUE_ORDER", "replay")
    replay_plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [128, 512]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )
    replay_sm2_iters = [inst for inst in replay_plan.queue(2) if inst.opcode is df.DataflowOpcode.ITER]

    assert [(inst.task_id, inst.task_range.length) for inst in replay_sm2_iters] == [
        (0, 64),
        (1, 128),
    ]


def test_scheduler_ready_time_tree_accounts_for_level0_long_first_leaf_order(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL_BUCKET_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL0_QUEUE_ORDER", "long_first")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [128, 512]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    task0_parent = [
        inst
        for inst in plan.instructions
        if inst.opcode is df.DataflowOpcode.REDUCE_UPDATE and inst.task_id == 0 and len(inst.input_slots) == 2
    ]

    assert len(task0_parent) == 1
    assert task0_parent[0].sm_id == 2


def test_scheduler_orders_level0_by_length_only_when_tree_criticality_also_increases(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL_BUCKET_TREE", "1")

    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL0_QUEUE_ORDER", "long_first")
    long_first_plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=8, cluster_size=8),
        range_lengths={"kv": [64, 1024, 256]},
        block_size=64,
        task_extents=(3,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )
    long_first_sm5_iters = [inst for inst in long_first_plan.queue(5) if inst.opcode is df.DataflowOpcode.ITER]
    assert [(inst.task_id, inst.task_range.length) for inst in long_first_sm5_iters] == [
        (2, 128),
        (1, 64),
    ]

    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL0_QUEUE_ORDER", "critical")
    critical_plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=8, cluster_size=8),
        range_lengths={"kv": [64, 1024, 256]},
        block_size=64,
        task_extents=(3,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )
    critical_sm5_iters = [inst for inst in critical_plan.queue(5) if inst.opcode is df.DataflowOpcode.ITER]

    assert [(inst.task_id, inst.task_range.length) for inst in critical_sm5_iters] == [
        (1, 64),
        (2, 128),
    ]

    dominance_plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=8, cluster_size=8),
        range_lengths={"kv": [64, 2560]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )
    dominance_sm6_iters = [inst for inst in dominance_plan.queue(6) if inst.opcode is df.DataflowOpcode.ITER]

    assert [(inst.task_id, inst.task_range.length) for inst in dominance_sm6_iters] == [
        (1, 256),
        (0, 64),
    ]


def test_scheduler_balanced_dag_replay_keeps_no_split_incumbent_for_trace15_like_tail(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_BALANCED_DAG_SCHED", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_BALANCED_DAG_SCORE", "replay")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_BALANCED_DAG_SPLIT_GAIN_BLOCKS", "0")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_HIER_CROSS_CLUSTER_SPLIT", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=96, cluster_size=16),
        range_lengths={
            "kv": [
                566,
                1251,
                1951,
                2662,
                3403,
                4188,
                5041,
                5961,
                6975,
                8112,
                9405,
                10902,
                12696,
                14919,
                17885,
                22527,
            ]
        },
        block_size=64,
        task_extents=(16,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    instructions = {inst.instruction_id: inst for inst in plan.instructions}
    split_tasks = {
        task_id
        for task_id in range(16)
        if len(
            {
                plan.topology.cluster_id(inst.sm_id)
                for inst in plan.instructions
                if inst.opcode is df.DataflowOpcode.ITER and inst.task_id == task_id
            }
        )
        > 1
    }
    task_hbm_sends = [
        comm
        for comm in plan.comms
        if comm.kind is df.DataflowCommKind.HBM_SEND and instructions[comm.target_instruction_id].task_id in split_tasks
    ]

    assert split_tasks == set()
    assert task_hbm_sends == []


def test_scheduler_balanced_dag_local_search_splits_long_suffix_from_incumbent(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_BALANCED_DAG_SCHED", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_BALANCED_DAG_LOCAL_SEARCH", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_BALANCED_DAG_HBM_PENALTY_BLOCKS", "0")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_HIER_CROSS_CLUSTER_SPLIT", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=6, cluster_size=2),
        range_lengths={"kv": [4096, 64, 64]},
        block_size=64,
        task_extents=(3,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        cluster_task_assignment=((0,), (1,), (2,)),
    )

    task0_iter_clusters = {
        plan.topology.cluster_id(inst.sm_id) for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER and inst.task_id == 0
    }
    instructions = {inst.instruction_id: inst for inst in plan.instructions}
    task0_hbm_sends = [
        comm for comm in plan.comms if comm.kind is df.DataflowCommKind.HBM_SEND and instructions[comm.target_instruction_id].task_id == 0
    ]

    assert task0_iter_clusters == {0, 2}
    assert len(task0_hbm_sends) == 1
    assert len(plan.slots) == 15


def test_scheduler_capacity_balance_splits_only_tasks_larger_than_cluster_capacity():
    topology = df.GPUTopology(sm_count=6, cluster_size=2)
    lengths = (448, 320, 256)
    incumbent = scheduler.load_balanced_streaming_chunks(
        topology,
        lengths,
        block_size=64,
        axis="kv",
        cluster_task_assignment=((0,), (1,), (2,)),
    )

    candidate = scheduler.capacity_balanced_streaming_chunks(
        topology,
        incumbent,
        block_size=64,
        capacity_blocks=3,
        min_segment_blocks=1,
        min_chunk_blocks=1,
        max_task_segments=2,
        assignment_beam=16,
    )

    assert candidate is not None
    split_tasks = {task_id for task_id, chunks in enumerate(candidate) if len({topology.cluster_id(chunk.sm_id) for chunk in chunks}) > 1}
    assert split_tasks == {0}
    for task_id, chunks in enumerate(candidate):
        assert [chunk.part_index for chunk in chunks] == list(range(len(chunks)))
        assert all(chunk.part_count == len(chunks) for chunk in chunks)
        assert len({chunk.sm_id for chunk in chunks}) == len(chunks)
        assert chunks[0].task_range.begin == 0
        assert chunks[-1].task_range.end == lengths[task_id]
        assert all(left.task_range.end == right.task_range.begin for left, right in zip(chunks, chunks[1:]))
    assert (
        max(
            sum(math.ceil(chunk.task_range.length / 64) for chunks in candidate for chunk in chunks if chunk.sm_id == sm_id)
            for sm_id in range(topology.sm_count)
        )
        <= 3
    )


def test_scheduler_critical_path_split_moves_large_suffix_without_tiny_chunks(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_SPLIT", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_CAPACITY_SEARCH_MAX_EVALS", "0")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_SUFFIX_BLOCKS", "16")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_MIN_CHUNK_BLOCKS", "4")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_EXTRA_CHUNK_PENALTY_US", "0")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_HIER_CROSS_CLUSTER_SPLIT", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=6, cluster_size=2),
        range_lengths={"kv": [4096, 64, 64]},
        block_size=64,
        task_extents=(3,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        cluster_task_assignment=((0,), (1,), (2,)),
    )

    task0_iter_clusters = {
        plan.topology.cluster_id(inst.sm_id) for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER and inst.task_id == 0
    }
    task0_iter_block_counts = [
        (inst.task_range.end - inst.task_range.begin) // 64
        for inst in plan.instructions
        if inst.opcode is df.DataflowOpcode.ITER and inst.task_id == 0
    ]
    instructions = {inst.instruction_id: inst for inst in plan.instructions}
    task0_hbm_sends = [
        comm for comm in plan.comms if comm.kind is df.DataflowCommKind.HBM_SEND and instructions[comm.target_instruction_id].task_id == 0
    ]

    assert 0 in task0_iter_clusters
    assert len(task0_iter_clusters) == 2
    assert min(task0_iter_block_counts) >= 4
    assert len(task0_hbm_sends) == 1


def test_scheduler_critical_path_split_rejects_tiny_suffix_candidates(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_SPLIT", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_CAPACITY_SEARCH_MAX_EVALS", "0")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_SUFFIX_BLOCKS", "2")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_MIN_CHUNK_BLOCKS", "4")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_EXTRA_CHUNK_PENALTY_US", "0")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_HIER_CROSS_CLUSTER_SPLIT", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=6, cluster_size=2),
        range_lengths={"kv": [4096, 64, 64]},
        block_size=64,
        task_extents=(3,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        cluster_task_assignment=((0,), (1,), (2,)),
    )

    task0_iter_clusters = {
        plan.topology.cluster_id(inst.sm_id) for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER and inst.task_id == 0
    }
    instructions = {inst.instruction_id: inst for inst in plan.instructions}
    task0_hbm_sends = [
        comm for comm in plan.comms if comm.kind is df.DataflowCommKind.HBM_SEND and instructions[comm.target_instruction_id].task_id == 0
    ]

    assert task0_iter_clusters == {0}
    assert task0_hbm_sends == []


def test_scheduler_critical_path_capacity_split_requires_joint_deadlock_free_queue_order(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_SPLIT", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_HIER_CROSS_CLUSTER_SPLIT", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_DIRECT_LEAF_ACC", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=112, cluster_size=16),
        range_lengths={
            "kv": [
                566,
                1251,
                1951,
                2662,
                3403,
                4188,
                5041,
                5961,
                6975,
                8112,
                9405,
                10902,
                12696,
                14919,
                17885,
                22527,
            ]
        },
        block_size=64,
        task_extents=(16,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        cluster_task_assignment=(
            (15,),
            (14, 3, 1),
            (4, 6, 0),
            (8, 13),
            (11, 7, 2),
            (10, 9, 5),
            (12,),
        ),
    )

    split_tasks = {
        task_id
        for task_id in range(16)
        if len(
            {
                plan.topology.cluster_id(inst.sm_id)
                for inst in plan.instructions
                if inst.opcode is df.DataflowOpcode.ITER and inst.task_id == task_id
            }
        )
        > 1
    }

    assert len(plan.instructions) > 0
    assert split_tasks == set()

    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL0_QUEUE_ORDER", "producer_ready")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_JOINT_SCHEDULE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_CAPACITY_SEARCH_MAX_EVALS", "1")
    monkeypatch.setenv(
        "TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_MAX_TASK_SEGMENTS",
        str(plan.topology.cluster_count),
    )
    monkeypatch.setenv(
        "TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_MAX_EXTRA_CHUNKS",
        str(plan.topology.sm_count + len(plan.task_range_lengths)),
    )
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_EXTRA_CHUNK_PENALTY_US", "0")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_HBM_PENALTY_US", "0")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_P95_WEIGHT", "0")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_RECV_WEIGHT", "0")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_TOTAL_RECV_WEIGHT", "0")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CRITICAL_PATH_MIN_GAIN_US", "0")
    balanced = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=112, cluster_size=16),
        range_lengths={
            "kv": [
                566,
                1251,
                1951,
                2662,
                3403,
                4188,
                5041,
                5961,
                6975,
                8112,
                9405,
                10902,
                12696,
                14919,
                17885,
                22527,
            ]
        },
        block_size=64,
        task_extents=(16,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        cluster_task_assignment=(
            (15,),
            (14, 3, 1),
            (4, 6, 0),
            (8, 13),
            (11, 7, 2),
            (10, 9, 5),
            (12,),
        ),
    )
    iter_instructions = [instruction for instruction in balanced.instructions if instruction.opcode is df.DataflowOpcode.ITER]
    balanced_split_tasks = {
        task_id
        for task_id in range(16)
        if len({balanced.topology.cluster_id(instruction.sm_id) for instruction in iter_instructions if instruction.task_id == task_id}) > 1
    }
    hbm_sends = [comm for comm in balanced.comms if comm.kind is df.DataflowCommKind.HBM_SEND]
    sm_block_loads = [
        sum(math.ceil(instruction.task_range.length / 64) for instruction in iter_instructions if instruction.sm_id == sm_id)
        for sm_id in range(balanced.topology.sm_count)
    ]

    assert balanced_split_tasks
    assert len(hbm_sends) == len(balanced_split_tasks)
    assert balanced.joint_execution_plan is not None
    joint = balanced.joint_execution_plan.require_valid(topology=balanced.topology)
    assert joint.schedule.comm_slot_intervals
    assert max(sm_block_loads) > 0


def test_scheduler_ready_time_tree_can_penalize_candidate_queue_tail(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL_BUCKET_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL0_QUEUE_ORDER", "long_first")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_QUEUE_WEIGHT", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=96, cluster_size=16),
        range_lengths={
            "kv": [
                566,
                1251,
                1951,
                2662,
                3403,
                4188,
                5041,
                5961,
                6975,
                8112,
                9405,
                10902,
                12696,
                14919,
                17885,
                22527,
            ]
        },
        block_size=64,
        task_extents=(16,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    task9_root = [
        inst
        for inst in plan.instructions
        if inst.opcode is df.DataflowOpcode.REDUCE_UPDATE and inst.task_id == 9 and inst.attrs.get("tree_level") == 3
    ]

    assert len(task9_root) == 1
    assert task9_root[0].sm_id != 86


def test_scheduler_can_place_root_reduce_on_late_input_producer(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL_BUCKET_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL0_QUEUE_ORDER", "long_first")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_ROOT_REDUCE_LATE_PRODUCER", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=96, cluster_size=16),
        range_lengths={
            "kv": [
                566,
                1251,
                1951,
                2662,
                3403,
                4188,
                5041,
                5961,
                6975,
                8112,
                9405,
                10902,
                12696,
                14919,
                17885,
                22527,
            ]
        },
        block_size=64,
        task_extents=(16,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    producer_sm_by_slot = {inst.output_slot: inst.sm_id for inst in plan.instructions if inst.output_slot is not None}
    task3_root = [
        inst
        for inst in plan.instructions
        if inst.opcode is df.DataflowOpcode.REDUCE_UPDATE and inst.task_id == 3 and inst.attrs.get("tree_level") == 2
    ]

    assert len(task3_root) == 1
    assert [producer_sm_by_slot[slot] for slot in task3_root[0].input_slots] == [30, 31]
    assert task3_root[0].sm_id == 31

    task2_root = [
        inst
        for inst in plan.instructions
        if inst.opcode is df.DataflowOpcode.REDUCE_UPDATE and inst.task_id == 2 and inst.attrs.get("tree_level") == 2
    ]

    assert len(task2_root) == 1
    assert [producer_sm_by_slot[slot] for slot in task2_root[0].input_slots] == [61, 63]
    assert task2_root[0].sm_id == 61


def test_scheduler_can_place_reduce_on_higher_level_producer(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL_BUCKET_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL0_QUEUE_ORDER", "long_first")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_REDUCE_HIGHER_LEVEL_PRODUCER", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [64, 256, 768]},
        block_size=64,
        task_extents=(3,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    producer_sm_by_slot = {inst.output_slot: inst.sm_id for inst in plan.instructions if inst.output_slot is not None}
    producer_level_by_slot = {
        inst.output_slot: inst.attrs.get("tree_level", 0) for inst in plan.instructions if inst.output_slot is not None
    }
    task2_root = [
        inst
        for inst in plan.instructions
        if inst.opcode is df.DataflowOpcode.REDUCE_UPDATE and inst.task_id == 2 and inst.attrs.get("tree_level") == 2
    ]

    assert len(task2_root) == 1
    root = task2_root[0]
    assert [producer_level_by_slot[slot] for slot in root.input_slots] == [1, 0]
    assert [producer_sm_by_slot[slot] for slot in root.input_slots] == [1, 2]
    assert root.sm_id == 1


def test_scheduler_chunk_swap_search_reduces_mixed_tail_loads(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CHUNK_SWAP_SEARCH", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=96, cluster_size=16),
        range_lengths={
            "kv": [
                566,
                1251,
                1951,
                2662,
                3403,
                4188,
                5041,
                5961,
                6975,
                8112,
                9405,
                10902,
                12696,
                14919,
                17885,
                22527,
            ]
        },
        block_size=64,
        task_extents=(16,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    iter_by_task_part = {
        (inst.task_id, inst.task_range.begin, inst.task_range.end): inst.sm_id
        for inst in plan.instructions
        if inst.opcode is df.DataflowOpcode.ITER
    }

    assert iter_by_task_part[(10, 8448, 9405)] == 95
    assert iter_by_task_part[(5, 3584, 4188)] == 86
    assert iter_by_task_part[(12, 12096, 12696)] == 61
    assert iter_by_task_part[(2, 1408, 1951)] == 57
    assert iter_by_task_part[(7, 4736, 5961)] == 63
    assert len(plan.slots) == 302
    assert len(plan.comms) == 180


def test_scheduler_streaming_tree_preserves_associative_operand_order(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL_BUCKET_TREE", "1")

    baseline = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [320]},
        block_size=64,
        task_extents=(1,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    producer_by_slot = {inst.output_slot: inst for inst in baseline.instructions if inst.output_slot is not None}

    def leaf_begins(slot_id):
        instruction = producer_by_slot[slot_id]
        if instruction.opcode is df.DataflowOpcode.ITER:
            return [instruction.task_range.begin]
        return [begin for input_slot in instruction.input_slots for begin in leaf_begins(input_slot)]

    finalize_instruction = next(inst for inst in baseline.instructions if inst.opcode is df.DataflowOpcode.FINALIZE)

    assert leaf_begins(finalize_instruction.input_slots[0]) == sorted(leaf_begins(finalize_instruction.input_slots[0]))

    default = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [320]},
        block_size=64,
        task_extents=(1,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    assert len(baseline.slots) == len(default.slots)
    assert len(baseline.comms) == len(default.comms)


def test_scheduler_leaf_normalization_does_not_permute_ranges(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_READY_TIME_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL_BUCKET_TREE", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL0_QUEUE_ORDER", "long_first")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_CHUNK_SWAP_SEARCH", "1")

    topology = df.GPUTopology(sm_count=96, cluster_size=16)
    config = df.DataflowSchedulerConfig.from_options(**scheduler_test_options())
    with df.use_scheduler_config(config):
        chunks = scheduler.load_balanced_streaming_chunks(
            topology,
            (
                566,
                1251,
                1951,
                2662,
                3403,
                4188,
                5041,
                5961,
                6975,
                8112,
                9405,
                10902,
                12696,
                14919,
                17885,
                22527,
            ),
            block_size=64,
            axis="kv",
        )
        optimized = scheduler.optimize_streaming_tree_leaf_orders(
            topology,
            chunks,
            block_size=64,
            streaming_tree_consumer="left",
            hierarchical_cross_cluster=False,
            ready_time_tree=True,
        )

    def chunk_order(task_chunks):
        return tuple(
            (
                chunk.part_index,
                chunk.sm_id,
                chunk.task_range.begin,
                chunk.task_range.end,
            )
            for chunk in task_chunks
        )

    assert [task_id for task_id, (before, after) in enumerate(zip(chunks, optimized)) if chunk_order(before) != chunk_order(after)] == []


def test_scheduler_streaming_reduce_keeps_slot_count_bounded_by_cluster_chunks():
    small = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [4096, 4096]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming",
    )
    large = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [8192, 8192]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming",
    )

    assert len(small.slots) == 8
    assert len(large.slots) == len(small.slots)
    assert len([inst for inst in large.instructions if inst.opcode is df.DataflowOpcode.ITER]) == 4


def test_scheduler_streaming_reduce_uses_nonuniform_chunks_to_balance_sm_loads():
    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"kv": [700, 300]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming",
    )

    iter_instructions = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    assert [(inst.task_id, inst.sm_id, inst.task_range.begin, inst.task_range.end) for inst in iter_instructions] == [
        (0, 0, 0, 512),
        (0, 1, 512, 700),
        (1, 1, 0, 300),
    ]

    sm1_iters = [inst for inst in plan.queue(1) if inst.opcode is df.DataflowOpcode.ITER]
    assert [(inst.task_id, inst.task_range.begin, inst.task_range.end) for inst in sm1_iters] == [
        (1, 0, 300),
        (0, 512, 700),
    ]
    sm_loads = {
        sm_id: sum(inst.task_range.end - inst.task_range.begin for inst in plan.queue(sm_id) if inst.opcode is df.DataflowOpcode.ITER)
        for sm_id in range(2)
    }
    assert sm_loads == {0: 512, 1: 488}


def test_scheduler_streaming_reduce_balances_chunk_targets_in_block_units():
    block_size = 64
    range_lengths = [10902, 6975, 3403]
    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=16, cluster_size=16),
        range_lengths={"kv": range_lengths},
        block_size=block_size,
        task_extents=(3,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming",
    )

    target_blocks = sum((length + block_size - 1) // block_size for length in range_lengths)
    target_blocks = (target_blocks + 16 - 1) // 16
    iter_instructions = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    chunk_blocks = [(inst.task_range.length + block_size - 1) // block_size for inst in iter_instructions]

    assert max(chunk_blocks) <= target_blocks
    assert all(comm.kind is not df.DataflowCommKind.HBM_SEND for comm in plan.comms)


def test_scheduler_streaming_reduce_can_force_hbm_fallback_inside_cluster():
    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [512]},
        block_size=64,
        task_extents=(1,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming",
        force_hbm_comms=True,
    )

    assert any(comm.kind is df.DataflowCommKind.HBM_SEND for comm in plan.comms)
    assert any(comm.kind is df.DataflowCommKind.HBM_RECV for comm in plan.comms)
    assert all(comm.kind is not df.DataflowCommKind.CLUSTER_SEND for comm in plan.comms)
    assert all(comm.kind is not df.DataflowCommKind.CLUSTER_RECV for comm in plan.comms)


def test_scheduler_pic_writes_readable_cluster_and_sm_queue_files(tmp_path):
    build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [256, 128]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming",
        pic=True,
        pic_dir=tmp_path,
    )

    txt_files = sorted(tmp_path.glob("*.txt"))
    md_files = sorted(tmp_path.glob("*.md"))
    assert len(txt_files) == 1
    assert len(md_files) == 1

    text = txt_files[0].read_text()
    markdown = md_files[0].read_text()
    assert "Dataflow Schedule" in text
    assert "scheduler_policy: cluster_local" in text
    assert "reduce_strategy: streaming" in text
    assert "cluster 0: SM0, SM1" in text
    assert "cluster 1: SM2, SM3" in text
    assert "SM0 [cluster 0]" in text
    assert "SM1 [cluster 0]" in text
    assert "ITER task=0 coords=(0,) range=[0, 128)" in text
    assert "REDUCE_UPDATE task=0 coords=(0,) inputs=(0,) output=1" in text
    assert "CLUSTER_SEND slot 1 -> 1 SM0->SM1" in text
    assert "```mermaid" in markdown
    assert '"SM0 [cluster 0]"' in markdown
    assert '"task0 [0,128)"' in markdown


def test_scheduler_pic_can_be_enabled_by_typed_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_SCHEDULE_PIC", "1")

    build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"kv": [128]},
        block_size=64,
        task_extents=(1,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming",
    )

    output_dir = tmp_path / "schedule_res"
    txt_files = sorted(output_dir.glob("*.txt"))
    md_files = sorted(output_dir.glob("*.md"))
    assert len(txt_files) == 1
    assert len(md_files) == 1
    assert "Dataflow Schedule" in txt_files[0].read_text()


def test_scheduler_uses_initial_cluster_barrier_phase_for_each_slot():
    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [256, 512]},
        block_size=128,
        task_extents=(2,),
    )

    cluster_comms = [comm for comm in plan.comms if comm.kind in (df.DataflowCommKind.CLUSTER_SEND, df.DataflowCommKind.CLUSTER_RECV)]

    assert [(comm.kind, comm.source_slot_id, comm.target_slot_id) for comm in cluster_comms] == [
        (df.DataflowCommKind.CLUSTER_SEND, 1, 1),
        (df.DataflowCommKind.CLUSTER_RECV, 1, 1),
        (df.DataflowCommKind.CLUSTER_SEND, 4, 4),
        (df.DataflowCommKind.CLUSTER_RECV, 4, 4),
    ]
    assert [comm.barrier_phase for comm in cluster_comms] == [0, 0, 0, 0]


def test_scheduler_streaming_cluster_chain_uses_initial_barrier_phase():
    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [1024]},
        block_size=64,
        task_extents=(1,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming",
    )

    cluster_comms = [comm for comm in plan.comms if comm.kind in (df.DataflowCommKind.CLUSTER_SEND, df.DataflowCommKind.CLUSTER_RECV)]

    assert len(cluster_comms) == 6
    assert all(comm.barrier_phase == 0 for comm in cluster_comms)


def test_scheduler_can_infer_range_axis_when_single_range_length_is_provided():
    program = (
        T.dataflow_program(task_domain=("batch", "head"))
        .partial(
            split_kv_no_stage_axis(Q="Q", K="K", V="V"),
            task_args=("seq", "head"),
        )
        .reduce(combine())
        .finalize(finalize(Output="O"))
    )

    plan = build_schedule(
        program,
        topology=df.GPUTopology(sm_count=1),
        range_lengths={"kv": 256},
        block_size=128,
        include_exit=False,
    )

    assert plan.range_axis == "kv"
    assert [inst.task_range.length for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER] == [128, 128]


def test_scheduler_range_offsets_shift_iter_task_ranges():
    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"kv": [128, 64]},
        range_offsets={"kv": [1024, 2048]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    iter_instructions = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    assert [(inst.task_id, inst.task_range.begin, inst.task_range.end) for inst in iter_instructions] == [
        (0, 1024, 1152),
        (1, 2048, 2112),
    ]


def test_scheduler_partial_only_emits_iter_without_reduce_finalize_or_comms():
    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=2),
        range_lengths={"kv": [128, 64]},
        block_size=64,
        task_extents=(2,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        partial_only=True,
    )

    assert [inst.opcode for inst in plan.instructions] == [
        df.DataflowOpcode.ITER,
        df.DataflowOpcode.ITER,
        df.DataflowOpcode.ITER,
    ]
    assert not plan.comms
    assert [(slot.task_id, slot.role) for slot in plan.slots] == [
        (0, "partial"),
        (0, "partial"),
        (1, "partial"),
    ]


def test_scheduler_root_reduce_cluster_candidates_stay_cluster_local():
    topology = df.GPUTopology(sm_count=8, cluster_size=4)

    assert scheduler.streaming_tree_reduce_candidate_sms(
        topology,
        left_sm=0,
        right_sm=2,
        prefer_late_input=False,
    ) == (0, 2)

    config = df.DataflowSchedulerConfig.from_options(root_reduce_cluster_candidates=True)
    with df.use_scheduler_config(config):
        assert scheduler.streaming_tree_reduce_candidate_sms(
            topology,
            left_sm=0,
            right_sm=2,
            prefer_late_input=True,
        ) == (0, 1, 2, 3)
        assert scheduler.streaming_tree_reduce_candidate_sms(
            topology,
            left_sm=0,
            right_sm=5,
            prefer_late_input=True,
        ) == (0, 5)


def test_scheduler_rebuilds_streaming_task_chunks_from_block_counts():
    chunks = (
        scheduler.StreamingChunk(
            task_id=0,
            sm_id=4,
            task_range=scheduler.TaskRange(axis="kv", begin=128, end=256),
            part_index=0,
            part_count=3,
        ),
        scheduler.StreamingChunk(
            task_id=0,
            sm_id=5,
            task_range=scheduler.TaskRange(axis="kv", begin=256, end=384),
            part_index=1,
            part_count=3,
        ),
        scheduler.StreamingChunk(
            task_id=0,
            sm_id=6,
            task_range=scheduler.TaskRange(axis="kv", begin=384, end=512),
            part_index=2,
            part_count=3,
        ),
    )

    rebuilt = scheduler.rebuild_streaming_task_chunks_with_block_counts(
        chunks,
        block_counts=(1, 3, 2),
        block_size=64,
    )

    assert rebuilt is not None
    assert [(chunk.sm_id, chunk.part_index, chunk.part_count) for chunk in rebuilt] == [
        (4, 0, 3),
        (5, 1, 3),
        (6, 2, 3),
    ]
    assert [(chunk.task_range.begin, chunk.task_range.end) for chunk in rebuilt] == [
        (128, 192),
        (192, 384),
        (384, 512),
    ]


def test_scheduler_streaming_tree_can_write_iter_directly_to_leaf_acc(monkeypatch):
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_DIRECT_LEAF_ACC", "1")
    monkeypatch.setenv("TEST_DATAFLOW_SCHEDULER_LEVEL_BUCKET_TREE", "1")

    plan = build_schedule(
        make_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"kv": [1024]},
        block_size=64,
        task_extents=(1,),
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
    )

    iter_instructions = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.ITER]
    reduce_updates = [inst for inst in plan.instructions if inst.opcode is df.DataflowOpcode.REDUCE_UPDATE]

    assert len(iter_instructions) == 4
    assert [len(inst.input_slots) for inst in reduce_updates] == [2, 2, 2]
    assert all(slot.role == "streaming_acc" for slot in plan.slots)
    assert len(plan.slots) == 7
    assert len(plan.comms) == 6


def test_scheduler_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="missing a reduce stage"):
        build_schedule(
            T.dataflow_program(task_domain=("batch", "head")).partial(
                split_kv(Q="Q", K="K", V="V"),
                task_args=("seq", "head"),
                range_axis="kv",
            ),
            topology=df.GPUTopology(sm_count=1),
            range_lengths={"kv": [128]},
            block_size=128,
        )

    with pytest.raises(ValueError, match="missing range_lengths entry"):
        build_schedule(
            make_program(),
            topology=df.GPUTopology(sm_count=1),
            range_lengths={"other": [128]},
            block_size=128,
        )

    with pytest.raises(ValueError, match="block_size"):
        build_schedule(
            make_program(),
            topology=df.GPUTopology(sm_count=1),
            range_lengths={"kv": [128]},
            block_size=0,
        )

    with pytest.raises(ValueError, match="task_extents product"):
        build_schedule(
            make_program(),
            topology=df.GPUTopology(sm_count=1),
            range_lengths={"kv": [128, 128, 128]},
            block_size=128,
            task_extents=(2, 2),
        )

    with pytest.raises(ValueError, match="range length must be positive"):
        build_schedule(
            make_program(),
            topology=df.GPUTopology(sm_count=1),
            range_lengths={"kv": [0]},
            block_size=128,
        )

    with pytest.raises(ValueError, match="sm_count"):
        df.GPUTopology(sm_count=0)
