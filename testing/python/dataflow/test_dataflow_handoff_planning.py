from __future__ import annotations

import copy

import pytest

import tilelang.language as T
import tilelang.dataflow as df
from tilelang import tvm


def make_pipeline_request(*, resident_second: bool = False):
    return df.DataflowPipelineRequest(
        transfers=(
            df.DataflowPipelineTransfer(
                destination_buffer_index=0,
                logical_extent=(32, 64),
                bytes_per_stage=4_096,
            ),
            df.DataflowPipelineTransfer(
                destination_buffer_index=1,
                logical_extent=(64, 64),
                bytes_per_stage=8_192,
                materialization=(df.DATAFLOW_PIPELINE_MATERIALIZE_RESIDENT if resident_second else df.DATAFLOW_PIPELINE_MATERIALIZE_COPY),
                async_permitted=not resident_second,
            ),
        ),
        gemms=(
            df.DataflowPipelineGemm(
                input_buffer_indices=(0, 1),
                accumulator_index=0,
            ),
        ),
        buffer_lifetimes=(
            df.DataflowPipelineBufferLifetime(
                buffer_index=0,
                producer_transfer_index=0,
                consumer_gemm_indices=(0,),
                release_after_gemm_index=0,
            ),
            df.DataflowPipelineBufferLifetime(
                buffer_index=1,
                producer_transfer_index=1,
                consumer_gemm_indices=(0,),
                release_after_gemm_index=0,
                allow_multiversion=not resident_second,
            ),
        ),
        stage_budget=3,
        max_outstanding=2,
        producer_threads=128,
        consumer_threads=128,
        synchronization_owner=df.DATAFLOW_PIPELINE_SYNC_PIPELINE,
        completion_semantics=df.DATAFLOW_PIPELINE_OWNED_COMPLETION,
        release_semantics=df.DATAFLOW_PIPELINE_OWNED_RELEASE,
    )


@T.prim_func
def pipeline_consumer_primfunc(
    left: T.Tensor((32, 192), T.float16),
    right: T.Tensor((192, 64), T.float16),
    output: T.Tensor((32, 64), T.float32),
):
    with T.Kernel(1, threads=32):
        left_shared = T.alloc_shared((32, 64), T.float16)
        right_shared = T.alloc_shared((64, 64), T.float16)
        accumulator = T.alloc_fragment((32, 64), T.float32)
        T.clear(accumulator)
        for tile in T.Pipelined(3, num_stages=3):
            T.copy(
                left[:, tile * 64 : (tile + 1) * 64],
                left_shared,
            )
            T.copy(
                right[tile * 64 : (tile + 1) * 64, :],
                right_shared,
            )
            T.gemm(left_shared, right_shared, accumulator)
        T.copy(accumulator, output)


def test_handoff_plan_derives_transfer_arena_and_lifetimes_from_pipeline():
    request = df.DataflowCrossHandlerHandoffRequest(
        consumer_stage_id=0,
        lookahead_distance=2,
        buffer_stages=3,
    )
    plan = df.plan_cross_handler_handoff(
        request,
        producer_stage_id=2,
        consumer_pipeline_request=make_pipeline_request(),
        task_coord_rank=2,
        max_shared_memory_bytes=256 * 1024,
    )

    assert plan.enabled
    assert plan.value_bindings == (0, 1)
    assert plan.selected_buffer_stages == 3
    assert plan.resource_budget_bytes == 256 * 1024
    assert plan.required_arena_bytes == plan.arena_bytes
    assert plan.arena_bytes == 2 * 3 * (4_096 + 8_192)
    assert [item.arena_offset for item in plan.transfer_plans] == [0, 24_576]
    assert [item.lookahead_slots for item in plan.transfer_plans] == [2, 2]
    assert [item.physical_buffer_stages for item in plan.transfer_plans] == [3, 3]
    assert plan.barrier_count == 6
    assert plan.wait_required and plan.release_required
    assert [item.release_phase for item in plan.buffer_lifetimes] == [
        "consumer_release",
        "consumer_release",
    ]
    assert df.DataflowCrossHandlerHandoffPlan.from_dict(plan.to_dict()) == plan


def test_handoff_plan_binds_selected_consumer_transfers_to_primfunc():
    pipeline_request = make_pipeline_request()
    pipeline_bound, _ = df.bind_pipeline_dataflow_plan(
        pipeline_consumer_primfunc,
        pipeline_request,
    )
    plan = df.plan_cross_handler_handoff(
        df.DataflowCrossHandlerHandoffRequest(
            consumer_stage_id=0,
            buffer_stages=2,
            value_bindings=(1,),
        ),
        producer_stage_id=2,
        consumer_pipeline_request=pipeline_request,
        task_coord_rank=1,
    )
    bound = df.bind_cross_handler_handoff_plan(
        pipeline_bound,
        plan,
        role="consumer",
        pipeline_request=pipeline_request,
    )

    assert str(bound.attrs[df.DATAFLOW_CROSS_HANDLER_HANDOFF_PLAN_FINGERPRINT_ATTR]) == plan.fingerprint
    assert str(bound.attrs[df.DATAFLOW_CROSS_HANDLER_HANDOFF_ROLE_ATTR]) == "consumer"
    transfer_indices = []

    def visit(node):
        if not isinstance(node, tvm.tir.Call):
            return
        value = node.annotations.get(df.DATAFLOW_CROSS_HANDLER_HANDOFF_TRANSFER_INDEX_ATTR)
        if value is not None:
            transfer_indices.append(int(value))

    tvm.tir.stmt_functor.post_order_visit(bound.body, visit)
    assert transfer_indices == [0]
    assert df.DATAFLOW_CROSS_HANDLER_HANDOFF_ARENA_OFFSET_ATTR in bound.script()


def test_handoff_plan_honors_explicit_value_bindings_without_shape_rules():
    request = df.DataflowCrossHandlerHandoffRequest(
        consumer_stage_id=0,
        buffer_stages=2,
        value_bindings=(1,),
    )
    plan = df.plan_cross_handler_handoff(
        request,
        producer_stage_id=2,
        consumer_pipeline_request=make_pipeline_request(),
        task_coord_rank=1,
    )

    assert plan.value_bindings == (1,)
    assert len(plan.transfer_plans) == 1
    assert plan.transfer_plans[0].logical_extent == (64, 64)
    assert plan.transfer_plans[0].buffer_stages == 2
    assert plan.transfer_plans[0].physical_buffer_stages == 3
    assert plan.arena_bytes == 24_576


@pytest.mark.parametrize(
    ("handoff_request", "pipeline", "error"),
    [
        (
            df.DataflowCrossHandlerHandoffRequest(
                consumer_stage_id=0,
                value_bindings=(2,),
            ),
            make_pipeline_request(),
            "missing consumer transfers",
        ),
        (
            df.DataflowCrossHandlerHandoffRequest(
                consumer_stage_id=0,
                value_bindings=(1,),
            ),
            make_pipeline_request(resident_second=True),
            "copy-materialized transfers",
        ),
    ],
)
def test_handoff_plan_rejects_invalid_pipeline_bindings(
    handoff_request,
    pipeline,
    error,
):
    with pytest.raises(df.DataflowCrossHandlerHandoffPlanningError, match=error):
        df.plan_cross_handler_handoff(
            handoff_request,
            producer_stage_id=2,
            consumer_pipeline_request=pipeline,
            task_coord_rank=1,
        )


def test_handoff_plan_records_disabled_and_resource_fallbacks():
    request = df.DataflowCrossHandlerHandoffRequest(
        consumer_stage_id=0,
        buffer_stages=2,
    )
    no_pipeline = df.plan_cross_handler_handoff(
        request,
        producer_stage_id=2,
        consumer_pipeline_request=None,
        task_coord_rank=1,
    )
    resource = df.plan_cross_handler_handoff(
        request,
        producer_stage_id=2,
        consumer_pipeline_request=make_pipeline_request(),
        task_coord_rank=1,
        max_shared_memory_bytes=1,
    )

    assert not no_pipeline.enabled
    assert no_pipeline.fallback_reason == "consumer_stage_has_no_pipeline_contract"
    assert no_pipeline.resource_budget_bytes is None
    assert no_pipeline.required_arena_bytes == 0
    assert not resource.enabled
    assert resource.fallback_reason == "handoff_arena_exceeds_shared_memory_budget"
    assert resource.resource_budget_bytes == 1
    assert resource.required_arena_bytes == 3 * (4_096 + 8_192)
    assert no_pipeline.arena_bytes == resource.arena_bytes == 0
    assert no_pipeline.decision.used_fallback
    assert resource.decision.used_fallback
    rejected = resource.decision.candidates[0]
    assert not rejected.legal
    assert rejected.resources.shared_memory_bytes == resource.required_arena_bytes
    assert rejected.resources.temporary_bytes == resource.required_arena_bytes

    tampered = copy.deepcopy(resource.to_dict())
    tampered["arena_bytes"] = 16
    with pytest.raises(ValueError, match="disabled handoff"):
        df.DataflowCrossHandlerHandoffPlan.from_dict(tampered)


@T.dataflow_intermediate
class Shard:
    value: T.int32


@T.dataflow_intermediate
class Full:
    value: T.int32


@T.dataflow.map(range=("begin", "end"))
def stage_a(task: T.int32, Source: T.Tensor((64,), T.int32)) -> Shard:
    return Shard(value=Source[T.dataflow_range_begin()])


@T.dataflow.map(range=("begin", "end"))
def stage_b(parts: list[Shard], task: T.int32) -> Shard:
    return Shard(value=parts[0].value)


def handoff_program(*, distance=1, budget=2):
    return (
        T.dataflow_program(
            task_domain=("task",),
            dynamic_ranges={"a": "a_ranges", "b": "b_ranges"},
        )
        .map(
            stage_a(Source="Source"),
            name="a",
            task_args=("task",),
            range_axis="a",
            pipeline_contract=make_pipeline_request(),
        )
        .reshared(
            input="a",
            name="exchange",
            output_type=Full,
            physical_output_type=Shard,
            output_arity=1,
            policy="hbm_all_gather",
        )
        .map(
            stage_b(),
            name="b",
            input="exchange",
            task_args=("task",),
            range_axis="b",
            handoff_contract=df.DataflowCrossHandlerHandoffRequest(
                consumer_stage_id=0,
                lookahead_distance=distance,
                buffer_stages=budget,
            ),
        )
    )


def test_scheduler_records_noncontiguous_future_task_queue_bindings():
    plan = df.schedule(
        handoff_program(distance=2),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"a": [16, 16, 16, 16], "b": [16, 16, 16, 16]},
        block_size=16,
        include_exit=False,
        task_coord_overrides=((3,), (9,), (14,), (27,)),
        stage_graph_task_weights=(9, 1, 4, 2),
        stage_graph_cluster_assignment=(0, 0, 0, 0),
    )

    assert len(plan.cross_handler_handoff_plans) == 1
    handoff_plan = plan.cross_handler_handoff_plans[0]
    assert handoff_plan.enabled
    bindings = plan.cross_handler_handoff_bindings
    assert [item.state for item in bindings] == [
        "active",
        "active",
        "tail",
        "tail",
    ]
    assert [item.consumer_task_coords for item in bindings] == [
        (14,),
        (27,),
        None,
        None,
    ]
    assert [item.arena_slot for item in bindings] == [0, 1, None, None]
    assert all(
        instruction.attrs.get("cross_handler_handoff_plan_fingerprint") == handoff_plan.fingerprint
        for instruction in plan.instructions
        if instruction.attrs.get("stage_id") in {0, 2}
    )


def test_scheduler_falls_back_when_no_queue_has_a_future_consumer():
    plan = df.schedule(
        handoff_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"a": [16, 16], "b": [16, 16]},
        block_size=16,
        include_exit=False,
        task_coord_overrides=((3,), (9,)),
        stage_graph_task_weights=(1, 1),
        stage_graph_cluster_assignment=(0, 1),
    )

    handoff_plan = plan.cross_handler_handoff_plans[0]
    assert not handoff_plan.enabled
    assert handoff_plan.fallback_reason == "handoff_queue_has_no_active_edges"
    assert handoff_plan.required_arena_bytes == 3 * (4_096 + 8_192)
    assert handoff_plan.arena_bytes == 0
    assert handoff_plan.decision.candidates[0].resources.shared_memory_bytes == (handoff_plan.required_arena_bytes)
    assert plan.cross_handler_handoff_bindings
    assert all(binding.state == "disabled" for binding in plan.cross_handler_handoff_bindings)
    assert all(
        instruction.attrs.get("cross_handler_handoff_plan_fingerprint") == handoff_plan.fingerprint
        for instruction in plan.instructions
        if instruction.attrs.get("stage_id") in {0, 2}
    )
