from __future__ import annotations

import copy
import os

import pytest
import torch

from tilelang import _ffi_api, tvm
import tilelang.language as T
import tilelang.dataflow as df
from tvm import tir


_HOPPER = df.TargetCapabilitySnapshot.for_cuda(
    (9, 0),
    compiler_version=(12, 8),
    max_dynamic_shared_memory=227_328,
)
_PORTABLE = df.TargetCapabilitySnapshot.for_cuda(
    (8, 0),
    supports_cluster_launch=False,
    compiler_version=(12, 8),
)


@T.dataflow_intermediate
class StreamShard:
    value: T.Tensor((64, 64), T.float16)


@T.dataflow_intermediate
class StreamFull:
    value: T.Tensor((2, 64, 64), T.float16)


@T.dataflow_intermediate
class StreamResult:
    value: T.Tensor((64, 64), T.float32)


@T.dataflow.map(
    range=("begin", "end"),
    threads=128,
    physical_contract=df.DataflowOperatorPhysicalContract(
        output_slot=df.DATAFLOW_OUTPUT_SLOT_DIRECT,
    ),
)
def stream_source(
    task: T.int32,
    Input: T.Tensor((2, 64, 64), T.float16),
) -> StreamShard:
    value = T.alloc_shared((64, 64), T.float16)
    source_index = T.cast(T.dataflow_range_begin(), "int32")
    T.copy(Input[source_index, :, :], value)
    return StreamShard(value=value)


@T.dataflow.map(
    range=("begin", "end"),
    threads=256,
    physical_contract=df.DataflowOperatorPhysicalContract(
        input_slots=df.DATAFLOW_INPUT_SLOTS_CONTIGUOUS,
    ),
)
def stream_consumer(
    parts: list[StreamShard],
    task: T.int32,
    Weight: T.Tensor((64, 64), T.float16),
) -> StreamResult:
    left = T.alloc_shared((64, 64), T.float16)
    right = T.alloc_shared((64, 64), T.float16)
    accumulator = T.alloc_fragment((64, 64), T.float32)
    T.clear(accumulator)
    for logical_tile in T.Pipelined(2, num_stages=2):
        T.copy(
            parts[logical_tile].value[:, :],
            left,
            valid_region=parts[logical_tile].value[:, :],
            synchronization_owner="pipeline",
        )
        T.copy(
            Weight,
            right,
            valid_region=Weight,
            synchronization_owner="pipeline",
        )
        T.gemm(left, right, accumulator)
    return StreamResult(value=accumulator)


@T.dataflow.map(
    range=("begin", "end"),
    threads=256,
)
def stream_discrete_consumer(
    parts: list[StreamShard],
    task: T.int32,
    Weight: T.Tensor((64, 64), T.float16),
) -> StreamResult:
    left = T.alloc_shared((64, 64), T.float16)
    right = T.alloc_shared((64, 64), T.float16)
    accumulator = T.alloc_fragment((64, 64), T.float32)
    T.clear(accumulator)
    for logical_tile in T.Pipelined(2, num_stages=2):
        T.copy(
            parts[logical_tile].value[:, :],
            left,
            valid_region=parts[logical_tile].value[:, :],
            synchronization_owner="pipeline",
        )
        T.copy(
            Weight,
            right,
            valid_region=Weight,
            synchronization_owner="pipeline",
        )
        T.gemm(left, right, accumulator)
    return StreamResult(value=accumulator)


@T.dataflow.map(
    range=("begin", "end"),
    threads=256,
    physical_contract=df.DataflowOperatorPhysicalContract(
        input_slots=df.DATAFLOW_INPUT_SLOTS_CONTIGUOUS,
    ),
)
def transport_copy_consumer(
    parts: list[StreamShard],
    task: T.int32,
) -> StreamResult:
    received = T.alloc_shared((64, 64), T.float16)
    result = T.alloc_shared((64, 64), T.float32)
    T.clear(result)
    for logical_tile in T.serial(2):
        T.copy(parts[logical_tile].value[:, :], received)
        T.sync_threads()
        for row, col in T.Parallel(64, 64):
            result[row, col] += T.cast(received[row, col], T.float32)
        T.sync_threads()
    return StreamResult(value=result)


@T.dataflow.map(
    range=("begin", "end"),
    threads=256,
)
def transport_discrete_copy_consumer(
    parts: list[StreamShard],
    task: T.int32,
) -> StreamResult:
    received = T.alloc_shared((64, 64), T.float16)
    result = T.alloc_shared((64, 64), T.float32)
    T.clear(result)
    for logical_tile in T.serial(2):
        T.copy(parts[logical_tile].value[:, :], received)
        T.sync_threads()
        for row, col in T.Parallel(64, 64):
            result[row, col] += T.cast(received[row, col], T.float32)
        T.sync_threads()
    return StreamResult(value=result)


@T.dataflow.finalize
def stream_finalize(
    result: StreamResult,
    Output: T.Tensor((64, 64), T.float32),
) -> None:
    row_begin = T.cast(T.dataflow_range_begin(), "int32") * 32
    for row, col in T.Parallel(32, 64):
        Output[row_begin + row, col] = result.value[row_begin + row, col]


def stream_pipeline_request():
    return df.DataflowPipelineRequest(
        transfers=(
            df.DataflowPipelineTransfer(
                destination_buffer_index=0,
                logical_extent=(64, 64),
                bytes_per_stage=8_192,
            ),
            df.DataflowPipelineTransfer(
                destination_buffer_index=1,
                logical_extent=(64, 64),
                bytes_per_stage=8_192,
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
            ),
        ),
        stage_budget=2,
        max_outstanding=2,
        producer_threads=128,
        consumer_threads=128,
        synchronization_owner=df.DATAFLOW_PIPELINE_SYNC_PIPELINE,
        completion_semantics=df.DATAFLOW_PIPELINE_OWNED_COMPLETION,
        release_semantics=df.DATAFLOW_PIPELINE_OWNED_RELEASE,
    )


def make_stream_program(
    *,
    family=df.DATAFLOW_TRANSPORT_STREAMED,
    consumer_access_order=df.DATAFLOW_TRANSPORT_LOGICAL_ORDER,
):
    consumer = stream_consumer if family == df.DATAFLOW_TRANSPORT_STREAMED else stream_discrete_consumer
    return (
        T.dataflow_program(
            task_domain=("task",),
            dynamic_ranges={
                "producer_range": "producer_ranges",
                "consumer_range": "consumer_ranges",
            },
        )
        .map(
            stream_source(Input="Input"),
            name="source",
            task_args=("task",),
            range_axis="producer_range",
        )
        .reshared(
            input="source",
            name="stream",
            output_type=StreamFull,
            physical_output_type=StreamShard,
            output_arity=2,
            transport_contract=df.DataflowResharedTransportRequest(
                family=family,
                logical_output_arity=2,
                physical_output_arity=2,
                consumer_access_order=consumer_access_order,
            ),
        )
        .map(
            consumer(Weight="Weight"),
            name="consumer",
            input="stream",
            task_args=("task",),
            range_axis="consumer_range",
            pipeline_contract=stream_pipeline_request(),
        )
        .finalize(stream_finalize(Output="Output"), input="consumer")
    )


def make_transport_copy_program(
    *,
    family,
    consumer_access_order=df.DATAFLOW_TRANSPORT_LOGICAL_ORDER,
):
    consumer = transport_copy_consumer if family == df.DATAFLOW_TRANSPORT_STREAMED else transport_discrete_copy_consumer
    return (
        T.dataflow_program(
            task_domain=("task",),
            dynamic_ranges={
                "producer_range": "producer_ranges",
                "consumer_range": "consumer_ranges",
            },
        )
        .map(
            stream_source(Input="Input"),
            name="source",
            task_args=("task",),
            range_axis="producer_range",
        )
        .reshared(
            input="source",
            name="transport",
            output_type=StreamFull,
            physical_output_type=StreamShard,
            output_arity=2,
            transport_contract=df.DataflowResharedTransportRequest(
                family=family,
                logical_output_arity=2,
                physical_output_arity=2,
                consumer_access_order=consumer_access_order,
            ),
        )
        .map(
            consumer(),
            name="consumer",
            input="transport",
            task_args=("task",),
            range_axis="consumer_range",
        )
        .finalize(stream_finalize(Output="Output"), input="consumer")
    )


@T.prim_func
def logical_streamed_copy(
    A: T.Tensor((2, 2, 8), "float16", scope="shared"),
    B: T.Tensor((2, 8), "float16", scope="shared"),
):
    with T.Kernel(128):
        for logical_tile in T.Pipelined(8, num_stages=2):
            T.copy(
                A[logical_tile // 2, logical_tile % 2, 0:8],
                B,
                valid_region=A[
                    logical_tile // 2,
                    logical_tile % 2,
                    0:8,
                ],
                synchronization_owner="pipeline",
            )


@T.prim_func
def logical_streamed_copy_with_let_indices(
    A: T.Tensor((2, 2, 8), "float16", scope="shared"),
    B: T.Tensor((2, 8), "float16", scope="shared"),
):
    with T.Kernel(128):
        for logical_tile in T.Pipelined(8, num_stages=2):
            physical_slot = logical_tile // 2
            source_tile = logical_tile % 2
            T.copy(
                A[physical_slot, source_tile, 0:8],
                B,
                valid_region=A[physical_slot, source_tile, 0:8],
                synchronization_owner="pipeline",
            )


def make_request(**updates):
    values = {
        "family": df.DATAFLOW_TRANSPORT_STREAMED,
        "logical_output_arity": 8,
        "physical_output_arity": 4,
        "consumer_access_order": df.DATAFLOW_TRANSPORT_LOGICAL_ORDER,
    }
    values.update(updates)
    return df.DataflowResharedTransportRequest(**values)


@pytest.mark.parametrize(
    ("physical_slot_bytes", "partition_bytes"),
    [
        (4_096, [2_048]),
        (16_384, [8_192]),
        (32_768, [8_192, 8_192]),
    ],
)
def test_streamed_plan_partitions_logical_payload_at_isa_limit(
    physical_slot_bytes,
    partition_bytes,
):
    plan = df.plan_reshared_transport(
        make_request(),
        cluster_size=2,
        physical_slot_bytes=physical_slot_bytes,
        target_capabilities=_HOPPER,
        receive_stage_count=2,
    )

    assert plan.family == df.DATAFLOW_TRANSPORT_STREAMED
    assert plan.logical_tiles_per_physical_slot == 2
    assert plan.logical_tile_bytes == physical_slot_bytes // 2
    assert [item.byte_count for item in plan.payload_partitions] == partition_bytes
    assert plan.transaction_bytes == 8_192
    assert plan.receive_stage_count == 2
    assert plan.credit_count == 2
    assert plan.remote_barrier_count == 0
    assert plan.publish_sync_required
    assert plan.release_sync_required


@pytest.mark.parametrize("cluster_size", [2, 4])
def test_streamed_plan_records_source_rank_locality_and_credit_lifetime(cluster_size):
    request = make_request(
        logical_output_arity=cluster_size * 4,
        physical_output_arity=cluster_size * 2,
    )
    plan = df.plan_reshared_transport(
        request,
        cluster_size=cluster_size,
        physical_slot_bytes=16_384,
        target_capabilities=_HOPPER,
        receive_stage_count=2,
    )

    assert plan.producer_slots_per_rank == 2
    rank_zero = [step for step in plan.steps if step.consumer_rank == 0]
    assert [step.logical_tile_index for step in rank_zero] == list(range(request.logical_output_arity))
    assert {step.locality for step in rank_zero} == {"local", "remote"}
    assert all(step.source_local_slot_index < plan.producer_slots_per_rank for step in rank_zero)
    assert len(plan.credit_lifetimes) == cluster_size * 2
    for lifetime in plan.credit_lifetimes:
        assert lifetime.acquire_steps == lifetime.release_steps
        assert all(step % 2 == lifetime.receive_stage for step in lifetime.acquire_steps)


def test_independent_consumer_order_changes_sequence_without_changing_tile_set():
    logical = df.plan_reshared_transport(
        make_request(),
        cluster_size=2,
        physical_slot_bytes=16_384,
        target_capabilities=_HOPPER,
    )
    independent = df.plan_reshared_transport(
        make_request(consumer_access_order=df.DATAFLOW_TRANSPORT_INDEPENDENT_ORDER),
        cluster_size=2,
        physical_slot_bytes=16_384,
        target_capabilities=_HOPPER,
    )

    logical_rank_one = [step.logical_tile_index for step in logical.steps if step.consumer_rank == 1]
    independent_rank_one = [step.logical_tile_index for step in independent.steps if step.consumer_rank == 1]
    assert logical_rank_one != independent_rank_one
    assert sorted(logical_rank_one) == sorted(independent_rank_one)


def test_auto_policy_selects_streamed_for_multi_stage_cluster_value():
    plan = df.plan_reshared_transport(
        make_request(family=df.DATAFLOW_TRANSPORT_AUTO),
        cluster_size=2,
        physical_slot_bytes=16_384,
        target_capabilities=_HOPPER,
        receive_stage_count=2,
    )

    assert plan.family == df.DATAFLOW_TRANSPORT_STREAMED
    assert plan.decision.selection_reason == "auto_transport_cost_and_resource_selection"
    assert len(plan.decision.candidates) == 3


def test_auto_policy_uses_hbm_without_cluster_transport():
    plan = df.plan_reshared_transport(
        make_request(family=df.DATAFLOW_TRANSPORT_AUTO),
        cluster_size=1,
        physical_slot_bytes=16_384,
        target_capabilities=_PORTABLE,
    )

    assert plan.family == df.DATAFLOW_TRANSPORT_HBM
    assert plan.receive_stage_count == 0
    assert not plan.publish_sync_required


def test_explicit_all_gather_degenerates_to_local_for_single_rank():
    plan = df.plan_reshared_transport(
        make_request(
            family=df.DATAFLOW_TRANSPORT_ALL_GATHER,
            logical_output_arity=2,
            physical_output_arity=2,
        ),
        cluster_size=1,
        physical_slot_bytes=16_384,
        target_capabilities=_PORTABLE,
    )

    assert plan.family == df.DATAFLOW_TRANSPORT_ALL_GATHER
    assert all(step.locality == "local" for step in plan.steps)
    assert all(not step.transport_required for step in plan.steps)
    assert plan.receive_stage_count == 0
    assert plan.remote_barrier_count == 0


def test_explicit_streamed_policy_is_structurally_rejected_without_capability():
    with pytest.raises(
        df.DataflowResharedTransportPlanningError,
        match="cluster DSM transport capability",
    ) as error:
        df.plan_reshared_transport(
            make_request(),
            cluster_size=2,
            physical_slot_bytes=16_384,
            target_capabilities=_PORTABLE,
        )

    decision = error.value.decision
    assert decision.selected_implementation is None
    assert decision.selection_reason == "explicit_transport_family_rejected"
    assert not decision.candidates[0].legal


def test_plan_serialization_and_fingerprint_fail_closed():
    plan = df.plan_reshared_transport(
        make_request(),
        cluster_size=2,
        physical_slot_bytes=32_768,
        target_capabilities=_HOPPER,
    )
    assert df.DataflowResharedTransportPlan.from_dict(plan.to_dict()) == plan

    forged = copy.deepcopy(plan.to_dict())
    forged["steps"][0]["source_rank"] = 1
    with pytest.raises(ValueError, match="typed topology|fingerprint"):
        df.DataflowResharedTransportPlan.from_dict(forged)


def test_streamed_plan_uses_available_consumer_threads_without_device_identity():
    named_target = df.TargetCapabilitySnapshot.for_cuda(
        (9, 0),
        compiler_version=(12, 8),
        device_name="diagnostic-only-name",
    )
    plan = df.plan_reshared_transport(
        make_request(),
        cluster_size=2,
        physical_slot_bytes=16_384,
        target_capabilities=named_target,
        available_threads=64,
    )

    assert plan.capability.registry_key == "cuda.cluster_transport.sm90.v1"
    assert plan.transfer_threads == 64
    assert plan.capability == df.resolve_reshared_transport_capability(_HOPPER)

    with pytest.raises(
        df.DataflowResharedTransportPlanningError,
        match="cooperative warp",
    ):
        df.plan_reshared_transport(
            make_request(),
            cluster_size=2,
            physical_slot_bytes=16_384,
            target_capabilities=_HOPPER,
            available_threads=16,
        )


def test_streamed_plan_binds_logical_copy_to_producer_local_slots():
    plan = df.plan_reshared_transport(
        make_request(),
        cluster_size=2,
        physical_slot_bytes=32,
        target_capabilities=_HOPPER,
        receive_stage_count=2,
    )
    bound = df.bind_reshared_transport_plan(
        logical_streamed_copy,
        plan,
        source_buffer_names=("A",),
    )
    copies = []

    def collect(node):
        if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.tileop.copy":
            copies.append(node)

    tir.stmt_functor.post_order_visit(bound.body, collect)
    assert len(copies) == 1
    parsed = _ffi_api.ParseOperator(copies[0])
    assert "src_block" in copies[0].annotations
    assert df.DATAFLOW_RESHARED_PAYLOAD_PARTITION_BYTES_ATTR in copies[0].annotations
    assert int(parsed.src.shape[0]) == plan.producer_slots_per_rank
    assert df.DATAFLOW_RESHARED_TRANSPORT_PLAN_FINGERPRINT_ATTR in bound.attrs
    assert str(bound.attrs[df.DATAFLOW_RESHARED_TRANSPORT_PLAN_FINGERPRINT_ATTR]) == plan.fingerprint


def test_streamed_binding_traces_let_derived_slot_indices_to_logical_loop():
    plan = df.plan_reshared_transport(
        make_request(),
        cluster_size=2,
        physical_slot_bytes=32,
        target_capabilities=_HOPPER,
        receive_stage_count=2,
    )
    bound = df.bind_reshared_transport_plan(
        logical_streamed_copy_with_let_indices,
        plan,
        source_buffer_names=("A",),
    )
    copies = []

    def collect(node):
        if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.tileop.copy":
            copies.append(node)

    tir.stmt_functor.post_order_visit(bound.body, collect)
    assert len(copies) == 1
    assert "src_block" in copies[0].annotations
    assert df.DATAFLOW_RESHARED_SOURCE_RANK_ATTR in copies[0].annotations


def test_independent_order_binding_derives_rank_and_source_tile_from_access_step():
    plan = df.plan_reshared_transport(
        make_request(consumer_access_order=df.DATAFLOW_TRANSPORT_INDEPENDENT_ORDER),
        cluster_size=2,
        physical_slot_bytes=32,
        target_capabilities=_HOPPER,
    )
    bound = df.bind_reshared_transport_plan(
        logical_streamed_copy,
        plan,
        source_buffer_names=("A",),
    )
    copies = []

    def collect(node):
        if isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.tileop.copy":
            copies.append(node)

    tir.stmt_functor.post_order_visit(bound.body, collect)
    parsed = _ffi_api.ParseOperator(copies[0])
    source_rank = copies[0].annotations[df.DATAFLOW_RESHARED_SOURCE_RANK_ATTR]

    assert plan.lowering_implementation_id == df.DATAFLOW_RESHARED_STREAMED_PUSH_IMPLEMENTATION
    assert plan.remote_barrier_count == plan.receive_stage_count == 2
    assert plan.publish_sync_required is False
    assert plan.release_sync_required is True
    assert "block_rank_in_cluster" in str(source_rank)
    assert isinstance(parsed.src_range[0].min, tir.FloorMod)
    assert isinstance(parsed.src_range[1].min, tir.FloorMod)
    assert "src_block" not in copies[0].annotations
    assert "dst_block" in copies[0].annotations
    assert "block_rank_in_cluster" in str(copies[0].annotations["dst_block"])
    assert df.DATAFLOW_RESHARED_CREDIT_TARGET_RANK_ATTR in copies[0].annotations
    assert int(copies[0].annotations[df.DATAFLOW_RESHARED_RECEIVE_STAGES_ATTR]) == 2


def test_non_moe_independent_stream_compiles_producer_push_transport():
    compiled = df.compile(
        make_stream_program(
            consumer_access_order=df.DATAFLOW_TRANSPORT_INDEPENDENT_ORDER,
        ),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"producer_range": [2], "consumer_range": [2]},
        block_size=1,
        task_extents=(1,),
        include_exit=False,
        target_override=_HOPPER,
        wrapper_name="dataflow_generic_streamed_push_transport",
    )

    assert "tl::tma_store_cluster" in compiled.wrapper_source
    assert "tl::cluster_pull" not in compiled.wrapper_source
    sync_names = {
        instruction.operator_name for instruction in compiled.plan.instructions if instruction.opcode is df.DataflowOpcode.CLUSTER_SYNC
    }
    assert "reshared_stream_publish" not in sync_names
    assert sync_names == {"reshared_stream_release"}
    transport_plan = compiled.plan.reshared_transport_plans[0][1]
    assert transport_plan.lowering_implementation_id == df.DATAFLOW_RESHARED_STREAMED_PUSH_IMPLEMENTATION
    assert transport_plan.remote_barrier_count == 2

    binding = compiled.decision_artifact().to_dict()["lowerings"]["transport"]["bindings"][0]
    assert binding["lowering_implementation_id"] == df.DATAFLOW_RESHARED_STREAMED_PUSH_IMPLEMENTATION


def test_streamed_push_requires_consumer_pipeline_lifetime():
    with pytest.raises(
        df.DataflowPrimFuncLoweringError,
        match="streamed producer-push transport requires a consumer pipeline contract",
    ):
        df.compile(
            make_transport_copy_program(
                family=df.DATAFLOW_TRANSPORT_STREAMED,
                consumer_access_order=df.DATAFLOW_TRANSPORT_INDEPENDENT_ORDER,
            ),
            topology=df.GPUTopology(sm_count=2, cluster_size=2),
            range_lengths={"producer_range": [2], "consumer_range": [2]},
            block_size=1,
            task_extents=(1,),
            include_exit=False,
            target_override=_HOPPER,
            wrapper_name="dataflow_invalid_streamed_push_without_pipeline",
        )


def test_non_moe_streamed_graph_compiles_generic_transport_and_artifact():
    compiled = df.compile(
        make_stream_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"producer_range": [2], "consumer_range": [2]},
        block_size=1,
        task_extents=(1,),
        include_exit=False,
        target_override=_HOPPER,
        wrapper_name="dataflow_generic_streamed_transport",
    )

    assert "tl::cluster_pull" in compiled.wrapper_source
    assert "tl::tma_store_cluster" not in compiled.wrapper_source
    assert not compiled.plan.comms
    assert {
        instruction.operator_name for instruction in compiled.plan.instructions if instruction.opcode is df.DataflowOpcode.CLUSTER_SYNC
    } == {"reshared_stream_publish", "reshared_stream_release"}
    for queue in compiled.plan.queues.values():
        assert queue[-1].operator_name == "reshared_stream_release"
    transport_plan = compiled.plan.reshared_transport_plans[0][1]
    assert transport_plan.transfer_threads == 128
    assert transport_plan.remote_barrier_count == 0

    artifact = compiled.decision_artifact().to_dict()
    binding = artifact["lowerings"]["transport"]["bindings"][0]
    assert binding["plan_fingerprint"] == transport_plan.fingerprint
    assert binding["primfunc_plan_fingerprint"] == transport_plan.fingerprint
    assert binding["lowering_implementation_id"] == df.DATAFLOW_RESHARED_STREAMED_PULL_IMPLEMENTATION

    forged = copy.deepcopy(artifact)
    forged.pop("fingerprint")
    forged["lowerings"]["transport"]["bindings"][0]["transaction_bytes"] += 16
    with pytest.raises(ValueError, match="transport binding"):
        df.DataflowDecisionArtifact.from_payload(forged)


def test_streamed_schedule_replay_preserves_target_and_transport_plan():
    program = make_stream_program()
    ranges = {"producer_range": [2], "consumer_range": [2]}
    plan = df.schedule(
        program,
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths=ranges,
        block_size=1,
        task_extents=(1,),
        include_exit=False,
        target_capabilities=_HOPPER,
    )
    replay = df.record_schedule_replay(
        program,
        plan,
        range_lengths=ranges,
        include_exit=False,
    )
    restored = df.DataflowScheduleReplay.from_dict(replay.to_dict())
    replayed = df.replay_schedule(program, restored)

    assert replayed.reshared_transport_plans == plan.reshared_transport_plans
    assert replayed.target_capabilities == _HOPPER

    forged = copy.deepcopy(replay.to_dict())
    forged["target_capabilities"]["supports_cluster_launch"] = False
    with pytest.raises(ValueError, match="fingerprint"):
        df.DataflowScheduleReplay.from_dict(forged)


def test_typed_slot_and_temporary_limits_are_enforced():
    with pytest.raises(ValueError, match="slot_bytes"):
        df.plan_reshared_transport(
            make_request(slot_bytes=1),
            cluster_size=2,
            physical_slot_bytes=16_384,
            target_capabilities=_HOPPER,
        )

    with pytest.raises(
        df.DataflowResharedTransportPlanningError,
        match="receive stages exceed",
    ):
        df.plan_reshared_transport(
            make_request(max_temporary_bytes=8_191),
            cluster_size=2,
            physical_slot_bytes=16_384,
            target_capabilities=_HOPPER,
        )


def test_non_moe_hbm_all_gather_and_streamed_are_numerically_equivalent():
    if os.environ.get("TILELANG_DATAFLOW_RUN_RESHARED_TRANSPORT_CUDA") != "1":
        pytest.skip("set TILELANG_DATAFLOW_RUN_RESHARED_TRANSPORT_CUDA=1 to run transport equivalence")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA unavailable")

    input_tensor = (
        torch.arange(2 * 64 * 64, device="cuda", dtype=torch.float32)
        .remainder(17)
        .sub(8)
        .mul_(0.03125)
        .to(torch.float16)
        .reshape(2, 64, 64)
    )
    expected = input_tensor.float().sum(dim=0)
    outputs = {}
    for family in (
        df.DATAFLOW_TRANSPORT_HBM,
        df.DATAFLOW_TRANSPORT_ALL_GATHER,
        df.DATAFLOW_TRANSPORT_STREAMED,
    ):
        compiled = df.compile(
            make_transport_copy_program(family=family),
            topology=df.GPUTopology(sm_count=2, cluster_size=2),
            range_lengths={"producer_range": [2], "consumer_range": [2]},
            block_size=1,
            task_extents=(1,),
            include_exit=False,
            target_override=_HOPPER,
            wrapper_name=f"dataflow_generic_transport_{family}",
        )
        output = torch.zeros((64, 64), device="cuda", dtype=torch.float32)
        compiled(Input=input_tensor, Output=output)
        torch.cuda.synchronize()
        outputs[family] = output
        torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)

    torch.testing.assert_close(
        outputs[df.DATAFLOW_TRANSPORT_HBM],
        outputs[df.DATAFLOW_TRANSPORT_ALL_GATHER],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        outputs[df.DATAFLOW_TRANSPORT_HBM],
        outputs[df.DATAFLOW_TRANSPORT_STREAMED],
        rtol=0.0,
        atol=0.0,
    )
