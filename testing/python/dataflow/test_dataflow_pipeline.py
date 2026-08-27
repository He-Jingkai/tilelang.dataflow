from __future__ import annotations

from dataclasses import replace

import pytest

import tilelang as tl
import tilelang.language as T
import tilelang.dataflow as df
from tilelang import tvm


@T.prim_func
def dual_input_gemm(
    left: T.Tensor((16, 32), T.float16),
    right: T.Tensor((32, 16), T.float16),
    output: T.Tensor((16, 16), T.float32),
):
    with T.Kernel(1, threads=32):
        left_shared = T.alloc_shared((16, 16), T.float16)
        right_shared = T.alloc_shared((16, 16), T.float16)
        accumulator = T.alloc_fragment((16, 16), T.float32)
        T.clear(accumulator)
        for tile in T.Pipelined(2, num_stages=2):
            T.copy(left[:, tile * 16 : (tile + 1) * 16], left_shared)
            T.copy(right[tile * 16 : (tile + 1) * 16, :], right_shared)
            T.gemm(left_shared, right_shared, accumulator)
        T.copy(accumulator, output)


@T.prim_func
def asymmetric_dual_input_gemm(
    left: T.Tensor((16, 48), T.float16),
    right: T.Tensor((48, 16), T.float16),
    output: T.Tensor((16, 16), T.float32),
):
    with T.Kernel(1, threads=32):
        left_shared = T.alloc_shared((16, 16), T.float16)
        right_shared = T.alloc_shared((16, 16), T.float16)
        accumulator = T.alloc_fragment((16, 16), T.float32)
        T.clear(accumulator)
        for tile in T.Pipelined(3, num_stages=3):
            T.copy(
                left[:, tile * 16 : (tile + 1) * 16],
                left_shared,
                valid_region=left,
                synchronization_owner="pipeline",
            )
            T.copy(
                right[tile * 16 : (tile + 1) * 16, :],
                right_shared,
                valid_region=right,
                synchronization_owner="pipeline",
            )
            T.gemm(left_shared, right_shared, accumulator)
        T.copy(accumulator, output)


@T.prim_func
def miswired_gemm(
    left: T.Tensor((16, 32), T.float16),
    right: T.Tensor((32, 16), T.float16),
    output: T.Tensor((16, 16), T.float32),
):
    with T.Kernel(1, threads=32):
        left_shared = T.alloc_shared((16, 16), T.float16)
        right_shared = T.alloc_shared((16, 16), T.float16)
        accumulator = T.alloc_fragment((16, 16), T.float32)
        T.clear(accumulator)
        for tile in T.Pipelined(2, num_stages=2):
            T.copy(left[:, tile * 16 : (tile + 1) * 16], left_shared)
            T.copy(right[tile * 16 : (tile + 1) * 16, :], right_shared)
            T.gemm(right_shared, right_shared, accumulator)
        T.copy(accumulator, output)


@T.prim_func
def aliased_copy_destinations(
    left: T.Tensor((16, 32), T.float16),
    right: T.Tensor((32, 16), T.float16),
    output: T.Tensor((16, 16), T.float32),
):
    with T.Kernel(1, threads=32):
        shared = T.alloc_shared((16, 16), T.float16)
        accumulator = T.alloc_fragment((16, 16), T.float32)
        T.clear(accumulator)
        for tile in T.Pipelined(2, num_stages=2):
            T.copy(left[:, tile * 16 : (tile + 1) * 16], shared)
            T.copy(right[tile * 16 : (tile + 1) * 16, :], shared)
            T.gemm(shared, shared, accumulator)
        T.copy(accumulator, output)


@T.prim_func
def stage_one_dual_input_gemm(
    left: T.Tensor((16, 32), T.float16),
    right: T.Tensor((32, 16), T.float16),
    output: T.Tensor((16, 16), T.float32),
):
    with T.Kernel(1, threads=32):
        left_shared = T.alloc_shared((16, 16), T.float16)
        right_shared = T.alloc_shared((16, 16), T.float16)
        accumulator = T.alloc_fragment((16, 16), T.float32)
        T.clear(accumulator)
        for tile in T.Pipelined(2, num_stages=1):
            T.copy(
                left[:, tile * 16 : (tile + 1) * 16],
                left_shared,
                valid_region=left,
                synchronization_owner="pipeline",
            )
            T.copy(
                right[tile * 16 : (tile + 1) * 16, :],
                right_shared,
                valid_region=right,
                synchronization_owner="pipeline",
            )
            T.gemm(left_shared, right_shared, accumulator)
        T.copy(accumulator, output)


@T.prim_func
def resident_input_gemm(
    left: T.Tensor((16, 16), T.float16),
    right: T.Tensor((32, 16), T.float16),
    output: T.Tensor((16, 16), T.float32),
):
    with T.Kernel(1, threads=32):
        left_shared = T.alloc_shared((16, 16), T.float16)
        right_shared = T.alloc_shared((16, 16), T.float16)
        accumulator = T.alloc_fragment((16, 16), T.float32)
        T.copy(left, left_shared)
        T.clear(accumulator)
        for tile in T.Pipelined(2, num_stages=2):
            T.copy(
                left_shared,
                left_shared,
                valid_region=left_shared,
                synchronization_owner="pipeline",
            )
            T.copy(
                right[tile * 16 : (tile + 1) * 16, :],
                right_shared,
                valid_region=right,
                synchronization_owner="pipeline",
            )
            T.gemm(left_shared, right_shared, accumulator)
        T.copy(accumulator, output)


@T.prim_func
def dispatch_alternative_gemm(
    left: T.Tensor((16, 32), T.float16),
    alternate: T.Tensor((16, 32), T.float16),
    right: T.Tensor((32, 16), T.float16),
    output: T.Tensor((16, 16), T.float32),
    choose_alternate: T.int32,
):
    with T.Kernel(1, threads=32):
        left_shared = T.alloc_shared((16, 16), T.float16)
        right_shared = T.alloc_shared((16, 16), T.float16)
        accumulator = T.alloc_fragment((16, 16), T.float32)
        T.clear(accumulator)
        for tile in T.Pipelined(2, num_stages=2):
            if choose_alternate != 0:
                T.copy(
                    alternate[:, tile * 16 : (tile + 1) * 16],
                    left_shared,
                    valid_region=alternate,
                    synchronization_owner="pipeline",
                )
            else:
                T.copy(
                    left[:, tile * 16 : (tile + 1) * 16],
                    left_shared,
                    valid_region=left,
                    synchronization_owner="pipeline",
                )
            T.copy(
                right[tile * 16 : (tile + 1) * 16, :],
                right_shared,
                valid_region=right,
                synchronization_owner="pipeline",
            )
            T.gemm(left_shared, right_shared, accumulator)
        T.copy(accumulator, output)


@T.prim_func
def multicast_without_permission(
    left: T.Tensor((16, 32), T.float16),
    right: T.Tensor((32, 16), T.float16),
    output: T.Tensor((16, 16), T.float32),
):
    with T.Kernel(1, threads=32):
        left_shared = T.alloc_shared((16, 16), T.float16)
        right_shared = T.alloc_shared((16, 16), T.float16)
        accumulator = T.alloc_fragment((16, 16), T.float32)
        T.clear(accumulator)
        for tile in T.Pipelined(2, num_stages=2):
            T.copy(
                left[:, tile * 16 : (tile + 1) * 16],
                left_shared,
                valid_region=left,
                synchronization_owner="pipeline",
                annotations={"cluster_mask": 3},
            )
            T.copy(
                right[tile * 16 : (tile + 1) * 16, :],
                right_shared,
                valid_region=right,
                synchronization_owner="pipeline",
            )
            T.gemm(left_shared, right_shared, accumulator)
        T.copy(accumulator, output)


@T.dataflow_intermediate
class PipelineOutput:
    value: T.Tensor((16, 16), T.float32)


@T.dataflow_intermediate
class PipelineDone:
    value: T.int32


@T.dataflow.map(range=("begin", "end"), threads=32)
def additive_transform(
    left: T.Tensor((16, 32), T.float16),
    right: T.Tensor((32, 16), T.float16),
) -> PipelineOutput:
    left_shared = T.alloc_shared((16, 16), T.float16)
    right_shared = T.alloc_shared((16, 16), T.float16)
    accumulator = T.alloc_fragment((16, 16), T.float32)
    T.clear(accumulator)
    for tile in T.Pipelined(2, num_stages=2):
        T.copy(
            left[:, tile * 16 : (tile + 1) * 16],
            left_shared,
            valid_region=left,
            synchronization_owner="pipeline",
        )
        T.copy(
            right[tile * 16 : (tile + 1) * 16, :],
            right_shared,
            valid_region=right,
            synchronization_owner="pipeline",
        )
        T.gemm(left_shared, right_shared, accumulator)
        T.gemm(left_shared, right_shared, accumulator)
    return PipelineOutput(value=accumulator)


@T.dataflow.map(range=("begin", "end"))
def consume_transform(parts: list[PipelineOutput]) -> PipelineDone:
    value = T.int32(0)
    for _ in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += T.int32(0)
    return PipelineDone(value=value)


def make_request(
    *,
    gemm_count: int = 1,
    diagnostic_name: str | None = None,
    **overrides,
):
    gemms = tuple(
        df.DataflowPipelineGemm(
            input_buffer_indices=(0, 1),
            accumulator_index=0,
            accumulator_dependency=True,
        )
        for _ in range(gemm_count)
    )
    consumers = tuple(range(gemm_count))
    values = {
        "transfers": (
            df.DataflowPipelineTransfer(
                destination_buffer_index=0,
                logical_extent=(16, 16),
                bytes_per_stage=512,
                async_permitted=True,
                multicast_permitted=True,
                eviction_hint=df.DATAFLOW_PIPELINE_EVICT_LAST,
            ),
            df.DataflowPipelineTransfer(
                destination_buffer_index=1,
                logical_extent=(16, 16),
                bytes_per_stage=512,
                async_permitted=True,
                eviction_hint=df.DATAFLOW_PIPELINE_EVICT_FIRST,
            ),
        ),
        "gemms": gemms,
        "buffer_lifetimes": (
            df.DataflowPipelineBufferLifetime(
                buffer_index=0,
                producer_transfer_index=0,
                consumer_gemm_indices=consumers,
                release_after_gemm_index=consumers[-1],
            ),
            df.DataflowPipelineBufferLifetime(
                buffer_index=1,
                producer_transfer_index=1,
                consumer_gemm_indices=consumers,
                release_after_gemm_index=consumers[-1],
            ),
        ),
        "stage_budget": None,
        "max_outstanding": 2,
        "producer_threads": 32,
        "consumer_threads": 32,
        "diagnostic_name": diagnostic_name,
    }
    values.update(overrides)
    return df.DataflowPipelineRequest(**values)


def additive_program(request):
    return (
        T.dataflow_program(
            task_domain=("task",),
            dynamic_ranges={"tile": "tiles", "sink": "sinks"},
        )
        .map(
            additive_transform(left="left", right="right"),
            name="additive_transform",
            range_axis="tile",
            pipeline_contract=request,
        )
        .reshared(
            input="additive_transform",
            output_type=PipelineOutput,
            physical_output_type=PipelineOutput,
            output_arity=1,
            transport_contract=df.DataflowResharedTransportRequest.from_legacy_policy(
                "hbm_all_gather",
                logical_output_arity=1,
                physical_output_arity=1,
            ),
        )
        .map(
            consume_transform(),
            input="reshared",
            range_axis="sink",
        )
    )


def test_multi_transfer_plan_records_requirements_resources_and_additive_groups():
    request = make_request(gemm_count=2, stage_budget=3)
    plan = df.plan_pipeline_dataflow(request)

    assert plan.selected_stages == 3
    assert plan.selected_max_outstanding == 2
    assert not plan.used_fallback
    assert plan.implementation_requirements.mode == df.DATAFLOW_PIPELINE_MODE_PIPELINED
    assert plan.implementation_requirements.async_transfer_indices == (0, 1)
    assert plan.implementation_requirements.producer_partitions == (None, None)
    assert plan.implementation_requirements.additive_gemm_groups == ((0, 1),)
    assert plan.implementation_requirements.buffer_versions == (3, 3)
    assert plan.implementation_requirements.release_after_gemm_indices == (1, 1)
    assert plan.resources.shared_memory_bytes == 3 * 1024
    assert plan.resources.barrier_count == 4
    assert plan.resources.transaction_bytes == 1024


def test_resource_limit_selects_a_semantically_equivalent_synchronous_fallback():
    request = make_request(max_shared_memory_bytes=1024)
    plan = df.plan_pipeline_dataflow(request, auto_stage_budget=3)

    assert plan.selected_stages == 1
    assert plan.used_fallback
    assert plan.fallback_reasons == ("shared_memory_budget",)
    assert plan.implementation_requirements.mode == df.DATAFLOW_PIPELINE_MODE_SYNCHRONOUS
    assert plan.implementation_requirements.async_transfer_indices == ()
    assert plan.implementation_requirements.additive_gemm_groups == ((0,),)
    assert plan.implementation_requirements.release_after_gemm_indices == (0, 0)
    assert plan.resources.shared_memory_bytes == 1024
    assert plan.request_fingerprint == request.fingerprint


def test_resource_limit_selects_budgeted_per_transfer_buffer_versions():
    request = make_request(
        stage_budget=3,
        max_outstanding=3,
        max_shared_memory_bytes=2560,
    )
    plan = df.plan_pipeline_dataflow(request)

    assert plan.selected_stages == 3
    assert not plan.used_fallback
    assert plan.implementation_requirements.buffer_versions == (3, 2)
    assert plan.resources.shared_memory_bytes == 2560


def test_non_dataflow_asymmetric_pipeline_materializes_typed_rings():
    request = make_request(
        stage_budget=3,
        max_outstanding=3,
        max_shared_memory_bytes=2560,
        producer_threads=64,
        consumer_threads=32,
    )
    request = replace(
        request,
        transfers=tuple(replace(transfer, producer_partition=index) for index, transfer in enumerate(request.transfers)),
    )
    bound, plan = df.bind_pipeline_dataflow_plan(
        asymmetric_dual_input_gemm,
        request,
    )

    assert plan.selected_stages == 3
    assert plan.implementation_requirements.buffer_versions == (3, 2)
    target = tvm.target.Target("cuda -arch=sm_90a")
    with target:
        source = tl.lower(bound, target=target).kernel_source
    assert "uint64_t mbarrier_mem[8];" in source
    assert source.count("arrive_and_expect_tx(1024)") == 1
    assert "mbarrier[((int)threadIdx.x)].arrive_and_expect_tx(1024);" in source


def test_pipeline_request_and_plan_round_trip_fail_closed():
    request = make_request(gemm_count=2, diagnostic_name="first-source")
    renamed = make_request(gemm_count=2, diagnostic_name="renamed-source")
    restored_request = df.dataflow_operation_request_from_dict(request.to_dict())
    plan = df.plan_pipeline_dataflow(request)

    assert restored_request == replace(request, diagnostic_name=None)
    assert request.fingerprint == renamed.fingerprint
    assert df.DataflowPipelinePlan.from_dict(plan.to_dict()) == plan

    forged = plan.to_dict()
    forged["selected_stages"] = 1
    with pytest.raises(ValueError):
        df.DataflowPipelinePlan.from_dict(forged)

    unknown = request.to_dict()
    unknown["workload_selector"] = "forbidden"
    with pytest.raises(ValueError, match="unknown fields"):
        df.dataflow_operation_request_from_dict(unknown)

    invalid_lifetime = request.to_dict()
    invalid_lifetime["buffer_lifetimes"] = [dict(item) for item in invalid_lifetime["buffer_lifetimes"]]
    invalid_lifetime["buffer_lifetimes"][0]["consumer_gemm_indices"] = [0]
    invalid_lifetime["buffer_lifetimes"][0]["release_after_gemm_index"] = 0
    with pytest.raises(ValueError, match="consumers do not match"):
        df.dataflow_operation_request_from_dict(invalid_lifetime)


def test_partitioned_producers_round_trip_and_bind_to_transfer_tile_ops():
    request = make_request(producer_threads=64)
    request = replace(
        request,
        transfers=(
            replace(request.transfers[0], producer_partition=0),
            replace(request.transfers[1], producer_partition=1),
        ),
    )

    bound, plan = df.bind_pipeline_dataflow_plan(dual_input_gemm, request)

    assert plan.implementation_requirements.producer_partitions == (0, 1)
    assert df.DataflowPipelinePlan.from_dict(plan.to_dict()) == plan
    observed = []

    def collect_partition(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.tileop.copy":
            partition = node.annotations.get(df.DATAFLOW_PIPELINE_PRODUCER_PARTITION_ATTR)
            if partition is not None:
                observed.append(int(partition))

    tvm.tir.stmt_functor.post_order_visit(bound.body, collect_partition)
    assert observed == [0, 1]
    producer_thread_budgets = []

    def collect_producer_threads(node):
        if isinstance(node, tvm.tir.For):
            value = node.annotations.get(df.DATAFLOW_PIPELINE_PRODUCER_THREADS_ATTR)
            if value is not None:
                producer_thread_budgets.append(int(value))

    tvm.tir.stmt_functor.post_order_visit(
        bound.body,
        collect_producer_threads,
    )
    assert producer_thread_budgets == [64]

    with pytest.raises(ValueError, match="canonical contiguous"):
        replace(
            request,
            transfers=(
                replace(request.transfers[0], producer_partition=1),
                replace(request.transfers[1], producer_partition=1),
            ),
        )
    with pytest.raises(ValueError, match="must all declare"):
        replace(
            request,
            transfers=(
                replace(request.transfers[0], producer_partition=0),
                replace(request.transfers[1], producer_partition=None),
            ),
        )


def test_non_dataflow_primfunc_binds_the_same_dual_input_contract():
    bound, plan = df.bind_pipeline_dataflow_plan(dual_input_gemm, make_request())

    assert plan.selected_stages == 2
    assert str(bound.attrs[df.DATAFLOW_PIPELINE_PLAN_FINGERPRINT_ATTR]) == plan.fingerprint
    assert int(bound.attrs[df.DATAFLOW_PIPELINE_PLAN_SCHEMA_ATTR]) == df.DATAFLOW_PIPELINE_PLAN_SCHEMA_VERSION
    observed_versions = []

    def collect_versions(node):
        if (
            isinstance(node, tvm.tir.Call)
            and isinstance(node.op, tvm.ir.Op)
            and node.op.name == "tl.tileop.copy"
            and df.DATAFLOW_PIPELINE_BUFFER_VERSIONS_ATTR in node.annotations
        ):
            observed_versions.append(int(node.annotations[df.DATAFLOW_PIPELINE_BUFFER_VERSIONS_ATTR]))

    tvm.tir.stmt_functor.post_order_visit(bound.body, collect_versions)
    assert observed_versions == [2, 2]

    with pytest.raises(df.DataflowPipelinePlanningError, match="GEMM count"):
        df.plan_primfunc_pipeline_dataflow(
            dual_input_gemm,
            make_request(gemm_count=2),
        )
    with pytest.raises(df.DataflowPipelinePlanningError, match="buffer edges"):
        df.plan_primfunc_pipeline_dataflow(miswired_gemm, make_request())
    with pytest.raises(df.DataflowPipelinePlanningError, match="aliases"):
        df.plan_primfunc_pipeline_dataflow(
            aliased_copy_destinations,
            make_request(),
        )


def test_resident_pipeline_buffer_is_typed_versioned_and_not_materialized():
    request = make_request()
    request = replace(
        request,
        transfers=(
            replace(
                request.transfers[0],
                materialization=df.DATAFLOW_PIPELINE_MATERIALIZE_RESIDENT,
                async_permitted=False,
                multicast_permitted=False,
            ),
            request.transfers[1],
        ),
        buffer_lifetimes=(
            replace(request.buffer_lifetimes[0], allow_multiversion=False),
            request.buffer_lifetimes[1],
        ),
    )

    bound, plan = df.bind_pipeline_dataflow_plan(resident_input_gemm, request)

    assert df.DataflowPipelinePlan.from_dict(plan.to_dict()) == plan
    assert plan.implementation_requirements.transfer_materializations == (
        df.DATAFLOW_PIPELINE_MATERIALIZE_RESIDENT,
        df.DATAFLOW_PIPELINE_MATERIALIZE_COPY,
    )
    assert plan.implementation_requirements.buffer_versions == (1, 2)
    assert plan.resources.shared_memory_bytes == 1024
    assert plan.resources.transaction_bytes == 512
    observed = []

    def collect_materialization(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.tileop.copy":
            materialization = node.annotations.get(df.DATAFLOW_PIPELINE_MATERIALIZATION_ATTR)
            if materialization is not None:
                observed.append(str(materialization.value))

    tvm.tir.stmt_functor.post_order_visit(bound.body, collect_materialization)
    assert observed == [
        df.DATAFLOW_PIPELINE_MATERIALIZE_RESIDENT,
        df.DATAFLOW_PIPELINE_MATERIALIZE_COPY,
    ]

    with pytest.raises(
        df.DataflowPipelinePlanningError,
        match="source and destination regions differ",
    ):
        df.bind_pipeline_dataflow_plan(dual_input_gemm, request)


def test_dispatch_alternatives_share_one_logical_transfer():
    bound, plan = df.bind_pipeline_dataflow_plan(
        dispatch_alternative_gemm,
        make_request(),
    )

    assert plan.implementation_requirements.transfer_count == 2
    assert str(bound.attrs[df.DATAFLOW_PIPELINE_PLAN_FINGERPRINT_ATTR]) == plan.fingerprint

    consumed_modes = []

    def collect_copy_mode(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.tileop.copy":
            mode = node.annotations.get("tl.transfer_pipeline_sync_consumed")
            if mode is not None:
                consumed_modes.append(int(mode))

    tvm.tir.stmt_functor.post_order_visit(bound.body, collect_copy_mode)
    assert consumed_modes == [1, 1, 1]


def test_mixed_transfer_plan_binds_sync_mode_to_all_dispatch_alternatives():
    request = make_request()
    request = replace(
        request,
        transfers=(
            replace(request.transfers[0], async_permitted=False),
            request.transfers[1],
        ),
    )
    bound, plan = df.bind_pipeline_dataflow_plan(
        dispatch_alternative_gemm,
        request,
    )

    assert plan.implementation_requirements.mode == (df.DATAFLOW_PIPELINE_MODE_PIPELINED)
    assert plan.implementation_requirements.async_transfer_indices == (1,)
    consumed_modes = []

    def collect_copy_mode(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.tileop.copy":
            mode = node.annotations.get("tl.transfer_pipeline_sync_consumed")
            if mode is not None:
                consumed_modes.append(int(mode))

    tvm.tir.stmt_functor.post_order_visit(bound.body, collect_copy_mode)
    assert consumed_modes == [2, 2, 1]


def test_illegal_multicast_fails_closed_at_primfunc_binding():
    request = make_request()
    request = replace(
        request,
        transfers=(
            replace(request.transfers[0], multicast_permitted=False),
            request.transfers[1],
        ),
    )

    with pytest.raises(
        df.DataflowPipelinePlanningError,
        match="multicast without permission",
    ):
        df.bind_pipeline_dataflow_plan(multicast_without_permission, request)


def test_resource_plan_forces_physical_synchronous_pipeline():
    request = make_request(
        stage_budget=3,
        max_shared_memory_bytes=1024,
    )
    bound, plan = df.bind_pipeline_dataflow_plan(
        stage_one_dual_input_gemm,
        request,
    )
    assert plan.implementation_requirements.mode == (df.DATAFLOW_PIPELINE_MODE_SYNCHRONOUS)
    assert plan.fallback_reasons == ("shared_memory_budget",)

    target = tvm.target.Target("cuda -arch=sm_90a")
    mod = tvm.IRModule.from_expr(bound.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)
    with target:
        mod = tl.transform.MaterializeLogicalGemm()(mod)
        mod = tl.transform.LayoutReducer()(mod)
        mod = tl.transform.ProducerConsumerWarpSpecialized()(mod)
        assert not mod["main"].attrs.get("tl_tiled_ws_applied")
        mod = tl.transform.PipelinePlanning()(mod)

    decisions = list(mod["main"].attrs["tl.pipeline_lowering_decisions"])
    assert len(decisions) == 1
    assert str(getattr(decisions[0]["selected_implementation"], "value", "")) == "synchronous"
    assert int(decisions[0]["fallback"]) == 1

    consumed_modes = []

    def collect_copy(node):
        if not isinstance(node, tvm.tir.Call):
            return
        mode = node.annotations.get("tl.transfer_pipeline_sync_consumed")
        if mode is not None:
            consumed_modes.append(int(mode))

    tvm.tir.stmt_functor.post_order_visit(
        mod["main"].body,
        collect_copy,
    )
    assert consumed_modes == [2, 2]


def test_non_moe_dataflow_map_records_two_transfers_and_additive_gemms():
    request = make_request(gemm_count=2, stage_budget=2)
    compiled = df.compile(
        additive_program(request),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"tile": [16], "sink": [16]},
        block_size=16,
        task_extents=(1,),
        include_exit=False,
        mode="inspect",
        inspection_stage="ir",
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (8, 0),
            compiler_version=(12, 8),
            max_dynamic_shared_memory=163_840,
        ),
    )

    handler = compiled.primfunc_lowering.handlers[0]
    assert handler.pipeline_dataflow_plan is not None
    assert handler.pipeline_dataflow_plan.request_fingerprint == request.fingerprint
    assert str(handler.prim_func.attrs[df.DATAFLOW_PIPELINE_PLAN_FINGERPRINT_ATTR]) == handler.pipeline_dataflow_plan.fingerprint
    record = next(
        item
        for item in compiled.decision_artifact().to_dict()["operation_contracts"]["records"]
        if item["request"]["kind"] == "pipeline_dataflow"
    )
    assert record["selection_state"] == "selected"
    assert record["decisions"][0]["selected_implementation"] == (df.DATAFLOW_PIPELINE_PLAN_IMPLEMENTATION)
    assert record["lowering_plans"][0]["implementation_requirements"]["additive_gemm_groups"] == [[0, 1]]


def test_non_moe_dataflow_pipeline_records_physical_sm80_bindings():
    request = make_request(gemm_count=2, stage_budget=2)
    compiled = df.compile(
        additive_program(request),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"tile": [16], "sink": [16]},
        block_size=16,
        task_extents=(1,),
        include_exit=False,
        mode="inspect",
        inspection_stage="cuda",
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (8, 0),
            compiler_version=(12, 8),
            max_dynamic_shared_memory=163_840,
        ),
    )

    decisions = compiled.decision_artifact().to_dict()
    binding = decisions["lowerings"]["pipeline"]["dataflow_bindings"][0]
    assert binding["schema_version"] == (df.DATAFLOW_PIPELINE_BINDING_SCHEMA_VERSION)
    assert binding["selected_stages"] == 2
    assert binding["additive_gemm_groups"] == [[0, 1]]
    assert [transfer["producer_partition"] for transfer in binding["transfers"]] == [None, None]
    assert binding["pipeline"]["selected_implementation"] == ("software_pipeline")
    assert len(binding["transfers"]) == 2
    assert all(len(transfer["alternatives"]) == 1 for transfer in binding["transfers"])
    assert len(binding["gemms"]) == 2
    assert {gemm["selected_implementation"] for gemm in binding["gemms"]} == {"cuda.mma.sync"}
