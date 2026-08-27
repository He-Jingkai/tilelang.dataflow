from __future__ import annotations

import os

import pytest
import torch

import tilelang.language as T
import tilelang.dataflow as df
import tvm
from tilelang.engine.phase import slice_handoff_layout_for_single_stage
from tilelang.layout import Fragment


_HOPPER = df.TargetCapabilitySnapshot.for_cuda(
    (9, 0),
    compiler_version=(12, 8),
    max_dynamic_shared_memory=232_448,
)


def test_single_stage_handoff_layout_slices_replicated_fragment_inputs():
    layout = Fragment(
        [2, 4],
        forward_thread_fn=lambda stage, index, rep: index * 2 + rep,
        forward_index_fn=lambda stage, index: tvm.runtime.convert([stage, index]),
        replicate=2,
    )

    sliced = slice_handoff_layout_for_single_stage(layout, [4])
    mapped = sliced.map_forward_index([tvm.tir.IntImm("int32", 3)])

    assert list(sliced.get_input_shape()) == [4]
    assert len(mapped) == 1
    assert int(tvm.arith.Analyzer().simplify(mapped[0])) == 3


@T.dataflow_intermediate
class HandoffShard:
    value: T.Tensor((64, 64), T.float16)


@T.dataflow_intermediate
class HandoffFull:
    value: T.Tensor((2, 64, 64), T.float16)


@T.dataflow_intermediate
class HandoffResult:
    value: T.Tensor((64, 64), T.float32)


@T.dataflow.map(
    range=("begin", "end"),
    threads=256,
    physical_contract=df.DataflowOperatorPhysicalContract(
        output_slot=df.DATAFLOW_OUTPUT_SLOT_DIRECT,
    ),
)
def handoff_stage_a(
    task: T.int32,
    Left: T.Tensor((4, 64, 128), T.float16),
    Right: T.Tensor((4, 128, 64), T.float16),
) -> HandoffShard:
    left_shared = T.alloc_shared((64, 64), T.float16)
    right_shared = T.alloc_shared((64, 64), T.float16)
    accumulator = T.alloc_fragment((64, 64), T.float32)
    result = T.alloc_shared((64, 64), T.float16)
    T.clear(accumulator)
    for tile in T.Pipelined(2, num_stages=2):
        T.copy(
            Left[task, :, tile * 64 : (tile + 1) * 64],
            left_shared,
            valid_region=Left[task, :, tile * 64 : (tile + 1) * 64],
            synchronization_owner="pipeline",
        )
        T.copy(
            Right[task, tile * 64 : (tile + 1) * 64, :],
            right_shared,
            valid_region=Right[task, tile * 64 : (tile + 1) * 64, :],
            synchronization_owner="pipeline",
        )
        T.gemm(left_shared, right_shared, accumulator)
    T.copy(accumulator, result)
    return HandoffShard(value=result)


@T.dataflow.map(
    range=("begin", "end"),
    threads=256,
    physical_contract=df.DataflowOperatorPhysicalContract(
        input_slots=df.DATAFLOW_INPUT_SLOTS_CONTIGUOUS,
    ),
)
def handoff_stage_b(
    parts: list[HandoffShard],
    task: T.int32,
    Weight: T.Tensor((2, 64, 64), T.float16),
    Output: T.Tensor((4, 64, 64), T.float32),
) -> HandoffResult:
    left_shared = T.alloc_shared((64, 64), T.float16)
    right_shared = T.alloc_shared((64, 64), T.float16)
    accumulator = T.alloc_fragment((64, 64), T.float32)
    T.clear(accumulator)
    for tile in T.Pipelined(2, num_stages=2):
        T.copy(
            parts[tile].value,
            left_shared,
            valid_region=parts[tile].value,
            synchronization_owner="pipeline",
        )
        T.copy(
            Weight[tile, :, :],
            right_shared,
            valid_region=Weight[tile, :, :],
            synchronization_owner="pipeline",
        )
        T.gemm(left_shared, right_shared, accumulator)
    T.copy(accumulator, Output[task, :, :])
    return HandoffResult(value=accumulator)


def make_pipeline_request(*, first_async: bool = True):
    return df.DataflowPipelineRequest(
        transfers=(
            df.DataflowPipelineTransfer(
                destination_buffer_index=0,
                logical_extent=(64, 64),
                bytes_per_stage=8_192,
                async_permitted=first_async,
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


def handoff_program(*, enabled: bool, lookahead_distance: int = 1):
    program = (
        T.dataflow_program(
            task_domain=("task",),
            dynamic_ranges={
                "a_range": "a_ranges",
                "b_range": "b_ranges",
            },
        )
        .map(
            handoff_stage_a(Left="Left", Right="Right"),
            name="a",
            task_args=("task",),
            range_axis="a_range",
            range_tile=1,
            pipeline_contract=make_pipeline_request(),
        )
        .reshared(
            input="a",
            name="exchange",
            output_type=HandoffFull,
            physical_output_type=HandoffShard,
            output_arity=2,
            policy="hbm_all_gather",
        )
    )
    map_kwargs = {
        "name": "b",
        "input": "exchange",
        "task_args": ("task",),
        "range_axis": "b_range",
        "pipeline_contract": make_pipeline_request(first_async=False),
    }
    if enabled:
        map_kwargs["handoff_contract"] = df.DataflowCrossHandlerHandoffRequest(
            consumer_stage_id=0,
            lookahead_distance=lookahead_distance,
            buffer_stages=2,
        )
    return program.map(
        handoff_stage_b(Weight="Weight", Output="Output"),
        **map_kwargs,
    )


def compile_handoff(
    *,
    enabled: bool,
    tasks: int = 3,
    target: df.TargetCapabilitySnapshot = _HOPPER,
    memory_policy: str | None = None,
    task_coords: tuple[tuple[int, ...], ...] | None = None,
):
    kwargs = {}
    if memory_policy is not None:
        kwargs["memory_policy"] = memory_policy
    return df.compile(
        handoff_program(enabled=enabled),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={
            "a_range": [2] * tasks,
            "b_range": [1] * tasks,
        },
        block_size=1,
        task_extents=None if task_coords is not None else (tasks,),
        task_coord_overrides=task_coords,
        include_exit=False,
        target_override=target,
        wrapper_name=("dataflow_generic_handoff_enabled" if enabled else "dataflow_generic_handoff_disabled"),
        **kwargs,
    )


def test_generic_handoff_materializes_typed_arena_and_physical_tma():
    compiled = compile_handoff(enabled=True)
    plan = compiled.plan.cross_handler_handoff_plans[0]

    assert plan.enabled
    assert len(plan.transfer_plans) == 2
    assert compiled.wrapper_spec.handoff_arena_bytes == plan.arena_bytes
    assert compiled.wrapper_spec.handoff_plan_arenas[0].plan_fingerprint == (plan.fingerprint)
    assert compiled.validate_memory_layout().valid
    assert "dataflow_primfunc_handoff_field" in compiled.wrapper_source
    assert "handler_args.handoff_arena_slot" in compiled.wrapper_source
    assert "handler_args.handoff_stage_count == 0u ? 0u" in (compiled.wrapper_source)

    handlers = {
        handler.cross_handler_handoff_role: handler
        for handler in compiled.primfunc_lowering.handlers
        if handler.cross_handler_handoff_role is not None
    }
    assert set(handlers) == {"producer", "consumer"}
    artifact = compiled.decision_artifact().to_dict()
    handoff_binding = artifact["lowerings"]["handoff"]["bindings"][0]
    assert handoff_binding["schema_version"] == 2
    assert handoff_binding["producer_handler_ids"] == [handlers["producer"].handler_id]
    assert handoff_binding["consumer_handler_ids"] == [handlers["consumer"].handler_id]
    assert {(item["transfer_index"], item["stage_index"]) for item in handoff_binding["producer_transfers"]} == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }
    handoff_copy_requests = [request for request in artifact["lowerings"]["copy"]["requests"] if request["handoff_producer"] is not None]
    assert len(handoff_copy_requests) == 4
    producer_pipeline_binding = next(
        binding
        for binding in artifact["lowerings"]["pipeline"]["dataflow_bindings"]
        if binding["handler_id"] == handlers["producer"].handler_id
    )
    assert len(producer_pipeline_binding["transfers"]) == 2
    producer_params = handlers["producer"].params
    consumer_params = handlers["consumer"].params
    assert {param.tensor_arg_index for param in producer_params if param.role is df.DataflowHandlerParamRole.TENSOR_ARG} == {0, 1, 2, 3}
    assert sum(param.role is df.DataflowHandlerParamRole.HANDOFF_TRANSFER_BUFFER for param in producer_params) == 4
    assert sum(param.role is df.DataflowHandlerParamRole.HANDOFF_TRANSFER_BUFFER for param in consumer_params) == 2
    assert "tl.cross_handler_handoff_producer_transfer" not in (compiled.primfunc_lowering.cuda_source)
    assert "tl.tileop.copy" not in compiled.primfunc_lowering.cuda_source


def test_generic_handoff_queue_tail_uses_noop_metadata_and_valid_pointer_slot():
    compiled = compile_handoff(enabled=True, tasks=3)
    bindings = compiled.plan.cross_handler_handoff_bindings

    assert any(binding.state == "active" for binding in bindings)
    assert any(binding.state == "tail" for binding in bindings)
    assert all(binding.stage_count == 0 and binding.arena_slot is None for binding in bindings if binding.state == "tail")
    assert "handler_args.handoff_stage_count == 0u ? 0u" in (compiled.wrapper_source)


def test_generic_handoff_combined_resource_fallback_preserves_explicit_shared_policy():
    constrained = df.TargetCapabilitySnapshot.for_cuda(
        (9, 0),
        compiler_version=(12, 8),
        max_dynamic_shared_memory=140_000,
    )
    compiled = compile_handoff(
        enabled=True,
        target=constrained,
        memory_policy="shared",
    )
    plan = compiled.plan.cross_handler_handoff_plans[0]

    assert not plan.enabled
    assert plan.fallback_reason == "handoff_arena_exceeds_shared_memory_budget"
    assert plan.resource_budget_bytes == 0
    assert plan.required_arena_bytes > 0
    assert plan.decision.candidates[0].resources.shared_memory_bytes == (plan.required_arena_bytes)
    assert compiled.wrapper_spec.handoff_arena_bytes == 0
    assert compiled.memory_plan.policy.mode == "shared"
    assert compiled.memory_plan.selected_candidate.placement == "shared"
    artifact = compiled.decision_artifact().to_dict()
    handoff_contracts = [
        contract for contract in artifact["operation_contracts"]["records"] if contract["request"]["kind"] == "cross_handler_handoff"
    ]
    assert len(handoff_contracts) == 1
    recorded_plan = handoff_contracts[0]["lowering_plans"][0]
    assert recorded_plan["resource_budget_bytes"] == 0
    assert recorded_plan["required_arena_bytes"] == plan.required_arena_bytes


def test_non_moe_handoff_runtime_matches_disabled_pipeline():
    if os.environ.get("TILELANG_DATAFLOW_RUN_HANDOFF_CUDA") != "1":
        pytest.skip("set TILELANG_DATAFLOW_RUN_HANDOFF_CUDA=1 to run handoff equivalence")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA unavailable")

    torch.manual_seed(0)
    left = torch.randn((4, 64, 128), device="cuda", dtype=torch.float16)
    right = torch.randn((4, 128, 64), device="cuda", dtype=torch.float16)
    weight = torch.randn((2, 64, 64), device="cuda", dtype=torch.float16)
    outputs = {}
    task_coords = ((0,), (2,), (3,))
    for enabled in (False, True):
        compiled = compile_handoff(
            enabled=enabled,
            tasks=len(task_coords),
            task_coords=task_coords,
        )
        output = torch.zeros((4, 64, 64), device="cuda", dtype=torch.float32)
        compiled(Left=left, Right=right, Weight=weight, Output=output)
        torch.cuda.synchronize()
        outputs[enabled] = output

    torch.testing.assert_close(
        outputs[True],
        outputs[False],
        rtol=2e-2,
        atol=2e-2,
    )
