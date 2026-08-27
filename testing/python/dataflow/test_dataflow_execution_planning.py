from __future__ import annotations

from dataclasses import replace
import json
import random

import pytest

import tilelang.language as T
import tilelang.dataflow as df
from testing.python.dataflow.test_dataflow_scheduler_config import make_replay_program
from tilelang.dataflow.experimental import debug_handlers as dataflow_debug
from tilelang.dataflow.specialization import capture_operator_specialization


def make_target(
    compute_capability=(9, 0),
    *,
    sm_count=16,
    shared_memory=232_448,
    max_threads=1024,
    supports_cluster_launch=None,
):
    return df.TargetCapabilitySnapshot.for_cuda(
        compute_capability,
        compiler_version=(12, 8),
        max_cluster_size=16,
        supports_cluster_launch=supports_cluster_launch,
        max_dynamic_shared_memory=shared_memory,
        multiprocessor_count=sm_count,
        max_threads_per_block=max_threads,
        max_registers_per_block=65_536,
    )


def make_request(
    *,
    task_extent=512,
    first_output=2048,
    second_output=4096,
    dtype="float8_e4m3fn",
    transport_family=df.DATAFLOW_TRANSPORT_AUTO,
):
    return df.DataflowExecutionRequest(
        task_extent=task_extent,
        stages=(
            df.DataflowExecutionStageRequest(
                output_extent=first_output,
                reduction_extent=second_output,
                input_dtype=dtype,
                weight_dtype=dtype,
                output_dtype=dtype,
                projection_count=2,
            ),
            df.DataflowExecutionStageRequest(
                output_extent=second_output,
                reduction_extent=first_output,
                input_dtype=dtype,
                weight_dtype=dtype,
                output_dtype="float16",
                input_from_previous_stage=True,
            ),
        ),
        linked_stage_pairs=((0, 1),),
        transport_family=transport_family,
    )


def portable_override(*, topology=None, task_tile=32):
    if topology is None:
        topology = df.GPUTopology(4, 2)
    stage = df.DataflowExecutionStageOverride(
        tile_n=64,
        tile_k=64,
        compute_threads=128,
        consumer_threads=128,
        gemm_family=df.DATAFLOW_EXECUTION_GEMM_PORTABLE,
        transfer_family=df.DATAFLOW_EXECUTION_TRANSFER_SYNCHRONOUS,
        input_distribution=df.DATAFLOW_EXECUTION_INPUT_COOPERATIVE,
        split_producers=False,
    )
    return df.DataflowExecutionOverride(
        topology=topology,
        task_tile_extent=task_tile,
        stages=(stage, stage),
        transport_family=df.DATAFLOW_TRANSPORT_HBM,
        handoff_stages=0,
    )


def test_execution_stage_request_accepts_native_tilelang_dtype():
    stage = df.DataflowExecutionStageRequest(
        output_extent=64,
        reduction_extent=128,
        input_dtype=T.float16,
        weight_dtype=T.float8_e4m3fn,
        output_dtype=T.float32,
    )

    assert all(isinstance(dtype, type(T.float16)) for dtype in (stage.input_dtype, stage.weight_dtype, stage.output_dtype))
    assert stage.input_dtype == T.float16
    assert stage.weight_dtype == T.float8_e4m3fn
    assert stage.output_dtype == T.float32
    assert not df.require_dataflow_dtype(stage.input_dtype).is_float8
    assert df.require_dataflow_dtype(stage.weight_dtype).is_float8
    assert stage.to_dict()["input_dtype"] == "float16"
    assert stage.to_dict()["weight_dtype"] == "float8_e4m3fn"
    assert stage.to_dict()["output_dtype"] == "float32"
    assert df.DataflowExecutionStageRequest.from_dict(stage.to_dict()) == stage


def test_native_tilelang_dtype_specialization_is_canonical():
    dtype = T.float16

    def uses_native_dtype():
        return dtype

    snapshot = capture_operator_specialization(uses_native_dtype)

    assert [(entry.name, entry.canonical_value) for entry in snapshot.entries] == [("dtype", "float16")]


def test_execution_auto_policy_generates_legal_target_aware_candidates():
    plan = df.plan_execution(make_request(), target_capabilities=make_target())

    assert plan.planner_version == df.DATAFLOW_EXECUTION_PLANNER_VERSION
    assert len(plan.candidates) >= 4
    assert plan.selected_evaluation.legal
    assert plan.selected_candidate.stages[0].gemm_family in {
        df.DATAFLOW_EXECUTION_GEMM_WARP_GROUP,
        df.DATAFLOW_EXECUTION_GEMM_WARP_GROUP_SMALL_M,
    }
    assert plan.selected_candidate.stages[0].transfer_family == (df.DATAFLOW_EXECUTION_TRANSFER_TMA)
    assert plan.resources.shared_memory_bytes > 0
    assert plan.resources.register_count > 0
    assert plan.resources.pipeline_shared_memory_budgets[0] is not None
    assert all(item.legal or item.rejection_reasons for item in plan.candidates)


def test_execution_auto_policy_generates_small_m_acceleration_from_logical_extent():
    plan = df.plan_execution(
        make_request(
            task_extent=16,
            first_output=256,
            second_output=256,
        ),
        target_capabilities=make_target(sm_count=1),
    )

    assert plan.selected_candidate.task_tile_extent == 16
    assert all(stage.gemm_family == df.DATAFLOW_EXECUTION_GEMM_WARP_GROUP_SMALL_M for stage in plan.selected_candidate.stages)


def test_execution_plan_round_trip_is_strict_and_fingerprinted():
    plan = df.plan_execution(make_request(), target_capabilities=make_target())
    payload = json.loads(json.dumps(plan.to_dict()))
    restored = df.DataflowExecutionPlan.from_dict(payload)

    assert restored == plan
    assert restored.fingerprint == plan.fingerprint
    with pytest.raises(ValueError, match="unknown fields"):
        df.DataflowExecutionPlan.from_dict({**payload, "printer_hint": "ignored"})
    with pytest.raises(ValueError, match="fingerprint"):
        df.DataflowExecutionPlan.from_dict({**payload, "fingerprint": "0" * 64})


@pytest.mark.parametrize("seed", (7, 19, 31))
def test_execution_randomized_shapes_and_topologies_are_legal(seed):
    rng = random.Random(seed)
    cluster_size = rng.choice((1, 2, 4))
    first_output = cluster_size * 64 * rng.randint(2, 8)
    second_output = cluster_size * 64 * rng.randint(2, 8)
    topology = df.GPUTopology(sm_count=cluster_size * 4, cluster_size=cluster_size)
    request = make_request(
        task_extent=rng.randint(2, 16) * 32,
        first_output=first_output,
        second_output=second_output,
        dtype="float16",
    )
    override = portable_override(topology=topology)

    plan = df.plan_execution(
        request,
        override=override,
        target_capabilities=make_target(
            (8, 0),
            sm_count=topology.sm_count,
            supports_cluster_launch=cluster_size > 1,
        ),
    )

    assert plan.explicit_override is True
    assert plan.selected_candidate.topology == topology
    assert plan.selected_evaluation.legal
    assert plan.selected_candidate.stages[0].tile_n == 64
    assert plan.selected_candidate.stages[1].tile_k == 64


def test_execution_explicit_acceleration_rejects_unsupported_target():
    accelerated = df.DataflowExecutionStageOverride(
        tile_n=128,
        tile_k=128,
        compute_threads=256,
        consumer_threads=128,
        gemm_family=df.DATAFLOW_EXECUTION_GEMM_WARP_GROUP,
        transfer_family=df.DATAFLOW_EXECUTION_TRANSFER_TMA,
        input_distribution=df.DATAFLOW_EXECUTION_INPUT_UNICAST,
        split_producers=True,
    )
    override = df.DataflowExecutionOverride(
        topology=df.GPUTopology(1, 1),
        task_tile_extent=64,
        stages=(accelerated, replace(accelerated, consumer_threads=256)),
        transport_family=df.DATAFLOW_TRANSPORT_HBM,
    )

    with pytest.raises(df.DataflowExecutionPlanningError) as captured:
        df.plan_execution(
            make_request(),
            override=override,
            target_capabilities=make_target((8, 0), sm_count=1),
        )

    payload = captured.value.to_dict()
    reasons = payload["candidates"][0]["rejection_reasons"]
    assert "stage_0_target_tma_unsupported" in reasons
    assert "stage_0_target_warp_group_gemm_unsupported" in reasons
    assert payload["request"]["fingerprint"] == make_request().fingerprint


def test_execution_pipeline_ownership_is_decided_per_stage():
    first = df.DataflowExecutionStageOverride(
        tile_n=128,
        tile_k=128,
        handler_extent=128,
        compute_threads=256,
        consumer_threads=128,
        pipeline_stages=2,
        max_outstanding=2,
        gemm_family=df.DATAFLOW_EXECUTION_GEMM_WARP_GROUP,
        transfer_family=df.DATAFLOW_EXECUTION_TRANSFER_TMA,
        input_distribution=df.DATAFLOW_EXECUTION_INPUT_MULTICAST,
        split_producers=True,
    )
    second = replace(
        first,
        handler_extent=256,
        consumer_threads=256,
        input_distribution=df.DATAFLOW_EXECUTION_INPUT_COOPERATIVE,
        split_producers=False,
    )

    plan = df.plan_execution(
        make_request(task_extent=64, first_output=256, second_output=512),
        override=df.DataflowExecutionOverride(
            topology=df.GPUTopology(2, 2),
            task_tile_extent=64,
            stages=(first, second),
            transport_family=df.DATAFLOW_TRANSPORT_STREAMED,
        ),
        target_capabilities=make_target(sm_count=2),
    )

    assert tuple(stage.common_pipeline for stage in plan.selected_candidate.stages) == (False, True)


def test_execution_explicit_override_cannot_bypass_thread_or_shared_limits():
    too_many_threads = portable_override(topology=df.GPUTopology(4, 2))
    too_many_threads = replace(
        too_many_threads,
        stages=tuple(replace(stage, compute_threads=1056, consumer_threads=1056) for stage in too_many_threads.stages),
    )
    with pytest.raises(df.DataflowExecutionPlanningError, match="threads_per_block"):
        df.plan_execution(
            make_request(dtype="float16"),
            override=too_many_threads,
            target_capabilities=make_target((8, 0), sm_count=4),
        )

    with pytest.raises(df.DataflowExecutionPlanningError, match="shared_memory"):
        df.plan_execution(
            make_request(dtype="float16"),
            override=portable_override(),
            target_capabilities=make_target(
                (8, 0),
                sm_count=4,
                shared_memory=8 * 1024,
            ),
        )


def test_execution_explicit_override_cannot_bypass_topology_or_warp_group_limits():
    oversized_topology = portable_override(topology=df.GPUTopology(8, 2))
    with pytest.raises(df.DataflowExecutionPlanningError) as captured:
        df.plan_execution(
            make_request(dtype="float16"),
            override=oversized_topology,
            target_capabilities=make_target((8, 0), sm_count=4),
        )
    assert "target_multiprocessor_count_limit_exceeded" in (captured.value.to_dict()["candidates"][0]["rejection_reasons"])

    accelerated = df.DataflowExecutionStageOverride(
        tile_n=128,
        tile_k=128,
        compute_threads=256,
        consumer_threads=64,
        gemm_family=df.DATAFLOW_EXECUTION_GEMM_WARP_GROUP,
        transfer_family=df.DATAFLOW_EXECUTION_TRANSFER_TMA,
        input_distribution=df.DATAFLOW_EXECUTION_INPUT_COOPERATIVE,
        split_producers=False,
    )
    with pytest.raises(df.DataflowExecutionPlanningError) as captured:
        df.plan_execution(
            make_request(),
            override=df.DataflowExecutionOverride(
                topology=df.GPUTopology(1, 1),
                task_tile_extent=64,
                stages=(accelerated, accelerated),
                transport_family=df.DATAFLOW_TRANSPORT_ALL_GATHER,
            ),
            target_capabilities=make_target(sm_count=1),
        )
    assert any(
        "tma_consumer_requires_complete_warp_group" in reason for reason in captured.value.to_dict()["candidates"][0]["rejection_reasons"]
    )


def test_streamed_transport_resources_use_stage_lifetime_peak():
    request = df.DataflowExecutionRequest(
        task_extent=32,
        stages=(
            df.DataflowExecutionStageRequest(
                output_extent=1024,
                reduction_extent=256,
                input_dtype="float8_e4m3fn",
                weight_dtype="float8_e4m3fn",
                output_dtype="float8_e4m3fn",
                projection_count=2,
            ),
            df.DataflowExecutionStageRequest(
                output_extent=512,
                reduction_extent=1024,
                input_dtype="float8_e4m3fn",
                weight_dtype="float8_e4m3fn",
                output_dtype="float16",
                input_from_previous_stage=True,
            ),
        ),
        linked_stage_pairs=((0, 1),),
    )
    first = df.DataflowExecutionStageOverride(
        tile_n=128,
        tile_k=128,
        handler_extent=512,
        compute_threads=128,
        consumer_threads=128,
        pipeline_stages=3,
        max_outstanding=2,
        gemm_family=df.DATAFLOW_EXECUTION_GEMM_WARP_GROUP_SMALL_M,
        transfer_family=df.DATAFLOW_EXECUTION_TRANSFER_TMA,
        input_distribution=df.DATAFLOW_EXECUTION_INPUT_UNICAST,
        split_producers=True,
    )
    second = replace(
        first,
        handler_extent=256,
        pipeline_stages=2,
        max_outstanding=2,
        input_distribution=df.DATAFLOW_EXECUTION_INPUT_COOPERATIVE,
        split_producers=False,
    )
    plan = df.plan_execution(
        request,
        override=df.DataflowExecutionOverride(
            topology=df.GPUTopology(2, 2),
            task_tile_extent=32,
            stages=(first, second),
            transport_family=df.DATAFLOW_TRANSPORT_STREAMED,
        ),
        target_capabilities=make_target(
            sm_count=2,
            shared_memory=120_000,
        ),
    )

    assert plan.resources.shared_memory_bytes == 96_480
    assert plan.resources.shared_memory_bytes <= 120_000


def test_execution_control_reserve_keeps_legal_large_tile_candidate():
    request = make_request(
        task_extent=1024,
        first_output=2048,
        second_output=7168,
    )
    first = df.DataflowExecutionStageOverride(
        tile_n=128,
        tile_k=128,
        handler_extent=128,
        compute_threads=256,
        consumer_threads=256,
        pipeline_stages=3,
        max_outstanding=2,
        gemm_family=df.DATAFLOW_EXECUTION_GEMM_WARP_GROUP,
        transfer_family=df.DATAFLOW_EXECUTION_TRANSFER_TMA,
        input_distribution=df.DATAFLOW_EXECUTION_INPUT_UNICAST,
        split_producers=True,
        wait_depth=1,
    )
    second = replace(
        first,
        tile_n=256,
        handler_extent=1792,
        pipeline_stages=2,
        consumer_threads=256,
        input_distribution=df.DATAFLOW_EXECUTION_INPUT_COOPERATIVE,
        split_producers=False,
        wait_depth=0,
    )

    plan = df.plan_execution(
        request,
        override=df.DataflowExecutionOverride(
            topology=df.GPUTopology(112, 4),
            task_tile_extent=64,
            stages=(first, second),
            transport_family=df.DATAFLOW_TRANSPORT_ALL_GATHER,
        ),
        target_capabilities=make_target(
            sm_count=132,
            shared_memory=227_328,
        ),
    )

    assert plan.selected_evaluation.legal
    assert plan.resources.shared_memory_bytes <= 227_328
    assert df.DATAFLOW_EXECUTION_SHARED_CONTROL_RESERVE_BYTES == 1024


def test_pipeline_budget_reserves_transport_specific_reshared_storage():
    request = df.DataflowExecutionRequest(
        task_extent=32,
        stages=(
            df.DataflowExecutionStageRequest(
                output_extent=1024,
                reduction_extent=256,
                input_dtype="float8_e4m3fn",
                weight_dtype="float8_e4m3fn",
                output_dtype="float8_e4m3fn",
                projection_count=2,
            ),
            df.DataflowExecutionStageRequest(
                output_extent=512,
                reduction_extent=1024,
                input_dtype="float8_e4m3fn",
                weight_dtype="float8_e4m3fn",
                output_dtype="float16",
                input_from_previous_stage=True,
            ),
        ),
        linked_stage_pairs=((0, 1),),
    )
    first = df.DataflowExecutionStageOverride(
        tile_n=128,
        tile_k=128,
        handler_extent=512,
        compute_threads=128,
        consumer_threads=128,
        pipeline_stages=3,
        # Keep every stage outstanding here so the transport-specific hardware
        # budget is observable. A 3-stage/2-outstanding request intentionally
        # caps this aggregate at its prefetch-only preferred budget instead.
        max_outstanding=3,
        gemm_family=df.DATAFLOW_EXECUTION_GEMM_WARP_GROUP_SMALL_M,
        transfer_family=df.DATAFLOW_EXECUTION_TRANSFER_TMA,
        input_distribution=df.DATAFLOW_EXECUTION_INPUT_UNICAST,
        split_producers=True,
    )
    second = replace(
        first,
        handler_extent=256,
        pipeline_stages=2,
        max_outstanding=2,
        input_distribution=df.DATAFLOW_EXECUTION_INPUT_COOPERATIVE,
        split_producers=False,
    )
    target = make_target(sm_count=2, shared_memory=140_000)

    def plan(family):
        return df.plan_execution(
            request,
            override=df.DataflowExecutionOverride(
                topology=df.GPUTopology(2, 2),
                task_tile_extent=32,
                stages=(first, second),
                transport_family=family,
            ),
            target_capabilities=target,
        )

    all_gather = plan(df.DATAFLOW_TRANSPORT_ALL_GATHER)
    streamed = plan(df.DATAFLOW_TRANSPORT_STREAMED)
    all_gather_budget = all_gather.resources.pipeline_shared_memory_budgets[0]
    streamed_budget = streamed.resources.pipeline_shared_memory_budgets[0]

    assert all_gather_budget is not None
    assert streamed_budget is not None
    # The logical value is 32 KiB. Streamed transport keeps only this rank's
    # 16 KiB physical producer slot, while all-gather materializes all 32 KiB.
    assert streamed_budget - all_gather_budget == 16 * 1024


def test_execution_full_partition_supports_a_tail_output_tile():
    request = df.DataflowExecutionRequest(
        task_extent=64,
        stages=(
            df.DataflowExecutionStageRequest(
                output_extent=256,
                reduction_extent=384,
                input_dtype="float8_e4m3fn",
                weight_dtype="float8_e4m3fn",
                output_dtype="float8_e4m3fn",
                projection_count=2,
            ),
            df.DataflowExecutionStageRequest(
                output_extent=896,
                reduction_extent=256,
                input_dtype="float8_e4m3fn",
                weight_dtype="float8_e4m3fn",
                output_dtype="float16",
                input_from_previous_stage=True,
            ),
        ),
        linked_stage_pairs=((0, 1),),
        uniform_stage_implementation=True,
    )
    first = df.DataflowExecutionStageOverride(
        tile_n=128,
        tile_k=128,
        handler_extent=128,
        compute_threads=256,
        consumer_threads=128,
        pipeline_stages=2,
        max_outstanding=2,
        gemm_family=df.DATAFLOW_EXECUTION_GEMM_WARP_GROUP,
        transfer_family=df.DATAFLOW_EXECUTION_TRANSFER_TMA,
        input_distribution=df.DATAFLOW_EXECUTION_INPUT_UNICAST,
        split_producers=True,
    )
    second = replace(
        first,
        tile_n=256,
        handler_extent=448,
        consumer_threads=256,
        input_distribution=df.DATAFLOW_EXECUTION_INPUT_COOPERATIVE,
        split_producers=False,
    )
    plan = df.plan_execution(
        request,
        override=df.DataflowExecutionOverride(
            topology=df.GPUTopology(2, 2),
            task_tile_extent=64,
            stages=(first, second),
            transport_family=df.DATAFLOW_TRANSPORT_ALL_GATHER,
        ),
        target_capabilities=make_target(sm_count=2),
    )

    assert plan.selected_candidate.stages[1].full_partition_pipeline is True
    assert plan.selected_candidate.stages[1].handler_extent == 448


def test_execution_auto_policy_uses_structured_portable_fallback():
    plan = df.plan_execution(
        make_request(dtype="float16"),
        target_capabilities=make_target((8, 0), sm_count=8, shared_memory=163_840),
    )

    assert plan.used_fallback is True
    assert plan.selection_reason == "capability_portable_fallback"
    assert all(stage.gemm_family == df.DATAFLOW_EXECUTION_GEMM_PORTABLE for stage in plan.selected_candidate.stages)
    assert plan.selected_candidate.transport_family == df.DATAFLOW_TRANSPORT_ALL_GATHER


def test_execution_request_has_no_diagnostic_identity_in_selection_key():
    request = make_request()
    payload = request.canonical_payload()

    assert "name" not in json.dumps(payload).lower()
    assert "operator" not in json.dumps(payload).lower()
    assert df.DataflowExecutionRequest.from_dict(request.to_dict()) == request


def compile_with_execution_plan(
    *,
    override=None,
    semantic_config=None,
    compile_flags=(),
):
    @df.jit(cache=False)
    def factory(execution_override=None, semantic=None, **compile_options):
        df.plan_execution(make_request(dtype="float16"), override=execution_override)
        return df.make_kernel_spec(
            make_replay_program(),
            topology=df.GPUTopology(sm_count=4, cluster_size=1),
            range_lengths={"kv": (128, 256)},
            block_size=64,
            task_extents=(2,),
            mode="debug",
            semantic_config=semantic,
            _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
            **compile_options,
        )

    options = {"compile_flags": compile_flags} if compile_flags else {}
    return factory(
        execution_override=override,
        semantic=semantic_config,
        target_override=make_target((8, 0), sm_count=4),
        **options,
    )


def test_jit_captures_execution_plan_in_artifact_and_cache_fingerprint():
    first = compile_with_execution_plan()
    explicit = compile_with_execution_plan(override=portable_override(topology=df.GPUTopology(4, 1)))
    first_plans = first.decision_artifact().to_dict()["scheduler"]["execution_plans"]
    explicit_plans = explicit.decision_artifact().to_dict()["scheduler"]["execution_plans"]

    assert len(first_plans) == 1
    assert len(explicit_plans) == 1
    assert first_plans[0]["selection_reason"] == "capability_portable_fallback"
    assert explicit_plans[0]["selection_reason"] == "explicit_typed_override"
    assert first.compile_config.fingerprint != explicit.compile_config.fingerprint
    assert first.compile_config.options_dict()["execution_plans"] == tuple(first_plans)
    selected = first.decision_artifact().to_dict()["governance"]["selected_implementations"]
    assert any(item["implementation_id"] == df.DATAFLOW_EXECUTION_PLANNER_IMPLEMENTATION for item in selected)


def test_fast_math_is_typed_semantic_config_and_compiler_owned_flag():
    strict = compile_with_execution_plan()
    fast = compile_with_execution_plan(semantic_config=df.DataflowSemanticConfig(fast_math=True))

    assert "--use_fast_math" not in strict.compile_config.options_dict().get("compile_flags", ())
    assert "--use_fast_math" in fast.compile_config.options_dict()["compile_flags"]
    assert fast.compile_config.semantic_config.fast_math is True
    assert strict.compile_config.fingerprint != fast.compile_config.fingerprint
    with pytest.raises(ValueError, match="compiler-owned"):
        compile_with_execution_plan(compile_flags=("--use_fast_math",))


def test_target_resource_schema_round_trip_and_legacy_compatibility():
    current = make_target()
    restored = df.TargetCapabilitySnapshot.from_dict(current.to_dict())
    assert restored == current
    assert restored.multiprocessor_count == 16
    assert restored.max_threads_per_block == 1024
    assert restored.max_registers_per_block == 65_536

    legacy_payload = current.to_dict()
    legacy_payload["schema_version"] = 3
    for name in (
        "multiprocessor_count",
        "max_threads_per_block",
        "max_registers_per_block",
        "fingerprint",
    ):
        legacy_payload.pop(name, None)
    legacy = df.TargetCapabilitySnapshot.from_dict(legacy_payload)
    assert legacy.schema_version == 3
    assert legacy.multiprocessor_count is None
