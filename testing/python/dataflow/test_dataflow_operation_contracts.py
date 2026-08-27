from __future__ import annotations

from dataclasses import replace
from datetime import date
import json

import pytest

import tilelang.language as T
import tilelang.dataflow as df
from tilelang.dataflow.experimental import debug_handlers as dataflow_debug
from tilelang.dataflow.operation_contracts import (
    DATAFLOW_HANDOFF_QUEUE_IMPLEMENTATION,
    DATAFLOW_LAYOUT_LINEAR_IMPLEMENTATION,
    DATAFLOW_RANGE_STAGE_GRAPH_IMPLEMENTATION,
    DATAFLOW_TRANSPORT_ALL_GATHER_IMPLEMENTATION,
    selected_operation_decision,
)


@T.dataflow_intermediate
class StatValue:
    total: T.int32


@T.dataflow.iter(range=("begin", "end"))
def window_stat(batch: T.int32, Source: T.Tensor((64,), T.int32)) -> StatValue:
    raise AssertionError("debug handler compilation must not execute operator bodies")


@T.dataflow.iter(range=("begin", "end"))
def segment_stat(batch: T.int32, Source: T.Tensor((64,), T.int32)) -> StatValue:
    raise AssertionError("debug handler compilation must not execute operator bodies")


@T.dataflow.reduce(associative=True)
def merge_stat(left: StatValue, right: StatValue) -> StatValue:
    return StatValue(total=left.total + right.total)


@T.dataflow.finalize
def store_stat(
    value: StatValue,
    batch: T.int32,
    Output: T.Tensor((1,), T.int32),
) -> None:
    Output[batch] = value.total


@T.dataflow_intermediate
class GraphShard:
    value: T.int32


@T.dataflow_intermediate
class GraphFull:
    value: T.int32


@T.dataflow_intermediate
class GraphOutput:
    value: T.int32


@T.dataflow_intermediate(
    layout_contracts=df.DataflowTensorLayoutRequest(
        field_index=0,
        logical_rank=1,
        layout_family=df.DATAFLOW_LAYOUT_LINEAR,
    )
)
class LayoutStat:
    values: T.Tensor((2,), T.int32)


@T.dataflow.map(range=("begin", "end"))
def graph_source(task: T.int32, Source: T.Tensor((64,), T.int32)) -> GraphShard:
    raise AssertionError("debug handler compilation must not execute operator bodies")


@T.dataflow.map(range=("begin", "end"))
def graph_sink(
    parts: list[GraphShard],
    task: T.int32,
    Bias: T.Tensor((64,), T.int32),
) -> GraphOutput:
    raise AssertionError("debug handler compilation must not execute operator bodies")


@T.dataflow.iter(range=("begin", "end"))
def layout_stat(Source: T.Tensor((64,), T.int32)) -> LayoutStat:
    first = T.int32(0)
    second = T.int32(0)
    for index in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        first += Source[index]
        second += Source[index] * T.int32(2)
    return LayoutStat(values=(first, second))


@T.dataflow.reduce
def merge_layout_stat(items: list[LayoutStat]) -> LayoutStat:
    first = T.int32(0)
    second = T.int32(0)
    for item in items:
        first += item.values[0]
        second += item.values[1]
    return LayoutStat(values=(first, second))


@T.dataflow.finalize
def store_layout_stat(
    value: LayoutStat,
    Output: T.Tensor((1, 2), T.int32),
) -> None:
    Output[T.dataflow_task_id(), 0] = value.values[0]
    Output[T.dataflow_task_id(), 1] = value.values[1]


def requests(diagnostic_name: str | None = None):
    return (
        df.DataflowRangeCoarseningRequest(
            logical_tile_extent=16,
            logical_range_extent=64,
            handler_range_extent=32,
            output_tile_arity=2,
            diagnostic_name=diagnostic_name,
        ),
        df.DataflowPipelineRequest(
            transfers=(
                df.DataflowPipelineTransfer(
                    destination_buffer_index=0,
                    logical_extent=(16, None),
                    bytes_per_stage=128,
                    producer_partition=0,
                ),
                df.DataflowPipelineTransfer(
                    destination_buffer_index=1,
                    logical_extent=(None, 16),
                    bytes_per_stage=128,
                    producer_partition=1,
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
            stage_budget=3,
            max_outstanding=2,
            producer_threads=128,
            consumer_threads=128,
            diagnostic_name=diagnostic_name,
        ),
        df.DataflowResharedTransportRequest(
            family=df.DATAFLOW_TRANSPORT_ALL_GATHER,
            logical_output_arity=4,
            physical_output_arity=2,
            slot_bytes=128,
            diagnostic_name=diagnostic_name,
        ),
        df.DataflowCrossHandlerHandoffRequest(
            consumer_stage_id=0,
            lookahead_distance=1,
            buffer_stages=2,
            value_bindings=(0, 2),
            diagnostic_name=diagnostic_name,
        ),
        df.DataflowTensorLayoutRequest(
            field_index=0,
            logical_rank=2,
            layout_family=df.DATAFLOW_LAYOUT_MATRIX_SWIZZLE,
            major_axis=1,
            alignment_bytes=16,
            diagnostic_name=diagnostic_name,
        ),
    )


@pytest.mark.parametrize("operation_request", requests("debug-only-name"))
def test_operation_request_round_trip_is_canonical_and_versioned(operation_request):
    payload = operation_request.to_dict(include_diagnostics=True)
    restored = df.dataflow_operation_request_from_dict(payload)

    assert restored == operation_request
    assert restored.fingerprint == operation_request.fingerprint
    assert payload["schema_version"] == df.DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION
    assert payload["kind"] in df.DATAFLOW_OPERATION_CONTRACT_KINDS
    assert len(payload["fingerprint"]) == 64


def test_operation_request_fingerprints_ignore_only_diagnostic_names():
    baseline = requests("first-source-name")
    renamed = requests("unrelated-second-name")

    assert [item.fingerprint for item in baseline] == [item.fingerprint for item in renamed]
    assert all("diagnostic_name" not in item.to_dict() for item in baseline)
    assert replace(baseline[0], handler_range_extent=48).fingerprint != (baseline[0].fingerprint)


def test_operation_schema_contains_no_workload_or_diagnostic_identity_fields():
    schema = json.dumps(
        df.dataflow_operation_contract_schema_dict(),
        sort_keys=True,
    ).lower()

    for forbidden in (
        "expert",
        "moe",
        "gate_up",
        "trace_identity",
        "model_identity",
        "operator_name",
        "diagnostic_name",
    ):
        assert forbidden not in schema


def test_serialized_operation_requests_fail_closed_on_schema_shape():
    payload = requests()[0].to_dict()

    missing_schema = dict(payload)
    missing_schema.pop("schema_version")
    with pytest.raises(ValueError, match="missing versioned fields"):
        df.dataflow_operation_request_from_dict(missing_schema)

    unknown = dict(payload)
    unknown["expert_count"] = 8
    with pytest.raises(ValueError, match="unknown fields.*expert_count"):
        df.dataflow_operation_request_from_dict(unknown)

    stale = dict(payload)
    stale["schema_version"] = payload["schema_version"] + 1
    with pytest.raises(ValueError, match="Unsupported Dataflow range.*schema version"):
        df.dataflow_operation_request_from_dict(stale)


def test_predicated_range_contract_represents_a_partial_tail_tile():
    request = df.DataflowRangeCoarseningRequest(
        logical_tile_extent=256,
        logical_range_extent=448,
        handler_range_extent=448,
        remainder_policy=df.DATAFLOW_REMAINDER_PREDICATE,
    )

    assert request.tiles_per_handler == 2
    with pytest.raises(ValueError, match="exact range remainder policy"):
        replace(request, remainder_policy=df.DATAFLOW_REMAINDER_EXACT)


def stat_program(iter_operator, axis: str, request):
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={axis: "lengths"})
        .partial(
            iter_operator(Source="Source"),
            task_args=("batch",),
            range_axis=axis,
            range_contract=request,
        )
        .reduce(merge_stat())
        .finalize(store_stat(Output="Output"))
    )


def compile_debug(program, axis: str):
    return df.compile(
        program,
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={axis: [64]},
        block_size=32,
        task_extents=(1,),
        include_exit=False,
        mode="debug",
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
            max_dynamic_shared_memory=232_448,
        ),
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )


def test_two_non_moe_programs_reuse_the_same_typed_contract():
    request = requests()[0]
    window = stat_program(window_stat, "window", request)
    segment = stat_program(segment_stat, "segment", request)

    assert window.stages[0].attrs["range_contract"] is request
    assert segment.stages[0].attrs["range_contract"] is request
    assert window.stages[0].attrs["range_contract"].fingerprint == (segment.stages[0].attrs["range_contract"].fingerprint)


def test_semantic_request_changes_compile_cache_key_but_diagnostic_name_does_not():
    baseline = compile_debug(
        stat_program(window_stat, "window", requests("first")[0]),
        "window",
    )
    renamed = compile_debug(
        stat_program(window_stat, "window", requests("second")[0]),
        "window",
    )
    changed = compile_debug(
        stat_program(
            window_stat,
            "window",
            replace(requests("first")[0], handler_range_extent=48),
        ),
        "window",
    )

    assert baseline.compile_config.program_fingerprint == (renamed.compile_config.program_fingerprint)
    assert baseline.compile_config.fingerprint == renamed.compile_config.fingerprint
    assert changed.compile_config.program_fingerprint != (baseline.compile_config.program_fingerprint)
    assert changed.compile_config.fingerprint != baseline.compile_config.fingerprint


def test_pipeline_request_records_a_typed_logical_plan_before_codegen_selection():
    program = (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"window": "lengths"})
        .partial(
            window_stat(Source="Source"),
            task_args=("batch",),
            range_axis="window",
            pipeline_contract=requests()[1],
        )
        .reduce(merge_stat())
        .finalize(store_stat(Output="Output"))
    )
    compiled = compile_debug(program, "window")

    record = next(
        item
        for item in compiled.decision_artifact().to_dict()["operation_contracts"]["records"]
        if item["request"]["kind"] == "pipeline_dataflow"
    )
    assert record["selection_state"] == "selected"
    assert len(record["lowering_plans"]) == 1
    assert record["lowering_plans"][0]["selected_stages"] == 3
    assert record["decisions"][0]["selected_implementation"] == (df.DATAFLOW_PIPELINE_PLAN_IMPLEMENTATION)


def contract_spec(
    implementation_id: str,
    *,
    kind: str,
    versions=(df.DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION,),
    state=df.DataflowImplementationState.STABLE,
    default_enabled=True,
    removal_date=None,
):
    return df.DataflowImplementationSpec(
        implementation_id=implementation_id,
        domain="test.operation",
        state=state,
        owner="QA",
        legality_contract="typed operation test legality",
        benchmark_evidence=("testing/python/dataflow/test_dataflow_operation_contracts.py",),
        default_enabled=default_enabled,
        removal_date=removal_date,
        contract_kinds=(kind,),
        contract_schema_versions=tuple(versions),
    )


def test_contract_registry_rejects_lifecycle_kind_and_schema_mismatches():
    request = requests()[0]
    retired = contract_spec(
        "test.range.retired",
        kind=request.KIND,
        state=df.DataflowImplementationState.RETIRED,
        default_enabled=False,
        removal_date="2027-01-01",
    )
    expired = contract_spec(
        "test.range.expired",
        kind=request.KIND,
        state=df.DataflowImplementationState.EXPERIMENTAL,
        default_enabled=False,
        removal_date="2026-01-01",
    )
    wrong_kind = contract_spec(
        "test.pipeline.only",
        kind=df.DataflowPipelineRequest.KIND,
    )
    wrong_schema = contract_spec(
        "test.range.future-schema",
        kind=request.KIND,
        versions=(request.schema_version + 1,),
    )
    registry = df.DataflowImplementationRegistry((retired, expired, wrong_kind, wrong_schema))

    with pytest.raises(ValueError, match="unregistered"):
        registry.require_contract_compatible(
            "test.unregistered",
            request,
            selected_explicitly=True,
        )
    with pytest.raises(ValueError, match="retired"):
        registry.require_contract_compatible(
            retired.implementation_id,
            request,
            selected_explicitly=True,
        )
    with pytest.raises(ValueError, match="expired"):
        registry.require_contract_compatible(
            expired.implementation_id,
            request,
            selected_explicitly=True,
            today=date(2026, 7, 30),
        )
    with pytest.raises(ValueError, match="does not implement contract kind"):
        registry.require_contract_compatible(
            wrong_kind.implementation_id,
            request,
            selected_explicitly=True,
        )
    with pytest.raises(ValueError, match="does not support.*schema version"):
        registry.require_contract_compatible(
            wrong_schema.implementation_id,
            request,
            selected_explicitly=True,
        )


def typed_stage_graph(transport_family=df.DATAFLOW_TRANSPORT_ALL_GATHER):
    range_request = df.DataflowRangeCoarseningRequest(
        logical_tile_extent=32,
        logical_range_extent=32,
        handler_range_extent=32,
    )
    return (
        T.dataflow_program(task_domain=("task",), dynamic_ranges={"x": "xs", "y": "ys"})
        .map(
            graph_source(Source="Source"),
            name="source",
            task_args=("task",),
            range_axis="x",
            range_contract=range_request,
        )
        .reshared(
            input="source",
            output_type=GraphFull,
            physical_output_type=GraphShard,
            output_arity=2,
            transport_contract=df.DataflowResharedTransportRequest(
                family=transport_family,
                logical_output_arity=2,
                physical_output_arity=2,
            ),
        )
        .map(
            graph_sink(Bias="Bias"),
            name="sink",
            input="reshared",
            task_args=("task",),
            range_axis="y",
            range_contract=range_request,
            handoff_contract=df.DataflowCrossHandlerHandoffRequest(
                consumer_stage_id=0,
                buffer_stages=2,
            ),
        )
    )


def compile_typed_stage_graph(program):
    return df.compile(
        program,
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"x": [64, 64], "y": [64, 64]},
        block_size=32,
        task_extents=(2,),
        include_exit=False,
        mode="debug",
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
            max_dynamic_shared_memory=232_448,
        ),
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )


def test_decision_artifact_records_actual_typed_stage_graph_selections():
    compiled = compile_typed_stage_graph(typed_stage_graph())

    artifact = compiled.decision_artifact().to_dict()
    contracts = artifact["operation_contracts"]
    assert contracts["schema"] == df.dataflow_operation_contract_schema_dict()
    assert len(contracts["records"]) == 4
    assert all(record["selection_state"] == "selected" for record in contracts["records"])
    selected = {decision["selected_implementation"] for record in contracts["records"] for decision in record["decisions"]}
    assert selected == {
        DATAFLOW_RANGE_STAGE_GRAPH_IMPLEMENTATION,
        DATAFLOW_TRANSPORT_ALL_GATHER_IMPLEMENTATION,
        DATAFLOW_HANDOFF_QUEUE_IMPLEMENTATION,
    }
    assert all(
        "diagnostic_name" not in record["request"] and all(decision["candidates"] for decision in record["decisions"])
        for record in contracts["records"]
    )
    selected_lifecycle = {record["implementation_id"] for record in artifact["governance"]["selected_implementations"]}
    assert selected <= selected_lifecycle


def test_explicit_hbm_transport_family_does_not_require_a_legacy_force_flag():
    compiled = compile_typed_stage_graph(typed_stage_graph(df.DATAFLOW_TRANSPORT_HBM))

    assert compiled.plan.comms
    assert {comm.kind for comm in compiled.plan.comms} == {df.DataflowCommKind.HBM_SEND, df.DataflowCommKind.HBM_RECV}


def test_typed_layout_uses_existing_primfunc_lowering_and_records_selection():
    program = (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"window": "lengths"})
        .partial(
            layout_stat(Source="Source"),
            task_args=("batch",),
            range_axis="window",
        )
        .reduce(merge_layout_stat())
        .finalize(store_layout_stat(Output="Output"))
    )
    compiled = df.compile(
        program,
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"window": [32]},
        block_size=32,
        task_extents=(1,),
        include_exit=False,
        mode="inspect",
        inspection_stage="ir",
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
            max_dynamic_shared_memory=232_448,
        ),
    )

    records = compiled.decision_artifact().to_dict()["operation_contracts"]["records"]
    layout_record = next(record for record in records if record["request"]["kind"] == "tensor_layout")
    assert layout_record["selection_state"] == "selected"
    assert layout_record["decisions"][0]["selected_implementation"] == (DATAFLOW_LAYOUT_LINEAR_IMPLEMENTATION)


def test_operation_decision_round_trip_preserves_fallback_rejection_and_resources():
    request = requests()[0]
    rejected = df.DataflowOperationCandidate(
        implementation_id="test.range.large",
        legal=False,
        resources=df.DataflowOperationResourceEstimate(shared_memory_bytes=262_144),
        rejection_reasons=("shared_memory_limit",),
    )
    decision = selected_operation_decision(
        request,
        DATAFLOW_RANGE_STAGE_GRAPH_IMPLEMENTATION,
        resources=df.DataflowOperationResourceEstimate(
            shared_memory_bytes=32_768,
            slot_bytes=4_096,
        ),
        selection_reason="resource_fallback",
        used_fallback=True,
        rejected_candidates=(rejected,),
    )

    restored = df.DataflowOperationDecision.from_dict(decision.to_dict())
    assert restored == decision
    assert restored.used_fallback is True
    assert restored.candidates[0].rejection_reasons == ("shared_memory_limit",)
    assert restored.resources.slot_bytes == 4_096


def test_lowering_boundary_v10_owns_specialization_layout_and_terminal_plans():
    contract = df.DATAFLOW_LOWERING_BOUNDARY_CONTRACT

    assert contract.schema_version == 10
    assert contract.operation_contract_schema_version == (df.DATAFLOW_OPERATION_CONTRACT_SCHEMA_VERSION)
    assert "pipeline_dataflow_physical_binding" in contract.dataflow_outputs
    assert "reshared_transport_plan" in contract.dataflow_outputs
    assert "reshared_transport_physical_binding" in contract.dataflow_outputs
    assert "cross_handler_handoff_plan" in contract.dataflow_outputs
    assert "cross_handler_handoff_runtime_binding" in contract.dataflow_outputs
    assert "typed_execution_request" in contract.dataflow_owned_inputs
    assert "typed_execution_override" in contract.dataflow_owned_inputs
    assert "execution_candidate_plan" in contract.dataflow_outputs
    assert "execution_resource_estimate" in contract.dataflow_outputs
    assert "operator_closure_values" in contract.dataflow_owned_inputs
    assert "typed_operator_physical_contract" in contract.dataflow_owned_inputs
    assert "typed_tensor_argument_layout" in contract.dataflow_owned_inputs
    assert "canonical_specialization_snapshot" in contract.dataflow_outputs
    assert "inferred_reshared_logical_type" in contract.dataflow_outputs
    assert "terminal_no_value_completion" in contract.dataflow_outputs
    assert "transport_source_rank_mapping" in contract.tilelang_owned_inputs
    assert "transport_family_and_physical_implementation" in contract.tilelang_outputs
    with pytest.raises(ValueError, match="Unsupported Dataflow lowering boundary"):
        replace(contract, schema_version=6)
    with pytest.raises(ValueError, match="operation contract metadata is stale"):
        replace(contract, operation_contract_schema_version=0)
