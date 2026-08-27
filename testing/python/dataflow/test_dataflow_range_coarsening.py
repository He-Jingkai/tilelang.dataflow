from __future__ import annotations

from dataclasses import replace
import json

import pytest

import tilelang.language as T
import tilelang.dataflow as df
from tilelang.dataflow.experimental import debug_handlers as dataflow_debug
from tilelang.dataflow.body_ir import lower_operator_call_to_body_ir
from tilelang.dataflow.decision_artifact import (
    validate_dataflow_decision_artifact_payload,
)
from tilelang.dataflow.ir import get_intermediate_type


@T.dataflow_intermediate
class RangeShard:
    value: T.Tensor((2, 8), T.float16)


@T.dataflow_intermediate
class RangeFull:
    value: T.Tensor((8, 8), T.float16)


@T.dataflow_intermediate
class RangeDone:
    value: T.int32


@T.dataflow_intermediate
class RankThreeOutput:
    value: T.Tensor((2, 3, 4), T.float16)


@T.dataflow.map(range=("begin", "end"))
def fused_transform(
    task: T.int32,
    source: T.Tensor((256,), T.float16),
) -> RangeShard:
    raise AssertionError("debug handler compilation must not execute operator bodies")


@T.dataflow.map(range=("begin", "end"))
def consume_transform(
    parts: list[RangeShard],
    task: T.int32,
) -> RangeDone:
    raise AssertionError("debug handler compilation must not execute operator bodies")


@T.dataflow.map(
    range=("begin", "end"),
)
def static_range_variant() -> RangeDone:
    if T.dataflow_range_tiles_per_handler() == 2:
        scratch = T.alloc_fragment((1,), "int32")  # noqa: F841 - asserted source contract
        selected = T.int32(2)
        return RangeDone(value=selected)
    else:
        scratch = T.alloc_fragment((1,), "int32")  # noqa: F841 - asserted source contract
        selected = T.int32(1)
        return RangeDone(value=selected)


@T.dataflow.map(range=("begin", "end"))
def raw_range_tiles_marker() -> RangeDone:
    scratch = T.alloc_fragment((T.dataflow_range_tiles_per_handler(),), "int32")
    for tile in T.serial(T.dataflow_range_tiles_per_handler()):
        scratch[tile] = tile
    return RangeDone(value=T.int32(scratch[0]))


@T.dataflow.map(range=("begin", "end"))
def bare_range_tiles_marker() -> RangeDone:
    scratch = T.alloc_fragment((dataflow_range_tiles_per_handler,), "int32")  # noqa: F821
    return RangeDone(value=T.int32(scratch[0]))


def make_request(*, diagnostic_name: str | None = None):
    return df.DataflowRangeCoarseningRequest(
        logical_tile_extent=16,
        logical_range_extent=64,
        handler_range_extent=32,
        output_tile_arity=2,
        diagnostic_name=diagnostic_name,
    )


def make_program(*, diagnostic_name: str | None = None):
    request = make_request(diagnostic_name=diagnostic_name)
    return (
        T.dataflow_program(
            task_domain=("task",),
            dynamic_ranges={"source_range": "sources", "sink_range": "sinks"},
        )
        .map(
            fused_transform(source="source"),
            name="source_transform",
            task_args=("task",),
            range_axis="source_range",
            range_contract=request,
        )
        .reshared(
            input="source_transform",
            output_type=RangeFull,
            physical_output_type=RangeShard,
            output_arity=4,
            transport_contract=df.DataflowResharedTransportRequest.from_legacy_policy(
                "cluster_shared_all_gather",
                logical_output_arity=8,
                physical_output_arity=4,
            ),
        )
        .map(
            consume_transform(),
            name="sink_transform",
            input="reshared",
            task_args=("task",),
            range_axis="sink_range",
            range_contract=request,
        )
    )


def build_schedule(program=None, *, resource_budget_bytes=None):
    return df.schedule(
        make_program() if program is None else program,
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"source_range": [128], "sink_range": [128]},
        block_size=32,
        task_extents=(1,),
        include_exit=False,
        range_resource_budget_bytes=resource_budget_bytes,
        target_capabilities=df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
            max_cluster_size=2,
        ),
    )


def test_non_moe_stage_graph_fuses_contiguous_tiles_from_one_body():
    plan = build_schedule(resource_budget_bytes=1 << 20)

    assert len(plan.range_coarsening_plans) == 2
    assert {stage_id for stage_id, _ in plan.range_coarsening_plans} == {0, 2}
    for _, range_plan in plan.range_coarsening_plans:
        assert range_plan.logical_tile_count == 4
        assert range_plan.tiles_per_handler == 2
        assert range_plan.handler_count == 2
        assert range_plan.selected_handler_range_extent == 32
        assert not range_plan.used_fallback

    map_instructions = [instruction for instruction in plan.instructions if instruction.opcode is df.DataflowOpcode.MAP]
    assert len(map_instructions) == 8
    assert {instruction.operator_name for instruction in map_instructions} == {
        "fused_transform",
        "consume_transform",
    }
    assert all(
        "range_coarsening_plan_fingerprint" in instruction.attrs and "range_output_mapping" in instruction.attrs
        for instruction in map_instructions
    )
    assert {instruction.task_range.length for instruction in map_instructions} == {32}
    assert {instruction.attrs["range_origin"] for instruction in map_instructions} == {
        0,
        64,
    }


def test_predicated_tail_records_rank_independent_output_layout():
    request = df.DataflowRangeCoarseningRequest(
        logical_tile_extent=24,
        logical_range_extent=70,
        handler_range_extent=48,
        output_tile_arity=2,
        remainder_policy=df.DATAFLOW_REMAINDER_PREDICATE,
    )
    plan = df.plan_range_coarsening(request)

    assert plan.logical_tile_count == 3
    assert plan.handler_count == 2
    assert [mapping.range_begin for mapping in plan.output_mappings] == [0, 48]
    assert [mapping.range_end for mapping in plan.output_mappings] == [48, 70]
    assert [mapping.padded_range_end for mapping in plan.output_mappings] == [48, 96]
    assert plan.output_mappings[1].physical_output_indices == (0,)
    assert plan.output_mappings[1].padding_output_indices == (1,)
    assert plan.output_mappings[1].has_remainder

    intermediate = get_intermediate_type(RankThreeOutput)
    assert intermediate is not None
    assert (
        df.estimate_range_output_bytes_per_tile(
            intermediate,
            output_tile_arity=2,
        )
        == 24
    )


def test_dependence_and_resource_limits_coarsen_without_a_second_function():
    request = df.DataflowRangeCoarseningRequest(
        logical_tile_extent=16,
        logical_range_extent=128,
        handler_range_extent=64,
        output_tile_arity=4,
    )
    accelerated = df.plan_range_coarsening(
        request,
        resource_bytes_per_tile=16 * 1024,
        resource_budget_bytes=128 * 1024,
    )
    portable = df.plan_range_coarsening(
        request,
        resource_bytes_per_tile=16 * 1024,
        resource_budget_bytes=40 * 1024,
    )
    ordered = df.plan_range_coarsening(
        replace(request, dependence=df.DATAFLOW_RANGE_ORDERED),
    )

    assert accelerated.tiles_per_handler == 4
    assert not accelerated.used_fallback
    assert portable.tiles_per_handler == 2
    assert portable.handler_count == 4
    assert portable.selected_resource_bytes == 32 * 1024
    assert portable.fallback_reasons == ("resource_budget",)
    assert ordered.tiles_per_handler == 1
    assert ordered.fallback_reasons == ("dependence_ordered",)


def test_exact_remainder_uses_the_largest_legal_resource_fallback():
    request = df.DataflowRangeCoarseningRequest(
        logical_tile_extent=16,
        logical_range_extent=128,
        handler_range_extent=64,
        output_tile_arity=4,
        remainder_policy=df.DATAFLOW_REMAINDER_EXACT,
    )
    plan = df.plan_range_coarsening(
        request,
        resource_bytes_per_tile=16 * 1024,
        resource_budget_bytes=48 * 1024,
    )

    assert plan.tiles_per_handler == 2
    assert plan.selected_handler_range_extent == 32
    assert plan.handler_count == 4
    assert plan.fallback_reasons == ("resource_budget", "exact_divisibility")
    assert all(not mapping.has_remainder for mapping in plan.output_mappings)


def test_exact_runtime_extent_records_divisibility_fallback():
    request = df.DataflowRangeCoarseningRequest(
        logical_tile_extent=16,
        handler_range_extent=64,
        output_tile_arity=4,
        remainder_policy=df.DATAFLOW_REMAINDER_EXACT,
    )
    plan = df.plan_range_coarsening(request, logical_range_extent=96)

    assert plan.requested_tiles_per_handler == 4
    assert plan.tiles_per_handler == 3
    assert plan.selected_handler_range_extent == 48
    assert plan.handler_count == 2
    assert plan.fallback_reasons == ("exact_divisibility",)


def test_range_plan_deserialization_rejects_type_coercion_and_derived_tampering():
    payload = df.plan_range_coarsening(make_request()).to_dict()

    wrong_type = dict(payload)
    wrong_type["logical_tile_count"] = str(payload["logical_tile_count"])
    with pytest.raises(df.DataflowRangeCoarseningError):
        df.DataflowRangeCoarseningPlan.from_dict(wrong_type)

    forged_mapping = dict(payload)
    forged_mapping["output_mappings"] = [dict(mapping) for mapping in payload["output_mappings"]]
    forged_mapping["output_mappings"][0]["padded_range_end"] += 16
    forged_mapping["output_mappings"][0]["padded_range_extent"] += 16
    forged_mapping["output_mappings"][0]["has_remainder"] = True
    with pytest.raises(ValueError, match="output mappings are inconsistent"):
        df.DataflowRangeCoarseningPlan.from_dict(forged_mapping)

    unknown_reason = dict(payload)
    unknown_reason["fallback_reasons"] = ["model_specific_fallback"]
    with pytest.raises(ValueError, match="unsupported fallback reason"):
        df.DataflowRangeCoarseningPlan.from_dict(unknown_reason)


def test_range_plan_round_trip_and_fingerprint_ignore_diagnostic_name():
    first = df.plan_range_coarsening(make_request(diagnostic_name="first"))
    renamed = df.plan_range_coarsening(make_request(diagnostic_name="renamed"))

    assert first == renamed
    assert first.fingerprint == renamed.fingerprint
    assert df.DataflowRangeCoarseningPlan.from_dict(first.to_dict()) == first

    stale = first.to_dict()
    stale["handler_count"] += 1
    with pytest.raises(ValueError):
        df.DataflowRangeCoarseningPlan.from_dict(stale)


def test_raw_body_specialization_consumes_selected_range_plan_value():
    body = lower_operator_call_to_body_ir(
        static_range_variant(),
        "map",
        specialization_constants={
            df.DATAFLOW_RANGE_TILES_PER_HANDLER_SPECIALIZATION: 2,
        },
    )

    assert body.tilelang_body == (
        'scratch = T.alloc_fragment((1,), "int32")',
        "selected = T.int32(2)",
    )


def test_range_tiles_marker_is_replaced_with_selected_plan_literal():
    body = lower_operator_call_to_body_ir(
        raw_range_tiles_marker(),
        "map",
        specialization_constants={
            df.DATAFLOW_RANGE_TILES_PER_HANDLER_SPECIALIZATION: 2,
        },
    )

    assert body.tilelang_body == (
        "scratch = T.alloc_fragment((2,), 'int32')",
        "for tile in T.serial(2):\n    scratch[tile] = tile",
    )
    assert all("dataflow_range_tiles_per_handler" not in statement for statement in body.tilelang_body)


def test_range_tiles_marker_requires_compiler_selected_plan():
    with pytest.raises(
        NotImplementedError,
        match="requires a selected range coarsening plan",
    ):
        lower_operator_call_to_body_ir(raw_range_tiles_marker(), "map")


def test_range_tiles_marker_rejects_legacy_bare_identifier():
    with pytest.raises(
        NotImplementedError,
        match="bare dataflow_range_tiles_per_handler is unsupported",
    ):
        lower_operator_call_to_body_ir(
            bare_range_tiles_marker(),
            "map",
            specialization_constants={
                df.DATAFLOW_RANGE_TILES_PER_HANDLER_SPECIALIZATION: 2,
            },
        )


def test_range_tiles_marker_rejects_direct_python_execution():
    with pytest.raises(RuntimeError, match="only valid inside a lowered Dataflow handler"):
        T.dataflow_range_tiles_per_handler()


def test_decision_artifact_records_selected_range_plans_and_resources():
    compiled = df.compile(
        make_program(diagnostic_name="artifact-only"),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"source_range": [128], "sink_range": [128]},
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
    artifact = compiled.decision_artifact().to_dict()
    records = [record for record in artifact["operation_contracts"]["records"] if record["request"]["kind"] == "range_coarsening"]

    assert artifact["scheduler"]["range_resource_budget_bytes"] == 232_448
    assert len(records) == 2
    assert all(record["selection_state"] == "selected" for record in records)
    assert all(len(record["lowering_plans"]) == 1 for record in records)
    assert all(
        record["decisions"][0]["selection_reason"] == "typed_range_coarsening_plan" and not record["decisions"][0]["used_fallback"]
        for record in records
    )

    tampered = json.loads(compiled.decision_artifact().canonical_json)
    range_record = next(record for record in tampered["operation_contracts"]["records"] if record["request"]["kind"] == "range_coarsening")
    decision = df.DataflowOperationDecision.from_dict(range_record["decisions"][0])
    range_record["decisions"][0] = replace(
        decision,
        used_fallback=True,
        selection_reason="range_resource_fallback",
    ).to_dict()
    with pytest.raises(ValueError, match="decision changed its plan"):
        validate_dataflow_decision_artifact_payload(tampered)
