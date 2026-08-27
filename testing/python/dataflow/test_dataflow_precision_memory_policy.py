from __future__ import annotations

import pytest

import tilelang.language as T
import tilelang.dataflow as df
from tilelang.dataflow.experimental import debug_handlers as dataflow_debug


def make_precision_program(
    *,
    allowed_dtypes=("float16", "float32"),
    strict_dtype="float32",
    minimum_dtype="float16",
):
    accumulator_dtype = strict_dtype
    constants = {"accumulator_dtype": accumulator_dtype}

    @T.dataflow_intermediate(specialization_constants=constants)
    class Partial:
        value: T.Tensor((1,), T.float32)

    @T.dataflow.iter(
        range=("begin", "end"),
        specialization_constants=constants,
        physical_contract=df.DataflowOperatorPhysicalContract(
            output_slot=df.DATAFLOW_OUTPUT_SLOT_DIRECT,
        ),
    )
    def split(Source: T.Tensor((64,), T.float32)) -> Partial:
        value = T.alloc_shared((1,), "float32")
        value[0] = T.float32(0)
        for index in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
            value[0] += Source[index]
        return Partial(value=value)

    @T.dataflow.reduce(
        associative=True,
        specialization_constants=constants,
        physical_contract=df.DataflowOperatorPhysicalContract(
            output_slot=df.DATAFLOW_OUTPUT_SLOT_DIRECT,
        ),
        accumulator_contracts={
            "accumulator_dtype": df.DataflowAccumulatorContract(
                contract_id="synthetic_sum",
                allowed_dtypes=allowed_dtypes,
                strict_dtype=strict_dtype,
                minimum_dtype=minimum_dtype,
                reduction_order="scheduler_plan_associative_binary",
            ),
        },
    )
    def reduce(left: Partial, right: Partial) -> Partial:
        accumulator = T.alloc_shared((1,), accumulator_dtype)
        value = T.alloc_shared((1,), "float32")
        accumulator[0] = T.cast(left.value[0], accumulator_dtype)
        accumulator[0] += T.cast(right.value[0], accumulator_dtype)
        value[0] = T.cast(accumulator[0], "float32")
        return Partial(value=value)

    @T.dataflow.finalize(specialization_constants=constants)
    def finalize(
        partial: Partial,
        batch: T.int32,
        Output: T.Tensor((1,), T.float32),
    ) -> None:
        Output[batch] = partial.value[0]

    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"items": "lengths"})
        .partial(split(Source="Source"), task_args=("batch",), range_axis="items")
        .reduce(reduce())
        .finalize(finalize(Output="Output"))
    )


def compile_debug(program, *, semantic_config=None, memory_policy=None):
    return df.compile(
        program,
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"items": (64,)},
        block_size=32,
        task_extents=(1,),
        semantic_config=semantic_config,
        memory_policy=memory_policy,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
            max_dynamic_shared_memory=232_448,
        ),
    )


def compile_inspect(program, *, semantic_config=None):
    return df.compile(
        program,
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"items": (64,)},
        block_size=32,
        task_extents=(1,),
        semantic_config=semantic_config,
        memory_policy="shared",
        mode="inspect",
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
            max_dynamic_shared_memory=232_448,
        ),
    )


def test_precision_policy_resolves_strict_fast_and_explicit_without_workload_tolerance_registry():
    program = make_precision_program()
    strict = df.resolve_program_precision(program, df.DataflowPrecisionPolicy())
    fast_policy = df.DataflowPrecisionPolicy(
        mode="fast",
        error_budget=df.DataflowErrorBudget(
            max_absolute_error=2e-3,
            max_relative_error=1e-2,
        ),
    )
    fast = df.resolve_program_precision(program, fast_policy)
    explicit = df.resolve_program_precision(
        program,
        df.DataflowPrecisionPolicy(
            mode="explicit",
            accumulator_dtype="float32",
        ),
    )

    assert strict.resolutions[0].selected_dtype == "float32"
    assert fast.resolutions[0].selected_dtype == "float16"
    assert fast.resolutions[0].error_budget == fast_policy.error_budget
    assert explicit.resolutions[0].selected_dtype == "float32"
    assert "tolerance" not in df.DataflowDTypeInfo.__dataclass_fields__
    with pytest.raises(ValueError, match="only valid for mode='fast'"):
        df.DataflowPrecisionPolicy(
            mode="strict",
            error_budget=df.DataflowErrorBudget(max_absolute_error=1e-3),
        )


def test_precision_policy_supports_a_second_non_mla_contract_and_fails_closed():
    program = make_precision_program(
        allowed_dtypes=("float32",),
        strict_dtype="float32",
        minimum_dtype="float32",
    )
    fast = df.resolve_program_precision(
        program,
        df.DataflowPrecisionPolicy(
            mode="fast",
            error_budget=df.DataflowErrorBudget(max_relative_error=1e-6),
        ),
    )

    assert fast.resolutions[0].selected_dtype == "float32"
    with pytest.raises(ValueError, match="requires a non-zero explicit error_budget"):
        df.DataflowPrecisionPolicy(mode="fast")
    with pytest.raises(ValueError, match="is not allowed by contract"):
        df.resolve_program_precision(
            program,
            df.DataflowPrecisionPolicy(
                mode="explicit",
                accumulator_dtype="float16",
            ),
        )


def test_precision_plan_enters_compile_and_artifact_fingerprints():
    program = make_precision_program()
    strict = compile_debug(program)
    explicit = compile_debug(
        program,
        semantic_config=df.DataflowSemanticConfig(
            precision=df.DataflowPrecisionPolicy(
                mode="explicit",
                accumulator_dtype="float16",
            )
        ),
    )

    assert strict.compile_config.fingerprint != explicit.compile_config.fingerprint
    assert strict.artifact_fingerprint != explicit.artifact_fingerprint
    strict_precision = strict.decision_artifact().to_dict()["lowerings"]["precision"]
    explicit_precision = explicit.decision_artifact().to_dict()["lowerings"]["precision"]
    assert strict_precision["plan"]["resolutions"][0]["selected_dtype"] == "float32"
    assert explicit_precision["plan"]["resolutions"][0]["selected_dtype"] == "float16"


def test_selected_accumulator_dtype_is_applied_during_primfunc_lowering():
    program = make_precision_program()
    strict = compile_inspect(program)
    fast = compile_inspect(
        program,
        semantic_config=df.DataflowSemanticConfig(
            precision=df.DataflowPrecisionPolicy(
                mode="fast",
                error_budget=df.DataflowErrorBudget(max_relative_error=1e-2),
            )
        ),
    )

    def binary_reduce_script(compiled):
        return next(
            handler.prim_func.script()
            for handler in compiled.primfunc_lowering.handlers
            if handler.operator_kind == "reduce" and handler.handler_variant_key.reduce_arity.arity_class == "binary"
        )

    strict_script = binary_reduce_script(strict)
    fast_script = binary_reduce_script(fast)
    assert '"float16"' not in strict_script
    assert '"float16"' in fast_script
    assert strict_script != fast_script


def test_memory_policy_selects_profitable_legal_candidate_and_records_rejections():
    policy = df.DataflowMemoryPolicy()
    shared = df.make_memory_candidate(
        candidate_id="shared",
        placement="shared",
        shared_memory_bytes=300_000,
        shared_slot_bytes=100_000,
        primfunc_scratch_bytes=200_000,
        target_shared_memory_limit=232_448,
    )
    scratch = df.make_memory_candidate(
        candidate_id="scratch_backed",
        placement="scratch_backed",
        shared_memory_bytes=220_000,
        shared_slot_bytes=0,
        primfunc_scratch_bytes=220_000,
        scratch_backed_slot_count=4,
        scratch_backed_slot_bytes=100_000,
        target_shared_memory_limit=232_448,
    )

    plan = df.select_memory_plan(policy, (shared, scratch))
    assert plan is not None
    assert plan.selected_candidate.placement == "scratch_backed"
    assert not shared.legal
    assert "target_shared_memory_limit_exceeded" in shared.rejection_reasons[0]


def test_memory_policy_defers_to_hbm_and_explicit_modes_fail_closed():
    shared = df.make_memory_candidate(
        candidate_id="shared",
        placement="shared",
        shared_memory_bytes=300_000,
        shared_slot_bytes=100_000,
        primfunc_scratch_bytes=200_000,
        target_shared_memory_limit=232_448,
    )
    assert (
        df.select_memory_plan(
            df.DataflowMemoryPolicy(),
            (shared,),
            allow_deferred_hbm=True,
        )
        is None
    )
    with pytest.raises(df.DataflowMemoryPlanningError, match="is illegal"):
        df.select_memory_plan(
            df.DataflowMemoryPolicy(mode="shared"),
            (shared,),
        )

    hbm = df.make_memory_candidate(
        candidate_id="hbm_direct_global",
        placement="hbm_direct_global",
        shared_memory_bytes=180_000,
        shared_slot_bytes=0,
        primfunc_scratch_bytes=180_000,
        hbm_direct_global_slot_count=4,
        target_shared_memory_limit=232_448,
    )
    plan = df.select_memory_plan(df.DataflowMemoryPolicy(), (shared, hbm))
    assert plan is not None
    assert plan.selected_candidate.placement == "hbm_direct_global"
    assert plan.selection_reason == "shared_and_scratch_illegal_hbm_fallback"
