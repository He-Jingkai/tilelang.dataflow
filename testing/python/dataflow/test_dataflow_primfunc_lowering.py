from __future__ import annotations

from dataclasses import replace
import random

import pytest

from tilelang import _ffi_api
import tilelang.language as T
import tilelang.dataflow as df
import tilelang.dataflow.body_ir as body_ir_module
import tilelang.dataflow.compiler as compiler_module
import tilelang.dataflow.primfunc_lowering as primfunc_lowering_module
import tilelang.dataflow.scheduler as scheduler_module
from tilelang.layout import make_wgmma_swizzled_layout
from tilelang.dataflow.experimental import debug_handlers as dataflow_debug
from tvm import ir, tir
from tvm.tir.stmt_functor import post_order_visit

from tilelang.dataflow.body_ir import (
    AssignIR,
    AugAssignIR,
    BinaryOpIR,
    CallIR,
    CastIR,
    FieldElementAccessIR,
    FieldAccessIR,
    LiteralIR,
    DataflowAccumulatorIR,
    DataflowLoopIR,
    ScalarVarIR,
    TensorLoadIR,
    TensorStoreIR,
    TupleExprIR,
    DataflowBodyIRLoweringError,
    lower_operator_call_to_body_ir,
)


def test_public_handler_lowering_surface_only_exposes_primfunc():
    import importlib

    assert df.SUPPORTED_HANDLER_LOWERINGS == (df.PRIMFUNC_HANDLER_LOWERING,)
    assert not hasattr(df, "TIR_U32_HANDLER_LOWERING")
    assert not hasattr(df, "SCALAR_U32_DEBUG_HANDLER_LOWERING")
    assert not hasattr(df, "TENSOR_U32_DEBUG_HANDLER_LOWERING")
    # A stale scikit-build editable finder may still remember the removed source
    # path and report FileNotFoundError instead of ModuleNotFoundError.  Either
    # exception proves that the retired lowering is no longer importable.
    with pytest.raises((ModuleNotFoundError, FileNotFoundError)):
        importlib.import_module("tilelang.dataflow.tir_lowering")


def test_static_parallel_extent_legalizer_pads_nondivisible_work():
    source = """for row, col in T.Parallel(rows, cols):
    Output[row, col] = Input[row, col]
"""

    legalized = primfunc_lowering_module.legalize_static_parallel_extents_source(
        source,
        {"rows": 32, "cols": 256},
        thread_count=384,
    )

    assert "T.Parallel(rows, 264)" in legalized
    assert "if col < 256:" in legalized
    assert "Output[row, col] = Input[row, col]" in legalized

    assert (
        primfunc_lowering_module.legalize_static_parallel_extents_source(
            source,
            {"rows": 32, "cols": 256},
            thread_count=256,
        )
        == source
    )


def test_thread_limited_linking_detects_cluster_collectives_structurally():
    @T.prim_func
    def cluster_collective():
        with T.Kernel(1, threads=128):
            T.cluster_sync()

    @T.prim_func
    def cta_collective():
        with T.Kernel(1, threads=128):
            T.sync_threads()

    assert primfunc_lowering_module.prim_func_has_cluster_collective(cluster_collective)
    assert not primfunc_lowering_module.prim_func_has_cluster_collective(cta_collective)


@T.dataflow_intermediate
class ScalarInter:
    value: T.int32


@T.dataflow_intermediate
class StatsInter:
    total: T.int32
    maximum: T.int32


@T.dataflow_intermediate
class ScoreInter:
    score: T.int32


@T.dataflow_intermediate
class TensorFieldInter:
    score: T.int32
    vec: T.Tensor((2,), T.int32)


@T.dataflow_intermediate
class SlicedTensorInter:
    vec: T.Tensor((2, 3), T.int32)


@T.dataflow_intermediate
class TinyMLAPartial:
    m: T.float32
    l: T.float32
    o: T.Tensor((2,), T.float32)


@T.dataflow.iter(range=("begin", "end"))
def split_scalar(batch: T.int32, Source: T.Tensor((16,), T.int32)) -> ScalarInter:
    value = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += Source[i]
    return ScalarInter(value=value)


@T.dataflow.iter(range=("begin", "end"))
def split_scalar_alt(A: T.Tensor((32,), T.int32)) -> ScalarInter:
    total = T.int32(1)
    for k in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        total += A[k] * T.int32(2)
    return ScalarInter(value=total)


@T.dataflow.iter(range=("begin", "end"))
def split_stats(Source: T.Tensor((16,), T.int32)) -> StatsInter:
    total = T.int32(0)
    maximum = T.int32(-2147483647)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        total += Source[i]
        maximum = T.max(maximum, Source[i])
    return StatsInter(total=total, maximum=maximum)


@T.dataflow.iter(range=("begin", "end"))
def split_by_head(head: T.int32, Source: T.Tensor((2, 64), T.int32)) -> ScalarInter:
    value = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += Source[head, i]
    return ScalarInter(value=value)


@T.dataflow.iter(range=("begin", "end"), threads=128)
def split_bounded_transfer(
    batch: T.int32,
    Source: T.Tensor((2, 64), T.float16),
) -> ScalarInter:
    range_begin = T.cast(T.dataflow_range_begin(), "int32")
    range_end = T.cast(T.dataflow_range_end(), "int32")
    Scratch = T.alloc_shared((16,), "float16")
    T.copy(
        Source[batch, range_begin : range_begin + 16],
        Scratch,
        valid_region=Source[batch, 0:range_end],
        oob_fill=0,
        allow_async=False,
    )
    return ScalarInter(value=T.int32(Scratch[0]))


@T.dataflow.reduce
def reduce_scalar(items: list[ScalarInter]) -> ScalarInter:
    value = T.int32(0)
    for item in items:
        value += item.value
    return ScalarInter(value=value)


@T.dataflow.reduce(associative=True)
def associative_sum_scalar(left: ScalarInter, right: ScalarInter) -> ScalarInter:
    return ScalarInter(value=left.value + right.value)


@T.dataflow.reduce(associative=True)
def associative_max_scalar(left: ScalarInter, right: ScalarInter) -> ScalarInter:
    return ScalarInter(value=T.max(left.value, right.value))


@T.dataflow.reduce(associative=True)
def associative_stats(left: StatsInter, right: StatsInter) -> StatsInter:
    return StatsInter(
        total=left.total + right.total,
        maximum=T.max(left.maximum, right.maximum),
    )


@T.dataflow.reduce(associative=True)
def associative_affine(left: StatsInter, right: StatsInter) -> StatsInter:
    return StatsInter(
        total=left.total * right.total,
        maximum=left.total * right.maximum + left.maximum,
    )


@T.dataflow.finalize
def finalize_scalar(inter: ScalarInter, Output: T.Tensor((2,), T.int32)) -> None:
    Output[T.dataflow_task_id()] = inter.value


@T.dataflow.finalize
def finalize_stats(
    inter: StatsInter,
    Total: T.Tensor((1,), T.int32),
    Maximum: T.Tensor((1,), T.int32),
) -> None:
    Total[T.dataflow_task_id()] = inter.total
    Maximum[T.dataflow_task_id()] = inter.maximum


@T.dataflow.finalize
def finalize_scalar_by_coords(
    inter: ScalarInter,
    batch: T.int32,
    head: T.int32,
    Output: T.Tensor((2, 2), T.int32),
) -> None:
    Output[batch, head] = inter.value


@T.dataflow.finalize
def finalize_unannotated_output(inter: ScalarInter, Output) -> None:
    Output[T.dataflow_task_id()] = inter.value


@T.dataflow.iter(range=("begin", "end"))
def split_wrong_return_field(Source) -> ScalarInter:
    value = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += Source[i]
    return ScalarInter(wrong=value)


@T.dataflow.iter(range=("begin", "end"))
def split_positional_return(Source) -> ScalarInter:
    value = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += Source[i]
    return ScalarInter(value)


@T.dataflow.iter(range=("begin", "end"))
def split_unbound_accumulator_init(Source) -> ScalarInter:
    value = T.int32(offset)  # noqa: F821 - deliberately invalid lowering input
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += Source[i]
    return ScalarInter(value=value)


@T.dataflow.iter(range=("begin", "end"))
def split_unbound_scalar(Source) -> ScalarInter:
    value = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += Source[i] + offset  # noqa: F821 - deliberately invalid lowering input
    return ScalarInter(value=value)


@T.dataflow.iter(range=("begin", "end"))
def split_for_else(Source) -> ScalarInter:
    value = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += Source[i]
    else:
        value += T.int32(1)
    return ScalarInter(value=value)


@T.dataflow.iter(range=("begin", "end"))
def split_unannotated_tensor(Source) -> ScalarInter:
    value = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += Source[i]
    return ScalarInter(value=value)


@T.dataflow.reduce
def reduce_wrong_field_base(items: list[ScalarInter]) -> ScalarInter:
    value = T.int32(0)
    for _item in items:
        value += other.value  # noqa: F821 - deliberately invalid lowering input
    return ScalarInter(value=value)


@T.dataflow.reduce
def reduce_wrong_field_name(items: list[ScalarInter]) -> ScalarInter:
    value = T.int32(0)
    for item in items:
        value += item.missing
    return ScalarInter(value=value)


@T.dataflow.reduce
def reduce_cast_scalar_field(items: list[ScalarInter]) -> ScalarInter:
    value = T.int32(0)
    for item in items:
        value += T.int32(item.value)
    return ScalarInter(value=value)


@T.dataflow.finalize
def finalize_wrong_index(inter: ScalarInter, Output) -> None:
    Output[0] = inter.value


@T.dataflow.finalize
def finalize_undeclared_output(inter: ScalarInter, Output) -> None:
    Missing[T.dataflow_task_id()] = inter.value  # noqa: F821 - deliberately invalid lowering input


@T.dataflow.finalize
def finalize_intermediate_target(inter: ScalarInter, Output) -> None:
    inter[T.dataflow_task_id()] = inter.value


@T.dataflow.finalize
def finalize_wrong_field_name(inter: ScalarInter, Output) -> None:
    Output[T.dataflow_task_id()] = inter.missing


@T.dataflow.iter(range=("begin", "end"))
def split_score(Source: T.Tensor((16,), T.int32)) -> ScoreInter:
    score = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        score += Source[i]
    return ScoreInter(score=score)


@T.dataflow.reduce
def reduce_score(items: list[ScoreInter]) -> ScoreInter:
    score = T.int32(0)
    for item in items:
        score += item.score
    return ScoreInter(score=score)


@T.dataflow.finalize
def finalize_score(inter: ScoreInter, Output: T.Tensor((1,), T.int32)) -> None:
    Output[T.dataflow_task_id()] = inter.score


@T.dataflow.iter(range=("begin", "end"))
def split_tensor_field(A: T.Tensor((64,), T.int32), B: T.Tensor((64,), T.int32)) -> TensorFieldInter:
    score = T.int32(0)
    v0 = T.int32(0)
    v1 = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        score += A[i]
        v0 += A[i] + B[i]
        v1 += A[i] * T.int32(2) + B[i]
    return TensorFieldInter(score=score, vec=(v0, v1))


@T.dataflow.iter(range=("begin", "end"))
def tiny_mla_split(
    head: T.int32,
    Q: T.Tensor((2, 2), T.float32),
    KV: T.Tensor((64, 2), T.float32),
) -> TinyMLAPartial:
    m = T.float32(-3.402823e38)
    l = T.float32(0)
    o0 = T.float32(0)
    o1 = T.float32(0)
    for kv in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        o0 = o0 * T.exp2(m - T.max(m, Q[head, 0] * KV[kv, 0])) + KV[kv, 0] * T.exp2(
            Q[head, 0] * KV[kv, 0] - T.max(m, Q[head, 0] * KV[kv, 0])
        )
        o1 = o1 * T.exp2(m - T.max(m, Q[head, 0] * KV[kv, 0])) + KV[kv, 1] * T.exp2(
            Q[head, 0] * KV[kv, 0] - T.max(m, Q[head, 0] * KV[kv, 0])
        )
        l = l * T.exp2(m - T.max(m, Q[head, 0] * KV[kv, 0])) + T.exp2(Q[head, 0] * KV[kv, 0] - T.max(m, Q[head, 0] * KV[kv, 0]))
        m = T.max(m, Q[head, 0] * KV[kv, 0])
    return TinyMLAPartial(m=m, l=l, o=(o0, o1))


@T.dataflow.reduce
def reduce_tensor_field(items: list[TensorFieldInter]) -> TensorFieldInter:
    score = T.int32(0)
    v0 = T.int32(0)
    v1 = T.int32(0)
    for item in items:
        score += item.score
        v0 += item.vec[0]
        v1 += item.vec[1]
    return TensorFieldInter(score=score, vec=(v0, v1))


@T.dataflow.reduce
def tiny_mla_reduce(items: list[TinyMLAPartial]) -> TinyMLAPartial:
    m = T.float32(-3.402823e38)
    l = T.float32(0)
    o0 = T.float32(0)
    o1 = T.float32(0)
    for item in items:
        o0 = o0 * T.exp2(m - T.max(m, item.m)) + item.o[0] * T.exp2(item.m - T.max(m, item.m))
        o1 = o1 * T.exp2(m - T.max(m, item.m)) + item.o[1] * T.exp2(item.m - T.max(m, item.m))
        l = l * T.exp2(m - T.max(m, item.m)) + item.l * T.exp2(item.m - T.max(m, item.m))
        m = T.max(m, item.m)
    return TinyMLAPartial(m=m, l=l, o=(o0, o1))


@T.dataflow.reduce(associative=True, threads=32)
def associative_tiny_mla_reduce(
    left: TinyMLAPartial,
    right: TinyMLAPartial,
) -> TinyMLAPartial:
    merged_m = T.alloc_shared((1,), "float32")
    merged_l = T.alloc_shared((1,), "float32")
    merged_o = T.alloc_shared((2,), "float32")
    left_scale = T.alloc_shared((1,), "float32")
    right_scale = T.alloc_shared((1,), "float32")
    for h in T.Parallel(1):
        merged_m[h] = T.max(left.m, right.m)
        left_scale[h] = T.exp2(left.m - merged_m[h])
        right_scale[h] = T.exp2(right.m - merged_m[h])
        merged_l[h] = left.l * left_scale[h] + right.l * right_scale[h]
    T.sync_threads()
    for d in T.Parallel(2):
        merged_o[d] = left.o[d] * left_scale[0] + right.o[d] * right_scale[0]
    T.sync_threads()
    return TinyMLAPartial(m=merged_m[0], l=merged_l[0], o=merged_o)


@T.dataflow.finalize
def finalize_tensor_field(
    inter: TensorFieldInter,
    ScoreOut: T.Tensor((2,), T.int32),
    Vec0Out: T.Tensor((2,), T.int32),
    Vec1Out: T.Tensor((2,), T.int32),
) -> None:
    ScoreOut[T.dataflow_task_id()] = inter.score
    Vec0Out[T.dataflow_task_id()] = inter.vec[0]
    Vec1Out[T.dataflow_task_id()] = inter.vec[1]


@T.dataflow.finalize
def tiny_mla_finalize(
    inter: TinyMLAPartial,
    Out0: T.Tensor((2,), T.float32),
    Out1: T.Tensor((2,), T.float32),
) -> None:
    Out0[T.dataflow_task_id()] = inter.o[0] / inter.l
    Out1[T.dataflow_task_id()] = inter.o[1] / inter.l


def make_scalar_program(split_op=split_scalar, reduce_op=reduce_scalar):
    iter_call = split_op(A="A") if split_op is split_scalar_alt else split_op(Source="Source")
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            iter_call,
            task_args=("batch",),
            range_axis="kv",
        )
        .reduce(reduce_op())
        .finalize(finalize_scalar(Output="Output"))
    )


def make_stats_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(split_stats(Source="Source"), task_args=("batch",), range_axis="kv")
        .reduce(associative_stats())
        .finalize(finalize_stats(Total="Total", Maximum="Maximum"))
    )


def make_bounded_transfer_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_bounded_transfer(Source="Source"),
            task_args=("batch",),
            range_axis="kv",
        )
        .reduce(associative_sum_scalar())
        .finalize(finalize_scalar(Output="Output"))
    )


def test_associative_reduce_generates_passthrough_binary_and_ordered_left_fold():
    compiled = df.compile(
        make_scalar_program(reduce_op=associative_sum_scalar),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [4, 8, 12]},
        task_extents=(3,),
        block_size=4,
        mode="debug",
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
        ),
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )
    reduce_handlers = {
        handler.handler_variant_key.reduce_arity.arity_class: handler for handler in result.handlers if handler.operator_kind == "reduce"
    }
    assert set(reduce_handlers) == {"passthrough", "binary", "generic"}

    passthrough = reduce_handlers["passthrough"]
    passthrough_script = passthrough.prim_func.script()
    assert passthrough.reduce_input_slot_count == 1
    assert "Item0" in passthrough_script
    assert "Item1" not in passthrough_script
    assert "input_count >" not in passthrough_script
    passthrough_instruction = next(
        instruction for instruction in compiled.plan.instructions if instruction.handler_variant_key == passthrough.handler_variant_key
    )
    assert passthrough_instruction.value_forward is not None
    assert passthrough_instruction.value_forward.source_slot_id == (passthrough_instruction.input_slots[0])
    assert passthrough_instruction.value_forward.output_slot_id == (passthrough_instruction.output_slot)

    binary = reduce_handlers["binary"]
    binary_script = binary.prim_func.script()
    assert binary.reduce_input_slot_count == 2
    assert "Item0[0] + Item1[0]" in binary_script
    assert "Merge" not in binary_script
    assert "input_count >" not in binary_script

    generic = reduce_handlers["generic"]
    generic_script = generic.prim_func.script()
    assert generic.reduce_input_slot_count == 3
    assert "Merge" in generic_script
    assert 'if T.Cast("int32", input_count) > 1:' in generic_script
    assert 'if T.Cast("int32", input_count) > 2:' in generic_script


def test_associative_reduce_links_variant_specific_wrapper_adapters():
    compiled = df.compile(
        make_scalar_program(reduce_op=associative_sum_scalar),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [4, 8, 12]},
        task_extents=(3,),
        block_size=4,
        include_exit=False,
        mode="executable",
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
        ),
    )

    sources = {
        handler.handler_variant_key.reduce_arity.arity_class: next(
            source.body_source for source in compiled.wrapper_spec.handler_sources if source.handler_id == handler.handler_id
        )
        for handler in compiled.wrapper_spec.handlers
        if handler.operator_kind == "reduce"
    }
    assert set(sources) == {"passthrough", "binary", "generic"}
    assert "if (output_value != reduce_items)" in sources["passthrough"]
    assert "handler_args.input_slot_count != 2u" in sources["binary"]
    assert "handler_args.input_slot_count < 3u" in sources["generic"]
    assert "handler_args.input_slot_count > 3u" in sources["generic"]


def test_associative_max_and_structured_reducers_use_the_generic_contract():
    target = df.TargetCapabilitySnapshot.for_cuda(
        (9, 0),
        compiler_version=(12, 8),
    )
    max_compiled = df.compile(
        make_scalar_program(reduce_op=associative_max_scalar),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [8]},
        task_extents=(1,),
        block_size=4,
        mode="debug",
        target_override=target,
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )
    max_lowering = df.lower_program_handlers_to_primfuncs(
        max_compiled.program,
        max_compiled.wrapper_spec,
        max_compiled.tensor_arg_plan,
        plan=max_compiled.plan,
        lower_to_cuda=False,
    )
    max_script = next(handler.prim_func.script() for handler in max_lowering.handlers if handler.operator_kind == "reduce")
    assert "T.max(Item0[0], Item1[0])" in max_script

    stats_compiled = df.compile(
        make_stats_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [8]},
        task_extents=(1,),
        block_size=4,
        mode="debug",
        target_override=target,
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )
    stats_lowering = df.lower_program_handlers_to_primfuncs(
        stats_compiled.program,
        stats_compiled.wrapper_spec,
        stats_compiled.tensor_arg_plan,
        plan=stats_compiled.plan,
        lower_to_cuda=False,
    )
    stats_script = next(handler.prim_func.script() for handler in stats_lowering.handlers if handler.operator_kind == "reduce")
    assert "Item0_total[0] + Item1_total[0]" in stats_script
    assert "T.max(Item0_maximum[0], Item1_maximum[0])" in stats_script
    assert "Out_total[0] = _dataflow_merge_total_0[0]" in stats_script
    assert "Out_maximum[0] = _dataflow_merge_maximum_0[0]" in stats_script


def test_associative_expression_materializes_all_fields_before_aliasable_output():
    program = (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(split_stats(Source="Source"), task_args=("batch",), range_axis="kv")
        .reduce(associative_affine())
        .finalize(finalize_stats(Total="Total", Maximum="Maximum"))
    )
    compiled = df.compile(
        program,
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"kv": [16]},
        task_extents=(1,),
        block_size=4,
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        mode="debug",
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
        ),
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )
    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )
    binary_script = next(
        handler.prim_func.script()
        for handler in result.handlers
        if handler.operator_kind == "reduce" and handler.handler_variant_key.reduce_arity.arity_class == "binary"
    )

    total_store = binary_script.index("Out_total[0] =")
    maximum_store = binary_script.index("Out_maximum[0] =")
    assert "Item0_total[0] * Item1_total[0]" in binary_script[:total_store]
    assert "Item0_total[0] * Item1_maximum[0] + Item0_maximum[0]" in binary_script[:total_store]
    assert total_store < maximum_store


def test_associative_passthrough_uses_copy_for_distinct_storage():
    compiled = df.compile(
        make_scalar_program(reduce_op=associative_sum_scalar),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [4]},
        task_extents=(1,),
        block_size=4,
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming",
        mode="executable",
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
        ),
    )

    reductions = compiled.decision_artifact().to_dict()["lowerings"]["reduction"]["instructions"]
    assert len(reductions) == 1
    forward = reductions[0]["value_forward"]
    assert reductions[0]["arity_class"] == "passthrough"
    assert reductions[0]["reduction_order"] == "value_forward"
    assert forward["selected_storage"] == "copy"
    assert forward["allocation_owner_slot_id"] == forward["output_slot_id"]
    assert compiled.validate_memory_layout().errors == ()


def test_value_forward_planner_aliases_only_preexisting_storage_identity():
    compiled = df.compile(
        make_scalar_program(reduce_op=associative_sum_scalar),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [4]},
        task_extents=(1,),
        block_size=4,
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming",
        mode="debug",
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
        ),
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )
    passthrough = next(instruction for instruction in compiled.plan.instructions if instruction.value_forward is not None)
    source_slot_id = passthrough.value_forward.source_slot_id
    output_slot_id = passthrough.value_forward.output_slot_id
    shared_storage_id = (
        max(
            (slot.shared_storage_id for slot in compiled.plan.slots if slot.shared_storage_id is not None),
            default=-1,
        )
        + 1
    )
    global_storage_id = (
        max(
            (slot.global_storage_id for slot in compiled.plan.slots if slot.global_storage_id is not None),
            default=-1,
        )
        + 1
    )
    colocated_plan = replace(
        compiled.plan,
        slots=tuple(
            replace(
                slot,
                shared_storage_id=shared_storage_id,
                global_storage_id=global_storage_id,
            )
            if slot.slot_id in {source_slot_id, output_slot_id}
            else slot
            for slot in compiled.plan.slots
        ),
    )

    aliased_plan = scheduler_module.plan_value_forward_storage(colocated_plan)
    aliased_instruction = next(
        instruction for instruction in aliased_plan.instructions if instruction.instruction_id == passthrough.instruction_id
    )
    slots_by_id = {slot.slot_id: slot for slot in aliased_plan.slots}

    assert aliased_instruction.value_forward.alias_owner_slot_id == source_slot_id
    assert slots_by_id[output_slot_id].alias_of_slot_id == source_slot_id
    assert slots_by_id[output_slot_id].allocation_owner_slot_id == source_slot_id


def test_raw_associative_online_merge_generates_binary_and_generic_variants():
    compiled = df.compile(
        make_tiny_mla_program(reduce_op=associative_tiny_mla_reduce),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [4, 8, 12]},
        task_extents=(3,),
        block_size=4,
        mode="debug",
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
        ),
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )
    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )
    reduce_handlers = {
        handler.handler_variant_key.reduce_arity.arity_class: handler for handler in result.handlers if handler.operator_kind == "reduce"
    }

    binary_script = reduce_handlers["binary"].prim_func.script()
    assert "Item0_m[0]" in binary_script
    assert "Item1_m[0]" in binary_script
    assert "Out_o[d]" in binary_script
    assert "merged_o = T.alloc_buffer" not in binary_script
    assert "input_count >" not in binary_script

    generic_script = reduce_handlers["generic"].prim_func.script()
    assert "Item2_m[0]" in generic_script
    assert 'if T.Cast("int32", input_count) > 1:' in generic_script
    assert 'if T.Cast("int32", input_count) > 2:' in generic_script
    assert "for item in range(input_count)" not in generic_script


def test_raw_associative_direct_output_requires_distinct_plan_storage():
    compiled = df.compile(
        make_tiny_mla_program(reduce_op=associative_tiny_mla_reduce),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"kv": [16]},
        task_extents=(1,),
        block_size=4,
        include_exit=False,
        scheduler_policy="cluster_local",
        reduce_strategy="streaming_tree",
        mode="debug",
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
        ),
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    assert primfunc_lowering_module.plan_reduce_input_output_storage_may_alias(compiled.plan)
    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )
    binary_script = next(
        handler.prim_func.script()
        for handler in result.handlers
        if handler.operator_kind == "reduce" and handler.handler_variant_key.reduce_arity.arity_class == "binary"
    )

    assert "merged_o = T.alloc_buffer" in binary_script
    assert "Out_o[_dataflow_o_i0] = merged_o[_dataflow_o_i0]" in binary_script


def make_head_program():
    return (
        T.dataflow_program(task_domain=("head",), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_by_head(Source="Source"),
            task_args=("head",),
            range_axis="kv",
        )
        .reduce(reduce_scalar())
        .finalize(finalize_scalar(Output="Output"))
    )


def make_unannotated_tensor_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_unannotated_tensor(Source="Source"),
            task_args=("batch",),
            range_axis="kv",
        )
        .reduce(reduce_scalar())
        .finalize(finalize_scalar(Output="Output"))
    )


def make_score_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_score(Source="Source"),
            task_args=("batch",),
            range_axis="kv",
        )
        .reduce(reduce_score())
        .finalize(finalize_score(Output="Output"))
    )


def make_coord_output_program():
    return (
        T.dataflow_program(task_domain=("batch", "head"), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_scalar(Source="Source"),
            task_args=("batch", "head"),
            range_axis="kv",
        )
        .reduce(reduce_scalar())
        .finalize(finalize_scalar_by_coords(Output="Output"))
    )


def make_tensor_field_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_tensor_field(A="A", B="B"),
            task_args=("batch",),
            range_axis="kv",
        )
        .reduce(reduce_tensor_field())
        .finalize(finalize_tensor_field(ScoreOut="ScoreOut", Vec0Out="Vec0Out", Vec1Out="Vec1Out"))
    )


def make_tiny_mla_program(reduce_op=tiny_mla_reduce):
    return (
        T.dataflow_program(task_domain=("head",), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            tiny_mla_split(Q="Q", KV="KV"),
            task_args=("head",),
            range_axis="kv",
        )
        .reduce(reduce_op())
        .finalize(tiny_mla_finalize(Out0="Out0", Out1="Out1"))
    )


def make_specialized_raw_tilelang_program(*, rows: int = 4, cols: int = 8):
    constants = {"rows": rows, "cols": cols}

    @T.dataflow_intermediate(specialization_constants=constants)
    class SpecializedTensorInter:
        vec: T.Tensor((rows, cols), T.int32)

    @T.dataflow.iter(range=("begin", "end"), specialization_constants=constants)
    def split_specialized_raw(Source: T.Tensor((rows, cols), T.int32)) -> SpecializedTensorInter:
        Scratch = T.alloc_shared((rows, cols), "int32")
        for i, j in T.Parallel(rows, cols):
            Scratch[i, j] = Source[i, j]
        return SpecializedTensorInter(vec=Scratch)

    @T.dataflow.reduce(specialization_constants=constants)
    def reduce_specialized_raw(items: list[SpecializedTensorInter]) -> SpecializedTensorInter:
        Vec = T.alloc_shared((rows, cols), "int32")
        for i, j in T.Parallel(rows, cols):
            Vec[i, j] = items[0].vec[i, j]
        return SpecializedTensorInter(vec=Vec)

    @T.dataflow.finalize(specialization_constants=constants)
    def finalize_specialized_raw(
        inter: SpecializedTensorInter,
        Output: T.Tensor((rows, cols), T.int32),
    ) -> None:
        for i, j in T.Parallel(rows, cols):
            Output[i, j] = inter.vec[i, j]

    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_specialized_raw(Source="Source"),
            range_axis="kv",
        )
        .reduce(reduce_specialized_raw())
        .finalize(finalize_specialized_raw(Output="Output"))
    )


def make_sliced_raw_tilelang_program():
    @T.dataflow.iter(range=("begin", "end"))
    def split_sliced_raw(Source: T.Tensor((4, 5), T.int32)) -> SlicedTensorInter:
        Scratch = T.alloc_shared((4, 5), "int32")
        for i, j in T.Parallel(4, 5):
            Scratch[i, j] = Source[i, j]
        return SlicedTensorInter(vec=Scratch[1:3, 2:5])

    @T.dataflow.reduce
    def reduce_sliced_raw(items: list[SlicedTensorInter]) -> SlicedTensorInter:
        Vec = T.alloc_shared((2, 3), "int32")
        for i, j in T.Parallel(2, 3):
            Vec[i, j] = items[0].vec[i, j]
        return SlicedTensorInter(vec=Vec)

    @T.dataflow.finalize
    def finalize_sliced_raw(inter: SlicedTensorInter, Output: T.Tensor((2, 3), T.int32)) -> None:
        for i, j in T.Parallel(2, 3):
            Output[i, j] = inter.vec[i, j]

    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_sliced_raw(Source="Source"),
            range_axis="kv",
        )
        .reduce(reduce_sliced_raw())
        .finalize(finalize_sliced_raw(Output="Output"))
    )


def make_raw_tilelang_return_ws_program():
    @T.dataflow_intermediate
    class ReturnWsTensorInter:
        vec: T.Tensor((2, 3), T.int32)

    @T.dataflow.iter(
        range=("begin", "end"),
        threads=384,
        physical_contract=df.DataflowOperatorPhysicalContract(
            return_warp_groups=(0, 1),
        ),
    )
    def split_return_ws(Source: T.Tensor((2, 3), T.int32)) -> ReturnWsTensorInter:
        Scratch = T.alloc_shared((2, 3), "int32")
        for i, j in T.Parallel(2, 3):
            Scratch[i, j] = Source[i, j]
        return ReturnWsTensorInter(vec=Scratch)

    @T.dataflow.reduce
    def reduce_return_ws(items: list[ReturnWsTensorInter]) -> ReturnWsTensorInter:
        Vec = T.alloc_shared((2, 3), "int32")
        for i, j in T.Parallel(2, 3):
            Vec[i, j] = items[0].vec[i, j]
        return ReturnWsTensorInter(vec=Vec)

    @T.dataflow.finalize
    def finalize_return_ws(inter: ReturnWsTensorInter, Output: T.Tensor((2, 3), T.int32)) -> None:
        for i, j in T.Parallel(2, 3):
            Output[i, j] = inter.vec[i, j]

    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_return_ws(Source="Source"),
            range_axis="kv",
        )
        .reduce(reduce_return_ws())
        .finalize(finalize_return_ws(Output="Output"))
    )


def make_raw_tilelang_direct_output_program():
    @T.dataflow_intermediate
    class DirectOutputInter:
        vec: T.Tensor((2, 3), T.int32)

    @T.dataflow.iter(
        range=("begin", "end"),
        threads=128,
        physical_contract=df.DataflowOperatorPhysicalContract(
            output_slot=df.DATAFLOW_OUTPUT_SLOT_DIRECT,
        ),
    )
    def split_direct_output(Source: T.Tensor((2, 3), T.int32)) -> DirectOutputInter:
        Scratch = T.alloc_shared((2, 3), "int32")
        for i, j in T.Parallel(2, 3):
            Scratch[i, j] = Source[i, j]
        return DirectOutputInter(vec=Scratch)

    @T.dataflow.reduce
    def reduce_direct_output(items: list[DirectOutputInter]) -> DirectOutputInter:
        Vec = T.alloc_shared((2, 3), "int32")
        for i, j in T.Parallel(2, 3):
            Vec[i, j] = items[0].vec[i, j]
        return DirectOutputInter(vec=Vec)

    @T.dataflow.finalize
    def finalize_direct_output(
        inter: DirectOutputInter,
        Output: T.Tensor((2, 3), T.int32),
    ) -> None:
        for i, j in T.Parallel(2, 3):
            Output[i, j] = inter.vec[i, j]

    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_direct_output(Source="Source"),
            range_axis="kv",
        )
        .reduce(reduce_direct_output())
        .finalize(finalize_direct_output(Output="Output"))
    )


def make_raw_map_list_input_and_finalize_range_program():
    @T.dataflow_intermediate
    class RawShard:
        vec: T.Tensor((2,), T.int32)

    @T.dataflow_intermediate
    class RawHidden:
        vec: T.Tensor((2,), T.int32)

    @T.dataflow.map(range=("begin", "end"), threads=128)
    def raw_map_list_source(Source: T.Tensor((4,), T.int32)) -> RawShard:
        Vec = T.alloc_shared((2,), "int32")
        for i in T.Parallel(2):
            Vec[i] = Source[T.dataflow_range_begin() + i]
        return RawShard(vec=Vec)

    @T.dataflow.map(range=("begin", "end"))
    def raw_map_list_consumer(parts: list[RawShard], Bias: T.Tensor((4,), T.int32)) -> RawHidden:
        Vec = T.alloc_shared((2,), "int32")
        for i in T.Parallel(2):
            slot = T.int32(i)
            Vec[i] = parts[slot].vec[i] + Bias[T.dataflow_range_begin() + i]
        return RawHidden(vec=Vec)

    @T.dataflow.finalize
    def raw_finalize_with_range(hidden: RawHidden, Output: T.Tensor((4,), T.int32)) -> None:
        for i in T.Parallel(2):
            Output[T.dataflow_range_begin() + i] = hidden.vec[i]

    return (
        T.dataflow_program(
            task_domain=("batch",),
            dynamic_ranges={"map1_range": "map1_ranges", "map2_range": "map2_ranges"},
        )
        .map(raw_map_list_source(Source="Source"), name="map1", range_axis="map1_range")
        .reshared(
            input="map1",
            name="gather",
            output_type=RawShard,
            physical_output_type=RawShard,
            output_arity=2,
            policy="hbm_all_gather",
        )
        .map(raw_map_list_consumer(Bias="Bias"), name="map2", input="gather", range_axis="map2_range")
        .finalize(raw_finalize_with_range(Output="Output"), input="map2")
    )


def make_raw_map_list_input_copy_program():
    @T.dataflow_intermediate
    class RawCopyShard:
        vec: T.Tensor((2,), T.int32)

    @T.dataflow_intermediate
    class RawCopyHidden:
        vec: T.Tensor((2,), T.int32)

    @T.dataflow.map(range=("begin", "end"))
    def raw_map_copy_source(Source: T.Tensor((4,), T.int32)) -> RawCopyShard:
        Vec = T.alloc_shared((2,), "int32")
        for i in T.Parallel(2):
            Vec[i] = Source[T.dataflow_range_begin() + i]
        return RawCopyShard(vec=Vec)

    @T.dataflow.map(range=("begin", "end"))
    def raw_map_copy_consumer(parts: list[RawCopyShard], selector: T.int32) -> RawCopyHidden:
        Vec = T.alloc_shared((2,), "int32")
        slot = selector
        T.copy(parts[slot].vec[0:2], Vec)
        return RawCopyHidden(vec=Vec)

    @T.dataflow.finalize
    def raw_map_copy_finalize(hidden: RawCopyHidden, Output: T.Tensor((4,), T.int32)) -> None:
        for i in T.Parallel(2):
            Output[T.dataflow_range_begin() + i] = hidden.vec[i]

    return (
        T.dataflow_program(
            task_domain=("selector",),
            dynamic_ranges={"map1_range": "map1_ranges", "map2_range": "map2_ranges"},
        )
        .map(raw_map_copy_source(Source="Source"), name="map1", range_axis="map1_range")
        .reshared(
            input="map1",
            name="gather",
            output_type=RawCopyShard,
            physical_output_type=RawCopyShard,
            output_arity=2,
            policy="hbm_all_gather",
        )
        .map(
            raw_map_copy_consumer(),
            name="map2",
            input="gather",
            task_args=("selector",),
            range_axis="map2_range",
        )
        .finalize(raw_map_copy_finalize(Output="Output"), input="map2")
    )


def make_wgmma_slot_layout_program():
    @T.dataflow_intermediate(primfunc_slot_layout={"value": "wgmma_k_major"})
    class WgmmaShard:
        value: T.Tensor((64, 128), T.float8_e4m3fn)

    @T.dataflow_intermediate
    class WgmmaDone:
        value: T.int32

    @T.dataflow.map(range=("begin", "end"))
    def wgmma_slot_source() -> WgmmaShard:
        Value = T.alloc_shared((64, 128), "float8_e4m3fn")
        T.clear(Value)
        return WgmmaShard(value=Value)

    @T.dataflow.map(range=("begin", "end"))
    def wgmma_slot_consumer(parts: list[WgmmaShard]) -> WgmmaDone:
        Weight = T.alloc_shared((128, 128), "float8_e4m3fn")
        Accum = T.alloc_fragment((64, 128), "float32")
        T.wgmma_gemm(parts[0].value[0:64, 0:128], Weight, Accum, transpose_B=True)
        return WgmmaDone(value=0)

    return (
        T.dataflow_program(
            task_domain=("batch",),
            dynamic_ranges={"map1_range": "map1_ranges", "map2_range": "map2_ranges"},
        )
        .map(wgmma_slot_source(), name="map1", range_axis="map1_range")
        .reshared(
            input="map1",
            name="gather",
            output_type=WgmmaShard,
            physical_output_type=WgmmaShard,
            output_arity=1,
            policy="hbm_all_gather",
        )
        .map(wgmma_slot_consumer(), name="map2", input="gather", range_axis="map2_range")
    )


def make_contiguous_wgmma_slot_layout_program():
    @T.dataflow_intermediate(primfunc_slot_layout={"value": "wgmma_k_major"})
    class ContiguousWgmmaShard:
        value: T.Tensor((64, 128), T.float8_e4m3fn)

    @T.dataflow_intermediate
    class ContiguousWgmmaDone:
        value: T.int32

    @T.dataflow.map(range=("begin", "end"))
    def contiguous_wgmma_source() -> ContiguousWgmmaShard:
        Value = T.alloc_shared((64, 128), "float8_e4m3fn")
        T.clear(Value)
        return ContiguousWgmmaShard(value=Value)

    @T.dataflow.map(
        range=("begin", "end"),
        threads=128,
        physical_contract=df.DataflowOperatorPhysicalContract(
            input_slots=df.DATAFLOW_INPUT_SLOTS_CONTIGUOUS,
        ),
    )
    def contiguous_wgmma_consumer(parts: list[ContiguousWgmmaShard]) -> ContiguousWgmmaDone:
        Weight = T.alloc_shared((128, 128), "float8_e4m3fn")
        Accum = T.alloc_fragment((64, 128), "float32")
        for slot in T.serial(0, 2):
            T.wgmma_gemm(
                parts[slot].value[0:64, 0:128],
                Weight,
                Accum,
                transpose_B=True,
            )
        return ContiguousWgmmaDone(value=0)

    return (
        T.dataflow_program(
            task_domain=("batch",),
            dynamic_ranges={"map1_range": "map1_ranges", "map2_range": "map2_ranges"},
        )
        .map(contiguous_wgmma_source(), name="map1", range_axis="map1_range")
        .reshared(
            input="map1",
            name="gather",
            output_type=ContiguousWgmmaShard,
            physical_output_type=ContiguousWgmmaShard,
            output_arity=2,
            policy="cluster_shared_all_gather",
        )
        .map(
            contiguous_wgmma_consumer(),
            name="map2",
            input="gather",
            range_axis="map2_range",
        )
    )


def make_unannotated_finalize_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(
            split_scalar(Source="Source"),
            task_args=("batch",),
            range_axis="kv",
        )
        .reduce(reduce_scalar())
        .finalize(finalize_unannotated_output(Output="Output"))
    )


def collect_call_names(expr):
    names = []
    if isinstance(expr, CallIR):
        names.append(expr.name)
        for arg in expr.args:
            names.extend(collect_call_names(arg))
    elif isinstance(expr, CastIR):
        names.extend(collect_call_names(expr.value))
    elif isinstance(expr, BinaryOpIR):
        names.extend(collect_call_names(expr.left))
        names.extend(collect_call_names(expr.right))
    elif isinstance(expr, TensorLoadIR):
        for index in expr.indices:
            names.extend(collect_call_names(index))
    elif isinstance(expr, (FieldAccessIR, FieldElementAccessIR)):
        names.extend(collect_call_names(expr.base))
    return names


def test_body_ir_uses_structured_nodes_for_iter_body():
    body_ir = lower_operator_call_to_body_ir(split_scalar(Source="Source"), "iter")

    assert body_ir.operator_kind == "iter"
    assert body_ir.operator_name == "split_scalar"
    assert body_ir.accumulators == (DataflowAccumulatorIR(name="value", dtype="int32", init=CastIR("int32", LiteralIR(0))),)
    assert body_ir.loop == DataflowLoopIR(
        var=ScalarVarIR("i"),
        iterable_kind="dataflow_range",
        body=(
            AugAssignIR(
                target=ScalarVarIR("value"),
                op="+",
                value=TensorLoadIR("Source", (ScalarVarIR("i"),)),
            ),
        ),
    )
    assert body_ir.returns == {"value": ScalarVarIR("value")}
    assert body_ir.stores == ()

    alt_ir = lower_operator_call_to_body_ir(split_scalar_alt(A="A"), "iter")
    assert alt_ir.accumulators == (DataflowAccumulatorIR(name="total", dtype="int32", init=CastIR("int32", LiteralIR(1))),)
    assert isinstance(alt_ir.loop, DataflowLoopIR)
    assert alt_ir.loop.var == ScalarVarIR("k")
    update = alt_ir.loop.body[0]
    assert isinstance(update, AugAssignIR)
    assert update.target == ScalarVarIR("total")
    assert update.value == BinaryOpIR(
        op="*",
        left=TensorLoadIR("A", (ScalarVarIR("k"),)),
        right=CastIR("int32", LiteralIR(2)),
    )
    assert alt_ir.returns == {"value": ScalarVarIR("total")}


def test_body_ir_allows_iter_task_scalar_parameter_indices():
    body_ir = lower_operator_call_to_body_ir(split_by_head(Source="Source"), "iter")

    assert isinstance(body_ir.loop, DataflowLoopIR)
    update = body_ir.loop.body[0]
    assert isinstance(update, AugAssignIR)
    assert update.value == TensorLoadIR(
        "Source",
        (
            ScalarVarIR("head"),
            ScalarVarIR("i"),
        ),
    )


def test_body_ir_uses_structured_nodes_for_reduce_and_finalize():
    reduce_ir = lower_operator_call_to_body_ir(reduce_scalar(), "reduce")

    assert reduce_ir.operator_kind == "reduce"
    assert reduce_ir.operator_name == "reduce_scalar"
    assert reduce_ir.accumulators == (DataflowAccumulatorIR(name="value", dtype="int32", init=CastIR("int32", LiteralIR(0))),)
    assert reduce_ir.loop == DataflowLoopIR(
        var=ScalarVarIR("item"),
        iterable_kind="items",
        body=(
            AugAssignIR(
                target=ScalarVarIR("value"),
                op="+",
                value=FieldAccessIR(ScalarVarIR("item"), "value"),
            ),
        ),
    )
    assert reduce_ir.returns == {"value": ScalarVarIR("value")}
    assert reduce_ir.stores == ()

    finalize_ir = lower_operator_call_to_body_ir(finalize_scalar(Output="Output"), "finalize")

    assert finalize_ir.operator_kind == "finalize"
    assert finalize_ir.operator_name == "finalize_scalar"
    assert finalize_ir.accumulators == ()
    assert finalize_ir.loop is None
    assert finalize_ir.returns == {}
    assert finalize_ir.stores == (
        TensorStoreIR(
            tensor_name="Output",
            index=ScalarVarIR("T.dataflow_task_id"),
            value=FieldAccessIR(ScalarVarIR("inter"), "value"),
        ),
    )


def test_body_ir_lowers_tensor_field_indexed_accesses():
    field_element_cls = getattr(body_ir_module, "FieldElementAccessIR", None)
    assert field_element_cls is not None

    reduce_ir = lower_operator_call_to_body_ir(reduce_tensor_field(), "reduce")
    assert isinstance(reduce_ir.loop, DataflowLoopIR)
    score_update, vec0_update, vec1_update = reduce_ir.loop.body
    assert score_update.value == FieldAccessIR(ScalarVarIR("item"), "score")
    assert isinstance(vec0_update.value, field_element_cls)
    assert vec0_update.value.base == ScalarVarIR("item")
    assert vec0_update.value.field_name == "vec"
    assert vec0_update.value.indices == (0,)
    assert vec0_update.value.flat_index == 0
    assert isinstance(vec1_update.value, field_element_cls)
    assert vec1_update.value.indices == (1,)
    assert vec1_update.value.flat_index == 1

    finalize_ir = lower_operator_call_to_body_ir(
        finalize_tensor_field(ScoreOut="ScoreOut", Vec0Out="Vec0Out", Vec1Out="Vec1Out"),
        "finalize",
    )
    assert finalize_ir.stores[0].value == FieldAccessIR(ScalarVarIR("inter"), "score")
    assert isinstance(finalize_ir.stores[1].value, field_element_cls)
    assert finalize_ir.stores[1].value.field_name == "vec"
    assert finalize_ir.stores[1].value.flat_index == 0
    assert isinstance(finalize_ir.stores[2].value, field_element_cls)
    assert finalize_ir.stores[2].value.field_name == "vec"
    assert finalize_ir.stores[2].value.flat_index == 1


def test_body_ir_lowers_online_softmax_iter_assignments_and_math_calls():
    body_ir = lower_operator_call_to_body_ir(tiny_mla_split(Q="Q", KV="KV"), "iter")

    assert body_ir.accumulators[0] == DataflowAccumulatorIR(
        name="m",
        dtype="float32",
        init=CastIR("float32", LiteralIR(-3.402823e38)),
    )
    assert isinstance(body_ir.loop, DataflowLoopIR)
    assert all(isinstance(stmt, AssignIR) for stmt in body_ir.loop.body)
    assert [stmt.target for stmt in body_ir.loop.body] == [
        ScalarVarIR("o0"),
        ScalarVarIR("o1"),
        ScalarVarIR("l"),
        ScalarVarIR("m"),
    ]
    call_names = [name for stmt in body_ir.loop.body for name in collect_call_names(stmt.value)]
    assert "T.max" in call_names
    assert "T.exp2" in call_names


def test_body_ir_lowers_online_softmax_reduce_assignments_and_tensor_fields():
    body_ir = lower_operator_call_to_body_ir(tiny_mla_reduce(), "reduce")

    assert body_ir.accumulators[0].init == CastIR("float32", LiteralIR(-3.402823e38))
    assert isinstance(body_ir.loop, DataflowLoopIR)
    assert all(isinstance(stmt, AssignIR) for stmt in body_ir.loop.body)
    call_names = [name for stmt in body_ir.loop.body for name in collect_call_names(stmt.value)]
    assert "T.max" in call_names
    assert "T.exp2" in call_names
    assert any(isinstance(stmt.value, BinaryOpIR) and "T.exp2" in collect_call_names(stmt.value) for stmt in body_ir.loop.body)


def test_body_ir_lowers_finalize_field_arithmetic_expression():
    body_ir = lower_operator_call_to_body_ir(tiny_mla_finalize(Out0="Out0", Out1="Out1"), "finalize")

    assert len(body_ir.stores) == 2
    for store in body_ir.stores:
        assert isinstance(store.value, BinaryOpIR)
        assert store.value.op == "/"
        assert isinstance(store.value.left, FieldElementAccessIR)
        assert isinstance(store.value.right, FieldAccessIR)


def test_body_ir_lowers_finalize_multidimensional_task_coord_store():
    body_ir = lower_operator_call_to_body_ir(finalize_scalar_by_coords(Output="Output"), "finalize")

    assert body_ir.stores == (
        TensorStoreIR(
            tensor_name="Output",
            index=TupleExprIR((ScalarVarIR("batch"), ScalarVarIR("head"))),
            value=FieldAccessIR(ScalarVarIR("inter"), "value"),
        ),
    )


def test_body_ir_rejects_wrong_return_field():
    with pytest.raises(DataflowBodyIRLoweringError, match="primfunc handler lowering .* return fields"):
        lower_operator_call_to_body_ir(split_wrong_return_field(Source="Source"), "iter")


def test_body_ir_rejects_positional_return_constructor():
    with pytest.raises(DataflowBodyIRLoweringError, match="primfunc handler lowering .* keyword field"):
        lower_operator_call_to_body_ir(split_positional_return(Source="Source"), "iter")


def test_body_ir_rejects_unbound_accumulator_init():
    with pytest.raises(DataflowBodyIRLoweringError, match="primfunc handler lowering .* typed scalar literal"):
        lower_operator_call_to_body_ir(split_unbound_accumulator_init(Source="Source"), "iter")


def test_body_ir_rejects_finalize_wrong_index():
    with pytest.raises(DataflowBodyIRLoweringError, match="primfunc handler lowering .*T\\.dataflow_task_id"):
        lower_operator_call_to_body_ir(finalize_wrong_index(Output="Output"), "finalize")


def test_body_ir_rejects_finalize_wrong_target():
    with pytest.raises(DataflowBodyIRLoweringError, match="primfunc handler lowering .* bound output parameter"):
        lower_operator_call_to_body_ir(finalize_undeclared_output(Output="Output"), "finalize")

    with pytest.raises(DataflowBodyIRLoweringError, match="primfunc handler lowering .* bound output parameter"):
        lower_operator_call_to_body_ir(finalize_intermediate_target(Output="Output"), "finalize")


def test_body_ir_rejects_iter_unbound_scalar():
    with pytest.raises(DataflowBodyIRLoweringError, match="primfunc handler lowering .* unsupported scalar 'offset'"):
        lower_operator_call_to_body_ir(split_unbound_scalar(Source="Source"), "iter")


def test_body_ir_rejects_reduce_wrong_field_base():
    with pytest.raises(DataflowBodyIRLoweringError, match="primfunc handler lowering .* unsupported reduce expression"):
        lower_operator_call_to_body_ir(reduce_wrong_field_base(), "reduce")


def test_body_ir_rejects_unknown_reduce_field_name():
    with pytest.raises(DataflowBodyIRLoweringError, match="primfunc handler lowering .* unknown intermediate field 'missing'"):
        lower_operator_call_to_body_ir(reduce_wrong_field_name(), "reduce")


def test_body_ir_rejects_unknown_finalize_field_name():
    with pytest.raises(DataflowBodyIRLoweringError, match="primfunc handler lowering .* unknown intermediate field 'missing'"):
        lower_operator_call_to_body_ir(finalize_wrong_field_name(Output="Output"), "finalize")


def test_body_ir_rejects_for_else():
    with pytest.raises(DataflowBodyIRLoweringError, match="primfunc handler lowering .* for-else"):
        lower_operator_call_to_body_ir(split_for_else(Source="Source"), "iter")


def test_primfunc_generation_uses_body_ir_not_fixed_template():
    compiled = df.compile(
        make_scalar_program(split_scalar_alt),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [32]},
        block_size=128,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )

    assert len(result.handlers) == 3
    iter_handler = result.handlers[0]
    assert iter_handler.global_symbol == "dataflow_primfunc_split_scalar_alt_device"
    assert isinstance(iter_handler.prim_func, tir.PrimFunc)
    script = iter_handler.prim_func.script()
    assert "A" in script
    assert "total" in script
    assert "range_begin" in script
    assert "range_end" in script
    assert "for k in range(range_begin" in script
    assert "* 2" in script
    assert "Source" not in script


def test_dataflow_range_fact_flows_into_public_transfer_contract():
    compiled = df.compile(
        make_bounded_transfer_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [13, 29]},
        task_extents=(2,),
        block_size=128,
        mode="debug",
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
        ),
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )
    original_plan = compiled.plan

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )

    assert compiled.plan == original_plan
    iter_handler = next(handler for handler in result.handlers if handler.operator_kind == "iter")
    range_end_param = next(param for param in iter_handler.params if param.role is df.DataflowHandlerParamRole.RANGE_END)
    range_end_var = iter_handler.prim_func.params[range_end_param.ordinal]

    copy_calls = []

    def collect_copy(node):
        if isinstance(node, tir.Call) and isinstance(node.op, ir.Op) and node.op.name == "tl.tileop.copy":
            copy_calls.append(node)

    post_order_visit(iter_handler.prim_func.body, collect_copy)
    assert len(copy_calls) == 1
    parsed_copy = _ffi_api.ParseOperator(copy_calls[0])
    contract = parsed_copy.transfer_contract
    assert contract is not None
    assert contract.schema_version == 1
    assert contract.sync_owner == int(T.TransferSynchronizationOwner.TRANSFER)

    bound_vars = []
    for region_range in contract.src_valid_region.region:
        post_order_visit(
            region_range.min,
            lambda node: bound_vars.append(node) if isinstance(node, tir.Var) else None,
        )
        post_order_visit(
            region_range.extent,
            lambda node: bound_vars.append(node) if isinstance(node, tir.Var) else None,
        )
    if not any(var.same_as(range_end_var) for var in bound_vars):
        matching_bindings = []

        def collect_range_binding(node):
            if isinstance(node, tir.LetStmt) and any(node.var.same_as(var) for var in bound_vars):
                value_vars = []
                post_order_visit(
                    node.value,
                    lambda value_node: value_vars.append(value_node) if isinstance(value_node, tir.Var) else None,
                )
                if any(var.same_as(range_end_var) for var in value_vars):
                    matching_bindings.append(node)

        post_order_visit(iter_handler.prim_func.body, collect_range_binding)
        assert matching_bindings
    planned_identity = next(
        instruction.handler_identity
        for instruction in compiled.plan.instructions
        if instruction.handler_identity is not None and instruction.handler_identity.operator_kind == "iter"
    )
    assert iter_handler.handler_identity == planned_identity
    assert iter_handler.handler_identity.binding_key == (
        planned_identity.operator_id,
        "iter",
    )
    assert iter_handler.handler_variant_key.base_identity == (iter_handler.handler_identity)
    decisions = (
        replace(
            compiled,
            primfunc_lowering=result,
        )
        .decision_artifact()
        .to_dict()
    )
    transfer_requests = decisions["lowerings"]["copy"]["requests"]
    assert len(transfer_requests) == 1
    request = transfer_requests[0]
    assert request["contract_schema_version"] == 1
    assert request["allow_async"] is False
    assert request["synchronization_owner"] == "transfer"
    assert request["lowering_plan_schema_version"] == 1
    assert request["selected_implementation"] == "common.simt"
    assert request["selection_supported"] is True
    assert request["asynchronous"] is False
    assert request["uses_tma_descriptor"] is False
    assert request["requires_post_fill"] is False
    assert request["selection_reason"] == ("transfer contract requires synchronous execution")


def test_primfunc_handler_binding_is_independent_of_diagnostic_names():
    compiled = df.compile(
        make_scalar_program(split_scalar_alt),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [32]},
        block_size=128,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )
    renamed_wrapper = replace(
        compiled.wrapper_spec,
        handlers=tuple(
            replace(
                handler,
                operator_name=f"diagnostic_handler_{handler.handler_id}",
                symbol_name=f"diagnostic_symbol_{handler.handler_id}",
            )
            for handler in compiled.wrapper_spec.handlers
        ),
    )

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        renamed_wrapper,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )

    assert [handler.handler_identity for handler in result.handlers] == [handler.handler_identity for handler in renamed_wrapper.handlers]
    assert [handler.operator_name for handler in result.handlers] == [
        "diagnostic_handler_0",
        "diagnostic_handler_1",
        "diagnostic_handler_2",
    ]


def test_randomized_iter_specializations_preserve_identity_through_linking():
    rng = random.Random(0)
    lengths = list(range(16, 129, 16)) + [17, 23, 47]
    rng.shuffle(lengths)
    compiled = df.compile(
        make_scalar_program(split_scalar_alt),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": lengths},
        block_size=128,
        task_extents=(len(lengths),),
        iter_range_buckets=tuple(range(1, 9)) + ("generic",),
        iter_range_bucket_size=16,
        iter_range_exact_lengths=(17, 23, 47),
    )

    scheduled_variants = []
    for instruction in compiled.plan.instructions:
        variant = instruction.handler_variant_key
        if variant is None or variant.base_identity.operator_kind != "iter":
            continue
        if variant not in scheduled_variants:
            scheduled_variants.append(variant)
    wrapper_variants = [handler.handler_variant_key for handler in compiled.wrapper_spec.handlers if handler.operator_kind == "iter"]
    lowered_variants = [handler.handler_variant_key for handler in compiled.primfunc_lowering.handlers if handler.operator_kind == "iter"]

    assert wrapper_variants == scheduled_variants
    assert lowered_variants == scheduled_variants
    assert len({variant.binding_key for variant in scheduled_variants}) == 1
    assert {source.handler_id for source in compiled.wrapper_spec.handler_sources} == {
        handler.handler_id for handler in compiled.wrapper_spec.handlers
    }


def test_primfunc_generation_threads_iter_task_scalar_parameters():
    compiled = df.compile(
        make_head_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [64, 64]},
        task_extents=(2,),
        block_size=32,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )

    iter_handler = result.handlers[0]
    iter_script = iter_handler.prim_func.script()
    assert iter_handler.task_param_names == ("head",)
    assert iter_handler.task_param_dtypes == ("int32",)
    assert "head" in iter_script
    assert "Source[head, i]" in iter_script
    assert "range_begin" in iter_script
    assert "range_end" in iter_script
    assert "task_id" in iter_script
    assumptions = []
    tir.stmt_functor.post_order_visit(
        iter_handler.prim_func.body,
        lambda node: (
            assumptions.append(node.args[0]) if isinstance(node, tir.Call) and getattr(node.op, "name", None) == "tir.assume" else None
        ),
    )
    assert len(assumptions) == 1
    assert "head" in str(assumptions[0])


def test_primfunc_generation_threads_finalize_task_scalar_parameters_and_output_shape():
    compiled = df.compile(
        make_coord_output_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [16, 16, 16, 16]},
        task_extents=(2, 2),
        block_size=16,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )

    finalize_handler = result.handlers[2]
    finalize_script = finalize_handler.prim_func.script()
    assert finalize_handler.task_param_names == ("batch", "head")
    assert finalize_handler.task_param_dtypes == ("int32", "int32")
    assert 'Output = T.match_buffer(Output_handle, (2, 2), "int32"' in finalize_script
    assert "Output[batch, head] = Inter[0]" in finalize_script
    assumptions = []
    tir.stmt_functor.post_order_visit(
        finalize_handler.prim_func.body,
        lambda node: (
            assumptions.append(node.args[0]) if isinstance(node, tir.Call) and getattr(node.op, "name", None) == "tir.assume" else None
        ),
    )
    assert len(assumptions) == 2
    assumption_vars = {str(param) for condition in assumptions for param in tir.analysis.undefined_vars(condition)}
    assert assumption_vars == {
        "batch",
        "head",
    }


def test_primfunc_generation_uses_single_non_value_return_field():
    compiled = df.compile(
        make_score_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [16]},
        block_size=128,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )

    iter_script = result.handlers[0].prim_func.script()
    reduce_script = result.handlers[1].prim_func.script()
    finalize_script = result.handlers[2].prim_func.script()
    assert "score" in iter_script
    assert "value" not in iter_script
    assert "for i in range(range_begin" in iter_script
    assert "range_end - range_begin" in iter_script
    assert "for item in range(input_count)" in reduce_script
    assert "Inter[0]" in finalize_script
    assert "Output[task_id]" in finalize_script


def test_primfunc_generation_lowers_reduce_cast_field_access():
    compiled = df.compile(
        make_scalar_program(reduce_op=reduce_cast_scalar_field),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [16]},
        block_size=128,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )

    reduce_script = result.handlers[1].prim_func.script()
    assert "for item in range(input_count)" in reduce_script
    assert "Items[item]" in reduce_script


def test_primfunc_generation_uses_max_range_length_for_iter_shape_and_loop():
    compiled = df.compile(
        make_scalar_program(split_scalar_alt),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [16, 32]},
        block_size=128,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )

    script = result.handlers[0].prim_func.script()
    assert 'A = T.match_buffer(A_handle, (32,), "int32"' in script
    assert "for k in range(range_begin" in script
    assert "range_end - range_begin" in script


def test_primfunc_generation_uses_fixed_shape_tensor_field_extents():
    compiled = df.compile(
        make_tensor_field_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )

    iter_script = result.handlers[0].prim_func.script()
    reduce_script = result.handlers[1].prim_func.script()
    finalize_script = result.handlers[2].prim_func.script()

    assert 'Out_score = T.match_buffer(Out_score_handle, (1,), "int32"' in iter_script
    assert 'Out_vec = T.match_buffer(Out_vec_handle, (2,), "int32"' in iter_script
    assert "Out_vec[0] = v0" in iter_script
    assert "Out_vec[1] = v1" in iter_script
    assert 'Item0_score = T.match_buffer(Item0_score_handle, (1,), "int32"' in reduce_script
    assert 'Item0_vec = T.match_buffer(Item0_vec_handle, (2,), "int32"' in reduce_script
    assert 'Item1_score = T.match_buffer(Item1_score_handle, (1,), "int32"' in reduce_script
    assert 'Item1_vec = T.match_buffer(Item1_vec_handle, (2,), "int32"' in reduce_script
    assert "T.if_then_else(item == 0, Item0_vec[0], Item1_vec[0])" in reduce_script
    assert "T.if_then_else(item == 0, Item0_vec[1], Item1_vec[1])" in reduce_script
    assert 'Inter_vec = T.match_buffer(Inter_vec_handle, (2,), "int32"' in finalize_script
    assert "Inter_vec[0]" in finalize_script
    assert "Inter_vec[1]" in finalize_script


def test_primfunc_generation_lowers_tiny_mla_math_source():
    compiled = df.compile(
        make_tiny_mla_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [64, 64]},
        task_extents=(2,),
        block_size=32,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )

    iter_script = result.handlers[0].prim_func.script()
    reduce_script = result.handlers[1].prim_func.script()
    finalize_script = result.handlers[2].prim_func.script()
    assert "T.max" in iter_script
    assert "T.exp2" in iter_script
    assert "Q[head, 0]" in iter_script
    assert "KV[kv, 0]" in iter_script
    assert "T.max" in reduce_script
    assert "T.exp2" in reduce_script
    assert "Out0[task_id] = Inter_o[0] / Inter_l[0]" in finalize_script
    assert "Out1[task_id] = Inter_o[1] / Inter_l[0]" in finalize_script


def test_primfunc_generation_resolves_operator_factory_specialization_constants():
    compiled = df.compile(
        make_specialized_raw_tilelang_program(rows=4, cols=8),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [1]},
        block_size=1,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )

    iter_script = result.handlers[0].prim_func.script()
    reduce_script = result.handlers[1].prim_func.script()
    finalize_script = result.handlers[2].prim_func.script()

    assert 'Source = T.match_buffer(Source_handle, (4, 8), "int32"' in iter_script
    assert 'Out = T.match_buffer(Out_handle, (4, 8), "int32"' in iter_script
    assert 'T.alloc_buffer((4, 8), "int32"' in iter_script
    assert "for i, j in T.grid(4, 8)" in iter_script or "for i in T.parallel(4)" in iter_script
    assert 'Items = T.match_buffer(Items_handle, (1, 4, 8), "int32"' in reduce_script
    assert 'Out = T.match_buffer(Out_handle, (4, 8), "int32"' in reduce_script
    assert 'Inter = T.match_buffer(Inter_handle, (4, 8), "int32"' in finalize_script
    assert 'Output = T.match_buffer(Output_handle, (4, 8), "int32"' in finalize_script


def test_primfunc_generation_stores_raw_tilelang_return_slices_with_offsets():
    compiled = df.compile(
        make_sliced_raw_tilelang_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [1]},
        block_size=1,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )

    iter_script = result.handlers[0].prim_func.script()
    assert 'Out = T.match_buffer(Out_handle, (2, 3), "int32"' in iter_script
    assert "Out[_dataflow_vec_i0, _dataflow_vec_i1] = Scratch[_dataflow_vec_i0 + 1, _dataflow_vec_i1 + 2]" in iter_script


def test_primfunc_generation_wraps_raw_tilelang_return_store_in_ws():
    compiled = df.compile(
        make_raw_tilelang_return_ws_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [1]},
        block_size=1,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )

    iter_script = result.handlers[0].prim_func.script()
    assert "if tx >= 0 and tx < 256:" in iter_script
    assert 'T.attr(0, "warp_specialize", 1)' in iter_script
    ws_start = iter_script.index("if tx >= 0 and tx < 256:")
    return_store = iter_script.index("Out[_dataflow_vec_i0, _dataflow_vec_i1] = Scratch[_dataflow_vec_i0, _dataflow_vec_i1]")
    assert ws_start < return_store


def test_primfunc_generation_aliases_raw_tilelang_return_to_output_buffer():
    compiled = df.compile(
        make_raw_tilelang_direct_output_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [1]},
        block_size=1,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )

    iter_script = result.handlers[0].prim_func.script()
    assert "Scratch = T.alloc_buffer" not in iter_script
    assert "Out[i, j] = Source[i, j]" in iter_script
    assert "_dataflow_vec_i" not in iter_script


def test_primfunc_generation_rewrites_raw_map_list_inputs_and_finalize_range():
    compiled = df.compile(
        make_raw_map_list_input_and_finalize_range_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"map1_range": [4], "map2_range": [4]},
        task_extents=(1,),
        block_size=2,
        include_exit=False,
        force_hbm_comms=True,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
        map_input_scope="shared",
    )

    map2_script = result.handlers[1].prim_func.script()
    assert 'scope="shared"' in map2_script
    finalize_script = result.handlers[2].prim_func.script()
    assert "parts[" not in map2_script
    assert "Item0" in map2_script
    assert "Item1" in map2_script
    assert "T.if_then_else(slot == 0" in map2_script
    assert "range_begin" in finalize_script
    assert "Output[range_begin" in finalize_script
    assert "Inter[i]" in finalize_script


def test_primfunc_generation_rewrites_raw_map_list_input_copy():
    compiled = df.compile(
        make_raw_map_list_input_copy_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"map1_range": [4], "map2_range": [4]},
        task_extents=(1,),
        block_size=2,
        include_exit=False,
        force_hbm_comms=True,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
    )

    map2_script = result.handlers[1].prim_func.script()
    assert "parts[" not in map2_script
    assert "T.copy" in map2_script
    assert "Item0" in map2_script
    assert "Item1" in map2_script
    assert "if slot == 0" in map2_script
    assert "if slot == 1" in map2_script


def test_primfunc_generation_preserves_wgmma_slot_layout_across_map_stages():
    compiled = df.compile(
        make_wgmma_slot_layout_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"map1_range": [1], "map2_range": [1]},
        task_extents=(1,),
        block_size=128,
        include_exit=False,
        force_hbm_comms=True,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
        map_input_scope="shared",
    )

    def buffer_layout(func, buffer_name):
        buffer = next(buffer for buffer in func.buffer_map.values() if buffer.name == buffer_name)
        layouts = []

        def collect(node):
            if not isinstance(node, tir.Block) or "layout_map" not in node.annotations:
                return
            for data_var, layout in node.annotations["layout_map"].items():
                if data_var.same_as(buffer.data):
                    layouts.append(layout)

        post_order_visit(func.body, collect)
        assert len(layouts) == 1
        return buffer, layouts[0]

    map1 = result.handlers[0].prim_func
    map2 = result.handlers[1].prim_func
    map1_output, map1_layout = buffer_layout(map1, "Out")
    map2_input, map2_layout = buffer_layout(map2, "Item0")

    assert map1_layout.is_equal(make_wgmma_swizzled_layout(map1_output))
    assert map2_layout.is_equal(make_wgmma_swizzled_layout(map2_input))
    assert map1_layout.is_equal(map2_layout)


def test_primfunc_generation_packs_contiguous_map_inputs_into_one_swizzled_buffer():
    compiled = df.compile(
        make_contiguous_wgmma_slot_layout_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"map1_range": [256], "map2_range": [256]},
        task_extents=(1,),
        block_size=128,
        include_exit=False,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    result = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
        map_input_scope="shared",
    )

    map2 = result.handlers[1].prim_func
    map2_script = map2.script()
    packed_input = next(buffer for buffer in map2.buffer_map.values() if buffer.name == "Items")
    layouts = []

    def collect(node):
        if not isinstance(node, tir.Block) or "layout_map" not in node.annotations:
            return
        for data_var, layout in node.annotations["layout_map"].items():
            if data_var.same_as(packed_input.data):
                layouts.append(layout)

    post_order_visit(map2.body, collect)
    assert tuple(int(extent) for extent in packed_input.shape) == (2, 64, 128)
    assert len(layouts) == 1
    assert layouts[0].is_equal(make_wgmma_swizzled_layout(packed_input))
    assert "Item0" not in map2_script
    assert "Item1" not in map2_script
    assert "parts[" not in map2_script
    assert "T.region(Items[slot, 0, 0], 1, 1, 64, 128)" in map2_script


def test_primfunc_linking_passes_contiguous_map_input_base_from_first_slot():
    compiled = df.compile(
        make_contiguous_wgmma_slot_layout_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"map1_range": [256], "map2_range": [256]},
        task_extents=(1,),
        block_size=128,
        include_exit=False,
        mode="executable",
        target_override=df.TargetCapabilitySnapshot.for_cuda(
            (9, 0),
            compiler_version=(12, 8),
        ),
    )

    assert compiled.primfunc_lowering is not None
    map2_handler = next(handler for handler in compiled.primfunc_lowering.handlers if handler.operator_name == "contiguous_wgmma_consumer")
    assert map2_handler.contiguous_map_input_count == 2
    source = compiled.wrapper_source
    signature_start = source.index(f"void {map2_handler.device_symbol}(")
    signature_end = source.index(") {", signature_start)
    signature = source[signature_start:signature_end]
    assert "const fp8_e4_t* __restrict__ Items" in signature
    assert "Item0" not in signature
    assert "Item1" not in signature

    adapter_start = source.index("TL_DEVICE void dataflow_handler_1_contiguous_wgmma_consumer(")
    adapter_end = source.index("TL_DEVICE bool dataflow_dispatch_handler(", adapter_start)
    adapter = source[adapter_start:adapter_end]
    assert "handler_args.input_slot_count < 2u" in adapter
    assert "input_slots[handler_args.input_slot_offset]" in adapter
    assert "input_slot_offset + 1u" not in adapter


def test_primfunc_linking_rejects_noncontiguous_packed_map_input_offsets():
    compiled = df.compile(
        make_contiguous_wgmma_slot_layout_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"map1_range": [256], "map2_range": [256]},
        task_extents=(1,),
        block_size=128,
        include_exit=False,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )
    lowering = df.lower_program_handlers_to_primfuncs(
        compiled.program,
        compiled.wrapper_spec,
        compiled.tensor_arg_plan,
        plan=compiled.plan,
        lower_to_cuda=False,
        map_input_scope="shared",
    )
    map2_instruction = next(
        instruction for instruction in compiled.plan.instructions if instruction.operator_name == "contiguous_wgmma_consumer"
    )
    broken_slot_id = map2_instruction.input_slots[1]
    broken_slots = list(compiled.packed_plan.slots)
    broken_slots[broken_slot_id] = replace(
        broken_slots[broken_slot_id],
        shared_offset=broken_slots[broken_slot_id].shared_offset + 16,
    )
    broken_plan = replace(compiled.packed_plan, slots=tuple(broken_slots))

    with pytest.raises(ValueError, match="contiguous map inputs"):
        compiler_module.validate_contiguous_primfunc_map_inputs(
            lowering,
            compiled.plan,
            broken_plan,
        )


def test_raw_map_dynamic_intermediate_tile_call_expands_to_static_slots():
    source = """
for slot in T.serial(0, 2):
    T.wgmma_gemm(
        parts[slot].value[0:64, 0:128],
        weights,
        accum,
        transpose_B=True,
    )
"""
    field = primfunc_lowering_module.PrimFuncField(
        name="value",
        dtype="float8_e4m3fn",
        shape=(64, 128),
        numel=64 * 128,
    )

    rewritten = primfunc_lowering_module.rewrite_raw_map_intermediate_fields(
        source,
        collection_names=("parts",),
        fields_by_name={"value": field},
        input_names=({"value": "Item0_value"}, {"value": "Item1_value"}),
    )

    assert "parts[" not in rewritten
    assert "T.if_then_else" not in rewritten
    assert "if slot == 0:" in rewritten
    assert "if slot == 1:" in rewritten
    assert "T.wgmma_gemm(Item0_value[0:64, 0:128]" in rewritten
    assert "T.wgmma_gemm(Item1_value[0:64, 0:128]" in rewritten


def test_inspect_ir_stores_handlers_without_generating_placeholder_wrapper():
    compiled = df.compile(
        make_scalar_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [16]},
        block_size=128,
        mode="inspect",
        inspection_stage="ir",
    )

    assert compiled.wrapper_spec.handler_lowering == df.PRIMFUNC_HANDLER_LOWERING
    assert compiled.primfunc_lowering is not None
    assert [handler.global_symbol for handler in compiled.primfunc_lowering.handlers] == [
        "dataflow_primfunc_split_scalar_device",
        "dataflow_primfunc_reduce_scalar_device",
        "dataflow_primfunc_finalize_scalar_device",
    ]
    assert [handler.device_symbol for handler in compiled.primfunc_lowering.handlers] == [
        "dataflow_primfunc_split_scalar_device_kernel",
        "dataflow_primfunc_reduce_scalar_device_kernel",
        "dataflow_primfunc_finalize_scalar_device_kernel",
    ]

    dumped = compiled.dump_plan()
    assert dumped["primfunc_lowering"] is not None
    assert dumped["primfunc_lowering"]["handlers"] == [handler.to_dict() for handler in compiled.primfunc_lowering.handlers]

    assert compiled.wrapper_source == ""
    assert compiled.wrapper_spec.handler_sources == ()


def test_compile_with_primfunc_lowering_generates_cuda_from_body_ir():
    compiled = df.compile(
        make_scalar_program(split_scalar_alt),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [32]},
        block_size=16,
        wrapper_name="dataflow_primfunc_body_ir_wrapper",
        mode="inspect",
        inspection_stage="cuda",
    )

    assert compiled.primfunc_lowering is not None
    assert compiled.artifact_contract == df.DATAFLOW_ARTIFACT_INSPECTION
    assert compiled.executable is False
    assert "no executable wrapper was generated" in compiled.non_executable_reason
    cuda_source = compiled.primfunc_lowering.cuda_source
    assert cuda_source
    assert "dataflow_primfunc_split_scalar_alt_device" in cuda_source
    assert "dataflow_primfunc_reduce_scalar_device" in cuda_source
    assert "dataflow_primfunc_finalize_scalar_device" in cuda_source
    assert "Source" not in cuda_source
    assert "A" in cuda_source
    assert "* 2" in cuda_source or "*2" in cuda_source


@pytest.mark.parametrize(
    "retired_option,value",
    (
        ("handler_lowering", "primfunc"),
        ("lower_primfunc_handlers", False),
        ("link_primfunc_handlers", False),
        ("primfunc_reduce_staging", "auto"),
    ),
)
def test_compile_rejects_retired_handler_options(retired_option, value):
    with pytest.raises(ValueError, match="retired"):
        df.compile(
            make_scalar_program(),
            topology=df.GPUTopology(sm_count=1, cluster_size=1),
            range_lengths={"kv": [16]},
            block_size=128,
            **{retired_option: value},
        )


def test_compile_with_primfunc_lowering_rejects_runtime_launch_until_linked():
    compiled = df.compile(
        make_scalar_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [16]},
        block_size=128,
        mode="inspect",
        inspection_stage="ir",
    )

    assert compiled.artifact_contract == df.DATAFLOW_ARTIFACT_INSPECTION
    assert compiled.executable is False
    assert "not lowered to CUDA" in compiled.non_executable_reason

    with pytest.raises(df.DataflowArtifactNotExecutableError, match="not lowered to CUDA"):
        compiled()
    with pytest.raises(df.DataflowArtifactNotExecutableError, match="not lowered to CUDA"):
        compiled.persistent_executable()


def test_primfunc_source_factory_receives_stage_graph_metadata():
    seen = {}

    @T.dataflow.map(range=("begin", "end"))
    def source_factory_map1(Input: T.Tensor((16,), T.int32)) -> ScalarInter:
        value = T.int32(0)
        for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
            value += Input[i]
        return ScalarInter(value=value)

    def source_factory_map2_source(**kwargs):
        seen.update(kwargs)
        global_symbol = kwargs["global_symbol"]
        return f"""
@T.prim_func
def {global_symbol}(Item0: T.Tensor((1,), "int32"), Item1: T.Tensor((1,), "int32"), Out: T.Tensor((1,), "int32"), task: T.int32, range_begin: T.uint32, range_end: T.uint32, task_id: T.uint32):
    T.func_attr({{"global_symbol": "{global_symbol}", "tl.dataflow_device_function": True, "tir.noalias": True}})
    with T.Kernel(1, threads=1) as _dataflow_pid:
        Out[0] = Item0[0] + Item1[0] + task
"""

    @T.dataflow.map(range=("begin", "end"), primfunc_source_factory=source_factory_map2_source)
    def source_factory_map2(parts: list[ScalarInter], task: T.int32) -> ScoreInter:
        raise AssertionError("source factory supplies this handler")

    @T.dataflow.finalize
    def source_factory_finalize(inter: ScoreInter, Output: T.Tensor((1,), T.int32)) -> None:
        Output[T.dataflow_task_id()] = inter.score

    program = (
        T.dataflow_program(task_domain=("task",), dynamic_ranges={"map1_range": "map1_ranges", "map2_range": "map2_ranges"})
        .map(source_factory_map1(Input="Input"), name="map1", task_args=("task",), range_axis="map1_range")
        .reshared(
            input="map1",
            name="gather",
            output_type=ScalarInter,
            physical_output_type=ScalarInter,
            output_arity=2,
            policy="hbm_all_gather",
        )
        .map(source_factory_map2(), name="map2", input="gather", task_args=("task",), range_axis="map2_range")
        .finalize(source_factory_finalize(Output="Output"), input="map2")
    )

    compiled = df.compile(
        program,
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"map1_range": [16], "map2_range": [16]},
        block_size=8,
        task_extents=(1,),
        include_exit=False,
        mode="inspect",
        inspection_stage="ir",
    )

    assert seen["max_input_slots"] == 2
    assert seen["stage"].name == "map2"
    assert seen["operator_call"].name == "source_factory_map2"
    assert seen["tensor_arg_plan"] is compiled.tensor_arg_plan
    assert [param.name for param in seen["task_params"]] == ["task"]


def test_primfunc_generation_rejects_unannotated_iter_tensor_parameter():
    compiled = df.compile(
        make_unannotated_tensor_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [16]},
        block_size=128,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    with pytest.raises(
        df.DataflowPrimFuncLoweringError,
        match="primfunc handler lowering requires tensor parameter 'Source' to use T.Tensor annotation",
    ):
        df.lower_program_handlers_to_primfuncs(
            compiled.program,
            compiled.wrapper_spec,
            compiled.tensor_arg_plan,
            plan=compiled.plan,
            lower_to_cuda=False,
        )


def test_primfunc_generation_rejects_unannotated_finalize_output_parameter():
    compiled = df.compile(
        make_unannotated_finalize_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [16]},
        block_size=128,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )

    with pytest.raises(
        df.DataflowPrimFuncLoweringError,
        match="primfunc handler lowering requires finalize output 'Output' to use T.Tensor annotation",
    ):
        df.lower_program_handlers_to_primfuncs(
            compiled.program,
            compiled.wrapper_spec,
            compiled.tensor_arg_plan,
            plan=compiled.plan,
            lower_to_cuda=False,
        )
