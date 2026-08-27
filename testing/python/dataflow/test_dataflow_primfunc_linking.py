from __future__ import annotations

import os
import re
from dataclasses import replace

import pytest

import tilelang.language as T
import tilelang.dataflow as df
from tilelang.dataflow.experimental import debug_handlers as dataflow_debug


@T.dataflow_intermediate
class ScalarInter:
    value: T.int32


@T.dataflow_intermediate
class PairInter:
    a: T.float32
    b: T.Tensor((2,), T.float16)


@T.dataflow_intermediate
class PairScalarInter:
    a: T.int32
    b: T.int32


@T.dataflow_intermediate
class TensorFieldInter:
    score: T.int32
    vec: T.Tensor((2,), T.int32)


@T.dataflow.iter(range=("begin", "end"))
def split_scalar(Source: T.Tensor((64,), T.int32)) -> ScalarInter:
    value = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += Source[i]
    return ScalarInter(value=value)


@T.dataflow.iter(range=("begin", "end"), threads=128)
def split_scalar_cta_wide(Source: T.Tensor((64,), T.int32)) -> ScalarInter:
    value = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += Source[i]
    return ScalarInter(value=value)


@T.dataflow.iter(range=("begin", "end"))
def split_by_head(head: T.int32, Source: T.Tensor((2, 64), T.int32)) -> ScalarInter:
    value = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += Source[head, i]
    return ScalarInter(value=value)


@T.dataflow.iter(range=("begin", "end"))
def split_pair_layout(Source: T.Tensor((64,), T.float32), Bias: T.Tensor((2,), T.float16)) -> PairInter:
    total = T.float32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        total += Source[i]
    return PairInter(a=total, b=Bias)


@T.dataflow.iter(range=("begin", "end"))
def split_pair(A: T.Tensor((64,), T.int32), B: T.Tensor((64,), T.int32)) -> PairScalarInter:
    a = T.int32(0)
    b = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        a += A[i]
        b += B[i]
    return PairScalarInter(a=a, b=b)


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
def split_pair_reversed_return(A: T.Tensor((64,), T.int32), B: T.Tensor((64,), T.int32)) -> PairScalarInter:
    a = T.int32(0)
    b = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        a += A[i]
        b += B[i]
    return PairScalarInter(b=b, a=a)


@T.dataflow.iter(range=("begin", "end"))
def split_scalar_two_inputs(
    Source: T.Tensor((64,), T.int32),
    Bias: T.Tensor((64,), T.int32),
) -> ScalarInter:
    value = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += Source[i] + Bias[i]
    return ScalarInter(value=value)


@T.dataflow.iter(range=("begin", "end"))
def split_scalar_unused_input(Source: T.Tensor((64,), T.int32)) -> ScalarInter:
    value = T.int32(0)
    for _ in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += T.int32(1)
    return ScalarInter(value=value)


@T.dataflow.iter(range=("begin", "end"))
def split_scalar_unused_out_collision(Out: T.Tensor((64,), T.int32)) -> ScalarInter:
    value = T.int32(0)
    for _ in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += T.int32(1)
    return ScalarInter(value=value)


@T.dataflow.map(range=("begin", "end"))
def terminal_side_effect_map1(Source: T.Tensor((64,), T.int32)) -> ScalarInter:
    value = T.int32(0)
    for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end()):
        value += Source[i]
    return ScalarInter(value=value)


@T.dataflow.map(range=("begin", "end"))
def terminal_side_effect_map2(parts: list[ScalarInter], Output: T.Tensor((64,), T.int32)) -> ScalarInter:
    for _ in T.Parallel(1):
        Output[T.dataflow_range_begin()] = parts[0].value
    return ScalarInter(value=0)


@T.dataflow.map(range=("begin", "end"))
def four_input_map2(parts: list[ScalarInter], Output: T.Tensor((64,), T.int32)) -> ScalarInter:
    for _ in T.Parallel(1):
        Output[T.dataflow_range_begin()] = parts[0].value + parts[1].value + parts[2].value + parts[3].value
    return ScalarInter(value=0)


@T.dataflow.reduce
def reduce_scalar(items: list[ScalarInter]) -> ScalarInter:
    value = T.int32(0)
    for item in items:
        value += item.value
    return ScalarInter(value=value)


@T.dataflow.reduce
def reduce_pair_layout(items: list[PairInter]) -> PairInter:
    total = T.float32(0)
    for item in items:
        total += item.a
    return PairInter(a=total, b=items[0].b)


@T.dataflow.reduce
def reduce_pair(items: list[PairScalarInter]) -> PairScalarInter:
    a = T.int32(0)
    b = T.int32(0)
    for item in items:
        a += item.a
        b += item.b
    return PairScalarInter(a=a, b=b)


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
def reduce_pair_reversed_return(items: list[PairScalarInter]) -> PairScalarInter:
    a = T.int32(0)
    b = T.int32(0)
    for item in items:
        a += item.a
        b += item.b
    return PairScalarInter(b=b, a=a)


@T.dataflow.finalize
def finalize_scalar(inter: ScalarInter, Output: T.Tensor((2,), T.int32)) -> None:
    Output[T.dataflow_task_id()] = inter.value


@T.dataflow.finalize
def finalize_scalar_by_coords(
    inter: ScalarInter,
    batch: T.int32,
    head: T.int32,
    Output: T.Tensor((2, 2), T.int32),
) -> None:
    Output[batch, head] = inter.value


@T.dataflow.finalize
def finalize_pair_layout(inter: PairInter, Output: T.Tensor((2,), T.float32)) -> None:
    Output[T.dataflow_task_id()] = inter.a


@T.dataflow.finalize
def finalize_pair(
    inter: PairScalarInter,
    OA: T.Tensor((2,), T.int32),
    OB: T.Tensor((2,), T.int32),
) -> None:
    OA[T.dataflow_task_id()] = inter.a
    OB[T.dataflow_task_id()] = inter.b


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
def finalize_pair_reversed_outputs(
    inter: PairScalarInter,
    OA: T.Tensor((2,), T.int32),
    OB: T.Tensor((2,), T.int32),
) -> None:
    OB[T.dataflow_task_id()] = inter.b
    OA[T.dataflow_task_id()] = inter.a


@T.dataflow.finalize
def finalize_pair_a_only(
    inter: PairScalarInter,
    OA: T.Tensor((2,), T.int32),
) -> None:
    OA[T.dataflow_task_id()] = inter.a


def make_scalar_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(split_scalar(Source="Source"), task_args=("batch",), range_axis="kv")
        .reduce(reduce_scalar())
        .finalize(finalize_scalar(Output="Output"))
    )


def make_scalar_cta_wide_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(split_scalar_cta_wide(Source="Source"), task_args=("batch",), range_axis="kv")
        .reduce(reduce_scalar())
        .finalize(finalize_scalar(Output="Output"))
    )


def make_head_program():
    return (
        T.dataflow_program(task_domain=("head",), dynamic_ranges={"kv": "seq_lens"})
        .partial(split_by_head(Source="Source"), task_args=("head",), range_axis="kv")
        .reduce(reduce_scalar())
        .finalize(finalize_scalar(Output="Output"))
    )


def make_coord_output_program():
    return (
        T.dataflow_program(task_domain=("batch", "head"), dynamic_ranges={"kv": "seq_lens"})
        .partial(split_scalar(Source="Source"), task_args=("batch", "head"), range_axis="kv")
        .reduce(reduce_scalar())
        .finalize(finalize_scalar_by_coords(Output="Output"))
    )


def make_pair_layout_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(split_pair_layout(Source="Source", Bias="Bias"), task_args=("batch",), range_axis="kv")
        .reduce(reduce_pair_layout())
        .finalize(finalize_pair_layout(Output="Output"))
    )


def make_pair_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(split_pair(A="A", B="B"), task_args=("batch",), range_axis="kv")
        .reduce(reduce_pair())
        .finalize(finalize_pair(OA="OA", OB="OB"))
    )


def make_pair_reversed_return_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(split_pair_reversed_return(A="A", B="B"), task_args=("batch",), range_axis="kv")
        .reduce(reduce_pair_reversed_return())
        .finalize(finalize_pair(OA="OA", OB="OB"))
    )


def make_pair_reversed_finalize_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(split_pair(A="A", B="B"), task_args=("batch",), range_axis="kv")
        .reduce(reduce_pair())
        .finalize(finalize_pair_reversed_outputs(OA="OA", OB="OB"))
    )


def make_pair_a_only_finalize_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(split_pair(A="A", B="B"), task_args=("batch",), range_axis="kv")
        .reduce(reduce_pair())
        .finalize(finalize_pair_a_only(OA="OA"))
    )


def make_scalar_two_input_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(split_scalar_two_inputs(Source="Source", Bias="Bias"), task_args=("batch",), range_axis="kv")
        .reduce(reduce_scalar())
        .finalize(finalize_scalar(Output="Output"))
    )


def make_tensor_field_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(split_tensor_field(A="A", B="B"), task_args=("batch",), range_axis="kv")
        .reduce(reduce_tensor_field())
        .finalize(finalize_tensor_field(ScoreOut="ScoreOut", Vec0Out="Vec0Out", Vec1Out="Vec1Out"))
    )


def make_scalar_unused_input_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(split_scalar_unused_input(Source="Source"), task_args=("batch",), range_axis="kv")
        .reduce(reduce_scalar())
        .finalize(finalize_scalar(Output="Output"))
    )


def make_scalar_unused_out_collision_program():
    return (
        T.dataflow_program(task_domain=("batch",), dynamic_ranges={"kv": "seq_lens"})
        .partial(split_scalar_unused_out_collision(Out="Out"), task_args=("batch",), range_axis="kv")
        .reduce(reduce_scalar())
        .finalize(finalize_scalar(Output="Output"))
    )


def make_terminal_map_side_effect_program():
    return (
        T.dataflow_program(
            task_domain=("batch",),
            dynamic_ranges={"map1_range": "map1_ranges", "map2_range": "map2_ranges"},
        )
        .map(
            terminal_side_effect_map1(Source="Source"),
            name="map1",
            task_args=("batch",),
            range_axis="map1_range",
        )
        .reshared(
            input="map1",
            name="gather",
            output_type=ScalarInter,
            physical_output_type=ScalarInter,
            output_arity=1,
            policy="hbm_all_gather",
        )
        .map(
            terminal_side_effect_map2(Output="Output"),
            name="map2",
            input="gather",
            task_args=("batch",),
            range_axis="map2_range",
        )
    )


def make_four_input_map_program():
    return (
        T.dataflow_program(
            task_domain=("batch",),
            dynamic_ranges={"map1_range": "map1_ranges", "map2_range": "map2_ranges"},
        )
        .map(
            terminal_side_effect_map1(Source="Source"),
            name="map1",
            task_args=("batch",),
            range_axis="map1_range",
            range_tile=16,
        )
        .reshared(
            input="map1",
            name="gather",
            output_type=ScalarInter,
            physical_output_type=ScalarInter,
            output_arity=4,
            policy="cluster_shared_all_gather",
        )
        .map(
            four_input_map2(Output="Output"),
            name="map2",
            input="gather",
            task_args=("batch",),
            range_axis="map2_range",
            range_tile=16,
        )
    )


def test_handler_abi_layout_matches_runtime_slot_packing():
    from tilelang.dataflow.handler_abi import build_handler_abi

    compiled = df.compile(
        make_scalar_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )
    abi = build_handler_abi(compiled.program, compiled.wrapper_spec, compiled.tensor_arg_plan, compiled.plan)

    assert [handler.operator_kind for handler in abi.handlers] == ["iter", "reduce", "finalize"]
    assert abi.field_layouts[0].name == "value"
    assert abi.field_layouts[0].dtype == "int32"
    assert abi.field_layouts[0].c_type == "int32_t"
    assert abi.field_layouts[0].offset == 0
    assert abi.field_layouts[0].numel == 1
    assert abi.slot_bytes == compiled.packed_plan.slots[0].bytes
    assert abi.max_reduce_input_slots == 2


def test_handler_abi_multifield_layout_matches_runtime_slot_packing():
    from tilelang.dataflow.handler_abi import build_handler_abi

    compiled = df.compile(
        make_pair_layout_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )
    abi = build_handler_abi(compiled.program, compiled.wrapper_spec, compiled.tensor_arg_plan, compiled.plan)

    assert [(field.name, field.offset, field.numel) for field in abi.field_layouts] == [
        ("a", 0, 1),
        ("b", 4, 2),
    ]
    assert abi.slot_bytes == compiled.packed_plan.slots[0].bytes


def test_handler_abi_indexes_tensor_bindings_by_handler_parameter():
    from tilelang.dataflow.handler_abi import build_handler_abi

    compiled = df.compile(
        make_pair_layout_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )
    abi = build_handler_abi(compiled.program, compiled.wrapper_spec, compiled.tensor_arg_plan, compiled.plan)

    assert abi.tensor_index("iter", "split_pair_layout", "Source") == 0
    assert abi.tensor_index("iter", "split_pair_layout", "Bias") == 1
    assert abi.tensor_index("finalize", "finalize_pair_layout", "Output") == 2
    assert abi.tensor_indices_for_handler("iter", "split_pair_layout") == {"Source": 0, "Bias": 1}

    with pytest.raises(KeyError, match="Missing Dataflow tensor binding"):
        abi.tensor_index("reduce", "reduce_pair_layout", "Source")


def test_handler_abi_computes_multifield_offsets_without_runtime_compile():
    from tilelang.dataflow.handler_abi import field_layouts_for_intermediate

    intermediate = df.get_intermediate_type(PairInter)
    layouts, total_bytes = field_layouts_for_intermediate(intermediate)

    assert [(field.name, field.dtype, field.c_type, field.offset, field.numel) for field in layouts] == [
        ("a", "float32", "float", 0, 1),
        ("b", "float16", "half_t", 4, 2),
    ]
    assert total_bytes == 16


def test_handler_abi_rejects_dynamic_shape_extents_with_context():
    from tilelang.dataflow.handler_abi import field_layouts_for_intermediate
    from tilelang.dataflow.ir import IntermediateField, IntermediateType

    intermediate = IntermediateType(
        name="BadInter",
        fields=(
            IntermediateField(
                name="values",
                annotation=None,
                shape=("n",),
                dtype="float32",
            ),
        ),
        python_class=object,
    )

    with pytest.raises(NotImplementedError, match="field 'values'"):
        field_layouts_for_intermediate(intermediate)


def test_tilelang_cuda_codegen_can_emit_device_function_for_dataflow_handler():
    compiled = df.compile(
        make_scalar_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
        mode="inspect",
        inspection_stage="cuda",
    )

    assert compiled.primfunc_lowering is not None
    source = compiled.primfunc_lowering.cuda_source
    for handler in compiled.primfunc_lowering.handlers:
        symbol = handler.device_symbol
        device_definition = rf"__device__\s+__forceinline__\s+void\s+{re.escape(symbol)}\s*\([^;{{]*\)\s*\{{"
        old_kernel = rf'extern\s+"C"\s+__global__\s+void\s+{re.escape(symbol)}\s*\('
        assert len(re.findall(device_definition, source, flags=re.S)) == 1
        assert not re.search(old_kernel, source)


def test_primfunc_cuda_source_preserves_logical_device_abi():
    compiled = df.compile(
        make_scalar_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
        mode="inspect",
        inspection_stage="cuda",
    )

    assert compiled.primfunc_lowering is not None
    source = compiled.primfunc_lowering.cuda_source
    handlers = {handler.operator_kind: handler for handler in compiled.primfunc_lowering.handlers}

    assert cuda_device_signature(source, handlers["iter"].device_symbol) == (
        "void dataflow_primfunc_split_scalar_device_kernel(const int* Source, int* Out, uint range_begin, uint range_end, uint task_id)"
    )
    assert cuda_device_signature(source, handlers["reduce"].device_symbol) == (
        "void dataflow_primfunc_reduce_scalar_device_kernel(const int* Item0, const int* Item1, int* Out, uint input_count, uint task_id)"
    )
    assert "dataflow_primfunc_reduce_scalar_device_kernel(const int* __restrict__ Item0" not in source
    assert "dataflow_primfunc_reduce_scalar_device_kernel(const int* Item0, const int* Item1, int* Out" in source
    assert cuda_device_signature(source, handlers["finalize"].device_symbol) == (
        "void dataflow_primfunc_finalize_scalar_device_kernel(const int* Inter, int* Output, uint task_id)"
    )


def test_primfunc_handlers_use_logical_device_abi():
    compiled = df.compile(
        make_scalar_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
        mode="inspect",
        inspection_stage="ir",
    )

    assert compiled.primfunc_lowering is not None
    handlers = {handler.operator_kind: handler for handler in compiled.primfunc_lowering.handlers}

    iter_script = handlers["iter"].prim_func.script()
    reduce_script = handlers["reduce"].prim_func.script()
    finalize_script = handlers["finalize"].prim_func.script()

    assert handlers["iter"].global_symbol == "dataflow_primfunc_split_scalar_device"
    assert handlers["iter"].device_symbol == "dataflow_primfunc_split_scalar_device_kernel"
    assert "range_begin" in iter_script
    assert "range_end" in iter_script
    assert "task_id" in iter_script
    assert "for i in range(range_begin" in iter_script
    assert "range_end - range_begin" in iter_script
    assert "Source" in iter_script
    assert "Out" in iter_script

    assert handlers["reduce"].global_symbol == "dataflow_primfunc_reduce_scalar_device"
    assert handlers["reduce"].device_symbol == "dataflow_primfunc_reduce_scalar_device_kernel"
    assert "Item0" in reduce_script
    assert "Item1" in reduce_script
    assert "input_count" in reduce_script
    assert "task_id" in reduce_script
    assert 'for item in range(T.Cast("int32", input_count))' in reduce_script

    assert handlers["finalize"].global_symbol == "dataflow_primfunc_finalize_scalar_device"
    assert handlers["finalize"].device_symbol == "dataflow_primfunc_finalize_scalar_device_kernel"
    assert "task_id" in finalize_script
    assert "Inter[0]" in finalize_script
    assert "Output[task_id]" in finalize_script


def test_primfunc_iter_handler_records_thread_count_and_kernel_threads():
    compiled = df.compile(
        make_scalar_cta_wide_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=64,
        mode="inspect",
        inspection_stage="ir",
    )

    assert compiled.primfunc_lowering is not None
    handlers = {handler.operator_kind: handler for handler in compiled.primfunc_lowering.handlers}
    iter_script = handlers["iter"].prim_func.script()

    assert handlers["iter"].thread_count == 128
    assert 'T.launch_thread("threadIdx.x", 128)' in iter_script


def test_compile_links_primfunc_device_handlers_into_wrapper_source():
    compiled = df.compile(
        make_scalar_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
        wrapper_name="dataflow_primfunc_linked_source",
    )

    source = compiled.wrapper_source
    assert compiled.primfunc_lowering is not None
    assert "__device__" in source
    assert "dataflow_primfunc_split_scalar_device_kernel" in source
    assert "dataflow_primfunc_split_scalar_device" in source
    assert "tensor_args[0].data_ptr" in source
    assert "tensor_args[1].data_ptr" in source
    assert "dataflow_primfunc_split_scalar_device_kernel(Source_tensor, output_value" in source
    assert "dataflow_primfunc_reduce_scalar_device_kernel(item0, item1, output_value" in source
    assert "dataflow_primfunc_finalize_scalar_device_kernel(inter_value, Output_tensor" in source
    assert "PrimFunc Dataflow handler" not in source


def test_linked_primfunc_map_side_effect_tensor_args_keep_mutable_pointer():
    compiled = df.compile(
        make_terminal_map_side_effect_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"map1_range": [64], "map2_range": [64]},
        block_size=32,
        wrapper_name="dataflow_primfunc_terminal_map_side_effect_source",
    )

    source = compiled.wrapper_source

    assert re.search(
        r"void dataflow_primfunc_terminal_side_effect_map2_device_kernel\("
        r"const int\* __restrict__ Item0, int\* __restrict__ Out, "
        r"int\* __restrict__ Output, uint range_begin",
        source,
    )
    assert "int32_t *Output_tensor = reinterpret_cast<int32_t *>" in source
    assert "const int32_t *Output_tensor" not in source
    assert (
        "dataflow_primfunc_terminal_side_effect_map2_device_kernel("
        "dataflow_primfunc_slot_field<int32_t>(shared_base, "
        "slots[input_slots[handler_args.input_slot_offset + 0u]], 0u), "
        "output_value, Output_tensor"
    ) in source


def test_linked_primfunc_compacts_non_global_map_input_slots():
    compiled = df.compile(
        make_four_input_map_program(),
        topology=df.GPUTopology(sm_count=4, cluster_size=4),
        range_lengths={"map1_range": [64], "map2_range": [16]},
        block_size=16,
        include_exit=False,
    )

    source = compiled.wrapper_source
    assert "const uint32_t input_slot0_id" not in source
    assert "const tl::DataflowSlot &input_slot0" not in source
    assert "const int32_t *item0 =" not in source
    assert "const int32_t *item3 =" not in source
    assert (
        "dataflow_primfunc_slot_field<int32_t>(shared_base, slots[input_slots[handler_args.input_slot_offset + 0u]], 0u);"
    ) not in source
    assert (
        "dataflow_primfunc_slot_field<int32_t>(shared_base, slots[input_slots[handler_args.input_slot_offset + 3u]], 0u);"
    ) not in source
    assert ("dataflow_primfunc_slot_field<int32_t>(shared_base, slots[input_slots[handler_args.input_slot_offset + 0u]], 0u)") in source
    assert ("dataflow_primfunc_slot_field<int32_t>(shared_base, slots[input_slots[handler_args.input_slot_offset + 3u]], 0u)") in source


def test_linked_primfunc_iter_adapter_passes_task_coords():
    compiled = df.compile(
        make_head_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [64, 64]},
        task_extents=(2,),
        block_size=32,
        wrapper_name="dataflow_primfunc_head_task_coord_linked_source",
    )

    assert compiled.primfunc_lowering is not None
    handlers = {handler.operator_kind: handler for handler in compiled.primfunc_lowering.handlers}
    iter_symbol = handlers["iter"].device_symbol
    assert cuda_device_signature(compiled.primfunc_lowering.cuda_source, iter_symbol) == (
        "void dataflow_primfunc_split_by_head_device_kernel("
        "const int* Source, int* Out, int head, uint range_begin, uint range_end, uint task_id)"
    )
    assert "handler_args.task_coord_count < 1u" in compiled.wrapper_source
    assert wrapper_call_line(compiled.wrapper_source, iter_symbol) == (
        "dataflow_primfunc_split_by_head_device_kernel("
        "Source_tensor, output_value, task_coords[handler_args.task_coord_offset], "
        "handler_args.range_begin, handler_args.range_end, handler_args.task_id);"
    )


def test_linked_primfunc_iter_adapter_supports_cta_wide_threads():
    compiled = df.compile(
        make_scalar_cta_wide_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=64,
        wrapper_name="dataflow_primfunc_cta_wide_linked_source",
    )

    assert compiled.primfunc_lowering is not None
    handlers = {handler.operator_kind: handler for handler in compiled.primfunc_lowering.handlers}
    iter_symbol = handlers["iter"].device_symbol
    handler_body = wrapper_handler_body(compiled.wrapper_source, iter_symbol)

    assert handlers["iter"].thread_count == 128
    assert "!tl::dataflow_is_leader_thread()" not in handler_body
    assert (
        f"{iter_symbol}(Source_tensor, output_value, handler_args.range_begin, handler_args.range_end, handler_args.task_id);"
    ) in handler_body
    assert "__syncthreads();" in handler_body


def test_linked_primfunc_finalize_adapter_passes_task_coords():
    compiled = df.compile(
        make_coord_output_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [16, 16, 16, 16]},
        task_extents=(2, 2),
        block_size=16,
        wrapper_name="dataflow_primfunc_finalize_task_coord_linked_source",
    )

    assert compiled.primfunc_lowering is not None
    handlers = {handler.operator_kind: handler for handler in compiled.primfunc_lowering.handlers}
    finalize_symbol = handlers["finalize"].device_symbol
    assert cuda_device_signature(compiled.primfunc_lowering.cuda_source, finalize_symbol) == (
        "void dataflow_primfunc_finalize_scalar_by_coords_device_kernel(const int* Inter, int* Output, int batch, int head, uint task_id)"
    )
    assert "handler_args.task_coord_count < 2u" in compiled.wrapper_source
    assert wrapper_call_line(compiled.wrapper_source, finalize_symbol) == (
        "dataflow_primfunc_finalize_scalar_by_coords_device_kernel("
        "inter_value, Output_tensor, task_coords[handler_args.task_coord_offset], "
        "task_coords[handler_args.task_coord_offset + 1u], handler_args.task_id);"
    )


def test_linked_queue_terminal_finalize_omits_redundant_completion_barrier():
    compiled = df.compile(
        make_scalar_cta_wide_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=64,
        wrapper_name="dataflow_terminal_finalize_completion_proof",
        primfunc_thread_count_overrides={2: 128},
    )

    handlers = {handler.operator_kind: handler for handler in compiled.primfunc_lowering.handlers}
    finalize_symbol = handlers["finalize"].device_symbol
    finalize_body = wrapper_handler_body(
        compiled.wrapper_source,
        finalize_symbol,
    )

    assert handlers["finalize"].thread_count > 1
    assert "__syncthreads();" not in finalize_body
    finalize_case = compiled.wrapper_source.index(f"case {handlers['finalize'].handler_id}u:")
    assert "return false;" in compiled.wrapper_source[finalize_case : compiled.wrapper_source.index("default:", finalize_case)]


def test_terminal_finalize_proof_is_disabled_by_any_nonterminal_use():
    from tilelang.dataflow.compiler import queue_terminal_finalize_variants

    compiled = df.compile(
        make_scalar_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [64, 64]},
        block_size=64,
        mode="debug",
        _experimental_debug_handler=dataflow_debug.EMPTY_HANDLER.name,
    )
    finalizers = tuple(instruction for instruction in compiled.plan.instructions if instruction.opcode is df.DataflowOpcode.FINALIZE)
    assert len(finalizers) == 2
    first = finalizers[0]
    first_queue = compiled.plan.queues[first.sm_id]
    exit_instruction = next(instruction for instruction in first_queue if instruction.opcode is df.DataflowOpcode.EXIT)
    nonterminal_plan = replace(
        compiled.plan,
        queues={
            **compiled.plan.queues,
            first.sm_id: (
                first,
                replace(exit_instruction, instruction_id=10_000),
                *tuple(
                    instruction
                    for instruction in first_queue
                    if instruction.instruction_id != first.instruction_id and instruction.opcode is not df.DataflowOpcode.EXIT
                ),
                exit_instruction,
            ),
        },
    )

    assert queue_terminal_finalize_variants(nonterminal_plan) == frozenset()


def test_linked_primfunc_handlers_generate_multifield_slot_adapters():
    compiled = df.compile(
        make_pair_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
        wrapper_name="dataflow_primfunc_pair_linked_source",
    )

    source = compiled.wrapper_source
    assert compiled.primfunc_lowering is not None
    handlers = {handler.operator_kind: handler for handler in compiled.primfunc_lowering.handlers}
    assert cuda_device_signature(compiled.primfunc_lowering.cuda_source, handlers["iter"].device_symbol) == (
        "void dataflow_primfunc_split_pair_device_kernel("
        "const int* A, const int* B, int* Out_a, int* Out_b, uint range_begin, uint range_end, uint task_id)"
    )
    assert "dataflow_primfunc_split_pair_device" in source
    assert "dataflow_primfunc_reduce_pair_device" in source
    assert "dataflow_primfunc_finalize_pair_device" in source
    assert (
        "dataflow_primfunc_split_pair_device_kernel("
        "A_tensor, B_tensor, output_a, output_b, "
        "handler_args.range_begin, handler_args.range_end, handler_args.task_id)"
    ) in source
    assert "output_a" in source
    assert "output_b" in source
    assert "output_a = dataflow_primfunc_slot_field<int32_t>(shared_base, output_slot, 0u)" in source
    assert "output_b = dataflow_primfunc_slot_field<int32_t>(shared_base, output_slot, 4u)" in source
    assert "item0_a" in source
    assert "item1_a" in source
    assert "item0_b" in source
    assert "item1_b" in source
    assert compiled.packed_plan.slots[0].bytes == 16


def test_linked_primfunc_multifield_adapters_use_declaration_order_for_reversed_returns():
    compiled = df.compile(
        make_pair_reversed_return_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
        wrapper_name="dataflow_primfunc_pair_reversed_return",
    )

    handlers = {handler.operator_kind: handler for handler in compiled.primfunc_lowering.handlers}
    iter_symbol = handlers["iter"].device_symbol
    reduce_symbol = handlers["reduce"].device_symbol

    assert cuda_device_signature(compiled.primfunc_lowering.cuda_source, iter_symbol) == (
        "void dataflow_primfunc_split_pair_reversed_return_device_kernel("
        "const int* A, const int* B, int* Out_a, int* Out_b, uint range_begin, uint range_end, uint task_id)"
    )
    assert wrapper_call_line(compiled.wrapper_source, iter_symbol) == (
        "dataflow_primfunc_split_pair_reversed_return_device_kernel("
        "A_tensor, B_tensor, output_a, output_b, "
        "handler_args.range_begin, handler_args.range_end, handler_args.task_id);"
    )
    assert cuda_device_signature(compiled.primfunc_lowering.cuda_source, reduce_symbol) == (
        "void dataflow_primfunc_reduce_pair_reversed_return_device_kernel("
        "const int* Item0_a, const int* Item0_b, const int* Item1_a, const int* Item1_b, "
        "int* Out_a, int* Out_b, uint input_count, uint task_id)"
    )
    assert wrapper_call_line(compiled.wrapper_source, reduce_symbol) == (
        "dataflow_primfunc_reduce_pair_reversed_return_device_kernel("
        "item0_a, item0_b, item1_a, item1_b, output_a, output_b, "
        "handler_args.input_slot_count, handler_args.task_id);"
    )


def test_linked_primfunc_finalize_adapter_preserves_output_binding_order():
    compiled = df.compile(
        make_pair_reversed_finalize_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
        wrapper_name="dataflow_primfunc_pair_reversed_finalize",
    )

    handlers = {handler.operator_kind: handler for handler in compiled.primfunc_lowering.handlers}
    symbol = handlers["finalize"].device_symbol
    assert cuda_device_signature(compiled.primfunc_lowering.cuda_source, symbol) == (
        "void dataflow_primfunc_finalize_pair_reversed_outputs_device_kernel("
        "const int* Inter_a, const int* Inter_b, int* OA, int* OB, uint task_id)"
    )
    assert wrapper_call_line(compiled.wrapper_source, symbol) == (
        "dataflow_primfunc_finalize_pair_reversed_outputs_device_kernel(inter_a, inter_b, OA_tensor, OB_tensor, handler_args.task_id);"
    )


def test_linked_primfunc_finalize_adapter_omits_unused_intermediate_fields():
    compiled = df.compile(
        make_pair_a_only_finalize_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
        wrapper_name="dataflow_primfunc_pair_a_only_finalize",
    )

    handlers = {handler.operator_kind: handler for handler in compiled.primfunc_lowering.handlers}
    symbol = handlers["finalize"].device_symbol
    assert cuda_device_signature(compiled.primfunc_lowering.cuda_source, symbol) == (
        "void dataflow_primfunc_finalize_pair_a_only_device_kernel(const int* Inter_a, int* OA, uint task_id)"
    )
    call_line = wrapper_call_line(compiled.wrapper_source, symbol)
    assert call_line == "dataflow_primfunc_finalize_pair_a_only_device_kernel(inter_a, OA_tensor, handler_args.task_id);"
    assert "inter_b" not in call_line


def test_linked_primfunc_scalar_adapter_supports_multi_tensor_iter_handler():
    compiled = df.compile(
        make_scalar_two_input_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
    )

    source = compiled.wrapper_source
    handlers = {handler.operator_kind: handler for handler in compiled.primfunc_lowering.handlers}
    assert cuda_device_signature(compiled.primfunc_lowering.cuda_source, handlers["iter"].device_symbol) == (
        "void dataflow_primfunc_split_scalar_two_inputs_device_kernel("
        "const int* Source, const int* Bias, int* Out, uint range_begin, uint range_end, uint task_id)"
    )
    assert "Source_tensor" in source
    assert "Bias_tensor" in source
    assert "dataflow_primfunc_split_scalar_two_inputs_device_kernel(Source_tensor, Bias_tensor, output_value" in source


def test_linked_primfunc_handlers_generate_tensor_field_slot_adapters():
    compiled = df.compile(
        make_tensor_field_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
        wrapper_name="dataflow_primfunc_tensor_field_linked_source",
    )

    source = compiled.wrapper_source
    handlers = {handler.operator_kind: handler for handler in compiled.primfunc_lowering.handlers}
    assert cuda_device_signature(compiled.primfunc_lowering.cuda_source, handlers["iter"].device_symbol) == (
        "void dataflow_primfunc_split_tensor_field_device_kernel("
        "const int* A, const int* B, int* Out_score, int* Out_vec, "
        "uint range_begin, uint range_end, uint task_id)"
    )
    assert cuda_device_signature(compiled.primfunc_lowering.cuda_source, handlers["reduce"].device_symbol) == (
        "void dataflow_primfunc_reduce_tensor_field_device_kernel("
        "const int* Item0_score, const int* Item0_vec, const int* Item1_score, const int* Item1_vec, "
        "int* Out_score, int* Out_vec, "
        "uint input_count, uint task_id)"
    )
    assert cuda_device_signature(compiled.primfunc_lowering.cuda_source, handlers["finalize"].device_symbol) == (
        "void dataflow_primfunc_finalize_tensor_field_device_kernel("
        "const int* Inter_score, const int* Inter_vec, int* ScoreOut, int* Vec0Out, int* Vec1Out, uint task_id)"
    )
    assert "output_score = dataflow_primfunc_slot_field<int32_t>(shared_base, output_slot, 0u)" in source
    assert "output_vec = dataflow_primfunc_slot_field<int32_t>(shared_base, output_slot, 4u)" in source
    assert "dataflow_primfunc_load_reduce_items" not in source
    assert "dataflow_primfunc_slot_global_field" not in source
    assert "_dataflow_use_global_slots" not in source
    assert "item0_score_global" not in source
    assert "item1_vec_global" not in source
    assert "item0_score = dataflow_primfunc_slot_field<int32_t>(shared_base, input_slot0, 0u);" in source
    assert "item1_vec = dataflow_primfunc_slot_field<int32_t>(shared_base, input_slot1, 4u);" in source
    assert (
        "dataflow_primfunc_reduce_tensor_field_device_kernel("
        "item0_score, item0_vec, item1_score, item1_vec, output_score, output_vec, "
        "handler_args.input_slot_count, handler_args.task_id);"
    ) in source
    assert compiled.packed_plan.slots[0].bytes == 16


def test_linked_primfunc_scalar_adapter_accepts_eliminated_unused_tensor_binding():
    compiled = df.compile(
        make_scalar_unused_input_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
    )

    assert compiled.primfunc_lowering is not None
    iter_artifact = next(
        artifact
        for handler, artifact in zip(
            compiled.primfunc_lowering.handlers,
            compiled.primfunc_lowering.codegen_artifacts,
        )
        if handler.operator_kind == "iter"
    )
    assert iter_artifact.params_for_role(df.DataflowHandlerParamRole.TENSOR_ARG) == ()


def test_handler_metadata_rejects_tensor_and_output_slot_role_collision():
    with pytest.raises(
        df.DataflowPrimFuncLoweringError,
        match=r"assigns conflicting roles to parameter 'Out'",
    ):
        df.compile(
            make_scalar_unused_out_collision_program(),
            topology=df.GPUTopology(sm_count=1, cluster_size=1),
            range_lengths={"kv": [64]},
            block_size=32,
        )


def test_primfunc_inspection_launch_is_rejected_without_wrapper_generation():
    unlinked = df.compile(
        make_scalar_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
        mode="inspect",
        inspection_stage="cuda",
    )
    assert unlinked.wrapper_source == ""
    with pytest.raises(NotImplementedError, match="no executable wrapper was generated"):
        unlinked()

    linked = df.compile(
        make_scalar_program(),
        topology=df.GPUTopology(sm_count=1, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
    )
    assert linked.options["link_primfunc_handlers"] is True
    assert "dataflow_primfunc_split_scalar_device_kernel" in linked.wrapper_source


def cuda_device_signature(source: str, symbol: str) -> str:
    match = re.search(
        rf"__device__\s+__forceinline__\s+void\s+{re.escape(symbol)}\s*\(([^)]*)\)",
        source,
        flags=re.S,
    )
    assert match is not None
    params = re.sub(r"\s+", " ", match.group(1).replace("__restrict__", "")).strip()
    params = re.sub(r"\s*,\s*", ", ", params)
    params = re.sub(r"\s+\*", "* ", params)
    return f"void {symbol}({params})"


def wrapper_call_line(source: str, symbol: str) -> str:
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{symbol}("):
            return stripped
    raise AssertionError(f"missing wrapper call for {symbol}")


def wrapper_handler_body(source: str, symbol: str) -> str:
    call_index = source.find(f"  {symbol}(")
    assert call_index != -1
    function_start = source.rfind("TL_DEVICE void", 0, call_index)
    assert function_start != -1
    next_function = source.find("\nTL_DEVICE void", function_start + 1)
    if next_function == -1:
        next_function = len(source)
    return source[function_start:next_function]


def require_executable_cuda():
    try:
        from tilelang.contrib import nvcc
    except Exception as err:
        pytest.skip(f"CUDA toolkit/NVCC unavailable: {err}")
    try:
        compiler = nvcc.get_nvcc_compiler()
    except Exception as err:
        pytest.skip(f"CUDA toolkit/NVCC unavailable: {err}")
    if not os.path.exists(compiler):
        pytest.skip(f"NVCC not found at {compiler}")

    try:
        from cuda.bindings import driver
    except Exception as err:
        pytest.skip(f"CUDA driver bindings unavailable: {err}")

    result = driver.cuInit(0)[0]
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA driver unavailable: {result}")
    result, count = driver.cuDeviceGetCount()
    if result != driver.CUresult.CUDA_SUCCESS or count == 0:
        pytest.skip(f"CUDA device unavailable: {result}, count={count}")

    result, device = driver.cuDeviceGet(0)
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA device 0 unavailable: {result}")
    result, context = driver.cuDevicePrimaryCtxRetain(device)
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA primary context unavailable: {result}")
    result, previous_context = driver.cuCtxGetCurrent()
    if result != driver.CUresult.CUDA_SUCCESS:
        driver.cuDevicePrimaryCtxRelease(device)
        pytest.skip(f"CUDA current context query failed: {result}")
    restore_result = driver.CUresult.CUDA_SUCCESS
    try:
        result = driver.cuCtxSetCurrent(context)[0]
    finally:
        restore_result = driver.cuCtxSetCurrent(previous_context)[0]
        driver.cuDevicePrimaryCtxRelease(device)
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA context activation failed: {result}")
    if restore_result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA context restore failed: {restore_result}")


def require_cluster_launch_cuda():
    require_executable_cuda()

    from cuda.bindings import driver

    result, device = driver.cuDeviceGet(0)
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA device 0 unavailable: {result}")

    cluster_launch_attr = getattr(
        driver.CUdevice_attribute,
        "CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH",
        None,
    )
    if cluster_launch_attr is None:
        pytest.skip("CUDA driver bindings do not expose cluster launch support attribute")
    result, supported = driver.cuDeviceGetAttribute(cluster_launch_attr, device)
    if result != driver.CUresult.CUDA_SUCCESS:
        pytest.skip(f"CUDA cluster launch support query failed: {result}")
    if not supported:
        pytest.skip("CUDA device does not support cluster launch")


def check_cuda(driver, result, action: str):
    if result != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{action} failed: {result}")


def test_linked_primfunc_handlers_execute_tensor_dataflow_on_cuda():
    require_executable_cuda()

    import ctypes
    from cuda.bindings import driver

    result, device = driver.cuDeviceGet(0)
    check_cuda(driver, result, "cuDeviceGet")
    result, context = driver.cuDevicePrimaryCtxRetain(device)
    check_cuda(driver, result, "cuDevicePrimaryCtxRetain")
    result, previous_context = driver.cuCtxGetCurrent()
    if result != driver.CUresult.CUDA_SUCCESS:
        driver.cuDevicePrimaryCtxRelease(device)
        check_cuda(driver, result, "cuCtxGetCurrent")

    allocations = []
    try:
        try:
            result = driver.cuCtxSetCurrent(context)[0]
            check_cuda(driver, result, "cuCtxSetCurrent")

            input_values = list(range(1, 129))
            output_count = 2
            input_bytes = len(input_values) * ctypes.sizeof(ctypes.c_int32)
            output_bytes = output_count * ctypes.sizeof(ctypes.c_int32)
            input_host = (ctypes.c_int32 * len(input_values))(*input_values)
            output_zero = (ctypes.c_int32 * output_count)()
            output_host = (ctypes.c_int32 * output_count)()

            result, input_ptr = driver.cuMemAlloc(input_bytes)
            check_cuda(driver, result, "cuMemAlloc(input)")
            allocations.append(input_ptr)
            result, output_ptr = driver.cuMemAlloc(output_bytes)
            check_cuda(driver, result, "cuMemAlloc(output)")
            allocations.append(output_ptr)
            check_cuda(driver, driver.cuMemcpyHtoD(input_ptr, input_host, input_bytes)[0], "cuMemcpyHtoD(input)")
            check_cuda(
                driver,
                driver.cuMemcpyHtoD(output_ptr, output_zero, output_bytes)[0],
                "cuMemcpyHtoD(output)",
            )

            compiled = df.compile(
                make_scalar_program(),
                topology=df.GPUTopology(sm_count=4, cluster_size=1),
                range_lengths={"kv": [128, 96]},
                task_extents=(2,),
                block_size=32,
                wrapper_name="dataflow_primfunc_linked_executable",
            )
            assert any(comm.kind is df.DataflowCommKind.HBM_SEND for comm in compiled.plan.comms)
            assert any(comm.kind is df.DataflowCommKind.HBM_RECV for comm in compiled.plan.comms)

            execution = compiled(Source=int(input_ptr), Output=int(output_ptr))
            check_cuda(
                driver,
                driver.cuMemcpyDtoH(output_host, output_ptr, output_bytes)[0],
                "cuMemcpyDtoH(output)",
            )

            assert execution.tensor_arg_count == 2
            assert list(output_host) == [
                sum(input_values[:128]),
                sum(input_values[:96]),
            ]
        finally:
            for ptr in reversed(allocations):
                driver.cuMemFree(ptr)
    finally:
        restore_result = driver.cuCtxSetCurrent(previous_context)[0]
        driver.cuDevicePrimaryCtxRelease(device)
        check_cuda(driver, restore_result, "cuCtxSetCurrent(previous)")


def test_linked_primfunc_handlers_execute_tensor_field_dataflow_on_cuda():
    require_executable_cuda()

    import ctypes
    from cuda.bindings import driver

    result, device = driver.cuDeviceGet(0)
    check_cuda(driver, result, "cuDeviceGet")
    result, context = driver.cuDevicePrimaryCtxRetain(device)
    check_cuda(driver, result, "cuDevicePrimaryCtxRetain")
    result, previous_context = driver.cuCtxGetCurrent()
    if result != driver.CUresult.CUDA_SUCCESS:
        driver.cuDevicePrimaryCtxRelease(device)
        check_cuda(driver, result, "cuCtxGetCurrent")

    allocations = []
    try:
        try:
            result = driver.cuCtxSetCurrent(context)[0]
            check_cuda(driver, result, "cuCtxSetCurrent")

            a_values = list(range(1, 129))
            b_values = list(range(3, 131))
            output_count = 2
            input_bytes = len(a_values) * ctypes.sizeof(ctypes.c_int32)
            output_bytes = output_count * ctypes.sizeof(ctypes.c_int32)
            a_host = (ctypes.c_int32 * len(a_values))(*a_values)
            b_host = (ctypes.c_int32 * len(b_values))(*b_values)
            output_zero = (ctypes.c_int32 * output_count)()
            score_host = (ctypes.c_int32 * output_count)()
            vec0_host = (ctypes.c_int32 * output_count)()
            vec1_host = (ctypes.c_int32 * output_count)()

            result, a_ptr = driver.cuMemAlloc(input_bytes)
            check_cuda(driver, result, "cuMemAlloc(A)")
            allocations.append(a_ptr)
            result, b_ptr = driver.cuMemAlloc(input_bytes)
            check_cuda(driver, result, "cuMemAlloc(B)")
            allocations.append(b_ptr)
            result, score_ptr = driver.cuMemAlloc(output_bytes)
            check_cuda(driver, result, "cuMemAlloc(ScoreOut)")
            allocations.append(score_ptr)
            result, vec0_ptr = driver.cuMemAlloc(output_bytes)
            check_cuda(driver, result, "cuMemAlloc(Vec0Out)")
            allocations.append(vec0_ptr)
            result, vec1_ptr = driver.cuMemAlloc(output_bytes)
            check_cuda(driver, result, "cuMemAlloc(Vec1Out)")
            allocations.append(vec1_ptr)

            check_cuda(driver, driver.cuMemcpyHtoD(a_ptr, a_host, input_bytes)[0], "cuMemcpyHtoD(A)")
            check_cuda(driver, driver.cuMemcpyHtoD(b_ptr, b_host, input_bytes)[0], "cuMemcpyHtoD(B)")
            check_cuda(driver, driver.cuMemcpyHtoD(score_ptr, output_zero, output_bytes)[0], "cuMemcpyHtoD(ScoreOut)")
            check_cuda(driver, driver.cuMemcpyHtoD(vec0_ptr, output_zero, output_bytes)[0], "cuMemcpyHtoD(Vec0Out)")
            check_cuda(driver, driver.cuMemcpyHtoD(vec1_ptr, output_zero, output_bytes)[0], "cuMemcpyHtoD(Vec1Out)")

            compiled = df.compile(
                make_tensor_field_program(),
                topology=df.GPUTopology(sm_count=4, cluster_size=1),
                range_lengths={"kv": [128, 96]},
                task_extents=(2,),
                block_size=32,
                wrapper_name="dataflow_primfunc_tensor_field_executable",
            )
            assert any(comm.kind is df.DataflowCommKind.HBM_SEND for comm in compiled.plan.comms)
            assert any(comm.kind is df.DataflowCommKind.HBM_RECV for comm in compiled.plan.comms)

            execution = compiled(
                A=int(a_ptr),
                B=int(b_ptr),
                ScoreOut=int(score_ptr),
                Vec0Out=int(vec0_ptr),
                Vec1Out=int(vec1_ptr),
            )
            check_cuda(driver, driver.cuMemcpyDtoH(score_host, score_ptr, output_bytes)[0], "cuMemcpyDtoH(ScoreOut)")
            check_cuda(driver, driver.cuMemcpyDtoH(vec0_host, vec0_ptr, output_bytes)[0], "cuMemcpyDtoH(Vec0Out)")
            check_cuda(driver, driver.cuMemcpyDtoH(vec1_host, vec1_ptr, output_bytes)[0], "cuMemcpyDtoH(Vec1Out)")

            expected_score = [sum(a_values[:128]), sum(a_values[:96])]
            expected_vec0 = [
                sum(a + b for a, b in zip(a_values[:128], b_values[:128])),
                sum(a + b for a, b in zip(a_values[:96], b_values[:96])),
            ]
            expected_vec1 = [
                sum(a * 2 + b for a, b in zip(a_values[:128], b_values[:128])),
                sum(a * 2 + b for a, b in zip(a_values[:96], b_values[:96])),
            ]

            assert execution.tensor_arg_count == 5
            assert list(score_host) == expected_score
            assert list(vec0_host) == expected_vec0
            assert list(vec1_host) == expected_vec1
        finally:
            for ptr in reversed(allocations):
                driver.cuMemFree(ptr)
    finally:
        restore_result = driver.cuCtxSetCurrent(previous_context)[0]
        driver.cuDevicePrimaryCtxRelease(device)
        check_cuda(driver, restore_result, "cuCtxSetCurrent(previous)")


def test_linked_primfunc_handlers_execute_tensor_field_cluster_dataflow_on_cuda():
    require_cluster_launch_cuda()

    import ctypes
    from cuda.bindings import driver

    result, device = driver.cuDeviceGet(0)
    check_cuda(driver, result, "cuDeviceGet")
    result, context = driver.cuDevicePrimaryCtxRetain(device)
    check_cuda(driver, result, "cuDevicePrimaryCtxRetain")
    result, previous_context = driver.cuCtxGetCurrent()
    if result != driver.CUresult.CUDA_SUCCESS:
        driver.cuDevicePrimaryCtxRelease(device)
        check_cuda(driver, result, "cuCtxGetCurrent")

    allocations = []
    try:
        try:
            result = driver.cuCtxSetCurrent(context)[0]
            check_cuda(driver, result, "cuCtxSetCurrent")

            a_values = list(range(1, 65))
            b_values = list(range(3, 67))
            output_count = 2
            input_bytes = len(a_values) * ctypes.sizeof(ctypes.c_int32)
            output_bytes = output_count * ctypes.sizeof(ctypes.c_int32)
            a_host = (ctypes.c_int32 * len(a_values))(*a_values)
            b_host = (ctypes.c_int32 * len(b_values))(*b_values)
            output_zero = (ctypes.c_int32 * output_count)()
            score_host = (ctypes.c_int32 * output_count)()
            vec0_host = (ctypes.c_int32 * output_count)()
            vec1_host = (ctypes.c_int32 * output_count)()

            result, a_ptr = driver.cuMemAlloc(input_bytes)
            check_cuda(driver, result, "cuMemAlloc(A)")
            allocations.append(a_ptr)
            result, b_ptr = driver.cuMemAlloc(input_bytes)
            check_cuda(driver, result, "cuMemAlloc(B)")
            allocations.append(b_ptr)
            result, score_ptr = driver.cuMemAlloc(output_bytes)
            check_cuda(driver, result, "cuMemAlloc(ScoreOut)")
            allocations.append(score_ptr)
            result, vec0_ptr = driver.cuMemAlloc(output_bytes)
            check_cuda(driver, result, "cuMemAlloc(Vec0Out)")
            allocations.append(vec0_ptr)
            result, vec1_ptr = driver.cuMemAlloc(output_bytes)
            check_cuda(driver, result, "cuMemAlloc(Vec1Out)")
            allocations.append(vec1_ptr)

            check_cuda(driver, driver.cuMemcpyHtoD(a_ptr, a_host, input_bytes)[0], "cuMemcpyHtoD(A)")
            check_cuda(driver, driver.cuMemcpyHtoD(b_ptr, b_host, input_bytes)[0], "cuMemcpyHtoD(B)")
            check_cuda(driver, driver.cuMemcpyHtoD(score_ptr, output_zero, output_bytes)[0], "cuMemcpyHtoD(ScoreOut)")
            check_cuda(driver, driver.cuMemcpyHtoD(vec0_ptr, output_zero, output_bytes)[0], "cuMemcpyHtoD(Vec0Out)")
            check_cuda(driver, driver.cuMemcpyHtoD(vec1_ptr, output_zero, output_bytes)[0], "cuMemcpyHtoD(Vec1Out)")

            compiled = df.compile(
                make_tensor_field_program(),
                topology=df.GPUTopology(sm_count=2, cluster_size=2),
                range_lengths={"kv": [64]},
                block_size=32,
                wrapper_name="dataflow_primfunc_tensor_field_cluster_executable",
            )
            assert any(comm.kind is df.DataflowCommKind.CLUSTER_SEND for comm in compiled.plan.comms)
            assert any(comm.kind is df.DataflowCommKind.CLUSTER_RECV for comm in compiled.plan.comms)
            assert not any(comm.kind is df.DataflowCommKind.HBM_SEND for comm in compiled.plan.comms)
            assert not any(comm.kind is df.DataflowCommKind.HBM_RECV for comm in compiled.plan.comms)

            execution = compiled(
                A=int(a_ptr),
                B=int(b_ptr),
                ScoreOut=int(score_ptr),
                Vec0Out=int(vec0_ptr),
                Vec1Out=int(vec1_ptr),
            )
            check_cuda(driver, driver.cuMemcpyDtoH(score_host, score_ptr, output_bytes)[0], "cuMemcpyDtoH(ScoreOut)")
            check_cuda(driver, driver.cuMemcpyDtoH(vec0_host, vec0_ptr, output_bytes)[0], "cuMemcpyDtoH(Vec0Out)")
            check_cuda(driver, driver.cuMemcpyDtoH(vec1_host, vec1_ptr, output_bytes)[0], "cuMemcpyDtoH(Vec1Out)")

            assert execution.cluster_dim == (2, 1, 1)
            assert list(score_host) == [sum(a_values), 0]
            assert list(vec0_host) == [sum(a + b for a, b in zip(a_values, b_values)), 0]
            assert list(vec1_host) == [sum(a * 2 + b for a, b in zip(a_values, b_values)), 0]
        finally:
            for ptr in reversed(allocations):
                driver.cuMemFree(ptr)
    finally:
        restore_result = driver.cuCtxSetCurrent(previous_context)[0]
        driver.cuDevicePrimaryCtxRelease(device)
        check_cuda(driver, restore_result, "cuCtxSetCurrent(previous)")


def test_linked_primfunc_handlers_execute_tensor_field_cluster_hbm_multitask_dataflow_on_cuda():
    require_cluster_launch_cuda()

    import ctypes
    from cuda.bindings import driver

    result, device = driver.cuDeviceGet(0)
    check_cuda(driver, result, "cuDeviceGet")
    result, context = driver.cuDevicePrimaryCtxRetain(device)
    check_cuda(driver, result, "cuDevicePrimaryCtxRetain")
    result, previous_context = driver.cuCtxGetCurrent()
    if result != driver.CUresult.CUDA_SUCCESS:
        driver.cuDevicePrimaryCtxRelease(device)
        check_cuda(driver, result, "cuCtxGetCurrent")

    allocations = []
    try:
        try:
            result = driver.cuCtxSetCurrent(context)[0]
            check_cuda(driver, result, "cuCtxSetCurrent")

            a_values = list(range(1, 129))
            b_values = list(range(3, 131))
            output_count = 2
            input_bytes = len(a_values) * ctypes.sizeof(ctypes.c_int32)
            output_bytes = output_count * ctypes.sizeof(ctypes.c_int32)
            a_host = (ctypes.c_int32 * len(a_values))(*a_values)
            b_host = (ctypes.c_int32 * len(b_values))(*b_values)
            output_zero = (ctypes.c_int32 * output_count)()
            score_host = (ctypes.c_int32 * output_count)()
            vec0_host = (ctypes.c_int32 * output_count)()
            vec1_host = (ctypes.c_int32 * output_count)()

            result, a_ptr = driver.cuMemAlloc(input_bytes)
            check_cuda(driver, result, "cuMemAlloc(A)")
            allocations.append(a_ptr)
            result, b_ptr = driver.cuMemAlloc(input_bytes)
            check_cuda(driver, result, "cuMemAlloc(B)")
            allocations.append(b_ptr)
            result, score_ptr = driver.cuMemAlloc(output_bytes)
            check_cuda(driver, result, "cuMemAlloc(ScoreOut)")
            allocations.append(score_ptr)
            result, vec0_ptr = driver.cuMemAlloc(output_bytes)
            check_cuda(driver, result, "cuMemAlloc(Vec0Out)")
            allocations.append(vec0_ptr)
            result, vec1_ptr = driver.cuMemAlloc(output_bytes)
            check_cuda(driver, result, "cuMemAlloc(Vec1Out)")
            allocations.append(vec1_ptr)

            check_cuda(driver, driver.cuMemcpyHtoD(a_ptr, a_host, input_bytes)[0], "cuMemcpyHtoD(A)")
            check_cuda(driver, driver.cuMemcpyHtoD(b_ptr, b_host, input_bytes)[0], "cuMemcpyHtoD(B)")
            check_cuda(driver, driver.cuMemcpyHtoD(score_ptr, output_zero, output_bytes)[0], "cuMemcpyHtoD(ScoreOut)")
            check_cuda(driver, driver.cuMemcpyHtoD(vec0_ptr, output_zero, output_bytes)[0], "cuMemcpyHtoD(Vec0Out)")
            check_cuda(driver, driver.cuMemcpyHtoD(vec1_ptr, output_zero, output_bytes)[0], "cuMemcpyHtoD(Vec1Out)")

            compiled = df.compile(
                make_tensor_field_program(),
                topology=df.GPUTopology(sm_count=4, cluster_size=2),
                range_lengths={"kv": [128, 96]},
                task_extents=(2,),
                block_size=32,
                wrapper_name="dataflow_primfunc_tensor_field_cluster_hbm_multi_executable",
            )
            cluster_sends = [comm for comm in compiled.plan.comms if comm.kind is df.DataflowCommKind.CLUSTER_SEND]
            cluster_recvs = [comm for comm in compiled.plan.comms if comm.kind is df.DataflowCommKind.CLUSTER_RECV]
            hbm_sends = [comm for comm in compiled.plan.comms if comm.kind is df.DataflowCommKind.HBM_SEND]
            hbm_recvs = [comm for comm in compiled.plan.comms if comm.kind is df.DataflowCommKind.HBM_RECV]
            assert len(cluster_sends) == 2
            assert len(cluster_recvs) == 2
            assert len(hbm_sends) == 3
            assert len(hbm_recvs) == 3

            execution = compiled(
                A=int(a_ptr),
                B=int(b_ptr),
                ScoreOut=int(score_ptr),
                Vec0Out=int(vec0_ptr),
                Vec1Out=int(vec1_ptr),
            )
            check_cuda(driver, driver.cuMemcpyDtoH(score_host, score_ptr, output_bytes)[0], "cuMemcpyDtoH(ScoreOut)")
            check_cuda(driver, driver.cuMemcpyDtoH(vec0_host, vec0_ptr, output_bytes)[0], "cuMemcpyDtoH(Vec0Out)")
            check_cuda(driver, driver.cuMemcpyDtoH(vec1_host, vec1_ptr, output_bytes)[0], "cuMemcpyDtoH(Vec1Out)")

            expected_score = [sum(a_values[:128]), sum(a_values[:96])]
            expected_vec0 = [
                sum(a + b for a, b in zip(a_values[:128], b_values[:128])),
                sum(a + b for a, b in zip(a_values[:96], b_values[:96])),
            ]
            expected_vec1 = [
                sum(a * 2 + b for a, b in zip(a_values[:128], b_values[:128])),
                sum(a * 2 + b for a, b in zip(a_values[:96], b_values[:96])),
            ]

            assert execution.cluster_dim == (2, 1, 1)
            assert list(score_host) == expected_score
            assert list(vec0_host) == expected_vec0
            assert list(vec1_host) == expected_vec1
        finally:
            for ptr in reversed(allocations):
                driver.cuMemFree(ptr)
    finally:
        restore_result = driver.cuCtxSetCurrent(previous_context)[0]
        driver.cuDevicePrimaryCtxRelease(device)
        check_cuda(driver, restore_result, "cuCtxSetCurrent(previous)")


def test_linked_primfunc_handlers_execute_hbm_dataflow_on_cuda():
    require_executable_cuda()

    compiled = df.compile(
        make_scalar_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=1),
        range_lengths={"kv": [64]},
        block_size=32,
        wrapper_name="dataflow_primfunc_hbm_executable",
    )

    assert any(comm.kind is df.DataflowCommKind.HBM_SEND for comm in compiled.plan.comms)
    assert any(comm.kind is df.DataflowCommKind.HBM_RECV for comm in compiled.plan.comms)


def test_linked_primfunc_handlers_keep_cluster_comm_in_wrapper_source():
    compiled = df.compile(
        make_scalar_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"kv": [64]},
        block_size=32,
        wrapper_name="dataflow_primfunc_cluster_source",
    )

    assert any(comm.kind is df.DataflowCommKind.CLUSTER_SEND for comm in compiled.plan.comms)
    assert any(comm.kind is df.DataflowCommKind.CLUSTER_RECV for comm in compiled.plan.comms)
    assert "tl::cluster_sync();" in compiled.wrapper_source
    assert "dataflow_primfunc_split_scalar_device" in compiled.wrapper_source


def test_linked_primfunc_tensor_field_handlers_keep_cluster_comm_in_wrapper_source():
    compiled = df.compile(
        make_tensor_field_program(),
        topology=df.GPUTopology(sm_count=2, cluster_size=2),
        range_lengths={"kv": [64]},
        block_size=32,
        wrapper_name="dataflow_primfunc_tensor_field_cluster_source",
    )

    assert any(comm.kind is df.DataflowCommKind.CLUSTER_SEND for comm in compiled.plan.comms)
    assert any(comm.kind is df.DataflowCommKind.CLUSTER_RECV for comm in compiled.plan.comms)
    assert not any(comm.kind is df.DataflowCommKind.HBM_SEND for comm in compiled.plan.comms)
    assert not any(comm.kind is df.DataflowCommKind.HBM_RECV for comm in compiled.plan.comms)
    assert "kDataflowClusterSize = 2u" in compiled.wrapper_source
    assert "tl::cluster_sync();" in compiled.wrapper_source
    assert "dataflow_primfunc_split_tensor_field_device_kernel" in compiled.wrapper_source
    assert "dataflow_primfunc_reduce_tensor_field_device_kernel" in compiled.wrapper_source
    assert "dataflow_primfunc_load_reduce_items" not in compiled.wrapper_source
    assert "item0_vec" in compiled.wrapper_source
    assert "item1_vec" in compiled.wrapper_source
