import pytest

import tilelang as tl
import tilelang.language as T
from tilelang import tvm


def materialize_logical_row_closure(arch: str):
    logical_rows = 32
    columns = 64
    reduction_columns = 16

    @T.prim_func
    def kernel(
        lhs: T.Tensor((logical_rows, reduction_columns), T.float16),
        rhs: T.Tensor((reduction_columns, columns), T.float16),
        values_input: T.Tensor((columns, columns), T.float16),
        output: T.Tensor((logical_rows, columns), T.float32),
    ):
        with T.Kernel(1, threads=128):
            lhs_shared = T.alloc_shared(
                (logical_rows, reduction_columns),
                T.float16,
            )
            rhs_shared = T.alloc_shared(
                (reduction_columns, columns),
                T.float16,
            )
            values = T.alloc_shared((columns, columns), T.float16)
            scores = T.alloc_fragment((logical_rows, columns), T.float32)
            probabilities = T.alloc_shared(
                (logical_rows, columns),
                T.float16,
            )
            accumulator = T.alloc_fragment(
                (logical_rows, columns),
                T.float32,
            )
            running_max = T.alloc_fragment((logical_rows,), T.float32)
            previous_max = T.alloc_fragment((logical_rows,), T.float32)
            history_scale = T.alloc_fragment((logical_rows,), T.float32)
            tile_sum = T.alloc_fragment((logical_rows,), T.float32)
            running_sum = T.alloc_shared((logical_rows,), T.float32)

            # Initialization intentionally precedes the GEMM that establishes
            # the source buffer's physical row extent.
            T.online_softmax_initialize(
                scores,
                running_max,
                running_sum,
                logical_rows,
            )
            T.copy(lhs, lhs_shared)
            T.copy(rhs, rhs_shared)
            T.copy(values_input, values)
            T.fill(accumulator, 0)
            T.gemm(
                lhs_shared,
                rhs_shared,
                scores,
                clear_accum=True,
                logical_shape=(logical_rows, columns, reduction_columns),
            )
            T.online_softmax_update(
                scores,
                values,
                probabilities,
                accumulator,
                running_max,
                previous_max,
                history_scale,
                tile_sum,
                running_sum,
                history_scale,
                logical_rows,
                logical_rows,
                columns,
                columns,
                False,
                T.GemmWarpPolicy.FullCol,
            )
            T.copy(accumulator[0:logical_rows, 0:columns], output)

    target = tvm.target.Target(f"cuda -arch={arch}")
    mod = tvm.IRModule({"main": kernel})
    mod = tvm.tir.transform.BindTarget(target)(mod)
    with target:
        return tl.transform.MaterializeLogicalGemm()(mod)["main"]


def collect_allocations(func):
    result = {}

    def collect(node):
        if isinstance(node, tvm.tir.Block):
            for buffer in node.alloc_buffers:
                result[buffer.name] = buffer

    tvm.tir.stmt_functor.post_order_visit(func.body, collect)
    return result


@pytest.mark.parametrize(
    ("state_rows", "output_rows", "columns", "value_columns"),
    [(2, 2, 16, 8), (4, 2, 32, 16)],
)
def test_online_softmax_update_is_an_inline_target_independent_composition(
    state_rows,
    output_rows,
    columns,
    value_columns,
):
    @T.prim_func
    def kernel(
        score_input: T.Tensor((state_rows, columns), T.float32),
        value_input: T.Tensor((columns, value_columns), T.float16),
        output: T.Tensor((output_rows, value_columns), T.float32),
    ):
        with T.Kernel(1, threads=32):
            scores = T.alloc_fragment((state_rows, columns), T.float32)
            values = T.alloc_shared((columns, value_columns), T.float16)
            probabilities = T.alloc_shared((output_rows, columns), T.float16)
            accumulator = T.alloc_fragment(
                (output_rows, value_columns),
                T.float32,
            )
            running_max = T.alloc_fragment((state_rows,), T.float32)
            previous_max = T.alloc_fragment((state_rows,), T.float32)
            history_scale = T.alloc_fragment((state_rows,), T.float32)
            tile_sum = T.alloc_fragment((state_rows,), T.float32)
            running_sum = T.alloc_fragment((state_rows,), T.float32)

            T.copy(score_input, scores)
            T.copy(value_input, values)
            T.fill(accumulator, 0)
            T.online_softmax_initialize(
                scores,
                running_max,
                running_sum,
                state_rows,
            )
            T.online_softmax_update(
                scores,
                values,
                probabilities,
                accumulator,
                running_max,
                previous_max,
                history_scale,
                tile_sum,
                running_sum,
                history_scale,
                state_rows,
                output_rows,
                columns,
                value_columns,
                False,
                T.GemmWarpPolicy.FullCol,
            )
            T.copy(accumulator, output)

    call_names = []
    tvm.tir.stmt_functor.post_order_visit(
        kernel.body,
        lambda node: call_names.append(node.op.name) if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op) else None,
    )
    assert call_names.count("tl.tileop.reduce") == 2
    assert call_names.count("tl.tileop.gemm") == 1

    target = tvm.target.Target("cuda -arch=sm_80")
    mod = tvm.IRModule.from_expr(kernel.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(target)(mod)
    with target:
        mod = tl.transform.PipelinePlanning()(mod)
    assert isinstance(mod["main"], tvm.tir.PrimFunc)


@pytest.mark.parametrize(
    ("arch", "expected_rows"),
    [("sm_90", 64), ("sm_80", 32)],
)
def test_online_softmax_logical_row_closure_tracks_gemm_physical_shape(
    arch,
    expected_rows,
):
    func = materialize_logical_row_closure(arch)
    allocations = collect_allocations(func)

    for name in (
        "scores",
        "running_max",
        "previous_max",
        "history_scale",
        "tile_sum",
        "running_sum",
    ):
        buffer = next(buffer for key, buffer in allocations.items() if name in key)
        assert int(buffer.shape[0]) == expected_rows

    row_blocks = []

    def collect_row_blocks(node):
        if isinstance(node, tvm.tir.Block) and node.name_hint.startswith("logical_row_"):
            row_blocks.append(node)

    tvm.tir.stmt_functor.post_order_visit(func.body, collect_row_blocks)
    assert {block.name_hint for block in row_blocks} == {
        "logical_row_initialization",
        "logical_row_recurrence",
    }

    recurrence = next(block for block in row_blocks if block.name_hint == "logical_row_recurrence")
    row_loop_extents = []
    region_extents = []

    def collect_closure_extents(node):
        if isinstance(node, tvm.tir.For) and int(node.extent) in (32, 64):
            row_loop_extents.append(int(node.extent))
        if isinstance(node, tvm.tir.Call) and getattr(node.op, "name", None) == "tl.tileop.region":
            region_extents.extend(int(value) for value in node.args[2:])

    tvm.tir.stmt_functor.post_order_visit(
        recurrence.body,
        collect_closure_extents,
    )
    assert expected_rows in row_loop_extents
    assert expected_rows in region_extents
    assert all(
        int(region.region[0].extent) == expected_rows
        for region in [*recurrence.reads, *recurrence.writes]
        if region.buffer.name
        in {
            "scores",
            "running_max",
            "previous_max",
            "history_scale",
            "tile_sum",
            "running_sum",
        }
    )

    reductions = []
    tvm.tir.stmt_functor.post_order_visit(
        recurrence.body,
        lambda node: (
            reductions.append(node) if isinstance(node, tvm.tir.Call) and getattr(node.op, "name", None) == "tl.tileop.reduce" else None
        ),
    )
    assert len(reductions) == 2
    for reduction in reductions:
        for region_load in reduction.args[:2]:
            row_index = region_load.indices[0]
            assert isinstance(row_index, tvm.tir.Ramp)
            assert int(row_index.lanes) == expected_rows
