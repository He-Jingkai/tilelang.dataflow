"""Tests for TileLang `LowerTileOp` copy annotations affecting cp.async sync."""

import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm.tir.stmt_functor import post_order_visit


def _count_calls(func: tvm.tir.PrimFunc):
    counts = {}

    def _visit(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op):
            name = str(node.op.name)
            counts[name] = counts.get(name, 0) + 1

    post_order_visit(func.body, _visit)
    return counts


def test_lower_tile_op_respects_copy_annotation_for_pipeline_managed_cp_async():
    target = tvm.target.Target("cuda -arch=sm_80")

    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float32),
        B: T.Tensor((16,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 16)
        S = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        T.copy(
            A[0:16],
            S,
            annotations={"no_implicit_async_commit_wait": T.int32(1)},
        )
        B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tl.transform.LowerTileOp()(mod)
    calls = _count_calls(mod["main"])

    assert calls.get("tl.ptx_cp_async", 0) > 0
    assert calls.get("tir.ptx_commit_group", 0) == 0
    assert calls.get("tir.ptx_wait_group", 0) == 0


def test_lower_tile_op_respects_copy_annotation_for_explicit_async_copy():
    target = tvm.target.Target("cuda -arch=sm_80")

    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float32),
        B: T.Tensor((16,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 16)
        S = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        T.async_copy(
            A[0:16],
            S,
            annotations={"no_implicit_async_commit_wait": T.int32(1)},
        )
        B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tl.transform.LowerTileOp()(mod)
    calls = _count_calls(mod["main"])

    assert calls.get("tl.ptx_cp_async", 0) > 0
    assert calls.get("tir.ptx_commit_group", 0) == 0
    assert calls.get("tir.ptx_wait_group", 0) == 0


def test_lower_tile_op_respects_parallel_loop_async_annotation_without_pipeline_context():
    target = tvm.target.Target("cuda -arch=sm_80")

    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float32),
        B: T.Tensor((16,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 16)
        S = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        for i in T.parallel(
            16,
            annotations={"parallel_async_without_async_commit_wait": T.bool(True)},
        ):
            S[i] = A[i]
        B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)
    calls = _count_calls(mod["main"])

    assert calls.get("tl.ptx_cp_async", 0) > 0
    assert calls.get("tir.ptx_commit_group", 0) == 0
    assert calls.get("tir.ptx_wait_group", 0) == 0


def test_lower_tile_op_remaps_shared_parameter_buffer_map_for_wgmma():
    target = tvm.target.Target("cuda -arch=sm_90a")

    @T.prim_func
    def before(A_handle: T.handle):
        T.func_attr({"global_symbol": "main", "target": target})
        A = T.match_buffer(
            A_handle,
            (64, 128),
            dtype="float8_e4m3fn",
            scope="shared",
        )
        T.launch_thread("blockIdx.x", 1)
        T.launch_thread("threadIdx.x", 128)
        B = T.alloc_buffer((128, 128), dtype="float8_e4m3fn", scope="shared")
        C = T.alloc_buffer((64, 128), dtype="float32", scope="local.fragment")
        T.wgmma_gemm(A, B, C, transpose_B=True)

    original_buffer = next(iter(before.buffer_map.values()))
    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tl.transform.LayoutInference()(mod)
        mod = tl.transform.LowerTileOp()(mod)

    lowered = mod["main"]
    lowered_buffer = next(iter(lowered.buffer_map.values()))
    access_ptr_vars = []

    def collect_access_ptr_vars(node):
        if (
            isinstance(node, tvm.tir.Call)
            and isinstance(node.op, tvm.ir.Op)
            and node.op.name == "tir.tvm_access_ptr"
            and isinstance(node.args[1], tvm.tir.Var)
            and node.args[1].name == "A"
        ):
            access_ptr_vars.append(node.args[1])

    post_order_visit(lowered.body, collect_access_ptr_vars)

    assert not lowered_buffer.data.same_as(original_buffer.data)
    assert access_ptr_vars
    assert all(var.same_as(lowered_buffer.data) for var in access_ptr_vars)


if __name__ == "__main__":
    tilelang.testing.main()
