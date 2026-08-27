from __future__ import annotations

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tvm import tir


def logical_gemm(m: int, *, n: int = 64, k: int = 16):
    @T.prim_func
    def main(
        A: T.Tensor((m, k), T.float16),
        B: T.Tensor((k, n), T.float16),
        C: T.Tensor((m, n), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((m, k), T.float16)
            B_shared = T.alloc_shared((k, n), T.float16)
            C_local = T.alloc_fragment((m, n), T.float32)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                clear_accum=True,
                logical_shape=(m, n, k),
            )
            T.copy(C_local, C)

    return main


def logical_gemm_with_reusable_shared_upper_bound(m: int = 32):
    @T.prim_func
    def main(
        A: T.Tensor((m, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        C: T.Tensor((m, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            earlier_shared = T.alloc_shared((4096,), T.float16)
            A_shared = T.alloc_shared((m, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((m, 64), T.float32)
            T.fill(earlier_shared, 0)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                clear_accum=True,
                logical_shape=(m, 64, 16),
            )
            T.copy(C_local, C)

    return main


def logical_gemm_with_physical_storage(m: int = 32, *, n: int = 64, k: int = 16):
    @T.prim_func
    def main(
        A: T.Tensor((m, k), T.float16),
        B: T.Tensor((k, n), T.float16),
        C: T.Tensor((m, n), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, k), T.float16)
            B_shared = T.alloc_shared((k, n), T.float16)
            C_local = T.alloc_fragment((64, n), T.float32)
            T.copy(A, A_shared[0:m, 0:k])
            T.copy(B, B_shared)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                clear_accum=True,
                logical_shape=(m, n, k),
            )
            T.copy(C_local[0:m, 0:n], C)

    return main


def logical_gemm_without_copy_producer(m: int = 32):
    @T.prim_func
    def main(
        A: T.Tensor((m, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        C: T.Tensor((m, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((m, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((m, 64), T.float32)
            for i, j in T.Parallel(m, 16):
                A_shared[i, j] = A[i, j]
            T.copy(B, B_shared)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                clear_accum=True,
                logical_shape=(m, 64, 16),
            )
            T.copy(C_local, C)

    return main


def logical_gemm_accumulator_chain(m: int):
    @T.prim_func
    def main(
        A0: T.Tensor((m, 32), T.float16),
        B0: T.Tensor((32, 64), T.float16),
        A1: T.Tensor((m, 16), T.float16),
        B1: T.Tensor((16, 64), T.float16),
        C: T.Tensor((m, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A0_shared = T.alloc_shared((m, 32), T.float16)
            B0_shared = T.alloc_shared((32, 64), T.float16)
            A1_shared = T.alloc_shared((m, 16), T.float16)
            B1_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((m, 64), T.float32)
            T.copy(A0, A0_shared)
            T.copy(B0, B0_shared)
            T.copy(A1, A1_shared)
            T.copy(B1, B1_shared)
            T.gemm(
                A0_shared,
                B0_shared,
                C_local,
                clear_accum=True,
                logical_shape=(m, 64, 32),
            )
            T.gemm(
                A1_shared,
                B1_shared,
                C_local,
                logical_shape=(m, 64, 16),
            )
            T.copy(C_local, C)

    return main


def materialize(func, arch: str, *, pass_config: dict | None = None):
    target = tvm.target.Target(f"cuda -arch={arch}")
    mod = tvm.IRModule({"main": func})
    mod = tir.transform.BindTarget(target)(mod)
    with tvm.transform.PassContext(opt_level=3, config=pass_config or {}):
        return tilelang.transform.MaterializeLogicalGemm()(mod)["main"]


def calls(func: tir.PrimFunc, op_name: str) -> list[tir.Call]:
    result: list[tir.Call] = []

    def collect(node):
        if isinstance(node, tir.Call) and getattr(node.op, "name", None) == op_name:
            result.append(node)

    tir.stmt_functor.post_order_visit(func.body, collect)
    return result


def collect_allocations(func: tir.PrimFunc) -> dict[str, tir.Buffer]:
    result: dict[str, tir.Buffer] = {}

    def collect(node):
        if isinstance(node, tir.Block):
            for buffer in node.alloc_buffers:
                result[buffer.name] = buffer

    tir.stmt_functor.post_order_visit(func.body, collect)
    return result


def test_logical_gemm_materializer_is_noop_without_gemm_or_thread_extent():
    @T.prim_func
    def main(A: T.Tensor((1,), T.float32)):
        A[0] = A[0] + T.float32(1)

    target = tvm.target.Target("cuda -arch=sm_90")
    bound = tir.transform.BindTarget(target)(tvm.IRModule({"main": main}))["main"]
    with target:
        after = tilelang.transform.MaterializeLogicalGemm()(tvm.IRModule({"main": bound}))["main"]

    tvm.ir.assert_structural_equal(bound, after)


@pytest.mark.parametrize("logical_m", [8, 16, 32, 48])
def test_logical_gemm_streams_logical_m_shared_a_without_physical_padding(logical_m):
    func = materialize(logical_gemm(logical_m), "sm_90")
    plans = list(func.attrs["tl.gemm_lowering_plans"])

    assert len(plans) == 1
    plan = plans[0]
    assert plan.implementation_id == "cuda.wgmma.rs.shared_a"
    assert plan.synchronous is True
    assert [int(value) for value in plan.logical_shape] == [logical_m, 64, 16]
    assert [int(value) for value in plan.physical_shape] == [64, 64, 16]
    assert plan.requires_padding is True
    assert plan.requires_materialization is True
    assert plan.additional_shared_memory_bytes == 0

    gemm = calls(func, "tl.tileop.gemm")[0]
    assert int(gemm.args[5]) == 64
    allocations = collect_allocations(func)
    a_shared = next(buffer for name, buffer in allocations.items() if "A_shared" in name)
    c_local = next(buffer for name, buffer in allocations.items() if "C_local" in name)
    assert [int(value) for value in a_shared.shape] == [logical_m, 16]
    assert [int(value) for value in c_local.shape] == [64, 64]
    assert len(calls(func, "tl.tileop.fill")) == 0

    a_copy = calls(func, "tl.tileop.copy")[0]
    assert len(a_copy.args) == 2
    assert [int(value) for value in a_copy.args[0].args[2:]] == [logical_m, 16]
    assert [int(value) for value in a_copy.args[1].args[2:]] == [logical_m, 16]
    requirements = {item.buffer_role: item for item in plan.temporary_requirements}
    assert set(requirements) == {"C"}
    assert requirements["C"].lifetime_end == "after_last_logical_output_consumer"
    assert requirements["C"].initialization_required is False
    assert int(gemm.annotations["wgmma_rs_shared_a_logical_m"]) == logical_m


def test_logical_gemm_preserves_direct_m64_wgmma_path():
    func = materialize(logical_gemm(64), "sm_90")
    plan = list(func.attrs["tl.gemm_lowering_plans"])[0]

    assert plan.implementation_id == "cuda.wgmma.async"
    assert plan.synchronous is False
    assert plan.requires_padding is False
    assert plan.requires_materialization is False
    assert [int(value) for value in collect_allocations(func)["A_shared"].shape] == [64, 16]


def test_logical_gemm_shared_a_rs_needs_no_shared_padding_budget():
    func = materialize(
        logical_gemm(32),
        "sm_90",
        pass_config={"tl.logical_gemm_max_shared_memory_bytes": 4095},
    )
    plan = list(func.attrs["tl.gemm_lowering_plans"])[0]

    assert plan.implementation_id == "cuda.wgmma.rs.shared_a"
    assert plan.requires_padding is True
    assert plan.additional_shared_memory_bytes == 0
    assert int(calls(func, "tl.tileop.gemm")[0].args[5]) == 64
    assert len(calls(func, "tl.tileop.fill")) == 0


def test_logical_gemm_defers_inconclusive_raw_shared_upper_bound_to_async_ss():
    func = materialize(
        logical_gemm_with_reusable_shared_upper_bound(),
        "sm_90",
        pass_config={"tl.logical_gemm_max_shared_memory_bytes": 8192},
    )
    plan = list(func.attrs["tl.gemm_lowering_plans"])[0]

    assert plan.implementation_id == "cuda.wgmma.async"
    assert plan.requires_materialization is True
    assert plan.additional_shared_memory_bytes == 1024
    assert "final peak validation is deferred" in plan.selection_reason


def test_logical_gemm_does_not_initialize_unused_rows_in_preallocated_a_storage():
    func = materialize(logical_gemm_with_physical_storage(), "sm_90")
    plan = list(func.attrs["tl.gemm_lowering_plans"])[0]

    assert plan.requires_padding is True
    assert plan.requires_materialization is True
    assert plan.additional_shared_memory_bytes == 0
    assert plan.additional_fragment_bytes == 0
    a_copy = calls(func, "tl.tileop.copy")[0]
    assert len(a_copy.args) == 2
    assert [int(value) for value in a_copy.args[0].args[2:]] == [32, 16]
    assert [int(value) for value in a_copy.args[1].args[2:]] == [32, 16]


def test_logical_gemm_needs_no_a_padding_without_a_copy_producer():
    func = materialize(logical_gemm_without_copy_producer(), "sm_90")
    plan = list(func.attrs["tl.gemm_lowering_plans"])[0]
    requirements = {item.buffer_role: item for item in plan.temporary_requirements}

    assert set(requirements) == {"C"}
    assert requirements["C"].initialization_required is False
    assert len(calls(func, "tl.tileop.fill")) == 0


def test_logical_gemm_does_not_group_synchronous_shared_a_rs_chain():
    func = materialize(logical_gemm_accumulator_chain(32), "sm_90")
    gemms = calls(func, "tl.tileop.gemm")

    assert len(gemms) == 2
    assert len(calls(func, "tl.tileop.wgmma_gemm")) == 0
    assert int(func.attrs["tl.logical_gemm_grouped_wgmma_chain_count"]) == 0
    assert all(call.annotations["gemm_lowering_implementation"] == "cuda.wgmma.rs.shared_a" for call in gemms)
    assert len(calls(func, "tl.wait_wgmma")) == 0


def test_logical_gemm_groups_adjacent_direct_m64_wgmma_accumulator_chain():
    func = materialize(logical_gemm_accumulator_chain(64), "sm_90")
    gemms = calls(func, "tl.tileop.wgmma_gemm")

    assert len(gemms) == 2
    assert len(calls(func, "tl.tileop.gemm")) == 0
    assert int(func.attrs["tl.logical_gemm_grouped_wgmma_chain_count"]) == 1
    assert [int(call.args[15]) for call in gemms] == [-1, -1]
    assert len(calls(func, "tl.wait_wgmma")) == 1
    assert [int(call.annotations["wgmma_emit_arrive"]) for call in gemms] == [1, 0]
    assert [int(call.annotations["wgmma_emit_commit"]) for call in gemms] == [0, 1]
    assert [int(call.annotations["wgmma_emit_fence_before"]) for call in gemms] == [0, 0]
    assert [int(call.annotations["wgmma_emit_fence_after"]) for call in gemms] == [0, 0]
    assert len(calls(func, "tl.warpgroup_fence_operand")) == 2
    assert [int(call.annotations["wgmma_additive_group_index"]) for call in gemms] == [0, 1]
    assert [int(call.annotations["wgmma_additive_group_size"]) for call in gemms] == [2, 2]


def test_logical_gemm_does_not_group_mma_accumulator_chain():
    func = materialize(logical_gemm_accumulator_chain(32), "sm_80")
    gemms = calls(func, "tl.tileop.gemm")

    assert int(func.attrs["tl.logical_gemm_grouped_wgmma_chain_count"]) == 0
    assert [int(call.args[15]) for call in gemms] == [0, 0]
    assert len(calls(func, "tl.wait_wgmma")) == 0
    assert all("wgmma_additive_group_size" not in call.annotations for call in gemms)


def test_logical_gemm_does_not_treat_later_copy_as_a_producer():
    @T.prim_func
    def main(
        A: T.Tensor((32, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        C: T.Tensor((32, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((32, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((32, 64), T.float32)
            T.copy(B, B_shared)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                clear_accum=True,
                logical_shape=(32, 64, 16),
            )
            T.copy(A, A_shared)
            T.copy(C_local, C)

    func = materialize(main, "sm_90")
    a_copy = next(call for call in calls(func, "tl.tileop.copy") if "A_shared" in call.args[1].args[0].buffer.name)

    assert [int(value) for value in a_copy.args[0].args[2:]] == [32, 16]
    assert len(calls(func, "tl.tileop.fill")) == 0


def test_logical_gemm_does_not_use_conditional_copy_outside_its_branch():
    @T.prim_func
    def main(
        A: T.Tensor((32, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        C: T.Tensor((32, 64), T.float32),
        predicate: T.int32,
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((32, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((32, 64), T.float32)
            if predicate != 0:
                T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                clear_accum=True,
                logical_shape=(32, 64, 16),
            )
            T.copy(C_local, C)

    func = materialize(main, "sm_90")

    assert len(calls(func, "tl.tileop.fill")) == 0


def test_logical_gemm_full_copy_produces_sliced_k_operand_padding():
    @T.prim_func
    def main(
        A: T.Tensor((32, 32), T.float16),
        B: T.Tensor((16, 64), T.float16),
        C: T.Tensor((32, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((32, 32), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((32, 64), T.float32)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(
                A_shared[0:32, 0:16],
                B_shared,
                C_local,
                clear_accum=True,
                logical_shape=(32, 64, 16),
            )
            T.copy(C_local, C)

    func = materialize(main, "sm_90")
    a_copy = calls(func, "tl.tileop.copy")[0]

    assert [int(value) for value in a_copy.args[0].args[2:]] == [32, 32]
    assert [int(value) for value in a_copy.args[1].args[2:]] == [32, 32]
    assert len(calls(func, "tl.tileop.fill")) == 0


def test_logical_gemm_global_disable_is_a_generic_fallback():
    func = materialize(
        logical_gemm(32),
        "sm_90",
        pass_config={"tl.disable_logical_gemm_padding": True},
    )
    plan = list(func.attrs["tl.gemm_lowering_plans"])[0]

    assert plan.implementation_id == "cuda.mma.sync"
    assert plan.requires_materialization is False


def test_logical_gemm_sm80_preserves_unpadded_mma_path():
    func = materialize(logical_gemm(32), "sm_80")
    plan = list(func.attrs["tl.gemm_lowering_plans"])[0]

    assert plan.implementation_id == "cuda.mma.sync"
    assert [int(value) for value in plan.physical_shape] == [32, 64, 16]
    assert len(calls(func, "tl.tileop.fill")) == 0


def test_logical_gemm_sm80_shrinks_preallocated_physical_storage_to_logical_m():
    func = materialize(logical_gemm_with_physical_storage(), "sm_80")
    plan = list(func.attrs["tl.gemm_lowering_plans"])[0]

    assert plan.implementation_id == "cuda.mma.sync"
    assert plan.requires_padding is False
    assert plan.requires_materialization is True
    assert int(calls(func, "tl.tileop.gemm")[0].args[5]) == 32
    assert len(calls(func, "tl.tileop.fill")) == 0


def test_logical_gemm_scalar_fallback_preserves_valid_output_shape():
    func = materialize(
        logical_gemm(8),
        "sm_90",
        pass_config={"tl.disable_logical_gemm_padding": True},
    )
    plan = list(func.attrs["tl.gemm_lowering_plans"])[0]

    assert plan.implementation_id == "cuda.scalar.sync"
    assert [int(value) for value in plan.logical_shape] == [8, 64, 16]
    assert [int(value) for value in plan.physical_shape] == [8, 64, 16]


@pytest.mark.parametrize(
    ("logical_shape", "message"),
    [
        ((32, 32, 16), "logical N must equal"),
        ((32, 64, 8), "logical K must equal"),
    ],
)
def test_logical_gemm_unimplemented_n_or_k_padding_fails_closed(logical_shape, message):
    @T.prim_func
    def main(
        A: T.Tensor((32, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        C: T.Tensor((32, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((32, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((32, 64), T.float32)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local, logical_shape=logical_shape)
            T.copy(C_local, C)

    with pytest.raises(tvm.error.InternalError, match=message):
        materialize(main, "sm_90")


def test_gemm_contract_rejects_padding_value_buffer_reads_and_side_effects():
    a = tir.decl_buffer((32, 16), "float16", name="A", scope="shared")
    b = tir.decl_buffer((16, 64), "float16", name="B", scope="shared")
    c = tir.decl_buffer((32, 64), "float32", name="C", scope="local.fragment")

    with pytest.raises(ValueError, match="cannot read a buffer"):
        T.gemm_contract((32, 64, 16), padding_value=a[0, 0])

    base_call = T.gemm(a, b, c, logical_shape=(32, 64, 16))
    unsafe_contract = tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.gemm_contract"),
        32,
        64,
        16,
        a[0, 0],
        1,
    )
    unsafe_call = tir.Call(
        base_call.dtype,
        base_call.op,
        [*base_call.args[:-1], unsafe_contract],
        annotations=base_call.annotations,
    )
    with pytest.raises(tvm.error.InternalError, match="cannot read a buffer"):
        tilelang._ffi_api.ParseOperator(unsafe_call)

    opaque_contract = T.gemm_contract(
        (32, 64, 16),
        padding_value=tir.call_extern("float16", "opaque_padding"),
    )
    opaque_call = tir.Call(
        base_call.dtype,
        base_call.op,
        [*base_call.args[:-1], opaque_contract],
        annotations=base_call.annotations,
    )
    with pytest.raises(tvm.error.InternalError, match="pure scalar expression"):
        tilelang._ffi_api.ParseOperator(opaque_call)
