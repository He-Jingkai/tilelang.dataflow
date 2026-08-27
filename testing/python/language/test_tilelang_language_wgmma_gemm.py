import pytest
import math
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _make_wgmma_kernel(gemm_op):
    @T.prim_func
    def main(
        A: T.Tensor((64, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        D: T.Tensor((64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float16)

            T.copy(A[0:64, 0:16], A_shared)
            T.copy(B[0:16, 0:64], B_shared)
            gemm_op(A_shared, B_shared, C_local)
            T.wait_wgmma(0)
            T.copy(C_local, D[0:64, 0:64])

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
@pytest.mark.parametrize(
    "gemm_api",
    [T.wgmma_gemm],
)
def test_wgmma_gemm_has_no_implicit_wait(gemm_api):
    kernel = tilelang.compile(_make_wgmma_kernel(lambda A, B, C: gemm_api(A, B, C, clear_accum=True)), target="cuda")
    src = kernel.get_kernel_source()

    assert src.count("tl::wait_wgmma<0>();") == 1


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_gemm_dispatch_has_no_implicit_wait():
    kernel = tilelang.compile(
        _make_wgmma_kernel(lambda A, B, C: T.wgmma_gemm(A, B, C, clear_accum=True)),
        target="cuda",
    )

    assert kernel.get_kernel_source().count("tl::wait_wgmma<0>();") == 1


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_ptx_stmatrix_frontend_lowers_to_cuda_helper():
    @T.prim_func
    def main(D: T.Tensor((16,), T.float16)):
        with T.Kernel(1, threads=32):
            S = T.alloc_shared((16,), T.float16)
            T.ptx_stmatrix(
                False,
                1,
                T.access_ptr(S[0], "w", extent=16),
                T.int32(0),
            )
            T.copy(S, D)

    kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    assert "tl::ptx_stmatrix_x1" in kernel.get_kernel_source()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_gemm_rejects_mma_fallback():
    @T.prim_func
    def main(
        A: T.Tensor((32, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        D: T.Tensor((32, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((32, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((32, 64), T.float16)

            T.copy(A[0:32, 0:16], A_shared)
            T.copy(B[0:16, 0:64], B_shared)
            T.wgmma_gemm(A_shared, B_shared, C_local, clear_accum=True)
            T.wait_wgmma(0)
            T.copy(C_local, D[0:32, 0:64])

    with pytest.raises(
        tvm.error.InternalError,
        match=r"T\.wgmma_gemm\(\) requires Hopper WGMMA lowering",
    ):
        tilelang.compile(main, target="cuda")


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_padded_wgmma_gemm_m32_uses_physical_m64_and_matches_torch():
    @T.prim_func
    def main(
        A: T.Tensor((32, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        D: T.Tensor((32, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float32)

            T.clear(A_shared)
            T.copy(A[0:32, 0:16], A_shared[0:32, 0:16])
            T.copy(B[0:16, 0:64], B_shared)
            T.padded_wgmma_gemm(A_shared, B_shared, C_local, logical_m=32, clear_accum=True)
            T.wait_wgmma(0)
            for i, j in T.Parallel(32, 64):
                D[i, j] = C_local[i, j]

    kernel = tilelang.compile(main, target="cuda")
    source = kernel.get_kernel_source()
    assert "wgmma" in source or "warpgroup" in source

    torch.manual_seed(0)
    a = torch.randn((32, 16), device="cuda", dtype=torch.float16)
    b = torch.randn((16, 64), device="cuda", dtype=torch.float16)
    out = torch.empty((32, 64), device="cuda", dtype=torch.float32)
    kernel(a, b, out)
    torch.testing.assert_close(out, (a @ b).to(torch.float32), atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_logical_wgmma_gemm_m32_materializes_padding_and_matches_torch():
    @T.prim_func
    def main(
        A: T.Tensor((32, 512), T.float16),
        B: T.Tensor((64, 512), T.float16),
        D: T.Tensor((32, 64), T.float32),
    ):
        with T.Kernel(1, threads=256):
            A_shared = T.alloc_shared((32, 512), T.float16)
            B_shared = T.alloc_shared((64, 512), T.float16)
            C_local = T.alloc_fragment((32, 64), T.float32)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.wgmma_gemm(
                A_shared,
                B_shared,
                C_local,
                logical_m=32,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=True,
            )
            T.wait_wgmma(0)
            T.copy(C_local, D)

    kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    source = kernel.get_kernel_source()
    assert "wgmma" in source or "warpgroup" in source

    torch.manual_seed(2026)
    a = torch.randn((32, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((64, 512), device="cuda", dtype=torch.float16)
    out = torch.empty((32, 64), device="cuda", dtype=torch.float32)
    kernel(a, b, out)
    torch.testing.assert_close(out, a.float() @ b.float().T, atol=3e-2, rtol=3e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
@pytest.mark.parametrize("logical_m", [2, 8, 16, 32, 48])
def test_logical_gemm_streams_shared_a_through_wgmma_rs(logical_m):
    reduction = 64

    @T.prim_func
    def main(
        A: T.Tensor((logical_m, reduction), T.float16),
        B: T.Tensor((64, reduction), T.float16),
        D: T.Tensor((logical_m, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((logical_m, reduction), T.float16)
            B_shared = T.alloc_shared((64, reduction), T.float16)
            C_local = T.alloc_fragment((logical_m, 64), T.float32)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                transpose_B=True,
                clear_accum=True,
                logical_shape=(logical_m, 64, reduction),
            )
            T.copy(C_local, D)

    tilelang.disable_cache()
    try:
        kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    finally:
        tilelang.enable_cache()
    source = kernel.get_kernel_source()
    assert "wgmma_rs" in source
    if logical_m >= 16:
        assert "ptx_ldmatrix" in source
    else:
        assert "A_ring" in source and "half_t(0x0p+0f" in source
    assert "A_ring[32]" in source
    assert "% 2" in source
    assert "warpgroup_wait<0>" in source
    assert "warpgroup_wait<2>" not in source

    torch.manual_seed(2030 + logical_m)
    a = torch.randn((logical_m, reduction), device="cuda", dtype=torch.float16)
    b = torch.randn((64, reduction), device="cuda", dtype=torch.float16)
    out = torch.empty((logical_m, 64), device="cuda", dtype=torch.float32)
    kernel(a, b, out)
    torch.testing.assert_close(out, a.float() @ b.float().T, atol=3e-2, rtol=3e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_logical_gemm_shared_a_rs_accumulates_q_and_qpe():
    logical_m = 32

    @T.prim_func
    def main(
        A: T.Tensor((logical_m, 64), T.float16),
        B: T.Tensor((64, 64), T.float16),
        APe: T.Tensor((logical_m, 32), T.float16),
        BPe: T.Tensor((64, 32), T.float16),
        D: T.Tensor((logical_m, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((logical_m, 64), T.float16)
            B_shared = T.alloc_shared((64, 64), T.float16)
            A_pe_shared = T.alloc_shared((logical_m, 32), T.float16)
            B_pe_shared = T.alloc_shared((64, 32), T.float16)
            C_local = T.alloc_fragment((logical_m, 64), T.float32)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.copy(APe, A_pe_shared)
            T.copy(BPe, B_pe_shared)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                transpose_B=True,
                clear_accum=True,
                logical_shape=(logical_m, 64, 64),
            )
            T.gemm(
                A_pe_shared,
                B_pe_shared,
                C_local,
                transpose_B=True,
                logical_shape=(logical_m, 64, 32),
            )
            T.copy(C_local, D)

    kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    source = kernel.get_kernel_source()
    assert source.count("wgmma_rs") == 2

    torch.manual_seed(2071)
    a = torch.randn((logical_m, 64), device="cuda", dtype=torch.float16)
    b = torch.randn((64, 64), device="cuda", dtype=torch.float16)
    a_pe = torch.randn((logical_m, 32), device="cuda", dtype=torch.float16)
    b_pe = torch.randn((64, 32), device="cuda", dtype=torch.float16)
    out = torch.empty((logical_m, 64), device="cuda", dtype=torch.float32)
    kernel(a, b, a_pe, b_pe, out)
    expected = a.float() @ b.float().T + a_pe.float() @ b_pe.float().T
    torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_logical_wgmma_gemm_m32_accumulates_async_pair():
    @T.prim_func
    def main(
        A: T.Tensor((32, 512), T.float16),
        B: T.Tensor((64, 512), T.float16),
        APe: T.Tensor((32, 64), T.float16),
        BPe: T.Tensor((64, 64), T.float16),
        D: T.Tensor((32, 64), T.float32),
    ):
        with T.Kernel(1, threads=256):
            A_shared = T.alloc_shared((32, 512), T.float16)
            B_shared = T.alloc_shared((64, 512), T.float16)
            A_pe_shared = T.alloc_shared((32, 64), T.float16)
            B_pe_shared = T.alloc_shared((64, 64), T.float16)
            C_local = T.alloc_fragment((32, 64), T.float32)

            T.copy(A, A_shared)
            T.copy(APe, A_pe_shared)
            T.copy(B, B_shared)
            T.copy(BPe, B_pe_shared)
            T.wgmma_gemm(
                A_shared,
                B_shared,
                C_local,
                logical_m=32,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=True,
                emit_commit=False,
                emit_fence_after=False,
            )
            T.wgmma_gemm(
                A_pe_shared,
                B_pe_shared,
                C_local,
                logical_m=32,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
                emit_arrive=False,
                emit_fence_before=False,
            )
            T.wait_wgmma(0)
            T.copy(C_local, D)

    kernel = tilelang.compile(main, target="cuda -arch=sm_90a")

    torch.manual_seed(2027)
    a = torch.randn((32, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((64, 512), device="cuda", dtype=torch.float16)
    a_pe = torch.randn((32, 64), device="cuda", dtype=torch.float16)
    b_pe = torch.randn((64, 64), device="cuda", dtype=torch.float16)
    out = torch.empty((32, 64), device="cuda", dtype=torch.float32)
    kernel(a, b, a_pe, b_pe, out)
    expected = a.float() @ b.float().T + a_pe.float() @ b_pe.float().T
    torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_padded_wgmma_gemm_m32_transpose_b_matches_torch():
    @T.prim_func
    def main(
        A: T.Tensor((32, 512), T.float16),
        B: T.Tensor((64, 512), T.float16),
        D: T.Tensor((32, 64), T.float32),
    ):
        with T.Kernel(1, threads=256):
            A_shared = T.alloc_shared((64, 512), T.float16)
            B_shared = T.alloc_shared((64, 512), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float32)

            T.copy(A[0:32, 0:512], A_shared[0:32, 0:512])
            for i, j in T.Parallel(32, 512):
                A_shared[32 + i, j] = T.float16(0)
            T.copy(B[0:64, 0:512], B_shared)
            T.padded_wgmma_gemm(
                A_shared,
                B_shared,
                C_local,
                logical_m=32,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=True,
            )
            T.wait_wgmma(0)
            for i, j in T.Parallel(32, 64):
                D[i, j] = C_local[i, j]

    tilelang.disable_cache()
    try:
        kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    finally:
        tilelang.enable_cache()
    source = kernel.get_kernel_source()
    assert "wgmma" in source or "warpgroup" in source

    torch.manual_seed(1)
    a = torch.randn((32, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((64, 512), device="cuda", dtype=torch.float16)
    out = torch.empty((32, 64), device="cuda", dtype=torch.float32)
    kernel(a, b, out)
    torch.testing.assert_close(out, a.float() @ b.float().T, atol=3e-2, rtol=3e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_padded_wgmma_gemm_m32_accumulates_transpose_b_tiles():
    @T.prim_func
    def main(
        A: T.Tensor((32, 512), T.float16),
        B: T.Tensor((64, 512), T.float16),
        APe: T.Tensor((32, 64), T.float16),
        BPe: T.Tensor((64, 64), T.float16),
        D: T.Tensor((32, 64), T.float32),
    ):
        with T.Kernel(1, threads=256):
            A_shared = T.alloc_shared((64, 512), T.float16)
            B_shared = T.alloc_shared((64, 512), T.float16)
            A_pe_shared = T.alloc_shared((64, 64), T.float16)
            B_pe_shared = T.alloc_shared((64, 64), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float32)

            T.copy(A[0:32, 0:512], A_shared[0:32, 0:512])
            T.copy(APe[0:32, 0:64], A_pe_shared[0:32, 0:64])
            for i, j in T.Parallel(32, 512):
                A_shared[32 + i, j] = T.float16(0)
            for i, j in T.Parallel(32, 64):
                A_pe_shared[32 + i, j] = T.float16(0)
            T.copy(B[0:64, 0:512], B_shared)
            T.copy(BPe[0:64, 0:64], B_pe_shared)
            T.padded_wgmma_gemm(
                A_shared,
                B_shared,
                C_local,
                logical_m=32,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=True,
            )
            T.padded_wgmma_gemm(
                A_pe_shared,
                B_pe_shared,
                C_local,
                logical_m=32,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
            )
            T.wait_wgmma(0)
            for i, j in T.Parallel(32, 64):
                D[i, j] = C_local[i, j]

    kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    source = kernel.get_kernel_source()
    assert "wgmma" in source or "warpgroup" in source

    torch.manual_seed(2)
    a = torch.randn((32, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((64, 512), device="cuda", dtype=torch.float16)
    a_pe = torch.randn((32, 64), device="cuda", dtype=torch.float16)
    b_pe = torch.randn((64, 64), device="cuda", dtype=torch.float16)
    out = torch.empty((32, 64), device="cuda", dtype=torch.float32)
    kernel(a, b, a_pe, b_pe, out)
    expected = a.float() @ b.float().T + a_pe.float() @ b_pe.float().T
    torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_padded_wgmma_gemm_m32_reduce_max_matches_torch():
    @T.prim_func
    def main(
        A: T.Tensor((32, 512), T.float16),
        B: T.Tensor((64, 512), T.float16),
        D: T.Tensor((32,), T.float32),
    ):
        with T.Kernel(1, threads=256):
            A_shared = T.alloc_shared((64, 512), T.float16)
            B_shared = T.alloc_shared((64, 512), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float32)
            row_max = T.alloc_fragment((64,), T.float32)

            T.copy(A[0:32, 0:512], A_shared[0:32, 0:512])
            for i, j in T.Parallel(32, 512):
                A_shared[32 + i, j] = T.float16(0)
            T.copy(B[0:64, 0:512], B_shared)
            T.padded_wgmma_gemm(
                A_shared,
                B_shared,
                C_local,
                logical_m=32,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=True,
            )
            T.wait_wgmma(0)
            T.fill(row_max, -T.infinity("float32"))
            T.reduce_max(C_local, row_max, dim=1, clear=False)
            for i in T.Parallel(32):
                D[i] = row_max[i]

    kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    source = kernel.get_kernel_source()
    assert "wgmma" in source or "warpgroup" in source

    torch.manual_seed(3)
    a = torch.randn((32, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((64, 512), device="cuda", dtype=torch.float16)
    out = torch.empty((32,), device="cuda", dtype=torch.float32)
    kernel(a, b, out)
    expected = (a.float() @ b.float().T).max(dim=1).values
    torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_padded_wgmma_mla_single_tile_softmax_pv_matches_torch():
    scale = 1.4426950408889634 / math.sqrt(512 + 64)

    @T.prim_func
    def main(
        Q: T.Tensor((32, 512), T.float16),
        QPe: T.Tensor((32, 64), T.float16),
        KV: T.Tensor((64, 512), T.float16),
        KPe: T.Tensor((64, 64), T.float16),
        D: T.Tensor((32, 512), T.float32),
    ):
        with T.Kernel(1, threads=256):
            Q_shared = T.alloc_shared((64, 512), T.float16)
            Q_pe_shared = T.alloc_shared((64, 64), T.float16)
            KV_shared = T.alloc_shared((64, 512), T.float16)
            K_pe_shared = T.alloc_shared((64, 64), T.float16)
            S_shared = T.alloc_shared((32, 64), T.float16)
            row_sum_shared = T.alloc_shared((32,), T.float32)
            O_shared = T.alloc_shared((32, 512), T.float16)
            acc_s = T.alloc_fragment((64, 64), T.float32)
            acc_o = T.alloc_fragment((32, 512), T.float32)
            row_max = T.alloc_fragment((64,), T.float32)
            row_sum = T.alloc_fragment((64,), T.float32)

            T.copy(Q[0:32, 0:512], Q_shared[0:32, 0:512])
            T.copy(QPe[0:32, 0:64], Q_pe_shared[0:32, 0:64])
            for i, j in T.Parallel(32, 512):
                Q_shared[32 + i, j] = T.float16(0)
            for i, j in T.Parallel(32, 64):
                Q_pe_shared[32 + i, j] = T.float16(0)
            T.copy(KV[0:64, 0:512], KV_shared)
            T.copy(KPe[0:64, 0:64], K_pe_shared)
            T.fill(acc_o, 0)

            T.padded_wgmma_gemm(
                Q_shared,
                KV_shared,
                acc_s,
                logical_m=32,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=True,
            )
            T.padded_wgmma_gemm(
                Q_pe_shared,
                K_pe_shared,
                acc_s,
                logical_m=32,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
            )
            T.wait_wgmma(0)
            for i, j in T.Parallel(64, 64):
                acc_s[i, j] *= T.float32(scale)
            T.fill(row_max, -T.infinity("float32"))
            T.reduce_max(acc_s, row_max, dim=1, clear=False)
            for i, j in T.Parallel(64, 64):
                acc_s[i, j] = T.exp2(acc_s[i, j] - row_max[i])
            T.fill(row_sum, 0)
            T.reduce_sum(acc_s, row_sum, dim=1)
            for i in T.Parallel(32):
                row_sum_shared[i] = row_sum[i]
            T.sync_threads()
            for i, j in T.Parallel(32, 64):
                S_shared[i, j] = acc_s[i, j]
            T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Parallel(32, 512):
                O_shared[i, j] = acc_o[i, j]
            T.sync_threads()
            for i, j in T.Parallel(32, 512):
                D[i, j] = T.cast(O_shared[i, j], "float32") / row_sum_shared[i]

    kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    source = kernel.get_kernel_source()
    assert "wgmma" in source or "warpgroup" in source

    torch.manual_seed(4)
    q = torch.randn((32, 512), device="cuda", dtype=torch.float16)
    q_pe = torch.randn((32, 64), device="cuda", dtype=torch.float16)
    kv = torch.randn((64, 512), device="cuda", dtype=torch.float16)
    k_pe = torch.randn((64, 64), device="cuda", dtype=torch.float16)
    out = torch.empty((32, 512), device="cuda", dtype=torch.float32)
    kernel(q, q_pe, kv, k_pe, out)
    scores = (q.float() @ kv.float().T + q_pe.float() @ k_pe.float().T) / math.sqrt(512 + 64)
    weights = torch.softmax(scores, dim=1)
    expected = weights @ kv.float()
    torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_gemm_local_p_single_warpgroup_consumes_qk_accumulator_layout():
    @T.prim_func
    def main(
        Q: T.Tensor((64, 64), T.float16),
        K: T.Tensor((64, 64), T.float16),
        V: T.Tensor((64, 64), T.float16),
        D: T.Tensor((64, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            Q_shared = T.alloc_shared((64, 64), T.float16)
            K_shared = T.alloc_shared((64, 64), T.float16)
            V_shared = T.alloc_shared((64, 64), T.float16)
            S_frag = T.alloc_fragment((64, 64), T.float32)
            P_frag = T.alloc_fragment((64, 64), T.float16)
            O_frag = T.alloc_fragment((64, 64), T.float32)

            T.copy(Q, Q_shared)
            T.copy(K, K_shared)
            T.copy(V, V_shared)
            T.wgmma_gemm(
                Q_shared,
                K_shared,
                S_frag,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=True,
            )
            T.wait_wgmma(0)
            for i, j in T.Parallel(64, 64):
                P_frag[i, j] = T.cast(S_frag[i, j], "float16")
            T.wgmma_gemm_local_p(
                P_frag,
                V_shared,
                O_frag,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=True,
            )
            T.wait_wgmma(0)
            T.copy(O_frag, D)

    kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    source = kernel.get_kernel_source()
    assert "wgmma_ss" in source
    assert "wgmma_rs" in source

    torch.manual_seed(8)
    q = (torch.randn((64, 64), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    k = (torch.randn((64, 64), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    v = (torch.randn((64, 64), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    out = torch.empty((64, 64), device="cuda", dtype=torch.float32)
    kernel(q, k, v, out)
    expected = ((q.float() @ k.float().T).to(torch.float16)).float() @ v.float()
    torch.testing.assert_close(out, expected, atol=5e-2, rtol=5e-2)


def test_wgmma_gemm_local_p_rejects_transposed_operands():
    with pytest.raises(ValueError, match="local-P"):

        @T.prim_func
        def main(
            P: T.Tensor((64, 64), T.float16),
            V: T.Tensor((64, 64), T.float16),
            D: T.Tensor((64, 64), T.float32),
        ):
            with T.Kernel(1, threads=128):
                P_frag = T.alloc_fragment((64, 64), T.float16)
                V_shared = T.alloc_shared((64, 64), T.float16)
                O_frag = T.alloc_fragment((64, 64), T.float32)

                T.copy(P, P_frag)
                T.copy(V, V_shared)
                T.wgmma_gemm_local_p(
                    P_frag,
                    V_shared,
                    O_frag,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.wait_wgmma(0)
                T.copy(O_frag, D)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_gemm_local_p_rejects_multi_column_warpgroup_without_remote_p():
    @T.prim_func
    def main(
        Q: T.Tensor((64, 64), T.float16),
        K: T.Tensor((64, 64), T.float16),
        V: T.Tensor((64, 64), T.float16),
        D: T.Tensor((64, 64), T.float32),
    ):
        with T.Kernel(1, threads=256):
            Q_shared = T.alloc_shared((64, 64), T.float16)
            K_shared = T.alloc_shared((64, 64), T.float16)
            V_shared = T.alloc_shared((64, 64), T.float16)
            S_frag = T.alloc_fragment((64, 64), T.float32)
            P_frag = T.alloc_fragment((64, 64), T.float16)
            O_frag = T.alloc_fragment((64, 64), T.float32)

            T.copy(Q, Q_shared)
            T.copy(K, K_shared)
            T.copy(V, V_shared)
            T.wgmma_gemm(
                Q_shared,
                K_shared,
                S_frag,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=True,
            )
            T.wait_wgmma(0)
            for i, j in T.Parallel(64, 64):
                P_frag[i, j] = T.cast(S_frag[i, j], "float16")
            T.wgmma_gemm_local_p(
                P_frag,
                V_shared,
                O_frag,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=True,
            )
            T.wait_wgmma(0)
            T.copy(O_frag, D)

    tilelang.disable_cache()
    try:
        with pytest.raises(ValueError, match="local/remote-P split"):
            tilelang.compile(main, target="cuda -arch=sm_90a")
    finally:
        tilelang.enable_cache()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_gemm_local_remote_p_two_warpgroups_split_output_halves():
    @T.prim_func
    def main(
        P0: T.Tensor((64, 64), T.float16),
        P1: T.Tensor((64, 64), T.float16),
        V0: T.Tensor((64, 512), T.float16),
        V1: T.Tensor((64, 512), T.float16),
        D: T.Tensor((64, 512), T.float32),
    ):
        with T.Kernel(1, threads=256):
            P0_shared = T.alloc_shared((64, 64), T.float16)
            P1_shared = T.alloc_shared((64, 64), T.float16)
            V0_left_shared = T.alloc_shared((64, 256), T.float16)
            V0_right_shared = T.alloc_shared((64, 256), T.float16)
            V1_left_shared = T.alloc_shared((64, 256), T.float16)
            V1_right_shared = T.alloc_shared((64, 256), T.float16)
            P0_frag = T.alloc_fragment((64, 64), T.float16)
            P1_frag = T.alloc_fragment((64, 64), T.float16)
            O_left = T.alloc_fragment((64, 256), T.float32)
            O_right = T.alloc_fragment((64, 256), T.float32)
            tx = T.get_thread_binding()

            T.copy(V0[:, 0:256], V0_left_shared)
            T.copy(V0[:, 256:512], V0_right_shared)
            T.copy(V1[:, 0:256], V1_left_shared)
            T.copy(V1[:, 256:512], V1_right_shared)
            if tx < 128:
                T.copy(P0, P0_frag)
                T.copy(P0_frag, P0_shared)
                T.fill(O_left, 0)
            if tx >= 128:
                T.copy(P1, P1_frag)
                T.copy(P1_frag, P1_shared)
                T.fill(O_right, 0)
            T.sync_threads()
            if tx < 128:
                T.wgmma_gemm_local_p(
                    P0_frag,
                    V0_left_shared,
                    O_left,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wgmma_gemm(
                    P1_shared,
                    V1_left_shared,
                    O_left,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wait_wgmma(0)
                T.copy(O_left, D[:, 0:256])
            if tx >= 128:
                T.wgmma_gemm_local_p(
                    P1_frag,
                    V1_right_shared,
                    O_right,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wgmma_gemm(
                    P0_shared,
                    V0_right_shared,
                    O_right,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wait_wgmma(0)
                T.copy(O_right, D[:, 256:512])

    tilelang.disable_cache()
    try:
        kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    finally:
        tilelang.enable_cache()

    source = kernel.get_kernel_source()
    assert source.count("wgmma_rs") == 2
    assert source.count("wgmma_ss") == 2
    assert "ptx_mma" not in source
    assert "ldmatrix" not in source

    torch.manual_seed(9)
    p0 = (torch.randn((64, 64), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    p1 = (torch.randn((64, 64), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    v0 = (torch.randn((64, 512), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    v1 = (torch.randn((64, 512), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    out = torch.empty((64, 512), device="cuda", dtype=torch.float32)
    kernel(p0, p1, v0, v1, out)
    expected = p0.float() @ v0.float() + p1.float() @ v1.float()
    torch.testing.assert_close(out, expected, atol=7e-2, rtol=7e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_gemm_local_remote_p_two_warpgroups_softmax_split():
    @T.prim_func
    def main(
        S0: T.Tensor((64, 64), T.float32),
        S1: T.Tensor((64, 64), T.float32),
        V0: T.Tensor((64, 512), T.float16),
        V1: T.Tensor((64, 512), T.float16),
        D: T.Tensor((64, 512), T.float32),
    ):
        with T.Kernel(1, threads=256):
            M0_shared = T.alloc_shared((64,), T.float32)
            M1_shared = T.alloc_shared((64,), T.float32)
            L0_shared = T.alloc_shared((64,), T.float32)
            L1_shared = T.alloc_shared((64,), T.float32)
            M_shared = T.alloc_shared((64,), T.float32)
            L_shared = T.alloc_shared((64,), T.float32)
            P0_shared = T.alloc_shared((64, 64), T.float16)
            P1_shared = T.alloc_shared((64, 64), T.float16)
            V0_left_shared = T.alloc_shared((64, 256), T.float16)
            V0_right_shared = T.alloc_shared((64, 256), T.float16)
            V1_left_shared = T.alloc_shared((64, 256), T.float16)
            V1_right_shared = T.alloc_shared((64, 256), T.float16)
            S0_frag = T.alloc_fragment((64, 64), T.float32)
            S1_frag = T.alloc_fragment((64, 64), T.float32)
            P0_frag = T.alloc_fragment((64, 64), T.float16)
            P1_frag = T.alloc_fragment((64, 64), T.float16)
            M0_frag = T.alloc_fragment((64,), T.float32)
            M1_frag = T.alloc_fragment((64,), T.float32)
            L0_frag = T.alloc_fragment((64,), T.float32)
            L1_frag = T.alloc_fragment((64,), T.float32)
            O_left = T.alloc_fragment((64, 256), T.float32)
            O_right = T.alloc_fragment((64, 256), T.float32)
            tx = T.get_thread_binding()

            T.copy(V0[:, 0:256], V0_left_shared)
            T.copy(V0[:, 256:512], V0_right_shared)
            T.copy(V1[:, 0:256], V1_left_shared)
            T.copy(V1[:, 256:512], V1_right_shared)
            if tx < 128:
                T.copy(S0, S0_frag)
                T.fill(M0_frag, -T.infinity("float32"))
                T.reduce_max(S0_frag, M0_frag, dim=1, clear=False)
                T.copy(M0_frag, M0_shared)
            if tx >= 128:
                T.copy(S1, S1_frag)
                T.fill(M1_frag, -T.infinity("float32"))
                T.reduce_max(S1_frag, M1_frag, dim=1, clear=False)
                T.copy(M1_frag, M1_shared)
            T.sync_threads()
            if tx < 128:
                for i in T.Parallel(64):
                    M_shared[i] = T.max(M0_shared[i], M1_shared[i])
                for i, j in T.Parallel(64, 64):
                    S0_frag[i, j] = T.exp2(S0_frag[i, j] - M_shared[i])
                T.fill(L0_frag, 0)
                T.reduce_sum(S0_frag, L0_frag, dim=1)
                T.copy(L0_frag, L0_shared)
            if tx >= 128:
                for i, j in T.Parallel(64, 64):
                    S1_frag[i, j] = T.exp2(S1_frag[i, j] - M_shared[i])
                T.fill(L1_frag, 0)
                T.reduce_sum(S1_frag, L1_frag, dim=1)
                T.copy(L1_frag, L1_shared)
            T.sync_threads()
            if tx < 128:
                for i in T.Parallel(64):
                    L_shared[i] = L0_shared[i] + L1_shared[i]
                for i, j in T.Parallel(64, 64):
                    P0_frag[i, j] = T.cast(T.fast_fdiv(S0_frag[i, j], L_shared[i]), "float16")
                T.copy(P0_frag, P0_shared)
                T.fill(O_left, 0)
            if tx >= 128:
                for i, j in T.Parallel(64, 64):
                    P1_frag[i, j] = T.cast(T.fast_fdiv(S1_frag[i, j], L_shared[i]), "float16")
                T.copy(P1_frag, P1_shared)
                T.fill(O_right, 0)
            T.sync_threads()
            if tx < 128:
                T.wgmma_gemm_local_p(
                    P0_frag,
                    V0_left_shared,
                    O_left,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wgmma_gemm(
                    P1_shared,
                    V1_left_shared,
                    O_left,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wait_wgmma(0)
                T.copy(O_left, D[:, 0:256])
            if tx >= 128:
                T.wgmma_gemm_local_p(
                    P1_frag,
                    V1_right_shared,
                    O_right,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wgmma_gemm(
                    P0_shared,
                    V0_right_shared,
                    O_right,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wait_wgmma(0)
                T.copy(O_right, D[:, 256:512])

    tilelang.disable_cache()
    try:
        kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    finally:
        tilelang.enable_cache()

    source = kernel.get_kernel_source()
    assert source.count("wgmma_rs") == 2
    assert source.count("wgmma_ss") == 2
    assert "ptx_mma" not in source
    assert "ldmatrix" not in source
    assert "__fdividef" in source

    torch.manual_seed(10)
    s0 = torch.randn((64, 64), device="cuda", dtype=torch.float32) * 0.25
    s1 = torch.randn((64, 64), device="cuda", dtype=torch.float32) * 0.25
    v0 = (torch.randn((64, 512), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    v1 = (torch.randn((64, 512), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    out = torch.empty((64, 512), device="cuda", dtype=torch.float32)
    kernel(s0, s1, v0, v1, out)

    logits = torch.cat([s0, s1], dim=1) * math.log(2.0)
    weights = torch.softmax(logits, dim=1)
    expected = weights[:, :64] @ v0.float() + weights[:, 64:] @ v1.float()
    torch.testing.assert_close(out, expected, atol=7e-2, rtol=7e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_gemm_local_remote_p_logical_m32_physical_m64():
    @T.prim_func
    def main(
        P0: T.Tensor((32, 64), T.float16),
        P1: T.Tensor((32, 64), T.float16),
        V0: T.Tensor((64, 512), T.float16),
        V1: T.Tensor((64, 512), T.float16),
        D: T.Tensor((32, 512), T.float32),
    ):
        with T.Kernel(1, threads=256):
            P0_shared = T.alloc_shared((64, 64), T.float16)
            P1_shared = T.alloc_shared((64, 64), T.float16)
            V0_left_shared = T.alloc_shared((64, 256), T.float16)
            V0_right_shared = T.alloc_shared((64, 256), T.float16)
            V1_left_shared = T.alloc_shared((64, 256), T.float16)
            V1_right_shared = T.alloc_shared((64, 256), T.float16)
            P0_frag = T.alloc_fragment((64, 64), T.float16)
            P1_frag = T.alloc_fragment((64, 64), T.float16)
            O_left = T.alloc_fragment((64, 256), T.float32)
            O_right = T.alloc_fragment((64, 256), T.float32)
            tx = T.get_thread_binding()

            T.clear(P0_shared)
            T.clear(P1_shared)
            T.copy(P0, P0_shared[0:32, 0:64])
            T.copy(P1, P1_shared[0:32, 0:64])
            T.copy(V0[:, 0:256], V0_left_shared)
            T.copy(V0[:, 256:512], V0_right_shared)
            T.copy(V1[:, 0:256], V1_left_shared)
            T.copy(V1[:, 256:512], V1_right_shared)
            if tx < 128:
                T.fill(P0_frag, 0)
                T.copy(P0, P0_frag[0:32, 0:64])
                T.fill(O_left, 0)
            if tx >= 128:
                T.fill(P1_frag, 0)
                T.copy(P1, P1_frag[0:32, 0:64])
                T.fill(O_right, 0)
            T.sync_threads()
            if tx < 128:
                T.wgmma_gemm_local_p(
                    P0_frag,
                    V0_left_shared,
                    O_left,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wgmma_gemm(
                    P1_shared,
                    V1_left_shared,
                    O_left,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wait_wgmma(0)
                T.copy(O_left[0:32, 0:256], D[:, 0:256])
            if tx >= 128:
                T.wgmma_gemm_local_p(
                    P1_frag,
                    V1_right_shared,
                    O_right,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wgmma_gemm(
                    P0_shared,
                    V0_right_shared,
                    O_right,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wait_wgmma(0)
                T.copy(O_right[0:32, 0:256], D[:, 256:512])

    tilelang.disable_cache()
    try:
        kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    finally:
        tilelang.enable_cache()

    source = kernel.get_kernel_source()
    assert source.count("wgmma_rs") == 2
    assert source.count("wgmma_ss") == 2
    assert "ptx_mma" not in source
    assert "ldmatrix" not in source

    torch.manual_seed(11)
    p0 = (torch.randn((32, 64), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    p1 = (torch.randn((32, 64), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    v0 = (torch.randn((64, 512), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    v1 = (torch.randn((64, 512), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    out = torch.empty((32, 512), device="cuda", dtype=torch.float32)
    kernel(p0, p1, v0, v1, out)
    expected = p0.float() @ v0.float() + p1.float() @ v1.float()
    torch.testing.assert_close(out, expected, atol=7e-2, rtol=7e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_wgmma_gemm_local_remote_p_logical_m32_k32_split():
    @T.prim_func
    def main(
        P0: T.Tensor((32, 32), T.float16),
        P1: T.Tensor((32, 32), T.float16),
        V0: T.Tensor((32, 512), T.float16),
        V1: T.Tensor((32, 512), T.float16),
        D: T.Tensor((32, 512), T.float32),
    ):
        with T.Kernel(1, threads=256):
            P0_shared = T.alloc_shared((64, 32), T.float16)
            P1_shared = T.alloc_shared((64, 32), T.float16)
            V_shared = T.alloc_shared((64, 512), T.float16)
            P0_frag = T.alloc_fragment((64, 32), T.float16)
            P1_frag = T.alloc_fragment((64, 32), T.float16)
            O_left = T.alloc_fragment((64, 256), T.float32)
            O_right = T.alloc_fragment((64, 256), T.float32)
            tx = T.get_thread_binding()

            T.clear(P0_shared)
            T.clear(P1_shared)
            T.copy(P0, P0_shared[0:32, 0:32])
            T.copy(P1, P1_shared[0:32, 0:32])
            T.copy(V0, V_shared[0:32, 0:512])
            T.copy(V1, V_shared[32:64, 0:512])
            if tx < 128:
                T.fill(P0_frag, 0)
                T.copy(P0, P0_frag[0:32, 0:32])
                T.fill(O_left, 0)
            if tx >= 128:
                T.fill(P1_frag, 0)
                T.copy(P1, P1_frag[0:32, 0:32])
                T.fill(O_right, 0)
            T.sync_threads()
            if tx < 128:
                T.wgmma_gemm_local_p(
                    P0_frag,
                    V_shared[0:32, 0:256],
                    O_left,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wgmma_gemm(
                    P1_shared,
                    V_shared[32:64, 0:256],
                    O_left,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wait_wgmma(0)
                T.copy(O_left[0:32, 0:256], D[:, 0:256])
            if tx >= 128:
                T.wgmma_gemm_local_p(
                    P1_frag,
                    V_shared[32:64, 256:512],
                    O_right,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wgmma_gemm(
                    P0_shared,
                    V_shared[0:32, 256:512],
                    O_right,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wait_wgmma(0)
                T.copy(O_right[0:32, 0:256], D[:, 256:512])

    tilelang.disable_cache()
    try:
        kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    finally:
        tilelang.enable_cache()

    source = kernel.get_kernel_source()
    assert source.count("wgmma_rs") == 2
    assert source.count("wgmma_ss") == 2
    assert "ptx_mma" not in source
    assert "ldmatrix" not in source

    torch.manual_seed(12)
    p0 = (torch.randn((32, 32), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    p1 = (torch.randn((32, 32), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    v0 = (torch.randn((32, 512), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    v1 = (torch.randn((32, 512), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    out = torch.empty((32, 512), device="cuda", dtype=torch.float32)
    kernel(p0, p1, v0, v1, out)
    expected = p0.float() @ v0.float() + p1.float() @ v1.float()
    torch.testing.assert_close(out, expected, atol=7e-2, rtol=7e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "128-thread WGMMA output fragments cannot currently be reassembled into "
        "one 256-thread logical fragment; Dataflow local/remote-P integration "
        "should store split output fields instead."
    ),
)
def test_wgmma_gemm_local_remote_p_logical_m32_k32_reassembles_output_fragment():
    @T.prim_func
    def main(
        P0: T.Tensor((32, 32), T.float16),
        P1: T.Tensor((32, 32), T.float16),
        V0: T.Tensor((32, 512), T.float16),
        V1: T.Tensor((32, 512), T.float16),
        D: T.Tensor((32, 512), T.float32),
    ):
        with T.Kernel(1, threads=256):
            P0_shared = T.alloc_shared((64, 32), T.float16)
            P1_shared = T.alloc_shared((64, 32), T.float16)
            V0_left_shared = T.alloc_shared((32, 256), T.float16)
            V0_right_shared = T.alloc_shared((32, 256), T.float16)
            V1_left_shared = T.alloc_shared((32, 256), T.float16)
            V1_right_shared = T.alloc_shared((32, 256), T.float16)
            P0_frag = T.alloc_fragment((64, 32), T.float16)
            P1_frag = T.alloc_fragment((64, 32), T.float16)
            O_left = T.alloc_fragment((64, 256), T.float32)
            O_right = T.alloc_fragment((64, 256), T.float32)
            O = T.alloc_fragment((32, 512), T.float32)
            tx = T.get_thread_binding()

            T.clear(P0_shared)
            T.clear(P1_shared)
            T.copy(P0, P0_shared[0:32, 0:32])
            T.copy(P1, P1_shared[0:32, 0:32])
            T.copy(V0[:, 0:256], V0_left_shared)
            T.copy(V0[:, 256:512], V0_right_shared)
            T.copy(V1[:, 0:256], V1_left_shared)
            T.copy(V1[:, 256:512], V1_right_shared)
            T.fill(O, 0)
            if tx < 128:
                T.fill(P0_frag, 0)
                T.copy(P0, P0_frag[0:32, 0:32])
                T.fill(O_left, 0)
            if tx >= 128:
                T.fill(P1_frag, 0)
                T.copy(P1, P1_frag[0:32, 0:32])
                T.fill(O_right, 0)
            T.sync_threads()
            if tx < 128:
                T.wgmma_gemm_local_p(
                    P0_frag,
                    V0_left_shared,
                    O_left,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wgmma_gemm(
                    P1_shared,
                    V1_left_shared,
                    O_left,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wait_wgmma(0)
                for i, j in T.Parallel(32, 256):
                    O[i, j] = O_left[i, j]
            if tx >= 128:
                T.wgmma_gemm_local_p(
                    P1_frag,
                    V1_right_shared,
                    O_right,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wgmma_gemm(
                    P0_shared,
                    V0_right_shared,
                    O_right,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.wait_wgmma(0)
                for i, j in T.Parallel(32, 256):
                    O[i, j + 256] = O_right[i, j]
            T.sync_threads()
            T.copy(O, D)

    tilelang.disable_cache()
    try:
        kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    finally:
        tilelang.enable_cache()

    source = kernel.get_kernel_source()
    assert source.count("wgmma_rs") == 2
    assert source.count("wgmma_ss") == 2
    assert "ptx_mma" not in source
    assert "ldmatrix" not in source

    torch.manual_seed(13)
    p0 = (torch.randn((32, 32), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    p1 = (torch.randn((32, 32), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    v0 = (torch.randn((32, 512), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    v1 = (torch.randn((32, 512), device="cuda", dtype=torch.float16) * 0.125).to(torch.float16)
    out = torch.empty((32, 512), device="cuda", dtype=torch.float32)
    kernel(p0, p1, v0, v1, out)
    expected = p0.float() @ v0.float() + p1.float() @ v1.float()
    torch.testing.assert_close(out, expected, atol=7e-2, rtol=7e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_padded_wgmma_gemm_m32_inside_pipeline_compiles_to_wgmma():
    @T.prim_func
    def main(
        A: T.Tensor((32, 256), T.float16),
        B: T.Tensor((64, 256), T.float16),
        D: T.Tensor((32, 64), T.float32),
    ):
        with T.Kernel(1, threads=256):
            A_shared = T.alloc_shared((64, 256), T.float16)
            B_shared = T.alloc_shared((64, 256), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float32)

            for _tile in T.Pipelined(1, num_stages=2):
                T.copy(A[0:64, 0:256], A_shared)
                T.copy(B[0:64, 0:256], B_shared)
                T.padded_wgmma_gemm(
                    A_shared,
                    B_shared,
                    C_local,
                    logical_m=32,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.wait_wgmma(0)

            for i, j in T.Parallel(32, 64):
                D[i, j] = C_local[i, j]

    kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    source = kernel.get_kernel_source()
    assert "wgmma" in source or "warpgroup" in source
    assert "tl::tma_load" in source

    torch.manual_seed(5)
    a = torch.randn((32, 256), device="cuda", dtype=torch.float16)
    b = torch.randn((64, 256), device="cuda", dtype=torch.float16)
    out = torch.empty((32, 64), device="cuda", dtype=torch.float32)
    kernel(a, b, out)
    torch.testing.assert_close(out, a.float() @ b.float().T, atol=3e-2, rtol=3e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_padded_wgmma_gemm_sliced_wide_shared_uses_region_k_inside_pipeline():
    @T.prim_func
    def main(
        A: T.Tensor((64, 512), T.float16),
        B: T.Tensor((64, 512), T.float16),
        D: T.Tensor((32, 64), T.float32),
    ):
        with T.Kernel(1, threads=256):
            A_shared = T.alloc_shared((64, 512), T.float16)
            B_shared = T.alloc_shared((64, 512), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float32)

            for _tile in T.Pipelined(1, num_stages=1):
                T.copy(A[0:64, 0:512], A_shared)
                T.copy(B[0:64, 0:512], B_shared)
                T.padded_wgmma_gemm(
                    A_shared[0:64, 128:192],
                    B_shared[0:64, 128:192],
                    C_local,
                    logical_m=32,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.wait_wgmma(0)

            for i, j in T.Parallel(32, 64):
                D[i, j] = C_local[i, j]

    kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    source = kernel.get_kernel_source()
    assert "tl::tma_load" in source
    assert "wgmma" in source or "warpgroup" in source

    torch.manual_seed(6)
    a = torch.randn((64, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((64, 512), device="cuda", dtype=torch.float16)
    out = torch.empty((32, 64), device="cuda", dtype=torch.float32)
    kernel(a, b, out)
    expected = a[:32, 128:192].float() @ b[:, 128:192].float().T
    torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_padded_wgmma_gemm_sliced_tma_store_region_feeds_sliced_wgmma():
    @T.prim_func
    def main(
        A: T.Tensor((64, 512), T.float16),
        B: T.Tensor((64, 512), T.float16),
        D: T.Tensor((32, 64), T.float32),
    ):
        with T.Kernel(1, threads=256):
            A_shared = T.alloc_shared((64, 512), T.float16)
            B_shared = T.alloc_shared((64, 512), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float32)

            for _tile in T.Pipelined(1, num_stages=1):
                T.copy(A[0:64, 128:192], A_shared[0:64, 128:192])
                T.copy(B[0:64, 128:192], B_shared[0:64, 128:192])
                T.padded_wgmma_gemm(
                    A_shared[0:64, 128:192],
                    B_shared[0:64, 128:192],
                    C_local,
                    logical_m=32,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.wait_wgmma(0)

            for i, j in T.Parallel(32, 64):
                D[i, j] = C_local[i, j]

    kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    source = kernel.get_kernel_source()
    assert "tl::tma_load" in source
    assert "wgmma" in source or "warpgroup" in source

    torch.manual_seed(7)
    a = torch.randn((64, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((64, 512), device="cuda", dtype=torch.float16)
    out = torch.empty((32, 64), device="cuda", dtype=torch.float32)
    kernel(a, b, out)
    expected = a[:32, 128:192].float() @ b[:, 128:192].float().T
    torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)
