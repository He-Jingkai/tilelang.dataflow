import tilelang
import tilelang.language as T
import tilelang.testing
import pytest


@tilelang.testing.requires_cuda_compute_version_eq(8, 0)
def test_copy_and_async_copy_gemm_codegen_equivalent_sm80():
    """For SM80, T.copy(global->shared) may lower to cp.async.

    This test checks that the explicit form:
      T.async_copy(...) + T.ptx_wait_group(0)
    produces identical CUDA source as:
      T.copy(...)

    This is intentionally a codegen equivalence test (not a perf test).
    """

    M = 256
    N = 256
    K = 128
    block_M = 128
    block_N = 128
    block_K = 32

    @T.prim_func
    def matmul_relu_kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.async_copy(A[by * block_M, ko * block_K], A_shared)
                T.ptx_wait_group(0)

                T.async_copy(B[ko * block_K, bx * block_N], B_shared)
                T.ptx_wait_group(0)

                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            T.copy(C_local, C[by * block_M, bx * block_N])

    async_matmul_relu = matmul_relu_kernel

    @T.prim_func
    def matmul_relu_kernel(  # noqa: F811
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            T.copy(C_local, C[by * block_M, bx * block_N])

    sync_matmul_relu = matmul_relu_kernel

    # Compile both and compare the generated CUDA source.
    async_kernel = tilelang.compile(async_matmul_relu, target="cuda")
    sync_kernel = tilelang.compile(sync_matmul_relu, target="cuda")

    async_src = async_kernel.get_kernel_source()
    sync_src = sync_kernel.get_kernel_source()

    assert async_src == sync_src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_predicated_async_copy_gemm_keeps_global_cp_async_source():
    """Predicated async_copy feeding GEMM must not stage cp.async source through local memory."""

    M = 32
    N = 64
    K = 512
    block_n = 64

    @T.prim_func
    def main(
        Q: T.Tensor((M, K), T.float16),
        KV: T.Tensor((16, 22527, 1, K), T.float16),
        Out: T.Tensor((M, N), T.float16),
        batch: T.int32,
        range_begin: T.int32,
        range_end: T.int32,
    ):
        with T.Kernel(1, threads=128):
            Q_shared = T.alloc_shared((M, K), T.float16)
            KV_shared = T.alloc_shared((block_n, K), T.float16)
            S_shared = T.alloc_shared((M, N), T.float16)
            acc = T.alloc_fragment((M, N), T.float32)
            acc_o = T.alloc_fragment((M, K), T.float32)
            T.copy(Q, Q_shared)
            T.fill(acc, 0)
            T.fill(acc_o, 0)
            full_tile_count = T.floordiv(range_end - range_begin, block_n)
            for tile in T.Pipelined(full_tile_count, num_stages=0):
                kv_start = range_begin + tile * block_n
                T.async_copy(KV[batch, kv_start : kv_start + block_n, 0, 0:K], KV_shared)
                T.ptx_wait_group(0)
                T.sync_threads()
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.copy(acc, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
            T.copy(acc, Out)

    kernel = tilelang.compile(main, out_idx=[2], target="cuda")
    src = kernel.get_kernel_source()
    print("=== predicated async_copy feeding GEMM codegen ===")
    print(src)
    assert "cp_async_gs_conditional<" in src, "Expected predicated cp.async in generated CUDA source"
    assert "KV_local_cast" not in src, "cp.async source must not be a local staging buffer"
    assert "cp_async_gs_conditional<8>((&(KV_shared_local_cast" not in src
    assert "(&(KV[" in src, "Expected cp.async source pointer to reference global KV"


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_async_copy_dynamic_src_upper_bound_stays_global_cp_async():
    """Tail async_copy can add a dynamic source bound without local staging."""

    K = 128
    block_n = 64

    @T.prim_func
    def main(
        KV: T.Tensor((16, 22527, 1, K), T.float16),
        Out: T.Tensor((1,), T.float16),
        batch: T.int32,
        range_begin: T.int32,
        range_end: T.int32,
    ):
        with T.Kernel(1, threads=128):
            KV_shared = T.alloc_shared((block_n, K), T.float16)
            T.async_copy(
                KV[batch, range_begin : range_begin + block_n, 0, 0:K],
                KV_shared,
                src_upper_bounds={1: range_end},
            )
            T.ptx_wait_group(0)
            T.sync_threads()
            Out[0] = KV_shared[0, 0]

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    assert "cp_async_gs_conditional<" in src
    assert "range_end" in src
    assert "KV_local_cast" not in src
    assert "(&(KV[" in src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_async_copy_assume_src_in_bounds_uses_unpredicated_cp_async():
    """Full-tile async_copy can opt out of source predicates when caller proves bounds."""

    K = 128
    block_n = 64

    @T.prim_func
    def main(
        KV: T.Tensor((16, 22527, 1, K), T.float16),
        Out: T.Tensor((1,), T.float16),
        batch: T.int32,
        range_begin: T.int32,
    ):
        with T.Kernel(1, threads=128):
            KV_shared = T.alloc_shared((block_n, K), T.float16)
            T.async_copy(
                KV[batch, range_begin : range_begin + block_n, 0, 0:K],
                KV_shared,
                assume_src_in_bounds=True,
            )
            T.ptx_wait_group(0)
            T.sync_threads()
            Out[0] = KV_shared[0, 0]

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    assert "cp_async_gs<" in src
    assert "cp_async_gs_conditional<" not in src
    assert "KV_local_cast" not in src
    assert "(&(KV[" in src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_async_copy_4d_region_to_2d_shared_preserves_all_columns():
    """Explicit async_copy must keep vector-lane offsets for swizzled shared destinations."""

    torch = pytest.importorskip("torch")

    @T.prim_func
    def main(
        A: T.Tensor((1, 64, 1, 512), T.float16),
        D: T.Tensor((64, 512), T.float16),
    ):
        with T.Kernel(1, threads=256):
            S = T.alloc_shared((64, 512), T.float16)
            T.async_copy(A[0, 0:64, 0, 0:512], S)
            T.ptx_wait_group(0)
            T.sync_threads()
            T.copy(S, D)

    kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    source = kernel.get_kernel_source()
    assert "cp_async" in source

    data = torch.arange(64 * 512, device="cuda", dtype=torch.float16).reshape(1, 64, 1, 512)
    out = torch.empty((64, 512), device="cuda", dtype=torch.float16)
    kernel(data, out)
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), data[0, :, 0, :].float(), rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
def test_async_copy_4d_region_to_gemm_shared_matches_torch():
    """A 4D global region copied with cp.async must land in the layout expected by GEMM."""

    torch = pytest.importorskip("torch")

    @T.prim_func
    def main(
        Q: T.Tensor((32, 512), T.float16),
        KV: T.Tensor((1, 64, 1, 512), T.float16),
        Out: T.Tensor((32, 64), T.float32),
    ):
        with T.Kernel(1, threads=256):
            Q_shared = T.alloc_shared((32, 512), T.float16)
            KV_shared = T.alloc_shared((64, 512), T.float16)
            acc = T.alloc_fragment((32, 64), T.float32)
            T.copy(Q, Q_shared)
            T.async_copy(KV[0, 0:64, 0, 0:512], KV_shared)
            T.ptx_wait_group(0)
            T.sync_threads()
            T.gemm(
                Q_shared,
                KV_shared,
                acc,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
                clear_accum=True,
            )
            T.copy(acc, Out)

    kernel = tilelang.compile(main, target="cuda -arch=sm_90a")
    source = kernel.get_kernel_source()
    assert "cp_async" in source

    torch.manual_seed(11)
    q = torch.randn((32, 512), device="cuda", dtype=torch.float16)
    kv = torch.randn((1, 64, 1, 512), device="cuda", dtype=torch.float16)
    out = torch.empty((32, 64), device="cuda", dtype=torch.float32)
    kernel(q, kv, out)
    torch.cuda.synchronize()

    expected = q.float() @ kv[0, :, 0, :].float().T
    torch.testing.assert_close(out, expected, rtol=3e-2, atol=3e-2)


if __name__ == "__main__":
    tilelang.testing.main()
