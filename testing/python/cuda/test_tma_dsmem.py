"""Regression tests for SM-to-SM cluster copy."""

import torch
import tilelang
import tilelang.language as T
import tilelang.testing
import tvm
import numpy as np


def make_store_cluster_kernel(N: int):
    @T.prim_func
    def kernel(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(2, threads=128, cluster_dims=(2, 1, 1)) as pid:
            s_src = T.alloc_shared((N,), "float32")
            s_dst = T.alloc_shared((N,), "float32")
            s_barrier = T.alloc_cluster_barrier([1])

            T.fill(s_src, 0.0)
            T.fill(s_dst, 0.0)
            T.cluster_sync()

            if pid == 0:
                for i in T.Parallel(N):
                    s_src[i] = A[i]
                T.copy_cluster(s_src, s_dst, dst_block=1, remote_barrier=s_barrier[0])

            if pid == 1:
                T.mbarrier_wait_parity(s_barrier[0], 0)
                for i in T.Parallel(N):
                    B[i] = s_dst[i]

    return kernel


def make_large_store_cluster_kernel(byte_count: int):
    @T.prim_func
    def kernel(
        A: T.Tensor((byte_count,), "uint8"),
        B: T.Tensor((byte_count,), "uint8"),
    ):
        with T.Kernel(2, threads=256, cluster_dims=(2, 1, 1)) as pid:
            s_src = T.alloc_shared((byte_count,), "uint8")
            s_dst = T.alloc_shared((byte_count,), "uint8")
            s_barrier = T.alloc_cluster_barrier([1])

            for chunk in T.serial((byte_count + 255) // 256):
                offset = chunk * 256 + T.get_thread_binding()
                if offset < byte_count:
                    s_src[offset] = A[offset]
                    s_dst[offset] = T.cast(0, "uint8")
            T.cluster_sync()

            if pid == 0:
                T.copy_cluster(
                    s_src,
                    s_dst,
                    dst_block=1,
                    remote_barrier=s_barrier[0],
                )

            if pid == 1:
                T.mbarrier_wait_parity(s_barrier[0], 0)
                T.sync_threads()
                for chunk in T.serial((byte_count + 255) // 256):
                    offset = chunk * 256 + T.get_thread_binding()
                    if offset < byte_count:
                        B[offset] = s_dst[offset]

    return kernel


def make_large_store_cluster_high_destination_kernel(byte_count: int):
    @T.prim_func
    def kernel(
        A: T.Tensor((byte_count,), "uint8"),
        B: T.Tensor((byte_count,), "uint8"),
    ):
        with T.Kernel(2, threads=256, cluster_dims=(2, 1, 1)) as pid:
            arena = T.alloc_shared((2 * byte_count,), "uint8")
            s_barrier = T.alloc_cluster_barrier([1])

            for chunk in T.serial((byte_count + 255) // 256):
                offset = chunk * 256 + T.get_thread_binding()
                if offset < byte_count:
                    arena[offset] = A[offset]
                    arena[byte_count + offset] = T.cast(0, "uint8")
            T.cluster_sync()

            if pid == 0:
                T.copy_cluster(
                    arena[0:byte_count],
                    arena[byte_count : 2 * byte_count],
                    dst_block=1,
                    remote_barrier=s_barrier[0],
                )

            if pid == 1:
                T.mbarrier_wait_parity(s_barrier[0], 0)
                T.sync_threads()
                for chunk in T.serial((byte_count + 255) // 256):
                    offset = chunk * 256 + T.get_thread_binding()
                    if offset < byte_count:
                        B[offset] = arena[byte_count + offset]

    return kernel


def make_store_cluster_producer_wg_kernel(N: int):
    @T.prim_func
    def kernel(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(2, threads=256, cluster_dims=(2, 1, 1)) as pid:
            s_src = T.alloc_shared((N,), "float32")
            s_dst = T.alloc_shared((N,), "float32")
            s_barrier = T.alloc_cluster_barrier([1])

            T.fill(s_src, 0.0)
            T.fill(s_dst, 0.0)
            T.cluster_sync()

            if pid == 0:
                for i in T.Parallel(N):
                    s_src[i] = A[i]
                T.sync_threads()
                if T.get_thread_binding() >= 128:
                    T.copy_cluster(
                        s_src,
                        s_dst,
                        dst_block=1,
                        remote_barrier=s_barrier[0],
                        leader_thread_extent=128,
                    )

            if pid == 1:
                T.mbarrier_wait_parity(s_barrier[0], 0)
                T.sync_threads()
                for i in T.Parallel(N):
                    B[i] = s_dst[i]

    return kernel


def make_store_cluster_ring_kernel(
    cluster_size: int,
    steps: int,
    bytes_per_stage: int,
    *,
    use_consumed: bool = True,
    use_credit: bool = True,
):
    if cluster_size <= 0 or cluster_size & (cluster_size - 1):
        raise ValueError("cluster_size must be a positive power of two")

    @T.prim_func
    def kernel(B: T.Tensor((cluster_size, steps), "int32")):
        with T.Kernel(
            cluster_size,
            threads=384,
            cluster_dims=(cluster_size, 1, 1),
        ) as pid:
            source = T.alloc_shared((bytes_per_stage,), "uint8")
            receive = T.alloc_shared((2, bytes_per_stage), "uint8")
            ready = T.alloc_cluster_barrier([1, 1])
            if use_consumed:
                consumed = T.alloc_barrier([2, 2])
            if use_credit:
                credit = T.alloc_cluster_barrier([2, 2])

            for i in T.serial((bytes_per_stage + 383) // 384):
                source_offset = i * 384 + T.get_thread_binding()
                if source_offset < bytes_per_stage:
                    source[source_offset] = T.cast(pid + 1, "uint8")
            T.cluster_sync()

            if T.get_thread_binding() >= 256:
                for step in T.serial(steps):
                    stage = step & 1
                    phase = (step // 2) & 1
                    if use_consumed:
                        T.mbarrier_wait_parity(consumed[stage], phase ^ 1)
                    if use_credit and step >= 2:
                        T.mbarrier_wait_parity(
                            credit[stage],
                            ((step // 2) - 1) & 1,
                        )
                    destination_rank = (T.block_rank_in_cluster() - step) & (cluster_size - 1)
                    T.copy_cluster(
                        source,
                        receive[stage, :],
                        dst_block=destination_rank,
                        remote_barrier=ready[stage],
                        leader_thread_extent=128,
                    )

            if T.get_thread_binding() < 256:
                for step in T.serial(steps):
                    stage = step & 1
                    phase = (step // 2) & 1
                    T.mbarrier_wait_parity(ready[stage], phase)
                    if T.get_thread_binding() == 0:
                        B[pid, step] = T.cast(receive[stage, 0], "int32")
                    if T.shuffle_elect(128):
                        if use_consumed:
                            T.mbarrier_arrive(consumed[stage])
                        if use_credit and step + 2 < steps:
                            next_source_rank = (T.block_rank_in_cluster() + step + 2) & (cluster_size - 1)
                            T.mbarrier_arrive(credit[stage], next_source_rank)

    return kernel


def make_cluster_barrier_ring_kernel(cluster_size: int, rank_delta: int):
    @T.prim_func
    def kernel(B: T.Tensor((cluster_size,), "int32")):
        with T.Kernel(
            cluster_size,
            threads=128,
            cluster_dims=(cluster_size, 1, 1),
        ) as pid:
            credit = T.alloc_cluster_barrier([1])
            if T.get_thread_binding() == 0:
                target = (T.block_rank_in_cluster() + rank_delta) & (cluster_size - 1)
                T.mbarrier_arrive(credit[0], target)
            T.mbarrier_wait_parity(credit[0], 0)
            if T.get_thread_binding() == 0:
                B[pid] = T.block_rank_in_cluster() + 1

    return kernel


def make_store_cluster_simt_no_barrier_kernel(N: int):
    """No remote_barrier -> SIMT fallback always taken; cluster_sync() orders stores."""

    @T.prim_func
    def kernel(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(2, threads=128, cluster_dims=(2, 1, 1)) as pid:
            s_src = T.alloc_shared((N,), "float32")
            s_dst = T.alloc_shared((N,), "float32")

            T.fill(s_src, 0.0)
            T.fill(s_dst, 0.0)
            T.cluster_sync()

            if pid == 0:
                for i in T.Parallel(N):
                    s_src[i] = A[i]
                # No remote_barrier: cluster copy lowering takes the SIMT path.
                # All threads write into block 1's s_dst via map_shared_rank.
                T.copy_cluster(s_src, s_dst, dst_block=1)

            # Full cluster barrier: ensures all map_shared_rank stores from
            # block 0 are visible in block 1's address space before block 1
            # reads s_dst.
            T.cluster_sync()

            if pid == 1:
                for i in T.Parallel(N):
                    B[i] = s_dst[i]

    return kernel


def make_cluster_pull_kernel(N: int):
    @T.prim_func
    def kernel(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((2, N), "float32"),
    ):
        with T.Kernel(2, threads=128, cluster_dims=(2, 1, 1)) as pid:
            s_src = T.alloc_shared((N,), "float32")
            s_dst = T.alloc_shared((N,), "float32")

            for i in T.Parallel(N):
                s_src[i] = A[i] + T.cast(pid * N, "float32")
                s_dst[i] = 0.0
            T.cluster_sync()

            source_rank = 1 - pid
            T.cluster_pull(s_dst[0], s_src[0], source_rank, N * 4, 0, 128)
            T.sync_threads()
            for i in T.Parallel(N):
                B[pid, i] = s_dst[i]

    return kernel


def test_cluster_pull_uses_registered_intrinsic():
    """Internal cluster transport must not be encoded as an external ABI call."""
    prim_func = make_cluster_pull_kernel(128)
    call_ops = []

    def collect_call(node):
        if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op):
            call_ops.append(node.op.name)

    tvm.tir.stmt_functor.post_order_visit(prim_func.body, collect_call)
    assert "tl.cluster_pull" in call_ops
    assert "tir.call_extern" not in call_ops


def make_store_cluster_simt_barrier_kernel(M: int, N_full: int, N_tile: int):
    """2-D slice copy that forces the SIMT fallback even though remote_barrier is set.

    s_src / s_dst are allocated with inner dimension N_full, but only the
    first N_tile columns are copied.  Because N_tile < N_full the
    is_contiguous_region() check fails: the inner-dim extent of the copy
    region (N_tile) does not equal the buffer shape (N_full).

    Cluster copy lowering falls back to map_shared_rank stores and, because
    remote_barrier was supplied, automatically appends:
        __syncthreads();
        if (threadIdx.x == 0) s_barrier[0].arrive(1u);
    Block 1 therefore waits on the same mbarrier as in the fast-path API,
    verifying that ptx_arrive_cluster_barrier is injected and functional.
    """

    @T.prim_func
    def kernel(
        A: T.Tensor((M, N_tile), "float32"),
        B: T.Tensor((M, N_tile), "float32"),
    ):
        with T.Kernel(2, threads=128, cluster_dims=(2, 1, 1)) as pid:
            # Deliberately wider buffer: N_full > N_tile so the slice
            # [0:M, 0:N_tile] is non-contiguous in row-major storage.
            s_src = T.alloc_shared((M, N_full), "float32")
            s_dst = T.alloc_shared((M, N_full), "float32")
            s_barrier = T.alloc_cluster_barrier([1])

            T.fill(s_src, 0.0)
            T.fill(s_dst, 0.0)
            T.cluster_sync()

            if pid == 0:
                for i, j in T.Parallel(M, N_tile):
                    s_src[i, j] = A[i, j]

                # [0:M, 0:N_tile] inner-dim extent N_tile != N_full
                # contiguity check fails, so this uses the SIMT fallback.
                # Compiler auto-injects: __syncthreads() +
                #   if (t == 0) s_barrier[0].arrive(1u);
                T.copy_cluster(
                    s_src[0:M, 0:N_tile],
                    s_dst[0:M, 0:N_tile],
                    dst_block=1,
                    remote_barrier=s_barrier[0],
                )

            if pid == 1:
                # Block 1 waits on the auto-injected ptx_arrive_cluster_barrier.
                T.mbarrier_wait_parity(s_barrier[0], 0)
                for i, j in T.Parallel(M, N_tile):
                    B[i, j] = s_dst[i, j]

    return kernel


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_store_cluster():
    """Fast path: T.copy_cluster emits tl::tma_store_cluster."""
    N = 128
    prim_func = make_store_cluster_kernel(N)
    mod = tilelang.compile(prim_func, out_idx=[1], execution_backend="cython")

    src = mod.get_kernel_source()
    assert "tl::tma_store_cluster" in src, (
        "Expected tl::tma_store_cluster in generated kernel source; "
        "T.copy_cluster(dst_block=..., remote_barrier=...) may have regressed "
        f"to the SIMT fallback.\nKernel source:\n{src}"
    )

    A = torch.arange(N, dtype=torch.float32, device="cuda")
    B = mod(A)
    np.testing.assert_allclose(
        B.cpu().numpy(),
        A.cpu().numpy(),
        rtol=0,
        atol=0,
        err_msg="tma_store_cluster copy produced wrong result",
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_store_cluster_copies_slot_larger_than_32k():
    byte_count = 33024
    mod = tilelang.compile(
        make_large_store_cluster_kernel(byte_count),
        out_idx=[1],
        execution_backend="cython",
    )

    src = mod.get_kernel_source()
    assert src.count("tl::tma_store_cluster") == 1
    a = torch.arange(byte_count, dtype=torch.int32, device="cuda").remainder(251).to(torch.uint8)
    b = mod(a)
    np.testing.assert_array_equal(b.cpu().numpy(), a.cpu().numpy())


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_store_cluster_from_producer_warpgroup():
    """A nonzero producer warpgroup can elect the bulk DSM TMA issuer."""
    N = 128
    mod = tilelang.compile(
        make_store_cluster_producer_wg_kernel(N),
        out_idx=[1],
        execution_backend="cython",
    )

    src = mod.get_kernel_source()
    assert "tl::tma_store_cluster" in src
    assert "tl::tl_shuffle_elect<128>()" in src

    A = torch.arange(N, dtype=torch.float32, device="cuda")
    B = mod(A)
    np.testing.assert_allclose(B.cpu().numpy(), A.cpu().numpy(), rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_store_cluster_streaming_ring():
    cluster_size = 16
    steps = cluster_size
    mod = tilelang.compile(
        make_store_cluster_ring_kernel(cluster_size, steps, 8192),
        out_idx=[0],
        execution_backend="cython",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
    )

    B = mod()
    expected = np.empty((cluster_size, steps), dtype=np.int32)
    for destination in range(cluster_size):
        for step in range(steps):
            expected[destination, step] = (destination + step) % cluster_size + 1
    np.testing.assert_array_equal(B.cpu().numpy(), expected)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_cluster_barrier_ring_arrival():
    cluster_size = 16
    mod = tilelang.compile(
        make_cluster_barrier_ring_kernel(cluster_size, 2),
        out_idx=[0],
        execution_backend="cython",
    )
    B = mod()
    np.testing.assert_array_equal(
        B.cpu().numpy(),
        np.arange(1, cluster_size + 1, dtype=np.int32),
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_store_cluster_simt_no_barrier():
    """SIMT fallback (no remote_barrier): map_shared_rank + cluster_sync ordering."""
    N = 128
    prim_func = make_store_cluster_simt_no_barrier_kernel(N)
    mod = tilelang.compile(prim_func, out_idx=[1], execution_backend="cython")

    src = mod.get_kernel_source()
    assert "map_shared_rank" in src, f"Expected map_shared_rank in generated source for no-barrier SIMT fallback.\nKernel source:\n{src}"
    assert "tl::tma_store_cluster" not in src, f"No-barrier path must NOT emit tl::tma_store_cluster.\nKernel source:\n{src}"

    A = torch.arange(N, dtype=torch.float32, device="cuda")
    B = mod(A)
    np.testing.assert_allclose(
        B.cpu().numpy(),
        A.cpu().numpy(),
        rtol=0,
        atol=0,
        err_msg="SIMT no-barrier cluster copy produced wrong result",
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_cluster_pull():
    """Destination CTA pulls a peer's DSM tile into local shared memory."""
    N = 2048
    prim_func = make_cluster_pull_kernel(N)
    mod = tilelang.compile(prim_func, out_idx=[1], execution_backend="cython")

    src = mod.get_kernel_source()
    assert "tl::cluster_pull" in src

    A = torch.arange(N, dtype=torch.float32, device="cuda")
    B = mod(A)
    expected = torch.stack((A + N, A))
    torch.testing.assert_close(B, expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_store_cluster_multi_tma_barrier():
    """Multi-TMA path: non-contiguous 2-D slice decomposed into N row TMA calls.

    The compiler decomposes a non-full-span 2-D slice (shape [M, N_tile] inside
    a buffer of shape [M, N_full]) into M individual tma_store_cluster calls -
    one per contiguous row.  The mbarrier arrive_count is updated to M so the
    destination CTA's wait(0) completes only after all rows are transferred.
    """
    M, N_full, N_tile = 4, 64, 32  # M rows, each N_tile elements

    prim_func = make_store_cluster_simt_barrier_kernel(M, N_full, N_tile)
    mod = tilelang.compile(prim_func, out_idx=[1], execution_backend="cython")

    src = mod.get_kernel_source()
    # Multi-TMA path must emit M separate tma_store_cluster calls, not SIMT stores.
    assert "tl::tma_store_cluster" in src, f"Expected tl::tma_store_cluster for multi-TMA row decomposition.\nKernel source:\n{src}"
    assert "map_shared_rank" not in src, f"Multi-TMA path must NOT fall back to map_shared_rank.\nKernel source:\n{src}"
    # The barrier must be initialised with arrive_count == M (one per TMA call).
    assert f"s_barrier[0].init({M})" in src, f"Expected barrier arrive_count={M} for {M}-row decomposition.\nKernel source:\n{src}"
    # Exactly M tma_store_cluster calls should appear in the source.
    assert src.count("tl::tma_store_cluster") == M, (
        f"Expected exactly {M} tma_store_cluster calls, got {src.count('tl::tma_store_cluster')}.\nKernel source:\n{src}"
    )

    A = torch.arange(M * N_tile, dtype=torch.float32, device="cuda").reshape(M, N_tile)
    B = mod(A)
    np.testing.assert_allclose(
        B.cpu().numpy(),
        A.cpu().numpy(),
        rtol=0,
        atol=0,
        err_msg="Multi-TMA row-decomposed cluster copy produced wrong result",
    )


def make_store_cluster_3d_multi_tma_kernel(D: int, M: int, N_full: int, N_tile: int):
    """3-D slice copy decomposed into D*M tma_store_cluster calls.

    Buffer shape is [D, M, N_full]; the copy region is [0:D, 0:M, 0:N_tile].
    Because N_tile < N_full the innermost dim is not full-span.
    MakeTMARows recurses twice (once on dim 0, once on dim 1) producing D*M
    contiguous-row TMA calls and sets barrier arrive_count = D*M.
    """

    @T.prim_func
    def kernel(
        A: T.Tensor((D, M, N_tile), "float32"),
        B: T.Tensor((D, M, N_tile), "float32"),
    ):
        with T.Kernel(2, threads=D * M * N_tile, cluster_dims=(2, 1, 1)) as pid:
            s_src = T.alloc_shared((D, M, N_full), "float32")
            s_dst = T.alloc_shared((D, M, N_full), "float32")
            s_barrier = T.alloc_cluster_barrier([1])

            T.fill(s_src, 0.0)
            T.fill(s_dst, 0.0)
            T.cluster_sync()

            if pid == 0:
                for d, i, j in T.Parallel(D, M, N_tile):
                    s_src[d, i, j] = A[d, i, j]

                T.copy_cluster(
                    s_src[0:D, 0:M, 0:N_tile],
                    s_dst[0:D, 0:M, 0:N_tile],
                    dst_block=1,
                    remote_barrier=s_barrier[0],
                )

            if pid == 1:
                T.mbarrier_wait_parity(s_barrier[0], 0)
                for d, i, j in T.Parallel(D, M, N_tile):
                    B[d, i, j] = s_dst[d, i, j]

    return kernel


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_store_cluster_3d_multi_tma():
    """3-D multi-TMA: recursive decomposition produces D*M separate TMA calls.

    With D=2 and M=4, the two-level recursion over dims 0 and 1 yields 8
    contiguous-row tma_store_cluster calls and initialises the barrier with
    arrive_count=8.
    """
    D, M, N_full, N_tile = 2, 4, 32, 16  # D*M*N_tile == 128 == thread count

    prim_func = make_store_cluster_3d_multi_tma_kernel(D, M, N_full, N_tile)
    mod = tilelang.compile(prim_func, out_idx=[1], execution_backend="cython")

    src = mod.get_kernel_source()
    n_expected = D * M
    assert "tl::tma_store_cluster" in src, f"Expected tl::tma_store_cluster for 3-D multi-TMA.\nKernel source:\n{src}"
    assert f"s_barrier[0].init({n_expected})" in src, (
        f"Expected barrier arrive_count={n_expected} for {D}x{M} decomposition.\nKernel source:\n{src}"
    )
    assert src.count("tl::tma_store_cluster") == n_expected, (
        f"Expected exactly {n_expected} tma_store_cluster calls, got {src.count('tl::tma_store_cluster')}.\nKernel source:\n{src}"
    )

    A = torch.arange(D * M * N_tile, dtype=torch.float32, device="cuda").reshape(D, M, N_tile)
    B = mod(A)
    np.testing.assert_allclose(
        B.cpu().numpy(),
        A.cpu().numpy(),
        rtol=0,
        atol=0,
        err_msg="3-D multi-TMA cluster copy produced wrong result",
    )


if __name__ == "__main__":
    tilelang.testing.main()
