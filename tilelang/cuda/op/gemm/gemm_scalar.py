from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang import language as T
from tvm.target import Target
from tvm.ir import Range
from tvm import tir


GEMM_INST_CUDA_SCALAR = "cuda.scalar"


class GemmCudaScalar(GemmBase):
    """CUDA scalar fallback for GEMM shapes that are too small for MMA tiles."""

    def infer_layout(self, target: Target, thread_nums: int):
        return {}

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tir.Var,
        mbar_phase_expr: tir.PrimExpr | None = None,
    ):
        M, N, K = self.M, self.N, self.K
        A_buf = self.ARegion.buffer
        B_buf = self.BRegion.buffer
        C_buf = self.CRegion.buffer
        trans_A = self.trans_A
        trans_B = self.trans_B
        clear_accum = self.clear_accum
        accum_dtype = self.accum_dtype

        a0 = self.ARegion.region[0].min
        a1 = self.ARegion.region[1].min
        b0 = self.BRegion.region[0].min
        b1 = self.BRegion.region[1].min
        c0 = self.CRegion.region[0].min
        c1 = self.CRegion.region[1].min
        element_count = int(M) * int(N)
        thread_count = int(thread_bounds.extent)
        chunks = (element_count + thread_count - 1) // thread_count

        @T.prim_func
        def gemm_cuda_scalar() -> None:
            accum = T.alloc_local((1,), accum_dtype)
            for chunk in T.serial(0, chunks):
                linear = chunk * thread_count + thread_var
                if linear < element_count:
                    i = linear // N
                    j = linear - i * N
                    accum[0] = T.cast(0, accum_dtype)
                    for k in T.serial(0, K):
                        accum[0] += T.cast(
                            A_buf[a0 + (k if trans_A else i), a1 + (i if trans_A else k)]
                            * B_buf[b0 + (j if trans_B else k), b1 + (k if trans_B else j)],
                            accum_dtype,
                        )
                    if clear_accum:
                        C_buf[c0 + i, c1 + j] = accum[0]
                    else:
                        C_buf[c0 + i, c1 + j] += accum[0]

        return gemm_cuda_scalar
