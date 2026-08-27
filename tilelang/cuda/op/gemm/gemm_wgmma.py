from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.layout import (
    make_full_bank_swizzled_layout,
    make_half_bank_swizzled_layout,
    make_quarter_bank_swizzled_layout,
    make_linear_layout,
    Layout,
)
from tilelang.cuda.intrinsics.macro.wgmma_macro_generator import (
    TensorCoreIntrinEmitter,
)
from tilelang.utils.language import is_shared, is_fragment
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.ir import Range
from tvm import tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify
from collections.abc import Callable


GEMM_INST_WGMMA = "cuda.wgmma"


class GemmWGMMA(GemmBase):
    def annotation_value(self, key: str):
        annotations = getattr(self.gemm_node, "annotations", None)
        if annotations is None:
            return None
        try:
            return annotations.get(key, None)
        except AttributeError:
            try:
                return annotations[key]
            except KeyError:
                return None

    def annotation_bool(self, key: str, default: bool) -> bool:
        value = self.annotation_value(key)
        if value is None:
            return default
        if hasattr(value, "value"):
            return bool(value.value)
        try:
            return bool(int(value))
        except (TypeError, ValueError):
            return bool(value)

    def rs_a_is_local_p(self) -> bool:
        return self.annotation_bool("wgmma_rs_a_is_local_p", False)

    def annotation_int(self, key: str) -> int | None:
        value = self.annotation_value(key)
        if value is None:
            return None
        if hasattr(value, "value"):
            value = value.value
        return int(value)

    def make_local_p_layout_emitter(self, target: Target, thread_nums: int) -> TensorCoreIntrinEmitter:
        """Build the QK-accumulator layout used by FlashMLA-style local-P PV."""
        p_n = int(self.K)
        m_warp, n_warp = self.policy.compute_warp_partition(
            self.M,
            p_n,
            thread_nums,
            target,
            GEMM_INST_WGMMA,
        )
        if n_warp != 1:
            raise ValueError(
                "T.wgmma_gemm_local_p currently requires a single local-P "
                "column partition. Multi-column warpgroup QK tiles need an "
                "explicit local/remote-P split before PV WGMMA RS."
            )
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(p_n // n_warp)
        return TensorCoreIntrinEmitter(
            a_dtype=self.in_dtype,
            b_dtype=self.in_dtype,
            accum_dtype=self.in_dtype,
            a_transposed=False,
            b_transposed=False,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=p_n,
        )

    def validate_local_p_a_capacity(self, a_layout: Layout, mma_emitter: TensorCoreIntrinEmitter) -> None:
        output_numel = 1
        for extent in a_layout.get_output_shape():
            output_numel *= int(extent)
        actual_regs = (output_numel * self.A.dtype.bits + 31) // 32
        required_regs = mma_emitter.wgmma_a_regs
        if actual_regs < required_regs:
            raise ValueError(
                "T.wgmma_gemm_local_p local-P A fragment has insufficient "
                f"register capacity for WGMMA RS: actual={actual_regs}, "
                f"required={required_regs}. Multi-column warpgroup QK tiles "
                "need an explicit local/remote-P split before PV WGMMA RS."
            )

    def infer_shared_layout(self, continuity: int) -> Callable[[tir.Buffer], Layout]:
        """Infer the swizzle layout for shared memory based on continuity.

        WGMMA can directly use shared memory as input, so the swizzle layout must
        match the tensor core's access pattern. The swizzle granularity is determined
        by the continuous dimension size:
          - 128B swizzle (Full):    continuity % (vectorized_size * 8) == 0
          - 64B swizzle (Half):     continuity % (vectorized_size * 4) == 0
          - 32B swizzle (Quarter):  continuity % (vectorized_size * 2) == 0
          - Linear (no swizzle):    otherwise

        See: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
        """
        vectorized_size = 128 // self.in_dtype.bits

        def layout_or_linear(layout_factory, buffer: tir.Buffer) -> Layout:
            try:
                stride = int(buffer.shape[-2])
            except (IndexError, TypeError, ValueError):
                return make_linear_layout(buffer)
            if stride % 8 != 0:
                return make_linear_layout(buffer)
            return layout_factory(buffer)

        if continuity % (vectorized_size * 8) == 0:
            return lambda buffer: layout_or_linear(make_full_bank_swizzled_layout, buffer)
        elif continuity % (vectorized_size * 4) == 0:
            return lambda buffer: layout_or_linear(make_half_bank_swizzled_layout, buffer)
        elif continuity % (vectorized_size * 2) == 0:
            return lambda buffer: layout_or_linear(make_quarter_bank_swizzled_layout, buffer)
        else:
            return make_linear_layout

    def infer_layout(self, target: Target, thread_nums: int):
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_WGMMA)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        mma_emitter = TensorCoreIntrinEmitter(
            a_dtype=self.in_dtype,
            b_dtype=self.in_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
            instruction_n=self.annotation_int("wgmma_instruction_n"),
        )
        a_is_k_major = not self.trans_A
        b_is_k_major = self.trans_B
        a_continuity = self.K if a_is_k_major else mma_emitter.wgmma_inst_m
        b_continuity = self.K if b_is_k_major else mma_emitter.wgmma_inst_n
        if self.is_gemm_ss():
            return {
                # WGMMA does not support padding
                self.A: self.infer_shared_layout(a_continuity)(self.A),
                self.B: self.infer_shared_layout(b_continuity)(self.B),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_rs():
            if self.rs_a_is_local_p():
                a_layout = self.make_local_p_layout_emitter(target, thread_nums).make_mma_store_layout(self.A)
                self.validate_local_p_a_capacity(a_layout, mma_emitter)
            else:
                a_layout = mma_emitter.make_mma_load_layout(self.A, matrix="A")
            return {
                self.A: a_layout,
                self.B: self.infer_shared_layout(b_continuity)(self.B),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        else:
            raise ValueError(f"Unsupported gemm combination for wgmma, A: {self.A.scope()}, B: {self.B.scope()}")

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tir.Var,
        mbar_phase_expr: tir.PrimExpr | None = None,
    ):
        thread_nums = thread_bounds.extent
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_WGMMA)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        mma_emitter = TensorCoreIntrinEmitter(
            a_dtype=self.in_dtype,
            b_dtype=self.in_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
            thread_var=thread_var,
            instruction_n=self.annotation_int("wgmma_instruction_n"),
        )

        if self.A in layout_map:
            mma_emitter._assign_a_shared_layout(layout_map[self.A])
        if self.B in layout_map:
            mma_emitter._assign_b_shared_layout(layout_map[self.B])

        # Get base offsets from regions
        # All dimensions may have offsets, including the matrix dimensions
        # However, for WGMMA, we pass the Buffer directly and handle offsets
        # through proper indexing in the access_ptr call or buffer slicing

        # We use region for memory input to support strided gemm
        # T.gemm(A_shared[0:128, :], B_shared, C_local)
        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        clear_accum = self.clear_accum
        wg_wait = self.wg_wait
        emit_arrive = self.annotation_bool("wgmma_emit_arrive", True)
        emit_commit = self.annotation_bool("wgmma_emit_commit", True)
        emit_fence_before = self.annotation_bool("wgmma_emit_fence_before", True)
        emit_fence_after = self.annotation_bool("wgmma_emit_fence_after", True)
        shared_a_logical_m = self.annotation_int("wgmma_rs_shared_a_logical_m")

        if self.is_gemm_ss():
            # For WGMMA, we need to handle buffer region offsets
            # If there are offsets, we create a BufferLoad inside the prim_func
            # to properly generate offset access

            @T.prim_func
            def _gemm_ssr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                # Perform Matrix Multiplication with offset consideration
                if shared_a_logical_m is not None:
                    mma_emitter.wgmma_shared_a_rs(
                        A_region,
                        B_region,
                        C_region,
                        shared_a_logical_m,
                        clear_accum,
                    )
                else:
                    mma_emitter.wgmma(
                        A_region,
                        B_region,
                        C_region,
                        clear_accum,
                        wg_wait,
                        emit_arrive,
                        emit_commit,
                        emit_fence_before,
                        emit_fence_after,
                    )

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_ssr, inline_let=True)
        elif self.is_gemm_rs():

            @T.prim_func
            def _gemm_rsr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                mma_emitter.wgmma(
                    A_region,
                    B_region,
                    C_region,
                    clear_accum,
                    wg_wait,
                    emit_arrive,
                    emit_commit,
                    emit_fence_before,
                    emit_fence_after,
                )

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_rsr, inline_let=True)
        raise ValueError(f"Unsupported gemm combination for wgmma, A: {self.A.scope()}, B: {self.B.scope()}")

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_sr(self) -> bool:
        return is_shared(self.A) and is_fragment(self.B)

    def is_gemm_rs(self) -> bool:
        return is_fragment(self.A) and is_shared(self.B)

    def is_gemm_rr(self) -> bool:
        return is_fragment(self.A) and is_fragment(self.B)
