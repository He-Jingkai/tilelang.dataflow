from tilelang import tvm as tvm
from tvm import tir
from tvm.target import Target
from tvm.ir.base import Node
from tvm.ir import Range
from tvm.runtime import Scriptable
import tvm_ffi
from .registry import resolve_gemm_impl
from tilelang import _ffi_api
from tilelang.utils.target import target_is_cuda


@tvm_ffi.register_global_func("tl.gemm.infer_layout")
def gemm_infer_layout(gemm, target: Target, thread_bounds: Range):
    thread_nums = thread_bounds.extent
    return gemm.infer_layout(target, thread_nums)


@tvm_ffi.register_global_func("tl.gemm.lower")
def gemm_lower(
    gemm,
    layout_map,
    target: Target,
    thread_bounds: Range,
    thread_var: tir.Var,
    mbar_phase_expr: tir.PrimExpr,
):
    # We pass thread_bounds rather than thread_extents because tcgen5mma need to check this
    stmt = gemm.lower(layout_map, target, thread_bounds, thread_var, mbar_phase_expr)
    return stmt


@tvm_ffi.register_object("tl.Gemm")
class Gemm(Node, Scriptable):
    # FFI fields (LLVM/MLIR-style lowerCamel via reflection):
    # a, b, c, aPtr, bPtr, cPtr, m, n, k, transA, transB,
    # strideA, strideB, offsetA, offsetB, clearAccum, kPack, wgWait, policy
    #
    # Backward-compat alias properties are provided below to support old names.

    # Backward-compat alias properties (old API → new FFI fields)
    @property
    def A(self):
        return self.a

    @property
    def B(self):
        return self.b

    @property
    def C(self):
        return self.c

    @property
    def APtr(self):
        return self.aPtr

    @property
    def BPtr(self):
        return self.bPtr

    @property
    def CPtr(self):
        return self.cPtr

    @property
    def M(self):
        return self.m

    @property
    def N(self):
        return self.n

    @property
    def K(self):
        return self.k

    @property
    def trans_A(self):
        return self.transA

    @property
    def trans_B(self):
        return self.transB

    @property
    def stride_A(self):
        return self.strideA

    @property
    def stride_B(self):
        return self.strideB

    @property
    def offset_A(self):
        return self.offsetA

    @property
    def offset_B(self):
        return self.offsetB

    @property
    def clear_accum(self):
        return self.clearAccum

    @property
    def k_pack(self):
        return self.kPack

    @property
    def wg_wait(self):
        return self.wgWait

    @property
    def is_tcgen05(self):
        return getattr(self, "isTcgen05", False)

    @property
    def is_wgmma(self):
        return getattr(self, "isWgmma", False)

    @property
    def sf_a_id(self):
        return self.sfAId

    @property
    def sf_b_id(self):
        return self.sfBId

    def infer_layout(self, target: Target, thread_nums: int):
        """Infer the layout for the GEMM operation based on target architecture."""
        gemm_inst = self._select_gemm_instruction(thread_nums, target)
        impl_class = self._get_implementation_class(gemm_inst, target)
        return impl_class(self).infer_layout(target, thread_nums)

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tir.Var,
        mbar_phase_expr: tir.PrimExpr,
    ):
        """Lower the GEMM operation to TIR statements based on target architecture."""
        thread_nums = thread_bounds.extent
        gemm_inst = self._select_gemm_instruction(thread_nums, target)
        impl_class = self._get_implementation_class(gemm_inst, target)
        return impl_class(self).lower(layout_map, target, thread_bounds, thread_var, mbar_phase_expr)

    def _select_gemm_instruction(self, thread_nums: int, target: Target) -> str:
        """Select the appropriate GEMM instruction key based on target and thread configuration.

        The selection logic chooses:
        1. TCGEN5MMA for Blackwell architecture
        2. WGMMA for Hopper architecture with sufficient matrix size and warp count
        3. MFMA for CDNA (AMD) architecture
        4. MMA for CUDA architecture when the shape satisfies MMA tile constraints
        5. Scalar for CPU target or CUDA shapes that cannot use MMA

        Args:
            thread_nums: Number of threads in the block
            target: Target architecture

        Returns:
            The selected backend-specific GEMM instruction key.
        """
        if self.should_use_cuda_scalar_fallback(target):
            return "cuda.scalar"
        return str(_ffi_api.GemmGetGemmInstructionKey(self, int(thread_nums), target))

    def should_use_cuda_scalar_fallback(self, target: Target) -> bool:
        if not target_is_cuda(target) or self.is_wgmma or self.is_tcgen05:
            return False
        try:
            m_extent = int(self.M)
            n_extent = int(self.N)
            k_extent = int(self.K)
        except (TypeError, ValueError):
            return False
        if m_extent >= 64:
            return False
        return not self.cuda_mma_tile_shape_supported(m_extent, n_extent, k_extent)

    def cuda_mma_tile_shape_supported(self, m_extent: int, n_extent: int, k_extent: int) -> bool:
        # The warp-level CUDA MMA emitter supports m16n8k{8,16,32,64,128,256}
        # tiles. Hopper WGMMA still requires M >= 64, but M=16/32 should fall
        # through to cuda.mma instead of the slow scalar fallback.
        try:
            input_bits = self.A.dtype.bits
        except AttributeError:
            return False
        micro_k = min(256 // input_bits, k_extent)
        return (
            m_extent >= 16
            and m_extent % 16 == 0
            and n_extent >= 8
            and n_extent % 8 == 0
            and k_extent >= micro_k
            and k_extent % micro_k == 0
        )

    def _get_implementation_class(self, gemm_inst: str, target: Target):
        """Get the appropriate implementation class for the given GEMM instruction key.

        Args:
            gemm_inst: The selected backend-specific GEMM instruction key
            target: Target architecture

        Returns:
            The implementation class for the instruction key

        Raises:
            NotImplementedError: If the instruction key is not supported
            ValueError: If the instruction key is unknown
        """
        return resolve_gemm_impl(gemm_inst, target)
