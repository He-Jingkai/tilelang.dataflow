from tilelang import tvm as tvm
from tvm.ir.base import Node
from tvm.runtime import Scriptable
import tvm_ffi
from tvm.target import Target
from tilelang import _ffi_api


@tvm_ffi.register_object("tl.Fill")
class Fill(Node, Scriptable): ...


@tvm_ffi.register_object("tl.AtomicAdd")
class AtomicAdd(Node, Scriptable): ...


@tvm_ffi.register_object("tl.Copy")
class Copy(Node, Scriptable): ...


@tvm_ffi.register_object("tl.TransferContract")
class TransferContract(Node, Scriptable):
    schema_version: int
    src_valid_region: tvm.tir.BufferRegion
    oob_fill: tvm.tir.PrimExpr
    allow_async: bool
    sync_owner: int


@tvm_ffi.register_object("tl.TransferLoweringPlan")
class TransferLoweringPlan(Node, Scriptable):
    schema_version: int
    implementation_id: str
    supported: bool
    asynchronous: bool
    uses_tma_descriptor: bool
    requires_post_fill: bool
    selection_reason: str
    rejected_candidates: list[str]


@tvm_ffi.register_object("tl.Conv2DIm2Col")
class Conv2DIm2ColOp(Node, Scriptable): ...


@tvm_ffi.register_object("tl.GemmWarpPolicy")
class GemmWarpPolicy(Node, Scriptable):
    policy_type: int
    m_warp: int
    n_warp: int

    def compute_warp_partition(self, M: int, N: int, block_size: int, target: Target, gemm_inst: str):
        _ffi_api.GemmWarpPolicyComputeWarpPartition(self, int(M), int(N), int(block_size), target, gemm_inst)
        return self.m_warp, self.n_warp


@tvm_ffi.register_object("tl.GemmContract")
class GemmContract(Node, Scriptable):
    schema_version: int
    logical_m: int
    logical_n: int
    logical_k: int
    padding_value: tvm.tir.PrimExpr
    allow_padding: bool


@tvm_ffi.register_object("tl.GemmTemporaryRequirement")
class GemmTemporaryRequirement(Node, Scriptable):
    buffer_role: str
    storage_scope: str
    logical_shape: list[int]
    physical_shape: list[int]
    neutral_value: tvm.tir.PrimExpr
    initialization_required: bool
    lifetime_start: str
    lifetime_end: str
    estimated_bytes: int
    additional_bytes: int


@tvm_ffi.register_object("tl.GemmLoweringPlan")
class GemmLoweringPlan(Node, Scriptable):
    schema_version: int
    implementation_id: str
    supported: bool
    synchronous: bool
    logical_shape: list[int]
    physical_shape: list[int]
    requires_padding: bool
    requires_materialization: bool
    temporary_requirements: list[GemmTemporaryRequirement]
    additional_shared_memory_bytes: int
    additional_fragment_bytes: int
    estimated_resource_bytes: int
    selection_reason: str
    rejected_candidates: list[str]


@tvm_ffi.register_object("tl.GemmSPWarpPolicy")
class GemmSPWarpPolicy(Node, Scriptable):
    policy_type: int
    m_warp: int
    n_warp: int

    def compute_warp_partition(self, M: int, N: int, block_size: int, target: Target, gemm_inst: str):
        _ffi_api.GemmSPWarpPolicyComputeWarpPartition(self, int(M), int(N), int(block_size), target, gemm_inst)
        return self.m_warp, self.n_warp


@tvm_ffi.register_object("tl.FinalizeReducerOp")
class FinalizeReducerOp(Node, Scriptable): ...


@tvm_ffi.register_object("tl.ParallelOp")
class ParallelOp(Node, Scriptable): ...


@tvm_ffi.register_object("tl.ReduceOp")
class ReduceOp(Node, Scriptable): ...


@tvm_ffi.register_object("tl.CumSumOp")
class CumSumOp(Node, Scriptable): ...


@tvm_ffi.register_object("tl.RegionOp")
class RegionOp(Node, Scriptable): ...


@tvm_ffi.register_object("tl.ReduceType")
class ReduceType(Node, Scriptable): ...
