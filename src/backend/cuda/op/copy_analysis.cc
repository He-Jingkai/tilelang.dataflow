/*!
 * \file tl/backend/cuda/op/copy_analysis.cc
 * \brief CUDA copy instruction classification helpers.
 */

#include "backend/cuda/op/copy.h"

#include "op/builtin.h"
#include "op/utils.h"
#include "target/utils.h"

#include <tvm/tir/transform.h>

#include <sstream>
#include <utility>

namespace tvm {
namespace tl {
namespace cuda {

using namespace tir;

namespace {

PrimExpr TMABytesFromElements(PrimExpr elements, DataType dtype) {
  PrimExpr elements_i64 = cast(DataType::Int(64), elements);
  int bits = dtype.bits();
  if (bits % 8 == 0) {
    return elements_i64 * IntImm(DataType::Int(64), bits / 8);
  }
  return FloorDiv(elements_i64 * IntImm(DataType::Int(64), bits) +
                      IntImm(DataType::Int(64), 7),
                  IntImm(DataType::Int(64), 8));
}

PrimExpr TMABitsFromElements(PrimExpr elements, DataType dtype) {
  return cast(DataType::Int(64), elements) *
         IntImm(DataType::Int(64), dtype.bits());
}

bool GetBoolAnnotation(const CopyNode &op, const char *key) {
  if (auto val = op.annotations.Get(key)) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value != 0;
    }
  }
  return false;
}

bool GetDisableTMA(const CopyNode &op) {
  return GetBoolAnnotation(op, "disable_tma");
}

bool GetIsTmaCopy(const CopyNode &op) {
  return GetBoolAnnotation(op, "is_tma_copy");
}

int64_t GetClusterMask(const CopyNode &op) {
  if (auto val = op.annotations.Get("cluster_mask")) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value;
    }
  }
  return 0;
}

bool GetIsAsyncCopy(const CopyNode &op) {
  if (GetBoolAnnotation(op, "is_async_copy")) {
    return true;
  }
  return GetBoolAnnotation(op, "force_cp_async");
}

bool GetNoImplicitAsyncCommitWait(const CopyNode &op) {
  return GetBoolAnnotation(op, attr::kAsyncCopyNoImplicitCommitWait);
}

bool CheckGlobalStrides(const Buffer &buffer, arith::Analyzer *analyzer,
                        bool emit_diagnostics) {
  Array<PrimExpr> strides = buffer->strides;
  if (strides.empty()) {
    PrimExpr stride = 1;
    strides.resize(buffer->shape.size());
    for (int i = static_cast<int>(buffer->shape.size()) - 1; i >= 0; --i) {
      strides.Set(i, stride);
      stride *= buffer->shape[i];
    }
  }

  if (!strides.empty() &&
      analyzer->CanProve(strides[strides.size() - 1] != 1,
                         arith::ProofStrength::kSymbolicBound)) {
    if (emit_diagnostics) {
      DLOG(WARNING)
          << "TMA bulk copy requires contiguous innermost global stride"
          << ", but got " << strides[strides.size() - 1] << " for buffer "
          << buffer->name << ", fallback to normal copy.";
    }
    return false;
  }

  for (size_t i = 0; i + 1 < strides.size(); ++i) {
    PrimExpr stride_bytes = TMABytesFromElements(strides[i], buffer->dtype);
    if (analyzer->CanProve(
            FloorMod(stride_bytes, IntImm(DataType::Int(64), 16)) != 0,
            arith::ProofStrength::kSymbolicBound)) {
      if (emit_diagnostics) {
        DLOG(WARNING) << "TMA bulk copy cannot support a global stride of "
                      << stride_bytes << " for buffer " << buffer->name
                      << ", fallback to normal copy.";
      }
      return false;
    }
    if (const int64_t *stride =
            as_const_int(analyzer->Simplify(stride_bytes))) {
      if (*stride >= (int64_t{1} << 40)) {
        if (emit_diagnostics) {
          DLOG(WARNING) << "TMA bulk copy cannot support a global stride of "
                        << stride_bytes << " for buffer " << buffer->name
                        << ", fallback to normal copy.";
        }
        return false;
      }
    }
  }
  return true;
}

bool CheckBulkLoad(const CopyNode &op, Target target, arith::Analyzer *analyzer,
                   bool check_last_dim, bool emit_diagnostics) {
  if (!TargetHasBulkCopy(target)) {
    return false;
  }
  if (op.src.scope() != "global" ||
      (op.dst.scope() != "shared.dyn" && op.dst.scope() != "shared")) {
    return false;
  }
  if (check_last_dim &&
      analyzer->CanProve(
          FloorMod(
              TMABitsFromElements(op.src_range[op.src_range.size() - 1]->extent,
                                  op.src->dtype),
              IntImm(DataType::Int(64), 128)) != 0,
          arith::ProofStrength::kSymbolicBound)) {
    if (emit_diagnostics) {
      DLOG(WARNING)
          << "src range must have last dim multiple of 16 for tma bulk load "
          << op.src->name << " range "
          << op.src_range[op.src_range.size() - 1]->extent << " * "
          << op.src->dtype.bits() << " bits % 128 != 0";
    }
    return false;
  }

  if (op.src->dtype != op.dst->dtype) {
    if (emit_diagnostics) {
      DLOG(WARNING) << "src and dst must have the same dtype for tma load "
                    << op.src->name << " vs. " << op.dst->name << " dtype "
                    << op.src->dtype << " vs. " << op.dst->dtype
                    << " will be fallback to normal copy";
    }
    return false;
  }
  return CheckGlobalStrides(op.src, analyzer, emit_diagnostics);
}

bool CheckBulkStore(const CopyNode &op, Target target,
                    arith::Analyzer *analyzer, bool check_last_dim,
                    bool emit_diagnostics) {
  if (!TargetHasBulkCopy(target)) {
    return false;
  }
  if ((op.src.scope() != "shared.dyn" && op.src.scope() != "shared") ||
      op.dst.scope() != "global") {
    return false;
  }
  if (check_last_dim &&
      analyzer->CanProve(
          FloorMod(
              TMABitsFromElements(op.dst_range[op.dst_range.size() - 1]->extent,
                                  op.dst->dtype),
              IntImm(DataType::Int(64), 128)) != 0,
          arith::ProofStrength::kSymbolicBound)) {
    if (emit_diagnostics) {
      DLOG(WARNING)
          << "dst range must have last dim multiple of 16 for tma bulk store "
          << op.dst->name << " range "
          << op.dst_range[op.dst_range.size() - 1]->extent << " * "
          << op.dst->dtype.bits() << " bits % 128 != 0";
    }
    return false;
  }
  if (op.src->dtype != op.dst->dtype) {
    if (emit_diagnostics) {
      DLOG(WARNING) << "src and dst must have the same dtype for tma store "
                    << op.src->name << " vs. " << op.dst->name << " dtype "
                    << op.src->dtype << " vs. " << op.dst->dtype
                    << " will be fallback to normal copy";
    }
    return false;
  }
  return CheckGlobalStrides(op.dst, analyzer, emit_diagnostics);
}

bool CheckBulkCopy1D(const Buffer &global_tensor, const Buffer &shared_tensor,
                     const Array<Range> &global_range,
                     const Array<Range> &shared_range,
                     const LayoutMap &layout_map, arith::Analyzer *analyzer) {
  bool shared_is_contiguous = true;
  if (layout_map.count(shared_tensor)) {
    Layout existing =
        layout_map.Get(shared_tensor).value().as<Layout>().value();
    Layout linear_layout = makeLinearLayout(shared_tensor->shape);
    shared_is_contiguous = StructuralEqual()(existing, linear_layout);
  }

  bool global_is_contiguous = true;
  bool global_not_full_dim_encounter = false;
  for (int i = global_range.size() - 1; i >= 0; i--) {
    if (!global_not_full_dim_encounter) {
      if (!analyzer->CanProve(global_range[i]->extent ==
                                      global_tensor->shape[i] &&
                                  global_range[i]->min == 0,
                              arith::ProofStrength::kSymbolicBound)) {
        global_not_full_dim_encounter = true;
      }
    } else {
      if (!analyzer->CanProve(global_range[i]->extent == 1,
                              arith::ProofStrength::kSymbolicBound)) {
        global_is_contiguous = false;
        break;
      }
    }
  }

  PrimExpr shared_elements = 1;
  for (size_t i = 0; i < shared_range.size(); i++) {
    shared_elements *= shared_range[i]->extent;
  }
  PrimExpr global_elements = 1;
  for (size_t i = 0; i < global_range.size(); i++) {
    global_elements *= global_range[i]->extent;
  }
  bool element_match =
      analyzer->CanProveEqual(shared_elements, global_elements);
  return shared_is_contiguous && global_is_contiguous && element_match;
}

bool CheckBulkLoad1D(const CopyNode &op, Target target,
                     const LayoutMap &layout_map, arith::Analyzer *analyzer,
                     bool emit_diagnostics) {
  if (!CheckBulkLoad(op, target, analyzer, false, emit_diagnostics)) {
    return false;
  }
  return CheckBulkCopy1D(op.src, op.dst, op.src_range, op.dst_range, layout_map,
                         analyzer);
}

bool CheckBulkStore1D(const CopyNode &op, Target target,
                      const LayoutMap &layout_map, arith::Analyzer *analyzer,
                      bool emit_diagnostics) {
  if (!CheckBulkStore(op, target, analyzer, false, emit_diagnostics)) {
    return false;
  }
  return CheckBulkCopy1D(op.dst, op.src, op.dst_range, op.src_range, layout_map,
                         analyzer);
}

bool CheckLDSMCopy(const CopyNode &op, Target target) {
  return TargetHasLdmatrix(target) && IsSharedBuffer(op.src) &&
         IsFragmentBuffer(op.dst);
}

bool CheckSTSMCopy(const CopyNode &op, Target target) {
  return TargetHasStmatrix(target) && IsFragmentBuffer(op.src) &&
         IsSharedBuffer(op.dst);
}

bool CheckTMemLoad(const CopyNode &op, Target target) {
  return TargetHasTmem(target) && op.src.scope() == "shared.tmem" &&
         IsFragmentBuffer(op.dst);
}

bool CheckTMemStore(const CopyNode &op, Target target) {
  return TargetHasTmem(target) && IsFragmentBuffer(op.src) &&
         op.dst.scope() == "shared.tmem";
}

bool CheckCPAsyncCopyPreconditions(const CopyNode &op) {
  return IsGlobalBuffer(op.src) && IsSharedBuffer(op.dst) &&
         op.src->dtype == op.dst->dtype;
}

bool CheckCPAsyncCopy(const CopyNode &op, Target target,
                      const LayoutMap &layout_map, arith::Analyzer *analyzer) {
  if (!TargetHasAsyncCopy(target)) {
    return false;
  }
  if (!CheckCPAsyncCopyPreconditions(op)) {
    return false;
  }
  // Skip vectorize size checks here because the layout is not stable during
  // layout inference and transform classification.
  return true;
}

bool CanProveRangeInAllocation(const Range &range, const PrimExpr &extent,
                               arith::Analyzer *analyzer) {
  return analyzer->CanProve(range->min >= 0,
                            arith::ProofStrength::kSymbolicBound) &&
         analyzer->CanProve(range->min + range->extent <= extent,
                            arith::ProofStrength::kSymbolicBound);
}

bool IsTMADtypeSupported(DataType dtype) {
  if (dtype.is_float4_e2m1fn() || dtype.is_bfloat16() || dtype.is_float8()) {
    return true;
  }
  if (dtype.is_float()) {
    return dtype.bits() == 8 || dtype.bits() == 16 || dtype.bits() == 32 ||
           dtype.bits() == 64;
  }
  if (dtype.is_int() || dtype.is_uint()) {
    return dtype.bits() == 8 || dtype.bits() == 16 || dtype.bits() == 32 ||
           dtype.bits() == 64;
  }
  return false;
}

bool CheckTMABoxExtent(const PrimExpr &extent, bool innermost, DataType dtype,
                       arith::Analyzer *analyzer, std::string *reason) {
  const int64_t *value = as_const_int(analyzer->Simplify(extent));
  if (value == nullptr || *value <= 0) {
    *reason = "TMA box extents must be positive compile-time integers";
    return false;
  }
  if (!innermost && *value > 256) {
    *reason = "TMA non-innermost box extents must not exceed 256 elements";
    return false;
  }
  if (innermost && dtype.bits() <= 0) {
    *reason = "TMA requires a byte-addressable source dtype";
    return false;
  }
  return true;
}

bool CheckContractTMARegions(const CopyNode &op,
                             const TransferLoweringContext &context,
                             std::string *reason) {
  arith::Analyzer local_analyzer;
  arith::Analyzer *analyzer =
      context.analyzer != nullptr ? context.analyzer : &local_analyzer;
  const size_t global_rank = op.src->shape.size();
  if (global_rank == 0 || global_rank > 5) {
    *reason = "TMA descriptors require source rank in [1, 5]";
    return false;
  }
  if (op.src_range.size() != global_rank ||
      op.dst_range.size() != op.dst->shape.size()) {
    *reason = "copy region rank must match its physical buffer rank";
    return false;
  }
  if (!IsTMADtypeSupported(op.src->dtype)) {
    *reason = "source dtype cannot be represented by a CUDA tensor map";
    return false;
  }
  if (!analyzer->CanProveEqual(op.src->elem_offset, 0)) {
    *reason = "TMA descriptor source buffers must have zero element offset";
    return false;
  }
  if (context.layout_map != nullptr && context.layout_map->count(op.src)) {
    *reason = "TMA descriptor source buffers cannot have a remapped layout";
    return false;
  }

  for (size_t axis = 0; axis < op.src->shape.size(); ++axis) {
    const int64_t *shape =
        as_const_int(analyzer->Simplify(op.src->shape[axis]));
    if (shape == nullptr || *shape <= 0) {
      *reason = "TMA descriptor source shapes must be positive compile-time "
                "integers";
      return false;
    }
    if (!CheckTMABoxExtent(op.src_range[axis]->extent,
                           axis + 1 == op.src_range.size(), op.src->dtype,
                           analyzer, reason)) {
      return false;
    }
  }
  Array<PrimExpr> global_strides = op.src->strides;
  if (global_strides.empty()) {
    PrimExpr stride = 1;
    global_strides.resize(global_rank);
    for (int axis = static_cast<int>(global_rank) - 1; axis >= 0; --axis) {
      global_strides.Set(axis, stride);
      stride *= op.src->shape[axis];
    }
  }
  if (global_strides.size() != global_rank) {
    *reason = "TMA descriptor stride rank must match source rank";
    return false;
  }
  for (size_t axis = 0; axis < global_strides.size(); ++axis) {
    PrimExpr stride_bytes = analyzer->Simplify(
        TMABytesFromElements(global_strides[axis], op.src->dtype));
    const int64_t *value = as_const_int(stride_bytes);
    if (value == nullptr || *value <= 0) {
      *reason = "TMA descriptor byte strides must be positive compile-time "
                "integers";
      return false;
    }
    if (axis + 1 < global_strides.size() &&
        (*value % 16 != 0 || *value >= (int64_t{1} << 40))) {
      *reason = "TMA descriptor outer byte strides must be 16-byte aligned "
                "and smaller than 2^40";
      return false;
    }
  }

  PrimExpr shared_offset = 0;
  PrimExpr shared_stride = 1;
  for (int axis = static_cast<int>(op.dst_range.size()) - 1; axis >= 0;
       --axis) {
    shared_offset += op.dst_range[axis]->min * shared_stride;
    shared_stride *= op.dst->shape[axis];
  }
  for (size_t axis = 0; axis < op.dst_range.size(); ++axis) {
    const int64_t *shape =
        as_const_int(analyzer->Simplify(op.dst->shape[axis]));
    if (shape == nullptr || *shape <= 0) {
      *reason = "TMA destination allocation shapes must be positive "
                "compile-time integers";
      return false;
    }
    if (!CanProveRangeInAllocation(op.dst_range[axis], op.dst->shape[axis],
                                   analyzer)) {
      *reason = "TMA destination region must be provably contained in its "
                "physical shared allocation";
      return false;
    }
  }
  if (context.layout_map == nullptr || !context.layout_map->count(op.dst)) {
    PrimExpr shared_offset_bytes =
        analyzer->Simplify(TMABytesFromElements(shared_offset, op.dst->dtype));
    if (!analyzer->CanProve(
            FloorMod(shared_offset_bytes, IntImm(DataType::Int(64), 16)) == 0,
            arith::ProofStrength::kSymbolicBound)) {
      *reason = "TMA shared destination address must be provably 16-byte "
                "aligned";
      return false;
    }
  }

  size_t src_axis = 0;
  size_t dst_axis = 0;
  while (src_axis < op.src_range.size() || dst_axis < op.dst_range.size()) {
    while (src_axis < op.src_range.size() &&
           analyzer->CanProveEqual(op.src_range[src_axis]->extent, 1)) {
      ++src_axis;
    }
    while (dst_axis < op.dst_range.size() &&
           analyzer->CanProveEqual(op.dst_range[dst_axis]->extent, 1)) {
      ++dst_axis;
    }
    if (src_axis == op.src_range.size() || dst_axis == op.dst_range.size()) {
      break;
    }
    if (!analyzer->CanProveEqual(op.src_range[src_axis]->extent,
                                 op.dst_range[dst_axis]->extent)) {
      *reason = "TMA source and destination non-unit extents must match in "
                "logical order";
      return false;
    }
    ++src_axis;
    ++dst_axis;
  }
  while (src_axis < op.src_range.size() &&
         analyzer->CanProveEqual(op.src_range[src_axis]->extent, 1)) {
    ++src_axis;
  }
  while (dst_axis < op.dst_range.size() &&
         analyzer->CanProveEqual(op.dst_range[dst_axis]->extent, 1)) {
    ++dst_axis;
  }
  if (src_axis != op.src_range.size() || dst_axis != op.dst_range.size()) {
    *reason = "TMA source and destination must have the same number of "
              "non-unit logical dimensions";
    return false;
  }
  return true;
}

bool ContractFillIsZero(const CopyNode &op, arith::Analyzer *analyzer) {
  PrimExpr fill = op.transfer_contract.value()->oob_fill;
  if (fill.dtype() != op.dst->dtype) {
    fill = Cast(op.dst->dtype, fill);
  }
  return analyzer->CanProveEqual(analyzer->Simplify(fill),
                                 make_zero(op.dst->dtype));
}

bool TensorMapOOBSatisfiesContract(const CopyNode &op,
                                   arith::Analyzer *analyzer) {
  if (!ContractFillIsZero(op, analyzer)) {
    return false;
  }

  const Array<Range> &valid_region =
      op.transfer_contract.value()->src_valid_region->region;
  ICHECK_EQ(op.src_range.size(), valid_region.size());
  ICHECK_EQ(op.src_range.size(), op.src->shape.size());
  for (size_t axis = 0; axis < op.src_range.size(); ++axis) {
    const Range &copy = op.src_range[axis];
    const Range &valid = valid_region[axis];
    PrimExpr copy_end = copy->min + copy->extent;
    PrimExpr valid_end = valid->min + valid->extent;

    // Tensor-map loads zero-fill coordinates outside [0, shape).  This is
    // sufficient when every in-bounds coordinate in the requested box is
    // also inside the contract's valid rectangle.
    bool lower_covered =
        analyzer->CanProve(valid->min <= 0,
                           arith::ProofStrength::kSymbolicBound) ||
        analyzer->CanProve(copy->min >= valid->min,
                           arith::ProofStrength::kSymbolicBound);
    bool upper_covered =
        analyzer->CanProve(valid_end >= op.src->shape[axis],
                           arith::ProofStrength::kSymbolicBound) ||
        analyzer->CanProve(copy_end <= valid_end,
                           arith::ProofStrength::kSymbolicBound);
    if (!lower_covered || !upper_covered) {
      return false;
    }
  }
  return true;
}

void AddRejectedCandidate(Array<String> *rejected, const char *candidate,
                          const std::string &reason) {
  std::ostringstream oss;
  oss << candidate << ": " << reason;
  rejected->push_back(String(oss.str()));
}

} // namespace

const char *CopyInstToString(CopyInst inst) {
  switch (inst) {
  case CopyInst::kNormal:
    return "Normal";
  case CopyInst::kLDSM:
    return "LDSM";
  case CopyInst::kSTSM:
    return "STSM";
  case CopyInst::kBulkLoad:
    return "BulkLoad";
  case CopyInst::kBulkStore:
    return "BulkStore";
  case CopyInst::kCPAsync:
    return "CPAsync";
  case CopyInst::kBulkLoad1D:
    return "BulkLoad1D";
  case CopyInst::kBulkStore1D:
    return "BulkStore1D";
  case CopyInst::kTMemLoad:
    return "TMemLoad";
  case CopyInst::kTMemStore:
    return "TMemStore";
  case CopyInst::kInvalid:
    return "Invalid";
  default:
    return "Unknown";
  }
}

bool CopyInstIsTMA(CopyInst inst) {
  return inst == CopyInst::kBulkLoad || inst == CopyInst::kBulkStore ||
         inst == CopyInst::kBulkLoad1D || inst == CopyInst::kBulkStore1D;
}

bool CopyInstIsCPAsync(CopyInst inst) { return inst == CopyInst::kCPAsync; }

namespace {

struct CopyFacts {
  bool cuda_like_target = false;
  bool has_layout_map = false;
  bool layout_dependent_tma_available = false;
  bool pass_context_disables_tma = false;
  bool explicit_tma = false;
  bool explicit_cp_async = false;
  bool no_implicit_async_commit_wait = false;
  bool disable_tma = false;
  int64_t cluster_mask = 0;
  bool can_bulk_load_1d = false;
  bool can_bulk_store_1d = false;
  bool can_bulk_load = false;
  bool can_bulk_store = false;
  bool can_bulk_load_ignore_last_dim = false;
  bool can_bulk_store_ignore_last_dim = false;
  bool can_cp_async = false;
  bool can_ldsm = false;
  bool can_stsm = false;
  bool can_tmem_load = false;
  bool can_tmem_store = false;
  std::string tma_unavailable_reason;
  std::string async_unavailable_reason;
};

bool IsCudaLikeTarget(Target target) {
  return target.defined() && (TargetIsCuda(target) || TargetIsCuTeDSL(target));
}

CopyInstSelection Supported(CopyInst inst) {
  return CopyInstSelection{inst, true, ""};
}

CopyInstSelection Unsupported(std::string reason) {
  return CopyInstSelection{CopyInst::kInvalid, false, std::move(reason)};
}

std::string MakeTmaUnavailableReason(const CopyNode &op) {
  std::ostringstream oss;
  oss << "T.tma_copy() requires TMA-capable target and global<->shared copy "
         "pattern, but TMA is not available for src="
      << op.src->name << ", dst=" << op.dst->name;
  return oss.str();
}

std::string MakeAsyncUnavailableReason(const CopyNode &op, Target target) {
  std::ostringstream oss;
  if (!target.defined()) {
    oss << "T.async_copy requires a defined target.";
  } else if (!TargetHasAsyncCopy(target)) {
    oss << "T.async_copy is only supported on targets with cp.async support "
           "(SM80+). Got target="
        << target;
  } else if (!IsGlobalBuffer(op.src) || !IsSharedBuffer(op.dst)) {
    oss << "T.async_copy only supports global->shared/shared.dyn copies. "
           "Got src="
        << op.src->name << " (scope=" << op.src.scope()
        << "), dst=" << op.dst->name << " (scope=" << op.dst.scope() << ").";
  } else if (op.src->dtype != op.dst->dtype) {
    oss << "T.async_copy requires equal byte-addressable dtypes. Got src "
           "dtype="
        << op.src->dtype << ", dst dtype=" << op.dst->dtype << ".";
  } else {
    oss << "Explicit async copy semantics require cp.async lowering, but "
           "constraints were not satisfied. Got src="
        << op.src->name << " (scope=" << op.src.scope()
        << ", dtype=" << op.src->dtype << "), dst=" << op.dst->name
        << " (scope=" << op.dst.scope() << ", dtype=" << op.dst->dtype << ").";
  }
  return oss.str();
}

bool IsAutoAsyncCopyEnabled(bool default_enabled) {
  using namespace tvm::transform;
  PassContext pass_ctx = PassContext::Current();
  return pass_ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(default_enabled))
      .value();
}

CopyInst SelectTmaInst(const CopyFacts &facts, bool allow_load,
                       bool allow_store, bool check_last_dim) {
  if (allow_load && facts.can_bulk_load_1d) {
    return CopyInst::kBulkLoad1D;
  }
  if (allow_store && facts.can_bulk_store_1d) {
    return CopyInst::kBulkStore1D;
  }
  if (allow_load && (check_last_dim ? facts.can_bulk_load
                                    : facts.can_bulk_load_ignore_last_dim)) {
    return CopyInst::kBulkLoad;
  }
  if (allow_store && (check_last_dim ? facts.can_bulk_store
                                     : facts.can_bulk_store_ignore_last_dim)) {
    return CopyInst::kBulkStore;
  }
  return CopyInst::kInvalid;
}

CopyInst SelectSyncLikeInst(const CopyFacts &facts) {
  if (facts.can_ldsm) {
    return CopyInst::kLDSM;
  }
  if (facts.can_stsm) {
    return CopyInst::kSTSM;
  }
  if (facts.can_tmem_load) {
    return CopyInst::kTMemLoad;
  }
  if (facts.can_tmem_store) {
    return CopyInst::kTMemStore;
  }
  return CopyInst::kNormal;
}

CopyFacts AnalyzeCopyFacts(const CopyNode &op, const CopyAnalysisContext &ctx) {
  CopyFacts facts;
  facts.cuda_like_target = IsCudaLikeTarget(ctx.target);
  facts.has_layout_map = ctx.layout_map != nullptr;
  facts.explicit_tma = GetIsTmaCopy(op);
  facts.explicit_cp_async = GetIsAsyncCopy(op);
  facts.no_implicit_async_commit_wait = GetNoImplicitAsyncCommitWait(op);
  facts.disable_tma = GetDisableTMA(op);
  facts.cluster_mask = GetClusterMask(op);
  facts.tma_unavailable_reason = MakeTmaUnavailableReason(op);
  facts.async_unavailable_reason = MakeAsyncUnavailableReason(op, ctx.target);
  facts.pass_context_disables_tma =
      tvm::transform::PassContext::Current()
          ->GetConfig<Bool>(kDisableTMALower, Bool(false))
          .value();

  if (!facts.cuda_like_target) {
    return facts;
  }

  arith::Analyzer local_analyzer;
  arith::Analyzer *analyzer =
      ctx.analyzer != nullptr ? ctx.analyzer : &local_analyzer;
  static const LayoutMap empty_layout_map;
  const LayoutMap &layout_map =
      ctx.layout_map != nullptr ? *ctx.layout_map : empty_layout_map;
  bool is_cutedsl = TargetIsCuTeDSL(ctx.target);
  facts.layout_dependent_tma_available =
      facts.has_layout_map && !is_cutedsl && !ctx.buffer_oob;

  if (facts.layout_dependent_tma_available) {
    facts.can_bulk_load_1d =
        CheckBulkLoad1D(op, ctx.target, layout_map, analyzer,
                        /*emit_diagnostics=*/false);
    facts.can_bulk_store_1d =
        CheckBulkStore1D(op, ctx.target, layout_map, analyzer,
                         /*emit_diagnostics=*/false);
  }

  if (facts.can_bulk_load_1d) {
    facts.can_bulk_load_ignore_last_dim = true;
    facts.can_bulk_load =
        CheckBulkLoad(op, ctx.target, analyzer, /*check_last_dim=*/true,
                      ctx.emit_diagnostics);
  } else {
    facts.can_bulk_load_ignore_last_dim =
        CheckBulkLoad(op, ctx.target, analyzer, /*check_last_dim=*/false,
                      ctx.emit_diagnostics);
    facts.can_bulk_load =
        CheckBulkLoad(op, ctx.target, analyzer, /*check_last_dim=*/true,
                      ctx.emit_diagnostics);
  }

  if (facts.can_bulk_store_1d) {
    facts.can_bulk_store_ignore_last_dim = true;
    facts.can_bulk_store =
        CheckBulkStore(op, ctx.target, analyzer, /*check_last_dim=*/true,
                       ctx.emit_diagnostics);
  } else {
    facts.can_bulk_store_ignore_last_dim =
        CheckBulkStore(op, ctx.target, analyzer, /*check_last_dim=*/false,
                       ctx.emit_diagnostics);
    facts.can_bulk_store =
        CheckBulkStore(op, ctx.target, analyzer, /*check_last_dim=*/true,
                       ctx.emit_diagnostics);
  }

  facts.can_cp_async = CheckCPAsyncCopy(op, ctx.target, layout_map, analyzer);
  facts.can_ldsm = CheckLDSMCopy(op, ctx.target);
  facts.can_stsm = CheckSTSMCopy(op, ctx.target);
  facts.can_tmem_load = CheckTMemLoad(op, ctx.target);
  facts.can_tmem_store = CheckTMemStore(op, ctx.target);
  return facts;
}

} // namespace

CopyInstSelection SelectCopyInstForLowering(const CopyNode &op,
                                            const CopyAnalysisContext &ctx) {
  if (op.transfer_contract.defined()) {
    TransferLoweringContext transfer_context;
    transfer_context.target = ctx.target;
    transfer_context.layout_map = ctx.layout_map;
    transfer_context.analyzer = ctx.analyzer;
    transfer_context.buffer_oob = ctx.buffer_oob;
    transfer_context.emit_diagnostics = ctx.emit_diagnostics;
    if (auto consumed = op.annotations.Get(kTransferPipelineSyncConsumed)) {
      if (const auto *value = consumed.value().as<IntImmNode>()) {
        transfer_context.pipeline_owns_synchronization =
            value->value == kTransferPipelineSyncManaged;
        transfer_context.force_synchronous =
            value->value == kTransferPipelineSyncFallback;
      }
    }
    TransferLoweringPlan plan = ResolveTransferLowering(op, transfer_context);
    if (!plan->supported) {
      return Unsupported(plan->selection_reason);
    }
    if (plan->implementation_id == kTransferImplCudaTMAFull ||
        plan->implementation_id == kTransferImplCudaTMATail) {
      return Supported(CopyInst::kBulkLoad);
    }
    if (plan->implementation_id == kTransferImplCudaCPAsync) {
      return Supported(CopyInst::kCPAsync);
    }
    ICHECK_EQ(plan->implementation_id, kTransferImplCommonSIMT)
        << "unknown transfer lowering implementation "
        << plan->implementation_id;
    return Supported(CopyInst::kNormal);
  }
  CopyFacts facts = AnalyzeCopyFacts(op, ctx);
  if (facts.cluster_mask != 0) {
    if (facts.can_bulk_load) {
      return Supported(CopyInst::kBulkLoad);
    }
    std::ostringstream oss;
    oss << "cluster_mask=0x" << std::hex << facts.cluster_mask
        << " requires descriptor-based TMA (kBulkLoad), but the copy does not "
           "meet TMA bulk-load constraints. src="
        << op.src->name << " (scope=" << op.src.scope()
        << "), dst=" << op.dst->name << " (scope=" << op.dst.scope() << ").";
    return Unsupported(oss.str());
  }

  if (facts.explicit_tma) {
    CopyInst inst =
        SelectTmaInst(facts, /*allow_load=*/true, /*allow_store=*/true,
                      /*check_last_dim=*/true);
    return inst == CopyInst::kInvalid
               ? Unsupported(facts.tma_unavailable_reason)
               : Supported(inst);
  }

  if (facts.explicit_cp_async || facts.no_implicit_async_commit_wait) {
    return facts.can_cp_async ? Supported(CopyInst::kCPAsync)
                              : Unsupported(facts.async_unavailable_reason);
  }

  if (!facts.disable_tma && !facts.pass_context_disables_tma) {
    CopyInst inst =
        SelectTmaInst(facts, /*allow_load=*/false, /*allow_store=*/true,
                      /*check_last_dim=*/true);
    if (inst != CopyInst::kInvalid) {
      return Supported(inst);
    }
  }

  return Supported(SelectSyncLikeInst(facts));
}

std::string ClassifyCopyForInstructionAnnotation(const CopyNode &op,
                                                 Target target,
                                                 bool in_pipeline) {
  if (op.transfer_contract.defined()) {
    const TransferContract &contract = op.transfer_contract.value();
    if (!in_pipeline ||
        contract->GetSyncOwner() != TransferSyncOwner::kPipeline) {
      return "sync";
    }
    TransferLoweringContext context;
    context.target = target;
    context.pipeline_owns_synchronization = true;
    TransferLoweringPlan plan = ResolveTransferLowering(op, context);
    if (!plan->supported) {
      return "sync";
    }
    if (plan->implementation_id == kTransferImplCudaTMAFull ||
        plan->implementation_id == kTransferImplCudaTMATail) {
      return "tma";
    }
    if (plan->implementation_id == kTransferImplCudaCPAsync) {
      return "cp_async";
    }
    return "sync";
  }
  CopyAnalysisContext ctx;
  ctx.target = target;
  CopyFacts facts = AnalyzeCopyFacts(op, ctx);
  if (!facts.cuda_like_target) {
    return "sync";
  }

  if (facts.cluster_mask != 0) {
    return facts.can_bulk_load ? "tma" : "sync";
  }

  if (facts.explicit_tma) {
    CopyInst inst =
        SelectTmaInst(facts, /*allow_load=*/true, /*allow_store=*/true,
                      /*check_last_dim=*/false);
    return CopyInstIsTMA(inst) ? "tma" : "sync";
  }

  if (facts.explicit_cp_async || facts.no_implicit_async_commit_wait) {
    return facts.can_cp_async ? "cp_async" : "sync";
  }

  if (in_pipeline && IsAutoAsyncCopyEnabled(/*default_enabled=*/false) &&
      facts.can_cp_async) {
    return "cp_async";
  }

  return "sync";
}

CopyInstSelection ClassifyWarpSpecializedProducerCopy(const CopyNode &op,
                                                      Target target) {
  if (op.transfer_contract.defined()) {
    const TransferContract &contract = op.transfer_contract.value();
    if (contract->GetSyncOwner() != TransferSyncOwner::kPipeline) {
      return Unsupported(
          "typed transfer synchronization is not owned by the pipeline");
    }
    TransferLoweringContext context;
    context.target = target;
    context.pipeline_owns_synchronization = true;
    TransferLoweringPlan plan = ResolveTransferLowering(op, context);
    if (!plan->supported) {
      return Unsupported(plan->selection_reason);
    }
    if (plan->implementation_id == kTransferImplCudaTMAFull ||
        plan->implementation_id == kTransferImplCudaTMATail) {
      return Supported(CopyInst::kBulkLoad);
    }
    if (plan->implementation_id == kTransferImplCudaCPAsync) {
      return Supported(CopyInst::kCPAsync);
    }
    return Supported(CopyInst::kNormal);
  }
  CopyAnalysisContext ctx;
  ctx.target = target;
  CopyFacts facts = AnalyzeCopyFacts(op, ctx);
  if (!facts.cuda_like_target) {
    return Supported(CopyInst::kNormal);
  }

  if (facts.cluster_mask != 0) {
    return facts.can_bulk_load ? Supported(CopyInst::kBulkLoad)
                               : Unsupported(facts.tma_unavailable_reason);
  }

  if (facts.explicit_tma) {
    CopyInst inst =
        SelectTmaInst(facts, /*allow_load=*/true, /*allow_store=*/false,
                      /*check_last_dim=*/false);
    return inst == CopyInst::kInvalid
               ? Unsupported(facts.tma_unavailable_reason)
               : Supported(inst);
  }

  if (facts.explicit_cp_async || facts.no_implicit_async_commit_wait) {
    return facts.can_cp_async ? Supported(CopyInst::kCPAsync)
                              : Unsupported(facts.async_unavailable_reason);
  }

  if (!facts.disable_tma) {
    CopyInst inst =
        SelectTmaInst(facts, /*allow_load=*/true, /*allow_store=*/false,
                      /*check_last_dim=*/true);
    if (inst != CopyInst::kInvalid) {
      return Supported(inst);
    }
  }

  return Supported(SelectSyncLikeInst(facts));
}

bool IsPipelineManagedCPAsyncCopy(const CopyNode &op, Target target) {
  if (op.transfer_contract.defined()) {
    const TransferContract &contract = op.transfer_contract.value();
    if (contract->GetSyncOwner() != TransferSyncOwner::kPipeline) {
      return false;
    }
    TransferLoweringContext context;
    context.target = target;
    context.pipeline_owns_synchronization = true;
    TransferLoweringPlan plan = ResolveTransferLowering(op, context);
    return plan->supported &&
           plan->implementation_id == kTransferImplCudaCPAsync;
  }
  CopyAnalysisContext ctx;
  ctx.target = target;
  CopyFacts facts = AnalyzeCopyFacts(op, ctx);
  if (!facts.cuda_like_target || facts.explicit_tma ||
      facts.explicit_cp_async) {
    return false;
  }
  return facts.can_cp_async;
}

TransferLoweringPlan
ResolveCudaTransferLowering(const CopyNode &op,
                            const TransferLoweringContext &context) {
  ICHECK(op.transfer_contract.defined());
  arith::Analyzer local_analyzer;
  arith::Analyzer *analyzer =
      context.analyzer != nullptr ? context.analyzer : &local_analyzer;
  const TransferContract &contract = op.transfer_contract.value();
  bool requires_contract_fill = TransferRequiresPostFill(op, analyzer);
  Array<String> rejected;

  TransferSyncOwner sync_owner = contract->GetSyncOwner();
  if (sync_owner == TransferSyncOwner::kCaller) {
    return MakeTransferLoweringPlan(
        kTransferImplCommonSIMT, /*supported=*/false,
        /*asynchronous=*/false, /*uses_tma_descriptor=*/false,
        /*requires_post_fill=*/false,
        "caller synchronization ownership has no common lowering consumer");
  }
  if (sync_owner == TransferSyncOwner::kPipeline &&
      !context.pipeline_owns_synchronization && !context.force_synchronous) {
    return MakeTransferLoweringPlan(
        kTransferImplCommonSIMT, /*supported=*/false,
        /*asynchronous=*/false, /*uses_tma_descriptor=*/false,
        /*requires_post_fill=*/false,
        "pipeline synchronization ownership was not consumed by a pipeline "
        "lowering pass");
  }
  if (context.force_synchronous) {
    ICHECK(sync_owner == TransferSyncOwner::kPipeline)
        << "only a pipeline-owned transfer can use the compiler's "
           "synchronous fallback";
    AddRejectedCandidate(&rejected, kTransferImplCudaTMAFull,
                         "pipeline planning selected synchronous fallback");
    AddRejectedCandidate(&rejected, kTransferImplCudaTMATail,
                         "pipeline planning selected synchronous fallback");
    AddRejectedCandidate(&rejected, kTransferImplCudaCPAsync,
                         "pipeline planning selected synchronous fallback");
    return MakeTransferLoweringPlan(
        kTransferImplCommonSIMT, /*supported=*/true,
        /*asynchronous=*/false, /*uses_tma_descriptor=*/false,
        /*requires_post_fill=*/false,
        "pipeline planning selected synchronous transfer fallback", rejected);
  }
  if (!contract->allow_async) {
    AddRejectedCandidate(&rejected, kTransferImplCudaTMAFull,
                         "transfer contract disallows asynchronous execution");
    AddRejectedCandidate(&rejected, kTransferImplCudaTMATail,
                         "transfer contract disallows asynchronous execution");
    AddRejectedCandidate(&rejected, kTransferImplCudaCPAsync,
                         "transfer contract disallows asynchronous execution");
    return MakeTransferLoweringPlan(
        kTransferImplCommonSIMT, /*supported=*/true,
        /*asynchronous=*/false, /*uses_tma_descriptor=*/false,
        /*requires_post_fill=*/false,
        "transfer contract requires synchronous execution", rejected);
  }

  CopyAnalysisContext copy_context;
  copy_context.target = context.target;
  copy_context.layout_map = context.layout_map;
  copy_context.analyzer = analyzer;
  copy_context.buffer_oob = context.buffer_oob;
  copy_context.emit_diagnostics = context.emit_diagnostics;
  CopyFacts facts = AnalyzeCopyFacts(op, copy_context);

  std::string tma_region_reason;
  bool tma_regions_legal =
      CheckContractTMARegions(op, context, &tma_region_reason);
  bool tma_enabled = !facts.disable_tma && !facts.pass_context_disables_tma;
  bool tma_legal = tma_enabled && facts.can_bulk_load && tma_regions_legal;
  if (!requires_contract_fill && tma_legal) {
    return MakeTransferLoweringPlan(
        kTransferImplCudaTMAFull, /*supported=*/true,
        /*asynchronous=*/true, /*uses_tma_descriptor=*/true,
        /*requires_post_fill=*/false,
        "full logical source region is valid for descriptor-based TMA load",
        rejected);
  }
  if (requires_contract_fill) {
    AddRejectedCandidate(&rejected, kTransferImplCudaTMAFull,
                         "logical source region is not provably fully valid");
  } else if (!tma_enabled) {
    AddRejectedCandidate(&rejected, kTransferImplCudaTMAFull,
                         "TMA lowering is disabled by compile policy");
  } else if (!facts.can_bulk_load) {
    AddRejectedCandidate(&rejected, kTransferImplCudaTMAFull,
                         facts.tma_unavailable_reason);
  } else if (!tma_regions_legal) {
    AddRejectedCandidate(&rejected, kTransferImplCudaTMAFull,
                         tma_region_reason);
  }

  if (requires_contract_fill && tma_legal) {
    bool requires_post_fill = !TensorMapOOBSatisfiesContract(op, analyzer);
    return MakeTransferLoweringPlan(
        kTransferImplCudaTMATail, /*supported=*/true,
        /*asynchronous=*/true, /*uses_tma_descriptor=*/true, requires_post_fill,
        requires_post_fill
            ? "tail/OOB TMA load followed by cooperative contract post-fill"
            : "tail/OOB TMA load whose tensor-bound zero-fill satisfies the "
              "contract",
        rejected);
  }
  if (!requires_contract_fill) {
    AddRejectedCandidate(&rejected, kTransferImplCudaTMATail,
                         "tail repair is unnecessary for a full valid tile");
  } else if (!tma_enabled) {
    AddRejectedCandidate(&rejected, kTransferImplCudaTMATail,
                         "TMA lowering is disabled by compile policy");
  } else if (!facts.can_bulk_load) {
    AddRejectedCandidate(&rejected, kTransferImplCudaTMATail,
                         facts.tma_unavailable_reason);
  } else if (!tma_regions_legal) {
    AddRejectedCandidate(&rejected, kTransferImplCudaTMATail,
                         tma_region_reason);
  }

  if (facts.can_cp_async &&
      (!requires_contract_fill || ContractFillIsZero(op, analyzer))) {
    return MakeTransferLoweringPlan(
        kTransferImplCudaCPAsync, /*supported=*/true,
        /*asynchronous=*/true, /*uses_tma_descriptor=*/false,
        /*requires_post_fill=*/false,
        requires_contract_fill
            ? "predicated zero-fill cp.async is legal for the transfer contract"
            : "full-tile cp.async is legal for the transfer contract",
        rejected);
  }
  AddRejectedCandidate(
      &rejected, kTransferImplCudaCPAsync,
      !facts.can_cp_async
          ? facts.async_unavailable_reason
          : "cp.async predication only supports a zero OOB fill value");
  return MakeTransferLoweringPlan(
      kTransferImplCommonSIMT, /*supported=*/true,
      /*asynchronous=*/false, /*uses_tma_descriptor=*/false,
      /*requires_post_fill=*/false,
      "no accelerated CUDA transfer candidate is legal", rejected);
}

} // namespace cuda
} // namespace tl
} // namespace tvm
