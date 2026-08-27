/*!
 * \file tl/backend/cuda/op/gemm.cc
 * \brief CUDA implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"

#include "op/builtin.h"
#include "op/tcgen5_meta.h"
#include "op/utils.h"
#include "target/utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace tvm {
namespace tl {

using namespace tir;

namespace cuda {

namespace {

constexpr const char *kCudaMMA = "cuda.mma";
constexpr const char *kCudaWGMMA = "cuda.wgmma";
constexpr const char *kCudaTCGEN05 = "cuda.tcgen05";

bool CheckWgmma(const GemmNode &op) {
  if (op.b_.scope() != "shared.dyn" && op.b_.scope() != "shared") {
    return false;
  }

  if (op.c_->dtype == DataType::Float(16)) {
    if (op.a_->dtype == DataType::Float(16) &&
        op.b_->dtype == DataType::Float(16))
      return op.k_ % 16 == 0;
    if (op.a_->dtype.is_float8() && op.b_->dtype.is_float8())
      return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;
    return false;
  }
  if (op.c_->dtype == DataType::Float(32)) {
    if (op.a_->dtype == DataType::Float(16) &&
        op.b_->dtype == DataType::Float(16))
      return op.k_ % 16 == 0;
    if (op.a_->dtype == DataType::BFloat(16) &&
        op.b_->dtype == DataType::BFloat(16))
      return op.k_ % 16 == 0;
    if (op.a_->dtype == DataType::Float(32) &&
        op.b_->dtype == DataType::Float(32))
      return (!op.transA_) && op.transB_ && op.k_ % 8 == 0;
    if (op.a_->dtype.is_float8() && op.b_->dtype.is_float8())
      return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;
    return false;
  }
  if (op.c_->dtype == DataType::Int(32)) {
    if (op.a_->dtype == DataType::Int(8) && op.b_->dtype == DataType::Int(8))
      return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;
    if (op.a_->dtype == DataType::Int(8) && op.b_->dtype == DataType::UInt(8))
      return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;
    if (op.a_->dtype == DataType::UInt(8) && op.b_->dtype == DataType::Int(8))
      return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;
    if (op.a_->dtype == DataType::UInt(8) && op.b_->dtype == DataType::UInt(8))
      return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;
    return false;
  }
  return false;
}

bool AllowTcgen5Mma(const GemmNode &op, Target target) {
  bool scope_ok = (IsSharedBuffer(op.a_) || op.a_.scope() == "shared.tmem") &&
                  IsSharedBuffer(op.b_) && op.c_.scope() == "shared.tmem";
  if (!TargetIsSm100(target) || !scope_ok)
    return false;
  DataType ab_dtype =
      (op.a_.scope() == "shared.tmem") ? op.b_->dtype : op.a_->dtype;
  return GetTCGEN5MMAMeta(op.m_, op.n_, op.k_, ab_dtype, op.c_->dtype).first;
}

bool AllowWgmma(const GemmNode &op, int block_size, Target target) {
  tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();

  int warp_size = TargetGetWarpSize(target);
  int num_warps = block_size / warp_size;
  return !ctxt->GetConfig(kDisableWGMMA, Optional<Bool>()).value_or(false) &&
         TargetIsHopper(target) && op.m_ >= 64 && num_warps % 4 == 0 &&
         CheckWgmma(op);
}

bool AllowWgmmaPhysicalShape(const GemmNode &op, int physical_m, int block_size,
                             Target target) {
  tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
  int warp_size = TargetGetWarpSize(target);
  int num_warps = block_size / warp_size;
  return !ctxt->GetConfig(kDisableWGMMA, Optional<Bool>()).value_or(false) &&
         TargetIsHopper(target) && physical_m >= 64 && num_warps % 4 == 0 &&
         CheckWgmma(op);
}

Array<Integer> GemmShape(int m, int n, int k) {
  return {Integer(m), Integer(n), Integer(k)};
}

bool IsSharedScope(const String &scope) {
  return scope == "shared" || scope == "shared.dyn";
}

int64_t MatrixBytes(int rows, int columns, DataType dtype) {
  return static_cast<int64_t>(rows) * columns * dtype.bits() * dtype.lanes() /
         8;
}

bool HasAllocationFor(const Buffer &buffer,
                      const GemmLoweringContext &context) {
  return context.allocated_buffers.count(buffer->data) != 0;
}

struct BufferExpansion {
  bool legal{false};
  int64_t additional_bytes{0};
};

BufferExpansion EstimateBufferExpansion(const Buffer &buffer,
                                        const BufferRegion &region,
                                        int matrix_axis, int physical_extent,
                                        const GemmLoweringContext &context) {
  if (!HasAllocationFor(buffer, context) || matrix_axis < 0 ||
      matrix_axis >= static_cast<int>(buffer->shape.size()) ||
      !buffer->strides.empty()) {
    return {};
  }
  Array<PrimExpr> shape = buffer->shape;
  if (auto planned = context.planned_buffer_shapes.Get(buffer->data)) {
    shape = planned.value();
  }
  if (shape.size() != buffer->shape.size() ||
      region->region.size() != buffer->shape.size()) {
    return {};
  }
  const int64_t *region_min = as_const_int(region->region[matrix_axis]->min);
  if (region_min == nullptr) {
    return {};
  }
  int64_t required_extent = *region_min + physical_extent;
  int64_t old_numel = 1;
  int64_t new_numel = 1;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    const int64_t *extent = as_const_int(shape[axis]);
    if (extent == nullptr || *extent <= 0) {
      return {};
    }
    old_numel *= *extent;
    new_numel *= axis == static_cast<size_t>(matrix_axis)
                     ? std::max(*extent, required_extent)
                     : *extent;
  }
  return {true, (new_numel - old_numel) * buffer->dtype.bits() *
                    buffer->dtype.lanes() / 8};
}

bool MmaShapeSupported(const GemmNode &op) {
  int m = op.logical_m();
  int n = op.logical_n();
  int k = op.logical_k();
  int input_bits = op.a_->dtype.bits();
  if (input_bits <= 0) {
    return false;
  }
  int micro_k = std::min(256 / input_bits, k);
  return m >= 16 && m % 16 == 0 && n >= 8 && n % 8 == 0 && k >= micro_k &&
         k % micro_k == 0 && op.c_.scope() == "local.fragment";
}

GemmLoweringPlan ResolveCudaGemmLowering(const GemmNode &op,
                                         const GemmLoweringContext &context) {
  Array<String> rejected;
  const int logical_m = op.logical_m();
  const int logical_n = op.logical_n();
  const int logical_k = op.logical_k();
  Array<Integer> logical_shape = GemmShape(logical_m, logical_n, logical_k);
  const bool explicit_wgmma = op.isWgmma_;
  const bool explicit_tcgen05 = op.isTcgen05_;
  tvm::transform::PassContext pass_context =
      tvm::transform::PassContext::Current();
  const bool logical_padding_disabled =
      pass_context->GetConfig(kDisableLogicalGemmPadding, Optional<Bool>())
          .value_or(false);

  if (explicit_tcgen05 ||
      (!explicit_wgmma && AllowTcgen5Mma(op, context.target))) {
    bool exact_logical_shape =
        logical_m == op.m_ && logical_n == op.n_ && logical_k == op.k_;
    if (AllowTcgen5Mma(op, context.target) && exact_logical_shape) {
      return MakeGemmLoweringPlan(kGemmImplCudaTCGen05, true, true,
                                  logical_shape, GemmShape(op.m_, op.n_, op.k_),
                                  false, false, {}, 0, 0, 0,
                                  "TCGEN05 operand, accumulator, target, and "
                                  "shape constraints are legal",
                                  rejected);
    }
    rejected.push_back(exact_logical_shape
                           ? "cuda.tcgen05.sync: target or operand contract is "
                             "not TCGEN05 legal"
                           : "cuda.tcgen05.sync: logical-shape materialization "
                             "is not implemented for TCGEN05");
    if (explicit_tcgen05) {
      return MakeGemmLoweringPlan(
          kGemmImplCudaTCGen05, false, true, logical_shape, logical_shape,
          false, false, {}, 0, 0, 0,
          "T.tcgen05_gemm() requires Blackwell TCGEN5MMA lowering; explicit "
          "request has no legal implementation",
          rejected);
    }
  }

  // Hopper WGMMA has a fixed 64-row instruction footprint. The logical
  // contract may request fewer rows; physical storage remains an ISA fact.
  constexpr int kWgmmaPhysicalM = 64;
  const int physical_m = std::max(op.m_, kWgmmaPhysicalM);
  bool wgmma_base_legal = AllowWgmmaPhysicalShape(
      op, physical_m, context.block_size, context.target);
  if (wgmma_base_legal && logical_m <= physical_m) {
    bool requires_padding = logical_m != physical_m;
    bool neutral_padding =
        !requires_padding || (op.gemm_contract.defined() &&
                              is_zero(op.gemm_contract.value()->padding_value));
    bool padding_allowed =
        !requires_padding ||
        (!logical_padding_disabled && op.gemm_contract.defined() &&
         op.gemm_contract.value()->allow_padding);
    // A logical-M shared operand can be loaded into the register-source WGMMA
    // fragment one K atom at a time.  This preserves the m64 ISA footprint
    // without materializing neutral rows in shared memory.  It is both the
    // short-lifetime choice when the complete K extent fits the register ring
    // and the resource-saving fallback when SS materialization is illegal.
    // Explicit T.wgmma_gemm remains on the asynchronous SS contract because
    // the streamed path owns its internal commit/wait pipeline and is
    // synchronous.
    bool shared_a_rs_legal =
        requires_padding && !explicit_wgmma && !op.transA_ &&
        IsSharedScope(op.a_.scope()) && IsSharedScope(op.b_.scope()) &&
        op.a_->dtype.bits() == 16 && op.b_->dtype.bits() == 16;
    // The RS implementation issues two K atoms per commit group and keeps two
    // groups in a four-atom register ring. Prefer it while the complete K
    // extent fits that ring: no source registers need to be recycled behind
    // an in-flight WGMMA group. Beyond that point use SS whenever the common
    // resource planner proves that materialization fits. This is derived from
    // the WGMMA issue/lifetime contract, not an operator or workload table.
    constexpr int kSharedARsCommitKAtoms = 2;
    constexpr int kSharedARsInFlightCommitGroups = 2;
    constexpr int kSharedARsRegisterRingKAtoms =
        kSharedARsCommitKAtoms * kSharedARsInFlightCommitGroups;
    const int shared_a_k_atom = std::max(1, 256 / op.a_->dtype.bits());
    const int shared_a_k_atoms =
        (logical_k + shared_a_k_atom - 1) / shared_a_k_atom;
    const bool shared_a_rs_fits_register_ring =
        shared_a_rs_legal && shared_a_k_atoms <= kSharedARsRegisterRingKAtoms;

    const int c_m_axis = static_cast<int>(op.c_->shape.size()) - 2;
    BufferExpansion streamed_c_expansion =
        shared_a_rs_legal
            ? EstimateBufferExpansion(op.c_, op.cRegion_, c_m_axis, physical_m,
                                      context)
            : BufferExpansion{};
    bool shared_a_rs_available = false;
    Array<GemmTemporaryRequirement> shared_a_rs_temporaries;
    int64_t shared_a_rs_additional_shared = 0;
    int64_t shared_a_rs_additional_fragment = 0;
    if (shared_a_rs_legal && padding_allowed && neutral_padding &&
        streamed_c_expansion.legal) {
      int64_t additional_c_bytes = streamed_c_expansion.additional_bytes;
      int64_t additional_shared =
          IsSharedScope(op.c_.scope()) ? additional_c_bytes : 0;
      int64_t additional_fragment =
          op.c_.scope() == "local.fragment" ? additional_c_bytes : 0;
      bool shared_upper_bound_is_conclusive =
          context.max_shared_memory_bytes < 0 ||
          context.current_shared_memory_bytes <=
              context.max_shared_memory_bytes;
      bool resource_legal =
          context.max_shared_memory_bytes < 0 ||
          (additional_shared <= context.max_shared_memory_bytes &&
           (!shared_upper_bound_is_conclusive ||
            context.current_shared_memory_bytes + additional_shared <=
                context.max_shared_memory_bytes));
      if (resource_legal) {
        shared_a_rs_temporaries = {MakeGemmTemporaryRequirement(
            "C", op.c_.scope(), {Integer(logical_m), Integer(logical_n)},
            {Integer(physical_m), Integer(op.n_)}, make_zero(op.c_->dtype),
            /*initialization_required=*/false, "before_first_gemm",
            "after_last_logical_output_consumer",
            MatrixBytes(physical_m, op.n_, op.c_->dtype), additional_c_bytes)};
        shared_a_rs_available = true;
        shared_a_rs_additional_shared = additional_shared;
        shared_a_rs_additional_fragment = additional_fragment;
      } else {
        rejected.push_back(
            "cuda.wgmma.rs.shared_a: C materialization exceeds the target "
            "resource limit");
      }
    } else if (shared_a_rs_legal && !padding_allowed) {
      rejected.push_back(
          "cuda.wgmma.rs.shared_a: logical shape requires padding but the "
          "contract disallows it");
    } else if (shared_a_rs_legal && !neutral_padding) {
      rejected.push_back(
          "cuda.wgmma.rs.shared_a: register padding requires the zero neutral "
          "value");
    } else if (shared_a_rs_legal && !streamed_c_expansion.legal) {
      rejected.push_back(
          "cuda.wgmma.rs.shared_a: physical C requires allocation-backed "
          "storage");
    }

    // Legacy SS materialization includes neutral padding initialization even
    // when the user allocation already has the physical instruction shape.
    bool requires_materialization = requires_padding;
    const int a_m_axis =
        static_cast<int>(op.a_->shape.size()) - (op.transA_ ? 1 : 2);
    BufferExpansion a_expansion =
        requires_materialization
            ? EstimateBufferExpansion(op.a_, op.aRegion_, a_m_axis, physical_m,
                                      context)
            : BufferExpansion{true, 0};
    BufferExpansion c_expansion =
        requires_materialization
            ? EstimateBufferExpansion(op.c_, op.cRegion_, c_m_axis, physical_m,
                                      context)
            : BufferExpansion{true, 0};
    bool can_materialize = a_expansion.legal && c_expansion.legal;
    int64_t additional_a_bytes = a_expansion.additional_bytes;
    int64_t additional_c_bytes = c_expansion.additional_bytes;
    int64_t additional_shared =
        IsSharedScope(op.a_.scope()) ? additional_a_bytes : 0;
    int64_t additional_fragment =
        op.a_.scope() == "local.fragment" ? additional_a_bytes : 0;
    if (op.c_.scope() == "local.fragment") {
      additional_fragment += additional_c_bytes;
    } else if (IsSharedScope(op.c_.scope())) {
      additional_shared += additional_c_bytes;
    }
    // Before storage planning, summing allocation sizes is only an upper bound:
    // buffers with disjoint lifetimes may later alias.  Use that bound when it
    // is conclusive, but defer an already-over-limit base allocation to the
    // common shared-memory planner.  The padding delta itself remains a hard
    // lower bound and can always be rejected against the target limit.
    bool shared_upper_bound_is_conclusive =
        context.max_shared_memory_bytes < 0 ||
        context.current_shared_memory_bytes <= context.max_shared_memory_bytes;
    bool resource_legal =
        context.max_shared_memory_bytes < 0 ||
        (additional_shared <= context.max_shared_memory_bytes &&
         (!shared_upper_bound_is_conclusive ||
          context.current_shared_memory_bytes + additional_shared <=
              context.max_shared_memory_bytes));
    bool resource_check_deferred = context.max_shared_memory_bytes >= 0 &&
                                   !shared_upper_bound_is_conclusive;

    // A short RS ring is useful when avoiding SS padding preserves another
    // shared-memory residency tier.  If both choices already occupy the same
    // tier, prefer SS: it leaves operand movement to the asynchronous WGMMA
    // path instead of adding register loads on the critical path.  This is a
    // resource/lifetime decision; it does not depend on an operator or shape
    // table.  Keep the conservative RS preference when the target budget is
    // unavailable.
    bool shared_a_rs_preserves_residency =
        shared_a_rs_fits_register_ring && !resource_check_deferred;
    if (shared_a_rs_fits_register_ring && context.max_shared_memory_bytes > 0 &&
        context.current_shared_memory_bytes >= 0 &&
        shared_upper_bound_is_conclusive) {
      const int64_t before_bytes =
          std::max<int64_t>(1, context.current_shared_memory_bytes);
      const int64_t after_bytes = std::max<int64_t>(
          1, context.current_shared_memory_bytes + additional_shared);
      const int64_t before_residency =
          context.max_shared_memory_bytes / before_bytes;
      const int64_t after_residency =
          context.max_shared_memory_bytes / after_bytes;
      shared_a_rs_preserves_residency = after_residency < before_residency;
    }

    if (padding_allowed && neutral_padding && can_materialize &&
        resource_legal && !shared_a_rs_preserves_residency) {
      Array<GemmTemporaryRequirement> temporaries;
      if (requires_padding) {
        temporaries.push_back(MakeGemmTemporaryRequirement(
            "A", op.a_.scope(), {Integer(logical_m), Integer(logical_k)},
            {Integer(physical_m), Integer(op.k_)}, make_zero(op.a_->dtype),
            /*initialization_required=*/true, "before_first_gemm",
            "after_last_gemm_consumer",
            MatrixBytes(physical_m, op.k_, op.a_->dtype), additional_a_bytes));
        temporaries.push_back(MakeGemmTemporaryRequirement(
            "C", op.c_.scope(), {Integer(logical_m), Integer(logical_n)},
            {Integer(physical_m), Integer(op.n_)}, make_zero(op.c_->dtype),
            /*initialization_required=*/false, "before_first_gemm",
            "after_last_logical_output_consumer",
            MatrixBytes(physical_m, op.n_, op.c_->dtype), additional_c_bytes));
      }
      return MakeGemmLoweringPlan(
          kGemmImplCudaWGMMA, true, false, logical_shape,
          GemmShape(physical_m, op.n_, op.k_), requires_padding,
          requires_materialization, temporaries, additional_shared,
          additional_fragment, additional_shared + additional_fragment,
          resource_check_deferred
              ? "WGMMA selected; the raw shared-allocation upper bound is "
                "inconclusive and final peak validation is deferred to the "
                "common shared-memory planner"
          : requires_padding
              ? "WGMMA selected with compiler-owned neutral logical-M padding"
              : "WGMMA physical shape is directly legal",
          rejected);
    }
    if (!padding_allowed) {
      rejected.push_back("cuda.wgmma.async: logical shape requires padding but "
                         "the contract disallows it");
    } else if (!neutral_padding) {
      rejected.push_back("cuda.wgmma.async: GEMM padding must use the "
                         "additive/multiplicative zero neutral value");
    } else if (!can_materialize) {
      rejected.push_back("cuda.wgmma.async: physical padding requires "
                         "allocation-backed A and C buffers");
    } else if (!resource_legal) {
      std::ostringstream os;
      os << "cuda.wgmma.async: shared-memory requirement "
         << (context.current_shared_memory_bytes + additional_shared)
         << " exceeds target limit " << context.max_shared_memory_bytes;
      rejected.push_back(os.str());
    }
    if (shared_a_rs_available) {
      return MakeGemmLoweringPlan(
          kGemmImplCudaWGMMASharedARS, true, true, logical_shape,
          GemmShape(physical_m, op.n_, op.k_), true, true,
          shared_a_rs_temporaries, shared_a_rs_additional_shared,
          shared_a_rs_additional_fragment,
          shared_a_rs_additional_shared + shared_a_rs_additional_fragment,
          shared_a_rs_fits_register_ring
              ? "WGMMA selected with logical-M shared A streamed through a "
                "bounded register-source ring"
              : "WGMMA selected with logical-M shared A streamed through a "
                "bounded register-source ring because SS materialization "
                "exceeds the available resource budget",
          rejected);
    }
  } else {
    rejected.push_back("cuda.wgmma.async: target, thread, dtype, scope, or K "
                       "constraints are not legal");
  }

  if (explicit_wgmma) {
    return MakeGemmLoweringPlan(
        kGemmImplCudaWGMMA, false, false, logical_shape, logical_shape, false,
        false, {}, 0, 0, 0,
        "T.wgmma_gemm() requires Hopper WGMMA lowering; explicit request has "
        "no legal implementation",
        rejected);
  }

  if (TargetHasSMVersionGE(context.target, 70) && MmaShapeSupported(op)) {
    return MakeGemmLoweringPlan(
        kGemmImplCudaMMA, true, true, logical_shape, logical_shape, false,
        op.m_ != logical_m, {}, 0, 0, 0,
        "MMA selected as the first legal unpadded implementation", rejected);
  }
  rejected.push_back("cuda.mma.sync: logical shape, scope, dtype, or target "
                     "constraints are not legal");

  return MakeGemmLoweringPlan(
      kGemmImplCudaScalar, true, true, logical_shape, logical_shape, false,
      op.m_ != logical_m, {}, 0, 0, 0,
      "scalar GEMM selected as the portable CUDA fallback", rejected);
}

void FatalWgmmaUnavailable(const GemmNode &op, Target target) {
  LOG(FATAL) << "T.wgmma_gemm() requires Hopper WGMMA lowering, but "
                "constraints were not satisfied. Got target="
             << target << ", A(scope=" << op.a_.scope()
             << ", dtype=" << op.a_->dtype << "), B(scope=" << op.b_.scope()
             << ", dtype=" << op.b_->dtype << "), C(scope=" << op.c_.scope()
             << ", dtype=" << op.c_->dtype << "), M=" << op.m_
             << ", N=" << op.n_ << ", K=" << op.k_ << ".";
}

void FatalTcgen5Unavailable(const GemmNode &op, Target target) {
  LOG(FATAL) << "T.tcgen05_gemm() requires Blackwell TCGEN5MMA lowering, "
                "but constraints were not satisfied. Got target="
             << target << ", A(scope=" << op.a_.scope()
             << ", dtype=" << op.a_->dtype << "), B(scope=" << op.b_.scope()
             << ", dtype=" << op.b_->dtype << "), C(scope=" << op.c_.scope()
             << ", dtype=" << op.c_->dtype << "), M=" << op.m_
             << ", N=" << op.n_ << ", K=" << op.k_ << ".";
}

std::pair<int, int>
ComputeDefaultWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                            int num_warps, int k_n_per_warp) {
  int m_warp = 1, n_warp = 1;
  constexpr int kMPerWarp = 16;

  ICHECK(M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  ICHECK(N % k_n_per_warp == 0)
      << "N must be divisible by " << k_n_per_warp << ", but got " << N;

  if (policy.isFullRow()) {
    m_warp = num_warps;
    n_warp = 1;
    if (M % (m_warp * kMPerWarp) != 0) {
      int max_m_warps = M / kMPerWarp;
      m_warp = max_m_warps;
      n_warp = num_warps / m_warp;
      if (n_warp == 0)
        n_warp = 1;
    }
  } else if (policy.isFullCol()) {
    m_warp = 1;
    n_warp = num_warps;
    if (N % (n_warp * k_n_per_warp) != 0) {
      int max_n_warps = N / k_n_per_warp;
      n_warp = max_n_warps;
      m_warp = num_warps / n_warp;
      if (m_warp == 0)
        m_warp = 1;
    }
  } else if (policy.isSquare()) {
    int max_m_warps = M / kMPerWarp;
    float ideal_ratio = N > 0 ? static_cast<float>(M) / N : 1.0f;

    int best_m = 1;
    int best_n = 1;
    float best_balance = std::numeric_limits<float>::max();
    for (int m = 1; m <= max_m_warps && m <= num_warps; m++) {
      int n = num_warps / m;

      float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(N) / (n * k_n_per_warp);
      if (m_per_warp < 1 || n_per_warp < 1)
        continue;
      if (m * n != num_warps)
        continue;

      float balance = std::abs(m_per_warp / n_per_warp - ideal_ratio);
      if (balance < best_balance) {
        best_balance = balance;
        best_m = m;
        best_n = n;
      }
    }

    m_warp = best_m;
    n_warp = best_n;
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }

  ICHECK(m_warp * n_warp == num_warps)
      << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", num_warps: " << num_warps;
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

std::pair<int, int> ComputeWgmmaWarpPartition(const GemmWarpPolicyNode &policy,
                                              int M, int N, int num_warps) {
  ICHECK(num_warps % 4 == 0) << "Warp-Group MMA requires 128*k threads.";

  int m_warp = 1, n_warp = 1;
  constexpr int kMPerWarp = 16;
  constexpr int kNPerWarp = 8;
  constexpr int kGroup = 4;

  ICHECK(M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  ICHECK(N % kNPerWarp == 0)
      << "N must be divisible by " << kNPerWarp << ", but got " << N;

  m_warp = kGroup;
  n_warp = num_warps / m_warp;

  if (policy.isFullRow()) {
    for (int cand = num_warps; cand >= kGroup; cand -= kGroup) {
      if (M % (cand * kMPerWarp) == 0) {
        m_warp = cand;
        n_warp = num_warps / m_warp;
        break;
      }
    }
  } else if (policy.isFullCol()) {
    int cand_n = n_warp;
    if (N % (cand_n * kNPerWarp) != 0) {
      int max_n = N / kNPerWarp;
      for (int n = std::min(cand_n, max_n); n >= 1; --n) {
        if (num_warps % n == 0 && (num_warps / n) % kGroup == 0) {
          n_warp = n;
          m_warp = num_warps / n_warp;
          break;
        }
      }
    }
  } else if (policy.isSquare()) {
    int max_m = M / kMPerWarp;
    int max_n = N / kNPerWarp;

    float ideal = N > 0 ? static_cast<float>(M) / N : 1.f;
    float best_score = std::numeric_limits<float>::max();
    int best_m = kGroup, best_n = n_warp;

    for (int m = kGroup; m <= num_warps && m <= max_m; m += kGroup) {
      if (num_warps % m)
        continue;
      int n = num_warps / m;
      if (n > max_n)
        continue;

      float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(N) / (n * kNPerWarp);
      float score = std::abs(m_per_warp / n_per_warp - ideal);

      if (score < best_score) {
        best_score = score;
        best_m = m;
        best_n = n;
      }
    }
    m_warp = best_m;
    n_warp = best_n;
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }

  ICHECK(m_warp * n_warp == num_warps)
      << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", num_warps: " << num_warps;
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

} // namespace

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    if (op.isWgmma_) {
      if (!AllowWgmma(op, block_size, target)) {
        FatalWgmmaUnavailable(op, target);
      }
      return kCudaWGMMA;
    }
    if (op.isTcgen05_) {
      if (!AllowTcgen5Mma(op, target)) {
        FatalTcgen5Unavailable(op, target);
      }
      return kCudaTCGEN05;
    }

    if (AllowTcgen5Mma(op, target)) {
      return kCudaTCGEN05;
    }
    if (AllowWgmma(op, block_size, target)) {
      return kCudaWGMMA;
    }
    return kCudaMMA;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    int num_warps = block_size / TargetGetWarpSize(target);
    if (gemm_inst == kCudaTCGEN05) {
      policy.m_warp = 1;
      policy.n_warp = num_warps;
      return {1, num_warps};
    }
    if (gemm_inst == kCudaWGMMA) {
      return ComputeWgmmaWarpPartition(policy, M, N, num_warps);
    }
    int k_n_per_warp =
        (TargetIsVolta(target) || TargetIsTuring(target)) ? 16 : 8;
    return ComputeDefaultWarpPartition(policy, M, N, num_warps, k_n_per_warp);
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    return gemm_inst == kCudaMMA;
  }

  static String InstructionKind(String gemm_inst) {
    if (gemm_inst == kCudaWGMMA) {
      return "wgmma";
    }
    if (gemm_inst == kCudaTCGEN05) {
      return "tcgen5mma";
    }
    if (gemm_inst == kCudaMMA) {
      return "mma";
    }
    return "unknown";
  }
};

} // namespace cuda

namespace {

bool MatchCudaGemmTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaGemm() {
  RegisterGemmImpl(GemmImpl{
      "cuda.Gemm",
      MatchCudaGemmTarget,
      cuda::Gemm::SelectInst,
      cuda::Gemm::ComputeWarpPartition,
      cuda::Gemm::ReuseExistingSharedLayout,
      cuda::Gemm::InstructionKind,
      cuda::ResolveCudaGemmLowering,
  });
  return true;
}

const bool cuda_gemm_registered = RegisterCudaGemm();

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tl.get_tcgen5_mma_meta", [](int M, int N, int K, DataType ab_dtype,
                                   DataType c_dtype, bool disable_2cta) {
        auto [success, meta] =
            GetTCGEN5MMAMeta(M, N, K, ab_dtype, c_dtype, disable_2cta);
        Array<Integer> result;
        if (success) {
          result.push_back(Integer(meta.atom_m));
          result.push_back(Integer(meta.atom_n));
          result.push_back(Integer(meta.atom_k));
          result.push_back(Integer(meta.enable_ws));
          result.push_back(Integer(meta.enable_2cta));
        }
        return result;
      });
  refl::GlobalDef().def(
      "tl.get_tcgen5_instr_desc",
      [](int atom_m, int atom_n, int atom_k, DataType ab_dtype,
         DataType c_dtype, bool a_is_k_major, bool b_is_k_major, int scale_in_a,
         int scale_in_b) {
        uint32_t desc = GetTCGEN5InstrDesc(atom_m, atom_n, atom_k, ab_dtype,
                                           c_dtype, a_is_k_major, b_is_k_major,
                                           scale_in_a, scale_in_b);
        return Integer(static_cast<int64_t>(desc));
      });
  refl::GlobalDef().def("tl.get_tcgen5_blockscaled_instr_desc",
                        [](int atom_m, int atom_n, DataType ab_dtype,
                           bool a_is_k_major, bool b_is_k_major, int scale_in_a,
                           int scale_in_b, int a_sf_id, int b_sf_id) {
                          uint32_t desc = GetTCGEN5BlockScaledInstrDesc(
                              atom_m, atom_n, ab_dtype, a_is_k_major,
                              b_is_k_major, scale_in_a, scale_in_b, a_sf_id,
                              b_sf_id);
                          return Integer(static_cast<int64_t>(desc));
                        });
}

} // namespace tl
} // namespace tvm
