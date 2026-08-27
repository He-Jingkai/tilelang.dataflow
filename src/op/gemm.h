/*!
 * \file tl/op/gemm.h
 * \brief Define gemm operator.
 *
 */

#ifndef TVM_TL_OP_GEMM_H_
#define TVM_TL_OP_GEMM_H_

#include "operator.h"

#include <cstdint>
#include <utility>

namespace tvm {

namespace tl {

using namespace tir;

constexpr int kGemmContractSchemaVersion = 1;
constexpr int kGemmLoweringPlanSchemaVersion = 1;

constexpr const char *kGemmImplCudaTCGen05 = "cuda.tcgen05.sync";
constexpr const char *kGemmImplCudaWGMMA = "cuda.wgmma.async";
constexpr const char *kGemmImplCudaWGMMASharedARS = "cuda.wgmma.rs.shared_a";
constexpr const char *kGemmImplCudaMMA = "cuda.mma.sync";
constexpr const char *kGemmImplCudaScalar = "cuda.scalar.sync";
constexpr const char *kGemmImplCommon = "common.gemm.sync";

/*! \brief Target-independent logical GEMM semantics. */
class GemmContractNode : public Object {
public:
  int schema_version{kGemmContractSchemaVersion};
  int logical_m{0};
  int logical_n{0};
  int logical_k{0};
  PrimExpr padding_value;
  bool allow_padding{true};

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.GemmContract", GemmContractNode,
                                    Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GemmContractNode>()
        .def_ro("schema_version", &GemmContractNode::schema_version)
        .def_ro("logical_m", &GemmContractNode::logical_m)
        .def_ro("logical_n", &GemmContractNode::logical_n)
        .def_ro("logical_k", &GemmContractNode::logical_k)
        .def_ro("padding_value", &GemmContractNode::padding_value)
        .def_ro("allow_padding", &GemmContractNode::allow_padding);
  }
};

class GemmContract : public ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GemmContract, ObjectRef,
                                             GemmContractNode);

  TVM_DLL explicit GemmContract(Array<PrimExpr> args);
  static const Op &Get();
};

/*! \brief One allocation requirement declared by a GEMM implementation. */
class GemmTemporaryRequirementNode : public Object {
public:
  String buffer_role;
  String storage_scope;
  Array<Integer> logical_shape;
  Array<Integer> physical_shape;
  PrimExpr neutral_value;
  bool initialization_required{true};
  String lifetime_start;
  String lifetime_end;
  int64_t estimated_bytes{0};
  int64_t additional_bytes{0};

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.GemmTemporaryRequirement",
                                    GemmTemporaryRequirementNode, Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GemmTemporaryRequirementNode>()
        .def_ro("buffer_role", &GemmTemporaryRequirementNode::buffer_role)
        .def_ro("storage_scope", &GemmTemporaryRequirementNode::storage_scope)
        .def_ro("logical_shape", &GemmTemporaryRequirementNode::logical_shape)
        .def_ro("physical_shape", &GemmTemporaryRequirementNode::physical_shape)
        .def_ro("neutral_value", &GemmTemporaryRequirementNode::neutral_value)
        .def_ro("initialization_required",
                &GemmTemporaryRequirementNode::initialization_required)
        .def_ro("lifetime_start", &GemmTemporaryRequirementNode::lifetime_start)
        .def_ro("lifetime_end", &GemmTemporaryRequirementNode::lifetime_end)
        .def_ro("estimated_bytes",
                &GemmTemporaryRequirementNode::estimated_bytes)
        .def_ro("additional_bytes",
                &GemmTemporaryRequirementNode::additional_bytes);
  }
};

class GemmTemporaryRequirement : public ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GemmTemporaryRequirement,
                                             ObjectRef,
                                             GemmTemporaryRequirementNode);
};

/*! \brief Structured target-resolved implementation of a GEMM contract. */
class GemmLoweringPlanNode : public Object {
public:
  int schema_version{kGemmLoweringPlanSchemaVersion};
  String implementation_id;
  bool supported{true};
  bool synchronous{true};
  Array<Integer> logical_shape;
  Array<Integer> physical_shape;
  bool requires_padding{false};
  bool requires_materialization{false};
  Array<GemmTemporaryRequirement> temporary_requirements;
  int64_t additional_shared_memory_bytes{0};
  int64_t additional_fragment_bytes{0};
  int64_t estimated_resource_bytes{0};
  String selection_reason;
  Array<String> rejected_candidates;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindTreeNode;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.GemmLoweringPlan", GemmLoweringPlanNode,
                                    Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GemmLoweringPlanNode>()
        .def_ro("schema_version", &GemmLoweringPlanNode::schema_version)
        .def_ro("implementation_id", &GemmLoweringPlanNode::implementation_id)
        .def_ro("supported", &GemmLoweringPlanNode::supported)
        .def_ro("synchronous", &GemmLoweringPlanNode::synchronous)
        .def_ro("logical_shape", &GemmLoweringPlanNode::logical_shape)
        .def_ro("physical_shape", &GemmLoweringPlanNode::physical_shape)
        .def_ro("requires_padding", &GemmLoweringPlanNode::requires_padding)
        .def_ro("requires_materialization",
                &GemmLoweringPlanNode::requires_materialization)
        .def_ro("temporary_requirements",
                &GemmLoweringPlanNode::temporary_requirements)
        .def_ro("additional_shared_memory_bytes",
                &GemmLoweringPlanNode::additional_shared_memory_bytes)
        .def_ro("additional_fragment_bytes",
                &GemmLoweringPlanNode::additional_fragment_bytes)
        .def_ro("estimated_resource_bytes",
                &GemmLoweringPlanNode::estimated_resource_bytes)
        .def_ro("selection_reason", &GemmLoweringPlanNode::selection_reason)
        .def_ro("rejected_candidates",
                &GemmLoweringPlanNode::rejected_candidates);
  }
};

class GemmLoweringPlan : public ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GemmLoweringPlan, ObjectRef,
                                             GemmLoweringPlanNode);
};

TVM_DLL GemmTemporaryRequirement MakeGemmTemporaryRequirement(
    String buffer_role, String storage_scope, Array<Integer> logical_shape,
    Array<Integer> physical_shape, PrimExpr neutral_value,
    bool initialization_required, String lifetime_start, String lifetime_end,
    int64_t estimated_bytes, int64_t additional_bytes);

TVM_DLL GemmLoweringPlan MakeGemmLoweringPlan(
    String implementation_id, bool supported, bool synchronous,
    Array<Integer> logical_shape, Array<Integer> physical_shape,
    bool requires_padding, bool requires_materialization,
    Array<GemmTemporaryRequirement> temporary_requirements,
    int64_t additional_shared_memory_bytes, int64_t additional_fragment_bytes,
    int64_t estimated_resource_bytes, String selection_reason,
    Array<String> rejected_candidates = {});

enum class GemmWarpPolicyType : uint8_t {
  kSquare = 0,
  kFullRow = 1,
  kFullCol = 2,
  kFree = 3,
};

/// Convert GemmWarpPolicyType enum to string for debugging
inline const char *GemmWarpPolicyTypeToString(GemmWarpPolicyType type) {
  switch (type) {
  case GemmWarpPolicyType::kSquare:
    return "Square";
  case GemmWarpPolicyType::kFullRow:
    return "FullRow";
  case GemmWarpPolicyType::kFullCol:
    return "FullCol";
  case GemmWarpPolicyType::kFree:
    return "Free";
  default:
    return "Unknown";
  }
}

class GemmWarpPolicyNode : public Object {
public:
  mutable int m_warp{0};
  mutable int n_warp{0};
  int policy_type;

  TVM_FFI_DECLARE_OBJECT_INFO("tl.GemmWarpPolicy", GemmWarpPolicyNode, Object);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GemmWarpPolicyNode>()
        .def_ro("policy_type", &GemmWarpPolicyNode::policy_type)
        .def_ro("m_warp", &GemmWarpPolicyNode::m_warp)
        .def_ro("n_warp", &GemmWarpPolicyNode::n_warp);
  }

  std::pair<int, int> computeWarpPartition(int M, int N, int block_size,
                                           Target target,
                                           String gemm_inst) const;

  bool isSquare() const {
    return policy_type == int(GemmWarpPolicyType::kSquare);
  }
  bool isFullRow() const {
    return policy_type == int(GemmWarpPolicyType::kFullRow);
  }
  bool isFullCol() const {
    return policy_type == int(GemmWarpPolicyType::kFullCol);
  }
  bool isFree() const { return policy_type == int(GemmWarpPolicyType::kFree); }
};

class GemmWarpPolicy : public ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GemmWarpPolicy, ObjectRef,
                                             GemmWarpPolicyNode);

  explicit GemmWarpPolicy(GemmWarpPolicyType policy_type) {
    auto node = tvm::ffi::make_object<GemmWarpPolicyNode>();
    node->policy_type = (int)policy_type;
    data_ = std::move(node);
  }

  explicit GemmWarpPolicy(int policy_type) {
    auto node = tvm::ffi::make_object<GemmWarpPolicyNode>();
    node->policy_type = policy_type;
    data_ = std::move(node);
  }

  explicit GemmWarpPolicy(int m_warp, int n_warp) {
    auto node = tvm::ffi::make_object<GemmWarpPolicyNode>();
    node->m_warp = m_warp;
    node->n_warp = n_warp;
    node->policy_type = (int)GemmWarpPolicyType::kFree;
    data_ = std::move(node);
  }
};

class GemmNode : public TileOperatorNode {
public:
  tir::Buffer a_, b_, c_;
  // BufferRegion for A, B and C
  BufferRegion aRegion_, bRegion_, cRegion_;
  bool transA_, transB_;
  int m_, n_, k_;
  int strideA_, strideB_;
  int offsetA_, offsetB_;
  PrimExpr clearAccum_ = const_false();
  tir::BufferLoad mbar_; // mbar is optional, only used for TCGEN5MMA
  Array<PrimExpr> cCoords_;
  // k_pack please ref to bitblas/tl/mfma_macro_generator.py::k_pack
  // only will be enabled under cdna mfma instructions
  int kPack_ = 1;
  int wgWait_ = 0;
  bool isWgmma_ = false;
  bool isTcgen05_ = false;
  mutable GemmWarpPolicy policy_;
  Map<String, ObjectRef> annotations_;
  BufferRegion sfaRegion_, sfbRegion_;
  PrimExpr sfAId_, sfBId_;
  Optional<GemmContract> gemm_contract;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.Gemm", GemmNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GemmNode>()
        .def_ro("a", &GemmNode::a_)
        .def_ro("b", &GemmNode::b_)
        .def_ro("c", &GemmNode::c_)
        .def_ro("aRegion", &GemmNode::aRegion_)
        .def_ro("bRegion", &GemmNode::bRegion_)
        .def_ro("cRegion", &GemmNode::cRegion_)
        .def_ro("transA", &GemmNode::transA_)
        .def_ro("transB", &GemmNode::transB_)
        .def_ro("m", &GemmNode::m_)
        .def_ro("n", &GemmNode::n_)
        .def_ro("k", &GemmNode::k_)
        .def_ro("strideA", &GemmNode::strideA_)
        .def_ro("strideB", &GemmNode::strideB_)
        .def_ro("offsetA", &GemmNode::offsetA_)
        .def_ro("offsetB", &GemmNode::offsetB_)
        .def_ro("clearAccum", &GemmNode::clearAccum_)
        .def_ro("mbar", &GemmNode::mbar_)
        .def_ro("cCoords", &GemmNode::cCoords_)
        .def_ro("kPack", &GemmNode::kPack_)
        .def_ro("wgWait", &GemmNode::wgWait_)
        .def_ro("isWgmma", &GemmNode::isWgmma_)
        .def_ro("isTcgen05", &GemmNode::isTcgen05_)
        .def_ro("policy", &GemmNode::policy_)
        .def_ro("annotations", &GemmNode::annotations_)
        .def_ro("sfaRegion", &GemmNode::sfaRegion_)
        .def_ro("sfbRegion", &GemmNode::sfbRegion_)
        .def_ro("sfAId", &GemmNode::sfAId_)
        .def_ro("sfBId", &GemmNode::sfBId_)
        .def_ro("gemm_contract", &GemmNode::gemm_contract);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;
  AccessRegions GetAccessRegions() const override;

  TileOperator Clone() const;

  // Target-specific GEMM instruction key.
  String getGemmInstructionKey(int block_size, Target target) const;
  String getGemmInstructionKind(int block_size, Target target) const;

  int logical_m() const {
    return gemm_contract.defined() ? gemm_contract.value()->logical_m : m_;
  }
  int logical_n() const {
    return gemm_contract.defined() ? gemm_contract.value()->logical_n : n_;
  }
  int logical_k() const {
    return gemm_contract.defined() ? gemm_contract.value()->logical_k : k_;
  }

private:
  mutable bool completed_ = false;
};

using GemmTargetPredicate = bool (*)(Target target);

struct GemmLoweringContext {
  Target target;
  int block_size{0};
  Map<Var, Buffer> allocated_buffers;
  Map<Var, Array<PrimExpr>> planned_buffer_shapes;
  int64_t current_shared_memory_bytes{0};
  int64_t max_shared_memory_bytes{-1};
};

using GemmLoweringResolver = GemmLoweringPlan (*)(
    const GemmNode &op, const GemmLoweringContext &context);

struct GemmImpl {
  const char *name;
  GemmTargetPredicate match_target;

  String (*select_inst)(const GemmNode &op, int block_size, Target target);

  std::pair<int, int> (*compute_warp_partition)(
      const GemmWarpPolicyNode &policy, int M, int N, int block_size,
      Target target, String gemm_inst);

  bool (*reuse_existing_shared_layout)(String gemm_inst);

  String (*instruction_kind)(String gemm_inst);

  GemmLoweringResolver resolve_lowering;
};

void RegisterGemmImpl(GemmImpl impl);

TVM_DLL GemmLoweringPlan
ResolveGemmLowering(const GemmNode &op, const GemmLoweringContext &context);
TVM_DLL int GemmLoweringRegistryVersion();

class Gemm : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Gemm, TileOperator, GemmNode);
  TVM_DLL Gemm(Array<PrimExpr> args,
               Map<String, ObjectRef> annotations = Map<String, ObjectRef>());
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_GEMM_H_
