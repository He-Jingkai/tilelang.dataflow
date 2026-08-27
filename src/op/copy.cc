/*!
 * \file tl/op/copy.cc
 * \brief Define the copy operator, backend dispatch, and shared normal-copy
 *        lowering helpers.
 */

#include "copy.h"
#include "../transform/common/loop_fusion_utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"
#include "utils.h"

#include "builtin.h"
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include <limits>
#include <sstream>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

TransferLoweringPlan
MakeTransferLoweringPlan(String implementation_id, bool supported,
                         bool asynchronous, bool uses_tma_descriptor,
                         bool requires_post_fill, String selection_reason,
                         Array<String> rejected_candidates) {
  ObjectPtr<TransferLoweringPlanNode> node =
      tvm::ffi::make_object<TransferLoweringPlanNode>();
  node->implementation_id = std::move(implementation_id);
  node->supported = supported;
  node->asynchronous = asynchronous;
  node->uses_tma_descriptor = uses_tma_descriptor;
  node->requires_post_fill = requires_post_fill;
  node->selection_reason = std::move(selection_reason);
  node->rejected_candidates = std::move(rejected_candidates);
  return TransferLoweringPlan(std::move(node));
}

Stmt LowerNormalCopy(const CopyNode &op, const LowerArgs &T,
                     arith::Analyzer *analyzer) {
  bool is_cpu_target = T.target->GetTargetDeviceType() == kDLCPU;
  auto simt_loop = op.MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));

  For vectorized_thread_loop;
  auto par_op = ParallelOp(fused_loop);

  if (is_cpu_target || IsLocalBuffer(op.src) || IsLocalBuffer(op.dst)) {
    if (IsLocalBuffer(op.src) && !IsLocalBuffer(op.dst)) {
      // A conflict write only occurs when multiple threads write to the same
      // global address. If any dst_range dimension's min depends on the thread
      // variable, each thread targets a distinct location and there is no
      // conflict.
      bool dst_depends_on_thread = false;
      for (const auto &range : op.dst_range) {
        if (tir::UsesVar(range->min, [&](const VarNode *v) {
              return v == T.thread_var.get();
            })) {
          dst_depends_on_thread = true;
          break;
        }
      }
      if (!dst_depends_on_thread) {
        DLOG(WARNING) << "Copy from local buffer `" << op.src->name << "` to "
                      << op.dst.scope() << " buffer `" << op.dst->name
                      << "` may cause conflicted write.";
      }
    }
    vectorized_thread_loop = VectorizeLoop(fused_loop, T.layout_map);
    return vectorized_thread_loop;
  }

  std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                    InferLevel::kFree};
  for (auto level : levels) {
    par_op->InferLayout({T.target,
                         T.thread_bounds,
                         T.layout_map,
                         analyzer,
                         false,
                         T.buffer_remap,
                         {}},
                        level);
  }
  auto loop_layout = par_op->GetLoopLayout();
  return LowerParallelLoop(par_op->GetRoot(), loop_layout, T.thread_var,
                           analyzer, T.layout_map,
                           par_op->GetPredicate(T.thread_var));
}

bool HasLegacyTransferSemanticAnnotation(
    const Map<String, ObjectRef> &annotations) {
  bool compiler_owned_streamed_push =
      annotations.count("tl.reshared_credit_target_rank") &&
      annotations.count("tl.reshared_receive_stages") &&
      annotations.count("tl.reshared_transport_plan_fingerprint");
  bool pipeline_sync_consumed = false;
  if (auto consumed = annotations.Get(kTransferPipelineSyncConsumed)) {
    if (const auto *value = consumed.value().as<IntImmNode>()) {
      pipeline_sync_consumed = value->value != 0;
    }
  }
  for (const auto &entry : annotations) {
    std::string name = entry.first;
    if (name == kTransferPipelineSyncConsumed ||
        (compiler_owned_streamed_push && name == "dst_block") ||
        (pipeline_sync_consumed &&
         (name == attr::kAsyncCopyNoImplicitCommitWait ||
          name == "is_tma_copy" || name == "barrier" ||
          name == "skip_expect_transaction" || name == "emit_arrive"))) {
      continue;
    }
    if (name == "assume_src_in_bounds" ||
        name.rfind("src_upper_bound_", 0) == 0 || name == "is_async_copy" ||
        name == "is_tma_copy" || name == "force_cp_async" ||
        name == attr::kAsyncCopyNoImplicitCommitWait || name == "barrier" ||
        name == "skip_expect_transaction" || name == "emit_arrive" ||
        name == "dst_block") {
      return true;
    }
  }
  return false;
}

namespace {

std::vector<CopyImpl> &CopyImplRegistry() {
  static std::vector<CopyImpl> registry;
  return registry;
}

std::vector<TransferLoweringImpl> &TransferLoweringImplRegistry() {
  static std::vector<TransferLoweringImpl> registry;
  return registry;
}

const CopyImpl &ResolveCopyImpl(Target target) {
  const auto &registry = CopyImplRegistry();
  const CopyImpl *best_impl = nullptr;
  int best_priority = std::numeric_limits<int>::min();
  for (const CopyImpl &impl : registry) {
    if (impl.match_target(target) && impl.priority >= best_priority) {
      best_impl = &impl;
      best_priority = impl.priority;
    }
  }
  ICHECK(best_impl != nullptr)
      << "tl.copy requires a target-specific implementation, but no copy "
         "implementation is registered for "
      << target->ToDebugString();
  return *best_impl;
}

LayoutMap InferCopyLayout(const CopyNode &op, const LayoutInferArgs &T,
                          InferLevel level) {
  return ResolveCopyImpl(T.target).infer_layout(op, T, level);
}

Stmt LowerCopyForTarget(const CopyNode &op, const LowerArgs &T,
                        arith::Analyzer *analyzer) {
  return ResolveCopyImpl(T.target).lower(op, T, analyzer);
}

std::vector<Conv2DIm2ColImpl> &Conv2DIm2ColImplRegistry() {
  static std::vector<Conv2DIm2ColImpl> registry;
  return registry;
}

const Conv2DIm2ColImpl &ResolveConv2DIm2ColImpl(Target target) {
  const auto &registry = Conv2DIm2ColImplRegistry();
  const Conv2DIm2ColImpl *best_impl = nullptr;
  int best_priority = std::numeric_limits<int>::min();
  for (const Conv2DIm2ColImpl &impl : registry) {
    if (impl.match_target(target) && impl.priority >= best_priority) {
      best_impl = &impl;
      best_priority = impl.priority;
    }
  }
  ICHECK(best_impl != nullptr)
      << "Conv2D im2col requires a target-specific implementation, but no "
         "implementation is registered for "
      << target->ToDebugString();
  return *best_impl;
}

Stmt LowerConv2DIm2ColForTarget(const Conv2DIm2ColOpNode &op,
                                const LowerArgs &T, arith::Analyzer *analyzer) {
  return ResolveConv2DIm2ColImpl(T.target).lower(op, T, analyzer);
}

} // namespace

void RegisterCopyImpl(CopyImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.infer_layout != nullptr);
  ICHECK(impl.lower != nullptr);
  CopyImplRegistry().push_back(impl);
}

void RegisterTransferLoweringImpl(TransferLoweringImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.resolve != nullptr);
  TransferLoweringImplRegistry().push_back(impl);
}

TransferLoweringPlan
ResolveTransferLowering(const CopyNode &op,
                        const TransferLoweringContext &context) {
  ICHECK(op.transfer_contract.defined())
      << "ResolveTransferLowering expects a copy with a transfer contract";
  const TransferLoweringImpl *best_impl = nullptr;
  int best_priority = std::numeric_limits<int>::min();
  for (const TransferLoweringImpl &impl : TransferLoweringImplRegistry()) {
    if (impl.match_target(context.target) && impl.priority >= best_priority) {
      best_impl = &impl;
      best_priority = impl.priority;
    }
  }
  if (best_impl != nullptr) {
    return best_impl->resolve(op, context);
  }
  return MakeTransferLoweringPlan(
      kTransferImplCommonSIMT, /*supported=*/true, /*asynchronous=*/false,
      /*uses_tma_descriptor=*/false,
      /*requires_post_fill=*/false,
      "target backend has no accelerated transfer implementation");
}

bool TransferRequiresPostFill(const CopyNode &op, arith::Analyzer *analyzer) {
  if (!op.transfer_contract.defined()) {
    return false;
  }
  arith::Analyzer local_analyzer;
  if (analyzer == nullptr) {
    analyzer = &local_analyzer;
  }
  const Array<Range> &valid_region =
      op.transfer_contract.value()->src_valid_region->region;
  ICHECK_EQ(op.src_range.size(), valid_region.size());
  ICHECK_EQ(op.src_range.size(), op.src->shape.size());
  for (size_t axis = 0; axis < op.src_range.size(); ++axis) {
    const Range &logical = op.src_range[axis];
    const Range &valid = valid_region[axis];
    PrimExpr logical_end = logical->min + logical->extent;
    PrimExpr valid_end = valid->min + valid->extent;
    if (!analyzer->CanProve(logical->min >= 0,
                            arith::ProofStrength::kSymbolicBound) ||
        !analyzer->CanProve(logical_end <= op.src->shape[axis],
                            arith::ProofStrength::kSymbolicBound) ||
        !analyzer->CanProve(logical->min >= valid->min,
                            arith::ProofStrength::kSymbolicBound) ||
        !analyzer->CanProve(logical_end <= valid_end,
                            arith::ProofStrength::kSymbolicBound)) {
      return true;
    }
  }
  return false;
}

void RegisterConv2DIm2ColImpl(Conv2DIm2ColImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.lower != nullptr);
  Conv2DIm2ColImplRegistry().push_back(impl);
}

const Op &TransferContract::Get() {
  static const Op &op = Op::Get("tl.transfer_contract");
  return op;
}

TransferContract::TransferContract(Array<PrimExpr> args) {
  ICHECK_EQ(args.size(), 4U)
      << "tl.transfer_contract expects valid source region, OOB fill, "
         "allow_async, and synchronization owner";

  AccessRegion valid_access = NormalizeToAccessRegion(args[0], kAccessRead);
  ICHECK_EQ(valid_access.access_mask, kAccessRead)
      << "transfer valid source region must use read access";
  const auto *allow_async = as_const_int(args[2]);
  const auto *sync_owner = as_const_int(args[3]);
  ICHECK(allow_async != nullptr && (*allow_async == 0 || *allow_async == 1))
      << "transfer allow_async must be a compile-time boolean, got " << args[2];
  ICHECK(sync_owner != nullptr)
      << "transfer synchronization owner must be a compile-time integer, got "
      << args[3];
  ICHECK_GE(*sync_owner, static_cast<int>(TransferSyncOwner::kTransfer));
  ICHECK_LE(*sync_owner, static_cast<int>(TransferSyncOwner::kCaller));
  ICHECK(args[1].dtype().is_scalar() && !args[1].dtype().is_handle())
      << "transfer OOB fill must be a scalar value, got " << args[1];
  ICHECK_LE(SideEffect(args[1]), CallEffectKind::kPure)
      << "transfer OOB fill must be a pure scalar expression and cannot read "
         "a buffer; pass a constant or scalar parameter";
  ICHECK(*allow_async != 0 ||
         *sync_owner == static_cast<int>(TransferSyncOwner::kTransfer))
      << "a transfer that disallows asynchronous execution must own its "
         "synchronization";

  ObjectPtr<TransferContractNode> node =
      tvm::ffi::make_object<TransferContractNode>();
  node->src_valid_region = valid_access.region;
  node->oob_fill = args[1];
  node->allow_async = *allow_async != 0;
  node->sync_owner = static_cast<int>(*sync_owner);
  data_ = std::move(node);
}

// Constructs a Copy operator node from call arguments and annotations.
// args[0]: source region, args[1]: destination region
// annotations: Map containing common SIMT hints and backend-specific metadata.
Copy::Copy(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ICHECK(args.size() == 2U || args.size() == 3U)
      << "tl.copy expects source, destination, and an optional transfer "
         "contract, got "
      << args.size() << " arguments";
  ObjectPtr<CopyNode> node = tvm::ffi::make_object<CopyNode>();
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->src = src_access.region->buffer;
  node->dst = dst_access.region->buffer;
  node->src_range = src_access.region->region;
  node->dst_range = dst_access.region->region;
  node->SetAccessRegions({src_access, dst_access});
  // Copy annotations from the Call node
  node->annotations = annotations;
  if (args.size() == 3U) {
    const auto *contract_call = args[2].as<CallNode>();
    ICHECK(contract_call != nullptr)
        << "the third tl.copy argument must be tl.transfer_contract(...), got "
        << args[2];
    const auto *contract_op = contract_call->op.as<OpNode>();
    ICHECK(contract_op != nullptr &&
           ffi::GetRef<Op>(contract_op).same_as(TransferContract::Get()))
        << "the third tl.copy argument must be tl.transfer_contract(...), got "
        << args[2];
    TransferContract contract(contract_call->args);
    if (auto consumed = node->annotations.Get(kTransferPipelineSyncConsumed)) {
      const auto *value = consumed.value().as<IntImmNode>();
      ICHECK(value != nullptr &&
             (value->value == kTransferPipelineSyncManaged ||
              value->value == kTransferPipelineSyncFallback))
          << kTransferPipelineSyncConsumed
          << " must be a compiler-produced synchronization mode";
      ICHECK(contract->GetSyncOwner() == TransferSyncOwner::kPipeline)
          << kTransferPipelineSyncConsumed
          << " is reserved for compiler-consumed pipeline ownership";
    }
    ICHECK(contract->src_valid_region->buffer.same_as(node->src))
        << "transfer valid source region must reference the copy source "
           "buffer; "
           "copy source="
        << node->src->name
        << ", valid-region buffer=" << contract->src_valid_region->buffer->name;
    ICHECK_EQ(contract->src_valid_region->region.size(), node->src_range.size())
        << "transfer valid source region rank must match the copy source rank";

    ICHECK(!HasLegacyTransferSemanticAnnotation(node->annotations))
        << "typed transfer contract cannot be combined with legacy transfer "
           "semantic annotations";

    node->transfer_contract = std::move(contract);
  }
  if (auto dst_block = node->annotations.Get("dst_block")) {
    if (auto int_imm = dst_block->as<IntImmNode>()) {
      if (int_imm->value != -1) {
        node->dst_block = Integer(int_imm->value);
      }
    } else {
      node->dst_block = Downcast<PrimExpr>(dst_block.value());
    }
  }
  if (auto src_block = node->annotations.Get("src_block")) {
    if (auto int_imm = src_block->as<IntImmNode>()) {
      if (int_imm->value != -1) {
        node->src_block = Integer(int_imm->value);
      }
    } else {
      node->src_block = Downcast<PrimExpr>(src_block.value());
    }
  }
  ICHECK(!(node->src_block.defined() && node->dst_block.defined()))
      << "tl.copy cannot select both source- and destination-driven cluster "
         "transport";
  data_ = std::move(node);
}

// Creates a shallow clone of this CopyNode.
TileOperator CopyNode::Clone() const {
  auto op = tvm::ffi::make_object<CopyNode>(*this);
  if (par_op_.defined()) {
    op->par_op_ = Downcast<ParallelOp>(par_op_->Clone());
  }
  return Copy(op);
}

// Creates iterator variables for dimensions with extent > 1.
Array<IterVar> CopyNode::MakeIterVars() const {
  // Choose the range set from the lowest-level memory scope between src and
  // dst. Scope levels: global < shared/shared.dyn/shared.tmem < local.fragment
  // (fragment)
  auto scope_level = [](const Buffer &b) -> int {
    String s = b.scope();
    if (s == "local.fragment" || s == "local")
      return 2;
    if (s == "shared" || s == "shared.dyn" || s == "shared.tmem")
      return 1;
    // default to global level for unknown scopes
    return 0;
  };

  int src_level = scope_level(src);
  int dst_level = scope_level(dst);
  bool base_is_src = (src_level >= dst_level);
  const Array<Range> &base_ranges = base_is_src ? src_range : dst_range;

  // Sanity check: when switching away from the original (src_range),
  // ensure the chosen base ranges are not provably smaller than the original
  // per dimension. This guards against generating undersized loop domains.
  // Improved logic: use two pointers to traverse both base_ranges and
  // src_range, skipping dimensions with extent == 1. The number of non-1
  // extents must match.
  arith::Analyzer analyzer;

  size_t base_dim = 0, src_dim = 0;
  while (base_dim < base_ranges.size() && src_dim < src_range.size()) {
    // Skip base extents that are 1
    while (base_dim < base_ranges.size() &&
           is_one(base_ranges[base_dim]->extent)) {
      ++base_dim;
    }
    // Skip src extents that are 1
    while (src_dim < src_range.size() && is_one(src_range[src_dim]->extent)) {
      ++src_dim;
    }
    // Both indices now at non-1, or at end
    if (base_dim < base_ranges.size() && src_dim < src_range.size()) {
      PrimExpr base_ext = base_ranges[base_dim]->extent;
      PrimExpr src_ext = src_range[src_dim]->extent;
      // Only fail if base extent is provably smaller than src extent
      if (analyzer.CanProve(base_ext < src_ext)) {
        std::ostringstream oss;
        oss << "Selected loop range is smaller than original src range at "
               "matched non-1 dimension: "
            << "base(extent=" << base_ext
            << ", scope=" << (base_is_src ? src.scope() : dst.scope())
            << ", min=" << base_ranges[base_dim]->min
            << ", base_dim=" << base_dim << ") < src(extent=" << src_ext
            << ", min=" << src_range[src_dim]->min << ", src_dim=" << src_dim
            << ", scope=" << src.scope() << ") for src=" << src->name
            << ", dst=" << dst->name << "\n";
        oss << "src buffer: " << src->name << ", scope=" << src.scope() << "\n";
        oss << "dst buffer: " << dst->name << ", scope=" << dst.scope() << "\n";
        oss << "base_ranges[" << base_dim
            << "]: min=" << base_ranges[base_dim]->min
            << ", extent=" << base_ext << "\n";
        oss << "src_ranges[" << src_dim << "]: min=" << src_range[src_dim]->min
            << ", extent=" << src_ext << "\n";
        LOG(FATAL) << oss.str();
      }
      ++base_dim;
      ++src_dim;
    }
  }

  // Any remaining unmatched dimensions in either range must all have extent ==
  // 1
  while (base_dim < base_ranges.size()) {
    ICHECK(is_one(base_ranges[base_dim]->extent))
        << "base_ranges has extra non-1 extent at dim " << base_dim;
    ++base_dim;
  }
  while (src_dim < src_range.size()) {
    ICHECK(is_one(src_range[src_dim]->extent))
        << "src_range has extra non-1 extent at dim " << src_dim;
    ++src_dim;
  }

  Array<IterVar> loop_vars;
  size_t idx = 0;
  for (size_t i = 0; i < base_ranges.size(); i++) {
    if (is_one(base_ranges[i]->extent))
      continue;
    Var var = Var(std::string{char('i' + idx)}, base_ranges[i]->extent->dtype);
    idx++;
    loop_vars.push_back(
        {Range(0, base_ranges[i]->extent), var, IterVarType::kDataPar});
  }
  return loop_vars;
}

// Generates index expressions for accessing src (src_dst=0) or dst (src_dst=1)
// buffers.
Array<PrimExpr> CopyNode::MakeIndices(const Array<IterVar> &ivs,
                                      int src_dst) const {
  Array<PrimExpr> indices;
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent))
      indices.push_back(ranges[i]->min);
    else {
      indices.push_back(ranges[i]->min + ivs[idx]->var);
      idx++;
    }
  }
  ICHECK(idx == ivs.size())
      << "idx = " << idx << ", ivs.size() = " << ivs.size()
      << "src name = " << src->name << ", dst name = " << dst->name;
  return indices;
}

// Builds a boundary predicate for memory accesses.
// Returns a conjunction of bounds checks, or empty PrimExpr if all checks pass.
PrimExpr CopyNode::MakePredicate(arith::Analyzer *analyzer,
                                 const Array<IterVar> &ivs,
                                 Array<PrimExpr> extents, int src_dst) const {
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;

  Array<PrimExpr> cond_list;
  ICHECK(extents.size() == ranges.size()) << extents << " " << ranges;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    PrimExpr index = ranges[i]->min;
    if (!is_one(ranges[i]->extent)) {
      index += ivs[idx]->var;
      ++idx;
    }
    PrimExpr cond = index < extents[i];
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    cond = index >= 0;
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
  }
  ICHECK_EQ(idx, ivs.size());
  if (cond_list.empty())
    return {};
  else {
    PrimExpr cond = cond_list[0];
    for (size_t i = 1; i < cond_list.size(); i++)
      cond = And(cond, cond_list[i]);
    return cond;
  }
}

PrimExpr CopyNode::MakeRegionPredicate(arith::Analyzer *analyzer,
                                       const Array<IterVar> &ivs,
                                       const Array<Range> &valid_region,
                                       int src_dst) const {
  Array<PrimExpr> indices = MakeIndices(ivs, src_dst);
  ICHECK_EQ(indices.size(), valid_region.size())
      << "copy valid region rank does not match its access rank";

  Array<PrimExpr> cond_list;
  for (size_t i = 0; i < indices.size(); i++) {
    PrimExpr upper_bound = valid_region[i]->min + valid_region[i]->extent;
    PrimExpr cond = indices[i] < upper_bound;
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    cond = indices[i] >= valid_region[i]->min;
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
  }
  if (cond_list.empty())
    return {};
  else {
    PrimExpr cond = cond_list[0];
    for (size_t i = 1; i < cond_list.size(); i++)
      cond = And(cond, cond_list[i]);
    return cond;
  }
}

// Constructs a SIMT-style nested loop that implements the copy.
For CopyNode::MakeSIMTLoop(arith::Analyzer *analyzer) const {
  Array<IterVar> loop_vars = MakeIterVars();
  bool is_scalar = loop_vars.empty();

  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);
  ICHECK(loop_vars.size() <= src_range.size())
      << "loop_vars.size() = " << loop_vars.size()
      << ", src_range.size() = " << src_range.size() << ", src = " << src->name
      << ", dst = " << dst->name;

  ICHECK(loop_vars.size() <= dst_range.size())
      << "loop_vars.size() = " << loop_vars.size()
      << ", dst_range.size() = " << dst_range.size() << ", src = " << src->name
      << ", dst = " << dst->name;

  Array<PrimExpr> src_indices = MakeIndices(loop_vars, 0);
  Array<PrimExpr> dst_indices = MakeIndices(loop_vars, 1);

  bool assume_src_in_bounds = annotations.count("assume_src_in_bounds");

  PrimExpr src_predicate;
  if (!assume_src_in_bounds) {
    Array<PrimExpr> src_upper_bounds = src->shape;
    for (size_t i = 0; i < src_upper_bounds.size(); ++i) {
      std::string key = "src_upper_bound_" + std::to_string(i);
      if (auto bound = annotations.Get(key)) {
        src_upper_bounds.Set(i, Downcast<PrimExpr>(bound.value()));
      }
    }
    src_predicate = MakePredicate(analyzer, loop_vars, src_upper_bounds, 0);
  }
  PrimExpr oob_fill = make_zero(dst->dtype);
  if (transfer_contract.defined()) {
    const TransferContract &contract = transfer_contract.value();
    PrimExpr valid_predicate = MakeRegionPredicate(
        analyzer, loop_vars, contract->src_valid_region->region, 0);
    if (src_predicate.defined() && valid_predicate.defined()) {
      src_predicate = analyzer->Simplify(And(src_predicate, valid_predicate));
    } else if (valid_predicate.defined()) {
      src_predicate = valid_predicate;
    }
    oob_fill = contract->oob_fill;
    if (oob_fill.dtype() != dst->dtype) {
      oob_fill = Cast(dst->dtype, oob_fill);
    }
  }
  PrimExpr dst_predicate = MakePredicate(analyzer, loop_vars, dst->shape, 1);

  PrimExpr value = BufferLoad(src, src_indices);
  if (src->dtype != dst->dtype)
    value = Cast(dst->dtype, value);
  if (src_predicate.defined())
    value = if_then_else(src_predicate, value, oob_fill);

  Stmt body = BufferStore(dst, value, dst_indices);
  if (dst_predicate.defined())
    body = IfThenElse(dst_predicate, body);
  if (is_scalar) {
    return For(Var("i"), 0, 1, ForKind::kSerial, body);
  }

  for (int i = loop_vars.size() - 1; i >= 0; i--) {
    Map<String, ObjectRef> loop_annotations;

    // Only attach the parallel related annotations on the outermost loop (i ==
    // 0)
    if (i == 0) {
      if (annotations.count(attr::kCoalescedWidth)) {
        loop_annotations.Set(attr::kCoalescedWidth,
                             annotations.Get(attr::kCoalescedWidth).value());
      }
      if (annotations.count(attr::kParallelLoopLayout)) {
        loop_annotations.Set(
            attr::kParallelLoopLayout,
            annotations.Get(attr::kParallelLoopLayout).value());
      }
    }

    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body, std::nullopt, loop_annotations);
  }
  return Downcast<For>(body);
}

Optional<For> CopyNode::MakeSIMTPostFillLoop(arith::Analyzer *analyzer) const {
  if (!transfer_contract.defined()) {
    return std::nullopt;
  }

  Array<IterVar> loop_vars = MakeIterVars();
  bool is_scalar = loop_vars.empty();
  for (const auto &iv : loop_vars) {
    analyzer->Bind(iv->var, iv->dom);
  }

  PrimExpr valid_predicate =
      MakePredicate(analyzer, loop_vars, src->shape, /*src_dst=*/0);
  PrimExpr contract_predicate = MakeRegionPredicate(
      analyzer, loop_vars, transfer_contract.value()->src_valid_region->region,
      /*src_dst=*/0);
  if (valid_predicate.defined() && contract_predicate.defined()) {
    valid_predicate =
        analyzer->Simplify(And(valid_predicate, contract_predicate));
  } else if (contract_predicate.defined()) {
    valid_predicate = contract_predicate;
  }
  if (!valid_predicate.defined() ||
      analyzer->CanProve(valid_predicate,
                         arith::ProofStrength::kSymbolicBound)) {
    return std::nullopt;
  }

  PrimExpr store_predicate = analyzer->Simplify(Not(valid_predicate));
  PrimExpr dst_predicate =
      MakePredicate(analyzer, loop_vars, dst->shape, /*src_dst=*/1);
  if (dst_predicate.defined()) {
    store_predicate = analyzer->Simplify(And(dst_predicate, store_predicate));
  }

  PrimExpr oob_fill = transfer_contract.value()->oob_fill;
  if (oob_fill.dtype() != dst->dtype) {
    oob_fill = Cast(dst->dtype, oob_fill);
  }
  Stmt body = IfThenElse(
      store_predicate,
      BufferStore(dst, oob_fill, MakeIndices(loop_vars, /*src_dst=*/1)));
  if (is_scalar) {
    return For(Var("i"), 0, 1, ForKind::kSerial, body);
  }

  for (int i = static_cast<int>(loop_vars.size()) - 1; i >= 0; --i) {
    Map<String, ObjectRef> loop_annotations;
    if (i == 0) {
      if (annotations.count(attr::kCoalescedWidth)) {
        loop_annotations.Set(attr::kCoalescedWidth,
                             annotations.Get(attr::kCoalescedWidth).value());
      }
      if (annotations.count(attr::kParallelLoopLayout)) {
        loop_annotations.Set(
            attr::kParallelLoopLayout,
            annotations.Get(attr::kParallelLoopLayout).value());
      }
    }
    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body, std::nullopt, loop_annotations);
  }
  return Downcast<For>(body);
}

LayoutMap CopyNode::InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const {
  return InferCopyLayout(*this, T, level);
}

LayoutMap CopyNode::InferSIMTLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  if (!par_op_.defined()) {
    arith::Analyzer analyzer;
    par_op_ = ParallelOp(MakeSIMTLoop(&analyzer));
  }
  return par_op_->InferLayout(T, level);
}
// Lowers the copy operation by dispatching to the selected target
// implementation.
Stmt CopyNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (transfer_contract.defined()) {
    TransferSyncOwner owner = transfer_contract.value()->GetSyncOwner();
    bool pipeline_sync_consumed = false;
    if (auto consumed = annotations.Get(kTransferPipelineSyncConsumed)) {
      if (const auto *value = consumed.value().as<IntImmNode>()) {
        pipeline_sync_consumed = value->value != 0;
      }
    }
    ICHECK(owner == TransferSyncOwner::kTransfer ||
           (owner == TransferSyncOwner::kPipeline && pipeline_sync_consumed))
        << "transfer contract synchronization owner '"
        << TransferSyncOwnerToString(owner)
        << "' was not consumed by the corresponding common lowering pass";
  }
  return LowerCopyForTarget(*this, T, analyzer);
}

// Constructs a Conv2DIm2ColOp node from call arguments.
// args: src, dst, nhw_step, c_step, kernel, stride, dilation, padding,
// eviction_policy
Conv2DIm2ColOp::Conv2DIm2ColOp(Array<PrimExpr> args,
                               Map<String, ObjectRef> annotations) {
  ObjectPtr<Conv2DIm2ColOpNode> node =
      tvm::ffi::make_object<Conv2DIm2ColOpNode>();
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->srcRegion_ = src_access.region;
  node->dstRegion_ = dst_access.region;
  node->SetAccessRegions({src_access, dst_access});
  node->src_ = node->srcRegion_->buffer;
  node->dst_ = node->dstRegion_->buffer;
  node->nhw_step_ = args[2];
  node->c_step_ = args[3];
  node->kernel_ = args[4].as<IntImm>().value()->value;
  node->stride_ = args[5].as<IntImm>().value()->value;
  node->dilation_ = args[6].as<IntImm>().value()->value;
  node->padding_ = args[7].as<IntImm>().value()->value;
  node->eviction_policy_ = args[8].as<IntImm>().value()->value;
  node->annotations_ = annotations;
  data_ = std::move(node);
}

// Creates a shallow copy of this Conv2DIm2ColOpNode.
TileOperator Conv2DIm2ColOpNode::Clone() const {
  auto op = tvm::ffi::make_object<Conv2DIm2ColOpNode>(*this);
  return Conv2DIm2ColOp(op);
}

Stmt Conv2DIm2ColOpNode::Lower(const LowerArgs &T,
                               arith::Analyzer *analyzer) const {
  return LowerConv2DIm2ColForTarget(*this, T, analyzer);
}

TVM_REGISTER_OP("tl.transfer_contract")
    .set_num_inputs(4)
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "transfer_contract")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

// Register the Copy operation with TVM's TIR system
// This makes the copy operation available for use in TVM programs
// - Takes source, destination, and an optional typed transfer contract.
// - Marked as opaque since it has side effects (memory writes)
TIR_REGISTER_TL_TILE_OP(Copy, copy)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.async_copy")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "async_copy")
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               Map<String, ObjectRef> ann = annotations;
                               ann.Set("is_async_copy",
                                       IntImm(DataType::Int(32), 1));
                               return Copy(args, ann);
                             })
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// Register the tma_copy operation — same as copy but forces TMA path
// and emits only expect_tx + tma_load (no wait).
TVM_REGISTER_OP("tl.tileop.tma_copy")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "tma_copy")
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               Map<String, ObjectRef> ann = annotations;
                               ann.Set("is_tma_copy",
                                       IntImm(DataType::Int(32), 1));
                               return Copy(args, ann);
                             })
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// Layout inference hook - returns empty map (no layout suggestions).
LayoutMap Conv2DIm2ColOpNode::InferLayout(const LayoutInferArgs &T,
                                          InferLevel level) const {
  return {};
}

// Register the Conv2DIm2Col operation with TVM's TIR system
// This operation performs im2col transformation for 2D convolutions using a
// target-specific lowering.
// - Takes 9 inputs: src_buffer, dst_buffer, nhw_step, c_step, kernel, stride,
// dilation, padding, eviction_policy
// - Marked as opaque since it has side effects (memory writes)
TIR_REGISTER_TL_TILE_OP(Conv2DIm2ColOp, c2d_im2col)
    .set_num_inputs(9)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TransferContractNode::RegisterReflection();
  TransferLoweringPlanNode::RegisterReflection();
  CopyNode::RegisterReflection();
  Conv2DIm2ColOpNode::RegisterReflection();
  refl::GlobalDef()
      .def("tl.ResolveTransferLowering",
           [](Call call, Target target) {
             TileOperator tile_op = ParseOperator(std::move(call));
             const auto *copy = tile_op.as<CopyNode>();
             ICHECK(copy != nullptr && copy->transfer_contract.defined())
                 << "tl.ResolveTransferLowering expects a tl.copy call with a "
                    "transfer contract";
             arith::Analyzer analyzer;
             TransferLoweringContext context;
             context.target = std::move(target);
             context.analyzer = &analyzer;
             if (auto consumed =
                     copy->annotations.Get(kTransferPipelineSyncConsumed)) {
               if (const auto *value = consumed.value().as<IntImmNode>()) {
                 context.pipeline_owns_synchronization =
                     value->value == kTransferPipelineSyncManaged;
                 context.force_synchronous =
                     value->value == kTransferPipelineSyncFallback;
               }
             }
             return ResolveTransferLowering(*copy, context);
           })
      .def("tl.TransferLoweringRegistryVersion", []() { return Integer(2); });
}
} // namespace tl
} // namespace tvm
