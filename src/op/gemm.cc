/*!
 * \file tl/op/gemm.cc
 * \brief Implementation of General Matrix Multiplication (GEMM) operators
 */

#include "gemm.h"

#include "builtin.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "utils.h"

#include <algorithm>
#include <sstream>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

GemmTemporaryRequirement MakeGemmTemporaryRequirement(
    String buffer_role, String storage_scope, Array<Integer> logical_shape,
    Array<Integer> physical_shape, PrimExpr neutral_value,
    bool initialization_required, String lifetime_start, String lifetime_end,
    int64_t estimated_bytes, int64_t additional_bytes) {
  ObjectPtr<GemmTemporaryRequirementNode> node =
      tvm::ffi::make_object<GemmTemporaryRequirementNode>();
  node->buffer_role = std::move(buffer_role);
  node->storage_scope = std::move(storage_scope);
  node->logical_shape = std::move(logical_shape);
  node->physical_shape = std::move(physical_shape);
  node->neutral_value = std::move(neutral_value);
  node->initialization_required = initialization_required;
  node->lifetime_start = std::move(lifetime_start);
  node->lifetime_end = std::move(lifetime_end);
  node->estimated_bytes = estimated_bytes;
  node->additional_bytes = additional_bytes;
  return GemmTemporaryRequirement(std::move(node));
}

GemmLoweringPlan MakeGemmLoweringPlan(
    String implementation_id, bool supported, bool synchronous,
    Array<Integer> logical_shape, Array<Integer> physical_shape,
    bool requires_padding, bool requires_materialization,
    Array<GemmTemporaryRequirement> temporary_requirements,
    int64_t additional_shared_memory_bytes, int64_t additional_fragment_bytes,
    int64_t estimated_resource_bytes, String selection_reason,
    Array<String> rejected_candidates) {
  ObjectPtr<GemmLoweringPlanNode> node =
      tvm::ffi::make_object<GemmLoweringPlanNode>();
  node->implementation_id = std::move(implementation_id);
  node->supported = supported;
  node->synchronous = synchronous;
  node->logical_shape = std::move(logical_shape);
  node->physical_shape = std::move(physical_shape);
  node->requires_padding = requires_padding;
  node->requires_materialization = requires_materialization;
  node->temporary_requirements = std::move(temporary_requirements);
  node->additional_shared_memory_bytes = additional_shared_memory_bytes;
  node->additional_fragment_bytes = additional_fragment_bytes;
  node->estimated_resource_bytes = estimated_resource_bytes;
  node->selection_reason = std::move(selection_reason);
  node->rejected_candidates = std::move(rejected_candidates);
  return GemmLoweringPlan(std::move(node));
}

const Op &GemmContract::Get() {
  static const Op &op = Op::Get("tl.gemm_contract");
  return op;
}

GemmContract::GemmContract(Array<PrimExpr> args) {
  ICHECK_EQ(args.size(), 5)
      << "tl.gemm_contract expects logical M/N/K, neutral padding value, "
         "and allow_padding";
  const int64_t *logical_m = as_const_int(args[0]);
  const int64_t *logical_n = as_const_int(args[1]);
  const int64_t *logical_k = as_const_int(args[2]);
  const int64_t *allow_padding = as_const_int(args[4]);
  ICHECK(logical_m != nullptr && logical_n != nullptr && logical_k != nullptr)
      << "logical GEMM shape must be compile-time integers";
  ICHECK_GT(*logical_m, 0);
  ICHECK_GT(*logical_n, 0);
  ICHECK_GT(*logical_k, 0);
  ICHECK(allow_padding != nullptr &&
         (*allow_padding == 0 || *allow_padding == 1))
      << "allow_padding must be a compile-time bool";
  ICHECK(args[3].dtype().is_scalar() && !args[3].dtype().is_handle())
      << "logical GEMM neutral padding must be a scalar expression";
  ICHECK_LE(SideEffect(args[3]), CallEffectKind::kPure)
      << "logical GEMM neutral padding must be a pure scalar expression and "
         "cannot read a buffer; pass a constant or scalar parameter";

  ObjectPtr<GemmContractNode> node = tvm::ffi::make_object<GemmContractNode>();
  node->logical_m = static_cast<int>(*logical_m);
  node->logical_n = static_cast<int>(*logical_n);
  node->logical_k = static_cast<int>(*logical_k);
  node->padding_value = args[3];
  node->allow_padding = *allow_padding != 0;
  data_ = std::move(node);
}

namespace {

std::vector<GemmImpl> &GemmImplRegistry() {
  static std::vector<GemmImpl> registry;
  return registry;
}

const GemmImpl &ResolveGemmImpl(Target target) {
  const auto &registry = GemmImplRegistry();
  const GemmImpl *matched_impl = nullptr;
  for (const GemmImpl &impl : registry) {
    if (impl.match_target(target)) {
      ICHECK(matched_impl == nullptr)
          << "tl.gemm found multiple target-specific implementations for "
          << target->ToDebugString() << ": " << matched_impl->name << " and "
          << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.gemm requires a target-specific implementation, but no gemm "
         "implementation is registered for "
      << target->ToDebugString();
  return *matched_impl;
}

} // namespace

void RegisterGemmImpl(GemmImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.select_inst != nullptr);
  ICHECK(impl.compute_warp_partition != nullptr);
  ICHECK(impl.reuse_existing_shared_layout != nullptr);
  ICHECK(impl.instruction_kind != nullptr);
  ICHECK(impl.resolve_lowering != nullptr);
  GemmImplRegistry().push_back(impl);
}

GemmLoweringPlan ResolveGemmLowering(const GemmNode &op,
                                     const GemmLoweringContext &context) {
  return ResolveGemmImpl(context.target).resolve_lowering(op, context);
}

int GemmLoweringRegistryVersion() { return kGemmLoweringPlanSchemaVersion; }

/**
 * @brief Construct a Gemm operator from serialized TL arguments.
 *
 * Deserializes operator parameters from `args` and resolves buffer references,
 * populating an internal GemmNode with buffers, transpose flags, M/N/K,
 * warp policy, clear_accum, strides, offsets, optional kPack/wg_wait, and
 * optional mbarrier.
 *
 * @param args Positional serialized arguments produced by the TL frontend:
 *   expected layout is:
 *     [Aptr, Bptr, Cptr, trans_A (Bool), trans_B (Bool),
 *      M (Int), N (Int), K (Int), policy (Int), clear_accum (Bool),
 *      stride_A (Int), stride_B (Int), offset_A (Int), offset_B (Int),
 *      (optional) kPack (Int), (optional) internal wg_wait (Int),
 *      (optional) mbar (BufferLoad), cCoord_y (PrimExpr), cCoord_x (PrimExpr)]
 */
Gemm::Gemm(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<GemmNode> node = tvm::ffi::make_object<GemmNode>();

  auto a_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto b_access = NormalizeToAccessRegion(args[1], kAccessRead);
  auto c_access = NormalizeToAccessRegion(args[2], kAccessReadWrite);

  node->aRegion_ = a_access.region;
  node->bRegion_ = b_access.region;
  node->cRegion_ = c_access.region;
  node->SetAccessRegions({a_access, b_access, c_access});

  node->a_ = node->aRegion_->buffer;
  node->b_ = node->bRegion_->buffer;
  node->c_ = node->cRegion_->buffer;
  node->transA_ = args[3].as<Bool>().value();
  node->transB_ = args[4].as<Bool>().value();
  node->m_ = args[5].as<IntImm>().value()->value;
  node->n_ = args[6].as<IntImm>().value()->value;
  node->k_ = args[7].as<IntImm>().value()->value;
  node->policy_ = GemmWarpPolicy(args[8].as<IntImm>().value()->value);
  node->clearAccum_ = args[9].as<PrimExpr>().value();
  node->strideA_ = args[10].as<IntImm>().value()->value;
  node->strideB_ = args[11].as<IntImm>().value()->value;
  node->offsetA_ = args[12].as<IntImm>().value()->value;
  node->offsetB_ = args[13].as<IntImm>().value()->value;
  if (args.size() > 14) {
    node->kPack_ = args[14].as<IntImm>().value()->value;
    if (node->kPack_ != 1 && node->kPack_ != 2) {
      ICHECK(false) << "kPack must be 1 or 2";
    }
  }
  if (args.size() > 15) {
    node->wgWait_ = args[15].as<IntImm>().value()->value;
  }
  if (auto val = annotations.Get("is_wgmma")) {
    const auto *int_val = val->as<IntImmNode>();
    ICHECK(int_val) << "is_wgmma annotation must be IntImmNode";
    node->isWgmma_ = int_val->value != 0;
  }
  if (auto val = annotations.Get("is_tcgen05")) {
    const auto *int_val = val->as<IntImmNode>();
    ICHECK(int_val) << "is_tcgen05 annotation must be IntImmNode";
    node->isTcgen05_ = int_val->value != 0;
  }
  if (args.size() > 16 && args[16]->IsInstance<BufferLoadNode>()) {
    node->mbar_ = Downcast<BufferLoad>(args[16]);
  }
  node->cCoords_ = Array<PrimExpr>(
      {args[17].as<PrimExpr>().value(), args[18].as<PrimExpr>().value()});
  size_t semantic_arg_count = args.size();
  for (size_t i = 19; i < args.size(); ++i) {
    const auto *contract_call = args[i].as<CallNode>();
    if (contract_call == nullptr) {
      continue;
    }
    const auto *contract_op = contract_call->op.as<OpNode>();
    if (contract_op == nullptr ||
        !ffi::GetRef<Op>(contract_op).same_as(GemmContract::Get())) {
      continue;
    }
    ICHECK(!node->gemm_contract.defined())
        << "tl.gemm accepts at most one logical GEMM contract";
    node->gemm_contract = GemmContract(contract_call->args);
    semantic_arg_count = std::min(semantic_arg_count, i);
  }
  if (semantic_arg_count > 19) {
    node->sfaRegion_ = NormalizeToBufferRegion(args[19]);
  }
  if (semantic_arg_count > 20) {
    node->sfbRegion_ = NormalizeToBufferRegion(args[20]);
  }
  if (semantic_arg_count > 21) {
    node->sfAId_ = args[21].as<PrimExpr>().value();
  }
  if (semantic_arg_count > 22) {
    node->sfBId_ = args[22].as<PrimExpr>().value();
  }
  if (node->gemm_contract.defined()) {
    const GemmContract &contract = node->gemm_contract.value();
    ICHECK_LE(contract->logical_m, node->m_)
        << "logical GEMM M cannot exceed the operand region M";
    ICHECK_LE(contract->logical_n, node->n_)
        << "logical GEMM N cannot exceed the operand region N";
    ICHECK_LE(contract->logical_k, node->k_)
        << "logical GEMM K cannot exceed the operand region K";
    ICHECK_EQ(contract->logical_n, node->n_)
        << "logical GEMM materialization currently supports M padding only; "
           "logical N must equal the operand region N";
    ICHECK_EQ(contract->logical_k, node->k_)
        << "logical GEMM materialization currently supports M padding only; "
           "logical K must equal the operand region K";
  }
  node->annotations_ = annotations;
  data_ = std::move(node);
}

AccessRegions GemmNode::GetAccessRegions() const {
  AccessRegions result;
  result.reads.push_back(aRegion_);
  result.reads.push_back(bRegion_);
  if (!is_one(clearAccum_)) {
    result.reads.push_back(cRegion_);
  }
  if (sfaRegion_.defined()) {
    result.reads.push_back(sfaRegion_);
  }
  if (sfbRegion_.defined()) {
    result.reads.push_back(sfbRegion_);
  }
  result.writes.push_back(cRegion_);
  return result;
}

TileOperator GemmNode::Clone() const {
  auto op = tvm::ffi::make_object<GemmNode>(*this);
  return Gemm(op);
}

String GemmNode::getGemmInstructionKey(int block_size, Target target) const {
  return ResolveGemmImpl(target).select_inst(*this, block_size, target);
}

String GemmNode::getGemmInstructionKind(int block_size, Target target) const {
  const GemmImpl &impl = ResolveGemmImpl(target);
  return impl.instruction_kind(impl.select_inst(*this, block_size, target));
}

std::pair<int, int> GemmWarpPolicyNode::computeWarpPartition(
    int M, int N, int block_size, Target target, String gemm_inst) const {
  return ResolveGemmImpl(target).compute_warp_partition(*this, M, N, block_size,
                                                        target, gemm_inst);
}

Stmt GemmNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (const auto f = ffi::Function::GetGlobal("tl.gemm.lower")) {
    PrimExpr mbar_phase = T.mbar_phase_expr;
    if (auto explicit_phase = GetAnnotatedMbarPhaseExpr(annotations_)) {
      mbar_phase = explicit_phase.value();
    }
    // NOTE(wt): Decide the instruction key and compute warp partition on Python
    // side.
    auto prim_func = Downcast<PrimFunc>(
        (*f)(tvm::ffi::GetRef<Gemm>(this), T.layout_map, T.target,
             T.thread_bounds, T.thread_var, mbar_phase));
    ICHECK(prim_func->attrs.defined());
    auto global_symbol =
        prim_func->attrs.GetAttr<tvm::ffi::String>("global_symbol");
    ICHECK(global_symbol.has_value());
    if (prim_func->body.as<BlockRealizeNode>()) {
      BlockRealize block_realize = Downcast<BlockRealize>(prim_func->body);
      auto block = block_realize->block;
      {
        BlockNode *n = block.CopyOnWrite();
        n->name_hint = global_symbol.value();
        n->annotations.Set(tl::attr::kLexicalAllocScope,
                           IntImm(DataType::Int(32), 1));
      }
      return BlockRealize(block_realize->iter_values, block_realize->predicate,
                          block);
    }
    // wrap with block realize node
    Map<String, ObjectRef> block_annotations;
    block_annotations.Set(tl::attr::kLexicalAllocScope,
                          IntImm(DataType::Int(32), 1));
    return BlockRealize(
        /*iter_values=*/Array<PrimExpr>(),
        /*predicate=*/const_true(),
        /*block=*/
        Block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
              /*name_hint=*/global_symbol.value(), prim_func->body,
              /*init=*/Optional<Stmt>(), /*alloc_buffers=*/{},
              /*match_buffers=*/{}, /*annotations=*/block_annotations));
  } else {
    LOG(FATAL) << "No lower function found for gemm";
    return Stmt();
  }
}

LayoutMap GemmNode::InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const {
  if (completed_)
    return {};
  LayoutMap results;
  if (const auto f = ffi::Function::GetGlobal("tl.gemm.infer_layout")) {
    auto inferred_layouts = Downcast<LayoutMap>(
        (*f)(tvm::ffi::GetRef<Gemm>(this), T.target, T.thread_bounds));
    // For MMA instructions, skip shared buffer layouts that are already
    // inferred by a prior operator to avoid layout conflicts when the same
    // shared buffer is consumed by multiple gemm ops with different transpose
    // semantics. WGMMA/TCGEN5MMA have strict shared memory layout requirements
    // and must always set their layouts.
    auto block_size = *as_const_int(T.thread_bounds->extent);
    String gemm_inst = getGemmInstructionKey(block_size, T.target);
    bool reuse_existing_shared_layout =
        ResolveGemmImpl(T.target).reuse_existing_shared_layout(gemm_inst);
    for (auto kv : inferred_layouts) {
      const Buffer &buf = kv.first;
      const Layout &layout = kv.second;
      if (reuse_existing_shared_layout && IsSharedBuffer(buf) &&
          T.layout_map.count(buf)) {
        continue;
      }
      if (auto frag = layout.as<Fragment>()) {
        results.Set(buf, frag.value()->BindThreadRange(T.thread_bounds));
      } else {
        results.Set(buf, layout);
      }
    }
  } else {
    LOG(FATAL) << "No infer layout function found for gemm";
  }

  completed_ = true;
  return results;
}

TIR_REGISTER_TL_TILE_OP(Gemm, gemm)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.wgmma_gemm")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "wgmma_gemm")
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               Map<String, ObjectRef> ann = annotations;
                               ann.Set("is_wgmma",
                                       IntImm(DataType::Int(32), 1));
                               return Gemm(args, ann);
                             })
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.tcgen05_gemm")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "tcgen05_gemm")
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               Map<String, ObjectRef> ann = annotations;
                               ann.Set("is_tcgen05",
                                       IntImm(DataType::Int(32), 1));
                               return Gemm(args, ann);
                             })
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.GemmWarpPolicy")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "GemmWarpPolicy");

TVM_REGISTER_OP("tl.gemm_contract")
    .set_num_inputs(5)
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "gemm_contract")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TVM_FFI_STATIC_INIT_BLOCK() {
  GemmNode::RegisterReflection();
  GemmContractNode::RegisterReflection();
  GemmTemporaryRequirementNode::RegisterReflection();
  GemmLoweringPlanNode::RegisterReflection();
  GemmWarpPolicyNode::RegisterReflection();
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.GemmWarpPolicyComputeWarpPartition",
                        [](GemmWarpPolicy policy, int M, int N, int block_size,
                           Target target, String gemm_inst) {
                          policy->computeWarpPartition(M, N, block_size, target,
                                                       gemm_inst);
                        });
  refl::GlobalDef()
      .def("tl.GemmGetGemmInstructionKey",
           [](Gemm gemm, int block_size, Target target) {
             return gemm->getGemmInstructionKey(block_size, target);
           })
      .def("tl.ResolveGemmLowering",
           [](Call call, Target target, int block_size,
              int current_shared_memory_bytes, int max_shared_memory_bytes) {
             TileOperator tile_op = ParseOperator(std::move(call));
             const auto *gemm = tile_op.as<GemmNode>();
             ICHECK(gemm != nullptr)
                 << "tl.ResolveGemmLowering expects a GEMM tile-op call";
             return ResolveGemmLowering(
                 *gemm, GemmLoweringContext{target,
                                            block_size,
                                            {},
                                            {},
                                            current_shared_memory_bytes,
                                            max_shared_memory_bytes});
           })
      .def("tl.GemmLoweringRegistryVersion",
           []() { return Integer(GemmLoweringRegistryVersion()); });
}

} // namespace tl
} // namespace tvm
