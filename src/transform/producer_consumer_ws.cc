/*!
 * \file producer_consumer_ws.cc
 * \brief Warp-specialized producer/consumer rewriting at the tile-op level.
 *
 * This pass runs **before** LayoutInference and LowerTileOp, operating on
 * high-level tile ops (`tl.tileop.copy`, `tl.tileop.gemm`, etc.).
 * It recognizes pipelined producer/consumer structure directly from tile-op
 * semantics and splits eligible loops into warp-specialized branches with
 * explicit barrier synchronization.
 *
 * The output IR is equivalent to a hand-written warp-specialized kernel:
 *   - TMA-annotated copies become `tl.tileop.tma_copy` with barrier refs
 *   - Barriers (`mbarrier_wait_parity`, `ptx_arrive_barrier`) are inserted
 *   - The loop body is wrapped in `if (threadIdx.x >= consumer_extent)`
 *
 * Limitations (v1):
 *   - Pure TMA pipelines only (no mixed TMA + cp.async)
 *   - Single pipelined loop per block
 *   - No pre-loop TMA prefetch / prologue optimizations
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <limits>
#include <unordered_map>

#include "../backend/cuda/op/copy.h"
#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/fill.h"
#include "../op/gemm.h"
#include "../op/operator.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "../target/utils.h"
#include "common/mbarrier.h"
#include "common/pipeline_utils.h"
#include "multi_version_buffer_rewriter.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

// ---------------------------------------------------------------------------
// Utility: flatten SeqStmt recursively
// ---------------------------------------------------------------------------
void FlattenSeqStmt(const Stmt &s, Array<Stmt> *out) {
  if (auto *seq = s.as<SeqStmtNode>()) {
    for (const auto &sub : seq->seq) {
      FlattenSeqStmt(sub, out);
    }
  } else {
    out->push_back(s);
  }
}

/// Annotation key marking that this function was transformed by the tiled WS
/// pass, so downstream passes can skip redundant transformations.
static constexpr const char *kTiledWSApplied = "tl_tiled_ws_applied";
static constexpr const char *kCrossHandlerHandoffRole =
    "tl.cross_handler_handoff_role";
static constexpr const char *kCrossHandlerHandoffEnabled =
    "tl.cross_handler_handoff_enabled";
static constexpr const char *kCrossHandlerHandoffTransferIndex =
    "tl.cross_handler_handoff_transfer_index";
static constexpr const char *kCrossHandlerHandoffProducerTransfer =
    "tl.cross_handler_handoff_producer_transfer";
static constexpr const char *kDataflowParamRoles = "tl.dataflow_param_roles";
static constexpr const char *kHandoffStageCountRole = "handoff_stage_count";

// ---------------------------------------------------------------------------
// PhaseCounter: local counter for correct barrier parity in guarded loops
// ---------------------------------------------------------------------------
struct PhaseCounter {
  Buffer buf;

  static PhaseCounter Create(const std::string &name) {
    return {decl_buffer({IntImm(DataType::Int(32), 1)}, DataType::Int(32), name,
                        "local")};
  }

  PrimExpr Load() const {
    return BufferLoad(buf, {IntImm(DataType::Int(32), 0)});
  }

  Stmt Init() const {
    return BufferStore(buf, IntImm(DataType::Int(32), 0),
                       {IntImm(DataType::Int(32), 0)});
  }

  Stmt Increment() const {
    return BufferStore(buf, Load() + 1, {IntImm(DataType::Int(32), 0)});
  }

  Stmt WrapLoopWithAlloc(Stmt loop) const {
    Stmt body = SeqStmt({Init(), std::move(loop)});
    body = DeclBuffer(buf, body);
    return Allocate(buf->data, buf->dtype, buf->shape, const_true(), body);
  }

  PrimExpr StageExpr(int num_stages) const {
    if (num_stages == 1)
      return IntImm(DataType::Int(32), 0);
    return FloorMod(Load(), num_stages);
  }

  PrimExpr ParityExpr(int num_stages) const {
    if (num_stages == 1)
      return FloorMod(Load(), 2);
    return FloorMod(FloorDiv(Load(), num_stages), 2);
  }
};

// ---------------------------------------------------------------------------
// StageExprReplacer: rewrite loop-var-based stage indexing to counter-based
// ---------------------------------------------------------------------------
class StageExprReplacer : public StmtExprMutator {
public:
  static Stmt Replace(const Stmt &stmt, Var loop_var, PrimExpr loop_min,
                      int num_stages, PrimExpr replacement) {
    StageExprReplacer r(std::move(loop_var), std::move(loop_min), num_stages,
                        std::move(replacement));
    return r.VisitStmt(stmt);
  }

private:
  StageExprReplacer(Var loop_var, PrimExpr loop_min, int num_stages,
                    PrimExpr replacement)
      : loop_var_(std::move(loop_var)), loop_min_(std::move(loop_min)),
        num_stages_(num_stages), replacement_(std::move(replacement)) {}

  PrimExpr VisitExpr_(const FloorModNode *op) final {
    if (is_const_int(op->b, num_stages_) && MatchLinearIdx(op->a)) {
      return replacement_;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  bool MatchLinearIdx(const PrimExpr &expr) const {
    if (expr.same_as(loop_var_))
      return true;
    if (const auto *sub = expr.as<SubNode>()) {
      if (sub->a.same_as(loop_var_)) {
        if (is_const_int(sub->b, 0))
          return true;
        if (sub->b.same_as(loop_min_))
          return true;
      }
    }
    return false;
  }

  Var loop_var_;
  PrimExpr loop_min_;
  int num_stages_;
  PrimExpr replacement_;
};

class PipelineBufferStageExprRewriter : public StmtExprMutator {
public:
  static Stmt Replace(const Stmt &stmt,
                      const std::unordered_map<Var, int, ObjectPtrHash,
                                               ObjectPtrEqual> &buffer_versions,
                      PrimExpr iteration) {
    PipelineBufferStageExprRewriter rewriter(buffer_versions,
                                             std::move(iteration));
    return rewriter.VisitStmt(stmt);
  }

private:
  PipelineBufferStageExprRewriter(
      const std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual>
          &buffer_versions,
      PrimExpr iteration)
      : buffer_versions_(buffer_versions), iteration_(std::move(iteration)) {}

  PrimExpr StageIndex(const Buffer &buffer) const {
    auto it = buffer_versions_.find(buffer->data);
    ICHECK(it != buffer_versions_.end() && it->second > 1);
    return FloorMod(iteration_, IntImm(DataType::Int(32), it->second));
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    if (buffer_versions_.count(load->buffer->data)) {
      ICHECK(!load->indices.empty());
      load.CopyOnWrite()->indices.Set(0, StageIndex(load->buffer));
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (buffer_versions_.count(store->buffer->data)) {
      ICHECK(!store->indices.empty());
      store.CopyOnWrite()->indices.Set(0, StageIndex(store->buffer));
    }
    return store;
  }

  const std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual>
      &buffer_versions_;
  PrimExpr iteration_;
};

// ---------------------------------------------------------------------------
// Statement classification
// ---------------------------------------------------------------------------

using BufferDataToBufferMap =
    std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual>;
using BufferSet = std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
using BufferMap =
    std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>;
using VarExprMap =
    std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>;
using StmtRewriteMap =
    std::unordered_map<Stmt, Stmt, ObjectPtrHash, ObjectPtrEqual>;
using BufferLayoutMap = std::unordered_map<Var, std::pair<Buffer, Layout>,
                                           ObjectPtrHash, ObjectPtrEqual>;

struct LocalAccessSummary {
  BufferSet read_buffers;
  BufferSet write_buffers;
  VarSet read_vars;
  VarSet def_vars;

  bool HasTrackedDefs() const {
    return !write_buffers.empty() || !def_vars.empty();
  }

  bool HasBranchPrivateBufferWrites() const { return !write_buffers.empty(); }
};

struct LocalLiveSet {
  BufferSet buffers;
  VarSet vars;

  bool NeedsAnyDef(const LocalAccessSummary &summary) const {
    for (const auto &buf : summary.write_buffers) {
      if (buffers.count(buf)) {
        return true;
      }
    }
    for (const auto &var : summary.def_vars) {
      if (vars.count(var)) {
        return true;
      }
    }
    return false;
  }

  void AddUses(const LocalAccessSummary &summary) {
    buffers.insert(summary.read_buffers.begin(), summary.read_buffers.end());
    vars.insert(summary.read_vars.begin(), summary.read_vars.end());
  }
};

static void MergeLocalAccessSummary(LocalAccessSummary *dst,
                                    const LocalAccessSummary &src) {
  dst->read_buffers.insert(src.read_buffers.begin(), src.read_buffers.end());
  dst->write_buffers.insert(src.write_buffers.begin(), src.write_buffers.end());
  dst->read_vars.insert(src.read_vars.begin(), src.read_vars.end());
  dst->def_vars.insert(src.def_vars.begin(), src.def_vars.end());
}

static Buffer CloneBranchPrivateBuffer(const Buffer &buffer,
                                       const std::string &suffix) {
  Type new_type = buffer->data->type_annotation;
  if (IsFragmentBuffer(buffer)) {
    const auto *ptr_type = buffer->data->type_annotation.as<PointerTypeNode>();
    ICHECK(ptr_type);
    new_type = PointerType(ptr_type->element_type, "local");
  }
  Var new_var(buffer->data->name_hint + suffix, new_type);
  return Buffer(new_var, buffer->dtype, buffer->shape, buffer->strides,
                buffer->elem_offset, buffer->name + suffix,
                buffer->data_alignment, buffer->offset_factor,
                buffer->buffer_type);
}

class BufferRemapper : public StmtExprMutator {
public:
  static Stmt Rewrite(const Stmt &stmt, const BufferMap &buffer_remap) {
    if (buffer_remap.empty()) {
      return stmt;
    }
    BufferRemapper remapper(buffer_remap);
    return remapper.VisitStmt(stmt);
  }

private:
  explicit BufferRemapper(const BufferMap &buffer_remap)
      : buffer_remap_(buffer_remap) {
    for (const auto &[old_buf, new_buf] : buffer_remap_) {
      var_remap_.emplace(old_buf->data, new_buf->data);
    }
  }

  Buffer RemapBuffer(const Buffer &buffer) const {
    auto it = buffer_remap_.find(buffer);
    if (it != buffer_remap_.end()) {
      return it->second;
    }
    return buffer;
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    auto it = var_remap_.find(ffi::GetRef<Var>(op));
    if (it != var_remap_.end()) {
      return it->second;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    Buffer new_buffer = RemapBuffer(load->buffer);
    if (!new_buffer.same_as(load->buffer)) {
      return BufferLoad(new_buffer, load->indices, load->predicate, load->span);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    Buffer new_buffer = RemapBuffer(store->buffer);
    if (!new_buffer.same_as(store->buffer)) {
      return BufferStore(new_buffer, store->value, store->indices,
                         store->predicate, store->span);
    }
    return store;
  }

  const BufferMap &buffer_remap_;
  VarExprMap var_remap_;
};

enum class TileStmtKind {
  kTmaProducer,     // TMA load producer (global->shared)
  kCpAsyncProducer, // Explicit cp.async / commit / wait_group producer stmt
  kSimtProducer, // Non-tile-op SIMT copy: For loop writing shared from global
  kConsumer,     // Compute (gemm, reduce, element-wise, etc.)
  kOther         // Unclassified
};

/// Detect if a statement is a SIMT global-to-shared memory copy.
/// Matches any statement that writes to shared memory and reads from global
/// memory, without reading shared or local buffers (which would indicate
/// consumer-side compute).  This is intentionally broader than "pure direct
/// copy" so that T.Parallel with complex indexing / if_then_else (later
/// lowered to cp.async) is also captured.
class SimtProducerDetector : public StmtExprVisitor {
public:
  static bool Detect(const Stmt &stmt) {
    SimtProducerDetector d;
    d(stmt);
    return d.writes_shared_ && d.reads_global_ && !d.reads_shared_local_;
  }

private:
  void VisitStmt_(const BufferStoreNode *op) final {
    if (IsSharedBuffer(op->buffer)) {
      writes_shared_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (IsGlobalBuffer(op->buffer)) {
      reads_global_ = true;
    }
    if (IsSharedBuffer(op->buffer) || IsLocalBuffer(op->buffer, true)) {
      reads_shared_local_ = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  bool writes_shared_{false};
  bool reads_global_{false};
  bool reads_shared_local_{false};
};

static Optional<Call> GetEvaluateCallInSimpleWrapper(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *call = eval->value.as<CallNode>()) {
      return ffi::GetRef<Call>(call);
    }
    return std::nullopt;
  }
  if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
    if (!if_stmt->else_case.defined()) {
      return GetEvaluateCallInSimpleWrapper(if_stmt->then_case);
    }
    return std::nullopt;
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    return GetEvaluateCallInSimpleWrapper(attr->body);
  }
  if (const auto *let = stmt.as<LetStmtNode>()) {
    return GetEvaluateCallInSimpleWrapper(let->body);
  }
  if (const auto *block = stmt.as<BlockNode>()) {
    return GetEvaluateCallInSimpleWrapper(block->body);
  }
  if (const auto *realize = stmt.as<BlockRealizeNode>()) {
    return GetEvaluateCallInSimpleWrapper(realize->block->body);
  }
  return std::nullopt;
}

class BufferDataToBufferCollector : public StmtExprVisitor {
public:
  static BufferDataToBufferMap Collect(const Stmt &stmt) {
    BufferDataToBufferCollector collector;
    collector.VisitStmt(stmt);
    return collector.result_;
  }

private:
  void VisitStmt_(const BlockRealizeNode *op) final {
    CollectBuffers(op->block);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BlockNode *op) final {
    CollectBuffers(ffi::GetRef<Block>(op));
    StmtExprVisitor::VisitStmt_(op);
  }

  void CollectBuffers(const Block &block) {
    for (const auto &buffer : block->alloc_buffers) {
      result_.emplace(buffer->data, buffer);
    }
  }

  BufferDataToBufferMap result_;
};

class LocalAccessCollector : public StmtExprVisitor {
public:
  static LocalAccessSummary Collect(const Stmt &stmt,
                                    const BufferDataToBufferMap &buffer_map) {
    LocalAccessCollector collector(buffer_map);
    collector.VisitStmt(stmt);
    return std::move(collector.summary_);
  }

private:
  explicit LocalAccessCollector(const BufferDataToBufferMap &buffer_map)
      : buffer_data_to_buffer_(buffer_map) {}

  static bool IsBranchPrivateBuffer(const Buffer &buffer) {
    return IsFragmentBuffer(buffer) || IsLocalBuffer(buffer, true);
  }

  void VisitStmt_(const LetStmtNode *op) final {
    VisitExpr(op->value);
    summary_.def_vars.insert(op->var);
    bound_vars_.insert(op->var);
    VisitStmt(op->body);
    bound_vars_.erase(op->var);
  }

  void VisitStmt_(const ForNode *op) final {
    VisitExpr(op->min);
    VisitExpr(op->extent);
    bound_vars_.insert(op->loop_var);
    VisitStmt(op->body);
    bound_vars_.erase(op->loop_var);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (IsBranchPrivateBuffer(op->buffer)) {
      summary_.read_buffers.insert(op->buffer);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (IsBranchPrivateBuffer(op->buffer)) {
      summary_.write_buffers.insert(op->buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const VarNode *op) final {
    Var var = ffi::GetRef<Var>(op);
    if (bound_vars_.count(var) || buffer_data_to_buffer_.count(var)) {
      return;
    }
    summary_.read_vars.insert(var);
  }

  void VisitExpr_(const CallNode *op) final {
    if (auto tile_op = ParseOperator(ffi::GetRef<Call>(op));
        tile_op.defined()) {
      AccessRegions access = tile_op->GetAccessRegions();
      for (const auto &region : access.reads) {
        if (IsBranchPrivateBuffer(region->buffer)) {
          summary_.read_buffers.insert(region->buffer);
        }
        VisitBufferRegion(region);
      }
      for (const auto &region : access.writes) {
        if (IsBranchPrivateBuffer(region->buffer)) {
          summary_.write_buffers.insert(region->buffer);
        }
        VisitBufferRegion(region);
      }
      return;
    }

    if (op->op.same_as(tl::access_ptr())) {
      ICHECK_EQ(op->args.size(), 3);
      const auto *base_load = op->args[0].as<BufferLoadNode>();
      ICHECK(base_load);
      if (IsBranchPrivateBuffer(base_load->buffer)) {
        int rw_mask = GetConstAccessMask(op->args[2]);
        if (rw_mask & 1) {
          summary_.read_buffers.insert(base_load->buffer);
        }
        if (rw_mask & 2) {
          summary_.write_buffers.insert(base_load->buffer);
        }
      }
      for (const auto &index : base_load->indices) {
        VisitExpr(index);
      }
      VisitExpr(op->args[1]);
      return;
    }

    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5);
      const auto *var = op->args[1].as<VarNode>();
      ICHECK(var);
      auto it = buffer_data_to_buffer_.find(ffi::GetRef<Var>(var));
      if (it != buffer_data_to_buffer_.end() &&
          IsBranchPrivateBuffer(it->second)) {
        int rw_mask = GetConstAccessMask(op->args[4]);
        if (rw_mask & 1) {
          summary_.read_buffers.insert(it->second);
        }
        if (rw_mask & 2) {
          summary_.write_buffers.insert(it->second);
        }
      }
      VisitExpr(op->args[2]);
      VisitExpr(op->args[3]);
      return;
    }

    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitBufferRegion(const BufferRegion &region) {
    for (const auto &range : region->region) {
      VisitExpr(range->min);
      VisitExpr(range->extent);
    }
  }

  int GetConstAccessMask(const PrimExpr &expr) const {
    if (const int64_t *imm = as_const_int(expr)) {
      return static_cast<int>(*imm);
    }
    return 3;
  }

  const BufferDataToBufferMap &buffer_data_to_buffer_;
  LocalAccessSummary summary_;
  VarSet bound_vars_;
};

enum class PreludeStmtPlacement : uint8_t {
  kKeepSharedPrelude,
  kProducerOnly,
  kConsumerOnly,
  kDuplicateToBoth,
};

static PreludeStmtPlacement
ClassifyPreludeStmt(const Stmt &stmt, const BufferDataToBufferMap &buffer_map,
                    const LocalLiveSet &shared_live_seed,
                    const LocalLiveSet &producer_live_seed,
                    const LocalLiveSet &consumer_live_seed) {
  LocalAccessSummary summary = LocalAccessCollector::Collect(stmt, buffer_map);
  if (!summary.HasTrackedDefs()) {
    return PreludeStmtPlacement::kKeepSharedPrelude;
  }

  if (shared_live_seed.NeedsAnyDef(summary)) {
    return PreludeStmtPlacement::kKeepSharedPrelude;
  }

  bool producer_needs = producer_live_seed.NeedsAnyDef(summary);
  bool consumer_needs = consumer_live_seed.NeedsAnyDef(summary);
  if (producer_needs && consumer_needs) {
    return PreludeStmtPlacement::kDuplicateToBoth;
  }
  if (producer_needs) {
    return PreludeStmtPlacement::kProducerOnly;
  }
  if (consumer_needs) {
    return PreludeStmtPlacement::kConsumerOnly;
  }
  // A pre-pipeline write to a fragment/local buffer cannot safely stay in the
  // shared prelude once the loop is split into producer and consumer thread
  // partitions.  In this pass, the producer partition is reserved for
  // global-to-shared transfer work, so unresolved branch-private initializers
  // belong in the consumer partition.
  if (summary.HasBranchPrivateBufferWrites()) {
    return PreludeStmtPlacement::kConsumerOnly;
  }
  return PreludeStmtPlacement::kKeepSharedPrelude;
}

static bool ContainsPtxCpAsync(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (found) {
      return;
    }
    if (const auto *call = node.as<CallNode>()) {
      if (call->op.same_as(builtin::ptx_cp_async()) ||
          call->op.same_as(tl::ptx_cp_async())) {
        found = true;
      }
    }
  });
  return found;
}

static bool IsPtxCommitGroup(const Stmt &stmt) {
  Optional<Call> call = GetEvaluateCallInSimpleWrapper(stmt);
  return call.defined() &&
         call.value()->op.same_as(builtin::ptx_commit_group());
}

static bool IsPtxWaitGroup(const Stmt &stmt) {
  Optional<Call> call = GetEvaluateCallInSimpleWrapper(stmt);
  return call.defined() && call.value()->op.same_as(builtin::ptx_wait_group());
}

static bool IsBarrierOrTmaControlCall(const CallNode *call) {
  return call->op.same_as(mbarrier_wait_parity()) ||
         call->op.same_as(mbarrier_expect_tx()) ||
         call->op.same_as(builtin::ptx_arrive_barrier()) ||
         call->op.same_as(tl::ptx_arrive_cluster_barrier()) ||
         call->op.same_as(builtin::ptx_arrive_barrier_expect_tx()) ||
         call->op.same_as(builtin::ptx_cp_async_barrier()) ||
         call->op.same_as(tl::ptx_cp_async_barrier_noinc()) ||
         call->op.same_as(tma_load()) || call->op.same_as(tma_load_im2col()) ||
         call->op.same_as(tma_store()) ||
         call->op.same_as(tma_store_arrive()) ||
         call->op.same_as(tma_store_wait()) ||
         call->op.same_as(builtin::tvm_storage_sync());
}

static bool HasGlobalToSharedCopyShape(const CopyNode *copy) {
  return copy != nullptr && IsGlobalBuffer(copy->src) &&
         IsSharedBuffer(copy->dst) && copy->src->dtype == copy->dst->dtype;
}

static bool IsResidentPipelineTransferStmt(const Stmt &stmt) {
  Optional<Call> call = GetEvaluateCallInSimpleWrapper(stmt);
  if (!call.defined()) {
    return false;
  }
  auto tile_op = ParseOperator(call.value());
  const auto *copy = tile_op.as<CopyNode>();
  if (copy == nullptr) {
    return false;
  }
  auto materialization = copy->annotations.Get(kPipelineMaterialization);
  if (!materialization.has_value()) {
    return false;
  }
  const auto *mode = materialization.value().as<StringImmNode>();
  ICHECK(mode != nullptr) << kPipelineMaterialization
                          << " must be a typed string";
  if (mode->value != kPipelineMaterializationResident) {
    return false;
  }
  ICHECK(copy->transfer_contract.defined() &&
         copy->transfer_contract.value()->GetSyncOwner() ==
             TransferSyncOwner::kPipeline)
      << "resident pipeline materialization requires pipeline sync ownership";
  ICHECK(copy->src->data.same_as(copy->dst->data) &&
         copy->src_range.size() == copy->dst_range.size())
      << "resident pipeline materialization must alias one exact region";
  for (size_t axis = 0; axis < copy->src_range.size(); ++axis) {
    ICHECK(StructuralEqual()(copy->src_range[axis], copy->dst_range[axis]))
        << "resident pipeline materialization must alias one exact region";
  }
  return true;
}

static TileOperator GetSimpleTileOperator(const Stmt &stmt) {
  Optional<Call> call = GetEvaluateCallInSimpleWrapper(stmt);
  if (!call.defined()) {
    return TileOperator();
  }
  return ParseOperator(call.value());
}

static bool IsDetachedHandoffProducerStmt(const Stmt &stmt) {
  TileOperator tile_op = GetSimpleTileOperator(stmt);
  const CopyNode *copy = tile_op.as<CopyNode>();
  return copy != nullptr &&
         copy->annotations.count(kCrossHandlerHandoffProducerTransfer);
}

static bool IsHandoffConsumerCopy(const CopyNode *copy) {
  return copy != nullptr &&
         copy->annotations.count(kCrossHandlerHandoffTransferIndex);
}

static PrimExpr RewriteCopyToTmaCopy(const Call &copy_call,
                                     const Buffer &barrier_buf,
                                     PrimExpr barrier_id);

static Stmt RewriteDetachedHandoffProducerStmt(const Stmt &stmt,
                                               const Buffer &barrier_buf,
                                               PrimExpr barrier_id) {
  class Rewriter : public StmtExprMutator {
  public:
    Rewriter(Buffer barrier_buf, PrimExpr barrier_id)
        : barrier_buf_(std::move(barrier_buf)),
          barrier_id_(std::move(barrier_id)) {}

    PrimExpr VisitExpr_(const CallNode *op) final {
      Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
      auto tile_op = ParseOperator(call);
      const auto *copy = tile_op.as<CopyNode>();
      if (copy == nullptr ||
          !copy->annotations.count(kCrossHandlerHandoffProducerTransfer)) {
        return call;
      }
      ICHECK(!rewritten_)
          << "one detached handoff statement must contain exactly one copy";
      Call rewritten =
          Downcast<Call>(RewriteCopyToTmaCopy(call, barrier_buf_, barrier_id_));
      auto annotations = rewritten->annotations;
      annotations.Set("emit_arrive", IntImm(DataType::Int(32), 1));
      rewritten_ = true;
      return Call(rewritten->dtype, rewritten->op, rewritten->args, annotations,
                  rewritten->span);
    }

    bool rewritten() const { return rewritten_; }

  private:
    Buffer barrier_buf_;
    PrimExpr barrier_id_;
    bool rewritten_{false};
  } rewriter(barrier_buf, std::move(barrier_id));

  Stmt result = rewriter(stmt);
  ICHECK(rewriter.rewritten())
      << "detached handoff producer statement lost its typed copy";
  return result;
}

static Stmt ReplaceDetachedHandoffProducerLeaf(const Stmt &stmt,
                                               Stmt replacement) {
  class Rewriter : public StmtExprMutator {
  public:
    explicit Rewriter(Stmt replacement)
        : replacement_(std::move(replacement)) {}

    Stmt VisitStmt_(const EvaluateNode *op) final {
      const auto *call = op->value.as<CallNode>();
      if (call != nullptr) {
        auto tile_op = ParseOperator(ffi::GetRef<Call>(call));
        const auto *copy = tile_op.as<CopyNode>();
        if (copy != nullptr &&
            copy->annotations.count(kCrossHandlerHandoffProducerTransfer)) {
          ICHECK(!rewritten_)
              << "one detached handoff statement must contain one leaf";
          rewritten_ = true;
          return replacement_;
        }
      }
      return StmtExprMutator::VisitStmt_(op);
    }

    bool rewritten() const { return rewritten_; }

  private:
    Stmt replacement_;
    bool rewritten_{false};
  } rewriter(std::move(replacement));

  Stmt result = rewriter(stmt);
  ICHECK(rewriter.rewritten())
      << "detached handoff producer statement lost its typed leaf";
  return result;
}

static cuda::CopyInstSelection ClassifyWarpSpecializedCopy(const CopyNode *copy,
                                                           Target target) {
  if (copy == nullptr) {
    return {cuda::CopyInst::kNormal, true, ""};
  }
  if (auto mode = copy->annotations.Get(kTransferPipelineSyncConsumed)) {
    if (const auto *value = mode.value().as<IntImmNode>();
        value != nullptr && value->value == kTransferPipelineSyncFallback) {
      return {cuda::CopyInst::kNormal, true,
              "typed pipeline selected synchronous transfer fallback"};
    }
  }
  return cuda::ClassifyWarpSpecializedProducerCopy(*copy, target);
}

static bool IsStreamedClusterPush(const CopyNode *copy) {
  if (copy == nullptr || !copy->dst_block.defined() ||
      copy->src_block.defined() ||
      !copy->annotations.count(kResharedCreditTargetRank) ||
      !copy->annotations.count(kResharedReceiveStages) ||
      !copy->annotations.count(kResharedPayloadPartitionBytes)) {
    return false;
  }
  auto family = copy->annotations.Get(kResharedTransportFamily);
  const auto *value =
      family.has_value() ? family.value().as<StringImmNode>() : nullptr;
  return value != nullptr && value->value == kResharedTransportStreamed;
}

static bool CheckPipelineManagedCPAsyncCopy(const CopyNode *copy,
                                            Target target) {
  if (copy == nullptr) {
    return false;
  }
  return cuda::IsPipelineManagedCPAsyncCopy(*copy, target);
}

static bool IsSyncGlobalToSharedCopyLikeStmt(const Stmt &stmt, Target target) {
  Optional<Call> call = GetEvaluateCallInSimpleWrapper(stmt);
  if (!call.defined()) {
    return false;
  }
  auto tile_op = ParseOperator(call.value());
  if (!tile_op.defined()) {
    return false;
  }
  const auto *copy = tile_op.as<CopyNode>();
  if (copy == nullptr) {
    return false;
  }

  cuda::CopyInstSelection result = ClassifyWarpSpecializedCopy(copy, target);
  return HasGlobalToSharedCopyShape(copy) && result.supported &&
         !cuda::CopyInstIsTMA(result.inst) &&
         !cuda::CopyInstIsCPAsync(result.inst);
}

static bool IsProducerMovableLoopPrefixStmt(const Stmt &stmt, Target target) {
  if (IsSyncGlobalToSharedCopyLikeStmt(stmt, target)) {
    return true;
  }

  bool has_allowed_work = false;
  bool has_disallowed = false;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (has_disallowed) {
      return;
    }
    if (const auto *call = node.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_storage_sync())) {
        const auto *scope = call->args[0].as<StringImmNode>();
        if (!scope ||
            (scope->value != "shared" && scope->value != "shared.dyn")) {
          has_disallowed = true;
          return;
        }
        has_allowed_work = true;
        return;
      }
      if (IsBarrierOrTmaControlCall(call)) {
        has_disallowed = true;
        return;
      }
    }
    if (const auto *ld = node.as<BufferLoadNode>()) {
      if (IsSharedBuffer(ld->buffer) || IsLocalBuffer(ld->buffer, true)) {
        has_disallowed = true;
        return;
      }
      if (IsGlobalBuffer(ld->buffer)) {
        has_allowed_work = true;
      }
    }
    if (const auto *st = node.as<BufferStoreNode>()) {
      if (IsSharedBuffer(st->buffer)) {
        has_allowed_work = true;
        return;
      }
      has_disallowed = true;
    }
  });
  return has_allowed_work && !has_disallowed;
}

/// Classify a tile-op copy as TMA load producer, cp.async producer, or
/// consumer using coarse pre-layout checks.
static TileStmtKind ClassifyCopy(const CopyNode *copy, Target target) {
  if (copy == nullptr) {
    return TileStmtKind::kConsumer;
  }
  if (IsStreamedClusterPush(copy)) {
    return TileStmtKind::kTmaProducer;
  }

  cuda::CopyInstSelection result = ClassifyWarpSpecializedCopy(copy, target);
  if (!result.supported) {
    return TileStmtKind::kConsumer;
  }
  if (cuda::CopyInstIsTMA(result.inst)) {
    return TileStmtKind::kTmaProducer;
  }
  if (cuda::CopyInstIsCPAsync(result.inst)) {
    return TileStmtKind::kCpAsyncProducer;
  }

  return TileStmtKind::kConsumer;
}

static bool PipelineConsumerCopyNeedsPartitionSync(const Stmt &stmt,
                                                   Target target) {
  Optional<Call> call = GetEvaluateCallInSimpleWrapper(stmt);
  if (!call.defined()) {
    return false;
  }
  auto tile_op = ParseOperator(call.value());
  const auto *copy = tile_op.as<CopyNode>();
  return copy != nullptr && copy->transfer_contract.defined() &&
         copy->transfer_contract.value()->GetSyncOwner() ==
             TransferSyncOwner::kPipeline &&
         IsSharedBuffer(copy->dst) &&
         ClassifyCopy(copy, target) == TileStmtKind::kConsumer;
}

/// Classify a single statement in the pipeline loop body.
TileStmtKind ClassifyStmt(const Stmt &stmt, Target target) {
  // Tile-op Calls: classify directly via CopyNode checks.
  if (auto *eval = stmt.as<EvaluateNode>()) {
    if (auto *call = eval->value.as<CallNode>()) {
      auto tile_op = ParseOperator(ffi::GetRef<Call>(call));
      if (tile_op.defined()) {
        if (auto *copy = tile_op.as<CopyNode>()) {
          return ClassifyCopy(copy, target);
        }
        // Conv2D im2col lowers to tma_load_im2col on Hopper — treat as TMA
        // producer so it goes to the producer warp group.
        if (tile_op.as<Conv2DIm2ColOpNode>()) {
          if (TargetIsHopper(target)) {
            return TileStmtKind::kTmaProducer;
          }
        }
        return TileStmtKind::kConsumer; // non-copy tile-op
      }
    }
  }
  // Explicit cp.async producer-side statements are already low-level builtins.
  if (ContainsPtxCpAsync(stmt) || IsPtxCommitGroup(stmt) ||
      IsPtxWaitGroup(stmt)) {
    return TileStmtKind::kCpAsyncProducer;
  }
  // Non-tile-op: check for SIMT global-to-shared copy.
  if (SimtProducerDetector::Detect(stmt)) {
    return TileStmtKind::kSimtProducer;
  }
  return TileStmtKind::kConsumer;
}

bool IsProducer(TileStmtKind kind) {
  return kind == TileStmtKind::kTmaProducer ||
         kind == TileStmtKind::kCpAsyncProducer ||
         kind == TileStmtKind::kSimtProducer;
}

// ---------------------------------------------------------------------------
// Helpers: create barrier IR nodes
// ---------------------------------------------------------------------------

static Stmt MakeParityWait(const Buffer &barrier_buf, PrimExpr barrier_id,
                           PrimExpr parity) {
  auto ref = MakeBarrierRef(barrier_buf, std::move(barrier_id));
  return Evaluate(Call(DataType::Handle(), mbarrier_wait_parity(),
                       {ref, std::move(parity)}));
}

static Stmt MakeArriveBarrier(const Buffer &barrier_buf, PrimExpr barrier_id) {
  auto ref = MakeBarrierRef(barrier_buf, std::move(barrier_id));
  return Evaluate(
      Call(DataType::Handle(), builtin::ptx_arrive_barrier(), {ref}));
}

static Stmt MakeWarpgroupLeaderArriveBarrier(const Buffer &barrier_buf,
                                             PrimExpr barrier_id) {
  constexpr int kWarpgroupThreadCount = 128;
  return IfThenElse(Call(DataType::Bool(), tl_shuffle_elect(),
                         {IntImm(DataType::Int(32), kWarpgroupThreadCount)}),
                    MakeArriveBarrier(barrier_buf, std::move(barrier_id)));
}

static Stmt MakeArriveBarrierExpectTx(const Buffer &barrier_buf,
                                      PrimExpr barrier_id,
                                      PrimExpr transaction_bytes) {
  auto ref = MakeBarrierRef(barrier_buf, std::move(barrier_id));
  return Evaluate(Call(DataType::Handle(),
                       builtin::ptx_arrive_barrier_expect_tx(),
                       {ref, std::move(transaction_bytes)}));
}

static Stmt MakeArriveClusterBarrier(const Buffer &barrier_buf,
                                     PrimExpr barrier_id,
                                     PrimExpr target_rank) {
  auto ref = MakeBarrierRef(barrier_buf, std::move(barrier_id));
  return Evaluate(Call(DataType::Handle(), tl::ptx_arrive_cluster_barrier(),
                       {ref, std::move(target_rank)}));
}

static Stmt MakeArriveClusterBarrier(const Buffer &barrier_buf,
                                     PrimExpr barrier_id, int target_rank) {
  return MakeArriveClusterBarrier(barrier_buf, std::move(barrier_id),
                                  IntImm(DataType::Int(32), target_rank));
}

static Stmt MakeWarpgroupLeaderArriveClusterBarrier(const Buffer &barrier_buf,
                                                    PrimExpr barrier_id,
                                                    PrimExpr target_rank) {
  constexpr int kWarpgroupThreadCount = 128;
  return IfThenElse(Call(DataType::Bool(), tl_shuffle_elect(),
                         {IntImm(DataType::Int(32), kWarpgroupThreadCount)}),
                    MakeArriveClusterBarrier(barrier_buf, std::move(barrier_id),
                                             std::move(target_rank)));
}

static Stmt MakeSharedStorageSync() {
  return Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),
                       {StringImm("shared")}));
}

static int64_t GetCopyClusterMask(const CopyNode *copy) {
  if (copy == nullptr) {
    return 0;
  }
  if (auto mask = copy->annotations.Get("cluster_mask")) {
    if (const auto *value = mask.value().as<IntImmNode>()) {
      return value->value;
    }
  }
  return 0;
}

static int GetCopyProducerPartition(const CopyNode *copy) {
  if (copy == nullptr) {
    return -1;
  }
  if (auto partition = copy->annotations.Get(kPipelineProducerPartition)) {
    const auto *value = partition.value().as<IntImmNode>();
    ICHECK(value != nullptr && value->value >= 0)
        << kPipelineProducerPartition
        << " must be a compile-time non-negative integer";
    return static_cast<int>(value->value);
  }
  return -1;
}

static int GetCopyPipelineBufferVersions(const CopyNode *copy, int num_stages) {
  if (copy == nullptr) {
    return num_stages;
  }
  if (auto annotation = copy->annotations.Get(kPipelineBufferVersions)) {
    const auto *value = annotation.value().as<IntImmNode>();
    ICHECK(value != nullptr && value->value > 0 && value->value <= num_stages)
        << kPipelineBufferVersions
        << " must be a compile-time positive integer no greater than the "
           "pipeline stage count";
    return static_cast<int>(value->value);
  }
  return num_stages;
}

static int GetStreamedClusterPushPartitionCount(const CopyNode *copy) {
  if (!IsStreamedClusterPush(copy)) {
    return 1;
  }
  auto raw = copy->annotations.Get(kResharedPayloadPartitionBytes);
  ICHECK(raw.has_value());
  Array<PrimExpr> partitions = Downcast<Array<PrimExpr>>(raw.value());
  ICHECK(!partitions.empty())
      << "streamed cluster push requires at least one payload partition";
  for (const PrimExpr &partition : partitions) {
    const int64_t *bytes = as_const_int(partition);
    ICHECK(bytes != nullptr && *bytes > 0)
        << "streamed cluster push partitions must be static and positive";
  }
  return static_cast<int>(partitions.size());
}

static PrimExpr GetStreamedClusterPushCreditTarget(const CopyNode *copy) {
  ICHECK(IsStreamedClusterPush(copy));
  auto target = copy->annotations.Get(kResharedCreditTargetRank);
  ICHECK(target.has_value());
  return Downcast<PrimExpr>(target.value());
}

static int MinRankInMask(int64_t mask) {
  ICHECK_GT(mask, 0);
  int rank = 0;
  while ((mask & 1) == 0) {
    mask >>= 1;
    ++rank;
  }
  return rank;
}

static int CountRanksInMask(int64_t mask) {
  ICHECK_GT(mask, 0);
  int count = 0;
  while (mask != 0) {
    count += mask & 1;
    mask >>= 1;
  }
  return count;
}

static PrimExpr CopyTransactionBytes(const CopyNode *copy) {
  ICHECK(copy != nullptr);
  PrimExpr elements = IntImm(DataType::Int(64), 1);
  for (const Range &range : copy->dst_range) {
    elements *= cast(DataType::Int(64), range->extent);
  }
  int bits = copy->dst->dtype.bits();
  return FloorDiv(elements * IntImm(DataType::Int(64), bits) +
                      IntImm(DataType::Int(64), 7),
                  IntImm(DataType::Int(64), 8));
}

// ---------------------------------------------------------------------------
// Convert tl.tileop.copy → tl.tileop.tma_copy with barrier annotation
// ---------------------------------------------------------------------------

/// Rewrite a `tl.tileop.copy` Call into a `tl.tileop.tma_copy` Call with
/// barrier reference.  The args (src/dst regions) are preserved; only the op
/// and annotations change.
static PrimExpr RewriteCopyToTmaCopy(const Call &copy_call,
                                     const Buffer &barrier_buf,
                                     PrimExpr barrier_id) {
  static const Op &tma_copy_op = Op::Get("tl.tileop.tma_copy");
  auto new_annotations = copy_call->annotations;
  new_annotations.Set("barrier", MakeBarrierRef(barrier_buf, barrier_id));
  new_annotations.Set("is_tma_copy", IntImm(DataType::Int(32), 1));
  if (new_annotations.Get(kPipelineProducerPartition)) {
    if (auto leader_extent = new_annotations.Get("leader_thread_extent")) {
      const auto *value = leader_extent.value().as<IntImmNode>();
      ICHECK(value != nullptr && value->value == 32)
          << "partitioned TMA transfer requires leader_thread_extent=32";
    }
    new_annotations.Set("leader_thread_extent", IntImm(DataType::Int(32), 32));
  }
  auto tile_op = ParseOperator(copy_call);
  if (const auto *copy = tile_op.as<CopyNode>();
      copy != nullptr && copy->transfer_contract.defined()) {
    new_annotations.Set(kTransferPipelineSyncConsumed,
                        IntImm(DataType::Int(32), 1));
  }
  return Call(copy_call->dtype, tma_copy_op, copy_call->args, new_annotations,
              copy_call->span);
}

/// Annotate SIMT producer statements so the enclosing transform owns cp.async
/// synchronization.
/// - ForNodes get `kParallelAsyncWithoutAsyncCommitWait = true` so
///   InjectPTXAsyncCopy does not emit commit_group + wait_group(0).
/// - Tile-op copy calls get `kAsyncCopyNoImplicitCommitWait` so copy.cc does
///   not emit its own implicit commit/wait either.
/// This allows the WS pass to emit its own commit_group +
/// cp_async_barrier_noinc, tying cp.async completion to the forward mbarrier.
class SimtProducerAnnotator : public StmtExprMutator {
public:
  static Stmt Annotate(const Stmt &stmt, Target target) {
    SimtProducerAnnotator a(std::move(target));
    return a.VisitStmt(stmt);
  }

private:
  explicit SimtProducerAnnotator(Target target) : target_(std::move(target)) {}

  Stmt VisitStmt_(const ForNode *op) final {
    Stmt body = VisitStmt(op->body);
    auto annotations = op->annotations;
    annotations.Set(attr::kParallelAsyncWithoutAsyncCommitWait, Bool(true));
    return For(op->loop_var, op->min, op->extent, op->kind, body,
               op->thread_binding, annotations, op->step, op->span);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    static const Op &copy_op = Op::Get("tl.tileop.copy");
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (!call->op.same_as(copy_op) || !CanUsePipelineManagedCPAsyncCopy(call)) {
      return call;
    }
    auto annotations = call->annotations;
    annotations.Set(attr::kAsyncCopyNoImplicitCommitWait,
                    IntImm(DataType::Int(32), 1));
    auto tile_op = ParseOperator(call);
    if (const auto *copy = tile_op.as<CopyNode>();
        copy != nullptr && copy->transfer_contract.defined()) {
      annotations.Set(kTransferPipelineSyncConsumed,
                      IntImm(DataType::Int(32), 1));
    }
    return Call(call->dtype, call->op, call->args, annotations, call->span);
  }

  bool CanUsePipelineManagedCPAsyncCopy(const Call &call) const {
    auto tile_op = ParseOperator(call);
    const auto *copy = tile_op.as<CopyNode>();
    if (copy == nullptr) {
      return false;
    }
    return CheckPipelineManagedCPAsyncCopy(copy, target_);
  }

  Target target_;
};

/// Copies that remain in the consumer partition are intentionally
/// synchronous.  Record that the warp-specialized pipeline consumed their
/// synchronization ownership so common copy lowering can select the legal
/// synchronous implementation alongside other asynchronous producers.
class ConsumerTransferFallbackAnnotator : public StmtExprMutator {
public:
  static Stmt Annotate(const Stmt &stmt) {
    ConsumerTransferFallbackAnnotator annotator;
    return annotator.VisitStmt(stmt);
  }

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    auto tile_op = ParseOperator(call);
    const auto *copy = tile_op.as<CopyNode>();
    if (copy == nullptr || !copy->transfer_contract.defined() ||
        copy->transfer_contract.value()->GetSyncOwner() !=
            TransferSyncOwner::kPipeline ||
        call->annotations.count(kTransferPipelineSyncConsumed)) {
      return call;
    }
    auto annotations = call->annotations;
    annotations.Set(kTransferPipelineSyncConsumed,
                    IntImm(DataType::Int(32), kTransferPipelineSyncFallback));
    return Call(call->dtype, call->op, call->args, annotations, call->span);
  }
};

class TileOpMbarPhaseAnnotator : public StmtExprMutator {
public:
  static Stmt Annotate(const Stmt &stmt, PrimExpr phase_expr) {
    TileOpMbarPhaseAnnotator annotator(std::move(phase_expr));
    return annotator.VisitStmt(stmt);
  }

private:
  explicit TileOpMbarPhaseAnnotator(PrimExpr phase_expr)
      : phase_expr_(std::move(phase_expr)) {}

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (!IsMbarPhaseConsumer(call)) {
      return call;
    }
    if (call->annotations.count(attr::kPipelineMbarPhaseExpr)) {
      return call;
    }
    auto annotations = call->annotations;
    annotations.Set(attr::kPipelineMbarPhaseExpr, phase_expr_);
    return Call(call->dtype, call->op, call->args, annotations, call->span);
  }

  bool IsMbarPhaseConsumer(const Call &call) const {
    auto tile_op = ParseOperator(call);
    return tile_op.defined() && (tile_op.as<CopyNode>() != nullptr ||
                                 tile_op.as<Conv2DIm2ColOpNode>() != nullptr ||
                                 tile_op.as<GemmNode>() != nullptr);
  }

  PrimExpr phase_expr_;
};

/// Annotate a tile-op Call (e.g., c2d_im2col) with a barrier reference.
/// The tile-op's Lower() is expected to check for the "barrier" annotation
/// and use it instead of allocating its own mbarrier.
static PrimExpr AnnotateTileOpBarrier(const Call &tile_call,
                                      const Buffer &barrier_buf,
                                      PrimExpr barrier_id) {
  auto new_annotations = tile_call->annotations;
  new_annotations.Set("barrier", MakeBarrierRef(barrier_buf, barrier_id));
  return Call(tile_call->dtype, tile_call->op, tile_call->args, new_annotations,
              tile_call->span);
}

struct BufferDataAccessInfo {
  bool read{false};
  bool write{false};

  bool HasAnyAccess() const { return read || write; }

  void Merge(const BufferDataAccessInfo &other) {
    read = read || other.read;
    write = write || other.write;
  }
};

struct BufferUsePositions {
  int first_read{-1};
  int last_access{-1};
};

struct PreludeTmaLoadPlan {
  Stmt stmt;
  int wait_pos{-1};
};

static BufferDataAccessInfo
AnalyzeBufferDataAccess(const Stmt &stmt, const Var &buffer_data,
                        const BufferDataToBufferMap &buffer_map) {
  class BufferDataAccessDetector : public StmtExprVisitor {
  public:
    BufferDataAccessDetector(const Var &buffer_data,
                             const BufferDataToBufferMap &buffer_map)
        : buffer_data_(buffer_data), buffer_map_(buffer_map) {}

    BufferDataAccessInfo Result() const { return result_; }

  private:
    void VisitExpr_(const BufferLoadNode *op) final {
      if (op->buffer->data.same_as(buffer_data_)) {
        result_.read = true;
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitStmt_(const BufferStoreNode *op) final {
      if (op->buffer->data.same_as(buffer_data_)) {
        result_.write = true;
      }
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitExpr_(const CallNode *op) final {
      if (auto tile_op = ParseOperator(ffi::GetRef<Call>(op));
          tile_op.defined()) {
        result_.Merge(GetTileOpBufferDataAccess(tile_op));
        StmtExprVisitor::VisitExpr_(op);
        return;
      }

      if (op->op.same_as(tl::access_ptr())) {
        ICHECK_EQ(op->args.size(), 3);
        const auto *base_load = op->args[0].as<BufferLoadNode>();
        ICHECK(base_load);
        if (base_load->buffer->data.same_as(buffer_data_)) {
          MarkAccess(op->args[2]);
        }
        for (const auto &index : base_load->indices) {
          VisitExpr(index);
        }
        VisitExpr(op->args[1]);
        return;
      }

      if (op->op.same_as(builtin::tvm_access_ptr())) {
        ICHECK_EQ(op->args.size(), 5);
        const auto *var = op->args[1].as<VarNode>();
        ICHECK(var);
        auto it = buffer_map_.find(ffi::GetRef<Var>(var));
        if (it != buffer_map_.end() && it->second->data.same_as(buffer_data_)) {
          MarkAccess(op->args[4]);
        }
        VisitExpr(op->args[2]);
        VisitExpr(op->args[3]);
        return;
      }

      StmtExprVisitor::VisitExpr_(op);
    }

    BufferDataAccessInfo
    GetTileOpBufferDataAccess(const TileOperator &tile_op) const {
      BufferDataAccessInfo access;
      AccessRegions regions = tile_op->GetAccessRegions();
      for (const auto &region : regions.reads) {
        if (region->buffer->data.same_as(buffer_data_)) {
          access.read = true;
        }
      }
      for (const auto &region : regions.writes) {
        if (region->buffer->data.same_as(buffer_data_)) {
          access.write = true;
        }
      }
      return access;
    }

    void MarkAccess(const PrimExpr &rw_expr) {
      int rw_mask = 3;
      if (const int64_t *imm = as_const_int(rw_expr)) {
        rw_mask = static_cast<int>(*imm);
      }
      if (rw_mask & 1) {
        result_.read = true;
      }
      if (rw_mask & 2) {
        result_.write = true;
      }
    }

    Var buffer_data_;
    const BufferDataToBufferMap &buffer_map_;
    BufferDataAccessInfo result_;
  };

  BufferDataAccessDetector detector(buffer_data, buffer_map);
  detector(stmt);
  return detector.Result();
}

static BufferDataAccessInfo
AnalyzeWgmmaIssueBufferDataAccess(const Stmt &stmt, const Var &buffer_data) {
  class WgmmaIssueAccessDetector : public StmtExprVisitor {
  public:
    explicit WgmmaIssueAccessDetector(const Var &buffer_data)
        : buffer_data_(buffer_data) {}

    BufferDataAccessInfo Result() const { return result_; }

  private:
    void VisitExpr_(const CallNode *op) final {
      auto tile_op = ParseOperator(ffi::GetRef<Call>(op));
      if (tile_op.defined()) {
        if (const auto *gemm = tile_op.as<GemmNode>();
            gemm != nullptr && gemm->isWgmma_) {
          AccessRegions regions = gemm->GetAccessRegions();
          for (const auto &region : regions.reads) {
            if (region->buffer->data.same_as(buffer_data_)) {
              result_.read = true;
            }
          }
          for (const auto &region : regions.writes) {
            if (region->buffer->data.same_as(buffer_data_)) {
              result_.write = true;
            }
          }
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    Var buffer_data_;
    BufferDataAccessInfo result_;
  };

  WgmmaIssueAccessDetector detector(buffer_data);
  detector(stmt);
  return detector.Result();
}

static bool ContainsWgmmaWait(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (found) {
      return;
    }
    if (const auto *call = node.as<CallNode>()) {
      if (call->op.same_as(tl::wait_wgmma()) ||
          call->op.same_as(tl::warpgroup_wait())) {
        found = true;
      }
    }
  });
  return found;
}

static bool ContainsDrainingWgmmaWait(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (found) {
      return;
    }
    const auto *call = node.as<CallNode>();
    if (call == nullptr ||
        (!call->op.same_as(tl::wait_wgmma()) &&
         !call->op.same_as(tl::warpgroup_wait())) ||
        call->args.empty()) {
      return;
    }
    const int64_t *pending_groups = as_const_int(call->args[0]);
    found = pending_groups != nullptr && *pending_groups == 0;
  });
  return found;
}

static bool IsImplicitlySynchronousWgmma(const Stmt &stmt, int consumer_threads,
                                         const Target &target) {
  Optional<Call> call = GetEvaluateCallInSimpleWrapper(stmt);
  if (!call.defined() || !call.value()->op.same_as(Gemm::Get())) {
    return false;
  }
  auto tile_op = ParseOperator(call.value());
  const auto *gemm = tile_op.as<GemmNode>();
  return gemm != nullptr &&
         gemm->getGemmInstructionKind(consumer_threads, target) == "wgmma";
}

struct ExternalizedWgmmaDrain {
  Stmt issue;
  Array<Stmt> drain;
};

class ExactCallReplacer : public StmtExprMutator {
public:
  static Stmt Replace(const Stmt &stmt, Call target, PrimExpr replacement) {
    ExactCallReplacer replacer(std::move(target), std::move(replacement));
    return replacer.VisitStmt(stmt);
  }

private:
  ExactCallReplacer(Call target, PrimExpr replacement)
      : target_(std::move(target)), replacement_(std::move(replacement)) {}

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (ffi::GetRef<Call>(op).same_as(target_)) {
      return replacement_;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Call target_;
  PrimExpr replacement_;
};

static Stmt MakeWgmmaOperandFence(const Buffer &buffer, int64_t element_count,
                                  int consumer_threads) {
  ICHECK_GT(element_count, 0);
  ICHECK_GT(consumer_threads, 0);
  int64_t total_bits =
      element_count * buffer->dtype.bits() * buffer->dtype.lanes();
  int64_t bits_per_register_partition =
      static_cast<int64_t>(consumer_threads) * 32;
  int64_t num_regs = (total_bits + bits_per_register_partition - 1) /
                     bits_per_register_partition;
  return Evaluate(
      Call(DataType::Handle(), warpgroup_fence_operand(),
           {StringImm(runtime::DLDataTypeToString(buffer->dtype)), buffer->data,
            buffer->elem_offset, IntImm(DataType::Int(32), num_regs)}));
}

static ExternalizedWgmmaDrain
ExternalizeImplicitWgmmaDrain(const Stmt &stmt, int consumer_threads,
                              const Target &target) {
  ExternalizedWgmmaDrain result{stmt, {}};
  Optional<Call> maybe_call = GetEvaluateCallInSimpleWrapper(stmt);
  if (!maybe_call.defined() || !maybe_call.value()->op.same_as(Gemm::Get()) ||
      maybe_call.value()->args.size() <= 15 ||
      !IsImplicitlySynchronousWgmma(stmt, consumer_threads, target)) {
    return result;
  }
  Call call = maybe_call.value();
  const int64_t *wait = as_const_int(call->args[15]);
  if (wait == nullptr || *wait != 0) {
    return result;
  }

  auto tile_op = ParseOperator(call);
  const auto *gemm = tile_op.as<GemmNode>();
  ICHECK(gemm != nullptr);
  Array<PrimExpr> args = call->args;
  args.Set(15, IntImm(call->args[15].dtype(), -1));
  Map<String, ObjectRef> annotations = call->annotations;
  annotations.Set("wgmma_emit_fence_after", Bool(false));
  Call async_issue(call->dtype, call->op, args, annotations, call->span);
  result.issue = ExactCallReplacer::Replace(stmt, call, async_issue);
  result.drain.push_back(Evaluate(
      Call(DataType::Handle(), wait_wgmma(), {IntImm(DataType::Int(32), 0)})));
  if (IsFragmentBuffer(gemm->a_)) {
    result.drain.push_back(MakeWgmmaOperandFence(
        gemm->a_, static_cast<int64_t>(gemm->m_) * gemm->k_, consumer_threads));
  }
  result.drain.push_back(MakeWgmmaOperandFence(
      gemm->c_, static_cast<int64_t>(gemm->m_) * gemm->n_, consumer_threads));
  return result;
}

static BufferUsePositions
AnalyzeConsumerBufferUsePositions(const Array<Stmt> &consumer_stmts,
                                  const Var &buffer_data,
                                  const BufferDataToBufferMap &buffer_map) {
  BufferUsePositions positions;
  BufferDataAccessInfo pending_wgmma_access;

  for (size_t ci = 0; ci < consumer_stmts.size(); ++ci) {
    BufferDataAccessInfo access =
        AnalyzeBufferDataAccess(consumer_stmts[ci], buffer_data, buffer_map);
    pending_wgmma_access.Merge(
        AnalyzeWgmmaIssueBufferDataAccess(consumer_stmts[ci], buffer_data));
    if (ContainsWgmmaWait(consumer_stmts[ci]) &&
        pending_wgmma_access.HasAnyAccess()) {
      // WGMMA reads shared operands asynchronously; the producer slot must not
      // be released until the matching wait drains the warpgroup queue.
      access.Merge(pending_wgmma_access);
      pending_wgmma_access = BufferDataAccessInfo{};
    }

    if (access.read && positions.first_read < 0) {
      positions.first_read = static_cast<int>(ci);
    }
    if (access.HasAnyAccess()) {
      positions.last_access = static_cast<int>(ci);
    }
  }

  return positions;
}

static bool CollectPreludeStmtsToPipelineLoop(const Stmt &stmt,
                                              const For &pipeline_loop,
                                              Array<Stmt> *prelude_stmts) {
  if (stmt.same_as(pipeline_loop)) {
    return true;
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (int i = 0; i < static_cast<int>(seq->seq.size()); ++i) {
      Array<Stmt> nested_prelude;
      if (CollectPreludeStmtsToPipelineLoop(seq->seq[i], pipeline_loop,
                                            &nested_prelude)) {
        for (int j = 0; j < i; ++j) {
          prelude_stmts->push_back(seq->seq[j]);
        }
        prelude_stmts->insert(prelude_stmts->end(), nested_prelude.begin(),
                              nested_prelude.end());
        return true;
      }
    }
    return false;
  }
  if (const auto *let = stmt.as<LetStmtNode>()) {
    return CollectPreludeStmtsToPipelineLoop(let->body, pipeline_loop,
                                             prelude_stmts);
  }
  if (const auto *realize = stmt.as<BlockRealizeNode>()) {
    return CollectPreludeStmtsToPipelineLoop(realize->block->body,
                                             pipeline_loop, prelude_stmts);
  }
  if (const auto *block = stmt.as<BlockNode>()) {
    return CollectPreludeStmtsToPipelineLoop(block->body, pipeline_loop,
                                             prelude_stmts);
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    return CollectPreludeStmtsToPipelineLoop(attr->body, pipeline_loop,
                                             prelude_stmts);
  }
  if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
    Array<Stmt> nested_prelude;
    if (CollectPreludeStmtsToPipelineLoop(if_stmt->then_case, pipeline_loop,
                                          &nested_prelude)) {
      prelude_stmts->insert(prelude_stmts->end(), nested_prelude.begin(),
                            nested_prelude.end());
      return true;
    }
    if (if_stmt->else_case.defined()) {
      nested_prelude.clear();
      if (CollectPreludeStmtsToPipelineLoop(if_stmt->else_case.value(),
                                            pipeline_loop, &nested_prelude)) {
        prelude_stmts->insert(prelude_stmts->end(), nested_prelude.begin(),
                              nested_prelude.end());
        return true;
      }
    }
  }
  return false;
}

static Optional<Var> ExtractProducerWriteBufferData(const Stmt &stmt) {
  Optional<Call> call = GetEvaluateCallInSimpleWrapper(stmt);
  if (!call.defined()) {
    return Optional<Var>();
  }
  auto tile_op = ParseOperator(call.value());
  if (!tile_op.defined()) {
    return Optional<Var>();
  }
  if (const auto *copy = tile_op.as<CopyNode>()) {
    if (IsSharedBuffer(copy->dst)) {
      return copy->dst->data;
    }
  }
  if (const auto *im2col = tile_op.as<Conv2DIm2ColOpNode>()) {
    if (IsSharedBuffer(im2col->dst_)) {
      return im2col->dst_->data;
    }
  }
  return Optional<Var>();
}

static int
FindFirstAsyncProducerConsumerRead(const Stmt &producer_stmt,
                                   const Array<Stmt> &consumer_compute_stmts,
                                   const BufferDataToBufferMap &buffer_map) {
  int earliest_read = static_cast<int>(consumer_compute_stmts.size());
  auto update_earliest_read = [&](const Var &buffer_data) {
    for (size_t ci = 0; ci < static_cast<size_t>(earliest_read); ++ci) {
      BufferDataAccessInfo access = AnalyzeBufferDataAccess(
          consumer_compute_stmts[ci], buffer_data, buffer_map);
      if (access.read) {
        earliest_read = static_cast<int>(ci);
        return;
      }
    }
  };
  if (Optional<Var> write_buffer_data =
          ExtractProducerWriteBufferData(producer_stmt)) {
    update_earliest_read(write_buffer_data.value());
  }
  PostOrderVisit(producer_stmt, [&](const ObjectRef &obj) {
    if (earliest_read == 0) {
      return;
    }
    if (const auto *store = obj.as<BufferStoreNode>()) {
      if (IsSharedBuffer(store->buffer)) {
        update_earliest_read(store->buffer->data);
      }
      return;
    }
    const auto *call = obj.as<CallNode>();
    if (!call || !(call->op.same_as(builtin::ptx_cp_async()) ||
                   call->op.same_as(tl::ptx_cp_async()))) {
      return;
    }
    PostOrderVisit(call->args[0], [&](const ObjectRef &ptr_obj) {
      if (earliest_read == 0) {
        return;
      }
      if (const auto *load = ptr_obj.as<BufferLoadNode>()) {
        if (IsSharedBuffer(load->buffer)) {
          update_earliest_read(load->buffer->data);
        }
        return;
      }
      const auto *ptr_call = ptr_obj.as<CallNode>();
      if (!ptr_call || !ptr_call->op.same_as(builtin::tvm_access_ptr())) {
        return;
      }
      const auto *var = ptr_call->args[1].as<VarNode>();
      if (!var) {
        return;
      }
      auto it = buffer_map.find(ffi::GetRef<Var>(var));
      if (it != buffer_map.end() && IsSharedBuffer(it->second)) {
        update_earliest_read(it->second->data);
      }
    });
  });
  return earliest_read;
}

static Stmt RewritePreludeTmaProducerStmt(const Stmt &stmt,
                                          const Buffer &barrier_buf,
                                          PrimExpr barrier_id) {
  class PreludeTmaProducerRewriter : public StmtExprMutator {
  public:
    PreludeTmaProducerRewriter(Buffer barrier_buf, PrimExpr barrier_id)
        : barrier_buf_(std::move(barrier_buf)),
          barrier_id_(std::move(barrier_id)) {}

    Stmt Rewrite(const Stmt &stmt) { return VisitStmt(stmt); }

  private:
    PrimExpr VisitExpr_(const CallNode *op) final {
      Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
      if (rewritten_) {
        return call;
      }
      auto tile_op = ParseOperator(call);
      if (!tile_op.defined()) {
        return call;
      }
      PrimExpr rewritten_call;
      if (tile_op.as<CopyNode>()) {
        rewritten_call = RewriteCopyToTmaCopy(call, barrier_buf_, barrier_id_);
      } else if (tile_op.as<Conv2DIm2ColOpNode>()) {
        rewritten_call = AnnotateTileOpBarrier(call, barrier_buf_, barrier_id_);
      } else {
        return call;
      }
      Call new_call = Downcast<Call>(rewritten_call);
      auto annotations = new_call->annotations;
      annotations.Set("emit_arrive", IntImm(DataType::Int(32), 1));
      rewritten_ = true;
      return Call(new_call->dtype, new_call->op, new_call->args, annotations,
                  new_call->span);
    }

    Buffer barrier_buf_;
    PrimExpr barrier_id_;
    bool rewritten_{false};
  };

  PreludeTmaProducerRewriter rewriter(barrier_buf, std::move(barrier_id));
  return rewriter.Rewrite(stmt);
}

// ---------------------------------------------------------------------------
// Main rewriter
// ---------------------------------------------------------------------------

class ProducerConsumerWSRewriter : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f, std::string *rejection_reason) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "ProducerConsumerWS: target attribute is required";

    ProducerConsumerWSRewriter T;
    T.target_ = target.value();
    if (auto cluster_dims = f->GetAttr<Array<Integer>>("cluster_dims")) {
      T.cluster_size_ = ClusterSize(cluster_dims.value());
    } else {
      PostOrderVisit(f->body, [&](const ObjectRef &node) {
        const auto *block = node.as<BlockNode>();
        if (block == nullptr || !block->annotations.count("cluster_dims")) {
          return;
        }
        auto cluster_dims =
            block->annotations.Get("cluster_dims")->try_cast<Array<Integer>>();
        ICHECK(cluster_dims.has_value())
            << "cluster_dims must be an Array<Integer>";
        int cluster_size = ClusterSize(cluster_dims.value());
        ICHECK(T.cluster_size_ == 1 || T.cluster_size_ == cluster_size)
            << "conflicting cluster_dims annotations in one PrimFunc";
        T.cluster_size_ = cluster_size;
      });
    }
    if (auto handoff_role = f->GetAttr<String>(kCrossHandlerHandoffRole)) {
      T.cross_handler_handoff_role_ = handoff_role.value();
    }
    T.cross_handler_handoff_enabled_ =
        f->HasNonzeroAttr(kCrossHandlerHandoffEnabled);
    if (auto roles = f->GetAttr<Array<String>>(kDataflowParamRoles)) {
      ICHECK_EQ(roles.value().size(), f->params.size())
          << kDataflowParamRoles << " must cover every PrimFunc parameter";
      for (size_t i = 0; i < roles.value().size(); ++i) {
        if (roles.value()[i] != kHandoffStageCountRole) {
          continue;
        }
        ICHECK(!f->buffer_map.count(f->params[i]))
            << "handoff stage count must be a scalar parameter";
        ICHECK(!T.handoff_stage_count_var_.defined())
            << "PrimFunc has multiple handoff stage-count parameters";
        T.handoff_stage_count_var_ = f->params[i];
      }
    }
    f.CopyOnWrite()->body = T(f->body);
    if (rejection_reason != nullptr) {
      *rejection_reason = T.rejection_reason_;
    }

    if (T.ws_transformed_) {
      f = WithAttr(std::move(f), kTiledWSApplied, IntImm(DataType::Int(32), 1));
    }
    return f;
  }

private:
  static int ClusterSize(const Array<Integer> &cluster_dims) {
    int cluster_size = 1;
    for (const Integer &dim : cluster_dims) {
      ICHECK_GT(dim->value, 0) << "cluster_dims must be positive";
      cluster_size *= static_cast<int>(dim->value);
    }
    return cluster_size;
  }

  // --- Track threadIdx.x binding ---
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        thread_iv_ = iv;
        Optional<PrimExpr> old_num_threads = num_threads_;
        num_threads_ = std::nullopt;
        AttrStmt attr = Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));
        if (num_threads_.defined()) {
          PrimExpr nt = num_threads_.value();
          thread_iv_.CopyOnWrite()->dom = {0, nt};
          attr.CopyOnWrite()->node = thread_iv_;
          attr.CopyOnWrite()->value = nt;
        }
        num_threads_ = old_num_threads;
        thread_iv_ = {};
        return attr;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // --- Find the block containing the pipeline loop ---
  Stmt VisitStmt_(const BlockRealizeNode *op) final {
    if (!thread_iv_.defined())
      return StmtExprMutator::VisitStmt_(op);

    const Block &orig_block = op->block;

    // Find the pipelined loop.
    Optional<For> pipeline_loop_opt = FindPipelineLoop(orig_block->body);
    if (!pipeline_loop_opt.defined())
      return StmtExprMutator::VisitStmt_(op);
    For pipeline_loop = pipeline_loop_opt.value();

    auto num_stages_anno = pipeline_loop->annotations.Get("num_stages");
    if (!num_stages_anno)
      return StmtExprMutator::VisitStmt_(op);
    int num_stages =
        static_cast<int>(Downcast<Integer>(num_stages_anno.value())->value);
    if (num_stages < 1)
      return StmtExprMutator::VisitStmt_(op);

    // Flatten the loop body.
    Array<Stmt> flat_stmts;
    Stmt loop_body = pipeline_loop->body;
    if (auto *realize = loop_body.as<BlockRealizeNode>()) {
      loop_body = realize->block->body;
    }
    // Unwrap LetStmt chain that dominates the whole loop body.
    std::vector<std::pair<Var, PrimExpr>> outer_let_bindings;
    while (const auto *let = loop_body.as<LetStmtNode>()) {
      outer_let_bindings.emplace_back(let->var, let->value);
      loop_body = let->body;
    }
    // Unwrap a single IfThenElse wrapper (no else branch) so that
    // TMA producers inside conditional loop bodies can be classified.
    // Keep LetStmt chains inside the conditional separate so they stay
    // dominated by the original guard after rebuilding WS branches.
    Optional<PrimExpr> loop_body_condition;
    std::vector<std::pair<Var, PrimExpr>> inner_let_bindings;
    if (const auto *if_stmt = loop_body.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined()) {
        // Peel LetStmt chain from inside the conditional body. These
        // bindings must remain inside the guarded region.
        Stmt inner = if_stmt->then_case;
        while (const auto *let = inner.as<LetStmtNode>()) {
          inner_let_bindings.emplace_back(let->var, let->value);
          inner = let->body;
        }
        loop_body_condition = if_stmt->condition;
        loop_body = inner;
      }
    }
    FlattenSeqStmt(loop_body, &flat_stmts);
    Array<Stmt> materialized_stmts;
    Array<Stmt> detached_handoff_producer_stmts;
    for (const Stmt &stmt : flat_stmts) {
      if (IsResidentPipelineTransferStmt(stmt)) {
        continue;
      }
      if (IsDetachedHandoffProducerStmt(stmt)) {
        detached_handoff_producer_stmts.push_back(stmt);
        continue;
      }
      materialized_stmts.push_back(stmt);
    }
    flat_stmts = std::move(materialized_stmts);

    if (!detached_handoff_producer_stmts.empty() &&
        (!cross_handler_handoff_enabled_ ||
         cross_handler_handoff_role_ != "producer" ||
         !handoff_stage_count_var_.defined())) {
      rejection_reason_ =
          "detached handoff transfers require typed producer metadata";
      return StmtExprMutator::VisitStmt_(op);
    }

    // Classify statements into producer (TMA/SIMT copy) and consumer.
    std::vector<TileStmtKind> kinds;
    int num_tma = 0;
    int num_simt = 0;
    for (const Stmt &s : flat_stmts) {
      auto k = ClassifyStmt(s, target_);
      kinds.push_back(k);
      if (k == TileStmtKind::kTmaProducer)
        ++num_tma;
      if (k == TileStmtKind::kSimtProducer)
        ++num_simt;
    }

    // Require at least one TMA producer.
    if (num_tma == 0)
      return StmtExprMutator::VisitStmt_(op);

    // --- Build the WS transformation ---
    return BuildWSBlock(op, orig_block, pipeline_loop, num_stages, flat_stmts,
                        kinds, detached_handoff_producer_stmts,
                        outer_let_bindings, inner_let_bindings,
                        loop_body_condition);
  }

  Stmt
  BuildWSBlock(const BlockRealizeNode *orig_realize, const Block &orig_block,
               const For &pipeline_loop, int num_stages,
               const Array<Stmt> &flat_stmts,
               const std::vector<TileStmtKind> &kinds,
               const Array<Stmt> &detached_handoff_producer_stmts,
               const std::vector<std::pair<Var, PrimExpr>> &outer_let_bindings,
               const std::vector<std::pair<Var, PrimExpr>> &inner_let_bindings,
               Optional<PrimExpr> loop_body_condition = Optional<PrimExpr>()) {
    Var loop_var = pipeline_loop->loop_var;
    PrimExpr loop_min = pipeline_loop->min;
    PrimExpr loop_extent = pipeline_loop->extent;
    PrimExpr linear_idx = loop_var - loop_min;

    // A pipeline nested under ordinary serial loops can reuse the same
    // barriers across invocations.  Prefer a loop-derived global iteration
    // when the invocation path is provably contiguous; sparse or otherwise
    // unprovable paths retain the persistent phase-counter protocol.
    PipelineInvocationAnalysis invocation =
        AnalyzePipelineInvocations(orig_block->body, pipeline_loop, linear_idx);
    bool use_affine_iteration =
        invocation.affine_iteration.defined() && !loop_body_condition.defined();
    PrimExpr pipeline_iteration =
        use_affine_iteration ? invocation.affine_iteration.value() : linear_idx;
    PrimExpr base_stage_expr = FloorMod(pipeline_iteration, num_stages);
    PrimExpr base_parity_expr =
        FloorMod(FloorDiv(pipeline_iteration, num_stages), 2);
    bool phase_counter_block_scope =
        invocation.may_repeat && !use_affine_iteration;
    // Guarded iterations also need counters so skipped iterations do not
    // advance barrier state.
    bool needs_phase_counter =
        loop_body_condition.defined() || phase_counter_block_scope;
    Optional<PhaseCounter> producer_phase_counter;
    Optional<PhaseCounter> consumer_phase_counter;
    PrimExpr p_stage_expr = base_stage_expr;
    PrimExpr p_parity_expr = base_parity_expr;
    PrimExpr c_stage_expr = base_stage_expr;
    PrimExpr c_parity_expr = base_parity_expr;
    PrimExpr p_iteration_expr = pipeline_iteration;
    PrimExpr c_iteration_expr = pipeline_iteration;
    if (needs_phase_counter) {
      producer_phase_counter = PhaseCounter::Create("producer_phase_cnt");
      consumer_phase_counter = PhaseCounter::Create("consumer_phase_cnt");
      p_stage_expr = producer_phase_counter.value().StageExpr(num_stages);
      p_parity_expr = producer_phase_counter.value().ParityExpr(num_stages);
      c_stage_expr = consumer_phase_counter.value().StageExpr(num_stages);
      c_parity_expr = consumer_phase_counter.value().ParityExpr(num_stages);
      p_iteration_expr = producer_phase_counter.value().Load();
      c_iteration_expr = consumer_phase_counter.value().Load();
    }

    PrimExpr consumer_extent = thread_iv_->dom->extent;
    PrimExpr producer_extent = IntImm(DataType::Int(32), 128);
    if (auto producer_threads =
            pipeline_loop->annotations.Get(kPipelineProducerThreads)) {
      const auto *value = producer_threads.value().as<IntImmNode>();
      ICHECK(value != nullptr && value->value > 0)
          << kPipelineProducerThreads
          << " must be a compile-time positive integer";
      producer_extent = IntImm(DataType::Int(32), value->value);
    }
    common_prelude_rewrites_.clear();

    bool has_simt_producer = false;
    bool has_cp_async_producer = false;
    bool all_tma_producers_are_copies = true;
    int num_producer_groups = 0;
    std::vector<bool> multicast_producer_groups;
    std::vector<int64_t> producer_cluster_masks;
    std::vector<PrimExpr> producer_transaction_bytes;
    std::vector<int> producer_partitions;
    std::vector<int> producer_buffer_versions;
    std::vector<bool> streamed_cluster_push_groups;
    std::vector<int> streamed_cluster_push_partition_counts;
    std::vector<PrimExpr> streamed_cluster_push_credit_targets;
    std::vector<bool> handoff_consumer_groups;
    std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual>
        pipeline_buffer_versions;
    for (auto k : kinds) {
      if (k == TileStmtKind::kTmaProducer) {
        ++num_producer_groups;
      }
      if (k == TileStmtKind::kSimtProducer)
        has_simt_producer = true;
      if (k == TileStmtKind::kCpAsyncProducer)
        has_cp_async_producer = true;
    }
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (kinds[i] != TileStmtKind::kTmaProducer) {
        continue;
      }
      Optional<Call> call = GetEvaluateCallInSimpleWrapper(flat_stmts[i]);
      auto tile_op =
          call.defined() ? ParseOperator(call.value()) : TileOperator();
      const auto *copy = tile_op.as<CopyNode>();
      all_tma_producers_are_copies &= copy != nullptr;
      int64_t cluster_mask = GetCopyClusterMask(copy);
      multicast_producer_groups.push_back(cluster_mask > 0);
      producer_cluster_masks.push_back(cluster_mask);
      producer_partitions.push_back(GetCopyProducerPartition(copy));
      handoff_consumer_groups.push_back(IsHandoffConsumerCopy(copy));
      bool streamed_cluster_push = IsStreamedClusterPush(copy);
      streamed_cluster_push_groups.push_back(streamed_cluster_push);
      streamed_cluster_push_partition_counts.push_back(
          GetStreamedClusterPushPartitionCount(copy));
      streamed_cluster_push_credit_targets.push_back(
          streamed_cluster_push ? GetStreamedClusterPushCreditTarget(copy)
                                : PrimExpr(IntImm(DataType::Int(32), 0)));
      int buffer_versions = GetCopyPipelineBufferVersions(copy, num_stages);
      producer_buffer_versions.push_back(buffer_versions);
      if (streamed_cluster_push) {
        auto receive_stages = copy->annotations.Get(kResharedReceiveStages);
        const auto *value = receive_stages.has_value()
                                ? receive_stages.value().as<IntImmNode>()
                                : nullptr;
        if (value == nullptr || value->value != buffer_versions) {
          rejection_reason_ =
              "streamed cluster push receive stages must match the typed "
              "pipeline buffer versions";
          return ffi::GetRef<BlockRealize>(orig_realize);
        }
      }
      if (copy != nullptr && buffer_versions > 1 &&
          copy->annotations.count(kPipelineBufferVersions)) {
        auto [it, inserted] =
            pipeline_buffer_versions.emplace(copy->dst->data, buffer_versions);
        ICHECK(inserted || it->second == buffer_versions)
            << "pipeline copies targeting one buffer disagree on "
            << kPipelineBufferVersions;
      }
      producer_transaction_bytes.push_back(
          copy == nullptr ? PrimExpr(0) : CopyTransactionBytes(copy));
    }
    std::vector<int> detached_handoff_producer_partitions;
    detached_handoff_producer_partitions.reserve(
        detached_handoff_producer_stmts.size());
    for (const Stmt &stmt : detached_handoff_producer_stmts) {
      TileOperator tile_op = GetSimpleTileOperator(stmt);
      const CopyNode *copy = tile_op.as<CopyNode>();
      if (copy == nullptr ||
          ClassifyCopy(copy, target_) != TileStmtKind::kTmaProducer ||
          !HasGlobalToSharedCopyShape(copy)) {
        rejection_reason_ = "detached handoff transfer requires a legal TMA "
                            "global-to-shared copy";
        return ffi::GetRef<BlockRealize>(orig_realize);
      }
      if (GetCopyClusterMask(copy) != 0) {
        rejection_reason_ = "detached handoff multicast requires a "
                            "cluster-owned arena protocol";
        return ffi::GetRef<BlockRealize>(orig_realize);
      }
      detached_handoff_producer_partitions.push_back(
          GetCopyProducerPartition(copy));
    }
    bool any_handoff_consumer = std::any_of(handoff_consumer_groups.begin(),
                                            handoff_consumer_groups.end(),
                                            [](bool value) { return value; });
    bool all_handoff_consumer = !handoff_consumer_groups.empty() &&
                                std::all_of(handoff_consumer_groups.begin(),
                                            handoff_consumer_groups.end(),
                                            [](bool value) { return value; });
    if (any_handoff_consumer && (!cross_handler_handoff_enabled_ ||
                                 cross_handler_handoff_role_ != "consumer" ||
                                 !handoff_stage_count_var_.defined())) {
      rejection_reason_ =
          "handoff consumer transfers require typed consumer metadata";
      return ffi::GetRef<BlockRealize>(orig_realize);
    }
    bool has_partitioned_producer =
        std::any_of(producer_partitions.begin(), producer_partitions.end(),
                    [](int partition) { return partition >= 0; });
    bool has_partitioned_handoff_producer =
        std::any_of(detached_handoff_producer_partitions.begin(),
                    detached_handoff_producer_partitions.end(),
                    [](int partition) { return partition >= 0; });
    bool has_multicast_producer = std::any_of(multicast_producer_groups.begin(),
                                              multicast_producer_groups.end(),
                                              [](bool value) { return value; });
    bool has_streamed_cluster_push = std::any_of(
        streamed_cluster_push_groups.begin(),
        streamed_cluster_push_groups.end(), [](bool value) { return value; });
    bool streamed_cluster_push_one_shot =
        has_streamed_cluster_push && invocation.static_invocation_count > 0 &&
        invocation.static_invocation_count <=
            std::numeric_limits<int32_t>::max();
    auto stage_for_versions = [](PrimExpr iteration, int versions) -> PrimExpr {
      if (versions == 1) {
        return IntImm(DataType::Int(32), 0);
      }
      return FloorMod(std::move(iteration),
                      IntImm(DataType::Int(32), versions));
    };
    auto parity_for_versions = [](PrimExpr iteration,
                                  int versions) -> PrimExpr {
      return FloorMod(
          FloorDiv(std::move(iteration), IntImm(DataType::Int(32), versions)),
          IntImm(DataType::Int(32), 2));
    };
    const int64_t *static_loop_extent = as_const_int(loop_extent);
    auto reject_multicast = [&](std::string reason) -> Stmt {
      rejection_reason_ = std::move(reason);
      return ffi::GetRef<BlockRealize>(orig_realize);
    };
    if (has_multicast_producer && static_loop_extent == nullptr) {
      return reject_multicast(
          "dynamic multicast loop extent requires synchronous fallback");
    }
    if (has_multicast_producer && invocation.may_repeat) {
      return reject_multicast(
          "nested multicast pipeline requires synchronous fallback");
    }
    if (has_multicast_producer && cluster_size_ <= 1) {
      return reject_multicast(
          "multicast requires a multi-CTA cluster topology");
    }
    if (has_streamed_cluster_push && cluster_size_ <= 1) {
      return reject_multicast(
          "streamed cluster push requires a multi-CTA cluster topology");
    }
    if (has_multicast_producer && cluster_size_ > 16) {
      return reject_multicast(
          "TMA multicast masks support at most 16 CTA ranks");
    }
    int64_t valid_cluster_mask = (int64_t{1} << cluster_size_) - 1;
    for (int64_t mask : producer_cluster_masks) {
      if ((mask & ~valid_cluster_mask) != 0) {
        return reject_multicast(
            "multicast mask references a rank outside cluster_dims");
      }
    }
    if (has_multicast_producer &&
        (has_simt_producer || has_cp_async_producer)) {
      return reject_multicast(
          "multicast cannot share a forward barrier with SIMT or cp.async "
          "producers");
    }
    for (int g = 0; g < num_producer_groups; ++g) {
      if (handoff_consumer_groups[g] && multicast_producer_groups[g]) {
        return reject_multicast("handoff consumer multicast requires a "
                                "cluster-owned arena protocol");
      }
    }

    // --- Barrier allocation ---
    // Layout: [fwd rings] [backpressure rings]
    // [prelude_0..prelude_{P-1}] [consumer_ready_0..consumer_ready_{C*S-1}]
    // [consumer_consumed_0..consumer_consumed_{C*S-1}]
    // Each TMA group uses the typed version count of its destination buffer.
    // When SIMT producers are present, all producer types share the same
    // barrier group — the last forward arrive covers everything.
    std::vector<int> forward_barrier_bases;
    int num_fwd = 0;
    for (int g = 0; g < num_producer_groups; ++g) {
      forward_barrier_bases.push_back(num_fwd);
      num_fwd +=
          streamed_cluster_push_groups[g] && streamed_cluster_push_one_shot
              ? static_cast<int>(invocation.static_invocation_count)
          : multicast_producer_groups[g] ? static_cast<int>(*static_loop_extent)
                                         : producer_buffer_versions[g];
    }
    std::vector<int> backpressure_barrier_bases;
    int num_bp = 0;
    for (int g = 0; g < num_producer_groups; ++g) {
      backpressure_barrier_bases.push_back(num_bp);
      num_bp +=
          streamed_cluster_push_groups[g] && streamed_cluster_push_one_shot
              ? static_cast<int>(invocation.static_invocation_count)
              : producer_buffer_versions[g];
    }
    std::vector<int> cluster_backpressure_bases(num_producer_groups, -1);
    int num_cluster_backpressure = 0;
    for (int g = 0; g < num_producer_groups; ++g) {
      if (multicast_producer_groups[g]) {
        cluster_backpressure_bases[g] = num_cluster_backpressure;
        num_cluster_backpressure += producer_buffer_versions[g];
      }
    }
    Optional<Buffer> cluster_backpressure_buf;
    if (num_cluster_backpressure > 0) {
      cluster_backpressure_buf = CreateClusterMBarrierBuffer(
          "cluster_backpressure_mbarrier", num_cluster_backpressure);
    }

    buffer_data_to_buffer_ =
        BufferDataToBufferCollector::Collect(orig_block->body);
    Array<Stmt> consumer_compute_stmts;
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (!IsProducer(kinds[i])) {
        consumer_compute_stmts.push_back(flat_stmts[i]);
      }
    }
    std::vector<int> consumer_sync_groups(consumer_compute_stmts.size(), -1);
    std::vector<int> consumer_sync_release_positions;
    int num_consumer_sync_groups = 0;
    for (size_t i = 0; i < consumer_compute_stmts.size(); ++i) {
      if (PipelineConsumerCopyNeedsPartitionSync(consumer_compute_stmts[i],
                                                 target_)) {
        consumer_sync_groups[i] = num_consumer_sync_groups++;
        Optional<Var> write_buffer_data =
            ExtractProducerWriteBufferData(consumer_compute_stmts[i]);
        ICHECK(write_buffer_data.defined());
        BufferUsePositions positions = AnalyzeConsumerBufferUsePositions(
            consumer_compute_stmts, write_buffer_data.value(),
            buffer_data_to_buffer_);
        ICHECK_GE(positions.last_access, static_cast<int>(i));
        consumer_sync_release_positions.push_back(positions.last_access + 1);
      }
    }
    int num_consumer_sync_barriers = 2 * num_consumer_sync_groups * num_stages;

    Array<Stmt> prelude_stmts;
    CollectPreludeStmtsToPipelineLoop(orig_block->body, pipeline_loop,
                                      &prelude_stmts);
    std::vector<PreludeTmaLoadPlan> prelude_tma_plans;
    for (const Stmt &stmt : prelude_stmts) {
      if (ClassifyStmt(stmt, target_) != TileStmtKind::kTmaProducer) {
        continue;
      }
      Optional<Var> write_buffer_data = ExtractProducerWriteBufferData(stmt);
      if (!write_buffer_data.defined()) {
        continue;
      }
      BufferUsePositions positions = AnalyzeConsumerBufferUsePositions(
          consumer_compute_stmts, write_buffer_data.value(),
          buffer_data_to_buffer_);
      if (positions.first_read < 0) {
        continue;
      }
      prelude_tma_plans.push_back({stmt, positions.first_read});
    }

    int total_barriers = num_fwd + num_bp +
                         detached_handoff_producer_stmts.size() +
                         prelude_tma_plans.size() + num_consumer_sync_barriers;
    Buffer barrier_buf =
        CreateMBarrierBuffer(injected_mbarrier_name_, total_barriers);
    // arrive_counts are computed later (after producer_extent is finalized).

    std::vector<int> wait_insert_pos(num_producer_groups, 0);
    std::vector<int> arrive_insert_pos(
        num_producer_groups, static_cast<int>(consumer_compute_stmts.size()));
    int access_group_idx = 0;
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (kinds[i] != TileStmtKind::kTmaProducer) {
        continue;
      }
      Optional<Var> write_buffer_data =
          ExtractProducerWriteBufferData(flat_stmts[i]);
      if (write_buffer_data.defined()) {
        BufferUsePositions positions = AnalyzeConsumerBufferUsePositions(
            consumer_compute_stmts, write_buffer_data.value(),
            buffer_data_to_buffer_);
        if (positions.first_read >= 0) {
          wait_insert_pos[access_group_idx] = positions.first_read;
          arrive_insert_pos[access_group_idx] = positions.last_access + 1;
        } else if (positions.last_access >= 0) {
          wait_insert_pos[access_group_idx] = 0;
          arrive_insert_pos[access_group_idx] = positions.last_access + 1;
        }
      }
      ++access_group_idx;
    }

    // --- Adjust wait positions for SIMT/cp.async producers ---
    // SIMT and cp.async producers tie their completion to all forward barriers.
    // If a consumer reads any such shared destination before the first TMA
    // read, pull all waits earlier so the async producer is also covered.
    if (has_simt_producer || has_cp_async_producer) {
      int earliest_async_read = static_cast<int>(consumer_compute_stmts.size());
      for (size_t i = 0; i < flat_stmts.size(); ++i) {
        if (kinds[i] != TileStmtKind::kSimtProducer &&
            kinds[i] != TileStmtKind::kCpAsyncProducer) {
          continue;
        }
        int first_read = FindFirstAsyncProducerConsumerRead(
            flat_stmts[i], consumer_compute_stmts, buffer_data_to_buffer_);
        earliest_async_read = std::min(earliest_async_read, first_read);
      }
      // Pull all wait positions earlier if needed.
      for (int g = 0; g < num_producer_groups; ++g) {
        wait_insert_pos[g] = std::min(wait_insert_pos[g], earliest_async_read);
      }
    }

    // --- Determine if TMA barriers can be merged ---
    // Pure-TMA producers in one partition can share barriers when their
    // lifetimes already coincide. Distinct typed producer partitions can also
    // share one completion group by waiting before the earliest use and
    // releasing after the latest use. Each partition contributes its own
    // arrive-and-expect-tx, so this does not depend on producer warp ordering.
    bool partitions_can_share_barrier = true;
    for (int g = 1; g < num_producer_groups; ++g) {
      if (producer_partitions[g] != producer_partitions[0]) {
        partitions_can_share_barrier = false;
        break;
      }
    }
    bool versions_can_share_barrier = true;
    for (int g = 1; g < num_producer_groups; ++g) {
      if (producer_buffer_versions[g] != producer_buffer_versions[0]) {
        versions_can_share_barrier = false;
        break;
      }
    }
    bool can_merge_tma_forward_barriers =
        (num_producer_groups > 1) && !has_multicast_producer &&
        !has_streamed_cluster_push && !has_simt_producer &&
        !has_cp_async_producer &&
        (!any_handoff_consumer || all_handoff_consumer) &&
        (!any_handoff_consumer || versions_can_share_barrier);
    bool use_one_shot_completion =
        can_merge_tma_forward_barriers && !versions_can_share_barrier &&
        all_tma_producers_are_copies &&
        invocation.static_invocation_count > 0 &&
        invocation.static_invocation_count <=
            std::numeric_limits<int32_t>::max() &&
        (!invocation.may_repeat || use_affine_iteration);
    bool can_merge_tma_barriers =
        can_merge_tma_forward_barriers && versions_can_share_barrier;
    bool merge_forward_across_producer_partitions =
        can_merge_tma_forward_barriers && has_partitioned_producer &&
        !partitions_can_share_barrier;
    bool merge_across_producer_partitions = can_merge_tma_barriers &&
                                            has_partitioned_producer &&
                                            !partitions_can_share_barrier;
    if (merge_across_producer_partitions) {
      int earliest_wait =
          *std::min_element(wait_insert_pos.begin(), wait_insert_pos.end());
      int latest_release =
          *std::max_element(arrive_insert_pos.begin(), arrive_insert_pos.end());
      std::fill(wait_insert_pos.begin(), wait_insert_pos.end(), earliest_wait);
      std::fill(arrive_insert_pos.begin(), arrive_insert_pos.end(),
                latest_release);
    } else if (can_merge_tma_barriers) {
      can_merge_tma_barriers = partitions_can_share_barrier;
      for (int g = 1; g < num_producer_groups; ++g) {
        if (wait_insert_pos[g] != wait_insert_pos[0] ||
            arrive_insert_pos[g] != arrive_insert_pos[0]) {
          can_merge_tma_barriers = false;
          break;
        }
      }
    }
    std::vector<int> forward_wait_insert_pos = wait_insert_pos;
    if (can_merge_tma_forward_barriers) {
      int earliest_wait =
          *std::min_element(wait_insert_pos.begin(), wait_insert_pos.end());
      std::fill(forward_wait_insert_pos.begin(), forward_wait_insert_pos.end(),
                earliest_wait);
      num_fwd = use_one_shot_completion
                    ? static_cast<int>(invocation.static_invocation_count)
                    : num_stages;
      std::fill(forward_barrier_bases.begin(), forward_barrier_bases.end(), 0);
    }
    if (can_merge_tma_barriers) {
      // Equal-sized rings can also share one backpressure group.
      int merged_versions = producer_buffer_versions[0];
      num_bp = merged_versions;
      std::fill(backpressure_barrier_bases.begin(),
                backpressure_barrier_bases.end(), 0);
    }
    total_barriers = num_fwd + num_bp + detached_handoff_producer_stmts.size() +
                     prelude_tma_plans.size() + num_consumer_sync_barriers;
    barrier_buf = CreateMBarrierBuffer(injected_mbarrier_name_, total_barriers);

    constexpr int kWarpgroupThreadCount = 128;
    int consumer_warpgroup_count = 0;
    if (const int64_t *threads = as_const_int(consumer_extent);
        threads != nullptr && *threads >= kWarpgroupThreadCount &&
        *threads % kWarpgroupThreadCount == 0) {
      consumer_warpgroup_count =
          static_cast<int>(*threads / kWarpgroupThreadCount);
    }
    const int backpressure_barrier_groups =
        can_merge_tma_barriers ? 1 : num_producer_groups;
    std::vector<bool> elect_backpressure_release(backpressure_barrier_groups,
                                                 false);
    if (consumer_warpgroup_count > 0 && !has_multicast_producer) {
      for (int g = 0; g < backpressure_barrier_groups; ++g) {
        int release_pos = arrive_insert_pos[g];
        if (release_pos <= 0 ||
            release_pos > static_cast<int>(consumer_compute_stmts.size())) {
          continue;
        }
        const Stmt &release_predecessor =
            consumer_compute_stmts[release_pos - 1];
        elect_backpressure_release[g] =
            ContainsDrainingWgmmaWait(release_predecessor) ||
            IsImplicitlySynchronousWgmma(
                release_predecessor,
                static_cast<int>(*as_const_int(consumer_extent)), target_);
      }
    }
    auto make_backpressure_release = [&](int group,
                                         PrimExpr barrier_id) -> Stmt {
      int producer_group = can_merge_tma_barriers ? 0 : group;
      if (streamed_cluster_push_groups[producer_group]) {
        PrimExpr target_rank =
            streamed_cluster_push_credit_targets[producer_group];
        return elect_backpressure_release[group]
                   ? MakeWarpgroupLeaderArriveClusterBarrier(
                         barrier_buf, std::move(barrier_id),
                         std::move(target_rank))
                   : MakeArriveClusterBarrier(barrier_buf,
                                              std::move(barrier_id),
                                              std::move(target_rank));
      }
      return elect_backpressure_release[group]
                 ? MakeWarpgroupLeaderArriveBarrier(barrier_buf,
                                                    std::move(barrier_id))
                 : MakeArriveBarrier(barrier_buf, std::move(barrier_id));
    };
    bool externalize_static_wgmma_drain =
        !needs_phase_counter && static_loop_extent != nullptr &&
        std::any_of(elect_backpressure_release.begin(),
                    elect_backpressure_release.end(),
                    [](bool elected) { return elected; });
    bool unroll_static_affine_wgmma = externalize_static_wgmma_drain &&
                                      use_affine_iteration &&
                                      invocation.role_scope_liftable;

    std::vector<Array<Stmt>> producer_loop_prefix_stmts(num_producer_groups);
    std::vector<bool> moved_compute_stmts(consumer_compute_stmts.size(), false);
    int compute_cursor = 0;
    for (int ti = 0; ti < num_producer_groups; ++ti) {
      int wait_pos = wait_insert_pos[ti];
      if (wait_pos <= compute_cursor) {
        compute_cursor = std::max(compute_cursor, wait_pos);
        continue;
      }
      bool all_movable = true;
      for (int ci = compute_cursor; ci < wait_pos; ++ci) {
        if (!IsProducerMovableLoopPrefixStmt(consumer_compute_stmts[ci],
                                             target_)) {
          all_movable = false;
          break;
        }
      }
      if (all_movable) {
        for (int ci = compute_cursor; ci < wait_pos; ++ci) {
          producer_loop_prefix_stmts[ti].push_back(consumer_compute_stmts[ci]);
          moved_compute_stmts[ci] = true;
        }
      }
      compute_cursor = wait_pos;
    }

    bool producer_needs_full_thread_extent = false;
    for (size_t i = 0;
         i < flat_stmts.size() && !producer_needs_full_thread_extent; ++i) {
      if (kinds[i] == TileStmtKind::kSimtProducer ||
          IsSyncGlobalToSharedCopyLikeStmt(flat_stmts[i], target_)) {
        producer_needs_full_thread_extent = true;
      }
    }
    if (!producer_needs_full_thread_extent) {
      for (const auto &prefix_stmts : producer_loop_prefix_stmts) {
        for (const auto &stmt : prefix_stmts) {
          if (IsSyncGlobalToSharedCopyLikeStmt(stmt, target_)) {
            producer_needs_full_thread_extent = true;
            break;
          }
        }
        if (producer_needs_full_thread_extent) {
          break;
        }
      }
    }
    if (producer_needs_full_thread_extent) {
      // LowerTileOp will materialize these producer-side sync copies into
      // explicit SIMT global->shared loops. Keep the producer partition at the
      // original thread extent so the lowered thread mapping stays valid.
      producer_extent = consumer_extent;
    }
    if (has_partitioned_producer || has_partitioned_handoff_producer) {
      if (has_simt_producer || has_cp_async_producer) {
        return reject_multicast(
            "partitioned TMA producers cannot share a producer partition with "
            "SIMT or cp.async transfers");
      }
      const int64_t *static_producer_extent = as_const_int(producer_extent);
      if (static_producer_extent == nullptr ||
          *static_producer_extent % 32 != 0) {
        return reject_multicast(
            "partitioned TMA producers require a static warp-aligned producer "
            "thread extent");
      }
      int producer_warps = static_cast<int>(*static_producer_extent / 32);
      if (has_partitioned_producer) {
        for (int partition : producer_partitions) {
          if (partition < 0 || partition >= producer_warps) {
            return reject_multicast(
                "typed TMA producer partition exceeds the physical producer "
                "thread budget");
          }
        }
      }
      for (int partition : detached_handoff_producer_partitions) {
        if (partition >= producer_warps) {
          return reject_multicast(
              "typed handoff producer partition exceeds the physical producer "
              "thread budget");
        }
      }
    }

    auto guard_partition = [&](int partition, Stmt stmt) -> Stmt {
      if (partition < 0) {
        return stmt;
      }
      PrimExpr producer_warp =
          FloorDiv(thread_iv_->var, IntImm(DataType::Int(32), 32));
      return IfThenElse(EQ(producer_warp, IntImm(DataType::Int(32), partition)),
                        stmt);
    };
    auto guard_producer_partition = [&](int group, Stmt stmt) -> Stmt {
      return guard_partition(producer_partitions[group], std::move(stmt));
    };
    auto guard_pure_tma_issuer_warp = [&](int group, Stmt stmt) -> Stmt {
      // A pure TMA stage is reused by the same elected issuer that launches
      // the transfer. Letting every producer warp wait independently on the
      // one-bit mbarrier parity permits a delayed warp to miss two phase
      // transitions (ABA) and wait forever after the final generation. Keep
      // SIMT/cp.async and multicast participation unchanged; for ordinary TMA
      // the issuer warp alone owns the reuse wait.
      if (has_simt_producer || has_cp_async_producer ||
          multicast_producer_groups[group] || producer_partitions[group] >= 0) {
        return stmt;
      }
      // Producer bodies are subsequently rewritten from physical threadIdx.x
      // to a producer-local index. Add consumer_extent here so that rewrite
      // recovers the physical warp used by tl_shuffle_elect<producer_extent>.
      // This matters when the consumer extent is not a producer-warpgroup
      // multiple (for example, 32 consumer + 128 producer threads).
      PrimExpr physical_thread = thread_iv_->var + consumer_extent;
      PrimExpr physical_warp =
          FloorDiv(physical_thread, IntImm(DataType::Int(32), 32));
      PrimExpr producer_warp_count =
          FloorDiv(producer_extent, IntImm(DataType::Int(32), 32));
      return IfThenElse(EQ(FloorMod(physical_warp, producer_warp_count),
                           IntImm(DataType::Int(32), 0)),
                        std::move(stmt));
    };
    auto guard_handoff_consumer_prefix = [&](int group, PrimExpr iteration,
                                             Stmt stmt) -> Stmt {
      if (!handoff_consumer_groups[group]) {
        return stmt;
      }
      ICHECK(handoff_stage_count_var_.defined());
      PrimExpr stage_count =
          cast(iteration.dtype(), handoff_stage_count_var_.value());
      return IfThenElse(GE(iteration, stage_count), std::move(stmt));
    };

    // --- Compute arrive_counts (after producer_extent is finalized) ---
    // Forward arrive_count:
    //   - Pure TMA (possibly merged): 1 (leader thread only)
    //   - Mixed TMA with SIMT/cp.async: producer_extent (all producer threads)
    PrimExpr fwd_arrive_count =
        use_one_shot_completion ? PrimExpr(IntImm(DataType::Int(32), 1))
        : merge_forward_across_producer_partitions
            ? PrimExpr(IntImm(DataType::Int(32), num_producer_groups))
            : ((!has_simt_producer && !has_cp_async_producer)
                   ? PrimExpr(IntImm(DataType::Int(32), 1))
                   : producer_extent);
    Array<PrimExpr> arrive_counts;
    if (can_merge_tma_forward_barriers) {
      for (int i = 0; i < num_fwd; ++i) {
        arrive_counts.push_back(fwd_arrive_count);
      }
    } else {
      for (int g = 0; g < num_producer_groups; ++g) {
        int ring_size =
            streamed_cluster_push_groups[g] && streamed_cluster_push_one_shot
                ? static_cast<int>(invocation.static_invocation_count)
            : multicast_producer_groups[g]
                ? static_cast<int>(*static_loop_extent)
                : producer_buffer_versions[g];
        PrimExpr group_arrive_count =
            streamed_cluster_push_groups[g]
                ? PrimExpr(IntImm(DataType::Int(32),
                                  streamed_cluster_push_partition_counts[g]))
                : fwd_arrive_count;
        for (int stage = 0; stage < ring_size; ++stage) {
          arrive_counts.push_back(group_arrive_count);
        }
      }
    }
    for (int g = 0; g < backpressure_barrier_groups; ++g) {
      PrimExpr arrive_count =
          elect_backpressure_release[g]
              ? PrimExpr(IntImm(DataType::Int(32), consumer_warpgroup_count))
              : consumer_extent;
      int versions =
          streamed_cluster_push_groups[g] && streamed_cluster_push_one_shot
              ? static_cast<int>(invocation.static_invocation_count)
              : producer_buffer_versions[g];
      for (int stage = 0; stage < versions; ++stage) {
        arrive_counts.push_back(arrive_count);
      }
    }
    for (size_t i = 0; i < detached_handoff_producer_stmts.size(); ++i) {
      arrive_counts.push_back(IntImm(DataType::Int(32), 1));
    }
    for (size_t i = 0; i < prelude_tma_plans.size(); ++i) {
      arrive_counts.push_back(IntImm(DataType::Int(32), 1));
    }
    for (int i = 0; i < num_consumer_sync_barriers; ++i) {
      arrive_counts.push_back(consumer_extent);
    }
    Array<PrimExpr> cluster_backpressure_arrive_counts;
    for (int g = 0; g < num_producer_groups; ++g) {
      if (!multicast_producer_groups[g]) {
        continue;
      }
      PrimExpr arrive_count =
          consumer_extent * CountRanksInMask(producer_cluster_masks[g]);
      for (int stage = 0; stage < producer_buffer_versions[g]; ++stage) {
        cluster_backpressure_arrive_counts.push_back(arrive_count);
      }
    }

    Array<Stmt> block_completion_prearm_stmts;
    Array<Stmt> forward_prearm_stmts;
    if (has_streamed_cluster_push) {
      // This collective must remain outside the divergent producer/consumer
      // role scope and outside repeated pipeline invocations.
      block_completion_prearm_stmts.push_back(
          Evaluate(Call(DataType::Handle(), tl::cluster_sync(), {})));
    }
    if (use_one_shot_completion) {
      PrimExpr total_transaction_bytes = IntImm(DataType::Int(64), 0);
      for (const PrimExpr &transaction_bytes : producer_transaction_bytes) {
        total_transaction_bytes += transaction_bytes;
      }
      PrimExpr block_threads = consumer_extent + producer_extent;
      const int64_t *static_block_threads = as_const_int(block_threads);
      ICHECK(static_block_threads != nullptr && *static_block_threads > 0);
      int64_t arm_groups =
          (invocation.static_invocation_count + *static_block_threads - 1) /
          *static_block_threads;
      Var arm_group("pipeline_completion_arm_group", DataType::Int(32));
      PrimExpr barrier_id =
          arm_group * IntImm(DataType::Int(32), *static_block_threads) +
          thread_iv_->var;
      Stmt arm =
          IfThenElse(LT(barrier_id, IntImm(DataType::Int(32),
                                           invocation.static_invocation_count)),
                     MakeArriveBarrierExpectTx(barrier_buf, barrier_id,
                                               total_transaction_bytes));
      block_completion_prearm_stmts.push_back(
          For(arm_group, 0, IntImm(DataType::Int(32), arm_groups),
              ForKind::kSerial, arm));
      block_completion_prearm_stmts.push_back(MakeSharedStorageSync());
    }
    if (has_multicast_producer) {
      Array<Stmt> arm_stmts;
      for (int g = 0; g < num_producer_groups; ++g) {
        if (!multicast_producer_groups[g]) {
          continue;
        }
        for (int64_t iteration = 0; iteration < *static_loop_extent;
             ++iteration) {
          arm_stmts.push_back(MakeArriveBarrierExpectTx(
              barrier_buf,
              IntImm(DataType::Int(32), forward_barrier_bases[g] + iteration),
              producer_transaction_bytes[g]));
        }
      }
      forward_prearm_stmts.push_back(
          IfThenElse(EQ(thread_iv_->var, IntImm(thread_iv_->var.dtype(), 0)),
                     SeqStmt(arm_stmts)));
      forward_prearm_stmts.push_back(
          Evaluate(Call(DataType::Handle(), tl::cluster_sync(), {})));
    }

    Array<Stmt> handoff_consumer_prearm_stmts;
    if (any_handoff_consumer) {
      ICHECK(handoff_stage_count_var_.defined());
      auto prearm_stage = [&](PrimExpr barrier_id, int stage,
                              int arrive_count) {
        Array<Stmt> arrivals;
        for (int i = 0; i < arrive_count; ++i) {
          arrivals.push_back(MakeArriveBarrier(barrier_buf, barrier_id));
        }
        Stmt arrive = arrivals.size() == 1 ? arrivals[0] : SeqStmt(arrivals);
        PrimExpr has_stage =
            GT(handoff_stage_count_var_.value(),
               make_const(handoff_stage_count_var_.value().dtype(), stage));
        handoff_consumer_prearm_stmts.push_back(
            IfThenElse(logical_and(EQ(thread_iv_->var,
                                      make_const(thread_iv_->var.dtype(), 0)),
                                   has_stage),
                       std::move(arrive)));
      };
      if (can_merge_tma_forward_barriers) {
        int empty_arrivals =
            merge_forward_across_producer_partitions ? num_producer_groups : 1;
        for (int stage = 0; stage < producer_buffer_versions[0]; ++stage) {
          prearm_stage(
              IntImm(DataType::Int(32), forward_barrier_bases[0] + stage),
              stage, empty_arrivals);
        }
      } else {
        for (int g = 0; g < num_producer_groups; ++g) {
          if (!handoff_consumer_groups[g]) {
            continue;
          }
          for (int stage = 0; stage < producer_buffer_versions[g]; ++stage) {
            prearm_stage(
                IntImm(DataType::Int(32), forward_barrier_bases[g] + stage),
                stage, 1);
          }
        }
      }
    }

    // A non-multicast stage is initially empty, so its first producer use does
    // not need a consumer release.  Multicast keeps the explicit pre-release
    // protocol because participating and non-participating CTA ranks use
    // different backpressure barriers.
    bool skip_initial_backpressure_waits = !has_multicast_producer;
    Array<Stmt> initial_bp_release_stmts;
    if (!skip_initial_backpressure_waits) {
      for (int g = 0; g < (can_merge_tma_barriers ? 1 : num_producer_groups);
           ++g) {
        int bp_base = num_fwd + backpressure_barrier_bases[g];
        for (int s = 0; s < producer_buffer_versions[g]; ++s) {
          initial_bp_release_stmts.push_back(MakeArriveBarrier(
              barrier_buf, IntImm(DataType::Int(32), bp_base + s)));
        }
      }
    }

    std::vector<Array<Stmt>> prelude_waits_before_consumer(
        consumer_compute_stmts.size());
    PrimExpr prelude_wait_guard =
        EQ(c_iteration_expr, IntImm(DataType::Int(32), 0));
    int handoff_completion_barrier_base = num_fwd + num_bp;
    int prelude_barrier_base =
        handoff_completion_barrier_base +
        static_cast<int>(detached_handoff_producer_stmts.size());
    int consumer_ready_barrier_base =
        prelude_barrier_base + static_cast<int>(prelude_tma_plans.size());
    int consumer_consumed_barrier_base =
        consumer_ready_barrier_base + num_consumer_sync_groups * num_stages;
    for (size_t i = 0; i < prelude_tma_plans.size(); ++i) {
      PrimExpr barrier_id = IntImm(DataType::Int(32), prelude_barrier_base + i);
      Stmt rewritten_prelude = RewritePreludeTmaProducerStmt(
          prelude_tma_plans[i].stmt, barrier_buf, barrier_id);
      common_prelude_rewrites_.emplace(prelude_tma_plans[i].stmt,
                                       rewritten_prelude);
      int wait_pos = prelude_tma_plans[i].wait_pos;
      ICHECK_GE(wait_pos, 0);
      ICHECK_LT(wait_pos, static_cast<int>(consumer_compute_stmts.size()));
      prelude_waits_before_consumer[wait_pos].push_back(IfThenElse(
          prelude_wait_guard, MakeParityWait(barrier_buf, barrier_id,
                                             IntImm(DataType::Int(32), 0))));
    }

    // --- Build producer body ---
    // Producer structure (mixed TMA + SIMT/cp.async):
    //   bp_wait → SIMT copies (all threads, async) → TMA copies (leader) →
    //   commit + cp_async_barrier_noinc.
    // SIMT copies are placed after bp_wait but before TMA so cp.async
    // and TMA can overlap.

    // First pass: collect SIMT/cp.async producer stmts separately.
    Array<Stmt> simt_producer_stmts;
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (kinds[i] == TileStmtKind::kSimtProducer) {
        // Annotate ForNodes with kParallelAsyncWithoutAsyncCommitWait so
        // InjectPTXAsyncCopy (called from LowerTileOp) does not insert
        // commit+wait — the WS pass will emit its own commit+barrier_noinc.
        simt_producer_stmts.push_back(
            SimtProducerAnnotator::Annotate(flat_stmts[i], target_));
      } else if (kinds[i] == TileStmtKind::kCpAsyncProducer) {
        simt_producer_stmts.push_back(flat_stmts[i]);
      }
    }

    // Second pass: build the producer body with correct ordering.
    Array<Stmt> producer_stmts;
    int tma_idx = 0;
    int last_tma_idx = num_producer_groups - 1;
    bool simt_stmts_emitted = false;
    for (size_t i = 0; i < flat_stmts.size(); ++i) {
      if (kinds[i] == TileStmtKind::kTmaProducer) {
        int barrier_group = can_merge_tma_barriers ? 0 : tma_idx;
        int group_versions = producer_buffer_versions[barrier_group];
        PrimExpr group_stage_expr =
            stage_for_versions(p_iteration_expr, group_versions);
        PrimExpr group_parity_expr =
            parity_for_versions(p_iteration_expr, group_versions);
        bool streamed_one_shot = streamed_cluster_push_groups[tma_idx] &&
                                 streamed_cluster_push_one_shot;
        PrimExpr forward_stage_expr =
            (use_one_shot_completion || streamed_one_shot)
                ? p_iteration_expr
                : (can_merge_tma_forward_barriers ? p_stage_expr
                                                  : group_stage_expr);
        int fwd_base = forward_barrier_bases[tma_idx];
        int bp_base = num_fwd + backpressure_barrier_bases[barrier_group];
        PrimExpr fwd_id =
            IntImm(DataType::Int(32), fwd_base) +
            (multicast_producer_groups[tma_idx] ? p_iteration_expr
                                                : forward_stage_expr);
        PrimExpr bp_id =
            IntImm(DataType::Int(32), bp_base) +
            (streamed_one_shot
                 ? p_iteration_expr - IntImm(DataType::Int(32), group_versions)
                 : group_stage_expr);

        // Same-partition merged copies need one wait. Distinct producer
        // partitions must each observe the shared stage release before reuse.
        if (!can_merge_tma_barriers || merge_across_producer_partitions ||
            tma_idx == 0) {
          PrimExpr wait_parity =
              streamed_one_shot ? PrimExpr(IntImm(DataType::Int(32), 0))
              : skip_initial_backpressure_waits
                  ? bitwise_xor(group_parity_expr, IntImm(DataType::Int(32), 1))
                  : group_parity_expr;
          Stmt local_wait = MakeParityWait(barrier_buf, bp_id, wait_parity);
          if (skip_initial_backpressure_waits) {
            local_wait = IfThenElse(
                GE(p_iteration_expr, IntImm(DataType::Int(32), group_versions)),
                local_wait);
          }
          if (multicast_producer_groups[tma_idx]) {
            ICHECK(cluster_backpressure_buf.defined());
            PrimExpr cluster_bp_id =
                IntImm(DataType::Int(32), cluster_backpressure_bases[tma_idx]) +
                group_stage_expr;
            Stmt cluster_wait = MakeParityWait(
                cluster_backpressure_buf.value(), cluster_bp_id,
                bitwise_xor(group_parity_expr, IntImm(DataType::Int(32), 1)));
            PrimExpr rank =
                Call(DataType::Int(32), block_rank_in_cluster(), {});
            PrimExpr mask =
                IntImm(DataType::Int(32), producer_cluster_masks[tma_idx]);
            PrimExpr outside_mask =
                EQ(bitwise_and(right_shift(mask, rank),
                               IntImm(DataType::Int(32), 1)),
                   IntImm(DataType::Int(32), 0));
            producer_stmts.push_back(guard_handoff_consumer_prefix(
                tma_idx, p_iteration_expr,
                guard_producer_partition(
                    tma_idx,
                    IfThenElse(
                        EQ(rank, IntImm(DataType::Int(32),
                                        MinRankInMask(
                                            producer_cluster_masks[tma_idx]))),
                        cluster_wait, IfThenElse(outside_mask, local_wait)))));
          } else {
            producer_stmts.push_back(guard_handoff_consumer_prefix(
                tma_idx, p_iteration_expr,
                guard_producer_partition(
                    tma_idx, guard_pure_tma_issuer_warp(tma_idx, local_wait))));
          }
        }

        // After the first bp_wait, emit all SIMT/cp.async producers
        // followed immediately by commit_group so the hardware can start
        // the async transfers as early as possible, overlapping with TMA.
        if (!simt_stmts_emitted && !simt_producer_stmts.empty()) {
          for (const auto &s : simt_producer_stmts) {
            producer_stmts.push_back(s);
          }
          // Commit cp.async group right after issuing — the earlier the
          // commit, the more overlap with subsequent TMA loads.
          if (has_simt_producer || has_cp_async_producer) {
            producer_stmts.push_back(Evaluate(
                Call(DataType::Handle(), builtin::ptx_commit_group(), {})));
          }
          simt_stmts_emitted = true;
        }

        for (const auto &stmt : producer_loop_prefix_stmts[tma_idx]) {
          producer_stmts.push_back(guard_producer_partition(tma_idx, stmt));
        }
        // Convert copy → tma_copy with barrier, or annotate non-copy
        // TMA tile-ops (e.g. c2d_im2col) with barrier reference.
        const auto *eval = flat_stmts[i].as<EvaluateNode>();
        ICHECK(eval);
        Call tile_call = Downcast<Call>(eval->value);
        auto tile_op = ParseOperator(tile_call);
        PrimExpr tma_call;
        // For pure TMA, tell LowerTileOp to emit arrive inside the same
        // tl_shuffle_elect block (via emit_arrive annotation), producing
        // arrive_and_expect_tx instead of separate expect_tx + arrive.
        // Same-partition merged copies use one final arrival. Distinct
        // partitions each arrive because their leaders execute independently.
        bool emit_arrive_on_this =
            !has_simt_producer && !has_cp_async_producer &&
            !multicast_producer_groups[tma_idx] && !use_one_shot_completion &&
            (!can_merge_tma_forward_barriers ||
             merge_forward_across_producer_partitions ||
             tma_idx == last_tma_idx);

        if (tile_op.defined() && tile_op.as<CopyNode>()) {
          tma_call = RewriteCopyToTmaCopy(tile_call, barrier_buf, fwd_id);
          if (multicast_producer_groups[tma_idx] || use_one_shot_completion) {
            auto call = Downcast<Call>(tma_call);
            auto annos = call->annotations;
            annos.Set("skip_expect_transaction", IntImm(DataType::Int(32), 1));
            tma_call =
                Call(call->dtype, call->op, call->args, annos, call->span);
          }
        } else {
          // Non-copy TMA producer (e.g. Conv2DIm2ColOp): annotate with
          // barrier so Lower() uses the WS barrier instead of its own.
          tma_call = AnnotateTileOpBarrier(tile_call, barrier_buf, fwd_id);
        }
        if (emit_arrive_on_this) {
          auto call = Downcast<Call>(tma_call);
          auto annos = call->annotations;
          annos.Set("emit_arrive", IntImm(DataType::Int(32), 1));
          tma_call = Call(call->dtype, call->op, call->args, annos, call->span);
        }
        producer_stmts.push_back(guard_handoff_consumer_prefix(
            tma_idx, p_iteration_expr,
            guard_producer_partition(tma_idx, Evaluate(tma_call))));
        ++tma_idx;
      }
      // SIMT/cp.async producers are handled above (after first bp_wait).
      // Consumer/Other statements are skipped in producer.
    }
    // Detached handoff transfers write a compiler-owned arena for a future
    // handler. They are not members of this handler's forward/backpressure
    // rings: each transfer has a one-shot completion barrier, and the
    // producer branch waits for every issued TMA before the handler returns.
    Array<Stmt> detached_handoff_waits;
    for (size_t i = 0; i < detached_handoff_producer_stmts.size(); ++i) {
      PrimExpr barrier_id =
          IntImm(DataType::Int(32), handoff_completion_barrier_base + i);
      Stmt issue = RewriteDetachedHandoffProducerStmt(
          detached_handoff_producer_stmts[i], barrier_buf, barrier_id);
      issue = guard_partition(detached_handoff_producer_partitions[i],
                              std::move(issue));
      producer_stmts.push_back(std::move(issue));

      Stmt wait =
          MakeParityWait(barrier_buf, barrier_id, IntImm(DataType::Int(32), 0));
      wait = ReplaceDetachedHandoffProducerLeaf(
          detached_handoff_producer_stmts[i], std::move(wait));
      wait = guard_partition(detached_handoff_producer_partitions[i],
                             std::move(wait));
      detached_handoff_waits.push_back(std::move(wait));
    }
    for (const Stmt &wait : detached_handoff_waits) {
      producer_stmts.push_back(wait);
    }
    // Fallback: if there were no TMA producers to anchor the bp_wait,
    // emit SIMT stmts now (shouldn't happen in the mixed path).
    if (!simt_stmts_emitted && !simt_producer_stmts.empty()) {
      for (const auto &s : simt_producer_stmts) {
        producer_stmts.push_back(s);
      }
    }
    // When any producer-side work is not single-threaded pure-TMA, all
    // producer threads arrive on all forward barriers after finishing it.
    // SIMT copies (later lowered to cp.async by InjectPTXAsyncCopy) and
    // explicit cp.async groups use commit_group + cp_async_barrier_noinc
    // so the async copy completion drives the mbarrier arrival, allowing
    // TMA and cp.async to overlap.  Other groups use MakeArriveBarrier.
    if (has_simt_producer || has_cp_async_producer) {
      // Any SIMT producer will become cp.async after LowerTileOp.
      bool group_has_async_copy = has_simt_producer || has_cp_async_producer;
      for (int g = 0; g < num_producer_groups; ++g) {
        int fwd_base = forward_barrier_bases[g];
        PrimExpr fwd_id =
            IntImm(DataType::Int(32), fwd_base) +
            stage_for_versions(p_iteration_expr, producer_buffer_versions[g]);
        if (group_has_async_copy) {
          // Tie cp.async completion to the forward mbarrier.
          // commit_group was already emitted right after the cp.async
          // instructions (before TMA) to maximize overlap.
          producer_stmts.push_back(Evaluate(
              Call(DataType::Handle(), tl::ptx_cp_async_barrier_noinc(),
                   {MakeBarrierRef(barrier_buf, fwd_id)})));
        } else {
          producer_stmts.push_back(MakeArriveBarrier(barrier_buf, fwd_id));
        }
      }
    }
    // Phase counter increment at end of producer guarded iteration
    if (needs_phase_counter) {
      producer_stmts.push_back(producer_phase_counter.value().Increment());
    }

    // --- Build consumer body ---
    int consumer_forward_barrier_groups =
        can_merge_tma_forward_barriers ? 1 : num_producer_groups;
    Array<Stmt> consumer_stmts;
    std::vector<bool> arrive_emitted(backpressure_barrier_groups, false);
    for (size_t ci = 0; ci < consumer_compute_stmts.size(); ++ci) {
      for (const auto &stmt : prelude_waits_before_consumer[ci]) {
        consumer_stmts.push_back(stmt);
      }
      for (int g = 0; g < consumer_forward_barrier_groups; ++g) {
        if (forward_wait_insert_pos[g] == static_cast<int>(ci)) {
          int group_versions = can_merge_tma_forward_barriers
                                   ? num_stages
                                   : producer_buffer_versions[g];
          bool streamed_one_shot =
              streamed_cluster_push_groups[g] && streamed_cluster_push_one_shot;
          PrimExpr group_stage_expr =
              (use_one_shot_completion || streamed_one_shot)
                  ? c_iteration_expr
                  : stage_for_versions(c_iteration_expr, group_versions);
          PrimExpr group_parity_expr =
              (use_one_shot_completion || streamed_one_shot)
                  ? PrimExpr(IntImm(DataType::Int(32), 0))
                  : parity_for_versions(c_iteration_expr, group_versions);
          int fwd_base = forward_barrier_bases[g];
          bool multicast = multicast_producer_groups[g];
          PrimExpr fwd_id = IntImm(DataType::Int(32), fwd_base) +
                            (multicast ? c_iteration_expr : group_stage_expr);
          consumer_stmts.push_back(
              MakeParityWait(barrier_buf, fwd_id,
                             multicast ? PrimExpr(IntImm(DataType::Int(32), 0))
                                       : group_parity_expr));
        }
      }
      if (!moved_compute_stmts[ci]) {
        if (externalize_static_wgmma_drain) {
          const int64_t *consumer_threads = as_const_int(consumer_extent);
          ICHECK(consumer_threads != nullptr);
          ExternalizedWgmmaDrain externalized = ExternalizeImplicitWgmmaDrain(
              consumer_compute_stmts[ci], static_cast<int>(*consumer_threads),
              target_);
          consumer_stmts.push_back(externalized.issue);
          for (const Stmt &drain_stmt : externalized.drain) {
            consumer_stmts.push_back(drain_stmt);
          }
        } else {
          consumer_stmts.push_back(consumer_compute_stmts[ci]);
        }
        if (consumer_sync_groups[ci] >= 0) {
          PrimExpr barrier_id =
              IntImm(DataType::Int(32),
                     consumer_ready_barrier_base +
                         consumer_sync_groups[ci] * num_stages) +
              c_stage_expr;
          consumer_stmts.push_back(MakeArriveBarrier(barrier_buf, barrier_id));
          consumer_stmts.push_back(
              MakeParityWait(barrier_buf, barrier_id, c_parity_expr));
        }
      }
      for (int g = 0; g < num_consumer_sync_groups; ++g) {
        if (consumer_sync_release_positions[g] == static_cast<int>(ci + 1)) {
          PrimExpr barrier_id =
              IntImm(DataType::Int(32),
                     consumer_consumed_barrier_base + g * num_stages) +
              c_stage_expr;
          consumer_stmts.push_back(MakeArriveBarrier(barrier_buf, barrier_id));
          consumer_stmts.push_back(
              MakeParityWait(barrier_buf, barrier_id, c_parity_expr));
        }
      }
      for (int g = 0; g < backpressure_barrier_groups; ++g) {
        if (arrive_insert_pos[g] == static_cast<int>(ci + 1)) {
          int group_versions = producer_buffer_versions[g];
          PrimExpr group_stage_expr =
              stage_for_versions(c_iteration_expr, group_versions);
          if (multicast_producer_groups[g]) {
            ICHECK(cluster_backpressure_buf.defined());
            PrimExpr cluster_bp_id =
                IntImm(DataType::Int(32), cluster_backpressure_bases[g]) +
                group_stage_expr;
            PrimExpr rank =
                Call(DataType::Int(32), block_rank_in_cluster(), {});
            PrimExpr mask =
                IntImm(DataType::Int(32), producer_cluster_masks[g]);
            PrimExpr inside_mask = EQ(bitwise_and(right_shift(mask, rank),
                                                  IntImm(DataType::Int(32), 1)),
                                      IntImm(DataType::Int(32), 1));
            consumer_stmts.push_back(
                IfThenElse(inside_mask,
                           MakeArriveClusterBarrier(
                               cluster_backpressure_buf.value(), cluster_bp_id,
                               MinRankInMask(producer_cluster_masks[g]))));
          }
          int bp_base = num_fwd + backpressure_barrier_bases[g];
          PrimExpr bp_id =
              IntImm(DataType::Int(32), bp_base) +
              (streamed_cluster_push_groups[g] && streamed_cluster_push_one_shot
                   ? c_iteration_expr
                   : group_stage_expr);
          consumer_stmts.push_back(make_backpressure_release(g, bp_id));
          arrive_emitted[g] = true;
        }
      }
    }
    if (consumer_compute_stmts.empty()) {
      for (int g = 0; g < consumer_forward_barrier_groups; ++g) {
        int group_versions = can_merge_tma_forward_barriers
                                 ? num_stages
                                 : producer_buffer_versions[g];
        bool streamed_one_shot =
            streamed_cluster_push_groups[g] && streamed_cluster_push_one_shot;
        PrimExpr group_stage_expr =
            (use_one_shot_completion || streamed_one_shot)
                ? c_iteration_expr
                : stage_for_versions(c_iteration_expr, group_versions);
        PrimExpr group_parity_expr =
            (use_one_shot_completion || streamed_one_shot)
                ? PrimExpr(IntImm(DataType::Int(32), 0))
                : parity_for_versions(c_iteration_expr, group_versions);
        int fwd_base = forward_barrier_bases[g];
        bool multicast = multicast_producer_groups[g];
        PrimExpr fwd_id = IntImm(DataType::Int(32), fwd_base) +
                          (multicast ? c_iteration_expr : group_stage_expr);
        consumer_stmts.push_back(
            MakeParityWait(barrier_buf, fwd_id,
                           multicast ? PrimExpr(IntImm(DataType::Int(32), 0))
                                     : group_parity_expr));
      }
    }
    for (int g = 0; g < backpressure_barrier_groups; ++g) {
      if (!arrive_emitted[g] &&
          arrive_insert_pos[g] ==
              static_cast<int>(consumer_compute_stmts.size())) {
        int group_versions = producer_buffer_versions[g];
        PrimExpr group_stage_expr =
            stage_for_versions(c_iteration_expr, group_versions);
        if (multicast_producer_groups[g]) {
          ICHECK(cluster_backpressure_buf.defined());
          PrimExpr cluster_bp_id =
              IntImm(DataType::Int(32), cluster_backpressure_bases[g]) +
              group_stage_expr;
          PrimExpr rank = Call(DataType::Int(32), block_rank_in_cluster(), {});
          PrimExpr mask = IntImm(DataType::Int(32), producer_cluster_masks[g]);
          PrimExpr inside_mask = EQ(bitwise_and(right_shift(mask, rank),
                                                IntImm(DataType::Int(32), 1)),
                                    IntImm(DataType::Int(32), 1));
          consumer_stmts.push_back(IfThenElse(
              inside_mask, MakeArriveClusterBarrier(
                               cluster_backpressure_buf.value(), cluster_bp_id,
                               MinRankInMask(producer_cluster_masks[g]))));
        }
        int bp_base = num_fwd + backpressure_barrier_bases[g];
        PrimExpr bp_id =
            IntImm(DataType::Int(32), bp_base) +
            (streamed_cluster_push_groups[g] && streamed_cluster_push_one_shot
                 ? c_iteration_expr
                 : group_stage_expr);
        consumer_stmts.push_back(make_backpressure_release(g, bp_id));
      }
    }
    // Phase counter increment at end of consumer guarded iteration
    if (needs_phase_counter) {
      consumer_stmts.push_back(consumer_phase_counter.value().Increment());
    }

    // --- Wrap with let bindings and optional condition ---
    auto wrap_lets =
        [&](Stmt body,
            const std::vector<std::pair<Var, PrimExpr>> &bindings) -> Stmt {
      for (auto it = bindings.rbegin(); it != bindings.rend(); ++it) {
        body = LetStmt(it->first, it->second, body);
      }
      return body;
    };

    Stmt producer_body = wrap_lets(SeqStmt(producer_stmts), inner_let_bindings);
    Stmt consumer_body = wrap_lets(SeqStmt(consumer_stmts), inner_let_bindings);

    // Wrap in original condition if the loop body was guarded.
    if (loop_body_condition.defined()) {
      producer_body = IfThenElse(loop_body_condition.value(), producer_body);
      consumer_body = IfThenElse(loop_body_condition.value(), consumer_body);
    }

    producer_body = wrap_lets(producer_body, outer_let_bindings);
    consumer_body = wrap_lets(consumer_body, outer_let_bindings);

    // Rewrite shared-buffer stage indices when the barrier phase no longer
    // restarts from the local pipeline-loop index.
    if (needs_phase_counter || use_affine_iteration) {
      producer_body = StageExprReplacer::Replace(
          producer_body, loop_var, loop_min, num_stages, p_stage_expr);
      consumer_body = StageExprReplacer::Replace(
          consumer_body, loop_var, loop_min, num_stages, c_stage_expr);
      producer_body = PipelineBufferStageExprRewriter::Replace(
          producer_body, pipeline_buffer_versions, p_iteration_expr);
      consumer_body = PipelineBufferStageExprRewriter::Replace(
          consumer_body, pipeline_buffer_versions, c_iteration_expr);
    }
    producer_body =
        TileOpMbarPhaseAnnotator::Annotate(producer_body, p_parity_expr);
    consumer_body = ConsumerTransferFallbackAnnotator::Annotate(consumer_body);
    consumer_body =
        TileOpMbarPhaseAnnotator::Annotate(consumer_body, c_parity_expr);

    // --- Build loops (strip pipeline annotations) ---
    // WS handles pipeline overlap via barriers, so strip all pipeline-
    // related annotations to prevent PipelinePlanning / InjectSoftware-
    // Pipeline from re-pipelining the already WS-transformed loops.
    Map<String, Any> loop_annos;
    for (const auto &[key, value] : pipeline_loop->annotations) {
      if (key != "num_stages" && key != "tl_pipeline_order" &&
          key != "tl_pipeline_stage" && key != "software_pipeline_order" &&
          key != "software_pipeline_stage") {
        loop_annos.Set(key, value);
      }
    }

    bool unroll_physical_pipeline =
        unroll_static_affine_wgmma && !use_one_shot_completion;
    ForKind physical_loop_kind =
        unroll_physical_pipeline ? ForKind::kUnrolled : ForKind::kSerial;
    if (unroll_physical_pipeline) {
      loop_annos.Set(tir::attr::pragma_unroll_explicit, Bool(false));
    }
    For producer_loop(loop_var, loop_min, loop_extent, physical_loop_kind,
                      producer_body, Optional<IterVar>(), loop_annos);
    For consumer_loop(loop_var, loop_min, loop_extent, physical_loop_kind,
                      consumer_body, Optional<IterVar>(), loop_annos);

    // Wrap loops with phase counter allocation when needed.
    Stmt final_producer_loop = producer_loop;
    Stmt final_consumer_loop = consumer_loop;
    if (!handoff_consumer_prearm_stmts.empty()) {
      Array<Stmt> consumer_parts = handoff_consumer_prearm_stmts;
      consumer_parts.push_back(final_consumer_loop);
      final_consumer_loop = SeqStmt(consumer_parts);
    }
    if (needs_phase_counter && !phase_counter_block_scope) {
      final_producer_loop =
          producer_phase_counter.value().WrapLoopWithAlloc(producer_loop);
      final_consumer_loop =
          consumer_phase_counter.value().WrapLoopWithAlloc(consumer_loop);
    }

    // --- Rewrite threadIdx.x for producer partition ---
    // Producer: threadIdx.x - consumer_extent (maps to [0, producer_extent))
    Stmt rewritten_producer = PCThreadIdxRewriter::Rewrite(
        final_producer_loop, thread_iv_->var, thread_iv_->var - consumer_extent,
        producer_extent, false);
    // Consumer: threadIdx.x stays, but extent is consumer_extent
    Stmt rewritten_consumer = final_consumer_loop;

    shared_prelude_live_seed_ = {};
    producer_prelude_live_seed_ = {};
    consumer_prelude_live_seed_ = {};
    producer_prelude_live_seed_.AddUses(LocalAccessCollector::Collect(
        rewritten_producer, buffer_data_to_buffer_));
    consumer_prelude_live_seed_.AddUses(LocalAccessCollector::Collect(
        rewritten_consumer, buffer_data_to_buffer_));

    // Move pre-loop branch-private initialization next to the branch that
    // consumes it. Classification is based on downstream producer/consumer
    // uses of the values defined by each prelude statement.
    extracted_producer_init_ = {};
    extracted_consumer_init_ = {};

    Array<IntImm> ws_partition = {Downcast<IntImm>(producer_extent),
                                  Downcast<IntImm>(consumer_extent)};

    // First pass: find and extract consumer-only pre-loop statements
    // by doing a dry replacement that populates extracted_consumer_init_.
    Stmt dummy_producer = rewritten_producer;
    const Stmt &dummy_consumer = rewritten_consumer;
    Stmt dummy_ws_branch = IfThenElse(GE(thread_iv_->var, consumer_extent),
                                      dummy_producer, dummy_consumer);
    dummy_ws_branch = AttrStmt(ws_partition, attr::kWarpSpecializationScope, 0,
                               dummy_ws_branch);
    Stmt dummy_ws = dummy_ws_branch;
    if (!initial_bp_release_stmts.empty() || !forward_prearm_stmts.empty()) {
      Array<Stmt> ws_parts;
      for (const Stmt &stmt : initial_bp_release_stmts) {
        Stmt release = stmt;
        if (phase_counter_block_scope) {
          release = IfThenElse(EQ(consumer_phase_counter.value().Load(),
                                  IntImm(DataType::Int(32), 0)),
                               release);
        }
        ws_parts.push_back(
            IfThenElse(LT(thread_iv_->var, consumer_extent), release));
      }
      ws_parts.push_back(MakeSharedStorageSync());
      for (const Stmt &stmt : forward_prearm_stmts) {
        ws_parts.push_back(stmt);
      }
      ws_parts.push_back(dummy_ws);
      dummy_ws = SeqStmt(ws_parts);
    }
    ReplaceResult replaced = ReplacePipelineLoopInStmt(
        orig_block->body, pipeline_loop, dummy_ws, consumer_extent);

    // Producer and consumer partitions cannot safely share the same block-level
    // local/fragment buffers after tiled WS is introduced before
    // LayoutInference: a single fragment layout cannot represent both thread
    // ranges. Clone every branch-private buffer touched by the producer so
    // LayoutInference can infer an independent producer-side thread range.
    BufferMap producer_buffer_remap;
    Array<Buffer> producer_private_buffers;
    {
      BufferSet block_alloc_buffers;
      for (const auto &buffer : orig_block->alloc_buffers) {
        block_alloc_buffers.insert(buffer);
      }
      LocalAccessSummary producer_access = LocalAccessCollector::Collect(
          rewritten_producer, buffer_data_to_buffer_);
      for (const auto &stmt : extracted_producer_init_) {
        MergeLocalAccessSummary(
            &producer_access,
            LocalAccessCollector::Collect(stmt, buffer_data_to_buffer_));
      }
      auto maybe_clone = [&](const Buffer &buffer) {
        if (!buffer.defined() ||
            !(IsFragmentBuffer(buffer) || IsLocalBuffer(buffer)) ||
            !block_alloc_buffers.count(buffer) ||
            producer_buffer_remap.count(buffer)) {
          return;
        }
        Buffer cloned = CloneBranchPrivateBuffer(buffer, "_producer_ws");
        producer_buffer_remap.emplace(buffer, cloned);
        producer_private_buffers.push_back(cloned);
      };
      for (const auto &buffer : producer_access.read_buffers) {
        maybe_clone(buffer);
      }
      for (const auto &buffer : producer_access.write_buffers) {
        maybe_clone(buffer);
      }
    }
    if (!producer_buffer_remap.empty()) {
      rewritten_producer =
          BufferRemapper::Rewrite(rewritten_producer, producer_buffer_remap);
      Array<Stmt> remapped_producer_init;
      for (const auto &stmt : extracted_producer_init_) {
        remapped_producer_init.push_back(
            BufferRemapper::Rewrite(stmt, producer_buffer_remap));
      }
      extracted_producer_init_ = remapped_producer_init;
    }

    // If branch-local prelude init/copy was extracted, rebuild with it inside
    // the corresponding WS branch so each branch initializes its own local
    // state before entering the pipelined loop.
    if (!extracted_producer_init_.empty() ||
        !extracted_consumer_init_.empty()) {
      Stmt enriched_producer = rewritten_producer;
      if (!extracted_producer_init_.empty()) {
        Array<Stmt> producer_parts;
        for (const auto &s : extracted_producer_init_) {
          producer_parts.push_back(PCThreadIdxRewriter::Rewrite(
              s, thread_iv_->var, thread_iv_->var - consumer_extent,
              producer_extent, false));
        }
        producer_parts.push_back(rewritten_producer);
        enriched_producer = producer_parts.size() == 1
                                ? producer_parts[0]
                                : SeqStmt(producer_parts);
      }
      Array<Stmt> consumer_parts;
      for (const auto &s : extracted_consumer_init_) {
        consumer_parts.push_back(s);
      }
      consumer_parts.push_back(rewritten_consumer);
      Stmt enriched_consumer = consumer_parts.size() == 1
                                   ? consumer_parts[0]
                                   : SeqStmt(consumer_parts);
      Stmt scoped_producer = enriched_producer;
      const Stmt &scoped_consumer = enriched_consumer;
      Stmt ws_body_branch = IfThenElse(GE(thread_iv_->var, consumer_extent),
                                       scoped_producer, scoped_consumer);
      ws_body_branch = AttrStmt(ws_partition, attr::kWarpSpecializationScope, 0,
                                ws_body_branch);
      // Second pass: replace again with the enriched WS body.
      // extracted_consumer_init_ is already empty (stmts were removed
      // from the prelude in the first pass result).
      // We need to replace in the ALREADY-modified body from pass 1.
      // But ReplacePipelineLoopInStmt finds the pipeline_loop by
      // pointer comparison, which won't match in the modified tree.
      // Instead, just substitute the dummy_ws in the replaced result.
      // Since dummy_ws_branch appears exactly once in replaced.stmt, do a
      // simple statement replacement on the branch placeholder stmt.  The
      // optional initial backpressure-release wrapper is outside this
      // placeholder and must not be duplicated during substitution.
      class SubstWsBody : public StmtExprMutator {
      public:
        SubstWsBody(const Stmt &old_ws, const Stmt &new_ws)
            : old_(old_ws), new_(new_ws) {}
        Stmt VisitStmt(const Stmt &stmt) final {
          if (stmt.same_as(old_)) {
            return new_;
          }
          return StmtExprMutator::VisitStmt(stmt);
        }
        Stmt old_, new_;
      };
      SubstWsBody subst(dummy_ws_branch, ws_body_branch);
      replaced.stmt = subst(replaced.stmt);
    }
    ICHECK(replaced.found)
        << "ProducerConsumerWS: failed to replace pipeline loop";
    Stmt new_block_body = SinkGuardedConsumerPostlude::Rewrite(
        replaced.stmt, thread_iv_->var, consumer_extent);
    bool lift_role_scope =
        invocation.role_scope_loop.defined() &&
        invocation.role_scope_liftable &&
        (!invocation.may_repeat || unroll_static_affine_wgmma);
    if (lift_role_scope) {
      bool lifted = false;
      new_block_body = LiftNestedWarpSpecialization::Rewrite(
          new_block_body, invocation.role_scope_loop.value()->loop_var,
          thread_iv_->var, consumer_extent, ws_partition, &lifted);
      ICHECK(lifted) << "ProducerConsumerWS: failed to lift a proven nested "
                        "warp-specialization scope";
    }
    if (phase_counter_block_scope) {
      new_block_body =
          consumer_phase_counter.value().WrapLoopWithAlloc(new_block_body);
      new_block_body =
          producer_phase_counter.value().WrapLoopWithAlloc(new_block_body);
    }
    if (!block_completion_prearm_stmts.empty()) {
      Array<Stmt> block_parts;
      for (const Stmt &stmt : block_completion_prearm_stmts) {
        block_parts.push_back(stmt);
      }
      block_parts.push_back(new_block_body);
      new_block_body = SeqStmt(block_parts);
    }

    // --- Update block ---
    Block new_block = orig_block;
    auto *block_ptr = new_block.CopyOnWrite();
    block_ptr->body = new_block_body;
    for (const auto &buffer : producer_private_buffers) {
      block_ptr->alloc_buffers.push_back(buffer);
    }

    // Add barrier buffer to alloc_buffers.
    block_ptr->alloc_buffers.push_back(barrier_buf);
    if (cluster_backpressure_buf.defined()) {
      block_ptr->alloc_buffers.push_back(cluster_backpressure_buf.value());
    }

    // Add barrier_init annotation.
    Map<Var, Array<PrimExpr>> barrier_init_map;
    barrier_init_map.Set(barrier_buf->data, arrive_counts);
    if (cluster_backpressure_buf.defined()) {
      barrier_init_map.Set(cluster_backpressure_buf.value()->data,
                           cluster_backpressure_arrive_counts);
    }
    auto ann = block_ptr->annotations;
    if (ann.count("barrier_init")) {
      auto existing =
          Downcast<Map<Var, Array<PrimExpr>>>(ann.Get("barrier_init").value());
      for (auto [k, v] : existing) {
        barrier_init_map.Set(k, v);
      }
    }
    ann.Set("barrier_init", barrier_init_map);
    block_ptr->annotations = std::move(ann);

    // Update thread extent at the tiled WS level so LayoutInference sees
    // the producer branch as live and can analyze explicit TMA copies.
    num_threads_ = consumer_extent + producer_extent;
    ws_transformed_ = true;

    // Rebuild BlockRealize.
    BlockRealize new_realize = ffi::GetRef<BlockRealize>(orig_realize);
    new_realize.CopyOnWrite()->block = new_block;
    return new_realize;
  }

  struct PipelineLetBinding {
    Var var;
    PrimExpr value;
  };

  struct PipelineInvocationPath {
    std::vector<For> enclosing_loops;
    std::vector<PrimExpr> guards;
    std::unordered_map<Var, PipelineLetBinding, ObjectPtrHash, ObjectPtrEqual>
        let_bindings;
  };

  struct PipelineInvocationAnalysis {
    bool may_repeat{false};
    Optional<PrimExpr> affine_iteration;
    int64_t static_invocation_count{0};
    Optional<For> outermost_repeated_loop;
    Optional<For> role_scope_loop;
    bool role_scope_liftable{false};
  };

  class PipelineOuterCollectiveDetector : public StmtExprVisitor {
  public:
    static bool Detect(const Stmt &stmt, const For &pipeline_loop) {
      PipelineOuterCollectiveDetector detector(pipeline_loop);
      detector.VisitStmt(stmt);
      return detector.found_;
    }

  private:
    explicit PipelineOuterCollectiveDetector(For pipeline_loop)
        : pipeline_loop_(std::move(pipeline_loop)) {}

    void VisitStmt_(const ForNode *op) final {
      if (ffi::GetRef<For>(op).same_as(pipeline_loop_)) {
        return;
      }
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitExpr_(const CallNode *op) final {
      if (op->op.same_as(builtin::tvm_storage_sync()) ||
          op->op.same_as(builtin::tvm_thread_allreduce()) ||
          op->op.same_as(tl::cluster_sync())) {
        found_ = true;
        return;
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    For pipeline_loop_;
    bool found_{false};
  };

  static bool UsesAnyLoopVar(const PrimExpr &expr,
                             const std::vector<For> &enclosing_loops) {
    return UsesVar(expr, [&](const VarNode *var) {
      Var handle = ffi::GetRef<Var>(var);
      return std::any_of(
          enclosing_loops.begin(), enclosing_loops.end(),
          [&](const For &loop) { return handle.same_as(loop->loop_var); });
    });
  }

  PrimExpr ExpandLoopDependentLets(const PrimExpr &expr,
                                   const PipelineInvocationPath &path) const {
    Map<Var, PrimExpr> substitutions;
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      const auto *var = node.as<VarNode>();
      if (var == nullptr) {
        return;
      }
      auto binding = path.let_bindings.find(ffi::GetRef<Var>(var));
      if (binding == path.let_bindings.end() ||
          !UsesAnyLoopVar(binding->second.value, path.enclosing_loops)) {
        return;
      }
      substitutions.Set(binding->second.var, binding->second.value);
    });
    return substitutions.empty() ? expr : tir::Substitute(expr, substitutions);
  }

  bool CollectPipelineInvocationPath(const Stmt &stmt, const For &pipeline_loop,
                                     PipelineInvocationPath path,
                                     PipelineInvocationPath *result) const {
    if (stmt.same_as(pipeline_loop)) {
      *result = std::move(path);
      return true;
    }
    if (const auto *for_node = stmt.as<ForNode>()) {
      path.enclosing_loops.push_back(ffi::GetRef<For>(for_node));
      return CollectPipelineInvocationPath(for_node->body, pipeline_loop,
                                           std::move(path), result);
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &child : seq->seq) {
        if (ContainsPipelineLoop(child, pipeline_loop)) {
          return CollectPipelineInvocationPath(child, pipeline_loop,
                                               std::move(path), result);
        }
      }
      return false;
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      PrimExpr value = ExpandLoopDependentLets(let->value, path);
      path.let_bindings.insert_or_assign(
          let->var, PipelineLetBinding{let->var, std::move(value)});
      return CollectPipelineInvocationPath(let->body, pipeline_loop,
                                           std::move(path), result);
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      if (!is_one(realize->predicate)) {
        path.guards.push_back(
            ExpandLoopDependentLets(realize->predicate, path));
      }
      return CollectPipelineInvocationPath(realize->block->body, pipeline_loop,
                                           std::move(path), result);
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      return CollectPipelineInvocationPath(block->body, pipeline_loop,
                                           std::move(path), result);
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return CollectPipelineInvocationPath(attr->body, pipeline_loop,
                                           std::move(path), result);
    }
    if (const auto *if_then_else = stmt.as<IfThenElseNode>()) {
      PrimExpr condition =
          ExpandLoopDependentLets(if_then_else->condition, path);
      if (ContainsPipelineLoop(if_then_else->then_case, pipeline_loop)) {
        path.guards.push_back(std::move(condition));
        return CollectPipelineInvocationPath(
            if_then_else->then_case, pipeline_loop, std::move(path), result);
      }
      if (if_then_else->else_case.defined() &&
          ContainsPipelineLoop(if_then_else->else_case.value(),
                               pipeline_loop)) {
        path.guards.push_back(Not(condition));
        return CollectPipelineInvocationPath(if_then_else->else_case.value(),
                                             pipeline_loop, std::move(path),
                                             result);
      }
    }
    return false;
  }

  PipelineInvocationAnalysis
  AnalyzePipelineInvocations(const Stmt &stmt, const For &pipeline_loop,
                             const PrimExpr &inner_linear_idx) const {
    PipelineInvocationAnalysis analysis;
    PipelineInvocationPath path;
    if (!CollectPipelineInvocationPath(stmt, pipeline_loop, {}, &path)) {
      return analysis;
    }

    for (const For &loop : path.enclosing_loops) {
      const int64_t *extent = as_const_int(loop->extent);
      analysis.may_repeat |= extent == nullptr || *extent > 1;
    }
    if (std::any_of(
            path.enclosing_loops.begin(), path.enclosing_loops.end(),
            [](const For &loop) { return loop->kind != ForKind::kSerial; })) {
      return analysis;
    }
    for (const PrimExpr &guard : path.guards) {
      if (SideEffect(guard) > CallEffectKind::kPure) {
        return analysis;
      }
    }
    const int64_t *pipeline_extent = as_const_int(pipeline_loop->extent);
    if (pipeline_extent != nullptr && *pipeline_extent > 0) {
      int64_t invocation_count = *pipeline_extent;
      bool static_domain = true;
      for (const For &loop : path.enclosing_loops) {
        const int64_t *extent = as_const_int(loop->extent);
        if (extent == nullptr || *extent <= 0 ||
            invocation_count > std::numeric_limits<int64_t>::max() / *extent) {
          static_domain = false;
          break;
        }
        invocation_count *= *extent;
      }
      if (static_domain) {
        analysis.static_invocation_count = invocation_count;
      }
    }
    if (!analysis.may_repeat) {
      if (!path.enclosing_loops.empty()) {
        analysis.role_scope_loop = path.enclosing_loops.front();
        analysis.role_scope_liftable = !PipelineOuterCollectiveDetector::Detect(
            analysis.role_scope_loop.value(), pipeline_loop);
      }
      return analysis;
    }

    if (UsesAnyLoopVar(pipeline_loop->min, path.enclosing_loops) ||
        UsesAnyLoopVar(pipeline_loop->extent, path.enclosing_loops)) {
      return analysis;
    }
    for (const For &loop : path.enclosing_loops) {
      if (UsesAnyLoopVar(loop->min, path.enclosing_loops) ||
          UsesAnyLoopVar(loop->extent, path.enclosing_loops)) {
        return analysis;
      }
    }

    arith::Analyzer analyzer;
    for (const PrimExpr &guard : path.guards) {
      if (!UsesAnyLoopVar(guard, path.enclosing_loops)) {
        continue;
      }
      // A loop-dependent guard is safe only when it selects a prefix of one
      // repeated serial loop.  More general sparse or multi-dimensional paths
      // continue to use the persistent phase counter.
      if (path.enclosing_loops.size() != 1) {
        return analysis;
      }
      const For &loop = path.enclosing_loops.front();
      PrimExpr next_guard =
          tir::Substitute(guard, {{loop->loop_var, loop->loop_var + 1}});
      if (!analyzer.CanProve(Or(Not(next_guard), guard))) {
        return analysis;
      }
    }

    PrimExpr iteration;
    for (const For &loop : path.enclosing_loops) {
      PrimExpr loop_linear = loop->loop_var - loop->min;
      iteration = iteration.defined() ? iteration * loop->extent + loop_linear
                                      : loop_linear;
    }
    ICHECK(iteration.defined());
    iteration = iteration * pipeline_loop->extent + inner_linear_idx;
    analysis.affine_iteration = analyzer.Simplify(iteration);
    for (const For &loop : path.enclosing_loops) {
      const int64_t *extent = as_const_int(loop->extent);
      if (extent == nullptr || *extent > 1) {
        analysis.outermost_repeated_loop = loop;
        break;
      }
    }
    ICHECK(analysis.outermost_repeated_loop.defined());
    analysis.role_scope_loop = analysis.outermost_repeated_loop;
    analysis.role_scope_liftable = !PipelineOuterCollectiveDetector::Detect(
        analysis.role_scope_loop.value(), pipeline_loop);
    return analysis;
  }

  // --- Find the first For loop with num_stages annotation ---
  Optional<For> FindPipelineLoop(const Stmt &stmt) {
    if (auto *for_node = stmt.as<ForNode>()) {
      if (for_node->annotations.Get("num_stages") &&
          !PipelineDataflowForcesSynchronous(ffi::GetRef<For>(for_node))) {
        return ffi::GetRef<For>(for_node);
      }
      return FindPipelineLoop(for_node->body);
    }
    // Walk through the control-flow and wrapper nodes that may contain a
    // pipeline loop.  Range-coarsened kernels commonly place a pipeline
    // inside an ordinary outer loop and a data-dependent guard.
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        if (Optional<For> result = FindPipelineLoop(s); result.defined()) {
          return result;
        }
      }
    }
    if (auto *let = stmt.as<LetStmtNode>()) {
      return FindPipelineLoop(let->body);
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      return FindPipelineLoop(realize->block->body);
    }
    if (auto *block = stmt.as<BlockNode>()) {
      return FindPipelineLoop(block->body);
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      return FindPipelineLoop(attr->body);
    }
    if (auto *if_then_else = stmt.as<IfThenElseNode>()) {
      if (Optional<For> result = FindPipelineLoop(if_then_else->then_case);
          result.defined()) {
        return result;
      }
      if (if_then_else->else_case.defined()) {
        return FindPipelineLoop(if_then_else->else_case.value());
      }
    }
    return std::nullopt;
  }

  bool ContainsPipelineLoop(const Stmt &stmt, const For &pipeline_loop) const {
    if (stmt.same_as(pipeline_loop)) {
      return true;
    }
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        if (ContainsPipelineLoop(s, pipeline_loop)) {
          return true;
        }
      }
      return false;
    }
    if (auto *let = stmt.as<LetStmtNode>()) {
      return ContainsPipelineLoop(let->body, pipeline_loop);
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      return ContainsPipelineLoop(realize->block->body, pipeline_loop);
    }
    if (auto *block = stmt.as<BlockNode>()) {
      return ContainsPipelineLoop(block->body, pipeline_loop);
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      return ContainsPipelineLoop(attr->body, pipeline_loop);
    }
    if (auto *for_node = stmt.as<ForNode>()) {
      return ContainsPipelineLoop(for_node->body, pipeline_loop);
    }
    if (auto *if_then_else = stmt.as<IfThenElseNode>()) {
      return ContainsPipelineLoop(if_then_else->then_case, pipeline_loop) ||
             (if_then_else->else_case.defined() &&
              ContainsPipelineLoop(if_then_else->else_case.value(),
                                   pipeline_loop));
    }
    return false;
  }

  struct ReplaceResult {
    Stmt stmt;
    bool found{false};
  };

  class SinkGuardedConsumerPostlude : public StmtExprMutator {
  public:
    static Stmt Rewrite(const Stmt &stmt, Var thread_var,
                        PrimExpr consumer_extent) {
      SinkGuardedConsumerPostlude sinker(std::move(thread_var),
                                         std::move(consumer_extent));
      return sinker.VisitStmt(stmt);
    }

  private:
    SinkGuardedConsumerPostlude(Var thread_var, PrimExpr consumer_extent)
        : thread_var_(std::move(thread_var)),
          consumer_extent_(std::move(consumer_extent)) {}

    static bool SameExpr(const PrimExpr &lhs, const PrimExpr &rhs) {
      return ExprDeepEqual()(lhs, rhs);
    }

    bool IsWSBranchStmt(const Stmt &stmt, IfThenElse *branch) const {
      const auto *if_node = stmt.as<IfThenElseNode>();
      if (!if_node || !if_node->else_case.defined()) {
        return false;
      }
      const auto *ge = if_node->condition.as<GENode>();
      if (!ge) {
        return false;
      }
      const auto *lhs = ge->a.as<VarNode>();
      if (!lhs || !ffi::GetRef<Var>(lhs).same_as(thread_var_)) {
        return false;
      }
      if (!SameExpr(ge->b, consumer_extent_)) {
        return false;
      }
      *branch = ffi::GetRef<IfThenElse>(if_node);
      return true;
    }

    bool IsWSBranch(const Stmt &stmt, Stmt *container,
                    IfThenElse *branch) const {
      if (IsWSBranchStmt(stmt, branch)) {
        *container = stmt;
        return true;
      }
      const auto *attr_node = stmt.as<AttrStmtNode>();
      if (!attr_node || attr_node->attr_key != attr::kWarpSpecializationScope) {
        return false;
      }
      if (!IsWSBranchStmt(attr_node->body, branch)) {
        return false;
      }
      *container = stmt;
      return true;
    }

    bool IsGuardedConsumerStmt(const Stmt &stmt, Stmt *body) const {
      const auto *if_node = stmt.as<IfThenElseNode>();
      if (!if_node || if_node->else_case.defined()) {
        return false;
      }
      const auto *lt = if_node->condition.as<LTNode>();
      if (!lt) {
        return false;
      }
      const auto *lhs = lt->a.as<VarNode>();
      if (!lhs || !ffi::GetRef<Var>(lhs).same_as(thread_var_)) {
        return false;
      }
      if (!SameExpr(lt->b, consumer_extent_)) {
        return false;
      }
      *body = if_node->then_case;
      return true;
    }

    static Stmt AppendToStmt(const Stmt &stmt, const Array<Stmt> &suffix) {
      if (suffix.empty()) {
        return stmt;
      }
      Array<Stmt> seq;
      if (const auto *seq_stmt = stmt.as<SeqStmtNode>()) {
        for (const auto &s : seq_stmt->seq) {
          seq.push_back(s);
        }
      } else {
        seq.push_back(stmt);
      }
      for (const auto &s : suffix) {
        seq.push_back(s);
      }
      return seq.size() == 1 ? seq[0] : SeqStmt(seq);
    }

    Stmt UpdateWSBranchContainer(const Stmt &container,
                                 const IfThenElse &branch,
                                 const Array<Stmt> &consumer_postlude) const {
      auto *branch_ptr = const_cast<IfThenElse &>(branch).CopyOnWrite();
      ICHECK(branch_ptr->else_case.defined());
      branch_ptr->else_case =
          AppendToStmt(branch_ptr->else_case.value(), consumer_postlude);
      if (container.same_as(branch)) {
        return branch;
      }
      AttrStmt attr = Downcast<AttrStmt>(container);
      attr.CopyOnWrite()->body = branch;
      return attr;
    }

    Stmt VisitStmt_(const SeqStmtNode *op) final {
      Array<Stmt> visited;
      for (const auto &stmt : op->seq) {
        visited.push_back(VisitStmt(stmt));
      }

      Array<Stmt> rebuilt;
      for (int i = 0; i < static_cast<int>(visited.size()); ++i) {
        Stmt ws_container;
        IfThenElse ws_branch;
        if (!IsWSBranch(visited[i], &ws_container, &ws_branch)) {
          rebuilt.push_back(visited[i]);
          continue;
        }

        Array<Stmt> consumer_postlude;
        int j = i + 1;
        for (; j < static_cast<int>(visited.size()); ++j) {
          Stmt body;
          if (!IsGuardedConsumerStmt(visited[j], &body)) {
            break;
          }
          consumer_postlude.push_back(body);
        }
        if (consumer_postlude.empty()) {
          rebuilt.push_back(visited[i]);
          continue;
        }

        rebuilt.push_back(UpdateWSBranchContainer(ws_container, ws_branch,
                                                  consumer_postlude));
        i = j - 1;
      }

      return rebuilt.size() == 1 ? rebuilt[0] : SeqStmt(rebuilt);
    }

    Var thread_var_;
    PrimExpr consumer_extent_;
  };

  class LiftNestedWarpSpecialization : public StmtExprMutator {
  public:
    static Stmt Rewrite(const Stmt &stmt, Var outer_loop_var, Var thread_var,
                        PrimExpr consumer_extent, Array<IntImm> ws_partition,
                        bool *lifted) {
      LiftNestedWarpSpecialization rewriter(
          std::move(outer_loop_var), std::move(thread_var),
          std::move(consumer_extent), std::move(ws_partition));
      Stmt result = rewriter.VisitStmt(stmt);
      *lifted = rewriter.lifted_;
      return result;
    }

  private:
    enum class Role { kProducer, kConsumer };

    class RoleProjector : public StmtExprMutator {
    public:
      static Stmt Project(const Stmt &stmt, Var thread_var,
                          PrimExpr consumer_extent, Role role,
                          int *selected_scopes) {
        RoleProjector projector(std::move(thread_var),
                                std::move(consumer_extent), role);
        Stmt result = projector.VisitStmt(stmt);
        *selected_scopes = projector.selected_scopes_;
        return result;
      }

    private:
      RoleProjector(Var thread_var, PrimExpr consumer_extent, Role role)
          : thread_var_(std::move(thread_var)),
            consumer_extent_(std::move(consumer_extent)), role_(role) {}

      bool IsProducerCondition(const PrimExpr &condition) const {
        if (const auto *ge = condition.as<GENode>()) {
          return ge->a.same_as(thread_var_) &&
                 ExprDeepEqual()(ge->b, consumer_extent_);
        }
        if (const auto *le = condition.as<LENode>()) {
          return ExprDeepEqual()(le->a, consumer_extent_) &&
                 le->b.same_as(thread_var_);
        }
        return false;
      }

      bool IsConsumerCondition(const PrimExpr &condition) const {
        const auto *lt = condition.as<LTNode>();
        return lt != nullptr && lt->a.same_as(thread_var_) &&
               ExprDeepEqual()(lt->b, consumer_extent_);
      }

      Stmt SelectBranch(const IfThenElseNode *op, bool select_then) {
        if (select_then) {
          return VisitStmt(op->then_case);
        }
        return op->else_case.defined()
                   ? VisitStmt(op->else_case.value())
                   : Stmt(Evaluate(IntImm(DataType::Int(32), 0)));
      }

      Stmt VisitStmt_(const AttrStmtNode *op) final {
        if (op->attr_key == attr::kWarpSpecializationScope) {
          ++selected_scopes_;
          return VisitStmt(op->body);
        }
        return StmtExprMutator::VisitStmt_(op);
      }

      Stmt VisitStmt_(const IfThenElseNode *op) final {
        if (IsProducerCondition(op->condition)) {
          return SelectBranch(op, role_ == Role::kProducer);
        }
        if (IsConsumerCondition(op->condition)) {
          return SelectBranch(op, role_ == Role::kConsumer);
        }
        return StmtExprMutator::VisitStmt_(op);
      }

      Var thread_var_;
      PrimExpr consumer_extent_;
      Role role_;
      int selected_scopes_{0};
    };

    LiftNestedWarpSpecialization(Var outer_loop_var, Var thread_var,
                                 PrimExpr consumer_extent,
                                 Array<IntImm> ws_partition)
        : outer_loop_var_(std::move(outer_loop_var)),
          thread_var_(std::move(thread_var)),
          consumer_extent_(std::move(consumer_extent)),
          ws_partition_(std::move(ws_partition)) {}

    Stmt VisitStmt_(const ForNode *op) final {
      if (!op->loop_var.same_as(outer_loop_var_)) {
        return StmtExprMutator::VisitStmt_(op);
      }

      Stmt loop = ffi::GetRef<For>(op);
      int producer_scopes = 0;
      int consumer_scopes = 0;
      Stmt producer =
          RoleProjector::Project(loop, thread_var_, consumer_extent_,
                                 Role::kProducer, &producer_scopes);
      Stmt consumer =
          RoleProjector::Project(loop, thread_var_, consumer_extent_,
                                 Role::kConsumer, &consumer_scopes);
      if (producer_scopes != 1 || consumer_scopes != 1) {
        return StmtExprMutator::VisitStmt_(op);
      }

      lifted_ = true;
      Stmt branch =
          IfThenElse(GE(thread_var_, consumer_extent_), producer, consumer);
      return AttrStmt(ws_partition_, attr::kWarpSpecializationScope, 0, branch);
    }

    Var outer_loop_var_;
    Var thread_var_;
    PrimExpr consumer_extent_;
    Array<IntImm> ws_partition_;
    bool lifted_{false};
  };

  Stmt GuardConsumerOnly(const Stmt &stmt, PrimExpr consumer_extent) {
    return IfThenElse(LT(thread_iv_->var, consumer_extent), stmt);
  }

  void SeedEnclosingLetUses(const Stmt &stmt, const For &pipeline_loop) {
    if (stmt.same_as(pipeline_loop)) {
      return;
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      if (ContainsPipelineLoop(let->body, pipeline_loop)) {
        shared_prelude_live_seed_.AddUses(LocalAccessCollector::Collect(
            Evaluate(let->value), buffer_data_to_buffer_));
        SeedEnclosingLetUses(let->body, pipeline_loop);
      }
      return;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &child : seq->seq) {
        if (ContainsPipelineLoop(child, pipeline_loop)) {
          SeedEnclosingLetUses(child, pipeline_loop);
          return;
        }
      }
      return;
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      SeedEnclosingLetUses(realize->block->body, pipeline_loop);
      return;
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      SeedEnclosingLetUses(block->body, pipeline_loop);
      return;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      SeedEnclosingLetUses(attr->body, pipeline_loop);
      return;
    }
    if (const auto *for_node = stmt.as<ForNode>()) {
      SeedEnclosingLetUses(for_node->body, pipeline_loop);
      return;
    }
    if (const auto *if_then_else = stmt.as<IfThenElseNode>()) {
      if (ContainsPipelineLoop(if_then_else->then_case, pipeline_loop)) {
        SeedEnclosingLetUses(if_then_else->then_case, pipeline_loop);
      } else if (if_then_else->else_case.defined() &&
                 ContainsPipelineLoop(if_then_else->else_case.value(),
                                      pipeline_loop)) {
        SeedEnclosingLetUses(if_then_else->else_case.value(), pipeline_loop);
      }
    }
  }

  ReplaceResult ReplacePipelineLoopInStmt(const Stmt &stmt,
                                          const For &pipeline_loop,
                                          const Stmt &ws_body,
                                          PrimExpr consumer_extent) {
    if (stmt.same_as(pipeline_loop)) {
      return {ws_body, true};
    }
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> new_seq;
      bool found = false;
      // First pass: find which child contains the pipeline loop.
      int loop_idx = -1;
      for (int i = 0; i < static_cast<int>(seq->seq.size()); ++i) {
        if (ContainsPipelineLoop(seq->seq[i], pipeline_loop)) {
          loop_idx = i;
          break;
        }
      }
      if (loop_idx < 0) {
        return {stmt, false};
      }
      // Let values enclosing the pipeline child are evaluated in the shared
      // prelude. Seed their uses before classifying sibling statements in
      // this SeqStmt; the recursive replacement reaches those lets too late
      // for the parent's backward liveness walk.
      SeedEnclosingLetUses(seq->seq[loop_idx], pipeline_loop);
      // Propagate liveness backwards through prelude statements so that
      // transitive dependencies are captured.  For example, if consumer
      // needs `m_start` and `m_start` is defined by a prelude statement
      // that reads `cur_batch_idx`, the loop defining `cur_batch_idx`
      // must also be visible to the consumer.
      {
        LocalLiveSet shared_live = shared_prelude_live_seed_;
        LocalLiveSet producer_live = producer_prelude_live_seed_;
        LocalLiveSet consumer_live = consumer_prelude_live_seed_;
        for (int i = loop_idx - 1; i >= 0; --i) {
          LocalAccessSummary summary = LocalAccessCollector::Collect(
              seq->seq[i], buffer_data_to_buffer_);
          if (!summary.HasTrackedDefs())
            continue;
          if (shared_live.NeedsAnyDef(summary)) {
            shared_live.AddUses(summary);
          }
          if (producer_live.NeedsAnyDef(summary)) {
            producer_live.AddUses(summary);
          }
          if (consumer_live.NeedsAnyDef(summary)) {
            consumer_live.AddUses(summary);
          }
        }
        shared_prelude_live_seed_ = shared_live;
        producer_prelude_live_seed_ = producer_live;
        consumer_prelude_live_seed_ = consumer_live;
      }
      // Classify pre-loop statements using branch-private def/use sets.
      // Shared-prelude statements stay in place; branch-private definitions
      // move next to the branch that consumes them, or are duplicated when
      // both producer and consumer need the same definition.
      bool pipeline_is_direct_child = seq->seq[loop_idx].same_as(pipeline_loop);
      for (int i = 0; i < loop_idx; ++i) {
        switch (ClassifyPreludeStmt(
            seq->seq[i], buffer_data_to_buffer_, shared_prelude_live_seed_,
            producer_prelude_live_seed_, consumer_prelude_live_seed_)) {
        case PreludeStmtPlacement::kProducerOnly:
          extracted_producer_init_.push_back(seq->seq[i]);
          break;
        case PreludeStmtPlacement::kConsumerOnly:
          if (pipeline_is_direct_child) {
            extracted_consumer_init_.push_back(seq->seq[i]);
          } else {
            // Preserve the original control scope when the pipeline is nested
            // more deeply. Moving an accumulator initializer through an outer
            // guard changes its dominance over post-pipeline consumers and can
            // unnecessarily extend fragment live ranges.
            new_seq.push_back(GuardConsumerOnly(seq->seq[i], consumer_extent));
          }
          break;
        case PreludeStmtPlacement::kDuplicateToBoth:
          extracted_producer_init_.push_back(seq->seq[i]);
          extracted_consumer_init_.push_back(seq->seq[i]);
          break;
        case PreludeStmtPlacement::kKeepSharedPrelude:
          if (auto it = common_prelude_rewrites_.find(seq->seq[i]);
              it != common_prelude_rewrites_.end()) {
            new_seq.push_back(it->second);
          } else {
            new_seq.push_back(seq->seq[i]);
          }
          break;
        }
      }
      // Replace the pipeline loop itself.
      ReplaceResult result = ReplacePipelineLoopInStmt(
          seq->seq[loop_idx], pipeline_loop, ws_body, consumer_extent);
      new_seq.push_back(result.stmt);
      // Guard post-loop siblings.
      for (int i = loop_idx + 1; i < static_cast<int>(seq->seq.size()); ++i) {
        new_seq.push_back(GuardConsumerOnly(seq->seq[i], consumer_extent));
      }
      return {new_seq.size() == 1 ? new_seq[0] : SeqStmt(new_seq), true};
    }
    if (auto *let = stmt.as<LetStmtNode>()) {
      // The LetStmt value is evaluated in the shared prelude (outside
      // both producer and consumer branches).  If it reads branch-private
      // buffers or vars defined by a prelude statement, that definition
      // must remain available in the shared scope.  Propagate such uses
      // into both live seeds before visiting the body so the upstream
      // prelude-statement classifier sees them when classifying the
      // surrounding SeqStmt.
      {
        LocalAccessSummary val_summary = LocalAccessCollector::Collect(
            Evaluate(let->value), buffer_data_to_buffer_);
        shared_prelude_live_seed_.AddUses(val_summary);
      }
      ReplaceResult result = ReplacePipelineLoopInStmt(
          let->body, pipeline_loop, ws_body, consumer_extent);
      if (!result.found) {
        return {stmt, false};
      }
      return {LetStmt(let->var, let->value, result.stmt), true};
    }
    if (auto *realize = stmt.as<BlockRealizeNode>()) {
      ReplaceResult result = ReplacePipelineLoopInStmt(
          realize->block->body, pipeline_loop, ws_body, consumer_extent);
      if (!result.found) {
        return {stmt, false};
      }
      Block block = realize->block;
      block.CopyOnWrite()->body = result.stmt;
      BlockRealize new_realize = ffi::GetRef<BlockRealize>(realize);
      new_realize.CopyOnWrite()->block = block;
      return {new_realize, true};
    }
    if (auto *block = stmt.as<BlockNode>()) {
      ReplaceResult result = ReplacePipelineLoopInStmt(
          block->body, pipeline_loop, ws_body, consumer_extent);
      if (!result.found) {
        return {stmt, false};
      }
      Block new_block = ffi::GetRef<Block>(block);
      new_block.CopyOnWrite()->body = result.stmt;
      return {new_block, true};
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      ReplaceResult result = ReplacePipelineLoopInStmt(
          attr->body, pipeline_loop, ws_body, consumer_extent);
      if (!result.found) {
        return {stmt, false};
      }
      AttrStmt new_attr = ffi::GetRef<AttrStmt>(attr);
      new_attr.CopyOnWrite()->body = result.stmt;
      return {new_attr, true};
    }
    if (auto *for_node = stmt.as<ForNode>()) {
      ReplaceResult result = ReplacePipelineLoopInStmt(
          for_node->body, pipeline_loop, ws_body, consumer_extent);
      if (!result.found) {
        return {stmt, false};
      }
      For new_for = ffi::GetRef<For>(for_node);
      new_for.CopyOnWrite()->body = result.stmt;
      return {new_for, true};
    }
    if (auto *if_then_else = stmt.as<IfThenElseNode>()) {
      ReplaceResult then_result = ReplacePipelineLoopInStmt(
          if_then_else->then_case, pipeline_loop, ws_body, consumer_extent);
      if (then_result.found) {
        IfThenElse new_if = ffi::GetRef<IfThenElse>(if_then_else);
        new_if.CopyOnWrite()->then_case = then_result.stmt;
        return {new_if, true};
      }
      if (if_then_else->else_case.defined()) {
        ReplaceResult else_result =
            ReplacePipelineLoopInStmt(if_then_else->else_case.value(),
                                      pipeline_loop, ws_body, consumer_extent);
        if (else_result.found) {
          IfThenElse new_if = ffi::GetRef<IfThenElse>(if_then_else);
          new_if.CopyOnWrite()->else_case = else_result.stmt;
          return {new_if, true};
        }
      }
    }
    return {stmt, false};
  }

  // --- PCThreadIdxRewriter (simplified for tile-op level) ---
  class PCThreadIdxRewriter : public StmtExprMutator {
  public:
    static Stmt Rewrite(Stmt stmt, Var thread_var, PrimExpr replaced,
                        PrimExpr thread_extent, bool do_shuffle) {
      PCThreadIdxRewriter r(std::move(thread_var), std::move(replaced),
                            std::move(thread_extent));
      return r(std::move(stmt));
    }

  private:
    PCThreadIdxRewriter(Var thread_var, PrimExpr replaced,
                        PrimExpr thread_extent)
        : thread_var_(std::move(thread_var)), replaced_(std::move(replaced)),
          thread_extent_(std::move(thread_extent)) {}

    PrimExpr VisitExpr_(const VarNode *var) final {
      if (ffi::GetRef<Var>(var).same_as(thread_var_)) {
        return replaced_;
      }
      return StmtExprMutator::VisitExpr_(var);
    }

    Var thread_var_;
    PrimExpr replaced_;
    PrimExpr thread_extent_;
  };

  // State
  Target target_;
  int cluster_size_{1};
  String cross_handler_handoff_role_;
  bool cross_handler_handoff_enabled_{false};
  Optional<Var> handoff_stage_count_var_;
  IterVar thread_iv_;
  Optional<PrimExpr> num_threads_; // total (consumer + producer)
  bool ws_transformed_{false};
  std::string rejection_reason_;
  BufferDataToBufferMap buffer_data_to_buffer_;
  StmtRewriteMap common_prelude_rewrites_;
  LocalLiveSet shared_prelude_live_seed_;
  LocalLiveSet producer_prelude_live_seed_;
  LocalLiveSet consumer_prelude_live_seed_;
  Array<Stmt> extracted_producer_init_;
  Array<Stmt> extracted_consumer_init_;
};

// ---------------------------------------------------------------------------
// Detect if manual WS is already present (skip if so)
// ---------------------------------------------------------------------------

class ManualWSDetector : public StmtExprVisitor {
public:
  static bool HasManualWS(const Stmt &stmt) {
    ManualWSDetector d;
    d(stmt);
    return d.found_;
  }

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    // Detect both the T.ws() language-level attr ("warp_specialize") and
    // the compiler-level attr (kWarpSpecializationScope).
    if (op->attr_key == "warp_specialize" ||
        op->attr_key == attr::kWarpSpecializationScope) {
      found_ = true;
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool found_{false};
};

/// Quick pre-scan: check if the function contains a pipelined loop (num_stages
/// >= 1) with at least one TMA load producer tile op and no manual layout
/// annotations (which are incompatible with early MVB expansion).
/// Check whether a layout annotation on a shared buffer is compatible with
/// TMA.  TMA supports identity (linear) layouts and the three standard
/// swizzle modes (32B / 64B / 128B).  Any other layout (e.g. padded,
/// Volta-style) cannot be used with TMA.
static bool IsTmaCompatibleLayout(const Layout &layout, const Buffer &buffer) {
  // Recognised swizzle → TMA with swizzle.
  if (DetectSwizzleMode(layout, buffer) != SwizzleMode::kNone) {
    return true;
  }
  // Identity / row-major linear → TMA without swizzle.
  if (StructuralEqual()(layout, makeLinearLayout(buffer->shape))) {
    return true;
  }
  return false;
}

class TiledWSCandidate : public StmtExprVisitor {
public:
  static bool Check(const Stmt &stmt, Target target) {
    TiledWSCandidate c;
    c.target_ = target;
    c(stmt);
    return c.has_pipeline_loop_ && c.has_tma_tile_op_;
  }

private:
  void VisitStmt_(const ForNode *op) final {
    bool old = in_pipeline_;
    if (!PipelineDataflowForcesSynchronous(ffi::GetRef<For>(op))) {
      if (auto anno = op->annotations.Get("num_stages")) {
        if (auto *imm = anno->as<IntImmNode>()) {
          if (imm->value >= 1) {
            has_pipeline_loop_ = true;
            in_pipeline_ = true;
          }
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
    in_pipeline_ = old;
  }

  void VisitExpr_(const CallNode *op) final {
    if (in_pipeline_ && !has_tma_tile_op_) {
      auto tile_op = ParseOperator(ffi::GetRef<Call>(op));
      if (auto *copy = tile_op.as<CopyNode>()) {
        if (ClassifyCopy(copy, target_) == TileStmtKind::kTmaProducer) {
          // If the destination buffer has a layout annotation, verify
          // that the layout is TMA-compatible (swizzle or linear).
          // Copies whose layout is incompatible with TMA cannot become
          // TMA producers.
          if (HasTmaCompatibleLayout(copy->dst)) {
            has_tma_tile_op_ = true;
          }
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BlockNode *op) final {
    // Collect layout_map entries so we can cross-check TMA copy targets.
    if (op->annotations.count("layout_map")) {
      auto anno = op->annotations.Get("layout_map");
      if (auto gmap = anno->as<Map<ObjectRef, ObjectRef>>(); gmap.has_value()) {
        for (const auto &[key, val] : gmap.value()) {
          Layout layout;
          if (auto l = val.as<Layout>(); l.has_value())
            layout = l.value();
          if (auto buf = key.as<Buffer>(); buf.has_value()) {
            layout_map_[buf.value()->data] = {buf.value(), layout};
          } else if (auto var = key.as<Var>(); var.has_value()) {
            for (const auto &buf : op->alloc_buffers) {
              if (buf->data.same_as(var.value())) {
                layout_map_[buf->data] = {buf, layout};
                break;
              }
            }
          }
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  /// A copy destination is TMA-compatible if it has no layout annotation,
  /// or its annotated layout is a recognised swizzle / linear layout.
  bool HasTmaCompatibleLayout(const Buffer &dst) const {
    auto it = layout_map_.find(dst->data);
    if (it == layout_map_.end()) {
      return true; // no annotation → identity layout → TMA OK
    }
    const auto &[buf, layout] = it->second;
    if (!layout.defined()) {
      return false; // annotation present but layout not parseable
    }
    return IsTmaCompatibleLayout(layout, buf);
  }

  Target target_;
  bool in_pipeline_{false};
  bool has_pipeline_loop_{false};
  bool has_tma_tile_op_{false};
  // Map from buffer data Var to (Buffer, Layout) for layout_map entries.
  BufferLayoutMap layout_map_;
};

} // namespace

// ---------------------------------------------------------------------------
// Pass registration
// ---------------------------------------------------------------------------

namespace {

int RequestedPipelineStages(const PrimFunc &f) {
  class Collector : public StmtExprVisitor {
  public:
    void VisitStmt_(const ForNode *op) final {
      if (auto stages = op->annotations.Get("num_stages")) {
        if (!PipelineDataflowForcesSynchronous(ffi::GetRef<For>(op))) {
          if (const auto *value = stages.value().as<IntImmNode>()) {
            max_stages = std::max(max_stages, static_cast<int>(value->value));
          }
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    int max_stages{0};
  } collector;
  collector(f->body);
  return collector.max_stages;
}

PrimFunc AppendPipelineDecision(PrimFunc f, int requested_stages,
                                String implementation, String reason,
                                bool fallback) {
  Array<Map<String, ObjectRef>> decisions;
  if (auto previous = f->GetAttr<Array<Map<String, ObjectRef>>>(
          kPipelineLoweringDecisions)) {
    decisions = previous.value();
  }
  Map<String, ObjectRef> decision;
  decision.Set("schema_version",
               Integer(kPipelineDecisionCurrentSchemaVersion));
  decision.Set("loop_index", Integer(static_cast<int>(decisions.size())));
  decision.Set("requested_stages", Integer(requested_stages));
  decision.Set("selected_implementation", StringImm(std::move(implementation)));
  decision.Set("fallback", Bool(fallback));
  decision.Set("selection_reason", StringImm(std::move(reason)));
  decisions.push_back(std::move(decision));
  f = WithAttr(std::move(f), kPipelineDecisionSchemaVersion,
               Integer(kPipelineDecisionCurrentSchemaVersion));
  return WithAttr(std::move(f), kPipelineLoweringDecisions, decisions);
}

} // namespace

tvm::transform::Pass ProducerConsumerWarpSpecialized() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    // Skip if disabled.
    if (ctx->GetConfig(kDisableWarpSpecialized, Optional<Bool>())
            .value_or(false)) {
      return f;
    }
    // Skip if the function already has manual WS.
    if (ManualWSDetector::HasManualWS(f->body)) {
      return f;
    }
    // Skip if TMA is not available.
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetHasBulkCopy(target.value())) {
      return f;
    }
    // Only apply MVB + WS if the function is a tiled WS candidate.
    if (!TiledWSCandidate::Check(f->body, target.value())) {
      DLOG(WARNING) << "[WS] skipped: no TMA copies in pipeline loop";
      return f;
    }
    DLOG(WARNING) << "[WS] candidate found, applying MVB + WS";
    int requested_stages = RequestedPipelineStages(f);
    // Expand shared buffers for pipelining before the WS split.
    // Keep the original so we can fall back if the WS rewriter doesn't fire
    // (e.g. non-tile-op consumers in the loop body).
    PrimFunc original_f = f;
    f = ApplyMultiVersionBufferRewriter(std::move(f));
    std::string rejection_reason;
    PrimFunc result =
        ProducerConsumerWSRewriter::Substitute(std::move(f), &rejection_reason);
    if (!result->HasNonzeroAttr(kTiledWSApplied)) {
      DLOG(WARNING) << "[WS] rewriter did not fire, falling back";
      // The TMA kernel needs warp specialization for correct pipelined
      // execution.  Since the tiled rewriter could not apply WS (e.g.
      // conditional loop body), strip pipeline annotations so that
      // PipelinePlanning / InjectSoftwarePipeline do not generate
      // broken non-WS TMA pipeline code.
      class SelectSynchronousPipelineFallback : public tir::StmtExprMutator {
      public:
        explicit SelectSynchronousPipelineFallback(String reason)
            : reason_(std::move(reason)) {}

        tir::Stmt VisitStmt_(const tir::ForNode *op) final {
          auto stmt = tir::StmtExprMutator::VisitStmt_(op);
          const auto *for_node = stmt.as<tir::ForNode>();
          ICHECK(for_node);
          if (for_node->annotations.count("num_stages")) {
            tir::For new_for = Downcast<tir::For>(stmt);
            auto *n = new_for.CopyOnWrite();
            n->annotations.Set(kPipelineDataflowMode,
                               StringImm(kPipelineDataflowModeSynchronous));
            n->annotations.Set(kPipelineDataflowFallbackReason,
                               StringImm(reason_));
            return std::move(new_for);
          }
          return stmt;
        }

      private:
        String reason_;
      };
      String fallback_reason =
          rejection_reason.empty()
              ? String("warp-specialization candidate could not be legally "
                       "rewritten")
              : String(rejection_reason);
      SelectSynchronousPipelineFallback selector(fallback_reason);
      auto synchronous = selector(original_f->body);
      auto *fn = original_f.CopyOnWrite();
      fn->body = synchronous;
      return original_f;
    }
    DLOG(WARNING) << "[WS] transformation applied successfully";
    return AppendPipelineDecision(
        std::move(result), requested_stages, "warp_specialized",
        "TMA producer/consumer dataflow accepted with compiler-owned "
        "barriers and buffer versioning",
        /*fallback=*/false);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ProducerConsumerWarpSpecialized",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ProducerConsumerWarpSpecialized",
                        ProducerConsumerWarpSpecialized);
  refl::GlobalDef().def("tl.transform.ProducerConsumerWarpSpecializedTiled",
                        ProducerConsumerWarpSpecialized);
}

} // namespace tl
} // namespace tvm
