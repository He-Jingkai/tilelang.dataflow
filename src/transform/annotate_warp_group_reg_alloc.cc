/*!
 * \file annotate_warp_group_reg_alloc.cc
 * \brief Annotate warp group reg alloc for warp specialization
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"
#include <functional>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

constexpr int kWarpGroupThreadCount = 128;
constexpr int kMinimumWarpGroupRegisters = 24;
constexpr int kSingleConsumerProducerRegisters = 40;
constexpr int kSingleConsumerRegisters = 232;
constexpr int kDefaultProducerRegisters = kMinimumWarpGroupRegisters;
constexpr int kDefaultConsumerRegisters = 240;
constexpr const char *kCrossHandlerHandoffEnabled =
    "tl.cross_handler_handoff_enabled";

template <typename F>
Stmt RewriteWarpSpecializationBody(const Stmt &stmt, F &&rewrite_if,
                                   bool *rewrote) {
  if (*rewrote) {
    return stmt;
  }

  if (const auto *if_node = stmt.as<IfThenElseNode>()) {
    *rewrote = true;
    return rewrite_if(ffi::GetRef<IfThenElse>(if_node));
  }

  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    Array<Stmt> new_seq;
    bool changed = false;
    for (const auto &sub_stmt : seq->seq) {
      Stmt rewritten =
          RewriteWarpSpecializationBody(sub_stmt, rewrite_if, rewrote);
      changed = changed || !rewritten.same_as(sub_stmt);
      new_seq.push_back(rewritten);
    }
    if (!changed) {
      return stmt;
    }
    return new_seq.size() == 1 ? new_seq[0] : SeqStmt(new_seq);
  }

  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    Stmt new_body =
        RewriteWarpSpecializationBody(attr->body, rewrite_if, rewrote);
    if (new_body.same_as(attr->body)) {
      return stmt;
    }
    return AttrStmt(attr->node, attr->attr_key, attr->value, new_body);
  }

  if (const auto *let_node = stmt.as<LetStmtNode>()) {
    Stmt new_body =
        RewriteWarpSpecializationBody(let_node->body, rewrite_if, rewrote);
    if (new_body.same_as(let_node->body)) {
      return stmt;
    }
    return LetStmt(let_node->var, let_node->value, new_body);
  }

  if (const auto *realize = stmt.as<BlockRealizeNode>()) {
    const Block &block = realize->block;
    Stmt new_body =
        RewriteWarpSpecializationBody(block->body, rewrite_if, rewrote);
    if (new_body.same_as(block->body)) {
      return stmt;
    }
    Block new_block(block->iter_vars, block->reads, block->writes,
                    block->name_hint, new_body, block->init,
                    block->alloc_buffers, block->match_buffers,
                    block->annotations);
    return BlockRealize(realize->iter_values, realize->predicate, new_block);
  }

  if (const auto *block = stmt.as<BlockNode>()) {
    Stmt new_body =
        RewriteWarpSpecializationBody(block->body, rewrite_if, rewrote);
    if (new_body.same_as(block->body)) {
      return stmt;
    }
    return Block(block->iter_vars, block->reads, block->writes,
                 block->name_hint, new_body, block->init, block->alloc_buffers,
                 block->match_buffers, block->annotations);
  }

  return stmt;
}

} // namespace

class SetMaxNRegCollector : public StmtExprVisitor {
public:
  struct Result {
    Array<IntImm> nreg;
    bool preserve_explicit_set_max_nreg{false};
  };

  static Result Collect(const PrimFunc &f) {
    SetMaxNRegCollector collector;
    collector(f->body);
    if (collector.warp_specialized_) {
      return {Array<IntImm>({}), true};
    }
    Array<IntImm> nreg = collector.has_no_set_max_nreg_
                             ? Array<IntImm>({IntImm(DataType::Int(32), -1),
                                              IntImm(DataType::Int(32), -1)})
                             : collector.nreg_;
    return {nreg, false};
  }

private:
  void VisitStmt_(const EvaluateNode *op) final {
    if (const CallNode *call = op->value.as<CallNode>()) {
      if (call->op.same_as(set_max_nreg())) {
        return;
      } else if (call->op.same_as(annotate_producer_reg_dealloc())) {
        auto reg_hint = call->args[0].as<IntImmNode>()->value;
        ICHECK(reg_hint <= 240 && reg_hint >= 24)
            << "Invalid reg hint: " << reg_hint;
        nreg_.Set(0, IntImm(DataType::Int(32), reg_hint));
      } else if (call->op.same_as(annotate_consumer_reg_alloc())) {
        auto reg_hint = call->args[0].as<IntImmNode>()->value;
        ICHECK(reg_hint <= 240 && reg_hint >= 24)
            << "Invalid reg hint: " << reg_hint;
        nreg_.Set(1, IntImm(DataType::Int(32), reg_hint));
      } else if (call->op.same_as(no_set_max_nreg())) {
        has_no_set_max_nreg_ = true;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == attr::kCustomWarpSpecialization) {
      warp_specialized_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  Array<IntImm> nreg_{IntImm(DataType::Int(32), 0),
                      IntImm(DataType::Int(32), 0)};
  bool has_no_set_max_nreg_ = false;
  bool warp_specialized_ = false;
};

class SimtCopyDetector : public StmtExprVisitor {
public:
  static bool Detect(const Stmt &stmt) {
    SimtCopyDetector detector;
    detector.VisitStmt(stmt);
    return detector.has_simt_copy_;
  }

private:
  void VisitStmt_(const EvaluateNode *op) final {
    if (const CallNode *call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::ptx_cp_async()) ||
          call->op.same_as(tl::ptx_cp_async())) {
        has_simt_copy_ = true;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    auto scope =
        runtime::StorageScope::Create(GetPtrStorageScope(op->buffer->data));
    if (scope.to_string() != "global") {
      has_simt_copy_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool has_simt_copy_{false};
};

class SetMaxNRegInjector : public StmtExprMutator {
public:
  static PrimFunc Inject(PrimFunc f) {
    auto T = SetMaxNRegInjector();
    SetMaxNRegCollector::Result result = SetMaxNRegCollector::Collect(f);
    T.nreg_ = result.nreg;
    T.preserve_explicit_set_max_nreg_ = result.preserve_explicit_set_max_nreg;
    T.normalize_cross_handler_reg_state_ =
        f->HasNonzeroAttr(kCrossHandlerHandoffEnabled);
    if (T.nreg_.empty()) {
      return f;
    }
    f.CopyOnWrite()->body = T(f->body);
    return f;
  }

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const CallNode *call = op->value.as<CallNode>()) {
      if (!preserve_explicit_set_max_nreg_ &&
          call->op.same_as(set_max_nreg())) {
        return StmtExprMutator::VisitStmt_(op);
      }
      if (call->op.same_as(annotate_producer_reg_dealloc()) ||
          call->op.same_as(annotate_consumer_reg_alloc()) ||
          call->op.same_as(no_set_max_nreg())) {
        // Remove annotations after they have been consumed by this pass.
        return Evaluate(0);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent &&
        Downcast<IterVar>(op->node)->thread_tag == "threadIdx.x") {
      thread_iv_ = Downcast<IterVar>(op->node);
      need_update_thread_extent_ = false;
      AttrStmt attr_stmt = Downcast<AttrStmt>(StmtExprMutator::VisitStmt_(op));
      if (need_update_thread_extent_) {
        thread_iv_.CopyOnWrite()->dom = {0, updated_thread_extent_.value()};
        attr_stmt.CopyOnWrite()->node = thread_iv_;
        attr_stmt.CopyOnWrite()->value = updated_thread_extent_.value();
      }
      thread_iv_ = {};
      return attr_stmt;
    } else if (op->attr_key == attr::kWarpSpecializationScope) {
      bool rewrote_ws_body = false;
      auto rewrite_if = [&](const IfThenElse &if_then_else) -> Stmt {
        auto producer_body = if_then_else->then_case;
        Optional<Stmt> consumer_body = if_then_else->else_case;
        // In some degenerate warp-specialized patterns (e.g., producer-only),
        // the consumer body may be absent. Handle gracefully by only
        // annotating the producer side when consumer is missing.

        auto dec_reg = nreg_[0].as<IntImmNode>()->value;
        auto inc_reg = nreg_[1].as<IntImmNode>()->value;

        auto consumer_reg_stmt = Evaluate(0);
        auto producer_reg_stmt = Evaluate(0);
        bool inject_reg_reallocation = false;
        int final_dec_reg = -1;
        int final_inc_reg = -1;

        // Default hints stay conservative: skip auto-injection when producer
        // contains SIMT copy-like statements. Explicit user hints should still
        // be honored even in that case.
        bool has_simt_copy = SimtCopyDetector::Detect(producer_body);
        bool has_explicit_hints = dec_reg != 0 || inc_reg != 0;

        if (dec_reg != -1 && inc_reg != -1 &&
            (has_explicit_hints || !has_simt_copy)) {
          int default_dec_reg = kDefaultProducerRegisters;
          int default_inc_reg = kDefaultConsumerRegisters;
          const auto *producer_partition = if_then_else->condition.as<GENode>();
          const auto *canonical_partition =
              if_then_else->condition.as<LENode>();
          const int64_t *consumer_thread_extent = nullptr;
          if (producer_partition != nullptr) {
            consumer_thread_extent = as_const_int(producer_partition->b);
          } else if (canonical_partition != nullptr) {
            consumer_thread_extent = as_const_int(canonical_partition->a);
          }
          if (consumer_thread_extent != nullptr &&
              *consumer_thread_extent == kWarpGroupThreadCount) {
            default_dec_reg = kSingleConsumerProducerRegisters;
            default_inc_reg = kSingleConsumerRegisters;
          }
          final_dec_reg = has_explicit_hints ? dec_reg : default_dec_reg;
          final_inc_reg = has_explicit_hints ? inc_reg : default_inc_reg;
          producer_reg_stmt =
              Evaluate(Call(DataType::Handle(), set_max_nreg(),
                            {IntImm(DataType::Int(32), final_dec_reg),
                             IntImm(DataType::Int(32), 0)}));
          consumer_reg_stmt =
              Evaluate(Call(DataType::Handle(), set_max_nreg(),
                            {IntImm(DataType::Int(32), final_inc_reg),
                             IntImm(DataType::Int(32), 1)}));
          inject_reg_reallocation = true;
        }

        auto normalize_reg_state = Evaluate(
            Call(DataType::Handle(), set_max_nreg(),
                 {IntImm(DataType::Int(32), kMinimumWarpGroupRegisters),
                  IntImm(DataType::Int(32), 0)}));
        bool normalize_roles =
            inject_reg_reallocation && normalize_cross_handler_reg_state_;
        if (normalize_roles) {
          ICHECK_GE(final_dec_reg, kMinimumWarpGroupRegisters)
              << "cross-handler producer register allocation is below the "
                 "PTX minimum";
        }

        if (consumer_body.defined()) {
          Array<Stmt> producer_stmts{producer_reg_stmt, producer_body};
          // setmaxnreg.dec is not idempotent. A multi-consumer pipeline may
          // already assign the producer the PTX minimum, in which case a
          // second dec-to-24 raises CUDA_ERROR_ILLEGAL_INSTRUCTION.
          if (normalize_roles && final_dec_reg != kMinimumWarpGroupRegisters) {
            producer_stmts.push_back(normalize_reg_state);
          }
          auto new_producer_body = SeqStmt(producer_stmts);
          Array<Stmt> consumer_stmts;
          consumer_stmts.push_back(consumer_reg_stmt);
          consumer_stmts.push_back(consumer_body.value());
          if (normalize_roles) {
            consumer_stmts.push_back(normalize_reg_state);
          }
          auto new_consumer_body = SeqStmt(consumer_stmts);
          // Keep the two role lifetimes independent through CUDA codegen.
          // A single if/else can make ptxas couple producer and consumer
          // locals even though the warp-group predicates are complementary.
          return SeqStmt(
              {IfThenElse(if_then_else->condition, new_producer_body),
               IfThenElse(Not(if_then_else->condition), new_consumer_body)});
        }

        Array<Stmt> producer_stmts{producer_reg_stmt, producer_body};
        if (normalize_roles && final_dec_reg != kMinimumWarpGroupRegisters) {
          producer_stmts.push_back(normalize_reg_state);
        }
        Stmt producer_role =
            IfThenElse(if_then_else->condition, SeqStmt(producer_stmts));
        if (normalize_roles) {
          // Warp groups outside a producer-only role retain the canonical
          // entry allocation and must also join the wrapper restoration.
          Stmt inactive_role =
              IfThenElse(Not(if_then_else->condition), normalize_reg_state);
          return SeqStmt({producer_role, inactive_role});
        }
        return producer_role;
      };

      Stmt new_body =
          RewriteWarpSpecializationBody(op->body, rewrite_if, &rewrote_ws_body);
      if (!rewrote_ws_body) {
        return StmtExprMutator::VisitStmt_(op);
      }
      return AttrStmt(op->node, op->attr_key, op->value, new_body);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Array<IntImm> nreg_;
  bool preserve_explicit_set_max_nreg_{false};
  bool normalize_cross_handler_reg_state_{false};
  IterVar thread_iv_;
  Optional<PrimExpr> updated_thread_extent_;
  bool need_update_thread_extent_ = false;
};

using namespace tir::transform;

tvm::transform::Pass AnnotateWarpGroupRegAlloc() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const PassContext &ctx) -> PrimFunc {
    return SetMaxNRegInjector::Inject(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AnnotateWarpGroupRegAlloc", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnnotateWarpGroupRegAlloc",
                        AnnotateWarpGroupRegAlloc);
}

} // namespace tl
} // namespace tvm
