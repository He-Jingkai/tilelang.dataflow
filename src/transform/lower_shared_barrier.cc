/*!
 *  \file lower_shared_barrier.cc
 *  \brief Convert shared.barrier buffers to plain shared + ptx init.
 */
#include "../op/builtin.h"
#include "tvm/ir/type.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <utility>

namespace tvm {
namespace tl {

namespace attr {
// BlockAttr, Recording the arrive counts for each barrier allocation
constexpr const char *kBarrierInit = "barrier_init";
} // namespace attr

using namespace tir;

class SharedBarrierRewriter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt body, bool disable_shuffle_elect = false) {
    SharedBarrierRewriter rewriter(disable_shuffle_elect);
    return rewriter(std::move(body));
  }

private:
  SharedBarrierRewriter(bool disable_shuffle_elect)
      : disable_shuffle_elect_(disable_shuffle_elect) {}

  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = tvm::ffi::GetRef<Block>(op);
    Array<Buffer> alloc_buffers = op->alloc_buffers;

    // Record the mapping from buffer data var to buffer for later lookup
    for (auto buffer : alloc_buffers) {
      buffer_map_.insert({buffer->data, buffer});
    }
    for (auto match_buffer : op->match_buffers) {
      buffer_map_.insert({match_buffer->buffer->data, match_buffer->buffer});
    }

    // Only check buffers allocated in THIS block, not accumulated from parent
    // blocks
    Array<Buffer> barrier_buffers;
    for (auto buffer : alloc_buffers) {
      const auto *ptr_type =
          buffer->data->type_annotation.as<PointerTypeNode>();
      if (!ptr_type)
        continue;
      auto storage_scope = ptr_type->storage_scope;
      if (storage_scope == "shared.barrier" ||
          storage_scope == "shared.cluster_barrier") {
        barrier_buffers.push_back(buffer);
        if (storage_scope == "shared.cluster_barrier") {
          has_cluster_barrier_ = true;
        }
      }
    }

    if (barrier_buffers.empty()) {
      return StmtExprMutator::VisitStmt_(op);
    }

    ICHECK(thread_var_.defined()) << "thread_var_ is not defined";
    ICHECK(thread_extent_.defined()) << "thread_extent_ is not defined";

    for (auto buffer : barrier_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }

    /*
    Transform:
        mbarrier_list = T.alloc_barrier(arrive_counts: list[int], "handle",
    scope="shared.barrier")

    into:
        # This is emitted by the definition of T.alloc_barrier
        mbarrier_list = T.alloc_buffer(len(arrive_counts), "handle",
    scope="shared.barrier")

        # This is emitted by this pass
        if tx == 0:
          for i in range(len(arrive_counts)):
            T.ptx_init_barrier_thread_count(mbarrier_list[i], arrive_counts[i])
    */

    // Extract the arrive counts from the block attr "barrier_init"
    // The attr is a Map<Var, Array<PrimExpr>> where key is buffer.data and
    // value is arrive counts
    ICHECK(op->annotations.count(attr::kBarrierInit))
        << "barrier_init is not defined";
    auto barrier_init_map = op->annotations.Get(attr::kBarrierInit)
                                ->as<Map<Var, Array<PrimExpr>>>()
                                .value();

    // Large uniform arrays are initialized by one thread per barrier. This
    // avoids serializing long TMA pipelines on a single elected thread while
    // retaining the compact elected-thread path for small or heterogeneous
    // barrier sets.
    constexpr int64_t kParallelBarrierInitThreshold = 32;
    const int64_t thread_extent = thread_extent_.as<IntImmNode>()->value;
    Array<Stmt> serial_init_calls;
    Array<Stmt> parallel_init_stmts;
    for (auto buffer : barrier_buffers) {
      auto data = buffer->data;
      ICHECK(barrier_init_map.count(data))
          << "Barrier buffer " << buffer->name
          << " not found in barrier_init annotation";
      auto arrive_counts = barrier_init_map.at(data);
      ICHECK(arrive_counts.size() ==
             static_cast<size_t>(buffer->shape[0].as<IntImmNode>()->value))
          << "The number of arrive counts (" << arrive_counts.size()
          << ") must match the barrier buffer size (" << buffer->shape[0]
          << ") for buffer " << buffer->name;

      const int64_t barrier_count = buffer->shape[0].as<IntImmNode>()->value;
      bool uniform_arrive_count = !arrive_counts.empty();
      for (size_t i = 1; i < arrive_counts.size(); ++i) {
        uniform_arrive_count =
            uniform_arrive_count &&
            StructuralEqual()(arrive_counts[0], arrive_counts[i]);
      }
      if (uniform_arrive_count &&
          barrier_count >= kParallelBarrierInitThreshold &&
          barrier_count <= thread_extent) {
        auto call =
            Call(DataType::Handle(), builtin::ptx_init_barrier_thread_count(),
                 {BufferLoad(buffer, {thread_var_->var}), arrive_counts[0]});
        parallel_init_stmts.push_back(
            IfThenElse(LT(thread_var_->var,
                          IntImm(thread_var_->var.dtype(), barrier_count)),
                       Evaluate(call), Stmt()));
        continue;
      }

      for (size_t i = 0; i < arrive_counts.size(); i++) {
        auto call =
            Call(DataType::Handle(), builtin::ptx_init_barrier_thread_count(),
                 {BufferLoad(buffer,
                             {IntImm(DataType::Int(32), static_cast<int>(i))}),
                  arrive_counts[i]});
        serial_init_calls.push_back(Evaluate(call));
      }
    }
    if (serial_init_calls.empty() && parallel_init_stmts.empty())
      return block;

    Array<Stmt> new_body;
    for (const auto &parallel_init : parallel_init_stmts) {
      new_body.push_back(parallel_init);
    }
    if (!serial_init_calls.empty()) {
      PrimExpr condition;
      if (!disable_shuffle_elect_) {
        condition =
            Call(DataType::Bool(), tl_shuffle_elect(), {thread_extent_});
      } else {
        condition = EQ(thread_var_->var, 0);
      }
      new_body.push_back(IfThenElse(condition,
                                    serial_init_calls.size() == 1
                                        ? serial_init_calls.back()
                                        : SeqStmt(serial_init_calls),
                                    Stmt()));
    }

    new_body.push_back(
        Evaluate(Call(DataType::Handle(), ptx_fence_barrier_init(), {})));
    new_body.push_back(Evaluate(
        Call(DataType::Handle(), builtin::tvm_storage_sync(),
             {StringImm(has_cluster_barrier_ ? "cluster" : "shared")})));
    new_body.push_back(block->body);

    block.CopyOnWrite()->body = SeqStmt(new_body);

    return StmtExprMutator::VisitStmt_(block.get());
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto buffer = load->buffer;
    if (buffer_remap_.count(buffer)) {
      auto new_buffer = buffer_remap_[load->buffer];
      return BufferLoad(new_buffer, load->indices);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto buffer = store->buffer;
    if (buffer_remap_.count(buffer)) {
      auto new_buffer = buffer_remap_[store->buffer];
      return BufferStore(new_buffer, store->value, store->indices);
    }
    return store;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_var_ = iv;
        thread_extent_ = op->value;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // This is a workaround for cpu backend,
  // we need to define a thread_var for the serial loop.
  IterVar thread_var_;
  PrimExpr thread_extent_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Buffer> buffer_remap_;
  // Mapping from data Var of a Buffer to Buffer, for lookup
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
  // Disable shuffle elect for the warp specialized kernel
  bool disable_shuffle_elect_;
  // Whether the block has a cluster barrier
  bool has_cluster_barrier_ = false;
};

PrimFunc LowerSharedBarrier(PrimFunc f, bool disable_shuffle_elect) {
  f.CopyOnWrite()->body =
      SharedBarrierRewriter::Rewrite(f->body, disable_shuffle_elect);
  return f;
}

namespace transform {
using namespace tir::transform;

tvm::transform::Pass LowerSharedBarrier() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    bool disable_shuffle_elect =
        ctx->GetConfig<Bool>(kDisableShuffleElect, Bool(false)).value();
    return tl::LowerSharedBarrier(std::move(f), disable_shuffle_elect);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerSharedBarrier", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerSharedBarrier", LowerSharedBarrier);
}

} // namespace transform
} // namespace tl
} // namespace tvm
