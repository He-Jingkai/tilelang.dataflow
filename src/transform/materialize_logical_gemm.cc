/*!
 * \file materialize_logical_gemm.cc
 * \brief Materialize target-selected physical GEMM shapes before layout
 * inference.
 */

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/fill.h"
#include "../op/gemm.h"
#include "../op/operator.h"
#include "../op/region.h"
#include "../op/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

constexpr const char *kLogicalGemmPaddingSource =
    "tl.logical_gemm_padding_source";
constexpr const char *kLogicalGemmPaddingBuffers =
    "tl.logical_gemm_padding_buffers";
constexpr const char *kLogicalGemmPaddingLogicalExtent =
    "tl.logical_gemm_padding_logical_extent";

bool IsGemmCall(const Call &call) {
  const auto *op = call->op.as<OpNode>();
  if (op == nullptr) {
    return false;
  }
  const String &name = op->name;
  return name == "tl.tileop.gemm" || name == "tl.tileop.wgmma_gemm" ||
         name == "tl.tileop.tcgen05_gemm";
}

bool IsCopyCall(const Call &call) {
  const auto *op = call->op.as<OpNode>();
  return op != nullptr && op->name == "tl.tileop.copy";
}

int64_t ConstantBufferBytes(const Buffer &buffer) {
  int64_t elements = 1;
  for (const PrimExpr &extent : buffer->shape) {
    const int64_t *value = as_const_int(extent);
    if (value == nullptr || *value <= 0) {
      return -1;
    }
    elements *= *value;
  }
  return elements * buffer->dtype.bits() * buffer->dtype.lanes() / 8;
}

bool IsSharedAllocation(const Buffer &buffer) {
  return buffer.scope() == "shared" || buffer.scope() == "shared.dyn";
}

class GemmAllocationCollector : public StmtExprVisitor {
public:
  void VisitExpr_(const CallNode *op) final {
    if (IsGemmCall(ffi::GetRef<Call>(op))) {
      ++gemm_call_count;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BlockNode *op) final {
    Block block = ffi::GetRef<Block>(op);
    for (const Buffer &buffer : op->alloc_buffers) {
      if (!allocated_buffers.count(buffer->data)) {
        allocated_buffers.Set(buffer->data, buffer);
        allocation_owner[buffer->data] = block;
        if (IsSharedAllocation(buffer)) {
          int64_t bytes = ConstantBufferBytes(buffer);
          if (bytes >= 0) {
            current_shared_memory_bytes += bytes;
          } else {
            shared_memory_is_dynamic = true;
          }
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      if (const int64_t *extent = as_const_int(op->value)) {
        block_size = std::max(block_size, static_cast<int>(*extent));
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  Map<Var, Buffer> allocated_buffers;
  std::unordered_map<Var, Block, ObjectPtrHash, ObjectPtrEqual>
      allocation_owner;
  int block_size{0};
  int gemm_call_count{0};
  int64_t current_shared_memory_bytes{0};
  bool shared_memory_is_dynamic{false};
};

struct PaddingFill {
  Var data;
  Array<Range> ranges;
  PrimExpr value;
  Block owner;
};

struct OperandPadding {
  Var data;
  Array<Range> logical_ranges;
  int matrix_axis{0};
  int physical_extent{0};
  PrimExpr value;
  bool requires_fallback_fill{false};
};

struct LogicalRowPaddingClosure {
  int logical_extent{0};
  int physical_extent{0};
  Map<Var, PrimExpr> buffer_axes;
};

struct LogicalGemmPlanningResult {
  Array<GemmLoweringPlan> ordered_plans;
  std::unordered_map<Call, GemmLoweringPlan, ObjectPtrHash, ObjectPtrEqual>
      plans_by_call;
  std::unordered_map<Call, OperandPadding, ObjectPtrHash, ObjectPtrEqual>
      copy_padding_by_call;
  Map<Var, Array<PrimExpr>> planned_buffer_shapes;
  std::vector<PaddingFill> padding_fills;
  std::unordered_map<Block, LogicalRowPaddingClosure, ObjectPtrHash,
                     ObjectPtrEqual>
      row_padding_closures;
  int64_t initial_shared_memory_bytes{0};
  int64_t additional_shared_memory_bytes{0};
  int block_size{0};
  bool requires_body_rewrite{false};
};

class LogicalGemmPlanner : public StmtExprVisitor {
public:
  LogicalGemmPlanner(
      Target target, int block_size, int64_t max_shared_memory_bytes,
      Map<Var, Buffer> allocated_buffers,
      std::unordered_map<Var, Block, ObjectPtrHash, ObjectPtrEqual>
          allocation_owner,
      int64_t initial_shared_memory_bytes)
      : target_(std::move(target)), block_size_(block_size),
        max_shared_memory_bytes_(max_shared_memory_bytes),
        allocated_buffers_(std::move(allocated_buffers)),
        allocation_owner_(std::move(allocation_owner)),
        current_shared_memory_bytes_(initial_shared_memory_bytes) {
    result_.initial_shared_memory_bytes = initial_shared_memory_bytes;
    result_.block_size = block_size;
  }

  LogicalGemmPlanningResult Plan(const Stmt &body) {
    VisitStmt(body);
    for (const Block &block : pending_row_padding_blocks_) {
      RecordLogicalRowPaddingClosure(block);
    }
    FinalizeOperandPadding();
    result_.planned_buffer_shapes = planned_buffer_shapes_;
    result_.additional_shared_memory_bytes =
        current_shared_memory_bytes_ - result_.initial_shared_memory_bytes;
    return std::move(result_);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    VisitExpr(op->condition);
    std::vector<Call> visible_before = visible_copy_calls_;
    VisitStmt(op->then_case);
    visible_copy_calls_ = visible_before;
    if (op->else_case.defined()) {
      VisitStmt(op->else_case.value());
    }
    visible_copy_calls_ = std::move(visible_before);
  }

  void VisitStmt_(const ForNode *op) final {
    std::vector<Call> visible_before = visible_copy_calls_;
    StmtExprVisitor::VisitStmt_(op);
    visible_copy_calls_ = std::move(visible_before);
  }

  void VisitStmt_(const WhileNode *op) final {
    std::vector<Call> visible_before = visible_copy_calls_;
    StmtExprVisitor::VisitStmt_(op);
    visible_copy_calls_ = std::move(visible_before);
  }

  void VisitStmt_(const BlockNode *op) final {
    StmtExprVisitor::VisitStmt_(op);
    if (op->annotations.Get(kLogicalGemmPaddingSource).has_value()) {
      pending_row_padding_blocks_.push_back(ffi::GetRef<Block>(op));
    }
  }

  void VisitExpr_(const CallNode *op) final {
    StmtExprVisitor::VisitExpr_(op);
    Call call = ffi::GetRef<Call>(op);
    if (IsCopyCall(call)) {
      visible_copy_calls_.push_back(call);
      return;
    }
    if (!IsGemmCall(call)) {
      return;
    }

    TileOperator tile_op = ParseOperator(call);
    const auto *gemm = tile_op.as<GemmNode>();
    ICHECK(gemm != nullptr);
    GemmLoweringContext context;
    context.target = target_;
    context.block_size = block_size_;
    context.allocated_buffers = allocated_buffers_;
    context.planned_buffer_shapes = planned_buffer_shapes_;
    context.current_shared_memory_bytes = current_shared_memory_bytes_;
    context.max_shared_memory_bytes = max_shared_memory_bytes_;
    GemmLoweringPlan plan = ResolveGemmLowering(*gemm, context);

    result_.ordered_plans.push_back(plan);
    result_.plans_by_call.emplace(call, plan);
    if (!plan->supported || !plan->requires_materialization) {
      return;
    }
    result_.requires_body_rewrite = true;

    ICHECK_EQ(plan->physical_shape.size(), 3U);
    int physical_m = static_cast<int>(plan->physical_shape[0]->value);
    if (!plan->requires_padding) {
      return;
    }
    const bool streams_shared_a =
        plan->implementation_id == kGemmImplCudaWGMMASharedARS;
    GemmTemporaryRequirement a_requirement;
    GemmTemporaryRequirement c_requirement;
    for (const GemmTemporaryRequirement &requirement :
         plan->temporary_requirements) {
      if (requirement->buffer_role == "A") {
        a_requirement = requirement;
      } else if (requirement->buffer_role == "C") {
        c_requirement = requirement;
      }
    }
    ICHECK(c_requirement.defined())
        << "logical GEMM padding plan must declare C storage requirements";
    if (!streams_shared_a) {
      ICHECK(a_requirement.defined())
          << "materialized logical GEMM A must declare a storage requirement";
      int a_matrix_axis =
          static_cast<int>(gemm->a_->shape.size()) - (gemm->transA_ ? 1 : 2);
      PrimExpr a_neutral = a_requirement->neutral_value;
      Array<Range> a_logical_ranges = gemm->aRegion_->region;
      a_logical_ranges.Set(
          a_matrix_axis,
          Range::FromMinExtent(
              a_logical_ranges[a_matrix_axis]->min,
              IntImm(a_logical_ranges[a_matrix_axis]->extent.dtype(),
                     gemm->logical_m())));
      ExpandBuffer(gemm->a_, gemm->aRegion_, a_matrix_axis, gemm->logical_m(),
                   physical_m, a_neutral, false);
      if (a_requirement->initialization_required) {
        AddOperandPadding(gemm->a_->data, a_logical_ranges, a_matrix_axis,
                          physical_m, a_neutral);
      }
    }
    ExpandBuffer(gemm->c_, gemm->cRegion_,
                 static_cast<int>(gemm->c_->shape.size()) - 2,
                 gemm->logical_m(), physical_m, c_requirement->neutral_value,
                 c_requirement->initialization_required);
    current_shared_memory_bytes_ += plan->additional_shared_memory_bytes;
  }

private:
  void RecordLogicalRowPaddingClosure(const Block &block) {
    const BlockNode *op = block.operator->();
    auto source_annotation = op->annotations.Get(kLogicalGemmPaddingSource);
    auto buffers_annotation = op->annotations.Get(kLogicalGemmPaddingBuffers);
    auto logical_annotation =
        op->annotations.Get(kLogicalGemmPaddingLogicalExtent);
    if (!source_annotation.has_value() || !buffers_annotation.has_value() ||
        !logical_annotation.has_value()) {
      return;
    }

    Var source = Downcast<Var>(source_annotation.value());
    Map<Var, PrimExpr> buffer_axes =
        Downcast<Map<Var, PrimExpr>>(buffers_annotation.value());
    PrimExpr logical_expr = Downcast<PrimExpr>(logical_annotation.value());
    const int64_t *logical_extent = as_const_int(logical_expr);
    ICHECK(logical_extent != nullptr && *logical_extent > 0)
        << "logical GEMM padding closure requires a positive static extent";
    auto source_shape = planned_buffer_shapes_.Get(source);
    if (!source_shape.has_value() || !buffer_axes.count(source)) {
      return;
    }
    const int64_t *source_axis_value = as_const_int(buffer_axes[source]);
    ICHECK(source_axis_value != nullptr);
    int source_axis = static_cast<int>(*source_axis_value);
    ICHECK_GE(source_axis, 0);
    ICHECK_LT(source_axis, static_cast<int>(source_shape.value().size()));
    const int64_t *physical_extent =
        as_const_int(source_shape.value()[source_axis]);
    ICHECK(physical_extent != nullptr);
    if (*physical_extent <= *logical_extent) {
      return;
    }

    for (const auto &[data, axis_expr] : buffer_axes) {
      ICHECK(allocated_buffers_.count(data))
          << "logical GEMM padding closure buffer must be allocation-backed";
      Buffer allocation = allocated_buffers_[data];
      const int64_t *axis_value = as_const_int(axis_expr);
      ICHECK(axis_value != nullptr);
      int axis = static_cast<int>(*axis_value);
      ICHECK_GE(axis, 0);
      ICHECK_LT(axis, static_cast<int>(allocation->shape.size()));
      Array<PrimExpr> shape = allocation->shape;
      if (auto planned = planned_buffer_shapes_.Get(data)) {
        shape = planned.value();
      }
      const int64_t *old_extent = as_const_int(shape[axis]);
      ICHECK(old_extent != nullptr);
      ICHECK(*old_extent == *logical_extent || *old_extent == *physical_extent)
          << "logical GEMM padding closure axis must match its logical or "
             "physical extent";
      if (*old_extent == *physical_extent) {
        continue;
      }
      int64_t other_elements = 1;
      for (int dim = 0; dim < static_cast<int>(shape.size()); ++dim) {
        if (dim == axis) {
          continue;
        }
        const int64_t *extent = as_const_int(shape[dim]);
        ICHECK(extent != nullptr && *extent > 0);
        other_elements *= *extent;
      }
      if (IsSharedAllocation(allocation)) {
        current_shared_memory_bytes_ +=
            (*physical_extent - *old_extent) * other_elements *
            allocation->dtype.bits() * allocation->dtype.lanes() / 8;
      }
      shape.Set(axis, IntImm(shape[axis].dtype(), *physical_extent));
      planned_buffer_shapes_.Set(data, shape);
    }
    result_.requires_body_rewrite = true;
    result_.row_padding_closures.emplace(
        block, LogicalRowPaddingClosure{static_cast<int>(*logical_extent),
                                        static_cast<int>(*physical_extent),
                                        buffer_axes});
  }

  bool CanExpandCopyProducer(const CopyNode &copy,
                             const OperandPadding &padding) const {
    if (!copy.dst->data.same_as(padding.data) ||
        copy.dst_range.size() != padding.logical_ranges.size() ||
        copy.src_range.size() < copy.dst_range.size() ||
        HasLegacyTransferSemanticAnnotation(copy.annotations)) {
      return false;
    }
    StructuralEqual equal;
    for (size_t axis = 0; axis < copy.dst_range.size(); ++axis) {
      if (axis == static_cast<size_t>(padding.matrix_axis)) {
        if (!equal(copy.dst_range[axis], padding.logical_ranges[axis])) {
          return false;
        }
        continue;
      }
      const Range &produced = copy.dst_range[axis];
      const Range &required = padding.logical_ranges[axis];
      if (!analyzer_.CanProve(produced->min <= required->min &&
                                  produced->min + produced->extent >=
                                      required->min + required->extent,
                              arith::ProofStrength::kSymbolicBound)) {
        return false;
      }
    }
    int leading_axes =
        static_cast<int>(copy.src_range.size() - copy.dst_range.size());
    int source_matrix_axis = leading_axes + padding.matrix_axis;
    if (source_matrix_axis < 0 ||
        source_matrix_axis >= static_cast<int>(copy.src_range.size())) {
      return false;
    }
    for (int axis = 0; axis < leading_axes; ++axis) {
      if (!is_one(copy.src_range[axis]->extent)) {
        return false;
      }
    }
    for (size_t axis = 0; axis < copy.dst_range.size(); ++axis) {
      if (!equal(copy.src_range[leading_axes + axis]->extent,
                 copy.dst_range[axis]->extent)) {
        return false;
      }
    }
    if (copy.transfer_contract.defined() &&
        !equal(copy.transfer_contract.value()->oob_fill, padding.value)) {
      return false;
    }
    return true;
  }

  void AddOperandPadding(const Var &data, const Array<Range> &logical_ranges,
                         int matrix_axis, int physical_extent,
                         const PrimExpr &value) {
    OperandPadding candidate{
        data, logical_ranges, matrix_axis, physical_extent, value, false};
    bool expanded_by_copy = false;
    for (const Call &call : visible_copy_calls_) {
      TileOperator tile_op = ParseOperator(call);
      const auto *copy = tile_op.as<CopyNode>();
      if (copy == nullptr || !CanExpandCopyProducer(*copy, candidate)) {
        continue;
      }
      RecordCopyPadding(call, candidate);
      expanded_by_copy = true;
    }
    candidate.requires_fallback_fill = !expanded_by_copy;

    StructuralEqual equal;
    for (OperandPadding &existing : operand_paddings_) {
      if (existing.data.same_as(data) && existing.matrix_axis == matrix_axis &&
          existing.physical_extent == physical_extent &&
          equal(existing.logical_ranges, logical_ranges) &&
          equal(existing.value, value)) {
        existing.requires_fallback_fill |= candidate.requires_fallback_fill;
        return;
      }
    }
    operand_paddings_.push_back(std::move(candidate));
  }

  void FinalizeOperandPadding() {
    for (const OperandPadding &padding : operand_paddings_) {
      if (padding.requires_fallback_fill) {
        Array<Range> fill_ranges = padding.logical_ranges;
        const Range &logical_axis = fill_ranges[padding.matrix_axis];
        const int64_t *logical_extent = as_const_int(logical_axis->extent);
        ICHECK(logical_extent != nullptr);
        fill_ranges.Set(padding.matrix_axis,
                        Range::FromMinExtent(
                            logical_axis->min + logical_axis->extent,
                            IntImm(logical_axis->extent.dtype(),
                                   padding.physical_extent - *logical_extent)));
        AddPaddingFill(padding.data, fill_ranges, padding.value);
      }
    }
  }

  void RecordCopyPadding(const Call &call, const OperandPadding &padding) {
    auto [it, inserted] = result_.copy_padding_by_call.emplace(call, padding);
    if (inserted) {
      return;
    }
    StructuralEqual equal;
    ICHECK_EQ(it->second.matrix_axis, padding.matrix_axis);
    ICHECK(equal(it->second.value, padding.value))
        << "one copy cannot materialize incompatible GEMM padding values";
    if (padding.physical_extent > it->second.physical_extent) {
      it->second = padding;
    }
  }

  void ExpandBuffer(const Buffer &buffer, const BufferRegion &region, int axis,
                    int logical_extent, int physical_extent,
                    PrimExpr neutral_value, bool emit_padding_fill) {
    ICHECK(allocated_buffers_.count(buffer->data))
        << "logical GEMM materialization requires an allocation-backed buffer";
    Buffer allocation = allocated_buffers_[buffer->data];
    ICHECK(allocation->strides.empty())
        << "logical GEMM materialization currently requires compact internal "
           "buffers; explicit strides must select an unpadded fallback";
    ICHECK_GE(axis, 0);
    ICHECK_LT(axis, static_cast<int>(allocation->shape.size()));
    ICHECK_EQ(region->region.size(), allocation->shape.size());
    const int64_t *region_min = as_const_int(region->region[axis]->min);
    ICHECK(region_min != nullptr)
        << "logical GEMM padding requires a compile-time matrix-axis offset";

    Array<PrimExpr> shape = allocation->shape;
    if (auto planned = planned_buffer_shapes_.Get(buffer->data)) {
      shape = planned.value();
    }
    const int64_t *old_extent = as_const_int(shape[axis]);
    ICHECK(old_extent != nullptr);
    int64_t required_extent = *region_min + physical_extent;
    if (*old_extent < required_extent) {
      shape.Set(axis, IntImm(shape[axis].dtype(), required_extent));
      planned_buffer_shapes_.Set(buffer->data, shape);
    }

    Array<Range> fill_ranges = region->region;
    fill_ranges.Set(
        axis, Range::FromMinExtent(region->region[axis]->min + logical_extent,
                                   IntImm(region->region[axis]->extent.dtype(),
                                          physical_extent - logical_extent)));
    if (emit_padding_fill) {
      AddPaddingFill(buffer->data, fill_ranges, neutral_value);
    }
  }

  void AddPaddingFill(const Var &data, const Array<Range> &ranges,
                      const PrimExpr &value) {
    auto owner_it = allocation_owner_.find(data);
    ICHECK(owner_it != allocation_owner_.end());
    StructuralEqual equal;
    Buffer buffer = allocated_buffers_[data];
    BufferRegion candidate(buffer, ranges);
    for (const PaddingFill &existing : result_.padding_fills) {
      if (!existing.data.same_as(data)) {
        continue;
      }
      BufferRegion previous(buffer, existing.ranges);
      if (equal(candidate, previous) && equal(value, existing.value)) {
        return;
      }
    }
    result_.padding_fills.push_back(
        PaddingFill{data, ranges, value, owner_it->second});
  }

  Target target_;
  int block_size_{0};
  int64_t max_shared_memory_bytes_{-1};
  Map<Var, Buffer> allocated_buffers_;
  std::unordered_map<Var, Block, ObjectPtrHash, ObjectPtrEqual>
      allocation_owner_;
  Map<Var, Array<PrimExpr>> planned_buffer_shapes_;
  std::vector<Call> visible_copy_calls_;
  std::vector<OperandPadding> operand_paddings_;
  std::vector<Block> pending_row_padding_blocks_;
  mutable arith::Analyzer analyzer_;
  int64_t current_shared_memory_bytes_{0};
  LogicalGemmPlanningResult result_;
};

PrimExpr MakeRegionCall(const BufferRegion &region, int access_mask) {
  Array<PrimExpr> args;
  Array<PrimExpr> mins;
  mins.reserve(region->region.size());
  for (const Range &range : region->region) {
    mins.push_back(range->min);
  }
  args.push_back(BufferLoad(region->buffer, mins));
  args.push_back(IntImm(DataType::Int(32), access_mask));
  for (const Range &range : region->region) {
    args.push_back(range->extent);
  }
  return Call(DataType::Handle(), RegionOp::Get(), args);
}

class LogicalRowPaddingRewriter : public StmtExprMutator {
public:
  explicit LogicalRowPaddingRewriter(LogicalRowPaddingClosure closure)
      : closure_(std::move(closure)) {}

  Stmt Rewrite(const Stmt &body) { return VisitStmt(body); }

  Array<BufferRegion> RewriteRegions(const Array<BufferRegion> &regions) {
    Array<BufferRegion> rewritten;
    rewritten.reserve(regions.size());
    for (const BufferRegion &region : regions) {
      auto axis_it = closure_.buffer_axes.find(region->buffer->data);
      if (axis_it == closure_.buffer_axes.end()) {
        rewritten.push_back(region);
        continue;
      }
      const int64_t *axis_value = as_const_int((*axis_it).second);
      ICHECK(axis_value != nullptr);
      int axis = static_cast<int>(*axis_value);
      ICHECK_LT(axis, static_cast<int>(region->region.size()));
      Array<Range> ranges = region->region;
      const int64_t *extent = as_const_int(ranges[axis]->extent);
      if (extent != nullptr && *extent == closure_.logical_extent) {
        ranges.Set(axis,
                   Range::FromMinExtent(ranges[axis]->min,
                                        IntImm(ranges[axis]->extent.dtype(),
                                               closure_.physical_extent)));
      }
      rewritten.push_back(BufferRegion(region->buffer, ranges));
    }
    return rewritten;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    const int64_t *extent = as_const_int(loop->extent);
    if (extent != nullptr && *extent == closure_.logical_extent) {
      loop.CopyOnWrite()->extent =
          IntImm(loop->extent.dtype(), closure_.physical_extent);
    }
    return loop;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto axis_it = closure_.buffer_axes.find(load->buffer->data);
    if (axis_it == closure_.buffer_axes.end()) {
      return load;
    }
    const int64_t *axis_value = as_const_int((*axis_it).second);
    ICHECK(axis_value != nullptr);
    int axis = static_cast<int>(*axis_value);
    ICHECK_LT(axis, static_cast<int>(load->indices.size()));
    const auto *ramp = load->indices[axis].as<RampNode>();
    if (ramp == nullptr) {
      return load;
    }
    const int64_t *lanes = as_const_int(ramp->lanes);
    if (lanes == nullptr || *lanes != closure_.logical_extent) {
      return load;
    }
    Array<PrimExpr> indices = load->indices;
    indices.Set(axis,
                Ramp(ramp->base, ramp->stride,
                     IntImm(ramp->lanes.dtype(), closure_.physical_extent),
                     ramp->span));
    return BufferLoad(load->buffer, indices, load->predicate, load->span);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (!call->op.same_as(RegionOp::Get()) || call->args.empty()) {
      return call;
    }
    const auto *load = call->args[0].as<BufferLoadNode>();
    if (load == nullptr || !closure_.buffer_axes.count(load->buffer->data)) {
      return call;
    }
    const int64_t *axis_value =
        as_const_int(closure_.buffer_axes[load->buffer->data]);
    ICHECK(axis_value != nullptr);
    int extent_index = 2 + static_cast<int>(*axis_value);
    ICHECK_LT(extent_index, static_cast<int>(call->args.size()));
    const int64_t *extent = as_const_int(call->args[extent_index]);
    if (extent == nullptr || *extent != closure_.logical_extent) {
      return call;
    }
    Array<PrimExpr> args = call->args;
    args.Set(extent_index, IntImm(call->args[extent_index].dtype(),
                                  closure_.physical_extent));
    return Call(call.dtype(), call->op, args, call->annotations, call->span);
  }

private:
  LogicalRowPaddingClosure closure_;
};

class LogicalGemmMaterializer : public StmtExprMutator {
public:
  explicit LogicalGemmMaterializer(LogicalGemmPlanningResult planning)
      : planning_(std::move(planning)) {}

  Stmt Rewrite(const Stmt &body) { return VisitStmt(body); }

  int GroupedWgmmaChainCount() const { return grouped_wgmma_chain_count_; }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> visited;
    visited.reserve(op->seq.size());
    for (const Stmt &stmt : op->seq) {
      visited.push_back(VisitStmt(stmt));
    }

    Array<Stmt> rewritten;
    rewritten.reserve(visited.size());
    for (int i = 0, n = static_cast<int>(visited.size()); i < n;) {
      int end = i + 1;
      while (end < n && CanGroupWgmma(visited[end - 1], visited[end])) {
        ++end;
      }
      if (end - i < 2) {
        rewritten.push_back(visited[i]);
        ++i;
        continue;
      }
      const int group_size = end - i;
      std::vector<std::pair<Buffer, int64_t>> register_operands =
          CollectGroupedWgmmaRegisterOperands(visited, i, end);
      for (const auto &[buffer, element_count] : register_operands) {
        rewritten.push_back(MakeWgmmaRegisterFence(buffer, element_count));
      }
      for (int index = 0; index < group_size; ++index) {
        rewritten.push_back(
            ConfigureGroupedWgmma(visited[i + index], index, group_size));
      }
      rewritten.push_back(Evaluate(Call(DataType::Handle(), wait_wgmma(),
                                        {IntImm(DataType::Int(32), 0)})));
      for (const auto &[buffer, element_count] : register_operands) {
        rewritten.push_back(MakeWgmmaRegisterFence(buffer, element_count));
      }
      ++grouped_wgmma_chain_count_;
      i = end;
    }
    if (rewritten.empty()) {
      return Evaluate(0);
    }
    if (rewritten.size() == 1) {
      return rewritten[0];
    }
    return SeqStmt(rewritten);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    Buffer remapped = RemapBuffer(load->buffer);
    if (!remapped.same_as(load->buffer)) {
      load.CopyOnWrite()->buffer = remapped;
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    Buffer remapped = RemapBuffer(store->buffer);
    if (!remapped.same_as(store->buffer)) {
      store.CopyOnWrite()->buffer = remapped;
    }
    return store;
  }

  Stmt VisitStmt_(const DeclBufferNode *op) final {
    DeclBuffer decl = Downcast<DeclBuffer>(StmtExprMutator::VisitStmt_(op));
    Buffer remapped = RemapBuffer(decl->buffer);
    if (!remapped.same_as(decl->buffer)) {
      decl.CopyOnWrite()->buffer = remapped;
    }
    return decl;
  }

  Stmt VisitStmt_(const AllocateNode *op) final {
    Allocate allocate = Downcast<Allocate>(StmtExprMutator::VisitStmt_(op));
    if (auto planned = planning_.planned_buffer_shapes.Get(op->buffer_var)) {
      allocate.CopyOnWrite()->extents = planned.value();
    }
    return allocate;
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    Block original_block = ffi::GetRef<Block>(op);
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    BlockNode *writer = block.CopyOnWrite();

    Array<Buffer> alloc_buffers;
    alloc_buffers.reserve(writer->alloc_buffers.size());
    for (const Buffer &buffer : writer->alloc_buffers) {
      alloc_buffers.push_back(RemapBuffer(buffer));
    }
    writer->alloc_buffers = std::move(alloc_buffers);

    writer->reads = RemapRegions(writer->reads);
    writer->writes = RemapRegions(writer->writes);
    Array<MatchBufferRegion> match_buffers;
    match_buffers.reserve(writer->match_buffers.size());
    for (const MatchBufferRegion &match : writer->match_buffers) {
      match_buffers.push_back(MatchBufferRegion(RemapBuffer(match->buffer),
                                                RemapRegion(match->source)));
    }
    writer->match_buffers = std::move(match_buffers);

    auto closure = planning_.row_padding_closures.find(original_block);
    if (closure != planning_.row_padding_closures.end()) {
      LogicalRowPaddingRewriter rewriter(closure->second);
      writer->reads = rewriter.RewriteRegions(writer->reads);
      writer->writes = rewriter.RewriteRegions(writer->writes);
      writer->body = rewriter.Rewrite(writer->body);
    }

    Array<Stmt> prefix;
    for (const PaddingFill &fill : planning_.padding_fills) {
      if (!fill.owner.same_as(original_block)) {
        continue;
      }
      Buffer original = FindAllocation(fill.data);
      BufferRegion region(RemapBuffer(original), fill.ranges);
      Map<String, ObjectRef> annotations;
      annotations.Set(attr::kPredicatedFillPartition,
                      IntImm(DataType::Int(32), 1));
      PrimExpr fill_call =
          Call(DataType::Handle(), Fill::Get(),
               {MakeRegionCall(region, kAccessWrite), fill.value}, annotations);
      prefix.push_back(Evaluate(fill_call));
    }
    if (!prefix.empty()) {
      prefix.push_back(writer->body);
      writer->body = SeqStmt(prefix);
    }
    return block;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call original_call = ffi::GetRef<Call>(op);
    auto plan_it = planning_.plans_by_call.find(original_call);
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    auto copy_padding_it = planning_.copy_padding_by_call.find(original_call);
    if (copy_padding_it != planning_.copy_padding_by_call.end()) {
      return ExpandCopyProducer(call, copy_padding_it->second);
    }
    if (plan_it == planning_.plans_by_call.end()) {
      return call;
    }
    const GemmLoweringPlan &plan = plan_it->second;
    ICHECK(plan->supported)
        << "logical GEMM has no legal lowering: " << plan->selection_reason;
    Map<String, ObjectRef> annotations = call->annotations;
    annotations.Set("gemm_lowering_implementation",
                    StringImm(plan->implementation_id));
    if (!plan->requires_materialization) {
      return Call(call.dtype(), call->op, call->args, annotations, call->span);
    }

    TileOperator tile_op = ParseOperator(call);
    const auto *gemm = tile_op.as<GemmNode>();
    ICHECK(gemm != nullptr);
    int physical_m = static_cast<int>(plan->physical_shape[0]->value);
    const bool streams_shared_a =
        plan->implementation_id == kGemmImplCudaWGMMASharedARS;
    Array<PrimExpr> args = call->args;
    const int a_matrix_axis =
        static_cast<int>(gemm->a_->shape.size()) - (gemm->transA_ ? 1 : 2);
    args.Set(
        0, ExpandMatrixRegion(args[0], a_matrix_axis,
                              streams_shared_a ? gemm->logical_m() : physical_m,
                              kAccessRead));
    args.Set(2, ExpandMatrixRegion(args[2],
                                   static_cast<int>(gemm->c_->shape.size()) - 2,
                                   physical_m, kAccessReadWrite));
    args.Set(5, IntImm(args[5].dtype(), physical_m));
    if (streams_shared_a) {
      annotations.Set("wgmma_rs_shared_a_logical_m",
                      IntImm(DataType::Int(32), gemm->logical_m()));
    }
    return Call(call.dtype(), call->op, args, annotations, call->span);
  }

private:
  Optional<Call> GroupableWgmma(const Stmt &stmt) const {
    const auto *evaluate = stmt.as<EvaluateNode>();
    if (evaluate == nullptr) {
      return std::nullopt;
    }
    const auto *call = evaluate->value.as<CallNode>();
    if (call == nullptr || !call->op.same_as(Gemm::Get()) ||
        call->args.size() <= 15) {
      return std::nullopt;
    }
    Optional<ObjectRef> implementation =
        call->annotations.Get("gemm_lowering_implementation");
    const auto *implementation_id =
        implementation.defined() ? implementation.value().as<StringImmNode>()
                                 : nullptr;
    if (implementation_id == nullptr ||
        implementation_id->value != kGemmImplCudaWGMMA) {
      return std::nullopt;
    }
    for (const char *key :
         {"wgmma_emit_arrive", "wgmma_emit_commit", "wgmma_emit_fence_before",
          "wgmma_emit_fence_after"}) {
      if (call->annotations.Get(key).has_value()) {
        return std::nullopt;
      }
    }
    const int64_t *wait = as_const_int(call->args[15]);
    if (wait == nullptr || *wait != 0) {
      return std::nullopt;
    }
    return ffi::GetRef<Call>(call);
  }

  bool CanGroupWgmma(const Stmt &left_stmt, const Stmt &right_stmt) const {
    Optional<Call> left = GroupableWgmma(left_stmt);
    Optional<Call> right = GroupableWgmma(right_stmt);
    if (!left.defined() || !right.defined() ||
        !is_zero(right.value()->args[9])) {
      return false;
    }
    StructuralEqual equal;
    return equal(left.value()->args[2], right.value()->args[2]) &&
           equal(left.value()->args[5], right.value()->args[5]) &&
           equal(left.value()->args[6], right.value()->args[6]) &&
           equal(left.value()->args[8], right.value()->args[8]);
  }

  Stmt ConfigureGroupedWgmma(const Stmt &stmt, int index,
                             int group_size) const {
    const auto *evaluate = stmt.as<EvaluateNode>();
    ICHECK(evaluate != nullptr);
    Call call = Downcast<Call>(evaluate->value);
    const bool first = index == 0;
    const bool last = index + 1 == group_size;
    Array<PrimExpr> args = call->args;
    args.Set(15, IntImm(call->args[15].dtype(), -1));
    Map<String, ObjectRef> annotations = call->annotations;
    annotations.Set("wgmma_emit_arrive",
                    IntImm(DataType::Int(32), first ? 1 : 0));
    annotations.Set("wgmma_emit_commit",
                    IntImm(DataType::Int(32), last ? 1 : 0));
    annotations.Set("wgmma_emit_fence_before", IntImm(DataType::Int(32), 0));
    annotations.Set("wgmma_emit_fence_after", IntImm(DataType::Int(32), 0));
    annotations.Set("wgmma_additive_group_index",
                    IntImm(DataType::Int(32), index));
    annotations.Set("wgmma_additive_group_size",
                    IntImm(DataType::Int(32), group_size));
    return Evaluate(Call(call.dtype(), Op::Get("tl.tileop.wgmma_gemm"), args,
                         annotations, call->span));
  }

  std::vector<std::pair<Buffer, int64_t>>
  CollectGroupedWgmmaRegisterOperands(const Array<Stmt> &stmts, int begin,
                                      int end) const {
    std::vector<std::pair<Buffer, int64_t>> operands;
    std::unordered_map<Var, size_t, ObjectPtrHash, ObjectPtrEqual>
        operand_indices;
    auto record = [&](const Buffer &buffer, int64_t element_count) {
      auto [it, inserted] =
          operand_indices.emplace(buffer->data, operands.size());
      if (inserted) {
        operands.emplace_back(buffer, element_count);
      } else {
        operands[it->second].second =
            std::max(operands[it->second].second, element_count);
      }
    };

    for (int index = begin; index < end; ++index) {
      const auto *evaluate = stmts[index].as<EvaluateNode>();
      ICHECK(evaluate != nullptr);
      auto tile_op = ParseOperator(Downcast<Call>(evaluate->value));
      const auto *gemm = tile_op.as<GemmNode>();
      ICHECK(gemm != nullptr);
      if (IsFragmentBuffer(gemm->a_)) {
        record(gemm->a_, static_cast<int64_t>(gemm->m_) * gemm->k_);
      }
      record(gemm->c_, static_cast<int64_t>(gemm->m_) * gemm->n_);
    }
    return operands;
  }

  Stmt MakeWgmmaRegisterFence(const Buffer &buffer,
                              int64_t element_count) const {
    ICHECK_GT(planning_.block_size, 0);
    ICHECK_GT(element_count, 0);
    int64_t total_bits =
        element_count * buffer->dtype.bits() * buffer->dtype.lanes();
    int64_t bits_per_register_partition =
        static_cast<int64_t>(planning_.block_size) * 32;
    int64_t num_regs = (total_bits + bits_per_register_partition - 1) /
                       bits_per_register_partition;
    return Evaluate(Call(DataType::Handle(), warpgroup_fence_operand(),
                         {StringImm(runtime::DLDataTypeToString(buffer->dtype)),
                          buffer->data, buffer->elem_offset,
                          IntImm(DataType::Int(32), num_regs)}));
  }

  PrimExpr ExpandCopyProducer(const Call &call, const OperandPadding &padding) {
    TileOperator tile_op = ParseOperator(call);
    const auto *copy = tile_op.as<CopyNode>();
    ICHECK(copy != nullptr);
    int leading_axes =
        static_cast<int>(copy->src_range.size() - copy->dst_range.size());
    int source_matrix_axis = leading_axes + padding.matrix_axis;

    Array<Range> source_ranges = copy->src_range;
    Array<Range> destination_ranges = copy->dst_range;
    source_ranges.Set(
        source_matrix_axis,
        Range::FromMinExtent(
            source_ranges[source_matrix_axis]->min,
            IntImm(source_ranges[source_matrix_axis]->extent.dtype(),
                   padding.physical_extent)));
    destination_ranges.Set(
        padding.matrix_axis,
        Range::FromMinExtent(
            destination_ranges[padding.matrix_axis]->min,
            IntImm(destination_ranges[padding.matrix_axis]->extent.dtype(),
                   padding.physical_extent)));

    Array<PrimExpr> args = call->args;
    args.Set(
        0, MakeRegionCall(BufferRegion(copy->src, source_ranges), kAccessRead));
    args.Set(1, MakeRegionCall(BufferRegion(copy->dst, destination_ranges),
                               kAccessWrite));
    if (!copy->transfer_contract.defined()) {
      PrimExpr valid_region =
          MakeRegionCall(BufferRegion(copy->src, copy->src_range), kAccessRead);
      PrimExpr contract =
          Call(DataType::Handle(), TransferContract::Get(),
               {valid_region, padding.value, IntImm(DataType::Int(32), 1),
                IntImm(DataType::Int(32),
                       static_cast<int>(TransferSyncOwner::kTransfer))});
      args.push_back(contract);
    }
    return Call(call.dtype(), call->op, args, call->annotations, call->span);
  }

  Buffer RemapBuffer(const Buffer &buffer) {
    auto cache_it = buffer_cache_.find(buffer);
    if (cache_it != buffer_cache_.end()) {
      return cache_it->second;
    }
    auto planned = planning_.planned_buffer_shapes.Get(buffer->data);
    if (!planned) {
      buffer_cache_.emplace(buffer, buffer);
      return buffer;
    }
    Buffer remapped(buffer->data, buffer->dtype, planned.value(),
                    buffer->strides, buffer->elem_offset, buffer->name,
                    buffer->data_alignment, buffer->offset_factor,
                    buffer->buffer_type, buffer->axis_separators, buffer->span);
    buffer_cache_.emplace(buffer, remapped);
    return remapped;
  }

  Buffer FindAllocation(const Var &data) const {
    for (const auto &[original, buffer] : buffer_cache_) {
      (void)original;
      if (buffer->data.same_as(data)) {
        return buffer;
      }
    }
    LOG(FATAL) << "cannot find materialized GEMM allocation for " << data;
    return Buffer();
  }

  BufferRegion RemapRegion(const BufferRegion &region) {
    return BufferRegion(RemapBuffer(region->buffer), region->region);
  }

  Array<BufferRegion> RemapRegions(const Array<BufferRegion> &regions) {
    Array<BufferRegion> result;
    result.reserve(regions.size());
    for (const BufferRegion &region : regions) {
      result.push_back(RemapRegion(region));
    }
    return result;
  }

  PrimExpr ExpandMatrixRegion(const PrimExpr &arg, int axis,
                              int physical_extent, int access_mask) {
    BufferRegion region = NormalizeToBufferRegion(arg);
    Array<Range> ranges = region->region;
    ICHECK_GE(axis, 0);
    ICHECK_LT(axis, static_cast<int>(ranges.size()));
    ranges.Set(axis, Range::FromMinExtent(ranges[axis]->min,
                                          IntImm(ranges[axis]->extent.dtype(),
                                                 physical_extent)));
    return MakeRegionCall(BufferRegion(RemapBuffer(region->buffer), ranges),
                          access_mask);
  }

  LogicalGemmPlanningResult planning_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>
      buffer_cache_;
  int grouped_wgmma_chain_count_{0};
};

LogicalGemmPlanningResult PlanLogicalGemms(const PrimFunc &func, Target target,
                                           int block_size,
                                           int64_t max_shared_memory_bytes) {
  GemmAllocationCollector collector;
  collector(func->body);
  if (collector.gemm_call_count == 0) {
    return {};
  }
  if (block_size <= 0) {
    block_size = collector.block_size;
  }
  ICHECK_GT(block_size, 0)
      << "logical GEMM planning requires a static thread extent";
  LogicalGemmPlanner planner(
      std::move(target), block_size,
      collector.shared_memory_is_dynamic ? -1 : max_shared_memory_bytes,
      collector.allocated_buffers, std::move(collector.allocation_owner),
      collector.current_shared_memory_bytes);
  return planner.Plan(func->body);
}

} // namespace

tvm::transform::Pass MaterializeLogicalGemm() {
  using namespace tir::transform;
  auto pass_func = [](PrimFunc func, const IRModule &module,
                      const tvm::transform::PassContext &ctx) {
    (void)module;
    Optional<Target> target = func->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "MaterializeLogicalGemm requires BindTarget to run first";
    int64_t max_shared_memory_bytes =
        ctx->GetConfig<Integer>(kLogicalGemmMaxSharedMemoryBytes)
            .value_or(Integer(-1))
            ->value;
    LogicalGemmPlanningResult planning =
        PlanLogicalGemms(func, target.value(), 0, max_shared_memory_bytes);
    if (planning.ordered_plans.empty()) {
      return func;
    }
    for (const GemmLoweringPlan &plan : planning.ordered_plans) {
      ICHECK(plan->supported)
          << "logical GEMM has no legal lowering: " << plan->selection_reason;
    }
    Array<GemmLoweringPlan> plans = planning.ordered_plans;
    int64_t additional_shared = planning.additional_shared_memory_bytes;
    PrimFunc rewritten = func;
    LogicalGemmMaterializer materializer(std::move(planning));
    rewritten.CopyOnWrite()->body = materializer.Rewrite(func->body);
    int grouped_wgmma_chains = materializer.GroupedWgmmaChainCount();
    rewritten = WithAttr(std::move(rewritten), "tl.gemm_lowering_plans", plans);
    rewritten =
        WithAttr(std::move(rewritten), "tl.gemm_lowering_registry_version",
                 Integer(GemmLoweringRegistryVersion()));
    rewritten = WithAttr(std::move(rewritten),
                         "tl.logical_gemm_additional_shared_memory_bytes",
                         Integer(additional_shared));
    rewritten = WithAttr(std::move(rewritten),
                         "tl.logical_gemm_grouped_wgmma_chain_count",
                         Integer(grouped_wgmma_chains));
    return rewritten;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MaterializeLogicalGemm", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.transform.MaterializeLogicalGemm", MaterializeLogicalGemm)
      .def("tl.ResolvePrimFuncGemmLoweringPlans",
           [](PrimFunc func, Target target, int block_size,
              int max_shared_memory_bytes) {
             return PlanLogicalGemms(func, std::move(target), block_size,
                                     max_shared_memory_bytes)
                 .ordered_plans;
           });
}

} // namespace tl
} // namespace tvm
