#pragma once

#include "common.h"
#include "dataflow_abi_generated.h"

namespace tl {

TL_DEVICE uint32_t dataflow_instruction_cluster_recv_count(
    const DataflowInstruction &instruction) {
  return instruction.flags & kDataflowInstructionClusterCommCountMask;
}

TL_DEVICE uint32_t dataflow_instruction_cluster_send_count(
    const DataflowInstruction &instruction) {
  return instruction.flags >> kDataflowInstructionClusterSendCountShift;
}

struct DataflowQueue {
  const DataflowInstruction *instructions;
  const uint32_t *offsets;
  const uint32_t *lengths;
};

TL_DEVICE uint32_t dataflow_thread_rank() {
  return static_cast<uint32_t>(
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));
}

TL_DEVICE uint32_t dataflow_num_threads() {
  return static_cast<uint32_t>(blockDim.x * blockDim.y * blockDim.z);
}

TL_DEVICE bool dataflow_is_leader_thread() {
  return dataflow_thread_rank() == 0;
}

TL_DEVICE uint32_t dataflow_cta_rank_in_grid() {
  return static_cast<uint32_t>(
      blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));
}

TL_DEVICE uint32_t dataflow_smid() {
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

TL_DEVICE uint8_t *dataflow_byte_ptr(void *base, uint32_t offset) {
  return reinterpret_cast<uint8_t *>(base) + offset;
}

TL_DEVICE const uint8_t *dataflow_byte_ptr(const void *base, uint32_t offset) {
  return reinterpret_cast<const uint8_t *>(base) + offset;
}

TL_DEVICE void *dataflow_slot_shared_ptr(void *shared_base,
                                         const DataflowSlot &slot) {
  return dataflow_byte_ptr(shared_base, slot.shared_offset);
}

TL_DEVICE const void *dataflow_slot_shared_ptr(const void *shared_base,
                                               const DataflowSlot &slot) {
  return dataflow_byte_ptr(shared_base, slot.shared_offset);
}

TL_DEVICE bool dataflow_slot_is_scratch_backed(const DataflowSlot &slot) {
  return (slot.flags & kDataflowSlotFlagScratchBacked) != 0u;
}

TL_DEVICE bool dataflow_slot_uses_hbm_direct_global(const DataflowSlot &slot) {
  return (slot.flags & kDataflowSlotFlagHBMDirectGlobal) != 0u;
}

TL_DEVICE bool dataflow_slot_uses_cluster_gated_push(const DataflowSlot &slot) {
  return (slot.flags & kDataflowSlotFlagClusterGatedPush) != 0u;
}

TL_DEVICE bool dataflow_slot_is_communicate(const DataflowSlot &slot) {
  return (slot.flags & kDataflowSlotFlagCommunicate) != 0u;
}

TL_DEVICE void *dataflow_slot_shared_ptr(void *shared_base, void *scratch_base,
                                         const DataflowSlot &slot) {
  return dataflow_byte_ptr(dataflow_slot_is_scratch_backed(slot) ? scratch_base
                                                                 : shared_base,
                           slot.shared_offset);
}

TL_DEVICE const void *dataflow_slot_shared_ptr(const void *shared_base,
                                               const void *scratch_base,
                                               const DataflowSlot &slot) {
  return dataflow_byte_ptr(dataflow_slot_is_scratch_backed(slot) ? scratch_base
                                                                 : shared_base,
                           slot.shared_offset);
}

TL_DEVICE void *dataflow_slot_global_ptr(void *global_base,
                                         const DataflowSlot &slot) {
  return dataflow_byte_ptr(global_base, slot.global_offset);
}

TL_DEVICE const void *dataflow_slot_global_ptr(const void *global_base,
                                               const DataflowSlot &slot) {
  return dataflow_byte_ptr(global_base, slot.global_offset);
}

TL_DEVICE void *dataflow_arg_ptr(void *arg_base, uint32_t offset) {
  return dataflow_byte_ptr(arg_base, offset);
}

TL_DEVICE const void *dataflow_arg_ptr(const void *arg_base, uint32_t offset) {
  return dataflow_byte_ptr(arg_base, offset);
}

TL_DEVICE DataflowHandlerArgs &dataflow_handler_args(void *arg_base,
                                                     uint32_t offset) {
  return *reinterpret_cast<DataflowHandlerArgs *>(
      dataflow_arg_ptr(arg_base, offset));
}

TL_DEVICE const DataflowHandlerArgs &dataflow_handler_args(const void *arg_base,
                                                           uint32_t offset) {
  return *reinterpret_cast<const DataflowHandlerArgs *>(
      dataflow_arg_ptr(arg_base, offset));
}

TL_DEVICE const DataflowTensorArg &
dataflow_tensor_arg(const DataflowTensorArg *tensor_args, uint32_t index) {
  return tensor_args[index];
}

TL_DEVICE uint32_t dataflow_queue_offset(const DataflowQueue &queue,
                                         uint32_t cta_rank) {
  return queue.offsets[cta_rank];
}

TL_DEVICE uint32_t dataflow_queue_length(const DataflowQueue &queue,
                                         uint32_t cta_rank) {
  return queue.lengths[cta_rank];
}

TL_DEVICE DataflowInstruction dataflow_queue_load(const DataflowQueue &queue,
                                                  uint32_t cta_rank,
                                                  uint32_t local_index) {
  return queue
      .instructions[dataflow_queue_offset(queue, cta_rank) + local_index];
}

TL_DEVICE bool dataflow_opcode_is_exit(const DataflowInstruction &inst) {
  return inst.opcode == static_cast<uint32_t>(DataflowOpcode::kExit);
}

TL_DEVICE bool
dataflow_opcode_is_cluster_sync(const DataflowInstruction &inst) {
  return inst.opcode == static_cast<uint32_t>(DataflowOpcode::kClusterSync);
}

TL_DEVICE uint32_t dataflow_select_index(uint32_t primary, uint32_t fallback) {
  return primary == kDataflowInvalidIndex ? fallback : primary;
}

TL_DEVICE uint32_t dataflow_min_u32(uint32_t lhs, uint32_t rhs) {
  return lhs < rhs ? lhs : rhs;
}

TL_DEVICE uint32_t dataflow_comm_bytes(const DataflowCommPlan &comm,
                                       const DataflowSlot &src_slot,
                                       const DataflowSlot &dst_slot) {
  if (comm.byte_count != 0) {
    return comm.byte_count;
  }
  return dataflow_min_u32(src_slot.bytes, dst_slot.bytes);
}

TL_DEVICE void *dataflow_comm_byte_ptr(void *ptr,
                                       const DataflowCommPlan &comm) {
  return reinterpret_cast<uint8_t *>(ptr) + comm.byte_offset;
}

TL_DEVICE const void *dataflow_comm_byte_ptr(const void *ptr,
                                             const DataflowCommPlan &comm) {
  return reinterpret_cast<const uint8_t *>(ptr) + comm.byte_offset;
}

} // namespace tl
