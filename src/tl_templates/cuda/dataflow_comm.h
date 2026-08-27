#pragma once

#include "barrier.h"
#include "dataflow_runtime.h"

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)) ||                      \
    (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 900))
#include "copy_sm90.h"
#endif

namespace tl {

TL_DEVICE bool dataflow_ptr_aligned_16(const void *ptr) {
  return (reinterpret_cast<unsigned long long>(ptr) & 0xfULL) == 0;
}

TL_DEVICE void dataflow_copy_bytes(void *dst, const void *src, uint32_t bytes) {
  uint32_t tid = dataflow_thread_rank();
  uint32_t stride = dataflow_num_threads();

  if (dataflow_ptr_aligned_16(dst) && dataflow_ptr_aligned_16(src)) {
    uint32_t vec_count = bytes >> 4;
    auto *dst_vec = reinterpret_cast<uint4 *>(dst);
    auto const *src_vec = reinterpret_cast<const uint4 *>(src);
    for (uint32_t i = tid; i < vec_count; i += stride) {
      dst_vec[i] = src_vec[i];
    }

    uint32_t tail_offset = vec_count << 4;
    auto *dst_bytes = reinterpret_cast<uint8_t *>(dst);
    auto const *src_bytes = reinterpret_cast<const uint8_t *>(src);
    for (uint32_t i = tail_offset + tid; i < bytes; i += stride) {
      dst_bytes[i] = src_bytes[i];
    }
    return;
  }

  auto *dst_bytes = reinterpret_cast<uint8_t *>(dst);
  auto const *src_bytes = reinterpret_cast<const uint8_t *>(src);
  for (uint32_t i = tid; i < bytes; i += stride) {
    dst_bytes[i] = src_bytes[i];
  }
}

template <typename BarrierType = uint64_t>
TL_DEVICE void dataflow_init_cluster_barrier(BarrierType &barrier,
                                             uint32_t arrive_count = 1) {
  if (dataflow_is_leader_thread()) {
    tl::mbarrier_init(barrier, arrive_count);
    tl::fence_barrier_init();
  }
  __syncthreads();
}

template <typename BarrierType = uint64_t>
TL_DEVICE void
dataflow_send_cluster_relaxed(void *dst_shared, const void *src_shared,
                              uint32_t dst_cta_rank, uint32_t bytes,
                              BarrierType &remote_barrier) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (dataflow_is_leader_thread()) {
    tl::tma_store_cluster(dst_shared, const_cast<void *>(src_shared),
                          static_cast<int>(dst_cta_rank), bytes,
                          remote_barrier);
  }
#else
  TILELANG_UNREACHABLE("Dataflow cluster send requires sm90+");
#endif
}

TL_DEVICE void dataflow_begin_cluster_send_batch() { __syncthreads(); }

template <typename BarrierType = uint64_t>
TL_DEVICE void dataflow_send_cluster(void *dst_shared, const void *src_shared,
                                     uint32_t dst_cta_rank, uint32_t bytes,
                                     BarrierType &remote_barrier) {
  dataflow_begin_cluster_send_batch();
  dataflow_send_cluster_relaxed(dst_shared, src_shared, dst_cta_rank, bytes,
                                remote_barrier);
}

template <typename BarrierType = uint64_t>
TL_DEVICE void dataflow_recv_cluster_relaxed(BarrierType &local_barrier,
                                             uint32_t parity) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (dataflow_is_leader_thread()) {
    tl::mbarrier_wait(local_barrier, static_cast<int>(parity & 1u));
  }
#else
  TILELANG_UNREACHABLE("Dataflow cluster recv requires sm90+");
#endif
}

TL_DEVICE void dataflow_complete_cluster_recv_batch() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (dataflow_is_leader_thread()) {
    tl::fence_proxy_async();
  }
  __syncthreads();
#else
  TILELANG_UNREACHABLE("Dataflow cluster recv requires sm90+");
#endif
}

template <typename BarrierType = uint64_t>
TL_DEVICE void dataflow_recv_cluster(BarrierType &local_barrier,
                                     uint32_t parity) {
  dataflow_recv_cluster_relaxed(local_barrier, parity);
  dataflow_complete_cluster_recv_batch();
}

TL_DEVICE void dataflow_store_flag_release(uint32_t *flags, uint32_t flag_index,
                                           uint32_t epoch) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  asm volatile("st.global.release.gpu.b32 [%0], %1;\n"
               :
               : "l"(flags + flag_index), "r"(epoch)
               : "memory");
#else
  __threadfence();
  reinterpret_cast<volatile uint32_t *>(flags)[flag_index] = epoch;
#endif
}

TL_DEVICE uint32_t dataflow_load_flag_acquire(const uint32_t *flags,
                                              uint32_t flag_index) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  uint32_t epoch;
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
               : "=r"(epoch)
               : "l"(flags + flag_index)
               : "memory");
  return epoch;
#else
  return reinterpret_cast<const volatile uint32_t *>(flags)[flag_index];
#endif
}

TL_DEVICE void dataflow_wait_hbm_flag(const uint32_t *flags,
                                      uint32_t flag_index, uint32_t epoch) {
  while (dataflow_load_flag_acquire(flags, flag_index) < epoch) {
    __nanosleep(64);
  }
  __syncthreads();
}

TL_DEVICE void dataflow_send_hbm_scalar(void *global_dst,
                                        const void *shared_src, uint32_t bytes,
                                        uint32_t *flags, uint32_t flag_index,
                                        uint32_t epoch) {
  __syncthreads();
  dataflow_copy_bytes(global_dst, shared_src, bytes);
  // Every thread publishes its striped payload stores before the leader flag.
  __threadfence();
  __syncthreads();
  if (dataflow_is_leader_thread()) {
    dataflow_store_flag_release(flags, flag_index, epoch);
  }
}

TL_DEVICE void dataflow_recv_hbm_scalar(void *shared_dst,
                                        const void *global_src, uint32_t bytes,
                                        const uint32_t *flags,
                                        uint32_t flag_index, uint32_t epoch) {
  dataflow_wait_hbm_flag(flags, flag_index, epoch);
  dataflow_copy_bytes(shared_dst, global_src, bytes);
  __syncthreads();
}

TL_DEVICE void dataflow_send_hbm(void *global_dst, const void *shared_src,
                                 uint32_t bytes, uint32_t *flags,
                                 uint32_t flag_index, uint32_t epoch,
                                 bool first_segment = true,
                                 bool last_segment = true) {
#if defined(TILELANG_DATAFLOW_HBM_USE_TMA) && defined(__CUDA_ARCH__) &&        \
    (__CUDA_ARCH__ >= 900)
  if (first_segment) {
    __syncthreads();
  }
  if (dataflow_is_leader_thread()) {
    if (first_segment) {
      tl::fence_proxy_async();
    }
    // The handoff payload is consumed shortly after publication.  Retain the
    // producer write in L2 until that single consumer reloads it.
    tl::tma_store<tl::CacheHintSm90::EVICT_LAST>(
        global_dst, const_cast<void *>(shared_src), bytes);
    tl::tma_store_arrive();
    tl::tma_store_wait<0>();
    dataflow_store_flag_release(flags, flag_index, epoch);
  }
  if (last_segment) {
    __syncthreads();
  }
#else
  dataflow_send_hbm_scalar(global_dst, shared_src, bytes, flags, flag_index,
                           epoch);
#endif
}

template <typename BarrierType = uint64_t>
TL_DEVICE void dataflow_recv_hbm_issue(void *shared_dst, const void *global_src,
                                       uint32_t bytes, const uint32_t *flags,
                                       uint32_t flag_index, uint32_t epoch,
                                       BarrierType &local_barrier) {
#if defined(TILELANG_DATAFLOW_HBM_USE_TMA) && defined(__CUDA_ARCH__) &&        \
    (__CUDA_ARCH__ >= 900)
  if (dataflow_is_leader_thread()) {
    while (dataflow_load_flag_acquire(flags, flag_index) < epoch) {
      __nanosleep(64);
    }
    tl::mbarrier_arrive_expect_tx(local_barrier, bytes);
    tl::tma_load(shared_dst, global_src, local_barrier, bytes);
  }
#else
  dataflow_recv_hbm_scalar(shared_dst, global_src, bytes, flags, flag_index,
                           epoch);
#endif
}

template <typename BarrierType = uint64_t>
TL_DEVICE void dataflow_recv_hbm_wait(uint32_t barrier_phase,
                                      BarrierType &local_barrier,
                                      bool last_segment = true) {
#if defined(TILELANG_DATAFLOW_HBM_USE_TMA) && defined(__CUDA_ARCH__) &&        \
    (__CUDA_ARCH__ >= 900)
  if (dataflow_is_leader_thread()) {
    tl::mbarrier_wait(local_barrier, static_cast<int>(barrier_phase & 1u));
    if (last_segment) {
      tl::fence_proxy_async();
    }
  }
  if (last_segment) {
    __syncthreads();
  }
#else
  (void)barrier_phase;
  (void)local_barrier;
  (void)last_segment;
#endif
}

template <typename BarrierType = uint64_t>
TL_DEVICE void
dataflow_recv_hbm(void *shared_dst, const void *global_src, uint32_t bytes,
                  const uint32_t *flags, uint32_t flag_index, uint32_t epoch,
                  uint32_t barrier_phase, BarrierType &local_barrier,
                  bool last_segment = true) {
  dataflow_recv_hbm_issue(shared_dst, global_src, bytes, flags, flag_index,
                          epoch, local_barrier);
  dataflow_recv_hbm_wait(barrier_phase, local_barrier, last_segment);
}

TL_DEVICE bool dataflow_hbm_recv_is_issued(const uint32_t *issued_words,
                                           uint32_t barrier_index) {
  return (issued_words[barrier_index >> 5] & (1u << (barrier_index & 31u))) !=
         0u;
}

TL_DEVICE void dataflow_hbm_recv_mark_issued(uint32_t *issued_words,
                                             uint32_t barrier_index) {
  issued_words[barrier_index >> 5] |= 1u << (barrier_index & 31u);
}

TL_DEVICE void dataflow_hbm_recv_clear_issued(uint32_t *issued_words,
                                              uint32_t barrier_index) {
  issued_words[barrier_index >> 5] &= ~(1u << (barrier_index & 31u));
}

template <typename BarrierType = uint64_t>
TL_DEVICE void
dataflow_try_recv_hbm_issue(void *shared_dst, const void *global_src,
                            uint32_t bytes, const uint32_t *flags,
                            uint32_t flag_index, uint32_t epoch,
                            BarrierType &local_barrier, uint32_t *issued_words,
                            uint32_t barrier_index, bool flag_ready = false) {
#if defined(TILELANG_DATAFLOW_HBM_USE_TMA) && defined(__CUDA_ARCH__) &&        \
    (__CUDA_ARCH__ >= 900)
  if (dataflow_is_leader_thread() &&
      !dataflow_hbm_recv_is_issued(issued_words, barrier_index) &&
      (flag_ready || dataflow_load_flag_acquire(flags, flag_index) >= epoch)) {
    tl::mbarrier_arrive_expect_tx(local_barrier, bytes);
    tl::tma_load(shared_dst, global_src, local_barrier, bytes);
    dataflow_hbm_recv_mark_issued(issued_words, barrier_index);
  }
#else
  (void)shared_dst;
  (void)global_src;
  (void)bytes;
  (void)flags;
  (void)flag_index;
  (void)epoch;
  (void)local_barrier;
  (void)issued_words;
  (void)barrier_index;
  (void)flag_ready;
#endif
}

template <typename BarrierType = uint64_t>
TL_DEVICE void dataflow_recv_hbm_wait_or_issue(
    void *shared_dst, const void *global_src, uint32_t bytes,
    const uint32_t *flags, uint32_t flag_index, uint32_t epoch,
    uint32_t barrier_phase, BarrierType &local_barrier, uint32_t *issued_words,
    uint32_t barrier_index, bool last_segment = true) {
#if defined(TILELANG_DATAFLOW_HBM_USE_TMA) && defined(__CUDA_ARCH__) &&        \
    (__CUDA_ARCH__ >= 900)
  if (dataflow_is_leader_thread()) {
    if (!dataflow_hbm_recv_is_issued(issued_words, barrier_index)) {
      while (dataflow_load_flag_acquire(flags, flag_index) < epoch) {
        __nanosleep(64);
      }
      tl::mbarrier_arrive_expect_tx(local_barrier, bytes);
      tl::tma_load(shared_dst, global_src, local_barrier, bytes);
    }
    tl::mbarrier_wait(local_barrier, static_cast<int>(barrier_phase & 1u));
    dataflow_hbm_recv_clear_issued(issued_words, barrier_index);
    if (last_segment) {
      tl::fence_proxy_async();
    }
  }
  if (last_segment) {
    __syncthreads();
  }
#else
  (void)issued_words;
  (void)barrier_index;
  (void)barrier_phase;
  (void)last_segment;
  dataflow_recv_hbm_scalar(shared_dst, global_src, bytes, flags, flag_index,
                           epoch);
#endif
}

TL_DEVICE void dataflow_send_hbm_direct_global(uint32_t *flags,
                                               uint32_t flag_index,
                                               uint32_t epoch) {
  __threadfence();
  __syncthreads();
  if (dataflow_is_leader_thread()) {
    dataflow_store_flag_release(flags, flag_index, epoch);
  }
}

TL_DEVICE void dataflow_recv_hbm_direct_global(const uint32_t *flags,
                                               uint32_t flag_index,
                                               uint32_t epoch) {
  dataflow_wait_hbm_flag(flags, flag_index, epoch);
}

template <typename BarrierType = uint64_t>
TL_DEVICE void
dataflow_comm_dispatch(const DataflowCommPlan &comm, const DataflowSlot *slots,
                       void *shared_base, void *scratch_base, void *global_base,
                       uint32_t *flags, BarrierType *barriers,
                       uint32_t *hbm_recv_issued = nullptr,
                       bool hbm_flag_ready = false) {
  if (comm.kind == static_cast<uint32_t>(DataflowCommKind::kNone)) {
    return;
  }

  const DataflowSlot &src_slot = slots[comm.src_slot_id];
  const DataflowSlot &dst_slot = slots[comm.dst_slot_id];
  uint32_t bytes = dataflow_comm_bytes(comm, src_slot, dst_slot);
  uint32_t barrier_index =
      dataflow_select_index(comm.barrier_index, dst_slot.barrier_index);

  switch (static_cast<DataflowCommKind>(comm.kind)) {
  case DataflowCommKind::kClusterSend:
    dataflow_send_cluster(
        dataflow_slot_shared_ptr(shared_base, scratch_base, dst_slot),
        dataflow_slot_shared_ptr(shared_base, scratch_base, src_slot),
        comm.peer_cta_rank, bytes, barriers[barrier_index]);
    break;
  case DataflowCommKind::kClusterRecv:
    dataflow_recv_cluster(barriers[barrier_index], comm.flag_epoch);
    break;
  case DataflowCommKind::kHBMSend:
    dataflow_send_hbm(
        dataflow_comm_byte_ptr(dataflow_slot_global_ptr(global_base, dst_slot),
                               comm),
        dataflow_comm_byte_ptr(
            dataflow_slot_shared_ptr(shared_base, scratch_base, src_slot),
            comm),
        bytes, flags,
        dataflow_select_index(comm.flag_index, dst_slot.flag_index),
        comm.flag_epoch, comm.segment_id == 0u,
        comm.segment_id + 1u == comm.segment_count);
    break;
  case DataflowCommKind::kHBMRecv:
    dataflow_recv_hbm(
        dataflow_comm_byte_ptr(
            dataflow_slot_shared_ptr(shared_base, scratch_base, dst_slot),
            comm),
        dataflow_comm_byte_ptr(dataflow_slot_global_ptr(global_base, dst_slot),
                               comm),
        bytes, flags,
        dataflow_select_index(comm.flag_index, dst_slot.flag_index),
        comm.flag_epoch, comm.barrier_phase, barriers[barrier_index],
        comm.segment_id + 1u == comm.segment_count);
    break;
  case DataflowCommKind::kHBMRecvIssue:
    if (hbm_recv_issued == nullptr) {
#if defined(__CUDA_ARCH__)
      __trap();
#else
      TILELANG_UNREACHABLE("split HBM receive requires runtime issue state");
#endif
    }
    dataflow_try_recv_hbm_issue(
        dataflow_comm_byte_ptr(
            dataflow_slot_shared_ptr(shared_base, scratch_base, dst_slot),
            comm),
        dataflow_comm_byte_ptr(dataflow_slot_global_ptr(global_base, dst_slot),
                               comm),
        bytes, flags,
        dataflow_select_index(comm.flag_index, dst_slot.flag_index),
        comm.flag_epoch, barriers[barrier_index], hbm_recv_issued,
        barrier_index, hbm_flag_ready);
    break;
  case DataflowCommKind::kHBMRecvWait:
    if (hbm_recv_issued == nullptr) {
#if defined(__CUDA_ARCH__)
      __trap();
#else
      TILELANG_UNREACHABLE("split HBM receive requires runtime issue state");
#endif
    }
    dataflow_recv_hbm_wait_or_issue(
        dataflow_comm_byte_ptr(
            dataflow_slot_shared_ptr(shared_base, scratch_base, dst_slot),
            comm),
        dataflow_comm_byte_ptr(dataflow_slot_global_ptr(global_base, dst_slot),
                               comm),
        bytes, flags,
        dataflow_select_index(comm.flag_index, dst_slot.flag_index),
        comm.flag_epoch, comm.barrier_phase, barriers[barrier_index],
        hbm_recv_issued, barrier_index,
        comm.segment_id + 1u == comm.segment_count);
    break;
  default:
#if defined(__CUDA_ARCH__)
    __trap();
#else
    TILELANG_UNREACHABLE("Unknown Dataflow communication kind");
#endif
  }
}

template <typename BarrierType = uint64_t>
TL_DEVICE void dataflow_comm_dispatch(const DataflowCommPlan &comm,
                                      const DataflowSlot *slots,
                                      void *shared_base, void *global_base,
                                      uint32_t *flags, BarrierType *barriers) {
  dataflow_comm_dispatch(comm, slots, shared_base, shared_base, global_base,
                         flags, barriers);
}

template <typename BarrierType = uint64_t>
TL_DEVICE void dataflow_comm_dispatch(const DataflowCommPlan *comm_plans,
                                      uint32_t comm_index,
                                      const DataflowSlot *slots,
                                      void *shared_base, void *global_base,
                                      uint32_t *flags, BarrierType *barriers) {
  dataflow_comm_dispatch(comm_plans[comm_index], slots, shared_base,
                         global_base, flags, barriers);
}

template <typename BarrierType = uint64_t>
TL_DEVICE void
dataflow_comm_dispatch(const DataflowCommPlan *comm_plans, uint32_t comm_index,
                       const DataflowSlot *slots, void *shared_base,
                       void *scratch_base, void *global_base, uint32_t *flags,
                       BarrierType *barriers) {
  dataflow_comm_dispatch(comm_plans[comm_index], slots, shared_base,
                         scratch_base, global_base, flags, barriers);
}

} // namespace tl
