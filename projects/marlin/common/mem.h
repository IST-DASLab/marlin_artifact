#pragma once
#include "base.h"

namespace marlin {
// Predicated asynchronous global->shared copy; used for inputs A where we apply
// predication to handle batchsizes that are not multiples of 16.
__device__ inline void cp_async4_pred_zfill(void *smem_ptr,
                                            const void *glob_ptr,
                                            bool pred = true,
                                            const bool zfill = false) {
  const int BYTES = 16;
  int src_in_bytes = (zfill ? 0 : BYTES);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("{\n"
               "   .reg .pred p;\n"
               "   setp.ne.b32 p, %0, 0;\n"
               "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
               "}\n" ::"r"((int)pred),
               "r"(smem), "l"(glob_ptr), "n"(BYTES), "r"(src_in_bytes));
}

__device__ inline void cp_async4_pred(void *smem_ptr, const void *glob_ptr,
                                      bool pred = true) {
  const int BYTES = 16;
  // int src_in_bytes = (zfill ? 0 : BYTES );
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("{\n"
               "   .reg .pred p;\n"
               "   setp.ne.b32 p, %0, 0;\n"
               "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
               "}\n" ::"r"((int)pred),
               "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

// Asynchronous global->shared copy with a cache hint indicating that the values
// may be evicted immediately; used for quantized weights B, which are only
// accessed precisely once and should thus not pollute the L2 cache which we
// need for inputs A and outputs C.
__device__ inline void cp_async4_stream(void *smem_ptr, const void *glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .b64 p;\n"
      "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
      "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
}

// Async copy fence.
__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most `n` async copy stages are still pending.
template <int n> __device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in tensor core layout.
__device__ inline void ldsm4(FragA &frag_a, const void *smem_ptr) {
  uint32_t *a = reinterpret_cast<uint32_t *>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(smem));
}

__device__ inline void ldsm4_m(FragM &frag_m, const void *smem_ptr) {
  uint32_t *a = reinterpret_cast<uint32_t *>(&frag_m);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  /* asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
  ); */
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(a[0]), "=r"(a[1])
               : "r"(smem));
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in tensor core layout.
__device__ inline void ldsm4_t(FragA &frag_a, const void *smem_ptr) {
  uint32_t *a = reinterpret_cast<uint32_t *>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
      : "r"(smem));
}

// Wait until barrier reaches `count`, then lock for current threadblock.
__device__ inline void barrier_acquire(int *lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

// Release barrier and increment visitation count.
__device__ inline void barrier_release(int *lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    // Make sure that all writes since acquiring this barrier are visible
    // globally, while releasing the barrier.
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                 :
                 : "l"(lock), "r"(val));
  }
}
} // namespace marlin