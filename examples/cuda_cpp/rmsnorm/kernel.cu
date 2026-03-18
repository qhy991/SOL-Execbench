#include "kernel.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>

/**
 * @brief Performs a sum-reduction within a single warp.
 *
 * Uses the __shfl_down_sync primitive for efficient, synchronization-free
 * communication between threads in a warp. All threads in the warp must call this function.
 *
 * @param val The float value each thread contributes to the sum.
 * @return The total sum, which is valid only in lane 0 of the warp.
 */
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * @brief CUDA kernel for RMS Normalization, optimized for hidden_size=4096.
 *
 * This kernel is specifically tailored for the B200 architecture by using:
 * - __nv_bfloat162 for vectorized memory access, doubling memory throughput.
 * - A large block size (1024 threads) to maximize parallelism.
 * - A two-stage reduction: a fast, parallel reduction in shared memory followed
 *   by an efficient warp-level reduction using shuffle instructions.
 * - All intermediate calculations are performed in FP32 for precision.
 * - Each block processes a single row (batch element), simplifying indexing and
 *   ensuring data locality.
 *
 * @param output Pointer to the output tensor data ([batch_size, 4096], bfloat16).
 * @param hidden_states Pointer to the input hidden_states tensor data ([batch_size, 4096], bfloat16).
 * @param weight Pointer to the weight tensor data ([4096], bfloat16).
 * @param batch_size The number of rows to process.
 * @param eps A small constant to add to the variance for numerical stability.
 */
__global__ void __launch_bounds__(1024, 1) rmsnorm_h4096_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ hidden_states,
    const __nv_bfloat16* __restrict__ weight,
    const int batch_size,
    const float eps) {

    // Each block processes one row. Grid dimension is the batch size.
    const int row_idx = blockIdx.x;
    if (row_idx >= batch_size) {
        return;
    }

    const int thread_idx = threadIdx.x;

    // hidden_size = 4096 -> 2048 bfloat162 elements.
    // blockDim.x = 1024 -> 2 bfloat162 elements per thread.
    constexpr int items_per_thread = 2;
    constexpr int hidden_size_vec = 2048; // 4096 / 2
    constexpr int hidden_size = 4096;

    // Pointers for the current row, cast for vectorized access
    const __nv_bfloat162* hidden_states_vec = reinterpret_cast<const __nv_bfloat162*>(hidden_states) + row_idx * hidden_size_vec;
    const __nv_bfloat162* weight_vec = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* output_vec = reinterpret_cast<__nv_bfloat162*>(output) + row_idx * hidden_size_vec;

    // Shared memory for block-wide reduction. Size is fixed (1024 floats).
    __shared__ float s_sum[1024];

    // --- Step 1: Calculate sum of squares per thread ---
    float thread_sum_sq = 0.0f;
    for (int i = 0; i < items_per_thread; ++i) {
        int col_idx = thread_idx + i * blockDim.x;
        __nv_bfloat162 h_vec = __ldg(hidden_states_vec + col_idx); // Use L1 cache streaming
        float2 h_f2 = __bfloat1622float2(h_vec);
        thread_sum_sq += h_f2.x * h_f2.x + h_f2.y * h_f2.y;
    }
    s_sum[thread_idx] = thread_sum_sq;
    __syncthreads();

    // --- Step 2: Block-wide reduction in shared memory ---
    // Reduce from 1024 -> 64 partial sums
    for (unsigned int s = 512; s > 32; s >>= 1) {
        if (thread_idx < s) {
            s_sum[thread_idx] += s_sum[thread_idx + s];
        }
        __syncthreads();
    }
    
    // The first two warps (64 threads) perform the final reduction steps
    if (thread_idx < 32) {
        // First warp sums its own partial sum with the corresponding partial sum from the second warp
        s_sum[thread_idx] += s_sum[thread_idx + 32];
        
        // Final reduction within the first warp using shuffle instructions
        float warp_total_sum = warp_reduce_sum(s_sum[thread_idx]);
        if (thread_idx == 0) {
            s_sum[0] = warp_total_sum;
        }
    }
    __syncthreads();

    // --- Step 3: Calculate inv_rms and broadcast via shared memory ---
    float inv_rms = s_sum[0]; // All threads read the final sum from thread 0's calculation
    if (thread_idx == 0) {
        float mean_sum_sq = inv_rms / hidden_size;
        inv_rms = rsqrtf(mean_sum_sq + eps);
        s_sum[0] = inv_rms; // Broadcast the final inv_rms value
    }
    __syncthreads();
    
    inv_rms = s_sum[0]; // All threads read the broadcasted value

    // --- Step 4: Apply normalization and scaling ---
    for (int i = 0; i < items_per_thread; ++i) {
        int col_idx = thread_idx + i * blockDim.x;
        __nv_bfloat162 h_vec = __ldg(hidden_states_vec + col_idx);
        __nv_bfloat162 w_vec = __ldg(weight_vec + col_idx);

        float2 h_f2 = __bfloat1622float2(h_vec);
        float2 w_f2 = __bfloat1622float2(w_vec);

        h_f2.x = (h_f2.x * inv_rms) * w_f2.x;
        h_f2.y = (h_f2.y * inv_rms) * w_f2.y;

        output_vec[col_idx] = __float22bfloat162_rn(h_f2);
    }
}

// Host launcher function implementation
void rmsnorm_h4096_launcher(
    torch::Tensor& output,
    const torch::Tensor& hidden_states,
    const torch::Tensor& weight,
    float eps,
    cudaStream_t stream) {

    const auto batch_size = hidden_states.size(0);
    
    // Kernel launch configuration
    // Each block of 1024 threads handles one row of the input tensor.
    dim3 blockDim(1024);
    dim3 gridDim(batch_size);

    // Get raw data pointers from PyTorch tensors
    auto* output_ptr = output.data_ptr<at::BFloat16>();
    const auto* hidden_states_ptr = hidden_states.data_ptr<at::BFloat16>();
    const auto* weight_ptr = weight.data_ptr<at::BFloat16>();
    
    // Launch the kernel
    rmsnorm_h4096_kernel<<<gridDim, blockDim, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(output_ptr),
        reinterpret_cast<const __nv_bfloat16*>(hidden_states_ptr),
        reinterpret_cast<const __nv_bfloat16*>(weight_ptr),
        batch_size,
        eps
    );

    // Check for any kernel launch errors in debug builds
    #ifndef NDEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
    #endif
}