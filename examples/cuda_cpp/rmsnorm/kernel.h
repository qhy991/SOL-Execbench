#pragma once

#ifndef RMSNORM_KERNEL_H
#define RMSNORM_KERNEL_H

#include <torch/extension.h>

/**
 * @brief Host-side launcher for the RMSNorm CUDA kernel.
 *
 * This function validates tensor shapes, sets up CUDA kernel launch parameters
 * (grid and block dimensions), and launches the rmsnorm_h4096_kernel on the
 * specified CUDA stream.
 *
 * @param output The output tensor, pre-allocated with the same shape as hidden_states.
 * @param hidden_states The input tensor of shape [batch_size, 4096].
 * @param weight The weight tensor of shape [4096].
 * @param eps A small float value to avoid division by zero.
 * @param stream The CUDA stream on which to launch the kernel.
 */
void rmsnorm_h4096_launcher(
    torch::Tensor& output,
    const torch::Tensor& hidden_states,
    const torch::Tensor& weight,
    float eps,
    cudaStream_t stream
);

#endif // RMSNORM_KERNEL_H