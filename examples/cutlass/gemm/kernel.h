#pragma once

#include <torch/extension.h>

void gemm_launcher(
    torch::Tensor& C,
    const torch::Tensor& A,
    const torch::Tensor& B,
    cudaStream_t s);
