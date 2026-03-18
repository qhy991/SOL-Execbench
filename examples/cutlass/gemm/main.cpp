#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"

torch::Tensor run(const torch::Tensor& A, const torch::Tensor& B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "inputs must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "inner dimensions must match");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "inputs must be contiguous");

    auto C = torch::empty({A.size(0), B.size(1)}, A.options());
    cudaStream_t current = at::cuda::getCurrentCUDAStream();
    gemm_launcher(C, A, B, current);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "CUTLASS GEMM");
}
