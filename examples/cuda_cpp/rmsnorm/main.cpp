#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"
#include <string>

// Helper function to check common tensor properties
void check_tensor(const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kBFloat16, name, " must have bfloat16 dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

/**
 * @brief Python-bindable 'run' function for RMSNorm.
 *
 * This function serves as the entry point from Python. It performs extensive
 * validation on the input tensors to ensure they meet the kernel's requirements.
 * It then allocates the output tensor and calls the CUDA kernel launcher.
 *
 * @param hidden_states Input tensor of shape [batch_size, 4096] and dtype bfloat16.
 * @param weight Weight tensor of shape [4096] and dtype bfloat16.
 * @return The output tensor with the same shape and dtype as hidden_states.
 */
torch::Tensor run(
    const torch::Tensor& hidden_states,
    const torch::Tensor& weight) {

    // --- Input Validation ---
    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be a 2D tensor, but got ", hidden_states.dim(), " dimensions");
    TORCH_CHECK(weight.dim() == 1, "weight must be a 1D tensor, but got ", weight.dim(), " dimensions");

    const int64_t hidden_size = hidden_states.size(1);
    
    TORCH_CHECK(hidden_size == 4096, "hidden_size must be 4096, but got ", hidden_size);
    TORCH_CHECK(weight.size(0) == hidden_size, "weight must have size ", hidden_size, ", but got ", weight.size(0));

    check_tensor(hidden_states, "hidden_states");
    check_tensor(weight, "weight");
    
    // --- Output Tensor Allocation ---
    auto output = torch::empty_like(hidden_states);

    // --- Kernel Execution ---
    const float eps = 1e-5f;

    // Get current CUDA stream from PyTorch's context
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch the kernel via the C++ wrapper function in the .cu file
    rmsnorm_h4096_launcher(
        output,
        hidden_states,
        weight,
        eps,
        stream
    );

    return output;
}

// --- Pybind11 Module Definition ---
// Exposes the 'run' function to Python, making it callable as a C++ extension.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "RMSNorm kernel for hidden_size=4096 (BFloat16, CUDA, B200 Optimized)");
}