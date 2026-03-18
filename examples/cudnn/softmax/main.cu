#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cudnn.h>
#include <stdexcept>
#include <string>

#define CUDNN_CHECK(expr)                                                  \
    do {                                                                   \
        cudnnStatus_t _status = (expr);                                    \
        if (_status != CUDNN_STATUS_SUCCESS) {                             \
            throw std::runtime_error(                                      \
                std::string("cuDNN error: ") + cudnnGetErrorString(_status)); \
        }                                                                  \
    } while (0)

torch::Tensor run(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");

    int batch_size = input.size(0);
    int hidden_size = input.size(1);

    auto output = torch::empty_like(input);

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnSetStream(handle, at::cuda::getCurrentCUDAStream()));

    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, hidden_size, 1, 1));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnSoftmaxForward(
        handle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, desc, input.data_ptr<float>(),
        &beta, desc, output.data_ptr<float>()));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    CUDNN_CHECK(cudnnDestroy(handle));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "cuDNN Softmax forward");
}
