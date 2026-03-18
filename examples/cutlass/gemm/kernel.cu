#include "kernel.h"

#include <cutlass/gemm/device/gemm.h>

using ColumnMajor = cutlass::layout::ColumnMajor;

using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                ColumnMajor,  // Layout of A matrix
                                                float,        // Data-type of B matrix
                                                ColumnMajor,  // Layout of B matrix
                                                float,        // Data-type of C matrix
                                                ColumnMajor>; // Layout of C matrix

void gemm_launcher(
    torch::Tensor& C,
    const torch::Tensor& A,
    const torch::Tensor& B,
    cudaStream_t s) {

    // PyTorch stores tensors row-major.  CUTLASS uses column-major.
    // row-major C(M,N) = A(M,K) @ B(K,N)
    //   is equivalent to
    // col-major C^T(N,M) = B^T(N,K) @ A^T(K,M)
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    float const* A_ptr = A.data_ptr<float>();
    float const* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    CutlassGemm gemm_operator;

    CutlassGemm::Arguments args(
        {N, M, K},           // Gemm problem dimensions (swapped M,N)
        {B_ptr, N},          // Tensor-ref for B^T (acts as A in col-major), lda=N
        {A_ptr, K},          // Tensor-ref for A^T (acts as B in col-major), ldb=K
        {C_ptr, N},          // Tensor-ref for source C^T, ldc=N
        {C_ptr, N},          // Tensor-ref for destination D^T, ldd=N
        {1.0f, 0.0f});       // Scalars: alpha, beta

    cutlass::Status status = gemm_operator(args, nullptr, s);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS GEMM failed: ", cutlassGetStatusString(status));
}
