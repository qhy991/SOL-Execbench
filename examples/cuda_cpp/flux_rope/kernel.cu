#include <torch/extension.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void rope_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_f,
    const __nv_bfloat16* __restrict__ sin_f,
    __nv_bfloat16* __restrict__ out,
    int B, int S, int H, int head_dim)
{
    int idx      = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    if (idx >= B * S * H * half_dim) return;

    int pair  = idx % half_dim;
    int h     = (idx / half_dim) % H;
    int s     = (idx / half_dim / H) % S;
    int b     = idx / half_dim / H / S;
    int base  = ((b * S + s) * H + h) * head_dim + pair * 2;
    int fbase = s * head_dim + pair * 2;

    float x1 = __bfloat162float(x[base]),      x2 = __bfloat162float(x[base+1]);
    float c1 = __bfloat162float(cos_f[fbase]),  c2 = __bfloat162float(cos_f[fbase+1]);
    float s1 = __bfloat162float(sin_f[fbase]),  s2 = __bfloat162float(sin_f[fbase+1]);

    out[base]   = __float2bfloat16(x1*c1 - x2*s1);
    out[base+1] = __float2bfloat16(x1*s2 + x2*c2);
}

torch::Tensor run(torch::Tensor query_or_key,
                  torch::Tensor freqs_cos, torch::Tensor freqs_sin) {
    const at::cuda::CUDAGuard guard(query_or_key.device());
    auto B = query_or_key.size(0), S = query_or_key.size(1);
    auto H = query_or_key.size(2), head_dim = query_or_key.size(3);
    auto output = torch::empty_like(query_or_key);
    int total_pairs = B * S * H * (head_dim / 2);
    rope_kernel<<<(total_pairs + 255) / 256, 256>>>(
        (const __nv_bfloat16*)query_or_key.data_ptr(),
        (const __nv_bfloat16*)freqs_cos.contiguous().data_ptr(),
        (const __nv_bfloat16*)freqs_sin.contiguous().data_ptr(),
        (__nv_bfloat16*)output.data_ptr(),
        B, S, H, head_dim);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "FLUX RoPE apply rotation (CUDA)");
}
