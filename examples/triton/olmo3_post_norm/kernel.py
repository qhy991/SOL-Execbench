import torch, triton, triton.language as tl

@triton.jit
def _rmsnorm_residual_kernel(x_ptr, res_ptr, w_ptr, out_ptr,
                              eps, hidden_size, BLOCK_SIZE: tl.constexpr):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size

    x = tl.load(x_ptr + row * hidden_size + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / hidden_size
    inv_rms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)

    # Round normalized to bf16 first (matches reference's .to(input_dtype))
    normed_bf16 = (x * inv_rms * w).to(tl.bfloat16)

    # Residual add: upcast both to f32, add, round to bf16
    # Matches PyTorch's bf16+bf16 semantics (upcast -> add -> round)
    res = tl.load(res_ptr + row * hidden_size + offs, mask=mask, other=0.0)
    out = (normed_bf16.to(tl.float32) + res.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + row * hidden_size + offs, out, mask=mask)

@torch.no_grad()
def run(sublayer_output, residual, weight, eps):
    shape = sublayer_output.shape
    hidden_size = shape[-1]
    n_rows = sublayer_output.numel() // hidden_size
    x_flat   = sublayer_output.contiguous().view(n_rows, hidden_size)
    res_flat = residual.contiguous().view(n_rows, hidden_size)
    output   = torch.empty_like(x_flat)
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    _rmsnorm_residual_kernel[(n_rows,)](
        x_flat, res_flat, weight.contiguous(), output,
        eps, hidden_size, BLOCK_SIZE=BLOCK_SIZE)
    return output.view(shape)
