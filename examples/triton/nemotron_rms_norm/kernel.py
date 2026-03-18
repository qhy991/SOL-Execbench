import torch, triton, triton.language as tl

@triton.jit
def _fused_add_rmsnorm_kernel(x_ptr, res_ptr, w_ptr, out_ptr,
                               eps, hidden_size, BLOCK_SIZE: tl.constexpr):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size

    # Load as bf16 -- match reference's bfloat16 inputs
    x_bf16   = tl.load(x_ptr   + row * hidden_size + offs, mask=mask, other=0.0)
    res_bf16 = tl.load(res_ptr  + row * hidden_size + offs, mask=mask, other=0.0)

    # Step 1: residual add with bf16 rounding (bf16+bf16 -> f32 -> bf16)
    x_sum_bf16 = (x_bf16.to(tl.float32) + res_bf16.to(tl.float32)).to(tl.bfloat16)

    # Step 2-4: RMSNorm in float32
    x = x_sum_bf16.to(tl.float32)
    var     = tl.sum(x * x, axis=0) / hidden_size
    inv_rms = tl.rsqrt(var + eps)

    # Step 5: round normalized to bf16, multiply weight (also bf16 multiply)
    w           = tl.load(w_ptr + offs, mask=mask, other=1.0)  # bf16
    x_norm_bf16 = (x * inv_rms).to(tl.bfloat16)
    out = (x_norm_bf16.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + row * hidden_size + offs, out, mask=mask)

@torch.no_grad()
def run(hidden_states, residual, weight, eps):
    shape       = hidden_states.shape
    hidden_size = shape[-1]
    n_rows      = hidden_states.numel() // hidden_size
    x_flat   = hidden_states.contiguous().view(n_rows, hidden_size)
    res_flat = residual.contiguous().view(n_rows, hidden_size)
    output   = torch.empty_like(x_flat)
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    _fused_add_rmsnorm_kernel[(n_rows,)](
        x_flat, res_flat, weight.contiguous(), output,
        eps, hidden_size, BLOCK_SIZE=BLOCK_SIZE)
    return output.view(shape)
