import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr,
    output_ptr,
    weight_ptr,
    eps,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Forward pass for RMS normalization with improved numerical stability."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load input as bfloat16, cast to float32 for computation
    x = tl.load(x_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)

    # Compute RMS in float32 for better precision
    x_squared = x_fp32 * x_fp32
    mean_squared = tl.sum(x_squared, axis=0) / n_cols
    rms = tl.sqrt(mean_squared + eps)

    # Normalize
    x_normed = x_fp32 / rms

    # Apply weight (also cast to float32)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    weight_fp32 = weight.to(tl.float32)
    output_fp32 = x_normed * weight_fp32

    # Cast back to original dtype and store
    output = output_fp32.to(x.dtype)
    tl.store(output_ptr + row_idx * n_cols + col_offsets, output, mask=mask)

def rmsnorm_fwd(x, weight, output):
    """Wrapper for Triton kernel (destination-passing style)."""
    batch_size, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (batch_size,)

    _rmsnorm_fwd_kernel[grid](
        x, output, weight, 1e-6, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
