import torch
import cuda.tile as ct
from math import ceil

ConstInt = ct.Constant[int]


def swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid):
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M):
    bid = ct.bid(0)
    return swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid)


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def matmul_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
    for k in range(num_tiles_k):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad).astype(dtype)
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad).astype(dtype)
        accumulator = ct.mma(a, b, accumulator)
    accumulator = ct.astype(accumulator, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=accumulator)


@ct.kernel
def add_kernel(A, B, C, tm: ConstInt, tn: ConstInt):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    zero_pad = ct.PaddingMode.ZERO
    a = ct.load(A, index=(bidx, bidy), shape=(tm, tn), padding_mode=zero_pad)
    b = ct.load(B, index=(bidx, bidy), shape=(tm, tn), padding_mode=zero_pad)
    ct.store(C, index=(bidx, bidy), tile=(a + b))


@torch.no_grad()
def run(attn_output, residual, o_proj_weight):
    shape = attn_output.shape
    M = attn_output.numel() // attn_output.shape[-1]
    K, N = attn_output.shape[-1], o_proj_weight.shape[0]

    a_flat = attn_output.contiguous().view(M, K)
    res_flat = residual.contiguous().view(M, N)

    # GEMM via cuTile
    tm, tn, tk = 128, 256, 64
    projected = torch.empty((M, N), device=a_flat.device, dtype=a_flat.dtype)
    grid_x = ceil(M / tm)
    grid_y = ceil(N / tn)
    grid = (grid_x * grid_y, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, matmul_kernel,
             (a_flat, o_proj_weight.t().contiguous(), projected, tm, tn, tk))

    # Residual add via cuTile
    output = torch.empty_like(projected)
    add_tm, add_tn = 32, 256
    add_grid = (ceil(M / add_tm), ceil(N / add_tn), 1)
    ct.launch(torch.cuda.current_stream(), add_grid, add_kernel,
             (projected, res_flat, output, add_tm, add_tn))

    return output.view(shape)
