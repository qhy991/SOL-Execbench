import torch
import torch.nn.functional as F

@torch.no_grad()
def run(x: torch.Tensor, gate_proj: torch.Tensor, up_proj: torch.Tensor) -> torch.Tensor:
    gate = F.linear(x, gate_proj)
    up = F.linear(x, up_proj)
    return F.gelu(gate, approximate='tanh') * up
