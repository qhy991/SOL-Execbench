import torch

def generate_inputs(axes_and_scalars, device):
    M = axes_and_scalars['M']
    N = axes_and_scalars['N']
    K = axes_and_scalars['K']
    grad_output = torch.randn(M, N, dtype=torch.float32, device=device) / N ** 0.5
    x = torch.randn(M, K, dtype=torch.float32, device=device) / K ** 0.5
    weight = torch.randn(N, K, dtype=torch.float32, device=device) / K ** 0.5
    return {'grad_output': grad_output, 'x': x, 'weight': weight}

@torch.no_grad()
def run(grad_output, x, weight):
    grad_input = grad_output @ weight
    grad_weight = grad_output.mT @ x
    return grad_input, grad_weight
