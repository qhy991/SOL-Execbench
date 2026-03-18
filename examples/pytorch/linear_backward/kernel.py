import torch

@torch.no_grad()
def run(grad_output, x, weight):
    grad_input = grad_output @ weight
    grad_weight = grad_output.mT @ x
    return grad_input, grad_weight
