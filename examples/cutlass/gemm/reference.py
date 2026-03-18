import torch

@torch.no_grad()
def run(A, B):
    return torch.mm(A, B)