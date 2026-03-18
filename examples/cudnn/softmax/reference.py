import torch

@torch.no_grad()
def run(input):
    return torch.softmax(input, dim=-1)