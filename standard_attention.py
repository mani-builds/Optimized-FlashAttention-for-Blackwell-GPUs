#!/usr/bin/env python
import torch

N = 1024
d = 54

torch.manual_seed(123)

q = torch.rand(N, d, device=torch.device(0), dtype=torch.float32)
k = torch.rand(N, d, device=torch.device(0), dtype=torch.float32)
v = torch.rand(N, d, device=torch.device(0), dtype=torch.float32)

def fwd_layer(q, k ,v):
    s = torch.matmul(q, k.T)
    p = torch.softmax(s, 1)
    o = torch.matmul(p , v)

    return o

print(fwd_layer(q, k, v))
