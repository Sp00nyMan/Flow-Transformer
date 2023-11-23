import torch
from model import *

input = torch.randn(1, 1, 8, 8)

model = FlowFormer(1,8, 2)
out = model(input)[0]
model.reverse(out)