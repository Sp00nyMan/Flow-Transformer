import torch
from model import *

input = torch.randn(1, 1, 8, 8)

model = Patchification(1, 2, 2)
out = model(input)
model.reverse(out)