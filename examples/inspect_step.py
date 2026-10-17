"""Run: uv run python examples/inspect_step.py"""
import numpy as np
from engine.tensor import Tensor
from nn.module import Parameter
from optim.adam import Adam

# A tiny regression expression isolates the chain rule from classifier details.
x = Tensor([2.0])
w = Parameter([3.0], prunable=True)
b = Parameter([1.0])
optimizer = Adam([w, b], lr=0.01)
prediction = x * (w * w.mask) + b
error = prediction - 5.0
loss = (error * error).sum()
loss.backward()
print('Before update:', {'prediction': prediction.item(), 'loss': loss.item(),
                         'weight_gradient': w.grad.item(), 'bias_gradient': b.grad.item()})
optimizer.step()
print('After Adam:', {'weight': w.item(), 'bias': b.item()})
optimizer.set_mask(w, [0])
optimizer.zero_grad()
error = x * (w * w.mask) + b - 5.0
(error * error).sum().backward()
optimizer.step()
print('After pruning:', {'weight': w.item(), 'gradient': w.grad.item(),
                         'moment': optimizer.state[w]['m'].item(),
                         'age': optimizer.state[w]['age'].item()})
optimizer.set_mask(w, [1])
optimizer.zero_grad()
error = x * (w * w.mask) + b - 5.0
(error * error).sum().backward()
optimizer.step()
print('After revival:', {'weight': w.item(), 'age': optimizer.state[w]['age'].item()})
