import torch
import torch.nn as nn
import numpy as np


# x = torch.tensor([1.0, 2.0, 3.0])
# print(x.shape)
# print(x.dtype)

# print(x * 2)
# print(x ** 2)

# print(torch.sin(x))

# x_np = np.array([1.0, 2.0, 3.0])
# print(x_np.shape)

# x_torch = torch.from_numpy(x_np)
# print(x_torch.shape)

# x_back = x_torch.numpy()
# print(x_back.shape)

# x = torch.tensor(3.0, requires_grad=True)
# y = x ** 2
# y.backward()
# print(x.grad)

# x = torch.tensor(2.0, requires_grad=True)
# y = torch.sin(x ** 2)
# y.backward()
# print(x.grad)

layer = nn.Linear(1, 1)   
print(layer.weight)       
print(layer.bias)  

x = torch.tensor([[3.0]])
y = layer(x)
print(y)
print(y.shape)

x = torch.tensor([[1.0], [1.0], [2.0], [2.0]])
y = layer(x)
print(y)
print(y.shape)

