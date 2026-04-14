import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Number of samples for each parameter
n_samples = 10000
file_name = 'data.pt'

# Random sampling
x = torch.FloatTensor(n_samples, 1).uniform_(0, 1)
t = torch.FloatTensor(n_samples, 1).uniform_(0, 1)
alpha = torch.exp(torch.FloatTensor(n_samples, 1).uniform_(np.log(0.1), np.log(10)))

# Evaluate analytical solution ("simulator")
u = torch.exp(-alpha * torch.pi**2 * t) * torch.sin(torch.pi * x)

# Stack inputs and compute mean
X = torch.cat([x, t, alpha], dim=1)
X_mean = X.mean(dim=0)
X_std = X.std(dim=0)
X_normalized = (X - X_mean) / X_std

# Save
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, file_name)
torch.save({
    'X': X_normalized,
    'u': u,
    'X_mean': X_mean,
    'X_std': X_std
}, save_path)

# print(f"X shape: {X.shape}")
# print(f"u shape: {u.shape}")
# print(f"alpha range: [{alpha.min():.3f}, {alpha.max():.3f}]")
# print(f"u range: [{u.min():.3f}, {u.max():.3f}]")
# print(f"Data saved to {save_path}")