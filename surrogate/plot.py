import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import MLP

# Parameters
trained_name = 'surrogate.pth'
data_name = 'data.pt'


# Load
script_dir = os.path.dirname(os.path.abspath(__file__))
trained_data = torch.load(os.path.join(script_dir, trained_name))
data = torch.load(os.path.join(script_dir, data_name))

# Model
model = MLP(**trained_data['model_params'])
model.load_state_dict(trained_data['model_state_dict'])
model.eval()


# Loss curves 
train_losses = trained_data['train_losses']
test_losses = trained_data['test_losses']

plt.figure()
plt.semilogy(train_losses, label='train')
plt.semilogy(test_losses, label='test')
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.title('Surrogate training')
plt.legend()
plt.show()


# Parity curve
ratio = 0.2

X, u = data['X'], data['u']

n_test = int(ratio * len(X))
X_test = X[-n_test:]
u_test = u[-n_test:]

with torch.no_grad():
    u_pred = model(X_test)

plt.figure()
plt.plot([0, 1], [0, 1], 'r', label='perfect', zorder=1)
plt.scatter(u_test.numpy(), u_pred.numpy(), s=1, alpha=0.3, zorder=10)
plt.xlabel('True u')
plt.ylabel('Predicted u')
plt.title('Parity plot')
plt.legend()
plt.show()


# Point distribution
# plt.figure()
# plt.hist(u_test.numpy(), bins=50)
# plt.xlabel('u')
# plt.ylabel('count')
# plt.title('Distribution of u values')
# plt.show()


# Comparison at fixed alpha
nx, nt = 100, 100
alpha_test = 0.1

x_plot = torch.linspace(0, 1, nx)
t_plot = torch.linspace(0, 1, nt)
X_grid, T_grid = torch.meshgrid(x_plot, t_plot, indexing='ij')

alpha_grid = torch.full_like(X_grid, alpha_test)
xt_alpha = torch.cat([
    X_grid.reshape(-1, 1),
    T_grid.reshape(-1, 1),
    alpha_grid.reshape(-1, 1)
], dim=1)

X_mean = data['X_mean']
X_std = data['X_std']

xt_alpha_normalized = (xt_alpha - X_mean) / X_std

# with torch.no_grad():
#     u_pred = model(xt_alpha).reshape(nx, nt)
with torch.no_grad():
    u_pred = model(xt_alpha_normalized).reshape(nx, nt)

u_analytical = torch.exp(-alpha_test * torch.pi**2 * T_grid) * torch.sin(torch.pi * X_grid)

error = torch.abs(u_pred - u_analytical)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

t_slices = [0.1, 0.5, 0.9]
x_np = x_plot.numpy()

for ax, t_val in zip(axes, t_slices):
    # find closest index
    t_idx = int(t_val * (nt - 1))
    
    ax.plot(x_np, u_analytical[:, t_idx].numpy(), 'k--', label='Analytical')
    ax.plot(x_np, u_pred[:, t_idx].numpy(), 'r', label='Surrogate')
    ax.set_title(f't = {t_val}')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

plt.suptitle(f'α = {alpha_test}')
plt.tight_layout()
plt.show()


# Error plots
# which alpha values have the worst error?
n_test = int(0.2 * len(X))
X_test = X[-n_test:]
u_test = u[-n_test:]

with torch.no_grad():
    u_pred_test = model(X_test)

abs_error = torch.abs(u_pred_test - u_test).squeeze()

# X_test columns are normalized x, t, alpha
# denormalize alpha to see which values are problematic
alpha_test_vals = X_test[:, 2] * X_std[2] + X_mean[2]

plt.figure()
plt.scatter(alpha_test_vals.numpy(), abs_error.numpy(), s=1, alpha=0.3)
plt.xlabel('alpha')
plt.ylabel('absolute error')
plt.title('Error vs alpha')
plt.show()




print(f"Max error: {error.max().item():.2e}")
print(f"Mean error: {error.mean().item():.2e}")

print(f"Max value of u_analytical at t=0.9 : {u_analytical[:, t_idx].max()}")

print(f"Max value at t=0.9: {u_analytical[:, -10].max().item():.2e}")
print(f"Max absolute error at t=0.9: {error[:, -10].max().item():.2e}")