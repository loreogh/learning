import torch
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import MLP

# Load saved file
script_dir = os.path.dirname(os.path.abspath(__file__))
load_path = os.path.join(script_dir, 'heat_pinn.pth')

loaded_data = torch.load(load_path)

model = MLP(**loaded_data['model_params'])
model.load_state_dict(loaded_data['model_state_dict'])
model.eval()

alpha = loaded_data['physics_params']['alpha']

# Plot parameters
nx = 100
nt = 100

x_plot = torch.linspace(0, 1, nx)
t_plot = torch.linspace(0, 1, nt)
X, T = torch.meshgrid(x_plot, t_plot, indexing='ij')

xt_plot = torch.cat([X.reshape(-1,1), T.reshape(-1,1)], dim=1)

with torch.no_grad():
    u_pred = model(xt_plot).reshape(nx, nt)

u_anal = torch.exp(-alpha * torch.pi**2 * T) * torch.sin(torch.pi * X)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im0 = axes[0].contourf(X.numpy(), T.numpy(), u_pred.numpy(), levels=50, cmap='hot')
axes[0].set_title('PINN')
axes[0].set_xlabel('x')
axes[0].set_ylabel('t')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].contourf(X.numpy(), T.numpy(), u_anal.numpy(), levels=50, cmap='hot')
axes[1].set_title('Analytical')
axes[1].set_xlabel('x')
plt.colorbar(im1, ax=axes[1])

error = torch.abs(u_pred - u_anal)
im2 = axes[2].contourf(X.numpy(), T.numpy(), error.numpy(), levels=50, cmap='hot')
axes[2].set_title('Error')
axes[2].set_xlabel('x')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()

print(f"Max error: {error.max().item():.2e}")
print(f"Mean error: {error.mean().item():.2e}")