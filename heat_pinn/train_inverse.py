import torch
from model import MLP
import os

# Network parameters
n_in = 2
n_out = 1
n_hidden = 32
n_layers = 4

n_cond = 100
n_pde = 1000

N_epoch = int(1e4)

alpha_guess = 0.5
alpha_real = 0.1

# Sampling
n_samp = 100
x_samp = torch.rand(n_samp, 1)
t_samp = torch.rand(n_samp, 1)
u_samp = torch.exp(-alpha_real * torch.pi**2 * t_samp) * torch.sin(torch.pi * x_samp)
xt_samp = torch.cat([x_samp, t_samp], dim=1)


# Loss parameters
lambda_ic = 10.0
lambda_bc = 10.0
lambda_data = 10.0


# Loss functions
def loss_pde(model, x, t, xt, alpha):
    
    u = model(xt)

    du_dt = torch.autograd.grad(u, t, 
            grad_outputs=torch.ones_like(u),
            create_graph=True)[0]
    
    du_dx = torch.autograd.grad(u, x, 
            grad_outputs=torch.ones_like(u),
            create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x, 
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True)[0]
    
    residual = du_dt - alpha * d2u_dx2

    return torch.mean(residual**2)

def loss_ic(model, n_points = n_cond):

    x = torch.rand(n_points, 1)
    t = torch.zeros(n_points, 1)
    xt = torch.cat([x, t], dim=1)

    u = model(xt)

    return torch.mean((u - torch.sin(torch.pi * x))**2)

def loss_bc1(model, n_points = n_cond):

    x = torch.zeros(n_points, 1)
    t = torch.rand(n_points, 1)
    xt = torch.cat([x, t], dim=1)

    u = model(xt)

    return torch.mean(u**2)

def loss_bc2(model, n_points = n_cond):

    x = torch.ones(n_points, 1)
    t = torch.rand(n_points, 1)
    xt = torch.cat([x, t], dim=1)

    u = model(xt)

    return torch.mean(u**2)


def loss_data(model, u_samp, xt_samp):

    u = model(xt_samp)

    return torch.mean((u - u_samp)**2)


# Model
model = MLP(n_in, n_out, n_hidden, n_layers)

# Random points
alpha = torch.nn.Parameter(torch.tensor([alpha_guess]))  

x = torch.rand(n_pde, 1).requires_grad_(True)
t = torch.rand(n_pde, 1).requires_grad_(True)
xt = torch.cat([x, t], dim=1)

# Optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': 1e-3},
    {'params': [alpha], 'lr': 1e-2}  # different lr for alpha
], )

# Optimization loop
for epoch in range(N_epoch):

    optimizer.zero_grad()
    loss_total = (loss_pde(model, x, t, xt, alpha) 
            + lambda_ic * loss_ic(model)
            + lambda_bc * loss_bc1(model)
            + lambda_bc * loss_bc2(model)
            + lambda_data * loss_data(model, u_samp, xt_samp))
    loss_total.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        l_pde = loss_pde(model, x, t, xt, alpha)
        l_ic = loss_ic(model)
        l_bc1 = loss_bc1(model)
        l_bc2 = loss_bc2(model)
        print(f"Epoch {epoch} | alpha: {alpha.item():.4f} | PDE: {l_pde.item():.2e} | IC: {l_ic.item():.2e} | BC1: {l_bc1.item():.2e} | BC2: {l_bc2.item():.2e}")


