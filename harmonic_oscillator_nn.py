import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# Neural network parameters
neurons = 16
N_epoch = 10000

# Parameters
t_final = 2*torch.pi
Nt = 100
lambda_ic = 1.0
omega = 1.0

x0 = 1.0
v0 = 0.0


# Model definition  
class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1,neurons)
        self.layer2 = nn.Linear(neurons,neurons)
        self.layer3 = nn.Linear(neurons,neurons)
        self.layer4 = nn.Linear(neurons,1)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.layer4(x)               # no activation on output layer
        return x


model = MLP()
 
# Loss function on the ODE value
def loss_ode(model, t, omega):

    x = model(t)  

    dx_dt = torch.autograd.grad(x, t, 
            grad_outputs=torch.ones_like(x),
            create_graph=True)[0]               
    d2x_dt2 = torch.autograd.grad(dx_dt, t,
            grad_outputs=torch.ones_like(dx_dt),
            create_graph=True)[0]     

    residual = d2x_dt2 + omega**2 * x

    # the loss function must return a scalar, but residual is a vector
    # hence I return the mean of residual^2.
    return torch.mean(residual**2)

# Loss value on the initial conditions
def loss_ic(model,x0_true,v0_true):
    
    t0 = torch.tensor([[0.0]], requires_grad=True)
    x0 = model(t0)

    dx0_dt0 = torch.autograd.grad(x0, t0, 
            grad_outputs=torch.ones_like(x0),
            create_graph=True)[0]
    
    residual = (x0 - x0_true)**2 + (dx0_dt0 - v0_true)**2

    return residual

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

# Time vector
t = torch.linspace(0, t_final, Nt).unsqueeze(1).requires_grad_(True)

# Optimization loop
for epoch in range(N_epoch):

    optimizer.zero_grad()    
    loss_total = loss_ode(model, t, omega) + lambda_ic * loss_ic(model, x0, v0)
    loss_total.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 500 == 0:
        l_ode = loss_ode(model, t, omega)
        l_ic = loss_ic(model, x0, v0)
        print(f"Epoch {epoch} | ODE: {l_ode.item():.2e} | IC: {l_ic.item():.2e} | Total: {loss_total.item():.2e}")


with torch.no_grad():
    t_plot = torch.linspace(0, t_final, 1000).unsqueeze(1)
    x_plot = model(t_plot)

t_np = t_plot.numpy()
x_np = x_plot.numpy()
x_anal = np.cos(omega * t_np)

plt.figure()
plt.plot(t_np, x_np, label='PINN')
plt.plot(t_np, x_anal, '--', label='Analytical')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('PINN vs Analytical solution')
plt.legend()
plt.show()