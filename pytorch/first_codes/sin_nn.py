import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# This is an actual neural network composed by different layerss
# and it will be applied to a tensor similarly to the single layer
class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1,16)
        self.layer2 = nn.Linear(16,16)
        self.layer3 = nn.Linear(16,1)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)               # no activation on output layer
        return x

# model definition  
model = MLP()

# Shows architecture of the neural network
# print(model)                      
# for p in model.parameters():
#     print(p.shape) 

# Counts the number of total parameters
# total = sum(p.numel() for p in model.parameters())
# print(f"Total parameters: {total}")

# Training data
x_train = torch.linspace(-torch.pi, torch.pi, 100).unsqueeze(1)
y_train = torch.sin(x_train)

# Loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Optimization loop
for epoch in range(5000):

    # reset gradient
    optimizer.zero_grad()
    
    # predict value
    y_pred = model(x_train)

    # compute loss
    loss = loss_fn(y_pred, y_train)

    # compute gradient
    loss.backward()

    # update parameters
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.2e}")


x_test = torch.linspace(-torch.pi, torch.pi, 1000).unsqueeze(1)
with torch.no_grad():
    y_test = model(x_test)

plt.figure()
plt.plot(x_test.numpy(), y_test.numpy(), label='MLP')
plt.plot(x_test.numpy(), torch.sin(x_test).numpy(), '--', label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('MLP vs sin(x)')
plt.legend()
plt.show()
