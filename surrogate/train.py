import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split, DataLoader
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import MLP

# Parameters
train_ratio = 0.8
load_name = 'data.pt'
save_name = 'surrogate.pth'
batch_size = 256

scheduler_step = 500
scheduler_gamma = 0.5

n_epochs = int(3000)

n_in = 3
n_hidden = 32
n_layers = 4
n_out = 1

# Model
model = MLP(n_in, n_out, n_hidden, n_layers)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
data = torch.load(os.path.join(script_dir, load_name))
X, u = data['X'], data['u']

# Create dataset
dataset = TensorDataset(X, u)

# Split data into training and testing
n_train = int(train_ratio * len(dataset))
n_test = len(dataset) - n_train
train_set, test_set = random_split(dataset, [n_train, n_test])

# Divide sets in batches 
# use drop_last=True to avoid residual batches
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Built in MSE loss function
loss_fn = nn.MSELoss()

# Training loop 
train_losses = []
test_losses = []
for epoch in range(n_epochs):

    model.train()
    train_loss = 0

    for X_batch, u_batch in train_loader:

        optimizer.zero_grad()
        u_pred = model(X_batch)
        loss = loss_fn(u_pred, u_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    scheduler.step()

    model.eval()
    with torch.no_grad():
        test_loss = 0
        for X_batch, u_batch in test_loader:

            u_pred = model(X_batch)
            test_loss += loss_fn(u_pred, u_batch).item()

        test_loss /= len(test_loader)

    test_losses.append(test_loss)

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Train: {train_loss:.2e} | Test: {test_loss:.2e}")


# Saving 
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, save_name)

torch.save({
    'model_state_dict': model.state_dict(),
    'model_params': {
        'n_in': n_in,
        'n_out': n_out,
        'n_hidden': n_hidden,
        'n_layers': n_layers
    },
    'train_losses': train_losses,
    'test_losses': test_losses
}, save_path)
