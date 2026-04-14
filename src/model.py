import torch
import torch.nn as nn


# Network definition
class MLP(nn.Module):
    def __init__(self, n_in, n_out, n_hidden, n_layers):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(n_in, n_hidden))
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
        layers.append(nn.Linear(n_hidden, n_out))

        self.layers = nn.ModuleList(layers)
        self.activation = nn.Tanh()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x) 
