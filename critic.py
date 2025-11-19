import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_layers=(256, 256)):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        # state: (B, state_dim)
        return self.net(state)  # (B,1)
