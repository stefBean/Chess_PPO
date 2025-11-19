import torch
import torch.nn as nn
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=(256, 256)):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        # state: (B, state_dim)
        return self.net(state)  # raw logits, no softmax

    def sample(self, state, legal_mask):
        """
        state:      (1, state_dim) tensor
        legal_mask: (1, action_dim) tensor with 0/1 entries
        """
        logits = self.forward(state)
        mask_tensor = (legal_mask == 0)
        logits = logits.masked_fill(mask_tensor, -1e9)

        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)

        return action.item(), logp, dist
