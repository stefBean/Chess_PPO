import torch
import torch.nn as nn
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=(256, 256), use_cnn: bool = True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_cnn = use_cnn and state_dim % 64 == 0
        self.channels = state_dim // 64

        if self.use_cnn:
            self.backbone = nn.Sequential(
                nn.Conv2d(self.channels, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            head_input = 64 * 8 * 8
        else:
            head_input = state_dim
        layers = []
        prev = head_input # state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        # state: (B, state_dim)
        if self.use_cnn:
            batch = state.shape[0]
            x = state.view(batch, self.channels, 8, 8)
            x = self.backbone(x)
            x = x.flatten(start_dim=1)
        else:
            x = state
        return self.net(x)

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
