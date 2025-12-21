import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_layers=(256, 256), use_cnn: bool = True):
        super().__init__()
        self.state_dim = state_dim
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
        prev = head_input
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
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
        return self.net(x)  # (B,1)
