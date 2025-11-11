# agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

import config
from networks import ChessBackbone


class Agent(nn.Module):
    def __init__(self, action_size: int):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.backbone = ChessBackbone()
        self.policy_head = nn.Linear(self.backbone.out_dim, action_size)
        self.value_head = nn.Linear(self.backbone.out_dim, 1)
        self.q_head = nn.Linear(self.backbone.out_dim, action_size)

        self.optimizer = optim.Adam(self.parameters(), lr=config.LEARNING_RATE)
        self.gamma = config.GAMMA
        self.epsilon = config.DQL_EPSILON_START

    # -----------------------------------------------------------
    # Action selection
    # -----------------------------------------------------------
    def act(self, state, legal_mask, explore=True):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            z = self.backbone(state_t)
            logits = self.policy_head(z).squeeze(0)
            logits[legal_mask == 0] = -1e9
            probs = F.softmax(logits, dim=-1)
            q_values = self.q_head(z).squeeze(0)
            q_values[legal_mask == 0] = -1e9

        if explore and random.random() < self.epsilon:
            legal_indices = np.nonzero(legal_mask)[0]
            return int(np.random.choice(legal_indices)), float(probs.max().item())
        else:
            # combine: mostly policy; tie-break by Q
            mixed = probs + 0.1 * F.softmax(q_values, dim=-1)
            action = int(torch.argmax(mixed).item())
            return action, float(probs[action].item())

    # -----------------------------------------------------------
    # Losses
    # -----------------------------------------------------------
    def compute_losses(self, batch):
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # shared encodings
        z = self.backbone(states_t)
        z_next = self.backbone(next_states_t)

        # ---- DQL target ----
        q_values = self.q_head(z)
        q_selected = q_values.gather(1, actions_t.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_head(z_next)
            q_target = rewards_t + self.gamma * (1 - dones_t) * q_next.max(1).values
        dql_loss = F.mse_loss(q_selected, q_target)

        # ---- Policy gradient (advantage actor-critic) ----
        logits = self.policy_head(z)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        values = self.value_head(z).squeeze(1)
        with torch.no_grad():
            advantages = q_target - values
        pg_loss = -(log_probs.gather(1, actions_t.view(-1, 1)).squeeze(1) * advantages).mean()
        value_loss = F.mse_loss(values, q_target)
        entropy = -(probs * log_probs).sum(dim=1).mean()

        total_loss = dql_loss + pg_loss + 0.5 * value_loss - 0.01 * entropy
        return total_loss, dql_loss.item(), pg_loss.item()

    # -----------------------------------------------------------
    # Optimization step
    # -----------------------------------------------------------
    def update(self, batch):
        self.optimizer.zero_grad()
        total_loss, dql, pg = self.compute_losses(batch)
        total_loss.backward()
        self.optimizer.step()
        self.epsilon = max(config.DQL_EPSILON_END, self.epsilon * config.DQL_EPSILON_DECAY)
        return {"loss": float(total_loss.item()), "dql": dql, "pg": pg}
