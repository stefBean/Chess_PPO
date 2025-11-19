# ppo_torch.py

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam

from actor import Actor
from critic import Critic


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.masks = []  # legal action masks

    def clear(self):
        self.__init__()

    def add(self, state, action, logprob, reward, done, value, mask):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.masks.append(mask)


class PPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 10,
        minibatch_size: int = 64,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        self.optim_actor = Adam(self.actor.parameters(), lr=actor_lr)
        self.optim_critic = Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = RolloutBuffer()

    # --------------------------------------------------
    # Acting
    # --------------------------------------------------
    def select_action(self, state_np: np.ndarray, legal_mask_np: np.ndarray):
        # state_np: (state_dim,)
        # legal_mask_np: (action_dim,)
        state = torch.from_numpy(state_np).float().to(self.device).unsqueeze(0)
        legal_mask = torch.from_numpy(legal_mask_np).to(self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.actor(state)
            mask_tensor = (legal_mask == 0)
            masked_logits = logits.masked_fill(mask_tensor, -1e9)
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
            value = self.critic(state).squeeze(-1)

        return int(action.item()), float(logprob.item()), float(value.item())

    def store_transition(
        self,
        state_np: np.ndarray,
        action: int,
        logprob: float,
        reward: float,
        done: bool,
        value: float,
        legal_mask_np: np.ndarray,
    ):
        self.buffer.add(
            state_np, action, logprob, reward, done, value, legal_mask_np
        )

    # --------------------------------------------------
    # GAE(λ)
    # --------------------------------------------------
    def compute_gae(self, rewards, dones, values):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)

        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0.0

        # Bootstrap with an extra value=0
        values_ext = torch.cat([values, torch.tensor([0.0], device=self.device)], dim=0)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_ext[t + 1] * (1.0 - dones[t]) - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    # --------------------------------------------------
    # PPO update
    # --------------------------------------------------
    def update(self):
        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32, device=self.device)
        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        dones = np.array(self.buffer.dones, dtype=np.float32)
        values = np.array(self.buffer.values, dtype=np.float32)
        masks = torch.tensor(np.array(self.buffer.masks), dtype=torch.float32, device=self.device)

        advantages, returns = self.compute_gae(rewards, dones, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.size(0)
        indices = np.arange(dataset_size)

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_masks = masks[mb_idx]

                # Actor
                logits = self.actor(mb_states)
                mask_tensor = (mb_masks == 0)
                masked_logits = logits.masked_fill(mask_tensor, -1e9)
                dist = Categorical(logits=masked_logits)

                new_logprobs = dist.log_prob(mb_actions)
                ratio = torch.exp(new_logprobs - mb_old_logprobs)

                unclipped = ratio * mb_advantages
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                actor_loss = -torch.mean(torch.min(unclipped, clipped))

                # Critic
                values_pred = self.critic(mb_states).squeeze(-1)
                critic_loss = torch.mean((mb_returns - values_pred) ** 2)

                # Entropy bonus
                entropy = dist.entropy().mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optim_actor.zero_grad()
                self.optim_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optim_actor.step()
                self.optim_critic.step()

        self.buffer.clear()
