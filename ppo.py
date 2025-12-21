# ppo_torch.py

import numpy as np
import torch
import torch.nn as nn
import os
from torch.distributions import Categorical
from torch.optim import Adam

from actor import Actor
from critic import Critic

## test
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.next_values = []
        self.masks = []  # legal action masks

    def clear(self):
        self.__init__()

    def add(self, state, action, logprob, reward, done, value, next_value, mask):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.next_values.append(next_value)
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
        entropy_coef: float = 0.02,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        value_clip: float = 0.2,
        target_kl: float = 0.02,
        reward_scale: float = 1.0,
        reward_clip: float | None = 2.0,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.value_clip = value_clip
        self.target_kl = target_kl
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip

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
        next_value: float,
        legal_mask_np: np.ndarray,
    ):
        self.buffer.add(
            state_np, action, logprob, reward, done, value, next_value, legal_mask_np
        )

    def evaluate_value(self, state_np: np.ndarray) -> float:
        state = torch.from_numpy(state_np).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(state).squeeze(-1)
        return float(value.item())

    # --------------------------------------------------
    # GAE(λ)
    # --------------------------------------------------
    def compute_gae(self, rewards, dones, values, next_values):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards = rewards * self.reward_scale
        if self.reward_clip is not None:
            rewards = torch.clamp(rewards, -self.reward_clip, self.reward_clip)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)
        next_values = torch.tensor(next_values, dtype=torch.float32, device=self.device)

        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0.0

        # Bootstrap with an extra value=0
        # values_ext = torch.cat([values, torch.tensor([0.0], device=self.device)], dim=0)

        for t in reversed(range(len(rewards))):
            # delta = rewards[t] + self.gamma * values_ext[t + 1] * (1.0 - dones[t]) - values_ext[t]
            # gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
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
        next_values = np.array(self.buffer.next_values, dtype=np.float32)
        masks = torch.tensor(np.array(self.buffer.masks), dtype=torch.float32, device=self.device)

        advantages, returns = self.compute_gae(rewards, dones, values, next_values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_values = torch.tensor(values, dtype=torch.float32, device=self.device)

        dataset_size = states.size(0)
        indices = np.arange(dataset_size)
        approx_kl = 0.0
        updates_run = 0

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
                mb_old_values = old_values[mb_idx]

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
                # critic_loss = torch.mean((mb_returns - values_pred) ** 2)
                value_pred_clipped = mb_old_values + torch.clamp(
                    values_pred - mb_old_values, -self.value_clip, self.value_clip
                )
                critic_loss = torch.mean(
                    torch.max(
                        (mb_returns - values_pred) ** 2,
                        (mb_returns - value_pred_clipped) ** 2,
                    )
                )

                # Entropy bonus
                entropy = dist.entropy().mean()
                # loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                loss = (
                        actor_loss
                        + self.value_coef * critic_loss
                        - self.entropy_coef * entropy
                )

                self.optim_actor.zero_grad()
                self.optim_critic.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                # nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optim_actor.step()
                self.optim_critic.step()

                approx_kl = torch.mean(mb_old_logprobs - new_logprobs).item()
                updates_run += 1
                if self.target_kl and approx_kl > 1.5 * self.target_kl:
                    break
            if self.target_kl and approx_kl > 1.5 * self.target_kl:
                break

        self.buffer.clear()
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "entropy_coef": float(self.entropy_coef),
            "approx_kl": approx_kl,
            "updates_run": updates_run,
        }

    # ==========================================================
    # Save & Load with auto-directory creation
    # ==========================================================
    def save(self, path_prefix: str = "models/ppo_chess"):
        directory = os.path.dirname(path_prefix)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"[INFO] Created directory: {directory}")

        actor_path = f"{path_prefix}_actor.pt"
        critic_path = f"{path_prefix}_critic.pt"

        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

        print(f"[INFO] Saved actor to  {actor_path}")
        print(f"[INFO] Saved critic to {critic_path}")

    def load(self, path_prefix: str = "models/ppo_chess", strict: bool = True):
        import os

        actor_path = f"{path_prefix}_actor.pt"
        critic_path = f"{path_prefix}_critic.pt"

        if not os.path.exists(actor_path) or not os.path.exists(critic_path):
            print(f"[WARN] No saved model found at: {path_prefix}_*.pt")
            return False

        self.actor.load_state_dict(torch.load(actor_path), strict=strict)
        self.critic.load_state_dict(torch.load(critic_path), strict=strict)

        print(f"[INFO] Loaded actor from  {actor_path}")
        print(f"[INFO] Loaded critic from {critic_path}")
        return True
