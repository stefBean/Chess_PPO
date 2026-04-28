import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from actor import Actor
from critic import Critic


class DopamineRolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.next_values = []
        self.masks = []
        self.terminals = []
        self.mood_scales = []

    def clear(self):
        self.__init__()

    def add(
        self,
        state,
        action,
        logprob,
        reward,
        done,
        value,
        next_value,
        mask,
        terminal,
        mood_scale,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.next_values.append(next_value)
        self.masks.append(mask)
        self.terminals.append(terminal)
        self.mood_scales.append(mood_scale)


class DopaminePolicyGradient:
    """Non-PPO agent with mood-modulated rewards (dopamine-inspired)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        epochs: int = 4,
        minibatch_size: int = 128,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.6,
        reward_scale: float = 1.0,
        reward_clip: float | None = 2.5,
        temperature: float = 1.0,
        temperature_min: float = 0.7,
        temperature_max: float = 1.3,
        mood_mean: float = 1.0,
        mood_std: float = 0.3,
        mood_smoothing: float = 0.2,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max

        self.mood_mean = mood_mean
        self.mood_std = mood_std
        self.mood_smoothing = mood_smoothing
        self.current_mood = mood_mean
        self.entropy_target = 0.0
        self.entropy_coef_min = entropy_coef
        self.entropy_coef_max = entropy_coef

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.optim_actor = Adam(self.actor.parameters(), lr=actor_lr)
        self.optim_critic = Adam(self.critic.parameters(), lr=critic_lr)
        self.buffer = DopamineRolloutBuffer()

    def _sample_mood_scale(self) -> float:
        instant = float(np.random.normal(loc=self.mood_mean, scale=self.mood_std))
        self.current_mood = (
            (1.0 - self.mood_smoothing) * self.current_mood
            + self.mood_smoothing * instant
        )
        return max(0.05, self.current_mood)

    def select_action(self, state_np: np.ndarray, legal_mask_np: np.ndarray):
        state = torch.from_numpy(state_np).float().to(self.device).unsqueeze(0)
        legal_mask = torch.from_numpy(legal_mask_np).to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(state) / max(self.temperature, 1e-6)
            masked_logits = logits.masked_fill(legal_mask == 0, -1e9)
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
            value = self.critic(state).squeeze(-1)
        return int(action.item()), float(logprob.item()), float(value.item())

    def evaluate_value(self, state_np: np.ndarray) -> float:
        state = torch.from_numpy(state_np).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(state).squeeze(-1)
        return float(value.item())

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
        terminal: bool | None = None,
    ):
        mood_scale = self._sample_mood_scale()
        adjusted_reward = reward * mood_scale
        self.buffer.add(
            state_np,
            action,
            logprob,
            adjusted_reward,
            done,
            value,
            next_value,
            legal_mask_np,
            terminal,
            mood_scale,
        )

    def _compute_gae(self, rewards, dones, values, next_values):
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards_t = rewards_t * self.reward_scale
        if self.reward_clip is not None:
            rewards_t = torch.clamp(rewards_t, -self.reward_clip, self.reward_clip)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values, dtype=torch.float32, device=self.device)
        next_values_t = torch.tensor(next_values, dtype=torch.float32, device=self.device)

        advantages = torch.zeros_like(rewards_t)
        gae = 0.0
        for t in reversed(range(len(rewards_t))):
            mask = 1.0 - dones_t[t]
            delta = rewards_t[t] + self.gamma * next_values_t[t] * mask - values_t[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
        returns = advantages + values_t
        return advantages, returns

    def update(self):
        if not self.buffer.states:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "normalized_entropy": 0.0,
                "entropy_coef": self.entropy_coef,
                "approx_kl": 0.0,
                "policy_shift_kl": 0.0,
                "policy_update_kl": 0.0,
                "advantages_raw_mean": 0.0,
                "advantages_raw_std": 0.0,
                "advantages_norm_mean": 0.0,
                "advantages_norm_std": 0.0,
                "expected_advantage": 0.0,
                "action_value_gap": 0.0,
                "temperature": self.temperature,
                "updates_run": 0,
                "mood_mean": self.current_mood,
                "mood_scale_mean": 1.0,
            }

        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32, device=self.device)
        masks = torch.tensor(np.array(self.buffer.masks), dtype=torch.float32, device=self.device)
        values = np.array(self.buffer.values, dtype=np.float32)
        next_values = np.array(self.buffer.next_values, dtype=np.float32)
        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        dones = np.array(self.buffer.dones, dtype=np.float32)

        advantages_raw, returns = self._compute_gae(rewards, dones, values, next_values)
        advantages = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std() + 1e-8)

        n = states.size(0)
        indices = np.arange(n)
        actor_losses, critic_losses, entropies, approx_kls = [], [], [], []
        updates_run = 0

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                mb = indices[start:start + self.minibatch_size]
                mb_states = states[mb]
                mb_actions = actions[mb]
                mb_adv = advantages[mb]
                mb_returns = returns[mb]
                mb_masks = masks[mb]
                mb_old_logprobs = old_logprobs[mb]

                logits = self.actor(mb_states) / max(self.temperature, 1e-6)
                masked_logits = logits.masked_fill(mb_masks == 0, -1e9)
                dist = Categorical(logits=masked_logits)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                policy_loss = -(new_logprobs * mb_adv.detach()).mean()
                actor_loss = policy_loss - self.entropy_coef * entropy

                values_pred = self.critic(mb_states).squeeze(-1)
                critic_loss = F.mse_loss(values_pred, mb_returns.detach())

                self.optim_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optim_actor.step()

                self.optim_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optim_critic.step()

                approx_kl = (mb_old_logprobs - new_logprobs).mean().item()
                approx_kls.append(float(approx_kl))
                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy.item()))
                updates_run += 1

        mood_scale_mean = float(np.mean(self.buffer.mood_scales)) if self.buffer.mood_scales else float(self.current_mood)
        self.buffer.clear()
        entropy_mean = float(np.mean(entropies)) if entropies else 0.0
        normalized_entropy = entropy_mean / np.log(masks.shape[1])
        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": entropy_mean,
            "normalized_entropy": float(normalized_entropy),
            "entropy_coef": self.entropy_coef,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "policy_shift_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "policy_update_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "advantages_raw_mean": float(advantages_raw.mean().item()),
            "advantages_raw_std": float(advantages_raw.std().item()),
            "advantages_norm_mean": float(advantages.mean().item()),
            "advantages_norm_std": float(advantages.std().item()),
            "expected_advantage": float(advantages_raw.mean().item()),
            "action_value_gap": 0.0,
            "temperature": self.temperature,
            "updates_run": updates_run,
            "mood_mean": float(self.current_mood),
            "mood_scale_mean": mood_scale_mean,
        }

    def save(self, path_prefix: str):
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pt")

    def load(self, path_prefix: str, strict: bool = True):
        actor_path = f"{path_prefix}_actor.pt"
        critic_path = f"{path_prefix}_critic.pt"
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device), strict=strict)
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device), strict=strict)
