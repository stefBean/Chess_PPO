# ppo_torch.py
import numpy as np
import torch
import torch.nn as nn
import os
import math
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
        self.masks = []
        self.final_value = 0.0

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
        self.final_value = next_value


class PPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.13,
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
        entropy_target: float = 0.6,
        entropy_coef_lr: float = 0.005,
        entropy_coef_min: float = 0.002,
        entropy_coef_max: float = 0.08,
        temperature: float = 1.0,
        temperature_lr: float = 0.01,
        temperature_min: float = 0.7,
        temperature_max: float = 1.0,
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
        self.entropy_target = entropy_target
        self.entropy_coef_lr = entropy_coef_lr
        self.entropy_coef_min = entropy_coef_min
        self.entropy_coef_max = entropy_coef_max
        self.temperature = temperature
        self.temperature_lr = temperature_lr
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max

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
            logits = self.actor(state) / max(self.temperature, 1e-6)
            mask_tensor = legal_mask == 0
            masked_logits = logits.masked_fill(mask_tensor, -1e9)
            dist = Categorical(logits=masked_logits)
            top2 = torch.topk(dist.probs, k=2, dim=-1).values
            gap = (top2[:, 0] - top2[:, 1]).mean()
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
    def compute_gae(self, rewards, dones, values, next_values, bootstrap_value: float):
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
            next_val = bootstrap_value if t == len(rewards) - 1 else next_values[t]
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    # def compute_gae(self, rewards, dones, values, last_value: float):
    #     rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
    #     dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
    #     values = torch.tensor(values, dtype=torch.float32, device=self.device)
    #
    #     advantages = torch.zeros_like(rewards, device=self.device)
    #     gae = 0.0
    #
    #     # Bootstrap with critic(last_state) instead of 0
    #     last_v = torch.tensor([last_value], dtype=torch.float32, device=self.device)
    #     values_ext = torch.cat([values, last_v], dim=0)
    #
    #     for t in reversed(range(len(rewards))):
    #         delta = rewards[t] + self.gamma * values_ext[t + 1] * (1.0 - dones[t]) - values_ext[t]
    #         gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
    #         advantages[t] = gae
    #
    #     returns = advantages + values
    #     return advantages, returns

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
        values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)

        bootstrap_value = 0.0 if len(dones) == 0 or dones[-1] == 1.0 else float(next_values[-1])
        advantages_raw, returns = self.compute_gae(rewards, dones, values, next_values, bootstrap_value)
        advantages_raw_mean = float(advantages_raw.mean().item()) if advantages_raw.numel() else 0.0
        advantages_raw_std = float(advantages_raw.std().item()) if advantages_raw.numel() else 0.0
        advantages = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std() + 1e-8)
        advantages_norm_mean = float(advantages.mean().item()) if advantages.numel() else 0.0
        advantages_norm_std = float(advantages.std().item()) if advantages.numel() else 0.0

        dataset_size = states.size(0)
        indices = np.arange(dataset_size)
        approx_kl = 0.0
        updates_run = 0

        # Snapshot of the old policy to quantify policy shift (KL divergence)
        with torch.no_grad():
            old_logits = self.actor(states) / max(self.temperature, 1e-6)
            old_masked_logits = old_logits.masked_fill(masks == 0, -1e9)

        entropy_terms = []
        normalized_entropy_terms = []
        expected_adv_terms = []
        action_value_gaps = []
        policy_shift_kls = []
        policy_update_kl = 0.0
        actor_losses = []
        critic_losses = []
        approx_kls = []

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
                mb_values_old = values_tensor[mb_idx]

                # Actor
                logits = self.actor(mb_states) / max(self.temperature, 1e-6)
                mask_tensor = mb_masks == 0
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
                if self.value_clip is not None:
                    values_pred_clipped = mb_values_old + torch.clamp(
                        values_pred - mb_values_old, -self.value_clip, self.value_clip
                    )
                    critic_loss_unclipped = (mb_returns - values_pred) ** 2
                    critic_loss_clipped = (mb_returns - values_pred_clipped) ** 2
                    critic_loss = torch.mean(torch.max(critic_loss_unclipped, critic_loss_clipped))
                else:
                    critic_loss = torch.mean((mb_returns - values_pred) ** 2)

                # Entropy bonus
                entropy = dist.entropy()
                legal_count = mb_masks.sum(dim=-1).clamp(min=1.0)
                normalization = torch.clamp(torch.log(legal_count), min=1e-8)
                normalized_entropy = entropy / normalization
                normalized_entropy_mean = normalized_entropy.mean()
                entropy_mean = entropy.mean()
                # loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                loss = (
                        actor_loss
                        + self.value_coef * critic_loss
                        - self.entropy_coef * normalized_entropy_mean
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

                # Track decision-quality diagnostics
                entropy_terms.append(entropy_mean.detach())
                normalized_entropy_terms.append(normalized_entropy_mean.detach())
                # Expected advantage under the new policy (before clipping)
                mb_advantages_raw = advantages_raw[mb_idx]
                expected_adv_terms.append(torch.mean(ratio * mb_advantages_raw).detach())

                actor_losses.append(actor_loss.detach())
                critic_losses.append(critic_loss.detach())
                approx_kls.append(torch.tensor(approx_kl, device=self.device))

                # Action-value gap: difference between the most and second-most probable legal actions
                probs = dist.probs
                top2 = torch.topk(probs, k=2, dim=1).values
                gap_batch = top2[:, 0] - top2[:, 1]
                action_value_gaps.append(gap_batch.mean().detach())

                # KL divergence to previous policy snapshot (policy shift)
                old_dist = Categorical(logits=old_masked_logits[mb_idx])
                policy_shift_kls.append(torch.distributions.kl_divergence(dist, old_dist).mean().detach())

                updates_run += 1
                if self.target_kl and approx_kl > 1.5 * self.target_kl:
                    break
            if self.target_kl and approx_kl > 1.5 * self.target_kl:
                break

        entropy_mean_value = float(torch.stack(entropy_terms).mean().item()) if entropy_terms else 0.0
        normalized_entropy_value = (
            float(torch.stack(normalized_entropy_terms).mean().item()) if normalized_entropy_terms else 0.0
        )

        # Dual controller: adjust entropy coefficient and temperature toward the target normalized entropy.
        if normalized_entropy_terms and self.entropy_target is not None:
            entropy_error = self.entropy_target - normalized_entropy_value
            updated_coef = self.entropy_coef + self.entropy_coef_lr * entropy_error
            self.entropy_coef = float(np.clip(updated_coef, self.entropy_coef_min, self.entropy_coef_max))

            updated_temp = self.temperature + self.temperature_lr * entropy_error
            self.temperature = float(np.clip(updated_temp, self.temperature_min, self.temperature_max))

        if dataset_size > 0:
            with torch.no_grad():
                new_logits = self.actor(states) / max(self.temperature, 1e-6)
                new_masked_logits = new_logits.masked_fill(masks == 0, -1e9)
                new_dist = Categorical(logits=new_masked_logits)
                old_dist_full = Categorical(logits=old_masked_logits)
                policy_update_kl = float(
                    torch.distributions.kl_divergence(old_dist_full, new_dist).mean().item()
                )

        self.buffer.clear()
        return {
            "entropy": entropy_mean_value,
            "normalized_entropy": normalized_entropy_value,
            "entropy_coef": float(self.entropy_coef),
            "temperature": float(self.temperature),
            "updates_run": updates_run,
            "expected_advantage": float(torch.stack(expected_adv_terms).mean().item()) if expected_adv_terms else 0.0,
            "action_value_gap": float(torch.stack(action_value_gaps).mean().item()) if action_value_gaps else 0.0,
            "policy_shift_kl": float(torch.stack(policy_shift_kls).mean().item()) if policy_shift_kls else 0.0,
            "policy_update_kl": policy_update_kl,
            "actor_loss": float(torch.stack(actor_losses).mean().item()) if actor_losses else 0.0,
            "critic_loss": float(torch.stack(critic_losses).mean().item()) if critic_losses else 0.0,
            "approx_kl": float(torch.stack(approx_kls).mean().item()) if approx_kls else 0.0,
            "advantages_raw_mean": advantages_raw_mean,
            "advantages_raw_std": advantages_raw_std,
            "advantages_norm_mean": advantages_norm_mean,
            "advantages_norm_std": advantages_norm_std,
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
