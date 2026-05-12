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
        self.td_errors = []
        self.dopamine_deltas = []
        self.mood_scales = []
        self.mood_labels = []

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
        td_error,
        mood_scale,
        mood_label,
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
        self.td_errors.append(td_error)
        self.dopamine_deltas.append(td_error)
        self.mood_scales.append(mood_scale)
        self.mood_labels.append(mood_label)


class DopaminePolicyGradient:
    """TD-error actor-critic agent using dopamine-inspired reward prediction error."""

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
        entropy_coef: float = 0.03,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.6,
        reward_scale: float = 1.0,
        reward_clip: float | None = None,
        temperature: float = 1.0,
        temperature_min: float = 0.7,
        temperature_max: float = 1.3,
        mood_mean: float = 1.0,
        mood_std: float = 1.3,
        mood_smoothing: float = 0.2,
        mood_mode_prob: float = 0.4,
        use_mood_modulation: bool = True,
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
        self.mood_mode_prob = mood_mode_prob
        self.use_mood_modulation = use_mood_modulation
        self.current_mood = mood_mean
        self.current_mood_label = "neutral"
        self.entropy_target = 0.0
        self.entropy_coef_min = entropy_coef
        self.entropy_coef_max = entropy_coef

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.optim_actor = Adam(self.actor.parameters(), lr=actor_lr)
        self.optim_critic = Adam(self.critic.parameters(), lr=critic_lr)
        self.buffer = DopamineRolloutBuffer()

    def _sample_mood_scale(self) -> tuple[float, str]:
        if not self.use_mood_modulation:
            self.current_mood = self.mood_mean
            self.current_mood_label = "neutral"
            return 1.0, self.current_mood_label

        mode_roll = float(np.random.random())
        mode_band = max(0.0, min(0.5, self.mood_mode_prob))
        if mode_roll < mode_band:
            label = "pessimistic"
            loc = self.mood_mean - self.mood_std
        elif mode_roll > 1.0 - mode_band:
            label = "optimistic"
            loc = self.mood_mean + self.mood_std
        else:
            label = "neutral"
            loc = self.mood_mean

        instant = float(np.random.normal(loc=loc, scale=max(self.mood_std, 1e-6)))
        self.current_mood = (
            (1.0 - self.mood_smoothing) * self.current_mood
            + self.mood_smoothing * instant
        )
        self.current_mood_label = label
        return max(0.05, self.current_mood), label

    def select_action(self, state_np: np.ndarray, legal_mask_np: np.ndarray):
        state = torch.from_numpy(state_np).float().to(self.device).unsqueeze(0)
        legal_mask = torch.from_numpy(legal_mask_np).to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(state) / max(self.temperature, 1e-6)
            masked_logits = logits.masked_fill(legal_mask == 0, -1e9)
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            legal_ids = torch.nonzero(legal_mask[0] > 0, as_tuple=False).squeeze(-1)

            if legal_ids.numel() == 0:
                raise RuntimeError("No legal actions available for DopaminePolicyGradient")

            legal_logits = logits[0, legal_ids]
            legal_logits = torch.nan_to_num(legal_logits, nan=0.0, posinf=1e4, neginf=-1e4)
            dist = Categorical(logits=legal_logits)
            sampled_offset = dist.sample()
            action = legal_ids[sampled_offset]
            logprob = dist.log_prob(sampled_offset)
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
        reward_signal = float(reward) * self.reward_scale
        if self.reward_clip is not None:
            reward_signal = float(np.clip(reward_signal, -self.reward_clip, self.reward_clip))
        done_float = 1.0 if done else 0.0
        td_error = reward_signal + self.gamma * float(next_value) * (1.0 - done_float) - float(value)
        mood_scale, mood_label = self._sample_mood_scale()
        self.buffer.add(
            state_np,
            action,
            logprob,
            reward_signal,
            done,
            value,
            next_value,
            legal_mask_np,
            terminal,
            td_error,
            mood_scale,
            mood_label,
        )

    def _compute_td_lambda_advantages(self, td_errors, dones):
        deltas_t = torch.tensor(td_errors, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        advantages = torch.zeros_like(deltas_t)
        gae = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for t in reversed(range(len(deltas_t))):
            mask = 1.0 - dones_t[t]
            gae = deltas_t[t] + self.gamma * self.lam * mask * gae
            advantages[t] = gae
        return advantages

    def _evaluate_legal_batch(self, states, actions, masks):
        logits = self.actor(states) / max(self.temperature, 1e-6)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        logprobs = []
        entropies = []
        for i in range(states.size(0)):
            legal_ids = torch.nonzero(masks[i] > 0, as_tuple=False).squeeze(-1)
            if legal_ids.numel() == 0:
                raise RuntimeError("Transition has no legal actions in DopaminePolicyGradient.update")
            legal_logits = logits[i, legal_ids]
            dist = Categorical(logits=legal_logits)
            matches = torch.nonzero(legal_ids == actions[i], as_tuple=False).squeeze(-1)
            if matches.numel() == 0:
                raise RuntimeError(
                    f"Stored action {int(actions[i].item())} is outside its legal mask; "
                    f"legal ids={legal_ids.detach().cpu().tolist()}"
                )
            logprobs.append(dist.log_prob(matches[0]))
            entropies.append(dist.entropy())
        return torch.stack(logprobs), torch.stack(entropies)

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
                "td_error_mean": 0.0,
                "td_error_abs_mean": 0.0,
                "td_error_std": 0.0,
                "td_error_positive_rate": 0.0,
                "td_lambda_adv_mean": 0.0,
                "td_lambda_adv_std": 0.0,
                "actor_signal_mean": 0.0,
                "actor_signal_std": 0.0,
                "legal_entropy": 0.0,
                "mood_mean": self.current_mood,
                "mood_scale_mean": 1.0,
                "mood_optimistic_rate": 0.0,
                "mood_pessimistic_rate": 0.0,
            }

        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        masks = torch.tensor(np.array(self.buffer.masks), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32, device=self.device)
        next_values = torch.tensor(self.buffer.next_values, dtype=torch.float32, device=self.device)
        td_errors = np.array(self.buffer.td_errors, dtype=np.float32)
        td_errors_t = torch.tensor(td_errors, dtype=torch.float32, device=self.device)

        critic_targets = rewards + self.gamma * next_values * (1.0 - dones)
        td_lambda_adv = self._compute_td_lambda_advantages(td_errors, self.buffer.dones)
        dopamine_adv = (td_lambda_adv - td_lambda_adv.mean()) / (td_lambda_adv.std(unbiased=False) + 1e-8)

        mood_scales_t = torch.tensor(self.buffer.mood_scales, dtype=torch.float32, device=self.device)
        actor_signal = dopamine_adv * mood_scales_t if self.use_mood_modulation else dopamine_adv

        n = states.size(0)
        indices = np.arange(n)
        actor_losses, critic_losses, entropies = [], [], []
        updates_run = 0

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                mb = indices[start:start + self.minibatch_size]
                mb_states = states[mb]
                mb_actions = actions[mb]
                mb_actor_signal = actor_signal[mb]
                mb_targets = critic_targets[mb]
                mb_masks = masks[mb]

                new_logprobs, entropy_values = self._evaluate_legal_batch(mb_states, mb_actions, mb_masks)
                entropy = entropy_values.mean()
                policy_loss = -(new_logprobs * mb_actor_signal.detach()).mean()
                actor_loss = policy_loss - self.entropy_coef * entropy

                values_pred = self.critic(mb_states).squeeze(-1)
                critic_loss = F.mse_loss(values_pred, mb_targets.detach())

                self.optim_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optim_actor.step()

                self.optim_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optim_critic.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy.item()))
                updates_run += 1

        mood_scale_mean = float(np.mean(self.buffer.mood_scales)) if self.buffer.mood_scales else float(self.current_mood)
        mood_labels = list(self.buffer.mood_labels)
        mood_optimistic_rate = float(np.mean([label == "optimistic" for label in mood_labels])) if mood_labels else 0.0
        mood_pessimistic_rate = float(np.mean([label == "pessimistic" for label in mood_labels])) if mood_labels else 0.0
        self.buffer.clear()
        entropy_mean = float(np.mean(entropies)) if entropies else 0.0
        legal_counts = masks.sum(dim=1).clamp_min(2.0)
        normalized_entropy = float((entropy_mean / torch.log(legal_counts).mean()).item()) if n else 0.0
        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": entropy_mean,
            "normalized_entropy": normalized_entropy,
            "entropy_coef": self.entropy_coef,
            "approx_kl": 0.0,
            "policy_shift_kl": 0.0,
            "policy_update_kl": 0.0,
            "advantages_raw_mean": float(td_lambda_adv.mean().item()),
            "advantages_raw_std": float(td_lambda_adv.std(unbiased=False).item()),
            "advantages_norm_mean": float(dopamine_adv.mean().item()),
            "advantages_norm_std": float(dopamine_adv.std(unbiased=False).item()),
            "expected_advantage": float(td_errors_t.mean().item()),
            "action_value_gap": 0.0,
            "temperature": self.temperature,
            "updates_run": updates_run,
            "td_error_mean": float(td_errors_t.mean().item()),
            "td_error_abs_mean": float(td_errors_t.abs().mean().item()),
            "td_error_std": float(td_errors_t.std(unbiased=False).item()),
            "td_error_positive_rate": float((td_errors_t > 0).float().mean().item()),
            "td_lambda_adv_mean": float(td_lambda_adv.mean().item()),
            "td_lambda_adv_std": float(td_lambda_adv.std(unbiased=False).item()),
            "actor_signal_mean": float(actor_signal.mean().item()),
            "actor_signal_std": float(actor_signal.std(unbiased=False).item()),
            "legal_entropy": entropy_mean,
            "mood_mean": float(self.current_mood),
            "mood_scale_mean": mood_scale_mean,
            "mood_optimistic_rate": mood_optimistic_rate,
            "mood_pessimistic_rate": mood_pessimistic_rate,
            "use_mood_modulation": self.use_mood_modulation,
        }

    def save(self, path_prefix: str):
        directory = os.path.dirname(path_prefix)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pt")

    def load(self, path_prefix: str, strict: bool = True):
        actor_path = f"{path_prefix}_actor.pt"
        critic_path = f"{path_prefix}_critic.pt"
        if not (os.path.exists(actor_path) and os.path.exists(critic_path)):
            print(f"[WARN] No saved actor/critic checkpoint found at: {path_prefix}_*.pt")
            return False
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device), strict=strict)
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device), strict=strict)
        print(f"[INFO] Loaded dopamine actor from  {actor_path}")
        print(f"[INFO] Loaded dopamine critic from {critic_path}")
        return True
