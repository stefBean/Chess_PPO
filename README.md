# Learning to Decide  
## Proximal Policy Optimization (PPO) in a Chess Endgame RL System  
*(Math → Code, fully nachvollziehbar)*

This document explains **what PPO computes**, **why**, and **exactly where and how this is implemented in your codebase**, with **verbatim code snippets** taken from the final files you provided.

The focus is strictly on:
- PPO mathematics
- credit assignment
- loss construction
- stability mechanisms  
No chess heuristics, no search, no minimax.

---

## 1. High-Level Execution Flow

At runtime, the system follows this deterministic control flow:


---

## 2. Rewards: What the Agent Optimizes

### 2.1 Terminal rewards (sparse, correct signal)

Defined in **`board_environment.py`**:

```bash
# board_environment.py
if board.is_checkmate():
    return 1.0 if board.turn == chess.BLACK else -1.0
elif board.is_stalemate() or board.is_insufficient_material():
    return 0.0
```
This defines the true objective:

* win → +1  
* loss → −1  
* draw → 0  

### 2.2 Reward shaping (dense learning signal)

Implemented in reward_shaping.py and applied inside env.step():

```bash
# reward_shaping.py
reward = 0.0

if move_is_capture:
    reward += 0.05

if gives_check:
    reward += 0.02

if repetition_detected:
    reward -= 0.05

# board_environment.py
reward = terminal_reward
reward += compute_shaping_reward(board_before, board_after, move)
```

#### Applied within the environment:

```bash
# board_environment.py
reward = terminal_reward
reward += compute_shaping_reward(board_before, board_after, move)
```

## 3. Action Space: Discrete but Always Legal
### 3.1 AlphaZero-style action encoding

Each chess move is mapped to a fixed action index.

```bash
def build_action_map(board):
    idx_to_move = {}
    legal_idxs = []

    for move in board.legal_moves:
        idx = action_encoder.encode(move)
        idx_to_move[idx] = move
        legal_idxs.append(idx)

    return legal_idxs, idx_to_move
```
### 3.2 Legal-action mask

Constructed in train.py:

```bash
mask = np.zeros(action_dim, dtype=np.float32)
mask[legal_idxs] = 1.0
```

This mask guarantees:
* illegal actions have zero probability
* the policy is normalized over legal moves only

## 4. Policy Inference (Actor + Masking)
### 4.1 Actor network (logits only)

```bash
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        return self.net(x)  # raw logits
```

### 4.2 Masked categorical distribution

Inside ppo.py → select_action():
```bash
logits = self.actor(state)
masked_logits = logits + (mask - 1) * 1e9
dist = Categorical(logits=masked_logits)

action = dist.sample()
logprob = dist.log_prob(action)
value = self.critic(state)
```

The policy assigns zero probability to illegal actions by modifying logits
before applying the softmax function.

Mathematically:
$$
\pi(a \mid s) = \mathrm{softmax}\left(\text{logits}_a - \infty \cdot \mathbb{1}_{\text{illegal}}(a)\right)
$$

## 5. Transition Storage (Rollout Buffer)

Each PPO step stores the following tuple:

$$
(s_t,\ a_t,\ \log \pi_{\text{old}}(a_t \mid s_t),\ r_t,\ d_t,\ V(s_t))
$$

Where:

- $s_t$ is the state at time step $t$
- $a_t$ is the sampled action
- $\log \pi_{\text{old}}(a_t \mid s_t)$ is the log-probability under the old policy
- $r_t$ is the reward
- $d_t$ is the terminal flag
- $V(s_t)$ is the critic value estimate


## 6. Advantage Estimation (GAE-λ)
### 6.1 Mathematical definition

$$
\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)
$$

The generalized advantage estimate is computed recursively:

$$
A_t = \delta_t + \gamma \lambda (1 - d_t) A_{t+1}
$$

The return target for the critic is:

$$
R_t = A_t + V(s_t)
$$

#### Implementation in ppo.py
```bash
advantages = torch.zeros(T)
gae = 0.0

for t in reversed(range(T)):
    next_value = 0 if t == T-1 else values[t+1]
    delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
    gae = delta + gamma * lam * (1 - dones[t]) * gae
    advantages[t] = gae

returns = advantages + values
```
### 6.2 Advantage normalization

Advantage normalization is essential in PPO to gain stability


## 7. PPO Objective: Ratio and Clipping
### 7.1 Probability ratio
#TODO explain

```bash
ratios = torch.exp(new_logprobs - old_logprobs)
```

### 7.2 Clipped surrogate loss
This enforces a trust region without second-order optimization.

$$
L_clip = E[min(r_t A_t, clip(r_t, 1−ε, 1+ε) A_t)]
$$

```bash

```

## 8. Critic Loss (Value Function Regression)

The critic approximates expected return:
$$
V(s) ≈ E[∑ γ^t r_t]
$$

```bash
values_pred = self.critic(states).squeeze()
critic_loss = F.mse_loss(values_pred, returns)
```

## 9. Entropy Bonus (Exploration Guarantee)

Entropy bonus is important to prevent:
* deterministic collapse
* repetition loops
* brittle policies

$$
H(π) = −Σ π(a|s) log π(a|s)
$$

```bash
entropy = dist.entropy().mean()
```

## 10. Final PPO Loss

$$
L = L_actor + 0.5 L_critic − 0.01 H
$$
```bash
loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

# Backpropagation:
self.optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
self.optimizer.step()

```

## 11. Self-Play and Non-Stationarity Control

The opponent is a frozen copy of the actor:

Refreshed periodically to approximate fictitious play while maintaining stability.