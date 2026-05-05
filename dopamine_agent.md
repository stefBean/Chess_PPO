# Dopamine Policy-Gradient Chess

This document explains the dopamine-inspired policy-gradient implementation in `dopamine_pg.py`, including the math,
and calls out the supporting files needed to assemble the full training and
inference pipeline. The project uses a chess endgame situation as environment, uses a dopamine-modulated
policy-gradient algorithm to train the neural network and to understand the training and quantification of the agent within and its decision process.

### Core Dopamine Agent Components (Code Map)

These files define the dopamine-inspired learning logic and the networks it trains:

- **`dopamine_pg.py`**: Policy-gradient algorithm, rollout buffer, GAE, entropy
  regularization, temperature control, and mood-modulated reward scaling.
- **`actor.py`**: Policy network that outputs logits over the action space.
- **`critic.py`**: Value network that estimates the state value \(V(s)\).

The dopamine pipeline plugs into the chess environment and encoders:

- **`board_environment.py`**: Gym-style chess environment, step/reset logic,
  termination detection, and rewards.
- **`board_encoding.py`**: Converts `chess.Board` to a tensor state
  representation (state vector).
- **`action_encoding.py`**: Maps `chess.Move` ↔ action indices for the policy.
- **`train.py`**: Drives self-play, collects rollouts, calls `update()`,
  manages curriculum sampling and opponents.
- **`play.py`**: Loads a trained policy for interactive play.
- **`reward_shaping.py`**: Adds shaped reward components used by the
  environment during non-terminal moves.

These files are the minimum set required to understand how `dopamine_pg.py` interacts
with the rest of the system during training and inference.


### Environment-Setup

The environment is easily implemented by installing `requirements.txt` and using Python 3.10 - 3.14.
After setup and installation of all requirements, training of the agent can be started.

It is best practice to compile all files before starting training with:

```bash
python -m compileall .
```

The training is done in steps with various opponents in order to pre-train the agent before training against itself.
Currentely, sections in `train.py` can be toggled to switch between agent modes and opponent stages.

Opponent implementations used by `train.py` in the given order:
1. **`opponent_random.py`**,
2. **`opponent_heuristic.py`**,
3. **`opponent_selfplay.py`**

Then run:
```bash
python3 train.py
```

After a full training process, or if using already pre-trained models from this project, you can run
```bash
python3 play.py
```

---
## 1. Policy and Value Networks

The actor network outputs logits for a categorical distribution over chess
actions, and the critic outputs a scalar value estimate:

$$
\pi_\theta(a \mid s) = \text{Categorical}(\text{logits}_\theta(s))
$$

$$
V_\phi(s) \approx \mathbb{E}[G_t \mid s_t = s]
$$

Unlike PPO clipping, `dopamine_pg.py` uses a direct policy-gradient objective
with entropy regularization and advantage weighting.

---
## 2. Action space & legality (chess-specific constraint)

Chess has **state-dependent legal moves**, but the policy outputs logits for a **fixed discrete action space** of size `encoder.ACT_DIM`.

### 2.1 Action encoding
- `action_encoding.py` defines `AlphaZeroActionEncoder` and the action id mapping.
- In `train.py`, legal chess moves are enumerated from `board.legal_moves`, encoded via `encoder.encode(mv, board)`, and placed into:
  - `idxs` (list of legal action ids)
  - `idx_to_move` (dict: action id → `chess.Move`)

### 2.2 Legal action mask
A binary vector `legal_mask_np` of shape `(action_dim,)` is built:

- `legal_mask_np[idxs] = 1.0`
- illegal actions remain 0

### 2.3 Masked logits sampling (in `dopamine_pg.py:select_action`)
Given actor logits $z(s)\in\mathbb{R}^{|\mathcal{A}|}$, illegal actions are suppressed:

- `masked_logits = logits.masked_fill(legal_mask == 0, -1e9)`
- legal indices are extracted (`legal_ids`), and a categorical distribution is built only over legal logits
- sample legal offset and map back to a global action id

This ensures:
$$
\pi_\theta(a\mid s)=0 \quad\text{for illegal } a
$$

Legality masking is essential in chess in order to turn a fixed-size action head into a valid
policy over a variable legal action set.

### 2.4 Temperature scaling
The policy uses a temperature $\tau$ to control exploration:

$$
z'_\theta(s) = \frac{z_\theta(s)}{\tau}
$$

In code:
- `logits = actor(state) / max(self.temperature, 1e-6)`
- then mask → categorical

Smaller $\tau$ makes the policy more peaked (more greedy); larger $\tau$ increases entropy.

---
## 3. Rollout Buffer

`DopamineRolloutBuffer` stores one entry per agent move:

- states $s_t$: flattened observation at time
- actions $a_t$: sampled action id
- log probabilities $\log \pi_{\theta_\text{old}}(a_t \mid s_t)$
- rewards $r_t$: dopamine-modulated reward attributed to that move
- done flags $d_t$: terminated / truncated used for GAE masking
- value estimates $V(s_t)$: before the step
- next values $V(s_{t+1})$: bootstrap value estimate
- legal action masks
- terminal: stored for analysis/diagnostics
- mood scales: per-transition stochastic multiplier sampled from smoothed mood dynamics

These are aggregated in `update()` for computing GAE and the policy/value losses.

### 3.1 Dopamine reward modulation
In `store_transition()`, each raw reward is transformed before entering the buffer:

$$
\tilde{r}_t = m_t \cdot r_t
$$

where mood scale $m_t$ is sampled from a Gaussian process with smoothing:

- `instant ~ Normal(mood_mean, mood_std)`
- `current_mood = (1 - mood_smoothing) * current_mood + mood_smoothing * instant`
- `mood_scale = max(0.05, current_mood)`

This introduces adaptive reward amplification/dampening analogous to dopamine-like variation.

---
## 4. Generalized Advantage Estimation (GAE)

### 4.1 Temporal-difference residuals
For each time step $t$:

$$
\delta_t = r_t + \gamma (1 - d_t) V(s_{t+1}) - V(s_t)
$$

with the convention that if terminal, the bootstrap term is masked out.

In `_compute_gae()`:

- `mask = 1 - dones[t]`
- `delta = rewards[t] + gamma * next_values[t] * mask - values[t]`

### 4.2 GAE recursion
The generalized advantage estimator is:

$$
A_t = \delta_t + \gamma \lambda (1 - d_t) A_{t+1}
$$

In code:

- iterate `t` backwards
- `gae = delta + gamma * lam * mask * gae`
- store `advantages[t] = gae`
- define returns:

$$
R_t = A_t + V(s_t)
$$

### 4.3 Reward scaling + clipping
Before computing GAE, rewards are transformed:

- `rewards_t = rewards_t * reward_scale`
- if configured: `rewards_t = clamp(rewards_t, -reward_clip, reward_clip)`

This stabilizes training under high-variance shaped rewards and mood-scaled transitions.

---
## 5. Optimization Objective (Non-PPO)

`update()` uses minibatch SGD over multiple epochs, but without PPO clipping.

### 5.1 Advantage normalization
Raw advantages are normalized:

$$
\hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + 10^{-8}}
$$

### 5.2 Policy loss
For sampled actions with new log-probabilities:

$$
L_\text{policy}(\theta) = -\mathbb{E}_t\left[\log \pi_\theta(a_t\mid s_t)\,\hat{A}_t\right]
$$

### 5.3 Entropy-regularized actor loss
$$
L_\text{actor} = L_\text{policy} - c_H\,\mathbb{E}_t[\mathcal{H}(\pi_\theta(\cdot\mid s_t))]
$$

where `c_H = entropy_coef`.

### 5.4 Critic loss
$$
L_\text{critic} = \text{MSE}(V_\phi(s_t), R_t)
$$

The actor and critic are optimized with separate Adam optimizers and gradient clipping:

- `clip_grad_norm_(actor.parameters(), max_grad_norm)`
- `clip_grad_norm_(critic.parameters(), max_grad_norm)`

---
## 6. Diagnostics and Logged Quantities

`update()` returns a diagnostics dictionary for monitoring:

- actor/critic losses
- entropy and normalized entropy
- approximate KL proxy `mean(old_logprob - new_logprob)`
- raw and normalized advantage statistics
- temperature
- updates run
- dopamine metrics: `mood_mean`, `mood_scale_mean`

These metrics can be logged in training loops to compare dopamine-modulated policy-gradient behavior versus PPO baselines.
