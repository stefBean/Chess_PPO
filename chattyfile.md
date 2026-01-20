# PPO Chess: Algorithm and Math Walkthrough

This document explains the PPO implementation in `ppo.py`, including the math,
and calls out the supporting files needed to assemble the full training and
inference pipeline.

## Core PPO Components (Code Map)

These files define the PPO learning logic and the networks it trains:

- **`ppo.py`**: PPO algorithm, rollout buffer, GAE, clipped objectives, entropy
  control, temperature control, and optimizer steps.
- **`actor.py`**: Policy network that outputs logits over the action space.
- **`critic.py`**: Value network that estimates the state value \(V(s)\).

The PPO pipeline plugs into the chess environment and encoders:

- **`board_environment.py`**: Gym-style chess environment, step/reset logic,
  termination detection, and rewards.
- **`board_encoding.py`**: Converts `chess.Board` to a tensor state
  representation (state vector).
- **`action_encoding.py`**: Maps `chess.Move` ↔ action indices for the policy.
- **`train.py`**: Drives self-play, collects rollouts, calls `PPO.update()`,
  manages curriculum sampling and opponents.
- **`play.py`**: Loads the PPO policy for interactive play.
- **`reward_shaping.py`**: Adds shaped reward components used by the
  environment during non-terminal moves.

These files are the minimum set required to understand how `ppo.py` interacts
with the rest of the system during training and inference.

## PPO: Key Ideas and Math (as implemented)

### 1. Policy and Value Networks

The actor network outputs logits for a categorical distribution over chess
actions, and the critic outputs a scalar value estimate:

\[
\pi_\theta(a \mid s) = \text{Categorical}(\text{logits}_\theta(s))
\]
\[
V_\phi(s) \approx \mathbb{E}[G_t \mid s_t = s]
\]

See the model definitions in `actor.py` and `critic.py`, and the policy/value
usage in `ppo.py`.

### 2. Rollout Buffer

`RolloutBuffer` stores per-step data:

- states \(s_t\)
- actions \(a_t\)
- log probabilities \(\log \pi_{\theta_\text{old}}(a_t \mid s_t)\)
- rewards \(r_t\)
- done flags \(d_t\)
- value estimates \(V(s_t)\)
- next values \(V(s_{t+1})\)
- legal action masks

These are aggregated in `PPO.update()` for computing GAE and the PPO loss.

### 3. Generalized Advantage Estimation (GAE)

`compute_gae()` implements GAE with reward scaling and optional clipping:

Reward preprocessing:
\[
\hat{r}_t = \text{clip}(r_t \cdot \text{reward\_scale}, \pm \text{reward\_clip})
\]

TD residual:
\[
\delta_t = \hat{r}_t + \gamma V(s_{t+1}) (1 - d_t) - V(s_t)
\]

GAE advantage:
\[
A_t = \delta_t + \gamma \lambda (1 - d_t) A_{t+1}
\]

Returns:
\[
R_t = A_t + V(s_t)
\]

Advantages are normalized before the policy update:
\[
\tilde{A}_t = \frac{A_t - \mu_A}{\sigma_A + 10^{-8}}
\]

### 4. PPO Clipped Surrogate Objective

The policy ratio:
\[
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}
           = \exp(\log \pi_\theta - \log \pi_{\theta_\text{old}})
\]

Clipped objective:
\[
L^\text{CLIP}(\theta) = \mathbb{E}\left[
\min\left(
r_t(\theta)\tilde{A}_t,
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\tilde{A}_t
\right)\right]
\]

In the code, this yields the actor loss:

\[
L_\text{actor} = -L^\text{CLIP}(\theta)
\]

### 5. Value Function Loss (with optional clipping)

When value clipping is enabled:

\[
V_\text{clip}(s_t) = V_\text{old}(s_t) +
\text{clip}(V_\phi(s_t) - V_\text{old}(s_t), -c, c)
\]

\[
L_\text{critic} = \mathbb{E}\left[
\max\left((R_t - V_\phi(s_t))^2, (R_t - V_\text{clip}(s_t))^2\right)
\right]
\]

If value clipping is disabled, the loss is the standard MSE:
\[
L_\text{critic} = \mathbb{E}[(R_t - V_\phi(s_t))^2]
\]

### 6. Entropy Bonus (with Legal-Move Normalization)

Entropy encourages exploration, but the legal move count varies by position.
`ppo.py` normalizes entropy by \(\log(\text{legal\_count})\):

\[
H(\pi) = -\sum_a \pi(a \mid s) \log \pi(a \mid s)
\]
\[
H_\text{norm} = \frac{H(\pi)}{\log(\text{legal\_count})}
\]

This normalized entropy is used in the combined loss:

\[
L = L_\text{actor} + c_v L_\text{critic} - c_H H_\text{norm}
\]

### 7. KL-Based Early Stopping

An approximate KL is computed in each minibatch:

\[
\widehat{KL} \approx \mathbb{E}\left[\log \pi_{\theta_\text{old}} - \log \pi_\theta\right]
\]

If it exceeds `1.5 * target_kl`, PPO stops early for the current update.

### 8. Dual Controller for Entropy and Temperature

The code uses a simple proportional controller to adjust both:

Error:
\[
e = H_\text{target} - H_\text{norm}
\]

Update:
$$
\[
c_H \leftarrow \text{clip}(c_H + \alpha e)
\]
\[
T \leftarrow \text{clip}(T + \beta e)
\]
$$
This keeps exploration near the target normalized entropy while also adapting
the temperature used to scale logits.

## How the PPO Loop Fits into Training

1. `train.py` collects rollouts via environment steps and opponent moves.
2. `ppo.py` stores transitions in `RolloutBuffer`.
3. `PPO.update()` computes GAE, normalizes advantages, and runs multiple
   epochs of minibatch PPO updates.
4. Actor/critic networks in `actor.py` and `critic.py` are optimized with Adam.
5. `action_encoding.py` provides the fixed action index mapping and legal masks.
6. `board_encoding.py` produces the input tensor state representation.

This is the end-to-end path from chess board to policy update.