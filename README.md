# PPO Chess

This document explains the PPO implementation in `ppo.py`, including the math,
and calls out the supporting files needed to assemble the full training and
inference pipeline. The project uses a chess endgame situation as environment, use the ppo algorithm 
to train the nn and to understand the training and quantification of the agent within and its decision process.

### Core PPO Components (Code Map)

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


### Environment-Setup

The environment is easily implemented by installing the `requirements.txt`and using Python 3.10 - 3.14
After the setup and installation of all requirements, the training of the agent can be started-

It is best practise to compile all files before starting the training with:

```bash
python -m compileall .  
```
The training is done in steps with various opponents in order to pre-train the agent before training against itself.
Currentely the files have been commented out within the `train.py` and need to be commented in again for the correct training. Notes are given within the file onto where to comment / uncomment sections.

Opponent implementations used by `train.py` in the given order:
1. **`opponent_random.py`**, 
2. **`opponent_heuristic.py`**, 
3. **`opponent_selfplay.py`**

Then run: 
```bash
python3 train.py
```

After a full training process, or if using the already pre-trained models from this project, you can run 
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

PPO replaces this with a **clipped surrogate** to prevent overly large policy updates.

Define the probability ratio

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)} 
{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} = \exp\left(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t)\right)
$$

Then the PPO surrogate is:

$$
L^\text{CLIP}_t(\theta) =
\min\left(r_t(\theta)A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,A_t\right)
$$

In `ppo.py:update()` this is computed via:

- `ratio = exp(new_logprobs - old_logprobs)`
- `unclipped = ratio * advantages`
- `clipped = clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages`
- `actor_loss = -mean(min(unclipped, clipped))`

So the code minimizes `actor_loss`, i.e. maximizes the surrogate.

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

### 2.3 Masked logits sampling (in `ppo.py:select_action`)
Given actor logits $z(s)\in\mathbb{R}^{|\mathcal{A}|}$, illegal actions are suppressed:

- `masked_logits = logits.masked_fill(legal_mask == 0, -1e9)`
- `dist = Categorical(logits=masked_logits)`
- sample `action = dist.sample()`

This ensures:
$$
\pi_\theta(a\mid s)=0 \quad\text{for illegal } a
$$

Legality masking is essential in chess in order to turn a fixed-size action head into a valid 
policy over a variable legal action set. Furthermore it prevents illegal-move probability mass, which could lead to poison learning - so the advantages would be assigned to impossible actions.
It makes the entropy itself meaningful by computing only over legal support. 

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

`RolloutBuffer` stores one entry per agent move:

- states $s_t$: flattened obersation at time
- actions $a_t$: sampled action id
- log probabilities $\log \pi_{\theta_\text{old}}(a_t \mid s_t)$
- rewards $r_t$: reward attributed to that move
- done flags $d_t$: terminated / truncated used for GAE masking
- value estimates $V(s_t)$: before the step
- next values $V(s_{t+1})$: bootstrap value estimate
- legal action masks
- terminal: Stored for analysis/diagnostics and to enable future bootstrapping-through-truncations

These are aggregated in `PPO.update()` for computing GAE and the PPO loss.

### 3.1 Transition collection in `train.py`
The flow per agent ply is:

1. build `state_np = flatten_obs(obs)`
2. build `legal_mask_np`
3. sample `(action_id, logprob, value) = agent.select_action(state_np, legal_mask_np)`
4. execute `obs_next, reward_agent, terminated, truncated, info = env.step(move)`
5. compute `terminal = terminated`
6. compute bootstrap:
   - `next_value = 0` if terminal
   - else `next_value = agent.evaluate_value(next_state_np)`
7. store:
   - `agent.store_transition(combined_reward, terminal, value, next_value, legal_mask_np)`

**Important:** the code passes `terminal` into the `done`/`dones` field.  
Therefore in GAE and bootstrapping, “done” means **terminal**.

---
## 4. Generalized Advantage Estimation (GAE)

### 4.1 Temporal-difference residuals
For each time step $t$:

$$
\delta_t = r_t + \gamma (1 - d_t) V(s_{t+1}) - V(s_t)
$$

with the convention that if terminal, the bootstrap term is masked out.

In `compute_gae()` implements GAE with reward scaling and optional clipping:

- `mask = 1 - dones[t]` (so terminal → mask 0)
- `next_val = bootstrap_value if last step else next_values[t]`
- `delta = reward[t] + gamma * next_val * mask - values[t]`

### 4.2 GAE recursion
The generalized advantage estimator is:

$$
A_t = \delta_t + \gamma \lambda (1 - d_t) A_{t+1}
$$


In code:

- iterate `t` backwards
- `gae = delta + gamma * gae_lambda * mask * gae`
- store `advantages[t] = gae`
- define returns:

$$
R_t = A_t + V(s_t)
$$

### 4.3 Reward scaling + clipping
Before computing GAE, rewards are transformed:

- `rewards = rewards * reward_scale`
- optionally `clamp(rewards, -reward_clip, +reward_clip)`

This changes the learning signal magnitude and can stabilize training when reward spikes exist.

$$
r_t \leftarrow c \cdot r_t
$$

$$
r_t \leftarrow \text{clip}(r_t, -r_{\max}, r_{\max})
$$

### 4.4 Advantage normalization
Inside `update()`:

- `advantages_raw` from GAE are logged (mean/std)
- training uses normalized advantages:
$$
\hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + 10^{-8}}
$$

Advantage normalization is important to keep the PPO's clipping scale sensitive and behave consistently. 
If advantages drift in magnitude, the same clip epsilon will behaive differently. Normalization makes the update regime more consistent.
---

## 5. PPO Clipped Surrogate Objective


The policy ratio:
$$
r_t(\theta)
= \frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
= \exp\left(
\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t)
\right)
$$

Clipped objective:
$$
L_t^{\text{CLIP}}(\theta)
=
\min\left(
r_t(\theta) A_t,\;
\text{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right) A_t
\right)
$$


In the code, this yields the actor loss:

$$
\mathcal L_{\text{actor}}
= -\mathbb{E}_t \left[L_t^{\text{CLIP}}(\theta)\right]
$$

### 5.1 Data tensors
`update()` stacks the buffer into tensors:

- `states`: shape `(T, state_dim)`
- `actions`: shape `(T,)`
- `old_logprobs`: shape `(T,)`
- `masks`: shape `(T, action_dim)`
- `returns`, `advantages`: shape `(T,)`

### 5.2 Multi-epoch minibatch SGD
For `epochs` passes:
- shuffle indices
- for each minibatch:
  - compute new policy distribution (masked)
  - compute `actor_loss`, `critic_loss`, `entropy bonus`
  - optimize actor & critic (two optimizers, one backward pass)

---

## 6. Value Function Loss (with optional clipping)

The total loss (per minibatch) in code, when value clipping is enabled:

$$
\mathcal L
= \mathcal L_{\text{actor}} + c_V \mathcal L_{\text{critic}} - c_E \mathbb{E}_t \left[ H_{\text{norm}}(s_t) \right]
$$

where:
- $c_v$ = `value_coef`
- $c_e$ = `entropy_coef`
- $H_\text{norm}$ is **entropy normalized by legal action count** (details below)


If value clipping is disabled, the loss is the standard MSE:
$$
L_\text{critic} = \mathbb{E}[(R_t - V_\phi(s_t))^2]
$$


### 6.1 Critic loss with value clipping
PPO sometimes clips value updates similarly to policy updates. The code implements:

- `values_pred` = $V_\phi(s)$
- `values_pred_clipped` = $V_\text{old} + clamp(values_pred - V_\text{old}, -value_clip, +value_clip)$
- compare squared errors:
  - unclipped: $(R - V)^2$
  - clipped: $(R - V_\text{clipped})^2$
- take the max, then mean:

$$
\mathcal{L}_V = \mathbb{E}\left[\max\left((R_t - V_\phi)^2,\; (R_t - V_\text{clip})^2\right)\right]
$$

This is exactly what `ppo.py` does when `value_clip` is not `None`.

Clipping value updates helps the critic to avoid jumping too far, which could destabilize advantages and therefore also destabilize the actors update.
It is particularly relevant in sparse / spiky reward settings, which endgames tend to have in their terminal outcome.

### 6.2 Entropy bonus with legality normalization
Raw entropy of a categorical policy depends on the number of available actions. In chess, legal move counts vary widely, so the code normalizes:

- `entropy = dist.entropy()`  
- `legal_count = mb_masks.sum(dim=-1).clamp(min=1.0)`
- `normalization = log(legal_count)`  
- `normalized_entropy = entropy / normalization`

This attempts to measure “how close to uniform over legal moves” the policy is.  
If the policy were uniform over $n$ legal moves, entropy is $\log n$, so:

$$
H_\text{norm} \approx \frac{\log n}{\log n} = 1
$$

Raw entropy is not comparable across positions itself, because the legal move counts vary widley.
There $\log(n_\text{legal})$ divides to create a rough percent of maximum entropy over the legal moves.

---

## 7. KL-divergence usage

### 7.1 Approximate KL for early stopping (classic PPO trick)
Per minibatch:

- `approx_kl = mean(old_logprobs - new_logprobs)`

This is a common approximation to:
$$
D_{KL}(\pi_{\theta_\text{old}} \;\|\; \pi_\theta)
$$

If `approx_kl > 1.5 * target_kl`, the code breaks out early.

KL early stopping is a safety valve and acts as a practical trust-region guard. It prevents catastrophic policy shifts if one batch produces high-gradient updates.


### 7.2 Policy shift vs policy update KL (diagnostics)
The code also snapshots the **pre-update policy** across the entire dataset:

- `old_logits = actor(states) / temperature`
- `old_masked_logits = old_logits.masked_fill(masks == 0, -1e9)`

Then during minibatches:
- `policy_shift_kls.append( KL(dist_new || dist_old_snapshot).mean() )`

After all updates:
- `policy_update_kl = KL(old_dist_full, new_dist_full).mean()`

These are both valid “policy movement” signals, but note the direction:
- `policy_shift_kls` uses `kl_divergence(dist, old_dist)` (new vs old) per minibatch
- `policy_update_kl` uses `kl_divergence(old_dist_full, new_dist)` (old vs new) on full dataset

When interpreting logs, be consistent about KL direction.

---

## 8. Dual Controller for Entropy and Temperature

The code uses a simple proportional controller to adjust both:

- `entropy_coef` (how strongly entropy is rewarded)
- `temperature` (how stochastic the policy is)

Based on the error:

$$
e = H_\text{target} - H_\text{norm}
$$

Code:

- `entropy_error = entropy_target - normalized_entropy_value`
- `entropy_coef ← clip(entropy_coef + entropy_coef_lr * entropy_error, [min,max])`
- `temperature ← clip(temperature + temperature_lr * entropy_error, [min,max])`

So:
- if normalized entropy is **below** target → increase coef & temperature (more exploration)
- if above target → decrease them (more exploitation)

In `train.py`, the target is scheduled from a higher start to a lower end, pushing the agent from exploration → exploitation over training.

This keeps exploration near the target normalized entropy while also adapting
the temperature used to scale logits. 

In general it can be noticed that the temperature changes the sampling distribution of the sharpness directly, while the entropy coefficient shapes the optimization objective. 
Controlling both toward a normalized entropy target gives stable exploration scheduling in an environment where the action set size varies per state.

---

## 9. Logged decision-quality diagnostics (what they mean in THIS code)

`PPO.update()` returns a dict of metrics:

- `entropy` / `normalized_entropy`
- `entropy_coef`, `temperature`
- `actor_loss`, `critic_loss`
- `approx_kl`, `policy_shift_kl`, `policy_update_kl`
- `expected_advantage`
- `probability_gap`
- `advantages_*` stats

### 9.1 Expected advantage (as implemented)
The code logs:

- `expected_advantage = mean(ratio * advantages_raw)`

Interpretation:
- If the new policy puts more mass on “good” actions (positive advantages) and less on “bad” ones, this term tends to increase.
- It is **not** an expectation over all actions (it’s computed on sampled actions only, scaled by the ratio).

### 9.2 Advantage Estimator
The code computes:

- `probs = dist.probs`
- `top2 = topk(probs, k=2)`
- `gap = top2[:,0] - top2[:,1]`

This is strictly a **probability gap** between the most- and second-most probable **legal** actions under $\pi_\theta$:

$$
\text{gap}_\pi(s) = \max_a \pi(a\mid s) - \max_{a\neq a^*}\pi(a\mid s)
$$

---

## 10. Neural network details (Actor/Critic)

Both actor and critic support two modes:

### 10.1 CNN mode (enabled when `state_dim % 64 == 0`)
- Interpret state as `(channels, 8, 8)`
- 3 conv layers: `Conv2d(channels→64)`, then `64→64`, `64→64`, each with ReLU
- Flatten to `64*8*8`
- MLP head with `hidden_layers=(256,256)` by default

### 10.2 MLP mode (fallback)
- Direct MLP from state vector

### 10.3 Output heads
- Actor outputs logits of size `action_dim`
- Critic outputs scalar value `V(s)`

---

## 11. How the PPO Loop Fits into Training

1. `train.py` collects rollouts via environment steps and opponent moves.
2. `ppo.py` stores transitions in `RolloutBuffer`.
3. `PPO.update()` computes GAE, normalizes advantages, and runs multiple
   epochs of minibatch PPO updates.
4. Actor/critic networks in `actor.py` and `critic.py` are optimized with Adam.
5. `action_encoding.py` provides the fixed action index mapping and legal masks.
6. `board_encoding.py` produces the input tensor state representation.

This is the end-to-end path from chess board to policy update.

---

## 12. Common pitfalls and how the code addresses them

### 12.1 Mask/encoding mismatch
`train.py` includes a “final safety check”:
- if sampled move not legal, it prints FEN and exits.
This is essential because a mismatch between `legal_mask_np` and `idx_to_move` silently breaks training.

### 12.2 Bootstrap correctness
This implementation stores `next_value` per step, but also has a `bootstrap_value` fallback for the last step:
- terminal last step → bootstrap 0
- otherwise uses `next_values[-1]`

### 12.3 KL early stopping
If policy changes too much within an update (approx KL beyond threshold), PPO stops early. This prevents catastrophic updates in sparse/volatile reward regimes.

### 12.4 Variable legal move counts
Normalized entropy makes logs comparable across positions with 5 legal moves vs 50 legal moves.

---

## 13. Future extensions 

Probable future adaption in order to enhance the “decision making quantification” story beyond what is currently logged:

1. **Per-state KL heatmaps**
   - log KL and entropy per ply during endgames (already: `policy_metrics` snapshots captured in `train.py`).

2. **Separate value network temperature**
   - keep temperature only for sampling; evaluate policy (logprobs) without temperature for learning (more standard).
   - currently temperature affects both sampling and the logits used for update.

3. **Update game complexity**
   - update the `boad_environment.py`, `board_encoding.py` and `reward_shaping.py` to highten the game complexity for a stronger agent

4. **Dopamine comparison analysis**
   - use the implemented dopamine TD-error agents and deterministic evaluation suite below to compare self-play training against frozen-PPO-opponent training.
---


## Appendix A: “Where in code” quick pointers

```text
- Action sampling + masking + temperature: `ppo.py:select_action()`
- Rollout storage: `ppo.py:store_transition()` called from `train.py`
- Reward scaling/clipping + GAE(λ): `ppo.py:compute_gae()`
- PPO loss (actor/critic/entropy): `ppo.py:update()`
- Value clipping logic: `ppo.py:update()` under `if self.value_clip is not None`
- KL early stopping: `ppo.py:update()` after `approx_kl` computation
- Adaptive entropy/temperature: end of `ppo.py:update()`
- Actor/Critic architectures: `actor.py`, `critic.py`
- Dopamine TD-error learner: `dopamine_pg.py`
- Legal action map shared by agents/opponents: `action_encoding.py:build_action_map()`
- Self-play and frozen policy opponents: `opponent_selfplay.py`
- Deterministic dopamine comparison: `evaluate_dopamine.py`
```
---

# New Chapter BA 2: Dopamine-inspired TD-error agent system

The repository now supports a thesis comparison between PPO and two dopamine-inspired actor-critic agents. The dopamine agent is **not PPO-like**: it learns from an explicit temporal-difference reward-prediction error (RPE) signal:

```text
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
```

This signal is stored as `td_error` / `dopamine_delta` and is directly logged for analysis. Mood modulation is optional and secondary only; by default it is disabled (`use_mood_modulation=False`, `mood_mean=1.0`, `mood_std=0.0`).

## 14.1 Experiment modes

Training is now selected with `TRAINING_MODE`, not only by `TRAINING_AGENT`:

```bash
TRAINING_MODE=ppo python train.py
TRAINING_MODE=dopamine_self python train.py
TRAINING_MODE=dopamine_vs_ppo python train.py
```

Supported modes:

| Mode | Learning agent | Opponent regime | Checkpoint prefix | TensorBoard run prefix |
| --- | --- | --- | --- | --- |
| `ppo` | PPO | PPO/current-policy self-play snapshots | `models/ppo_chess` | `PPO_<timestamp>` |
| `dopamine_self` | Dopamine TD-error actor-critic | Dopamine self-play snapshots only | `models/dopamine_self_chess` | `DopamineSelf_<timestamp>` |
| `dopamine_vs_ppo` | Dopamine TD-error actor-critic | Frozen PPO actor loaded from `models/ppo_chess` | `models/dopamine_vs_ppo_chess` | `DopamineVsPPO_<timestamp>` |

The two dopamine modes are intended to differ only by opponent regime:

```text
dopamine_self   -> opponent = own dopamine self-play snapshots
dopamine_vs_ppo -> opponent = frozen PPO policy
```

All other settings should be kept equal for thesis comparison: architecture, learning rates, `gamma`, `lambda`, reward shaping, start positions, max timesteps, seeds, and evaluation suite.

## 14.2 Dopamine learner details

`dopamine_pg.py` exposes the same training-loop API as before:

```text
select_action(state_np, legal_mask_np)
evaluate_value(state_np)
store_transition(...)
update()
save(path_prefix)
load(path_prefix, strict=False)
```

For every transition, the rollout buffer stores:

```text
state, action, logprob, reward, done, value, next_value,
legal_mask, td_error / dopamine_delta, mood_scale
```

The critic target is one-step TD:

```text
V_target = r_t + gamma * V(s_{t+1}) * (1 - done_t)
critic_loss = MSE(V(s_t), V_target)
```

The actor is updated with a dopamine/RPE signal. The implementation computes TD(λ)-style advantages from the TD-error sequence:

```text
gae_t = delta_t + gamma * lambda * (1 - done_t) * gae_{t+1}
dopamine_adv = (gae - mean(gae)) / (std(gae) + 1e-8)
policy_loss = -logπ(a_t | s_t) * dopamine_adv
```

If mood modulation is explicitly enabled:

```bash
USE_MOOD_MODULATION=1 TRAINING_MODE=dopamine_self python train.py
```

then mood scales only the actor signal, not the reward or critic target:

```text
actor_signal = dopamine_adv * mood_scale
```

## 14.3 Legal-action sampling and action maps

Legal action handling is centralized in `action_encoding.py`:

```python
idxs, idx_to_move, legal_mask_np = build_action_map(board, encoder)
```

`build_action_map` iterates over `board.legal_moves`, encodes each move, ignores unencodable moves, rejects ids outside `[0, encoder.ACT_DIM)`, avoids duplicate ids, and returns the mask used by agents and opponents.

Dopamine action sampling no longer builds a distribution over all 4672 actions with illegal logits set to `-1e9`. Instead, it samples only among legal ids:

```python
legal_ids = torch.nonzero(legal_mask[0] > 0, as_tuple=False).squeeze(-1)
legal_logits = logits[0, legal_ids]
dist = Categorical(logits=legal_logits)
sampled_offset = dist.sample()
action = legal_ids[sampled_offset]
```

The self-play and frozen-policy opponents use the same legal-only strategy. If a sampled action cannot be mapped back to a move, the code raises a diagnostic error containing the sampled id, legal ids, FEN, and legal UCI moves.

## 14.4 Frozen PPO opponent

`FrozenPolicyOpponent` in `opponent_selfplay.py` is used for `TRAINING_MODE=dopamine_vs_ppo`. It:

- accepts an actor, encoder, `flatten_obs`, and device;
- deep-copies the actor;
- switches it to `eval()` mode;
- disables gradients with `requires_grad_(False)`;
- chooses moves under `torch.no_grad()`;
- samples only from legal action ids.

For `dopamine_vs_ppo`, the PPO checkpoint must exist:

```text
models/ppo_chess_actor.pt
models/ppo_chess_critic.pt
```

The PPO opponent is inference-only and its weights are never updated during dopamine training.

## 14.5 Checkpoints and shared dopamine initialization

Checkpoint namespaces are intentionally separate:

```text
models/ppo_chess
models/dopamine_self_chess
models/dopamine_vs_ppo_chess
models/dopamine_base_chess
```

To initialize both dopamine agents from the exact same base checkpoint for fair comparison:

```bash
INIT_DOPAMINE_BASE=1 TRAINING_MODE=dopamine_self python train.py
INIT_DOPAMINE_BASE=1 TRAINING_MODE=dopamine_vs_ppo python train.py
```

This creates or loads:

```text
models/dopamine_base_chess
```

and copies it into both dopamine experiment checkpoints:

```text
models/dopamine_self_chess
models/dopamine_vs_ppo_chess
```

Dopamine is **not** initialized from PPO unless explicitly requested:

```bash
INIT_DOPAMINE_FROM_PPO=1 TRAINING_MODE=dopamine_self python train.py
```

When combined with `INIT_DOPAMINE_BASE=1`, the shared dopamine base is created from PPO only if `INIT_DOPAMINE_FROM_PPO=1`; otherwise the base comes from the dopamine random initialization.

## 14.6 TensorBoard logging and metadata

Each run logs comparison metadata in hparams:

```text
training_mode
agent_type
opponent_type
checkpoint_prefix
initialized_from
fixed_opponent_prob
start_mode
seed
```

Dopamine runs additionally log explicit RPE metrics:

```text
Dopamine/td_error_mean
Dopamine/td_error_abs_mean
Dopamine/td_error_std
Dopamine/td_error_positive_rate
Dopamine/td_lambda_adv_mean
Dopamine/td_lambda_adv_std
Dopamine/actor_signal_mean
Dopamine/actor_signal_std
Dopamine/mood_scale_mean  # only when mood modulation is enabled
Policy/legal_entropy
Policy/temperature
Episode/return
Episode/length
```

These metrics make the dopamine signal directly measurable for comparison with PPO.

## 14.7 Deterministic dopamine comparison evaluation

Use the evaluation script after training both dopamine agents:

```bash
python evaluate_dopamine.py
```

It evaluates:

```text
models/dopamine_self_chess
models/dopamine_vs_ppo_chess
```

against the same fixed opponent suite:

```text
random
heuristic
frozen_ppo
dopamine_self_snapshot
dopamine_vs_ppo_snapshot
```

The script uses the same number of games, random seed, start mode, max ply, and alternating agent colors for both dopamine agents. Configure it with:

```bash
SEED=0 EVAL_GAMES=20 EVAL_START_MODE=endgame EVAL_MAX_PLY=160 python evaluate_dopamine.py
```

It writes:

```text
evaluation_results/dopamine_comparison.json
evaluation_results/dopamine_comparison.csv
```

Metrics include:

```text
win rate
draw rate
loss rate
average return
return per ply
average game length
mate/conversion rate
illegal action count
average TD error
TD error abs mean
policy entropy
legal candidate count
```

## 14.8 Recommended thesis workflow

1. Train or load PPO baseline:

```bash
TRAINING_MODE=ppo SEED=0 python train.py
```

2. Create a shared dopamine base and train dopamine self-play:

```bash
INIT_DOPAMINE_BASE=1 TRAINING_MODE=dopamine_self SEED=0 python train.py
```

3. Train dopamine against frozen PPO from the same dopamine base:

```bash
INIT_DOPAMINE_BASE=1 TRAINING_MODE=dopamine_vs_ppo SEED=0 python train.py
```

4. Run deterministic comparison:

```bash
SEED=0 EVAL_GAMES=20 python evaluate_dopamine.py
```

This workflow ensures the two dopamine agents differ only by opponent regime while using the same architecture, initialization policy, learning setup, and evaluation suite.
