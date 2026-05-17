import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import torch

from action_encoding import AlphaZeroActionEncoder, build_action_map
from board_encoding import BoardEncoding
from board_environment import Chess
from dopamine_pg import DopaminePolicyGradient

from opponent_selfplay import FrozenPolicyOpponent
from ppo import PPO
from train import (
    DOPAMINE_SELF_PREFIX,
    DOPAMINE_VS_PPO_PREFIX,
    flatten_obs,
)


@dataclass
class EvaluationConfig:
    games: int = int(os.getenv("EVAL_GAMES", "20"))
    seed: int = int(os.getenv("SEED", "0"))
    start_mode: str = os.getenv("EVAL_START_MODE", "endgame")
    max_ply: int = int(os.getenv("EVAL_MAX_PLY", "160"))
    history_length: int = 2
    endgame_max_extra_per_side: int = 4
    endgame_min_extra_total: int = 3
    require_pawn: bool = False
    checkpoint_step: Optional[int] = (
        int(os.getenv("EVAL_CHECKPOINT_STEP"))
        if os.getenv("EVAL_CHECKPOINT_STEP", "").strip()
        else None
    )


def make_env(config: EvaluationConfig) -> Tuple[Chess, BoardEncoding]:
    base_env = Chess(
        start_mode=config.start_mode,
        endgame_max_extra_per_side=config.endgame_max_extra_per_side,
        endgame_min_extra_total=config.endgame_min_extra_total,
        require_pawn=config.require_pawn,
    )
    return base_env, BoardEncoding(base_env, history_length=config.history_length)


def load_dopamine_agent(prefix: str, state_dim: int, action_dim: int, device: str) -> DopaminePolicyGradient:
    agent = DopaminePolicyGradient(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        lam=0.95,
        reward_scale=1.0,
        reward_clip=None,
        mood_mean=1.0,
        mood_std=0.0,
        use_mood_modulation=False,
        device=device,
    )
    if not agent.load(prefix, strict=False):
        raise RuntimeError(f"Missing dopamine checkpoint: {prefix}_*.pt")
    agent.actor.eval()
    agent.critic.eval()
    return agent


def load_ppo_actor(state_dim: int, action_dim: int, device: str):
    ppo = PPO(state_dim=state_dim, action_dim=action_dim, device=device)
    if not ppo.load("models/ppo_chess", strict=False):
        raise RuntimeError("Missing PPO checkpoint: models/ppo_chess_*.pt")
    ppo.actor.eval()
    for param in ppo.actor.parameters():
        param.requires_grad_(False)
    return ppo.actor


def deterministic_policy_stats(agent, state_np, legal_mask_np):
    state = torch.from_numpy(state_np).float().to(agent.device).unsqueeze(0)
    legal_mask = torch.from_numpy(legal_mask_np).to(agent.device).unsqueeze(0)
    with torch.no_grad():
        logits = agent.actor(state) / max(agent.temperature, 1e-6)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        legal_ids = torch.nonzero(legal_mask[0] > 0, as_tuple=False).squeeze(-1)
        if legal_ids.numel() == 0:
            raise RuntimeError("No legal ids for deterministic evaluation action")
        legal_logits = logits[0, legal_ids]
        dist = torch.distributions.Categorical(logits=legal_logits)
        offset = torch.argmax(legal_logits)
        action_id = int(legal_ids[offset].item())
        entropy = float(dist.entropy().item())
        legal_count = int(legal_ids.numel())
        value = float(agent.critic(state).squeeze(-1).item())
    return action_id, entropy, legal_count, value


def choose_opponent_move(opponent, board, observation):
    if hasattr(opponent, "select_move"):
        move, _ = opponent.select_move(board, observation)
        return move
    return opponent.choose_move(board)


def game_outcome(result: str, agent_color: chess.Color) -> str:
    if result == "1/2-1/2" or result == "*":
        return "draw"
    agent_wins = (result == "1-0" and agent_color == chess.WHITE) or (result == "0-1" and agent_color == chess.BLACK)
    return "win" if agent_wins else "loss"


def evaluate_one_game(agent, opponent, encoder, config: EvaluationConfig, game_idx: int, agent_color: chess.Color):
    seed = config.seed + game_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    base_env, env = make_env(config)
    obs, _ = env.reset(seed=seed, options={"max_ply": config.max_ply, "agent_color": agent_color})
    board = base_env._board
    start_fen = board.fen()
    done = False
    terminal = False
    total_return = 0.0
    td_errors: List[float] = []
    entropies: List[float] = []
    legal_counts: List[int] = []
    illegal_action_count = 0
    plies = 0

    while not done:
        state_np = flatten_obs(obs)
        idxs, idx_to_move, legal_mask_np = build_action_map(board, encoder)
        if not idxs:
            illegal_action_count += 1
            break

        action_id, entropy, legal_count, value = deterministic_policy_stats(agent, state_np, legal_mask_np)
        entropies.append(entropy)
        legal_counts.append(legal_count)
        move = idx_to_move.get(action_id)
        if move is None or move not in board.legal_moves:
            illegal_action_count += 1
            break

        obs_next, reward_agent, terminated, truncated, _ = env.step(move)
        combined_reward = reward_agent
        total_return += reward_agent
        plies += 1
        done = terminated or truncated
        terminal = terminated

        if not done:
            opp_move = choose_opponent_move(opponent, board, obs_next)
            if opp_move is None:
                done = True
                terminal = True
                combined_reward += 1.0
                total_return += 1.0
            else:
                obs_next, reward_opp, terminated_opp, truncated_opp, _ = env.step(opp_move)
                combined_reward += reward_opp
                total_return += reward_opp
                plies += 1
                done = terminated_opp or truncated_opp
                terminal = terminated_opp

        next_value = 0.0 if done else agent.evaluate_value(flatten_obs(obs_next))
        td_errors.append(combined_reward + agent.gamma * next_value * (1.0 - float(done)) - value)
        obs = obs_next

        if plies >= config.max_ply:
            done = True

    result = board.result()
    outcome = game_outcome(result, agent_color)
    agent_conversion = outcome == "win" and (terminal or board.is_checkmate())
    return {
        "start_fen": start_fen,
        "agent_color": "white" if agent_color == chess.WHITE else "black",
        "result": result,
        "outcome": outcome,
        "return": total_return,
        "length": plies,
        "return_per_ply": total_return / max(1, plies),
        "mate_conversion": agent_conversion,
        "illegal_action_count": illegal_action_count,
        "td_error_mean": float(np.mean(td_errors)) if td_errors else 0.0,
        "td_error_abs_mean": float(np.mean(np.abs(td_errors))) if td_errors else 0.0,
        "policy_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "legal_candidate_count": float(np.mean(legal_counts)) if legal_counts else 0.0,
    }


def aggregate_games(games: List[Dict]) -> Dict:
    n = max(1, len(games))
    wins = sum(g["outcome"] == "win" for g in games)
    draws = sum(g["outcome"] == "draw" for g in games)
    losses = sum(g["outcome"] == "loss" for g in games)
    return {
        "games": len(games),
        "win_rate": wins / n,
        "draw_rate": draws / n,
        "loss_rate": losses / n,
        "average_return": float(np.mean([g["return"] for g in games])) if games else 0.0,
        "return_per_ply": float(np.mean([g["return_per_ply"] for g in games])) if games else 0.0,
        "average_game_length": float(np.mean([g["length"] for g in games])) if games else 0.0,
        "mate_conversion_rate": sum(g["mate_conversion"] for g in games) / n,
        "illegal_action_count": int(sum(g["illegal_action_count"] for g in games)),
        "average_td_error": float(np.mean([g["td_error_mean"] for g in games])) if games else 0.0,
        "td_error_abs_mean": float(np.mean([g["td_error_abs_mean"] for g in games])) if games else 0.0,
        "policy_entropy": float(np.mean([g["policy_entropy"] for g in games])) if games else 0.0,
        "legal_candidate_count": float(np.mean([g["legal_candidate_count"] for g in games])) if games else 0.0,
    }


def evaluate_dopamine_comparison():
    config = EvaluationConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    encoder = AlphaZeroActionEncoder()

    base_env, env = make_env(config)
    obs, _ = env.reset(seed=config.seed, options={"max_ply": config.max_ply, "agent_color": chess.WHITE})
    state_dim = flatten_obs(obs).shape[0]
    action_dim = encoder.ACT_DIM

    agents = {
        "dopamine_self": load_dopamine_agent(DOPAMINE_SELF_PREFIX, state_dim, action_dim, device),
        "dopamine_vs_ppo": load_dopamine_agent(DOPAMINE_VS_PPO_PREFIX, state_dim, action_dim, device),
    }
    ppo_actor = load_ppo_actor(state_dim, action_dim, device)

    protocol_pairings = {
        "dopamine_self": (
            "dopamine_self_snapshot",
            FrozenPolicyOpponent(
                agents["dopamine_self"].actor,
                encoder,
                flatten_obs,
                torch.device(device),
                deterministic=True,
            ),
        ),
        "dopamine_vs_ppo": (
            "frozen_ppo",
            FrozenPolicyOpponent(
                ppo_actor,
                encoder,
                flatten_obs,
                torch.device(device),
                deterministic=True,
            ),
        ),
    }

    details = []
    aggregates = []
    for agent_name, agent in agents.items():
        opponent_name, opponent = protocol_pairings[agent_name]
        games = []
        for game_idx in range(config.games):
            agent_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
            game = evaluate_one_game(agent, opponent, encoder, config, game_idx, agent_color)
            game.update({"agent": agent_name, "opponent": opponent_name, "game_index": game_idx})
            games.append(game)
            details.append(game)
        aggregate = aggregate_games(games)
        aggregate.update({"agent": agent_name, "opponent": opponent_name})
        aggregates.append(aggregate)

    os.makedirs("evaluation_results", exist_ok=True)
    output_suffix = f"_step_{config.checkpoint_step}" if config.checkpoint_step is not None else ""
    json_path = f"evaluation_results/dopamine_comparison{output_suffix}.json"
    csv_path = f"evaluation_results/dopamine_comparison{output_suffix}.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config.__dict__,
                "protocol": "isolated_dopamine_pairings",
                "allowed_pairings": {
                    "dopamine_self": "dopamine_self_snapshot",
                    "dopamine_vs_ppo": "frozen_ppo",
                },
                "checkpoint_prefixes": {
                    "dopamine_self": DOPAMINE_SELF_PREFIX,
                    "dopamine_vs_ppo": DOPAMINE_VS_PPO_PREFIX,
                    "frozen_ppo": "models/ppo_chess",
                },
                "aggregates": aggregates,
                "games": details,
            },
            f,
            indent=2,
        )

    fieldnames = [
        "agent",
        "opponent",
        "games",
        "win_rate",
        "draw_rate",
        "loss_rate",
        "average_return",
        "return_per_ply",
        "average_game_length",
        "mate_conversion_rate",
        "illegal_action_count",
        "average_td_error",
        "td_error_abs_mean",
        "policy_entropy",
        "legal_candidate_count",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregates)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    evaluate_dopamine_comparison()
