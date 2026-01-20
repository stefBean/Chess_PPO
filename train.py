# train.py
import chess
import numpy as np
import torch
import subprocess
import webbrowser
import time
import random
import json
import os
import sys
import threading

from board_environment import Chess
from board_encoding import BoardEncoding
from action_encoding import AlphaZeroActionEncoder
from ppo import PPO
from opponent_random import RandomOpponent
from opponent_heuristic import HeuristicOpponent
from opponent_selfplay import SelfPlayOpponent, build_action_map, OpponentPool
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from inspect_board import BoardAnalyzer
from collections import deque

from copy import deepcopy

def flatten_obs(obs):
    return obs.astype(np.float32).ravel()  # (2240,)

def move_requires_promotion(board, move):
    """Return True if a pawn moves into last rank and must promote."""
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != 1:  # not a pawn
        return False

    to_rank = move.to_square // 8
    if (piece.color and to_rank == 7) or (not piece.color and to_rank == 0):
        return True
    return False

def quick_eval_gate(
    actor,
    encoder,
    flatten_obs_fn,
    device,
    games: int = 2,
    max_plies: int = 40,
    history_length: int = 2,
    curriculum_stage: int = 0,
    temperature: float = 1.0,
    endgame_max_extra_per_side: int = 4,
    endgame_min_extra_total: int = 3,
    require_pawn: bool = False,
):
    """Lightweight evaluation to decide if a snapshot enters the pool."""
    base_env = Chess(
        start_mode="curriculum_endgame",
        endgame_max_extra_per_side=endgame_max_extra_per_side,
        endgame_min_extra_total=endgame_min_extra_total,
        require_pawn=require_pawn,
    )
    env_eval = BoardEncoding(base_env, history_length=history_length)
    random_opp = RandomOpponent()
    total_score = 0.0
    actor.eval()

    for _ in range(games):
        obs, _ = env_eval.reset(options={"curriculum_stage": curriculum_stage})
        board = base_env._board
        done = False
        plies = 0

        while not done and plies < max_plies:
            state_np = flatten_obs_fn(obs)
            idxs, idx_to_move = build_action_map(board, encoder)
            if not idxs:
                total_score -= 0.5
                break

            legal_mask_np = np.zeros(encoder.ACT_DIM, dtype=np.float32)
            legal_mask_np[idxs] = 1.0

            state = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
            legal_mask = torch.from_numpy(legal_mask_np).to(device).unsqueeze(0)

            with torch.no_grad():
                logits = actor(state) / max(temperature, 1e-6)
                masked_logits = logits.masked_fill(legal_mask == 0, -1e9)
                dist = torch.distributions.Categorical(logits=masked_logits)
                action = dist.sample()

            move = idx_to_move.get(int(action.item()))
            if move is None:
                total_score -= 0.5
                break

            obs, reward, terminated, truncated, _ = env_eval.step(move)
            done = terminated or truncated
            plies += 1

            if done:
                total_score += reward
                break

            opp_move = random_opp.choose_move(board)
            if opp_move is None:
                total_score += 1.0
                break

            obs, reward2, terminated, truncated, _ = env_eval.step(opp_move)
            done = terminated or truncated
            plies += 1

            if done:
                total_score += (reward + reward2)

    actor.train()
    return total_score / max(1, games)

def print_game_with_analysis(moves, start_fen, episode, analyzer, outcome_summary, endgame_start):
    print(
        f"\n[GAME] Episode {episode} | start FEN: {start_fen} | "
        f"result: {outcome_summary['result']} | winner: {outcome_summary['winner_label']}"
    )
    print(
        f"[GAME] Agent: {outcome_summary['agent_side']} | Opponent: {outcome_summary['opponent_side']} "
        f"({outcome_summary['opponent_label']})"
    )
    if endgame_start["ply"] is not None:
        print(
            f"[GAME] Endgame starts at ply {endgame_start['ply']} "
            f"({endgame_start['reason']})"
        )
    board = chess.Board(start_fen)
    print(analyzer.describe_board(board))
    for entry in moves:
        move = chess.Move.from_uci(entry["uci"])
        if move not in board.legal_moves:
            print(
                f"[GAME] Illegal move encountered in log: {entry['uci']} "
                f"(ply {entry['ply']}, player {entry['player']})"
            )
            break
        board.push(move)
        reward_total = entry.get("reward_total", 0.0)
        reward_terminal = entry.get("reward_terminal", 0.0)
        reward_shaping = entry.get("reward_shaping", 0.0)
        check_before = entry.get("check_before")
        check_after = entry.get("check_after")
        check_note = ""
        if check_before is not None or check_after is not None:
            check_note = f" | check_before={check_before} check_after={check_after}"
        print(
            f"\n[PLY {entry['ply']}] {entry['player']} "
            f"{entry.get('san', '')} ({entry['uci']}) via {entry.get('by', 'agent')} | "
            f"reward {reward_total:+.3f} (terminal {reward_terminal:+.3f}, shaping {reward_shaping:+.3f})"
            f"{check_note}"
        )
        snapshot = entry.get("endgame_snapshot")
        if snapshot:
            print(
                f"[ENDGAME SNAPSHOT] FEN: {snapshot['fen']}\n"
                f"{snapshot['analysis']}"
            )
        policy_metrics = entry.get("policy_metrics")
        value_before = entry.get("value_before")
        value_after = entry.get("value_after")

        if policy_metrics:
            print(
                f"[POLICY] entropy={policy_metrics['entropy']:.3f} "
                f"gap={policy_metrics['action_value_gap']:.3f} "
                f"top={policy_metrics['top_moves']} "
                f"V_before={value_before:.3f} V_after={value_after if value_after is not None else float('nan'):.3f}"
            )
        print(analyzer.describe_board(board))

def resolve_outcome(result, agent_color, opponent_label):
    if result == "1-0":
        winner_side = "white"
    elif result == "0-1":
        winner_side = "black"
    elif result == "1/2-1/2":
        winner_side = "draw"
    else:
        winner_side = "unknown"

    agent_side = "white" if agent_color == chess.WHITE else "black"
    opponent_side = "black" if agent_side == "white" else "white"

    if winner_side == "draw":
        winner_label = "draw"
    elif winner_side == agent_side:
        winner_label = "agent"
    elif winner_side == opponent_side:
        winner_label = opponent_label
    else:
        winner_label = "unknown"

    return {
        "result": result,
        "winner_side": winner_side,
        "winner_label": winner_label,
        "agent_side": agent_side,
        "opponent_side": opponent_side,
        "opponent_label": opponent_label,
    }


def build_policy_metrics(agent, state_np, legal_mask_np, idx_to_move, board, top_k=3):
    state = torch.from_numpy(state_np).float().to(agent.device).unsqueeze(0)
    legal_mask = torch.from_numpy(legal_mask_np).to(agent.device).unsqueeze(0)
    with torch.no_grad():
        logits = agent.actor(state) / max(agent.temperature, 1e-6)
        masked_logits = logits.masked_fill(legal_mask == 0, -1e9)
        dist = torch.distributions.Categorical(logits=masked_logits)
        entropy = float(dist.entropy().item())
        probs = dist.probs.squeeze(0)

    legal_count = int(legal_mask_np.sum())
    k = max(1, min(top_k, legal_count))
    top_probs, top_indices = torch.topk(probs, k)
    top_moves = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        move = idx_to_move.get(int(idx))
        if move is None:
            continue
        top_moves.append(
            {
                "uci": move.uci(),
                "san": board.san(move),
                "prob": float(prob),
            }
        )

    action_value_gap = 0.0
    if legal_count >= 2:
        with torch.no_grad():
            top2 = torch.topk(probs, k=2, dim=-1).values.squeeze(0)
            action_value_gap = float((top2[0] - top2[1]).item())

    return {
        "entropy": entropy,
        "top_moves": top_moves,
        "action_value_gap": action_value_gap,
    }


def build_endgame_snapshot(board, analyzer):
    analysis = analyzer.analyze(board)
    return {
        "fen": board.fen(),
        "board": board.unicode(),
        "analysis": analyzer.format_analysis(analysis),
        "endgame": analysis.endgame,
        "endgame_reason": analysis.endgame_reason,
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_game_ply = None #240
    curriculum_stage = 0
    endgame_max_extra_per_side = 4
    endgame_min_extra_total = 3
    require_pawn = False
    gate_hits = 0
    max_stage = 3
    threshold = 0.60

    # base_env = Chess()
    # base_env = Chess(start_mode="endgame", endgame_max_extra_per_side=3)
    # env = BoardEncoding(base_env, history_length=2)
    # Start games from random endgame positions to focus learning on
    # conversion/defense scenarios.
    base_env = Chess(
        start_mode="curriculum_endgame",
        endgame_max_extra_per_side=endgame_max_extra_per_side,
        endgame_min_extra_total=endgame_min_extra_total,
        require_pawn=require_pawn,
    )
    env = BoardEncoding(base_env, history_length=2)
    encoder = AlphaZeroActionEncoder()

    # Get initial observation to determine dimensions
    obs, info = env.reset(options={"curriculum_stage": curriculum_stage})
    state_dim = flatten_obs(obs).shape[0]
    action_dim = encoder.ACT_DIM

    print(f"State dim: {state_dim}, Action dim: {action_dim}, Device: {device}")

    entropy_target_start = 0.75
    entropy_target_end = 0.20

    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.13,
        epochs=10,
        minibatch_size=64,
        device=device,
        entropy_coef=0.02,
        value_coef=0.5,
        max_grad_norm=0.6,
        value_clip=0.25,
        target_kl=0.015,
        reward_scale=0.85,
        reward_clip=2.5,
        entropy_target=entropy_target_start,
        entropy_coef_lr=0.005,
        entropy_coef_min=0.002,
        entropy_coef_max=0.10,
        temperature=1.0,
        temperature_lr=0.01,
        temperature_min=0.7,
        temperature_max=1.0,
    )

    #agent.load("models/ppo_chess")
    #opponent = RandomOpponent()
    # opponent = HeuristicOpponent()
    opponent_pool = OpponentPool(max_size=8, p_latest=0.7, seed=0)
    fixed_opponents = [("random", RandomOpponent()), ("heuristic", HeuristicOpponent())]
    fixed_opponent_prob = 0.25
    max_timesteps = 400000 #200000 #500
    steps_per_update = 6144 #4096 #256 #
    timestep = 0
    episode = 0
    # entropy_coef = agent.entropy_coef
    # entropy_coef_min = 0.005
    # entropy_decay = 0.97
    smoothed_return = None
    return_window = deque(maxlen=50)



    #run_name = f"{opponent.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    agent.load("models/ppo_chess", strict=False)

    # Choose how the agent learns: against a static opponent or via self-play.
    use_self_play = True
    # opponent = RandomOpponent()
    # opponent = HeuristicOpponent()
    self_play_opponent = SelfPlayOpponent(
        agent_actor=agent.actor,
        encoder=encoder,
        flatten_obs=flatten_obs,
        device=agent.device,
        refresh_interval=steps_per_update,
        pool_size=6,
        sample_past_prob=0.4,
        temperature=agent.temperature,
    )

    opponent_pool.add(agent.actor)
    update_count = 0
    snapshot_every = 10
    opponent_name = "SelfPlay" #if use_self_play else opponent.__class__.__name__
    run_name = f"{opponent_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = f"runs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    analyzer = BoardAnalyzer()
    print_game_every = 1
    endgame_log_plies = 30
    top_k_policy_moves = 3
    game_log_path = os.path.join(log_dir, "games.jsonl")
    hparams = {
        "learning_rate_actor": 3e-4,
        "learning_rate_critic": 1e-3,
        "batch_size": 384,
        "epochs_per_update": 5,
        "opponent": opponent_name,
        "steps_per_update": steps_per_update,
        "entropy_coef_init": agent.entropy_coef,
        "entropy_coef_min": agent.entropy_coef_min,
        "entropy_coef_max": agent.entropy_coef_max,
        "entropy_target_start": entropy_target_start,
        "entropy_target_end": entropy_target_end,
        "value_coef": 0.65,
        "selfplay_snapshot_pool": 6,
        "target_kl": 0.015,
        "reward_scale": 0.85,
        "temperature_init": agent.temperature,
        "temperature_min": agent.temperature_min,
        "temperature_max": agent.temperature_max,
    }
    try:
        tb_process = subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", "--logdir", "runs", "--port", "6006"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        def _open_tb():
            time.sleep(2)
            try:
                webbrowser.open_new_tab("http://localhost:6006")
            except Exception:
                pass

        threading.Thread(target=_open_tb, daemon=True).start()

    except Exception as e:
        print(f"[WARN] Could not launch TensorBoard automatically: {e}")

    while timestep < max_timesteps:
        agent_color = chess.WHITE if episode % 2 == 0 else chess.BLACK
        obs, info = env.reset(options={"max_ply": max_game_ply, "agent_color": agent_color, "curriculum_stage": curriculum_stage})
        board = base_env._board
        done = False
        agent_color = board.turn  # chess.WHITE or chess.BLACK; fixed for this episode
        starting_fen = board.fen()
        game_moves = []
        final_info = None
        opponent_label = "old_model" if use_self_play else "opponent"
        endgame_started = False
        endgame_start_ply = None
        endgame_reason = None
        endgame_capture_count = 0

        if use_self_play:
            snap = opponent_pool.sample()
            self_play_opponent.load_snapshot(snap)

        ep_reward = 0.0
        ep_len = 0

        # Ensure the frozen self-play opponent periodically syncs with the
        # learning agent so the sparring partner improves over time.
        if use_self_play:
            #self_play_opponent.maybe_refresh(agent.actor, timestep)
            self_play_opponent.begin_episode()
        active_fixed_opponent = None
        if np.random.rand() < fixed_opponent_prob:
            active_fixed_opponent = random.choice(fixed_opponents)
            opponent_label = active_fixed_opponent[0]

        while not done:
            state_np = flatten_obs(obs)
            board_before = board.copy(stack=False)
            board_analysis = analyzer.analyze(board_before)
            if board_analysis.endgame and not endgame_started:
                endgame_started = True
                endgame_start_ply = base_env.move_count
                endgame_reason = board_analysis.endgame_reason
            if timestep % 200 == 0:
                print(f"timestep={timestep} ep_len={ep_len} fen={board.fen()}")

            # ----------------------------------------
            # BUILD LEGAL MOVES + INDEX MAP
            # ----------------------------------------
            legal_moves = list(board.legal_moves)
            idxs = []
            idx_to_move = {}

            for mv in legal_moves:
                # Enforce promotion legality
                if move_requires_promotion(board, mv) and mv.promotion is None:
                    # Skip illegal non-promotion pawn move
                    continue

                try:
                    idx = encoder.encode(mv, board)
                    idxs.append(idx)
                    idx_to_move[idx] = mv
                except:
                    continue

            # idxs, idx_to_move = build_action_map(board, encoder)

            if not idxs:
                reward = -1.0
                done = True
                terminal = True
                agent.store_transition(
                    state_np,
                    0,
                    0.0,
                    reward,
                    done,
                    0.0,
                    0.0,
                    np.zeros(action_dim, dtype=np.float32),
                    terminal,
                )
                break

            legal_mask_np = np.zeros(action_dim, dtype=np.float32)
            legal_mask_np[idxs] = 1.0

            action_id, logprob, value = agent.select_action(state_np, legal_mask_np)
            policy_metrics = None
            if endgame_started and endgame_capture_count < endgame_log_plies:
                policy_metrics = build_policy_metrics(
                    agent,
                    state_np,
                    legal_mask_np,
                    idx_to_move,
                    board_before,
                    top_k=top_k_policy_moves,
                )
            move = idx_to_move.get(action_id)
            if move is None:
                raise RuntimeError(
                    f"Action id {action_id} not in idx_to_move (mask/encoding mismatch). "
                    f"Legal idx count={len(idxs)} fen={board.fen()}"
                )


            # Final safety check
            if move not in board.legal_moves:
                print("ILLEGAL MOVE PRODUCED:", move.uci())
                print("Board FEN:", board.fen())
                print("Legal moves:", [m.uci() for m in board.legal_moves])
                raise SystemExit("Stopping due to masking/encoding mismatch")

            # ----------------------------------------
            # STEP ENVIRONMENT
            # ----------------------------------------
            ply_count = base_env.move_count + 1
            san_move = board_before.san(move)
            obs_next, reward_agent, terminated, truncated, info_agent = env.step(move)
            done = terminated or truncated
            terminal = terminated
            check_before = board_before.is_check()
            check_after = base_env._board.is_check()
            if done:
                final_info = info_agent

            # Opponent (self-play) move before storing transition so the reward
            # reflects the full ply outcome.
            combined_reward = reward_agent
            ep_reward += reward_agent
            ep_len += 1
            endgame_snapshot = None
            if endgame_started and endgame_capture_count < endgame_log_plies:
                endgame_snapshot = build_endgame_snapshot(board_before, analyzer)
                endgame_capture_count += 1
            value_after = None
            if not done:
                value_after = agent.evaluate_value(flatten_obs(obs_next))
            game_moves.append(
                {
                    "ply": ply_count,
                    "player": "white" if board_before.turn == chess.WHITE else "black",
                    "uci": move.uci(),
                    "san": san_move,
                    "by": "agent",
                    "reward_total": info_agent.get("reward_total", reward_agent),
                    "reward_terminal": info_agent.get("reward_terminal", 0.0),
                    "reward_shaping": info_agent.get("reward_shaping", 0.0),
                    "reward_breakdown": info_agent.get("reward_breakdown", {}),
                    "endgame_snapshot": endgame_snapshot,
                    "policy_metrics": policy_metrics,
                    "value_before": value,
                    "value_after": value_after,
                    "check_before": check_before,
                    "check_after": check_after,
                }
            )
            # obs_next, reward_agent, terminated, truncated, info = env.step(move)
            # done = terminated or truncated

            # Opponent (self-play) move before storing transition so the reward
            # reflects the full ply outcome.
            # combined_reward = reward_agent
            # ep_reward += reward_agent
            # ep_len += 1
            if not done:
                if not endgame_started:
                    opp_analysis = analyzer.analyze(board)
                    if opp_analysis.endgame:
                        endgame_started = True
                        endgame_start_ply = base_env.move_count
                        endgame_reason = opp_analysis.endgame_reason
                #if use_self_play:
                # opp_move, _ = self_play_opponent.select_move(board, obs_next)
                #else:
                   # opp_move = opponent.choose_move(board)
                if active_fixed_opponent is None:
                    opp_move, _ = self_play_opponent.select_move(board, obs_next)
                else:
                    opp_move = active_fixed_opponent[1].choose_move(board)
                if opp_move is None:
                    done = True
                    terminal = True
                    combined_reward = reward_agent + 1.0
                else:
                    opp_board_before = board.copy(stack=False)
                    opp_ply = base_env.move_count + 1
                    opp_san = opp_board_before.san(opp_move)
                    obs_after_opp, reward_opp, terminated_opp, truncated_opp, info_opp = env.step(opp_move)
                    # obs_after_opp, reward_opp, terminated_opp, truncated_opp, info = env.step(opp_move)
                    combined_reward += reward_opp
                    ep_reward += reward_opp
                    ep_len += 1
                    done = terminated_opp or truncated_opp
                    terminal = terminated_opp
                    opp_check_before = opp_board_before.is_check()
                    opp_check_after = base_env._board.is_check()
                    if done:
                        final_info = info_opp
                    obs_next = obs_after_opp
                    opp_snapshot = None
                    if endgame_started and endgame_capture_count < endgame_log_plies:
                        opp_snapshot = build_endgame_snapshot(opp_board_before, analyzer)
                        endgame_capture_count += 1
                    game_moves.append(
                        {
                            "ply": opp_ply,
                            "player": "white" if opp_board_before.turn == chess.WHITE else "black",
                            "uci": opp_move.uci(),
                            "san": opp_san,
                            # "by": active_fixed_opponent[0] if active_fixed_opponent else "self_play_snapshot",
                            "by": opponent_label,
                            "reward_total": info_opp.get("reward_total", reward_opp),
                            "reward_terminal": info_opp.get("reward_terminal", 0.0),
                            "reward_shaping": info_opp.get("reward_shaping", 0.0),
                            "reward_breakdown": info_opp.get("reward_breakdown", {}),
                            "endgame_snapshot": opp_snapshot,
                            "policy_metrics": None,
                            "value_before": None,
                            "value_after": None,
                            "check_before": opp_check_before,
                            "check_after": opp_check_after,
                        }
                    )

            # Bootstrap value for the next state to stabilize advantage estimates.
            next_value = 0.0
            if not terminal:
                next_state_np = flatten_obs(obs_next)
                next_value = agent.evaluate_value(next_state_np)

            agent.store_transition(
                state_np,
                action_id,
                logprob,
                combined_reward,
                terminal,
                value,
                next_value,
                legal_mask_np,
            )

            # ep_reward += combined_reward
            # ep_len += 1
            timestep += 1
            obs = obs_next
            board = base_env._board

            if timestep % steps_per_update == 0:
                print(f"\nPPO UPDATE @ timestep {timestep}, episode {episode}")
                progress = min(1.0, timestep / max_timesteps)
                agent.entropy_target = entropy_target_start + (entropy_target_end - entropy_target_start) * progress

                metrics = agent.update()
                self_play_opponent.temperature = agent.temperature

                update_count += 1

                if update_count % snapshot_every == 0:
                    gate_score = quick_eval_gate(
                        agent.actor,
                        encoder,
                        flatten_obs,
                        agent.device,
                        curriculum_stage=curriculum_stage,
                        temperature=agent.temperature,
                        endgame_max_extra_per_side=endgame_max_extra_per_side,
                        endgame_min_extra_total=endgame_min_extra_total,
                        require_pawn=require_pawn,
                    )
                    if gate_score >= -0.15:
                        opponent_pool.add(agent.actor)
                        print(
                            f"[POOL] Added snapshot (gate score={gate_score:.3f}); pool size={len(opponent_pool.snapshots)}")
                    else:
                        print(f"[POOL] Skipped snapshot (gate score={gate_score:.3f})")
                    if gate_score >= threshold:
                        gate_hits += 1
                    else:
                        gate_hits = 0

                    if gate_hits >= 2 and curriculum_stage < max_stage:
                        curriculum_stage += 1
                        gate_hits = 0
                        print(f"[CURRICULUM] Promoted to stage {curriculum_stage}")
                # entropy_coef = max(entropy_coef_min, entropy_coef * entropy_decay)
                # agent.entropy_coef = entropy_coef

                # Adaptive entropy controller (normalized entropy target)
                entropy_target_start = 0.60  # high exploration early
                entropy_target_end = 0.20  # more deterministic later
                entropy_target_decay = 0.985  # per update
                entropy_target = entropy_target_start

                alpha_lr = 0.002  # controller step size (small!)
                alpha_min = 0.003
                alpha_max = 0.05
                # last_value = 0.0
                # if not done:
                #     state_np = flatten_obs(obs).astype(np.float32)
                #     with torch.no_grad():
                #         s = torch.from_numpy(state_np).float().to(device).unsqueeze(0)
                #         last_value = float(agent.critic(s).squeeze(-1).item())
                # metrics = agent.update(last_value=last_value)

                writer.add_scalar("Loss/actor", metrics["actor_loss"], timestep)
                writer.add_scalar("Loss/critic", metrics["critic_loss"], timestep)
                writer.add_scalar("Loss/entropy", metrics["entropy"], timestep)
                writer.add_scalar("Loss/entropy_normalized", metrics["normalized_entropy"], timestep)
                writer.add_scalar("Loss/entropy_coef", metrics["entropy_coef"], timestep)
                writer.add_scalar("KL/approx_kl", metrics["approx_kl"], timestep)
                writer.add_scalar("KL/policy_shift", metrics["policy_shift_kl"], timestep)
                writer.add_scalar("KL/policy_update", metrics["policy_update_kl"], timestep)
                writer.add_scalar("Advantage/raw_mean", metrics["advantages_raw_mean"], timestep)
                writer.add_scalar("Advantage/raw_std", metrics["advantages_raw_std"], timestep)
                writer.add_scalar("Advantage/normalized_mean", metrics["advantages_norm_mean"], timestep)
                writer.add_scalar("Advantage/normalized_std", metrics["advantages_norm_std"], timestep)
                writer.add_scalar("Policy/expected_advantage", metrics["expected_advantage"], timestep)
                writer.add_scalar("Policy/action_value_gap", metrics["action_value_gap"], timestep)
                writer.add_scalar("Policy/policy_entropy", metrics["entropy"], timestep)
                writer.add_scalar("Policy/policy_entropy_normalized", metrics["normalized_entropy"], timestep)
                writer.add_scalar("Policy/entropy_target", agent.entropy_target, timestep)
                writer.add_scalar("Policy/entropy_schedule_progress", progress, timestep)
                writer.add_scalar("Policy/entropy_schedule_progress", progress, timestep)
                writer.add_scalar("Policy/temperature", metrics["temperature"], timestep)
                writer.add_scalar("Optimization/minibatches", metrics["updates_run"], timestep)
                writer.add_scalar("Timesteps/timestep", timestep, timestep)
                writer.flush()

            if timestep >= max_timesteps:
                done = True
                break

        # final_info = info

        episode += 1
        return_per_ply = ep_reward / max(1, ep_len)
        if smoothed_return is None:
            smoothed_return = ep_reward
        else:
            smoothed_return = 0.9 * smoothed_return + 0.1 * ep_reward
        return_window.append(ep_reward)
        rolling_return = float(np.mean(return_window))

        result_string = (final_info or {}).get("result", board.result())
        outcome_summary = resolve_outcome(result_string, agent_color, opponent_label)
        if endgame_start_ply is None:
            endgame_start_ply = None
            endgame_reason = "not_reached"
        endgame_start = {"ply": endgame_start_ply, "reason": endgame_reason or "not_reached"}
        # agent_color = np.random.choice([chess.WHITE, chess.BLACK])
        try:
            with open(game_log_path, "a", encoding="utf-8") as f:
                json.dump(
                    {
                        "episode": episode,
                        # "agent_color": "white" if agent_color == chess.WHITE else "black",
                        # "result": (final_info or {}).get("result", board.result()),
                        "agent_color": outcome_summary["agent_side"],
                        "opponent_color": outcome_summary["opponent_side"],
                        "opponent_label": opponent_label,
                        "result": outcome_summary["result"],
                        "winner": outcome_summary["winner_label"],
                        "terminal_reason": (final_info or {}).get("terminal_reason"),
                        "forced_draw": (final_info or {}).get("forced_draw", False),
                        "start_fen": starting_fen,
                        "endgame_start_ply": endgame_start["ply"],
                        "endgame_start_reason": endgame_start["reason"],
                        "episode_return": ep_reward,
                        "episode_length": ep_len,
                        "return_per_ply": return_per_ply,
                        "return_smooth": smoothed_return,
                        "return_window_avg": rolling_return,
                        "moves": game_moves,
                    },
                    f,
                )
                f.write("\n")
        except Exception as e:
            print(f"[WARN] Failed to log game for episode {episode}: {e}")

        if print_game_every > 0 and episode % print_game_every == 0:
            print_game_with_analysis(game_moves, starting_fen, episode, analyzer,  outcome_summary, endgame_start)

        # print(f"Episode {episode} | Return={ep_reward:.2f} | Length={ep_len}")
        # if smoothed_return is None:
        #    smoothed_return = ep_reward
        # else:
        #     smoothed_return = 0.9 * smoothed_return + 0.1 * ep_reward
        if print_game_every > 0 and episode % print_game_every == 0:
            print_game_with_analysis(game_moves, starting_fen, episode, analyzer, outcome_summary, endgame_start)

        print(
            f"Episode {episode} | Return={ep_reward:.2f} | Return/Ply={return_per_ply:.3f} | "
            f"Smooth={smoothed_return:.2f} | WindowAvg={rolling_return:.2f} | Length={ep_len}"
        )
        writer.add_scalar("Episode/return", ep_reward, episode)
        writer.add_scalar("Episode/return_smooth", smoothed_return, episode)
        writer.add_scalar("Episode/return_per_ply", return_per_ply, episode)
        writer.add_scalar("Episode/return_window_avg", rolling_return, episode)
        writer.add_scalar("Episode/length", ep_len, episode)

    print("Training finished.")
    writer.add_hparams(
        hparams,
        {
            "final_return": ep_reward,
            "final_length": ep_len,
            "actor_loss_last": metrics["actor_loss"],
            "critic_loss_last": metrics["critic_loss"],
        }
    )
    writer.close()
    agent.save("models/ppo_chess")


if __name__ == "__main__":
    main()
