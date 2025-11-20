# train.py

import numpy as np
import torch
import subprocess
import webbrowser
import time

from board_environment import Chess
from board_encoding import BoardEncoding
from action_encoding import AlphaZeroActionEncoder
from ppo import PPO
from opponent_random import RandomOpponent
from opponent_heuristic import HeuristicOpponent
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_env = Chess()
    env = BoardEncoding(base_env, history_length=2)
    encoder = AlphaZeroActionEncoder()

    # Get initial observation to determine dimensions
    obs, info = env.reset()
    state_dim = flatten_obs(obs).shape[0]
    action_dim = encoder.ACT_DIM

    print(f"State dim: {state_dim}, Action dim: {action_dim}, Device: {device}")

    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        epochs=4,
        minibatch_size=256,
        device=device,
    )

    agent.load("models/ppo_chess")
    opponent = RandomOpponent()
    # opponent = HeuristicOpponent()

    max_timesteps = 200000
    steps_per_update = 4096
    timestep = 0
    episode = 0

    run_name = f"{opponent.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = f"runs/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)
    hparams = {
        "learning_rate_actor": 3e-4,
        "learning_rate_critic": 1e-3,
        "batch_size": 256,
        "epochs_per_update": 10,
        "opponent": opponent.__class__.__name__,
        "steps_per_update": steps_per_update,
    }
    try:
        # Start TensorBoard as background process
        tb_process = subprocess.Popen(
            ["tensorboard", "--logdir", "runs", "--port", "6006"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)

        # Auto-open browser tab
        webbrowser.open("http://localhost:6006")

    except Exception as e:
        print(f"[WARN] Could not launch TensorBoard automatically: {e}")

    while timestep < max_timesteps:
        obs, info = env.reset()
        board = base_env._board
        done = False

        ep_reward = 0.0
        ep_len = 0

        while not done:
            state_np = flatten_obs(obs)

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

            if not idxs:
                # No legal moves => end episode
                reward = -1.0
                done = True
                agent.store_transition(
                    state_np,
                    0,
                    0.0,
                    reward,
                    done,
                    0.0,
                    np.zeros(action_dim, dtype=np.float32),
                )
                break

            # Build mask: 1.0 where legal, 0.0 otherwise
            legal_mask_np = np.zeros(action_dim, dtype=np.float32)
            legal_mask_np[idxs] = 1.0

            # ----------------------------------------
            # CHOOSE ACTION THROUGH PPO WITH MASK
            # ----------------------------------------
            action_id, logprob, value = agent.select_action(state_np, legal_mask_np)

            # Map selected action to legal move
            if action_id not in idx_to_move:
                # Very rare fallback: sample a legal move directly
                move = np.random.choice(legal_moves)
            else:
                move = idx_to_move[action_id]

            # Final safety check
            if move not in board.legal_moves:
                print("ILLEGAL MOVE PRODUCED:", move.uci())
                print("Board FEN:", board.fen())
                print("Legal moves:", [m.uci() for m in board.legal_moves])
                raise SystemExit("Stopping due to masking/encoding mismatch")

            # ----------------------------------------
            # STEP ENVIRONMENT
            # ----------------------------------------
            obs_next, reward, terminated, truncated, info = env.step(move)
            done = terminated or truncated

            # Store transition
            agent.store_transition(
                state_np,
                action_id,
                logprob,
                reward,
                done,
                value,
                legal_mask_np,
            )

            ep_reward += reward
            ep_len += 1
            timestep += 1

            obs = obs_next
            board = base_env._board

            # PPO update condition
            if timestep % steps_per_update == 0:
                print(f"\nPPO UPDATE @ timestep {timestep}, episode {episode}")
                metrics = agent.update()

                # Log to TensorBoard
                writer.add_scalar("Loss/actor", metrics["actor_loss"], timestep)
                writer.add_scalar("Loss/critic", metrics["critic_loss"], timestep)
                writer.add_scalar("Loss/entropy", metrics["entropy"], timestep)
                writer.add_scalar("Timesteps/timestep", timestep, timestep)
                writer.flush()

            if timestep >= max_timesteps:
                done = True
                break

        episode += 1
        print(f"Episode {episode} | Return={ep_reward:.2f} | Length={ep_len}")
        writer.add_scalar("Episode/return", ep_reward, episode)
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
