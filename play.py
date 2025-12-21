import argparse
import numpy as np
import chess

from board_environment import Chess
from board_encoding import BoardEncoding
from action_encoding import AlphaZeroActionEncoder
from ppo import PPO


def flatten_obs(obs):
    return obs.astype(np.float32).ravel()


def move_requires_promotion(board, move):
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return False

    to_rank = move.to_square // 8
    return (piece.color and to_rank == 7) or (not piece.color and to_rank == 0)


def build_legal_mask(board, encoder, action_dim):
    legal_moves = list(board.legal_moves)
    idxs = []
    idx_to_move = {}

    for mv in legal_moves:
        if move_requires_promotion(board, mv) and mv.promotion is None:
            mv = chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN)

        try:
            idx = encoder.encode(mv, board)
        except Exception:
            continue
        idxs.append(idx)
        idx_to_move[idx] = mv

    mask = np.zeros(action_dim, dtype=np.float32)
    mask[idxs] = 1.0
    return mask, idx_to_move


def parse_args():
    parser = argparse.ArgumentParser(description="Play against a trained PPO chess agent in the terminal.")
    parser.add_argument("--model-prefix", default="models/ppo_chess", help="Path prefix to saved PPO actor/critic weights.")
    parser.add_argument("--device", default="cpu", help="Device for inference (cpu or cuda).")
    parser.add_argument("--color", choices=["white", "black"], default="white", help="Choose your side.")
    parser.add_argument("--start-mode", choices=["standard", "endgame"], default="endgame", help="Game start mode.")
    parser.add_argument("--endgame-max-extra-per-side", type=int, default=3, help="Extra pieces per side for endgame starts.")
    parser.add_argument("--endgame-min-extra-total", type=int, default=1, help="Total extra pieces minimum for endgame starts.")
    parser.add_argument("--history-length", type=int, default=2, help="Observation history length used by the encoder.")
    return parser.parse_args()


def main():
    args = parse_args()

    base_env = Chess(
        start_mode=args.start_mode,
        endgame_max_extra_per_side=args.endgame_max_extra_per_side,
        endgame_min_extra_total=args.endgame_min_extra_total,
        require_pawn=True,
    )
    env = BoardEncoding(base_env, history_length=args.history_length)
    encoder = AlphaZeroActionEncoder()

    obs, info = env.reset()
    state_dim = flatten_obs(obs).shape[0]
    action_dim = encoder.ACT_DIM

    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        device=args.device,
    )
    agent.load(args.model_prefix, strict=False)

    human_is_white = args.color == "white"
    print(f"You are playing as {'White' if human_is_white else 'Black'}. Enter moves in UCI format (e.g., e2e4). Type 'quit' to exit.")

    board = base_env._board
    done = False
    current_obs = obs

    while not done:
        print("\nCurrent board:")
        print(base_env.render())
        print(f"Turn: {'White' if board.turn else 'Black'}")

        human_turn = board.turn == chess.WHITE if human_is_white else board.turn == chess.BLACK

        if human_turn:
            user_input = input("Your move (uci): ").strip().lower()
            if user_input in {"quit", "exit", "q"}:
                print("Exiting game.")
                break
            try:
                move = chess.Move.from_uci(user_input)
            except ValueError:
                print("Could not parse move. Please use UCI like e2e4 or e7e8q.")
                continue

            if move_requires_promotion(board, move) and move.promotion is None:
                move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)

            if move not in board.legal_moves:
                print("Illegal move. Try again.")
                continue

            current_obs, reward, terminated, truncated, info = env.step(move)
            done = terminated or truncated
        else:
            mask, idx_to_move = build_legal_mask(board, encoder, action_dim)
            if mask.sum() == 0:
                print("Agent has no legal moves. Game over.")
                break

            state_np = flatten_obs(current_obs)
            action_id, logprob, value = agent.select_action(state_np, mask)
            move = idx_to_move.get(action_id)
            if move is None:
                move = list(board.legal_moves)[0]

            print(f"Agent plays: {move.uci()}")
            current_obs, reward, terminated, truncated, info = env.step(move)
            done = terminated or truncated

        board = base_env._board

    print("\nFinal board:")
    print(base_env.render())
    print(f"Game result: {board.result()}")


if __name__ == "__main__":
    main()