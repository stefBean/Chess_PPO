from board_environment import Chess
from board_encoding import BoardEncoding
from inspect_board import BoardAnalyzer
import chess

def main():
    base_env = Chess()
    env = BoardEncoding(base_env, history_length=2)

    # Reset environment
    obs, info = env.reset()
    print("Initial observation shape:", obs.shape)

    analyzer = BoardAnalyzer()


    print("\nInitial board:")
    # print(base_env.render())
    print(analyzer.describe_board(base_env._board))


    # Pick a move
    move = chess.Move.from_uci("e2e4")

    # Step
    obs, reward, terminated, truncated, info = env.step(move)
    print("\nAfter move e2e4:")
    # print(base_env.render())
    print(analyzer.describe_board(base_env._board))

    print("Reward:", reward, "Terminated:", terminated, "Truncated:", truncated)
    print("Observation shape:", obs.shape)

    print("\nSample encoding slice (first 2 planes):")
    print(obs[:, :, :2])

if __name__ == "__main__":
    main()
