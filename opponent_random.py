# For a fast learning experience research suggest to implement first a random opponent -- further explanation needs to be added -- and later on a heuristic and only in the last step a self-play for long-term improvement

# opponent_random.py
import random
import chess

class RandomOpponent:
    def choose_move(self, board: chess.Board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        return random.choice(legal_moves)
