# opponent_heuristic.py
import chess

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

class HeuristicOpponent:
    def __init__(self, material_weight=1.0, mobility_weight=0.1):
        self.material_weight = material_weight
        self.mobility_weight = mobility_weight

    def evaluate_board(self, board: chess.Board):
        """Positive = good for side to move."""
        score = 0

        # Material
        for piece_type in PIECE_VALUES.keys():
            score += (
                len(board.pieces(piece_type, board.turn)) -
                len(board.pieces(piece_type, not board.turn))
            ) * PIECE_VALUES[piece_type] * self.material_weight

        # Mobility
        score += board.legal_moves.count() * self.mobility_weight

        return score

    def choose_move(self, board: chess.Board):
        best_move = None
        best_score = -1e9

        for mv in board.legal_moves:
            board.push(mv)
            score = self.evaluate_board(board)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = mv

        return best_move
