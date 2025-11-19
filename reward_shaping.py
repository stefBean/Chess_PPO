# reward_shaping.py

import chess

class RewardShaper:
    def __init__(self):
        # Tune these as needed
        self.capture_reward = 0.1
        self.check_reward = 0.05
        self.pawn_push_reward = 0.01
        self.castle_reward = 0.05
        self.piece_development_reward = 0.02
        self.repeat_penalty = -0.1
        self.stalling_penalty = -0.01

    def material_value(self, piece_type):
        """Rough material scoring; small because shaping must be tiny."""
        if piece_type == chess.PAWN:
            return 1.0
        if piece_type == chess.KNIGHT or piece_type == chess.BISHOP:
            return 3.0
        if piece_type == chess.ROOK:
            return 5.0
        if piece_type == chess.QUEEN:
            return 9.0
        return 0

    def shaped_reward(self, board_before, board_after, move, move_count):
        reward = 0.0

        # --- Capture ---
        if board_before.is_capture(move):
            captured_piece = board_before.piece_at(move.to_square)
            if captured_piece:
                reward += self.capture_reward * (self.material_value(captured_piece.piece_type) / 10.0)

        # --- Check ---
        if board_after.is_check():
            reward += self.check_reward

        # --- Pawn push forward ---
        piece = board_before.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            from_rank = chess.square_rank(move.from_square)
            to_rank = chess.square_rank(move.to_square)
            if piece.color and to_rank > from_rank:    # white pawn moves up
                reward += self.pawn_push_reward
            if not piece.color and to_rank < from_rank:  # black pawn moves down
                reward += self.pawn_push_reward

        # --- Castling ---
        if board_before.is_castling(move):
            reward += self.castle_reward

        # --- Development ---
        if piece and piece.piece_type in {chess.KNIGHT, chess.BISHOP}:
            starting_squares = {
                chess.WHITE: {1, 6},   # b1, g1
                chess.BLACK: {57, 62}  # b8, g8
            } if piece.piece_type == chess.KNIGHT else {
                chess.WHITE: {2, 5},   # c1, f1
                chess.BLACK: {58, 61}  # c8, f8
            }
            if move.from_square in starting_squares[piece.color]:
                reward += self.piece_development_reward

        # --- Repetitive movement penalty ---
        if board_after.can_claim_threefold_repetition():
            reward += self.repeat_penalty

        # --- Stall penalty (too many moves) ---
        if move_count > 300:
            reward += self.stalling_penalty

        return reward
