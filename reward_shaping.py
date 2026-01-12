# reward_shaping.py

import chess
import math

class RewardShaper:
    def __init__(self):
        # Tune these as needed
        # self.capture_reward = 0.1
        # self.check_reward = 0.05
        # self.pawn_push_reward = 0.01
        # self.castle_reward = 0.05
        # self.piece_development_reward = 0.02
        # self.repeat_penalty = -0.1

        # endgame focused reward
        self.capture_reward = 0.08
        self.check_reward = 0.04
        self.pawn_push_reward = 0.02
        self.promotion_bonus = 0.35
        self.king_activity_weight = 0.01
        self.mate_pressure_bonus = 0.05
        self.repeat_penalty = -0.08
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

    def _pawn_progress(self, board_before: chess.Board, board_after: chess.Board, move: chess.Move) -> float:
        """Reward pawns that advance toward promotion or convert."""
        pawn_reward = 0.0
        piece = board_before.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            from_rank = chess.square_rank(move.from_square)
            to_rank = chess.square_rank(move.to_square)
            direction = 1 if piece.color == chess.WHITE else -1
            progress = (to_rank - from_rank) * direction
            if progress > 0:
                pawn_reward += self.pawn_push_reward * progress

            if move.promotion:
                pawn_reward += self.promotion_bonus

        # Bonus for any pawn that reaches the 7th rank (or 2nd for black)
        for color in [chess.WHITE, chess.BLACK]:
            target_rank = 6 if color else 1
            advanced_pawns = [sq for sq in board_after.pieces(chess.PAWN, color) if
                              chess.square_rank(sq) == target_rank]
            if advanced_pawns:
                pawn_reward += 0.02 * len(advanced_pawns)

        return pawn_reward

    def _king_activity(self, board_before: chess.Board, board_after: chess.Board, move: chess.Move) -> float:
        """Encourage kings to centralize and approach the opponent in endgames."""
        activity = 0.0
        piece = board_before.piece_at(move.from_square)
        if piece and piece.piece_type == chess.KING:
            center = (3.5, 3.5)

            def dist_to_center(square):
                r, f = chess.square_rank(square), chess.square_file(square)
                return math.hypot(r - center[0], f - center[1])

            before_dist = dist_to_center(move.from_square)
            after_dist = dist_to_center(move.to_square)
            activity += self.king_activity_weight * max(0.0, before_dist - after_dist)

            # Encourage opposition: move closer to the enemy king
            enemy_king_sq = board_after.king(not piece.color)
            if enemy_king_sq is not None:
                def taxi_distance(a, b):
                    return abs(chess.square_rank(a) - chess.square_rank(b)) + abs(
                        chess.square_file(a) - chess.square_file(b))

                dist_before = taxi_distance(move.from_square, enemy_king_sq)
                dist_after = taxi_distance(move.to_square, enemy_king_sq)
                activity += self.king_activity_weight * 0.5 * max(0.0, dist_before - dist_after)

        return activity

    def _mate_pressure(self, board_after: chess.Board) -> float:
        """Reward states that strongly restrict the opponent and signal mate is near."""
        pressure = 0.0
        if board_after.is_check():
            legal_moves = board_after.legal_moves.count()
            scarcity_factor = max(0.0, (6 - legal_moves) / 6.0)
            pressure += self.mate_pressure_bonus * (1.0 + scarcity_factor)
        return pressure

    def shaped_reward(self, board_before, board_after, move, move_count):
        reward = 0.0

    def _blank_breakdown(self) -> dict:
        return {
            "capture": 0.0,
            "check": 0.0,
            "pawn_progress": 0.0,
            "king_activity": 0.0,
            "mate_pressure": 0.0,
            "repetition": 0.0,
            "stalling": 0.0,
            "total": 0.0,
        }

    def shaped_reward_breakdown(self, board_before, board_after, move, move_count) -> dict:
        breakdown = self._blank_breakdown()

        # --- Capture ---
        if board_before.is_capture(move):
            captured_piece = board_before.piece_at(move.to_square)
            if captured_piece:
                breakdown["capture"] = (
                        self.capture_reward * (self.material_value(captured_piece.piece_type) / 10.0)
                )
                # reward += self.capture_reward * (self.material_value(captured_piece.piece_type) / 10.0)

        # --- Check ---
        if board_after.is_check():
            # reward += self.check_reward

        # # --- Pawn push forward ---
        # piece = board_before.piece_at(move.from_square)
        # if piece and piece.piece_type == chess.PAWN:
        #     from_rank = chess.square_rank(move.from_square)
        #     to_rank = chess.square_rank(move.to_square)
        #     if piece.color and to_rank > from_rank:    # white pawn moves up
        #         reward += self.pawn_push_reward
        #     if not piece.color and to_rank < from_rank:  # black pawn moves down
        #         reward += self.pawn_push_reward
        #
        # # --- Castling ---
        # if board_before.is_castling(move):
        #     reward += self.castle_reward
        #
        # # --- Development ---
        # if piece and piece.piece_type in {chess.KNIGHT, chess.BISHOP}:
        #     starting_squares = {
        #         chess.WHITE: {1, 6},   # b1, g1
        #         chess.BLACK: {57, 62}  # b8, g8
        #     } if piece.piece_type == chess.KNIGHT else {
        #         chess.WHITE: {2, 5},   # c1, f1
        #         chess.BLACK: {58, 61}  # c8, f8
        #     }
        #     if move.from_square in starting_squares[piece.color]:
        #         reward += self.piece_development_reward

        # --- Endgame pawn and king incentives ---
        # reward += self._pawn_progress(board_before, board_after, move)
        # reward += self._king_activity(board_before, board_after, move)
        # reward += self._mate_pressure(board_after)

        # --- Repetitive movement penalty ---
            breakdown["check"] = self.check_reward

        breakdown["pawn_progress"] = self._pawn_progress(board_before, board_after, move)
        breakdown["king_activity"] = self._king_activity(board_before, board_after, move)
        breakdown["mate_pressure"] = self._mate_pressure(board_after)
        if board_after.can_claim_threefold_repetition():
            # reward += self.repeat_penalty
            breakdown["repetition"] = self.repeat_penalty

        # --- Stall penalty (too many moves) ---
        if move_count > 160:
            # reward += self.stalling_penalty

        # return reward
            breakdown["stalling"] = self.stalling_penalty

        breakdown["total"] = sum(
            breakdown[key]
            for key in (
                "capture",
                "check",
                "pawn_progress",
                "king_activity",
                "mate_pressure",
                "repetition",
                "stalling",
            )
        )
        return breakdown

    def shaped_reward(self, board_before, board_after, move, move_count):
        breakdown = self.shaped_reward_breakdown(board_before, board_after, move, move_count)
        return breakdown["total"]
