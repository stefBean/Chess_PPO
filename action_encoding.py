import chess
import numpy as np

QUEEN_DIRS = [8, -8, 1, -1, 9, 7, -7, -9]
KNIGHT_DIRS = [17, 15, 10, 6, -6, -10, -15, -17]

PROMOTION_DIRS = {
    chess.WHITE: [8, 7, 9],
    chess.BLACK: [-8, -7, -9],
}

PROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


class AlphaZeroActionEncoder:
    ACTIONS_PER_SQ = 73
    ACT_DIM = 64 * ACTIONS_PER_SQ  # 4672

    # --------------------------------------------
    # Move -> index
    # --------------------------------------------
    def encode(self, move: chess.Move, board: chess.Board) -> int:
        from_sq = move.from_square
        to_sq = move.to_square
        base = from_sq * self.ACTIONS_PER_SQ
        diff = to_sq - from_sq
        color = board.turn

        # Sliding (0–55)
        idx = self._encode_sliding(diff)
        if idx is not None:
            return base + idx

        # Knight (56–63)
        idx = self._encode_knight(diff)
        if idx is not None:
            return base + idx

        # Underpromotion (64–72)
        if move.promotion in PROMOTION_PIECES:
            idx = self._encode_underpromo(diff, move.promotion, color)
            if idx is not None:
                return base + idx

        # Queen promotions behave like sliding
        if move.promotion == chess.QUEEN:
            idx = self._encode_sliding(diff)
            if idx is not None:
                return base + idx

        raise ValueError("Move cannot be encoded: " + move.uci())

    def _encode_sliding(self, diff):
        for dir_idx, d in enumerate(QUEEN_DIRS):
            if d == 0: continue
            if diff % d == 0:
                step = diff // d
                if 1 <= step <= 7:
                    return dir_idx * 7 + (step - 1)
        return None

    def _encode_knight(self, diff):
        try:
            idx = KNIGHT_DIRS.index(diff)
        except ValueError:
            return None
        return 56 + idx

    def _encode_underpromo(self, diff, promo_piece, color):
        dirs = PROMOTION_DIRS[color]
        try:
            dir_idx = dirs.index(diff)
        except ValueError:
            return None

        try:
            promo_idx = PROMOTION_PIECES.index(promo_piece)
        except ValueError:
            return None

        return 64 + dir_idx * 3 + promo_idx

    # --------------------------------------------
    # index -> Move
    # --------------------------------------------
    def decode(self, action_id: int, board: chess.Board) -> chess.Move:
        from_sq = action_id // self.ACTIONS_PER_SQ
        local = action_id % self.ACTIONS_PER_SQ
        color = chess.WHITE if board.turn else chess.BLACK

        # Sliding
        if local < 56:
            dir_idx = local // 7
            step = (local % 7) + 1
            to_sq = from_sq + QUEEN_DIRS[dir_idx] * step
            return chess.Move(from_sq, to_sq)

        # Knight
        if local < 64:
            to_sq = from_sq + KNIGHT_DIRS[local - 56]
            return chess.Move(from_sq, to_sq)

        # Underpromotion
        offset = local - 64
        dir_idx = offset // 3
        promo_idx = offset % 3
        to_sq = from_sq + PROMOTION_DIRS[color][dir_idx]
        promo_piece = PROMOTION_PIECES[promo_idx]
        return chess.Move(from_sq, to_sq, promotion=promo_piece)

    # --------------------------------------------
    # Legal mask for policy learning
    # --------------------------------------------
    def legal_mask(self, board: chess.Board):
        mask = np.zeros(self.ACT_DIM, dtype=np.float32)
        for move in board.legal_moves:
            try:
                idx = self.encode(move, board)
                mask[idx] = 1.0
            except ValueError:
                continue
        return mask


def build_action_map(board: chess.Board, encoder: AlphaZeroActionEncoder):
    """Return encoded legal ids, id-to-move map, and legal mask for a board."""
    idxs = []
    idx_to_move = {}
    seen = set()
    legal_mask_np = np.zeros(encoder.ACT_DIM, dtype=np.float32)

    for move in board.legal_moves:
        try:
            idx = encoder.encode(move, board)
        except ValueError:
            continue
        if idx < 0 or idx >= encoder.ACT_DIM:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        idxs.append(idx)
        idx_to_move[idx] = move
        legal_mask_np[idx] = 1.0

    return idxs, idx_to_move, legal_mask_np
