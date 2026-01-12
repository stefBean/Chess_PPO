from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import chess


@dataclass(frozen=True)
class BoardAnalysis:
    material_white: int
    material_black: int
    material_balance: int
    total_material: int
    endgame: bool
    endgame_reason: str
    mobility: int
    in_check: bool
    pawn_files_white: Dict[str, int]
    pawn_files_black: Dict[str, int]
    doubled_pawns_white: int
    doubled_pawns_black: int
    isolated_pawns_white: int
    isolated_pawns_black: int


class BoardAnalyzer:
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }

    def analyze(self, board: chess.Board) -> BoardAnalysis:
        material_white = 0
        material_black = 0
        for piece_type, value in self.piece_values.items():
            material_white += len(board.pieces(piece_type, chess.WHITE)) * value
            material_black += len(board.pieces(piece_type, chess.BLACK)) * value
        total_material = material_white + material_black
        endgame, endgame_reason = self._endgame_status(board, total_material)

        pawn_files_white = self._pawn_files(board, chess.WHITE)
        pawn_files_black = self._pawn_files(board, chess.BLACK)

        doubled_white = sum(max(0, count - 1) for count in pawn_files_white.values())
        doubled_black = sum(max(0, count - 1) for count in pawn_files_black.values())
        isolated_white = self._isolated_pawn_count(pawn_files_white)
        isolated_black = self._isolated_pawn_count(pawn_files_black)

        return BoardAnalysis(
            material_white=material_white,
            material_black=material_black,
            material_balance=material_white - material_black,
            total_material=total_material,
            endgame=endgame,
            endgame_reason=endgame_reason,
            mobility=board.legal_moves.count(),
            in_check=board.is_check(),
            pawn_files_white=pawn_files_white,
            pawn_files_black=pawn_files_black,
            doubled_pawns_white=doubled_white,
            doubled_pawns_black=doubled_black,
            isolated_pawns_white=isolated_white,
            isolated_pawns_black=isolated_black,
        )

    def describe_board(self, board: chess.Board) -> str:
        analysis = self.analyze(board)
        return f"{board.unicode()}\n\n{self.format_analysis(analysis)}"

    def format_analysis(self, analysis: BoardAnalysis) -> str:
        white_files = self._format_pawn_files(analysis.pawn_files_white)
        black_files = self._format_pawn_files(analysis.pawn_files_black)
        return (
            "Analysis:\n"
            f"  Material (W/B): {analysis.material_white}/{analysis.material_black} "
            f"(balance {analysis.material_balance:+d})\n"
            f"  Total material: {analysis.total_material}\n"
            f"  Endgame: {analysis.endgame} ({analysis.endgame_reason})\n"
            f"  Mobility (side to move): {analysis.mobility}\n"
            f"  In check: {analysis.in_check}\n"
            f"  Pawn files W: {white_files}\n"
            f"  Pawn files B: {black_files}\n"
            f"  Doubled pawns W/B: {analysis.doubled_pawns_white}/{analysis.doubled_pawns_black}\n"
            f"  Isolated pawns W/B: {analysis.isolated_pawns_white}/{analysis.isolated_pawns_black}"
        )

    def _pawn_files(self, board: chess.Board, color: bool) -> Dict[str, int]:
        counts = {file_name: 0 for file_name in chess.FILE_NAMES}
        for square in board.pieces(chess.PAWN, color):
            file_name = chess.FILE_NAMES[chess.square_file(square)]
            counts[file_name] += 1
        return counts

    def _isolated_pawn_count(self, pawn_files: Dict[str, int]) -> int:
        isolated = 0
        file_names = list(chess.FILE_NAMES)
        for idx, file_name in enumerate(file_names):
            count = pawn_files[file_name]
            if count == 0:
                continue
            left = pawn_files[file_names[idx - 1]] if idx > 0 else 0
            right = pawn_files[file_names[idx + 1]] if idx < len(file_names) - 1 else 0
            if left == 0 and right == 0:
                isolated += count
        return isolated

    def _format_pawn_files(self, pawn_files: Dict[str, int]) -> str:
        return " ".join(f"{file_name}:{count}" for file_name, count in pawn_files.items())

    def _endgame_status(self, board: chess.Board, total_material: int) -> tuple[bool, str]:
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        if total_material <= 14:
            return True, "material<=14"
        if queens == 0 and total_material <= 20:
            return True, "no_queens_material<=20"
        return False, "midgame"