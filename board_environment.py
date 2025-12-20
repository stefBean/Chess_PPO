# -*- coding: utf-8 -*-
"""Base environment for the game of chess.

This module contains a basic `Chess` environment. It relies heavily on the
`python-chess` package, which implements the underlying game mechanics.

"""

from typing import Tuple, Optional, Dict, List
from reward_shaping import RewardShaper

import random
import chess
import gymnasium as gym


class Chess(gym.Env):
    """Base environment for the game of chess.

    This env does not have a built-in opponent; moves are made for both the
    black and the white player in turn. At any given timestep, the env expects a
    legal move for the current player, otherwise an Error is raised.

    The agent is awarded a reward of +1 if the white player makes a winning move
    and -1 if the black player makes a winning move. All other rewards are zero.
    Since the winning player is always the last one to move, this is the only
    way to assign meaningful rewards based on the outcome of the game.

    Observations and actions are represented as `Board` and `Move` objects,
    respectively. The actual encoding as numpy arrays is left to wrapper classes
    for flexibility and separation of concerns (see the `wrappers` module for
    examples). As a consequence, the `observation_space` and `action_space`
    members are set to `None`.

    Observation:
        Type: chess.Board

        Note: Modifying the returned `Board` instance does not modify the
        internal state of this env.

    Actions:
        Type: chess.Move

    Reward:
        +1/-1 if white or black makes a winning move, respectively.

    Starting State:
        The usual initial board position for chess, as defined by FIDE

    Episode Termination:
        Either player wins.
        The game ends in a draw (e.g. stalemate, insufficient matieral,
        fifty-move rule, threefold repetition)

        Note: Surrendering is not an option.

    UPDATE 20.12.25 - Endgame learning only to maximize learning curve of agent without the need of a full game due to hardware capacities

    """

    # We deliberately use the render mode 'unicode' instead of the canonical
    # 'ansi' mode, since the output string contains non-ascii characters.
    meta = {
        'render.modes': ['unicode']
    }

    action_space = None
    observation_space = None

    reward_range = (-1, 1)

    """Maps game outcomes returned by `chess.Board.result()` to rewards."""
    _rewards: Dict[str, float] = {
        '*': 0.0,  # Game not over yet
        '1/2-1/2': 0.0,  # Draw
        '1-0': +1.0,  # White wins
        '0-1': -1.0,  # Black wins
    }

    def __init__(
            self,
            start_mode: str = "standard",
            endgame_max_extra_per_side: int = 3,
            endgame_min_extra_total: int = 2,
            require_pawn: bool = True,
            require_material_edge: bool = True,
    ) -> None:
    # def __init__(self) -> None:
        #: The underlying chess.Board instance that represents the game.
        self._board: Optional[chess.Board] = None
        self.shaper = RewardShaper()
        self.move_count = 0
        self.start_mode = start_mode
        self.endgame_max_extra_per_side = endgame_max_extra_per_side
        self.endgame_min_extra_total = endgame_min_extra_total
        self.require_pawn = require_pawn
        self.require_material_edge = require_material_edge

        #: Indicates whether the env has been reset since it has been created
        #: or the previous game has ended.
        self._ready: bool = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            import random, numpy as np
            np.random.seed(seed)
            random.seed(seed)

        # self._board = chess.Board()
        self.move_count = 0

        if self.start_mode == "endgame":
            self._board = self._generate_endgame_board()
        else:
            self._board = chess.Board()

        self._ready = True
        info = {}
        return self._observation(), info

    def step(self, action: chess.Move):

        assert self._ready, "Cannot call env.step() before calling reset()"

        if action not in self._board.legal_moves:
            raise ValueError(
                f"Illegal move {action} for board position {self._board.fen()}"
            )
        board_before = self._board.copy(stack=False)
        self._board.push(action)
        board_after = self._board
        self.move_count += 1

        reward_terminal = self._reward()
        terminated = board_after.is_game_over()
        truncated = False

        if not terminated:
            reward_shaping = self.shaper.shaped_reward(
                board_before,
                board_after,
                action,
                self.move_count
            )
        else:
            reward_shaping = 0.0

        reward = reward_terminal + reward_shaping

        observation = self._observation()
        info = {
            "fen": self._board.fen(),
            "result": self._board.result(),
            "legal_moves": list(map(str, self._board.legal_moves))
        }

        if terminated:
            self._ready = False

        return observation, reward, terminated, truncated, info

    def render(self, mode: str = 'unicode') -> Optional[str]:
        """
        Renders the current board position.

        The following render modes are supported:

        - unicode: Returns a string (str) representation of the current
          position, using non-ascii characters to represent individual pieces.

        Args:
            mode: (see above)
        """

        board = self._board if self._board else chess.Board()

        if mode == 'unicode':
            return board.unicode()

        else:
            return super(Chess, self).render(mode=mode)

    @property
    def legal_moves(self) -> List[chess.Move]:
        """Legal moves for the current player."""
        assert self._ready, "Cannot compute legal moves before calling reset()"

        return list(self._board.legal_moves)

    def _observation(self) -> chess.Board:
        """Returns the current board position."""
        return self._board.copy()

    def _reward(self) -> float:
        """Returns the reward for the most recent move."""
        result = self._board.result()
        reward = Chess._rewards[result]

        return reward

    # ------------------------------------------------------------------
    # Endgame initialization helpers
    # ------------------------------------------------------------------
    def _generate_endgame_board(self) -> chess.Board:
        """Create a random endgame position for self-play training.

        The generator limits material to a handful of pieces per side to
        focus the agent on converting endgame advantages. Boards that are
        immediately terminal (checkmate, stalemate, insufficient material) are
        discarded to ensure the agent always has to make at least one move.
        """

        max_attempts = 100
        piece_pool = [
            chess.QUEEN,
            chess.ROOK,
            chess.BISHOP,
            chess.KNIGHT,
            chess.PAWN,
        ]
        material_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }

        for _ in range(max_attempts):
            board = chess.Board(None)

            white_king = random.choice(chess.SQUARES)
            black_candidates = [
                sq for sq in chess.SQUARES
                if sq != white_king and chess.square_distance(sq, white_king) > 1
            ]
            if not black_candidates:
                continue
            black_king = random.choice(black_candidates)

            board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))

            occupied = {white_king, black_king}
            total_extra = 0
            placed_pawn = False

            for color in (chess.WHITE, chess.BLACK):
                extras = random.randint(1, self.endgame_max_extra_per_side)
                for _ in range(extras):
                    piece_type = random.choice(piece_pool)
                    square = self._sample_square(board, occupied, piece_type, color)
                    if square is None:
                        continue
                    board.set_piece_at(square, chess.Piece(piece_type, color))
                    occupied.add(square)
                    total_extra += 1
                    if piece_type == chess.PAWN:
                        placed_pawn = True

            # Require at least one non-king piece to avoid trivial king vs king
            if total_extra < self.endgame_min_extra_total:
                continue

            if self.require_pawn and not placed_pawn:
                continue

            board.turn = random.choice([chess.WHITE, chess.BLACK])

            if not board.is_valid():
                continue
            if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
                continue

            if self.require_material_edge:
                balance = 0
                for square, piece in board.piece_map().items():
                    if piece.piece_type not in material_values:
                        continue
                    value = material_values[piece.piece_type]
                    balance += value if piece.color == chess.WHITE else -value
                if abs(balance) < 1:
                    continue

            # Ensure at least one legal move exists for the side to play
            if any(True for _ in board.legal_moves):
                return board

        raise RuntimeError("Failed to generate a valid endgame board after many attempts")

    def _sample_square(
            self,
            board: chess.Board,
            occupied: set,
            piece_type: chess.PieceType,
            color: chess.Color,
    ) -> Optional[int]:
        """Choose a random legal square for the given piece type."""

        candidates = []
        for sq in chess.SQUARES:
            if sq in occupied:
                continue
            if piece_type == chess.PAWN:
                rank = chess.square_rank(sq)
                if rank in (0, 7):
                    continue
            candidates.append(sq)

        if not candidates:
            return None

        random.shuffle(candidates)
        for sq in candidates:
            board.set_piece_at(sq, chess.Piece(piece_type, color))
            if board.is_valid():
                return sq
            board.remove_piece_at(sq)

        return None

    def _repr_svg_(self) -> str:
        """Returns an SVG representation of the current board position"""
        board = self._board if self._board else chess.Board()
        return str(board._repr_svg_())