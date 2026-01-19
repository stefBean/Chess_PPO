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
import itertools


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
            endgame_scenarios: Optional[List[Dict[str, List[chess.PieceType]]]] = None,
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
        self.endgame_scenarios = endgame_scenarios
        self.endgame_max_extra_per_side = endgame_max_extra_per_side
        self.endgame_min_extra_total = endgame_min_extra_total
        self.require_pawn = require_pawn
        self.require_material_edge = require_material_edge
        self.max_ply: int | None = 200 #200
        self.no_progress = 0
        self._full_endgame_scenarios = self._enumerate_endgame_scenarios()
        self.curriculum_stage = 0
        self._curriculum_sorted = None
        self._curriculum_stage_fracs = [0.10, 0.30, 0.60, 1.0]

        #: Indicates whether the env has been reset since it has been created
        #: or the previous game has ended.
        self._ready: bool = False

    def reset(self,*, seed=None, options=None):
        if seed is not None:
            import random, numpy as np
            np.random.seed(seed)
            random.seed(seed)

        # self._board = chess.Board()
        self.move_count = 0
        options = options or {}
        # self.max_ply = int(options.get("max_ply", self.max_ply))
        max_ply_option = options.get("max_ply", self.max_ply)
        if max_ply_option is None:
            self.max_ply = None
        else:
            self.max_ply = int(max_ply_option)
        desired_color = options.get("agent_color")
        self.no_progress = 0
        stage = options.get("curriculum_stage")
        if stage is not None:
            try:
                self.curriculum_stage = int(stage)
            except (TypeError, ValueError):
                pass

        if self.start_mode == "endgame":
            self._board = self._generate_endgame_board()
        elif self.start_mode == "curriculum_endgame":
            self._board = self._generate_curriculum_endgame_board()
        else:
            self._board = chess.Board()

        if desired_color in (chess.WHITE, chess.BLACK):
            self._board.turn = desired_color

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
        mover = board_before.turn
        self._board.push(action)
        board_after = self._board
        self.move_count += 1
        is_capture = board_before.is_capture(action)
        moved_piece = board_before.piece_at(action.from_square)
        is_pawn = (moved_piece is not None and moved_piece.piece_type == chess.PAWN)

        if is_capture or is_pawn:
            self.no_progress = 0
        else:
            self.no_progress += 1

        truncated = False
        forced_draw = False
        #if self.move_count >= self.max_ply:
        max_ply_limit = self.max_ply is not None and self.max_ply > 0
        if max_ply_limit and self.move_count >= self.max_ply:
            terminated = False
            forced_draw = True
            truncated = True
        else:
            terminated = board_after.is_game_over()

        # if self.move_count >= self.max_ply:
        #    truncated = True

        # reward_terminal = self._reward()
        # terminated = board_after.is_game_over()
        # truncated = False
        reward_terminal = 0.0
        reward_terminal_bonus = 0.0
        terminal_reason = None
        done = terminated or truncated
        if done:
        #if terminated:
            #res = board_after.result()
            #if res == "1-0":
            #    reward_terminal = +1.0 if mover == chess.WHITE else -1.0
            #elif res == "0-1":
            #    reward_terminal = +1.0 if mover == chess.BLACK else -1.0
            #else:
            #    reward_terminal = 0.0
            if forced_draw:
                reward_terminal = 0.0
                terminal_reason = "max_ply"
            else:
                res = board_after.result()
                if res == "1-0":
                    reward_terminal = +1.0 if mover == chess.WHITE else -1.0
                elif res == "0-1":
                    reward_terminal = +1.0 if mover == chess.BLACK else -1.0
                else:
                    reward_terminal = 0.0
                if board_after.is_checkmate():
                    terminal_reason = "checkmate"
                elif board_after.is_stalemate():
                    terminal_reason = "stalemate"
                elif board_after.is_insufficient_material():
                    terminal_reason = "insufficient_material"
                else:
                    terminal_reason = "rule_termination"
                if reward_terminal != 0.0:
                    reward_terminal_bonus = self.shaper.terminal_bonus(board_after)
                    reward_terminal += reward_terminal_bonus

        if not done: #terminated:
            # reward_shaping = self.shaper.shaped_reward
            reward_breakdown = self.shaper.shaped_reward_breakdown(
                board_before,
                board_after,
                action,
                self.move_count,
            )
            reward_shaping = reward_breakdown["total"]
        else:
            reward_breakdown = self.shaper._blank_breakdown()
            reward_shaping = 0.0
            if reward_terminal_bonus != 0.0:
                reward_breakdown["terminal_bonus"] = reward_terminal_bonus
                reward_breakdown["total"] = reward_terminal_bonus

        reward = reward_terminal + reward_shaping

        observation = self._observation()
        result_string = "1/2-1/2" if forced_draw else self._board.result()
        info = {
            "result": result_string,
            "legal_moves": list(map(str, self._board.legal_moves)),
            "terminal_reason": terminal_reason,
            "forced_draw": forced_draw,
            "reward_terminal": reward_terminal,
            "reward_terminal_bonus": reward_terminal_bonus,
            "reward_shaping": reward_shaping,
            "reward_total": reward,
            "reward_breakdown": reward_breakdown,
        }

        if done:
        # if terminated or truncated:
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

    def _generate_curriculum_endgame_board(self) -> chess.Board:
        """Create focused endgame tasks (e.g., K+P vs K, K+B+P vs K)."""

        # scenarios = self.endgame_scenarios or [
        #     {"white": [chess.KING, chess.PAWN], "black": [chess.KING]},
        #     {"white": [chess.KING, chess.BISHOP, chess.PAWN], "black": [chess.KING]},
        #     {"white": [chess.KING, chess.KNIGHT, chess.PAWN], "black": [chess.KING]},
        #     {"white": [chess.KING, chess.ROOK, chess.PAWN], "black": [chess.KING]},
        #     {"white": [chess.KING, chess.QUEEN], "black": [chess.KING, chess.PAWN]},
        #     {"white": [chess.KING, chess.ROOK], "black": [chess.KING, chess.PAWN]},
        #     {"white": [chess.KING, chess.QUEEN], "black": [chess.KING, chess.ROOK]},
        # ]
        """Create focused endgame tasks (e.g., K+P vs K, K+B+P vs K).

                This version enumerates all material configurations up to the configured
                `endgame_max_extra_per_side` so the agent cycles through every classical
                endgame permutation instead of a small handcrafted list.
                """

        scenarios = self.endgame_scenarios or self._full_endgame_scenarios
        if self._curriculum_sorted is None:
            # scenarios = self.endgame_scenarios or self._enumerate_endgame_scenarios()
            self._curriculum_sorted = sorted(scenarios, key=self._scenario_difficulty)

        stage = max(0, min(self.curriculum_stage, len(self._curriculum_stage_fracs) - 1))
        frac = self._curriculum_stage_fracs[stage]
        k = max(1, int(len(self._curriculum_sorted) * frac))
        candidate_scenarios = self._curriculum_sorted[:k]
        difficulty_bias = 0.6 + 0.4 * stage
        scenario_weights = [
            (self._scenario_difficulty(scenario) + 1e-3) ** difficulty_bias
            for scenario in candidate_scenarios
        ]

        # Also allow mirroring colors for diversity
        for _ in range(400):
            scenario = random.choices(candidate_scenarios, weights=scenario_weights, k=1)[0]
            if random.random() < 0.5:
                scenario = {"white": scenario["black"], "black": scenario["white"]}

            board = chess.Board(None)
            occupied = set()

            # Place kings first with distance
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
            occupied.update([white_king, black_king])

            if not self._place_piece_set(board, occupied, scenario["white"], chess.WHITE):
                continue
            if not self._place_piece_set(board, occupied, scenario["black"], chess.BLACK):
                continue

            if not board.is_valid():
                continue
            if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
                continue
            if any(True for _ in board.legal_moves):
                board.turn = random.choice([chess.WHITE, chess.BLACK])
                return board

        raise RuntimeError("Failed to generate a curriculum endgame board")

    def _place_piece_set(self, board: chess.Board, occupied: set, pieces: List[chess.PieceType],
                         color: chess.Color) -> bool:
        """Place the given list of pieces randomly on the board."""
        for piece_type in pieces:
            if piece_type == chess.KING:
                continue
            square = self._sample_square(board, occupied, piece_type, color)
            if square is None:
                return False
            board.set_piece_at(square, chess.Piece(piece_type, color))
            occupied.add(square)
        return True

    def _enumerate_endgame_scenarios(self) -> List[Dict[str, List[chess.PieceType]]]:
        """Enumerate all compact endgame material configurations.

        Scenarios cover every combination of up to `endgame_max_extra_per_side`
        extra pieces for each color drawn from the canonical endgame set
        (queen, rook, bishop, knight, pawn). Kings are always included.
        """

        piece_pool = [
            chess.PAWN,
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
        ]
        scenarios: List[Dict[str, List[chess.PieceType]]] = []

        def expand_combo(count: int):
            for combo in itertools.combinations_with_replacement(piece_pool, count):
                yield list(combo)

        for white_count in range(0, self.endgame_max_extra_per_side + 1):
            for black_count in range(0, self.endgame_max_extra_per_side + 1):
                if white_count + black_count < self.endgame_min_extra_total:
                    continue
                for white_combo in expand_combo(white_count):
                    for black_combo in expand_combo(black_count):
                        if self.require_pawn and chess.PAWN not in (white_combo + black_combo):
                            continue
                        scenarios.append(
                            {
                                "white": [chess.KING] + white_combo,
                                "black": [chess.KING] + black_combo,
                            }
                        )

        # simple fallback scenario in case strict filters remove all
        if not scenarios:
            scenarios.append({"white": [chess.KING, chess.PAWN], "black": [chess.KING]})

        return scenarios

    def _scenario_difficulty(self, scenario) -> float:
        values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9
        }
        w = sum(values.get(p, 0) for p in scenario["white"] if p != chess.KING)
        b = sum(values.get(p, 0) for p in scenario["black"] if p != chess.KING)
        return (w + b) + 0.3 * abs(w - b)

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