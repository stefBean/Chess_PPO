import copy
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import chess

from action_encoding import AlphaZeroActionEncoder


class SelfPlayOpponent:
    """A frozen copy of the current PPO policy used for self-play.

    The opponent's actor weights are periodically refreshed from the learning
    agent to provide a moving target while keeping the opponent stable between
    refreshes. Action selection mirrors the masking logic used by the agent so
    that only legal chess moves are sampled.
    """

    def __init__(
        self,
        agent_actor: torch.nn.Module,
        encoder: AlphaZeroActionEncoder,
        flatten_obs: Callable[[np.ndarray], np.ndarray],
        device: torch.device,
        refresh_interval: int = 4096,
    ) -> None:
        self.encoder = encoder
        self.flatten_obs = flatten_obs
        self.device = device
        self.refresh_interval = refresh_interval
        self.steps_since_refresh = 0

        # Hold a frozen copy of the actor network for inference only.
        self.actor = copy.deepcopy(agent_actor).to(self.device)
        self.actor.eval()

    def maybe_refresh(self, agent_actor: torch.nn.Module) -> None:
        """Refresh the frozen actor weights after a certain number of steps."""
        if self.steps_since_refresh >= self.refresh_interval:
            self.actor.load_state_dict(copy.deepcopy(agent_actor.state_dict()))
            self.steps_since_refresh = 0

    def select_move(
        self, board: chess.Board, observation: np.ndarray
    ) -> Tuple[chess.Move, Dict[str, np.ndarray]]:
        """Sample a legal move using the frozen actor policy.

        Returns both the chosen move and the computed action map so callers can
        reuse it if needed.
        """
        state_np = self.flatten_obs(observation)
        idxs, idx_to_move = build_action_map(board, self.encoder)

        if not idxs:
            return None, {"legal_mask": None, "idx_to_move": idx_to_move}

        legal_mask_np = np.zeros(self.encoder.ACT_DIM, dtype=np.float32)
        legal_mask_np[idxs] = 1.0

        state = torch.from_numpy(state_np).float().to(self.device).unsqueeze(0)
        legal_mask = torch.from_numpy(legal_mask_np).to(self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.actor(state)
            masked_logits = logits.masked_fill(legal_mask == 0, -1e9)
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()

        move = idx_to_move.get(int(action.item()))
        if move is None:
            legal_moves = list(board.legal_moves)
            move = np.random.choice(legal_moves)

        self.steps_since_refresh += 1
        return move, {"legal_mask": legal_mask_np, "idx_to_move": idx_to_move}


def build_action_map(
    board: chess.Board, encoder: AlphaZeroActionEncoder
) -> Tuple[List[int], Dict[int, chess.Move]]:
    """Return the encoded legal action ids and mapping to chess.Move objects."""
    legal_moves = list(board.legal_moves)
    idxs: List[int] = []
    idx_to_move: Dict[int, chess.Move] = {}

    for mv in legal_moves:
        try:
            idx = encoder.encode(mv, board)
            idxs.append(idx)
            idx_to_move[idx] = mv
        except Exception:
            continue
    return idxs, idx_to_move
