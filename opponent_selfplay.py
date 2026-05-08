import copy
import random
from collections import deque
from typing import Callable, Deque, Dict, Tuple

import numpy as np
import torch
import chess

from action_encoding import AlphaZeroActionEncoder, build_action_map


class SelfPlayOpponent:
    """A frozen copy of the current agent policy used for self-play.

    The opponent's actor weights are refreshed from snapshots of the current
    agent policy. Action selection samples only from encoded legal chess moves.
    """

    def __init__(
        self,
        agent_actor: torch.nn.Module,
        encoder: AlphaZeroActionEncoder,
        flatten_obs: Callable[[np.ndarray], np.ndarray],
        device: torch.device,
        refresh_interval: int = 4096,
        pool_size: int = 6,
        sample_past_prob: float = 0.35,
        temperature: float = 1.15,
    ) -> None:
        self.encoder = encoder
        self.flatten_obs = flatten_obs
        self.device = device
        self.refresh_interval = refresh_interval
        self.sample_past_prob = sample_past_prob
        self.temperature = temperature

        # Hold a frozen copy of the actor network for inference only.
        self.actor = copy.deepcopy(agent_actor).to(self.device)
        self.opponent_actor = copy.deepcopy(agent_actor).to(self.device)

        initial_state = copy.deepcopy(agent_actor.state_dict())
        self.latest_state = initial_state
        self.snapshots: Deque[dict] = deque([initial_state], maxlen=pool_size)
        self.last_snapshot_step = 0

    def begin_episode(self) -> None:
        """Choose an opponent snapshot for the upcoming episode."""
        if len(self.snapshots) > 1 and random.random() < self.sample_past_prob:
            state_dict = random.choice(list(self.snapshots)[:-1])
        else:
            state_dict = self.latest_state

        self.opponent_actor.load_state_dict(copy.deepcopy(state_dict))
        self.opponent_actor.eval()

    # def maybe_refresh(self, agent_actor: torch.nn.Module, global_step: int) -> None:
    #     """Refresh the snapshot pool on a fixed cadence."""
    #     if global_step - self.last_snapshot_step < self.refresh_interval:
    #         return
    #
    #     state_dict = copy.deepcopy(agent_actor.state_dict())
    #     self.latest_state = state_dict
    #     self.snapshots.append(state_dict)
    #     self.actor.load_state_dict(state_dict)
    #     self.last_snapshot_step = global_step

    def load_snapshot(self, snapshot_state_dict) -> None:
        if snapshot_state_dict is None:
            return
        snapshot_copy = copy.deepcopy(snapshot_state_dict)
        self.latest_state = snapshot_copy
        self.snapshots.append(snapshot_copy)
        self.opponent_actor.load_state_dict(snapshot_copy, strict=True)

    def select_move(
        self, board: chess.Board, observation: np.ndarray
    ) -> Tuple[chess.Move, Dict[str, np.ndarray]]:
        """Sample a legal move using the frozen actor policy.

        Returns both the chosen move and the computed action map so callers can
        reuse it if needed.
        """
        state_np = self.flatten_obs(observation)
        idxs, idx_to_move, legal_mask_np = build_action_map(board, self.encoder)

        if not idxs:
            return None, {"legal_mask": None, "idx_to_move": idx_to_move}

        state = torch.from_numpy(state_np).float().to(self.device).unsqueeze(0)
        legal_mask = torch.from_numpy(legal_mask_np).to(self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.opponent_actor(state) / max(self.temperature, 1e-3)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            legal_ids = torch.nonzero(legal_mask[0] > 0, as_tuple=False).squeeze(-1)
            legal_logits = logits[0, legal_ids]
            dist = torch.distributions.Categorical(logits=legal_logits)
            sampled_offset = dist.sample()
            action = legal_ids[sampled_offset]

        action_id = int(action.item())
        move = idx_to_move.get(action_id)
        if move is None:
            raise RuntimeError(
                "Self-play opponent sampled action without a move mapping; "
                f"sampled_action_id={action_id}; "
                f"legal_ids={idxs}; "
                f"fen={board.fen()}; "
                f"legal_uci_moves={[m.uci() for m in board.legal_moves]}"
            )

        return move, {"legal_mask": legal_mask_np, "idx_to_move": idx_to_move}


class FrozenPolicyOpponent:
    """Inference-only policy opponent with legal-id-only action sampling."""

    def __init__(
        self,
        actor: torch.nn.Module,
        encoder: AlphaZeroActionEncoder,
        flatten_obs: Callable[[np.ndarray], np.ndarray],
        device: torch.device,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> None:
        self.encoder = encoder
        self.flatten_obs = flatten_obs
        self.device = device
        self.temperature = temperature
        self.deterministic = deterministic
        self.actor = copy.deepcopy(actor).to(self.device)
        self.actor.eval()
        for param in self.actor.parameters():
            param.requires_grad_(False)

    def select_move(
        self, board: chess.Board, observation: np.ndarray
    ) -> Tuple[chess.Move, Dict[str, np.ndarray]]:
        state_np = self.flatten_obs(observation)
        idxs, idx_to_move, legal_mask_np = build_action_map(board, self.encoder)
        if not idxs:
            return None, {"legal_mask": None, "idx_to_move": idx_to_move}

        state = torch.from_numpy(state_np).float().to(self.device).unsqueeze(0)
        legal_mask = torch.from_numpy(legal_mask_np).to(self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.actor(state) / max(self.temperature, 1e-6)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            legal_ids = torch.nonzero(legal_mask[0] > 0, as_tuple=False).squeeze(-1)
            legal_logits = logits[0, legal_ids]
            if self.deterministic:
                sampled_offset = torch.argmax(legal_logits)
            else:
                dist = torch.distributions.Categorical(logits=legal_logits)
                sampled_offset = dist.sample()
            action = legal_ids[sampled_offset]

        action_id = int(action.item())
        move = idx_to_move.get(action_id)
        if move is None:
            raise RuntimeError(
                "Frozen policy opponent sampled action without a move mapping; "
                f"sampled_action_id={action_id}; "
                f"legal_ids={idxs}; "
                f"fen={board.fen()}; "
                f"legal_uci_moves={[m.uci() for m in board.legal_moves]}"
            )
        return move, {"legal_mask": legal_mask_np, "idx_to_move": idx_to_move}


# Snapshot pool for current agent policies.
class OpponentPool:
    def __init__(self, max_size: int = 8, p_latest: float = 0.7, seed: int = 0):
        self.max_size = max_size
        self.p_latest = p_latest
        self.rng = np.random.default_rng(seed)
        self.snapshots = []  # list of state_dicts (CPU tensors)

    def add(self, actor: torch.nn.Module) -> None:
        # store on CPU to avoid GPU memory growth
        state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
        self.snapshots.append(state)
        if len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)

    def sample(self):
        if not self.snapshots:
            return None
        if len(self.snapshots) == 1:
            return self.snapshots[-1]
        if self.rng.random() < self.p_latest:
            return self.snapshots[-1]
        idx = self.rng.integers(0, len(self.snapshots) - 1)  # exclude latest
        return self.snapshots[int(idx)]