import argparse
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any


def load_games(log_path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    """Load recorded training games from a JSONL log file."""
    games: List[Dict[str, Any]] = []
    if not log_path.exists():
        raise FileNotFoundError(f"No game log found at {log_path}")

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                games.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit is not None and len(games) >= limit:
                break
    return games


def format_moves(moves: Iterable[Dict[str, Any]]) -> str:
    """Return a human-readable string of moves with SAN and UCI."""
    parts = []
    for move in moves:
        ply = move.get("ply")
        parts.append(
            f"{ply:>3}: {move.get('player','?'):>5} | {move.get('san','?'):8} ({move.get('uci','?')}) via {move.get('by','agent')}"
        )
    return "\n".join(parts)


def describe_game(game: Dict[str, Any]) -> str:
    """Render a single game summary for inspection."""
    header = (
        f"Episode {game.get('episode','?')}: agent as {game.get('agent_color','?')}, "
        f"result {game.get('result','?')} ({game.get('terminal_reason')})"
    )
    start_fen = game.get("start_fen", "<unknown>")
    moves_text = format_moves(game.get("moves", []))
    return f"{header}\nStart FEN: {start_fen}\nMoves:\n{moves_text}\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect recorded training games.")
    parser.add_argument(
        "log_path",
        type=Path,
        help="Path to the games.jsonl file produced during training (e.g., runs/SelfPlay_.../games.jsonl).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of most recent games to display.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    games = load_games(args.log_path, limit=args.limit)
    if not games:
        print("No games found.")
        return

    for game in games:
        print(describe_game(game))
        print("-" * 40)


if __name__ == "__main__":
    main()