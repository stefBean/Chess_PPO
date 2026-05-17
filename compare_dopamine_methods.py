"""Print equal-step comparisons for the two dopamine training regimes.

The script compares:
  * return curves from ``runs/*/games.jsonl``;
  * win-rate curves from the same game logs;
  * fixed evaluation checkpoints from ``evaluation_results/dopamine_comparison*.json``.

It intentionally samples both training regimes on the same step grid so a longer
run does not get an unfair advantage in the printed curve summaries.
"""

from __future__ import annotations

import json
import os
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

TRAINING_METHODS = {
    "dopamine_self": "Dopamine self-play",
    "dopamine_vs_ppo": "Dopamine vs PPO",
}
RUN_PREFIXES = {
    "dopamine_self": "DopamineSelf_",
    "dopamine_vs_ppo": "DopamineVsPPO_",
}


@dataclass(frozen=True)
class EpisodePoint:
    step: int
    episode: int
    episode_return: float
    rolling_return: float
    rolling_win_rate: float


@dataclass(frozen=True)
class CurveSample:
    step: int
    self_return: float
    vs_ppo_return: float
    return_delta: float
    self_win_rate: float
    vs_ppo_win_rate: float
    win_rate_delta: float


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[WARN] Ignoring invalid {name}={raw!r}; using {default}.")
        return default


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed JSON in {path}:{line_number}: {exc}")


def _discover_run_dirs(runs_dir: Path, method: str) -> List[Path]:
    prefix = RUN_PREFIXES[method]
    if not runs_dir.exists():
        return []
    return sorted(
        [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)],
        key=lambda p: p.stat().st_mtime,
    )


def _selected_run_dirs(runs_dir: Path, method: str) -> List[Path]:
    env_name = "DOPAMINE_SELF_RUN" if method == "dopamine_self" else "DOPAMINE_VS_PPO_RUN"
    explicit = os.getenv(env_name, "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"{env_name} points to missing run directory: {path}")
        return [path]

    run_dirs = _discover_run_dirs(runs_dir, method)
    if not run_dirs:
        return []
    if os.getenv("COMPARE_ALL_RUNS", "0") == "1":
        return run_dirs
    return [run_dirs[-1]]


def _load_episode_curve(run_dirs: Sequence[Path], rolling_window: int) -> List[EpisodePoint]:
    rows: List[dict] = []
    for run_dir in run_dirs:
        game_log = run_dir / "games.jsonl"
        if not game_log.exists():
            print(f"[WARN] No games.jsonl found for {run_dir}; skipping.")
            continue
        rows.extend(_read_jsonl(game_log))

    points: List[EpisodePoint] = []
    returns: deque[float] = deque(maxlen=max(1, rolling_window))
    wins: deque[float] = deque(maxlen=max(1, rolling_window))
    cumulative_steps = 0
    last_step = 0

    for fallback_episode, row in enumerate(rows, start=1):
        episode = int(row.get("episode", fallback_episode))
        episode_return = float(row.get("episode_return", 0.0))
        episode_length = int(row.get("episode_length", 0) or 0)
        logged_step = row.get("training_timestep")
        if logged_step is None:
            cumulative_steps += max(1, episode_length)
            step = cumulative_steps
        else:
            step = int(logged_step)
            if step <= last_step:
                step = last_step + max(1, episode_length)
        last_step = step

        returns.append(episode_return)
        wins.append(1.0 if row.get("winner") == "agent" else 0.0)
        points.append(
            EpisodePoint(
                step=step,
                episode=episode,
                episode_return=episode_return,
                rolling_return=float(mean(returns)),
                rolling_win_rate=float(mean(wins)),
            )
        )
    return points


def _steps_grid(max_shared_step: int, num_points: int, explicit_steps: str = "") -> List[int]:
    if explicit_steps.strip():
        parsed = sorted({int(item.strip()) for item in explicit_steps.split(",") if item.strip()})
        return [step for step in parsed if 0 < step <= max_shared_step]
    if max_shared_step <= 0:
        return []
    num_points = max(1, num_points)
    if num_points == 1:
        return [max_shared_step]
    return sorted({max(1, round(max_shared_step * idx / num_points)) for idx in range(1, num_points + 1)})


def _point_at_or_before(points: Sequence[EpisodePoint], step: int) -> EpisodePoint | None:
    selected = None
    for point in points:
        if point.step <= step:
            selected = point
        else:
            break
    return selected


def build_curve_samples(curves: Dict[str, List[EpisodePoint]], steps: Sequence[int]) -> List[CurveSample]:
    samples: List[CurveSample] = []
    for step in steps:
        self_point = _point_at_or_before(curves["dopamine_self"], step)
        vs_ppo_point = _point_at_or_before(curves["dopamine_vs_ppo"], step)
        if self_point is None or vs_ppo_point is None:
            continue
        samples.append(
            CurveSample(
                step=step,
                self_return=self_point.rolling_return,
                vs_ppo_return=vs_ppo_point.rolling_return,
                return_delta=vs_ppo_point.rolling_return - self_point.rolling_return,
                self_win_rate=self_point.rolling_win_rate,
                vs_ppo_win_rate=vs_ppo_point.rolling_win_rate,
                win_rate_delta=vs_ppo_point.rolling_win_rate - self_point.rolling_win_rate,
            )
        )
    return samples


def _format_float(value: float, width: int = 8, precision: int = 3) -> str:
    return f"{value:{width}.{precision}f}"


def print_curve_comparison(samples: Sequence[CurveSample], rolling_window: int) -> None:
    print("\n=== Equal-step training curve comparison ===")
    print(f"Rolling window: {rolling_window} completed games")
    if not samples:
        print("No overlapping training steps found for the two methods.")
        return
    print(
        "step".rjust(10),
        "self_ret".rjust(10),
        "vs_ppo_ret".rjust(10),
        "Δret".rjust(10),
        "self_win".rjust(10),
        "vs_ppo_win".rjust(10),
        "Δwin".rjust(10),
    )
    for sample in samples:
        print(
            f"{sample.step:10d}",
            _format_float(sample.self_return, 10),
            _format_float(sample.vs_ppo_return, 10),
            _format_float(sample.return_delta, 10),
            _format_float(sample.self_win_rate, 10),
            _format_float(sample.vs_ppo_win_rate, 10),
            _format_float(sample.win_rate_delta, 10),
        )


def _checkpoint_step(path: Path, payload: dict) -> int | None:
    for source in (payload.get("config", {}), payload):
        for key in ("training_step", "checkpoint_step", "step", "timestep"):
            value = source.get(key) if isinstance(source, dict) else None
            if value is not None:
                return int(value)
    match = re.search(r"(?:step|checkpoint)[_-]?(\d+)", path.stem)
    return int(match.group(1)) if match else None


def _load_evaluation_files(evaluation_dir: Path) -> List[tuple[int | None, Path, dict]]:
    if not evaluation_dir.exists():
        return []
    loaded = []
    for path in sorted(evaluation_dir.glob("dopamine_comparison*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[WARN] Skipping malformed evaluation file {path}: {exc}")
            continue
        loaded.append((_checkpoint_step(path, payload), path, payload))
    return sorted(loaded, key=lambda item: (-1 if item[0] is None else item[0], item[1].name))


def print_fixed_evaluation_checkpoints(evaluation_dir: Path) -> None:
    print("\n=== Fixed evaluation checkpoints ===")
    evaluation_files = _load_evaluation_files(evaluation_dir)
    if not evaluation_files:
        print(f"No evaluation JSON files found in {evaluation_dir}.")
        print("Run: SEED=0 EVAL_GAMES=20 python evaluate_dopamine.py")
        return

    for step, path, payload in evaluation_files:
        checkpoint_label = f"step {step}" if step is not None else "final/current checkpoint"
        print(f"\n[{checkpoint_label}] {path}")
        aggregates = payload.get("aggregates", [])
        by_opponent: Dict[str, Dict[str, dict]] = {}
        for aggregate in aggregates:
            agent = aggregate.get("agent")
            opponent = aggregate.get("opponent")
            if agent in TRAINING_METHODS and opponent:
                by_opponent.setdefault(opponent, {})[agent] = aggregate

        if not by_opponent:
            print("  No dopamine_self/dopamine_vs_ppo aggregate rows found.")
            continue

        print(
            "opponent".ljust(28),
            "self_win".rjust(9),
            "vsppo_win".rjust(9),
            "Δwin".rjust(9),
            "self_ret".rjust(10),
            "vsppo_ret".rjust(10),
            "Δret".rjust(10),
        )
        for opponent in sorted(by_opponent):
            self_row = by_opponent[opponent].get("dopamine_self")
            vs_ppo_row = by_opponent[opponent].get("dopamine_vs_ppo")
            if not self_row or not vs_ppo_row:
                continue
            self_win = float(self_row.get("win_rate", 0.0))
            vs_ppo_win = float(vs_ppo_row.get("win_rate", 0.0))
            self_ret = float(self_row.get("average_return", 0.0))
            vs_ppo_ret = float(vs_ppo_row.get("average_return", 0.0))
            print(
                opponent.ljust(28),
                _format_float(self_win, 9),
                _format_float(vs_ppo_win, 9),
                _format_float(vs_ppo_win - self_win, 9),
                _format_float(self_ret, 10),
                _format_float(vs_ppo_ret, 10),
                _format_float(vs_ppo_ret - self_ret, 10),
            )


def main() -> None:
    runs_dir = Path(os.getenv("RUNS_DIR", "runs"))
    evaluation_dir = Path(os.getenv("EVALUATION_DIR", "evaluation_results"))
    rolling_window = _env_int("COMPARE_WINDOW", 50)
    num_points = _env_int("COMPARE_POINTS", 10)
    explicit_steps = os.getenv("COMPARE_STEPS", "")

    curves: Dict[str, List[EpisodePoint]] = {}
    for method, label in TRAINING_METHODS.items():
        run_dirs = _selected_run_dirs(runs_dir, method)
        if not run_dirs:
            print(f"[WARN] No run directories found for {label} in {runs_dir}.")
            curves[method] = []
            continue
        print(f"Using {label} run(s): {', '.join(str(path) for path in run_dirs)}")
        curves[method] = _load_episode_curve(run_dirs, rolling_window)
        if curves[method]:
            print(
                f"  {len(curves[method])} games, "
                f"steps {curves[method][0].step}..{curves[method][-1].step}"
            )
        else:
            print("  No games loaded.")

    if curves["dopamine_self"] and curves["dopamine_vs_ppo"]:
        max_shared_step = min(curves["dopamine_self"][-1].step, curves["dopamine_vs_ppo"][-1].step)
        steps = _steps_grid(max_shared_step, num_points, explicit_steps)
        samples = build_curve_samples(curves, steps)
        print_curve_comparison(samples, rolling_window)
    else:
        print_curve_comparison([], rolling_window)

    print_fixed_evaluation_checkpoints(evaluation_dir)


if __name__ == "__main__":
    main()