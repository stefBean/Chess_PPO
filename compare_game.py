# generate_eval_positions.py
import json
from board_environment import Chess

env = Chess(
    start_mode="curriculum_endgame",
    endgame_max_extra_per_side=4,
    endgame_min_extra_total=3,
    require_pawn=False,
)

positions = []
for i in range(100):
    env.reset(options={"curriculum_stage": 3, "agent_color": None})
    positions.append(env._board.fen())

with open("eval_positions.json", "w") as f:
    json.dump(positions, f, indent=2)