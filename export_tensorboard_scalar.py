from pathlib import Path
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS_DIR = Path("runs")
OUT_FILE = Path("tensorboard_scalars_export_2.csv")

rows = []

event_files = list(RUNS_DIR.rglob("events.out.tfevents.*"))

for event_file in event_files:
    run_dir = event_file.parent
    run_name = str(run_dir.relative_to(RUNS_DIR))

    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()

    for tag in accumulator.Tags().get("scalars", []):
        for event in accumulator.Scalars(tag):
            rows.append({
                "run": run_name,
                "tag": tag,
                "step": event.step,
                "wall_time": event.wall_time,
                "value": event.value,
            })

df = pd.DataFrame(rows)
df = df.drop_duplicates(subset=["run", "tag", "step", "value"])
df.to_csv(OUT_FILE, index=False)

print("Event files found:", len(event_files))
print("Rows exported:", len(df))
print("Runs:")
print(df["run"].unique())
print("Tags:")
print(df["tag"].unique())