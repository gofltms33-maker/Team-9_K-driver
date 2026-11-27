"""Freeze a saved neural checkpoint and its exact evaluation configuration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def freeze(source, output, checkpoint="model.pt", lane_duration=None):
    config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    for filename, digest in config["protected_hashes"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise ValueError("Protected road/traffic file changed: " + filename)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source / checkpoint, output / "model.pt")
    config["stage"] = "Frozen neural candidate; no training updates in this folder"
    config["frozen_checkpoint"] = dict(source=str(source), checkpoint=checkpoint,
        sha256=hashlib.sha256((output / "model.pt").read_bytes()).hexdigest())
    if lane_duration is not None:
        config["actuator_experiment"] = dict(original_lane_change_duration=config["action"]["lane_change_duration"],
            lane_change_duration=lane_duration, no_weight_changes=True)
        config["action"]["lane_change_duration"] = lane_duration
    (output / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for relative in ("env/sumo_env.py", "env/mdp_config.py", "sensor_history.py", "physical_layout.py", "action_labels.py", "utils/networks.py", "algorithms/ppo.py", "evaluate_imitation.py"):
        target = output / "evaluation_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative, target)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--lane-duration", type=float)
    args = parser.parse_args()
    freeze(args.source, args.output, args.checkpoint, args.lane_duration)
    print("Frozen", args.output)
