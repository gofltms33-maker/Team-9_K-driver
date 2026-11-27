"""Replay a BC/PPO-architecture actor using its saved observation and route task.

Example: python evaluate_imitation.py results/bc_curriculum_recovery_v11 --episodes 1 --gui
No expert driver, recovery rule, or training is used.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import torch

from env.sumo_env import SumoHighwayEnv
from algorithms.ppo import PPO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1200)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--full-route", action="store_true",
                        help="Evaluate a curriculum actor on the full original navigation plan")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    cfg = json.loads((args.run / "run_config.json").read_text(encoding="utf-8"))
    base = Path(cfg["base"])
    for filename, digest in cfg["protected_hashes"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise ValueError("Saved road/traffic fixture changed: " + filename)
    road = runpy.run_path(str(base / "road_config.py"))
    route = None
    simulation = dict(cfg["simulation"])
    variant = cfg["args"].get("route_variant", "normal")
    if variant == "detour":
        from route_variants import straight_detour
        plan = straight_detour(base)
        route = plan["route"]
        simulation["max_steps"] = json.loads((args.run / "navigation_plan.json").read_text())["time_budget_steps"]
    prefix = cfg["args"].get("route_prefix")
    if args.full_route:
        prefix = None
    if prefix not in (None, "None"):
        if route is None:
            route = ET.parse(base / "sumo/highway.rou.xml").getroot().find("route[@id='r0']").get("edges").split()
        route = route[:int(prefix)]
    env = SumoHighwayEnv(str((base / "sumo/highway.sumocfg").resolve()),
                         road["ROAD"], road["EGO"], simulation, cfg["observation"], cfg["action"],
                         cfg["reward"], traffic=road["TRAFFIC"], gui=args.gui, planned_route=route)
    if env.lc_safety_enabled or env.lane_change_cooldown:
        raise ValueError("Saved configuration enables policy assists; use an unassisted run")
    from sensor_history import with_sensor_history
    env = with_sensor_history(env,cfg["observation"])
    torch.set_num_threads(1)
    agent = PPO(env.obs_dim, 2, **{**cfg["hparams"], "device": "cpu"})
    agent.load(str(args.run / "model.pt"))
    rows = []
    try:
        for seed in range(args.seed_start, args.seed_start + args.episodes):
            np.random.seed(seed)
            obs, _ = env.reset(seed=seed)
            distance = 0.0
            max_command_error, command_mismatches = 0.0, 0
            for step in range(1, env.max_steps + 1):
                obs, _, terminated, truncated, info = env.step(agent.predict(obs))
                distance += info["progress_m"]
                command_error = info.get("longitudinal_command_error")
                if command_error is not None:
                    max_command_error = max(max_command_error, command_error)
                    command_mismatches += int(command_error > 1e-4)
                if terminated or truncated:
                    break
            row = dict(seed=seed, steps=step, end_reason=info["end_reason"],
                       physical_progress_m=round(distance, 2), control_source=info["control_source"])
            row.update(sumo_speed_mode=env.policy_speed_mode, sumo_lane_change_mode=env.policy_lane_change_mode, max_command_error=max_command_error,
                       command_mismatch_steps=command_mismatches)
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    result = Path("outputs") / (args.run.name + "_evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    result.write_text(json.dumps(dict(run=str(args.run), route_prefix=prefix, full_route_override=args.full_route,
                                     current_actuator=True,
                                     custom_policy_assists=False, sumo_speed_mode=env.policy_speed_mode,
                                     sumo_lane_change_mode=env.policy_lane_change_mode,
                                     native_sumo_safety_constraints=env.policy_speed_mode == 31 or env.policy_lane_change_mode != 0,
                                     all_sumo_speed_checks_disabled=env.policy_speed_mode == 96,
                                     episodes=rows), indent=2), encoding="utf-8")
    print("Report:", result)
