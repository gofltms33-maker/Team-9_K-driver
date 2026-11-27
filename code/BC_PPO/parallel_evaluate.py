"""Parallel full-route evaluation of a frozen neural policy and saved SUMO modes.

Example:
    python parallel_evaluate.py results/frozen_candidate --episodes 100 --seed-start 5000 --workers 4

No custom driving gate, expert driver, autonomous lane choice, route recovery,
or training is used. Configured SUMO speed/lane safety modes remain explicit.
Use --require-course-profile to require direct 96/0 control and redacted lane
guidance. Archived native-safety/lane-guidance results do not qualify for it.
"""
import argparse
import atexit
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
_ENV = _AGENT = _CONFIG = _EXPECTED_ROUTE = None
_REQUIRE_COURSE_PROFILE = False


def validate_course_config(config):
    """Reject non-course settings without silently changing the experiment."""
    action, observation = config["action"], config["observation"]
    if int(action.get("policy_speed_mode", -1)) != 96 or int(action.get("policy_lane_change_mode", 0)) != 0:
        raise ValueError("Course profile requires policy_speed_mode=96 and policy_lane_change_mode=0; archived native-safety results are excluded")
    if action.get("lane_change_safety", {}).get("enabled", True) or float(action.get("lane_change_cooldown_s", 0)) != 0:
        raise ValueError("Course profile forbids custom lane safety gates and cooldowns")
    if observation.get("redact_lane_guidance") is not True:
        raise ValueError("Course profile requires observation.redact_lane_guidance=true")
    frames=observation.get("sensor_history_frames",1)
    if isinstance(frames,bool) or not isinstance(frames,int) or not 1<=frames<=4:
        raise ValueError("Sensor history must contain1..4 measured frames")
    if not observation.get("lane_guidance") or not observation.get("next_junction_guidance"):
        raise ValueError("Course profile requires reserved redacted columns 38:55 and coarse navigation columns 55:60")
    if not isinstance(observation.get("route_distance_feature", False), bool):
        raise ValueError("observation.route_distance_feature must be a boolean")
    if observation.get("route_distance_feature", False):
        scale = float(observation.get("route_distance_scale_m", 10000.0))
        if not math.isfinite(scale) or scale != 10000.0:
            raise ValueError("Course physical progress input requires route_distance_scale_m=10000")
    for section in (config, config.get("simulation", {}), action):
        if any(section.get(name, False) for name in ("native_driver", "autodriver", "auto_driver",
                "automatic_reroute", "auto_reroute", "reroute_if_missed")):
            raise ValueError("Course profile forbids native expert driving and automatic route recovery")
    metadata = config.get("course_profile", {})
    if metadata.get("redacted_columns", [38, 55]) != [38, 55] or metadata.get("coarse_navigation_columns", [55, 60]) != [55, 60]:
        raise ValueError("Course profile navigation column contract does not match 38:55 redacted / 55:60 coarse")


def validate_course_runtime(observation, speed_mode, lane_mode, route_distance_feature=False,
                            sensor_history_frames=1):
    import numpy as np
    array = np.asarray(observation)
    if sensor_history_frames>1:
        from sensor_history import validate_history_data
        validate_history_data(array.reshape(1,-1),dict(sensor_history_frames=sensor_history_frames,
            redact_lane_guidance=True,route_distance_feature=route_distance_feature))
    if int(speed_mode) != 96 or int(lane_mode) != 0:
        raise RuntimeError("Course profile runtime modes must remain 96/0")
    if array.ndim != 1 or len(array) < 60 or not np.isfinite(array[38:55]).all():
        raise RuntimeError("Course profile reserved observation block must be finite")
    if route_distance_feature:
        if not 0 <= array[53] <= 1 or np.any(array[38:53] != 0) or array[54] != 0:
            raise RuntimeError("Only physical progress in column 53 may be nonzero; it must be in [0,1]")
    elif np.any(array[38:55] != 0):
        raise RuntimeError("Course profile observation columns 38:55 must be exactly zero at every step")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(7):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt==6:raise
            time.sleep(min(.05*2**attempt,.5))


def read_json(path):
    """Retry short Windows file-sharing locks without discarding episode data."""
    for attempt in range(7):
        try:return json.loads(Path(path).read_text(encoding='utf-8'))
        except PermissionError:
            if attempt==6:raise
            time.sleep(min(.05*2**attempt,.5))


def resolve_input(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def close_worker():
    global _ENV
    if _ENV is not None:
        try:
            _ENV.close()
        except Exception:
            pass
        _ENV = None


def initialize_worker(config, model, expected_route, require_course_profile=False):
    global _ENV, _AGENT, _CONFIG, _EXPECTED_ROUTE, _REQUIRE_COURSE_PROFILE
    import torch
    from algorithms.ppo import PPO
    from train_warmstart_ppo import saved_environment
    torch.set_num_threads(1)
    _CONFIG, _EXPECTED_ROUTE = config, tuple(expected_route)
    _REQUIRE_COURSE_PROFILE = require_course_profile
    if require_course_profile:
        validate_course_config(config)
    _ENV = saved_environment(config, prefix=None)
    _AGENT = PPO(_ENV.obs_dim, 2, **{**config["hparams"], "device": "cpu"})
    _AGENT.load(model)
    _AGENT.policy.eval()
    atexit.register(close_worker)


def error_row(seed, error):
    return dict(seed=seed, end_reason="evaluation_error", arrival=False, collided=False,
        teleported=False, red_light_violation=False, steps=0, seconds=0.0,
        physical_progress_m=0.0, ep_return=0.0, protocol_valid=False,
        error=str(error), traceback=traceback.format_exc(limit=5))


def evaluate_episode(seed):
    global _ENV
    import numpy as np
    from env.sumo_env import traci
    from train_warmstart_ppo import saved_environment
    steps = 0
    progress = total_return = 0.0
    collided = teleported = red_light = False
    errors = []
    route_distance_feature = bool(_CONFIG["observation"].get("route_distance_feature", False))
    sensor_history_frames = int(_CONFIG["observation"].get("sensor_history_frames",1))
    row = dict(seed=seed, course_profile_required=_REQUIRE_COURSE_PROFILE,
               route_distance_feature=route_distance_feature,
               sensor_history_frames=sensor_history_frames,
               course_profile_verified=False)
    try:
        if _ENV is None:
            _ENV = saved_environment(_CONFIG, prefix=None)
        np.random.seed(seed)
        obs, reset_info = _ENV.reset(seed=seed)
        row['initial_traffic']=reset_info.get('initial_traffic',getattr(_ENV,'initial_traffic',{}))
        actual_speed = int(traci.vehicle.getSpeedMode(_ENV.ego_id))
        actual_lane = int(traci.vehicle.getLaneChangeMode(_ENV.ego_id))
        if _REQUIRE_COURSE_PROFILE:
            validate_course_runtime(obs, actual_speed, actual_lane, route_distance_feature,sensor_history_frames)
        expected_speed = int(_CONFIG["action"].get("policy_speed_mode", 0))
        expected_lane = int(_CONFIG["action"].get("policy_lane_change_mode", 0))
        row.update(actual_speed_mode=actual_speed, actual_lane_change_mode=actual_lane,
                   configured_speed_mode=expected_speed, configured_lane_change_mode=expected_lane,
                   sumo_version=list(traci.getVersion()))
        if actual_speed != expected_speed or actual_lane != expected_lane:
            raise RuntimeError("Actual SUMO modes differ from the frozen configuration")
        if actual_lane not in (0, 256, 512):
            raise RuntimeError("Autonomous lane selection is forbidden in policy evaluation")
        route_unchanged = tuple(traci.vehicle.getRoute(_ENV.ego_id)) == _EXPECTED_ROUTE
        if not route_unchanged:
            raise RuntimeError("AV was not inserted on the full original route")
        mode_unchanged = True
        mismatches = observed_commands = 0
        maximum_error = total_error = 0.0
        requested_last = realized_last = None
        info, terminated, truncated = {}, False, False
        for steps in range(1, _ENV.max_steps + 1):
            obs, reward, terminated, truncated, info = _ENV.step(_AGENT.predict(obs))
            progress += float(info.get("progress_m", 0.0))
            total_return += float(reward)
            collided |= bool(info.get("collided", False) or info.get("end_reason") == "collision")
            teleported |= bool(info.get("teleported", False) or info.get("end_reason") == "teleport")
            red_light |= bool(info.get("red_light_violation", False) or info.get("end_reason") == "red_light")
            if _REQUIRE_COURSE_PROFILE:
                validate_course_runtime(obs, info.get("sumo_speed_mode", actual_speed),
                                        info.get("sumo_lane_change_mode", actual_lane), route_distance_feature,sensor_history_frames)
            command_error = info.get("longitudinal_command_error")
            if command_error is not None and math.isfinite(float(command_error)):
                observed_commands += 1
                magnitude = abs(float(command_error))
                maximum_error = max(maximum_error, magnitude)
                total_error += magnitude
                mismatches += int(magnitude > 1e-4)
            requested_last = info.get("requested_speed")
            realized_last = info.get("realized_speed")
            if _ENV.ego_id in traci.vehicle.getIDList():
                if _REQUIRE_COURSE_PROFILE:
                    validate_course_runtime(obs, traci.vehicle.getSpeedMode(_ENV.ego_id),
                                            traci.vehicle.getLaneChangeMode(_ENV.ego_id), route_distance_feature,sensor_history_frames)
                route_unchanged &= tuple(traci.vehicle.getRoute(_ENV.ego_id)) == _EXPECTED_ROUTE
                # Modes are checked at reset and at the last live state of each
                # episode. Per-step environment metadata also exposes changes.
                mode_unchanged &= int(info.get("sumo_speed_mode", actual_speed)) == actual_speed
                mode_unchanged &= int(info.get("sumo_lane_change_mode", actual_lane)) == actual_lane
                if terminated or truncated or steps == _ENV.max_steps:
                    mode_unchanged &= int(traci.vehicle.getSpeedMode(_ENV.ego_id)) == actual_speed
                    mode_unchanged &= int(traci.vehicle.getLaneChangeMode(_ENV.ego_id)) == actual_lane
            route_unchanged &= not bool(info.get("rerouted", False)) and _ENV.reroute_count == 0
            if info.get("control_source") != "policy":
                raise RuntimeError("A non-policy controller was used")
            if terminated or truncated:
                break
        reason = info.get("end_reason") or "time_limit"
        arrival = bool(reason == "arrival" and not collided and not teleported and route_unchanged)
        row.update(end_reason=reason, arrival=arrival, collided=collided, teleported=teleported,
            red_light_violation=red_light, steps=steps, seconds=round(steps * _ENV.step_length, 4),
            physical_progress_m=round(progress, 3), ep_return=round(total_return, 5),
            terminated=bool(terminated), truncated=bool(truncated), full_original_route=True,
            route_unchanged=route_unchanged, control_source="policy", mode_unchanged=mode_unchanged,
            protocol_valid=bool(route_unchanged and mode_unchanged),
            course_profile_verified=bool(_REQUIRE_COURSE_PROFILE and route_unchanged and mode_unchanged),
            command_observations=observed_commands, command_mismatch_steps=mismatches,
            max_command_error=maximum_error,
            mean_command_error=total_error / observed_commands if observed_commands else 0.0,
            last_requested_speed=None if requested_last is None else float(requested_last),
            last_realized_speed=None if realized_last is None else float(realized_last))
    except Exception as exc:
        error = error_row(seed, exc)
        error.update(row)
        error.update(end_reason="evaluation_error", arrival=False, protocol_valid=False,
                     collided=collided, teleported=teleported, red_light_violation=red_light,
                     steps=steps, seconds=round(steps * _CONFIG["simulation"]["step_length"], 4),
                     physical_progress_m=round(progress, 3), ep_return=round(total_return, 5))
        row = error
        close_worker()
    return row


def worker_main(task_path):
    """Standalone process protocol: task/results/logs are ordinary files only."""
    task = read_json(task_path)
    sys.path.insert(0, task["project_root"])
    out = Path(task["output"])
    completed = 0
    try:
        initialize_worker(task["config"], task["model"], task["expected_route"], task.get("require_course_profile", False))
        for seed in task["seeds"]:
            row = evaluate_episode(seed)
            atomic_json(out / "episodes" / f"seed_{seed}.json", row)
            completed += 1
            atomic_json(Path(task_path).with_suffix(".status.json"),
                        dict(status="running", completed=completed, assigned=len(task["seeds"])))
    except Exception as exc:
        for seed in task["seeds"]:
            destination = out / "episodes" / f"seed_{seed}.json"
            if not destination.exists():
                atomic_json(destination, error_row(seed, exc))
        atomic_json(Path(task_path).with_suffix(".status.json"),
                    dict(status="failed", completed=completed, error=str(exc)))
        return 1
    finally:
        close_worker()
    atomic_json(Path(task_path).with_suffix(".status.json"),
                dict(status="completed", completed=completed, assigned=len(task["seeds"])))
    return 0


def summarize(rows, planned, integrity=None, finished=False, require_course_profile=False,
              route_distance_feature=False):
    counts = Counter(row["end_reason"] for row in rows)
    collisions = sum(bool(row.get("collided")) for row in rows)
    arrivals = sum(bool(row.get("arrival")) for row in rows)
    evaluation_valid = bool(finished and len(rows) == planned
        and all(row.get("protocol_valid", False) for row in rows)
        and counts["evaluation_error"] == 0 and integrity and all(integrity.values()))
    if require_course_profile:
        evaluation_valid &= all(row.get("course_profile_verified", False) for row in rows)
    metrics = dict(episodes=planned, completed_records=len(rows), end_reason_counts=dict(counts),
        collision_episodes=collisions, arrival_episodes=arrivals,
        collision_rate=collisions / planned, arrival_rate=arrivals / planned,
        red_light_rate=sum(bool(row.get("red_light_violation")) for row in rows) / planned,
        teleport_rate=sum(bool(row.get("teleported")) for row in rows) / planned,
        evaluation_error_rate=counts["evaluation_error"] / planned,
        failed_episodes=planned - arrivals, evaluation_valid=evaluation_valid,
        course_profile_required=require_course_profile,
        course_profile_verified=bool(evaluation_valid) if require_course_profile else None,
        physical_progress_input_enabled=route_distance_feature,
        collision_target_met=bool(evaluation_valid and collisions / planned < .05),
        completion_target_met=bool(evaluation_valid and arrivals / planned >= .95),
        perfect_completion_observed=bool(evaluation_valid and arrivals == planned))
    metrics["task_target_met"] = metrics["collision_target_met"] and metrics["completion_target_met"]
    metrics["note"] = "All planned seeds are in the denominator; errors are failures. Finite-sample results do not guarantee future success."
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-course-profile", action="store_true",
                        help="Require modes 96/0 and verify redacted lane guidance at every step")
    args = parser.parse_args()
    if min(args.episodes, args.workers, args.batch_size) < 1:
        parser.error("Episodes, workers and batch size must be positive")
    run = args.run.resolve()
    config_path, model_path = run / "run_config.json", run / args.checkpoint
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    route_distance_feature = bool(config["observation"].get("route_distance_feature", False))
    if args.require_course_profile:
        validate_course_config(config)
    config["base"] = str(resolve_input(config["base"]))
    protected = {str(resolve_input(path)): value for path, value in config["protected_hashes"].items()}
    for path, expected in protected.items():
        if digest(path) != expected:
            raise ValueError("Protected road/traffic hash mismatch: " + path)
    config["protected_hashes"] = protected
    base = Path(config["base"])
    route_root = ET.parse(base / "sumo/highway.rou.xml").getroot()
    route_node = route_root.find("route[@id='r0']")
    if route_node is None:
        raise ValueError("Original r0 route is missing")
    expected_route = route_node.get("edges").split()
    # Constructor checks reject autonomous lane modes and custom policy assists
    # before starting a process pool or any simulation.
    from train_warmstart_ppo import saved_environment
    probe = saved_environment(config, prefix=None)
    probe.close()
    inputs = {str(config_path): hashlib.sha256(config_bytes).hexdigest(), str(model_path): digest(model_path)}
    for path in [*protected, str(base / "road_config.py"), str(base / "sumo/highway.sumocfg"),
                 str(base / "sumo/highway.net.xml"), str(base / "sumo/highway.rou.xml")]:
        inputs[path] = digest(path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fUTC")
    out = (args.output or ROOT / "outputs" / (run.name + "_parallel_" + stamp)).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "episodes").mkdir()
    (out / "run_config.source.json").write_bytes(config_bytes)
    atomic_json(out / "run_config.json", config)
    shutil.copyfile(model_path, out / "model.pt")
    if digest(out / "model.pt") != inputs[str(model_path)]:
        raise ValueError("Model changed while creating its evaluation snapshot")
    shutil.copyfile(__file__, out / "parallel_evaluate.py")
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    workers = min(args.workers, len(seeds))
    manifest = dict(source_run=str(run), source_checkpoint=args.checkpoint,
        created_utc=datetime.now(timezone.utc).isoformat(), seeds=seeds, workers=workers,
        batch_size=args.batch_size, input_hashes_before=inputs, full_original_route=expected_route,
        configured_speed_mode=config["action"].get("policy_speed_mode", 0),
        configured_lane_change_mode=config["action"].get("policy_lane_change_mode", 0),
        course_profile_required=args.require_course_profile,
        measured_sensor_history_frames=config["observation"].get("sensor_history_frames",1),
        physical_progress_input=dict(enabled=route_distance_feature,
            column=53 if route_distance_feature else None,
            scale_m=config["observation"].get("route_distance_scale_m", 10000.0) if route_distance_feature else None,
            source="clip(SUMO vehicle.getDistance / route_distance_scale_m, 0, 1)" if route_distance_feature else None,
            lane_instruction=False, action_override=False),
        claim_scope="Course profile only; archived native-safety results excluded" if args.require_course_profile else "Saved configuration only; not certified as a course profile",
        worker_transport="Standalone subprocesses; ordinary task/result/log files, no multiprocessing pipes",
        custom_policy_assists=False, native_expert_driver=False,
        protocol="Configured SUMO constraints are disclosed; no autonomous lane selection or custom driving rules")
    atomic_json(out / "manifest.json", manifest)
    rows, started = [], time.monotonic()
    atomic_json(out / "status.json", dict(status="running", **summarize(rows, len(seeds),
        require_course_profile=args.require_course_profile, route_distance_feature=route_distance_feature)))
    (out / "workers").mkdir()
    processes = []
    for number in range(workers):
        assigned = seeds[number::workers]
        task_path = out / "workers" / f"worker_{number}.json"
        atomic_json(task_path, dict(project_root=str(ROOT), output=str(out), config=config,
            model=str(out / "model.pt"), expected_route=expected_route, seeds=assigned,
            require_course_profile=args.require_course_profile))
        try:
            with (out / "workers" / f"worker_{number}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, str(out / "parallel_evaluate.py"),
                    "--worker-task", str(task_path)], cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            processes.append(dict(process=process, seeds=assigned, accounted=False))
        except Exception as exc:
            for seed in assigned:
                atomic_json(out / "episodes" / f"seed_{seed}.json", error_row(seed, exc))
    seen = set()
    while True:
        for worker in processes:
            code = worker["process"].poll()
            if code is not None and not worker["accounted"]:
                for seed in worker["seeds"]:
                    destination = out / "episodes" / f"seed_{seed}.json"
                    if not destination.exists():
                        atomic_json(destination, error_row(seed, f"Worker exited with code {code} before producing this result"))
                worker["accounted"] = True
        changed = False
        for seed in seeds:
            if seed in seen:
                continue
            destination = out / "episodes" / f"seed_{seed}.json"
            if destination.exists():
                row = read_json(destination)
                rows.append(row)
                seen.add(seed)
                changed = True
                print(json.dumps({k: row[k] for k in ("seed", "end_reason", "steps", "physical_progress_m")}), flush=True)
        if changed:
            rows.sort(key=lambda row: row["seed"])
            partial = summarize(rows, len(seeds), require_course_profile=args.require_course_profile,
                                route_distance_feature=route_distance_feature)
            atomic_json(out / "evaluation.json", dict(manifest=manifest, status="running", metrics=partial, episodes=rows))
            atomic_json(out / "status.json", dict(status="running", elapsed_seconds=round(time.monotonic() - started, 2), **partial))
        if len(seen) == len(seeds) and all(worker["accounted"] for worker in processes):
            break
        time.sleep(1)
    integrity = {path: digest(path) == expected if Path(path).exists() else False for path, expected in inputs.items()}
    integrity["frozen_model_snapshot"] = digest(out / "model.pt") == inputs[str(model_path)]
    acceptance = summarize(rows, len(seeds), integrity=integrity, finished=True,
                           require_course_profile=args.require_course_profile,
                           route_distance_feature=route_distance_feature)
    final = dict(manifest=manifest, status="completed", input_integrity=integrity,
                 elapsed_seconds=round(time.monotonic() - started, 2), metrics=acceptance, episodes=rows)
    atomic_json(out / "evaluation.json", final)
    atomic_json(out / "acceptance.json", acceptance)
    atomic_json(out / "status.json", dict(status="completed", **acceptance))
    print("REPORT", out, json.dumps(acceptance), flush=True)
    return 0 if acceptance["evaluation_valid"] else 2


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-task":
        raise SystemExit(worker_main(sys.argv[2]))
    raise SystemExit(main())
