"""Collect native demonstrations, pretrain the PPO actor, and evaluate without PPO updates."""
import argparse
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import runpy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.ppo import PPO
from env.sumo_env import SumoHighwayEnv, traci
from env.mdp_config import SIMULATION, OBSERVATION, ACTION
from train import HPARAMS, REWARD
from action_labels import lateral_label


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def make_env(base, native=False, route_variant="normal", route_prefix=None):
    road = runpy.run_path(str(base / "road_config.py"))
    simulation, route = dict(SIMULATION), None
    if route_variant == "detour":
        from route_variants import straight_detour
        plan = straight_detour(base)
        route = plan["route"]
        simulation["max_steps"] = math.ceil(SIMULATION["max_steps"] *
                                           plan["variant_distance_m"] / plan["original_distance_m"])
    if route_prefix is not None:
        if route_prefix < 2:
            raise ValueError("Curriculum routes need at least two edges")
        if route is None:
            route = ET.parse(base / "sumo/highway.rou.xml").getroot().find("route[@id='r0']").get("edges").split()
        route = route[:route_prefix]
    return SumoHighwayEnv(str((base / "sumo/highway.sumocfg").resolve()),
                         road["ROAD"], road["EGO"], simulation, OBSERVATION, ACTION,
                         REWARD, traffic=road["TRAFFIC"], native_driver=native,
                         planned_route=route)


def collect(base, out, episodes, recovery_policy=None, failure_targeted=False, seed_start=0,
            route_variant="normal", route_prefix=None, curriculum_mix=False,
            recovery_lead_steps=None, recovery_window=None, expand_recovery_input=False, failure_only=False,
             collection_seeds=None, expert_tau=None, minimum_successful_episodes=5):
    env = make_env(base, native=True, route_variant="normal" if route_variant == "mixed" else route_variant,
                   route_prefix=route_prefix)
    learner = None
    if recovery_policy:
        recovery_config = json.loads((recovery_policy.parent / "run_config.json").read_text(encoding="utf-8"))
        learner = PPO(env.obs_dim, 2, **{**recovery_config["hparams"], "device": "cpu"})
        state = torch.load(recovery_policy, map_location="cpu", weights_only=True)
        if expand_recovery_input:
            for name in ("trunk.0.weight", "accel_trunk.0.weight"):
                if name in state:
                    extra = learner.policy.state_dict()[name].shape[1] - state[name].shape[1]
                    if extra < 0:
                        raise ValueError("Cannot discard recovery actor inputs")
                    state[name] = F.pad(state[name], (0, extra))
        learner.policy.load_state_dict(state)
    obs_rows, actions, ids, lane_masks, accel_masks = [], [], [], [], []
    reports = []
    rejected_accel = boundary = 0
    try:
        for seed in (collection_seeds if collection_seeds is not None else range(seed_start, seed_start + episodes)):
            np.random.seed(seed)
            episode_variant = ("detour" if seed % 2 else "normal") if route_variant == "mixed" else route_variant
            episode_prefix = (5, 16, None, None)[(seed - seed_start) % 4] if curriculum_mix else route_prefix
            if route_variant == "mixed" or curriculum_mix:
                setup = make_env(base, native=True, route_variant=episode_variant, route_prefix=episode_prefix)
                env.planned_route, env.max_steps = setup.planned_route, setup.max_steps
            env.native_driver = learner is None
            obs, _ = env.reset(seed=seed)
            if learner is None and expert_tau is not None:
                traci.vehicle.setTau("ego", expert_tau)
            failure_step = None
            critical_step = None
            if learner is not None and failure_targeted:
                # First pass locates this learner's actual failure, not a fixed road location.
                last_motion_step = 0
                for failure_step in range(1, env.max_steps + 1):
                    obs, _, done, truncated, failure_info = env.step(learner.predict(obs))
                    if failure_info.get("speed") is not None and failure_info["speed"] > 1:
                        last_motion_step = failure_step
                    if done or truncated:
                        break
                critical_step = (last_motion_step if failure_info["end_reason"] in
                                 ("idle", "route_blocked", "stuck") else failure_step)
                if failure_only and failure_info["end_reason"] == "arrival":
                    reports.append(dict(seed=seed, end_reason="learner_already_arrived",
                                        retained=False, samples=0, learner_failure_step=failure_step))
                    print("DEMONSTRATION", json.dumps(reports[-1]), flush=True)
                    continue
                np.random.seed(seed)
                obs, _ = env.reset(seed=seed)
            if learner is not None:
                # Collect expert recovery labels from learner-induced states.
                # The learner prefix is not labelled as expert behavior.
                prefix = (max(1, int(failure_step * (0.55 if seed % 2 == 0 else 0.75)))
                          if failure_step is not None else 10 + (seed % 4) * 10)
                if recovery_lead_steps is not None and critical_step is not None:
                    prefix = max(1, critical_step - recovery_lead_steps)
                for _ in range(prefix):
                    obs, _, done, truncated, _ = env.step(learner.predict(obs))
                    if done or truncated:
                        raise RuntimeError("Learner prefix ended before expert recovery")
                env.native_driver = True
                traci.vehicle.setSpeedMode("ego", 31)
                traci.vehicle.setLaneChangeMode("ego", 1621)
                traci.vehicle.changeLane("ego", traci.vehicle.getLaneIndex("ego"), 0)
                traci.vehicle.setSpeed("ego", -1)
                if expert_tau is not None:
                    traci.vehicle.setTau("ego", expert_tau)
            start = len(obs_rows)
            for step in range(env.max_steps):
                before_speed = traci.vehicle.getSpeed("ego")
                edge = traci.vehicle.getRoadID("ego")
                lane = traci.vehicle.getLaneIndex("ego")
                lane_id = traci.vehicle.getLaneID("ego")
                links = traci.lane.getLinks(lane_id)
                next_obs, _, done, truncated, info = env.step([0, 0])
                if not done and not truncated and (recovery_window is None or step < recovery_window):
                    speed = traci.vehicle.getSpeed("ego")
                    dv = (speed - before_speed) / env.step_length
                    raw = dv / (env.max_accel if dv >= 0 else env.max_decel)
                    same_edge = edge == traci.vehicle.getRoadID("ego")
                    command, lane_valid = lateral_label(lane_id, lane, links,
                        traci.vehicle.getLaneID("ego"), traci.vehicle.getLaneIndex("ego"))
                    valid_accel = abs(raw) <= 1.0001
                    rejected_accel += int(not valid_accel)
                    boundary += int(not same_edge)
                    obs_rows.append(obs)
                    actions.append([np.clip(raw, -1, 1), command])
                    ids.append(seed)
                    lane_masks.append(lane_valid)
                    accel_masks.append(valid_accel)
                obs = next_obs
                if done or truncated:
                    break
            report = dict(seed=seed, steps=step + 1, end_reason=info["end_reason"],
                          route_variant=episode_variant, time_budget_steps=env.max_steps,
                          route_prefix=episode_prefix,
                          samples=len(obs_rows) - start, learner_failure_step=failure_step,
                          learner_end_reason=failure_info["end_reason"] if failure_step is not None else None,
                          learner_critical_step=critical_step,
                          handover_step=prefix if learner is not None else 0)
            if info["end_reason"] != "arrival":
                # Do not silently label a failed rescue as a successful expert.
                for rows in [obs_rows, actions, ids, lane_masks, accel_masks]:
                    del rows[start:]
                report["retained"] = False
            else:
                report["retained"] = True
            reports.append(report)
            print("DEMONSTRATION", json.dumps(report), flush=True)
    finally:
        env.close()
    data = dict(observation=np.asarray(obs_rows, np.float32),
                action=np.asarray(actions, np.float32), episode=np.asarray(ids),
                lane_valid=np.asarray(lane_masks), accel_valid=np.asarray(accel_masks))
    np.savez_compressed(out / "demonstrations.npz", **data)
    write_json(out / "data_audit.json", dict(episodes=reports, samples=len(ids),
               lane_counts={str(k): int(np.sum(data["action"][:, 1] == k)) for k in [-1, 0, 1]},
               excluded_edge_transitions=boundary, excluded_accelerations=rejected_accel,
               label_schema=3,
               expert_suffix=learner is not None,
               label_definition="t observation -> executed acceleration and lateral change including junction entry/exit; unrepresentable negative internal lane targets masked"))
    if len(set(ids)) < minimum_successful_episodes:
        raise RuntimeError(f"Fewer than {minimum_successful_episodes} successful expert episodes remain; inspect data_audit.json")
    return data


def fit(data, out, epochs, lane_weight_power=0.5, brake_weight=1.0,
        init_model=None, bc_lr=3e-4, expand_input=False, launch_weight=1.0,
        hazard_weight=1.0, balanced_sampling=False, checkpoint_every=0):
    agent = PPO(data["observation"].shape[1], 2, **{**HPARAMS, "device": "cpu"})
    if init_model:
        state = torch.load(init_model, map_location="cpu", weights_only=True)
        if expand_input:
            old = state["trunk.0.weight"]
            extra = data["observation"].shape[1] - old.shape[1]
            if extra < 0:
                raise ValueError("Cannot discard checkpoint observation features")
            state["trunk.0.weight"] = F.pad(old, (0, extra))
        agent.policy.load_state_dict(state)
    # Freeze critic and exploration scale: BC pretrains only the shared features and actor.
    agent.policy.log_std.requires_grad_(False)
    for p in agent.policy.v_head.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in agent.policy.parameters() if p.requires_grad], lr=bc_lr)
    obs = torch.from_numpy(data["observation"])
    accel = torch.from_numpy(data["action"][:, :1])
    labels = torch.from_numpy((1 - data["action"][:, 1]).astype(np.int64))
    lm, am = torch.from_numpy(data["lane_valid"]), torch.from_numpy(data["accel_valid"])
    heldout = data["episode"] % 5 == 4
    train = np.flatnonzero(~heldout)
    valid = np.flatnonzero(heldout)
    if len(valid) == 0:
        raise ValueError("At least 5 demonstration episodes are required for an episode-level split")
    counts = torch.bincount(labels[train][lm[train]], minlength=3).float().clamp(min=1)
    weights = (counts.sum() / counts).pow(lane_weight_power)
    weights /= weights.mean()
    all_obs, all_actions = data["observation"], data["action"]
    hazards = (((all_obs[:, 31] < .75) & (all_obs[:, 32] < .35) & (all_obs[:, 0] > .03))
               | ((all_obs[:, 9] < .3) & (all_obs[:, 10] < -.015)))
    starts = (all_obs[:, 0] < .03) & (all_actions[:, 0] > .1) & (all_obs[:, 31] > .75)
    hazard_indices, start_indices = train[hazards[train]], train[starts[train]]
    def loss(indices):
        mu, _, logits, _ = agent.policy(obs[indices])
        if am[indices].any():
            target = accel[indices][am[indices]]
            importance = torch.where(target < -0.1, brake_weight, 1.0)
            # Reweight genuine expert targets in observed hazard states only.
            # This changes supervised learning, never an evaluated action.
            observed = obs[indices][am[indices]]
            hazard = ((observed[:, 31:32] < .75) & (observed[:, 32:33] < .35)
                      & (observed[:, :1] > .03))
            hazard |= ((observed[:, 9:10] < .3) & (observed[:, 10:11] < -.015))
            importance *= torch.where(hazard, hazard_weight, 1.0)
            # Balance rare demonstrated restarts; never override an evaluated action.
            restart = ((obs[indices][am[indices], :1] < .03) & (target > .1)
                       & (obs[indices][am[indices], 31:32] > .75))
            importance *= torch.where(restart, launch_weight, 1.0)
            regression = ((mu[am[indices]] - target).square() * importance).mean()
        else:
            regression = mu.sum() * 0
        classification = F.cross_entropy(logits[lm[indices]], labels[indices][lm[indices]], weight=weights) if lm[indices].any() else logits.sum() * 0
        return regression + classification, regression, classification
    best, history = float("inf"), []
    for epoch in range(epochs):
        np.random.shuffle(train)
        optimization_indices = train
        if balanced_sampling:
            pieces = []
            if len(hazard_indices):
                pieces.append(np.random.choice(hazard_indices, len(train) * 2 // 5))
            if len(start_indices):
                pieces.append(np.random.choice(start_indices, len(train) // 5))
            pieces.append(np.random.choice(train, len(train) - sum(len(p) for p in pieces)))
            optimization_indices = np.concatenate(pieces)
            np.random.shuffle(optimization_indices)
        agent.policy.train()
        for start in range(0, len(optimization_indices), 512):
            total, _, _ = loss(optimization_indices[start:start + 512])
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
            optimizer.step()
        agent.policy.eval()
        with torch.no_grad():
            total, regression, classification = loss(valid)
        row = dict(epoch=epoch + 1, validation_loss=total.item(),
                   accel_mse=regression.item(), lane_loss=classification.item())
        history.append(row)
        if total.item() < best:
            best = total.item()
            agent.save(str(out / "model.pt"))
        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            agent.save(str(out / f"model_epoch_{epoch + 1}.pt"))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("BC", json.dumps(row), flush=True)
    agent.save(str(out / "model_last.pt"))
    write_json(out / "bc_history.json", history)
    agent.load(str(out / "model.pt"))
    return agent


def evaluate(agent, base, out, episodes, seed_start=100, route_variant="normal", route_prefix=None):
    env = make_env(base, route_variant=route_variant, route_prefix=route_prefix)
    if env.native_driver or env.lc_safety_enabled or env.lane_change_cooldown:
        raise ValueError("Policy evaluation must not use native driving or lane safety/cooldown assists")
    reports = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            np.random.seed(seed)
            obs, _ = env.reset(seed=seed)
            progress = changes = 0
            total = 0
            for step in range(env.max_steps):
                obs, reward, done, truncated, info = env.step(agent.predict(obs))
                total += reward
                progress += info["progress_m"]
                changes += int(info["actual_lane_change"])
                if done or truncated:
                    break
            row = dict(seed=seed, steps=step + 1, end_reason=info["end_reason"],
                       route_variant=route_variant, time_budget_steps=env.max_steps,
                       route_prefix=route_prefix,
                       control_source=info["control_source"],
                       physical_progress_m=round(progress, 2), lane_changes=changes,
                       ep_return=round(total, 3), edge=traci.vehicle.getRoadID("ego")
                       if "ego" in traci.vehicle.getIDList() else None)
            reports.append(row)
            write_json(out / "evaluation.json", reports)
            print("EVALUATION", json.dumps(row), flush=True)
    finally:
        env.close()
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("results/envfix_20261118_140436"))
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed-start", type=int, default=100)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--recovery-policy", type=Path)
    parser.add_argument("--lane-weight-power", type=float, default=0.5)
    parser.add_argument("--failure-targeted", action="store_true")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--brake-weight", type=float, default=1.0)
    parser.add_argument("--init-model", type=Path)
    parser.add_argument("--bc-lr", type=float, default=3e-4)
    parser.add_argument("--route-variant", choices=["normal", "detour", "mixed"], default="normal")
    parser.add_argument("--expand-input", action="store_true")
    parser.add_argument("--route-prefix", type=int)
    parser.add_argument("--launch-weight", type=float, default=1.0)
    parser.add_argument("--curriculum-mix", action="store_true")
    parser.add_argument("--hazard-weight", type=float, default=1.0)
    parser.add_argument("--recovery-lead-steps", type=int)
    parser.add_argument("--recovery-window", type=int)
    parser.add_argument("--ego-speed-cap", type=float,
                        help="Experimental fixed AV maximum in m/s; leaves HV and road limits intact")
    parser.add_argument("--max-steps", type=int,
                        help="Explicit experimental time budget at the existing simulation step length")
    parser.add_argument("--front-safety-features", action="store_true",
                        help="Append observed TTC/headway/braking margin (no action override)")
    parser.add_argument("--nearby-traffic-features", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--signal-safety-feature", action="store_true")
    parser.add_argument("--semantic-lane-branches", action="store_true")
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--failure-only", action="store_true")
    parser.add_argument("--seed-list", type=int, nargs="+")
    parser.add_argument("--expert-tau", type=float,
                        help="Headway of the native demonstration teacher only; never applied to the learner")
    args = parser.parse_args()
    if args.expert_tau is not None and (not math.isfinite(args.expert_tau) or args.expert_tau <= 0):
        parser.error("--expert-tau must be finite and positive")
    if args.seed_list is not None and len(set(args.seed_list)) != len(args.seed_list):
        parser.error("--seed-list must not repeat episodes")
    if args.ego_speed_cap is not None:
        if not math.isfinite(args.ego_speed_cap) or args.ego_speed_cap <= 0:
            parser.error("--ego-speed-cap must be finite and positive")
        SIMULATION["ego_speed_cap_mps"] = args.ego_speed_cap
    if args.max_steps is not None:
        if args.max_steps < 1:
            parser.error("--max-steps must be positive")
        SIMULATION["max_steps"] = args.max_steps
    if args.front_safety_features:
        OBSERVATION["front_safety_features"] = True
    if args.signal_safety_feature:
        OBSERVATION["signal_safety_feature"] = True
    if args.semantic_lane_branches:
        OBSERVATION["semantic_lane_branches"] = True
    if args.nearby_traffic_features:
        OBSERVATION["nearby_traffic_features"] = True
        OBSERVATION["nearby_traffic_count"] = 8
    if any(value is not None and value < 1 for value in
           (args.recovery_lead_steps, args.recovery_window)):
        parser.error("Recovery lead and window must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    out = args.output or Path("results") / datetime.now().strftime("bc_warmstart_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    for relative in ["train_bc_ppo.py", "action_labels.py", "route_variants.py", "train.py",
                     "env/sumo_env.py", "env/mdp_config.py", "algorithms/ppo.py", "utils/networks.py"]:
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).resolve().parent / relative, target)
    shutil.copyfile(args.base / "road_config.py", out / "source/env/road_config.py")
    protected = [args.base / "sumo" / f for f in ["highway.net.xml", "highway.rou.xml"]]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    write_json(out / "run_config.json", dict(stage="BC only; critic untrained", base=str(args.base),
               hparams=HPARAMS, observation=OBSERVATION, action=ACTION, simulation=SIMULATION,
               reward=REWARD, protected_hashes=hashes, args={k: str(v) for k, v in vars(args).items()}))
    if args.route_variant in ("detour", "mixed"):
        from route_variants import straight_detour
        plan = straight_detour(args.base)
        plan["time_budget_steps"] = make_env(args.base, route_variant="detour").max_steps
        plan["time_budget_rule"] = "Same seconds per route metre as the normal route"
        write_json(out / "navigation_plan.json", plan)
    data = dict(np.load(args.data)) if args.data else None
    if data is not None:
        audit_path = args.data.parent / "data_audit.json"
        if audit_path.exists() and json.loads(audit_path.read_text()).get("label_schema") not in (2, 3):
            raise ValueError("Legacy action labels omit junction-exit actions; recollect schema-2 data")
        expected_dim = make_env(args.base).obs_dim
        if data["observation"].shape[1] != expected_dim:
            raise ValueError(f"Dataset has {data['observation'].shape[1]} features; environment needs {expected_dim}")
    if data is None or args.recovery_policy or args.route_variant in ("detour", "mixed"):
        extra = collect(args.base, out, args.episodes, args.recovery_policy,
                        args.failure_targeted, args.seed_start, args.route_variant, args.route_prefix, args.curriculum_mix,
                        args.recovery_lead_steps, args.recovery_window, args.expand_input, args.failure_only, args.seed_list, args.expert_tau)
        if data is not None:
            extra["episode"] += int(data["episode"].max()) + 1
            data = {k: np.concatenate([data[k], extra[k]]) for k in data}
        else:
            data = extra
        np.savez_compressed(out / "demonstrations.npz", **data)
    if args.collect_only:
        assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        print("COLLECTED", out, data["observation"].shape, flush=True)
        raise SystemExit(0)
    agent = fit(data, out, args.epochs, args.lane_weight_power, args.brake_weight,
                args.init_model, args.bc_lr, args.expand_input, args.launch_weight, args.hazard_weight,
                args.balanced_sampling, args.checkpoint_every)
    reports = evaluate(agent, args.base, out, args.eval_episodes, args.eval_seed_start,
                       "normal" if args.route_variant == "mixed" else args.route_variant, args.route_prefix)
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    print("RESULT", str(out), "success", sum(r["end_reason"] == "arrival" for r in reports), "/", len(reports), flush=True)
