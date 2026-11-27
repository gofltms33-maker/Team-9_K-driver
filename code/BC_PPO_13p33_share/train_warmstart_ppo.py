"""Unassisted PPO fine-tuning with demonstration regularization and a separate critic.

Uses saved roads/HV demand and saved BC observation/action definitions. No road
builder, teacher takeover, action filter, lane oracle or automatic route change.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import runpy
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
import torch.nn.functional as F

from algorithms.ppo import PPO
from env.sumo_env import SumoHighwayEnv
from utils.buffer import RolloutBuffer
from utils.logger import RunLogger


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def distribution(policy, obs, temperature):
    mu, std, logits, _ = policy(obs)
    return Normal(mu, std), Categorical(logits=logits / temperature)


def demonstration_lane_weight_power(config,override=None):
    """Preserve explicitly archived navigation supervision across BC -> PPO."""
    value=float(override if override is not None else
                config.get('navigation_supervision',{}).get('lane_weight_power',.5))
    if not math.isfinite(value) or value<0:raise ValueError('Invalid demonstration lane weighting')
    return value


def sampled_action(policy, obs, temperature, acceleration_only=False, lane_only=False):
    if acceleration_only and lane_only:raise ValueError('Select one trained control axis')
    normal, categorical = distribution(policy, obs, temperature)
    acceleration = normal.mean if lane_only else normal.sample()
    lane = categorical.logits.argmax(-1) if acceleration_only else categorical.sample()
    action = torch.cat((acceleration, (1.0 - lane).unsqueeze(-1)), dim=-1)
    log_prob = torch.zeros_like(acceleration[:,0]) if lane_only else normal.log_prob(acceleration).sum(-1)
    if not acceleration_only:
        log_prob = log_prob + categorical.log_prob(lane)
    return action, log_prob


def action_log_prob(policy, obs, action, temperature, acceleration_only=False, lane_only=False):
    if acceleration_only and lane_only:raise ValueError('Select one trained control axis')
    normal, categorical = distribution(policy, obs, temperature)
    lane = (1 - action[:, 1]).long()
    logp, entropy = normal.log_prob(action[:, :1]).sum(-1), normal.entropy().sum(-1)
    if lane_only:logp,entropy=torch.zeros_like(logp),torch.zeros_like(entropy)
    if not acceleration_only:
        logp, entropy = logp + categorical.log_prob(lane), entropy + categorical.entropy()
    return logp, entropy


def freeze_for_lateral_ppo(policy):
    """Learning parameter choice only: keep the entire longitudinal actor fixed."""
    if not hasattr(policy,'lane_adapter'):raise ValueError('Lateral-only refinement requires a learned lane adapter')
    policy.requires_grad_(False)
    policy.lane_adapter.requires_grad_(True)


class SeparateCritic(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.trunk = copy.deepcopy(policy.trunk)
        self.head = copy.deepcopy(policy.v_head)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.requires_grad_(True)

    def forward(self, observation):
        return self.head(self.trunk(observation)).squeeze(-1)


def observed_risk_cost(obs, visibility, vmax, max_decel, dt, signal_horizon=150.0,
                       red_braking=5.4, red_weight=.5):
    """Reward shaping from observed quantities; never modifies an action."""
    speed = float(obs[0]) * vmax
    cost = 0.0
    if obs[10] < .999:  # Existing no-leader sentinel is (distance=1, relative speed=1).
        gap = max(float(obs[9]) * visibility, 0)
        closing = max(-float(obs[10]) * vmax, 0)
        headway_cost = max(1 - gap / (2 + 1.5 * speed), 0) if speed > .5 else 0
        ttc_cost = max(1 - gap / (3 * closing), 0) if closing > .1 else 0
        cost += .15 * headway_cost + .4 * ttc_cost
    if obs[31] < .75 and speed > .5:
        stopping = speed * speed / (2 * red_braking) + speed * dt
        distance = float(obs[32]) * signal_horizon
        cost += red_weight * max(1 - distance / max(stopping + 3, 1), 0)
    # Detailed lane arrows are deliberately redacted in the course profile.
    # Treating those zeros as "no permitted route lane" would penalize every
    # branch approach, including an already correct lane. Risk here uses only
    # observable leader/signal geometry; route completion has its own reward.
    return cost


def saved_environment(config, prefix=None, gui=False):
    base = Path(config["base"])
    road = runpy.run_path(str(base / "road_config.py"))
    route = None
    if prefix is not None:
        route = ET.parse(base / "sumo/highway.rou.xml").getroot().find("route[@id='r0']").get("edges").split()[:prefix]
    env = SumoHighwayEnv(str((base / "sumo/highway.sumocfg").resolve()),
                         road["ROAD"], road["EGO"], config["simulation"], config["observation"],
                         config["action"], config["reward"], traffic=road["TRAFFIC"], planned_route=route,gui=gui)
    if env.native_driver or env.lc_safety_enabled or env.lane_change_cooldown:
        raise ValueError("Expert driving and policy assists are forbidden")
    from sensor_history import with_sensor_history
    return with_sensor_history(env,config["observation"])


def evaluate(agent, config, episodes, seed_start, prefix=None):
    env = saved_environment(config, prefix)
    rows = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            np.random.seed(seed)
            obs, _ = env.reset(seed=seed)
            progress, total = 0.0, 0.0
            for step in range(1, env.max_steps + 1):
                obs, reward, terminated, truncated, info = env.step(agent.predict(obs))
                progress += info["progress_m"]
                total += reward
                if terminated or truncated:
                    break
            rows.append(dict(seed=seed, steps=step, end_reason=info["end_reason"],
                             physical_progress_m=round(progress, 2), ep_return=round(total, 3),
                             control_source=info["control_source"]))
    finally:
        env.close()
    return rows


def summarize(rows):
    n = len(rows)
    result = {reason: sum(r["end_reason"] == reason for r in rows) / n
              for reason in ("arrival", "collision", "red_light", "route_blocked", "idle", "teleport", "stuck", "time_limit")}
    result["progress_m"] = float(np.mean([r["physical_progress_m"] for r in rows]))
    result["ep_return"] = float(np.mean([r["ep_return"] for r in rows]))
    return result


def selection_score(metrics):
    # At equal completion, fewer collisions take precedence over driving farther
    # before crashing. Standing still still cannot outrank a completing policy.
    return (metrics["arrival"], -metrics["collision"], -metrics["teleport"],
            -metrics["red_light"], metrics["progress_m"])


def demonstration_batch(data, rng, batch_size):
    # Balance observed hazard states against ordinary driving, always with real
    # expert labels. This is an optimization sampler, not a driving controller.
    regular, hazard = data["regular"], data["hazard"]
    starts = data.get("starts", np.array([], dtype=int))
    pieces = []
    recovery = data.get("recovery", np.array([], dtype=int))
    if len(hazard):
        pieces.append(rng.choice(hazard, batch_size * (3 if len(recovery) else 4) // 10))
    if len(starts):
        pieces.append(rng.choice(starts, batch_size // 5))
    if len(recovery):
        pieces.append(rng.choice(recovery, batch_size // 4))
    pieces.append(rng.choice(regular, batch_size - sum(len(part) for part in pieces)))
    indices = np.concatenate(pieces)
    return indices


def censored_acceleration_error(mu, target, normalized_speed, vmax=33.33, decel=5.4, dt=.5):
    """Inverse-dynamics targets are sets when commanded speed clips at zero.

    Any sufficiently negative acceleration produces the same demonstrated stop.
    Genuine restart targets retain ordinary error, including its nonzero gradient
    when the learner currently predicts a negative acceleration.
    """
    stop_bound = -normalized_speed * vmax / (decel * dt)
    demonstrated_stop = (target <= 0) & (target <= stop_bound + 1e-4)
    return torch.where(demonstrated_stop, (mu - stop_bound).clamp_min(0), mu - target)


def potential_risk_adjustment(before_risk, after_risk, gamma, terminated, coefficient):
    """gamma*Phi(next)-Phi(current), Phi=-coefficient*risk; terminal Phi=0."""
    return coefficient * (before_risk - gamma * (0.0 if terminated else after_risk))


def intervention_adjusted_reward(reward, requested_speed, realized_speed,
                                 max_accel, max_decel, dt, coefficient=0.0):
    """Training reward only: charge bounded native longitudinal intervention.

    The simulator has already executed the policy action. Its speed correction
    is normalized by the largest physical speed change in one control step.
    Missing terminal speeds carry no inferred cost. No action is replaced.
    """
    if requested_speed is None or realized_speed is None:
        return reward, 0.0
    if not math.isfinite(requested_speed) or not math.isfinite(realized_speed):
        return reward, 0.0
    max_delta = max(max_accel, max_decel) * dt
    if not math.isfinite(max_delta) or max_delta <= 0:
        raise ValueError("A positive finite physical one-step speed delta is required")
    cost = min(abs(requested_speed - realized_speed) / max_delta, 1.0)
    # Preserve the original reward exactly at the default coefficient.
    return (reward if coefficient == 0 else reward - coefficient * cost), cost


def predictive_safety_loss(obs, mu, vmax=33.33, visibility=120.0, signal_horizon=150.0,
                           max_accel=5.4, max_decel=5.4, dt=.5, speed_cap=33.33,
                           reaction_time=None, comfort_decel=None):
    """Differentiable one-step training cost, never an inference-time action filter.

    Uses only observed velocity, leader distance/relative velocity, signal and
    road limit. TraCI leader gap already excludes minGap; 0.5 m is extra buffer.
    This auxiliary objective makes the learner's own predicted acceleration
    receive a gradient for insufficient braking even before a rare collision.
    """
    speed = obs[:, :1] * vmax
    scale = torch.where(mu >= 0, max_accel, max_decel)
    next_speed = (speed + mu * scale * dt).clamp(0, speed_cap)
    leader_speed = (speed + obs[:, 10:11] * vmax).clamp_min(0)
    next_gap = obs[:, 9:10] * visibility + (leader_speed - next_speed) * dt
    reaction = dt if reaction_time is None else reaction_time
    braking = max_decel if comfort_decel is None else comfort_decel
    # A faster leader does not cancel reaction headway. Clamp only the relative
    # braking term, otherwise closing-speed changes can leave almost zero gap.
    stopping_gap = next_speed * reaction + (next_speed.square() - leader_speed.square()).clamp_min(0) / (2 * braking)
    present = (obs[:, 10:11] < .999).float()
    leader_cost = present * (((stopping_gap + .5 - next_gap) / 20).clamp_min(0).square()
                             + 5 * ((.5 - next_gap) / 5).clamp_min(0).square())
    next_signal_distance = obs[:, 32:33] * signal_horizon - next_speed * dt
    signal_cost = (1 - obs[:, 31:32]).clamp(0, 1) * (
        ((next_speed.square() / 4 + next_speed * dt + 3 - next_signal_distance) / 20).clamp_min(0).square()
        + 5 * ((3 - next_signal_distance) / 5).clamp_min(0).square())
    overspeed_cost = ((next_speed - obs[:, 37:38] * vmax) / vmax).clamp_min(0).square()
    return (leader_cost + signal_cost + overspeed_cost).mean()


def spatial_collision_costs(obs, mu, vmax=33.33, visibility=120.0, dt=.5, lane_width=3.2,lookahead=1.):
    """Training-only predicted overlap costs for neural left/keep/right decisions.

    Nearby features contain front-bumper positions. Moving vehicles' headings
    are inferred from observed velocity to approximate body centers/footprints.
    No route, priority, future simulator state, or action override is queried.
    SUMO moves vehicles longitudinally before applying lane changes. A requested
    adjacent lane is therefore occupied by the ego from dt onward, while the
    keep action also retains the ego's current occupancy at time zero.
    """
    if obs.shape[1] < 100:
        raise ValueError("Spatial cost requires eight observed nearby vehicles")
    nearby = obs[:, -40:].reshape(-1, 8, 5)
    speed = obs[:, :1] * vmax
    next_speed = (speed + mu * 5.4 * dt).clamp(0, vmax)
    other_forward = speed + nearby[:, :, 2] * vmax
    other_left = nearby[:, :, 3] * vmax
    other_speed = torch.sqrt(other_forward.square() + other_left.square()).clamp_min(.1)
    heading_forward = torch.where(other_speed > .1, other_forward / other_speed, 1.0)
    heading_left = torch.where(other_speed > .1, other_left / other_speed, 0.0)
    forward_position = nearby[:, :, 0] * visibility + 2.5 * (1 - heading_forward)
    left_position = nearby[:, :, 1] * visibility - 2.5 * heading_left
    # Nominal passenger-car footprints, including SUMO minGap along the road.
    length_extent = 2.5 + 2.5 * heading_forward.abs() + heading_left.abs() + 3.0
    width_extent = 1.0 + 2.5 * heading_left.abs() + heading_forward.abs() + .2
    shifts = torch.as_tensor([lane_width, 0.0, -lane_width], dtype=obs.dtype, device=obs.device)
    candidates = []
    for horizon in (0.0, dt, max(lookahead, dt)):
        forward = forward_position + (nearby[:, :, 2] * vmax - (next_speed - speed)) * horizon
        left = left_position + nearby[:, :, 3] * vmax * horizon
        longitudinal_overlap = (1 - forward.abs() / length_extent).clamp_min(0)
        lateral_overlap = (1 - (left.unsqueeze(-1) - shifts).abs() / width_extent.unsqueeze(-1)).clamp_min(0)
        overlap = longitudinal_overlap.unsqueeze(-1) * lateral_overlap * nearby[:, :, 4:5]
        at_horizon = overlap.max(dim=1).values
        if horizon == 0.0:
            # Shifting the ego fully sideways at t=0 would incorrectly reject
            # a neighboring vehicle that clears before the lane-change phase.
            at_horizon = at_horizon * at_horizon.new_tensor([0.0, 1.0, 0.0])
        candidates.append(at_horizon)
    costs = torch.stack(candidates).max(dim=0).values
    invalid = torch.stack(((obs[:, 22] < 0).float(), torch.zeros_like(obs[:, 22]),
                            (obs[:, 24] < 0).float()), dim=-1)
    return costs + 2 * invalid


def demonstration_loss(policy, data, indices, censored_stop=False):
    obs = data["observation"][indices]
    targets = data["action"][indices]
    mu, _, logits, _ = policy(obs)
    accel_mask, lane_mask = data["accel_valid"][indices], data["lane_valid"][indices]
    regression = mu.sum() * 0
    if accel_mask.any():
        weights = torch.where((obs[accel_mask, :1] < .03) & (targets[accel_mask, :1] > .1), 5.0, 1.0)
        error = (censored_acceleration_error(mu[accel_mask], targets[accel_mask, :1], obs[accel_mask, :1])
                 if censored_stop else mu[accel_mask] - targets[accel_mask, :1])
        regression = (error.square() * weights).mean()
    classification = logits.sum() * 0
    if lane_mask.any():
        classification = F.cross_entropy(logits[lane_mask], (1 - targets[lane_mask, 1]).long(),
                                          weight=data["class_weights"])
    return regression + classification


def load_demonstrations(paths, obs_dim, heldout=False, expected_observation=None, lane_weight_power=.5):
    arrays = []
    for path in paths:
        route_distance_enabled = False
        source = {}
        audit = path.parent / "data_audit.json"
        audit_data = json.loads(audit.read_text(encoding="utf-8")) if audit.exists() else {}
        if audit_data.get("label_schema") not in (2, 3):
            raise ValueError("Only audited schema-2/3 demonstrations are accepted: " + str(path))
        if expected_observation is not None:
            metadata = path.parent / "run_config.json"
            source = json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else {}
            source_obs = source.get("observation", audit_data.get("observation", {}))
            for key,default in (("visibility",120.),("nav_horizon",300.),("signal_horizon",150.)):
                if not math.isclose(float(source_obs.get(key,default)),float(expected_observation.get(key,default)),rel_tol=1e-9):
                    raise ValueError("Observation normalization differs for "+key+": "+str(path))
            if int(source_obs.get("sensor_history_frames", 1)) != int(expected_observation.get("sensor_history_frames", 1)):
                raise ValueError("Sensor history layout differs: " + str(path))
            if bool(source_obs.get("retain_overlap_neighbors", False)) != bool(expected_observation.get("retain_overlap_neighbors", False)):
                raise ValueError("Neighbor observation semantics changed; recollect demonstrations: " + str(path))
            if bool(source_obs.get('physical_layout_features',False))!=bool(expected_observation.get('physical_layout_features',False)):
                raise ValueError('Physical own-layout inputs differ: '+str(path))
            if bool(source_obs.get("semantic_lane_branches", False)) != bool(expected_observation.get("semantic_lane_branches", False)):
                raise ValueError("Navigation observation semantics changed; recollect demonstrations: " + str(path))
            if bool(source_obs.get("redact_lane_guidance", False)) != bool(expected_observation.get("redact_lane_guidance", False)):
                raise ValueError("Detailed lane guidance redaction differs: " + str(path))
            route_distance_enabled = bool(expected_observation.get("route_distance_feature", False))
            if bool(source_obs.get("route_distance_feature", False)) != route_distance_enabled:
                raise ValueError("Route distance observation feature differs: " + str(path))
            if route_distance_enabled:
                source_scale = float(source_obs.get("route_distance_scale_m", 10000.0))
                expected_scale = float(expected_observation.get("route_distance_scale_m", 10000.0))
                if (not math.isfinite(source_scale) or not math.isfinite(expected_scale)
                        or source_scale <= 0 or expected_scale <= 0
                        or not math.isclose(source_scale, expected_scale, rel_tol=0.0, abs_tol=1e-9)):
                    raise ValueError("Route distance normalization scale differs or is invalid: " + str(path))
        with np.load(path) as archive:
            a = {key: archive[key] for key in ("observation", "action", "accel_valid", "lane_valid", "episode")}
        # Priorities affect training-row sampling only. Near-failure recovery
        # beginnings contain the learner-induced state before long easy suffixes.
        # Derivatives of already audited teacher data can explicitly mark an
        # offline curriculum subset; no label or policy input is synthesized.
        a['priority_mask'] = np.full(len(a['episode']),bool(audit_data.get('training_focus',False)),bool)
        lead=source.get('collection',{}).get('lead_steps')
        if audit_data.get('expert_suffix') and lead is not None and int(lead)<=12:
            for seed in np.unique(a['episode']):
                a['priority_mask'][np.flatnonzero(a['episode']==seed)[:16]]=True
        a["recovery_mask"] = np.full(len(a["episode"]),
            audit_data.get("expert_suffix", False) or audit_data.get("learner_observed_states", False), bool)
        if audit_data.get("native_counterfactual_queries"):
            # "Without TraCI speed commands" does not undo speedMode=96:
            # ignoreRed/junction flags still affect the native speed proposal.
            # These speeds are diagnostics, not a restored mode-31 expert.
            a["accel_valid"][:] = False
            # Prioritize genuine model-proposal labels immediately before this
            # learner's failures, rather than oversampling an entire easy lap.
            a["recovery_mask"][:] = False
            for episode in audit_data.get("episodes", []):
                if episode["end_reason"] != "arrival":
                    indices = np.flatnonzero(a["episode"] == episode["seed"])
                    a["recovery_mask"][indices[-48:]] = True
        if a["observation"].shape[1] != obs_dim:
            raise ValueError("Demonstration observation mismatch")
        if expected_observation:
            from sensor_history import validate_history_data
            validate_history_data(a["observation"],expected_observation)
        if route_distance_enabled:
            if obs_dim <= 53:
                raise ValueError("Route distance requires reserved observation column 53")
            distance = a["observation"][:, 53]
            if not (np.isfinite(distance).all() and ((distance >= 0) & (distance <= 1)).all()):
                raise ValueError("Route distance column 53 must be finite and within [0, 1]: " + str(path))
        if expected_observation and expected_observation.get("redact_lane_guidance", False):
            redacted = a["observation"][:, 38:55]
            if route_distance_enabled:
                # Measured odometer/GPS distance is the sole permitted use of
                # this reserved slot; turn/lane guidance remains unavailable.
                redacted = np.concatenate((redacted[:, :15], redacted[:, 16:]), axis=1)
            if not np.all(redacted == 0):
                raise ValueError("Redacted demonstrations contain detailed lane guidance: " + str(path))
        if obs_dim >= 60 and audit_data.get("label_schema") == 2:
            # Only archived schema-2 data marked native right-at-exit moves from
            # internal lane zero as valid, although TraCI discards target -1.
            # Internal lanes have all five connectivity values == 1 (even
            # nonexistent offsets), zero turn markings and normalized index 0.
            # A normal lane at index 0 necessarily has absent right offsets.
            # Schema 3 already masks these using the actual pre-action lane.
            # Redacted lane fields cannot be used to infer internal-lane IDs.
            if expected_observation and expected_observation.get("redact_lane_guidance", False):
                raise ValueError("Schema-2 data must be upgraded before lane-field redaction")
            o = a["observation"]
            internal_zero = ((o[:, 21:26] == 1).all(axis=1) & (o[:, 38:53] == 0).all(axis=1)
                             & (o[:, 54] == 0))
            a["lane_valid"][internal_zero & (a["action"][:, 1] < 0)] = False
        # Do not teach anticipation of an unobserved red-to-green switch.
        green_switch = ((a["episode"][:-1] == a["episode"][1:]) &
                        (a["observation"][:-1, 31] < .25) &
                        (a["observation"][1:, 31] > .75) & (a["action"][:-1, 0] > .1))
        a["accel_valid"][np.flatnonzero(green_switch)] = False
        # Never optimize on held-out demonstration episodes.
        keep = a["episode"] % 5 == 4 if heldout else a["episode"] % 5 != 4
        arrays.append({key: value[keep] for key, value in a.items()})
    merged = {key: np.concatenate([a[key] for a in arrays]) for key in arrays[0]}
    obs = merged["observation"]
    hazard = ((obs[:, 31] < .75) & (obs[:, 32] < .35) & (obs[:, 0] > .03))
    hazard |= (obs[:, 9] < .3) & (obs[:, 10] < -.015)
    data = {key: torch.from_numpy(merged[key]) for key in ("observation", "action", "accel_valid", "lane_valid")}
    data["regular"] = np.arange(len(obs))
    data["hazard"] = np.flatnonzero(hazard)
    data["starts"] = np.flatnonzero((obs[:, 0] < .03) & (merged["action"][:, 0] > .1)
                                    & (obs[:, 31] > .75) & merged["accel_valid"])
    data["recovery"] = np.flatnonzero(merged["recovery_mask"])
    data['priority'] = np.flatnonzero(merged['priority_mask'] &
                                     (merged['accel_valid'] | merged['lane_valid']))
    counts = torch.bincount((1 - data["action"][data["lane_valid"], 1]).long(), minlength=3).float().clamp_min(1)
    weights = (counts.sum() / counts).pow(lane_weight_power)
    data["class_weights"] = weights / weights.mean()
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-run", type=Path, default=Path("results/bc_curriculum_stage3_v12"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs="+", default=[Path("results/bc_curriculum_stage3_v12/demonstrations.npz"), Path("results/bc_curriculum_full_v13/demonstrations.npz")])
    parser.add_argument("--steps", type=int, default=40960)
    parser.add_argument("--rollout", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=.995)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--clip", type=float, default=.1)
    parser.add_argument("--target-kl", type=float, default=.01)
    parser.add_argument("--bc-coef", type=float, default=.5)
    parser.add_argument('--lane-weight-power',type=float,
                        help='BC lane-class weighting; defaults to archived navigation supervision or legacy .5')
    parser.add_argument("--std", type=float, default=.06)
    parser.add_argument("--lane-temperature", type=float, default=.5)
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed-start", type=int, default=1500)
    parser.add_argument("--route-prefix", type=int)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--train-seed-start",type=int,default=20000)
    parser.add_argument("--train-seed-stop",type=int,default=1000000,
                        help="Exclusive upper bound; choose disjoint training/evaluation seed ranges")
    parser.add_argument("--reward-scale", type=float, default=.02)
    parser.add_argument("--collision-penalty", type=float, default=100.0)
    parser.add_argument("--red-light-penalty", type=float, default=100.0)
    parser.add_argument("--failure-penalty", type=float, default=30.0)
    parser.add_argument("--arrival-bonus", type=float, default=150.0)
    parser.add_argument("--selection-collision-limit", type=float,
                        help="Optional checkpoint selection constraint: collision rate strictly below this limit")
    parser.add_argument("--risk-coef", type=float, default=1.0)
    parser.add_argument("--red-braking", type=float, default=2.0,
                        help="Comfortable deceleration used in reward shaping only")
    parser.add_argument("--red-weight", type=float, default=2.0)
    parser.add_argument("--censored-stop-loss", action="store_true")
    parser.add_argument("--suffix-curriculum", action="store_true",
                        help="Training only: mix full starts and valid later starts, always evaluate full circuit")
    parser.add_argument("--curriculum-warmup-seconds", type=float, default=600.0)
    parser.add_argument("--acceleration-only", action="store_true",
                        help="PPO updates the learned acceleration tower; learned lane tower stays deterministic and frozen")
    parser.add_argument('--lane-only',action='store_true',
                        help='PPO updates only the learned lane adapter; acceleration is frozen and deterministic')
    parser.add_argument("--risk-shaping", choices=("cost", "potential"), default="cost")
    parser.add_argument("--predictive-safety-coef", type=float, default=0.0)
    parser.add_argument("--strict-speed-mode", action="store_true",
                        help="Explicitly disable all SUMO speed safety checks (mode 96)")
    parser.add_argument("--predictive-reaction-time", type=float, default=.5)
    parser.add_argument("--intervention-coef", type=float, default=0.0,
                        help="Training-only penalty for normalized requested/realized speed corrections")
    args = parser.parse_args()
    if args.lane_only and args.acceleration_only:parser.error('Choose only one frozen-axis mode')
    if min(args.steps, args.rollout, args.epochs, args.eval_every, args.eval_episodes) < 1:
        parser.error("Step, update and evaluation counts must be positive")
    if not 0<=args.train_seed_start<args.train_seed_stop:
        parser.error('Training seed bounds must be increasing nonnegative integers')
    if not (0 < args.std < 1 and 0 < args.lane_temperature <= 2):
        parser.error("Invalid exploration parameters")
    if not math.isfinite(args.intervention_coef) or args.intervention_coef < 0:
        parser.error("--intervention-coef must be finite and nonnegative")
    for name in ("collision_penalty", "red_light_penalty", "failure_penalty", "arrival_bonus"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error("--" + name.replace("_", "-") + " must be finite and positive")
    if args.selection_collision_limit is not None and (
            not math.isfinite(args.selection_collision_limit) or not 0 < args.selection_collision_limit <= 1):
        parser.error("--selection-collision-limit must be finite and in (0, 1]")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.init_run / "run_config.json").read_text(encoding="utf-8"))
    config['initialization']=dict(run=str(args.init_run),model_sha256=hashlib.sha256((args.init_run/'model.pt').read_bytes()).hexdigest())
    args.lane_weight_power=demonstration_lane_weight_power(config,args.lane_weight_power)
    config["stage"] = "PPO fine-tuning with BC auxiliary loss and separate critic"
    if args.strict_speed_mode:
        config["action"]["policy_speed_mode"] = 96
    config["hparams"].update(lr=args.lr, n_steps=args.rollout, n_epochs=args.epochs,
                              gamma=args.gamma, gae_lambda=args.gae_lambda,
                              clip_eps=args.clip, target_kl=args.target_kl)
    config["args"] = {key: str(value) for key, value in vars(args).items()}
    config["args"]["route_variant"] = "normal"
    config["reward"] = {**config["reward"], "speed_weight": 0.0, "progress_weight": .01,
                         "collision_penalty": args.collision_penalty, "red_light_penalty": args.red_light_penalty,
                         "failure_penalty": args.failure_penalty, "arrival_bonus": args.arrival_bonus,
                         "overspeed_penalty": .5, "close_gap_penalty": .1,
                         "idle_penalty": .05, "lane_change_penalty": .02}
    config["fine_tuning"] = {"separate_critic": True, "reward_scale": args.reward_scale,
                              "observed_risk_cost": True, "lane_temperature": args.lane_temperature,
                              "redacted_lane_arrow_risk_removed": True,
                              "fixed_acceleration_std": args.std, "bc_coefficient": args.bc_coef,
                              "red_braking": args.red_braking, "red_weight": args.red_weight,
                              "training_suffix_curriculum": args.suffix_curriculum,
                              "acceleration_only": args.acceleration_only,
                              "lane_only": args.lane_only,
                              "risk_shaping": args.risk_shaping,
                              "predictive_safety_coefficient": args.predictive_safety_coef,
                              "predictive_reaction_time": args.predictive_reaction_time,
                              "native_speed_intervention_coefficient": args.intervention_coef,
                              "native_speed_intervention_normalization": "abs(requested-realized)/(max(accel,decel)*dt), capped at 1; training reward only",
                              "selection_collision_limit": args.selection_collision_limit,
                              "curriculum_warmup_seconds_max": args.curriculum_warmup_seconds,
                              "train_seed_range": [args.train_seed_start, args.train_seed_stop], "selection_seeds": [args.eval_seed_start, args.eval_seed_start + args.eval_episodes - 1]}
    for filename, digest in config["protected_hashes"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest
    write_json(out / "run_config.json", config)
    for relative in ("train_warmstart_ppo.py", "sensor_history.py", "physical_layout.py", "env/sumo_env.py", "utils/networks.py", "utils/buffer.py", "algorithms/ppo.py"):
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative, destination)
    env = saved_environment(config, args.route_prefix)
    if args.suffix_curriculum and args.route_prefix is not None:
        raise ValueError("Suffix training requires full-route evaluation")
    base_route = ET.parse(Path(config["base"]) / "sumo/highway.rou.xml").getroot().find("route[@id='r0']").get("edges").split()
    eligible_starts = []
    if args.suffix_curriculum:
        import sumolib
        net = sumolib.net.readNet(str(Path(config["base"]) / "sumo/highway.net.xml"))
        eligible_starts = [index for index in range(1, len(base_route) - 1)
                           if net.getEdge(base_route[index]).getLength() > 100]
    agent = PPO(env.obs_dim, 2, **{**config["hparams"], "device": "cpu"})
    agent.load(str(args.init_run / "model.pt"))
    policy = agent.policy
    policy.v_head.requires_grad_(False)
    policy.log_std.requires_grad_(False)
    if args.acceleration_only:
        if config["hparams"]["policy_type"] not in ("split_hybrid", "lane_residual_hybrid", "set_residual_hybrid", "metric_set_residual_hybrid", "invariant_metric_set_hybrid"):
            raise ValueError("Acceleration-only PPO requires independent split actor towers")
        policy.trunk.requires_grad_(False)
        policy.lane_head.requires_grad_(False)
        policy.mu_head.requires_grad_(False)
        if hasattr(policy, "lane_adapter"):
            policy.lane_adapter.requires_grad_(False)
        if hasattr(policy, "accel_adapter"):
            policy.accel_trunk.requires_grad_(False)
            policy.accel_head.requires_grad_(False)
    if args.lane_only:freeze_for_lateral_ppo(policy)
    else:
        with torch.no_grad():policy.log_std.fill_(math.log(args.std))
    actor = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=args.lr)
    critic = SeparateCritic(policy)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    # PPO's constructor seeds Torch from the archived initialization config.
    # Restore this run's explicit seed before sampling training actions.
    torch.manual_seed(args.seed)
    demos = load_demonstrations(args.data, env.obs_dim, expected_observation=config["observation"],
                                lane_weight_power=args.lane_weight_power)
    logger = RunLogger(str(out))
    episodes, evaluations = [], []
    started = time.monotonic()
    def reset():
        seed = int(rng.integers(args.train_seed_start, args.train_seed_stop))
        start_index = 0
        if args.suffix_curriculum and eligible_starts and rng.random() >= .4:
            start_index = int(rng.choice(eligible_starts))
        if args.suffix_curriculum:
            env.planned_route = base_route[start_index:] if start_index else None
            env.warmup_steps = int(config["simulation"]["warmup_steps"])
            if start_index:
                env.warmup_steps += int(rng.uniform(0, args.curriculum_warmup_seconds) / env.step_length)
        env.training_start_index = start_index
        np.random.seed(seed)
        observation, _ = env.reset(seed=seed)
        return observation, seed
    def value(observation):
        with torch.no_grad():
            return float(critic(torch.as_tensor(observation).unsqueeze(0)).item())
    def checkpoint_score(metrics):
        score = selection_score(metrics)
        if args.selection_collision_limit is None:
            return score
        return (metrics["collision"] < args.selection_collision_limit, *score)
    best = None
    obs = None
    step_count, update, ep_return, ep_length, ep_progress = 0, 0, 0.0, 0, 0.0
    ep_intervention_cost, ep_intervention_steps = 0.0, 0
    try:
        # Baseline is included in checkpoint selection so regression is visible.
        baseline = evaluate(agent, config, args.eval_episodes, args.eval_seed_start, args.route_prefix)
        metrics = summarize(baseline)
        best = checkpoint_score(metrics)
        agent.save(str(out / "model.pt"))
        write_json(out / "evaluation.json", baseline)
        evaluations.append(dict(steps=0, **metrics))
        logger.log_eval(0, metrics, str(metrics), len(baseline))
        obs, train_seed = reset()
        while step_count < args.steps:
            buffer = RolloutBuffer()
            intervention_costs = []
            for _ in range(min(args.rollout, args.steps - step_count)):
                with torch.no_grad():
                    action_t, logp_t = sampled_action(policy, torch.as_tensor(obs).unsqueeze(0), args.lane_temperature, args.acceleration_only,args.lane_only)
                action = action_t[0].numpy().astype(np.float32)
                next_obs, reward, terminated, truncated, info = env.step(action)
                shaped_reward, intervention_cost = intervention_adjusted_reward(
                    reward, info.get("requested_speed"), info.get("realized_speed"),
                    env.max_accel, env.max_decel, env.step_length, args.intervention_coef)
                intervention_costs.append(intervention_cost)
                if args.risk_shaping == "potential":
                    before_risk = observed_risk_cost(obs, env.visibility, env.vmax, env.max_decel, env.step_length, env.signal_horizon, args.red_braking, args.red_weight)
                    after_risk = (0.0 if terminated else observed_risk_cost(next_obs, env.visibility, env.vmax, env.max_decel, env.step_length, env.signal_horizon, args.red_braking, args.red_weight))
                    shaped_reward += potential_risk_adjustment(before_risk, after_risk, args.gamma, terminated, args.risk_coef)
                elif not (terminated or truncated):
                    shaped_reward -= args.risk_coef * observed_risk_cost(next_obs, env.visibility, env.vmax, env.max_decel, env.step_length, env.signal_horizon, args.red_braking, args.red_weight)
                stored_reward = args.reward_scale * shaped_reward
                if truncated and not terminated:
                    stored_reward += args.gamma * value(next_obs)
                buffer.add(obs, action, logp_t.item(), stored_reward, value(obs), terminated or truncated)
                step_count += 1
                ep_return += reward
                ep_progress += info["progress_m"]
                ep_length += 1
                ep_intervention_cost += intervention_cost
                ep_intervention_steps += int(intervention_cost > 1e-4)
                if terminated or truncated:
                    episodes.append(dict(steps=step_count, seed=train_seed, ep_return=ep_return, length=ep_length,
                                         progress_m=ep_progress, end_reason=info["end_reason"],
                                         native_speed_intervention_cost_sum=ep_intervention_cost,
                                         native_speed_intervention_penalty_sum=args.intervention_coef * ep_intervention_cost,
                                         native_speed_intervention_steps=ep_intervention_steps,
                                         native_speed_intervention_fraction=ep_intervention_steps / ep_length,
                                         route_start_index=env.training_start_index))
                    logger.log_episode(step_count, ep_return, ep_length)
                    obs, train_seed = reset()
                    ep_return, ep_length, ep_progress = 0.0, 0, 0.0
                    ep_intervention_cost, ep_intervention_steps = 0.0, 0
                else:
                    obs = next_obs
            advantages, returns = buffer.compute_gae(value(obs), args.gamma, args.gae_lambda)
            observations, actions, old_logp = buffer.to_tensors("cpu")
            advantages = torch.from_numpy(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            returns = torch.from_numpy(returns)
            warmup = update < args.warmup_updates
            totals = {key: [] for key in ("policy_loss", "bc_loss", "value_loss", "approx_kl", "clip_frac")}
            stop_actor = False
            for _ in range(args.epochs):
                for batch in np.array_split(rng.permutation(len(buffer)), math.ceil(len(buffer) / 128)):
                    predicted_value = critic(observations[batch])
                    critic_loss = F.mse_loss(predicted_value, returns[batch])
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), .5)
                    critic_optimizer.step()
                    totals["value_loss"].append(critic_loss.item())
                    if warmup or stop_actor:
                        continue
                    new_logp, entropy = action_log_prob(policy, observations[batch], actions[batch], args.lane_temperature, args.acceleration_only,args.lane_only)
                    log_ratio = new_logp - old_logp[batch]
                    ratio = log_ratio.exp()
                    with torch.no_grad():
                        kl = ((ratio - 1) - log_ratio).mean().item()
                    if kl > 1.5 * args.target_kl:
                        stop_actor = True
                        continue
                    surrogate = torch.minimum(ratio * advantages[batch], ratio.clamp(1 - args.clip, 1 + args.clip) * advantages[batch])
                    policy_loss = -surrogate.mean()
                    anchor = demonstration_loss(policy, demos, demonstration_batch(demos, rng, 256), args.censored_stop_loss)
                    safety_loss = policy_loss * 0
                    if args.predictive_safety_coef:
                        mean_acceleration = policy(observations[batch])[0]
                        safety_loss = predictive_safety_loss(observations[batch], mean_acceleration,
                            env.vmax, env.visibility, env.signal_horizon, env.max_accel,
                            env.max_decel, env.step_length, env.control_speed_cap, args.predictive_reaction_time)
                    loss = policy_loss + args.bc_coef * anchor + args.predictive_safety_coef * safety_loss
                    actor.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], .5)
                    actor.step()
                    totals["policy_loss"].append(policy_loss.item())
                    totals["bc_loss"].append(anchor.item())
                    totals["approx_kl"].append(kl)
                    totals["clip_frac"].append(float(((ratio - 1).abs() > args.clip).float().mean()))
            update += 1
            stats = {key: float(np.mean(vals)) if vals else 0.0 for key, vals in totals.items()}
            recent = episodes[-20:]
            stats.update(update=update, warmup=float(warmup), early_stop=float(stop_actor),
                         native_speed_intervention_mean=float(np.mean(intervention_costs)),
                         native_speed_intervention_fraction=float(np.mean(np.asarray(intervention_costs) > 1e-4)),
                         native_speed_intervention_penalty_mean=args.intervention_coef * float(np.mean(intervention_costs)),
                         ep_rew_mean=float(np.mean([r["ep_return"] for r in recent])) if recent else 0.0,
                         training_arrival=float(np.mean([r["end_reason"] == "arrival" for r in recent])) if recent else 0.0,
                         elapsed_seconds=round(time.monotonic() - started, 1), std=policy.current_std)
            logger.log_train(step_count, stats)
            agent.save(str(out / "model_latest.pt"))
            torch.save(dict(critic=critic.state_dict(), actor_optimizer=actor.state_dict(),
                            critic_optimizer=critic_optimizer.state_dict(), steps=step_count), out / "training_state.pt")
            logger.save_csv()
            write_json(out / "episodes.json", episodes)
            write_json(out / "status.json", dict(status="running", steps=step_count, target_steps=args.steps, **stats))
            if update % args.eval_every == 0 or step_count == args.steps:
                env.close()
                rows = evaluate(agent, config, args.eval_episodes, args.eval_seed_start, args.route_prefix)
                metrics = summarize(rows)
                evaluations.append(dict(steps=step_count, **metrics))
                directory = out / "evaluations"
                directory.mkdir(exist_ok=True)
                write_json(directory / f"evaluation_{step_count}.json", rows)
                logger.log_eval(step_count, metrics, str(metrics), len(rows))
                candidate_score = checkpoint_score(metrics)
                if candidate_score > best:
                    best = candidate_score
                    agent.save(str(out / "model.pt"))
                    write_json(out / "evaluation.json", rows)
                    write_json(out / "best_checkpoint.json", dict(steps=step_count, **metrics))
                obs, train_seed = reset()
                ep_return, ep_length, ep_progress = 0.0, 0, 0.0
        agent.save(str(out / "model_last.pt"))
        write_json(out / "evaluations.json", evaluations)
        for filename, digest in config["protected_hashes"].items():
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest
        write_json(out / "status.json", dict(status="completed", steps=step_count,
                                             elapsed_seconds=round(time.monotonic() - started, 1),
                                             best_selection_score=best))
    except BaseException as error:
        write_json(out / "status.json", dict(status="failed", steps=step_count, error=repr(error)))
        raise
    finally:
        env.close()
        logger.close()


if __name__ == "__main__":
    main()
