"""Fit a learned longitudinal tower on audited demonstrations, preserving BC lane behavior."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn

from algorithms.ppo import PPO
from train_warmstart_ppo import (load_demonstrations, demonstration_batch, evaluate,
                                 summarize, write_json, saved_environment, censored_acceleration_error,
                                 selection_score, predictive_safety_loss, spatial_collision_costs)
import torch.nn.functional as F


def acceleration_loss(policy, data, indices, underbrake_weight, launch_weight=5.0, residual_penalty=0.0,
                      censored_stop=False, safety_coefficient=0.0, lane_coefficient=0.0,
                      lane_safety_coefficient=0.0, spatial_safety_coefficient=0.0,
                      reference_policy=None, reference_coefficient=0.0, reaction_time=None,
                      lane_reference_coefficient=None, comfort_decel=None, risk_weighted_reference=False,
                      safety_scales=None,spatial_horizon=1.,brake_weight_scope='hazard'):
    obs = data["observation"][indices]
    target = data["action"][indices, :1]
    valid = data["accel_valid"][indices]
    mu, _, logits, _ = policy(obs)
    weights = torch.where((obs[:, :1] < .03) & (target > .1), launch_weight, 1.0)
    hazard = (((obs[:, 31:32] < .75) & (obs[:, 32:33] < .35) & (obs[:, :1] > .03)) |
              ((obs[:, 9:10] < .3) & (obs[:, 10:11] < -.015)))
    if brake_weight_scope not in ('hazard','all'):
        raise ValueError('Unknown supervised braking-weight scope')
    # A close leader can require deceleration before relative velocity becomes
    # negative. The optional all-scope uses real teacher braking targets rather
    # than making stronger supervision depend on that particular hazard test.
    # This changes the training objective only, never inference-time actions.
    braking = (target < -.05) & (mu > target)
    if brake_weight_scope == 'hazard':braking &= hazard
    weights *= torch.where(braking, underbrake_weight, 1.0)
    error = censored_acceleration_error(mu, target, obs[:, :1]) if censored_stop else mu - target
    loss = (error.square() * weights)[valid].mean() if valid.any() else mu.sum() * 0
    if residual_penalty:
        correction = policy.accel_head(policy.accel_trunk(policy.local_features(obs)))
        loss += residual_penalty * correction.square().mean()
    if safety_coefficient:
        loss += safety_coefficient * predictive_safety_loss(obs, mu, reaction_time=reaction_time,
            comfort_decel=comfort_decel, **(safety_scales or {}))
    if lane_coefficient:
        lane_mask = data["lane_valid"][indices]
        if lane_mask.any():
            loss += lane_coefficient * F.cross_entropy(logits[lane_mask],
                (1 - data["action"][indices, 1][lane_mask]).long(), weight=data["class_weights"])
    if lane_safety_coefficient or spatial_safety_coefficient:
        scales=safety_scales or {}
        costs = spatial_collision_costs(obs, mu,lookahead=spatial_horizon,
            **{k:scales[k] for k in ('vmax','visibility','dt') if k in scales})
        if lane_safety_coefficient:
            loss += lane_safety_coefficient * (torch.softmax(logits / .5, -1) * costs.detach()).sum(-1).mean()
        if spatial_safety_coefficient:
            chosen = logits.detach().argmax(-1)
            loss += spatial_safety_coefficient * costs.gather(1, chosen.unsqueeze(-1)).mean()
    if reference_policy is not None:
        with torch.no_grad():
            reference_mu, _, reference_logits, _ = reference_policy(obs)
        lane_reference = reference_coefficient if lane_reference_coefficient is None else lane_reference_coefficient
        loss += reference_coefficient * (mu - reference_mu).square().mean()
        lane_kl = F.kl_div(F.log_softmax(logits, -1), F.softmax(reference_logits, -1), reduction="none").sum(-1)
        if risk_weighted_reference:
            with torch.no_grad():
                reference_cost = spatial_collision_costs(obs, reference_mu).gather(1, reference_logits.argmax(-1, keepdim=True)).squeeze(-1)
                reference_weight = 1 / (1 + 10 * reference_cost)
            lane_kl = lane_kl * reference_weight
        loss += lane_reference * lane_kl.mean()
    return loss


def lane_balanced_batch(data, rng, size, fraction):
    """Mix natural training rows with equal left/keep/right teacher labels.

    This changes supervised-data sampling only. It never masks policy actions,
    selects simulator scenarios, or changes the evaluation distribution.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("Lane class fraction must be within [0,1]")
    if "lane_class_rows" not in data:
        actions = data["action"][:, 1].cpu().numpy()
        valid = data["lane_valid"].cpu().numpy()
        regular = np.asarray(data["regular"])
        data["lane_class_rows"] = [regular[valid[regular] & (actions[regular] == label)]
                                   for label in (1, 0, -1)]
    count = int(size * fraction) // 3
    pieces = []
    for rows in data["lane_class_rows"]:
        if len(rows) and count:
            pieces.append(rng.choice(rows, count))
    pieces.append(rng.choice(data["regular"], size - sum(map(len, pieces))))
    return np.concatenate(pieces)


def stop_hold_batch(data,rng,size,fraction,lane_fraction=0.,uniform=False):
    """Training-only sampling of actual expert stops, excluding restart labels."""
    if 'stop_hold_rows' not in data:
        obs=data['observation'].numpy();action=data['action'].numpy()
        regular=np.asarray(data['regular']);valid=data['accel_valid'].numpy()
        stopped=(obs[:,0]<.03)&(action[:,0]<=0)&valid
        waiting=((obs[:,9]<.1)&(obs[:,10]<.999))|((obs[:,31]<.25)&(obs[:,32]<.1))
        data['stop_hold_rows']=regular[(stopped&waiting)[regular]]
    rows=data['stop_hold_rows'];count=int(size*fraction) if len(rows) else 0;rest=size-count
    if not rest:return rng.choice(rows,count)
    base=(lane_balanced_batch(data,rng,rest,lane_fraction) if lane_fraction else
          rng.choice(data['regular'],rest) if uniform else demonstration_batch(data,rng,rest))
    return np.concatenate((base,rng.choice(rows,count))) if count else base


def configure_trainable_parameters(policy, args):
    """Select BC actor parameters while keeping critic heads/std frozen."""
    train_backbone = bool(getattr(args, "train_backbone", False))
    train_lane_backbone = bool(getattr(args, "train_lane_backbone", False))
    train_accel_backbone = bool(getattr(args, "train_accel_backbone", False))
    if train_backbone and (args.lane_only or not args.train_lanes):
        raise ValueError("Backbone training requires joint acceleration/lane refinement")
    policy.requires_grad_(False)
    if args.lane_only:
        acceleration_parameters = []
    elif args.set_residual:
        policy.accel_adapter.requires_grad_(True)
        acceleration_parameters = list(policy.accel_adapter.parameters())
        if train_backbone or train_accel_backbone:
            policy.accel_trunk.requires_grad_(True)
            policy.accel_head.requires_grad_(True)
            acceleration_parameters += list(policy.accel_trunk.parameters()) + list(policy.accel_head.parameters())
    else:
        policy.accel_trunk.requires_grad_(True)
        policy.accel_head.requires_grad_(True)
        acceleration_parameters = list(policy.accel_trunk.parameters()) + list(policy.accel_head.parameters())
    groups = [dict(params=acceleration_parameters, lr=args.lr)] if acceleration_parameters else []
    if args.train_lanes:
        if args.lane_residual or args.set_residual:
            policy.lane_adapter.requires_grad_(True)
            lane_parameters = list(policy.lane_adapter.parameters())
            if train_backbone or train_lane_backbone:
                policy.trunk.requires_grad_(True)
                policy.lane_head.requires_grad_(True)
                lane_parameters += list(policy.trunk.parameters()) + list(policy.lane_head.parameters())
        else:
            policy.trunk.requires_grad_(True)
            policy.lane_head.requires_grad_(True)
            lane_parameters = list(policy.trunk.parameters()) + list(policy.lane_head.parameters())
        groups.append(dict(params=lane_parameters, lr=args.lane_lr))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-run", type=Path, default=Path("results/bc_hazard_v15"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", nargs="+", type=Path, default=[Path("results/bc_curriculum_stage3_v12/demonstrations.npz"), Path("results/bc_curriculum_full_v13/demonstrations.npz")])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--underbrake-weight", type=float, default=2.0)
    parser.add_argument('--brake-weight-scope',choices=('hazard','all'),default='hazard',
                        help='Apply asymmetric supervised braking error to hazard rows or all actual braking labels')
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed-start", type=int, default=1500)
    parser.add_argument("--nearby-traffic-features", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--launch-weight", type=float, default=5.0)
    parser.add_argument("--residual-penalty", type=float, default=0.0)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--kinematic-features", action="store_true")
    parser.add_argument("--censored-stop-loss", action="store_true")
    parser.add_argument("--rollout-every", type=int, default=0)
    parser.add_argument("--rollout-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--predictive-safety-coef", type=float, default=0.0)
    parser.add_argument("--train-lanes", action="store_true")
    parser.add_argument("--lane-lr", type=float, default=1e-5)
    parser.add_argument("--lane-coef", type=float, default=1.0)
    parser.add_argument("--lane-safety-coef", type=float, default=0.0)
    parser.add_argument("--spatial-safety-coef", type=float, default=0.0)
    parser.add_argument('--spatial-safety-horizon',type=float,default=1.)
    parser.add_argument("--reference-coef", type=float, default=0.0)
    parser.add_argument("--lane-residual", action="store_true")
    parser.add_argument("--lane-weight-power", type=float, default=.5)
    parser.add_argument("--predictive-reaction-time", type=float, default=.5)
    parser.add_argument("--set-residual", action="store_true")
    parser.add_argument("--lane-reference-coef", type=float)
    parser.add_argument("--predictive-comfort-decel", type=float, default=5.4)
    parser.add_argument("--risk-weighted-reference", action="store_true")
    parser.add_argument("--lane-only", action="store_true")
    parser.add_argument("--metric-sensors", action="store_true")
    parser.add_argument("--invariant-base", action="store_true",
                        help="Use learned set encoders for all nearby-car paths in the policy")
    parser.add_argument("--train-backbone", action="store_true",
                        help="Train both actor backbones/heads in addition to residual adapters")
    parser.add_argument("--train-lane-backbone", action="store_true",
                        help="Train the navigation backbone/head as well as its adapter; acceleration can stay frozen")
    parser.add_argument("--train-accel-backbone", action="store_true",
                        help="Train the complete longitudinal actor, independently of navigation")
    parser.add_argument("--uniform-sampling", action="store_true",
                        help="Sample training rows uniformly instead of hazard/start/recovery balancing")
    parser.add_argument("--lane-class-fraction", type=float, default=0.0,
                        help="Training only: mix this fraction of class-balanced lane labels into natural batches")
    parser.add_argument('--stop-hold-fraction',type=float,default=0.,
                        help='Training-only batch fraction from actual expert stopped/waiting labels')
    args = parser.parse_args()
    if args.lane_only and (not args.train_lanes or not (args.set_residual or args.lane_residual)):
        parser.error("--lane-only requires --train-lanes and a residual lane actor")
    if args.metric_sensors and not args.set_residual:
        parser.error("--metric-sensors requires --set-residual")
    if args.invariant_base and not (args.metric_sensors and args.set_residual):
        parser.error("--invariant-base requires --set-residual --metric-sensors")
    if args.train_backbone and (args.lane_only or not args.train_lanes):
        parser.error("--train-backbone requires --train-lanes and cannot be combined with --lane-only")
    if args.train_lane_backbone and (not args.lane_only or args.train_backbone):
        parser.error("--train-lane-backbone requires --lane-only and cannot be combined with --train-backbone")
    if args.train_accel_backbone and (args.lane_only or not args.set_residual):
        parser.error("--train-accel-backbone requires --set-residual without --lane-only")
    if not math.isfinite(args.lane_class_fraction) or not 0 <= args.lane_class_fraction <= 1:
        parser.error("--lane-class-fraction must be finite and within [0,1]")
    if not math.isfinite(args.stop_hold_fraction) or not 0<=args.stop_hold_fraction<=1:
        parser.error('Stop-hold fraction must be within[0,1]')
    if not math.isfinite(args.spatial_safety_horizon) or args.spatial_safety_horizon<=0:
        parser.error('Spatial safety horizon must be positive')
    if args.lane_class_fraction and (not args.train_lanes or args.lane_weight_power != 0 or args.uniform_sampling):
        parser.error("Class-balanced lane batches require --train-lanes --lane-weight-power 0 and no --uniform-sampling")
    torch.set_num_threads(1)
    torch.manual_seed(83)
    rng = np.random.default_rng(83)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.init_run / "run_config.json").read_text(encoding="utf-8"))
    original_policy_type = config["hparams"]["policy_type"]
    if original_policy_type == "invariant_metric_set_hybrid":
        args.invariant_base = True
    if original_policy_type in ("metric_set_residual_hybrid","invariant_metric_set_hybrid") and not args.metric_sensors:
        parser.error("Keep --metric-sensors when refining a metric-input actor")
    config["initialization"] = dict(run=str(args.init_run), model_sha256=hashlib.sha256((args.init_run / "model.pt").read_bytes()).hexdigest())
    for inherited in ("fine_tuning", "perception_fix", "frozen_checkpoint"):
        config.pop(inherited, None)
    config["stage"] = "Local longitudinal neural actor fitted with BC; lane tower frozen"
    config["hparams"]["policy_type"] = "residual_hybrid" if args.residual else "local_hybrid"
    if args.split:
        config["hparams"]["policy_type"] = "split_hybrid"
    if args.lane_residual:
        config["hparams"]["policy_type"] = "lane_residual_hybrid"
    if args.set_residual:
        config["hparams"]["policy_type"] = "set_residual_hybrid"
    if args.metric_sensors:
        config["hparams"]["policy_type"] = "metric_set_residual_hybrid"
    if args.invariant_base:
        config["hparams"]["policy_type"] = "invariant_metric_set_hybrid"
    config["stage"] = "BC refinement of learned acceleration and lane actors" if args.train_lanes else "BC longitudinal refinement with frozen learned lane actor"
    if args.lane_only:
        config["stage"] = "BC learned lane-adapter refinement; complete acceleration actor frozen"
    if args.train_lane_backbone:
        config["stage"] = "BC complete learned navigation actor refinement; complete acceleration actor frozen"
    if args.train_backbone:
        config["stage"] = "BC joint actor refinement including acceleration/navigation backbones and residual adapters"
    if args.train_accel_backbone:
        config["stage"] = "BC full longitudinal actor refinement, with separately configured navigation training"
    if args.nearby_traffic_features:
        config["observation"]["nearby_traffic_features"] = True
    if args.kinematic_features:
        config["observation"].update(front_safety_features=True, signal_safety_feature=True)
    config["args"] = {k: str(v) for k, v in vars(args).items()}
    config["bc_refinement"] = dict(acceleration_lr=0.0 if args.lane_only else args.lr, lane_lr=args.lane_lr,
        train_lanes=args.train_lanes, lane_only=args.lane_only, train_backbone=args.train_backbone,
        frozen_navigation_backbone=bool((args.lane_residual or args.set_residual) and not (args.train_backbone or args.train_lane_backbone)),
        train_lane_backbone=args.train_lane_backbone,train_accel_backbone=args.train_accel_backbone,
        sampler="mixed_lane_classes_and_natural_rows" if args.lane_class_fraction else (
            "uniform_training_rows" if args.uniform_sampling else "balanced_hazard_start_recovery"),
        lane_class_fraction=args.lane_class_fraction,
        brake_weight_scope=args.brake_weight_scope,underbrake_weight=args.underbrake_weight,
        stop_hold_fraction=args.stop_hold_fraction,spatial_safety_horizon=args.spatial_safety_horizon,
        class_weight_distribution="unweighted; batch proportions explicit" if args.lane_class_fraction else "uniform valid training rows",
        predictive_safety_coefficient=args.predictive_safety_coef, comfort_decel=args.predictive_comfort_decel,
        reaction_time=args.predictive_reaction_time, reference_coefficient=args.reference_coef,
        lane_reference_coefficient=args.lane_reference_coef, risk_weighted_reference=args.risk_weighted_reference)
    config["args"].update(route_prefix="None", route_variant="normal")
    config["data_sources"] = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in args.data]
    write_json(out / "run_config.json", config)
    for relative in ("train_local_bc.py", "train_warmstart_ppo.py", "sensor_history.py", "physical_layout.py", "utils/networks.py", "algorithms/ppo.py", "env/sumo_env.py"):
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative, destination)
    env = saved_environment(config)
    safety_scales=dict(vmax=env.vmax,visibility=env.visibility,signal_horizon=env.signal_horizon,
        max_accel=env.max_accel,max_decel=env.max_decel,dt=env.step_length,speed_cap=env.control_speed_cap)
    config['bc_refinement']['predictive_safety_physical_scales']=safety_scales
    write_json(out / "run_config.json", config)
    agent = PPO(env.obs_dim, 2, **{**config["hparams"], "device": "cpu"})
    state = torch.load(args.init_run / "model.pt", map_location="cpu", weights_only=True)
    if args.metric_sensors and original_policy_type not in ("metric_set_residual_hybrid","invariant_metric_set_hybrid"):
        from utils.networks import convert_metric_adapter_weights
        state = convert_metric_adapter_weights(state)
    for name in ("trunk.0.weight", "accel_trunk.0.weight"):
        if name in state:
            extra = agent.policy.state_dict()[name].shape[1] - state[name].shape[1]
            if extra < 0:
                raise ValueError("Cannot discard archived actor inputs")
            state[name] = torch.nn.functional.pad(state[name], (0, extra))
    missing, unexpected = agent.policy.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith(("accel_", "kinematic_scales", "lane_adapter.")) for key in missing):
        raise ValueError((missing, unexpected))
    if args.residual and "accel_head.weight" not in state:
        nn.init.zeros_(agent.policy.accel_head.weight)
        nn.init.zeros_(agent.policy.accel_head.bias)
    if args.split and "accel_head.weight" not in state:
        agent.policy.accel_trunk.load_state_dict(agent.policy.trunk.state_dict())
        agent.policy.accel_head.load_state_dict(agent.policy.mu_head.state_dict())
    policy = agent.policy
    reference = copy.deepcopy(policy).requires_grad_(False).eval() if args.reference_coef or args.lane_reference_coef else None
    groups = configure_trainable_parameters(policy, args)
    optimizer = torch.optim.Adam(groups)
    extra_loss_args = (args.lane_coef if args.train_lanes else 0.0, args.lane_safety_coef,
                       args.spatial_safety_coef, reference, args.reference_coef, args.predictive_reaction_time,
                       args.lane_reference_coef, args.predictive_comfort_decel, args.risk_weighted_reference,
                       safety_scales,args.spatial_safety_horizon,args.brake_weight_scope)
    data = load_demonstrations(args.data, env.obs_dim, expected_observation=config["observation"], lane_weight_power=args.lane_weight_power)
    validation = load_demonstrations(args.data, env.obs_dim, heldout=True, expected_observation=config["observation"], lane_weight_power=args.lane_weight_power)
    best, history = float("inf"), []
    rollouts, rollout_best = [], None
    if args.rollout_every:
        baseline_rows = evaluate(agent, config, args.rollout_episodes, args.eval_seed_start)
        baseline_metrics = summarize(baseline_rows)
        (out / "evaluations").mkdir(exist_ok=True)
        write_json(out / "evaluations/epoch_0.json", baseline_rows)
        rollout_best = selection_score(baseline_metrics)
        agent.save(str(out / "model_rollout_best.pt"))
        rollouts.append(dict(epoch=0, **baseline_metrics))
        write_json(out / "bc_selection.json", rollouts[-1])
    for epoch in range(args.epochs):
        policy.train()
        for _ in range(math.ceil(len(data["regular"]) / 512)):
            indices = (stop_hold_batch(data,rng,512,args.stop_hold_fraction,args.lane_class_fraction,args.uniform_sampling)
                       if args.stop_hold_fraction else lane_balanced_batch(data, rng, 512, args.lane_class_fraction)
                       if args.lane_class_fraction else rng.choice(data["regular"], size=512)
                       if args.uniform_sampling else demonstration_batch(data, rng, 512))
            loss = acceleration_loss(policy, data, indices, args.underbrake_weight, args.launch_weight, args.residual_penalty, args.censored_stop_loss, args.predictive_safety_coef, *extra_loss_args)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
            optimizer.step()
        policy.eval()
        with torch.no_grad():
            ordinary = acceleration_loss(policy, validation, validation["regular"], args.underbrake_weight, args.launch_weight, args.residual_penalty, args.censored_stop_loss, args.predictive_safety_coef, *extra_loss_args).item()
            hazard = acceleration_loss(policy, validation, validation["hazard"], args.underbrake_weight, args.launch_weight, args.residual_penalty, args.censored_stop_loss, args.predictive_safety_coef, *extra_loss_args).item()
        total = (ordinary + hazard) / 2
        history.append(dict(epoch=epoch + 1, validation_loss=total, ordinary=ordinary, hazard=hazard))
        if total < best:
            best = total
            agent.save(str(out / "model.pt"))
        if args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0:
            agent.save(str(out / f"model_epoch_{epoch + 1}.pt"))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print("BC", history[-1], flush=True)
            write_json(out / "status.json", dict(status="training", completed_epoch=epoch + 1, target_epochs=args.epochs))
        if args.rollout_every and (epoch + 1) % args.rollout_every == 0:
            rows = evaluate(agent, config, args.rollout_episodes, args.eval_seed_start)
            metrics = summarize(rows)
            write_json(out / "evaluations" / f"epoch_{epoch + 1}.json", rows)
            rollouts.append(dict(epoch=epoch + 1, **metrics))
            write_json(out / "bc_rollouts.json", rollouts)
            print("BC ROLLOUT", rollouts[-1], flush=True)
            if selection_score(metrics) > rollout_best:
                rollout_best = selection_score(metrics)
                agent.save(str(out / "model_rollout_best.pt"))
                write_json(out / "bc_selection.json", rollouts[-1])
    agent.save(str(out / "model_last.pt"))
    write_json(out / "bc_history.json", history)
    if args.rollout_every:
        shutil.copyfile(out / "model.pt", out / "model_validation_best.pt")
        shutil.copyfile(out / "model_rollout_best.pt", out / "model.pt")
    agent.load(str(out / "model.pt"))
    rows = evaluate(agent, config, args.eval_episodes, args.eval_seed_start)
    write_json(out / "evaluation.json", rows)
    write_json(out / "metrics.json", summarize(rows))
    for filename, digest in config["protected_hashes"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest
    write_json(out / "status.json", dict(status="completed", completed_epoch=args.epochs))
    print("RESULT", out, summarize(rows), flush=True)


if __name__ == "__main__":
    main()
