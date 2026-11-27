"""학습된 모델을 sumo-gui로 재생하고 주행 지표를 출력한다.

사용법:
    python test.py results/run_20260824_153012/model.pt   ← 모델 경로 지정
    python test.py                                        ← 최근 run 자동 선택
    python test.py --episodes 5 --nogui                   ← 화면 없이 지표만 측정
    python test.py results/bc_example/model.pt --algorithm bc --nogui

빨간 차가 ego. 앞차가 느리면 감속하고 앞이 비면 가속하는지 관찰해보자.
"""
import argparse
import json
import glob
import os
import sys

# 화면 없이(--nogui) 평가할 때는 libsumo 로 빠르게, 화면이 필요하면 TraCI (sumo-gui).
if "--nogui" in sys.argv and "--no-libsumo" not in sys.argv:
    os.environ["USE_LIBSUMO"] = "1"
else:
    os.environ["USE_LIBSUMO"] = "0"

import torch

from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION, OBSERVATION, ACTION
from env import road_builder
from env.sumo_env import SumoHighwayEnv

from algorithms.ppo import PPO
from algorithms.bc import BCPolicy
from utils.evaluator import evaluate_policy

# train.py와 동일한 네트워크 구조로 만들어야 가중치를 불러올 수 있으므로
# 하이퍼파라미터를 그대로 가져온다 (구조 관련 항목만 실제로 쓰임).
from train import HPARAMS, RESULTS_DIR, REWARD  # ← 보상도 train과 동일 조건

BASE = os.path.dirname(os.path.abspath(__file__))


def find_latest_model() -> str:
    """results/*/model.pt 중 가장 최근에 저장된 것을 찾는다."""
    candidates = glob.glob(os.path.join(BASE, RESULTS_DIR, "*", "model.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"'{RESULTS_DIR}/' 안에 model.pt가 없습니다. "
            f"먼저 train.py 또는 train_bc.py로 학습하세요.")
    return max(candidates, key=os.path.getmtime)


def load_agent(model_path, obs_dim, act_dim, algorithm="auto"):
    """BC 체크포인트와 기존 PPO state_dict를 구분해 평가 정책을 복원한다."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("지원하지 않는 모델 파일 형식입니다.")
    if "policy" in checkpoint and "optimizer" in checkpoint:
        checkpoint = checkpoint["policy"]    # 학습 중 중간 저장(checkpoint.pt): 정책 부분만 본다
    if checkpoint.get("algorithm") == "bc":
        detected = "bc"
    elif checkpoint.get("algorithm") == "dqn":
        detected = "dqn"
    elif "trunk.0.weight" in checkpoint and "mu_head.weight" in checkpoint:
        detected = "ppo"
    else:
        raise ValueError("현재 BC 체크포인트 또는 PPO state_dict가 아닙니다.")
    if algorithm != "auto" and algorithm != detected:
        raise ValueError(f"--algorithm {algorithm}과 모델 유형 {detected}가 다릅니다.")

    if detected == "dqn":
        from algorithms.dqn import DQN
        if checkpoint.get("obs_dim") != obs_dim:
            raise ValueError(f"DQN 모델 관측 차원 {checkpoint.get('obs_dim')} != 환경 {obs_dim}")
        agent = DQN(obs_dim=obs_dim, hidden_sizes=tuple(checkpoint["hidden_sizes"]),
                    activation=checkpoint.get("activation", "relu"), buffer_size=1, device="cpu")
        agent.q.load_state_dict(checkpoint["state_dict"])
        agent.q.eval()
    elif detected == "bc":
        if not {"state_dim", "action_dim", "state_dict"}.issubset(checkpoint):
            raise ValueError("현재 BCPolicy 형식이 아닙니다. 현재 train_bc.py로 학습한 모델을 사용하세요.")
        dims = (checkpoint["state_dim"], checkpoint["action_dim"])
        if dims != (obs_dim, act_dim):
            raise ValueError(f"BC 모델 입출력 차원 {dims} != 환경 {(obs_dim, act_dim)}")
        agent = BCPolicy.load(model_path, device="cpu")
    else:
        agent = PPO(obs_dim=obs_dim, act_dim=act_dim, **HPARAMS)
        try:
            agent.policy.load_state_dict(checkpoint)
        except RuntimeError as exc:
            raise ValueError("PPO 모델과 현재 환경 차원 또는 HPARAMS의 네트워크 구조가 다릅니다.") from exc
        agent.policy.eval()
    print(f"평가 알고리즘: {detected.upper()}")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습된 모델 재생/평가")
    parser.add_argument("model", nargs="?", default=None,
                        help="model.pt 경로 (생략 시 최근 run 자동 선택)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="재생/평가할 에피소드 수")
    parser.add_argument("--nogui", action="store_true",
                        help="화면 없이 주행 지표만 측정 (빠름)")
    parser.add_argument("--algorithm", choices=["auto", "ppo", "bc", "dqn"], default="auto",
                        help="기본은 모델 파일에서 PPO/BC 자동 판별")
    parser.add_argument("--no-libsumo", action="store_true", help="--nogui 여도 TraCI 사용")
    parser.add_argument("--stochastic", action="store_true",
                        help="탐험을 켠 채(학습 때처럼 확률적으로) 평가. 기본은 탐험 없이(가장 확률 높은 행동)")
    parser.add_argument("--ensemble", default=None,
                        help="DQN 전용: 같은 실험의 다른 모델들(쉼표로 구분)을 함께 불러 Q값을 평균해 행동을 고른다. "
                             "예) --ensemble eval_models\\model_00819200.pt,eval_models\\model_00921600.pt")
    parser.add_argument("--warmup", type=float, default=400.0,
                        help="학습차를 넣기 전에 일반차로 도로를 채우는 시간(초). 기본 400초 "
                             "(일반차는 지도 입구에서 생겨 들어오므로, 바로 출발하면 도로가 거의 비어 있다). "
                             "0 이면 예전처럼 2초만 정착")
    parser.add_argument("--warmup-jitter", type=float, default=100.0,
                        help="도로 채우는 시간을 판마다 ± 이 값(초) 안에서 무작위로 바꾼다 (기본 100)")
    parser.add_argument("--side-scan", choices=["auto", "on", "off"], default="auto",
                        help="옆 차로 근거리 감지. auto 면 모델이 학습된 실험의 run_config.json 을 따른다")
    parser.add_argument("--save-paths", default=None,
                        help="평가 판들의 주행 경로를 이 파일(.csv.gz)로 저장 (경로 그림용)")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes는 1 이상이어야 합니다.")

    model_path = args.model or find_latest_model()
    if not os.path.isabs(model_path):
        model_path = os.path.join(BASE, model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
    print(f"모델 로드: {model_path}")

    # 주의: 현재 env/ 설정과 다른 설정으로 학습된 모델이면 성능이 이상할 수 있다.
    # 그 run의 설정 스냅샷은 model.pt와 같은 폴더의 *_config.py 에서 확인 가능.
    # 관측 이력·가시거리를 바꾸기 전에 학습한 모델(관측 40차원, 가시거리 50 m)도 평가할 수 있게
    # 모델의 입력 차원을 보고 맞춘다.
    _ck = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(_ck, dict) and "policy" in _ck and "optimizer" in _ck:
        _ck = _ck["policy"]
    _in = _ck.get("obs_dim") if isinstance(_ck, dict) and "obs_dim" in _ck else (
        _ck["trunk.0.weight"].shape[1] if isinstance(_ck, dict) and "trunk.0.weight" in _ck else None)
    if _in == 40 and int(OBSERVATION.get("history", 1)) != 1:
        OBSERVATION = {**OBSERVATION, "history": 1, "visibility": 50.0}
        print("※ 예전 관측(40차원, 가시거리 50 m)으로 학습한 모델 → 그 설정으로 평가합니다")

    _step = float(SIMULATION.get("step_length", 0.5))
    SIMULATION = {**SIMULATION, "warmup_steps": max(int(SIMULATION.get("warmup_steps", 4)),
                                                    int(round(args.warmup / _step))),
                  "warmup_jitter_steps": int(round(min(args.warmup_jitter, args.warmup) / _step))}
    print(f"도로 채움: 학습차 투입 전 {SIMULATION['warmup_steps'] * _step:.0f}초 "
          f"± {SIMULATION['warmup_jitter_steps'] * _step:.0f}초 (판마다 무작위 → 출발 때 신호 위상도 판마다 다름)")

    # 옆 차로 근거리 감지: 모델이 학습된 설정 그대로 평가한다
    _ss = None
    _rc_traffic = {}
    if args.side_scan != "auto":
        _ss = args.side_scan == "on"
    else:
        _d = os.path.dirname(os.path.abspath(model_path))
        for _ in range(3):
            _rc = os.path.join(_d, "run_config.json")
            if os.path.exists(_rc):
                with open(_rc, encoding="utf-8") as f:
                    _cfg = json.load(f)
                _ss = bool(_cfg.get("side_scan", False))
                _rc_traffic = {k: _cfg[k] for k in ("s2_merge_vph", "s4_scale") if k in _cfg}
                break
            _d = os.path.dirname(_d)
    if _ss is not None:
        OBSERVATION = {**OBSERVATION, "side_scan": _ss}
    if _rc_traffic:
        TRAFFIC = {**TRAFFIC, **_rc_traffic}
    print(f"S2 합류 추가 교통 {TRAFFIC.get('s2_merge_vph', 0):.0f}대/시간, S4 회전교차로 배율 {TRAFFIC.get('s4_scale', 1)}")
    print(f"옆 차로 근거리 감지: {'켬' if OBSERVATION.get('side_scan') else '끔'}")

    # 평가를 여러 개 동시에 돌려도 서로의 도로 파일을 덮어쓰지 않게 실행마다 따로 만든다 (끝나면 지움)
    sumo_dir = os.path.join(BASE, "env", f"sumo_test_{os.getpid()}")
    sumocfg = road_builder.build(ROAD, EGO, TRAFFIC, sumo_dir)
    env = SumoHighwayEnv(
        cfg_path=sumocfg,
        road=ROAD, ego=EGO,
        mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
        action=ACTION, reward=REWARD,
        traffic=TRAFFIC,
        gui=not args.nogui,
        gui_autostart=False,   # ▶(플레이) 버튼을 눌러야 에피소드 시작
    )

    if not args.nogui:
        print("※ 창이 뜨면 ▶(플레이) 버튼을 눌러야 에피소드가 시작됩니다.")
        print("  에피소드가 끝나면 다음 에피소드 창이 다시 뜨고, 또 ▶를 누르면 됩니다.")

    _warm_counts = []
    _plain_reset = env.reset

    def _reset_count(*a, **k):
        out = _plain_reset(*a, **k)
        _warm_counts.append(int(getattr(env, "warmup_vehicle_count", 0)))
        return out
    env.reset = _reset_count

    # 진행률: 출발 때 목적지까지 남은 경로 거리 대비, 끝날 때까지 줄인 거리 (도착이면 100%)
    # 끝난 이유 세부: 시간 초과로 묶이는 것(1,000초 / 오래 정지 / 경로 이탈 한도 / 돌아갈 길 없음)을 나눠 센다
    _prog = {"r0": None, "last": None}
    _ep_prog, _ep_km, _ends = [], [], []
    _plain_reset2, _plain_step2 = env.reset, env.step

    def _reset_prog(*a, **k):
        out = _plain_reset2(*a, **k)
        _prog["r0"] = _prog["last"] = env._remaining_route_m()
        return out

    def _step_prog(action):
        out = _plain_step2(action)
        _, _, term, trunc, info = out
        from env.sumo_env import traci as _tr
        if env.ego_id in _tr.vehicle.getIDList():
            rem = env._remaining_route_m()
            if rem is not None:
                _prog["last"] = rem
        if term or trunc:
            r0, last = _prog["r0"], _prog["last"]
            if info.get("arrived"):
                frac, km = 1.0, (r0 or 0.0) / 1000.0
            elif r0:
                done_m = max(0.0, r0 - (last if last is not None else r0))
                frac, km = min(done_m / r0, 1.0), done_m / 1000.0
            else:
                frac, km = 0.0, 0.0
            _ep_prog.append(frac)
            _ep_km.append(km)
            _ends.append("collision" if info.get("collided") else "arrival" if info.get("arrived")
                         else "red_light" if info.get("red_light_violation") else "stuck" if info.get("stuck_timeout")
                         else "no_way_back" if info.get("no_way_back") else "reroute_limit" if info.get("reroute_limit")
                         else "max_steps")
        return out
    env.reset, env.step = _reset_prog, _step_prog

    try:
        agent = load_agent(model_path, env.observation_space.shape[0],
                           env.action_space.shape[0], args.algorithm)
        if args.ensemble:
            import types
            import numpy as _np
            _qs = [agent.q]
            for _p in [x.strip() for x in args.ensemble.split(",") if x.strip()]:
                if not os.path.isabs(_p):
                    _p = os.path.join(BASE, _p)
                _a = load_agent(_p, env.observation_space.shape[0], env.action_space.shape[0], "dqn")
                _qs.append(_a.q)
                print(f"  함께 쓰는 모델: {_p}")

            @torch.no_grad()
            def _q_mean(self, obs):
                o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                return _np.mean([q(o).squeeze(0).numpy() for q in _qs], axis=0)
            agent.q_values = types.MethodType(_q_mean, agent)
            print(f"앙상블: 모델 {len(_qs)}개의 Q값 평균으로 행동 선택")
        rows = []
        if args.save_paths:
            from env.sumo_env import traci
            cnt = {"ep": -1, "t": 0}
            orig_reset, orig_step = env.reset, env.step

            def _reset(*a, **k):
                cnt["ep"] += 1
                cnt["t"] = 0
                return orig_reset(*a, **k)

            def _step(action):
                out = orig_step(action)
                _, r, term, trunc, info = out
                x = y = v = None
                road, lane = "", ""
                try:
                    if env.ego_id in traci.vehicle.getIDList():
                        x, y = traci.vehicle.getPosition(env.ego_id)
                        v = traci.vehicle.getSpeed(env.ego_id)
                        road = traci.vehicle.getRoadID(env.ego_id)
                        lane = traci.vehicle.getLaneIndex(env.ego_id)
                except Exception:  # noqa: BLE001
                    pass
                end = ""
                if term or trunc:
                    end = ("collision" if info.get("collided") else "arrival" if info.get("arrived")
                           else "red_light" if info.get("red_light_violation") else "stuck" if info.get("stuck_timeout")
                           else "no_way_back" if info.get("no_way_back") else "reroute_limit" if info.get("reroute_limit")
                           else "max_steps")
                rows.append([cnt["ep"], cnt["t"], "" if x is None else round(x, 1), "" if y is None else round(y, 1),
                             "" if v is None else round(v, 2), road, lane, round(float(r), 4),
                             int(bool(info.get("rerouted"))), end])
                cnt["t"] += 1
                return out
            env.reset, env.step = _reset, _step
        print(f"평가 방식: {'탐험 켬 (확률적)' if args.stochastic else '탐험 없음 (결정적)'}")
        metrics = evaluate_policy(agent, env, n_episodes=args.episodes,
                                  deterministic=not args.stochastic)
    finally:
        env.close()
        import shutil
        shutil.rmtree(sumo_dir, ignore_errors=True)
        if args.save_paths and rows:
            import csv
            import gzip
            os.makedirs(os.path.dirname(os.path.abspath(args.save_paths)), exist_ok=True)
            with gzip.open(args.save_paths, "wt", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["episode", "t", "x", "y", "speed", "road_id", "lane", "reward", "rerouted", "end_reason"])
                w.writerows(rows)
            print(f"주행 경로 저장: {args.save_paths}")

    print(f"\n─── 주행 지표 ({args.episodes} 에피소드 평균) ───")
    print(f"  충돌률       : {metrics.collision_rate:.0%}")
    print(f"  완주율       : {metrics.success_rate:.0%}")
    if _ep_prog:
        print(f"  진행률       : {sum(_ep_prog) / len(_ep_prog):.0%}  (평균 {sum(_ep_km) / len(_ep_km):.2f} km 주행)")
    print(f"  시간초과율   : {metrics.timeout_rate:.0%}")
    if _ends:
        _n = len(_ends)
        _c = {k: _ends.count(k) / _n for k in ("max_steps", "stuck", "reroute_limit", "no_way_back")}
        print(f"    └ 1,000초 초과 {_c['max_steps']:.0%} / 오래 서 있어서 종료 {_c['stuck']:.0%} / "
              f"경로 이탈 한도 {_c['reroute_limit']:.0%} / 돌아갈 길 없음 {_c['no_way_back']:.0%}")
    print(f"  신호위반율   : {metrics.red_light_rate:.0%}")
    print(f"  평균 속도    : {metrics.mean_speed:.2f} m/s "
          f"({metrics.mean_speed * 3.6:.1f} km/h)")
    print(f"  평균 차간거리: {metrics.mean_gap:.2f} m")
    print(f"  최소 차간거리: {metrics.min_gap:.2f} m")
    print(f"  차선변경     : {metrics.lane_changes:.1f} 회/에피소드")
    print(f"  평균 보상    : {metrics.ep_return:.2f}")
    print(f"  평균 길이    : {metrics.ep_length:.0f} 스텝")
    if _warm_counts:
        print(f"  출발 시 도로 위 일반차: 평균 {sum(_warm_counts) / len(_warm_counts):.0f}대 "
              f"(최소 {min(_warm_counts)}, 최대 {max(_warm_counts)})")
