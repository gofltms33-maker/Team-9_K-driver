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
from env.sumo_env import SumoHighwayEnv, BACKEND

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
    elif "trunk.0.weight" in checkpoint and "mu_head.weight" in checkpoint:
        detected = "ppo"
    else:
        raise ValueError("현재 BC 체크포인트 또는 PPO state_dict가 아닙니다.")
    if algorithm != "auto" and algorithm != detected:
        raise ValueError(f"--algorithm {algorithm}과 모델 유형 {detected}가 다릅니다.")

    if detected == "bc":
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
    parser.add_argument("--hold", action="store_true",
                        help="영상 녹화용: 마지막 판이 끝나도 SUMO 창을 닫지 않고 엔터를 누를 때까지 기다림")
    parser.add_argument("--nogui", action="store_true",
                        help="화면 없이 주행 지표만 측정 (빠름)")
    parser.add_argument("--algorithm", choices=["auto", "ppo", "bc"], default="auto",
                        help="기본은 모델 파일에서 PPO/BC 자동 판별")
    parser.add_argument("--no-libsumo", action="store_true", help="--nogui 여도 TraCI 사용")
    parser.add_argument("--pij", choices=["auto", "on", "off", "sig", "dist", "strict"], default="auto",
                        help="교차로 안 비보호 신호 유지 관측. auto = 학습 때 설정, on/off = 강제, "
                             "sig/dist = 신호값만/거리만 (원인 분리 실험용), strict = 교차로 id·진입 링크 검증 후 적용")
    parser.add_argument("--obs", choices=["auto", "new", "old"], default="auto",
                        help="교차로 관측 버전. auto = 학습 폴더의 mdp_config 값, new = 현재 mdp_config, old = 추가 전")
    parser.add_argument("--stochastic", action="store_true",
                        help="탐험을 켠 채(학습 때처럼 확률적으로) 평가. 기본은 탐험 없이(가장 확률 높은 행동)")
    parser.add_argument("--save-paths", default=None,
                        help="평가 판들의 주행 경로를 이 파일(.csv.gz)로 저장 (경로 그림용)")
    parser.add_argument("--seed-start", type=int, default=None,
                        help="판마다 시드를 이 값부터 1씩 늘려 고정 (예: 1000). 같은 값을 쓰면 여러 모델을 "
                             "똑같은 교통 상황으로 비교할 수 있다. 생략하면 매번 무작위")
    parser.add_argument("--cfg", choices=["current", "train"], default="current",
                        help="REWARD·ACTION·SIMULATION 설정. current = 지금 env/mdp_config (예전 평가와 같은 조건, 기본), "
                             "train = 학습 폴더의 run_config.json 보상 + mdp_config.py 의 ACTION·SIMULATION "
                             "(학습 뒤에 생긴 lane_change_carry_s 같은 기능은 꺼짐). 관측은 둘 다 --obs/--pij 규칙을 따름")
    parser.add_argument("--start-left", nargs=3, type=float, default=None, metavar=("대수/h", "시작초", "끝초"),
                        help="교통량 늘린 버전: 출발 구간 염곡 비보호 좌회전 줄에 대기 차를 만들 교통류 (예: 400 270 530). --heavy-traffic 과 함께")
    parser.add_argument("--ramp-split", action="store_true",
                        help="교통량 늘린 버전: 서초IC 램프 대기줄 차를 양재 교차로에서 우회전·직진·좌회전으로 나눔 (기본: 전부 우회전, 아직 조정 중)")
    parser.add_argument("--hotspot-window", nargs=4, type=float, default=None, metavar=("램프시작", "램프끝", "공사시작", "공사끝"),
                        help="교통량 늘린 버전: --hotspot-traffic 차를 넣을 시뮬레이션 시각(초) 구간. 시드마다 내 차 도착 시각에 맞춘다")
    parser.add_argument("--hotspot-traffic", nargs=2, type=float, default=None, metavar=("서초IC램프", "공사"),
                        help="교통량 늘린 버전: --heavy-traffic 과 함께, 서초IC 출구 램프·언주로 공사 구간에 대기줄을 만들 차를 시간당 이만큼 추가 (예: 900 900)")
    parser.add_argument("--heavy-traffic", nargs=2, type=float, default=None, metavar=("나머지", "고속도로"),
                        help="교통량 늘린 버전: 출발 구간(정문~첫 우회전~염곡 비보호 좌회전)은 원래 교통량, 나머지 도로와 고속도로만 배수로 (예: 2 3)")
    parser.add_argument("--vph", type=float, default=None, metavar="대수",
                        help="배경 교통량 vehs_per_hour (기본 800). 교통량 늘린 버전 예: 1600 = 2배. 학습·공식 평가는 기본값")
    parser.add_argument("--yellow-obs", choices=["auto", "on", "off"], default="auto",
                        help="노란불을 멈출(0)/지나갈(1) 로 나눠 관측 (Y). 관측 크기 그대로라 예전 모델에도 켤 수 있음")
    parser.add_argument("--queue-mask", choices=["auto", "on", "off"], default="auto",
                        help="Q: 정지선 앞에 줄 선 차가 있으면 교차로 상대 차 관측을 없음으로 (기본 auto = 학습 때 설정)")
    parser.add_argument("--stalled-block", choices=["auto", "on", "off"], default="auto",
                        help="S: 교차로 안에 멈춰 내 진로를 막는 차가 있으면 우선 신호여도 양보 필요 1 (기본 auto = 학습 때 설정)")
    parser.add_argument("--yield-cut", choices=["auto", "on", "off"], default="auto",
                        help="상대 도착이 8초 이상이면 [39] 양보 필요를 0 으로 (B, 관측 수정). 관측 크기 그대로라 예전 모델에도 켤 수 있음")
    parser.add_argument("--foe-scan-internal", choices=["auto", "on", "off"], default="auto",
                        help="상대 차 탐색에 교차로 내부 차로 포함(A, 관측 버그 수정). auto = 학습 때 설정. "
                             "관측 크기가 그대로라 예전 모델에도 켤 수 있음")
    parser.add_argument("--warmup", default=None, metavar="초",
                        help="출발 전에 배경 교통을 먼저 흘려 도로를 채우는 시간. '600' = 10분 고정, "
                             "'300-600' = 판마다 무작위. 기본은 mdp_config 의 warmup_s (300~600). '0' = 예전처럼 빈 도로")
    args = parser.parse_args()
    if args.warmup is not None:
        from train import parse_warmup
        SIMULATION = {**SIMULATION, "warmup_s": parse_warmup(args.warmup)}
    _w = SIMULATION.get("warmup_s", 0) or 0
    _w = _w if isinstance(_w, (list, tuple)) else (_w, _w)
    print(f"교통 채우기: 출발 전 {float(_w[0]):.0f}~{float(_w[1]):.0f}초" + (" (빈 도로에서 출발)" if float(_w[1]) <= 0 else ""))
    if args.episodes < 1:
        parser.error("--episodes는 1 이상이어야 합니다.")

    model_path = args.model or find_latest_model()
    if not os.path.isabs(model_path):
        model_path = os.path.join(BASE, model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
    print(f"모델 로드: {model_path}")
    print(f"SUMO 연결 방식: {BACKEND}" + ("" if BACKEND == "libsumo" or not args.nogui else "  ← libsumo 를 못 불러옴. cmd 에서 set PYTHONPATH= 를 먼저 실행하세요 (느려짐)"))

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

    # 교차로 관측(비보호 신호값·상대 차 찾는 방법)은 모델이 학습한 그대로 맞춰야 한다 (2026-11-24 부터 여러 번 바뀜).
    # auto: 학습 폴더에 복사된 mdp_config.py 의 값을 쓴다. 항목이 없으면 예전 관측(추가 전).
    _JKEYS = ("permissive_signal", "junction_foe_mode", "foe_horizon_m", "foe_horizon_s", "neighbor_scan",
              "leader_signals", "permissive_in_junction", "exit_lane_aware", "nav_horizon_s_fast",
              "internal_chain_via", "link_cache_clear", "foe_scan_internal", "gap_margin_obs", "yield_horizon_cut", "yellow_decision_obs", "yellow_decision_decel",
              "stalled_foe_block", "stalled_foe_speed", "stalled_foe_dist", "queue_yield_mask", "far_lead_obs", "far_lead_range",
              "gap_margin_scale", "gap_margin_accel")
    _OLD = {"permissive_signal": None, "junction_foe_mode": "sumo", "neighbor_scan": False,
            "leader_signals": False, "permissive_in_junction": False, "exit_lane_aware": False,
            "nav_horizon_s_fast": 0.0, "internal_chain_via": False, "link_cache_clear": False,
            "foe_scan_internal": False, "gap_margin_obs": False, "yield_horizon_cut": False, "yellow_decision_obs": False,
            "stalled_foe_block": False, "queue_yield_mask": False, "far_lead_obs": False}
    _mode = args.obs
    if _mode == "auto":
        _run = os.path.dirname(os.path.abspath(model_path))
        if os.path.basename(_run) == "eval_models":
            _run = os.path.dirname(_run)
        _cfg = os.path.join(_run, "mdp_config.py")
        if os.path.exists(_cfg):
            _ns = {}
            try:
                with open(_cfg, encoding="utf-8") as _f:
                    exec(compile(_f.read(), _cfg, "exec"), _ns)   # 학습 때 저장한 설정 파일 (우리가 만든 것)
                _ro = _ns.get("OBSERVATION", {})
                _jo = {k: _ro[k] for k in _JKEYS if k in _ro}
                OBSERVATION = {**OBSERVATION, **_OLD, **_jo}
                print(f"교차로 관측: 학습 때 설정 사용 → 신호 'g'={OBSERVATION.get('permissive_signal')}, "
                      f"상대 차={OBSERVATION.get('junction_foe_mode')}, 옆 차로 직접 스캔={OBSERVATION.get('neighbor_scan', False)}, "
                      f"앞차 깜빡이={OBSERVATION.get('leader_signals', False)}, "
                      f"교차로 안 비보호 유지={OBSERVATION.get('permissive_in_junction', False)}")
            except Exception as _e:  # noqa: BLE001
                print(f"※ 학습 폴더 mdp_config.py 를 못 읽음 ({_e}) → 현재 설정으로 평가")
    elif _mode == "old":
        OBSERVATION = {**OBSERVATION, **_OLD}
        print("※ 비보호 신호·교차로 상대 차 관측 추가 전 관측으로 평가합니다")
    if args.pij != "auto":
        OBSERVATION = {**OBSERVATION, "permissive_in_junction": {"on": True, "off": False}.get(args.pij, args.pij)}
        print(f"※ 교차로 안 비보호 신호 유지 관측을 강제로 {args.pij}")
    if args.vph is not None:
        TRAFFIC = {**TRAFFIC, "vehs_per_hour": float(args.vph)}
        print(f"※ 배경 교통량 vehs_per_hour = {args.vph:.0f} (교통량 늘린 버전, 학습·공식 평가는 800)")
    if args.yellow_obs != "auto":
        OBSERVATION = {**OBSERVATION, "yellow_decision_obs": args.yellow_obs == "on"}
        print(f"※ 노란불 멈춤/지나감 관측(Y) 을 강제로 {args.yellow_obs}")
    if args.queue_mask != "auto":
        OBSERVATION = {**OBSERVATION, "queue_yield_mask": args.queue_mask == "on"}
        print(f"※ 정지선 앞 줄이 있으면 상대 차 관측 없음 (Q) 을 강제로 {args.queue_mask}")
    if args.stalled_block != "auto":
        OBSERVATION = {**OBSERVATION, "stalled_foe_block": args.stalled_block == "on"}
        print(f"※ 교차로 안 멈춘 차가 내 진로를 막으면 양보 1 (S) 을 강제로 {args.stalled_block}")
    if args.yield_cut != "auto":
        OBSERVATION = {**OBSERVATION, "yield_horizon_cut": args.yield_cut == "on"}
        print(f"※ 먼 상대(8초 이상)는 양보 필요 0 (B) 을 강제로 {args.yield_cut}")
    if args.foe_scan_internal != "auto":
        OBSERVATION = {**OBSERVATION, "foe_scan_internal": args.foe_scan_internal == "on"}
        print(f"※ 상대 차 탐색에 교차로 내부 차로 포함(A)을 강제로 {args.foe_scan_internal}")
    # 모델 입력 크기로 앞차 깜빡이 관측(40 ↔ 42) 을 최종 확인 (설정 사본이 없거나 달라도 모델에 맞춘다)
    if _in in (40, 42) and int(OBSERVATION.get("history", 1)) == 1:
        _want = (_in == 42)
        if bool(OBSERVATION.get("leader_signals", False)) != _want:
            OBSERVATION = {**OBSERVATION, "leader_signals": _want}
            print(f"※ 모델 입력 {_in}개 → 앞차 깜빡이 관측 {'켬' if _want else '끔'}")

    # --cfg train: 학습 당시 보상(run_config.json, CLI 감점 포함)과 ACTION·SIMULATION(학습 폴더 mdp_config.py) 복원.
    # 보상은 행동에 영향을 주지 않지만(보상 숫자만 달라짐), ACTION 의 lane_change_carry_s 처럼 학습 뒤에 생긴 기능은 주행을 바꾼다.
    if args.cfg == "train":
        _run = os.path.dirname(os.path.abspath(model_path))
        if os.path.basename(_run) == "eval_models":
            _run = os.path.dirname(_run)
        _rc = os.path.join(_run, "run_config.json")
        _mc = os.path.join(_run, "mdp_config.py")
        if not (os.path.exists(_rc) and os.path.exists(_mc)):
            raise SystemExit(f"--cfg train: 학습 폴더에 run_config.json / mdp_config.py 가 없습니다: {_run}")
        with open(_rc, encoding="utf-8") as _f:
            REWARD = dict(json.load(_f)["reward"])
        _ns = {}
        with open(_mc, encoding="utf-8") as _f:
            exec(compile(_f.read(), _mc, "exec"), _ns)
        _w0 = SIMULATION.get("warmup_s")
        ACTION = dict(_ns["ACTION"])
        SIMULATION = dict(_ns["SIMULATION"])
        if args.warmup is not None:
            SIMULATION["warmup_s"] = _w0
        print(f"설정: 학습 당시 (보상 = run_config.json, ACTION·SIMULATION = 학습 폴더 mdp_config.py, "
              f"차로 변경 이어받기 {float(ACTION.get('lane_change_carry_s', 0.0) or 0.0):.1f}초)")
    else:
        print(f"설정: 현재 env/mdp_config (차로 변경 이어받기 {float(ACTION.get('lane_change_carry_s', 0.0) or 0.0):.1f}초, "
              f"보상은 학습 당시와 다를 수 있음)")

    # 평가를 여러 개 동시에 돌려도 서로의 도로 파일을 덮어쓰지 않게 실행마다 따로 만든다 (끝나면 지움)
    sumo_dir = os.path.join(BASE, "env", f"sumo_test_{os.getpid()}")
    sumocfg = road_builder.build(ROAD, EGO, TRAFFIC, sumo_dir)
    if args.heavy_traffic is not None:
        from utils.heavy_traffic import scale_rou
        hs = args.hotspot_traffic or (None, None)
        scale_rou(sumo_dir, others=float(args.heavy_traffic[0]), highway=float(args.heavy_traffic[1]),
                  ramp_vph=hs[0], cz_vph=hs[1],
                  ramp_win=args.hotspot_window[0:2] if args.hotspot_window else None,
                  cz_win=args.hotspot_window[2:4] if args.hotspot_window else None,
                  ramp_split=args.ramp_split,
                  left_vph=args.start_left[0] if args.start_left else None,
                  left_win=args.start_left[1:3] if args.start_left else None)
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

    try:
        agent = load_agent(model_path, env.observation_space.shape[0],
                           env.action_space.shape[0], args.algorithm)
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
        # 시드를 안 주면 실행마다 무작위로 정한다. 예전에는 PPO 생성자가 np.random.seed(0) 을 불러서
        # '무작위' 평가가 사실 매번 같은 100판이었다 (2026-11-24 발견). 정한 시드는 출력해서 재현할 수 있게 한다.
        if args.seed_start is None:
            args.seed_start = int.from_bytes(os.urandom(4), "little") % 1_000_000
            print(f"시드 (무작위로 정함): {args.seed_start} ~ {args.seed_start + args.episodes - 1}"
                  f"  ← 같은 판을 다시 보려면 --seed-start {args.seed_start}")
        else:
            print(f"시드 고정: {args.seed_start} ~ {args.seed_start + args.episodes - 1}")
        metrics = evaluate_policy(agent, env, n_episodes=args.episodes,
                                  deterministic=not args.stochastic, seed_start=args.seed_start,
                                  verbose=True)
    finally:
        if args.hold and not args.nogui:
            try:
                input("\n[--hold] 주행이 끝났습니다. SUMO 창은 열려 있습니다 (녹화를 먼저 멈추고 저장하세요). "
                      "엔터를 누르면 창을 닫습니다...")
            except (EOFError, KeyboardInterrupt):
                pass
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
    print(f"  진행률       : {metrics.progress:.0%}  (평균 {metrics.km_driven:.2f} km 주행)")
    print(f"  시간초과율   : {metrics.timeout_rate:.0%}  (그중 오래 서 있어서 종료 {metrics.stuck_rate:.0%})")
    print(f"  신호위반율   : {metrics.red_light_rate:.0%}")
    print(f"  평균 속도    : {metrics.mean_speed:.2f} m/s "
          f"({metrics.mean_speed * 3.6:.1f} km/h)")
    print(f"  평균 차간거리: {metrics.mean_gap:.2f} m")
    print(f"  최소 차간거리: {metrics.min_gap:.2f} m")
    print(f"  차선변경     : {metrics.lane_changes:.1f} 회/에피소드")
    print(f"  평균 보상    : {metrics.ep_return:.2f}")
    print(f"  평균 길이    : {metrics.ep_length:.0f} 스텝")
