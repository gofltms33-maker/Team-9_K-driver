"""SUMO 고속도로 RL 환경 — Gymnasium 인터페이스 + TraCI.

┌─────────────────────────────────────────────────────────────┐
│  강화학습 루프에서 이 클래스의 역할                              │
│                                                             │
│    obs, _ = env.reset()          # 에피소드 시작, 초기 관측     │
│    obs, r, term, trunc, _ = env.step(action)  # 1스텝 진행    │
│                                                             │
│  내부적으로는 TraCI(Traffic Control Interface)라는 소켓 API로  │
│  SUMO 시뮬레이터를 원격 조종한다:                               │
│    traci.simulationStep()  → 시뮬레이션을 1스텝 진행            │
│    traci.vehicle.setSpeed() → 차량 속도 명령                   │
│    traci.vehicle.getSpeed() 등 → 상태 조회                     │
└─────────────────────────────────────────────────────────────┘

관측/행동/보상의 구체적인 정의는 configs/mdp_config.py 에 있고,
이 파일은 그 설정을 받아서 실행만 한다. (MDP를 바꿀 때 이 파일 수정 불필요)
"""
import os
import re
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# SUMO 설치 폴더의 tools/ 안에 traci, sumolib 파이썬 모듈이 들어있다.
# pip로도 설치했지만, 버전 불일치를 피하려면 SUMO_HOME 쪽을 우선 경로에 추가.
#
# SUMO_HOME이 설정 안 돼 있으면 OS별 기본 설치 경로를 순서대로 탐색해서
# 자동으로 잡아준다 (환경변수 설정을 깜빡해도 동작하도록).
if "SUMO_HOME" not in os.environ:
    _candidates = [
        # macOS (Homebrew) — Apple Silicon / Intel
        "/opt/homebrew/opt/sumo/share/sumo",
        "/usr/local/opt/sumo/share/sumo",
        # Linux (apt)
        "/usr/share/sumo",
        # Windows (기본 인스톨러 경로)
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
    ]
    for _p in _candidates:
        if os.path.isdir(_p):
            os.environ["SUMO_HOME"] = _p
            print(f"[sumo_env] SUMO_HOME 자동 감지: {_p}")
            break
    else:
        raise RuntimeError(
            "SUMO_HOME 환경변수가 없고 기본 경로에서도 SUMO를 찾지 못했습니다.\n"
            "  Windows: 인스톨러 재설치(SUMO_HOME 체크) 또는 환경변수 수동 설정\n"
            "  macOS  : ./setup_mac.sh 실행 또는 README_MAC.md 참고\n"
            "  Linux  : sudo apt install sumo sumo-tools 후 "
            "export SUMO_HOME=/usr/share/sumo")

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

# 화면 없이 학습·평가할 때는 libsumo(SUMO 를 파이썬 안에서 직접 실행)를 쓰면 TraCI 소켓 통신이
# 없어져 훨씬 빠르다. train.py / test.py --nogui 가 USE_LIBSUMO=1 을 켠다 (pip install libsumo).
# 설치 안 돼 있거나 화면(sumo-gui)이 필요하면 기존 TraCI 를 쓴다. 두 방식의 함수 이름은 같다.
if os.environ.get("USE_LIBSUMO") == "1":
    try:
        import libsumo as traci
        BACKEND = "libsumo"
    except ImportError:
        import traci
        BACKEND = "traci"
else:
    import traci
    BACKEND = "traci"
from sumolib import checkBinary

# SUMO 시작 실패 때 다시 시도할 예외 (libsumo 에는 traci.exceptions 모듈이 없어서 따로 모은다)
_START_ERRORS = tuple(e for e in (
    getattr(getattr(traci, "exceptions", None), "FatalTraCIError", None),
    getattr(traci, "TraCIException", None),
    getattr(traci, "FatalTraCIError", None),
) if isinstance(e, type)) + (OSError, RuntimeError)


class SumoHighwayEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg_path, road, ego, mdp_sim, mdp_obs, action, reward,
                 traffic=None, gui=False, gui_autostart=True):
        """
        Args:
            cfg_path: road_builder.build()가 만든 .sumocfg 경로
            road:     configs.road_config.ROAD
            ego:      configs.road_config.EGO (가속/감속 한계가 여기서 옴)
            mdp_sim:  configs.mdp_config.SIMULATION
            mdp_obs:  configs.mdp_config.OBSERVATION
            action:   configs.mdp_config.ACTION (행동 차원 수)
            reward:   configs.mdp_config.REWARD
            traffic:  configs.road_config.TRAFFIC (프리필/컨트롤러 정보).
                None이면 프리필 없이 flow만으로 동작 (하위호환)
            gui:      True면 sumo-gui(화면 있는 버전)로 실행
            gui_autostart: (gui=True일 때만 의미)
                True  → 창이 뜨면 시뮬레이션이 자동으로 시작
                False → 사용자가 ▶(플레이) 버튼을 누를 때까지 대기.
                        sumo-gui는 --start 옵션이 없으면 TraCI의
                        simulationStep() 호출도 플레이 전까지 블록되므로,
                        에피소드 전체가 "▶를 눌러야 시작"되는 효과가 난다.
        """
        super().__init__()
        self.cfg_path = cfg_path
        self.gui = gui
        self.gui_autostart = gui_autostart

        # ----- 도로/차량 설정에서 필요한 값들 -----
        self.vmax = float(road["speed_limit"])   # 속도 정규화 기준
        self.num_lanes = int(road["num_lanes"])
        # 시작 차선: "center"면 중앙 차선(차선수//2)으로 자동 결정.
        #   3차선 → 1번(가운데), 5차선 → 2번. 차선 수를 바꿔도 항상 가운데라
        #   좌/우 양쪽으로 변경 여지가 대칭으로 생긴다 (탐험에 유리).
        _dl = ego["depart_lane"]
        if str(_dl).lower() == "center":
            self.ego_depart_lane = str(int(road["num_lanes"]) // 2)
        else:
            self.ego_depart_lane = str(_dl)
        # ---- 프리필(도로 전체 사전 배치) 설정 ----
        self.traffic = traffic or {}
        pf = (self.traffic.get("prefill") or {})
        self.prefill_enabled = bool(pf.get("enabled", False))
        self.prefill_start = float(pf.get("start_offset", 80.0))
        self.prefill_end_margin = float(pf.get("end_margin", 50.0))
        self.prefill_jitter = float(pf.get("spacing_jitter", 0.3))
        # ---- 차선변경 안전 게이트 설정 (mdp_config.ACTION["lane_change_safety"]) ----
        lcs = (action.get("lane_change_safety") or {})
        self.lc_safety_enabled = bool(lcs.get("enabled", True))
        self.lc_front_min = float(lcs.get("min_front_gap", 5.0))   # m
        self.lc_rear_min = float(lcs.get("min_rear_gap", 5.0))     # m
        self.lc_front_tau = float(lcs.get("front_tau", 0.3))       # s
        self.lc_rear_tau = float(lcs.get("rear_tau", 0.5))         # s
        self.lc_spawn_protect = float(lcs.get("spawn_protect", 60.0))  # m
        self.lc_horizon = float(lcs.get("approach_horizon", 1.0))       # s
        self.road_length = float(road["length"])
        self.traffic_max_speed = float(self.traffic.get("max_speed", self.vmax))
        # 차선당 배치 간격: 지정값이 없으면 교통류 이론대로 자동 계산
        #   차선당 유량 q_lane = (vehs/h) / 3600 / 차선수  [대/초]
        #   간격 = 속도 / q_lane   (그 속도로 흐를 때 차 사이 평균 거리)
        if pf.get("spacing"):
            self.prefill_spacing = float(pf["spacing"])
        else:
            vph = float(self.traffic.get("vehs_per_hour", 1800))
            q_lane = max(vph / 3600.0 / max(self.num_lanes, 1), 1e-6)
            self.prefill_spacing = self.traffic_max_speed / q_lane
        # 컨트롤러 타입 샘플링 준비 (rou.xml의 car_<이름> vType과 대응)
        ctrls = self.traffic.get("controllers") or {}
        if ctrls:
            names = list(ctrls.keys())
            probs = np.array([float(ctrls[n].get("probability", 1.0)) for n in names])
            self._prefill_types = [f"car_{n}" for n in names]
            self._prefill_probs = probs / probs.sum()
        else:
            self._prefill_types, self._prefill_probs = ["car"], np.array([1.0])
        self.ego_depart_speed = str(ego["depart_speed"])
        # 행동 a ∈ [-1,1] 을 실제 가속도로 변환할 때 쓰는 스케일:
        #   a = +1 → +max_accel (풀 액셀),  a = -1 → -max_decel (풀 브레이크)
        self.max_accel = float(ego["accel"])   # m/s^2
        self.max_decel = float(ego["decel"])   # m/s^2 (양수로 저장, 부호는 계산 시)

        # ----- MDP 설정 -----
        self.step_length = float(mdp_sim["step_length"])
        self.max_steps = int(mdp_sim["max_steps"])
        self.stuck_timeout = float(mdp_sim.get("stuck_timeout_s", 180.0))
        self._stopped_steps = 0
        self.reroute_count = 0          # 이번 에피소드의 경로 재탐색 횟수
        self._last_limit = None         # 직전 도로 제한속도 (교차로 안에서 쓰기)
        self._last_reroute_edge = None  # 같은 갈림길 앞 반복 재탐색은 한 번으로 센다
        # 학습 때만 쓰는 '코스 중간 출발' 확률. 기본 0 (test.py·데이터 수집·평가는 항상 정문 출발).
        # train.py 가 mdp_config 의 train_random_start_prob 로 켠다.
        self.random_start_prob = 0.0
        self.queue_start_prob = 0.0  # explicitly enabled by training only
        self.queue_episode_steps = int(float(mdp_sim.get("train_queue_seconds", 120)) / self.step_length)
        self.queue_stopped_prob = float(mdp_sim.get("train_queue_stopped_prob", 0.0))
        self._queue_episode = False
        self._queue_start = None
        self._queue_trials = 0
        self._queue_successes = 0
        self._start_cands = None
        self.start_edge = None
        self.warmup_steps = int(mdp_sim["warmup_steps"])
        # 교통 채우기: ego 를 넣기 전에 배경 교통만 이 시간(초)만큼 먼저 돌린다. (최소, 최대) 사이에서 판마다 무작위.
        # 0 이면 예전처럼 거의 빈 도로(시뮬레이션 2초)에서 출발. 도로가 정상 교통량으로 차는 데 약 5~10분 걸린다.
        # 판마다 출발 시각이 달라지므로 신호 타이밍도 판마다 달라진다.
        _w = mdp_sim.get("warmup_s", 0) or 0
        self.warmup_range = (float(_w[0]), float(_w[1])) if isinstance(_w, (list, tuple)) else (float(_w), float(_w))
        self.visibility = float(mdp_obs["visibility"])   # 관측 거리 W
        # 내비게이션·공사 안내 / 신호등을 보는 거리 (env/mdp_config.py)
        self.nav_horizon = float(mdp_obs.get("nav_horizon", 300.0))
        self.nav_horizon_s = float(mdp_obs.get("nav_horizon_s", 0.0))
        # 빠른 도로(제한속도 >= nav_fast_speed)에서는 안내를 nav_horizon_s_fast 초 앞부터 (0 = 끔).
        # 서초IC 출구(3차로 이동)를 15초(약 350 m) 전에야 안내받아, 옆 차에 막히면 못 빠지던 문제 (2026-11-25).
        self.nav_horizon_s_fast = float(mdp_obs.get("nav_horizon_s_fast", 0.0) or 0.0)
        self.nav_fast_speed = float(mdp_obs.get("nav_fast_speed", 20.0))
        # 신호 없는 갈림길(고속도로 출구·램프)에서 경로를 벗어나는 '출구 전용' 차로를 차로 연결성 [21..25] 에
        # '그 갈림길에서 끝나는 차로'로 표시 (2026-11-25). WBf_b 0번(R2907 램프 전용)을 몰라서, 앞쪽 양재IC 루프
        # '오른쪽' 안내를 따라 그 차로로 옮겼다가 R2907 로 빠지던 문제. 신호 교차로(회전 전용 차로)에는 적용 안 함.
        self.exit_lane_aware = bool(mdp_obs.get("exit_lane_aware", False))
        self._exit_only_cache = {}
        self.nav_norm = float(mdp_obs.get("nav_norm_m", self.nav_horizon))
        self.signal_horizon = float(mdp_obs.get("signal_horizon", 150.0))
        # 교차로 안에 서 있는 차도 앞차로 감지할지 (_junction_obstacle). False 면 예전처럼 getLeader 만.
        self.junction_obstacle = bool(mdp_obs.get("junction_obstacle", False))
        self.start_boost = dict(mdp_sim.get("train_start_boost", {}) or {})
        # 지정하면 코스 중간 출발을 이 도로들에서만 한다 (취약 구간 집중 연습). 비우면 기존처럼 전 구간.
        self.start_only = set(mdp_sim.get("train_start_only", []) or [])
        # 원래 경로(r0) 밖의 도로에서 출발해 r0 에 합류하는 학습 판 (학습 때만, 중간 출발 판 중 이 확률).
        # 예: R2907(양재IC 진입 램프) -> NB_e 0번 가속 차로 -> 본선 합류. 경로를 잘못 들었을 때만 겪는 상황이라
        # 학습 중에 거의 안 나와서, 일부러 그 상황에서 출발시킨다 (2026-11-25, SAFE10·11 1086/1023 램프 끝 정지).
        self.extra_starts = list(mdp_sim.get("train_extra_starts", []) or [])
        self.extra_start_prob = float(mdp_sim.get("train_extra_start_prob", 0.0) or 0.0)
        self._extra_routes = None
        # 차로 연결성([21..25])을 다음 도로까지 따라가며 볼 거리 (m). None 이면 예전처럼 지금 도로 안·시야 거리만.
        _leh = mdp_obs.get("lane_end_horizon")
        self.lane_end_horizon = float(_leh) if _leh else None
        # 비보호 초록('g') 신호값. None 이면 예전처럼 'G' 와 같은 1.0
        _ps = mdp_obs.get("permissive_signal")
        self.permissive_signal = float(_ps) if _ps is not None else None
        # 비보호('g') 링크로 교차로에 들어간 뒤에도(교차로 안) 신호값을 비보호(permissive_signal)로 유지하고
        # 정지선 거리는 0(이미 넘음)으로 둔다. 예전엔 교차로 안 = '앞에 신호 없음' = [1, 1] 이라
        # 일반 초록으로 직진·좌회전하는 상황과 관측이 같아져, 정지선을 넘는 순간 +0.6~0.7 로 급가속해
        # 맞은편 직진 차와 충돌했다 (SAFE11 4.1만 50000번대 비보호 좌회전 충돌 5판 모두, 2026-11-25).
        # 방식: False/"off" = 예전(교차로 안 = 신호 없음), True/"on" = 비보호 신호값 + 정지선 0,
        #       "sig" = 신호값만 비보호 유지(거리는 예전), "dist" = 거리만 0(신호값은 예전)  ← 원인 분리 실험용,
        #       "strict" = "on" 과 같되 지금 교차로 id·내부 차로가 캐시한 진입 링크와 일치할 때만 (2026-11-25)
        _pij = mdp_obs.get("permissive_in_junction", False)
        self.pij_mode = {True: "on", False: "off", None: "off"}.get(_pij, str(_pij))
        if self.pij_mode not in ("off", "on", "sig", "dist", "strict"):
            raise ValueError(f"permissive_in_junction 값이 이상합니다: {_pij!r}")
        self.permissive_in_junction = self.pij_mode != "off"
        # 교차로 내부 차로 이어가기에서 연결의 경유(via) 내부 차로(좌회전 뒷부분 :ar_yg_23 등)까지 포함할지.
        # False = 예전(뒷부분 빠짐 → 통과 시간·상대 차로·양보 필요가 틀림). SAFE11 까지는 False 로 학습 (2026-11-25)
        self.internal_chain_via = bool(mdp_obs.get("internal_chain_via", False))
        # 내 링크 캐시(_ego_link_info)를 경로 끝·링크 못 찾음·예외 때도 지울지. False = 예전 (2026-11-25)
        self.link_cache_clear = bool(mdp_obs.get("link_cache_clear", False))
        # A. 상대 차를 거슬러 찾을 때 교차로 내부 차로(연결 구간) 위의 차도 포함할지.
        #    False = 예전: 회전교차로처럼 짧은 구간이 이어지면 상대 차가 앞 교차로를 지나는 순간 관측에서 사라졌다 (2026-11-25)
        self.foe_scan_internal = bool(mdp_obs.get("foe_scan_internal", False))
        # 양보할 상대가 foe_horizon_s(8초) 보다 늦게 닿으면 [39] '양보 필요' 를 0 으로 (지금은 상관없는 먼 차).
        # False = 예전: 8초에서 잘린 [38]=1 과 양보=1 이 함께 나와, 멀리 거의 서 있는 차를 하염없이 기다렸다 (2026-11-25)
        self.yield_horizon_cut = bool(mdp_obs.get("yield_horizon_cut", False))
        # Y. 노란불 관측을 '멈출 노란불(빨강처럼 0)' / '지나갈 노란불(초록처럼 1)' 로 나눠 보여줄지.
        #    노란불이 켜진 순간 필요한 감속(v^2 / 2d)이 yellow_decision_decel 이하면 멈춤, 넘으면 지나감 (그 신호 동안 고정).
        #    False = 예전: 노란불은 늘 0.5 → 결정적 정책이 어정쩡하게 약하게 밟다가 빨간불에 넘던 문제 (2026-11-26)
        self.yellow_decision_obs = bool(mdp_obs.get("yellow_decision_obs", False))
        # S. 교차로 안에 멈춰 서서 내 진로(내부 차로 중심선)를 실제로 막고 있는 상대 차는, 내 신호가 우선이어도
        #    [39] '양보 필요' = 1 로 보여줄지. False = 예전: 초록불 직진(우선)이라 양보 0 → 맞은편 좌회전 대기차가
        #    교차로 안에서 내 차로를 막고 서 있는데도 그대로 들어가 부딪혔다 (ar_ys_11·ar_nh_11, 2026-11-26)
        self.stalled_foe_block = bool(mdp_obs.get("stalled_foe_block", False))
        self.stalled_foe_speed = float(mdp_obs.get("stalled_foe_speed", 1.0))
        self.stalled_foe_dist = float(mdp_obs.get("stalled_foe_dist", 2.0))
        # Q. 내 차로에서 정지선보다 앞에 다른 차(줄)가 있으면 나는 교차로 맨 앞이 아니므로 [38..39] 상대 차 관측을
        #    '없음'(1, 0) 으로. False = 예전: 정지선 60 m 안이면 줄 뒤 30~40 m 에서도 '양보 필요=1' 이 떠서
        #    줄에 붙지 않고 멀리 서서 기다렸다 (출발 구간 ar_hq_gate_out·ar_yj_f0b·ar_yg_a0_in, 2026-11-26)
        self.queue_yield_mask = bool(mdp_obs.get("queue_yield_mask", False))
        # F. 먼 앞차 관측 (2026-11-26): 같은 차로 앞차를 far_lead_range(150 m)까지 따로 본다 → 관측 끝에 2개
        #    [먼 앞차까지 거리 / far_lead_range (없으면 1), 그 차 속도 / 제한속도 (없으면 1)].
        #    기존 앞차 관측(visibility 50 m)은 그대로. 예전엔 대기줄 끝이 50 m 보다 멀면 '앞차 없음'으로 보여
        #    빨간불 정지선 기준으로 줄 뒤 수십 m 에 섰다 (뒤차들까지 줄줄이 멀리 섬).
        self.far_lead_obs = bool(mdp_obs.get("far_lead_obs", False))
        self.far_lead_range = float(mdp_obs.get("far_lead_range", 150.0))
        self.yellow_decision_decel = float(mdp_obs.get("yellow_decision_decel", 4.0))
        self._yobs_latch = None
        # B. [42] 여유 시간 관측: (양보할 상대 도착 시간 - 내가 교차로를 빠져나가는 예상 시간) / gap_margin_scale.
        #    양보할 상대가 없으면 1. 켜면 관측이 1개 늘어난다 (맨 뒤에 붙음) (2026-11-25)
        self.gap_margin_obs = bool(mdp_obs.get("gap_margin_obs", False))
        self.gap_margin_scale = float(mdp_obs.get("gap_margin_scale", 12.0))
        self.gap_margin_accel = float(mdp_obs.get("gap_margin_accel", 2.0))
        self._gap_margin = None
        # 교차로 상대 차 관측 방법: "route" (내부 차로 엇갈림 직접 계산) / "sumo" (예전 getJunctionFoes)
        self.junction_foe_mode = str(mdp_obs.get("junction_foe_mode", "sumo"))
        # 바로 옆 차로 앞뒤 차 관측에 직접 스캔(앞뒤 연결 구간 포함)도 합칠지. False 면 예전처럼 getNeighbors 만.
        self.neighbor_scan = bool(mdp_obs.get("neighbor_scan", False))
        self.foe_horizon_m = float(mdp_obs.get("foe_horizon_m", 150.0))   # route_time 전용
        self.foe_horizon_s = float(mdp_obs.get("foe_horizon_s", 8.0))
        self._upstream_cache = {}       # 들어오는 차로 → [(거슬러 올라간 차로, 정지선까지 더할 거리)]
        self._jfoe_cache = {}           # 교차로 → {내부 차로: (들어오는 차로, 나가는 도로, 링크 번호)}
        self._ego_link_prio = None      # 교차로 들어가기 전 내 링크가 우선(G)이었는지 (교차로 안에서 씀)
        self._foe_state = None          # 마지막 관측 때 교차로 상대 상태 (양보 위반 감점용)
        self._ego_link_info = None      # (신호 문자, 방향, 교차로 id) — 앞 교차로에서 내 링크
        self._ego_link_chain = ()       # 그 링크의 교차로 내부 차로들 (진입 링크 검증용, "strict")
        self._foe_vid = None            # [38..39] 에 잡힌 상대 차 id (분석 기록용, 주행에 영향 없음)
        self._pij_applied = False       # 이번 관측에서 교차로 안 비보호 유지가 실제로 적용됐는지 (분석 기록용)
        self._gb_reason = ""            # 좌회전 대기차 회피 감점 판정 이유 (검증용)
        self._jw_info = None            # 교차로 안 회전 대기차 접근 감점 판정 (검증용)
        self._le_info = None            # 차로 끝 유지 감점 판정 (검증용)
        self._pl_info = None            # 비보호 좌회전 추가 감점 판정 정보 (검증용)
        self._sumonet = None            # 정적 도로망 (sumolib, 처음 쓸 때 한 번 읽음)
        self._nav_key = None            # 경로별 안내표 캐시
        self._nav_tab = None
        self._route_cache_t = None      # [데이터 기록용] 같은 시각엔 getBestLanes 를 한 번만
        self._route_cache = None
        # 학습 중 데이터 기록기 (env/step_recorder.py). None 이면 기록 안 함.
        self.recorder = None
        self.recording = True           # 평가 주행 동안 잠시 끄는 스위치
        self._last_obs = None
        # 밀도 정규화용 "차량 1대가 점유하는 길이" (차체 + 최소 간격)
        self.density_veh_len = float(mdp_obs.get("density_veh_length", 7.5))
        self.act_dim = int(action["dim"])
        self.accel_index = int(action.get("accel_index", 0))
        self.lane_change_index = int(action.get("lane_change_index", 1))
        self.lane_change_threshold = float(action.get("lane_change_threshold", 0.5))
        self.lane_change_duration = float(action.get("lane_change_duration", 1.0))
        # 교차로 안에서 낸 차로 변경 명령을 버리지 않고, 교차로를 빠져나온 뒤 이 시간(초) 안에 실행한다 (0 = 끔).
        # 정책이 합류 교차로(:NB_m2907) 안에서 이미 '왼쪽'을 명령하는데 교차로 안이라 버려지고,
        # 가속 차로(NB_e 0번)에 들어서면 '유지'로 바뀌어 차로 끝에 서 버리던 문제 (SAFE10·11 1086 등, 2026-11-25).
        # 명령은 정책이 낸 것 그대로이고(사람이 깜빡이를 미리 켜 두는 것과 같음), 실행 때 기존 안전 검사를 그대로 거친다.
        self.lc_carry_s = float(action.get("lane_change_carry_s", 0.0) or 0.0)
        self._pending_lc = None         # (방향, 명령한 스텝)
        self.rw = reward

        # GUI는 ego를 따라가며 가까이 보이도록 기본값을 둔다.
        self.gui_zoom = float(mdp_sim.get("gui_zoom", 1200.0))
        self.gui_delay = float(mdp_sim.get("gui_delay", 100.0))
        self.gui_track_ego = bool(mdp_sim.get("gui_track_ego", True))

        self.ego_id = "ego"        # SUMO 안에서 ego 차량을 식별하는 ID
        self.step_count = 0
        self._started = False      # SUMO 프로세스가 떠 있는지 여부

        # Gymnasium 행동 공간:
        #   action[0] = 종방향 가감속 raw ∈ [-1,1]
        #   action[1] = 차선변경 raw ∈ [-1,1], 환경 내부에서 {-1,0,+1}로 양자화
        # PPO의 Gaussian 정책과 호환하기 위해 Box를 유지한다. 실제 적용된
        # 차선변경 명령은 info["lane_change"]에 정확한 -1/0/+1로 반환한다.
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.act_dim,), dtype=np.float32)
        # 관측: 19차원, 범위 [-1,1]. 레이아웃은 env/mdp_config.py 주석 참고.
        #   [0] 내 속도, [1..12] 3개 차선 × (선행/후행 상대거리·상대속도),
        #   [13..15] 3개 차선의 연결성(끊김 정도),
        #   시야 차선: 좌2, 좌1, 현재, 우1, 우2 (5개) — 3차선 도로에서
        #   맨 왼쪽에 있어도 오프셋 -2로 맨 오른쪽 차선까지 보인다.
        self.obs_offsets = (+2, +1, 0, -1, -2)
        n_lanes_view = len(self.obs_offsets)
        # + 신호(상태, 거리) 2 + 내비(거리, 좌, 우) 3 + 공사 거리 1 + 지금 도로 제한속도 1
        # + 교차로에서 만날 차(가장 가까운 차의 충돌 지점까지 거리, 내가 양보해야 하는지) 2
        self.obs_dim_single = 1 + n_lanes_view * 4 + n_lanes_view + n_lanes_view + 2 + 3 + 1 + 1 + 2
        # [40..41] 같은 차로 앞차의 방향지시등 (좌, 우). 켜면 관측이 2개 늘어난다 (2026-11-24).
        self.leader_signals = bool(mdp_obs.get("leader_signals", False))
        if self.leader_signals:
            self.obs_dim_single += 2
        if bool(mdp_obs.get("gap_margin_obs", False)):
            self.obs_dim_single += 1    # [42] 여유 시간 (맨 뒤)
        if bool(mdp_obs.get("far_lead_obs", False)):
            self.obs_dim_single += 2    # F. 먼 앞차 거리·속도 (맨 뒤)
        self._lead_vid = None           # 마지막으로 찾은 '같은 차로 앞차' id (방향지시등 관측용)
        # 관측 이력(POMDP): 정책 입력 = 최근 history 개 관측을 이어 붙인 것 [o_{t-k+1}, ..., o_t].
        # 한 장면만으로는 앞차가 지금 감속 중인지 알 수 없어서, 최근 몇 스텝을 함께 보여준다.
        # 데이터 기록(observation 칼럼)은 지금 시점 관측 o_t 하나만 저장한다 (이력은 앞 행들로 복원).
        self.obs_history = max(int(mdp_obs.get("history", 1)), 1)
        self.obs_dim = self.obs_dim_single * self.obs_history
        self._hist = None
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.obs_dim,),
                                            dtype=np.float32)

    # ==================================================================
    # SUMO 프로세스 시작
    # ==================================================================
    def _start_sumo(self):
        # gui=True면 화면이 있는 sumo-gui, 아니면 화면 없는 sumo (학습용, 훨씬 빠름)
        binary = checkBinary("sumo-gui" if self.gui else "sumo")
        cmd = [
            binary, "-c", self.cfg_path,
            "--step-length", str(self.step_length),
            # 충돌한 차량을 시뮬레이션에서 제거. (기본값 teleport는 순간이동시켜버려서
            # RL 학습에는 부적합. remove로 해야 "충돌 → 에피소드 종료"가 깔끔함)
            "--collision.action", "remove",
            # 교차로 안에서 부딪히는 것도 충돌로 판정 (신호 위반·끼어들기를 실제 사고로 배우게)
            "--collision.check-junctions", "true",
            "--no-warnings", "true",   # 콘솔 경고 끄기
            "--no-step-log", "true",   # 스텝 로그 끄기
            # 매 에피소드 다른 시드 → 배경 교통 패턴이 달라져서 과적합 방지
            "--seed", str(np.random.randint(0, 100000)),
        ]
        if self.gui:
            cmd += ["--quit-on-end"]      # 에피소드 종료 시 창 정리
            cmd += ["--delay", str(self.gui_delay)]
            if self.gui_autostart:
                cmd += ["--start"]        # 자동 재생
            # --start가 없으면: 창이 뜬 뒤 ▶(플레이)를 눌러야 진행된다.
            #   매 에피소드가 reset()에서 SUMO를 새로 띄우므로,
            #   에피소드마다 ▶를 눌러 시작하게 된다.
        # SUMO를 자식 프로세스로 띄우고 소켓 연결.
        # 수천 번 껐다 켜다 보면 가끔 SUMO 가 뜨자마자 연결이 끊기는 일이 있어서
        # (예: "Connection closed by SUMO") 잠깐 쉬었다가 몇 번 다시 시도한다. 주행 방식에는 영향 없음.
        import time
        for attempt in range(5):
            try:
                traci.start(cmd)
                break
            except _START_ERRORS as e:
                try:
                    traci.close()
                except Exception:  # noqa: BLE001  반쯤 열린 연결 정리 (없으면 무시)
                    pass
                if attempt == 4:
                    raise
                print(f"[SUMO 시작 실패 {attempt + 1}/5] {e!r} → {2 * (attempt + 1)}초 뒤 다시 시도")
                time.sleep(2 * (attempt + 1))
                cmd[cmd.index("--seed") + 1] = str(np.random.randint(0, 100000))
        self._started = True

    # ==================================================================
    # 에피소드 초기화
    # ==================================================================
    def _prefill_traffic(self):
        """도로 전 구간에 배경차를 일정 간격(±jitter)으로 미리 배치한다.

        flow는 도로 입구에서만 차를 만들기 때문에, 프리필이 없으면
        에피소드 초반의 긴 도로가 텅 비어 "빈 도로 주행"만 잔뜩 학습된다.
        프리필은 첫 스텝부터 ego가 실제 교통류 한가운데서 시작하게 만든다.

        - 위치: start_offset ~ (길이 - end_margin), 차선별 독립 배치
        - 간격: prefill_spacing × (1 ± jitter) 무작위
        - 타입: controllers 확률 비율로 샘플링 (rou.xml의 vType과 동일)
        - 속도: departSpeed="max" → 각 개체가 "자기 허용 최고속도"로 시작.
          컨트롤러별 max_speed/speedDev 차등이 프리필에도 그대로 반영되고,
          개체별 무작위 speedFactor 때문에 숫자 속도를 주면 SUMO가
          "허용속도 초과"로 거부할 수 있는 문제도 함께 피한다.
        배치된 차들은 SUMO의 각자 컨트롤러(IDM 등)가 정상적으로 운전한다.
        """
        rng = np.random.default_rng()
        count = 0
        # 차선별 간격 배율 (없으면 전 차선 동일). 인덱스 0 = 맨 오른쪽.
        factors = (self.traffic.get("prefill", {})
                   .get("lane_spacing_factor")
                   or [1.0] * self.num_lanes)
        for lane in range(self.num_lanes):
            f = float(factors[lane]) if lane < len(factors) else 1.0
            lane_spacing = self.prefill_spacing * f
            pos = self.prefill_start + rng.uniform(0, lane_spacing)
            while pos < self.road_length - self.prefill_end_margin:
                vid = f"pre_{lane}_{count}"
                vtype = str(rng.choice(self._prefill_types, p=self._prefill_probs))
                try:
                    traci.vehicle.add(
                        vid, "r0", typeID=vtype,
                        departLane=str(lane),
                        departPos=f"{pos:.1f}",
                        departSpeed="max")
                    count += 1
                except traci.TraCIException:
                    pass  # 겹침 등으로 실패한 자리는 건너뜀
                step = lane_spacing * rng.uniform(
                    1.0 - self.prefill_jitter, 1.0 + self.prefill_jitter)
                pos += max(step, 10.0)
        return count

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._pending_lc = None
        self._queue_episode = False
        self._queue_start = None

        # SUMO 상태를 완전히 초기화하는 가장 확실한 방법은 껐다 켜는 것.
        if self._started:
            traci.close()
            self._started = False
        self._start_sumo()

        # 도로 전 구간에 배경차를 미리 배치 (첫 스텝부터 실제 교통류 상태)
        if self.prefill_enabled:
            self._prefill_traffic()
        # 배치된 차들이 자리를 잡도록 몇 스텝만 정착시킨다.
        for _ in range(self.warmup_steps):
            traci.simulationStep()
        # 교통 채우기 (warmup_s): ego 없이 배경 교통만 먼저 흘려 도로를 정상 교통량으로 만든다.
        lo, hi = self.warmup_range
        if hi > 0.0:
            w_sec = float(np.random.uniform(lo, hi)) if hi > lo else lo
            # 화면(GUI)에서는 교통 채우는 동안만 지연 0 으로 빨리 돌리고, ego 출발 때 원래 지연으로 되돌린다.
            # 화면 속도만 바뀌고 시뮬레이션 결과는 같다 (2026-11-25)
            if self.gui:
                try:
                    traci.gui.setDelay("View #0", 0.0)
                except Exception:  # noqa: BLE001
                    pass
            for _ in range(int(w_sec / self.step_length)):
                traci.simulationStep()
            if self.gui:
                try:
                    traci.gui.setDelay("View #0", self.gui_delay)
                except Exception:  # noqa: BLE001
                    pass
        self.warmup_used_s = traci.simulation.getTime()

        # ego 차량 투입 — 기본은 경로 시작(현대차 본사 정문, 0 m).
        # 학습 때만(random_start_prob > 0) 일부 판은 원래 경로 위 다른 지점에서 출발한다.
        self.start_edge = self._add_ego(random_start=(np.random.rand() < self.random_start_prob))

        # ── ego 삽입 보증 ──
        # 시작 차선 입구가 잠시 막혀 있으면 SUMO가 삽입을 "지연"시킨다. 그 경우 ego 없이
        # 에피소드가 진행되어 즉시 종료되는 버그가 되므로, 삽입될 때까지 (최대 30스텝) 기다린다.
        # 코스 중간 출발이 30스텝 안에 안 되면 원래 출발점(정문)으로 다시 넣는다.
        # 정문 출발은 최대 240스텝(2분) 기다린다: 교통을 채운 뒤(warmup_s)에는 정문 출구 도로(37 m)에서
        # 출발하는 배경 교통이 정문 신호에 줄을 서 도로를 꽉 채우므로, 신호가 바뀌어 줄이 빠질 때까지
        # 기다려야 한다 (예전 30스텝 = 15초로는 'ego 삽입 실패' 로 평가가 멈췄다, 2026-11-24).
        for attempt in range(2):
            traci.simulationStep()  # ego가 실제로 도로에 나타나도록 1스텝 진행
            _wait = 0
            _limit = 30 if self.start_edge is not None else 240
            while self.ego_id not in traci.vehicle.getIDList() and _wait < _limit:
                traci.simulationStep()
                _wait += 1
            if self.ego_id in traci.vehicle.getIDList() or self.start_edge is None:
                break
            try:
                traci.vehicle.remove(self.ego_id)
            except traci.TraCIException:
                pass
            self._queue_episode = False
            self._queue_start = None
            self.start_edge = self._add_ego(random_start=False, allow_queue=False)
        if self.ego_id not in traci.vehicle.getIDList():
            raise RuntimeError(
                "ego 삽입 실패: 출발 도로 입구가 2분 넘게 막혀 있습니다. "
                "TRAFFIC['vehs_per_hour']를 낮추거나 depart_speed를 확인하세요.")

        # GUI 기본 시점: ego를 화면 중앙에서 추적하고 충분히 가까이 확대한다.
        # 긴 직선도로 전체를 한 화면에 맞추는 기본 뷰는 차량이 점/삼각형처럼
        # 보이기 쉬우므로 학습/평가 관찰용으로는 이 설정이 더 직관적이다.
        if self.gui and self.ego_id in traci.vehicle.getIDList():
            try:
                view_ids = traci.gui.getIDList()
                if view_ids:
                    view_id = view_ids[0]
                    # tracking을 먼저 켜고 zoom을 마지막에 적용해야 SUMO가
                    # 차량 추적을 시작하면서 viewport를 다시 덮어쓰지 않는다.
                    if self.gui_track_ego:
                        traci.gui.trackVehicle(view_id, self.ego_id)
                    traci.gui.setZoom(view_id, self.gui_zoom)
            except traci.TraCIException:
                pass

        self.step_count = 0
        self._stopped_steps = 0
        self._prev_rem = None
        self._best_rem = None           # 신기록 방식 진행 보상: 이번 판 목적지까지 남은 거리의 최솟값
        self._prev_lane_phi = None      # 차로 안내 보상의 직전 퍼텐셜 (_lane_guide_potential)
        self.reroute_count = 0
        self._last_reroute_edge = None
        self._last_limit = None
        self._ego_link_prio = None
        self._ego_link_info = None
        self._ego_link_chain = ()
        self._foe_vid = None
        self._pij_applied = False
        self._yellow_latch = None
        self._ctx_hold = None
        self._ctx_sig = (False, None)
        self._ctx_last = None
        self._ctx_pot = None
        self._far_lead_id = None
        self._signal_queue_last = None
        obs = self._get_obs()
        self._last_obs = obs
        if self.recorder is not None and self.recording:
            self.recorder.on_reset()
        return self._stacked(obs, reset=True), {"queue_start": self._queue_start}

    def _stacked(self, obs, reset: bool = False):
        """최근 obs_history 개 관측을 이어 붙인 정책 입력. 판 시작은 첫 관측으로 채운다."""
        if self.obs_history == 1:
            return obs
        if reset or self._hist is None:
            self._hist = [obs] * self.obs_history
        else:
            self._hist = self._hist[1:] + [obs]
        return np.concatenate(self._hist).astype(np.float32)

    # ==================================================================
    # 관측 만들기 (레이아웃은 env/mdp_config.py 주석 참고)
    #
    # 핵심 설계: 특정 도로에 대한 하드코딩 없이, 매 스텝 SUMO 네트워크를
    # 실시간 조회해서 "지금 내 주변"을 계산한다. 그래서 도로 구조가
    # 어떻게 바뀌어도(차선 수, 합류, 차선 감소 등) 이 코드는 그대로 동작한다.
    #
    # 사용하는 TraCI API:
    #   getLeader / getFollower : 같은 차선의 앞/뒤 차량 (edge 경계도 넘어 탐색)
    #   getNeighbors(mode)      : 옆 차선의 앞/뒤 차량.
    #       mode 비트: bit0 = 0:왼쪽 / 1:오른쪽,  bit1 = 0:후행 / 1:선행
    #   lane.getLinks           : 차선이 다음 edge로 이어지는지 (연결성 판단)
    # ==================================================================

    # (차선 오프셋, 선행차 getNeighbors mode, 후행차 mode)
    #   오프셋 +1 = 왼쪽 차선 (SUMO는 차선 index가 왼쪽으로 갈수록 커짐)
    _NEIGHBOR_MODES = {+1: (0b10, 0b00),   # 왼쪽: 선행=2, 후행=0
                       -1: (0b11, 0b01)}   # 오른쪽: 선행=3, 후행=1

    def _far_leader(self):
        """F. 같은 차로(경로를 따라) far_lead_range m 안의 가장 가까운 앞차 (거리 m, 속도). 없으면 None."""
        try:
            res = traci.vehicle.getLeader(self.ego_id, self.far_lead_range)
        except traci.TraCIException:
            return None
        if res is None or res[0] == "" or not (0.0 <= res[1] <= self.far_lead_range):
            self._far_lead_id = None
            return None
        self._far_lead_id = res[0]
        return float(res[1]), float(traci.vehicle.getSpeed(res[0]))

    def _neighbor(self, offset: int, leader: bool):
        """오프셋 차선(-1=오른쪽, 0=현재, +1=왼쪽)의 가장 가까운
        선행(leader=True)/후행 차량을 (거리 m, 그 차량 속도)로 반환.
        visibility 밖이거나 없으면 None."""
        if offset == 0:
            if leader:
                res = traci.vehicle.getLeader(self.ego_id, self.visibility)
                cands = []
                if res is not None and res[0] != "" and 0.0 <= res[1] <= self.visibility:
                    cands.append((float(res[1]), traci.vehicle.getSpeed(res[0]), res[0]))
                if self.junction_obstacle:
                    jo = self._junction_obstacle()
                    if jo is not None:
                        cands.append((jo[0], jo[1], self._jo_vid))
                if not cands:
                    self._lead_vid = None
                    return None
                g, v, vid = min(cands, key=lambda c: c[0])
                self._lead_vid = vid
                return g, v
            else:
                vid, gap = traci.vehicle.getFollower(self.ego_id, self.visibility)
                if vid == "":
                    return None
        elif abs(offset) >= 2:
            # 두 칸 이상 옆 차선: SUMO getNeighbors는 "바로 옆"만 지원하므로
            # 차선 직접 스캔(_scan_lane_neighbors — 같은 edge의 차량 목록을
            # 훑어 가장 가까운 앞/뒤를 찾음)으로 감지한다.
            front, rear = self._scan_lane_neighbors(offset)
            res = front if leader else rear
            if res is None or res[0] < 0 or res[0] > self.visibility:
                return None
            return float(res[0]), float(res[1])
        else:
            mode_leader, mode_follower = self._NEIGHBOR_MODES[offset]
            pairs = traci.vehicle.getNeighbors(
                self.ego_id, mode_leader if leader else mode_follower)
            if self.neighbor_scan:
                # 바로 옆 차로도 직접 스캔(앞뒤 연결 구간 포함)과 합쳐 더 가까운 쪽을 쓴다.
                # getNeighbors 는 합류·교차로 직후 뒤쪽 연결 구간의 차를 못 봤다 (2026-11-24).
                front, rear = self._scan_lane_neighbors(offset)
                sc = front if leader else rear
                cands = []
                if pairs:
                    pv, pg = min(pairs, key=lambda p: p[1])
                    if 0.0 <= pg <= self.visibility:
                        cands.append((float(pg), traci.vehicle.getSpeed(pv)))
                if sc is not None and sc[0] <= self.visibility:
                    cands.append((max(float(sc[0]), 0.0), float(sc[1])))
                return min(cands) if cands else None
            if not pairs:
                return None
            vid, gap = min(pairs, key=lambda p: p[1])   # 가장 가까운 차량

        if gap < 0 or gap > self.visibility:
            return None
        return float(gap), traci.vehicle.getSpeed(vid)

    def _junction_internal_lanes(self, node_id):
        """교차로 node_id 안쪽 차로(':node_번호_차로') 목록. 교차로마다 한 번만 찾아 둔다."""
        cache = self.__dict__.setdefault("_jlanes_cache", {})
        if node_id not in cache:
            pre = f":{node_id}_"
            cache[node_id] = [l for l in traci.lane.getIDList()
                              if l.startswith(pre) and re.fullmatch(r"\d+_\d+", l[len(pre):])]
        return cache[node_id]

    def _junction_obstacle(self):
        """앞 교차로(또는 지금 지나는 교차로) 안의 차 중, 내 진행 방향 앞·내 차폭 안에 있는 가장 가까운 차.

        getLeader 는 내 경로와 같은 교차로 안쪽 차로의 차만 앞차로 본다. 그래서 같은 입구 차로에서
        좌회전하려고 교차로 안에 멈춰 선 차(직진·좌회전 공용 차로)처럼 '경로는 다르지만 실제로 내 앞을
        막고 있는' 차가 안 보여서, 관측상 '앞이 비었음' 인 채로 추돌했다 (남부순환로 분석, 2026-11-23).
        사람이 교차로 안에 서 있는 차를 눈으로 보는 것과 같은 정보다. 반환: (거리 m, 그 차 속도) 또는 None.
        """
        try:
            ego = self.ego_id
            road = traci.vehicle.getRoadID(ego)
            if road.startswith(":"):
                node_id = road[1:].rsplit("_", 1)[0]
            else:
                lane = traci.vehicle.getLaneID(ego)
                if traci.lane.getLength(lane) - traci.vehicle.getLanePosition(ego) > self.visibility:
                    return None
                node_id = self._net().getEdge(road).getToNode().getID()
            lanes = self._junction_internal_lanes(node_id)
            if not lanes:
                return None
            x, y = traci.vehicle.getPosition(ego)
            ang = np.radians(traci.vehicle.getAngle(ego))      # SUMO: 0 = 북쪽, 시계 방향
            hx, hy = np.sin(ang), np.cos(ang)
            best = None
            for ln in lanes:
                for v in traci.lane.getLastStepVehicleIDs(ln):
                    if v == ego:
                        continue
                    # 상대 차의 앞 범퍼·가운데·뒤 범퍼 세 점 중 하나라도 내 앞, 내 차폭 안이면 막고 있는 차.
                    # (앞 범퍼만 보면 좌회전하며 앞이 꺾인 차는 차체 뒷부분이 내 차로에 걸쳐 있어도 놓친다)
                    vx, vy = traci.vehicle.getPosition(v)
                    va = np.radians(traci.vehicle.getAngle(v))
                    L = traci.vehicle.getLength(v)
                    gap = None
                    for f in (0.0, 0.5, 1.0):
                        px, py = vx - f * L * np.sin(va), vy - f * L * np.cos(va)
                        dx, dy = px - x, py - y
                        lon = dx * hx + dy * hy                # 내 진행 방향으로 얼마나 앞인가
                        lat = abs(dx * hy - dy * hx)           # 옆으로 얼마나 떨어졌나
                        if lon > 0.0 and lat <= self.JUNCTION_OBSTACLE_LAT:
                            gap = lon if gap is None else min(gap, lon)
                    if gap is not None and gap <= self.visibility and (best is None or gap < best[0]):
                        best = (float(gap), traci.vehicle.getSpeed(v))
                        self._jo_vid = v
            return best
        except (traci.TraCIException, KeyError, AttributeError):
            return None

    JUNCTION_OBSTACLE_LAT = 2.0   # m. 이 옆 거리 안이면 '내 앞을 막는 차' (차폭 약 1.8 m 두 대의 절반 합 + 여유)

    def _lane_guide_potential(self):
        """차로 안내 보상용 퍼텐셜 Φ = −(경로로 이어지는 차로까지 필요한 차로 변경 수).

        좌회전·우회전·출구 모두 같은 규칙: SUMO getBestLanes 의 offset (경로를 계속 따라가려면
        몇 차로 옮겨야 하는지) 을 쓴다. "왼쪽일수록" 같은 고정 규칙이 아니다.
        지금 차로로 갈 수 있는 거리가 안내 거리(관측의 내비 안내와 같은 거리) 보다 멀면 0
        → 에이전트가 볼 수 없는 먼 회전 때문에 보상이 생기지 않게 한다.
        교차로 안이거나 조회 실패면 None (직전 값 유지).
        """
        rl = self._route_lanes()
        if not rl:
            return None
        try:
            cur = int(traci.vehicle.getLaneIndex(self.ego_id))
        except traci.TraCIException:
            return None
        if cur not in rl:
            return None
        remain, offset, _ = rl[cur]
        horizon = self._nav_horizon_m()
        base = 0.0 if (offset == 0 or remain > horizon) else -float(abs(offset))
        sc = float(self.rw.get("shared_lane_cost", 0.0))
        if sc > 0.0 and base == 0.0 and self._on_shared_lane(cur):
            base -= sc
        return base

    def _on_shared_lane(self, cur):
        """지금 차로가 경로 밖 방향(좌·우회전 등)과 같이 쓰는 차로이고, 같은 도로에 경로 방향만 가는 차로가 있으면 True.
        예: 남부순환로 ar_ns_f0 2번 차로(직진+좌회전) — 좌회전 차가 교차로 안에서 맞은편을 기다리며 서 있다."""
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            route = traci.vehicle.getRoute(self.ego_id)
            ri = traci.vehicle.getRouteIndex(self.ego_id)
        except traci.TraCIException:
            return False
        if edge.startswith(":") or ri + 1 >= len(route):
            return False
        nxt = route[ri + 1]
        cache = self.__dict__.setdefault("_shared_cache", {})
        key = (edge, nxt)
        if key not in cache:
            info = {}
            try:
                for ln in self._net().getEdge(edge).getLanes():
                    tos = {c.getTo().getID() for c in ln.getOutgoing()}
                    info[ln.getIndex()] = (nxt in tos, tos == {nxt})
            except KeyError:
                info = {}
            pure = any(v[1] for v in info.values())
            cache[key] = {i: (v[0] and not v[1] and pure) for i, v in info.items()}
        return bool(cache[key].get(cur, False))

    def _route_lanes(self):
        """[데이터 기록 + 차로 안내 '보상' 전용 — 관측에는 쓰지 않는다]
        현재 edge 의 차로별로 "경로를 따라 차로 변경 없이 더 갈 수 있는 거리"를 구한다.
        (어느 차로로 가야 하는지 알려주는 세밀한 정보라 정책 입력에 넣으면 안 된다)

        반환: {차로 index: (남은 거리 m, 경로 차로까지 필요한 변경 수(+왼쪽), 목적지까지 거리 m)}
              교차로 내부이거나 조회 실패면 None.
        SUMO getBestLanes 는 ego 의 실제 경로(r0)를 기준으로, 각 차로에서 차로 변경 없이
        갈 수 있는 길이를 준다. 좌회전 차로·출구 차로가 아닌 곳에 있으면 그 값이 짧다.
        """
        t = traci.simulation.getTime()
        if self._route_cache_t == t:
            return self._route_cache
        out = None
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if not edge.startswith(":"):
                pos = traci.vehicle.getLanePosition(self.ego_id)
                route = traci.vehicle.getRoute(self.ego_id)
                last = route[-1]
                d_end = traci.vehicle.getDrivingDistance(
                    self.ego_id, last, max(traci.lane.getLength(f"{last}_0") - 0.1, 0.0))
                if d_end < 0:                   # 계산 불가(INVALID) → 아주 멀다고 본다
                    d_end = float("inf")
                out = {}
                for q in traci.vehicle.getBestLanes(self.ego_id):
                    lane_id, length, offset = q[0], q[1], q[3]
                    if lane_id.rsplit("_", 1)[0] != edge:
                        continue
                    out[int(lane_id.rsplit("_", 1)[1])] = (float(length) - pos, int(offset), d_end)
        except traci.TraCIException:
            out = None
        self._route_cache_t, self._route_cache = t, out
        return out

    def _lane_connectivity(self, offset: int) -> float:
        """오프셋 차선의 '끊김 정도'를 반환.

        -1 : 그 차선이 존재하지 않음
         1 : visibility 이내에서 계속 이어짐
        0~1: (차선이 끝나는 지점까지의 거리 / visibility)
             → 값이 작을수록 곧 차선이 사라진다는 경고

        일반화 원리: 차선의 남은 길이와, 차선 끝에서 다음 edge로의
        연결(link) 존재 여부를 SUMO 네트워크에서 직접 조회한다.
        차선 감소/합류 지점은 "링크 없는 차선의 끝"으로 자연히 감지된다.
        """
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):          # 교차로(junction) 내부 통과 중
                return 1.0
            lane_index = traci.vehicle.getLaneIndex(self.ego_id) + offset
            if lane_index < 0 or lane_index >= traci.edge.getLaneNumber(edge):
                return -1.0                   # 그런 차선이 없음
            lane_id = f"{edge}_{lane_index}"
            remaining = (traci.lane.getLength(lane_id)
                         - traci.vehicle.getLanePosition(self.ego_id))
            if self.lane_end_horizon is not None:
                return self._lane_end_lookahead(lane_id, remaining)
            if remaining > self.visibility:
                return 1.0                    # 시야 내에선 계속 이어짐
            if traci.lane.getLinks(lane_id):
                return 1.0                    # edge가 끝나도 다음으로 연결됨
            return float(np.clip(remaining / self.visibility, 0.0, 1.0))
        except traci.TraCIException:
            return 1.0                        # 조회 실패 시 중립값

    def _nav_horizon_m(self) -> float:
        """내비게이션 안내 거리 (m) = max(nav_horizon, 초 x 지금 도로 제한속도). 빠른 도로는 nav_horizon_s_fast 초."""
        v = self._speed_limit()
        sec = self.nav_horizon_s
        if self.nav_horizon_s_fast > 0.0 and v >= self.nav_fast_speed:
            sec = self.nav_horizon_s_fast
        return max(self.nav_horizon, sec * v)

    def _exit_only(self, lane_id: str, nxt_edge) -> bool:
        """이 차로가 경로를 벗어나는 '출구 전용' 차로인가: 끝에서 경로의 다음 도로로 가는 연결이 없고,
        같은 도로의 다른 차로는 그 다음 도로로 이어지며, 끝 지점이 신호 교차로가 아님 (고속도로 출구·램프 갈림길)."""
        if not nxt_edge:
            return False
        key = (lane_id, nxt_edge)
        hit = self._exit_only_cache.get(key)
        if hit is not None:
            return hit
        out = False
        try:
            net = self._net()
            ln = net.getLane(lane_id)
            edge = ln.getEdge()
            if edge.getToNode().getType() not in ("traffic_light", "traffic_light_unregulated", "traffic_light_right_on_red"):
                goes = lambda L: nxt_edge in {c.getTo().getID() for c in L.getOutgoing()}
                if ln.getOutgoing() and not goes(ln) and any(goes(o) for o in edge.getLanes() if o is not ln):
                    out = True
        except (KeyError, AttributeError):
            out = False
        self._exit_only_cache[key] = out
        return out

    def _lane_end_lookahead(self, lane_id: str, remaining: float) -> float:
        """차로를 다음 도로까지 따라가며 lane_end_horizon(m) 안에서 차로가 끝나는지 본다.

        반환: 1 = horizon 안에서 계속 이어짐, 0~1 = (끝나는 곳까지 거리 / horizon).
        예전에는 지금 도로 안에서만 봐서, 다음 도로에서 사라지는 차로(예: NB_e 1번 → NB_f 0번,
        6→5차로 감소)가 끝나기 50 m(고속에서 약 2초) 전에야 보였다. 그래서 '왼쪽으로 옮겨라'는 보상의
        이유가 관측에 없어 차로 결정이 50:50 으로 갈렸다 (NB_f 정지 14%, 2026-11-24).
        다음 차로는 ego 경로를 따라가는 연결을 우선 고르고, 없으면 첫 연결을 따라간다.
        """
        H = self.lane_end_horizon
        dist, cur = remaining, lane_id
        route = traci.vehicle.getRoute(self.ego_id)
        ri = max(traci.vehicle.getRouteIndex(self.ego_id), 0)
        for _ in range(12):                     # 짧은 도로가 이어져도 무한 반복하지 않게
            if dist >= H:
                return 1.0
            links = traci.lane.getLinks(cur)
            if not links:
                return float(np.clip(dist / H, 0.0, 1.0))
            nxt_edge = route[ri + 1] if ri + 1 < len(route) else None
            if self.exit_lane_aware and self._exit_only(cur, nxt_edge):
                return float(np.clip(dist / H, 0.0, 1.0))   # 출구 전용 차로: 경로로는 여기서 끝남
            pick = next((lk for lk in links if lk[0].rsplit("_", 1)[0] == nxt_edge), links[0])
            if pick[0].rsplit("_", 1)[0] == nxt_edge:
                ri += 1
            via_len = float(pick[7]) if len(pick) > 7 else 0.0
            cur = pick[0]
            dist += via_len + traci.lane.getLength(cur)
        return 1.0

    def _lane_density(self, offset: int) -> float:
        """오프셋 차선의 '전방 차량 밀도'를 0~1로 반환.

        밀도 = (시야 W 이내 전방 차량 수) / (그 구간의 최대 수용 대수)
               최대 수용 대수 = 사용가능거리 / density_veh_length

        인코딩 관례 (막힘 정도로 읽히도록):
            1.0 : 차선이 없거나 사실상 진입 불가  ← "꽉 찬 것"처럼 취급해서
                  에이전트가 자연스럽게 회피하게 만든다
            0.0 : 텅 빈 차선
        일반화: 차선이 시야보다 짧으면 link를 따라 다음 차선까지 이어서 센다.
        곧 끊기는 차선(link 없음)은 남은 거리만큼만 세므로 분모가 줄어
        같은 대수여도 밀도가 높게 나온다 → "끊기는 차선에 차가 몰려있음" 신호.
        """
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):          # 교차로 내부: 중립값
                return 0.0
            lane_index = traci.vehicle.getLaneIndex(self.ego_id) + offset
            if lane_index < 0 or lane_index >= traci.edge.getLaneNumber(edge):
                return 1.0                    # 차선 없음 = 꽉 참 취급
            lane_id = f"{edge}_{lane_index}"
            ego_pos = traci.vehicle.getLanePosition(self.ego_id)
            remaining = traci.lane.getLength(lane_id) - ego_pos
            links = traci.lane.getLinks(lane_id)

            # 이 차선에서 앞으로 실제로 쓸 수 있는 거리
            usable = self.visibility if links else min(self.visibility, remaining)
            if usable <= 0:
                return 1.0

            count = 0
            # (a) 현재 차선 구간의 전방 차량
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                if vid == self.ego_id:
                    continue
                d = traci.vehicle.getLanePosition(vid) - ego_pos
                if 0.0 < d <= min(self.visibility, remaining):
                    count += 1
            # (b) 시야가 edge 끝을 넘고 다음 차선으로 이어지면 이어서 센다
            if links and remaining < self.visibility:
                next_lane_id = links[0][0]    # 첫 번째 연결 차선
                for vid in traci.lane.getLastStepVehicleIDs(next_lane_id):
                    if traci.vehicle.getLanePosition(vid) <= \
                            self.visibility - remaining:
                        count += 1

            max_count = usable / self.density_veh_len
            return float(np.clip(count / max_count, 0.0, 1.0))
        except traci.TraCIException:
            return 0.0

    # ==================================================================
    # ego 투입 (기본: 정문 출발 / 학습 때 일부: 원래 경로 위 다른 지점 출발)
    # ==================================================================
    def _start_candidates(self, route):
        """원래 경로의 도로 조각 중 출발해도 되는 것: 100 m 이상, 회전교차로 안이 아님,
        목적지 바로 앞 두 조각 제외, 첫 조각(정문 진입로)은 기본 출발이라 제외."""
        net = self._net()
        ring = set()
        for rb in net.getRoundabouts():
            for e in rb.getEdges():
                ring.add(e if isinstance(e, str) else e.getID())
        out = []
        for k in range(1, len(route) - 2):
            try:
                L = net.getEdge(route[k]).getLength()
            except KeyError:
                continue
            if L >= 100.0 and route[k] not in ring:
                out.append((k, L))
        return out

    def _start_weights(self, route, cands):
        """방향 전환(좌·우회전·출구·유턴) 직전 조각에서 더 자주 출발하도록 가중치를 준다.

        어려운 상황을 더 자주 연습하게 하는 출발 위치 분포일 뿐, 주행에는 관여하지 않는다
        (평가는 항상 정문 출발)."""
        tab = self._nav_table(route)
        w = np.array([3.0 if (k in tab or (k + 1) in tab) else 1.0 for k, _ in cands])
        # 지정한 구간에서 더 자주 출발 (train_start_boost: {도로 id: 배수}). 평가에서 반복해서 막히는데
        # 학습 판에서는 거의 겪지 않는 상황을 집중 연습시키려는 것 (NB_f 차로 끝: 12만 스텝 동안 20번뿐).
        for i, (k, _) in enumerate(cands):
            w[i] *= float(self.start_boost.get(route[k], 1.0))
            if self.start_only and route[k] not in self.start_only:
                w[i] = 0.0
        if w.sum() <= 0:
            raise ValueError(f"--start-only 도로 {sorted(self.start_only)} 중 출발 가능한 곳(경로 위, 100 m 이상)이 없습니다")
        return w / w.sum()

    def _extra_start_routes(self, route):
        """extra_starts 의 각 도로에서 출발해 원래 경로 r0 에 합류하는 경로들. [(route id, 출발 도로, 길이)]

        출발 도로에서 다음 도로들을 따라가며(최대 6단계) r0 위의 도로(정문 진입로 제외)를 처음 만나면
        [출발 도로 ... 합류 도로] + r0 의 나머지로 경로를 만든다. 못 찾으면 그 도로는 건너뛴다.
        """
        net = self._net()
        pos = {e: i for i, e in enumerate(route)}     # 같은 도로가 두 번 나오면 뒤쪽 위치
        out = []
        for e0 in self.extra_starts:
            try:
                start = net.getEdge(e0)
            except KeyError:
                print(f"[출발] --extra-start 도로 {e0} 가 도로망에 없어 건너뜀")
                continue
            frontier, seen, found = [[e0]], {e0}, None
            for _ in range(6):
                nxt = []
                for path in frontier:
                    for succ in net.getEdge(path[-1]).getOutgoing():
                        sid = succ.getID()
                        if sid in pos and pos[sid] >= 1:
                            found = path + [sid]
                            break
                        if sid not in seen and not sid.startswith(":"):
                            seen.add(sid)
                            nxt.append(path + [sid])
                    if found:
                        break
                if found:
                    break
                frontier = nxt
            if not found:
                print(f"[출발] --extra-start 도로 {e0} 에서 원래 경로로 합류하는 길을 못 찾아 건너뜀")
                continue
            k = pos[found[-1]]
            rid = f"r0_extra_{e0}"
            edges = found + list(route[k + 1:])
            out.append((rid, edges, e0, float(start.getLength())))
        return out

    def _add_ego(self, random_start: bool, allow_queue: bool = True):
        """ego 를 넣고, 코스 중간에서 출발했으면 그 도로 id 를, 정문 출발이면 None 을 돌려준다."""
        start = None
        if allow_queue and self.queue_start_prob > 0 and np.random.rand() < self.queue_start_prob:
            from env.queue_curriculum import add_queue_start
            self._queue_trials += 1
            # Natural queues are intermittent at the default traffic density.
            # Search different signal phases BEFORE ego insertion, without changing signals.
            for attempt in range(25):
                start = add_queue_start(self, traci)
                if start is not None:
                    break
                if attempt < 24:
                    for _ in range(max(1, int(5 / self.step_length))):
                        traci.simulationStep()
            self.warmup_used_s = traci.simulation.getTime()
            if start is not None:
                self._queue_successes += 1
                random_start = False
            if self._queue_trials == 1 or self._queue_trials % 10 == 0:
                print(f"[queue-start] found={self._queue_successes}/{self._queue_trials} "
                      f"last={self._queue_start}", flush=True)
        if random_start and self.extra_starts and np.random.rand() < self.extra_start_prob:
            route = list(traci.route.getEdges("r0"))
            if self._extra_routes is None:
                self._extra_routes = self._extra_start_routes(route)
            if self._extra_routes:
                rid, edges, e0, L = self._extra_routes[np.random.randint(len(self._extra_routes))]
                if rid not in traci.route.getIDList():
                    traci.route.add(rid, edges)
                pos = float(np.random.uniform(10.0, max(L - 80.0, 11.0)))
                traci.vehicle.add(self.ego_id, rid, typeID="ego_type",
                                  departLane="random", departPos=f"{pos:.1f}", departSpeed="avg")
                start = e0
                random_start = False
        if random_start:
            route = list(traci.route.getEdges("r0"))
            if self._start_cands is None:
                self._start_cands = self._start_candidates(route)
            if self._start_cands:
                if getattr(self, "_start_w", None) is None or len(self._start_w) != len(self._start_cands):
                    self._start_w = self._start_weights(route, self._start_cands)
                k, L = self._start_cands[np.random.choice(len(self._start_cands), p=self._start_w)]
                rid = f"r0_from{k}"
                if rid not in traci.route.getIDList():
                    traci.route.add(rid, route[k:])
                # 도로 조각 안의 무작위 위치(앞 10 m ~ 끝 80 m 전), 무작위 차로, 그 차로의 평균 속도.
                # 끝에 너무 가까우면 출발하자마자 신호·분기에 닿아 손쓸 틈 없이 위반·실패하므로 80 m 띄운다.
                pos = float(np.random.uniform(10.0, max(L - 80.0, 11.0)))
                traci.vehicle.add(self.ego_id, rid, typeID="ego_type",
                                  departLane="random", departPos=f"{pos:.1f}", departSpeed="avg")
                start = route[k]
        if start is None:
            traci.vehicle.add(self.ego_id, "r0", typeID="ego_type",
                              departLane=self.ego_depart_lane,
                              departPos="0",
                              departSpeed=self.ego_depart_speed)
        # ★ 중요: speedMode=0 → SUMO의 내장 안전장치(자동 안전제동 등)를 전부 끔.
        #   이걸 꺼야 RL이 속도를 온전히 제어하고, 잘못 배우면 실제로 충돌한다.
        #   (켜두면 SUMO가 알아서 브레이크를 밟아줘서 충돌이 거의 안 나고,
        #    그러면 "충돌 회피"라는 학습 신호 자체가 사라진다)
        #   신호·교차로 통행 규칙도 SUMO가 대신 지켜주지 않는다 (보조 규칙 금지).
        #   대신 관측에 앞 신호 상태·정지선까지 거리를 넣고(_signal_features),
        #   교차로 안 충돌도 충돌로 판정한다(--collision.check-junctions).
        traci.vehicle.setSpeedMode(self.ego_id, 0)
        # SUMO의 자율 차선변경은 끄고, RL이 내리는 TraCI 차선변경 명령만 사용한다.
        traci.vehicle.setLaneChangeMode(self.ego_id, 0)
        return start

    # ==================================================================
    # 신호등 · 내비게이션 · 공사 구간 관측 (지도 수준 정보만, 차로 안내 없음)
    # ==================================================================
    JUNCTION_LOOK = 60.0     # 앞 교차로를 보는 거리 (m)
    FOE_SCALE = 50.0         # 상대 차의 충돌 지점까지 거리 정규화 (m)

    def _junction_foe_features(self):
        """앞 교차로(JUNCTION_LOOK m 안)에서 내 진로와 엇갈릴 차 중 충돌 지점에 가장 가까운 차.

        [충돌 지점까지 그 차의 거리 / FOE_SCALE (0~1, 없으면 1), 내가 양보해야 하는 차면 1 아니면 0]
        맞은편 직진 차(비보호 좌회전), 교차 방향 차, 합류해 들어오는 차가 여기에 잡힌다.
        옆 차로 앞뒤 차만 보던 기존 관측으로는 교차로 안의 상대가 안 보여 피할 수가 없었다.
        사람이 교차로에서 고개를 돌려 보는 정보와 같다 (규칙·차로 안내 아님).
        """
        if self.junction_foe_mode in ("route", "route_time"):
            return self._route_foe_features()
        try:
            foes = traci.vehicle.getJunctionFoes(self.ego_id, self.JUNCTION_LOOK)
        except (traci.TraCIException, AttributeError):
            return [1.0, 0.0]
        best = None
        for f in foes:
            try:
                ego_d, foe_d, ego_resp = float(f[1]), float(f[2]), bool(f[7])
            except (IndexError, TypeError, ValueError):
                continue
            if ego_d < 0 or foe_d < 0:
                continue
            if best is None or foe_d < best[0]:
                best = (foe_d, 1.0 if ego_resp else 0.0)
        if best is None:
            return [1.0, 0.0]
        return [float(np.clip(best[0] / self.FOE_SCALE, 0.0, 1.0)), best[1]]

    @staticmethod
    def _prio(ch):
        """링크 상태 문자가 우선권(멈추지 않고 지나갈 권리)이면 True. G=우선 초록, M=우선 비신호, O=신호 꺼짐."""
        return ch in ("G", "M", "O")

    def _internal_chain(self, lane_id):
        """내부 차로에서 시작해 교차로를 빠져나갈 때까지 이어지는 내부 차로들 (좌회전 대기 지점 차로 포함).
        internal_chain_via 면 경유(via) 내부 차로까지 따라간다 (_internal_chain_full)."""
        if self.internal_chain_via:
            return self._internal_chain_full(lane_id)
        chain, cur = [], lane_id
        for _ in range(4):
            chain.append(cur)
            nxt = None
            for lk in traci.lane.getLinks(cur):
                if lk[0].startswith(":"):
                    nxt = lk[0]
                    break
            if nxt is None or nxt in chain:
                break
            cur = nxt
        return chain

    @staticmethod
    def _internal_chain_full(lane_id):
        """_internal_chain 과 같되, 연결의 경유(via) 내부 차로까지 따라간다 (분석·"strict" 검증용).
        SUMO 는 좌회전 뒷부분(:ar_yg_23, :ar_rb_J_8, :ar_ys_22)을 '다음 일반 차로로 가는 연결의 via' 로 저장해서
        _internal_chain 은 이 부분을 빠뜨린다 (2026-11-25 확인). 기존 관측과 맞추려고 _internal_chain 은 그대로 둔다."""
        chain, cur = [], lane_id
        for _ in range(6):
            chain.append(cur)
            nxt = None
            for lk in traci.lane.getLinks(cur):
                if lk[0].startswith(":"):
                    nxt = lk[0]
                    break
                if len(lk) > 4 and lk[4] and lk[4].startswith(":"):
                    nxt = lk[4]
                    break
            if nxt is None or nxt in chain:
                break
            cur = nxt
        return chain

    def _internal_origin(self, node_id):
        """교차로 node_id 의 내부 차로마다 (들어오는 일반 차로, 나가는 도로, 그 차로의 링크 번호)."""
        if node_id in self._jfoe_cache:
            return self._jfoe_cache[node_id]
        table = {}
        for e in self._net().getNode(node_id).getIncoming():
            for ln in e.getLanes():
                lid = ln.getID()
                for i, lk in enumerate(traci.lane.getLinks(lid)):
                    if not lk[4]:
                        continue
                    to_edge = lk[0].rsplit("_", 1)[0]
                    for il in self._internal_chain(lk[4]):
                        table.setdefault(il, (lid, to_edge, i))
        self._jfoe_cache[node_id] = table
        return table

    def _upstream_lanes(self, src):
        """들어오는 차로 src 와 그 앞(상류) 차로들을 foe_horizon_m 까지: [(차로, 그 차로 끝 → 정지선 거리)]."""
        if src in self._upstream_cache:
            return self._upstream_cache[src]
        net = self._net()
        out, frontier, seen = [(src, 0.0)], [(src, 0.0)], {src}
        while frontier:
            lid, off = frontier.pop()
            try:
                lane = net.getLane(lid)
            except (KeyError, AttributeError):
                continue
            base = off + lane.getLength()
            if base >= self.foe_horizon_m:
                continue
            for up in lane.getIncoming():
                uid = up.getID()
                if uid in seen:
                    continue
                seen.add(uid)
                up_off = base
                if self.foe_scan_internal:
                    # up → lid 연결의 교차로 내부 차로(들) 위의 차도 본다
                    via = None
                    for c in up.getOutgoing():
                        if c.getToLane().getID() == lid:
                            via = c.getViaLaneID()
                            break
                    if via:
                        try:
                            ch = self._internal_chain(via)
                            lens = [traci.lane.getLength(x) for x in ch]
                        except traci.TraCIException:
                            ch, lens = [], []
                        for i, x in enumerate(ch):
                            if x not in seen:
                                seen.add(x)
                                out.append((x, base + sum(lens[i + 1:])))
                        up_off = base + sum(lens)
                out.append((uid, up_off))
                frontier.append((uid, up_off))
        self._upstream_cache[src] = out
        return out

    def _route_foe_features(self):
        """[38..39] 직접 계산: 내 경로가 지나갈 교차로 내부 차로와 엇갈리는 내부 차로 위·그쪽으로 다가오는 차.

        getJunctionFoes 는 ar_yg 비보호 좌회전에서 맞은편 직진 차가 와도 항상 비어 있었다 (2026-11-24).
        여기서는 SUMO 가 교차로 우선순위 계산에 쓰는 '엇갈리는 내부 차로' 정보를 그대로 쓴다.
          - 이미 그 내부 차로 위에 있는 차: 거리 0
          - 들어오는 차로에서 다가오는 차: 정지선까지 거리 (그 차의 다음 도로가 그 링크 방향이고,
            그 링크 신호가 빨강/노랑이 아닐 때만)
        반환: [가장 가까운 상대 거리 / FOE_SCALE (없으면 1), 내가 양보해야 하면 1]
        양보 = 내 링크가 우선권 없음(비보호 g 등) 이고 상대 링크는 우선권 있음.
        """
        ego = self.ego_id
        self._foe_state = None              # (상대 도착까지 초, 내가 양보해야 함, 내 정지선까지 m) — 양보 위반 감점용
        self._foe_vid = None
        self._gap_margin = None
        try:
            lane = traci.vehicle.getLaneID(ego)
            if not lane:
                return [1.0, 0.0]
            ego_dist = 0.0
            ego_len = traci.vehicle.getLength(ego)
            if lane.startswith(":"):
                chain = self._internal_chain(lane)
                node_id = lane[1:].rsplit("_", 2)[0]
                ego_prio = True if self._ego_link_prio is None else self._ego_link_prio
                # 교차로 안: 남은 내부 차로 길이 + 차 길이 만큼 더 가야 빠져나감
                clear_d = (sum(traci.lane.getLength(x) for x in chain)
                           - traci.vehicle.getLanePosition(ego) + ego_len)
            else:
                dist = traci.lane.getLength(lane) - traci.vehicle.getLanePosition(ego)
                ego_dist = dist
                if dist > self.JUNCTION_LOOK:
                    self._ego_link_prio = None
                    self._ego_link_info = None
                    self._ego_link_chain = ()
                    return [1.0, 0.0]
                route = traci.vehicle.getRoute(ego)
                ri = traci.vehicle.getRouteIndex(ego)
                if ri + 1 >= len(route):
                    if self.link_cache_clear:
                        self._clear_link_cache()
                    return [1.0, 0.0]
                if self.queue_yield_mask:
                    ld = traci.vehicle.getLeader(ego, dist)
                    if ld and ld[0] and ld[1] < dist - 1.0:
                        return [1.0, 0.0]       # Q. 정지선 앞에 줄 선 차가 있음 → 아직 내 차례 아님
                link = None
                for lk in traci.lane.getLinks(lane):
                    if lk[4] and lk[0].rsplit("_", 1)[0] == route[ri + 1]:
                        link = lk
                        break
                if link is None:
                    if self.link_cache_clear:
                        self._clear_link_cache()
                    return [1.0, 0.0]       # 이 차로로는 경로 방향으로 못 감 (차로 관측이 따로 알려줌)
                chain = self._internal_chain(link[4])
                node_id = link[4][1:].rsplit("_", 2)[0]
                ego_prio = self._prio(link[5])
                self._ego_link_prio = ego_prio
                # 내 링크 정보 (신호 상태 문자, 방향 l/s/r/t, 교차로 id). 교차로 안에서도 유지.
                self._ego_link_info = (link[5], link[6], node_id)
                self._ego_link_chain = tuple(self._internal_chain_full(link[4]))
                clear_d = dist + sum(traci.lane.getLength(x) for x in chain) + ego_len
            origin = self._internal_origin(node_id)
            foe_lanes = set()
            for il in chain:
                foe_lanes.update(traci.lane.getInternalFoes(il))
            foe_lanes -= set(chain)
            # 값: "route" = 정지선까지 거리(m) / FOE_SCALE,  "route_time" = 교차로에 닿기까지 시간(s) / foe_horizon_s
            timed = self.junction_foe_mode == "route_time"
            scale = self.foe_horizon_s if timed else self.FOE_SCALE
            best = None                      # (값, 상대 링크 우선권)
            blocked = False                  # S. 멈춘 상대가 내 진로를 막고 있음
            seen = set()
            for fl in foe_lanes:
                org = origin.get(fl)
                st = traci.lane.getLinks(org[0])[org[2]][5] if org else "G"
                for v in traci.lane.getLastStepVehicleIDs(fl):
                    if v == ego:
                        continue
                    if (self.stalled_foe_block and ego_prio and not blocked
                            and traci.vehicle.getSpeed(v) < self.stalled_foe_speed
                            and self._blocks_path(v, chain)):
                        blocked = True              # S. 내 진로 위에 멈춰 있음 → 우선이어도 양보
                        best = (0.0, True, traci.vehicle.getSpeed(v))
                        self._foe_vid = v
                    elif best is None or best[0] > 0.0:
                        best = (0.0, self._prio(st), traci.vehicle.getSpeed(v))   # 이미 엇갈리는 내부 차로 위에 있음
                        self._foe_vid = v
                if org is None or org[:2] in seen:
                    continue
                seen.add(org[:2])
                if st in ("r", "u", "s", "y"):
                    continue                 # 그 방향이 빨강/노랑이면 들어오지 않는다고 본다
                src, to_edge, _ = org
                src_edge = src.rsplit("_", 1)[0]
                lanes = self._upstream_lanes(src) if timed else [(src, 0.0)]
                for lid, off in lanes:
                    L = traci.lane.getLength(lid)
                    for v in traci.lane.getLastStepVehicleIDs(lid):
                        if v == ego:
                            continue
                        d = off + L - traci.vehicle.getLanePosition(v)
                        if timed:
                            if d > self.foe_horizon_m:
                                continue
                            val = d / max(traci.vehicle.getSpeed(v), 1.0)
                        else:
                            if d > self.FOE_SCALE:
                                continue
                            val = d
                        if blocked or (best is not None and val >= best[0]):
                            continue
                        r_v = traci.vehicle.getRoute(v)
                        k = traci.vehicle.getRouteIndex(v)
                        rest = list(r_v[k:k + 8])
                        if src_edge not in rest:
                            continue
                        j = rest.index(src_edge)
                        if j + 1 >= len(rest) or rest[j + 1] != to_edge:
                            continue         # 그 차는 다른 방향으로 감 (엇갈리지 않음)
                        best = (val, self._prio(st), traci.vehicle.getSpeed(v))
                        self._foe_vid = v
            if best is None:
                return [1.0, 0.0]
            must_yield = 1.0 if ((not ego_prio or blocked) and best[1]) else 0.0
            if must_yield and timed and self.yield_horizon_cut and best[0] >= self.foe_horizon_s:
                must_yield = 0.0
            if must_yield and timed:
                v_e = traci.vehicle.getSpeed(ego)
                a_e = self.gap_margin_accel
                t_pass = (-v_e + np.sqrt(v_e * v_e + 2.0 * a_e * max(clear_d, 0.0))) / a_e
                self._gap_margin = float(best[0]) - float(t_pass)
            if timed:
                # (상대 도착 초, 내가 양보해야 함, 내 정지선까지 m, 상대 속도, 내가 교차로를 빠져나갈 거리 m)
                self._foe_state = (float(best[0]), bool(must_yield), float(ego_dist), float(best[2]),
                                   float(clear_d))
            return [float(np.clip(best[0] / scale, 0.0, 1.0)), must_yield]
        except (traci.TraCIException, KeyError, IndexError, ValueError):
            if self.link_cache_clear:
                self._clear_link_cache()
            return [1.0, 0.0]

    def _blocks_path(self, vid, chain):
        """S. 차 vid 의 앞·가운데·뒤 중 한 점이라도 내 내부 차로 중심선에서 stalled_foe_dist m 안이면 True."""
        x, y = traci.vehicle.getPosition(vid)
        ang = np.radians(traci.vehicle.getAngle(vid))
        L = traci.vehicle.getLength(vid)
        hx, hy = np.sin(ang), np.cos(ang)
        pts = [(x, y), (x - 0.5 * L * hx, y - 0.5 * L * hy), (x - L * hx, y - L * hy)]
        lim = self.stalled_foe_dist
        for il in chain:
            shp = traci.lane.getShape(il)
            for (ax, ay), (bx, by) in zip(shp, shp[1:]):
                dx, dy = bx - ax, by - ay
                n2 = dx * dx + dy * dy
                for px, py in pts:
                    t = 0.0 if n2 <= 0 else min(max(((px - ax) * dx + (py - ay) * dy) / n2, 0.0), 1.0)
                    qx, qy = ax + t * dx - px, ay + t * dy - py
                    if qx * qx + qy * qy < lim * lim:
                        return True
        return False

    def _clear_link_cache(self):
        self._ego_link_prio = None
        self._ego_link_info = None
        self._ego_link_chain = ()

    def _stopline_cost(self, v, accel_raw):
        """정지선 감점의 크기(0~1.5). 빨간불이거나, 노란불이 켜진 순간 yellow_stop_decel 이하로 멈출 수 있었으면
        정지선을 서 있는 앞차처럼 본다: 필요 감속이 기준을 넘는 만큼 + 정지선 10 m 안에서 굴러가는 것."""
        tl = self._next_tls()
        self._ctx_sig = (False, None)
        if tl is None or tl[2] > 80.0:
            self._yellow_latch = None
            return 0.0
        tid, idx, d, st = tl
        c = st.lower()
        key = (tid, idx)
        if c in ("r", "u"):
            must = True
        elif c == "y":
            latch = getattr(self, "_yellow_latch", None)
            if latch is None or latch[0] != key:
                need = v * v / (2.0 * max(d, 0.1))
                latch = (key, need <= float(self.rw.get("yellow_stop_decel", 4.0)))
                self._yellow_latch = latch
            must = latch[1]
        else:
            self._yellow_latch = None
            must = False
        self._ctx_sig = (bool(must), float(d))
        if not must:
            return 0.0
        cost = 0.0
        comf = float(self.rw.get("stopline_comfort_decel", 2.5))
        need = v * v / (2.0 * max(d, 0.1))
        if need > comf:
            cost += min((need - comf) / comf, 1.0)
        if self.rw.get("ctx_reward", False):
            # 상황별 보상 모드: 정지선 쪽은 제동 여유(필요 감속 > 편안한 감속)만 본다. 빨간불이라는 이유로
            # 다가가기·가속을 막던 항목(10 m 안 굴러감, 50 m 안 가속, 15 m 안 기어 나가기)은 _ctx_reward 의
            # 정차 후 미세 전진 감점으로 대신한다.
            return cost
        if d < 10.0 and v > 0.3 and accel_raw > -0.2:
            cost += 0.5
        # 멈춰야 하는 신호를 향해 가속하면 감점 (2026-11-24 보완). 예전엔 정지선 10 m 밖에서의 가속은
        # 필요 감속이 작아 감점이 없어서, 앞차가 빠지자 노란불 34 m 앞에서 1.7 → 8.3 m/s 로 가속해 넘었다.
        acc_rng = float(self.rw.get("stopline_accel_range", 0.0))
        # 줄 선 앞차 판정: queue_gap_far 면 150 m(far_lead_range)까지 본다 (2026-11-26). 예전엔 50 m 까지만 봐서,
        # 줄 끝이 50 m 밖이면 줄 끝까지 다가가는 것도 '빨간불 쪽 가속'으로 감점돼 줄 간격 감점과 서로 반대로 당겼다.
        _far = bool(self.rw.get("queue_gap_far", False))
        if acc_rng > 0.0 and d < acc_rng and accel_raw > 0.0:
            # 정지선 앞에 줄 선 차가 있으면(앞차가 정지선보다 가까움) 그 차를 따라 붙는 건 정상 → 감점 안 함
            lead = self._far_leader() if _far else self._neighbor(0, leader=True)
            if lead is None or lead[0] >= d:
                cost += accel_raw
        # 기어 나가기 감점 (2026-11-26): 멈춰야 하는 정지선 creep_range m 안에서 가속 명령(> 0)을 내면 고정 감점.
        # 예전 감점은 0.3 m/s 넘게 굴러갈 때 위주라, 정지선 바로 앞에 서 있다가 조금씩 밀고 나가는 시작을 거의 못 막았다
        # (SAFE13+B+Y 기본 200판 신호위반 20건이 전부 0.6~3 m/s 기어 나가기). 줄 선 앞차를 따라가는 건 제외.
        creep = float(self.rw.get("stopline_creep_penalty", 0.0))
        if creep > 0.0 and d < float(self.rw.get("stopline_creep_range", 15.0)) and accel_raw > 0.0:
            lead = self._far_leader() if _far else self._neighbor(0, leader=True)
            if lead is None or lead[0] >= d:
                cost += creep
        return cost

    def _signal_queue_cost(self, speed):
        """State cost only: never modifies the policy action or SUMO controls."""
        self._signal_queue_last = None
        weight = float(self.rw.get("signal_queue_weight", 0.0))
        if weight <= 0 or traci.vehicle.getRoadID(self.ego_id).startswith(":"):
            return 0.0
        tls = self._next_tls()
        if tls is None or tls[2] > 200 or tls[3].lower() not in ("r", "u"):
            return 0.0
        lead = self._far_leader()
        if lead is None or lead[1] >= 3 or lead[0] >= tls[2] - 1:
            return 0.0
        gap, lead_speed = lead
        lo = float(self.rw.get("queue_gap_min", 5.0))
        hi = float(self.rw.get("queue_gap_free", 8.0))
        far = min(max(gap - hi, 0.0) / 40.0, 1.0)
        near = min(max(lo - gap, 0.0) / max(lo, 0.1), 1.0)
        settle = min(abs(speed - lead_speed) / 3.0, 1.0) if gap <= hi else 0.0
        cost = weight * min(far + near + settle, 1.0)
        self._signal_queue_last = dict(gap=gap, far=far, near=near, settle=settle, cost=cost)
        return cost

    def _safe_signal_queue(self, leader, speed, accel_raw):
        """Optional reward exception, never an action filter.

Only normal queues outside junctions qualify. Existing TTC, obstacle, close-gap,
stopline and collision costs remain active. Include one more action interval in
the braking margin so fast approaches do not receive this exception.
"""
        if not self.rw.get("signal_queue_exempt_conflicts", False):
            return False
        queue = self._signal_queue_last
        if queue is None or leader is None or abs(leader[0] - queue["gap"]) > 0.5:
            return False
        vid = self._far_lead_id
        if not vid or traci.vehicle.getRoadID(vid).startswith(":"):
            return False
        gap = leader[0]
        minimum = float(self.rw.get("queue_gap_min", 5.0))
        if gap < minimum:
            return False
        accel = accel_raw * (self.max_accel if accel_raw >= 0 else self.max_decel)
        next_speed = max(speed + accel * self.step_length, 0.0)
        remaining = gap - minimum - next_speed * self.step_length
        if remaining <= 0:
            return next_speed <= 0.05
        need = next_speed * next_speed / (2 * remaining)
        return need <= float(self.rw.get("stopline_comfort_decel", 2.0))

    def _ctx_reward(self, v, accel_raw):
        """상황별 보상 (2026-11-26, REWARD ctx_reward). 상황을 먼저 나누고 그 상황에 맞는 항목만 감점한다.
        상황 판정은 관측에 있는 정보(앞차 150 m 거리·속도, 신호·정지선 거리, 양보 필요, 내 링크 종류)로만 한다.
          - 양보: 지금 양보해야 하거나 우선권 없는 링크(비보호 g, 합류 m, 정지 s, 점멸 o)이거나 교차로 안 -> 감점 없음
          - 목표 정차 위치: 정지선 앞에 서 있는 앞차가 있으면 그 앞차(대기열), 없고 멈춰야 하는 신호면 정지선
          - 정차: 목표에서 너무 멀면(> queue_gap_free) 멀수록 감점, 줄에서 너무 가까우면(< queue_gap_min) 감점,
                  범위 안이면 정차 유지 상태로 기억
          - 정차 유지 중 미세 전진: 목표가 그대로인데(정지선: 아직 멈춰야 함 / 줄: 앞차 아직 정지이고 간격 4 m 미만) 가속 -> 감점
          - 재출발: 정차 유지 중 목표가 풀렸는데(초록불, 앞차 출발) 계속 서 있으면 감점
        반환: (감점, 상황 이름)."""
        rw = self.rw
        w_gap = float(rw.get("queue_gap_penalty", 0.2))
        w_creep = float(rw.get("ctx_creep_penalty", 1.0))
        w_restart = float(rw.get("ctx_restart_penalty", 0.1))
        lo = float(rw.get("queue_gap_min", 2.5))
        hi = float(rw.get("queue_gap_free", 8.0))
        scale = float(rw.get("queue_gap_scale", 40.0))
        must, d = getattr(self, "_ctx_sig", (False, None))
        li, fs = self._ego_link_info, self._foe_state
        try:
            in_j = traci.vehicle.getRoadID(self.ego_id).startswith(":")
        except traci.TraCIException:
            in_j = False
        hold = getattr(self, "_ctx_hold", None)
        stopped = v < 0.3
        # ---- 정지선 준수 검사 (2026-11-27 보완): 양보 예외와 무관하게 먼저 한다 ----
        # 빨간불(또는 멈춰야 하는 노란불)일 때도 '양보 필요' 가 1 이 되는 경우가 있어, 예전에는 양보 예외에 걸려 정지선 보호가 통째로 빠졌다.
        #  (a) 정지선 ctx_line_zone m 안에서 조금이라도 움직이는데(0.05 m/s 초과), 지금 낸 감속으로는 정지선 0.3 m 전에 못 선다
        #      -> 감점. 예전엔 0.3 m/s 이하나 '브레이크 명령(-0.2 이하)' 이면 면제라, 아주 느리게 넘거나 약하게 밟으며 넘는 걸 못 막았다.
        #  (b) 정지선 앞에 선 뒤(정차 유지) 아직 멈춰야 하는데 가속 -> 미세 전진 감점.
        pre_cost, pre_ctx = 0.0, None
        line_hold = False
        if must and d is not None and not in_j:
            zone = float(rw.get("ctx_line_zone", 3.0))
            if d < zone and v > 0.05:
                dec = -accel_raw * self.max_decel if accel_raw < 0.0 else 0.0
                stop_d = v * v / (2.0 * dec) if dec > 0.1 else float("inf")
                if stop_d > d - 0.3:
                    pre_cost += w_creep
                    pre_ctx = "line_roll"
            if pre_ctx is None and hold is not None and hold[0] == "s" and accel_raw > 0.0:
                pre_cost += w_creep
                pre_ctx = "creep"
            line_hold = stopped and d <= hi
        # 양보 예외는 '신호로 멈춰야 하는' 상황이 아닐 때만 (2026-11-27 보완). 빨간불에서도 교차 방향 차 때문에 [39] 양보 = 1 이
        # 떠서, 빨간불 대기줄(정문 출구·ar_yj_f0b 등)이 통째로 양보 예외가 돼 줄 간격 학습 신호가 없었다
        # (TEST31CTX 출발 구간: 8 m 넘게 정차한 시간의 약 45% 가 이 경우).
        # 지퍼 합류(Z, 2026-11-27 추가): 차로 감소 지점에서 번갈아 들어가야 하는 링크. 예전엔 목록에 없어서 지퍼 합류 직전에도
        # 줄 간격 압박·다가가기 보상이 그대로 걸렸다 (VID39AGG 125회: 서초IC 지퍼 합류 :SC_EB_n5_0 에서 합류 차와 충돌).
        yield_ctx = (fs is not None and bool(fs[1]) and not must) or (li is not None and li[0] in ("g", "m", "s", "o"))
        if in_j or yield_ctx:
            self._ctx_hold = ("s", round(d, 1)) if line_hold else None
            self._ctx_pot = None
            return pre_cost, (pre_ctx or "yield")
        lead = self._far_leader()
        queue = lead is not None and lead[1] < 0.5 and (d is None or lead[0] < d - 1.0)
        cost, ctx = pre_cost, (pre_ctx or "free")
        # 퍼텐셜 보상 (ctx_potential, 2026-11-27): 정차 목표(서 있는 앞차 또는 멈춰야 하는 정지선)가 같은 동안
        # Phi = -min(max(목표까지 거리 - queue_gap_free, 0), 60) 의 변화량 x ctx_potential 을 매 스텝 준다.
        # 멈춘 뒤에만 신호가 있던 줄 간격 감점과 달리, 다가가는 순간마다 1 m 당 바로 보상 (물러나면 같은 만큼 감점).
        # 목표가 바뀌면(다른 앞차, 정지선 <-> 줄) 그 스텝은 보상하지 않고 기준만 새로 잡는다.
        w_pot = float(rw.get("ctx_potential", 0.0))
        slow_v = float(rw.get("ctx_slow_v", 3.0))
        # 느린 줄: 앞차가 ctx_slow_v 미만으로 서다 가다 하는 경우까지 같은 목표로 본다 (2026-11-27 보완).
        # 예전엔 '서 있는 앞차(0.5 m/s 미만)' 만 목표라, 앞차가 기어가는 순간 기준점이 지워져 그 사이 벌어진 간격이 공짜였다
        # (TEST29POT: 지퍼 합류로 서다 가다 하는 SC_EB_4 에서 앞차가 움직이는데 서 있던 시간 14.5 s -> 105 s).
        slow_lead = lead is not None and lead[1] < slow_v and (d is None or lead[0] < d - 1.0)
        if w_pot > 0.0:
            if slow_lead:
                tkey, tdist = ("q", getattr(self, "_far_lead_id", None)), lead[0]
            elif must and d is not None:
                tkey, tdist = ("s", None), d
            else:
                tkey, tdist = None, None
            prev = getattr(self, "_ctx_pot", None)
            if tkey is not None:
                phi = -min(max(tdist - hi, 0.0), 60.0)
                if prev is not None and prev[0] == tkey:
                    dphi = phi - prev[1]
                    # 다가가는 보상은 목표 앞(queue_gap_min)에 편안한 감속(stopline_comfort_decel)으로 설 수 있을 때만 (2026-11-27).
                    # TEST32 25회: 서 있는 앞차로 5.8 m/s 로 다가가며 거의 안 밟다 추돌. 물러나는 감점은 그대로.
                    comf = float(rw.get("stopline_comfort_decel", 2.5))
                    need = v * v / (2.0 * max(tdist - lo, 0.5))
                    if dphi > 0.0 and need > comf:
                        dphi = 0.0
                    cost -= w_pot * dphi                   # 다가가면 phi 증가 -> 보상 (cost 음수)
                self._ctx_pot = (tkey, phi)
            else:
                self._ctx_pot = None
        if queue or (must and d is not None):
            dist = lead[0] if queue else d
            kind = "q" if queue else "s"
            if hold is not None and hold[0] == "q" and accel_raw > 0.0 and queue and dist < 4.0:
                cost += w_creep
                ctx = "creep"
            # 정지선 바로 앞(ctx_line_zone m 안)을 멈출 생각 없이 굴러가면 감점 (2026-11-27). 상황별 보상에서 예전 '정지선 10 m 안
            # 굴러감' 항목을 뺐더니, 멀리 섰다가 정지선 쪽으로 기어 오던 차가 그대로 넘어가 신호위반했다 (SAFE28CTX 중간 평가 2건).
            # 다가가기는 막지 않도록 범위를 정지선 3 m 안으로 좁혔다.
            if stopped:
                if dist > hi:
                    cost += w_gap * min((dist - hi) / scale, 1.0)
                    ctx = "stop_far"
                    self._ctx_hold = None
                else:
                    if dist < lo and kind == "q":
                        cost += w_gap * min((lo - dist) / lo, 1.0)
                    self._ctx_hold = (kind, round(dist, 1))
                    if ctx == "free":
                        ctx = "hold"
            elif ctx == "free":
                ctx = "approach"
        else:
            if stopped and not must and lead is not None and lead[1] > 0.5 and hi < lead[0] < 60.0:
                # 앞차가 출발해 멀어지는데 계속 서 있음 (예전엔 한 번 가까이 섰던 경우에만 감점 -> 멀리 서 있다 떠나는 경우가 빠졌다)
                cost += w_restart
                ctx = "restart_wait"
            elif hold is not None and stopped and not must:
                cost += w_restart                    # 초록불로 풀렸는데 계속 서 있음
                ctx = "restart_wait"
            elif hold is not None and not stopped:
                self._ctx_hold = None
                ctx = "restart"
            elif lead is not None and v > 1.0 and lead[1] > 1.0:
                ctx = "follow"
        return cost, ctx

    def _leader_signal_features(self):
        """같은 차로 앞차(교차로 안에서 내 앞을 막는 차 포함)의 방향지시등 [좌, 우] (켜짐 1, 꺼짐·앞차 없음 0).

        사람이 앞차의 깜빡이를 보고 '저 차는 좌회전하려고 교차로에서 설 수 있다'고 미리 아는 정보.
        초록불에 좌회전 대기로 서 있는 차를 '곧 출발할 차'로 보고 브레이크를 풀다 추돌하던
        남부순환로 충돌(SAFE2 9.4만 8건 중 4건)을 겨냥 (2026-11-24). SUMO 는 회전 차가 교차로에
        다가가면 깜빡이를 켠다 (남부순환로 좌회전 차: 접근 중 68%, 교차로 안 85% 켜짐; 직진 차는 거의 안 켬).
        """
        vid = self._lead_vid
        if not vid:
            return [0.0, 0.0]
        try:
            sig = int(traci.vehicle.getSignals(vid))
        except traci.TraCIException:
            return [0.0, 0.0]
        return [1.0 if sig & 2 else 0.0, 1.0 if sig & 1 else 0.0]   # bit1 = 좌, bit0 = 우

    def _permissive_left_cost(self, v, accel_raw):
        """비보호 좌회전(내 링크 신호 g + 방향 좌회전)에서 양보할 상대가 있을 때의 추가 감점 크기 (0~1).

        - 내가 교차로를 빠져나가는 예상 시간 t_pass: 지금 속도에서 permissive_left_accel(2 m/s2)로 가속한다고 보고
          '정지선까지 + 교차로 내부 차로 + 차 길이'를 지나는 시간.
        - 양보할 상대 도착 시간 t_foe (150 m 상대 차 탐지값) < t_pass + 여유(permissive_left_margin) 이면 위험.
          그때 진입 구간(정지선 max(10 m, 2초 x 속도) 안 또는 교차로 안)에서 들어가고 있거나
          (속도 > 1 m/s 이고 세게 밟지 않음) 가속하면 감점.
        - 안전한 틈(t_foe >= t_pass + 여유)이면 감점 해제 -> 틈이 나면 가도 되고, 서서 기다리는 것도 감점 없음.
        - 회전교차로 내부·합류(신호 없는 우선/zipper 교차로라 상태가 'g'가 아님), 직진·우회전 링크에는
          적용하지 않는다 (기존 yield_penalty 만).
        - 주의: ar_rb_J 는 회전교차로 '진입 전' 신호 교차로(WBf_f -> ar_rb_f_app 비보호 좌회전)라 적용 대상이다.
        """
        self._pl_info = None
        fs, li = self._foe_state, self._ego_link_info
        if fs is None or li is None or len(fs) < 5:
            return 0.0
        state, direction, node = li
        if state != "g" or direction not in ("l", "L"):
            return 0.0
        t_foe, must_y, d_stop, v_foe, clear_d = fs[:5]
        if not must_y:
            return 0.0
        if d_stop > max(10.0, 2.0 * v):
            return 0.0
        a = float(self.rw.get("permissive_left_accel", 2.0))
        t_pass = (-v + np.sqrt(v * v + 2.0 * a * max(clear_d, 0.0))) / a
        margin = float(self.rw.get("permissive_left_margin", 1.5))
        need = t_pass + margin
        going = (v > 1.0 and accel_raw > -0.5) or accel_raw > 0.0
        self._pl_info = (round(t_foe, 1), round(t_pass, 1), round(clear_d, 1), going)
        if t_foe >= need:
            return 0.0                                   # 안전한 틈 -> 감점 해제
        if not going:
            return 0.0                                   # 서서 기다리거나 세게 밟는 중 -> 감점 없음
        return float(min(1.0, (need - t_foe) / need))

    def _avoidable_left_waiter(self, leader):
        """좌회전 대기차를 안전하게 피할 수 있는데 그 뒤에 있는지 판정. (True/False, 이유 문자열)

        모두 만족해야 True:
          1) 같은 차로 앞차가 40 m 안에 서 있음(1 m/s 미만)이고 좌측 방향지시등을 켬
          2) 내가 다음 교차로에서 직진 (내 링크 방향 s)
          3) 앞 신호가 초록(G/g)이거나 신호 없음 (빨강·노랑 대기줄이면 피해도 소용없음 -> 감점 없음)
          4) 바로 옆 차로(좌·우) 중 하나가
             - 경로로 이어지고(다음 도로로 가는 링크 있음) 경로 안내상 필요한 차로 변경 수가 늘지 않으며
             - 기존 차선변경 안전 조건(_lane_change_is_safe: 앞뒤 간격, 접근 속도)을 통과하고
             - 앞·뒤 차와의 TTC 가 3초 이상 (뒤차가 빠르게 다가오면 회피 불가)
        """
        if leader is None or leader[1] >= 1.0 or leader[0] > 40.0:
            return False, "앞차 정지 아님"
        vid = self._lead_vid
        try:
            if not vid or not (int(traci.vehicle.getSignals(vid)) & 2):
                return False, "앞차 좌측 깜빡이 없음"
            ego = self.ego_id
            edge = traci.vehicle.getRoadID(ego)
            if edge.startswith(":"):
                return False, "교차로 안"
            lane = traci.vehicle.getLaneID(ego)
            route = traci.vehicle.getRoute(ego)
            ri = traci.vehicle.getRouteIndex(ego)
            if ri + 1 >= len(route):
                return False, "경로 끝"
            nxt = route[ri + 1]
            my_link = None
            for lk in traci.lane.getLinks(lane):
                if lk[0].rsplit("_", 1)[0] == nxt:
                    my_link = lk
                    break
            if my_link is None or my_link[6] != "s":
                return False, "직진 아님"
            if not self._green_ahead():
                return False, "신호 대기"
            cur = traci.vehicle.getLaneIndex(ego)
            lanes = {ln.getIndex(): ln for ln in self._net().getEdge(edge).getLanes()}
            rl = self._route_lanes() or {}
            cur_off = abs(rl[cur][1]) if cur in rl else 0
            v = traci.vehicle.getSpeed(ego)
            reasons = []
            for d in (-1, 1):
                ln = lanes.get(cur + d)
                if ln is None:
                    continue
                tag = f"{cur + d}번 차로"
                if nxt not in {c.getTo().getID() for c in ln.getOutgoing()}:
                    reasons.append(tag + " 경로 끊김")
                    continue
                if (cur + d) in rl and abs(rl[cur + d][1]) > cur_off:
                    reasons.append(tag + " 경로 이탈 위험")
                    continue
                if not self._lane_change_is_safe(d):
                    reasons.append(tag + " 간격 부족")
                    continue
                front, rear = self._scan_lane_neighbors(d)
                if rear is not None and rear[1] > v and rear[0] / (rear[1] - v) < 3.0:
                    reasons.append(tag + " 뒤차 접근(TTC<3초)")
                    continue
                if front is not None and v > front[1] and front[0] / (v - front[1]) < 3.0:
                    reasons.append(tag + " 앞차 TTC<3초")
                    continue
                return True, tag + "으로 회피 가능"
            return False, "회피 불가: " + (", ".join(reasons) if reasons else "옆 차로 없음")
        except (traci.TraCIException, KeyError):
            return False, "조회 실패"

    def _junction_waiter_cost(self, leader, v, accel_raw):
        """교차로 안(또는 깜빡이를 켠) 회전 대기차를 향해 브레이크 없이 다가가는 정도 (0~1).

        남부순환로 2번 차로(직진+좌회전 공용) 충돌 분석 (SAFE9·SAFE10, 2026-11-25):
        좌회전 차가 교차로 안(:ar_ys_14, :ar_nh_14)에 들어가 깜빡이를 켠 채 맞은편 차를 기다리며 0 m/s 로 서 있는데,
        ego 는 그 차를 30~50 m 앞에서 보고도 가속 0 근처로 5~6 m/s 를 유지하며 '곧 비켜 주겠지' 하고 들어가 추돌.
        모두 만족하면 감점:
          - 같은 차로 앞차(교차로 안에서 내 앞을 막는 차 포함)가 1 m/s 미만으로 서 있고
          - 그 차가 교차로 안에 있거나 좌·우 방향지시등을 켬 (회전하려고 기다리는 차)
          - 거리 < zone = 교차로 안 차면 max(30 m, 2.5초 x 내 속도), 교차로 밖 깜빡이 차면 max(15 m, 1.5초 x 내 속도)
          - 내 속도 > 1 m/s 이고 브레이크를 밟지 않음 (가속 명령 > -0.3)
        크기 = 1 - 거리/zone (가까울수록 큼). 세게 밟고 있거나 이미 섰으면 0.
        """
        self._jw_info = None
        if leader is None or leader[1] >= 1.0:
            return 0.0
        gap = float(leader[0])
        vid = self._lead_vid
        try:
            in_jct = bool(vid) and traci.vehicle.getRoadID(vid).startswith(":")
            blink = bool(vid) and (int(traci.vehicle.getSignals(vid)) & 3) != 0
        except traci.TraCIException:
            return 0.0
        if not (in_jct or blink):
            return 0.0
        # 교차로 안에 선 차는 넓게(30 m / 2.5초), 교차로 밖에서 깜빡이만 켠 차는 좁게(15 m / 1.5초).
        # (검증: 대기줄에서 28 m 앞 깜빡이 차를 향해 1~2 m/s 로 줄을 당기는 것까지 감점되던 것을 뺌)
        zone = max(30.0, 2.5 * v) if in_jct else max(15.0, 1.5 * v)
        if gap >= zone:
            return 0.0
        going = v > 1.0 and accel_raw > -0.3
        cost = (1.0 - gap / zone) if going else 0.0
        self._jw_info = (round(gap, 1), round(v, 1), round(float(accel_raw), 2), in_jct, blink, round(cost, 2))
        return cost

    def _lane_end_hold_cost(self, lane_change, lane_change_applied):
        """내 차로가 곧 끝나는데(차로 감소·진입 램프 가속 차로) 안전하게 옮길 옆 차로가 있는데도 유지하는 정도 (0~1).

        SAFE10 1086 분석 (2026-11-25): R2907 진입 램프가 붙는 NB_e 0번 차로(NB_f 로 이어지지 않음)에서
        왼쪽 차로 변경이 끝까지 안전했는데도 차로 명령이 계속 '유지'라 차로 끝에 서서 오래 서 있음으로 종료.
        모두 만족하면 감점:
          - 교차로 안이 아니고, 내 차로가 앞 lane_end_zone(150 m) 안에서 끝남 (다음 도로로 이어지는 연결 없음)
          - 바로 옆 차로(좌·우) 중 하나가 내 차로보다 더 멀리 이어지고, 경로 안내상 필요한 변경 수가 늘지 않으며,
            기존 차선변경 안전 조건(_lane_change_is_safe)을 통과함
          - 이번 스텝에 그 쪽으로 차로를 실제로 바꾸지 않음
        크기 = 1 - 끝까지 거리/zone (끝에 가까울수록, 끝에 서 있으면 1). 옮길 수 없으면(간격 부족 등) 0.
        """
        self._le_info = None
        self._le_block = None
        try:
            ego = self.ego_id
            edge = traci.vehicle.getRoadID(ego)
            if edge.startswith(":"):
                return 0.0
            Z = float(self.rw.get("lane_end_zone", 150.0))
            cur = traci.vehicle.getLaneIndex(ego)
            pos = traci.vehicle.getLanePosition(ego)
            n = traci.edge.getLaneNumber(edge)

            def end_dist(idx):
                lid = f"{edge}_{idx}"
                rem = traci.lane.getLength(lid) - pos
                h0 = self.lane_end_horizon
                self.lane_end_horizon = Z
                try:
                    r = self._lane_end_lookahead(lid, rem)
                finally:
                    self.lane_end_horizon = h0
                return r * Z            # r == 1 이면 Z 이상 이어짐

            d_own = end_dist(cur)
            if d_own >= Z:
                return 0.0
            rl = self._route_lanes() or {}
            own_off = abs(rl[cur][1]) if cur in rl else 0
            why = []
            for d in (1, -1):
                j = cur + d
                if j < 0 or j >= n:
                    continue
                if end_dist(j) <= d_own + 1.0:
                    why.append(f"{j}번 차로도 끝남")
                    continue
                if j in rl and abs(rl[j][1]) > own_off:
                    why.append(f"{j}번 차로 경로 이탈 위험")
                    continue
                if not self._lane_change_is_safe(d):
                    why.append(f"{j}번 차로 간격 부족")
                    self._le_block = (d_own, Z)       # 옮겨야 하는데 틈이 없음 (ctx_merge_penalty 용)
                    continue
                self._le_block = None
                if lane_change == d and lane_change_applied:
                    self._le_info = (round(d_own, 1), f"{j}번 차로로 옮기는 중", 0.0)
                    return 0.0
                cost = 1.0 - d_own / Z
                self._le_info = (round(d_own, 1), f"{j}번 차로로 옮길 수 있는데 유지", round(cost, 2))
                return cost
            self._le_info = (round(d_own, 1), "옮길 수 없음: " + (", ".join(why) or "옆 차로 없음"), 0.0)
            return 0.0
        except (traci.TraCIException, KeyError):
            return 0.0

    def _green_ahead(self):
        """앞 신호가 초록(G/g)이거나 앞에 신호가 없으면 True (빨강·노랑 대기줄에서는 차로를 바꿔도 소용없음)."""
        tl = self._next_tls()
        if tl is None or tl[2] > 80.0:
            return True
        return tl[3] in ("G", "g")

    def _free_side_lane(self):
        """경로로 이어지는 바로 옆 차로(좌·우) 중, 앞 30 m 가 비어 있는 차로가 있으면 True."""
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return False
            route = traci.vehicle.getRoute(self.ego_id)
            ri = traci.vehicle.getRouteIndex(self.ego_id)
            nxt = route[ri + 1] if ri + 1 < len(route) else None
            cur = traci.vehicle.getLaneIndex(self.ego_id)
            lanes = {ln.getIndex(): ln for ln in self._net().getEdge(edge).getLanes()}
        except (traci.TraCIException, KeyError):
            return False
        for d in (-1, 1):
            ln = lanes.get(cur + d)
            if ln is None:
                continue
            if nxt is not None and nxt not in {c.getTo().getID() for c in ln.getOutgoing()}:
                continue                         # 그 차로로는 경로를 못 이어감
            front, _ = self._scan_lane_neighbors(d)
            if front is None or front[0] > 30.0:
                return True
        return False

    def _speed_limit(self) -> float:
        """지금 달리는 차로의 제한속도 (m/s). 교차로 안이면 직전 도로 값을 쓴다."""
        try:
            lim = float(traci.vehicle.getAllowedSpeed(self.ego_id))
            if traci.vehicle.getRoadID(self.ego_id).startswith(":") and self._last_limit:
                return self._last_limit
            self._last_limit = lim
            return lim
        except traci.TraCIException:
            return self._last_limit or self.vmax

    def _net(self):
        if self._sumonet is None:
            import re
            import sumolib
            with open(self.cfg_path, encoding="utf-8") as f:
                m = re.search(r'<net-file value="([^"]+)"', f.read())
            path = os.path.join(os.path.dirname(os.path.abspath(self.cfg_path)), m.group(1))
            self._sumonet = sumolib.net.readNet(path)
            self._construction = {e.getID() for e in self._sumonet.getEdges()
                                  if "공사" in (e.getName() or "")}
        return self._sumonet

    def _next_tls(self):
        """경로상 다음 신호 (tlsID, linkIndex, 거리 m, 상태문자) 또는 None."""
        try:
            lst = traci.vehicle.getNextTLS(self.ego_id)
        except traci.TraCIException:
            return None
        return lst[0] if lst else None

    def _signal_features(self):
        self._pij_applied = False
        if self.pij_mode != "off" and self.permissive_signal is not None:
            li = self._ego_link_info
            if li is not None and li[0] == "g":
                try:
                    road = traci.vehicle.getRoadID(self.ego_id)
                    lane = traci.vehicle.getLaneID(self.ego_id)
                except traci.TraCIException:
                    road, lane = "", ""
                if road.startswith(":"):
                    ok = True
                    if self.pij_mode == "strict":
                        # 지금 교차로가 캐시한 교차로이고, 지금 내부 차로가 캐시한 진입 링크의 내부 차로일 때만
                        ok = (road[1:].rsplit("_", 1)[0] == li[2]) and (lane in self._ego_link_chain)
                    if ok:
                        self._pij_applied = True
                        if self.pij_mode in ("on", "strict"):
                            return [self.permissive_signal, 0.0]   # 비보호 좌회전(등) 진행 중: 신호 비보호, 정지선 이미 넘음
                        base = self._signal_features_base()
                        if self.pij_mode == "sig":
                            return [self.permissive_signal, base[1]]
                        return [base[0], 0.0]                      # "dist"
        return self._signal_features_base()

    def _signal_features_base(self):
        """예전 방식 신호 관측: 경로상 다음 신호의 상태와 정지선 거리 (교차로 안이면 그 다음 신호)."""
        nxt = self._next_tls()
        if nxt is None or nxt[2] > self.signal_horizon:
            return [1.0, 1.0]
        raw = nxt[3]
        c = raw.lower()
        state = 0.0 if c in ("r", "u") else (0.5 if c == "y" else 1.0)
        if c == "y" and self.yellow_decision_obs:
            key = (nxt[0], nxt[1])
            if self._yobs_latch is None or self._yobs_latch[0] != key:
                try:
                    v = traci.vehicle.getSpeed(self.ego_id)
                except traci.TraCIException:
                    v = 0.0
                need = v * v / (2.0 * max(nxt[2], 0.1))
                self._yobs_latch = (key, need <= self.yellow_decision_decel)
            state = 0.0 if self._yobs_latch[1] else 1.0
        elif c != "y":
            self._yobs_latch = None
        if raw == "g" and self.permissive_signal is not None:
            state = self.permissive_signal       # 비보호: 갈 수 있지만 맞은편 차에 양보해야 함
        return [state, float(np.clip(nxt[2] / self.signal_horizon, 0.0, 1.0))]

    @staticmethod
    def _lateral(e, succ, ahead=40.0):
        """succ 를 ahead m 들어간 지점이 e 의 끝 방향 기준 왼쪽(+)/오른쪽(-)으로 몇 m 인지."""
        sh = e.getShape()
        (x0, y0), (x1, y1) = sh[-2][:2], sh[-1][:2]
        L = float(np.hypot(x1 - x0, y1 - y0)) or 1.0
        hx, hy = (x1 - x0) / L, (y1 - y0) / L
        pts = [p[:2] for p in succ.getShape()]
        px, py = pts[-1]
        acc = 0.0
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            d = float(np.hypot(bx - ax, by - ay))
            if d > 0 and acc + d >= ahead:
                t = (ahead - acc) / d
                px, py = ax + t * (bx - ax), ay + t * (by - ay)
                break
            acc += d
        return hx * (py - y1) - hy * (px - x1)

    def _nav_table(self, route):
        """경로의 각 지점(k: route[k] → route[k+1])이 좌/우 방향 전환인지 표로 만든다."""
        key = tuple(route)
        if self._nav_key == key:
            return self._nav_tab
        net = self._net()
        tab = {}
        for k in range(len(route) - 1):
            try:
                e, nxt = net.getEdge(route[k]), net.getEdge(route[k + 1])
            except KeyError:
                continue
            conns = [c for ln in e.getLanes() for c in ln.getOutgoing()]
            succ = {c.getTo().getID() for c in conns}
            mine = [c for c in conns if c.getTo().getID() == nxt.getID()]
            if len(succ) < 2 or not mine:
                continue                      # 갈림길이 아니면 안내할 것 없음
            d = mine[0].getDirection()
            if d in ("l", "L", "t"):
                tab[k] = "L"
            elif d in ("r", "R"):
                tab[k] = "R"
            elif nxt.getPriority() < e.getPriority():
                # 직진 방향이지만 낮은 등급 도로로 갈라져 나감(출구): 다른 갈래보다 오른쪽/왼쪽
                a = self._lateral(e, nxt)
                others = [self._lateral(e, net.getEdge(o)) for o in succ if o != nxt.getID()]
                if others and all(a < o for o in others):
                    tab[k] = "R"
                elif others and all(a > o for o in others):
                    tab[k] = "L"
        self._nav_key, self._nav_tab = key, tab
        return tab

    def _nav_features(self):
        """[다음 방향 전환까지 거리, 좌, 우, 공사 구간까지 거리] — 없거나 멀면 거리 1."""
        out = [1.0, 0.0, 0.0, 1.0]
        try:
            route = traci.vehicle.getRoute(self.ego_id)
            k0 = max(traci.vehicle.getRouteIndex(self.ego_id), 0)
            net = self._net()
            tab = self._nav_table(route)
            cur = traci.vehicle.getRoadID(self.ego_id)
            for k in sorted(t for t in tab if t >= k0):
                if route[k] == cur:           # 지금 이 edge 끝에서 방향 전환 (정지선에 서 있으면 0)
                    lane = traci.vehicle.getLaneID(self.ego_id)
                    d = max(traci.lane.getLength(lane) - traci.vehicle.getLanePosition(self.ego_id), 0.0)
                else:
                    L = net.getEdge(route[k]).getLength()
                    d = traci.vehicle.getDrivingDistance(self.ego_id, route[k], max(L - 0.1, 0.0))
                if d < 0:
                    continue                  # 이미 지난 지점 (교차로 안 등)
                horizon = self._nav_horizon_m()
                if d <= horizon:
                    out[0] = d / self.nav_norm
                    out[1 if tab[k] == "L" else 2] = 1.0
                break
            if cur in self._construction:
                out[3] = 0.0
            else:
                for e in route[k0:]:
                    if e in self._construction:
                        d = traci.vehicle.getDrivingDistance(self.ego_id, e, 0.0)
                        if 0.0 <= d <= self.nav_horizon:
                            out[3] = d / self.nav_horizon
                        break
        except Exception as e:                # noqa: BLE001  관측 계산 실패로 학습이 멈추지 않게
            if not getattr(self, "_nav_warned", False):
                print(f"[관측] 내비게이션/공사 정보 계산 실패 (중립값 사용): {e!r}")
                self._nav_warned = True
        return [float(np.clip(x, 0.0, 1.0)) for x in out]

    def _remaining_route_m(self):
        """지금 경로를 따라 목적지(경로 마지막 도로 끝)까지 남은 거리(m). 못 구하면 None."""
        try:
            route = traci.vehicle.getRoute(self.ego_id)
            last = route[-1]
            L = traci.lane.getLength(last + "_0")
            d = traci.vehicle.getDrivingDistance(self.ego_id, last, max(L - 0.1, 0.0))
        except traci.TraCIException:
            return None
        if d is None or d < 0 or d > 1e6:
            return None
        return float(d)

    REJOIN_WINDOW = 3000.0   # 원래 경로 중 이 거리(m) 안의 지점으로만 복귀 (가장 이른 지점 우선)

    def _edge_time(self, eid):
        e = self._net().getEdge(eid)
        return e.getLength() / max(e.getSpeed(), 1.0)

    def _reroute_if_missed(self):
        """경로가 이어지지 않는 차로로 갈림길 직전까지 왔으면 '원래 경로로 돌아가는 길'을 다시 찾는다.

        내비게이션의 '경로 재탐색'과 같다. 학습차가 좌회전·출구 차로로 들어가지 못한 채
        정지선/분기점 바로 앞(남은 거리 < max(5 m, 1.5초 주행 거리))까지 오면,
        지금 차로로 갈 수 있는 길들 중에서 원래 경로의 가까운 지점(앞으로 REJOIN_WINDOW m 안)으로
        돌아가는 가장 빠른 길을 고르고, 그 뒤로는 원래 경로를 그대로 따른다.
        예) 본사에서 나와 염곡 좌회전 대기줄에 못 끼어들면 → 직진 → 구룡사거리 유턴 → 양재대로 서행으로 합류.
        어느 차로로 가라거나 언제 바꾸라는 안내는 하지 않는다 (놓친 뒤에 새 길을 줄 뿐).
        """
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return False
            if self._last_reroute_edge is not None and edge != self._last_reroute_edge:
                # 그 도로를 벗어났으면 '같은 갈림길 앞 반복' 이 아니다. 우회로를 돌아 같은 곳에서
                # 또 놓치면 새로 놓친 것으로 센다 (안 그러면 같은 고리를 무한히 돌 수 있음)
                self._last_reroute_edge = None
            route = list(traci.vehicle.getRoute(self.ego_id))
            k = traci.vehicle.getRouteIndex(self.ego_id)
            if k < 0 or k + 1 >= len(route) or route[k] != edge:
                return False
            lane = traci.vehicle.getLaneID(self.ego_id)
            links = traci.lane.getLinks(lane)
            if not links:
                return False                       # 끝나는 차로(차로 수 감소): 경로를 놓친 게 아니라 옆 차로로 합류해야 하는 상황
            if any(l[0].rsplit("_", 1)[0] == route[k + 1] for l in links):
                return False                       # 지금 차로로 경로가 이어짐
            remaining = traci.lane.getLength(lane) - traci.vehicle.getLanePosition(self.ego_id)
            if remaining > max(5.0, traci.vehicle.getSpeed(self.ego_id) * 1.5):
                return False                       # 아직 차로를 바꿀 여유가 있음

            # 복귀 후보: 원래 경로에서 앞으로 REJOIN_WINDOW m 안의 지점들.
            # 그중 '가장 이른 지점'으로 돌아가는 길만 고른다. 목적지 직행이나 경로 일부를 건너뛰는
            # 지름길은 허용하지 않는다 (예전에는 '남은 시간 최소'로 골라서, 차로를 일부러 놓치면
            # 서초IC·남부순환로 루프를 건너뛰고 체육관으로 직행하는 지름길이 생겼다 = 보상 해킹).
            cands, dist = [], 0.0
            for jj in range(k + 1, len(route)):
                cands.append(jj)
                dist += self._net().getEdge(route[jj]).getLength()
                if dist > self.REJOIN_WINDOW:
                    break
            best = None
            for jj in cands:                       # 이른 지점부터
                for l in links:
                    succ = l[0].rsplit("_", 1)[0]
                    r = traci.simulation.findRoute(succ, route[jj])
                    if r.edges and (best is None or r.travelTime < best[0]):
                        best = (r.travelTime, jj, list(r.edges))
                if best is not None:
                    break                          # 이 지점으로 돌아갈 수 있으면 더 뒤는 보지 않는다
            if best is None:
                return None                        # 원래 경로로 돌아갈 길이 없음 → 그 판 종료
            _, j, path = best
            traci.vehicle.setRoute(self.ego_id, [edge] + path + route[j + 1:])
            self._nav_key = None                   # 내비 안내표 다시 계산
            # 같은 도로(같은 갈림길 앞)에서 차로를 또 바꿔 다시 재탐색한 것은 한 번 놓친 것으로 센다
            if edge == self._last_reroute_edge:
                return False
            self._last_reroute_edge = edge
            self.reroute_count += 1
            return True
        except traci.TraCIException:
            return False

    def get_detected_vehicles(self):
        """현재 관측 시야에 들어온(=감지된) 배경차들의 좌표 목록을 반환.

        데이터셋 수집용 부가 정보 — 관측 벡터에는 들어가지 않는다.
        "감지"의 정의는 관측과 동일: 관측 대상 차선(obs_offsets 범위)에서
        ego 기준 종방향 ±visibility(W) 이내에 있는 차량.

        Returns: list[dict]  각 원소 =
            {"id": 차량 id, "x": 절대 x, "y": 절대 y,
             "rel_pos": ego 기준 종방향 상대거리(m, 앞 +), "speed": m/s,
             "lane_offset": ego 차선 기준 오프셋(+왼쪽)}
        ego가 없으면 빈 리스트.
        """
        out = []
        if self.ego_id not in traci.vehicle.getIDList():
            return out
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return out
            ego_lane = traci.vehicle.getLaneIndex(self.ego_id)
            ego_pos = traci.vehicle.getLanePosition(self.ego_id)
            n_lanes = traci.edge.getLaneNumber(edge)
            for offset in self.obs_offsets:
                li = ego_lane + offset
                if li < 0 or li >= n_lanes:
                    continue
                for vid in traci.lane.getLastStepVehicleIDs(f"{edge}_{li}"):
                    if vid == self.ego_id:
                        continue
                    rel = traci.vehicle.getLanePosition(vid) - ego_pos
                    if abs(rel) > self.visibility:
                        continue
                    x, y = traci.vehicle.getPosition(vid)
                    out.append({"id": vid, "x": float(x), "y": float(y),
                                "rel_pos": float(rel),
                                "speed": float(traci.vehicle.getSpeed(vid)),
                                "lane_offset": int(offset)})
        except traci.TraCIException:
            pass
        return out

    def _get_obs(self):
        # ego가 이미 사라졌다면(충돌/도착) 0벡터 반환 — 종료 직전 마지막 관측용
        if self.ego_id not in traci.vehicle.getIDList():
            return np.zeros(self.obs_dim_single, dtype=np.float32)

        v = traci.vehicle.getSpeed(self.ego_id)
        obs = [v / self.vmax]                      # [0] 내 절대속도

        # ----- 5개 차선(좌2→좌1→현재→우1→우2)의 선행/후행 상대거리·상대속도 -----
        for offset in self.obs_offsets:
            lead = self._neighbor(offset, leader=True)
            if lead is None:
                obs += [1.0, 1.0]                  # 없음: 멀고 + 위협 없음
            else:
                gap, lv = lead
                obs += [gap / self.visibility,
                        float(np.clip((lv - v) / self.vmax, -1.0, 1.0))]

            foll = self._neighbor(offset, leader=False)
            if foll is None:
                obs += [-1.0, -1.0]                # 없음: 멀고 + 위협 없음
            else:
                gap, fv = foll
                obs += [-gap / self.visibility,    # 후행은 음수 부호
                        float(np.clip((fv - v) / self.vmax, -1.0, 1.0))]

        # ----- 5개 차선의 연결성 (같은 순서) -----
        for offset in self.obs_offsets:
            obs.append(self._lane_connectivity(offset))

        # ----- 5개 차선의 전방 차량 밀도 (같은 순서) -----
        for offset in self.obs_offsets:
            obs.append(self._lane_density(offset))

        # ----- 신호등 / 내비게이션 / 공사 구간 (mdp_config 의 관측 [31..36]) -----
        obs += self._signal_features()
        obs += self._nav_features()
        obs.append(self._speed_limit() / self.vmax)   # [37] 지금 도로 제한속도 (표지판 정보)
        obs += self._junction_foe_features()           # [38..39] 교차로에서 만날 차
        if self.leader_signals:
            obs += self._leader_signal_features()      # [40..41] 앞차 방향지시등 (좌, 우)
        if self.gap_margin_obs:
            gm = self._gap_margin                      # _junction_foe_features 가 방금 계산
            obs.append(1.0 if gm is None else float(np.clip(gm / self.gap_margin_scale, -1.0, 1.0)))   # [42]
        if self.far_lead_obs:
            fl = self._far_leader()
            obs.append(1.0 if fl is None else float(np.clip(fl[0] / self.far_lead_range, 0.0, 1.0)))
            obs.append(1.0 if fl is None else float(np.clip(fl[1] / max(self._speed_limit(), 1.0), 0.0, 1.0)))

        return np.clip(np.array(obs, dtype=np.float32), -1.0, 1.0)

    # ==================================================================
    # 1 스텝 진행: 행동 적용 → 시뮬레이션 1스텝 → 보상/종료 판정
    # ==================================================================
    def _quantize_lane_change(self, raw: float) -> int:
        """연속 policy 출력을 실제 차선 명령 {-1,0,+1}로 변환한다.

        +1 = 왼쪽, 0 = 유지, -1 = 오른쪽. SUMO 차선 index가 왼쪽으로
        갈수록 증가하는 규약과 동일해서 해석이 직관적이다.
        """
        raw = float(np.clip(raw, -1.0, 1.0))
        if raw >= self.lane_change_threshold:
            return 1
        if raw <= -self.lane_change_threshold:
            return -1
        return 0

    def _scan_lane_neighbors(self, direction: int):
        """목표 차선을 직접 스캔해 (가장 가까운 앞차 gap·속도, 뒷차 gap·속도) 반환.

        getNeighbors의 백업. getNeighbors는 SUMO 차선변경모델의 이웃 캐시를
        읽는데, "방금 스폰된 차량"은 다음 스텝까지 이 캐시에 없어서
        도로 입구 근처에서 사각지대가 생긴다. 차선의 차량 목록을 직접
        훑으면 이미 삽입된 차량은 전부 잡힌다.
        """
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            li = traci.vehicle.getLaneIndex(self.ego_id) + direction
            # 존재하지 않는 차선이면 조회하지 않는다.
            # (없는 lane id를 traci.lane.*로 조회하면 SUMO가 매번
            #  에러 로그를 찍으므로, 범위 검사를 먼저 해서 조용히 건너뛴다)
            if edge.startswith(":") or li < 0 \
                    or li >= traci.edge.getLaneNumber(edge):
                return None, None
            lane_id = f"{edge}_{li}"
            my_pos = traci.vehicle.getLanePosition(self.ego_id)
            my_len = traci.vehicle.getLength(self.ego_id)
            front, rear = None, None
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                d = traci.vehicle.getLanePosition(vid) - my_pos
                if d >= 0:
                    g = d - traci.vehicle.getLength(vid)
                    if front is None or g < front[0]:
                        front = (g, traci.vehicle.getSpeed(vid))
                else:
                    g = -d - my_len
                    if rear is None or g < rear[0]:
                        rear = (g, traci.vehicle.getSpeed(vid))
            # 목표 차로의 '앞뒤 도로'까지 본다 (2026-11-24). 같은 도로 안만 보면 합류·교차로 직후에
            # 바로 뒤 연결 구간(:NB_m1470_1 등)에서 35 m/s 로 오는 차, 교차로 직전에 앞 연결 구간에 있는
            # 차가 안 보여 차선변경 충돌 7건 중 6건이 났다.
            for lid, extra in self._lane_upstream(lane_id):
                for vid in traci.lane.getLastStepVehicleIDs(lid):
                    if vid == self.ego_id:
                        continue
                    g = my_pos - my_len + extra + traci.lane.getLength(lid) - traci.vehicle.getLanePosition(vid)
                    if rear is None or g < rear[0]:
                        rear = (g, traci.vehicle.getSpeed(vid))
            to_end = traci.lane.getLength(lane_id) - my_pos
            if to_end < self.visibility:
                for lid, extra in self._lane_downstream(lane_id):
                    for vid in traci.lane.getLastStepVehicleIDs(lid):
                        if vid == self.ego_id:
                            continue
                        g = to_end + extra + traci.vehicle.getLanePosition(vid) - traci.vehicle.getLength(vid)
                        if front is None or g < front[0]:
                            front = (g, traci.vehicle.getSpeed(vid))
            return front, rear
        except traci.TraCIException:
            return None, None

    UPSTREAM_LOOK = 80.0     # 차선변경 검사: 목표 차로 뒤쪽(이전 연결 구간·이전 도로)을 볼 거리 (m)

    def _lane_upstream(self, lane_id):
        """lane_id 로 들어오는 내부 차로(교차로·합류부 연결 구간)와 그 앞 도로 차로.
        [(차로, 그 차로 끝 → lane_id 시작까지 거리)]. 처음 한 번만 계산해 둔다."""
        cache = self.__dict__.setdefault("_lane_up_cache", {})
        if lane_id in cache:
            return cache[lane_id]
        out = []
        try:
            lane = self._net().getLane(lane_id)
            for inl in lane.getIncoming():
                for c in inl.getOutgoing():
                    if c.getToLane().getID() != lane_id:
                        continue
                    via = c.getViaLaneID()
                    chain = self._internal_chain(via) if via else []
                    lens = [traci.lane.getLength(x) for x in chain]
                    for i, x in enumerate(chain):
                        out.append((x, sum(lens[i + 1:])))
                    if sum(lens) < self.UPSTREAM_LOOK:
                        out.append((inl.getID(), sum(lens)))
        except (KeyError, AttributeError, traci.TraCIException):
            pass
        cache[lane_id] = out
        return out

    def _lane_downstream(self, lane_id):
        """lane_id 끝에서 이어지는 내부 차로(교차로 연결 구간). [(차로, lane_id 끝 → 그 차로 시작까지 거리)]."""
        cache = self.__dict__.setdefault("_lane_down_cache", {})
        if lane_id in cache:
            return cache[lane_id]
        out = []
        try:
            for lk in traci.lane.getLinks(lane_id):
                if not lk[4]:
                    continue
                acc = 0.0
                for x in self._internal_chain(lk[4]):
                    out.append((x, acc))
                    acc += traci.lane.getLength(x)
        except traci.TraCIException:
            pass
        cache[lane_id] = out
        return out

    def _lane_change_is_safe(self, direction: int) -> bool:
        """목표 차선으로 지금 이동해도 측면충돌이 나지 않는지 검사한다.

        3중 안전 규칙:
          (0) 입구 보호구간: ego가 도로 시작 spawn_protect(m) 이내면 변경 금지.
              flow가 입구(0m)에서 차를 새로 만드는데, 생성 "직전" 차량은
              어떤 API로도 미리 볼 수 없어 원천적으로 사각지대이기 때문.
          (1) 앞: gap > min_front_gap + front_tau × 내속도
              + 접근 여유: gap > (내속도 − 앞차속도)⁺ × 1s   (내가 더 빠를 때)
          (2) 뒤: gap > min_rear_gap + rear_tau × 뒤차속도
              + 접근 여유: gap > (뒤차속도 − 내속도)⁺ × 1s   (뒤가 더 빠를 때)
        이웃은 getNeighbors 결과와 차선 직접 스캔 결과 중 "더 가까운 쪽"을
        쓴다 (스폰 직후 차량 사각지대 보완).
        """
        # "도로 시작" = ego 가 출발한 지점. 예전 직선 도로에선 차로 위치와 같았지만,
        # 도로망에서는 edge 마다 차로 위치가 0부터 다시 시작해서 모든 edge 의 앞 60 m 가
        # 변경 금지가 돼 버린다 (짧은 edge 에선 차로 변경이 아예 불가능). 주행 거리로 판단한다.
        # 출발한 도로(정문 진입로) 위에서만 적용한다. 본선에 나온 뒤에는 바로 바꿀 수 있어야
        # 50 m 남짓한 구간에서 좌회전 대기줄로 끼어드는 상황이 성립한다.
        # 코스 중간 출발(start_edge 가 있음)은 도로 한가운데 넣는 것이라 입구 보호가 필요 없다.
        # (적용하면 도로 끝 근처에서 출발했을 때 60 m 를 못 채워 차로를 영영 못 바꾼다)
        if (self.start_edge is None
                and traci.vehicle.getRouteIndex(self.ego_id) <= 0
                and traci.vehicle.getDistance(self.ego_id) < self.lc_spawn_protect):
            return False

        v = traci.vehicle.getSpeed(self.ego_id)
        scan_front, scan_rear = self._scan_lane_neighbors(direction)

        def closer(a, b):
            if a is None:
                return b
            if b is None:
                return a
            return a if a[0] <= b[0] else b

        lead = closer(self._neighbor(direction, leader=True), scan_front)
        if lead is not None:
            gap, lv = lead
            if gap <= self.lc_front_min + self.lc_front_tau * v:
                return False
            if gap <= max(v - lv, 0.0) * self.lc_horizon:  # 시간여유 내 접촉 위험
                return False

        foll = closer(self._neighbor(direction, leader=False), scan_rear)
        if foll is not None:
            gap, fv = foll
            if gap <= self.lc_rear_min + self.lc_rear_tau * fv:
                return False
            if gap <= max(fv - v, 0.0) * self.lc_horizon:  # 시간여유 내 접촉 위험
                return False
        return True

    def _carry_lane_change(self, cmd: int) -> int:
        """교차로 안에서 낸 차로 변경 명령을 기억했다가, 빠져나온 뒤 '유지' 명령이면 그 명령으로 바꿔 돌려준다.

        - 교차로 안 + 변경 명령: 기억 (이 명령은 교차로 안이라 실행 안 됨. 불가능한 요청 감점도 안 줌)
        - 교차로 밖: 기억한 명령이 lane_change_carry_s 안이고, 지금 명령이 '유지'이며, 지금 차로가 150 m 안에서 끝나면
          기억한 방향으로 실행 시도 (차로가 계속 이어지면 기억을 버림)
          (안전 검사에 걸리면 기억은 유지 → 다음 스텝 다시 시도). 반대 방향 명령이 오거나 시간이 지나면 버림.
        """
        try:
            road = traci.vehicle.getRoadID(self.ego_id)
        except traci.TraCIException:
            return cmd
        now = self.step_count
        if road.startswith(":"):
            if cmd != 0:
                self._pending_lc = (cmd, now)
            return cmd
        p = self._pending_lc
        if p is None:
            return cmd
        if (now - p[1]) * self.step_length > self.lc_carry_s or cmd == -p[0]:
            self._pending_lc = None
            return cmd
        if cmd != 0:
            return cmd
        # 지금 차로가 앞 150 m 안에서 끝날 때(합류 가속 차로·차로 감소)만 이어받는다.
        # 보통 교차로를 지나 차로가 계속 이어지면 버림 → 교차로 안에서 흔들린 명령이 엉뚱한 차로 변경으로 이어지지 않게.
        try:
            lid = traci.vehicle.getLaneID(self.ego_id)
            rem = traci.lane.getLength(lid) - traci.vehicle.getLanePosition(self.ego_id)
            h0 = self.lane_end_horizon
            self.lane_end_horizon = 150.0
            try:
                ends = self._lane_end_lookahead(lid, rem) < 1.0
            finally:
                self.lane_end_horizon = h0
        except traci.TraCIException:
            ends = False
        if not ends:
            self._pending_lc = None
            return cmd
        return p[0]

    def _apply_lane_change(self, lane_change: int) -> bool:
        """가능하고 "안전한" 경우에만 차선변경을 적용하고 성공 여부를 반환.

        안전 게이트(_lane_change_is_safe)를 통과하지 못한 명령은 실행하지
        않고 False를 반환한다 → step()에서 "불가능한 명령"과 동일하게
        invalid_action_penalty가 부과된다. 즉 에이전트 입장에서
        "위험한 변경 = 실행도 안 되고 감점"이라, 측면충돌로 즉사하며
        배우는 대신 안전한 타이밍을 고르는 쪽으로 학습이 유도된다.
        """
        self._lc_impossible = False          # 그 방향에 차로가 아예 없음 (교차로 안 포함)
        if self.ego_id not in traci.vehicle.getIDList():
            return False
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                # 교차로 안 변경 명령: 이어받기(lane_change_carry_s)가 켜져 있으면 기억해 두므로 '불가능한 요청'이 아님
                self._lc_impossible = lane_change != 0 and self.lc_carry_s <= 0.0
                return False
            current = traci.vehicle.getLaneIndex(self.ego_id)
            n_lanes = traci.edge.getLaneNumber(edge)
            target = current + lane_change
            if target < 0 or target >= n_lanes:
                self._lc_impossible = True
                return False
            # ★ 측면충돌 방지: 실제 이동 명령(0이 아닌 변경)에만 안전 검사
            if (lane_change != 0 and self.lc_safety_enabled
                    and not self._lane_change_is_safe(lane_change)):
                return False                 # 차로는 있지만 지금은 틈이 없음 (끼어들기 실패)
            traci.vehicle.changeLane(
                self.ego_id, target, self.lane_change_duration)
            return True
        except traci.TraCIException:
            return False

    def get_privileged_state(self) -> dict:
        """POMDP 데이터셋용 simulator-side state.

        agent가 실제로 받는 입력은 _get_obs()의 부분관측이고, 이 메서드는
        분석/지도신호용 privileged state만 수집한다. 정책 입력으로 쓰지 않는다.
        """
        if self.ego_id not in traci.vehicle.getIDList():
            return {"ego_alive": False}
        try:
            x, y = traci.vehicle.getPosition(self.ego_id)
            vehicles = []
            for vid in traci.vehicle.getIDList():
                try:
                    vx, vy = traci.vehicle.getPosition(vid)
                    vehicles.append({
                        "id": vid,
                        "x": float(vx),
                        "y": float(vy),
                        "speed": float(traci.vehicle.getSpeed(vid)),
                        "lane": int(traci.vehicle.getLaneIndex(vid)),
                        "lane_id": traci.vehicle.getLaneID(vid),
                        "road_id": traci.vehicle.getRoadID(vid),
                        "lane_pos": float(traci.vehicle.getLanePosition(vid)),
                    })
                except traci.TraCIException:
                    continue
            return {
                "ego_alive": True,
                "ego_x": float(x),
                "ego_y": float(y),
                "ego_speed": float(traci.vehicle.getSpeed(self.ego_id)),
                "ego_lane": int(traci.vehicle.getLaneIndex(self.ego_id)),
                "ego_lane_id": traci.vehicle.getLaneID(self.ego_id),
                "ego_road_id": traci.vehicle.getRoadID(self.ego_id),
                "ego_lane_pos": float(traci.vehicle.getLanePosition(self.ego_id)),
                "vehicles": vehicles,
            }
        except traci.TraCIException:
            return {"ego_alive": False}

    def step(self, action):
        # action = [가감속 raw, 차선변경 raw]
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size < self.act_dim:
            raise ValueError(
                f"action은 {self.act_dim}차원이어야 합니다. 받은 shape={np.asarray(action).shape}")

        accel_raw = float(np.clip(action_arr[self.accel_index], -1.0, 1.0))
        lane_raw = float(np.clip(action_arr[self.lane_change_index], -1.0, 1.0))
        lane_change = self._quantize_lane_change(lane_raw)
        lane_change_applied = False
        lane_before = None

        # 데이터 기록: 행동을 적용하기 "전" 시점 t 의 상태·상황 값을 먼저 잡아 둔다
        rec = self.recorder if (self.recorder is not None and self.recording) else None
        if rec is not None:
            rec.before_step(self._last_obs)

        # 빨간불 위반 판정용: 행동 전 시점의 다음 신호와 속도
        tls_before = self._next_tls() if self.ego_id in traci.vehicle.getIDList() else None
        v_before = traci.vehicle.getSpeed(self.ego_id) if tls_before is not None else 0.0

        # ----- (1) 종방향 + 횡방향 행동 적용 -----
        if self.ego_id in traci.vehicle.getIDList():
            v = traci.vehicle.getSpeed(self.ego_id)
            lane_before = int(traci.vehicle.getLaneIndex(self.ego_id))

            accel_cmd = (accel_raw * self.max_accel
                         if accel_raw >= 0.0 else accel_raw * self.max_decel)
            dv = accel_cmd * self.step_length
            traci.vehicle.setSpeed(
                self.ego_id, float(np.clip(v + dv, 0.0, self.vmax)))

            if self.lc_carry_s > 0.0:
                lane_change = self._carry_lane_change(lane_change)
            lane_change_applied = self._apply_lane_change(lane_change)
            if (self._pending_lc is not None and lane_change == self._pending_lc[0]
                    and lane_change_applied):
                self._pending_lc = None     # 이어받은(또는 같은) 명령이 실행됨

        # ----- (2) 시뮬레이션 1스텝 -----
        traci.simulationStep()
        self.step_count += 1

        # ----- (3) 이벤트/관측 -----
        collided = self.ego_id in traci.simulation.getCollidingVehiclesIDList()
        # 충돌로 제거된 차도 SUMO 에선 '도착 목록'에 들어가므로 충돌이면 도착이 아니다
        arrived = (self.ego_id in traci.simulation.getArrivedIDList()) and not collided
        alive = self.ego_id in traci.vehicle.getIDList()
        # SUMO 가 오래 못 움직인 ego 를 순간이동시켰는지 (정상 주행이 아니므로 '정지 시간 초과'로 끝낸다)
        teleported = self.ego_id in traci.simulation.getStartingTeleportIDList()
        lane_after = int(traci.vehicle.getLaneIndex(self.ego_id)) if alive else None

        # 좌회전·출구 차로를 놓쳤으면 원래 경로로 돌아가는 길을 다시 찾는다.
        # 돌아갈 길이 아예 없으면(None) 경로 이탈 감점 후 그 판을 끝낸다.
        rr = self._reroute_if_missed() if (alive and not collided) else False
        rerouted = rr is True
        no_way_back = rr is None

        # 빨간불에 정지선을 넘었는지.
        #  - 스텝 전에 빨간불이던 신호가 이번 스텝 안에 넘을 수 있는 거리(속도×0.5초 + 여유)에 있었고
        #  - 스텝 후 그 신호(같은 신호기 id)가 앞에 더 이상 없으면 넘은 것.
        # 신호의 '연결 번호'는 차로를 바꾸거나 경로를 다시 찾으면 바뀌므로 비교하지 않는다
        # (예전에는 그것 때문에 정지선 수십 m 앞에서 차로만 바꿔도 위반으로 잘못 판정됐다).
        red_run = False
        if (alive and not collided and tls_before is not None
                and tls_before[3].lower() in ("r", "u")):
            reach = v_before * self.step_length + 0.5 * self.max_accel * self.step_length ** 2 + 1.0
            if tls_before[2] <= reach:
                try:
                    now = traci.vehicle.getNextTLS(self.ego_id)
                except traci.TraCIException:
                    now = [tls_before]
                red_run = not any(t[0] == tls_before[0] and t[2] <= tls_before[2] + 0.5 for t in now)

        obs = self._get_obs()
        terminated, truncated = False, False
        ego_speed, ego_gap = None, None

        # ----- (4) 기존 보상 유지 -----
        if collided:
            reward = -float(self.rw["collision_penalty"])
            terminated = True
        elif arrived:
            reward = float(self.rw["arrival_bonus"])
            terminated = True
        elif not alive:
            reward = 0.0
            terminated = True
        else:
            ego_speed = traci.vehicle.getSpeed(self.ego_id)
            same_lane_leader = self._neighbor(0, leader=True)
            if same_lane_leader is not None:
                ego_gap = same_lane_leader[0]
            # 속도 보상은 '지금 도로 제한속도' 대비 (제한속도 이상은 더 주지 않음),
            # 제한속도를 overspeed_tolerance(10%) 넘게 초과하면 초과분에 비례해 감점.
            v_lim = max(self._speed_limit(), 1.0)
            sw = float(self.rw["speed_weight"])
            rem = self._remaining_route_m() if self.rw.get("progress_reward") else None
            if rem is not None and self.rw.get("progress_best_only", False):
                # 신기록 방식: 이번 판에서 목적지에 가장 가까이 갔던 거리(_best_rem)보다 더 가까워진 만큼만 보상.
                # 회전을 놓쳐 우회로가 붙으면 남은 거리가 늘어나므로, 원래 경로로 돌아와 예전 최고 지점을
                # 넘어설 때까지 진행 보상이 0 이다 → 같은 곳을 빙빙 돌며 진행 보상을 다시 모으는 꼼수가 없다.
                if self._best_rem is None:
                    reward = sw * min(ego_speed, v_lim) / v_lim
                else:
                    prog = max(self._best_rem - rem, 0.0) / (v_lim * self.step_length)
                    reward = sw * min(prog, 1.0)
                self._best_rem = rem if self._best_rem is None else min(self._best_rem, rem)
            elif rem is not None and self._prev_rem is not None:
                # 진행 거리 보상: 이번 스텝에 목적지까지 남은 경로 거리가 줄어든 만큼 (제한속도로 가면 +sw).
                # 길을 놓쳐 우회로가 붙으면 남은 거리가 늘어 음수 (하한 −sw × progress_floor).
                prog = (self._prev_rem - rem) / (v_lim * self.step_length)
                reward = sw * float(np.clip(prog, -float(self.rw.get("progress_floor", 20.0)), 1.0))
            else:
                reward = sw * min(ego_speed, v_lim) / v_lim
            self._prev_rem = rem
            over = ego_speed - v_lim * (1.0 + float(self.rw.get("overspeed_tolerance", 0.1)))
            if over > 0.0:
                reward -= float(self.rw.get("overspeed_penalty", 0.0)) * over / v_lim
            # 줄 간격 범위 (2026-11-26, queue_gap_min > 0 일 때): 나와 앞차가 둘 다 서 있는 대기줄에서는
            # '너무 가까움' 기준을 close_gap_threshold(6 m) 대신 queue_gap_min 으로 본다. 예전엔 줄에서 2~6 m 뒤에
            # 정상적으로 서 있어도 매 스텝 감점이라, 줄 간격 감점(8 m 넘으면)과 합쳐 감점 없는 구간이 6~8 m 뿐이었다.
            q_min = float(self.rw.get("queue_gap_min", 0.0))
            in_queue = (q_min > 0.0 and ego_speed < 1.0 and same_lane_leader is not None
                        and same_lane_leader[1] < 0.5)
            close_thr = q_min if in_queue else float(self.rw["close_gap_threshold"])
            if ego_gap is not None and ego_gap < close_thr:
                reward -= float(self.rw["close_gap_penalty"])
                if in_queue:
                    # 줄에서 queue_gap_min 보다 바짝 붙으면 더 감점 (출발할 때 앞차를 받던 0.3 m 사례)
                    reward -= float(self.rw.get("queue_gap_penalty", 0.0)) * min((q_min - ego_gap) / q_min, 1.0)
            # 앞차와 충돌까지 남은 시간(TTC)이 짧으면 감점: 이대로 가면 몇 초 뒤 닿는가
            tp = float(self.rw.get("ttc_penalty", 0.0))
            if tp > 0.0 and same_lane_leader is not None:
                lead_gap, lead_v = same_lane_leader
                closing = ego_speed - lead_v
                thr = float(self.rw.get("ttc_threshold", 2.0))
                if closing > 0.1:
                    ttc = lead_gap / closing
                    if ttc < thr:
                        reward -= tp * (thr - ttc) / thr
            # 가까이 서 있는 앞차(교차로 안에 선 차 포함)를 향해 가속하면 감점
            cap = float(self.rw.get("close_accel_penalty", 0.0))
            if cap > 0.0 and same_lane_leader is not None and accel_raw > 0.0:
                lead_gap, lead_v = same_lane_leader
                if lead_v < 1.0 and lead_gap < 2.0 + 1.5 * ego_speed:
                    reward -= cap * accel_raw
            # 서 있는 앞차 감점: 앞차 2 m 뒤에 멈추는 데 필요한 감속이 기준을 넘으면 (브레이크를 늦게·약하게)
            obp = float(self.rw.get("obstacle_penalty", 0.0))
            if obp > 0.0 and same_lane_leader is not None and same_lane_leader[1] < 1.0:
                comf = float(self.rw.get("stopline_comfort_decel", 2.5))
                d_ob = max(same_lane_leader[0] - 2.0, 0.1)
                need = ego_speed * ego_speed / (2.0 * d_ob)
                if need > comf:
                    reward -= obp * min((need - comf) / comf, 1.0)
            # 정지선 감점: 빨간불 / 멈출 수 있을 때 켜진 노란불의 정지선을 서 있는 앞차처럼 본다
            slp = float(self.rw.get("stopline_penalty", 0.0))
            if slp > 0.0:
                reward -= slp * self._stopline_cost(ego_speed, accel_raw)
            # 양보 위반: 양보해야 할 상대가 곧 오는데 교차로로 들어가고 있으면 감점
            yp = float(self.rw.get("yield_penalty", 0.0))
            fs = self._foe_state
            if yp > 0.0 and fs is not None:
                yt = float(self.rw.get("yield_time", 3.0))
                t_foe, must_y, d_stop, v_foe = fs[:4]
                strong = bool(self.rw.get("yield_strict", False))
                zone = max(10.0, 2.0 * ego_speed) if strong else max(5.0, 1.5 * ego_speed)
                if must_y and t_foe < yt and d_stop <= zone:
                    # (강화) 이미 세게 밟고 있으면(−0.5 이하) 양보 중이므로 감점 안 함
                    if ego_speed > 1.0 and not (strong and accel_raw <= -0.5):
                        reward -= yp * (1.0 - t_foe / yt)
                    # (강화) 서 있다가 양보할 상대가 오는데 출발(가속)하면 감점 — 회전교차로 정지 출발 충돌
                    if strong and accel_raw > 0.0:
                        reward -= yp * accel_raw
                # (강화) 우선권이 있어도, 움직이는 상대가 1.5초 안에 같은 곳에 오면 감점 — 공사 구간 합류 끼어들기
                ct = float(self.rw.get("conflict_time", 0.0))
                # 교차로에 아직 안 들어온(t>0) 상대만 — 이미 교차로 안에서 같은 출구로 합류해 앞서가는 차는 위협이 아님
                if strong and ct > 0.0 and not must_y and v_foe > 2.0 and 0.0 < t_foe < ct                         and d_stop <= zone and ego_speed > 1.0 and accel_raw > -0.5:
                    reward -= yp * (1.0 - t_foe / ct)

            # 비보호 좌회전 추가 양보 감점 (기존 yield_penalty 는 그대로, 비보호 좌회전 g + l 에만 추가)
            plp = float(self.rw.get("permissive_left_penalty", 0.0))
            if plp > 0.0:
                c_pl = self._permissive_left_cost(ego_speed, accel_raw)
                if c_pl > 0.0:
                    reward -= plp * c_pl

            # ---- 이유 없는 정지: 매 스텝 바로 감점 ----
            # 앞 60 m 안에 빨간불·노란불이 없고, 15 m 안에 앞차도 없는데 서 있으면.
            # (180초 뒤의 stuck_penalty 만으로는 할인 때문에 지금 기준 거의 0 이라
            #  "애매하면 서 있기"를 막지 못했다)
            ip = float(self.rw.get("idle_penalty", 0.0))
            if ip > 0.0 and ego_speed < 0.5:
                tls = self._next_tls()
                held_by_signal = (tls is not None and tls[2] <= 60.0
                                  and tls[3].lower() in ("r", "y", "u"))
                held_by_leader = ego_gap is not None and ego_gap < 15.0
                # 비보호('g') 링크에서 우선권 있는 상대 차를 기다리는 중이면 정당한 양보 대기 (2026-11-25).
                # 예전엔 이것도 감점돼 판당 ~10초, ~0.6점 → "기다리지 말고 들어가라" 압력이 됐다.
                held_by_yield = False
                if self.rw.get("idle_exempt_yield", False):
                    li, fs = self._ego_link_info, self._foe_state
                    held_by_yield = li is not None and li[0] == "g" and fs is not None and bool(fs[1])
                if not held_by_signal and not held_by_leader and not held_by_yield:
                    reward -= ip

            # ---- 줄 간격 감점 (2026-11-26): 신호 대기 줄에서 앞차와 너무 멀리 떨어져 서 있으면 ----
            # 나도 거의 서 있고(1 m/s 미만) 같은 차로 앞차도 서 있으며 그 앞차가 정지선보다 앞(줄)일 때만,
            # 간격이 queue_gap_free m 를 넘는 만큼 감점 (최대 queue_gap_penalty). 진행 보상은 총합이 같고
            # 정지 감점은 신호·양보 대기에서 면제라, 줄 뒤 30~40 m 에 서 있어도 손해가 없던 문제 대상.
            qgp = float(self.rw.get("queue_gap_penalty", 0.0))
            reward -= self._signal_queue_cost(ego_speed)
            self._ctx_last = None
            if self.rw.get("ctx_reward", False):
                if slp <= 0.0:
                    self._stopline_cost(ego_speed, accel_raw)   # 신호 상태(멈춰야 하는지·정지선 거리)만 갱신, 감점은 안 씀
                c_ctx, n_ctx = self._ctx_reward(ego_speed, accel_raw)
                reward -= c_ctx
                self._ctx_last = (n_ctx, c_ctx)
                qgp = 0.0                            # 예전 줄 간격 감점은 끈다 (상황별 보상이 대신)
            q_lead = same_lane_leader
            if qgp > 0.0 and self.rw.get("queue_gap_far", False):
                q_lead = self._far_leader()         # F. 150 m 까지 (관측과 같은 거리)
            # 줄 판정 속도: 기본은 나 < 1 m/s, 앞차 < 0.5 m/s (둘 다 멈춤). queue_gap_slow_v > 0 이면 둘 다 그 속도 미만
            # (천천히 움직이는 줄에서 앞차가 출발했는데 따라가지 않는 경우까지, 2026-11-26)
            q_v = float(self.rw.get("queue_gap_slow_v", 0.0))
            ego_slow = ego_speed < (q_v if q_v > 0.0 else 1.0)
            lead_slow = q_lead is not None and q_lead[1] < (q_v if q_v > 0.0 else 0.5)
            # 양보 예외 (queue_gap_exempt_yield): 지금 양보해야 하거나(비보호 좌회전·합류·회전교차로에서 상대 차가 옴),
            # 다음 링크가 우선권 없는 링크(비보호 g, 합류 m, 정지 s, 점멸 o)이거나, 교차로 안이면 줄 간격 감점 안 함
            q_exempt = False
            if qgp > 0.0 and self.rw.get("queue_gap_exempt_yield", False):
                li_q, fs_q = self._ego_link_info, self._foe_state
                in_junction = self.ego_id in traci.vehicle.getIDList() and traci.vehicle.getRoadID(self.ego_id).startswith(":")
                q_exempt = (in_junction or (fs_q is not None and bool(fs_q[1]))
                            or (li_q is not None and li_q[0] in ("g", "m", "s", "o")))
            if qgp > 0.0 and ego_slow and lead_slow and not q_exempt:
                q_gap = float(q_lead[0])
                free = float(self.rw.get("queue_gap_free", 8.0))
                tls = self._next_tls()
                ahead_of_line = tls is None or q_gap < float(tls[2]) - 1.0
                if ahead_of_line and q_gap > free:
                    reward -= qgp * min((q_gap - free) / float(self.rw.get("queue_gap_scale", 30.0)), 1.0)

            # ---- 막힘 페널티: 느린 앞차 뒤에 갇혀 있는 동안 매 스텝 감점 ----
            # "차선을 유지한 채 막혀 있기"와 "변경으로 탈출하기" 사이에
            # 즉각적인 보상 차등을 만들어 추월 학습 신호를 조밀하게 한다.
            # (자세한 설계 이유는 env/mdp_config.py의 REWARD 주석 참고)
            # 초록불 막힘 감점: 초록불(또는 신호 없음)인데 같은 차로 앞차가 서 있어 막혀 있고,
            # 경로로 이어지는 옆 차로 앞이 비어 있으면 매 스텝 감점 → 기다리지 말고 옆 차로로 피하게 유도.
            # 남부순환로 2번 차로(직진+좌회전)에서 좌회전 대기차 뒤에 서서 기다리기만 하던 행동 대상 (2026-11-24).
            gbp = float(self.rw.get("green_blocked_penalty", 0.0))
            if gbp > 0.0:
                avoid_ok, self._gb_reason = self._avoidable_left_waiter(same_lane_leader)
                changed = lane_change != 0 and lane_change_applied
                if avoid_ok and not changed:
                    reward -= gbp       # 안전하게 피할 수 있는데 계속 그 차 뒤에 있음

            # 교차로 안 회전 대기차 접근 감점: 교차로 안(또는 깜빡이 켠) 서 있는 회전 대기차를 향해 브레이크 없이 다가감
            jwp = float(self.rw.get("junction_waiter_penalty", 0.0))
            safe_signal_queue = self._safe_signal_queue(same_lane_leader, ego_speed, accel_raw)
            if jwp > 0.0:
                c_jw = self._junction_waiter_cost(same_lane_leader, ego_speed, accel_raw)
                if c_jw > 0.0 and not safe_signal_queue:
                    reward -= jwp * c_jw

            # 차로 끝 유지 감점: 내 차로가 곧 끝나는데 안전하게 옮길 옆 차로가 있는데도 유지
            lep = float(self.rw.get("lane_end_penalty", 0.0))
            if lep > 0.0:
                c_le = self._lane_end_hold_cost(lane_change, lane_change_applied)
                if c_le > 0.0:
                    reward -= lep * c_le
                # 합류 압박 (ctx_merge_penalty, 2026-11-27): 경로상 내 차로가 곧 끝나서 옆 차로로 옮겨야 하는데 틈이 없을 때,
                # 속도를 그대로 유지하면 (속도/제한속도) x (끝에 가까운 정도) 만큼 감점 -> 속도를 줄여 틈을 만드는 쪽을 배우게.
                # TEST31 75회·TEST32 25회 서초IC 출구, TEST32 50회 W_in: 차로 변경 명령은 냈지만 안전 검사에 거절당한 채
                # 속도를 안 줄여 그대로 지나쳐 경로를 놓쳤다. 강제 감속이 아니라 보상.
                w_mg = float(self.rw.get("ctx_merge_penalty", 0.0))
                blk = getattr(self, "_le_block", None)
                if w_mg > 0.0 and blk is not None and not (lane_change != 0 and lane_change_applied):
                    d_own, Zm = blk
                    reward -= w_mg * min(ego_speed / max(v_lim, 1.0), 1.0) * (1.0 - d_own / Zm)

            bp = float(self.rw.get("blocked_penalty", 0.0))
            if (bp > 0.0 and not safe_signal_queue and ego_gap is not None
                    and ego_gap < float(self.rw.get("blocked_gap", 20.0))
                    and ego_speed < float(self.rw.get("blocked_speed_frac", 0.7))
                        * v_lim):
                reward -= bp

            # ---- 차선변경 보상 항 ----
            # lane_change        : 이번 스텝에 policy가 낸 양자화 명령 (-1/0/+1)
            # lane_change_applied: 그 명령이 실제로 실행됐는지
            # 주의: _apply_lane_change는 유지 명령(0)에도 현재 차선으로
            # changeLane을 호출해 True를 반환한다(차선 고정 용도).
            # 따라서 "실제 변경"은 lane_change != 0 과 함께 판정해야 한다.
            if lane_change != 0 and lane_change_applied:
                # 실행된 변경마다 소액 감점 → 이득 없는 지그재그 억제.
                # 변경 덕에 속도 보상이 그 이상 늘면 순이득이므로,
                # "필요할 때만 바꾸는" 정책으로 수렴하게 만드는 항이다.
                reward -= float(self.rw.get("lane_change_penalty", 0.0))
            elif lane_change != 0 and getattr(self, "_lc_impossible", True):
                # 그 방향에 차로가 아예 없는데 변경을 요청한 경우만 감점 (맨 오른쪽에서 '오른쪽' 등).
                # 이게 공짜면 정책이 '유지'와 '불가능한 요청'을 같은 뜻으로 섞어 써서, 차로 행동의 확률이
                # 어느 쪽으로도 확실해지지 않는다 (탐험 없이 평가하면 필요한 곳에서도 안 바꿈).
                # 차로는 있지만 틈이 없어 거절된 요청(끼어들기 시도 실패)은 감점하지 않는다.
                reward -= float(self.rw.get("invalid_action_penalty", 0.0))

            # ---- 차로 안내 보상 (퍼텐셜 기반 shaping) ----
            # 회전·출구가 안내 거리 안에 들어오면, 경로로 이어지는 차로에 한 칸 가까워질 때 +w,
            # 멀어질 때 −w. 왔다 갔다 하면 합이 0 이라 "보상 모으기" 꼼수가 생기지 않는다.
            # 틀린 차로에서 회전이 안내 거리에 들어오면 그 순간 −w×(필요 변경 수) 를 받고,
            # 제 차로로 옮기면 돌려받는다. 경로를 다시 찾은 스텝은 기준만 새로 잡는다
            # (놓친 뒤 경로가 바뀌어 퍼텐셜이 0 이 되면서 reroute_penalty 를 상쇄하지 않게).
            lg = float(self.rw.get("lane_guide_weight", 0.0))
            wl = float(self.rw.get("wrong_lane_penalty", 0.0))
            if lg > 0.0 or wl > 0.0:
                phi = self._lane_guide_potential()
                if phi is not None:
                    if lg > 0.0 and self._prev_lane_phi is not None and not rerouted:
                        reward += lg * (phi - self._prev_lane_phi)
                    self._prev_lane_phi = phi
                # 틀린 차로 매 스텝 감점: 회전·출구가 안내 거리 안인데 경로로 이어지지 않는 차로에 있으면
                # 매 스텝 −wl × (옮겨야 할 칸 수). 위 퍼텐셜 보상은 '지금 옮기나 나중에 옮기나' 같아서
                # 매 스텝 유지와 변경의 이득 차이가 거의 없었다 → 늦게 옮길수록 손해가 되게 한다.
                # 감점만 있어서 왔다 갔다 하는 꼼수가 없고, 틈이 없어 못 옮길 때는 모든 행동이 똑같이 감점.
                if wl > 0.0 and self._prev_lane_phi is not None and self._prev_lane_phi < 0.0:
                    reward -= wl * abs(self._prev_lane_phi)

        if red_run and not terminated:
            reward -= float(self.rw.get("red_light_penalty", 0.0))
            terminated = True

        # 경로 이탈(좌회전·출구·유턴 차로를 놓쳐 재탐색): 한 번마다 감점,
        # max_reroutes 번째면 그 판을 끝낸다 (우회로에서 계속 도는 시간 낭비 방지, 추가 감점 없음)
        reroute_end = False
        if (rerouted or no_way_back) and not terminated:
            key = "no_way_back_penalty" if no_way_back else "reroute_penalty"
            reward -= float(self.rw.get(key, self.rw.get("reroute_penalty", 0.0)))
            if no_way_back or self.reroute_count >= int(self.rw.get("max_reroutes", 3)):
                reroute_end = True

        queue_timeout = self._queue_episode and self.step_count >= self.queue_episode_steps
        if self.step_count >= self.max_steps or reroute_end or queue_timeout:
            truncated = True
        # ego 가 stuck_timeout_s 초 넘게 거의 서 있으면 시간 초과로 끝낸다 (보상 변화 없음).
        # 예: 경로로 이어지지 않는 차로 끝에 서서 영영 못 가는 경우. 신호 대기보다 충분히 길게.
        stuck_end = False
        if alive and not terminated:
            if traci.vehicle.getSpeed(self.ego_id) < 0.1:
                self._stopped_steps += 1
            else:
                self._stopped_steps = 0
            if self._stopped_steps * self.step_length >= self.stuck_timeout:
                truncated = True
                stuck_end = True
                reward -= float(self.rw.get("stuck_penalty", 0.0))
        if teleported and not collided and not arrived and not stuck_end:
            terminated, truncated, stuck_end = False, True, True
            reward = -float(self.rw.get("stuck_penalty", 0.0))

        info = {
            "collided": collided,
            "arrived": arrived,
            "speed": ego_speed,
            "gap": ego_gap,
            "accel_raw": accel_raw,
            "lane_change_raw": lane_raw,
            "lane_change": lane_change,
            "lane_change_applied": lane_change_applied,
            "lane_before": lane_before,
            "lane_after": lane_after,
            "red_light_violation": red_run,
            "rerouted": rerouted,
            "stuck_timeout": stuck_end,
            "reroute_limit": reroute_end,
            "no_way_back": bool(no_way_back),
            "queue_training": self._queue_episode,
            "queue_timeout": queue_timeout,
            "signal_queue": self._signal_queue_last,
            "remaining_m": self._prev_rem,   # 목적지까지 남은 경로 거리 (평가의 진행률 계산용)
        }
        if rec is not None:
            rec.after_step(action_arr, float(reward), terminated, truncated, info, obs)
        self._last_obs = obs
        return self._stacked(obs), float(reward), terminated, truncated, info

    # ==================================================================
    def close(self):
        """환경 정리. 학습이 끝나면 반드시 호출해서 SUMO 프로세스를 종료할 것."""
        if self._started:
            traci.close()
            self._started = False
