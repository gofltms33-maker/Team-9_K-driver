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

import traci
import traci.constants as tc
from sumolib import checkBinary


class SumoHighwayEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg_path, road, ego, mdp_sim, mdp_obs, action, reward,
                 traffic=None, gui=False, gui_autostart=True, native_driver=False,
                 planned_route=None):
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
        # Environment validation only: never enable implicitly during PPO training.
        self.native_driver = native_driver
        self.planned_route = tuple(planned_route) if planned_route is not None else None

        # ----- 도로/차량 설정에서 필요한 값들 -----
        self.vmax = float(road["speed_limit"])   # 속도 정규화 기준
        self.control_speed_cap = min(self.vmax, float(ego["max_speed"]),
                                     float(mdp_sim.get("ego_speed_cap_mps", self.vmax)))
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
        self.default_action_step_length = mdp_sim.get("default_action_step_length_s")
        self.ego_action_step_length = mdp_sim.get("ego_action_step_length_s")
        self.max_steps = int(mdp_sim["max_steps"])
        self.stuck_timeout = float(mdp_sim.get("stuck_timeout_s", 180.0))
        self._stopped_steps = 0
        self._idle_steps = 0
        self.idle_grace = float(mdp_sim.get("idle_grace_s", 3.0))
        self.idle_timeout = float(mdp_sim.get("idle_timeout_s", 30.0))
        self.reroute_count = 0          # 이번 에피소드의 경로 재탐색 횟수
        self.warmup_steps = int(mdp_sim["warmup_steps"])
        self.visibility = float(mdp_obs["visibility"])   # 관측 거리 W
        self.lane_guidance = bool(mdp_obs.get("lane_guidance", False))
        # Preserve archived tensor shapes while removing all detailed lane guidance.
        self.redact_lane_guidance = bool(mdp_obs.get("redact_lane_guidance", False))
        self.physical_layout_features = bool(mdp_obs.get('physical_layout_features',False))
        # Optional observable odometer, independent of route/lane answers. Reuse
        # a reserved redacted slot so archived policy tensor shapes stay fixed.
        self.route_distance_feature = bool(mdp_obs.get("route_distance_feature", False))
        self.route_distance_scale_m = float(mdp_obs.get("route_distance_scale_m", 10000.0))
        if self.route_distance_feature and (not self.lane_guidance or not self.redact_lane_guidance):
            raise ValueError("Route distance requires the reserved redacted lane block")
        if self.route_distance_feature and (not np.isfinite(self.route_distance_scale_m) or self.route_distance_scale_m <= 0):
            raise ValueError("route_distance_scale_m must be positive and finite")
        self.next_junction_guidance = bool(mdp_obs.get("next_junction_guidance", False))
        self.front_safety_features = bool(mdp_obs.get("front_safety_features", False))
        self.retain_overlap_neighbors = bool(mdp_obs.get("retain_overlap_neighbors", False))
        # Same dimension, different map semantics: saved runs must opt in.
        self.semantic_lane_branches = bool(mdp_obs.get("semantic_lane_branches", False))
        self._branch_direction_cache = {}
        self.signal_safety_feature = bool(mdp_obs.get("signal_safety_feature", False))
        self.nearby_traffic_features = bool(mdp_obs.get("nearby_traffic_features", False))
        self.nearby_traffic_count = int(mdp_obs.get("nearby_traffic_count", 8))
        # 내비게이션·공사 안내 / 신호등을 보는 거리 (env/mdp_config.py)
        self.nav_horizon = float(mdp_obs.get("nav_horizon", 300.0))
        self.signal_horizon = float(mdp_obs.get("signal_horizon", 150.0))
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
        self.lane_change_cooldown = float(action.get("lane_change_cooldown_s", 0.0))
        # Archived mode 0 is legacy (some SUMO junction checks remain). New
        # explicitly unassisted configurations use 96; expert validation uses 31.
        self.policy_speed_mode = int(action.get("policy_speed_mode", 0))
        self.policy_lane_change_mode = int(action.get("policy_lane_change_mode", 0))
        if self.policy_lane_change_mode not in (0, 256, 512):
            raise ValueError("Policy lane mode must disable autonomous lane selection (0/256/512)")
        self._last_lane_change_step = -1000000
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
        # + 신호(상태, 거리) 2 + 내비(거리, 좌, 우) 3 + 공사 거리 1
        self.obs_dim = 1 + n_lanes_view * 4 + n_lanes_view + n_lanes_view + 2 + 3 + 1 + 1
        self.obs_dim += 17 if self.lane_guidance else 0
        self.obs_dim += 5 if self.next_junction_guidance else 0
        self.obs_dim += 3 if self.front_safety_features else 0
        self.obs_dim += 1 if self.signal_safety_feature else 0
        self.obs_dim += 3 if self.physical_layout_features else 0
        self.obs_dim += 5 * self.nearby_traffic_count if self.nearby_traffic_features else 0
        if self.physical_layout_features and self.obs_dim!=107:
            raise ValueError('Own-layout sensors require the standard104-feature current-frame layout')
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
        if self.default_action_step_length is not None:
            cmd += ["--default.action-step-length", str(self.default_action_step_length)]
        traci.start(cmd)   # SUMO를 자식 프로세스로 띄우고 소켓 연결
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

        # ego 차량 투입 — 반드시 "도로 시작점(0m)"에서 출발
        insertion_route = self._ego_insertion_route()
        traci.vehicle.add(self.ego_id, insertion_route, typeID="ego_type",
                          departLane=self.ego_depart_lane,
                          departPos="0",
                          departSpeed=self.ego_depart_speed)
        traci.vehicle.setMaxSpeed(self.ego_id, self.control_speed_cap)
        if self.ego_action_step_length is not None:
            traci.vehicle.setActionStepLength(self.ego_id, float(self.ego_action_step_length))

        # ★ mode 96 disables all SUMO speed checks. Mode 0 is only the legacy
        # "most checks off" mode; its junction-internal checks may remain active.
        #   이걸 꺼야 RL이 속도를 온전히 제어하고, 잘못 배우면 실제로 충돌한다.
        #   (켜두면 SUMO가 알아서 브레이크를 밟아줘서 충돌이 거의 안 나고,
        #    그러면 "충돌 회피"라는 학습 신호 자체가 사라진다)
        #   신호·교차로 통행 규칙도 SUMO가 대신 지켜주지 않는다 (보조 규칙 금지).
        #   대신 관측에 앞 신호 상태·정지선까지 거리를 넣고(_signal_features),
        #   교차로 안 충돌도 충돌로 판정한다(--collision.check-junctions).
        traci.vehicle.setSpeedMode(self.ego_id, 31 if self.native_driver else self.policy_speed_mode)
        # SUMO의 자율 차선변경은 끄고, RL이 내리는 TraCI 차선변경 명령만 사용한다.
        traci.vehicle.setLaneChangeMode(self.ego_id, 1621 if self.native_driver else self.policy_lane_change_mode)

        traci.simulationStep()  # ego가 실제로 도로에 나타나도록 1스텝 진행

        # ── ego 삽입 보증 ──
        # 도로 입구는 flow가 차를 스폰하는 자리라, 시작 차선 입구가 잠시
        # 막혀 있으면 SUMO가 삽입을 "지연"시킨다. 그 경우 ego 없이
        # 에피소드가 진행되어 즉시 종료되는 버그가 되므로, 삽입될 때까지
        # (최대 30스텝) 기다린다. 지연되어도 요청한 차선은 유지된다.
        _wait = 0
        while self.ego_id not in traci.vehicle.getIDList() and _wait < 30:
            traci.simulationStep()
            _wait += 1
        if self.ego_id not in traci.vehicle.getIDList():
            raise RuntimeError(
                "ego 삽입 실패: 도로 입구가 30스텝 이상 막혀 있습니다. "
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

        if self.planned_route is not None:
            # Navigation plan set once before the episode; no automatic recovery.
            traci.vehicle.setRoute(self.ego_id, self.planned_route)
            self._nav_key = None
        if self.nearby_traffic_features:
            traci.vehicle.subscribeContext(self.ego_id, tc.CMD_GET_VEHICLE_VARIABLE,
                                           self.visibility, (tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_ANGLE))
        self.step_count = 0
        self._stopped_steps = 0
        self._idle_steps = 0
        self._last_lane_change_step = -1000000
        self.reroute_count = 0
        obs = self._get_obs()
        self._last_obs = obs
        if self.recorder is not None and self.recording:
            self.recorder.on_reset()
        return obs, {}

    def _ego_insertion_route(self):
        """Support explicitly assigned AV navigation starts at episode reset only.

        Original-route and prefix tasks keep their existing insertion behavior.
        The HV route definitions and road network are never altered.
        """
        if self.planned_route is None:
            return "r0"
        original = traci.route.getEdges("r0")
        if self.planned_route[0] == original[0]:
            return "r0"
        traci.route.add("av_navigation_route", self.planned_route)
        return "av_navigation_route"

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

    def _neighbor(self, offset: int, leader: bool):
        """오프셋 차선(-1=오른쪽, 0=현재, +1=왼쪽)의 가장 가까운
        선행(leader=True)/후행 차량을 (거리 m, 그 차량 속도)로 반환.
        visibility 밖이거나 없으면 None."""
        if offset == 0:
            if leader:
                res = traci.vehicle.getLeader(self.ego_id, self.visibility)
                if res is None:
                    return None
                vid, gap = res
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
            if (res is None or res[0] > self.visibility
                    or (res[0] < 0 and not self.retain_overlap_neighbors)):
                return None
            return max(float(res[0]), 0.0), float(res[1])
        else:
            mode_leader, mode_follower = self._NEIGHBOR_MODES[offset]
            pairs = traci.vehicle.getNeighbors(
                self.ego_id, mode_leader if leader else mode_follower)
            if not pairs:
                return None
            vid, gap = min(pairs, key=lambda p: p[1])   # 가장 가까운 차량

        if not vid or gap > self.visibility or (gap < 0 and not self.retain_overlap_neighbors):
            return None
        return max(float(gap), 0.0), traci.vehicle.getSpeed(vid)

    def _route_lanes(self):
        """[데이터 기록 전용 — 관측에는 쓰지 않는다]
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
            if remaining > self.visibility:
                return 1.0                    # 시야 내에선 계속 이어짐
            if traci.lane.getLinks(lane_id):
                return 1.0                    # edge가 끝나도 다음으로 연결됨
            return float(np.clip(remaining / self.visibility, 0.0, 1.0))
        except traci.TraCIException:
            return 1.0                        # 조회 실패 시 중립값

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
    # 신호등 · 내비게이션 · 공사 구간 관측 (지도 수준 정보만, 차로 안내 없음)
    # ==================================================================
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
        nxt = self._next_tls()
        if nxt is None or nxt[2] > self.signal_horizon:
            return [1.0, 1.0]
        c = nxt[3].lower()
        state = 0.0 if c in ("r", "u") else (0.5 if c == "y" else 1.0)
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

    def _branch_direction(self, source, target, raw_direction):
        """Describe a map-visible branch without supplying a route/lane answer.

        SUMO may call both a through road and a tangential exit 's'. A
        lower-priority exit is classified by its geometry relative to the other
        branches, as navigation already did. Every connection is described
        independently of the AV route, traffic, or selected driving action.
        """
        key = (source.getID(), target.getID(), raw_direction)
        if key in self._branch_direction_cache:
            return self._branch_direction_cache[key]
        direction = raw_direction
        if raw_direction == "s" and target.getPriority() < source.getPriority():
            others = [edge for edge in source.getOutgoing() if edge.getID() != target.getID()]
            if others:
                offset = self._lateral(source, target)
                other_offsets = [self._lateral(source, edge) for edge in others]
                if all(offset < other for other in other_offsets):
                    direction = "r"
                elif all(offset > other for other in other_offsets):
                    direction = "l"
        self._branch_direction_cache[key] = direction
        return direction

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
            d = self._branch_direction(e, nxt, mine[0].getDirection())
            if d in ("l", "L", "t"):
                tab[k] = "L"
            elif d in ("r", "R"):
                tab[k] = "R"
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
                if d <= self.nav_horizon:
                    out[0] = d / self.nav_horizon
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

    def _reroute_if_missed(self) -> bool:
        """경로가 이어지지 않는 차로로 갈림길 직전까지 왔으면 경로를 다시 찾는다.

        내비게이션의 '경로 재탐색'과 같다. 학습차가 좌회전·출구 차로로 들어가지 못한 채
        정지선/분기점 바로 앞(남은 거리 < max(5 m, 1.5초 주행 거리))까지 오면, 지금 차로로
        갈 수 있는 길들 중 목적지까지 가장 빠른 길로 경로를 바꾼다. 어느 차로로 가라거나
        언제 바꾸라는 안내는 하지 않는다 (놓친 뒤에 새 길을 줄 뿐).
        """
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                return False
            route = traci.vehicle.getRoute(self.ego_id)
            k = traci.vehicle.getRouteIndex(self.ego_id)
            if k < 0 or k + 1 >= len(route) or route[k] != edge:
                return False
            lane = traci.vehicle.getLaneID(self.ego_id)
            links = traci.lane.getLinks(lane)
            nxt = route[k + 1]
            if any(l[0].rsplit("_", 1)[0] == nxt for l in links):
                return False                       # 지금 차로로 경로가 이어짐
            remaining = traci.lane.getLength(lane) - traci.vehicle.getLanePosition(self.ego_id)
            if remaining > max(5.0, traci.vehicle.getSpeed(self.ego_id) * 1.5):
                return False                       # 아직 차로를 바꿀 여유가 있음
            dest = route[-1]
            best = None
            for l in links:
                succ = l[0].rsplit("_", 1)[0]
                r = traci.simulation.findRoute(succ, dest)
                if r.edges and (best is None or r.travelTime < best[0]):
                    best = (r.travelTime, list(r.edges))
            if best is None:
                return False                       # 갈 수 있는 길이 없음 (막다른 차로)
            traci.vehicle.setRoute(self.ego_id, [edge] + best[1])
            self._nav_key = None                   # 내비 안내표 다시 계산
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

    def _lane_marking_features(self):
        """Map-visible turn permissions, not SUMO best-lane or future traffic answers.

        Five relative lanes x (left/straight/right), distance to edge end, lane index.
        """
        edge = traci.vehicle.getRoadID(self.ego_id)
        lane = traci.vehicle.getLaneID(self.ego_id)
        index = traci.vehicle.getLaneIndex(self.ego_id)
        count = traci.edge.getLaneNumber(edge)
        features = []
        for offset in self.obs_offsets:
            target = index + offset
            directions = set()
            if 0 <= target < count and not edge.startswith(":"):
                links = traci.lane.getLinks(f"{edge}_{target}")
                if self.semantic_lane_branches:
                    net = self._net()
                    source = net.getEdge(edge)
                    directions = {self._branch_direction(source, net.getEdge(link[0].rsplit("_", 1)[0]), link[6])
                                  for link in links}
                else:
                    directions = {link[6] for link in links}
            features.extend([float(bool(directions & {"l", "L", "t"})),
                             float("s" in directions),
                             float(bool(directions & {"r", "R"}))])
        remaining = traci.lane.getLength(lane) - traci.vehicle.getLanePosition(self.ego_id)
        return features + [float(np.clip(remaining / self.nav_horizon, 0, 1)),
                           index / max(count - 1, 1)]

    def _next_junction_features(self):
        """Navigation for the next actual branch, including 'continue straight'.

        No best-lane answer, policy action or route change is computed here.
        """
        route = traci.vehicle.getRoute(self.ego_id)
        start = max(0, traci.vehicle.getRouteIndex(self.ego_id))
        current = traci.vehicle.getRoadID(self.ego_id)
        net = self._net()
        for k in range(start, len(route) - 1):
            edge = net.getEdge(route[k])
            outgoing = edge.getOutgoing()
            if len(outgoing) < 2:
                continue
            if route[k] == current:
                lane = traci.vehicle.getLaneID(self.ego_id)
                distance = traci.lane.getLength(lane) - traci.vehicle.getLanePosition(self.ego_id)
            else:
                distance = traci.vehicle.getDrivingDistance(self.ego_id, route[k], edge.getLength() - 0.1)
            if distance < 0:
                continue
            if distance > self.nav_horizon:
                break
            directions = {(self._branch_direction(edge, target, c.getDirection())
                           if self.semantic_lane_branches else c.getDirection()) for target, cs in outgoing.items()
                          if target.getID() == route[k + 1] for c in cs}
            return [float(np.clip(distance / self.nav_horizon, 0, 1)),
                    float(bool(directions & {"l", "L"})), float("s" in directions),
                    float(bool(directions & {"r", "R"})), float("t" in directions)]
        return [1.0, 0.0, 0.0, 0.0, 0.0]

    def _front_safety_observation(self):
        leader = self._neighbor(0, leader=True)
        if leader is None:
            return [1.0, 1.0, 1.0]
        gap, leader_speed = leader
        speed = traci.vehicle.getSpeed(self.ego_id)
        closing = max(speed - leader_speed, 0.0)
        ttc = gap / closing if closing > 0.1 else 10.0
        headway = gap / max(speed, 0.1)
        braking_distance = max(speed * speed - leader_speed * leader_speed, 0) / (2 * self.max_decel)
        margin = gap - braking_distance - speed * self.step_length
        return [float(np.clip(ttc / 10, 0, 1)), float(np.clip(headway / 3, 0, 1)),
                float(np.clip(margin / 30, -1, 1))]

    def _get_obs(self):
        # ego가 이미 사라졌다면(충돌/도착) 0벡터 반환 — 종료 직전 마지막 관측용
        if self.ego_id not in traci.vehicle.getIDList():
            return np.zeros(self.obs_dim, dtype=np.float32)

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
        obs.append(self._road_speed_limit() / self.vmax)
        if self.lane_guidance:
            obs.extend([0.0] * 17 if self.redact_lane_guidance else self._lane_marking_features())
            if self.route_distance_feature:
                obs[53] = float(np.clip(traci.vehicle.getDistance(self.ego_id)
                                        / self.route_distance_scale_m, 0.0, 1.0))
        if self.next_junction_guidance:
            obs.extend(self._next_junction_features())
        if self.front_safety_features:
            obs.extend(self._front_safety_observation())
        if self.signal_safety_feature:
            stopping = v * v / 4.0 + v * self.step_length
            obs.append(float(np.clip((obs[32] * self.signal_horizon - stopping) / 60, -1, 1)))
        if self.nearby_traffic_features:
            if self.physical_layout_features:
                # Measured own state only; no desired lane or route-lane answer.
                edge=traci.vehicle.getRoadID(self.ego_id)
                count=traci.edge.getLaneNumber(edge)
                index=traci.vehicle.getLaneIndex(self.ego_id)
                from physical_layout import encode_own_layout
                obs.extend(encode_own_layout(edge,index,count))
            obs.extend(self._nearby_traffic_observation())

        return np.clip(np.array(obs, dtype=np.float32), -1.0, 1.0)

    def _nearby_traffic_observation(self):
        """Observed nearby vehicle positions/velocities, including cross traffic.

        Ego-centric forward/left axes, nearest first, with an occupancy mask.
        No IDs, vehicle routes, priorities, predicted conflicts or actions enter.
        """
        x, y = traci.vehicle.getPosition(self.ego_id)
        angle = np.deg2rad(traci.vehicle.getAngle(self.ego_id))
        forward = np.array([np.sin(angle), np.cos(angle)])
        left = np.array([-forward[1], forward[0]])
        velocity = forward * traci.vehicle.getSpeed(self.ego_id)
        nearby = traci.vehicle.getContextSubscriptionResults(self.ego_id) or {}
        vehicles = []
        for vid, state in nearby.items():
            if vid == self.ego_id:
                continue
            delta = np.asarray(state[tc.VAR_POSITION]) - np.array([x, y])
            distance = float(np.linalg.norm(delta))
            if distance > self.visibility:
                continue
            heading = np.deg2rad(state[tc.VAR_ANGLE])
            other_velocity = state[tc.VAR_SPEED] * np.array([np.sin(heading), np.cos(heading)])
            relative_velocity = other_velocity - velocity
            features = [float(delta @ forward / self.visibility), float(delta @ left / self.visibility),
                        float(relative_velocity @ forward / self.vmax),
                        float(relative_velocity @ left / self.vmax), 1.0]
            vehicles.append((distance, features))
        vehicles.sort(key=lambda pair: pair[0])
        result = [value for _, features in vehicles[:self.nearby_traffic_count] for value in features]
        return result + [0.0] * (5 * self.nearby_traffic_count - len(result))

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
            front, rear = None, None
            for vid in traci.lane.getLastStepVehicleIDs(lane_id):
                d = traci.vehicle.getLanePosition(vid) - my_pos
                if d >= 0:
                    g = d - traci.vehicle.getLength(vid)
                    if front is None or g < front[0]:
                        front = (g, traci.vehicle.getSpeed(vid))
                else:
                    g = -d - traci.vehicle.getLength(self.ego_id)
                    if rear is None or g < rear[0]:
                        rear = (g, traci.vehicle.getSpeed(vid))
            return front, rear
        except traci.TraCIException:
            return None, None

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
        if traci.vehicle.getDistance(self.ego_id) < self.lc_spawn_protect:
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

    def _apply_lane_change(self, lane_change: int) -> bool:
        """가능하고 "안전한" 경우에만 차선변경을 적용하고 성공 여부를 반환.

        안전 게이트(_lane_change_is_safe)를 통과하지 못한 명령은 실행하지
        않고 False를 반환한다 → step()에서 "불가능한 명령"과 동일하게
        invalid_action_penalty가 부과된다. 즉 에이전트 입장에서
        "위험한 변경 = 실행도 안 되고 감점"이라, 측면충돌로 즉사하며
        배우는 대신 안전한 타이밍을 고르는 쪽으로 학습이 유도된다.
        """
        if self.ego_id not in traci.vehicle.getIDList():
            return False
        if lane_change != 0 and ((self.step_count - self._last_lane_change_step)
                                 * self.step_length < self.lane_change_cooldown):
            return False
        try:
            edge = traci.vehicle.getRoadID(self.ego_id)
            if edge.startswith(":"):
                # Relative API retains requests across junctions. Some native
                # exit maneuvers are not representable (e.g. right from internal
                # lane zero); do not silently claim those requests were applied.
                if traci.vehicle.getLaneIndex(self.ego_id) + lane_change < 0:
                    return False
                traci.vehicle.changeLaneRelative(self.ego_id, lane_change,
                                                 self.lane_change_duration)
                return True
            current = traci.vehicle.getLaneIndex(self.ego_id)
            n_lanes = traci.edge.getLaneNumber(edge)
            target = current + lane_change
            if target < 0 or target >= n_lanes:
                return False
            # ★ 측면충돌 방지: 실제 이동 명령(0이 아닌 변경)에만 안전 검사
            if (lane_change != 0 and self.lc_safety_enabled
                    and not self._lane_change_is_safe(lane_change)):
                return False
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

    def _road_speed_limit(self):
        try:
            return max(0.1, min(self.control_speed_cap, traci.lane.getMaxSpeed(
                traci.vehicle.getLaneID(self.ego_id))))
        except traci.TraCIException:
            return self.vmax

    def _signal_entry_snapshot(self):
        """Capture signal states for movements the AV can physically enter this step.

        A TLS link disappearing is not evidence of crossing a stop line: a lane
        change or reroute can change that link while the AV is still upstream.
        """
        edge = traci.vehicle.getRoadID(self.ego_id)
        if edge.startswith(":"):
            return None
        lane = traci.vehicle.getLaneID(self.ego_id)
        remaining = traci.lane.getLength(lane) - traci.vehicle.getLanePosition(self.ego_id)
        if remaining > self.vmax * self.step_length + 3.0:
            return None
        movements = []
        for index in range(traci.edge.getLaneNumber(edge)):
            from_lane = f"{edge}_{index}"
            for link in traci.lane.getLinks(from_lane, extended=True):
                movements.append((link[0], link[4], link[5].lower(), from_lane))
        return edge, movements

    @staticmethod
    def _internal_lane_reaches(via, target):
        pending, visited = [via], set()
        while pending and len(visited) < 32:
            lane = pending.pop()
            if lane == target:
                return True
            if not lane.startswith(":") or lane in visited:
                continue
            visited.add(lane)
            pending.extend(link[0] for link in traci.lane.getLinks(lane))
        return False

    def _red_light_crossed(self, snapshot, entered_lane):
        if snapshot is None:
            return False
        edge, movements = snapshot
        if entered_lane.rsplit("_", 1)[0] == edge:
            return False                  # Includes standing on the stop line.
        crossed = [(target, via, state, source) for target, via, state, source in movements
                   if target == entered_lane or via == entered_lane
                   or (entered_lane.startswith(":") and via
                       and self._internal_lane_reaches(via, entered_lane))]
        # If a very short junction was crossed in one step and several incoming
        # lanes merge, avoid assigning an ambiguous red movement as a violation.
        if not crossed or any(state not in ("r", "u") for _, _, state, _ in crossed):
            return False
        # SUMO advances traffic lights inside simulationStep. A car legitimately
        # starting on the first green step must not be charged for the previous red.
        for target, via, _, source in crossed:
            now = [link[5].lower() for link in traci.lane.getLinks(source, extended=True)
                   if link[0] == target and link[4] == via]
            if not now or any(state not in ("r", "u") for state in now):
                return False
        return True

    def _route_lane_blocked(self):
        """Diagnostic only: a stopped AV has reached the end of a wrong route lane.

        This does not choose a lane or expose SUMO's best-lane answer to a policy.
        """
        lane = traci.vehicle.getLaneID(self.ego_id)
        if lane.startswith(":") or traci.vehicle.getSpeed(self.ego_id) >= 0.1:
            return False
        remaining = traci.lane.getLength(lane) - traci.vehicle.getLanePosition(self.ego_id)
        if remaining > 3.0:
            return False
        return any(item[0] == lane and not item[4]
                   for item in traci.vehicle.getBestLanes(self.ego_id))

    def _required_wait(self):
        """Explain a necessary wait, without selecting or overriding an action."""
        v = traci.vehicle.getSpeed(self.ego_id)
        leader = self._neighbor(0, leader=True)
        if leader is not None and leader[0] <= max(8.0, 3.0 + v):
            return "leader"
        tls = self._next_tls()
        braking_distance = v * v / (2.0 * self.max_decel) + v * self.step_length
        if (tls is not None and tls[3].lower() in ("r", "u", "y")
                and tls[2] <= max(8.0, braking_distance + 3.0)):
            return "signal"
        lane = traci.vehicle.getLaneID(self.ego_id)
        if traci.lane.getLength(lane) - traci.vehicle.getLanePosition(self.ego_id) <= 15.0:
            links = traci.lane.getLinks(lane, extended=True)
            edge = traci.vehicle.getRoadID(self.ego_id)
            if not edge.startswith(":"):
                route = traci.vehicle.getRoute(self.ego_id)
                k = traci.vehicle.getRouteIndex(self.ego_id)
                if 0 <= k < len(route) - 1:
                    links = [link for link in links
                             if link[0].rsplit("_", 1)[0] == route[k + 1]]
            if any(link[3] or not link[2] or link[5].lower() in ("r", "u", "y")
                   for link in links):
                return "junction"
        return None

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_arr.size < self.act_dim:
            raise ValueError(f"Expected {self.act_dim} action dimensions")
        accel_raw = float(np.clip(action_arr[self.accel_index], -1.0, 1.0))
        lane_raw = float(np.clip(action_arr[self.lane_change_index], -1.0, 1.0))
        lane_change = self._quantize_lane_change(lane_raw)
        rec = self.recorder if self.recorder is not None and self.recording else None
        if rec is not None:
            rec.before_step(self._last_obs)

        edge_before = traci.vehicle.getRoadID(self.ego_id)
        lane_before = int(traci.vehicle.getLaneIndex(self.ego_id))
        distance_before = traci.vehicle.getDistance(self.ego_id)
        signal_entry = self._signal_entry_snapshot()
        road_limit = self._road_speed_limit()
        v = traci.vehicle.getSpeed(self.ego_id)
        accel_cmd = accel_raw * (self.max_accel if accel_raw >= 0 else self.max_decel)
        requested_speed = None
        if self.native_driver:
            traci.vehicle.setSpeed(self.ego_id, -1)
            lane_change_applied = False
        else:
            requested_speed = float(np.clip(v + accel_cmd * self.step_length, 0.0, self.control_speed_cap))
            traci.vehicle.setSpeed(self.ego_id, requested_speed)
            lane_change_applied = self._apply_lane_change(lane_change)
        traci.simulationStep()
        self.step_count += 1

        collided = self.ego_id in traci.simulation.getCollidingVehiclesIDList()
        # Keep HV teleport settings unchanged; an AV teleport is always a failure,
        # including same-step removal and reinsertion or teleport-to-destination.
        teleported = (self.ego_id in traci.simulation.getStartingTeleportIDList()
                      or self.ego_id in traci.simulation.getEndingTeleportIDList())
        arrived = (self.ego_id in traci.simulation.getArrivedIDList()
                   and not collided and not teleported)
        alive = self.ego_id in traci.vehicle.getIDList()
        realized_speed = traci.vehicle.getSpeed(self.ego_id) if alive else None
        lane_after = int(traci.vehicle.getLaneIndex(self.ego_id)) if alive else None
        actual_lane_change = bool(alive and not teleported
                                  and traci.vehicle.getRoadID(self.ego_id) == edge_before
                                  and lane_after != lane_before)
        if actual_lane_change:
            self._last_lane_change_step = self.step_count
        red_run = bool(alive and not collided and not teleported
                       and self._red_light_crossed(signal_entry,
                                                  traci.vehicle.getLaneID(self.ego_id)))
        # Judge the movement actually taken before navigation changes the route.
        # No rule-based recovery: the policy must follow the original route itself.
        rerouted = False
        obs = self._get_obs()
        terminated = truncated = False
        end_reason = None
        ego_speed = ego_gap = None
        physical_progress = 0.0
        waiting_reason = None
        route_blocked = False
        failure_penalty = float(self.rw.get("failure_penalty", 5.0))
        if collided:
            reward, terminated, end_reason = -float(self.rw["collision_penalty"]), True, "collision"
        elif teleported:
            reward, terminated, end_reason = -failure_penalty, True, "teleport"
        elif red_run:
            reward, terminated, end_reason = -float(self.rw["red_light_penalty"]), True, "red_light"
        elif arrived:
            reward, terminated, end_reason = float(self.rw["arrival_bonus"]), True, "arrival"
        elif not alive:
            reward, terminated, end_reason = -failure_penalty, True, "disappeared"
        else:
            ego_speed = traci.vehicle.getSpeed(self.ego_id)
            leader = self._neighbor(0, leader=True)
            ego_gap = leader[0] if leader is not None else None
            physical_progress = float(np.clip(
                traci.vehicle.getDistance(self.ego_id) - distance_before,
                0.0, self.vmax * self.step_length))
            reward = float(self.rw["speed_weight"]) * min(ego_speed / road_limit, 1.0)
            reward += float(self.rw.get("progress_weight", 0.0)) * physical_progress
            reward -= float(self.rw.get("overspeed_penalty", 0.0)) * max(
                ego_speed / road_limit - 1.0, 0.0)
            safe_gap = (float(self.rw["close_gap_threshold"])
                        + float(self.rw.get("close_gap_time_headway", 0.0)) * ego_speed)
            if ego_gap is not None and ego_speed > 0.5 and ego_gap < safe_gap:
                reward -= float(self.rw["close_gap_penalty"])
            waiting_reason = self._required_wait()
            route_blocked = self._route_lane_blocked()
            if (not waiting_reason and ego_gap is not None
                    and ego_gap < float(self.rw.get("blocked_gap", 20.0))
                    and ego_speed < float(self.rw.get("blocked_speed_frac", 0.7)) * road_limit):
                reward -= float(self.rw.get("blocked_penalty", 0.0))
            if actual_lane_change:
                reward -= float(self.rw.get("lane_change_penalty", 0.0))
            elif lane_change != 0 and not lane_change_applied:
                reward -= float(self.rw.get("invalid_action_penalty", 0.0))
            self._stopped_steps = self._stopped_steps + 1 if ego_speed < 0.1 else 0
            self._idle_steps = (self._idle_steps + 1
                                if ego_speed < 0.1 and waiting_reason is None else 0)
            idle_seconds = self._idle_steps * self.step_length
            if idle_seconds > self.idle_grace:
                reward -= float(self.rw.get("idle_penalty", 0.0))
            if idle_seconds >= self.idle_timeout:
                reward -= failure_penalty
                terminated, end_reason = True, "route_blocked" if route_blocked else "idle"
            elif self._stopped_steps * self.step_length >= self.stuck_timeout:
                reward -= failure_penalty
                terminated, end_reason = True, "stuck"
            elif self.step_count >= self.max_steps:
                truncated, end_reason = True, "time_limit"

        info = {
            "collided": collided, "arrived": arrived, "teleported": teleported,
            "control_source": "sumo_validation" if self.native_driver else "policy",
            "sumo_speed_mode": 31 if self.native_driver else self.policy_speed_mode,
            "sumo_lane_change_mode": 1621 if self.native_driver else self.policy_lane_change_mode,
            "requested_speed": requested_speed, "realized_speed": realized_speed,
            "longitudinal_command_error": (abs(requested_speed - realized_speed)
                if requested_speed is not None and realized_speed is not None else None),
            "end_reason": end_reason, "waiting_reason": waiting_reason,
            "route_blocked": route_blocked,
            "idle_seconds": self._idle_steps * self.step_length,
            "progress_m": physical_progress, "road_speed_limit": road_limit,
            "speed": ego_speed, "gap": ego_gap,
            "accel_raw": accel_raw, "lane_change_raw": lane_raw,
            "lane_change": lane_change, "lane_change_applied": lane_change_applied,
            "actual_lane_change": actual_lane_change,
            "lane_before": lane_before, "lane_after": lane_after,
            "red_light_violation": red_run, "rerouted": rerouted,
        }
        if rec is not None:
            rec.after_step(action_arr, float(reward), terminated, truncated, info, obs)
        self._last_obs = obs
        return obs, float(reward), terminated, truncated, info

    # ==================================================================
    def close(self):
        """환경 정리. 학습이 끝나면 반드시 호출해서 SUMO 프로세스를 종료할 것."""
        if self._started:
            traci.close()
            self._started = False
