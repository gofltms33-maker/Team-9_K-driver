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
        self._start_cands = None
        self.start_edge = None
        self.warmup_steps = int(mdp_sim["warmup_steps"])
        self.warmup_jitter_steps = int(mdp_sim.get("warmup_jitter_steps", 0))
        self.visibility = float(mdp_obs["visibility"])   # 관측 거리 W
        # 내비게이션·공사 안내 / 신호등을 보는 거리 (env/mdp_config.py)
        self.nav_horizon = float(mdp_obs.get("nav_horizon", 300.0))
        self.nav_horizon_s = float(mdp_obs.get("nav_horizon_s", 0.0))
        self.nav_norm = float(mdp_obs.get("nav_norm_m", self.nav_horizon))
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
        # 관측 이력(POMDP): 정책 입력 = 최근 history 개 관측을 이어 붙인 것 [o_{t-k+1}, ..., o_t].
        # 한 장면만으로는 앞차가 지금 감속 중인지 알 수 없어서, 최근 몇 스텝을 함께 보여준다.
        # 데이터 기록(observation 칼럼)은 지금 시점 관측 o_t 하나만 저장한다 (이력은 앞 행들로 복원).
        self.obs_history = max(int(mdp_obs.get("history", 1)), 1)
        # 옆 차로 근거리 감지. getNeighbors/차로 스캔은 "같은 도로 조각(edge)" 기준이라
        # 도로 조각 경계·교차로 안에 걸친 옆 차를 놓치고, 나와 겹쳐 있는 차(간격 < 0)는 버린다.
        # 켜면 내 주변 side_scan_m 안의 차를 실제 위치(내 진행방향 기준 앞뒤·좌우)로 찾아
        # 옆 차로 앞/뒤 차에 함께 반영하고, 겹친 차는 간격 0 으로 본다.
        self.side_scan = bool(mdp_obs.get("side_scan", False))
        self.side_scan_m = float(mdp_obs.get("side_scan_m", 12.0))
        self._near_t = None
        self._near = []
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

        # SUMO 상태를 완전히 초기화하는 가장 확실한 방법은 껐다 켜는 것.
        if self._started:
            traci.close()
            self._started = False
        self._start_sumo()

        # 도로 전 구간에 배경차를 미리 배치 (첫 스텝부터 실제 교통류 상태)
        if self.prefill_enabled:
            self._prefill_traffic()
        # 배치된 차들이 자리를 잡도록 몇 스텝만 정착시킨다.
        # 신호는 고정 주기라, 매번 같은 시간만 채우면 출발할 때 신호 위상이 매 판 똑같다.
        # 채우는 시간을 판마다 ±warmup_jitter_steps 안에서 무작위로 바꿔 신호 위상도 판마다 달라지게 한다.
        n_warm = self.warmup_steps
        if self.warmup_jitter_steps > 0:
            n_warm = max(0, n_warm + int(np.random.randint(-self.warmup_jitter_steps,
                                                           self.warmup_jitter_steps + 1)))
        for _ in range(n_warm):
            traci.simulationStep()
        self.warmup_vehicle_count = traci.vehicle.getIDCount()   # 학습차 투입 직전 도로 위 일반차 수

        # ego 차량 투입 — 기본은 경로 시작(현대차 본사 정문, 0 m).
        # 학습 때만(random_start_prob > 0) 일부 판은 원래 경로 위 다른 지점에서 출발한다.
        self.start_edge = self._add_ego(random_start=(np.random.rand() < self.random_start_prob))

        # ── ego 삽입 보증 ──
        # 시작 차선 입구가 잠시 막혀 있으면 SUMO가 삽입을 "지연"시킨다. 그 경우 ego 없이
        # 에피소드가 진행되어 즉시 종료되는 버그가 되므로, 삽입될 때까지 (최대 30스텝) 기다린다.
        # 코스 중간 출발이 30스텝 안에 안 되면 원래 출발점(정문)으로 다시 넣는다.
        for attempt in range(2):
            traci.simulationStep()  # ego가 실제로 도로에 나타나도록 1스텝 진행
            _wait = 0
            while self.ego_id not in traci.vehicle.getIDList() and _wait < 30:
                traci.simulationStep()
                _wait += 1
            if self.ego_id in traci.vehicle.getIDList() or self.start_edge is None:
                break
            try:
                traci.vehicle.remove(self.ego_id)
            except traci.TraCIException:
                pass
            self.start_edge = self._add_ego(random_start=False)
        if self.ego_id not in traci.vehicle.getIDList():
            # 도로를 채운 뒤에는 정문 입구가 잠깐 막혀 있을 수 있다 → 교통을 새로 뽑아 다시 시작 (최대 5번)
            self._insert_retry = getattr(self, "_insert_retry", 0) + 1
            if self._insert_retry <= 5:
                return self.reset(seed=None, options=options)
            self._insert_retry = 0
            raise RuntimeError(
                "ego 삽입 실패: 도로 입구가 30스텝 이상 막혀 있습니다. "
                "TRAFFIC['vehs_per_hour']를 낮추거나 depart_speed를 확인하세요.")

        self._insert_retry = 0
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
        self.reroute_count = 0
        self._last_reroute_edge = None
        self._last_limit = None
        obs = self._get_obs()
        self._last_obs = obs
        if self.recorder is not None and self.recording:
            self.recorder.on_reset()
        return self._stacked(obs, reset=True), {}

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

    def _neighbor(self, offset: int, leader: bool):
        """오프셋 차선(-1=오른쪽, 0=현재, +1=왼쪽)의 가장 가까운
        선행(leader=True)/후행 차량을 (거리 m, 그 차량 속도)로 반환.
        visibility 밖이거나 없으면 None."""
        res = self._neighbor_lane(offset, leader)
        if self.side_scan and offset != 0:
            near = self._near_side(offset, leader)
            if near is not None and (res is None or near[0] < res[0]):
                res = near
        return res

    def _near_vehicles(self):
        """내 주변 side_scan_m 안의 차들을 (앞뒤 거리, 좌우 거리/차로폭, 속도, 차 길이)로.
        같은 시각에는 한 번만 계산한다. 반대 방향·교차 방향 차(진행방향 차이 45° 초과)는 뺀다."""
        t = traci.simulation.getTime()
        if self._near_t == t:
            return self._near
        out = []
        try:
            ex, ey = traci.vehicle.getPosition(self.ego_id)
            ea = traci.vehicle.getAngle(self.ego_id)
            hx, hy = np.sin(np.radians(ea)), np.cos(np.radians(ea))
            w = max(traci.lane.getWidth(traci.vehicle.getLaneID(self.ego_id)), 2.5)
            r = self.side_scan_m + 10.0
            for vid in traci.vehicle.getIDList():
                if vid == self.ego_id:
                    continue
                x, y = traci.vehicle.getPosition(vid)
                dx, dy = x - ex, y - ey
                if dx * dx + dy * dy > r * r:
                    continue
                da = (traci.vehicle.getAngle(vid) - ea + 180.0) % 360.0 - 180.0
                if abs(da) > 45.0:
                    continue
                lon = dx * hx + dy * hy            # + 앞
                lat = (dx * -hy + dy * hx) / w     # + 왼쪽, 차로 폭 단위
                out.append((lon, lat, traci.vehicle.getSpeed(vid), traci.vehicle.getLength(vid)))
        except traci.TraCIException:
            out = []
        self._near_t, self._near = t, out
        return out

    def _near_side(self, offset: int, leader: bool):
        """옆 차로(offset) 근거리 앞/뒤 차를 실제 위치로 찾는다. 겹치면 간격 0."""
        best = None
        ego_len = None
        for lon, lat, sp, ln in self._near_vehicles():
            if abs(lat - offset) > 0.5:
                continue
            if leader and lon >= 0:
                gap = max(lon - ln, 0.0)
            elif not leader and lon < 0:
                if ego_len is None:
                    ego_len = traci.vehicle.getLength(self.ego_id)
                gap = max(-lon - ego_len, 0.0)
            else:
                continue
            if gap > self.side_scan_m:
                continue
            if best is None or gap < best[0]:
                best = (gap, sp)
        return best

    def _neighbor_lane(self, offset: int, leader: bool):
        """도로 조각(edge)·차로 기준 이웃 (getLeader/getFollower/getNeighbors/차로 스캔)."""
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
            if res is None or res[0] > self.visibility or (res[0] < 0 and not self.side_scan):
                return None
            return max(float(res[0]), 0.0), float(res[1])
        else:
            mode_leader, mode_follower = self._NEIGHBOR_MODES[offset]
            pairs = traci.vehicle.getNeighbors(
                self.ego_id, mode_leader if leader else mode_follower)
            if not pairs:
                return None
            vid, gap = min(pairs, key=lambda p: p[1])   # 가장 가까운 차량
            if gap < 0 and self.side_scan:
                gap = 0.0                                # 나와 겹친 옆 차

        if gap < 0 or gap > self.visibility:
            return None
        return float(gap), traci.vehicle.getSpeed(vid)

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
        return w / w.sum()

    def _add_ego(self, random_start: bool):
        """ego 를 넣고, 코스 중간에서 출발했으면 그 도로 id 를, 정문 출발이면 None 을 돌려준다."""
        start = None
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
                horizon = max(self.nav_horizon, self.nav_horizon_s * self._speed_limit())
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
                self._lc_impossible = lane_change != 0
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

            lane_change_applied = self._apply_lane_change(lane_change)

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
            if rem is not None and self._prev_rem is not None:
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
            if (ego_gap is not None and
                    ego_gap < float(self.rw["close_gap_threshold"])):
                reward -= float(self.rw["close_gap_penalty"])
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
                if not held_by_signal and not held_by_leader:
                    reward -= ip

            # ---- 막힘 페널티: 느린 앞차 뒤에 갇혀 있는 동안 매 스텝 감점 ----
            # "차선을 유지한 채 막혀 있기"와 "변경으로 탈출하기" 사이에
            # 즉각적인 보상 차등을 만들어 추월 학습 신호를 조밀하게 한다.
            # (자세한 설계 이유는 env/mdp_config.py의 REWARD 주석 참고)
            bp = float(self.rw.get("blocked_penalty", 0.0))
            if (bp > 0.0 and ego_gap is not None
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

        if self.step_count >= self.max_steps or reroute_end:
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
