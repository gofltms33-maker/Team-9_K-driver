"""학습 중 주행 데이터를 CSV 로 저장한다 (팀 데이터 칼럼 정의서 기준).

한 행 = 한 스텝 (0.5초, env/mdp_config.py 의 step_length).
학습(train.py --record)하는 동안 매 스텝 기록하고, 중간 평가 주행은 기록하지 않는다.

칼럼
  기본 19개   episode, t, observation, action_raw, action.*, reward, terminated, truncated,
              state, detected_vehicles, event.*, next_observation, next_state
  상황 1 (추월)            s1.* 5개
  상황 2 (차로 감소·합류)  s2.* 5개
  상황 3 (대기줄 진입)     s3.* 6개
  상황 4 (회전교차로)      s4.* 3개
  끼어들기 실패            s1/s2/s3.cutin_failed — 그 상황에서 목표 차로 쪽으로 차로 변경을 명령했는데
                           안전 판정에 막혀 실행되지 않았으면 True, 실행됐거나 시도 안 했으면 False,
                           그 상황이 아니면 빈칸
  사건                     event.lane_change_rejected (차로 변경 명령이 실행 안 됨, 상황 무관)
                           event.rerouted (좌회전·출구 차로를 놓쳐 경로를 다시 찾음 = 우회)
                           event.red_light_violation (빨간불에 정지선 통과)
  추가                     sim_time_s (시뮬레이션 시각, 초)

저장 형식: 압축 CSV (data_partNNNN.csv.gz). pandas.read_csv 로 바로 읽힌다.
  에피소드마다 압축 블록을 닫아서, 학습이 중간에 꺼져도 그때까지의 에피소드는 온전히 남는다.

참고
  - action_raw / action.* 는 정책이 낸 명령이다 (오프라인 RL 에 쓰는 값).
    SUMO 가 실제로 낸 가속도는 s2.ego_acceleration_mps2 에 있다.
  - state / next_state 는 ego 반경 NEAR_RADIUS(기본 100 m) 안의 차량만 담는다
    (전 차량을 매 스텝 담으면 실험 하나가 수십 GB 가 된다).
    차량마다 id·종류(vType)·위치·속도·가속도·차로·차로 위치가 있어서,
    같은 id 를 시간순으로 따라가면 보류한 평가 칼럼(뒤차 최대 감속도, 합류 소요 시간,
    회전교차로 대기 시간 등)을 학습이 끝난 뒤 계산할 수 있다.
  - 해당 상황·목표 차로·대상 차량이 없으면 빈칸, 실제 값이 0 이면 0.
  - 거리·간격은 관측과 같은 SUMO 유효 간격(최소 간격 제외)이며 앞뒤 차는 50 m 안에서 찾는다.
  - 기록 중 오류가 나도 학습은 멈추지 않는다 (해당 칸만 빈칸, 처음 몇 번만 경고 출력).
"""
import csv
import gzip
import json
import math
import os

import traci.constants as tc
from env.sumo_env import traci          # 환경과 같은 SUMO 연결 방식 (TraCI 또는 libsumo)

NEAR_RADIUS = 100.0      # state 에 담을 주변 차량 반경 (m)
LOOKAHEAD = 250.0        # 상황 1~4 목표 지점을 찾는 거리 (m)
QUEUE_ZONE = 100.0       # 상황 3: 목표 지점 상류 대기 구간 (m)
QUEUE_SPEED = 0.5        # 상황 3: 이 속도(m/s) 이하면 대기 차량
CIRC_RANGE = 80.0        # 상황 4: 회전 차량을 찾는 경로상 거리 (m)
SLOW_MARGIN = 2.0        # 상황 1: 앞차가 허용 속도보다 이만큼(m/s) 느리면 추월 검토

COLUMNS = [
    "episode", "t", "observation", "action_raw",
    "action.accel_raw", "action.lane_change_raw", "action.lane_change", "action.lane_change_applied",
    "reward", "terminated", "truncated", "state", "detected_vehicles",
    "event.collided", "event.arrived", "event.lane_before", "event.lane_after",
    "next_observation", "next_state",
    "s1.leader_gap_m", "s1.leader_relative_speed_mps", "s1.target_front_gap_m",
    "s1.target_rear_gap_m", "s1.target_rear_relative_speed_mps",
    "s2.lane_end_distance_m", "s2.target_front_gap_m", "s2.target_rear_gap_m",
    "s2.target_rear_relative_speed_mps", "s2.ego_acceleration_mps2",
    "s3.target_distance_m", "s3.lane_changes_remaining", "s3.queue_vehicle_count",
    "s3.target_front_gap_m", "s3.target_rear_gap_m", "s3.target_rear_relative_speed_mps",
    "s4.entry_distance_m", "s4.circulating_distance_m", "s4.circulating_speed_mps",
    "s1.cutin_failed", "s2.cutin_failed", "s3.cutin_failed",
    "event.lane_change_rejected", "event.rerouted", "event.red_light_violation",
    "sim_time_s",
]
S_COLS = [c for c in COLUMNS if c[:2] in ("s1", "s2", "s3", "s4") and not c.endswith("cutin_failed")]

_CTX_VARS = [tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_ACCELERATION, tc.VAR_LANE_INDEX,
             tc.VAR_LANE_ID, tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION, tc.VAR_TYPE]


def _r(x, nd=3):
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else round(x, nd)
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    try:                                    # numpy 값
        return round(float(x), nd)
    except (TypeError, ValueError):
        return x


def _js(x, nd=3):
    return json.dumps(_r(x, nd), ensure_ascii=False, separators=(",", ":"))


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".") if v == v else ""
    return v


class _Net:
    """정적 도로망 정보 (sumolib). 차로 연결, 차로가 합류로 사라지는지, 회전교차로 edge."""

    def __init__(self, net_path):
        import sumolib
        self.net = sumolib.net.readNet(net_path)
        self.prio = {e.getID(): e.getPriority() for e in self.net.getEdges()}
        incoming = {}
        for e in self.net.getEdges():
            for ln in e.getLanes():
                for c in ln.getOutgoing():
                    incoming.setdefault(c.getToLane().getID(), []).append(c.getFromLane())
        self.incoming = incoming
        ring = set()
        for rb in self.net.getRoundabouts():
            for e in rb.getEdges():
                ring.add(e if isinstance(e, str) else e.getID())
        self.ring = ring

    def lane(self, lane_id):
        return self.net.getLane(lane_id)

    def loses(self, from_lane, to_lane):
        """from_lane 이 to_lane 으로 들어갈 때 '사라지는 쪽'인지 (차로 감소·합류).
        같은 차로로 여러 차로가 모이면: 우선순위 높은 도로 쪽이 이어지고,
        같은 도로끼리면 차로 번호가 덜 바뀌는 쪽이 이어진다."""
        def key(l):
            return (-self.prio[l.getEdge().getID()], abs(l.getIndex() - to_lane.getIndex()))
        mine = key(from_lane)
        return any(key(o) < mine for o in self.incoming.get(to_lane.getID(), [])
                   if o.getID() != from_lane.getID())


class StepRecorder:
    def __init__(self, env, out_dir, rows_per_file=50_000, start_episode=0, start_part=0):
        self.env = env
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.rows_per_file = rows_per_file
        net_path = self._net_path(env.cfg_path)
        self.netinfo = _Net(net_path)
        self.episode = start_episode - 1        # 이어서 학습하면 그 다음 번호부터
        self.t = 0
        self.active = False
        self._file = None
        self._writer = None
        self._path = None
        self._rows_in_file = 0
        self._part = start_part                 # 이어서 학습하면 새 파일 번호부터
        self._warned = 0
        self.total_rows = 0
        self._state_t = None
        self._pre = None

    # ── 파일 ─────────────────────────────────────────────
    @staticmethod
    def _net_path(cfg_path):
        import re
        with open(cfg_path, encoding="utf-8") as f:
            m = re.search(r'<net-file value="([^"]+)"', f.read())
        return os.path.join(os.path.dirname(os.path.abspath(cfg_path)), m.group(1))

    def _new_part(self):
        self._path = os.path.join(self.out_dir, f"data_part{self._part:04d}.csv.gz")
        self._part += 1
        self._rows_in_file = 0
        with gzip.open(self._path, "wt", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(COLUMNS)

    def _begin_episode_block(self):
        self._file = gzip.open(self._path, "at", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file)

    def _end_episode_block(self):
        if self._file is not None:
            self._file.close()               # 에피소드마다 압축 블록을 닫는다 (중간에 꺼져도 안전)
            self._file = None
            self._writer = None

    def close(self):
        self._end_episode_block()

    def _warn(self, where, e):
        if self._warned < 5:
            print(f"[데이터 기록] {where} 계산 실패 (그 칸만 빈칸): {e!r}")
        self._warned += 1

    # ── 에피소드 / 스텝 ───────────────────────────────────
    def on_reset(self):
        self._end_episode_block()            # 평가 등으로 끊긴 이전 에피소드 정리
        if self._path is None or self._rows_in_file >= self.rows_per_file:
            self._new_part()                     # 파일은 에피소드 경계에서만 나눈다
        self._begin_episode_block()
        self.episode += 1
        self.t = 0
        self.active = True
        try:
            traci.vehicle.subscribeContext(self.env.ego_id, tc.CMD_GET_VEHICLE_VARIABLE,
                                           NEAR_RADIUS, _CTX_VARS)
        except traci.TraCIException as e:
            self._warn("주변 차량 구독", e)
        try:
            self._state_t = self._state(first=True)
        except Exception as e:                       # noqa: BLE001
            self._warn("state", e)
            self._state_t = {"ego_alive": False}

    def before_step(self, obs_t):
        if not self.active:
            return
        self._pre = {"obs": obs_t if obs_t is not None else [], "detected": [],
                     "s": dict.fromkeys(S_COLS), "time": traci.simulation.getTime()}
        try:
            self._pre["detected"] = self.env.get_detected_vehicles()
        except Exception as e:                       # noqa: BLE001
            self._warn("detected_vehicles", e)
        if self.env.ego_id in traci.vehicle.getIDList():
            for fn in (self._s1, self._s2, self._s3, self._s4):
                part = {}
                try:
                    fn(part)                         # 계산된 칸까지는 남는다
                except Exception as e:               # noqa: BLE001
                    self._warn(fn.__name__, e)
                self._pre["s"].update(part)

    def after_step(self, action, reward, terminated, truncated, info, next_obs):
        if not self.active or self._pre is None:
            return
        try:
            self._write(action, reward, terminated, truncated, info, next_obs)
        except Exception as e:                       # noqa: BLE001  기록 실패로 학습이 멈추지 않게
            self._warn("행 저장", e)
            self._pre = None
            self.t += 1
            if terminated or truncated:
                self._end_episode_block()
                self.active = False

    def _write(self, action, reward, terminated, truncated, info, next_obs):
        try:
            state_tp1 = self._state()
        except Exception as e:                       # noqa: BLE001
            self._warn("state", e)
            state_tp1 = {"ego_alive": False}
        p = self._pre
        row = {
            "episode": self.episode, "t": self.t,
            "observation": _js(list(map(float, p["obs"])), 4),
            "action_raw": _js(list(map(float, action[:2])), 4),
            "action.accel_raw": float(info["accel_raw"]),
            "action.lane_change_raw": float(info["lane_change_raw"]),
            "action.lane_change": int(info["lane_change"]),
            "action.lane_change_applied": bool(info["lane_change_applied"]),
            "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated),
            "state": _js(self._state_t), "detected_vehicles": _js(p["detected"]),
            "event.collided": bool(info["collided"]), "event.arrived": bool(info["arrived"]),
            "event.lane_before": info["lane_before"], "event.lane_after": info["lane_after"],
            "next_observation": _js(list(map(float, next_obs)), 4),
            "next_state": _js(state_tp1),
            "sim_time_s": p["time"],
            "event.rerouted": bool(info.get("rerouted", False)),
            "event.red_light_violation": bool(info.get("red_light_violation", False)),
        }
        row.update(p["s"])
        lc, applied = int(info["lane_change"]), bool(info["lane_change_applied"])
        row["event.lane_change_rejected"] = bool(lc != 0 and not applied)
        for n in (1, 2, 3):                          # 상황별 끼어들기 실패
            d = p["s"].get(f"_s{n}_dir")
            if d is not None:
                row[f"s{n}.cutin_failed"] = bool(d != 0 and lc == d and not applied)
        self._writer.writerow([_cell(row.get(c)) for c in COLUMNS])
        self._rows_in_file += 1
        self.total_rows += 1
        self._state_t = state_tp1
        self._pre = None
        self.t += 1
        if terminated or truncated:
            self._end_episode_block()
            self.active = False

    # ── state (ego 주변 NEAR_RADIUS) ─────────────────────
    def _state(self, first=False):
        ego = self.env.ego_id
        if ego not in traci.vehicle.getIDList():
            return {"ego_alive": False}
        res = {} if first else (traci.vehicle.getContextSubscriptionResults(ego) or {})
        if not res:                                  # 구독 결과가 아직 없을 때(에피소드 첫 스텝)
            ex, ey = traci.vehicle.getPosition(ego)
            for vid in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition(vid)
                if math.hypot(x - ex, y - ey) <= NEAR_RADIUS:
                    res[vid] = {tc.VAR_POSITION: (x, y), tc.VAR_SPEED: traci.vehicle.getSpeed(vid),
                                tc.VAR_ACCELERATION: traci.vehicle.getAcceleration(vid),
                                tc.VAR_LANE_INDEX: traci.vehicle.getLaneIndex(vid),
                                tc.VAR_LANE_ID: traci.vehicle.getLaneID(vid),
                                tc.VAR_ROAD_ID: traci.vehicle.getRoadID(vid),
                                tc.VAR_LANEPOSITION: traci.vehicle.getLanePosition(vid),
                                tc.VAR_TYPE: traci.vehicle.getTypeID(vid)}
        e = res.get(ego)
        if e is None:
            return {"ego_alive": False}
        vehicles = []
        for vid, v in res.items():
            if vid == ego:
                continue
            vehicles.append({"id": vid, "type": v[tc.VAR_TYPE],
                             "x": v[tc.VAR_POSITION][0], "y": v[tc.VAR_POSITION][1],
                             "speed": v[tc.VAR_SPEED], "accel": v[tc.VAR_ACCELERATION],
                             "lane": v[tc.VAR_LANE_INDEX], "lane_id": v[tc.VAR_LANE_ID],
                             "road_id": v[tc.VAR_ROAD_ID], "lane_pos": v[tc.VAR_LANEPOSITION]})
        return {"ego_alive": True,
                "ego_x": e[tc.VAR_POSITION][0], "ego_y": e[tc.VAR_POSITION][1],
                "ego_speed": e[tc.VAR_SPEED], "ego_accel": e[tc.VAR_ACCELERATION],
                "ego_lane": e[tc.VAR_LANE_INDEX], "ego_lane_id": e[tc.VAR_LANE_ID],
                "ego_road_id": e[tc.VAR_ROAD_ID], "ego_lane_pos": e[tc.VAR_LANEPOSITION],
                "vehicles": vehicles}

    # ── 공통 ─────────────────────────────────────────────
    def _gaps(self, d):
        """오프셋 d 차로의 (앞차 간격, 뒤차 간격, 뒤차 상대속도) — 50 m 안, 없으면 None."""
        v = traci.vehicle.getSpeed(self.env.ego_id)
        f = self.env._neighbor(d, leader=True)
        r = self.env._neighbor(d, leader=False)
        return (f[0] if f else None, r[0] if r else None, (r[1] - v) if r else None)

    def _where(self):
        ego = self.env.ego_id
        edge = traci.vehicle.getRoadID(ego)
        if edge.startswith(":"):
            return None
        route = traci.vehicle.getRoute(ego)
        k = traci.vehicle.getRouteIndex(ego)
        if k < 0 or k >= len(route) or route[k] != edge:
            return None
        return edge, traci.vehicle.getLaneIndex(ego), traci.vehicle.getLanePosition(ego), route, k

    def _walk_lane(self, edge, idx, pos, route, k):
        """차로 idx 를 경로를 따라 LOOKAHEAD 까지 따라간다.
        반환: ("end", 거리)       그 지점에서 차로가 끊기거나 합류로 사라짐
              ("offroute", 거리)  그 지점에서 경로가 다른 차로로만 이어짐
              ("ok", None)        LOOKAHEAD 안에서 이어짐 (또는 목적지 도달)"""
        N = self.netinfo
        lane = N.lane(f"{edge}_{idx}")
        dist = lane.getLength() - pos
        while dist <= LOOKAHEAD:
            if k + 1 >= len(route):
                return ("ok", None)
            outs = lane.getOutgoing()
            if not outs:
                return ("end", dist)
            conns = [c for c in outs if c.getTo().getID() == route[k + 1]]
            if not conns:
                return ("offroute", dist)
            to = conns[0].getToLane()
            if N.loses(lane, to):
                return ("end", dist)
            lane = to
            dist += lane.getLength()
            k += 1
        return ("ok", None)

    # ── 상황 1: 추월 ─────────────────────────────────────
    def _s1(self, out):
        ego = self.env.ego_id
        res = traci.vehicle.getLeader(ego, self.env.visibility)
        if not res or res[1] < 0 or res[1] > self.env.visibility:
            return out                               # 50 m 안에 앞차 없음 (getLeader 는 더 멀리도 돌려줌)
        lid, gap = res
        v = traci.vehicle.getSpeed(ego)
        lv = traci.vehicle.getSpeed(lid)
        out["s1.leader_gap_m"] = gap
        out["s1.leader_relative_speed_mps"] = lv - v
        if lv >= traci.vehicle.getAllowedSpeed(lid) - SLOW_MARGIN:
            return out
        w = self._where()
        if w is None:
            return out
        edge, idx, pos, route, k = w
        n = traci.edge.getLaneNumber(edge)
        for d in (+1, -1):                           # 왼쪽 우선, 안 되면 오른쪽
            j = idx + d
            if 0 <= j < n and self._walk_lane(edge, j, pos, route, k)[0] == "ok":
                out["_s1_dir"] = d
                fg, rg, rv = self._gaps(d)
                out["s1.target_front_gap_m"], out["s1.target_rear_gap_m"] = fg, rg
                out["s1.target_rear_relative_speed_mps"] = rv
                break
        return out

    # ── 상황 2: 차로 감소·합류 ───────────────────────────
    def _s2(self, out):
        out["s2.ego_acceleration_mps2"] = traci.vehicle.getAcceleration(self.env.ego_id)
        w = self._where()
        if w is None:
            return out
        edge, idx, pos, route, k = w
        kind, dist = self._walk_lane(edge, idx, pos, route, k)
        if kind != "end":
            return out
        out["s2.lane_end_distance_m"] = dist
        n = traci.edge.getLaneNumber(edge)
        for j in sorted((j for j in range(n) if j != idx), key=lambda j: abs(j - idx)):
            kj, dj = self._walk_lane(edge, j, pos, route, k)
            if kj == "ok" or (kj == "end" and dj > dist + 1.0):
                out["_s2_dir"] = (j > idx) - (j < idx)
                if abs(j - idx) <= 2:
                    fg, rg, rv = self._gaps(j - idx)
                    out["s2.target_front_gap_m"], out["s2.target_rear_gap_m"] = fg, rg
                    out["s2.target_rear_relative_speed_mps"] = rv
                break
        return out

    # ── 상황 3: 대기줄이 있는 차로로 진입 (좌회전·출구) ─
    def _s3(self, out):
        w = self._where()
        if w is None:
            return out
        edge, idx, pos, route, k0 = w
        ego = self.env.ego_id
        N = self.netinfo
        k = k0
        while k + 1 < len(route):
            e = N.net.getEdge(route[k])
            if k == k0:                              # 지금 edge: 남은 차로 길이 (정지선에 서 있으면 0)
                d_end = max(traci.lane.getLength(f"{edge}_{idx}") - pos, 0.0)
            else:
                d_end = traci.vehicle.getDrivingDistance(ego, route[k], max(e.getLength() - 0.1, 0.0))
            if d_end < 0 or d_end > LOOKAHEAD:
                return out
            nxt = route[k + 1]
            lanes = e.getLanes()
            S = [ln.getIndex() for ln in lanes
                 if any(c.getTo().getID() == nxt for c in ln.getOutgoing())]
            succ = {c.getTo().getID() for ln in lanes for c in ln.getOutgoing()}
            dirs = {c.getDirection() for ln in lanes for c in ln.getOutgoing()
                    if c.getTo().getID() == nxt}
            is_exit = N.prio.get(nxt, 0) < N.prio.get(route[k], 0)
            is_left = bool(dirs & {"l", "L", "t"})
            if S and len(S) < len(lanes) and len(succ) >= 2 and (is_left or is_exit):
                out["s3.target_distance_m"] = d_end
                if k == k0:
                    near = min(S, key=lambda j: abs(j - idx))
                    need = near - idx
                else:
                    rl = self.env._route_lanes() or {}
                    need = rl.get(idx, (0, 0, 0))[1]
                out["s3.lane_changes_remaining"] = abs(need)
                cnt = 0
                for j in S:
                    lid = f"{route[k]}_{j}"
                    L = traci.lane.getLength(lid)
                    for vid in traci.lane.getLastStepVehicleIDs(lid):
                        if vid != ego and traci.vehicle.getLanePosition(vid) >= L - QUEUE_ZONE \
                                and traci.vehicle.getSpeed(vid) <= QUEUE_SPEED:
                            cnt += 1
                out["s3.queue_vehicle_count"] = cnt
                d = (need > 0) - (need < 0)
                out["_s3_dir"] = d
                fg, rg, rv = self._gaps(d)
                out["s3.target_front_gap_m"], out["s3.target_rear_gap_m"] = fg, rg
                out["s3.target_rear_relative_speed_mps"] = rv
                return out
            k += 1
        return out

    # ── 상황 4: 회전교차로 진입 ──────────────────────────
    def _s4(self, out):
        N = self.netinfo
        if not N.ring:
            return out
        ego = self.env.ego_id
        edge = traci.vehicle.getRoadID(ego)
        if edge in N.ring:                           # 이미 회전 중
            return out
        route = traci.vehicle.getRoute(ego)
        k = max(traci.vehicle.getRouteIndex(ego), 0)
        entry = next((e for e in route[k:] if e in N.ring), None)
        if entry is None:
            return out
        d = traci.vehicle.getDrivingDistance(ego, entry, 0.0)
        if d < 0 or d > LOOKAHEAD:
            return out
        out["s4.entry_distance_m"] = d
        best = None
        for re_ in N.ring:
            for vid in traci.edge.getLastStepVehicleIDs(re_):
                if vid == ego:
                    continue
                dd = traci.vehicle.getDrivingDistance(vid, entry, 0.0)
                if 0.0 <= dd <= CIRC_RANGE and (best is None or dd < best[0]):
                    best = (dd, traci.vehicle.getSpeed(vid))
        if best:
            out["s4.circulating_distance_m"], out["s4.circulating_speed_mps"] = best
        return out


def resume_point(out_dir):
    """기존 데이터 폴더에서 (다음 에피소드 번호, 다음 파일 번호)를 찾는다."""
    import glob
    import re
    parts = sorted(glob.glob(os.path.join(out_dir, "data_part*.csv.gz")))
    if not parts:
        return 0, 0
    nxt_part = int(re.search(r"data_part(\d+)", parts[-1]).group(1)) + 1
    last_ep = -1
    csv.field_size_limit(2 ** 31 - 1)
    for path in reversed(parts):
        try:
            with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
                for row in csv.reader(f):
                    if row and row[0].isdigit():
                        last_ep = max(last_ep, int(row[0]))
        except (EOFError, OSError):          # 강제 종료로 끝이 잘린 파일: 읽힌 데까지
            pass
        if last_ep >= 0:
            break
    return last_ep + 1, nxt_part
