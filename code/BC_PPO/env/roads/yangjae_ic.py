"""양재IC (경부고속도로 × 양재대로) SUMO 도로망 생성기.

road_builder.build() 가 ROAD["layout"] == "yangjae_ic" 일 때 호출한다.
OpenStreetMap 실측 좌표(env/roads/yangjae_data.py)를 바탕으로 아래 파일을 만든다.

    highway.nod.xml / highway.edg.xml / highway.con.xml  → netconvert 입력
    highway.rou.xml                                     → 차량 타입 + 경로 + 교통류
    highway.scenery.xml                                 → 도로 테두리, 중앙분리대, 다리 난간,
                                                           지하차도 덮개, 건물, 잔디, 나무
    highway.gui.xml                                     → 검은 도로 / 입체(z) 표시 설정

구조 요약 (실제 양재IC와 동일한 클로버형):
    ┌ 경부고속도로(5차로×2)가 '다리'로 양재대로 위를 지나간다 (z = +8 m)
    ├ 양재대로 본선(2차로×2)은 양 끝에서 '지하차도'로 내려간다 (z = -7 m)
    │     서쪽: 매헌지하차도   동쪽: 염곡동서지하차도
    ├ 양재대로 측도(2차로×2) + 동행 C-D 도로가 루프/램프를 받는다
    ├ 루프 4개 (좌회전) + 바깥 램프 4개 (우회전)
    └ 서울만남의광장 휴게소 → 경부 상행 '1차로 지하차도' (경부 밑을 통과)

높이 표현: 모든 노드/엣지에 실제 z 좌표를 넣고, GUI를 trueZ(입체 순서) 모드로
띄운다. 그래서 다리 밑을 지나는 차는 다리에 가려지고, 지하차도 안의 차는
덮개 밑으로 사라진다. (평면 교차가 없으므로 서로 부딪히는 일도 없다.)

좌표: 위도 37.4636 / 경도 127.0394 원점 기준 미터 (x=동, y=북).
"""
import math
import os
import random

from env.roads import yangjae_data as D

# ──────────────────────────────────────────────────────────────
# 높이 (z). SUMO-GUI는 교차로(junction) 면을 z=4에 그리므로 지면을 5로 둬야
# 지면 위 차량(z+1=6)이 교차로 면에 가리지 않는다.
# ──────────────────────────────────────────────────────────────
Z_GROUND = 5.0
Z_BRIDGE = 13.0     # 경부 교량 상판 (지면 + 8 m)
Z_TUNNEL = -2.0     # 지하차도 바닥 (지면 - 7 m)

# 차로 폭 (m)
LW_G = 3.5          # 경부고속도로
LW_Y = 3.2          # 양재대로 본선/측도
LW_CD = 3.4         # C-D 도로
LW_R = 3.5          # 램프/루프

# 제한속도 (m/s). 경부 본선은 road_config의 ROAD["speed_limit"]를 쓴다.
V_YMAIN = 19.44     # 70 km/h
V_FRONT = 16.67     # 60 km/h
V_RAMP = 13.89      # 50 km/h
V_LOOP = 11.11      # 40 km/h

# 경부 본선 구간 (t: 교차 중심에서 하행 방향 거리, m)
T_NORTH, T_SOUTH = -800.0, 1050.0
# 양재대로 구간 (s: 서쪽 끝에서 동쪽으로의 거리, m)
S_WEST, S_EAST = 0.0, 1800.0

# 기본 교통량 (대/시간). TRAFFIC["vehs_per_hour"] / 800 비율로 전체가 스케일된다.
# 키 = 경로 이름 (아래 ROUTES 참고). TRAFFIC["ic_flows"]로 개별 덮어쓰기 가능.
DEFAULT_FLOWS = {
    "sb_thru": 800,      # 경부 하행 직진  (= ego 경로 r0)
    "nb_thru": 800,      # 경부 상행 직진
    "eb_main": 400,      # 양재대로 동행 본선 (지하차도 2개 통과)
    "wb_main": 400,      # 양재대로 서행 본선
    "eb_front": 220,     # 양재대로 동행 측도 직진
    "wb_front": 220,     # 양재대로 서행 측도 직진
    "sb_to_wb": 120,     # 경부하행 → 양재대로 서행 (바깥 램프)
    "sb_to_eb": 100,     # 경부하행 → 양재대로 동행 (루프)
    "nb_to_eb": 120,     # 경부상행 → 양재대로 동행 (바깥 램프)
    "nb_to_wb": 100,     # 경부상행 → 양재대로 서행 (루프)
    "eb_to_sb": 120,     # 양재대로 동행 → 경부하행 (바깥 램프)
    "eb_to_nb": 100,     # 양재대로 동행 → 경부상행 (루프)
    "wb_to_nb": 200,     # 양재대로 서행 → 경부상행 (2차로 램프)
    "wb_to_sb": 100,     # 양재대로 서행 → 경부하행 (루프)
    "svc_to_nb": 60,     # 서울만남의광장 → 경부상행 (1차로 지하차도)
}


# ══════════════════════════════════════════════════════════════
# 기하 유틸
# ══════════════════════════════════════════════════════════════
def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _norm(vx, vy):
    L = math.hypot(vx, vy) or 1e-9
    return vx / L, vy / L


class _Poly:
    """2D 폴리라인: 거리 s 위치의 점과 접선."""

    def __init__(self, pts):
        self.p = [tuple(map(float, q)) for q in pts]
        self.cum = [0.0]
        for a, b in zip(self.p, self.p[1:]):
            self.cum.append(self.cum[-1] + math.dist(a, b))
        self.length = self.cum[-1]

    def at(self, s):
        s = min(max(s, 0.0), self.length)
        lo, hi = 0, len(self.cum) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.cum[mid] <= s:
                lo = mid
            else:
                hi = mid
        a, b = self.p[lo], self.p[hi]
        seg = self.cum[hi] - self.cum[lo] or 1e-9
        f = (s - self.cum[lo]) / seg
        tx, ty = _norm(b[0] - a[0], b[1] - a[1])
        return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, tx, ty


def _resample(pts, step):
    P = _Poly(pts)
    n = max(2, int(math.ceil(P.length / step)) + 1)
    return [P.at(P.length * i / (n - 1))[:2] for i in range(n)]


def _gauss_smooth(pts, sigma, step=1.0, anchor=0.0):
    """폴리라인을 호 길이 기준 가우시안으로 부드럽게 만든다.

    - 양 끝은 '점 대칭 패딩'으로 처리해서 끝점 위치와 진행방향이 유지된다.
    - anchor > 0 이면 양 끝 anchor m 안쪽은 원래 형상을 점점 더 따르게 해서
      본선 차로와 맞물리는 접점/접선이 정확히 보존된다.
    결과: 곡률이 연속적으로 변하는(꺾임 없는) 곡선.
    """
    P = _resample(pts, step)
    n = len(P)
    r = max(1, int(3.0 * sigma / step))
    pad_l = [(2 * P[0][0] - P[min(i, n - 1)][0], 2 * P[0][1] - P[min(i, n - 1)][1])
             for i in range(r, 0, -1)]
    pad_r = [(2 * P[-1][0] - P[max(n - 1 - i, 0)][0], 2 * P[-1][1] - P[max(n - 1 - i, 0)][1])
             for i in range(1, r + 1)]
    Q = pad_l + P + pad_r
    w = [math.exp(-0.5 * (k * step / sigma) ** 2) for k in range(-r, r + 1)]
    ws = sum(w)
    out = []
    for i in range(n):
        x = sum(w[k] * Q[i + k][0] for k in range(2 * r + 1)) / ws
        y = sum(w[k] * Q[i + k][1] for k in range(2 * r + 1)) / ws
        out.append((x, y))
    if anchor > 0:
        L = step * (n - 1)
        for i in range(n):
            f = _smooth(min(i * step, L - i * step) / anchor)
            out[i] = (P[i][0] + (out[i][0] - P[i][0]) * f, P[i][1] + (out[i][1] - P[i][1]) * f)
    return out


# 양재대로 기준축 — 원본 10 m 점열을 2 m 간격 + 가우시안(σ=25 m)으로 매끄럽게
_Y = _Poly(_gauss_smooth(D.Y_AXIS, 25.0, step=2.0))
_GO, _GD = D.G_ORIGIN, D.G_DIR
_GR = (_GD[1], -_GD[0])          # 경부 하행 진행방향의 오른쪽(남서)


def _gpos(t, off):
    return (_GO[0] + _GD[0] * t + _GR[0] * off,
            _GO[1] + _GD[1] * t + _GR[1] * off)


def _ypos(s, off):
    x, y, tx, ty = _Y.at(s)
    return x + ty * off, y - tx * off          # 오른쪽 법선 = (ty, -tx)


def _ydir(s):
    return _Y.at(s)[2:]


# ── 높이 프로파일 ──
def _z_g(t):
    """경부: 교차 중심 ±36 m 교량 상판, 이후 140 m에 걸쳐 지면으로."""
    a = abs(t)
    if a <= 36.0:
        return Z_BRIDGE
    return Z_BRIDGE - (Z_BRIDGE - Z_GROUND) * _smooth((a - 36.0) / 140.0)


# 양재대로 본선 지하차도: (내려가기 시작, 바닥 도달, 올라가기 시작, 지면 복귀)
TUNNELS_Y = [(10.0, 130.0, 570.0, 690.0),       # 매헌지하차도 (서)
             (1050.0, 1170.0, 1385.0, 1505.0)]  # 염곡동서지하차도 (동)


def _z_main(s):
    for a, b, c, d in TUNNELS_Y:
        if a <= s <= d:
            if s < b:
                return Z_GROUND + (Z_TUNNEL - Z_GROUND) * _smooth((s - a) / (b - a))
            if s <= c:
                return Z_TUNNEL
            return Z_TUNNEL + (Z_GROUND - Z_TUNNEL) * _smooth((s - c) / (d - c))
    return Z_GROUND


def _z_flat(_):
    return Z_GROUND


# ══════════════════════════════════════════════════════════════
# 차로군(carriageway) 정의
#   sign : 기준축 기준 어느 쪽 (+ = 경부 하행 오른쪽/양재 동행 오른쪽)
#   inner: 중앙(왼쪽) 경계까지의 거리. 차로는 바깥쪽으로 쌓인다
#          → 차로가 늘어나도(가속/감속차로) 왼쪽 차로는 제자리 유지
#   dir  : 진행방향이 축 방향이면 +1, 반대면 -1
# ══════════════════════════════════════════════════════════════
class _CW:
    def __init__(self, axis, sign, inner, lw, dirn, zf):
        self.axis, self.sign, self.inner, self.lw = axis, sign, inner, lw
        self.dir, self.zf = dirn, zf

    def lane_off(self, n, i):
        """n차로 엣지의 i번 차로(0=맨 오른쪽) 중심 offset."""
        return self.sign * (self.inner + (n - 1 - i + 0.5) * self.lw)

    def center_off(self, n):
        return self.sign * (self.inner + n * self.lw / 2.0)

    def pos(self, st, off):
        return _gpos(st, off) if self.axis == "G" else _ypos(st, off)

    def heading(self, st):
        if self.axis == "G":
            d = _GD
        else:
            d = _ydir(st)
        return d[0] * self.dir, d[1] * self.dir


CW = {
    "SB": _CW("G", +1, 3.0, LW_G, +1, _z_g),     # 경부 하행 (부산 방향)
    "NB": _CW("G", -1, 3.0, LW_G, -1, _z_g),     # 경부 상행 (서울 방향)
    "EBm": _CW("Y", +1, 2.2, LW_Y, +1, _z_main),  # 양재대로 동행 본선
    "WBm": _CW("Y", -1, 2.2, LW_Y, -1, _z_main),  # 양재대로 서행 본선
    "EBf": _CW("Y", +1, 12.4, LW_Y, +1, _z_flat),  # 동행 측도
    "CD": _CW("Y", +1, 20.4, LW_CD, +1, _z_flat),  # 동행 C-D (루프 엇갈림 구간)
    "WBf": _CW("Y", -1, 13.0, LW_Y, -1, _z_flat),  # 서행 측도 (= C-D 겸용)
}


# ══════════════════════════════════════════════════════════════
# 네트워크 구성
# ══════════════════════════════════════════════════════════════
class _Net:
    def __init__(self):
        self.nodes = {}      # id -> (x, y, z)
        self.edges = {}      # id -> dict
        self.cons = []       # (from, to, fromLane, toLane)

    def node(self, nid, x, y, z):
        self.nodes[nid] = (x, y, z)

    def edge(self, eid, frm, to, n, lw, speed, prio, name, shape, kind):
        self.edges[eid] = dict(frm=frm, to=to, n=n, lw=lw, speed=speed,
                               prio=prio, name=name, shape=shape, kind=kind)

    def con(self, frm, to, fl, tl):
        self.cons.append((frm, to, fl, tl))


def _cw_shape(cw, a, b, n, off_fn=None, step=2.0):
    """차로군 엣지 형상: 축 거리 a→b, n차로 중심선 (x,y,z) 목록."""
    L = abs(b - a)
    k = max(1, int(math.ceil(L / step)))
    pts = []
    for i in range(k + 1):
        st = a + (b - a) * i / k
        off = off_fn(st) if off_fn else cw.center_off(n)
        x, y = cw.pos(st, off)
        pts.append((x, y, cw.zf(st)))
    return pts


def _bezier(p0, h0, p3, h3, n=8, k=0.42):
    """3차 베지어. k = 핸들 길이 비율 (클수록 양 끝이 직선에 가깝게 → 급커브 방지)."""
    d = math.dist(p0, p3)
    c1 = (p0[0] + h0[0] * d * k, p0[1] + h0[1] * d * k)
    c2 = (p3[0] - h3[0] * d * k, p3[1] - h3[1] * d * k)
    out = []
    for i in range(n + 1):
        t = i / n
        a, b, c, e = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3
        out.append((a * p0[0] + b * c1[0] + c * c2[0] + e * p3[0],
                    a * p0[1] + b * c1[1] + c * c2[1] + e * p3[1]))
    return out


def _ramp_shape(key, A, hA, B, hB, zfun, tan_len=55.0):
    """OSM 램프 중심선을 시작점 A/끝점 B(실제 붙을 차로 위치)에 맞게 휘어 붙인다.

    1) 양 끝 변위를 램프 길이 방향으로 부드럽게 감쇠시키며 전체를 이동
    2) 양 끝 55 m 는 본선 진행방향과 접하도록 베지어 곡선으로 교체
    3) 전체를 다시 한 번 가우시안으로 다듬어 곡률이 연속이 되게 함
    zfun(frac) → 램프 위 위치(0~1)의 높이
    """
    # OSM 점은 듬성듬성해서 꺾임이 있다 → 먼저 σ=9 m 로 곡선화
    P = _gauss_smooth(D.RAMPS[key], 9.0, step=1.0)
    pl = _Poly(P)
    L = pl.length
    dA = (A[0] - P[0][0], A[1] - P[0][1])
    dB = (B[0] - P[-1][0], B[1] - P[-1][1])
    lb = min(90.0, 0.42 * L)
    moved = []
    for i, q in enumerate(P):
        s = pl.cum[i]
        ws = 1.0 - _smooth(s / lb)
        we = 1.0 - _smooth((L - s) / lb)
        moved.append((q[0] + dA[0] * ws + dB[0] * we, q[1] + dA[1] * ws + dB[1] * we))
    mp = _Poly(moved)
    L2 = mp.length
    tl = min(tan_len, L2 * 0.25)
    qa = mp.at(tl)
    qb = mp.at(L2 - tl)
    head = _bezier(A, hA, qa[:2], qa[2:], int(tl))
    tail = _bezier(qb[:2], qb[2:], B, hB, int(tl))
    mid = [q for q, s in zip(mp.p, mp.cum) if tl + 1.0 < s < L2 - tl - 1.0]
    xy = head + mid + tail[1:]
    # 이어붙인 이음매까지 한 번 더 매끄럽게 (양 끝 접점·접선은 고정)
    xy = _gauss_smooth(xy, 6.0, step=1.0, anchor=6.0)
    xy = _resample(xy, 1.5)
    pp = _Poly(xy)
    return [(q[0], q[1], zfun(s / pp.length)) for q, s in zip(xy, pp.cum)]


def _lin_z(z0, z1):
    return lambda f: z0 + (z1 - z0) * _smooth(f)


def build_network(speed_g):
    """노드/엣지/연결을 구성해 _Net 으로 반환."""
    net = _Net()
    SB, NB, EBm, WBm, EBf, CD, WBf = (CW[k] for k in
                                       ("SB", "NB", "EBm", "WBm", "EBf", "CD", "WBf"))

    def cw_node(nid, cw, st, n):
        x, y = cw.pos(st, cw.center_off(n))
        net.node(nid, x, y, cw.zf(st))

    def cw_edge(eid, cw, a, b, n, frm, to, speed, prio, name, kind, off_fn=None):
        net.edge(eid, frm, to, n, cw.lw, speed, prio, name,
                 _cw_shape(cw, a, b, n, off_fn), kind)

    def lane_pt(cw, st, n, lanes):
        """n차로 엣지에서 주어진 차로들의 중심(평균) 위치 (x, y)."""
        off = sum(cw.lane_off(n, i) for i in lanes) / len(lanes)
        return cw.pos(st, off)

    # ── 1. 경부 하행 (SB) ─────────────────────────────────────
    t_d1471, t_m7252, t_d1469, t_m7085 = -312.5, -58.9, 90.1, 706.7
    t_x1471 = t_d1471 - 170.0                 # 감속차로 시작
    t_drop = t_m7085 + 220.0                  # 가속차로 끝
    G = "경부고속도로"
    for nid, t, n in [("SB_N", T_NORTH, 5), ("SB_x1471", t_x1471, 5),
                      ("SB_d1471", t_d1471, 6), ("SB_m7252", t_m7252, 5),
                      ("SB_d1469", t_d1469, 6), ("SB_m7085", t_m7085, 5),
                      ("SB_drop", t_drop, 6), ("SB_S", T_SOUTH, 5)]:
        cw_node(nid, SB, t, n)
    sb = [("SB_a", T_NORTH, t_x1471, 5, "SB_N", "SB_x1471"),
          ("SB_b", t_x1471, t_d1471, 6, "SB_x1471", "SB_d1471"),
          ("SB_c", t_d1471, t_m7252, 5, "SB_d1471", "SB_m7252"),
          ("SB_w", t_m7252, t_d1469, 6, "SB_m7252", "SB_d1469"),
          ("SB_d", t_d1469, t_m7085, 5, "SB_d1469", "SB_m7085"),
          ("SB_e", t_m7085, t_drop, 6, "SB_m7085", "SB_drop"),
          ("SB_f", t_drop, T_SOUTH, 5, "SB_drop", "SB_S")]
    for eid, a, b, n, f, t in sb:
        cw_edge(eid, SB, a, b, n, f, t, speed_g, 10, G + " 하행", "G")

    # ── 2. 경부 상행 (NB) ─────────────────────────────────────
    t_msvc, t_d7083, t_m1470, t_d7086, t_m2907 = 417.2, 254.6, 55.0, -57.6, -317.2
    t_dropA, t_dropB = t_m2907 - 150.0, t_m2907 - 300.0
    for nid, t, n in [("NB_S", T_SOUTH, 5), ("NB_msvc", t_msvc, 5),
                      ("NB_d7083", t_d7083, 6), ("NB_m1470", t_m1470, 5),
                      ("NB_d7086", t_d7086, 6), ("NB_m2907", t_m2907, 5),
                      ("NB_dropA", t_dropA, 7), ("NB_dropB", t_dropB, 6),
                      ("NB_N", T_NORTH, 5)]:
        cw_node(nid, NB, t, n)
    nb = [("NB_a", T_SOUTH, t_msvc, 5, "NB_S", "NB_msvc"),
          ("NB_b", t_msvc, t_d7083, 6, "NB_msvc", "NB_d7083"),
          ("NB_c", t_d7083, t_m1470, 5, "NB_d7083", "NB_m1470"),
          ("NB_w", t_m1470, t_d7086, 6, "NB_m1470", "NB_d7086"),
          ("NB_d", t_d7086, t_m2907, 5, "NB_d7086", "NB_m2907"),
          ("NB_e", t_m2907, t_dropA, 7, "NB_m2907", "NB_dropA"),
          ("NB_f", t_dropA, t_dropB, 6, "NB_dropA", "NB_dropB"),
          ("NB_g", t_dropB, T_NORTH, 5, "NB_dropB", "NB_N")]
    for eid, a, b, n, f, t in nb:
        cw_edge(eid, NB, a, b, n, f, t, speed_g, 10, G + " 상행", "G")

    # ── 3. 양재대로 본선 (지하차도 구간은 별도 엣지로 분리) ──────
    s_m = [S_WEST, 690.0, 1050.0, 1505.0]
    s_merge, s_split = 1680.0, 1674.5
    for i, s in enumerate(s_m):
        cw_node(f"EBm_{i}", EBm, s, 2)
        cw_node(f"WBm_{i}", WBm, s, 2)
    cw_node("YE_merge", EBm, s_merge, 2)
    cw_node("YE_split", WBm, s_split, 2)
    ebm_bounds = s_m + [s_merge]
    for i in range(4):
        a, b = ebm_bounds[i], ebm_bounds[i + 1]
        to = f"EBm_{i + 1}" if i < 3 else "YE_merge"
        cw_edge(f"EBm_{i}", EBm, a, b, 2, f"EBm_{i}", to, V_YMAIN, 9,
                "양재대로 (본선)", "Ymain")
    wbm_bounds = [s_split, 1505.0, 1050.0, 690.0, S_WEST]
    wnodes = ["YE_split", "WBm_3", "WBm_2", "WBm_1", "WBm_0"]
    for i in range(4):
        cw_edge(f"WBm_{i}", WBm, wbm_bounds[i], wbm_bounds[i + 1], 2,
                wnodes[i], wnodes[i + 1], V_YMAIN, 9, "양재대로 (본선)", "Ymain")
    # 동쪽 끝: 측도+본선 합류 / 분기
    x, y = _ypos(S_EAST, 2.2 + 2 * LW_Y)
    net.node("YE_E_out", x, y, Z_GROUND)
    x, y = _ypos(S_EAST, -(2.2 + 2 * LW_Y))
    net.node("YE_W_in", x, y, Z_GROUND)
    cw_edge("E_out", EBm, s_merge, S_EAST, 4, "YE_merge", "YE_E_out", V_YMAIN, 9,
            "양재대로", "Ymain")
    cw_edge("W_in", WBm, S_EAST, s_split, 4, "YE_W_in", "YE_split", V_YMAIN, 9,
            "양재대로", "Ymain")
    net.con("W_in", "WBm_0", 2, 0)
    net.con("W_in", "WBm_0", 3, 1)
    for i in range(3):
        net.con(f"EBm_{i}", f"EBm_{i + 1}", 0, 0)
        net.con(f"EBm_{i}", f"EBm_{i + 1}", 1, 1)
        net.con(f"WBm_{i}", f"WBm_{i + 1}", 0, 0)
        net.con(f"WBm_{i}", f"WBm_{i + 1}", 1, 1)
    net.con("EBm_3", "E_out", 0, 2)
    net.con("EBm_3", "E_out", 1, 3)

    # ── 4. 양재대로 동행 측도 + C-D ───────────────────────────
    s_d7084, s_dCD, s_m1469, s_d1470, s_mCD, s_m7083 = 667.9, 739.4, 798.4, 931.5, 973.7, 1106.9
    s_ef1, s_ef2 = s_m7083 + 130.0, s_m7083 + 280.0
    Y_F = "양재대로 (측도)"
    for nid, s, n in [("EBf_W", S_WEST, 2), ("EBf_d7084", s_d7084, 2),
                      ("EBf_dCD", s_dCD, 2), ("EBf_mCD", s_mCD, 2),
                      ("EBf_m7083", s_m7083, 3), ("EBf_drop1", s_ef1, 4),
                      ("EBf_drop2", s_ef2, 3)]:
        cw_node(nid, EBf, s, n)

    def ebf_taper(n):
        # 동쪽 끝에서 본선 옆(오른쪽 2차로 자리)으로 붙도록 offset을 줄인다
        target = (CW["EBm"].lane_off(4, 0) + CW["EBm"].lane_off(4, 1)) / 2.0  # E_out 0,1번 차로

        def f(s):
            base = EBf.center_off(n)
            return base + (target - base) * _smooth((s - 1540.0) / 140.0)
        return f
    cw_edge("EBf_a", EBf, S_WEST, s_d7084, 2, "EBf_W", "EBf_d7084", V_FRONT, 7, Y_F, "Yfront")
    cw_edge("EBf_b", EBf, s_d7084, s_dCD, 2, "EBf_d7084", "EBf_dCD", V_FRONT, 7, Y_F, "Yfront")
    cw_edge("EBf_c", EBf, s_dCD, s_mCD, 2, "EBf_dCD", "EBf_mCD", V_FRONT, 7, Y_F, "Yfront")
    cw_edge("EBf_d", EBf, s_mCD, s_m7083, 3, "EBf_mCD", "EBf_m7083", V_FRONT, 7, Y_F, "Yfront")
    cw_edge("EBf_e", EBf, s_m7083, s_ef1, 4, "EBf_m7083", "EBf_drop1", V_FRONT, 7, Y_F, "Yfront")
    cw_edge("EBf_f", EBf, s_ef1, s_ef2, 3, "EBf_drop1", "EBf_drop2", V_FRONT, 7, Y_F, "Yfront")
    cw_edge("EBf_g", EBf, s_ef2, s_merge, 2, "EBf_drop2", "YE_merge", V_FRONT, 7, Y_F,
            "Yfront", off_fn=ebf_taper(2))
    # C-D: 측도 0번 차로에서 갈라져 바깥으로 벌어졌다가 다시 측도로 합류
    cd_lane0_from = EBf.lane_off(2, 0)
    cd1 = CD.center_off(1)

    def cd_a_off(s):
        return cd_lane0_from + (cd1 - cd_lane0_from) * _smooth((s - s_dCD) / (s_m1469 - s_dCD))

    cd_c_from = CD.lane_off(2, 1)
    cd_c_to = EBf.lane_off(3, 0)

    def cd_c_off(s):
        return cd_c_from + (cd_c_to - cd_c_from) * _smooth((s - s_d1470) / (s_mCD - s_d1470))

    x, y = _ypos(s_m1469, CD.center_off(1))
    net.node("CD_m1469", x, y, Z_GROUND)
    x, y = _ypos(s_d1470, CD.center_off(2))
    net.node("CD_d1470", x, y, Z_GROUND)
    cw_edge("CD_a", CD, s_dCD, s_m1469, 1, "EBf_dCD", "CD_m1469", V_FRONT, 6,
            "양재대로 (C-D)", "CD", off_fn=cd_a_off)
    cw_edge("CD_b", CD, s_m1469, s_d1470, 2, "CD_m1469", "CD_d1470", V_FRONT, 6,
            "양재대로 (C-D)", "CD")
    cw_edge("CD_c", CD, s_d1470, s_mCD, 1, "CD_d1470", "EBf_mCD", V_FRONT, 6,
            "양재대로 (C-D)", "CD", off_fn=cd_c_off)

    # ── 5. 양재대로 서행 측도 ──────────────────────────────────
    s_wx, s_d2907, s_m7086, s_d7252, s_m1471 = 1300.0, 1145.0, 955.7, 854.6, 687.5
    s_wdrop = s_m1471 - 130.0
    for nid, s, n in [("WBf_x", s_wx, 2), ("WBf_d2907", s_d2907, 3),
                      ("WBf_m7086", s_m7086, 2), ("WBf_d7252", s_d7252, 3),
                      ("WBf_m1471", s_m1471, 2), ("WBf_drop", s_wdrop, 3),
                      ("WBf_W", S_WEST, 2)]:
        cw_node(nid, WBf, s, n)

    def wbf_taper(s):
        start = (WBm.lane_off(4, 0) + WBm.lane_off(4, 1)) / 2.0
        base = WBf.center_off(2)
        return start + (base - start) * _smooth((s_split - s) / 140.0)
    cw_edge("WBf_a", WBf, s_split, s_wx, 2, "YE_split", "WBf_x", V_FRONT, 7, Y_F,
            "Yfront", off_fn=wbf_taper)
    cw_edge("WBf_b", WBf, s_wx, s_d2907, 3, "WBf_x", "WBf_d2907", V_FRONT, 7, Y_F, "Yfront")
    cw_edge("WBf_c", WBf, s_d2907, s_m7086, 2, "WBf_d2907", "WBf_m7086", V_FRONT, 7, Y_F, "Yfront")
    cw_edge("WBf_w", WBf, s_m7086, s_d7252, 3, "WBf_m7086", "WBf_d7252", V_FRONT, 7, Y_F, "Yfront")
    cw_edge("WBf_d", WBf, s_d7252, s_m1471, 2, "WBf_d7252", "WBf_m1471", V_FRONT, 7, Y_F, "Yfront")
    cw_edge("WBf_e", WBf, s_m1471, s_wdrop, 3, "WBf_m1471", "WBf_drop", V_FRONT, 7, Y_F, "Yfront")
    cw_edge("WBf_f", WBf, s_wdrop, S_WEST, 2, "WBf_drop", "WBf_W", V_FRONT, 7, Y_F, "Yfront")
    net.con("W_in", "WBf_a", 0, 0)
    net.con("W_in", "WBf_a", 1, 1)

    # ── 6. 램프 / 루프 ────────────────────────────────────────
    def ramp(eid, key, n, frm, to, A, hA, B, hB, zfun, speed, prio, name, kind):
        shape = _ramp_shape(key, A, hA, B, hB, zfun)
        net.edge(eid, frm, to, n, LW_R, speed, prio, name, shape, kind)

    zg = _z_g
    # (1) 경부하행 → 양재대로 서행 (바깥 램프 1471)
    ramp("R1471", "R1471", 1, "SB_d1471", "WBf_m1471",
         lane_pt(SB, t_d1471, 6, [0]), SB.heading(t_d1471),
         lane_pt(WBf, s_m1471, 3, [0]), WBf.heading(s_m1471),
         _lin_z(zg(t_d1471), Z_GROUND), V_RAMP, 4, "양재IC 램프", "ramp")
    # (2) 양재대로 서행 → 경부하행 (루프 7252, 서쪽 사분면)
    ramp("L7252", "L7252", 1, "WBf_d7252", "SB_m7252",
         lane_pt(WBf, s_d7252, 3, [0]), WBf.heading(s_d7252),
         lane_pt(SB, t_m7252, 6, [0]), SB.heading(t_m7252),
         _lin_z(Z_GROUND, zg(t_m7252)), V_LOOP, 3, "양재IC 루프", "loop")
    # (3) 양재대로 서행 → 경부상행 (2차로 램프 2907)
    ramp("R2907", "R2907", 2, "WBf_d2907", "NB_m2907",
         lane_pt(WBf, s_d2907, 3, [0, 1]), WBf.heading(s_d2907),
         lane_pt(NB, t_m2907, 7, [0, 1]), NB.heading(t_m2907),
         _lin_z(Z_GROUND, zg(t_m2907)), V_RAMP, 4, "양재IC 램프", "ramp")
    # (4) 경부상행 → 양재대로 서행 (루프 7086, 북쪽 사분면)
    ramp("L7086", "L7086", 1, "NB_d7086", "WBf_m7086",
         lane_pt(NB, t_d7086, 6, [0]), NB.heading(t_d7086),
         lane_pt(WBf, s_m7086, 3, [0]), WBf.heading(s_m7086),
         _lin_z(zg(t_d7086), Z_GROUND), V_LOOP, 3, "양재IC 루프", "loop")
    # (5) 경부상행 → 양재대로 동행 (바깥 램프 7083)
    ramp("R7083", "R7083", 1, "NB_d7083", "EBf_m7083",
         lane_pt(NB, t_d7083, 6, [0]), NB.heading(t_d7083),
         lane_pt(EBf, s_m7083, 4, [0]), EBf.heading(s_m7083),
         _lin_z(zg(t_d7083), Z_GROUND), V_RAMP, 4, "양재IC 램프", "ramp")
    # (6) 양재대로 동행(C-D) → 경부상행 (루프 1470, 동쪽 사분면)
    ramp("L1470", "L1470", 1, "CD_d1470", "NB_m1470",
         lane_pt(CD, s_d1470, 2, [0]), CD.heading(s_d1470),
         lane_pt(NB, t_m1470, 6, [0]), NB.heading(t_m1470),
         _lin_z(Z_GROUND, zg(t_m1470)), V_LOOP, 3, "양재IC 루프", "loop")
    # (7) 경부하행 → 양재대로 동행(C-D) (루프 1469, 남쪽 사분면)
    ramp("L1469", "L1469", 1, "SB_d1469", "CD_m1469",
         lane_pt(SB, t_d1469, 6, [0]), SB.heading(t_d1469),
         lane_pt(CD, s_m1469, 2, [0]), CD.heading(s_m1469),
         _lin_z(zg(t_d1469), Z_GROUND), V_LOOP, 3, "양재IC 루프", "loop")
    # (8) 양재대로 동행 → 경부하행 (바깥 램프 7084)
    ramp("R7084", "R7084", 1, "EBf_d7084", "SB_m7085",
         lane_pt(EBf, s_d7084, 2, [0]), EBf.heading(s_d7084),
         lane_pt(SB, t_m7085, 6, [0]), SB.heading(t_m7085),
         _lin_z(Z_GROUND, zg(t_m7085)), V_RAMP, 4, "양재IC 램프", "ramp")
    # (9) 서울만남의광장 → 경부상행: 1차로 지하차도로 경부 밑을 통과
    #     (터널 구간과 지상 구간을 나눠 지상 합류부가 지하 높이로 그려지지 않게 한다)
    raw = _gauss_smooth(D.RAMPS["RSVC"], 9.0, step=2.0)
    rp = _Poly(raw)
    split_l = rp.length * 0.86                 # 이 지점 이후는 지면
    sx, sy = rp.at(split_l)[:2]
    net.node("SVC_start", raw[0][0], raw[0][1], Z_GROUND)
    net.node("SVC_up", sx, sy, Z_GROUND)
    B = lane_pt(NB, t_msvc, 6, [0])
    full = _ramp_shape("RSVC", raw[0], _norm(raw[1][0] - raw[0][0], raw[1][1] - raw[0][1]),
                       B, NB.heading(t_msvc), lambda f: Z_GROUND, tan_len=35.0)
    fp = _Poly([q[:2] for q in full])
    Lf = fp.length
    # 높이: 35 m 지면 → 100 m 걸쳐 하강 → 지하(경부 밑) → 100 m 걸쳐 상승
    d0, d1 = 35.0, 135.0
    u0 = Lf * 0.86 - 100.0
    u1 = Lf * 0.86

    def z_svc(l):
        if l <= d0:
            return Z_GROUND
        if l < d1:
            return Z_GROUND + (Z_TUNNEL - Z_GROUND) * _smooth((l - d0) / (d1 - d0))
        if l <= u0:
            return Z_TUNNEL
        if l < u1:
            return Z_TUNNEL + (Z_GROUND - Z_TUNNEL) * _smooth((l - u0) / (u1 - u0))
        return Z_GROUND
    pts3 = [(q[0], q[1], z_svc(s)) for q, s in zip(fp.p, fp.cum)]
    cut = next(i for i, s in enumerate(fp.cum) if s >= u1)
    t_part = pts3[:cut + 1]
    g_part = pts3[cut:]
    net.nodes["SVC_up"] = t_part[-1]
    net.edge("RSVC_t", "SVC_start", "SVC_up", 1, LW_R, V_LOOP, 4,
             "만남의광장 지하차도", t_part, "svc")
    net.edge("RSVC_g", "SVC_up", "NB_msvc", 1, LW_R, V_RAMP, 4,
             "만남의광장 진입로", g_part, "svc")
    net.con("RSVC_t", "RSVC_g", 0, 0)

    # ── 7. 차로 연결 (모든 분기/합류를 명시 — netconvert 추측에 맡기지 않음) ──
    c = net.con

    def shift(frm, to, n_from, k, start=0):
        """frm의 차로 start..n_from-1 → to의 (i-start+k) 차로."""
        for i in range(start, n_from):
            c(frm, to, i, i - start + k)

    # SB
    c("SB_a", "SB_b", 0, 0); shift("SB_a", "SB_b", 5, 1)          # 감속차로 생성
    c("SB_b", "R1471", 0, 0); shift("SB_b", "SB_c", 6, 0, start=1)
    c("L7252", "SB_w", 0, 0); shift("SB_c", "SB_w", 5, 1)          # 루프 합류 → 엇갈림차로
    c("SB_w", "L1469", 0, 0); shift("SB_w", "SB_d", 6, 0, start=1)
    c("R7084", "SB_e", 0, 0); shift("SB_d", "SB_e", 5, 1)          # 가속차로
    shift("SB_e", "SB_f", 6, 0, start=1)                           # 0번 차로 소멸
    # NB
    c("RSVC_g", "NB_b", 0, 0); shift("NB_a", "NB_b", 5, 1)
    c("NB_b", "R7083", 0, 0); shift("NB_b", "NB_c", 6, 0, start=1)
    c("L1470", "NB_w", 0, 0); shift("NB_c", "NB_w", 5, 1)
    c("NB_w", "L7086", 0, 0); shift("NB_w", "NB_d", 6, 0, start=1)
    c("R2907", "NB_e", 0, 0); c("R2907", "NB_e", 1, 1); shift("NB_d", "NB_e", 5, 2)
    shift("NB_e", "NB_f", 7, 0, start=1)
    shift("NB_f", "NB_g", 6, 0, start=1)
    # EB 측도 / C-D
    c("EBf_a", "R7084", 0, 0); c("EBf_a", "EBf_b", 0, 0); c("EBf_a", "EBf_b", 1, 1)
    c("EBf_b", "CD_a", 0, 0); c("EBf_b", "EBf_c", 0, 0); c("EBf_b", "EBf_c", 1, 1)
    c("L1469", "CD_b", 0, 0); c("CD_a", "CD_b", 0, 1)              # C-D 엇갈림 시작
    c("CD_b", "L1470", 0, 0); c("CD_b", "CD_c", 1, 0)              # 엇갈림 끝
    c("CD_c", "EBf_d", 0, 0); shift("EBf_c", "EBf_d", 2, 1)
    c("R7083", "EBf_e", 0, 0); shift("EBf_d", "EBf_e", 3, 1)
    shift("EBf_e", "EBf_f", 4, 0, start=1)
    shift("EBf_f", "EBf_g", 3, 0, start=1)
    c("EBf_g", "E_out", 0, 0); c("EBf_g", "E_out", 1, 1)
    # WB 측도
    c("WBf_a", "WBf_b", 0, 0); c("WBf_a", "WBf_b", 0, 1); c("WBf_a", "WBf_b", 1, 2)
    c("WBf_b", "R2907", 0, 0); c("WBf_b", "R2907", 1, 1)
    c("WBf_b", "WBf_c", 1, 0); c("WBf_b", "WBf_c", 2, 1)
    c("L7086", "WBf_w", 0, 0); shift("WBf_c", "WBf_w", 2, 1)
    c("WBf_w", "L7252", 0, 0); shift("WBf_w", "WBf_d", 3, 0, start=1)
    c("R1471", "WBf_e", 0, 0); shift("WBf_d", "WBf_e", 2, 1)
    shift("WBf_e", "WBf_f", 3, 0, start=1)
    return net


# 경로 (엣지 나열). r0 = ego 경로 = 경부 하행 직진.
ROUTES = {
    "sb_thru": "SB_a SB_b SB_c SB_w SB_d SB_e SB_f",
    "nb_thru": "NB_a NB_b NB_c NB_w NB_d NB_e NB_f NB_g",
    "eb_main": "EBm_0 EBm_1 EBm_2 EBm_3 E_out",
    "wb_main": "W_in WBm_0 WBm_1 WBm_2 WBm_3",
    "eb_front": "EBf_a EBf_b EBf_c EBf_d EBf_e EBf_f EBf_g E_out",
    "wb_front": "W_in WBf_a WBf_b WBf_c WBf_w WBf_d WBf_e WBf_f",
    "sb_to_wb": "SB_a SB_b R1471 WBf_e WBf_f",
    "sb_to_eb": "SB_a SB_b SB_c SB_w L1469 CD_b CD_c EBf_d EBf_e EBf_f EBf_g E_out",
    "nb_to_eb": "NB_a NB_b R7083 EBf_e EBf_f EBf_g E_out",
    "nb_to_wb": "NB_a NB_b NB_c NB_w L7086 WBf_w WBf_d WBf_e WBf_f",
    "eb_to_sb": "EBf_a R7084 SB_e SB_f",
    "eb_to_nb": "EBf_a EBf_b CD_a CD_b L1470 NB_w NB_d NB_e NB_f NB_g",
    "wb_to_nb": "W_in WBf_a WBf_b R2907 NB_e NB_f NB_g",
    "wb_to_sb": "W_in WBf_a WBf_b WBf_c WBf_w L7252 SB_w SB_d SB_e SB_f",
    "svc_to_nb": "RSVC_t RSVC_g NB_b NB_c NB_w NB_d NB_e NB_f NB_g",
}
EGO_ROUTE = "sb_thru"


def check_routes(net):
    """각 경로의 연속 엣지 사이에 차로 연결이 실제로 있는지 검사. 문제 목록 반환."""
    pairs = {(f, t) for f, t, _, _ in net.cons}
    bad = []
    for name, r in ROUTES.items():
        es = r.split()
        for a, b in zip(es, es[1:]):
            if a not in net.edges or b not in net.edges:
                bad.append(f"{name}: 엣지 없음 {a}/{b}")
            elif (a, b) not in pairs:
                bad.append(f"{name}: 연결 없음 {a}→{b}")
    for f, t, fl, tl in net.cons:
        if fl >= net.edges[f]["n"] or tl >= net.edges[t]["n"]:
            bad.append(f"차로 번호 초과 {f}_{fl}→{t}_{tl}")
        if net.edges[f]["to"] != net.edges[t]["frm"]:
            bad.append(f"노드 불일치 {f}→{t}")
    return bad


# ══════════════════════════════════════════════════════════════
# 파일 쓰기
# ══════════════════════════════════════════════════════════════
def _fmt3(p):
    return f"{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}"


def write_net_inputs(net, out_dir):
    nod = os.path.join(out_dir, "highway.nod.xml")
    edg = os.path.join(out_dir, "highway.edg.xml")
    con = os.path.join(out_dir, "highway.con.xml")
    # 들어오는 엣지 + 나가는 엣지가 모두 있는 노드만 priority 교차로
    deg_in, deg_out = {}, {}
    for e in net.edges.values():
        deg_out[e["frm"]] = deg_out.get(e["frm"], 0) + 1
        deg_in[e["to"]] = deg_in.get(e["to"], 0) + 1
    with open(nod, "w", encoding="utf-8") as f:
        f.write("<nodes>\n")
        for nid, (x, y, z) in net.nodes.items():
            typ = ' type="priority"' if deg_in.get(nid) and deg_out.get(nid) else ""
            f.write(f'    <node id="{nid}" x="{x:.2f}" y="{y:.2f}" z="{z:.2f}"{typ}/>\n')
        f.write("</nodes>\n")
    with open(edg, "w", encoding="utf-8") as f:
        f.write("<edges>\n")
        for eid, e in net.edges.items():
            shape = " ".join(_fmt3(p) for p in e["shape"])
            f.write(f'    <edge id="{eid}" from="{e["frm"]}" to="{e["to"]}" '
                    f'numLanes="{e["n"]}" speed="{e["speed"]:.2f}" priority="{e["prio"]}" '
                    f'width="{e["lw"]}" spreadType="center" name="{e["name"]}" '
                    f'disallow="pedestrian bicycle" shape="{shape}"/>\n')
        f.write("</edges>\n")
    with open(con, "w", encoding="utf-8") as f:
        f.write("<connections>\n")
        for frm, to, fl, tl in net.cons:
            f.write(f'    <connection from="{frm}" to="{to}" fromLane="{fl}" toLane="{tl}"/>\n')
        f.write("</connections>\n")
    return nod, edg, con


def write_routes(path, vtype_lines, flow_type, traffic, ego_cfg):
    """경로 + 교통류. 'r0'은 ego 전용 경로(경부 하행 직진)."""
    scale = float(traffic.get("vehs_per_hour", 800)) / 800.0
    flows = dict(DEFAULT_FLOWS)
    flows.update(traffic.get("ic_flows") or {})
    lines = ['<routes>',
             f'    <vType id="ego_type" accel="{ego_cfg["accel"]}" decel="{ego_cfg["decel"]}" '
             f'maxSpeed="{ego_cfg["max_speed"]}" color="1,0,0" vClass="passenger" '
             f'guiShape="passenger/sedan"/>',
             vtype_lines,
             f'    <route id="r0" edges="{ROUTES[EGO_ROUTE]}"/>']
    for name, edges in ROUTES.items():
        lines.append(f'    <route id="rt_{name}" edges="{edges}"/>')
    for name in ROUTES:
        vph = float(flows.get(name, 0)) * scale
        if vph <= 0:
            continue
        lines.append(f'    <flow id="f_{name}" type="{flow_type}" route="rt_{name}" '
                     f'begin="0" end="100000" period="exp({vph / 3600.0:.5f})" '
                     f'departLane="best" departSpeed="max" departPos="base"/>')
    lines.append('</routes>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def net_bounds(net):
    xs = [p[0] for e in net.edges.values() for p in e["shape"]]
    ys = [p[1] for e in net.edges.values() for p in e["shape"]]
    return min(xs), min(ys), max(xs), max(ys)


def write_gui(path, net, gui_delay, view_width=750.0, center=None):
    """검은 도로 + trueZ(입체 순서) GUI 설정. 시작 화면은 IC 중심."""
    x0, y0, x1, y1 = net_bounds(net)
    extent = max(x1 - x0, y1 - y0)
    zoom = extent / view_width * 100.0
    cx, cy = center or _GO
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"""<viewsettings>
    <scheme name="yangjae">
        <opengl trueZ="1"/>
        <background backgroundColor="86,104,74" showGrid="0"/>
        <edges laneEdgeMode="0" laneShowBorders="1" showLinkDecals="0" showLinkRules="0" showRails="0" hideConnectors="1" showSublanes="0" showDirection="0"/>
        <vehicles vehicleMode="0" vehicleQuality="2" showBlinker="1" vehicle_minSize="0"/>
        <junctions junctionMode="0" drawShape="1"/>
    </scheme>
    <viewport zoom="{zoom:.0f}" x="{cx:.1f}" y="{cy:.1f}" angle="0"/>
    <delay value="{gui_delay}"/>
</viewsettings>
""")


# ══════════════════════════════════════════════════════════════
# 조경 / 시각 요소 (물리와 무관한 polygon)
# trueZ 모드에서 polygon의 layer = 그려지는 높이(z).
#   차로는 엣지의 최저 z, 차량은 z+1, 교차로 면은 4에 그려진다.
# ══════════════════════════════════════════════════════════════
def _offset_line(pts, off):
    """폴리라인을 오른쪽으로 off 만큼 평행이동 (x,y만)."""
    out = []
    n = len(pts)
    for i in range(n):
        a = pts[max(i - 1, 0)]
        b = pts[min(i + 1, n - 1)]
        tx, ty = _norm(b[0] - a[0], b[1] - a[1])
        out.append((pts[i][0] + ty * off, pts[i][1] - tx * off))
    return out


def _trim_ends(pts, d):
    """폴리라인 양 끝을 d m 씩 잘라낸다 (짧으면 빈 목록)."""
    P = _resample(pts, 1.0)
    if len(P) * 1.0 <= 2 * d + 2:
        return []
    k = int(d)
    return P[k:len(P) - k]


def _band(pts, off_l, off_r):
    """중심선 기준 offset 띠 폴리곤 (off_l < off_r, 오른쪽 +)."""
    L = _offset_line(pts, off_l)
    R = _offset_line(pts, off_r)
    return L + R[::-1]


def _pstr(poly):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in poly)


def _pip(x, y, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _seg_dist(px, py, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    L2 = vx * vx + vy * vy or 1e-9
    t = max(0.0, min(1.0, ((px - a[0]) * vx + (py - a[1]) * vy) / L2))
    return math.hypot(px - a[0] - t * vx, py - a[1] - t * vy)


class _RoadIndex:
    """'이 점이 도로(테두리 포함) 위/근처인가' 빠른 판정용 격자."""

    def __init__(self, net, cell=25.0):
        self.cell = cell
        self.grid = {}
        for e in net.edges.values():
            hw = e["n"] * e["lw"] / 2.0 + 1.0
            sh = e["shape"]
            for a, b in zip(sh, sh[1:]):
                x0, x1 = sorted((a[0], b[0]))
                y0, y1 = sorted((a[1], b[1]))
                for gx in range(int((x0 - hw) // cell), int((x1 + hw) // cell) + 1):
                    for gy in range(int((y0 - hw) // cell), int((y1 + hw) // cell) + 1):
                        self.grid.setdefault((gx, gy), []).append((a, b, hw))

    def clearance(self, x, y):
        """가장 가까운 도로 가장자리까지 거리 (도로 위면 음수)."""
        best = 1e9
        gx, gy = int(x // self.cell), int(y // self.cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for a, b, hw in self.grid.get((gx + dx, gy + dy), ()):
                    best = min(best, _seg_dist(x, y, a, b) - hw)
        return best


def _blob(cx, cy, r, rng, n=12, rough=0.3):
    raw = [1.0 + rng.uniform(-rough, rough) for _ in range(n)]
    sm = [(raw[i - 1] + raw[i] + raw[(i + 1) % n]) / 3.0 for i in range(n)]
    return [(cx + r * sm[i] * math.cos(2 * math.pi * i / n),
             cy + r * sm[i] * math.sin(2 * math.pi * i / n)) for i in range(n)]


def write_scenery(path, net, seed=7, trees=False, buildings=False):
    rng = random.Random(seed)
    L = ['<additional>']
    pid = [0]

    def poly(shape, color, layer, fill=True, lw=None):
        pid[0] += 1
        extra = '' if fill else f' lineWidth="{lw}"'
        L.append(f'    <poly id="p{pid[0]}" color="{color}" fill="{1 if fill else 0}" '
                 f'layer="{layer:.2f}"{extra} shape="{_pstr(shape)}"/>')

    # ── (1) 도로 테두리(연석/갓길): 차로보다 살짝 아래에 밝은 회색 띠 ──
    #     지하로 내려가는 엣지(지하차도 진입부)는 밝은 띠가 경사 차로를 덮어
    #     하얗게 보이므로, 대신 훨씬 아래에 어두운 옹벽색 띠를 깐다.
    #     테두리는 교차로 면(z=4)보다 아래(3.6)에 깐다 → 합류/분류부에 회색 줄무늬가 안 생김.
    #     단, 경부 교량 구간(|t|<45 m)은 아래 도로를 덮어야 하므로 교량 높이에 둔다.
    CURB, WALL = "176,176,168", "58,58,56"
    Z_CURB = 3.6
    for eid, e in net.edges.items():
        sh = e["shape"]
        xy = [p[:2] for p in sh]
        zmin = min(p[2] for p in sh)
        hw = e["n"] * e["lw"] / 2.0
        lo, hi = (-hw, hw) if e.get("spread", "center") == "center" else (0.0, 2 * hw)
        if zmin < Z_GROUND - 0.1:
            poly(_band(xy, lo - 0.9, hi + 0.9), WALL, zmin - 3.0)
            continue
        if e.get("kind") == "G" and zmin > Z_GROUND + 0.5:
            def tg(p):
                return (p[0] - _GO[0]) * _GD[0] + (p[1] - _GO[1]) * _GD[1]
            on = [p[:2] for p in sh if abs(tg(p)) <= 45.0]
            if len(on) >= 2:
                poly(_band(on, lo - 0.7, hi + 0.7), CURB, zmin - 0.4)
        elif zmin > Z_GROUND + 0.5:      # 주변도로의 다리: 아래 도로를 덮도록 다리 높이에
            cut = _trim_ends(xy, 14.0)   # 끝(교차점)은 비워 둬야 교차점 위에 회색 띠가 안 생긴다
            if len(cut) >= 2:
                poly(_band(cut, lo - 0.7, hi + 0.7), CURB, zmin - 0.4)
        poly(_band(xy, lo - 0.7, hi + 0.7), CURB, Z_CURB)

    # ── (2) 경부 중앙분리대: 포장 + 가운데 콘크리트 방호벽 ──
    for eid in [k for k in net.edges if k.startswith("SB_")]:
        e = net.edges[eid]
        zmin = min(p[2] for p in e["shape"])
        sh = e["shape"]
        # 중앙선(축) 위치로 되돌린 폴리라인
        ts = [((p[0] - _GO[0]) * _GD[0] + (p[1] - _GO[1]) * _GD[1]) for p in (sh[0], sh[-1])]
        k = max(2, int(abs(ts[1] - ts[0]) / 6) + 1)
        axis = [_gpos(ts[0] + (ts[1] - ts[0]) * i / (k - 1), 0.0) for i in range(k)]
        hi = zmin > Z_GROUND + 0.5
        poly(_band(axis, -3.1, 3.1), "64,64,62", (zmin - 0.35) if hi else Z_CURB + 0.1)
        poly(_band(axis, -0.35, 0.35), "214,214,206", zmin + 0.2)
    # 양재대로 본선 사이 분리대 (지하로 내려가는 엣지는 생략 — 경사 차로를 덮지 않게)
    for eid in ("EBm_0", "EBm_1", "EBm_2", "EBm_3"):
        e = net.edges[eid]
        zmin = min(p[2] for p in e["shape"])
        if zmin < Z_GROUND - 0.1:
            continue
        sh = e["shape"]
        k = len(sh)
        x0, y0 = sh[0][:2]
        # 시작/끝 station
        s0 = _proj_y(x0, y0)
        s1 = _proj_y(*sh[-1][:2])
        axis = [_ypos(s0 + (s1 - s0) * i / max(k - 1, 1), 0.0) for i in range(k)]
        poly(_band(axis, -2.3, 2.3), "64,64,62", Z_CURB + 0.1)
        poly(_band(axis, -0.3, 0.3), "214,214,206", zmin + 0.2)

    # ── (3) 경부 교량 난간 (교량 구간 바깥 가장자리) ──
    for eid, sgn in (("SB_w", +1), ("NB_w", -1)):
        e = net.edges[eid]
        zmin = min(p[2] for p in e["shape"])
        outer = 3.0 + e["n"] * LW_G
        ts = [-48.0 + 96.0 * i / 24 for i in range(25)]
        line = [_gpos(t, sgn * outer) for t in ts]
        poly(_band(line, 0.0, sgn * 0.9) if sgn > 0 else _band(line, -0.9, 0.0),
             "222,222,214", zmin + 0.3)

    # ── (4) 지하차도 덮개: 콘크리트 + 점선 테두리 + 입구(어두운 띠) ──
    COVER, DASH, PORTAL = "128,130,122", "236,236,228", "18,18,18"
    for a, b, c_, d in TUNNELS_Y:
        ss = [b + (c_ - b) * i / 60 for i in range(61)]
        axis = [_ypos(s, 0.0) for s in ss]
        poly(_band(axis, -9.6, 9.6), COVER, 3.0)
        for side in (-9.6, 9.6):
            s = b
            while s < c_:
                s2 = min(s + 6.0, c_)
                seg = [_ypos(s + (s2 - s) * j / 3, side) for j in range(4)]
                poly(_band(seg, -0.35, 0.35), DASH, 3.2)
                s += 10.0
        for s in (b, c_):
            seg = [_ypos(s, o) for o in (-9.0, 9.0)]
            dx, dy = _ydir(s)
            poly([(seg[0][0] - dx * 1.5, seg[0][1] - dy * 1.5), (seg[1][0] - dx * 1.5, seg[1][1] - dy * 1.5),
                  (seg[1][0] + dx * 1.5, seg[1][1] + dy * 1.5), (seg[0][0] + dx * 1.5, seg[0][1] + dy * 1.5)],
                 PORTAL, 3.1)
    # 만남의광장 지하차도: 지하 바닥 높이인 구간을 덮개로
    sh = net.edges["RSVC_t"]["shape"]
    under = [p[:2] for p in sh if p[2] <= Z_TUNNEL + 0.05]
    if len(under) >= 2:
        poly(_band(under, -4.2, 4.2), COVER, 3.0)
        for q in (0, -1):
            p0 = under[q]
            p1 = under[1] if q == 0 else under[-2]
            dx, dy = _norm(p1[0] - p0[0], p1[1] - p0[1])
            nx, ny = dy, -dx
            poly([(p0[0] + nx * 3.8 - dx, p0[1] + ny * 3.8 - dy), (p0[0] - nx * 3.8 - dx, p0[1] - ny * 3.8 - dy),
                  (p0[0] - nx * 3.8 + dx * 1.5, p0[1] - ny * 3.8 + dy * 1.5),
                  (p0[0] + nx * 3.8 + dx * 1.5, p0[1] + ny * 3.8 + dy * 1.5)], PORTAL, 3.1)

    ridx = _RoadIndex(net)

    # ── 서울만남의광장 휴게소 주차장 (지하차도 출발점) ──
    pad = [_gpos(t, o) for t, o in ((392, 34), (458, 34), (466, 46), (466, 88), (392, 88))]
    poly(pad, "92,92,90", 0.8)
    poly(pad + [pad[0]], "176,176,168", 0.85, fill=False, lw=0.8)
    for t in range(398, 454, 3):                       # 주차 칸 선
        for o0, o1 in ((40, 50), (60, 70), (74, 84)):
            poly(_band([_gpos(t, o0), _gpos(t, o1)], -0.08, 0.08), "220,220,210", 0.9)
    if buildings:
        hall = [_gpos(t, o) for t, o in ((400, 51), (446, 51), (446, 58), (400, 58))]
        poly(hall, "196,184,160", 1.6)
        poly(hall + [hall[0]], "96,96,92", 1.7, fill=False, lw=0.5)

    # ── (5) 루프 안쪽 잔디 ──
    loops = []
    for eid in ("L7252", "L7086", "L1470", "L1469"):
        ring = [p[:2] for p in net.edges[eid]["shape"]]
        cx = sum(p[0] for p in ring) / len(ring)
        cy = sum(p[1] for p in ring) / len(ring)
        inner = []
        for x, y in ring:
            d = math.hypot(x - cx, y - cy) or 1
            k = max(0.0, (d - 4.5) / d)
            inner.append((cx + (x - cx) * k, cy + (y - cy) * k))
        loops.append(inner)
        poly(inner, "104,134,80", 0.5)

    # ── (6) 건물 / 주차장 (도로와 겹치는 것은 제외) ──
    bpolys = []
    roofs = ["168,166,160", "150,150,148", "184,178,166", "140,144,150", "196,192,184"]
    for kind, lv, pts in (D.BUILDINGS if buildings else []):
        if any(ridx.clearance(x, y) < 1.0 for x, y in pts):
            continue
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        if ridx.clearance(cx, cy) < 1.0:
            continue
        if kind == "P":
            poly(pts, "108,108,104", 0.8)
        else:
            h = 1.2 + min(lv, 30) * 0.12
            poly([(x + h, y - h) for x, y in pts], "40,46,38,110", 1.3)   # 옅은 바닥 그림자
            poly(pts, rng.choice(roofs), 1.6)
            poly(pts + [pts[0]], "96,96,92", 1.7, fill=False, lw=0.5)
        bpolys.append(pts)

    # ── (7) 나무: 루프 안쪽·IC 녹지는 빽빽하게, 나머지는 드문드문 ──
    if trees:
        palettes = [("46,110,44", "32,84,32", "96,158,84"),
                    ("38,98,40", "26,72,28", "84,146,74"),
                    ("56,122,50", "40,92,38", "110,168,92"),
                    ("30,88,36", "20,64,26", "76,138,70")]
        x0, y0, x1, y1 = net_bounds(net)
        waves = [(rng.uniform(0.008, 0.03), rng.uniform(0, 6.3), rng.uniform(0, 6.3),
                  rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(5)]

        def noise(x, y):
            # 저주파 노이즈 → 나무가 '숲 덩어리'와 '빈 땅'으로 자연스럽게 뭉친다
            v = sum(math.sin(f * (x * ax + y * ay) + p1) * math.cos(f * (y * ax - x * ay) + p2)
                    for f, p1, p2, ax, ay in waves) / len(waves)
            return max(0.15, min(1.8, 1.0 + 1.9 * v))
        step = 7.0
        y = y0 - 40
        tid = 0
        while y < y1 + 40:
            x = x0 - 40
            while x < x1 + 40:
                px = x + rng.uniform(-3, 3)
                py = y + rng.uniform(-3, 3)
                x += step
                in_loop = any(_pip(px, py, lp) for lp in loops)
                near_ic = math.hypot(px - _GO[0], py - _GO[1]) < 330
                p_keep = (0.55 if in_loop else (0.20 if near_ic else 0.05)) * noise(px, py)
                if rng.random() > p_keep:
                    continue
                r = rng.uniform(2.4, 4.4)
                if ridx.clearance(px, py) < r + 1.5:
                    continue
                if any(_pip(px, py, b) for b in bpolys) or _pip(px, py, pad):
                    continue
                body, dark, light = rng.choice(palettes)
                tid += 1
                poly(_blob(px + r * 0.3, py - r * 0.3, r, rng, n=10), "18,26,16,80", 1.0)
                poly(_blob(px, py, r, rng, n=10), body, 2.0)
                poly(_blob(px + r * 0.12, py - r * 0.12, r * 0.55, rng, n=8), dark, 3.0)
                if rng.random() < 0.5:     # 절반만 하이라이트 (폴리곤 수 절약)
                    poly(_blob(px - r * 0.3, py + r * 0.3, r * 0.3, rng, n=7, rough=0.2),
                         light, 3.5)
            y += step
    L.append('</additional>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def _proj_y(x, y):
    """점 (x,y)의 양재대로 축 station."""
    best, bs = 1e18, 0.0
    for i, (a, b) in enumerate(zip(_Y.p, _Y.p[1:])):
        vx, vy = b[0] - a[0], b[1] - a[1]
        L2 = vx * vx + vy * vy
        t = max(0.0, min(1.0, ((x - a[0]) * vx + (y - a[1]) * vy) / L2))
        d = math.hypot(x - a[0] - t * vx, y - a[1] - t * vy)
        if d < best:
            best, bs = d, _Y.cum[i] + t * math.sqrt(L2)
    return bs


def netconvert_args(nod, edg, con, out_net):
    return ["--node-files", nod, "--edge-files", edg, "--connection-files", con,
            "-o", out_net,
            "--offset.disable-normalization", "true",   # 조경 좌표와 맞추기 위해 원점 유지
            "--no-turnarounds", "true",
            "--junctions.corner-detail", "8",
            "--geometry.max-grade.fix", "false",
            "--no-warnings", "true"]
