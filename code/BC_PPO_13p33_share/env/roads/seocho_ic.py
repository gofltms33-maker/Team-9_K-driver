"""서초IC (경부고속도로 × 남부순환로) 를 양재IC처럼 손으로 다시 만든다.

OSM 실측 선형은 '위치 참고용'으로만 쓰고, 도로는 아래 규칙으로 새로 깐다.
  - 본선 4개(경부 상·하행, 남부순환로 동·서행)는 두 방향의 가운데 선(축)을 따라
    중앙분리대 안쪽 선을 고정하고, 차로가 늘거나 줄면 바깥쪽으로만 붙거나 빠진다.
  - 램프 6개는 본선 바깥 차로에서 접선 방향으로 갈라져 나와 OSM 선형을 따라가고,
    반대쪽 본선에 접선 방향으로 붙는다 (베지어 + 가우시안 다듬기).
  - 합류는 가속차로(차로 추가) → 일정 거리 뒤 지퍼 합류로 차로 감소,
    분류는 전용 차로(감속차로) 또는 바깥 차로 공유.
  - 경부는 남부순환로 위를 다리로 지나가고, 램프 높이는 양 끝에서 기울기 0 으로 이어진다.
구조 (OSM 기준):
  경부 하행 → 남부 서행  : 북서 직결 램프        남부 서행 → 경부 하행 : 북서 루프
  경부 하행 → 남부 동행  : 남서 루프            남부 동행 → 경부 하행 : 남서 직결 램프
  경부 상행 → 남부 동행  : 남동 직결 램프 (ego)  경부 상행 → 남부 서행 : 북동 루프
"""
import math

from env.roads import yangjae_ic as ic

CENTER = (-1180.0, 2200.0)
REMOVE_R = 330.0              # 이 반경 안의 기존(OSM) 도로는 지우고 새로 깐다
KEEP = {"w59054141_2", "w277756490_0"}     # 경계 밖으로 이어지는 경부 본선 (남쪽)

# OSM 원본 요소 (위치 참고용)
G_SB = ["w59054143_1", "w59054143_2", "w135131170_0", "w59054141_1"]
G_NB = ["w660731153_0", "w135131171_0", "w59054150_1"]
N_EB = ["w560620487_0", "w560620487_1", "w560620487_2", "w560620487_3", "w560620487_4"]
N_WB = ["w560607463_0", "w560607463_1", "w560607463_2", "w560607463_3"]
RAMP_OSM = {
    "SC_SB_WB": "w59054135_0",     # 경부 하행 → 남부 서행 (북서 직결)
    "SC_WB_SB": "w1251524454_0",   # 남부 서행 → 경부 하행 (북서 루프)
    "SC_SB_EB": "w1251524455_0",   # 경부 하행 → 남부 동행 (남서 루프)
    "SC_EB_SB": "w183519076_0",    # 남부 동행 → 경부 하행 (남서 직결)
    "SC_NB_EB": "w1251524457_0",   # 경부 상행 → 남부 동행 (남동 직결)
    "SC_NB_WB": "w1250979590_0",   # 경부 상행 → 남부 서행 (북동 루프)
}
G_BOUND_SB = "n732030910"      # 경부 하행이 남쪽 구간(w59054141_2)으로 나가는 노드
G_BOUND_NB = "n11091065444"    # 경부 상행이 남쪽 구간(w277756490_0)에서 들어오는 노드
N_BOUND_E = "n5406399769"      # 남부순환로 동쪽 경계 (양방향이 한 점에서 만남)

LW_G, LW_N, LW_R = 3.5, 3.3, 3.5
MED_G = 1.6                    # 경부 중앙분리대 (콘크리트 방호벽)
MED_N = 2.0                    # 남부순환로 중앙분리대 (동쪽 손으로 깐 구간과 같게)
END = {}                       # 동쪽 끝 정보 (다음 구간이 이어 붙일 때 사용)
Z0, ZB = ic.Z_GROUND, 12.0     # 지면, 경부 교량 (남쪽 구간 높이와 같게)
N_G, N_N = 4, 3                # 기본 차로 수
EXT_NORTH = 90.0               # 경부 북쪽 끝을 OSM 끝보다 더 늘리는 길이
EXT_WEST = 40.0


# ── 기하 도우미 ────────────────────────────────────────────────
def _chain(edges, ids):
    pts = []
    for k in ids:
        sh = [tuple(p[:2]) for p in edges[k]["shape"]]
        pts += sh if not pts else sh[1:]
    return pts


def _length(pts):
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


class Axis:
    """두 방향 차도의 가운데 선. s(호 길이)로 위치/방향을 돌려준다."""

    def __init__(self, fwd, bwd, ext0=0.0, ext1=0.0):
        # 주 방향 = 두 선의 시작→끝 평균 방향
        ax, ay = (fwd[-1][0] - fwd[0][0]) + (bwd[0][0] - bwd[-1][0]), \
                 (fwd[-1][1] - fwd[0][1]) + (bwd[0][1] - bwd[-1][1])
        L = math.hypot(ax, ay)
        d = (ax / L, ay / L)
        o = fwd[0]

        def tl(p):        # (진행 성분, 오른쪽 성분)
            return ((p[0] - o[0]) * d[0] + (p[1] - o[1]) * d[1],
                    (p[0] - o[0]) * d[1] - (p[1] - o[1]) * d[0])

        def interp(pts):
            q = sorted(tl(p) for p in pts)
            return q

        F, B = interp(fwd), interp(bwd)
        t0 = max(F[0][0], B[0][0]) - ext0
        t1 = min(F[-1][0], B[-1][0]) + ext1

        def lat(q, t):
            if t <= q[0][0]:
                return q[0][1]
            if t >= q[-1][0]:
                return q[-1][1]
            for (ta, la), (tb, lb) in zip(q, q[1:]):
                if ta <= t <= tb:
                    return la + (lb - la) * (t - ta) / max(tb - ta, 1e-9)
            return q[-1][1]

        mid = []
        t = t0
        while t <= t1 + 1e-6:
            m = 0.5 * (lat(F, t) + lat(B, t))
            mid.append((o[0] + d[0] * t + d[1] * m, o[1] + d[1] * t - d[0] * m))
            t += 2.0
        self.pts = ic._gauss_smooth(mid, 25.0, step=2.0)
        self.s = [0.0]
        for a, b in zip(self.pts, self.pts[1:]):
            self.s.append(self.s[-1] + math.dist(a, b))
        self.L = self.s[-1]

    def at(self, s):
        s = min(max(s, 0.0), self.L)
        for i in range(1, len(self.s)):
            if self.s[i] >= s:
                a, b = self.pts[i - 1], self.pts[i]
                f = (s - self.s[i - 1]) / max(self.s[i] - self.s[i - 1], 1e-9)
                h = ((b[0] - a[0]) / max(math.dist(a, b), 1e-9), (b[1] - a[1]) / max(math.dist(a, b), 1e-9))
                return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f), h
        return self.pts[-1], self.dir(self.L)

    def dir(self, s):
        return self.at(s)[1]

    def pos(self, s, off):
        """off > 0 = 진행 방향(축 +s) 기준 오른쪽."""
        (x, y), (hx, hy) = self.at(s)
        return (x + hy * off, y - hx * off)

    def proj(self, p):
        best = (1e18, 0.0)
        for i in range(len(self.pts) - 1):
            a, b = self.pts[i], self.pts[i + 1]
            L = math.dist(a, b)
            if L < 1e-9:
                continue
            u = max(0.0, min(1.0, ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / L ** 2))
            q = (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)
            dd = math.dist(p, q)
            if dd < best[0]:
                best = (dd, self.s[i] + u * L)
        return best[1]


class Carriageway:
    """축의 한쪽 차도. side=+1: 축 +s 방향으로 달림(오른쪽), -1: -s 방향(왼쪽)."""

    def __init__(self, axis, side, med, lw, zfun):
        self.ax, self.side, self.med, self.lw, self.z = axis, side, med, lw, zfun

    def off(self, n, lane=None):
        """차로 n 개짜리 구간의 중심(lane=None) 또는 lane 번 차로(0=바깥) 중심의 횡위치."""
        if lane is None:
            o = self.med / 2 + n * self.lw / 2
        else:
            o = self.med / 2 + (n - lane - 0.5) * self.lw
        return o * self.side      # 축 +s 기준 오른쪽(+)

    def heading(self, s):
        hx, hy = self.ax.dir(s)
        return (hx * self.side, hy * self.side)

    def lane_pt(self, s, n, lane=0):
        return self.ax.pos(s, self.off(n, lane))

    def shape(self, s_a, s_b, n, step=4.0):
        """진행 순서대로 (s_a → s_b) 중심선 3D 점."""
        k = max(2, int(abs(s_b - s_a) / step))
        out = []
        for i in range(k + 1):
            s = s_a + (s_b - s_a) * i / k
            x, y = self.ax.pos(s, self.off(n))
            out.append((x, y, self.z(s)))
        return out


def _heading_at(pts, i):
    a, b = pts[max(0, i - 1)], pts[min(len(pts) - 1, i + 1)]
    L = math.dist(a, b) or 1.0
    return ((b[0] - a[0]) / L, (b[1] - a[1]) / L)


def _ramp(osm_pts, A, hA, B, hB, zA, zB, cut=50.0):
    """A(본선 바깥 차로, 방향 hA) → OSM 선형 → B(반대 본선, 방향 hB)."""
    P = ic._resample(osm_pts, 2.0)
    s = [0.0]
    for a, b in zip(P, P[1:]):
        s.append(s[-1] + math.dist(a, b))
    L = s[-1]
    c = min(cut, L * 0.3)
    i1 = next(i for i, v in enumerate(s) if v >= c)
    i2 = max(i1 + 2, next(i for i, v in enumerate(s) if v >= L - c))
    mid = P[i1:i2 + 1]
    h1, h2 = _heading_at(P, i1), _heading_at(P, i2)
    head = ic._bezier(A, hA, mid[0], h1, n=12, k=0.38)
    tail = ic._bezier(mid[-1], h2, B, hB, n=12, k=0.38)
    xy = head[:-1] + mid + tail[1:]
    xy = ic._gauss_smooth(xy, 9.0, step=2.0, anchor=18.0)
    Ls = [0.0]
    for a, b in zip(xy, xy[1:]):
        Ls.append(Ls[-1] + math.dist(a, b))
    T = Ls[-1] or 1.0
    out = [(x, y, zA + (zB - zA) * ic._smooth(l / T)) for (x, y), l in zip(xy, Ls)]
    out = out[::3] + ([out[-1]] if (len(out) - 1) % 3 else [])
    return out


# ── 본체 ─────────────────────────────────────────────────────
def rebuild(nodes, edges, cons, speed_g):
    """nodes/edges/cons(양재IC 연결 목록)를 제자리에서 고친다."""
    for k in G_SB + G_NB + N_EB + N_WB + list(RAMP_OSM.values()) + list(KEEP):
        assert k in edges, f"서초IC: OSM 요소 {k} 없음 (OSM 파일이 바뀌었나요?)"
    sb = _chain(edges, G_SB)
    nb = _chain(edges, G_NB)
    eb = _chain(edges, N_EB)
    wb = _chain(edges, N_WB)
    rosm = {k: [tuple(p[:2]) for p in edges[v]["shape"]] for k, v in RAMP_OSM.items()}
    sp_n = edges[N_EB[1]]["speed"]
    pr_g, pr_n = edges[G_SB[0]]["prio"], edges[N_EB[1]]["prio"]
    bound = {k: nodes[k] for k in (G_BOUND_SB, G_BOUND_NB, N_BOUND_E)}

    # 지울 것: 반경 안의 기존 도로 (남쪽 경부 본선 2개는 유지)
    for k in [k for k, e in edges.items()
              if not k[0].isupper() and k not in KEEP
              and any(math.dist(p[:2], CENTER) < REMOVE_R for p in e["shape"])]:
        del edges[k]

    # 축: 경부(+s = 북쪽, 상행 방향), 남부(+s = 동쪽, 동행 방향)
    gA = Axis(nb, sb, ext0=0.0, ext1=EXT_NORTH)
    nA = Axis(eb, wb, ext0=EXT_WEST, ext1=0.0)
    # 경부 축 시작을 남쪽 경계 노드에 맞춘다
    s_gb = min(gA.proj(bound[G_BOUND_SB][:2]), gA.proj(bound[G_BOUND_NB][:2]))
    s_c = gA.proj(nA.pos(nA.proj(gA.pos(gA.L / 2, 0)), 0))      # 교차 지점 (경부 축 위)
    for _ in range(3):
        s_c = gA.proj(nA.pos(nA.proj(gA.pos(s_c, 0)), 0))
    s_nc = nA.proj(gA.pos(s_c, 0))

    def z_g(s):                       # 교차부 ±35 m 교량, 양쪽으로 110 m 동안 지면까지
        d = abs(s - s_c)
        if d <= 35:
            return ZB
        if d >= 145:
            return Z0
        return ZB + (Z0 - ZB) * ic._smooth((d - 35) / 110)

    def z_n(_):
        return Z0

    NB = Carriageway(gA, +1, MED_G, LW_G, z_g)
    SB = Carriageway(gA, -1, MED_G, LW_G, z_g)
    EB = Carriageway(nA, +1, MED_N, LW_N, z_n)
    WB = Carriageway(nA, -1, MED_N, LW_N, z_n)

    # 램프 접점 (OSM 끝점을 축에 투영)
    def ps(axis, p):
        return axis.proj(p)
    s_sbwb_d = ps(gA, rosm["SC_SB_WB"][0])
    s_wbsb_m = ps(gA, rosm["SC_WB_SB"][-1]) - 10.0      # 루프는 본선과 나란해진 뒤 합류
    s_sbeb_d = ps(gA, rosm["SC_SB_EB"][0])
    s_nbwb_d = ps(gA, rosm["SC_NB_WB"][0])
    s_ebsb_d = ps(nA, rosm["SC_EB_SB"][0])
    s_sbeb_m = ps(nA, rosm["SC_SB_EB"][-1])
    s_nbeb_m = ps(nA, rosm["SC_NB_EB"][-1])
    s_nbwb_m = ps(nA, rosm["SC_NB_WB"][-1])
    s_wbsb_d = ps(nA, rosm["SC_WB_SB"][0])
    s_sbwb_m = ps(nA, rosm["SC_SB_WB"][-1])
    s_n_end = nA.proj(bound[N_BOUND_E][:2])
    s_g_end = gA.L

    new_nodes, new_edges, new_cons, zipper = {}, {}, [], set()

    def node(nid, cw, s):
        x, y = cw.ax.pos(s, 0.0)                 # 노드는 축 위 (양방향 공용이면 하나)
        new_nodes[nid] = (x, y, cw.z(s))
        return nid

    def cw_node(nid, cw, s, n):
        x, y = cw.ax.pos(s, cw.off(n))
        new_nodes[nid] = (x, y, cw.z(s))
        return nid

    def main(eid, frm, to, cw, s_a, s_b, n, name, speed, prio):
        new_edges[eid] = dict(frm=frm, to=to, n=n, lw=cw.lw, speed=speed, prio=prio, name=name,
                              shape=cw.shape(s_a, s_b, n), spread="center")

    def ramp(eid, frm, to, key, cwA, sA, nA_, cwB, sB, nB_, speed, name="서초IC 램프"):
        A = cwA.lane_pt(sA, nA_, 0)
        B = cwB.lane_pt(sB, nB_, 0)
        shp = _ramp(rosm[key], A, cwA.heading(sA), B, cwB.heading(sB), cwA.z(sA), cwB.z(sB))
        new_edges[eid] = dict(frm=frm, to=to, n=1, lw=LW_R, speed=speed, prio=3, name=name,
                              shape=shp, spread="center")

    def con(a, b, fl, tl):
        new_cons.append((a, b, fl, tl))

    def straight(a, b, n, shift=0):
        for i in range(n):
            con(a, b, i, i + shift)

    sp_r, sp_l = 60 / 3.6, 40 / 3.6

    # ── 경부 하행 (북 → 남, s 감소) ── 4차로, 북서 루프 합류로 5, 남서 루프로 빠지며 4, 직결 합류로 5
    s_sbwb_d = min(s_sbwb_d, s_g_end - 40)
    cw_node("SC_SB_n0", SB, s_g_end, N_G)
    cw_node("SC_SB_n1", SB, s_sbwb_d, N_G)                   # 북서 직결 분류 (바깥 차로 공유)
    cw_node("SC_SB_n2", SB, s_wbsb_m, N_G)                   # 북서 루프 합류 (차로 추가)
    cw_node("SC_SB_n3", SB, s_sbeb_d, N_G + 1)               # 남서 루프 분류 (전용 차로)
    main("SC_SB_0", "SC_SB_n0", "SC_SB_n1", SB, s_g_end, s_sbwb_d, N_G, "경부고속도로", speed_g, pr_g)
    main("SC_SB_1", "SC_SB_n1", "SC_SB_n2", SB, s_sbwb_d, s_wbsb_m, N_G, "경부고속도로", speed_g, pr_g)
    main("SC_SB_2", "SC_SB_n2", "SC_SB_n3", SB, s_wbsb_m, s_sbeb_d, N_G + 1, "경부고속도로", speed_g, pr_g)
    main("SC_SB_3", "SC_SB_n3", G_BOUND_SB, SB, s_sbeb_d, s_gb, N_G, "경부고속도로", speed_g, pr_g)
    straight("SC_SB_0", "SC_SB_1", N_G)
    con("SC_SB_0", "SC_SB_WB", 0, 0)
    straight("SC_SB_1", "SC_SB_2", N_G, 1)
    con("SC_WB_SB", "SC_SB_2", 0, 0)
    for i in range(1, N_G + 1):
        con("SC_SB_2", "SC_SB_3", i, i - 1)
    con("SC_SB_2", "SC_SB_EB", 0, 0)
    straight("SC_SB_3", "w59054141_2", N_G, 1)               # 남쪽 구간은 5차로
    con("SC_EB_SB", "w59054141_2", 0, 0)

    # ── 경부 상행 (남 → 북) ── 5차로 중 바깥 차로는 남동 직결 전용, 4차로로 북쪽, 북동 루프는 공유 분류
    # 교량 구간(교차 지점 앞 35 m 부터)은 따로 끊는다. sumo-gui 는 엣지를 그 엣지의 가장 낮은 높이에
    # 그리기 때문에, 지면에서 올라오는 경사로와 한 엣지로 묶이면 교량이 지면 높이로 그려지고
    # 그 아래를 지나는 남부순환로 차량이 교량 위로 보인다.
    s_nb_deck = s_c - 35.0
    cw_node("SC_NB_nd", NB, s_nb_deck, N_G)
    cw_node("SC_NB_n1", NB, s_nbwb_d, N_G)
    cw_node("SC_NB_n2", NB, s_g_end, N_G)
    main("SC_NB_0", G_BOUND_NB, "SC_NB_nd", NB, s_gb, s_nb_deck, N_G, "경부고속도로", speed_g, pr_g)
    main("SC_NB_0b", "SC_NB_nd", "SC_NB_n1", NB, s_nb_deck, s_nbwb_d, N_G, "경부고속도로", speed_g, pr_g)
    main("SC_NB_1", "SC_NB_n1", "SC_NB_n2", NB, s_nbwb_d, s_g_end, N_G, "경부고속도로", speed_g, pr_g)
    for i in range(1, 5):
        con("w277756490_0", "SC_NB_0", i, i - 1)
    con("w277756490_0", "SC_NB_EB", 0, 0)
    straight("SC_NB_0", "SC_NB_0b", N_G)
    straight("SC_NB_0b", "SC_NB_1", N_G)
    con("SC_NB_0b", "SC_NB_WB", 0, 0)

    # ── 남부순환로 동행 (서 → 동) ──
    s_drop1 = min(s_sbeb_m + 90.0, s_nbeb_m - 40.0)
    s_drop2 = min(s_nbeb_m + 120.0, s_n_end - 35.0)
    cw_node("SC_EB_n0", EB, 0.0, N_N)
    cw_node("SC_EB_n1", EB, s_ebsb_d, N_N)                    # 남서 직결 분류 (공유)
    cw_node("SC_EB_n2", EB, s_sbeb_m, N_N)                    # 남서 루프 합류 (가속차로)
    cw_node("SC_EB_n3", EB, s_drop1, N_N)                     # 지퍼 합류
    cw_node("SC_EB_n4", EB, s_nbeb_m, N_N)                    # 남동 직결 합류 (가속차로)
    cw_node("SC_EB_n5", EB, s_drop2, N_N)                     # 지퍼 합류
    zipper |= {"SC_EB_n3", "SC_EB_n5"}
    nm = "남부순환로"
    main("SC_EB_0", "SC_EB_n0", "SC_EB_n1", EB, 0.0, s_ebsb_d, N_N, nm, sp_n, pr_n)
    main("SC_EB_1", "SC_EB_n1", "SC_EB_n2", EB, s_ebsb_d, s_sbeb_m, N_N, nm, sp_n, pr_n)
    main("SC_EB_2", "SC_EB_n2", "SC_EB_n3", EB, s_sbeb_m, s_drop1, N_N + 1, nm, sp_n, pr_n)
    main("SC_EB_3", "SC_EB_n3", "SC_EB_n4", EB, s_drop1, s_nbeb_m, N_N, nm, sp_n, pr_n)
    main("SC_EB_4", "SC_EB_n4", "SC_EB_n5", EB, s_nbeb_m, s_drop2, N_N + 1, nm, sp_n, pr_n)
    main("SC_EB_5", "SC_EB_n5", N_BOUND_E, EB, s_drop2, s_n_end, N_N, nm, sp_n, pr_n)
    straight("SC_EB_0", "SC_EB_1", N_N)
    con("SC_EB_0", "SC_EB_SB", 0, 0)
    straight("SC_EB_1", "SC_EB_2", N_N, 1)
    con("SC_SB_EB", "SC_EB_2", 0, 0)
    con("SC_EB_2", "SC_EB_3", 0, 0)
    for i in range(1, N_N + 1):
        con("SC_EB_2", "SC_EB_3", i, i - 1)
    straight("SC_EB_3", "SC_EB_4", N_N, 1)
    con("SC_NB_EB", "SC_EB_4", 0, 0)
    con("SC_EB_4", "SC_EB_5", 0, 0)
    for i in range(1, N_N + 1):
        con("SC_EB_4", "SC_EB_5", i, i - 1)

    # ── 남부순환로 서행 (동 → 서, s 감소) ──
    s_drop3 = max(s_nbwb_m - 90.0, s_wbsb_d + 40.0)
    cw_node("SC_WB_n1", WB, s_nbwb_m, N_N)                    # 북동 루프 합류 (가속차로)
    cw_node("SC_WB_n2", WB, s_drop3, N_N)                     # 지퍼 합류
    cw_node("SC_WB_n3", WB, s_wbsb_d, N_N)                    # 북서 루프 분류 (공유)
    cw_node("SC_WB_n4", WB, s_sbwb_m, N_N)                    # 북서 직결 합류 (차로 추가)
    cw_node("SC_WB_n5", WB, 0.0, N_N + 1)
    zipper.add("SC_WB_n2")
    main("SC_WB_0", N_BOUND_E, "SC_WB_n1", WB, s_n_end, s_nbwb_m, N_N, nm, sp_n, pr_n)
    main("SC_WB_1", "SC_WB_n1", "SC_WB_n2", WB, s_nbwb_m, s_drop3, N_N + 1, nm, sp_n, pr_n)
    main("SC_WB_2", "SC_WB_n2", "SC_WB_n3", WB, s_drop3, s_wbsb_d, N_N, nm, sp_n, pr_n)
    main("SC_WB_3", "SC_WB_n3", "SC_WB_n4", WB, s_wbsb_d, s_sbwb_m, N_N, nm, sp_n, pr_n)
    main("SC_WB_4", "SC_WB_n4", "SC_WB_n5", WB, s_sbwb_m, 0.0, N_N + 1, nm, sp_n, pr_n)
    straight("SC_WB_0", "SC_WB_1", N_N, 1)
    con("SC_NB_WB", "SC_WB_1", 0, 0)
    con("SC_WB_1", "SC_WB_2", 0, 0)
    for i in range(1, N_N + 1):
        con("SC_WB_1", "SC_WB_2", i, i - 1)
    straight("SC_WB_2", "SC_WB_3", N_N)
    con("SC_WB_2", "SC_WB_SB", 0, 0)
    straight("SC_WB_3", "SC_WB_4", N_N, 1)
    con("SC_SB_WB", "SC_WB_4", 0, 0)

    # ── 램프 ──
    ramp("SC_SB_WB", "SC_SB_n1", "SC_WB_n4", "SC_SB_WB", SB, s_sbwb_d, N_G, WB, s_sbwb_m, N_N + 1, sp_r)
    ramp("SC_WB_SB", "SC_WB_n3", "SC_SB_n2", "SC_WB_SB", WB, s_wbsb_d, N_N, SB, s_wbsb_m, N_G + 1, sp_l)
    ramp("SC_SB_EB", "SC_SB_n3", "SC_EB_n2", "SC_SB_EB", SB, s_sbeb_d, N_G + 1, EB, s_sbeb_m, N_N + 1, sp_l)
    ramp("SC_EB_SB", "SC_EB_n1", G_BOUND_SB, "SC_EB_SB", EB, s_ebsb_d, N_N, SB, s_gb, N_G + 1, sp_r)
    ramp("SC_NB_EB", G_BOUND_NB, "SC_EB_n4", "SC_NB_EB", NB, s_gb, N_G + 1, EB, s_nbeb_m, N_N + 1, sp_r)
    ramp("SC_NB_WB", "SC_NB_n1", "SC_WB_n1", "SC_NB_WB", NB, s_nbwb_d, N_G, WB, s_nbwb_m, N_N + 1, sp_l)

    # 경계 정리: 기존 노드를 새 차도 위치로 옮기고, 이어지는 기존 도로 끝을 60 m 에 걸쳐 맞춘다
    def move(nid, new_id, p_new, z_new):
        for k, e in edges.items():
            if e["frm"] == nid or e["to"] == nid:
                at_end = e["to"] == nid
                sh = e["shape"]
                e["shape"] = [(x, y, z) for (x, y), (_, _, z) in zip(_blend(sh, p_new, at_end), sh)]
                if at_end:
                    e["to"] = new_id
                else:
                    e["frm"] = new_id
        new_nodes[new_id] = (p_new[0], p_new[1], z_new)
    # 남쪽: 경부 5차로 구간 시작/끝을 새 축의 5차로 위치로
    move(G_BOUND_SB, G_BOUND_SB, SB.ax.pos(s_gb, SB.off(N_G + 1)), z_g(s_gb))
    move(G_BOUND_NB, G_BOUND_NB, NB.ax.pos(s_gb, NB.off(N_G + 1)), z_g(s_gb))
    # 동쪽: OSM 은 양방향을 한 점으로 묶어 두었으므로 동행/서행 노드를 나눈다 (서운로는 서행 쪽에 연결)
    p_eb = EB.ax.pos(s_n_end, EB.off(N_N))
    p_wb = WB.ax.pos(s_n_end, WB.off(N_N))
    END.clear()
    END.update(mid=nA.pos(s_n_end, 0.0), heading=nA.dir(s_n_end), med=MED_N,
               eb_node="SC_E_EB", wb_node="SC_E_WB", n=N_N, lw=LW_N)
    for k, e in list(edges.items()):
        if N_BOUND_E not in (e["frm"], e["to"]):
            continue
        if e.get("name") != "남부순환로":       # 서운로 등 옆길은 빼서 경계를 깔끔하게
            del edges[k]
            continue
        at_end = e["to"] == N_BOUND_E
        h = e["shape"][-1] if at_end else e["shape"][0]
        nxt = e["shape"][-2] if at_end else e["shape"][1]
        # 동쪽으로 나가는 남부 동행 → 동행 노드, 나머지(서행 유입, 서운로) → 서행 노드
        tgt, nid = (p_eb, "SC_E_EB") if (not at_end and e.get("name") == "남부순환로") else (p_wb, "SC_E_WB")
        sh = e["shape"]
        if e.get("name") == "남부순환로":
            # OSM 은 두 차도가 한 점으로 모이도록 끝을 휘어 놓았다 → 80 m 를 버리고 접선 베지어로 다시 잇는다
            q = sh if not at_end else sh[::-1]                # 경계 노드에서 출발
            acc, k = 0.0, 0
            while k < len(q) - 2 and acc < 80.0:
                acc += math.dist(q[k][:2], q[k + 1][:2])
                k += 1
            far = q[k]
            hf = _heading_at([p_[:2] for p_ in q], k)
            cw = EB if nid == "SC_E_EB" else WB
            h0 = cw.heading(s_n_end)
            if at_end:
                h0 = (-h0[0], -h0[1])                         # 노드에서 바깥쪽(동쪽)으로
            bez = ic._bezier(tgt, h0, far[:2], hf, n=16, k=0.4)
            q = [(x, y, Z0) for x, y in bez[:-1]] + list(q[k:])
            e["shape"] = q if not at_end else q[::-1]
        else:
            e["shape"] = [(x, y, z) for (x, y), (_, _, z) in zip(_blend(sh, tgt, at_end), sh)]
        if at_end:
            e["to"] = nid
        else:
            e["frm"] = nid
    new_edges["SC_EB_5"]["to"] = "SC_E_EB"
    new_edges["SC_WB_0"]["frm"] = "SC_E_WB"
    new_nodes["SC_E_EB"] = (p_eb[0], p_eb[1], Z0)
    new_nodes["SC_E_WB"] = (p_wb[0], p_wb[1], Z0)
    nodes.pop(N_BOUND_E, None)

    nodes.update(new_nodes)
    edges.update(new_edges)
    cons.extend(new_cons)
    return zipper


def _blend(sh, p, at_end, L=60.0):
    """끝점을 p 로 옮기되 L m 에 걸쳐 부드럽게."""
    pts = [q[:2] for q in sh]
    if not at_end:
        pts = pts[::-1]
    end = pts[-1]
    dx, dy = p[0] - end[0], p[1] - end[1]
    acc = [0.0]
    for a, b in zip(pts[::-1], pts[::-1][1:]):
        acc.append(acc[-1] + math.dist(a, b))
    acc = acc[::-1]                        # 끝에서부터의 거리
    out = [(x + dx * ic._smooth(max(0.0, 1 - d / L)), y + dy * ic._smooth(max(0.0, 1 - d / L)))
           for (x, y), d in zip(pts, acc)]
    return out if at_end else out[::-1]
