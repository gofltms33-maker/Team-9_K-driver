"""공사 구간 (시나리오 ② 공사 구간 합류): 언주로 남행(매봉 → 개포) 3차로 → 1차로.

  - 매봉에서 내려오는 언주로 남행(ar_ej_f0) 중간에 공사 구간을 둔다.
  - 안쪽(중앙분리대 쪽) 2개 차로를 막는다: 경고 구간(3차로) → 테이퍼 40 m(2차로, 지퍼) → 작업 구간 150 m(1차로, 지퍼)
    → 다시 3차로.
  - 막힌 차로 위에 주황색 작업 구역 면과 라바콘 줄을 그린다 (차로보다 높은 층 → 화면에서 확실히 막혀 보임).
  - 작업 구간 제한속도 30 km/h.
"""
import math

from env.roads import yangjae_ic as ic

EDGE = "ar_ej_f0x"          # 언주로 남행: 연결로 C 가 갈라지는 T자 아래 S자 구간 (개포 방향)
START_FRAC = 0.40          # 엣지 길이 대비 테이퍼 시작 위치
TAPER, WORK = 40.0, 150.0
WORK_SPEED = 30 / 3.6
POLYS = []                 # (점 목록, 색, 층, 채움)


def _arc(sh):
    s = [0.0]
    for a, b in zip(sh, sh[1:]):
        s.append(s[-1] + math.dist(a[:2], b[:2]))
    return s


def _at(sh, s_list, s):
    for i in range(1, len(s_list)):
        if s_list[i] >= s:
            a, b = sh[i - 1], sh[i]
            f = (s - s_list[i - 1]) / max(s_list[i] - s_list[i - 1], 1e-9)
            return tuple(a[k] + (b[k] - a[k]) * f for k in range(3))
    return sh[-1]


def _sub(sh, s_list, s_a, s_b):
    pts = [_at(sh, s_list, s_a)] + [p for p, s in zip(sh, s_list) if s_a < s < s_b] + [_at(sh, s_list, s_b)]
    return pts


def _shift(pts, off):
    """진행 기준 오른쪽(+)으로 off 만큼 평행이동."""
    out = []
    for i, p in enumerate(pts):
        a, b = pts[max(0, i - 1)], pts[min(len(pts) - 1, i + 1)]
        L = math.dist(a[:2], b[:2]) or 1.0
        hx, hy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        out.append((p[0] + hy * off, p[1] - hx * off, p[2]))
    return out


def rebuild(nodes, edges, cons, warp=None):
    if EDGE not in edges:
        return set()
    e = edges[EDGE]
    sh, n, lw = e["shape"], e["n"], e["lw"]
    if n < 3:
        return set()
    S = _arc(sh)
    L = S[-1]
    s1 = L * START_FRAC
    s2, s3 = s1 + TAPER, s1 + TAPER + WORK
    if s3 > L - 60:
        return set()
    to_node = e["to"]
    ids = [EDGE, EDGE + "_tp", EDGE + "_wz", EDGE + "_rc"]
    nn = ["c1", "c2", "c3"]
    nid = {k: f"{EDGE}_{k}" for k in nn}
    for k, s in zip(nn, (s1, s2, s3)):
        nodes[nid[k]] = _at(sh, S, s)
    base = dict(e)
    edges[ids[0]] = dict(base, to=nid["c1"], shape=_sub(sh, S, 0.0, s1))
    edges[ids[1]] = dict(base, frm=nid["c1"], to=nid["c2"], n=n - 1,
                         shape=_shift(_sub(sh, S, s1, s2), lw / 2), speed=min(base["speed"], 40 / 3.6),
                         name=(base.get("name") or "") + " (공사중)")   # 테이퍼부터 공사 구간
    edges[ids[2]] = dict(base, frm=nid["c2"], to=nid["c3"], n=1,
                         shape=_shift(_sub(sh, S, s2, s3), lw * (n - 1) / 2), speed=WORK_SPEED,
                         name=(base.get("name") or "") + " (공사중)")
    edges[ids[3]] = dict(base, frm=nid["c3"], to=to_node, shape=_sub(sh, S, s3, L))
    # 교차로 쪽 연결은 마지막 조각에서 나가도록
    for i, (a, b, fl, tl) in enumerate(cons):
        if a == EDGE:
            cons[i] = (ids[3], b, fl, tl)
    # 차로 연결: 3 → 2 (안쪽 차로 끝, 지퍼) → 1 (지퍼) → 3
    for i in range(n - 1):
        cons.append((ids[0], ids[1], i, i))
    cons.append((ids[0], ids[1], n - 1, n - 2))
    cons.append((ids[1], ids[2], 0, 0))
    for i in range(1, n - 1):
        cons.append((ids[1], ids[2], i, 0))
    for i in range(n):
        cons.append((ids[2], ids[3], 0, i))

    # 화면: 실제 공사장처럼 — 테이퍼 사선 줄무늬, 방호벽(빨강/흰), 라바콘, 화살표 표지판, 장비
    if warp is not None:
        POLYS.clear()
        hw = n * lw / 2
        open_edge = hw - lw                       # 열린 1차로의 왼쪽 경계 (중심선 기준 오른쪽 +)

        def frame(s_):
            p0 = _at(sh, S, max(0.0, min(L, s_)))
            p1 = _at(sh, S, max(0.0, min(L, s_ + 1.0)))
            q = math.dist(p0[:2], p1[:2]) or 1.0
            hx, hy = (p1[0] - p0[0]) / q, (p1[1] - p0[1]) / q
            return p0, (hx, hy), (hy, -hx)        # 위치, 진행방향, 오른쪽

        def P_(s_, w_):
            p0, h, r = frame(s_)
            return (p0[0] + r[0] * w_, p0[1] + r[1] * w_)

        def rect(s_, w_, a, b, col, lay, fill=True):
            """중심 (s_, w_), 진행방향 길이 a, 폭 b."""
            p0, h, r = frame(s_)
            c = (p0[0] + r[0] * w_, p0[1] + r[1] * w_)
            pts = [(c[0] + h[0] * x + r[0] * y, c[1] + h[1] * x + r[1] * y)
                   for x, y in ((-a / 2, -b / 2), (a / 2, -b / 2), (a / 2, b / 2), (-a / 2, b / 2))]
            POLYS.append((pts, col, lay, fill))

        def edge_w(s_):                           # 막힌 구역의 오른쪽 경계 (s_ 위치)
            if s_ < s1:
                return -hw
            if s_ < s2:
                return -hw + (open_edge + hw) * (s_ - s1) / TAPER
            if s_ <= s3:
                return open_edge
            return open_edge - (open_edge + hw) * min(1.0, (s_ - s3) / 15.0)

        # 1) 막힌 구역 바닥: 짙은 회색 (차로 위를 덮음)
        st = [s1 + k * 2.0 for k in range(int((s3 + 15 - s1) / 2.0) + 1)]
        left = [P_(x, -hw) for x in st]
        right = [P_(x, edge_w(x)) for x in st]
        POLYS.append((left + right[::-1], "88,86,82", 6.4, True))
        # 2) 테이퍼 사선 줄무늬 (주황/흰 번갈아)
        k = 0
        x = s1
        while x < s2 - 1.5:
            w0, w1 = -hw, edge_w(x + 1.5)
            col = "245,120,20" if k % 2 == 0 else "250,250,245"
            pts = [P_(x, w0), P_(x + 2.0, w0), P_(x + 3.5, edge_w(x + 3.5)), P_(x + 1.5, edge_w(x + 1.5))]
            POLYS.append((pts, col, 6.5, True))
            x += 2.0
            k += 1
        # 3) 방호벽: 경계선을 따라 빨강/흰 블록 (2 m 씩)
        x, k = s2, 0
        while x < s3:
            rect(x + 1.0, open_edge - 0.35, 2.0, 0.6, "210,30,30" if k % 2 == 0 else "245,245,245", 6.8)
            x += 2.0
            k += 1
        # 4) 라바콘: 테이퍼 경계 4 m 마다 (주황 + 흰 띠)
        x = s1 - 20
        while x < s2:
            w_ = edge_w(x) + 0.4 if x >= s1 else -hw + 0.6 + (x - (s1 - 20)) / 20 * 0.0
            p_ = P_(x, w_)
            for rr, col, lay in ((0.45, "255,100,0", 6.9), (0.22, "255,255,255", 7.0)):
                POLYS.append(([(p_[0] + rr * math.cos(t * math.pi / 5), p_[1] + rr * math.sin(t * math.pi / 5))
                               for t in range(10)], col, lay, True))
            x += 4.0
        # 5) 화살표 표지판 (테이퍼 시작, 막힌 차로 가운데): 검정 판 + 노란 화살표(오른쪽)
        sw = -hw + lw / 2
        rect(s1 + 2, sw, 1.2, 2.6, "20,20,20", 7.1)
        p0, h, r = frame(s1 + 2)
        c = (p0[0] + r[0] * sw, p0[1] + r[1] * sw)
        arrow = [(-0.1, -1.0), (-0.1, 0.4), (-0.4, 0.4), (0.0, 1.1), (0.4, 0.4), (0.1, 0.4), (0.1, -1.0)]
        POLYS.append(([(c[0] + r[0] * yy + h[0] * xx, c[1] + r[1] * yy + h[1] * xx) for xx, yy in arrow],
                      "255,210,0", 7.2, True))
        # 6) 장비: 굴착기(노랑) + 덤프트럭(주황) + 현장 사무실(흰)
        mid = (s2 + s3) / 2
        wz = (-hw + open_edge) / 2
        rect(mid - 40, wz, 6.0, 3.0, "245,190,0", 7.1)                  # 굴착기 몸체
        rect(mid - 40, wz, 2.6, 2.6, "60,60,60", 7.2)                   # 굴착기 회전부
        rect(mid - 34, wz + 0.3, 5.0, 0.7, "245,190,0", 7.2)            # 굴착기 팔
        rect(mid + 5, wz, 8.5, 2.6, "235,110,20", 7.1)                  # 덤프트럭 적재함
        rect(mid + 10.5, wz, 2.4, 2.6, "250,250,250", 7.2)              # 덤프트럭 운전석
        rect(mid + 45, wz, 7.0, 3.0, "235,235,230", 7.1)                # 현장 사무실
        rect(mid + 45, wz, 7.0, 0.5, "40,90,170", 7.2)                  # 사무실 띠
        # 7) 공사 안내판 (노란 마름모) 250 m, 150 m 앞 길가
        for dd in (250, 150):
            if s1 - dd < 25.0:                    # 교차로 안에 서지 않게
                continue
            p0, h, r = frame(s1 - dd)
            c = (p0[0] + r[0] * (hw + 2.8), p0[1] + r[1] * (hw + 2.8))
            POLYS.append(([(c[0] + 1.1 * math.cos(t * math.pi / 2), c[1] + 1.1 * math.sin(t * math.pi / 2))
                           for t in range(4)], "255,200,0", 7.0, True))
            POLYS.append(([(c[0] + 0.5 * math.cos(t * math.pi / 2), c[1] + 0.5 * math.sin(t * math.pi / 2))
                           for t in range(4)], "20,20,20", 7.1, True))
    return {nid["c1"], nid["c2"]}
