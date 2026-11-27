"""순환 코스의 일반도로(남부순환로·언주로·양재대로)를 한 줄씩 손으로 깐다.

  - 도로마다 차로 수·폭·중앙분리대가 처음부터 끝까지 일정 (지도 조각을 잇지 않음 → 끊김/뭉침 없음).
  - 축은 기준점들을 지나는 꺾은선을 크게 다듬은 완만한 곡선.
  - 교차로는 모두 평면 신호 사거리. 교차로 사이를 잇지 않는 팔은 실측 방향의 일직선 막다른 길
    (배경차 출발/도착점). 지하차도(양재·구룡·개포)는 만들지 않는다.
  - 차로별 회전: 바깥 = 우회전+직진, 가운데 = 직진, 안쪽 = 직진+좌회전.
  - 직선 구간 줄이기(warp)를 먼저 적용한 좌표 위에 깐다 → 변형으로 꺾이는 곳이 없다.
좌표는 모두 '원래(변형 전) 좌표'로 적고, rebuild 에서 warp 로 옮긴다.
"""
import math

from env.roads import yangjae_ic as ic
from env.roads import seocho_ic as sc

P = "ar_"
Z0 = sc.Z0
TUBE = 45.0
X_R = 170.0

# 교차로: 중심, 막다른 팔 [(이름 목록, 그쪽 방향의 한 점, 차로 수, 길이)]
XS = {
    "ys": dict(label="양재역", c=(-471.0, 2317.0),
               arms=[(("강남대로",), (-520.0, 2420.0), 3, 150.0), (("강남대로",), (-420.0, 2220.0), 3, 150.0)]),
    "nh": dict(label="논현로", c=(173.0, 2428.0),
               arms=[(("논현로",), (170.0, 2540.0), 3, 130.0), (("논현로",), (200.0, 2330.0), 3, 130.0)]),
    "mb": dict(label="매봉", c=(922.0, 2742.0),
               arms=[(("언주로",), (870.0, 2860.0), 3, 130.0)]),
    "kp": dict(label="개포", c=(1715.0, 1368.0),
               arms=[(("양재대로",), (1850.0, 1445.0), 4, 150.0),
                     (("개포지하도로", "언주로"), (1790.0, 1270.0), 3, 150.0)]),
    # 염곡사거리 (보기 좋게: 염곡동서지하차도가 끝나고 조금 지나서). 0번 팔(북서) 끝에서 유턴 — hq.py
    "yg": dict(label="염곡사거리", c=(887.0, 742.0),
               arms=[(("염곡로",), (677.0, 955.0), 3, 300.0), (("염곡로",), (992.0, 636.0), 3, 150.0)]),
    "gr": dict(label="구룡(양재대로×논현로)", c=(1021.0, 875.0),
               arms=[(("논현로",), (950.0, 975.0), 3, 130.0), (("논현로",), (1060.0, 830.0), 3, 130.0)]),
}

# 도로: 시작 → (교차로들) → 끝. 'guide' 는 교차로 사이 축이 지날 실측 점.
ROADS = [
    dict(key="ns", name="남부순환로", n=3, lw=3.3, med=2.0, speed=60 / 3.6, prio=7,
         start=("seocho",), stops=["ys", "nh", "mb"], end=("dead", (1053.0, 2815.0), 150.0), guide={}),
    dict(key="ej", name="언주로", n=3, lw=3.3, med=2.0, speed=60 / 3.6, prio=7,
         start=("X", "mb", (1001.0, 2541.0)), stops=["kp"], end=None,
         guide={0: [(1105.0, 2297.0), (1269.0, 1934.0), (1544.0, 1565.0)]}),
    dict(key="yj", name="양재대로", n=4, lw=3.2, med=None, speed=60 / 3.6, prio=7,
         start=("yangjae", "E_out", "W_in"), stops=["yg", "gr", "kp"], end=None,
         guide={2: [(1441.0, 1180.0)]}),
]
X_SPEED, X_PRIO, X_LW, X_MED = 50 / 3.6, 6, 3.3, 2.0


class Axis:
    def __init__(self, pts):
        self.pts = pts
        self.s = [0.0]
        for a, b in zip(pts, pts[1:]):
            self.s.append(self.s[-1] + math.dist(a, b))
        self.L = self.s[-1]

    def at(self, s):
        s = min(max(s, 0.0), self.L)
        lo, hi = 0, len(self.s) - 1
        while hi - lo > 1:
            m = (lo + hi) // 2
            if self.s[m] <= s:
                lo = m
            else:
                hi = m
        a, b = self.pts[lo], self.pts[hi]
        L = max(self.s[hi] - self.s[lo], 1e-9)
        f = (s - self.s[lo]) / L
        return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f), ((b[0] - a[0]) / L, (b[1] - a[1]) / L)

    def pos(self, s, off=0.0):
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
            d = math.dist(p, q)
            if d < best[0]:
                best = (d, self.s[i] + u * L)
        return best[1]

    def dist(self, p):
        return math.dist(p, self.pos(self.proj(p)))


def _unit(v):
    L = math.hypot(*v) or 1.0
    return (v[0] / L, v[1] / L)


def _pca_dir(pts):
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    syy = sum((p[1] - my) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    th = 0.5 * math.atan2(2 * sxy, sxx - syy)
    return (math.cos(th), math.sin(th))


def _hdg(sh, at_end):
    a, b = (sh[-2], sh[-1]) if at_end else (sh[0], sh[1])
    return math.atan2(b[1] - a[1], b[0] - a[0])


def rebuild(nodes, edges, cons, warp=None):
    w = warp or (lambda x, y: (x, y))
    W = lambda p: w(p[0], p[1])                                       # noqa: E731
    new_e, new_c, tls = {}, [], set()
    X = {k: dict(v, c=W(v["c"]), app=[], exit=[]) for k, v in XS.items()}

    def E(eid, frm, to, shape, n, name, lw, speed, prio):
        new_e[eid] = dict(frm=frm, to=to, n=n, lw=lw, speed=speed, prio=prio, name=name,
                          shape=shape, spread="center")

    # 막다른 팔 방향 (실측 점 PCA, 지우기 전에)
    def arm_dir(xc, names, toward):
        t = _unit((toward[0] - xc[0], toward[1] - xc[1]))
        pts = [p[:2] for e in edges.values() if (e.get("name") or "") in names
               for p in e["shape"] if 15 < math.dist(p[:2], xc) < 170
               and ((p[0] - xc[0]) * t[0] + (p[1] - xc[1]) * t[1]) > 0]
        if len(pts) < 4:
            return t
        d = _pca_dir(pts + [xc])
        return d if d[0] * t[0] + d[1] * t[1] > 0 else (-d[0], -d[1])

    for k, x in X.items():
        x["arm_dirs"] = [arm_dir(x["c"], names, W(tw)) for names, tw, _, _ in x["arms"]]

    # 도로 축 만들기 (지우기 전에 남부순환로 동쪽 방향 등 실측 참고)
    axes = {}
    for r in ROADS:
        pts, h0 = [], None
        st = r["start"]
        if st[0] == "seocho":
            E0 = sc.END
            m0, h0 = W(E0["mid"]), E0["heading"]
            r["_med"] = E0["med"]
            pts = [m0, (m0[0] + h0[0] * 80, m0[1] + h0[1] * 80)]
        elif st[0] == "yangjae":
            eo, wi = edges[st[1]], edges[st[2]]
            pa, pb = eo["shape"][-1][:2], wi["shape"][0][:2]
            m0 = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
            h0 = _unit((eo["shape"][-1][0] - eo["shape"][-3][0], eo["shape"][-1][1] - eo["shape"][-3][1]))
            sep = math.dist(pa, pb)
            r["_med"] = max(1.0, sep - r["n"] * r["lw"])
            pts = [m0, (m0[0] + h0[0] * 80, m0[1] + h0[1] * 80)]
        elif st[0] == "X":
            xc = X[st[1]]["c"]
            d = arm_dir(xc, (r["name"],), W(st[2]))
            r["_start_dir"] = d
            r["_med"] = r["med"]
            pts = [xc, (xc[0] + d[0] * 90, xc[1] + d[1] * 90)]
        for i, sk in enumerate(r["stops"]):
            for g in r["guide"].get(i, []):
                pts.append(W(g))
            pts.append(X[sk]["c"])
        if r["end"] and r["end"][0] == "dead":
            last = pts[-1]
            t = W(r["end"][1])
            d = _unit((t[0] - last[0], t[1] - last[1]))
            pts.append((last[0] + d[0] * r["end"][2], last[1] + d[1] * r["end"][2]))
        raw = ic._resample(pts, 2.0)
        axes[r["key"]] = Axis(ic._gauss_smooth(raw, 45.0, step=2.0, anchor=60.0))

    # 기존 도로 지우기: 축 주변(같은 이름이면 넓게), 교차로 주변. 대문자(양재IC·서초IC) 제외
    names = {r["name"] for r in ROADS} | {"양재지하차도", "구룡지하차도", "개포지하도로"}
    for eid in [k for k, e in edges.items() if not k[0].isupper() and not k.startswith(P)]:
        e = edges[eid]
        nm = e.get("name") or ""
        tube = 150.0 if nm in names else TUBE
        hit = False
        for p in e["shape"][::2] + [e["shape"][-1]]:
            q = p[:2]
            if any(math.dist(q, x["c"]) < X_R for x in X.values()):
                hit = True
                break
            for a in axes.values():
                s = a.proj(q)
                if -20 < s < a.L + 20 and a.dist(q) < tube:
                    hit = True
                    break
            if hit:
                break
        if hit:
            del edges[eid]

    # 교차로 노드
    for k, x in X.items():
        nodes[P + k] = (x["c"][0], x["c"][1], Z0)
        tls.add(P + k)

    # 도로 깔기
    for r in ROADS:
        ax = axes[r["key"]]
        n, lw, med = r["n"], r["lw"], r["_med"]
        o = med / 2 + n * lw / 2
        rk = P + r["key"]

        def line(s_a, s_b, side, step=4.0):
            k = max(2, int(abs(s_b - s_a) / step))
            return [(*ax.pos(s_a + (s_b - s_a) * i / k, side * o), Z0) for i in range(k + 1)]

        stops = [0.0] + [ax.proj(X[sk]["c"]) for sk in r["stops"]]
        fn = [None] + [P + sk for sk in r["stops"]]
        bn = [None] + [P + sk for sk in r["stops"]]
        st = r["start"]
        if st[0] == "seocho":
            fn[0], bn[0] = sc.END["eb_node"], sc.END["wb_node"]
        elif st[0] == "yangjae":
            fn[0], bn[0] = edges[st[1]]["to"], edges[st[2]]["frm"]
        else:
            fn[0] = bn[0] = P + st[1]
        if r["end"] and r["end"][0] == "dead":
            stops.append(ax.L)
            pe, pw = ax.pos(ax.L, o), ax.pos(ax.L, -o)
            nodes[rk + "_e_f"] = (pe[0], pe[1], Z0)
            nodes[rk + "_e_b"] = (pw[0], pw[1], Z0)
            fn.append(rk + "_e_f")
            bn.append(rk + "_e_b")
        segs = len(stops) - 1
        for i in range(segs):
            ef, eb = f"{rk}_f{i}", f"{rk}_b{i}"
            E(ef, fn[i], fn[i + 1], line(stops[i], stops[i + 1], +1), n, r["name"], lw, r["speed"], r["prio"])
            E(eb, bn[i + 1], bn[i], line(stops[i + 1], stops[i], -1), n, r["name"], lw, r["speed"], r["prio"])
            # 교차로 등록
            if fn[i + 1].startswith(P) and fn[i + 1][len(P):] in X:
                X[fn[i + 1][len(P):]]["app"].append(ef)
                X[fn[i + 1][len(P):]]["exit"].append(eb)
            if fn[i].startswith(P) and fn[i][len(P):] in X:
                X[fn[i][len(P):]]["exit"].append(ef)
                X[fn[i][len(P):]]["app"].append(eb)
        # 시작 경계를 기존 노드 위치에 맞추고 차로 연결
        if st[0] in ("seocho", "yangjae"):
            for eid in (f"{rk}_f0", f"{rk}_b0"):
                e = new_e[eid]
                nid = e["frm"] if eid.endswith("f0") else e["to"]
                at_end = e["to"] == nid
                sh = e["shape"]
                e["shape"] = [(x_, y_, z) for (x_, y_), (_, _, z) in
                              zip(sc._blend(sh, nodes[nid][:2], at_end, L=50.0), sh)]
            if st[0] == "seocho":
                # 서초IC 마지막 조각(SC_EB_5 / SC_WB_0)과 한 엣지로 합친다 → 경계 교차점(잔 조각) 없음
                f0, b0 = new_e[f"{rk}_f0"], new_e[f"{rk}_b0"]
                e5, w0 = edges.pop("SC_EB_5"), edges.pop("SC_WB_0")
                f0["frm"] = e5["frm"]
                f0["shape"] = list(e5["shape"]) + list(f0["shape"][1:])
                b0["to"] = w0["to"]
                b0["shape"] = list(b0["shape"]) + list(w0["shape"][1:])
                for i_, (x_, y_, fl, tl) in enumerate(cons):
                    if y_ == "SC_EB_5":
                        cons[i_] = (x_, f"{rk}_f0", fl, tl)
                    if x_ == "SC_WB_0":
                        cons[i_] = (f"{rk}_b0", y_, fl, tl)
                nodes.pop(sc.END["eb_node"], None)
                nodes.pop(sc.END["wb_node"], None)
            else:
                for i in range(n):
                    new_c.append((st[1], f"{rk}_f0", i, i))
                    new_c.append((f"{rk}_b0", st[2], i, i))

    # 막다른 팔
    for k, x in X.items():
        xc = x["c"]
        for j, ((nms, _, cn, L), d) in enumerate(zip(x["arms"], x["arm_dirs"])):
            co = X_MED / 2 + cn * X_LW / 2
            rgt = (d[1], -d[0])                     # 팔 방향(바깥쪽) 기준 오른쪽

            def pt(t, side):                        # side +1: 바깥으로 나가는 차도(오른쪽)
                return (xc[0] + d[0] * t + rgt[0] * co * side, xc[1] + d[1] * t + rgt[1] * co * side, Z0)

            kk = max(2, int(L / 4))
            ni, no = f"{P}{k}_a{j}_i", f"{P}{k}_a{j}_o"
            nodes[ni] = pt(L, -1)
            nodes[no] = pt(L, +1)
            ei, eo = f"{P}{k}_a{j}_in", f"{P}{k}_a{j}_out"
            E(ei, ni, P + k, [pt(L - L * i / kk, -1) for i in range(kk + 1)], cn, nms[-1] if len(nms) > 1 else nms[0],
              X_LW, X_SPEED, X_PRIO)
            E(eo, P + k, no, [pt(L * i / kk, +1) for i in range(kk + 1)], cn, nms[-1] if len(nms) > 1 else nms[0],
              X_LW, X_SPEED, X_PRIO)
            x["app"].append(ei)
            x["exit"].append(eo)

    # 교차로 차로별 회전
    for k, x in X.items():
        for a in x["app"]:
            ea = new_e[a]
            ha, na = _hdg(ea["shape"], True), ea["n"]
            for b in x["exit"]:
                eb = new_e[b]
                hb, nb = _hdg(eb["shape"], False), eb["n"]
                turn = math.degrees((hb - ha + math.pi) % (2 * math.pi) - math.pi)
                if abs(turn) > 150:
                    continue
                if turn < -40:
                    new_c.append((a, b, 0, 0))
                elif turn > 40:
                    new_c.append((a, b, na - 1, nb - 1))
                else:
                    for i in range(na):
                        new_c.append((a, b, i, min(nb - 1, max(0, i - (na - nb)))))
    edges.update(new_e)
    cons.extend(new_c)
    return tls
