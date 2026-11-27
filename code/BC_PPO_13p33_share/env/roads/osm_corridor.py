"""순환 코스 '주변 도로' 생성기 — OSM 실측 좌표를 다듬어 SUMO plain XML 로 만든다.

어제 양재IC(env/roads/yangjae_ic.py)를 만든 방식과 같다:
  - OSM 중심선을 가우시안으로 매끄럽게 (꺾임 제거, 교차로 쪽 끝점·방향은 고정)
  - 양방향 도로는 중앙선 기준 좌/우 차로군으로, 일방통행(분리된 차로)은 중심선 그대로
  - 다리/지하차도는 OSM layer → 높이(z), 경사는 끝 60 m 에서 완만하게
  - 순환 코스 중심선에서 CORRIDOR m 이내 도로만 남기고 그 밖은 잘라 막다른 접속로로
  - 서로 가까운 교차 노드(JOIN m 이내)는 하나의 교차로로 묶음
양재IC 영역은 yangjae_ic 가 담당하므로 그 도로에 가까운 부분은 잘라내고, 경계에서 이어붙인다.
"""
import json
import math
import os

from env.roads.yangjae_ic import _gauss_smooth, _smooth, _Poly, Z_GROUND

LAT0, LON0 = 37.4636, 127.0394
KX = math.cos(math.radians(LAT0)) * 111320
KY = 110574.0
CORRIDOR = 260.0     # 순환 코스 중심선에서 이 거리 안의 도로만
JOIN = 22.0          # 이 거리 안의 교차 노드는 하나로
Z_PER_LAYER = 7.0

# 순환 코스 중심선 (양재IC → 경부 → 서초IC → 남부순환로 → 매봉 → 언주로 → 개포 → 양재대로 → 양재IC)
LOOP = [(-600, 830), (-1180, 2200), (-850, 2250), (-470, 2310), (172, 2437), (918, 2732),
        (1270, 1935), (1544, 1565), (1716, 1355), (1440, 1180), (1028, 880), (741, 597)]

DEFAULT_LANES = {"motorway": 4, "trunk": 3, "primary": 3, "secondary": 2, "tertiary": 1}
SPEED = {"motorway": 27.78, "trunk": 22.22, "primary": 16.67, "secondary": 13.89, "tertiary": 11.11}
PRIO = {"motorway": 10, "trunk": 9, "primary": 7, "secondary": 5, "tertiary": 3}
LW = 3.3


def _xy(g):
    return ((g["lon"] - LON0) * KX, (g["lat"] - LAT0) * KY)


def _seg_d(p, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    L2 = vx * vx + vy * vy or 1e-9
    t = max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / L2))
    return math.hypot(p[0] - a[0] - t * vx, p[1] - a[1] - t * vy)


def _poly_d(p, pl):
    return min(_seg_d(p, a, b) for a, b in zip(pl, pl[1:]))


def load_ways(paths):
    ways, seen = [], set()
    for p in paths:
        if not os.path.exists(p):
            continue
        for w in json.load(open(p, encoding="utf-8"))["elements"]:
            if w["id"] in seen or "geometry" not in w:
                continue
            seen.add(w["id"])
            ways.append(w)
    return ways


def build(paths, exclude_fn=None):
    """반환: nodes{id:(x,y,z)}, edges{id:dict(frm,to,n,lw,speed,prio,name,shape,spread)}

    exclude_fn(x, y) -> True 이면 그 점은 다른 모듈(양재IC)이 담당하므로 잘라낸다.
    """
    ways = load_ways(paths)
    # 1) 코리도 안의 연속 구간으로 자르기
    pieces = []          # (way, [(osm_node_id, x, y)])
    for w in ways:
        hw = w["tags"]["highway"]
        base = hw.replace("_link", "")
        if base not in DEFAULT_LANES:
            continue
        lim = CORRIDOR if base in ("motorway", "trunk", "primary") else CORRIDOR * 0.6
        raw = [(nid, *_xy(g)) for nid, g in zip(w["nodes"], w["geometry"])]
        pts = [raw[0]]                     # 10 m 간격으로 촘촘히 (잘라낼 때 짧은 조각이 사라지지 않게)
        for (n1, x1, y1), (n2, x2, y2) in zip(raw, raw[1:]):
            k = int(math.hypot(x2 - x1, y2 - y1) // 10)
            for j in range(1, k + 1):
                f = j / (k + 1)
                pts.append((f"{w['id']}:{n1}:{j}", x1 + (x2 - x1) * f, y1 + (y2 - y1) * f))
            pts.append((n2, x2, y2))
        cur = []
        for q in pts:
            inside = _poly_d(q[1:], LOOP) <= lim and not (exclude_fn and exclude_fn(q[1], q[2]))
            if inside:
                cur.append(q)
            else:
                if len(cur) >= 2:
                    pieces.append((w, cur))
                cur = []
        if len(cur) >= 2:
            pieces.append((w, cur))

    # 2) 교차 노드 찾기: 여러 조각이 공유하는 OSM 노드 + 조각 양 끝
    use = {}
    for w, pts in pieces:
        for i, (nid, x, y) in enumerate(pts):
            use.setdefault(nid, [x, y, 0, set()])
            use[nid][2] += 1 if 0 < i < len(pts) - 1 else 2
            use[nid][3].add(int(w["tags"].get("layer", "0") or 0))
    jn = {nid for nid, v in use.items() if v[2] >= 2}
    # 3) 교차 노드 묶기: '짧은 도로 조각(JOIN m 미만)으로 직접 이어진' 노드끼리만 하나로
    #    (거리만으로 묶으면 나란히 달리는 상·하행 차로가 한 점으로 꼬인다)
    par = {nid: nid for nid in jn}

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for w, pts in pieces:
        idx = [i for i, q in enumerate(pts) if q[0] in jn]
        for a_, b_ in zip(idx, idx[1:]):
            L = sum(math.dist(pts[i][1:], pts[i + 1][1:]) for i in range(a_, b_))
            if L < JOIN:
                ra, rb = find(pts[a_][0]), find(pts[b_][0])
                if ra != rb:
                    par[ra] = rb
    groups = {}
    for nid in jn:
        groups.setdefault(find(nid), []).append(nid)
    cluster, nodes = {}, {}
    for root, mem in groups.items():
        rid = f"n{root}"
        cx = sum(use[m][0] for m in mem) / len(mem)
        cy = sum(use[m][1] for m in mem) / len(mem)
        lay = max((l for m in mem for l in use[m][3]), key=abs)
        nodes[rid] = (cx, cy, Z_GROUND + Z_PER_LAYER * lay)
        for m in mem:
            cluster[m] = rid

    # 4) 엣지 만들기
    edges = {}
    for w, pts in pieces:
        t = w["tags"]
        hw = t["highway"]
        base = hw.replace("_link", "")
        link = hw.endswith("_link")
        oneway = t.get("oneway") in ("yes", "1", "true") or link or base in ("motorway",) \
            or t.get("junction") == "roundabout"
        rev = t.get("oneway") == "-1"
        try:
            lanes = int(str(t.get("lanes", "")).split(";")[0])
        except ValueError:
            lanes = 1 if link else DEFAULT_LANES[base] * (1 if oneway else 2)
        lay = int(t.get("layer", "0") or 0)
        name = t.get("name", "")
        # 교차 노드에서 쪼개기
        idx = [i for i, q in enumerate(pts) if q[0] in jn]
        if 0 not in idx:
            idx = [0] + idx
        if len(pts) - 1 not in idx:
            idx.append(len(pts) - 1)
        for k, (a, b) in enumerate(zip(idx, idx[1:])):
            seg = pts[a:b + 1]
            na = cluster.get(seg[0][0]) or _end_node(nodes, seg[0], lay)
            nb = cluster.get(seg[-1][0]) or _end_node(nodes, seg[-1], lay)
            if na == nb:
                continue
            xy = [(q[1], q[2]) for q in seg]
            xy[0] = nodes[na][:2]
            xy[-1] = nodes[nb][:2]
            L = sum(math.dist(p, q) for p, q in zip(xy, xy[1:]))
            if L < 4:
                continue
            sm = _gauss_smooth(xy, 7.0, step=1.5, anchor=min(10.0, L / 3)) if L > 12 else xy
            pl = _Poly(sm)
            za, zb, zl = nodes[na][2], nodes[nb][2], Z_GROUND + Z_PER_LAYER * lay
            ramp = min(60.0, pl.length / 2)

            def z_at(s):
                if s < ramp:
                    return za + (zl - za) * _smooth(s / ramp)
                if s > pl.length - ramp:
                    return zb + (zl - zb) * _smooth((pl.length - s) / ramp)
                return zl
            shape = [(p[0], p[1], z_at(s)) for p, s in zip(pl.p, pl.cum)]
            eid = f"w{w['id']}_{k}"
            if oneway:
                if rev:
                    shape, na, nb = shape[::-1], nb, na
                edges[eid] = dict(frm=na, to=nb, n=max(1, lanes), lw=3.5 if base in ("motorway", "trunk") else LW,
                                  speed=SPEED[base] * (0.7 if link else 1), prio=PRIO[base] - (2 if link else 0),
                                  name=name, shape=shape, spread="center")
            else:
                n = max(1, lanes // 2)
                edges[eid + "f"] = dict(frm=na, to=nb, n=n, lw=LW, speed=SPEED[base], prio=PRIO[base],
                                        name=name, shape=shape, spread="right")
                edges[eid + "r"] = dict(frm=nb, to=na, n=n, lw=LW, speed=SPEED[base], prio=PRIO[base],
                                        name=name, shape=shape[::-1], spread="right")
    return nodes, edges


def _end_node(nodes, q, lay):
    nid = f"e{q[0]}"
    if nid not in nodes:
        nodes[nid] = (q[1], q[2], Z_GROUND + Z_PER_LAYER * lay)
    return nid
