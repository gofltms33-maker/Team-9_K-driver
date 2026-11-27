"""차로 수가 바뀌는 곳을 자연스럽게 (화면 표시).

SUMO 는 한 엣지 안에서 차로 폭이 일정해서 3차로 → 2차로가 되는 노드에서 바깥 가장자리가 '툭' 꺾인다.
도로망은 그대로 두고, 꺾인 자리에 도로와 같은 색의 '사선 면'(+ 연석 선)을 그려
넓은 쪽 바깥 가장자리 → 좁은 쪽 바깥 가장자리가 TAPER m 에 걸쳐 부드러운 곡선으로 이어지게 한다.
대상: 들어오는 엣지 1개 · 나가는 엣지 1개 뿐인 노드에서 차로 수가 다른 곳 (램프 합류·분류, 공사 구간 제외).
"""
import math

from env.roads import yangjae_ic as ic

TAPER = 45.0
ROAD_COL, CURB_COL = "0,0,0", "176,176,168"
LAY_ROAD, LAY_CURB = 4.5, 4.6


def _arc(sh):
    s = [0.0]
    for a, b in zip(sh, sh[1:]):
        s.append(s[-1] + math.dist(a[:2], b[:2]))
    return s


def _pt_dir(sh, s):
    S = _arc(sh)
    s = max(0.0, min(S[-1], s))
    for i in range(1, len(S)):
        if S[i] >= s:
            a, b = sh[i - 1], sh[i]
            f = (s - S[i - 1]) / max(S[i] - S[i - 1], 1e-9)
            L = math.dist(a[:2], b[:2]) or 1.0
            return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f), ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
    a, b = sh[-2], sh[-1]
    L = math.dist(a[:2], b[:2]) or 1.0
    return b[:2], ((b[0] - a[0]) / L, (b[1] - a[1]) / L)


def _outer(sh, s, hw):
    p, h = _pt_dir(sh, s)
    return (p[0] + h[1] * hw, p[1] - h[0] * hw), h


def patches(nodes, edges, skip_nodes=()):
    """이어지는 두 엣지(진행방향 차이 < 20°) 사이 이음매를 도로색 사각형으로 메운다
    (서로 다른 차로 수·노드 근처 연석 조각이 비쳐 보이는 '점점점' 방지)."""
    ins, outs = {}, {}
    for k, e in edges.items():
        outs.setdefault(e["frm"], []).append(k)
        ins.setdefault(e["to"], []).append(k)
    polys = []
    for nid in nodes:
        if nid in skip_nodes or abs(nodes[nid][2] - ic.Z_GROUND) > 0.5:   # 다리·지하 노드는 제외
            continue
        for a in ins.get(nid, []):
            for b in outs.get(nid, []):
                A, B = edges[a], edges[b]
                pa, ha = _pt_dir(A["shape"], 1e9)
                pb, hb = _pt_dir(B["shape"], 0.0)
                if ha[0] * hb[0] + ha[1] * hb[1] < math.cos(math.radians(20)):
                    continue
                if math.dist(pa, pb) > 25:
                    continue
                wa, wb = A["n"] * A["lw"] / 2, B["n"] * B["lw"] / 2
                ra, rb = (ha[1], -ha[0]), (hb[1], -hb[0])
                back = (pa[0] - ha[0] * 3.0, pa[1] - ha[1] * 3.0)
                fwd = (pb[0] + hb[0] * 3.0, pb[1] + hb[1] * 3.0)
                quad = [(back[0] - ra[0] * wa, back[1] - ra[1] * wa), (back[0] + ra[0] * wa, back[1] + ra[1] * wa),
                        (fwd[0] + rb[0] * wb, fwd[1] + rb[1] * wb), (fwd[0] - rb[0] * wb, fwd[1] - rb[1] * wb)]
                polys.append((quad, ROAD_COL, 4.4))
    return polys


def wedges(nodes, edges, skip=("_c1", "_c2", "_c3")):
    ins, outs = {}, {}
    for k, e in edges.items():
        outs.setdefault(e["frm"], []).append(k)
        ins.setdefault(e["to"], []).append(k)
    polys = []
    for nid in nodes:
        if any(nid.endswith(x) for x in skip):
            continue
        i_, o_ = ins.get(nid, []), outs.get(nid, [])
        if len(i_) != 1 or len(o_) != 1:
            continue
        A, B = edges[i_[0]], edges[o_[0]]
        if A["n"] == B["n"]:
            continue
        sa, sb = A["shape"], B["shape"]
        La, Lb = _arc(sa)[-1], _arc(sb)[-1]
        hwa, hwb = A["n"] * A["lw"] / 2, B["n"] * B["lw"] / 2
        if A["n"] > B["n"]:                       # 줄어듦: A 끝 바깥 → B 바깥 TAPER m 지점
            T = min(TAPER, Lb * 0.6)
            p0, h0 = _outer(sa, La, hwa)
            q0, _ = _outer(sb, 0.0, hwb)
            q1, h1 = _outer(sb, T, hwb)
            curve = ic._bezier(p0, h0, q1, h1, n=16, k=0.45)
            edge_pts = [_outer(sb, T - T * j / 12, hwb)[0] for j in range(13)]   # q1 → q0
            poly = curve + edge_pts[1:] + [p0]
        else:                                     # 늘어남: A 바깥 TAPER m 앞 → B 시작 바깥
            T = min(TAPER, La * 0.6)
            p1, h1 = _outer(sa, La - T, hwa)
            p0, _ = _outer(sa, La, hwa)
            q0, h0 = _outer(sb, 0.0, hwb)
            curve = ic._bezier(p1, h1, q0, h0, n=16, k=0.45)
            edge_pts = [_outer(sa, La - T * j / 12, hwa)[0] for j in range(13)]  # p0 → p1 (뒤로)
            poly = curve + [p0] + edge_pts[1:]
        polys.append((poly, ROAD_COL, LAY_ROAD))
        polys.append((ic._band(curve, -0.35, 0.35), CURB_COL, LAY_CURB))
    return polys


LINE_COL, LINE_W = "225,225,215", 0.30


def median_lines(nodes, edges, skip_nodes=(), prefixes=("ar_", "SC_")):
    """마주 보고 달리는 두 차도 사이 가운데에 연한 실선을 그린다 (어느 도로든 같은 모양).
    - 내 손으로 깐 도로(ar_*, SC_*)만 대상, 서로 반대 방향이고 안쪽 가장자리 사이가 6 m 이내인 구간.
    - 신호 교차로 중심 18 m 안은 비운다."""
    E = {k: e for k, e in edges.items() if k.startswith(prefixes) and e["n"] >= 2}

    def samples(e):
        sh = e["shape"]
        S = _arc(sh)
        out = []
        s = 0.0
        while s <= S[-1]:
            p, h = _pt_dir(sh, s)
            z = sh[min(range(len(S)), key=lambda i: abs(S[i] - s))][2]
            hw = e["n"] * e["lw"] / 2
            out.append(((p[0] - h[1] * hw, p[1] + h[0] * hw), h, z))     # 안쪽(왼쪽) 가장자리
            s += 2.0
        return out
    smp = {k: samples(e) for k, e in E.items()}
    box = {}
    for k, pts in smp.items():
        xs = [p[0][0] for p in pts]
        ys = [p[0][1] for p in pts]
        box[k] = (min(xs) - 8, min(ys) - 8, max(xs) + 8, max(ys) + 8)
    centers = [nodes[n][:2] for n in skip_nodes if n in nodes]
    polys = []
    keys = sorted(E)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ba, bb = box[a], box[b]
            if ba[2] < bb[0] or bb[2] < ba[0] or ba[3] < bb[1] or bb[3] < ba[1]:
                continue
            run = []
            for (p, h, z) in smp[a]:
                q = min(smp[b], key=lambda t: (t[0][0] - p[0]) ** 2 + (t[0][1] - p[1]) ** 2)
                d = math.dist(p, q[0])
                ok = d < 6.0 and (h[0] * q[1][0] + h[1] * q[1][1]) < -0.9 and abs(z - q[2]) < 0.5 \
                    and all(math.dist(p, c) > 18.0 for c in centers)
                if ok:
                    run.append((((p[0] + q[0][0]) / 2, (p[1] + q[0][1]) / 2), z))
                elif run:
                    polys += _emit(run)
                    run = []
            if run:
                polys += _emit(run)
    return polys


def _emit(run):
    if len(run) < 3:
        return []
    pts = [p for p, _ in run]
    z = max(z for _, z in run)
    return [(ic._band(pts, -LINE_W / 2, LINE_W / 2), LINE_COL, z + 0.2)]
