"""원지동 회전교차로 (서초종합체육관 앞) — 지하차도 위 회전교차로.

카카오맵(위성·일반지도)에서 읽은 위치를 기준으로 손으로 깐다.
  - 양재대로12길: 양재대로(양재IC 서쪽 측도)에서 남동쪽으로 갈라져 내려와,
    회전교차로 '아래'를 청계산지하차도(편도 2차로)로 지나 동쪽으로 이어진다.
  - 지상 측도(편도 1차로)가 지하차도 입구 앞에서 갈라져 회전교차로로 올라가고,
    회전교차로에서 나와 지하차도 출구 뒤에서 다시 본선에 합류한다 (가속차로 → 지퍼).
  - 회전교차로(반지름 18 m, 1차로)는 4갈래: 북서(측도), 동(측도), 남(양재대로12길 남쪽), 북(체육관).
  - 북쪽 갈래는 체육관 진입로: 회전교차로 → 공원 북쪽 가장자리를 따라 동북동 → 체육관 앞.
  - 높이: 지하차도 바닥 -2, 회전교차로·측도 지면 5 → 화면에서 위아래가 분명하다.
좌표는 원래(변형 전) 좌표. rebuild 에서 warp 로 옮긴다 (이 일대는 양재IC 와 함께 평행이동).
"""
import math

from env.roads import yangjae_ic as ic
from env.roads import seocho_ic as sc
from env.roads.arterials import Axis, _unit, _hdg

P = "ar_rb_"
Z0, ZT = ic.Z_GROUND, ic.Z_TUNNEL

J_REF = (-269.0, -326.0)               # 양재대로 ↔ 양재대로12길 교차점 (카카오맵)
RC = (-1.0, -584.0)                    # 회전교차로 중심 (카카오맵 위성)
RADIUS, RB_LW = 21.0, 4.5
AXIS = [(-269.0, -326.0), (-215.0, -388.0), (-162.0, -438.0), (-105.0, -505.0), (-47.0, -560.0),
        (0.0, -588.0), (52.0, -605.0), (103.0, -603.0), (231.0, -590.0), (340.0, -582.0), (430.0, -578.0)]
N_MAIN, LW, MED = 2, 3.3, 2.0
SIDE_LW, GAP = 3.5, 4.5
S_LEG_TO = (41.0, -713.0)              # 남쪽 갈래 방향
GYM = [(18.0, -537.0), (184.0, -465.0), (217.0, -479.0), (241.0, -500.0), (244.0, -507.0)]   # 26 → 27 → 28 → 체육관 입구
S_LEG_LEN = 190.0
# 체육관 건물 (카카오맵 일반지도 외곽선). 진입로 끝이 건물 북쪽 면에 닿는다.
GYM_BUILDING = [(212.0, -503.0), (240.0, -492.0), (276.0, -511.0), (293.0, -536.0), (281.0, -562.0),
                (220.0, -554.0), (207.0, -528.0)]
BUILDING_POLYS = []                    # (점 목록, 색, 층) — write_all 이 경관 파일에 넣는다 (변형 좌표)
D_SPLIT, PORTAL, FLAT = 175.0, 140.0, 60.0   # 회전교차로 중심(s)으로부터: 측도 분기/합류, 지하차도 시작, 바닥 평탄
ZIP_LEN = 60.0

GOAL_EDGE = P + "gym_in"               # 시험 차량 도착 엣지 (체육관 방향)
ROUNDABOUTS = []                       # write_all 이 <roundabout> 로 기록


def _split_at(nodes, edges, cons, eid, p, new_id, keep_first, nid):
    """eid 를 p 에서 가장 가까운 점으로 둘로 자른다. keep_first=True 면 앞쪽이 원래 id."""
    e = edges[eid]
    sh = e["shape"]
    k = min(range(1, len(sh) - 1), key=lambda i: math.dist(sh[i][:2], p))
    a = dict(e, shape=sh[:k + 1])
    b = dict(e, shape=sh[k:])
    nodes[nid] = tuple(sh[k])
    a["to"] = nid
    b["frm"] = nid
    if keep_first:
        edges[eid], edges[new_id] = a, b
        moved_from = eid                      # eid 에서 나가던 연결은 이제 new_id 에서 나간다
        for i, (x, y, fl, tl) in enumerate(cons):
            if x == moved_from:
                cons[i] = (new_id, y, fl, tl)
    else:
        edges[new_id], edges[eid] = a, b
        for i, (x, y, fl, tl) in enumerate(cons):
            if y == eid:
                cons[i] = (x, new_id, fl, tl)
    return sh[k]


def rebuild(nodes, edges, cons, warp=None):
    w = warp or (lambda x, y: (x, y))
    W = lambda p: w(p[0], p[1])                                      # noqa: E731
    new_e, new_c, zipper, tls = {}, [], set(), set()

    def E(eid, frm, to, shape, n, name, lw, speed, prio):
        new_e[eid] = dict(frm=frm, to=to, n=n, lw=lw, speed=speed, prio=prio, name=name,
                          shape=shape, spread="center")

    # ── 1) 양재대로 측도를 교차점 J 에서 자른다 ──
    jr = W(J_REF)
    pa = _split_at(nodes, edges, cons, "EBf_a", jr, "EBf_a0", keep_first=False, nid=P + "J_eb")
    pb = _split_at(nodes, edges, cons, "WBf_f", jr, "WBf_g", keep_first=True, nid=P + "J_wb")
    # 두 측도 노드를 하나의 신호 교차점으로 합친다 (노드는 가운데, 도로 모양은 그대로)
    J = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
    for k in ("EBf_a0", "EBf_a", "WBf_f", "WBf_g"):
        e = edges[k]
        if e["to"] in (P + "J_eb", P + "J_wb"):
            e["to"] = P + "J"
        else:
            e["frm"] = P + "J"
    nodes.pop(P + "J_eb", None)
    nodes.pop(P + "J_wb", None)
    nodes[P + "J"] = (J[0], J[1], Z0)
    tls.add(P + "J")

    # ── 2) 본선 축 ──
    pts = [J] + [W(p) for p in AXIS[1:]]
    raw = ic._resample(pts, 2.0)
    ax = Axis(ic._gauss_smooth(raw, 28.0, step=2.0, anchor=30.0))
    rc = W(RC)
    sR = ax.proj(rc)
    o_main = MED / 2 + N_MAIN * LW / 2

    def zt(s):
        d = abs(s - sR)
        if d <= FLAT:
            return ZT
        if d >= PORTAL:
            return Z0
        return ZT + (Z0 - ZT) * ic._smooth((d - FLAT) / (PORTAL - FLAT))

    def cw(s_a, s_b, side, n, zf=lambda s: Z0, step=4.0):
        o = MED / 2 + n * LW / 2
        k = max(2, int(abs(s_b - s_a) / step))
        return [(*ax.pos(s_a + (s_b - s_a) * i / k, side * o), zf(s_a + (s_b - s_a) * i / k))
                for i in range(k + 1)]

    def cw_node(nid, s, side, n):
        x, y = ax.pos(s, side * (MED / 2 + n * LW / 2))
        nodes[nid] = (x, y, Z0)

    sF0, sF1 = sR - D_SPLIT, sR + D_SPLIT
    L = ax.L
    sp_m, sp_s, sp_rb = 50 / 3.6, 40 / 3.6, 25 / 3.6
    tn, mn = "청계산지하차도", "양재대로12길"

    # 정방향 (J → 동쪽 끝), 오른쪽 = +1
    cw_node(P + "f_div", sF0, +1, N_MAIN)
    cw_node(P + "f_mrg", sF1, +1, N_MAIN)
    cw_node(P + "f_zip", sF1 + ZIP_LEN, +1, N_MAIN)
    cw_node(P + "f_end", L, +1, N_MAIN)
    E(P + "f_app", P + "J", P + "f_div", cw(0.0, sF0, +1, N_MAIN), N_MAIN, mn, LW, sp_m, 6)
    E(P + "f_tun", P + "f_div", P + "f_mrg", cw(sF0, sF1, +1, N_MAIN, zt), N_MAIN, tn, LW, sp_m, 6)
    E(P + "f_acc", P + "f_mrg", P + "f_zip", cw(sF1, sF1 + ZIP_LEN, +1, N_MAIN + 1), N_MAIN + 1, mn, LW, sp_m, 6)
    E(P + "f_dep", P + "f_zip", P + "f_end", cw(sF1 + ZIP_LEN, L, +1, N_MAIN), N_MAIN, mn, LW, sp_m, 6)
    # 역방향 (동쪽 끝 → J), 왼쪽 = -1
    cw_node(P + "b_end", L, -1, N_MAIN)
    cw_node(P + "b_div", sF1, -1, N_MAIN)
    cw_node(P + "b_mrg", sF0, -1, N_MAIN)
    cw_node(P + "b_zip", sF0 - ZIP_LEN, -1, N_MAIN)
    E(P + "b_app", P + "b_end", P + "b_div", cw(L, sF1, -1, N_MAIN), N_MAIN, mn, LW, sp_m, 6)
    E(P + "b_tun", P + "b_div", P + "b_mrg", cw(sF1, sF0, -1, N_MAIN, zt), N_MAIN, tn, LW, sp_m, 6)
    E(P + "b_acc", P + "b_mrg", P + "b_zip", cw(sF0, sF0 - ZIP_LEN, -1, N_MAIN + 1), N_MAIN + 1, mn, LW, sp_m, 6)
    E(P + "b_dep", P + "b_zip", P + "J", cw(sF0 - ZIP_LEN, 0.0, -1, N_MAIN), N_MAIN, mn, LW, sp_m, 6)
    zipper |= {P + "f_zip", P + "b_zip"}

    # ── 3) 회전교차로 갈래 방향과 노드 ──
    def leg_dir(theta_vec):
        return _unit(theta_vec)
    h_in = ax.at(sR - 30)[1]
    h_out = ax.at(sR + 30)[1]
    legs = {
        "nw": leg_dir((-h_in[0], -h_in[1])),
        "e": leg_dir(h_out),
        "s": leg_dir((W(S_LEG_TO)[0] - rc[0], W(S_LEG_TO)[1] - rc[1])),
        "n": leg_dir((W(GYM[0])[0] - rc[0], W(GYM[0])[1] - rc[1])),
    }
    ang = {k: math.atan2(d[1], d[0]) for k, d in legs.items()}
    order = sorted(legs, key=lambda k: ang[k])               # 반시계 방향 순서
    for k in order:
        d = legs[k]
        nodes[P + "n_" + k] = (rc[0] + d[0] * RADIUS, rc[1] + d[1] * RADIUS, Z0)
    rb_edges = []
    for i, k in enumerate(order):
        k2 = order[(i + 1) % len(order)]
        a0, a1 = ang[k], ang[k2]
        if a1 <= a0:
            a1 += 2 * math.pi
        m = max(4, int((a1 - a0) * RADIUS / 3))
        arc = [(rc[0] + RADIUS * math.cos(a0 + (a1 - a0) * j / m), rc[1] + RADIUS * math.sin(a0 + (a1 - a0) * j / m), Z0)
               for j in range(m + 1)]
        eid = f"{P}c_{k}_{k2}"
        E(eid, P + "n_" + k, P + "n_" + k2, arc, 1, "원지동 회전교차로", RB_LW, sp_rb, 8)
        rb_edges.append(eid)
    ROUNDABOUTS.clear()
    ROUNDABOUTS.append(([P + "n_" + k for k in order], rb_edges))

    def leg_pt(k, dist, side):
        """갈래 k 방향으로 중심에서 dist, 진행 기준 옆으로 side(+: 갈래 바깥 방향 기준 오른쪽)."""
        d = legs[k]
        r = (d[1], -d[0])
        return (rc[0] + d[0] * dist + r[0] * side, rc[1] + d[1] * dist + r[1] * side)

    so = SIDE_LW / 2 + 0.3                                    # 갈래 양방향 차로 사이 반폭

    def side_road(s_from, s_to, side, k, entering):
        """본선 바깥 차로 ↔ 회전교차로 갈래 노드. entering=True: 본선→회전교차로."""
        sg = 1 if s_to > s_from else -1
        o_side = MED / 2 + N_MAIN * LW + GAP + SIDE_LW / 2       # 나란히 달릴 때 축에서의 거리
        o_close = MED / 2 + N_MAIN * LW - SIDE_LW / 2            # 갈라지는 순간 (본선 바깥 차로와 겹침)
        out = []
        s_par_end = sR - sg * (FLAT + 5)
        # 본선과 나란한 구간
        n1 = max(3, int(abs(s_par_end - s_from) / 4))
        for i in range(n1 + 1):
            s = s_from + (s_par_end - s_from) * i / n1
            f = ic._smooth(min(1.0, abs(s - s_from) / 60.0))
            o = o_close + (o_side - o_close) * f
            out.append(ax.pos(s, side * o))
        # 회전교차로 갈래로 (접선 베지어). 들어가는 차로 = 갈래 바깥방향 기준 왼쪽(-), 나오는 차로 = 오른쪽(+)
        tgt = leg_pt(k, RADIUS + 1.0, -so if entering else so)
        h_end = (-legs[k][0], -legs[k][1])                    # 중심을 향해
        h0 = ax.at(s_par_end)[1]
        h0 = (h0[0] * sg, h0[1] * sg)
        bz = ic._bezier(out[-1], h0, tgt, h_end, n=14, k=0.45)
        return out[:-1] + bz

    # 정방향 측도: f_div → (북서 갈래) 회전교차로
    sh = side_road(sF0, sR, +1, "nw", True)
    E(P + "s_nw_in", P + "f_div", P + "n_nw", [(x, y, Z0) for x, y in sh], 1, mn, SIDE_LW, sp_s, 5)
    # 회전교차로 → (북서 갈래) → b_mrg (역방향 본선 합류)
    sh = side_road(sF0, sR, -1, "nw", False)[::-1]
    E(P + "s_nw_out", P + "n_nw", P + "b_mrg", [(x, y, Z0) for x, y in sh], 1, mn, SIDE_LW, sp_s, 5)
    # 역방향 측도: b_div → (동 갈래) 회전교차로
    sh = side_road(sF1, sR, -1, "e", True)
    E(P + "s_e_in", P + "b_div", P + "n_e", [(x, y, Z0) for x, y in sh], 1, mn, SIDE_LW, sp_s, 5)
    # 회전교차로 → (동 갈래) → f_mrg
    sh = side_road(sF1, sR, +1, "e", False)[::-1]
    E(P + "s_e_out", P + "n_e", P + "f_mrg", [(x, y, Z0) for x, y in sh], 1, mn, SIDE_LW, sp_s, 5)

    # 남쪽 갈래 (양방향 1차로, 막다른 길)
    def straight_leg(k, Lleg, name, eid_in, eid_out, nin, nout):
        a, b = RADIUS + 1.0, RADIUS + Lleg
        kk = max(3, int(Lleg / 4))
        p_in = [leg_pt(k, b - (b - a) * i / kk, -so) for i in range(kk + 1)]
        p_out = [leg_pt(k, a + (b - a) * i / kk, so) for i in range(kk + 1)]
        nodes[nin] = (*p_in[0], Z0)
        nodes[nout] = (*p_out[-1], Z0)
        E(eid_in, nin, P + "n_" + k, [(x, y, Z0) for x, y in p_in], 1, name, SIDE_LW, sp_s, 5)
        E(eid_out, P + "n_" + k, nout, [(x, y, Z0) for x, y in p_out], 1, name, SIDE_LW, sp_s, 5)
    straight_leg("s", S_LEG_LEN, "양재대로12길", P + "s_in", P + "s_out", P + "s_i", P + "s_o")
    # 남쪽 갈래 끝: 건물(트럭터미널 쪽)로 들어가는 것처럼 — 길 끝을 덮는 건물 하나
    d = legs["s"]
    r = (d[1], -d[0])
    c0 = (rc[0] + d[0] * (RADIUS + S_LEG_LEN + 22.0), rc[1] + d[1] * (RADIUS + S_LEG_LEN + 22.0))
    hw_, hl_ = 34.0, 24.0                      # 길 방향과 직각으로 넓은 건물
    s_bld = [(c0[0] + r[0] * a + d[0] * b, c0[1] + r[1] * a + d[1] * b)
             for a, b in ((-hw_, -hl_), (hw_, -hl_), (hw_, hl_), (-hw_, hl_))]

    # 북쪽 갈래: 체육관 진입로 (26 → 27 → 28 → 체육관 앞), 양방향 1차로
    g = [W(p) for p in GYM]
    start = leg_pt("n", RADIUS + 1.0, 0.0)
    path = [start] + g
    path = ic._gauss_smooth(ic._resample(path, 2.0), 8.0, step=2.0, anchor=6.0)
    gx = Axis(path)

    def gline(s_a, s_b, side):
        k = max(3, int(abs(s_b - s_a) / 3))
        return [(*gx.pos(s_a + (s_b - s_a) * i / k, side * so), Z0) for i in range(k + 1)]
    pe_in, pe_out = gx.pos(gx.L, so), gx.pos(gx.L, -so)
    nodes[P + "gym_i"] = (*pe_in, Z0)
    nodes[P + "gym_o"] = (*pe_out, Z0)
    E(P + "gym_in", P + "n_n", P + "gym_i", gline(0.0, gx.L, +1), 1, "체육관 진입로", SIDE_LW, sp_s, 4)
    E(P + "gym_out", P + "gym_o", P + "n_n", gline(gx.L, 0.0, -1), 1, "체육관 진입로", SIDE_LW, sp_s, 4)

    # 건물 (경관 다각형). 지붕 높이(층 8)가 차로보다 위라서 길이 건물 안으로 들어가는 것처럼 보인다.
    if warp is not None:
        BUILDING_POLYS.clear()
        BUILDING_POLYS.append(([W(p) for p in GYM_BUILDING], "150,156,168", 8.0))   # 서초종합체육관
        BUILDING_POLYS.append((s_bld, "150,150,150", 8.0))                           # 남쪽 건물

    # ── 4) 연결 ──
    for i in range(N_MAIN):
        new_c.append((P + "f_app", P + "f_tun", i, i))
        new_c.append((P + "b_app", P + "b_tun", i, i))
        new_c.append((P + "f_tun", P + "f_acc", i, i + 1))
        new_c.append((P + "b_tun", P + "b_acc", i, i + 1))
    new_c.append((P + "f_app", P + "s_nw_in", 0, 0))
    new_c.append((P + "b_app", P + "s_e_in", 0, 0))
    new_c.append((P + "s_e_out", P + "f_acc", 0, 0))
    new_c.append((P + "s_nw_out", P + "b_acc", 0, 0))
    for d in ("f", "b"):
        new_c.append((P + d + "_acc", P + d + "_dep", 0, 0))
        for i in range(1, N_MAIN + 1):
            new_c.append((P + d + "_acc", P + d + "_dep", i, i - 1))

    # 교차점 J: 차로별 회전
    apps = ["EBf_a0", "WBf_f", P + "b_dep"]
    exits = ["EBf_a", "WBf_g", P + "f_app"]
    allE = dict(edges)
    allE.update(new_e)
    for a in apps:
        ea = allE[a]
        ha, na = _hdg(ea["shape"], True), ea["n"]
        for b in exits:
            eb = allE[b]
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
    # 기존 양재IC 연결 중 J 로 바뀐 끝(EBf_a0 의 끝 / WBf_f 의 끝)에서 나가던 것은 없음 (원래 막다른 끝)
    edges.update(new_e)
    cons.extend(new_c)
    return tls, zipper
