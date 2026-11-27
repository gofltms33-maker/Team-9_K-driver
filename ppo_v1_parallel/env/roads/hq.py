"""현대자동차 본사(양재사옥) 와 염곡사거리 — 출발점.

배치 (보기 좋게): 양재IC 동쪽 염곡동서지하차도가 끝나고 → 조금 지나서 → 현대차 본사 → 염곡사거리.
  - 염곡사거리는 양재대로(arterials 의 'yg' 교차로)에 있는 평면 신호 사거리.
  - 본사 정문은 양재대로 동행(북동행) 차도 바깥쪽, 사거리 105 m 앞.
시나리오 ⑤ '사옥 출구 직후 좌회전':
  - 정문을 나와 우회전으로 양재대로에 붙은 뒤, 약 85 m 안에 가장 안쪽 좌회전 차로로 옮겨 사거리에서 좌회전.
    가장 안쪽 차로는 좌회전 전용 → 신호 대기 줄이 길게 서고, 줄 선 차가 틈을 안 주면 못 들어간다.
  - 좌회전 → 북서쪽 길 끝에서 유턴 → 사거리로 돌아와 우회전 → 양재대로 서행 → 양재IC → 경부 상행
    (카카오맵 경로: 출발 → 1 → 2(염곡사거리) → 5 → 4 → 3(유턴) → … → 양재IC).
"""
import math

from env.roads import yangjae_ic as ic
from env.roads.arterials import _unit
from env.roads.wonji import _split_at

P = "ar_hq_"
Z0 = ic.Z_GROUND
X_NODE = "ar_yg"                 # 염곡사거리 (arterials)
APPROACH = "ar_yj_f0"            # 양재IC 쪽 → 염곡사거리 (북동행)
NW_ARM = 0                       # arterials XS['yg'] 의 북서 팔 번호 (끝에서 유턴)
GATE_BACK = 105.0               # 70 m 였을 때 정문~정지선이 약 50 m 라 세 차로를 옮기기가 너무 빠듯했음.
                                 # 양재IC 지하차도 출구~염곡사거리 사이 길이(약 130 m) 안에서 최대한 뒤로 뺀 값
DRIVE_LEN = 40.0
BUILD_W, BUILD_D = 120.0, 85.0   # 본사 건물: 도로 방향 폭, 안쪽 깊이
BUILD_SETBACK = 25.0
BUILDING_POLYS = []
COVER_POLYS = []
START_EDGE = P + "gate_out"


def rebuild(nodes, edges, cons, warp=None):
    new_e, new_c = {}, []

    def E(eid, frm, to, shape, n, name, lw, speed, prio):
        new_e[eid] = dict(frm=frm, to=to, n=n, lw=lw, speed=speed, prio=prio, name=name,
                          shape=[(x, y, Z0) for x, y in shape], spread="center")

    # 1) 정문: 진입로(북동행)를 사거리 GATE_BACK m 앞에서 자른다
    sh = edges[APPROACH]["shape"]
    acc, gp = 0.0, sh[0]
    for a_, b_ in zip(sh[::-1], sh[::-1][1:]):
        acc += math.dist(a_[:2], b_[:2])
        if acc >= GATE_BACK:
            gp = b_
            break
    _split_at(nodes, edges, cons, APPROACH, gp[:2], APPROACH + "b", keep_first=True, nid=P + "G")
    ap = edges[APPROACH]
    n, lw = ap["n"], ap["lw"]
    for i in range(n):
        new_c.append((APPROACH, APPROACH + "b", i, i))
    g = nodes[P + "G"][:2]
    s2 = edges[APPROACH + "b"]["shape"]
    hf = _unit((s2[min(3, len(s2) - 1)][0] - g[0], s2[min(3, len(s2) - 1)][1] - g[1]))
    side = (hf[1], -hf[0])                                      # 진행 기준 오른쪽 = 본사 쪽
    edge_off = n * lw / 2 + 1.0
    gpos = (g[0] + side[0] * edge_off, g[1] + side[1] * edge_off)
    a_ = (gpos[0] + side[0] * DRIVE_LEN, gpos[1] + side[1] * DRIVE_LEN)
    so = 2.0

    def dl(p0, p1, off, k=10):
        return [(p0[0] + (p1[0] - p0[0]) * i / k + hf[0] * off, p0[1] + (p1[1] - p0[1]) * i / k + hf[1] * off)
                for i in range(k + 1)]
    Hi, Ho = P + "HQ_i", P + "HQ_o"
    nodes[Ho] = (a_[0] + hf[0] * so, a_[1] + hf[1] * so, Z0)
    nodes[Hi] = (a_[0] - hf[0] * so, a_[1] - hf[1] * so, Z0)
    E(P + "gate_out", Ho, P + "G", dl(a_, gpos, so), 1, "현대차 본사 정문", 3.5, 20 / 3.6, 3)
    E(P + "gate_in", P + "G", Hi, dl(gpos, a_, -so), 1, "현대차 본사 정문", 3.5, 20 / 3.6, 3)
    new_c.append((P + "gate_out", APPROACH + "b", 0, 0))       # 우회전 합류
    new_c.append((APPROACH, P + "gate_in", 0, 0))               # 우회전 진입

    # 2) 염곡사거리: 북동행 가장 안쪽 차로는 좌회전 전용 (직진 연결 제거)
    app = APPROACH + "b"
    na = edges[app]["n"]
    for i in range(len(cons) - 1, -1, -1):
        a, b, fl, tl = cons[i]
        if a == app and fl == na - 1 and b == "ar_yj_f1":
            del cons[i]

    # 3) 북서 팔 끝에서 유턴: 두 막다른 노드를 하나로 합치고 유턴 연결
    ei, eo = f"ar_yg_a{NW_ARM}_in", f"ar_yg_a{NW_ARM}_out"
    if ei in edges and eo in edges:
        ni, no = edges[ei]["frm"], edges[eo]["to"]
        m = ((nodes[ni][0] + nodes[no][0]) / 2, (nodes[ni][1] + nodes[no][1]) / 2, Z0)
        nodes[P + "U"] = m
        edges[ei]["frm"] = P + "U"
        edges[eo]["to"] = P + "U"
        nodes.pop(ni, None)
        nodes.pop(no, None)
        nn = edges[eo]["n"]
        new_c.append((eo, ei, nn - 1, edges[ei]["n"] - 1))

    # 3-1) 구룡사거리 유턴: 염곡 좌회전 대기줄에 끼어들지 못하고 직진했을 때
    #      다음 사거리(구룡)에서 유턴해 양재대로 서행으로 돌아와 원래 경로에 합류할 수 있게.
    #      (가장 안쪽 차로 → 반대편 가장 안쪽 차로, 신호 사거리의 일반적인 유턴)
    if "ar_yj_f1" in edges and "ar_yj_b1" in edges and edges["ar_yj_f1"]["to"] == edges["ar_yj_b1"]["frm"]:
        new_c.append(("ar_yj_f1", "ar_yj_b1", edges["ar_yj_f1"]["n"] - 1, edges["ar_yj_b1"]["n"] - 1))

    # 4) 본사 건물: 정문 안쪽, 도로와 나란한 큰 직사각형 (두 동)
    if warp is not None:
        BUILDING_POLYS.clear()
        c0 = (gpos[0] + side[0] * (DRIVE_LEN + BUILD_SETBACK + BUILD_D / 2 - 20.0),
              gpos[1] + side[1] * (DRIVE_LEN + BUILD_SETBACK + BUILD_D / 2 - 20.0))
        for off_along, wid in ((-BUILD_W * 0.27, BUILD_W * 0.42), (BUILD_W * 0.27, BUILD_W * 0.42)):
            cc = (c0[0] + hf[0] * off_along, c0[1] + hf[1] * off_along)
            pts = [(cc[0] + hf[0] * a + side[0] * b, cc[1] + hf[1] * a + side[1] * b)
                   for a, b in ((-wid / 2, -BUILD_D / 2), (wid / 2, -BUILD_D / 2),
                                (wid / 2, BUILD_D / 2), (-wid / 2, BUILD_D / 2))]
            BUILDING_POLYS.append((pts, "92,110,132", 8.0))
        # 정문 앞 광장 (진입로 끝을 덮음)
        pc = (gpos[0] + side[0] * (DRIVE_LEN + 6), gpos[1] + side[1] * (DRIVE_LEN + 6))
        plaza = [(pc[0] + hf[0] * a + side[0] * b, pc[1] + hf[1] * a + side[1] * b)
                 for a, b in ((-28, -10), (28, -10), (28, 12), (-28, 12))]
        BUILDING_POLYS.append((plaza, "150,150,146", 7.5))
    edges.update(new_e)
    cons.extend(new_c)
    return set(), set()
