"""순환 코스 안쪽 연결 도로 (가운데가 비어 보이지 않게).

  A: 양재역 남쪽 팔 끝 ─────────────── 염곡사거리 북서 팔 끝 (유턴 지점 그대로)
  B: 논현로 남쪽 팔 끝 ─────────────── 구룡사거리 북서 팔 끝
  C: 언주로(매봉~개포 사이 T자) ── B 와 십자 교차 ── A 에 T자로 붙음
규칙
  - 모든 연결 도로: 편도 3차로 + 중앙분리대 2 m (교차 도로 팔과 같은 단면) → 팔 끝과 차로가 정확히 이어진다.
  - 팔 끝에서는 팔의 진행방향 그대로 40 m 곧게 이어 나간 뒤 완만하게 꺾인다.
  - 새 교차점(A–C, B–C, C–언주로)은 모두 신호 교차로. 차로별 회전: 바깥=우, 가운데=직진, 안쪽=좌.
"""
import math

from env.roads import yangjae_ic as ic
from env.roads.arterials import Axis, _unit, _hdg, X_MED, X_LW
from env.roads.wonji import _split_at

P = "ar_cn_"
Z0 = ic.Z_GROUND
N, LW, MED = 3, X_LW, X_MED
O = MED / 2 + N * LW / 2
SPEED, PRIO = 50 / 3.6, 6
EJ_T_S = 330.0              # 언주로 남행(ar_ej_f0) 시작에서 T자 교차점까지 (m)
STRAIGHT = 40.0
C_ON_A = 0.58               # C 가 A 에 붙는 위치 (A 길이 비율)


def _arm_end(nodes, edges, key, j):
    """팔 끝: (가운데 점, 바깥 방향, 나가는 엣지 끝 노드, 들어오는 엣지 시작 노드)."""
    eo, ei = edges[f"ar_{key}_a{j}_out"], edges[f"ar_{key}_a{j}_in"]
    po, pi = eo["shape"][-1][:2], ei["shape"][0][:2]
    c = ((po[0] + pi[0]) / 2, (po[1] + pi[1]) / 2)
    d = _unit((eo["shape"][-1][0] - eo["shape"][-4][0], eo["shape"][-1][1] - eo["shape"][-4][1]))
    return c, d, eo["to"], ei["frm"], po, pi


def _seg_inter(a0, a1, b0, b1):
    d1 = (a1[0] - a0[0], a1[1] - a0[1])
    d2 = (b1[0] - b0[0], b1[1] - b0[1])
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((b0[0] - a0[0]) * d2[1] - (b0[1] - a0[1]) * d2[0]) / den
    u = ((b0[0] - a0[0]) * d1[1] - (b0[1] - a0[1]) * d1[0]) / den
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (a0[0] + d1[0] * t, a0[1] + d1[1] * t)
    return None


def _cross(ax1, ax2):
    for i in range(len(ax1.pts) - 1):
        for j in range(len(ax2.pts) - 1):
            p = _seg_inter(ax1.pts[i], ax1.pts[i + 1], ax2.pts[j], ax2.pts[j + 1])
            if p:
                return p
    return None


def _axis(c0, d0, c1, d1, mid=()):
    pts = [c0, (c0[0] + d0[0] * STRAIGHT, c0[1] + d0[1] * STRAIGHT)] + list(mid) + \
          [(c1[0] + d1[0] * STRAIGHT, c1[1] + d1[1] * STRAIGHT), c1]
    raw = ic._resample(pts, 2.0)
    return Axis(ic._gauss_smooth(raw, 30.0, step=2.0, anchor=STRAIGHT))


def rebuild(nodes, edges, cons, warp=None):
    new_e, new_c, tls = {}, [], set()

    def E(eid, frm, to, shape, name):
        new_e[eid] = dict(frm=frm, to=to, n=N, lw=LW, speed=SPEED, prio=PRIO, name=name,
                          shape=[(x, y, Z0) for x, y in shape], spread="center")

    # 팔 끝
    A0 = _arm_end(nodes, edges, "ys", 1)
    A1 = _arm_end(nodes, edges, "yg", 0)
    B0 = _arm_end(nodes, edges, "nh", 1)
    B1 = _arm_end(nodes, edges, "gr", 0)
    axA = _axis(A0[0], A0[1], A1[0], A1[1])
    axB = _axis(B0[0], B0[1], B1[0], B1[1])

    # 언주로 T자: 남행(ar_ej_f0) 을 EJ_T_S 지점에서, 북행(ar_ej_b0) 을 가장 가까운 점에서 자른다
    f0 = edges["ar_ej_f0"]["shape"]
    acc, tp = 0.0, f0[-1]
    for a_, b_ in zip(f0, f0[1:]):
        acc += math.dist(a_[:2], b_[:2])
        if acc >= EJ_T_S:
            tp = b_
            break
    pa = _split_at(nodes, edges, cons, "ar_ej_f0", tp[:2], "ar_ej_f0x", keep_first=True, nid=P + "T_f")
    pb = _split_at(nodes, edges, cons, "ar_ej_b0", tp[:2], "ar_ej_b0x", keep_first=True, nid=P + "T_b")
    T = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
    for k in ("ar_ej_f0", "ar_ej_f0x", "ar_ej_b0", "ar_ej_b0x"):
        e = edges[k]
        if e["to"] in (P + "T_f", P + "T_b"):
            e["to"] = P + "T"
        else:
            e["frm"] = P + "T"
    nodes.pop(P + "T_f", None)
    nodes.pop(P + "T_b", None)
    nodes[P + "T"] = (T[0], T[1], Z0)
    tls.add(P + "T")
    # C: T 에서 언주로와 직각으로 남서쪽(순환 안쪽)으로 → B 와 교차 → A 에 T자
    hf = _unit((edges["ar_ej_f0x"]["shape"][2][0] - T[0], edges["ar_ej_f0x"]["shape"][2][1] - T[1]))
    dC = (hf[1], -hf[0])                                  # 남행 기준 오른쪽 = 순환 안쪽(남서)
    xA = axA.pos(axA.L * C_ON_A)                          # A 의 이 비율 지점에 T자로 붙는다
    dA_in = _unit((T[0] - xA[0], T[1] - xA[1]))
    to_x = _unit((xA[0] - T[0], xA[1] - T[1]))
    dC = _unit((dC[0] + to_x[0], dC[1] + to_x[1]))         # 언주로와 직각과 목표 방향의 중간으로 출발
    axC = _axis(T, dC, xA, dA_in)
    xBC = _cross(axC, axB)
    xAC = xA
    for nm_, p_ in (("AC", xAC), ("BC", xBC)):
        if p_ is None:
            raise RuntimeError(f"연결 도로 교차점 {nm_} 계산 실패")
        nodes[P + nm_] = (p_[0], p_[1], Z0)
        tls.add(P + nm_)

    def lines(ax, s_a, s_b, step=4.0):
        k = max(2, int(abs(s_b - s_a) / step))
        sg = 1 if s_b > s_a else -1
        f = [ax.pos(s_a + (s_b - s_a) * i / k, sg * O) for i in range(k + 1)]
        return f

    # 도로별 구간 나누기 → 엣지
    def build_road(tag, ax, name, stops):
        """stops: [(s, 정방향 노드, 역방향 노드, 끝점 고정(정방향 시작/끝 좌표) 또는 None)]"""
        for i in range(len(stops) - 1):
            sa, fa, ba, fix_a = stops[i]
            sb, fb, bb, fix_b = stops[i + 1]
            fwd = lines(ax, sa, sb)
            bwd = lines(ax, sb, sa)
            if fix_a:
                fwd[0], bwd[-1] = fix_a
            if fix_b:
                fwd[-1], bwd[0] = fix_b
            E(f"{P}{tag}{i}f", fa, fb, fwd, name)
            E(f"{P}{tag}{i}b", bb, ba, bwd, name)

    sAC_A = axA.proj(xAC)
    build_road("A", axA, "양재역~염곡 연결로", [
        (0.0, A0[2], A0[3], (A0[4], A0[5])),
        (sAC_A, P + "AC", P + "AC", None),
        (axA.L, A1[3], A1[2], (A1[5], A1[4])),
    ])
    sBC_B = axB.proj(xBC)
    build_road("B", axB, "논현로~구룡 연결로", [
        (0.0, B0[2], B0[3], (B0[4], B0[5])),
        (sBC_B, P + "BC", P + "BC", None),
        (axB.L, B1[3], B1[2], (B1[5], B1[4])),
    ])
    sBC_C = axC.proj(xBC)
    build_road("C", axC, "언주로 연결로", [
        (0.0, P + "T", P + "T", None),
        (sBC_C, P + "BC", P + "BC", None),
        (axC.L, P + "AC", P + "AC", None),
    ])
    # 팔 끝 ↔ 연결로: 차로 그대로
    for key, j, road, first in (("ys", 1, "A", True), ("yg", 0, "A", False), ("nh", 1, "B", True), ("gr", 0, "B", False)):
        out_e, in_e = f"ar_{key}_a{j}_out", f"ar_{key}_a{j}_in"
        if first:
            cf, cb = f"{P}{road}0f", f"{P}{road}0b"
        else:
            cf, cb = f"{P}{road}1b", f"{P}{road}1f"       # 끝쪽: 역방향이 팔에서 나옴
        for i in range(N):
            new_c.append((out_e, cf, i, i))
            new_c.append((cb, in_e, i, i))

    # 신호 교차점 차로별 회전
    allE = dict(edges)
    allE.update(new_e)
    for node in tls:
        apps = [k for k, e in allE.items() if e["to"] == node]
        exits = [k for k, e in allE.items() if e["frm"] == node]
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
    # 언주로 직진 (T 에서 잘린 조각끼리): 위 회전 규칙의 직진으로 이미 포함됨
    edges.update(new_e)
    cons.extend(new_c)
    return tls
