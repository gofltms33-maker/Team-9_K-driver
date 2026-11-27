"""양재IC 안 경부고속도로 본선 폭 고르게 (울퉁불퉁 없애기).

원래는 램프가 합류할 때마다 차로가 하나 생기고(5 → 6), 다음 출구에서 그 차로가 빠지면서(6 → 5)
150~250 m 마다 본선 바깥 가장자리가 들어갔다 나왔다 했다.
→ 하행·상행 모두 IC 안에서는 6차로로 폭을 일정하게 두고(바깥 차로 = 보조 차로),
   출구는 바깥 차로에서 갈라지고(공유), 진입 램프는 바깥 차로에 지퍼로 합류한다.
"""
import math

WIDEN = ("SB_c", "SB_d", "NB_c", "NB_d")
STRAIGHT = [("SB_b", "SB_c"), ("SB_c", "SB_w"), ("SB_w", "SB_d"), ("SB_d", "SB_e"),
            ("NB_b", "NB_c"), ("NB_c", "NB_w"), ("NB_w", "NB_d")]
# SB_m7252 (루프 L7252 → 하행 본선) 는 지퍼에서 서로 기다리다 루프 쪽이 계속 멈추는 일이 생겨
# 일반 합류(루프가 본선 바깥 차로에 틈을 보고 들어감)로 둔다. 도로 모양은 그대로.
ZIPPER = {"SB_m7085", "NB_m1470"}


def _shift(pts, off):
    out = []
    for i, p in enumerate(pts):
        a, b = pts[max(0, i - 1)], pts[min(len(pts) - 1, i + 1)]
        L = math.dist(a[:2], b[:2]) or 1.0
        hx, hy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        out.append((p[0] + hy * off, p[1] - hx * off) + tuple(p[2:]))
    return out


def apply(edges, cons):
    if not all(k in edges for k in WIDEN):
        return set()
    for k in WIDEN:
        e = edges[k]
        e["shape"] = _shift(e["shape"], e["lw"] / 2)          # 안쪽(중앙분리대) 가장자리 고정, 바깥으로 한 차로
        e["n"] += 1
    pairs = set(STRAIGHT) | {("NB_d", "NB_e")}
    cons[:] = [c for c in cons if (c[0], c[1]) not in pairs]
    for a, b in STRAIGHT:
        for i in range(edges[a]["n"]):
            cons.append((a, b, i, i))
    for i in range(edges["NB_d"]["n"]):                        # 6 → 7 (R2907 2차로가 바깥에 붙음)
        cons.append(("NB_d", "NB_e", i, i + 1))
    cons[:] = [c for c in cons if not (c[0] == "R2907" and c[1] == "NB_e" and c[2] == 1)]
    cons.append(("R2907", "NB_e", 1, 0))
    return set(ZIPPER)
