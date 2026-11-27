"""경부고속도로 양재IC ~ 서초IC 구간 (지도 조각 5개씩 → 손으로 깐 한 줄).

  - 하행/상행 각 5차로, 차로 폭 3.5 m, 중앙분리대(콘크리트 방호벽) 폭은 양 끝 간격에 맞춰 서서히.
  - 축은 두 지도 차도의 가운데 선을 크게 다듬은 완만한 곡선, 양 끝은 서초IC·양재IC 방향과 접선으로 이어진다.
  - 높이는 지면 (서초IC 남쪽 다리 접속부도 지면으로 내려오게 서초IC 쪽을 고쳤다).
"""
import math

from env.roads import yangjae_ic as ic
from env.roads import seocho_ic as sc
from env.roads.arterials import Axis, _unit

SB_START, NB_END = "n732030910", "n11091065444"      # 서초IC 남쪽 경계 노드
N, LW = 5, 3.5
TAN = 130.0


def _chain(edges, start, fwd):
    out, cur = [], start
    for _ in range(40):
        nx = [k for k, v in edges.items() if (v["frm"] == cur if fwd else v["to"] == cur) and not k[0].isupper()
              and (v.get("name") or "").startswith("경부")]
        if not nx:
            break
        k = nx[0]
        out.append(k)
        cur = edges[k]["to"] if fwd else edges[k]["frm"]
    return out, cur


def rebuild(nodes, edges, cons, warp=None):
    sb, sb_end = _chain(edges, SB_START, True)          # 서초 → 양재 (하행)
    nb, nb_start = _chain(edges, NB_END, False)          # 양재 → 서초 (상행), 역순으로 모음
    if not sb or not nb:
        return
    sb_pts = [p for k in sb for p in edges[k]["shape"]]
    nb_pts = [p for k in nb[::-1] for p in edges[k]["shape"]]
    z0 = nodes[SB_START][2]
    p_s0, p_n0 = nodes[SB_START][:2], nodes[NB_END][:2]      # 서초 쪽 끝
    p_s1, p_n1 = nodes[sb_end][:2], nodes[nb_start][:2]      # 양재 쪽 끝
    M0 = ((p_s0[0] + p_n0[0]) / 2, (p_s0[1] + p_n0[1]) / 2)
    M1 = ((p_s1[0] + p_n1[0]) / 2, (p_s1[1] + p_n1[1]) / 2)
    # 양 끝 방향: 서초 쪽은 하행 본선이 들어오는 방향, 양재 쪽은 하행 SB_a 가 나가는 방향
    a = edges["SC_SB_3"]["shape"]
    h0 = _unit((a[-1][0] - a[-4][0], a[-1][1] - a[-4][1]))
    b = edges["SB_a"]["shape"]
    h1 = _unit((b[3][0] - b[0][0], b[3][1] - b[0][1]))
    # 가운데 선 표본: 하행 선의 각 점에서 상행 선까지의 가장 가까운 점과의 중점
    mids = []
    for i in range(0, len(sb_pts), 6):
        p = sb_pts[i][:2]
        q = min(nb_pts, key=lambda r: math.dist(r[:2], p))[:2]
        m = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        if math.dist(m, M0) > 150 and math.dist(m, M1) > 150:
            mids.append(m)
    mids.sort(key=lambda m: math.dist(m, M0))
    pts = [M0, (M0[0] + h0[0] * TAN, M0[1] + h0[1] * TAN)] + [(M1[0] - h1[0] * TAN, M1[1] - h1[1] * TAN), M1]
    ax = Axis(ic._gauss_smooth(ic._resample(pts, 2.0), 60.0, step=2.0, anchor=TAN))
    # 중앙분리대: 양 끝 실제 간격에 맞춤
    sep0, sep1 = math.dist(p_s0, p_n0), math.dist(p_s1, p_n1)
    med0, med1 = max(0.8, sep0 - N * LW), max(0.8, sep1 - N * LW)

    def off(s):
        f = ic._smooth(min(max(s / ax.L, 0.0), 1.0))
        return (med0 + (med1 - med0) * f) / 2 + N * LW / 2

    def line(s_a, s_b, side, step=6.0):
        k = max(2, int(abs(s_b - s_a) / step))
        return [(*ax.pos(s_a + (s_b - s_a) * i / k, side * off(s_a + (s_b - s_a) * i / k)), z0)
                for i in range(k + 1)]
    sb_shape = line(0.0, ax.L, +1)
    nb_shape = line(ax.L, 0.0, -1)
    sb_shape[0] = (*p_s0, z0)
    sb_shape[-1] = (*p_s1, nodes[sb_end][2])
    nb_shape[0] = (*p_n1, nodes[nb_start][2])
    nb_shape[-1] = (*p_n0, z0)
    proto = dict(edges[sb[0]])
    for k in sb + nb:
        del edges[k]
    edges["ar_gb_sb"] = dict(proto, frm=SB_START, to=sb_end, n=N, lw=LW, shape=sb_shape, spread="center",
                             name="경부고속도로")
    edges["ar_gb_nb"] = dict(proto, frm=nb_start, to=NB_END, n=N, lw=LW, shape=nb_shape, spread="center",
                             name="경부고속도로")
    ren = {k: "ar_gb_sb" for k in sb}
    ren.update({k: "ar_gb_nb" for k in nb})
    for i, (x, y, fl, tl) in enumerate(cons):
        cons[i] = (ren.get(x, x), ren.get(y, y), fl, tl)
