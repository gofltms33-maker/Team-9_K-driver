"""양재 순환 코스 전체 (ROAD["layout"] == "yangjae_loop_full").

= 어제 만든 양재IC(env/roads/yangjae_ic.py, 손으로 다듬은 클로버형 IC)
+ 그 밖의 순환 코스 주변 도로(env/roads/osm_corridor.py, OSM 실측을 다듬어 생성):
  경부 상행/하행 ~ 서초IC(전 램프) ~ 남부순환로(양재역 사거리·양재지하차도·논현로 교차) ~
  매봉(언주로) ~ 언주로(개포까지 모든 교차로) ~ 개포(개포지하도로) ~ 양재대로(구룡지하차도·논현로 교차) ~ 염곡
두 부분은 경계의 막다른 노드를 방향이 맞는 것끼리 합쳐서 이어붙인다.
"""
import math
import os

from env.roads import yangjae_ic as ic
from env.roads import osm_corridor as oc
from env.roads import seocho_ic as sc
from env.roads import arterials as ar
from env.roads import wonji as wj
from env.roads import hq
from env.roads import construction as cz
from env.roads import connectors as cnx
from env.roads import gyeongbu as gb
from env.roads import taper as tpr
from env.roads import yangjae_fix as yfx
from env.roads import traffic_mix as tm

SC_ZIPPER = set()      # 손으로 만든 구간의 지퍼 합류 노드 (build_network 가 채움)
X_TLS = set()          # 손으로 만든 평면 교차로 (신호)

OSM_FILES = ["loop_north.json", "loop_south.json"]

# ego 순환 경로 경유지: (엣지 id) 또는 (x, y, 진행방향 각도°)
EGO_VIA = ["ar_hq_gate_out", "ar_yg_a0_out", "ar_yg_a0_in",   # 현대차 본사 정문 → 염곡사거리 좌회전 → 유턴 → 양재대로 서행
           "L7252", "L1469", "L1470", "NB_g",  # 양재IC 고리 램프 3개 연속: 경부 하행 합류 → 양재대로 동행(연결로) 합류
                                               # → 경부 상행 합류 (합류·엇갈림을 최대한 많이 겪게, 카카오 경유 2)
           (-930, 1500, 110, "경부"),        # 경부 상행 (서초IC 한참 전)
           (-100, 2378, 10, "남부순환"),     # 남부순환로 동행 (양재역 지나서)
           "ar_ns_f2", "ar_ej_f0",           # 남부순환로 → 매봉 우회전 → 언주로 (안쪽 연결로로 질러가지 않게)
           (1270, 1935, -65, "언주"),        # 언주로 남동행
           (1300, 1080, -145, "양재대로"),   # 양재대로 남서행 (개포 지나서)
           "W_in", "WBf_f", "ar_rb_gym_in"]


def _heading(sh, at_end):
    a, b = (sh[-2], sh[-1]) if at_end else (sh[0], sh[1])
    L = math.dist(a[:2], b[:2]) or 1
    return ((b[0] - a[0]) / L, (b[1] - a[1]) / L)


def build_network(speed_g, osm_dir, warp=None):
    """warp 를 주면 (손으로 깐 남부순환로 제외) 전체를 변형한 뒤 남부순환로를 변형된 좌표 위에 깐다."""
    yn = ic.build_network(speed_g)
    ri = ic._RoadIndex(yn)
    paths = [os.path.join(osm_dir, f) for f in OSM_FILES]
    cn, ce = oc.build(paths, lambda x, y: ri.clearance(x, y) < 30)

    nodes = dict(yn.nodes)
    edges = {k: dict(v, spread="center") for k, v in yn.edges.items()}
    nodes.update(cn)
    edges.update(ce)

    # 경계 이어붙이기: 양재IC 막다른 노드 ↔ 주변도로 막다른 노드 (방향 일치, 150 m 이내)
    def dead_ends(es, ns):
        out_, in_ = {}, {}
        for eid, e in es.items():
            out_.setdefault(e["frm"], []).append(eid)
            in_.setdefault(e["to"], []).append(eid)
        src = [n for n in ns if n in out_ and n not in in_]     # 시작만 있는 노드
        snk = [n for n in ns if n in in_ and n not in out_]     # 끝만 있는 노드
        return src, snk, out_, in_
    y_src, y_snk, y_out, y_in = dead_ends(yn.edges, yn.nodes)
    c_src, c_snk, c_out, c_in = dead_ends(ce, cn)
    used = set()
    pairs = []
    # (a) 주변도로 끝(snk) → 양재IC 시작(src)
    for cnode in c_snk:
        h = _heading(ce[c_in[cnode][0]]["shape"], True)
        best = None
        for ynode in y_src:
            if ynode in used:
                continue
            d = math.dist(cn[cnode][:2], yn.nodes[ynode][:2])
            hy = _heading(yn.edges[y_out[ynode][0]]["shape"], False)
            if d < 150 and h[0] * hy[0] + h[1] * hy[1] > 0.6 and (best is None or d < best[0]):
                best = (d, ynode)
        if best:
            used.add(best[1])
            pairs.append((cnode, best[1]))
    # (b) 양재IC 끝(snk) → 주변도로 시작(src)
    for cnode in c_src:
        h = _heading(ce[c_out[cnode][0]]["shape"], False)
        best = None
        for ynode in y_snk:
            if ynode in used:
                continue
            d = math.dist(cn[cnode][:2], yn.nodes[ynode][:2])
            hy = _heading(yn.edges[y_in[ynode][0]]["shape"], True)
            if d < 150 and h[0] * hy[0] + h[1] * hy[1] > 0.6 and (best is None or d < best[0]):
                best = (d, ynode)
        if best:
            used.add(best[1])
            pairs.append((cnode, best[1]))
    for cnode, ynode in pairs:
        yp = yn.nodes[ynode]
        for eid, e in edges.items():
            if e["to"] == cnode:
                e["to"] = ynode
                e["shape"] = _blend_end(e["shape"], yp, at_end=True)
            if e["frm"] == cnode:
                e["frm"] = ynode
                e["shape"] = _blend_end(e["shape"], yp, at_end=False)
        nodes.pop(cnode, None)
    cons = list(yn.cons)
    SC_ZIPPER.clear()
    SC_ZIPPER.update(yfx.apply(edges, cons))            # 양재IC 경부 본선 폭 일정하게
    SC_ZIPPER.update(sc.rebuild(nodes, edges, cons, speed_g))
    X_TLS.clear()
    if warp is not None:
        warp_all(nodes, edges, warp)
    X_TLS.update(ar.rebuild(nodes, edges, cons, warp))
    t_, z_ = wj.rebuild(nodes, edges, cons, warp)
    X_TLS.update(t_)
    SC_ZIPPER.update(z_)
    t_, _ = hq.rebuild(nodes, edges, cons, warp)
    X_TLS.update(t_)
    X_TLS.update(cnx.rebuild(nodes, edges, cons, warp))
    gb.rebuild(nodes, edges, cons, warp)
    SC_ZIPPER.update(cz.rebuild(nodes, edges, cons, warp))
    used_n = {e["frm"] for e in edges.values()} | {e["to"] for e in edges.values()}
    for k in [k for k in nodes if k not in used_n]:
        del nodes[k]
    return nodes, edges, cons, pairs


def _blend_end(shape, p, at_end):
    """끝점을 p로 옮기면서 마지막 60 m 를 부드럽게 휘게."""
    sh = list(shape) if at_end else list(shape[::-1])
    L = 0.0
    cum = [0.0]
    for a, b in zip(sh[::-1], sh[::-1][1:]):
        L += math.dist(a[:2], b[:2])
        cum.append(L)
    cum = cum[::-1]                      # 각 점에서 끝까지 거리
    dx, dy, dz = p[0] - sh[-1][0], p[1] - sh[-1][1], p[2] - sh[-1][2]
    out = []
    for q, c in zip(sh, cum):
        w = 1 - ic._smooth(c / 60.0)
        out.append((q[0] + dx * w, q[1] + dy * w, q[2] + dz * w))
    return out if at_end else out[::-1]


def _nearest_edge(edges, x, y, ang, name=""):
    hx, hy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    best = None
    for eid, e in edges.items():
        if name and name not in (e.get("name") or ""):
            continue
        sh = e["shape"]
        for a, b in zip(sh, sh[1:]):
            vx, vy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(vx, vy) or 1e-9
            if (vx * hx + vy * hy) / L < 0.7:
                continue
            d = oc._seg_d((x, y), a[:2], b[:2])
            if best is None or d < best[0]:
                best = (d, eid)
    return best[1]


def ego_route(edges, warp=None):
    import heapq
    out = {}
    for eid, e in edges.items():
        out.setdefault(e["frm"], []).append(eid)
    length = {eid: sum(math.dist(a[:2], b[:2]) for a, b in zip(e["shape"], e["shape"][1:]))
              for eid, e in edges.items()}
    w = warp or (lambda x, y: (x, y))
    via = [v if isinstance(v, str) else _nearest_edge(edges, *w(v[0], v[1]), *v[2:]) for v in EGO_VIA]

    def sp(a, b):
        dist, prev = {a: 0.0}, {}
        pq = [(0.0, a)]
        while pq:
            d, e = heapq.heappop(pq)
            if e == b:
                break
            if d > dist[e]:
                continue
            for n in out.get(edges[e]["to"], []):
                if edges[n]["to"] == edges[e]["frm"] and edges[n]["frm"] == edges[e]["to"] and \
                        edges[n].get("spread") == "right":
                    continue                        # U턴 금지
                nd = d + length[n]
                if nd < dist.get(n, 1e18):
                    dist[n], prev[n] = nd, e
                    heapq.heappush(pq, (nd, n))
        if b not in dist:
            raise RuntimeError(f"경로 없음: {a} → {b}")
        path = [b]
        while path[-1] != a:
            path.append(prev[path[-1]])
        return path[::-1]
    route = [via[0]]
    for a, b in zip(via, via[1:]):
        route += sp(a, b)[1:]
    return route, via


MAIN_ROADS = {"경부고속도로", "남부순환로", "양재대로", "언주로", "영동3교"}
ANCHOR_MARGIN = 30.0


def prune_bands(nodes, edges, keep=(), warp=None):
    """고정 영역(교차로·IC) 사이 직선 구간에서는 순환 큰길만 남기고 골목·옆길·평행 도로를 지운다.
    고정 영역에 한 점이라도 걸친 도로는 그대로 둔다."""
    def inside(x, y):
        return any(math.hypot(x - ax, y - ay) < r + ANCHOR_MARGIN for ax, ay, r in anc)
    w = warp or (lambda x, y: (x, y))
    anc = [(*w(ax, ay), r) for _, ax, ay, r in ANCHORS]
    drop = []
    for eid, e in edges.items():
        if eid[0].isupper() or eid in keep or eid.startswith(ar.P):
            continue
        if (e.get("name") or "").split(" ")[0] in MAIN_ROADS:
            continue
        if not any(inside(p[0], p[1]) for p in e["shape"]):
            drop.append(eid)
    for eid in drop:
        del edges[eid]
    # 고정 영역 밖으로 뻗은 옆길 꼬리: 막다른 끝이 영역 밖이면 영역 경계에서 잘라낸다
    def first_out(sh):
        return next((i for i, p in enumerate(sh) if not inside(*p[:2])), None)
    for _ in range(40):
        adj = {}
        for eid, e in edges.items():
            adj.setdefault(e["frm"], set()).add(eid)
            adj.setdefault(e["to"], set()).add(eid)
        changed = False
        for eid in list(edges):
            e = edges.get(eid)
            if e is None or eid[0].isupper() or eid in keep or eid.startswith(ar.P):
                continue
            if (e.get("name") or "").split(" ")[0] in MAIN_ROADS:
                continue
            for end in ("to", "frm"):
                d = e[end]
                group = adj.get(d, set())
                nb = {edges[k]["frm"] if edges[k]["to"] == d else edges[k]["to"] for k in group}
                if len(nb) != 1 or d not in nodes or inside(*nodes[d][:2]):
                    continue                       # 막다른 끝이 아니거나 이미 영역 안
                if any(k[0].isupper() or k in keep or k.startswith(ar.P) for k in group):
                    continue
                cuts = {}
                for k2 in group:                   # 양방향 짝을 함께 처리 (막다른 점 → 반대쪽 순서로)
                    e2 = edges[k2]
                    sh = e2["shape"][::-1] if e2["to"] == d else e2["shape"]     # d 에서 출발
                    rev = sh[::-1]                                                # 반대쪽 끝에서 출발
                    k = first_out(rev)
                    cuts[k2] = None if not k else rev[:k]
                if any(c is None or len(c) < 2 for c in cuts.values()):
                    for k2 in group:               # 통째로 영역 밖 → 삭제
                        edges.pop(k2, None)
                        drop.append(k2)
                else:
                    for k2, c in cuts.items():     # c: 반대쪽 끝 → 경계 직전
                        e2 = edges[k2]
                        e2["shape"] = c if e2["to"] == d else c[::-1]
                    nodes[d] = tuple(next(iter(cuts.values()))[-1])
                changed = True
                break
        if not changed:
            break
    # 짧게 남은 막다른 조각(25 m 미만, 큰길 아님) 정리
    for _ in range(10):
        deg = {}
        for e in edges.values():
            deg[e["frm"]] = deg.get(e["frm"], 0) + 1
            deg[e["to"]] = deg.get(e["to"], 0) + 1
        short = [k for k, e in edges.items()
                 if not k[0].isupper() and k not in keep and not k.startswith(ar.P)
                 and sum(math.dist(a[:2], b[:2]) for a, b in zip(e["shape"], e["shape"][1:])) < 25
                 and (deg[e["frm"]] == 1 or deg[e["to"]] == 1)]
        if not short:
            break
        for k in short:
            del edges[k]
            drop.append(k)
    # 손으로 깐 도로망과 이어지지 않은 섬(남은 실측 조각) 지우기
    par = {}

    def find(a):
        while par.setdefault(a, a) != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for e in edges.values():
        par[find(e["frm"])] = find(e["to"])
    good = {find(e["frm"]) for k, e in edges.items() if k[0].isupper() or k.startswith(ar.P) or k in keep}
    for k in [k for k, e in edges.items() if find(e["frm"]) not in good]:
        del edges[k]
        drop.append(k)
    return drop


def add_dead_end_turns(nodes, edges, cons, max_gap=40.0):
    """지도 경계에서 잘린 막다른 길 끝에 유턴을 허용한다.

    사거리 옆 팔·지도 가장자리 도로는 실제로는 그 너머 도로로 이어지지만, 지도를 잘라 만들면서
    막다른 길이 됐다. 그대로 두면 좌회전 등을 놓친 학습차가 그쪽으로 들어갔을 때 돌아 나올 길이
    없어 갇힌다. 끝에서 반대 방향 도로로 돌아 나오게(= '한 블록 돌아 나오기'를 대신) 한다.
    대상: 막다른 곳에서 끝나는 도로 x 와, 그 근처(max_gap m) 막다른 곳에서 시작해 x 의 출발점으로
    돌아가는 도로 y 의 쌍. 일반 차량의 출발·도착 도로로는 그대로 쓰이도록 표시를 남긴다.
    """
    outs, ins = {}, {}
    for k, e in edges.items():
        outs.setdefault(e["frm"], []).append(k)
        ins.setdefault(e["to"], []).append(k)
    used, n = set(), 0
    for x, ex in list(edges.items()):
        if x in used:
            continue
        a = ex["to"]
        for y, ey in edges.items():
            if y == x or y in used or ey["to"] != ex["frm"]:
                continue
            b = ey["frm"]
            if set(outs.get(a, [])) - {y} or set(ins.get(b, [])) - {x}:
                continue                          # 둘 중 하나라도 다른 길로 이어지면 막다른 길이 아님
            if a != b:
                if math.dist(nodes[a][:2], nodes[b][:2]) > max_gap:
                    continue
                m = ((nodes[a][0] + nodes[b][0]) / 2, (nodes[a][1] + nodes[b][1]) / 2, nodes[a][2])
                nodes[a] = m
                ey["frm"] = a
                nodes.pop(b, None)
            ex["turn_end"], ey["turn_start"] = True, True
            # 막다른 길이라 어느 차로에서든 돌아 나올 수 있게 한다 (차로끼리 겹치지 않게 같은 번호로 연결)
            for i in range(ex["n"]):
                cons.append((x, y, i, min(i, ey["n"] - 1)))
            used |= {x, y}
            n += 1
            break
    return n


def write_all(out_dir, road, ego, traffic, vtypes_block, flow_type, gui_delay):
    osm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "osm")
    shrink = float(road.get("loop_shrink", SHRINK))
    warp = make_warp(shrink) if shrink < 0.999 else (lambda x, y: (x, y))
    nodes, edges, cons, pairs = build_network(float(road["speed_limit"]), osm_dir, warp)
    route, via = ego_route(edges, warp)
    prune_bands(nodes, edges, keep=set(route), warp=warp)
    add_dead_end_turns(nodes, edges, cons)         # 학습차 경로(route)는 이미 정해진 뒤라 바뀌지 않는다

    nod, edg, con = [os.path.join(out_dir, f"highway.{k}.xml") for k in ("nod", "edg", "con")]
    deg_in, deg_out = {}, {}
    for e in edges.values():
        deg_out[e["frm"]] = deg_out.get(e["frm"], 0) + 1
        deg_in[e["to"]] = deg_in.get(e["to"], 0) + 1
    # 합류 지점(들어오는 차선 합 > 나가는 차선): 램프가 본선 틈만 기다리다 갇히지 않도록
    # 지퍼 합류(번갈아 끼워주기)로 지정. 주변도로(실측) 노드에만 적용.
    ins_, outs_ = {}, {}
    for eid, e in edges.items():
        outs_.setdefault(e["frm"], []).append(eid)
        ins_.setdefault(e["to"], []).append(eid)
    zipper = set()
    for nid in nodes:
        i_, o_ = ins_.get(nid, []), outs_.get(nid, [])
        if len(o_) == 1 and len(i_) >= 2 and sum(edges[k]["n"] for k in i_) > edges[o_[0]]["n"] \
                and not any(k[0].isupper() for k in i_ + o_):
            zipper.add(nid)
    with open(nod, "w", encoding="utf-8") as f:
        f.write("<nodes>\n")
        for nid, (x, y, z) in nodes.items():
            if nid not in deg_in and nid not in deg_out:
                continue
            tp = ' type="zipper"' if nid in zipper or nid in SC_ZIPPER else ""
            if nid in X_TLS:
                tp = ' type="traffic_light"'

            f.write(f'    <node id="{nid}" x="{x:.2f}" y="{y:.2f}" z="{z:.2f}"{tp}/>\n')
        f.write("</nodes>\n")
    with open(edg, "w", encoding="utf-8") as f:
        f.write("<edges>\n")
        for eid, e in edges.items():
            shp = " ".join(f"{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}" for p in e["shape"])
            nm = (e.get("name") or "").replace('"', "")
            lane_w = e.get("lane_w") or {}
            f.write(f'    <edge id="{eid}" from="{e["frm"]}" to="{e["to"]}" numLanes="{e["n"]}" '
                    f'speed="{e["speed"]:.2f}" priority="{e["prio"]}" width="{e["lw"]}" '
                    f'spreadType="{e.get("spread", "center")}"' + (f' name="{nm}"' if nm else "") + ' '
                    f'disallow="pedestrian bicycle" shape="{shp}"' + ("/>\n" if not lane_w else ">\n"))
            if lane_w:
                for li, wv in sorted(lane_w.items()):
                    f.write(f'        <lane index="{li}" width="{wv:.2f}"/>\n')
                f.write("    </edge>\n")
        for rn, re_ in wj.ROUNDABOUTS:              # 원지동 회전교차로
            if all(x in edges for x in re_):
                f.write(f'    <roundabout nodes="{" ".join(rn)}" edges="{" ".join(re_)}"/>\n')
        f.write("</edges>\n")
    with open(con, "w", encoding="utf-8") as f:
        f.write("<connections>\n")
        for a, b, fl, tl in cons:
            f.write(f'    <connection from="{a}" to="{b}" fromLane="{fl}" toLane="{tl}"/>\n')
        f.write("</connections>\n")

    # 경로/교통류: 양재IC 기본 교통 + 주변도로 막다른 노드 사이 관통 교통(from/to, SUMO가 경로 탐색)
    # 양재IC 자체 교통류(경로 끝에서 차가 사라짐)는 끄고, 전체 도로망 막다른 길 사이 OD 교통으로 대체
    _tr = dict(traffic, ic_flows={k: 0 for k in ic.DEFAULT_FLOWS})
    ic.write_routes(os.path.join(out_dir, "highway.rou.xml"), tm.VTYPES, "hv", _tr, ego)
    with open(os.path.join(out_dir, "highway.rou.xml"), encoding="utf-8") as f:
        txt = f.read().rstrip()
    assert txt.endswith("</routes>")
    txt = txt[: -len("</routes>")]
    txt = txt.replace(f'<route id="r0" edges="{ic.ROUTES[ic.EGO_ROUTE]}"/>',
                      f'<route id="r0" edges="{" ".join(route)}"/>')
    scale = float(traffic.get("vehs_per_hour", 800)) / 800.0
    extra = tm.flows(edges, scale)          # 일반 50 % + 특수 5종 각 10 % (traffic_mix.py)
    with open(os.path.join(out_dir, "highway.rou.xml"), "w", encoding="utf-8") as f:
        f.write(txt + "\n".join(extra) + "\n</routes>\n")

    ic.write_gui(os.path.join(out_dir, "highway.gui.xml"), _NetView(nodes, edges), gui_delay, 2600.0,
                 center=warp(300.0, 1150.0))
    # 표시물(중앙분리대·지하차도 덮개·휴게소 등)은 원래 좌표로 만든 뒤 같은 변형을 적용
    # (양재IC 의 지하차도 덮개 등은 원래 좌표 기준으로 계산되므로) 원래 좌표로 만든 뒤 같은 변형을 적용하고,
    # 변형 뒤에 깐 남부순환로 구간은 최종 좌표로 따로 만들어 덧붙인다
    _orig = build_network(float(road["speed_limit"]), osm_dir)
    ns_ids = {k for k in _orig[1] if k.startswith(ar.P)}
    for k in ns_ids:
        del _orig[1][k]
    prune_bands(_orig[0], _orig[1], keep=set(route))
    scn = os.path.join(out_dir, "highway.scenery.xml")
    ic.write_scenery(scn, _NetView(_orig[0], _orig[1]), trees=False)
    warp_scenery_file(scn, warp)
    extra_polys = []
    for k, e in edges.items():                 # 남부순환로 구간: 연석 띠 (교차로 면 z=4 보다 아래)
        if not k.startswith(ar.P):
            continue
        lw_ = e.get("lane_w") or {}
        hw = sum(lw_.get(i, e["lw"]) for i in range(e["n"])) / 2 + 0.7
        xy = [p_[:2] for p_ in e["shape"]]
        zmin = min(p_[2] for p_ in e["shape"])
        if zmin < ic.Z_GROUND - 0.1:          # 지하차도: 경사 차로를 덮지 않게 훨씬 아래에 어두운 옹벽색
            extra_polys.append(f'    <poly id="ns_c_{k}" color="58,58,56" fill="1" layer="{zmin - 3.0:.2f}" '
                               f'shape="{ic._pstr(ic._band(xy, -hw - 0.2, hw + 0.2))}"/>')
            continue
        extra_polys.append(f'    <poly id="ns_c_{k}" color="176,176,168" fill="1" layer="3.60" '
                           f'shape="{ic._pstr(ic._band(xy, -hw, hw))}"/>')
    _tw = tpr.patches(nodes, edges, skip_nodes=X_TLS) + tpr.wedges(nodes, edges)
    for j, (pts_, col, lay) in enumerate(_tw):  # 이음매 메움 + 차로 수 바뀌는 곳 부드러운 사선
        extra_polys.append(f'    <poly id="ar_tw_{j}" color="{col}" fill="1" layer="{lay:.2f}" '
                           f'shape="{ic._pstr(pts_)}"/>')
    for j, (pts_, col, lay, fill) in enumerate(cz.POLYS):          # 공사 구간
        extra_polys.append(f'    <poly id="ar_cz_{j}" color="{col}" fill="{1 if fill else 0}" layer="{lay:.2f}"'
                           + ('' if fill else ' lineWidth="0.4"') + f' shape="{ic._pstr(pts_)}"/>')
    for j, (pts_, col, lay) in enumerate(hq.COVER_POLYS):
        extra_polys.append(f'    <poly id="ar_cov_{j}" color="{col}" fill="1" layer="{lay:.2f}" '
                           f'shape="{ic._pstr(pts_)}"/>')
    for j, (pts_, col, lay) in enumerate(wj.BUILDING_POLYS + hq.BUILDING_POLYS):     # 원지동: 체육관 등 건물
        extra_polys.append(f'    <poly id="ar_bld_{j}" color="{col}" fill="1" layer="{lay:.2f}" '
                           f'shape="{ic._pstr(pts_ + [pts_[0]])}"/>')
        extra_polys.append(f'    <poly id="ar_bld_{j}_e" color="90,92,100" fill="0" lineWidth="0.8" '
                           f'layer="{lay + 0.1:.2f}" shape="{ic._pstr(pts_ + [pts_[0]])}"/>')
    with open(scn, encoding="utf-8") as f:
        body = f.read().rstrip()
    i = body.rfind("</")
    with open(scn, "w", encoding="utf-8") as f:
        f.write(body[:i] + "\n".join(extra_polys) + "\n" + body[i:] + "\n")
    return nod, edg, con, route, pairs


class _NetView:
    """yangjae_ic 의 write_gui / write_scenery 가 쓰는 최소 인터페이스."""

    def __init__(self, nodes, edges):
        self.nodes, self.edges = nodes, edges


def netconvert_args(nod, edg, con, out_net):
    return ["--node-files", nod, "--edge-files", edg, "--connection-files", con, "-o", out_net,
            "--offset.disable-normalization", "true", "--no-turnarounds", "true",
            "--junctions.corner-detail", "8", "--geometry.max-grade.fix", "false",
            "--tls.guess", "true", "--tls.guess.threshold", "40", "--tls.default-type", "static",
            "--roundabouts.guess", "true", "--no-warnings", "true"]


# ══════════════════════════════════════════════════════════════
# 직선 구간 줄이기 (교차로/IC 모양은 그대로, 그 사이 도로만 짧게)
# ══════════════════════════════════════════════════════════════
# (이름, 중심 x, y, 고정 반경 m) — 반경 안은 모양 그대로 '평행이동'만 한다.
ANCHORS = [("양재IC", 40, 0, 420), ("서초IC", -1180, 2200, 330), ("양재역", -470, 2310, 190),
           ("논현로", 172, 2437, 160), ("매봉", 918, 2732, 190), ("개포", 1716, 1355, 180),
           ("양재대로x논현로", 1028, 880, 120)]
# 양재IC → 구룡 → 개포 는 '앞 고정영역을 기준으로 변 방향으로만' 줄인다 (옆으로 밀리지 않아 꺾임 없음).
# 값 = 앞 영역 대비 위치 차이에 곱할 비율 (-0.24 → 그 변을 원래의 약 60 % 로).
# 지하차도 → 현대차 → 염곡사거리 → 구룡이 차례로 들어갈 길이를 확보한다.
ANCHOR_CHAIN = {"양재대로x논현로": ("양재IC", -0.237), "개포": ("양재대로x논현로", -0.25)}
SHRINK = 0.6          # 고정 영역 중심들을 순환 중심 쪽으로 당기는 비율 (1 = 원래 크기)


def make_warp(shrink=SHRINK):
    """각 변(이웃한 고정영역 사이)을 따라 '그 방향으로만' 균일 압축 → 직선은 직선 그대로.

    고정영역 반경 안: 평행이동만. 변 위의 점: 변 방향 위치에 따라 양 끝 이동량을 선형 보간.
    """
    cx = sum(a[1] for a in ANCHORS) / len(ANCHORS)
    cy = sum(a[2] for a in ANCHORS) / len(ANCHORS)
    disp = {nm: ((cx + (x - cx) * shrink) - x, (cy + (y - cy) * shrink) - y) for nm, x, y, r in ANCHORS}
    pos = {nm: (x, y) for nm, x, y, r in ANCHORS}
    scale = (1.0 - shrink) / (1.0 - SHRINK) if SHRINK < 1 else 0.0
    for nm, (par, k) in ANCHOR_CHAIN.items():          # 순서대로 (부모가 먼저)
        k *= scale
        disp[nm] = (disp[par][0] + k * (pos[nm][0] - pos[par][0]), disp[par][1] + k * (pos[nm][1] - pos[par][1]))
    A = [(x, y, r, disp[nm][0], disp[nm][1]) for nm, x, y, r in ANCHORS]
    n = len(A)

    def warp(px, py):
        for x, y, r, dx, dy in A:
            if math.hypot(px - x, py - y) <= r:
                return px + dx, py + dy
        best = None
        for i in range(n):
            x1, y1, r1, dx1, dy1 = A[i]
            x2, y2, r2, dx2, dy2 = A[(i + 1) % n]
            L = math.hypot(x2 - x1, y2 - y1)
            ux, uy = (x2 - x1) / L, (y2 - y1) / L
            sp = (px - x1) * ux + (py - y1) * uy
            d = abs(-(px - x1) * uy + (py - y1) * ux)
            if 0 <= sp <= L and d < 280 and (best is None or d < best[0]):
                t = max(0.0, min(1.0, (sp - r1) / max(1.0, L - r1 - r2)))
                best = (d, dx1 + (dx2 - dx1) * t, dy1 + (dy2 - dy1) * t)
        if best is None:          # 순환 바깥쪽 끝 (경부 남쪽, 체육관 등): 가장 가까운 고정영역을 따라감
            x, y, r, dx, dy = min(A, key=lambda a: math.hypot(px - a[0], py - a[1]))
            return px + dx, py + dy
        return px + best[1], py + best[2]
    return warp


def warp_all(nodes, edges, warp):
    for k, (x, y, z) in list(nodes.items()):
        nodes[k] = (*warp(x, y), z)
    for e in edges.values():
        sh = [(*warp(p[0], p[1]), p[2]) for p in e["shape"]]
        # 변형 후 다시 한 번 살짝 다듬기 (끝점·방향 고정)
        if len(sh) > 4:
            xy = oc._gauss_smooth([p[:2] for p in sh], 4.0, step=1.5, anchor=8.0)
            pl = oc._Poly(xy)
            zs = [p[2] for p in sh]
            L0 = sum(math.dist(a[:2], b[:2]) for a, b in zip(sh, sh[1:])) or 1
            cum0 = [0.0]
            for a, b in zip(sh, sh[1:]):
                cum0.append(cum0[-1] + math.dist(a[:2], b[:2]))

            def zf(f):
                t = f * L0
                for i in range(len(cum0) - 1):
                    if cum0[i + 1] >= t:
                        g = (t - cum0[i]) / ((cum0[i + 1] - cum0[i]) or 1)
                        return zs[i] + (zs[i + 1] - zs[i]) * g
                return zs[-1]
            sh = [(q[0], q[1], zf(c / (pl.length or 1))) for q, c in zip(pl.p, pl.cum)]
        e["shape"] = sh


def warp_scenery_file(path, warp):
    import re
    with open(path, encoding="utf-8") as f:
        txt = f.read()

    def rep(m):
        pts = []
        for q in m.group(1).split():
            x, y = q.split(",")[:2]
            wx, wy = warp(float(x), float(y))
            pts.append(f"{wx:.2f},{wy:.2f}")
        return f'shape="{" ".join(pts)}"'
    with open(path, "w", encoding="utf-8") as f:
        f.write(re.sub(r'shape="([^"]+)"', rep, txt))
