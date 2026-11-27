"""교통량 늘린 버전 만들기 (test.py --heavy-traffic, train.py --traffic-mix 에서 사용).

road_builder.build() 가 만든 highway.rou.xml 의 배경 교통류 투입 빈도를 키운다.
  - 출발 구간(정문 출구 → 첫 우회전 ar_yj_f0* → 염곡 ar_yg 비보호 좌회전)을 지나는 교통류는 원래대로 둔다
    (교통량을 올리면 이 구간이 막혀 ego 가 출발 직후 갇혔다).
  - 경부고속도로("경부…" 도로)를 지나는 교통류는 highway 배, 나머지는 others 배.
"""
import os
import re

START_PREFIX = ("ar_hq_", "ar_yj_f0", "ar_yg_")
# 대기줄 만들기 (교통량 늘린 버전, 2026-11-26): 기존 교통류를 몇 배로 늘리는 대신, 대기줄 구간만 지나고 바로 빠지는
# 교통류를 새로 넣는다. (배수 방식은 램프를 지나는 교통류가 1개(시간당 100대)뿐이고 그 차들이 남부순환로를 계속 달려
# 남부순환로에만 줄이 생겼고, 공사 구간을 지나는 교통류는 시간당 50대뿐이라 줄이 안 생겼다.)
#  - 서초IC 출구: 경부 상행 → 램프(SC_NB_EB) → 남부순환로 첫 교차로(양재 ar_ys)에서 바로 우회전해 빠짐.
#    램프 차로는 남부순환로 맨 오른쪽 차로로만 이어지고 그 차로가 우회전 차로라, 빨간불마다 램프 쪽으로 줄이 생긴다.
#  - 공사 구간: 언주로 남행(ar_ej_f0) → 작업 구간(1차로) → 다음 도로(ar_yj_b2) 로 빠짐. 신호에서 몰려 온 차가 3→1 차로에서 밀린다.
#  - 램프 차를 양재 교차로에서 우회전만 시키면 남부순환로 오른쪽 차로에만 몰려서 (2026-11-26 사용자 피드백)
#    우회전 / 직진(다음 남현 교차로에서 빠짐) / 좌회전 으로 나눈다. 좌회전 차는 램프에서 나와 왼쪽 차로까지 옮겨 가야 한다.
HOT_FLOWS = {
    "ramp": [dict(frm="ar_gb_nb", via="SC_NB_EB", to="ar_ys_a1_out", share=0.35, tag="r"),   # 양재 우회전
             dict(frm="ar_gb_nb", via="SC_NB_EB", to="ar_nh_a1_out", share=0.35, tag="s"),   # 양재 직진 → 남현에서 빠짐
             dict(frm="ar_gb_nb", via="SC_NB_EB", to="ar_ys_a0_out", share=0.30, tag="l")],  # 양재 좌회전
    "cz": [dict(frm="ar_ej_f0", via="ar_ej_f0x_wz", to="ar_yj_b2", share=1.0, tag="")],
    # 출발 구간 좌회전 대기 (2026-11-27): 정문 출구 -> 우회전 -> 염곡(ar_yg) 비보호 좌회전 줄에 내 차 앞 대기 차를 만든다.
    # 양재대로를 따라오다 염곡에서 좌회전하는 차 60%, 정문에서 나와 같은 길로 가는 차 40%.
    # 양재대로 맨 왼쪽 차로(3번, 좌회전 차로로 그대로 이어짐)에서만 출발시킨다. 정문에서 나와 3개 차로를 건너가는 차까지 넣으면
    # 좌회전 차로가 꽉 찬 뒤 그 차들이 2번 차로에 서서 끼어들 틈을 기다리다 내 차가 추돌했다 (시드 1017 시험 3판 모두, 2026-11-27).
    "left": [dict(frm="ar_yj_f0", via="ar_yj_f0b", to="ar_yg_a0_out", share=1.0, tag="y", lane="3")],
}
# 대기줄 측정용 구간
Q_RAMP = ("SC_NB_EB",)
Q_NAMBU = ("SC_EB_4", "ar_ns_f0")
Q_CZ = ("ar_ej_f0x", "ar_ej_f0x_tp", "ar_ej_f0x_wz")


def scale_rou(sumo_dir, others=2.0, highway=3.0, ramp_vph=None, cz_vph=None, ramp_win=None, cz_win=None, ramp_split=False,
              left_vph=None, left_win=None):
    """ramp_vph / cz_vph: 서초IC 출구 램프·공사 구간에 대기줄을 만들 교통류를 시간당 이만큼 새로 넣는다.
    ramp_win / cz_win: (시작초, 끝초) 시뮬레이션 시각. 이 시간에만 넣는다 (없으면 내내). 공사 구간은 시간당 약 600대를
    넘으면 줄이 끝없이 길어져서, 내 차가 도착하기 전 몇 분만 몰아 넣어 도착할 때 적당한 줄이 있게 한다."""
    import sumolib
    net = sumolib.net.readNet(os.path.join(sumo_dir, "highway.net.xml"))
    rou = os.path.join(sumo_dir, "highway.rou.xml")
    with open(rou, encoding="utf-8") as f:
        lines = f.read().split("\n")
    n_start = n_hw = n_other = n_hot = 0
    out = []
    for ln in lines:
        m = re.search(r'<flow id="(f[cqr]\d+)"[^>]*from="([^"]+)" to="([^"]+)"', ln)
        mr = re.search(r'period="exp\(([\d.]+)\)"', ln)
        if m and mr:
            mv = re.search(r' via="([^"]+)"', ln)
            chain = [m.group(2)] + (mv.group(1).split() if mv else []) + [m.group(3)]
            path = []
            for x, y in zip(chain, chain[1:]):
                p = net.getShortestPath(net.getEdge(x), net.getEdge(y))[0]
                path += list(p or [])
            ids = [e.getID() for e in path]
            if any(i.startswith(START_PREFIX) for i in ids):
                k = 1.0
                n_start += 1
            elif any((e.getName() or "").startswith("경부") for e in path):
                k = highway
                n_hw += 1
            else:
                k = others
                n_other += 1
            if k != 1.0:
                ln = ln.replace(mr.group(0), f'period="exp({float(mr.group(1)) * k:.5f})"')
        out.append(ln)
    extra = []
    for key, vph, win in (("ramp", ramp_vph, ramp_win), ("cz", cz_vph, cz_win), ("left", left_vph, left_win)):
        if vph:
            b0, b1 = (float(win[0]), float(win[1])) if win else (0.0, 100000.0)
            flows = HOT_FLOWS[key]
            if key == "ramp" and not ramp_split:      # 예전 방식: 램프 차 전부 양재 우회전
                flows = [dict(HOT_FLOWS["ramp"][0], share=1.0, tag="")]
            for fl in flows:
                rate = float(vph) * fl["share"] / 3600.0
                vt, dl = "hv", fl.get("lane", "best")
                extra.append(f'    <flow id="fv_{key}{fl["tag"]}" type="{vt}" from="{fl["frm"]}" to="{fl["to"]}" via="{fl["via"]}" '
                             f'begin="{b0:.0f}" end="{b1:.0f}" period="exp({rate:.5f})" departLane="{dl}" departSpeed="max"/>')
    if extra:
        # SUMO 는 경로 파일을 출발 시각 순서로 읽는다. 시작 시각이 뒤섞이면 늦게 적힌 이른 교통류가 통째로 무시된다
        # (출발 좌회전 270 초가 서초IC 530 초·공사 1180 초 뒤에 적혀 한 대도 안 나왔다, 2026-11-27).
        extra.sort(key=lambda ln: float(re.search(r'begin="([\d.]+)"', ln).group(1)))
        i = max(k for k, ln in enumerate(out) if "</routes>" in ln)
        out[i:i] = extra
        n_hot = len(extra)
    with open(rou, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"[교통량 늘린 버전] 출발 구간 지나는 교통류 {n_start}개 그대로, 고속도로 {n_hw}개 x{highway}, 나머지 {n_other}개 x{others}"
          + (f", 대기줄용 교통류 추가 (서초IC 램프 {ramp_vph or 0:.0f}대/h, 공사 {cz_vph or 0:.0f}대/h, 출발 좌회전 {left_vph or 0:.0f}대/h)" if n_hot else ""))
