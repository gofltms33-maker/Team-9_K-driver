"""배경차 구성: 일반 50 % (Krauss 25 % + IDM 25 %) + 특수 5종 각 10 % (발표 슬라이드 '차량 Interaction 설정').

원칙: 어떤 상황도 '강제로 일으키지 않는다'.
  - 특정 차를 특정 시각·위치에 놓거나, 특정 차가 ego 를 막도록 짜지 않는다.
  - 성향이 다른 차들이 무작위 시각(지수분포 간격)·무작위 차로로 들어와 각자 갈 길을 간다.
  - 그 결과 1차로 저속차, 끼워 주지 않는 차, 늦게 끼어드는 차, 회전교차로 혼잡, 대기줄이
    '그럴 법한 확률로' 생긴다. 일어나는지는 그때그때 다르다.

  ① hv_slow     1차로 정속(저속) 주행: 희망속도가 낮고, 속도 때문에 차로를 바꾸지 않으며 우측 유지 성향이 없음
                → 어느 차로(1차로 포함)에 들어가든 그 자리에서 천천히 간다.
  ② hv_nonyield 양보 안 함: 협조 차선변경 0 (공사 구간·합류부에서 틈을 안 열어 줌), 차간거리 짧음
  ③ hv_cutin    늦게·과감하게 끼어듦: 경로상 필요한 차로 변경을 늦게 하고(전략 0.3), 작은 틈도 받아들임
  ④ 회전교차로 쪽 교통 (경로 기반): 원지동 회전교차로를 지나는 출발·도착 조합 → 그 일대 교통량이 늘어난다
  ⑤ 대기줄 차량 (경로 기반, hv_queue): 염곡사거리 좌회전 / 서초IC 출구를 지나는 차.
     줄에 서면 끼워 주지 않는 성향(협조 0, 차간 1 m) → 늦게 차로를 바꾸려는 차가 못 끼어드는 일이 생길 수 있다.
"""
import math
import random

BASE_VPH = 2400.0           # vehs_per_hour = 800 기준 배경(①②③+일반) 총량
SPECIAL_SHARE = 0.125       # 배경 대비 ④, ⑤ 각각 (→ 전체의 10 %)

VTYPES = """    <vType id="hv_normal" vClass="passenger" carFollowModel="Krauss" accel="2.6" decel="4.5" sigma="0.5"
           tau="1.0" minGap="2.5" speedFactor="normc(1.0,0.1,0.8,1.2)" color="210,210,210"
           lcStrategic="1.0" lcCooperative="0.8" lcSpeedGain="1.0" lcKeepRight="1.0"/>
    <vType id="hv_idm" vClass="passenger" carFollowModel="IDM" accel="2.6" decel="4.5"
           tau="1.0" minGap="2.5" speedFactor="normc(1.0,0.1,0.8,1.2)" color="160,160,190"
           lcStrategic="1.0" lcCooperative="0.8" lcSpeedGain="1.0" lcKeepRight="1.0"/>
    <vType id="hv_slow" vClass="passenger" carFollowModel="Krauss" accel="2.0" decel="4.5" sigma="0.3"
           tau="1.2" minGap="2.5" speedFactor="normc(0.7,0.05,0.6,0.8)" color="70,130,230"
           lcStrategic="1.0" lcCooperative="0.8" lcSpeedGain="0.0" lcKeepRight="0.0"/>
    <vType id="hv_nonyield" vClass="passenger" carFollowModel="Krauss" accel="3.0" decel="4.5" sigma="0.4"
           tau="1.0" minGap="1.5" speedFactor="normc(1.1,0.1,0.9,1.3)" color="240,140,30"
           lcStrategic="1.0" lcCooperative="0.0" lcCooperativeSpeed="0.0" lcSpeedGain="1.0" lcKeepRight="0.5"/>
    <vType id="hv_cutin" vClass="passenger" carFollowModel="Krauss" accel="3.0" decel="4.5" sigma="0.6"
           tau="1.0" minGap="2.0" speedFactor="normc(1.15,0.1,0.9,1.35)" color="220,60,200"
           lcStrategic="0.3" lcAssertive="1.8" lcImpatience="0.5" lcCooperative="0.3" lcSpeedGain="2.0"
           lcKeepRight="0.3"/>
    <vType id="hv_queue" vClass="passenger" carFollowModel="Krauss" accel="2.4" decel="4.5" sigma="0.5"
           tau="1.0" minGap="1.0" speedFactor="normc(1.0,0.1,0.8,1.2)" color="245,215,40"
           lcStrategic="1.0" lcCooperative="0.0" lcCooperativeSpeed="0.0" lcSpeedGain="0.2" lcKeepRight="1.0"/>
    <vType id="hv_round" vClass="passenger" carFollowModel="Krauss" accel="2.6" decel="4.5" sigma="0.5"
           tau="1.0" minGap="2.5" speedFactor="normc(1.0,0.1,0.8,1.2)" color="60,190,90"
           lcStrategic="1.0" lcCooperative="0.8" lcSpeedGain="1.0" lcKeepRight="1.0"/>
    <vTypeDistribution id="hv" vTypes="hv_normal hv_idm hv_slow hv_nonyield hv_cutin" probabilities="0.3125 0.3125 0.125 0.125 0.125"/>
"""

# ④ 회전교차로를 지나게 되는 출발 → 도착 (원지동)
ROUND_OD = [
    ("ar_rb_s_in", "ar_rb_gym_in"), ("ar_rb_gym_out", "ar_rb_s_out"),
    ("ar_rb_b_app", "ar_rb_gym_in"), ("ar_rb_b_app", "ar_rb_s_out"),
    ("ar_rb_s_in", "ar_rb_f_dep"), ("ar_rb_gym_out", "ar_rb_f_dep"),
    ("ar_rb_s_in", "WBf_g"), ("ar_rb_gym_out", "WBf_g"),
    ("EBf_a0", "ar_rb_gym_in"), ("EBf_a0", "ar_rb_s_out"),
]
# ⑤ 대기줄을 지나게 되는 출발 → (경유) → 도착
QUEUE_OD = [
    ("EBf_a0", "ar_yg_a0_out", None),            # 양재대로 동행 → 염곡사거리 좌회전
    ("ar_hq_gate_out", "ar_yg_a0_out", None),    # 본사에서 나와 좌회전 (ego 와 같은 길)
    ("ar_gb_nb", "ar_ns_f3", "SC_NB_EB"),        # 경부 상행 → 서초IC 출구 → 남부순환로
]


# S2 합류 지점: 램프·연결로가 본선에 붙는 곳의 '본선 위쪽' 출발 도로와, 합류 뒤 도로(경유)
#   (출발, 합류 뒤 경유 도로) — 합류 지점 본선 오른쪽 두 차로에 차를 보내 끼어들 틈을 실제로 다투게 한다
MERGE_OD = [
    ("SB_a", "SB_w"),          # 양재IC 고리 램프(L7252) → 경부 하행 합류
    ("NB_a", "NB_w"),          # 고리 램프(L1470) → 경부 상행 합류
    ("EBf_a0", "CD_b"),        # 고리 램프(L1469) → 양재대로 연결로 합류
    ("ar_ej_f0", "ar_ej_f0x_tp"),  # 언주로 남행 공사 구간 차로 감소
]


def flows(edges, scale, seed=7, s2_merge_vph=0.0, s4_scale=1.0):
    """배경 OD + ④⑤ flow 줄 목록."""
    rng = random.Random(seed)
    deg_in, deg_out = set(), set()
    for e in edges.values():
        deg_in.add(e["to"])
        deg_out.add(e["frm"])
    # 막다른 길 끝에 유턴을 붙인 도로(add_dead_end_turns)도 원래대로 출발·도착 도로로 쓴다
    srcs = [k for k, e in edges.items() if e["frm"] not in deg_in or e.get("turn_start")]
    snks = [k for k, e in edges.items() if e["to"] not in deg_out or e.get("turn_end")]
    # 배경: 각 출발에서 멀리 떨어진 도착 3곳, 출발 차로 수에 비례
    pairs = []
    for s in srcs:
        far = [t for t in snks if math.dist(edges[s]["shape"][0][:2], edges[t]["shape"][-1][:2]) > 700]
        for t in rng.sample(far, min(3, len(far))):
            pairs.append((s, t, edges[s]["n"]))
    wsum = sum(w for _, _, w in pairs) or 1.0
    total = BASE_VPH * scale
    out = []
    for k, (a, b, w) in enumerate(pairs):
        vph = total * w / wsum
        out.append(f'    <flow id="fc{k}" type="hv" from="{a}" to="{b}" begin="0" end="100000" '
                   f'period="exp({vph / 3600:.5f})" departLane="random" departSpeed="max"/>')
    sp = total * SPECIAL_SHARE
    rod = [(a, b) for a, b in ROUND_OD if a in edges and b in edges]
    for k, (a, b) in enumerate(rod):
        vph = sp * float(s4_scale) / max(1, len(rod))
        out.append(f'    <flow id="fr{k}" type="hv_round" from="{a}" to="{b}" begin="0" end="100000" '
                   f'period="exp({vph / 3600:.5f})" departLane="best" departSpeed="max"/>')
    qod = [(a, b, v) for a, b, v in QUEUE_OD if a in edges and b in edges and (v is None or v in edges)]
    for k, (a, b, v) in enumerate(qod):
        vph = sp / max(1, len(qod))
        via = f' via="{v}"' if v else ""
        out.append(f'    <flow id="fq{k}" type="hv_queue" from="{a}" to="{b}"{via} begin="0" end="100000" '
                   f'period="exp({vph / 3600:.5f})" departLane="best" departSpeed="max"/>')
    # S2: 합류 지점 본선 교통 (기본 0 = 예전과 같음). 도착지는 합류 뒤 멀리 떨어진 도로 중 무작위
    if s2_merge_vph and s2_merge_vph > 0:
        for k, (a, via) in enumerate((a, v) for a, v in MERGE_OD if a in edges and v in edges):
            far = [t for t in snks if math.dist(edges[via]["shape"][-1][:2], edges[t]["shape"][-1][:2]) > 500]
            b = rng.choice(far) if far else via
            for lane in (0, 1):
                if lane >= edges[a]["n"]:
                    continue
                out.append(f'    <flow id="fm{k}_{lane}" type="hv" from="{a}" to="{b}" via="{via}" begin="0" end="100000" '
                           f'period="exp({s2_merge_vph * scale / 2 / 3600:.5f})" departLane="{lane}" departSpeed="max"/>')
    return out
