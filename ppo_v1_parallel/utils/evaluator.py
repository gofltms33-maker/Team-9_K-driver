"""정책 평가 유틸 — 주행 지표 측정.

알고리즘과 분리해둔 이유: 평가는 "정책이 무엇으로 학습됐는지"와 무관하다.
`predict(obs, deterministic=True) -> action` 메서드만 있으면
PPO든 SAC든, 심지어 규칙 기반 컨트롤러든 똑같이 평가할 수 있다.

왜 별도 평가가 필요한가:
    학습 중 에피소드는 탐험 노이즈(σ)가 섞인 확률적 행동의 결과라
    "지금 정책의 실제 운전 실력"을 정확히 보여주지 못한다.
    그래서 주기적으로 노이즈 없는(deterministic) 정책으로 주행시켜
    충돌률 / 완주율 / 평균속도 / 차간거리를 따로 측정한다.
"""
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class DrivingMetrics:
    """평가 1회(n_episodes 평균)의 주행 지표."""
    collision_rate: float   # 충돌로 끝난 에피소드 비율 (0~1, 낮을수록 좋음)
    success_rate: float     # 도로 끝까지 완주한 비율 (0~1, 높을수록 좋음)
    timeout_rate: float     # 시간초과로 끝난 비율 (= 1 - 충돌 - 완주 - 신호위반)
    mean_speed: float       # 전 스텝 평균 주행속도 (m/s)
    mean_gap: float         # 선행차 감지 스텝에서의 평균 차간거리 (m)
    min_gap: float          # 관측된 최소 차간거리 (m) — 위험 운전 지표
    ep_return: float        # 에피소드당 평균 누적 보상
    ep_length: float        # 에피소드당 평균 스텝 수
    lane_changes: float     # 에피소드당 평균 "실행된" 차선변경 횟수
                            # (추월 시나리오에서 정책이 실제로 차선을
                            #  활용하는지 보는 핵심 지표)
    red_light_rate: float = 0.0  # 빨간불 위반으로 끝난 비율 (0~1). 시간초과와 따로 센다
                                 # (안 그러면 위반이 '시간초과'로 섞여 충돌률만 보면 좋아 보임)
    progress: float = 0.0        # 평균 경로 진행률 (0~1). 완주 판 = 1, 출발 못 하면 0.
                                 # 완주율이 0% 인 초반에도 "얼마나 멀리 가게 됐는지" 보이게 한다
    stuck_rate: float = 0.0      # 오래 서 있어서(stuck_timeout) 끝난 비율. timeout_rate 에 포함된 값
    km_driven: float = 0.0       # 에피소드당 평균 주행 거리 (km)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """콘솔 한 줄 출력용 문자열."""
        return (f"충돌률={self.collision_rate:.0%}  "
                f"완주율={self.success_rate:.0%}  "
                f"진행률={self.progress:.0%}  "
                f"정지종료={self.stuck_rate:.0%}  "
                f"주행={self.km_driven:.2f}km  "
                f"평균속도={self.mean_speed:.1f}m/s  "
                f"평균차간={self.mean_gap:.1f}m  "
                f"최소차간={self.min_gap:.1f}m  "
                f"신호위반={self.red_light_rate:.0%}  "
                f"차선변경={self.lane_changes:.1f}회  "
                f"보상={self.ep_return:.1f}")


def evaluate_policy(policy, env, n_episodes: int = 5,
                    deterministic: bool = True, seed_start=None, verbose: bool = False) -> DrivingMetrics:
    """정책을 n_episodes 주행시키고 주행 지표 평균을 반환.

    Args:
        policy: `predict(obs, deterministic) -> action` 을 가진 객체
                (PPOAgent, 또는 같은 인터페이스의 무엇이든)
        env:    SumoHighwayEnv (mean_gap 기본값용 visibility 속성만 사용)
        n_episodes: 평가 에피소드 수. 많을수록 지표가 안정적이지만 느리다
        deterministic: True면 탐험 노이즈 없이 평균 행동만 사용

    Note:
        모든 지표는 env.step()의 info(collided/arrived/speed/gap)에서 읽는다.
        관측(obs) 벡터의 구성과 무관하므로, 관측을 바꿔도 이 코드는 안 깨진다.
    """
    n_collide, n_arrive, n_timeout, n_red, n_stuck = 0, 0, 0, 0, 0
    speeds, gaps, returns, lengths, lane_changes = [], [], [], [], []
    progresses, kms = [], []

    for ep_i in range(n_episodes):
        if seed_start is not None:
            # 판마다 시드를 정해 두면 SUMO 교통 시드·교통 채우는 시간이 고정되어,
            # 여러 모델을 '똑같은 교통 상황 n판'으로 비교할 수 있다 (env 가 np.random 으로 정함).
            np.random.seed(int(seed_start) + ep_i)
        obs, _ = env.reset()
        done, ep_ret, ep_len = False, 0.0, 0
        ep_lc = 0
        ended_by = None
        first_rem = last_rem = None
        stuck = False
        where, ego_v, end_info = "", 0.0, {}

        while not done:
            action = policy.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            ep_len += 1
            done = terminated or truncated
            if verbose and not done:
                # 판이 끝난 위치를 알려주려고 매 스텝 ego 도로·속도를 기억 (종료 스텝에는 ego 가 이미 없음)
                try:
                    from env.sumo_env import traci as _tr
                    where = f"{_tr.vehicle.getRoadID(env.ego_id)} {_tr.vehicle.getLaneIndex(env.ego_id)}번 차로"
                    ego_v = float(_tr.vehicle.getSpeed(env.ego_id))
                except Exception:  # noqa: BLE001
                    pass
            if done:
                end_info = info

            # 실행된 차선변경 집계 (유지 명령의 changeLane 핀 동작은 제외)
            if info.get("lane_change", 0) != 0 and info.get("lane_change_applied"):
                ep_lc += 1

            if info.get("collided"):
                ended_by = "collision"
            elif info.get("arrived"):
                ended_by = "arrival"
            elif info.get("red_light_violation"):
                ended_by = "red_light"
            if info.get("stuck_timeout"):
                stuck = True

            rem = info.get("remaining_m")
            if rem is not None:
                if first_rem is None:
                    first_rem = float(rem)
                # 가장 가까이 갔던 값(최솟값)을 쓴다. 마지막 값을 쓰면 경로를 놓쳐 우회로가 붙거나
                # 지도 밖으로 빠진(no_way_back) 판은 남은 거리가 커져서 진행률이 0 으로 잘못 나온다.
                last_rem = float(rem) if last_rem is None else min(last_rem, float(rem))

            # 속도/차간거리는 env가 info로 넘겨주는 원시값(m/s, m)을 쓴다.
            # 관측(obs) 레이아웃이 바뀌어도 평가 코드는 영향받지 않는다.
            # 종료 스텝은 None (ego가 이미 제거됨) → 통계에서 제외.
            if info.get("speed") is not None:
                speeds.append(float(info["speed"]))
            if info.get("gap") is not None:
                gaps.append(float(info["gap"]))

        if ended_by == "collision":
            n_collide += 1
        elif ended_by == "arrival":
            n_arrive += 1
        elif ended_by == "red_light":
            n_red += 1
        else:
            n_timeout += 1
            n_stuck += int(stuck)

        # 진행률: 출발 때 남은 거리 대비 줄어든 비율 (우회로로 늘어나면 0 으로 자름)
        if ended_by == "arrival":
            progresses.append(1.0)
            kms.append((first_rem or 0.0) / 1000.0)
        elif first_rem and last_rem is not None:
            progresses.append(float(np.clip(1.0 - last_rem / first_rem, 0.0, 1.0)))
            kms.append(max(first_rem - last_rem, 0.0) / 1000.0)
        else:
            progresses.append(0.0)
            kms.append(0.0)

        if verbose:
            if ended_by == "collision":
                why = "충돌"
            elif ended_by == "arrival":
                why = "완주"
            elif ended_by == "red_light":
                why = "신호위반"
            elif end_info.get("no_way_back"):
                why = "경로 이탈 (돌아갈 길 없음)"
            elif end_info.get("reroute_limit"):
                why = "경로 재탐색 한도 초과"
            elif stuck:
                why = "오래 서 있음"
            else:
                why = "시간초과"
            t_s = ep_len * float(getattr(env, "step_length", 0.5))
            print(f"[{ep_i + 1}/{n_episodes}판] {why}  |  위치: {where or '?'}  속도 {ego_v:.1f} m/s  |  "
                  f"출발 후 {t_s:.0f}초  보상 {ep_ret:.1f}", flush=True)

        returns.append(ep_ret)
        lengths.append(ep_len)
        lane_changes.append(ep_lc)

    return DrivingMetrics(
        collision_rate=n_collide / n_episodes,
        success_rate=n_arrive / n_episodes,
        timeout_rate=n_timeout / n_episodes,
        red_light_rate=n_red / n_episodes,
        mean_speed=float(np.mean(speeds)) if speeds else 0.0,
        mean_gap=float(np.mean(gaps)) if gaps else float(env.visibility),
        min_gap=float(np.min(gaps)) if gaps else float(env.visibility),
        ep_return=float(np.mean(returns)),
        ep_length=float(np.mean(lengths)),
        lane_changes=float(np.mean(lane_changes)),
        progress=float(np.mean(progresses)),
        stuck_rate=n_stuck / n_episodes,
        km_driven=float(np.mean(kms)),
    )
