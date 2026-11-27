"""제출용 주행 데이터 수집 (학습 없음, 최종 모델 고정).

최종 모델(SAFE13 5.3만 + 관측 수정 B)을 결정적으로 주행시키며 env/step_recorder.py 칼럼으로 매 스텝 기록한다.
여러 사람이 나눠 모을 때는 --part 만 다르게 주면 시드·에피소드 번호가 겹치지 않는다.

예 (3명이 10^7 행을 나눠 모으기, 각자 약 333만 행):
    python collect_data.py --part 1 --rows 3334000 --workers 8
    python collect_data.py --part 2 --rows 3334000 --workers 8
    python collect_data.py --part 3 --rows 3334000 --workers 8

결과: data_collect/part<번호>/w<작업자>/data_partNNNN.csv.gz  (pandas.read_csv 로 바로 읽힘)
중간에 꺼져도 그때까지 끝난 에피소드는 온전히 남는다. 같은 --part 로 다시 실행하면 이어서 모은다
(이미 기록한 에피소드 수만큼 시드를 건너뛴다).
"""
import argparse
import glob
import gzip
import json
import multiprocessing as mp
import os
import runpy
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUN = os.path.join(BASE, "results", "SAFE13_20261125_185902")
DEFAULT_MODEL = os.path.join(DEFAULT_RUN, "eval_models", "model_00053248.pt")

# 학습 때 관측 설정을 맞추는 항목 (test.py 와 같음). 학습 사본에 없으면 예전 값.
JKEYS = ("permissive_signal", "junction_foe_mode", "foe_horizon_m", "foe_horizon_s", "neighbor_scan",
         "leader_signals", "permissive_in_junction", "exit_lane_aware", "nav_horizon_s_fast",
         "internal_chain_via", "link_cache_clear", "foe_scan_internal", "gap_margin_obs", "yield_horizon_cut",
         "gap_margin_scale", "gap_margin_accel", "yellow_decision_obs", "yellow_decision_decel")
OLD = {"permissive_signal": None, "junction_foe_mode": "sumo", "neighbor_scan": False, "leader_signals": False,
       "permissive_in_junction": False, "exit_lane_aware": False, "nav_horizon_s_fast": 0.0,
       "internal_chain_via": False, "link_cache_clear": False, "foe_scan_internal": False,
       "gap_margin_obs": False, "yield_horizon_cut": False, "yellow_decision_obs": False,
       "stalled_foe_block": False, "queue_yield_mask": False, "far_lead_obs": False}

PART_SEED_STRIDE = 1_000_000      # 사람(part)마다 시드 구간
WORKER_SEED_STRIDE = 100_000      # 작업자마다 시드 구간
PART_EP_STRIDE = 100_000_000      # 사람마다 에피소드 번호 구간
WORKER_EP_STRIDE = 1_000_000      # 작업자마다 에피소드 번호 구간 (utils/vec_env.py 와 같은 간격)


def _done_episodes(rec_dir):
    """이미 기록된 에피소드 수 (이어서 모을 때 시드를 건너뛰기 위해)."""
    n = 0
    for f in sorted(glob.glob(os.path.join(rec_dir, "data_part*.csv.gz"))):
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                eps = set()
                next(fh, None)
                for line in fh:
                    eps.add(line.split(",", 1)[0])
                n += len(eps)
        except (OSError, EOFError):
            pass                      # 쓰다 끊긴 마지막 파일
    return n


def _worker(wid, args, counter, stop):
    os.environ["USE_LIBSUMO"] = "1"
    sys.path.insert(0, BASE)
    os.chdir(BASE)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from env.road_config import ROAD, EGO, TRAFFIC
    from env import mdp_config as MC
    from env import road_builder
    from env.sumo_env import SumoHighwayEnv
    from env.step_recorder import StepRecorder
    from algorithms.ppo import PPO
    from utils.buffer import RolloutBuffer
    import train as TR

    snap = runpy.run_path(os.path.join(args.run, "mdp_config.py"))
    obs_cfg = {**MC.OBSERVATION, **OLD, **{k: snap["OBSERVATION"][k] for k in JKEYS if k in snap["OBSERVATION"]}}
    obs_cfg["yield_horizon_cut"] = args.yield_cut == "on"          # 최종 모델 = SAFE13 5.3만 + B
    with open(os.path.join(args.run, "run_config.json"), encoding="utf-8") as f:
        reward = dict(json.load(f)["reward"])                        # 학습 당시 보상 (reward 칼럼)
    sim, action = dict(snap["SIMULATION"]), dict(snap["ACTION"])

    sumo_dir = os.path.join(BASE, "env", f"sumo_collect_p{args.part}_w{wid}")
    cfg = road_builder.build(ROAD, EGO, TRAFFIC, sumo_dir)
    env = SumoHighwayEnv(cfg_path=cfg, road=ROAD, ego=EGO, mdp_sim=sim, mdp_obs=obs_cfg, action=action,
                         reward=reward, traffic=TRAFFIC, gui=False)
    env.random_start_prob = 0.0
    agent = PPO(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0],
                buffer=RolloutBuffer(), **TR.HPARAMS)
    agent.policy.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    agent.policy.eval()

    rec_dir = os.path.join(args.out, f"part{args.part}", f"w{wid}")
    os.makedirs(rec_dir, exist_ok=True)
    done = _done_episodes(rec_dir)
    ep0 = args.part * PART_EP_STRIDE + wid * WORKER_EP_STRIDE + done
    n_parts = len(glob.glob(os.path.join(rec_dir, "data_part*.csv.gz")))
    env.recorder = StepRecorder(env, rec_dir, start_episode=ep0, start_part=n_parts)
    seed = args.part * PART_SEED_STRIDE + wid * WORKER_SEED_STRIDE + done

    last = 0
    while not stop.is_set():
        try:
            np.random.seed(seed)
            obs, _ = env.reset(seed=seed)
            while True:
                a = np.ravel(agent.predict(obs, deterministic=True)).astype(np.float32)
                obs, r, te, tr, info = env.step(a)
                if te or tr:
                    break
        except Exception as e:  # noqa: BLE001  한 판 실패(출발 막힘 등)는 건너뛰고 계속
            print(f"[w{wid}] 시드 {seed} 건너뜀: {e!r}", flush=True)
        seed += 1
        rows = env.recorder.total_rows
        with counter.get_lock():
            counter.value += rows - last
            total = counter.value
        last = rows
        if total >= args.rows:
            stop.set()
    try:
        env.recorder.close()
        env.close()
    except Exception:  # noqa: BLE001
        pass


def main():
    ap = argparse.ArgumentParser(description="제출용 주행 데이터 수집 (학습 없음)")
    ap.add_argument("--part", type=int, required=True, help="나눠 모을 때 사람 번호 (1, 2, 3 ...). 시드·에피소드가 겹치지 않게 함")
    ap.add_argument("--rows", type=int, default=3_334_000, help="이번 실행에서 모을 행 수 (기본 333만 4천 = 10^7 / 3)")
    ap.add_argument("--workers", type=int, default=8, help="동시에 돌릴 SUMO 수 (CPU 코어 수보다 작게)")
    ap.add_argument("--out", default=os.path.join(BASE, "data_collect"), help="저장 폴더")
    ap.add_argument("--run", default=DEFAULT_RUN, help="학습 결과 폴더 (관측·보상 설정을 여기서 읽음)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="모델 파일 (기본 SAFE13 5.3만)")
    ap.add_argument("--yield-cut", choices=["on", "off"], default="on", help="관측 수정 B (기본 on = 최종 모델)")
    args = ap.parse_args()
    os.environ["PYTHONPATH"] = ""

    ctx = mp.get_context("spawn")
    counter, stop = ctx.Value("q", 0), ctx.Event()
    procs = [ctx.Process(target=_worker, args=(w, args, counter, stop), daemon=False) for w in range(args.workers)]
    for p in procs:
        p.start()
    t0 = time.time()
    print(f"수집 시작: part {args.part}, 목표 {args.rows:,} 행, 작업자 {args.workers}개 → {args.out}", flush=True)
    try:
        while any(p.is_alive() for p in procs):
            time.sleep(60)
            n, dt = counter.value, time.time() - t0
            rate = n / dt * 60 if dt > 0 else 0
            eta = (args.rows - n) / rate if rate > 0 else float("inf")
            print(f"[{dt / 60:6.1f}분] {n:,} / {args.rows:,} 행 ({n / args.rows:.1%}) | 분당 {rate:,.0f} 행 | "
                  f"남은 시간 약 {eta / 60:.1f}시간", flush=True)
    except KeyboardInterrupt:
        print("중단 요청: 진행 중인 에피소드까지 저장하고 멈춥니다...", flush=True)
        stop.set()
    for p in procs:
        p.join()
    print(f"끝: {counter.value:,} 행 ({(time.time() - t0) / 3600:.1f}시간)", flush=True)


if __name__ == "__main__":
    main()
