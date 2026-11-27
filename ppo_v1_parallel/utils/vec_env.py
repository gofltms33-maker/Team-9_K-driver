"""여러 SUMO 환경을 프로세스 여러 개로 동시에 돌리는 병렬 환경.

각 작업자(worker) 프로세스가 SumoHighwayEnv 하나를 갖고, 메인은 행동을 한꺼번에 보내고 결과를 모은다.
    - 판이 끝나면 작업자가 바로 새 판을 시작한다(자동 리셋).
      끝난 판의 마지막 관측은 info["final_obs"] 로 함께 돌려준다 (시간 제한 부트스트랩용).
    - 데이터 기록(--record)도 작업자마다 따로 해서 data/w<번호>/ 에 저장한다.
      에피소드 번호는 작업자마다 1,000,000 씩 떨어뜨려 겹치지 않게 한다.

PPO 처럼 환경 여러 개에서 동시에 경험을 모으면, 한 번의 업데이트에 여러 판의 서로 다른 상황이 섞여
학습이 한두 판의 운에 덜 휘둘리고, 같은 시간에 더 많은 경험을 모을 수 있다.
"""
import multiprocessing as mp
import os
import time

import numpy as np

EPISODE_OFFSET = 1_000_000


def _close_env(env):
    rows = 0
    if getattr(env, "recorder", None) is not None:
        try:
            env.recorder.close()
            rows = env.recorder.total_rows
        except Exception:  # noqa: BLE001
            pass
    try:
        env.close()
    except Exception:  # noqa: BLE001
        pass
    return rows


def _worker(remote, idx, env_kwargs, random_start_prob, record_dir, start_ep, start_part, seed):
    np.random.seed(seed)
    from env.sumo_env import SumoHighwayEnv
    env = SumoHighwayEnv(**env_kwargs)
    env.random_start_prob = random_start_prob
    env.queue_start_prob = float(env_kwargs["mdp_sim"].get("train_queue_start_prob", 0.0))
    if record_dir:
        from env.step_recorder import StepRecorder
        env.recorder = StepRecorder(env, record_dir, start_episode=start_ep, start_part=start_part)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, r, term, trunc, info = env.step(data)
                if term or trunc:
                    info = dict(info)
                    info["final_obs"] = obs
                    obs, _ = env.reset()
                remote.send((obs, r, term, trunc, info))
            elif cmd == "reset":
                obs, _ = env.reset()
                remote.send(obs)
            elif cmd == "close":
                remote.send(_close_env(env))
                break
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        # Ctrl+C 는 작업자에게도 전달된다: 데이터 파일만 깔끔히 닫고 끝낸다
        _close_env(env)


class SubprocVecEnv:
    def __init__(self, n_envs, env_kwargs, random_start_prob=0.0, record_root=None,
                 resume=False, base_seed=0):
        from env.step_recorder import resume_point
        ctx = mp.get_context("spawn")
        self.num_envs = int(n_envs)
        self.remotes, self.procs = [], []
        self.record_dirs = []
        t0 = int(time.time()) % 100000
        for i in range(self.num_envs):
            record_dir, start_ep, start_part = None, i * EPISODE_OFFSET, 0
            if record_root:
                record_dir = os.path.join(record_root, f"w{i}")
                os.makedirs(record_dir, exist_ok=True)
                if resume:
                    ep, part = resume_point(record_dir)
                    if part > 0:
                        start_ep, start_part = ep, part
            self.record_dirs.append(record_dir)
            a, b = ctx.Pipe()
            seed = (int(base_seed) * 7919 + i * 104729 + t0) % (2 ** 31 - 1)
            p = ctx.Process(target=_worker,
                            args=(b, i, env_kwargs[i] if isinstance(env_kwargs, (list, tuple)) else env_kwargs,
                                  random_start_prob, record_dir, start_ep, start_part, seed),
                            daemon=True)
            p.start()
            b.close()
            self.remotes.append(a)
            self.procs.append(p)

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return np.stack([r.recv() for r in self.remotes]).astype(np.float32)

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", np.asarray(a, dtype=np.float32)))
        out = [r.recv() for r in self.remotes]
        obs, rew, term, trunc, infos = zip(*out)
        return (np.stack(obs).astype(np.float32), np.asarray(rew, dtype=np.float32),
                np.asarray(term, dtype=bool), np.asarray(trunc, dtype=bool), list(infos))

    def close(self):
        """작업자를 정리하고, 저장한 데이터 행 수의 합을 돌려준다."""
        rows = 0
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:  # noqa: BLE001
                pass
        for r in self.remotes:
            try:
                if r.poll(60):
                    rows += int(r.recv())
            except Exception:  # noqa: BLE001
                pass
        for p in self.procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        return rows
