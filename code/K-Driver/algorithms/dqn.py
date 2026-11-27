"""DQN (Double DQN) — 알고리즘 본체.

PPO 와 같은 환경·관측(40차원)·보상을 쓰고, 행동만 이산으로 바꾼다.

    행동 15개 = 가감속 5단계 × 차로 3개
        가감속: 강한 감속 / 감속 / 유지 / 가속 / 강한 가속  (−1, −0.5, 0, +0.5, +1) × 최대 가감속
        차로  : 오른쪽 / 유지 / 왼쪽                         (−1, 0, +1)
    Q-network 는 관측을 받아 15개 행동 각각의 Q값을 한 번에 출력한다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DQN 요약 (Mnih et al., 2015) + Double DQN (van Hasselt et al., 2016)

Q-network Q_φ(s, a): 상태 s 에서 행동 a 를 했을 때 앞으로 받을 할인 보상 합의 추정.
행동 선택: a = argmax_a Q_φ(s, a)  (학습 중에는 ε-greedy 로 탐험)

TD target (Double DQN):
    a* = argmax_a' Q_φ(s', a')            ← 다음 행동은 online network 로 고르고
    y  = r + γ (1 − d) Q_φ̄(s', a*)       ← 그 값은 target network 로 계산 (과대평가 완화)
손실:
    L(φ) = E[ Huber(Q_φ(s, a) − y) ]

안정화 장치:
    Experience Replay : 경험을 버퍼에 쌓아두고 무작위 미니배치로 여러 번 재사용
                        (연속된 경험 사이의 상관을 깨고, 드문 상황도 반복 학습)
    Target Network    : φ̄ ← φ 를 K 번 업데이트마다 복사 (학습 목표가 급격히 움직이지 않게)
    ε-greedy          : ε 을 1.0 에서 0.05 로 선형 감소 (초반 탐험 → 점차 활용)

시간 제한으로 끊긴 판(truncated)은 MDP 가 끝난 게 아니므로 d = 0 으로 두고
다음 상태의 Q 로 부트스트랩한다 (d = 1 은 충돌·도착·빨간불 같은 진짜 종료만).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

train.py 가 PPO 와 똑같이 부를 수 있도록 인터페이스를 맞췄다:
    learn(), predict(), save()/load(), save_checkpoint()/load_checkpoint(),
    num_timesteps, n_updates
여기서 "업데이트 1번"은 PPO 와 같은 n_steps(2048) 스텝 단위로 센다
(로그·평가 간격·중간 저장을 PPO 와 같은 기준으로 비교하기 위해).
"""
import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from utils.networks import build_mlp
from utils.evaluator import evaluate_policy

ACCEL_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)
LANE_LEVELS = (-1.0, 0.0, 1.0)
ACTIONS = [(a, l) for a in ACCEL_LEVELS for l in LANE_LEVELS]   # index = 3 × 가감속 + 차로


class ArrayReplayBuffer:
    """고정 크기 순환 버퍼 (numpy 배열). 가득 차면 가장 오래된 경험부터 덮어쓴다."""

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(self.capacity, dtype=np.int64)
        self.rew = np.zeros(self.capacity, dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)
        self.pos, self.size = 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, act, rew, next_obs, done):
        i = self.pos
        self.obs[i], self.next_obs[i] = obs, next_obs
        self.act[i], self.rew[i], self.done[i] = act, rew, float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size: int, device):
        idx = self.rng.integers(0, self.size, size=batch_size)
        to = lambda x: torch.as_tensor(x[idx], device=device)
        return to(self.obs), to(self.act), to(self.rew), to(self.next_obs), to(self.done)


class DQN:
    def __init__(
        self,
        obs_dim: int,
        # ---- 최적화 ----
        lr: float = 1e-4,
        gamma: float = 0.95,
        batch_size: int = 128,
        buffer_size: int = 200_000,
        learning_starts: int = 10_000,   # 버퍼에 이만큼 쌓인 뒤부터 학습 시작
        train_freq: int = 4,             # 환경 스텝 몇 번마다 경사 하강 1번
        target_update: int = 2_000,      # 경사 하강 몇 번마다 target network 복사 (K)
        max_grad_norm: float = 10.0,
        # ---- 탐험 (ε-greedy) ----
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 300_000,  # 이 스텝까지 선형으로 줄인다
        eval_eps: float = 0.05,          # predict(deterministic=False) 일 때의 ε
        # ---- 네트워크 ----
        hidden_sizes=(256, 256),
        activation: str = "relu",
        # ---- 로그 단위 ----
        n_steps: int = 2048,
        # ---- 기타 ----
        seed: int = 0,
        device: str = "auto",
        **_ignored,                      # PPO 전용 설정이 섞여 들어와도 무시
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.obs_dim, self.n_actions = obs_dim, len(ACTIONS)
        self.hidden_sizes, self.activation = tuple(hidden_sizes), activation
        self.q = build_mlp(obs_dim, hidden_sizes, self.n_actions, activation).to(self.device)
        self.q_target = build_mlp(obs_dim, hidden_sizes, self.n_actions, activation).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)

        self.gamma, self.batch_size = gamma, batch_size
        self.learning_starts, self.train_freq = learning_starts, train_freq
        self.target_update, self.max_grad_norm = target_update, max_grad_norm
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.eval_eps, self.n_steps = eval_eps, n_steps

        self.replay = ArrayReplayBuffer(buffer_size, obs_dim, seed)
        self.rng = np.random.default_rng(seed)
        self.num_timesteps = 0
        self.n_updates = 0
        self.grad_steps = 0

    # ==================================================================
    # 행동
    # ==================================================================
    @staticmethod
    def to_env_action(idx: int) -> np.ndarray:
        """행동 번호 → 환경이 받는 [가감속 raw, 차선변경 raw]."""
        a, l = ACTIONS[int(idx)]
        return np.array([a, l], dtype=np.float32)

    def epsilon(self) -> float:
        frac = min(self.num_timesteps / max(self.eps_decay_steps, 1), 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def q_values(self, obs) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.q(obs_t).squeeze(0).cpu().numpy()

    def select(self, obs, eps: float) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(obs)))

    def predict(self, obs, deterministic: bool = True):
        """평가/실행용 (utils/evaluator.py 인터페이스). 결정적이면 argmax Q."""
        idx = self.select(obs, 0.0 if deterministic else self.eval_eps)
        return self.to_env_action(idx)

    # ==================================================================
    # 학습 한 번 (미니배치 경사 하강)
    # ==================================================================
    def train_step(self):
        o, a, r, o2, d = self.replay.sample(self.batch_size, self.device)
        q = self.q(o).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a_star = self.q(o2).argmax(1, keepdim=True)                 # online 으로 고르고
            q_next = self.q_target(o2).gather(1, a_star).squeeze(1)     # target 으로 평가
            y = r + self.gamma * (1.0 - d) * q_next
        loss = F.smooth_l1_loss(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.grad_steps += 1
        if self.grad_steps % self.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item()), float(q.mean().item())

    # ==================================================================
    # 메인 학습 루프
    # ==================================================================
    def learn(self, env, total_timesteps: int, logger=None,
              log_interval: int = 1, eval_interval: int = 0,
              eval_episodes: int = 5, callback=None, eval_env=None):
        if hasattr(env, "num_envs"):                       # 병렬 환경 (utils/vec_env.py)
            return self._learn_vec(env, total_timesteps, logger, log_interval,
                                   eval_interval * self.n_steps, eval_episodes, callback, eval_env)
        eval_env = eval_env or env
        obs, _ = env.reset()
        ep_ret, ep_len = 0.0, 0
        recent_returns = deque(maxlen=20)
        n_updates = self.n_updates

        while self.num_timesteps < total_timesteps:
            losses, qs, n_lc = [], [], 0
            for _ in range(self.n_steps):
                eps = self.epsilon()
                idx = self.select(obs, eps)
                next_obs, reward, terminated, truncated, _ = env.step(self.to_env_action(idx))
                self.num_timesteps += 1
                n_lc += int(ACTIONS[idx][1] != 0.0)

                # d = 1 은 진짜 종료(충돌·도착·빨간불)만. 시간 제한으로 끊긴 판은 부트스트랩.
                self.replay.add(obs, idx, reward, next_obs, terminated)

                ep_ret += reward
                ep_len += 1
                if (self.num_timesteps >= self.learning_starts
                        and len(self.replay) >= self.batch_size
                        and self.num_timesteps % self.train_freq == 0):
                    loss, qm = self.train_step()
                    losses.append(loss)
                    qs.append(qm)

                if terminated or truncated:
                    recent_returns.append(ep_ret)
                    if logger is not None:
                        logger.log_episode(self.num_timesteps, ep_ret, ep_len)
                    ep_ret, ep_len = 0.0, 0
                    obs, _ = env.reset()
                else:
                    obs = next_obs

            n_updates += 1
            self.n_updates = n_updates
            metrics = {
                "update": n_updates,
                "ep_rew_mean": float(np.mean(recent_returns)) if recent_returns else float("nan"),
                "epsilon": self.epsilon(),
                "td_loss": float(np.mean(losses)) if losses else float("nan"),
                "q_mean": float(np.mean(qs)) if qs else float("nan"),
                "p_lane_change": n_lc / self.n_steps,
                "buffer": len(self.replay),
            }
            if logger is not None:
                logger.log_train(self.num_timesteps, metrics,
                                 verbose=(n_updates % log_interval == 0))

            if eval_interval > 0 and n_updates % eval_interval == 0:
                m = evaluate_policy(self, eval_env, n_episodes=eval_episodes)
                if logger is not None:
                    logger.log_eval(self.num_timesteps, m.as_dict(), m.summary(), eval_episodes)
                obs, _ = env.reset()
                ep_ret, ep_len = 0.0, 0

            if callback is not None:
                callback(self, n_updates)

    # ==================================================================
    # 병렬 환경 학습 루프: 환경 N 개의 경험을 한 리플레이 버퍼에 같이 쌓는다
    # ==================================================================
    def _learn_vec(self, venv, total_timesteps, logger, log_interval,
                   eval_every_steps, eval_episodes, callback, eval_env):
        n = venv.num_envs
        obs = venv.reset()
        ep_ret, ep_len = np.zeros(n), np.zeros(n, dtype=int)
        recent_returns = deque(maxlen=20)
        n_updates = self.n_updates
        next_eval = ((self.num_timesteps // eval_every_steps) + 1) * eval_every_steps \
            if eval_every_steps > 0 else None
        credit = 0.0                       # 환경 스텝 train_freq 번마다 경사 하강 1번

        while self.num_timesteps < total_timesteps:
            losses, qs, n_lc, n_seen = [], [], 0, 0
            for _ in range(max(1, self.n_steps // n)):
                eps = self.epsilon()
                with torch.no_grad():
                    greedy = self.q(torch.as_tensor(obs, dtype=torch.float32,
                                                    device=self.device)).argmax(1).cpu().numpy()
                explore = self.rng.random(n) < eps
                idxs = np.where(explore, self.rng.integers(self.n_actions, size=n), greedy)
                acts = np.stack([self.to_env_action(k) for k in idxs])
                next_obs, r, term, trunc, infos = venv.step(acts)
                self.num_timesteps += n
                for i in range(n):
                    done_i = term[i] or trunc[i]
                    # 판이 끝났으면 next_obs 는 이미 새 판 첫 관측 → 끝난 판의 마지막 관측을 쓴다
                    nxt = infos[i]["final_obs"] if done_i else next_obs[i]
                    self.replay.add(obs[i], int(idxs[i]), float(r[i]), nxt, bool(term[i]))
                    n_lc += int(ACTIONS[int(idxs[i])][1] != 0.0)
                    ep_ret[i] += r[i]
                    ep_len[i] += 1
                    if done_i:
                        recent_returns.append(float(ep_ret[i]))
                        if logger is not None:
                            logger.log_episode(self.num_timesteps, float(ep_ret[i]), int(ep_len[i]))
                        ep_ret[i], ep_len[i] = 0.0, 0
                n_seen += n
                obs = next_obs
                if self.num_timesteps >= self.learning_starts and len(self.replay) >= self.batch_size:
                    credit += n / self.train_freq
                    while credit >= 1.0:
                        loss, qm = self.train_step()
                        losses.append(loss)
                        qs.append(qm)
                        credit -= 1.0

            n_updates += 1
            self.n_updates = n_updates
            metrics = {
                "update": n_updates,
                "ep_rew_mean": float(np.mean(recent_returns)) if recent_returns else float("nan"),
                "epsilon": self.epsilon(),
                "td_loss": float(np.mean(losses)) if losses else float("nan"),
                "q_mean": float(np.mean(qs)) if qs else float("nan"),
                "p_lane_change": n_lc / max(n_seen, 1),
                "buffer": len(self.replay),
            }
            if logger is not None:
                logger.log_train(self.num_timesteps, metrics, verbose=(n_updates % log_interval == 0))

            if next_eval is not None and self.num_timesteps >= next_eval:
                m = evaluate_policy(self, eval_env, n_episodes=eval_episodes)
                if logger is not None:
                    logger.log_eval(self.num_timesteps, m.as_dict(), m.summary(), eval_episodes)
                while next_eval <= self.num_timesteps:
                    next_eval += eval_every_steps

            if callback is not None:
                callback(self, n_updates)

    # ==================================================================
    # 저장 / 불러오기
    # ==================================================================
    def _model_dict(self):
        return {"algorithm": "dqn", "obs_dim": self.obs_dim, "n_actions": self.n_actions,
                "hidden_sizes": list(self.hidden_sizes), "activation": self.activation,
                "actions": [list(a) for a in ACTIONS], "state_dict": self.q.state_dict()}

    def save(self, path: str):
        torch.save(self._model_dict(), path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        if "policy" in ck and "optimizer" in ck:
            ck = ck["policy"]                     # 중간 저장(checkpoint.pt)도 그대로
        self.q.load_state_dict(ck["state_dict"])
        self.q_target.load_state_dict(ck["state_dict"])

    # 이어서 학습하기용. 리플레이 버퍼는 크기가 커서 저장하지 않는다
    # (이어서 학습하면 learning_starts 만큼 다시 쌓은 뒤 학습을 재개한다).
    def save_checkpoint(self, path: str, extra: dict = None):
        tmp = path + ".tmp"
        torch.save({"policy": self._model_dict(),
                    "target": self.q_target.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "num_timesteps": self.num_timesteps,
                    "n_updates": self.n_updates,
                    "grad_steps": self.grad_steps,
                    "extra": extra or {}}, tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path: str) -> dict:
        ck = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ck["policy"]["state_dict"])
        self.q_target.load_state_dict(ck["target"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.num_timesteps = int(ck["num_timesteps"])
        self.n_updates = int(ck["n_updates"])
        self.grad_steps = int(ck.get("grad_steps", 0))
        self.learning_starts = self.num_timesteps + self.learning_starts   # 버퍼를 다시 채운 뒤 학습
        return ck.get("extra", {})
