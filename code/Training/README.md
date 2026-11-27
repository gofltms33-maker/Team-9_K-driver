# Training

## BC + PPO

- [train_local_bc.py](../BC_PPO/train_local_bc.py): 기본 BC 학습
- [train_trust_region_correction.py](../BC_PPO/train_trust_region_correction.py): 실패 시연 보완
- [train_warmstart_ppo.py](../BC_PPO/train_warmstart_ppo.py): PPO 초기화 후 학습
- [train_bc_ppo.py](../BC_PPO/train_bc_ppo.py), [PPO 구현](../BC_PPO/algorithms/ppo.py)
- [학습 명령 기록](../BC_PPO/provenance/training_commands.original.json), [PPO 초기화 가중치](../BC_PPO/ppo_initialization/)

원본 학습 명령은 이전 PC의 절대 경로를 포함한다. 데이터 및 경로를 준비한 뒤 실행해야 한다. 원래 시연 NPZ가 모두 제공되지 않아 동일 학습의 완전한 재현은 불가능하다.

## P1\~P9 / DQN

- [train.py](../DQN/train.py), [train_bc.py](../DQN/train_bc.py)
- [PPO](../DQN/algorithms/ppo.py), [DQN](../DQN/algorithms/dqn.py), [BC](../DQN/algorithms/bc.py)
- [실험별 설정 및 결과](../DQN/experiments/)

모델 폴더에서 실행하는 기본 학습 진입점(최종 실험과 동일한 설정을 보장하지 않음):

```cmd
python train.py --algo ppo
python train.py --algo dqn
```

실험별 `run_config.json`에서 보상·하이퍼파라미터·교통 채움 시간을 확인한다. [DQN C4 설정](../DQN/experiments/DQN_C4_warmup100-300_final/run_config.json). 실제 재학습 전 `python train.py --help`와 설정 파일을 대조한다.
