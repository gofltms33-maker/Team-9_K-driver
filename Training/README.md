# Training

[프로젝트 홈](../README.md) · [Environment](../Environment/README.md) · [Evaluation](../Evaluation/README.md) · [Data Collection](../Data_Collection/README.md)

## PPO 실행

모든 명령은 저장소 루트에서 실행합니다. 이 진입점은 원본 `ppo_v1_parallel/train.py`를 실행하며 결과를 `ppo_v1_parallel/results/`에 저장합니다. 모델과 결과의 상대 경로는 **`ppo_v1_parallel/` 기준**입니다.

```powershell
python Training/train.py --help
python Training/train.py --name PPO_parallel --n-envs 8 --timesteps 200000 --lr 2e-5 --seed 0

# 모델 가중치를 가져와 새 실험 시작
python Training/train.py --name PPO_finetune --init-model results/SAFE13_20261125_185902/eval_models/model_00053248.pt --n-envs 8

# 기존 실험 이어서 학습 (해당 결과 폴더에 추가 기록)
python Training/train.py --resume results/SAFE13_20261125_185902
```

## 학습 설정과 출력

| 항목 | 위치 |
|---|---|
| 학습률·할인율·rollout·은닉층 | [train.py](../ppo_v1_parallel/train.py)의 `HPARAMS`와 CLI |
| PPO 업데이트 | [algorithms/ppo.py](../ppo_v1_parallel/algorithms/ppo.py) |
| 연속 가감속·이산 차로 선택 정책 | [utils/networks.py](../ppo_v1_parallel/utils/networks.py) |
| 병렬 환경 | [utils/vec_env.py](../ppo_v1_parallel/utils/vec_env.py) |
| 관측·행동·기본 보상 | [env/mdp_config.py](../ppo_v1_parallel/env/mdp_config.py) |
| 실험 보상 덮어쓰기 | `train.py`의 `REWARD_OVERRIDES`와 CLI |

출력에는 모델·체크포인트, `run_config.json`, 설정·학습 코드 사본, `training_log.csv`, `eval_log.csv`가 포함됩니다. TensorBoard를 설치한 경우 관련 로그를 기록할 수 있습니다. `--record`는 학습 중 압축 CSV 기록을 활성화합니다.

## 최종 모델 생성 과정

최종 발표 모델은 단일 새 학습 명령의 결과가 아니라 실패 사례 분석, SAFE5→SAFE17 이어 학습, 가중치 평균과 관측 수정을 거쳐 선정했습니다.

1. 충돌·신호 위반의 위치와 직전 행동을 분석합니다.
2. 늦은 제동·정지차 근접 가속·TTC·정지선·양보에 즉시 감점을 추가합니다.
3. 기존 정책에서 이어 학습합니다. 충돌 감점은 -5로 유지합니다.
4. SAFE13 53,248스텝과 SAFE17 `model.pt` 가중치를 7:3으로 평균합니다.
5. B·Y·S 관측 수정을 적용해 최종 평가합니다.

완성 가중치는 [`MIX13x17_w70/model.pt`](../ppo_v1_parallel/results/MIX13x17_w70/model.pt)에 포함되어 있습니다. 별도 가중치 혼합 실행 스크립트는 제공하지 않으며, 혼합 시 추가 학습은 수행하지 않았습니다. 원본 실행·설정·실험 기록은 [PPO 구현 안내](../ppo_v1_parallel/README.md)에 있습니다.

## 다른 알고리즘

- BC + PPO: [기존 패키지](../DQN_BC_PPO/BC_PPO/)와 [학습 안내](../DQN_BC_PPO/Training/README.md)
- 초기 PPO P1\~P9·Double DQN: [기존 구현](../DQN_BC_PPO/DQN/)
- 실패 원인과 정량 결과: [최종 발표 기반 보고서](../docs/PROJECT_REPORT.md#4-av-training-and-problem-solving)

기존 패키지는 모델별 관측·의존성이 다르므로 해당 폴더 안내에 따라 실행합니다.
