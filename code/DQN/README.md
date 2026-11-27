# DQN · 초기 PPO P1\~P9

[프로젝트 홈](../../README.md) · [전체 코드 안내](../README.md) · [BC+PPO](../BC_PPO/README.md)

`K-Driver_P1-P9, DQN.zip`의 구현을 정리한 폴더입니다. 기존 `code/K-Driver/`에서 위치만 변경했으며 DQN과 P1\~P9가 공유하는 환경·학습 코드와 모델·실험 기록을 함께 보존했습니다.

## 실행 준비

저장소 루트에서 다음 명령을 실행합니다. SUMO는 별도 설치하고 `SUMO_HOME`과 `PATH`를 설정합니다. [환경 안내](../Environment/README.md)를 참고합니다.

```powershell
cd code/DQN
python -m pip install torch numpy gymnasium matplotlib tensorboard traci sumolib
```

위 목록은 소스의 직접 의존성 기준이며 원본 버전 잠금 파일은 없습니다. libsumo를 사용하려면 설치된 SUMO와 호환되는 버전을 별도 설치합니다.

## 최종 DQN 평가

아래 명령은 `code/DQN/` 안에서 실행합니다.

```powershell
# GUI 재생
python test.py models/DQN_final_C4_870k.pt --algorithm dqn --side-scan on --warmup 200 --warmup-jitter 100 --episodes 1

# 교통 사전 형성 100~300초, 100회 평가
python test.py models/DQN_final_C4_870k.pt --algorithm dqn --side-scan on --warmup 200 --warmup-jitter 100 --nogui --episodes 100
```

최종 발표의 D4 87만 스텝 모델은 완주율 58%, 충돌률 7%, 신호 위반률 28%를 보고했습니다. 이번 폴더 정리에서 성능을 다시 측정하지 않았습니다. 발표 조건과 실험 설정은 [최종 실험 설정](experiments/DQN_C4_warmup100-300_final/run_config.json) 및 [보고서](../../docs/PROJECT_REPORT.md#6-model-b-double-dqn)를 참고합니다.

## 학습과 도로 확인

```powershell
python train.py --help
python train.py --algo dqn
python train.py --algo ppo
python view_road.py --play
```

위 학습 명령은 기본 진입점이며 과거 각 실험의 재현 명령 전체를 뜻하지 않습니다. 실험별 `run_config.json`과 옵션을 함께 확인합니다.

## 파일 구성

| 위치 | 내용 |
|---|---|
| [train.py](train.py), [test.py](test.py) | PPO/DQN 학습·평가 |
| [algorithms/](algorithms/) | DQN·PPO·BC 구현 |
| [env/](env/) | SUMO 환경, 도로, 관측·행동·보상, 기록기 |
| [utils/](utils/) | 정책 네트워크·버퍼·평가·로깅 |
| [models/DQN_final_C4_870k.pt](models/DQN_final_C4_870k.pt) | 최종 DQN 가중치 |
| [models/PPO_v1_best_720k.pt](models/PPO_v1_best_720k.pt) | 초기 PPO 비교 가중치 |
| [experiments/](experiments/) | P1\~P9·DQN 실험 설정, 가중치, 학습·평가 기록 |
| [collect_pomdp_data.py](collect_pomdp_data.py) | 데이터 수집; 현재 정책 옵션은 random·keep-lane·PPO·BC이며 DQN 옵션 없음 |

팀 최종 선정 순수 PPO 모델은 별도의 [ppo_v1_parallel](../../ppo_v1_parallel/README.md)에 있습니다. 이 폴더의 DQN·초기 PPO와 구분합니다.
