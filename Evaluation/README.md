# Evaluation

[프로젝트 홈](../README.md) · [Environment](../Environment/README.md) · [Training](../Training/README.md) · [Data Collection](../Data_Collection/README.md)

## 최종 AV 재현

저장소 루트에서 다음 명령을 실행합니다.

```powershell
# GUI 주행 재생
python Evaluation/evaluate.py --episodes 1

# 기본 교통 100회 평가
python Evaluation/evaluate.py --nogui --episodes 100 --seed-start 1000

# 증가 교통 100회 평가: 출발 구간 유지, 나머지 도로 ×2·고속도로 ×3
python Evaluation/evaluate.py --nogui --episodes 100 --seed-start 1000 --heavy-traffic 2 3
```

`evaluate.py`는 공통 `test.py`를 실행하며 다음 기본값을 적용합니다. 명시적으로 전달한 옵션은 유지합니다.

| 항목 | 기본값 |
|---|---|
| 가중치 | `results/MIX13x17_w70/model.pt` |
| B | `--yield-cut on`: 상대 차량 도착 8초 이상이면 양보 필요 0 |
| Y | `--yellow-obs on`: 노란불에서 필요 감속 4 m/s² 기준 멈춤·통과 관측 |
| S | `--stalled-block on`: 진로 2 m 안 교차로 정지차가 있으면 양보 필요 1 |
| 교통 형성 | `--warmup 300-600` |

모델·출력 상대 경로는 `ppo_v1_parallel/` 기준입니다. 다른 모델은 **첫 번째 인수**에 경로를 지정하고 관측 옵션을 직접 맞춥니다. 최종 모델용 B·Y·S 기본값은 다른 모델을 지정해도 유지됩니다.

```powershell
python Evaluation/evaluate.py --help
python Evaluation/evaluate.py results/MIX13x17_w70/model.pt --nogui --episodes 10 --seed-start 2000
```

## 최종 체크포인트와 보고 성능

체크포인트는 SAFE13 × 0.7 + SAFE17 × 0.3으로, 추가 학습 없이 평균한 가중치입니다. 최종 발표자료 35\~37쪽은 다음 성능을 보고합니다.

| 평가 | 횟수 | 완주 | 충돌 | 신호 위반 | 경로 이탈 |
|---|---:|---:|---:|---:|---:|
| 기본 교통, 300\~600초 교통 형성 | 400 | 85.25% | 3.25% | 9.75% | 1.5% |
| 증가 교통 | 100 | 84% | 5% | 6% | 3% |

기본 교통은 시드 1000·2000·50000·60000번대, 증가 교통은 1000번대를 사용했습니다. 위 100회 명령 하나가 발표의 기본 교통 400회 전체 평가를 의미하지는 않습니다. 이 정리 작업에서 성능 평가를 새로 실행하지 않았습니다.

## 평가 시 확인할 설정

- `--cfg current`가 기본입니다. `--cfg train`은 모델 폴더의 보상·행동·시뮬레이션 설정을 복원하므로 조건을 구분해 기록합니다.
- `--obs auto`는 모델 폴더의 관측 사본을 불러옵니다. 최종 B·Y·S는 위 진입점에서 명시적으로 켭니다.
- 도로는 현재 `env/road_config.py`로 다시 생성합니다. 모델만 옮기지 말고 결과 폴더의 설정 사본도 유지합니다.
- 관측·보상·교통 형성 조건과 평가 횟수가 다른 BC+PPO·DQN 결과는 [상세 보고서](../docs/PROJECT_REPORT.md#8-final-model-comparison)의 조건과 함께 해석합니다.

기존 모델 재현은 [BC+PPO·DQN 평가 안내](../code/Evaluation/README.md)를 참고합니다.
