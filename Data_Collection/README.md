# Data Collection

[프로젝트 홈](../README.md) · [Environment](../Environment/README.md) · [Training](../Training/README.md) · [Evaluation](../Evaluation/README.md)

## 수집 실행

저장소 루트에서 실행합니다. 이 진입점은 원본 `ppo_v1_parallel/collect_data.py`를 사용합니다.

```powershell
python Data_Collection/collect.py --help
python Data_Collection/collect.py --part 1 --rows 3334000 --workers 8
```

여러 사람이 수집할 때는 `--part 1`, `--part 2`, `--part 3`처럼 서로 다른 번호를 사용합니다. part·worker별로 시드·에피소드 번호를 분리합니다. 기본 결과는 `ppo_v1_parallel/data_collect/part<번호>/w<작업자>/data_partNNNN.csv.gz`에 저장됩니다. 같은 part와 출력 위치로 재실행하면 기존 에피소드를 기준으로 이어 수집하며, 에피소드 단위 저장으로 목표 행 수를 초과할 수 있습니다.

## 기본 모델과 재현 범위

현재 원본 수집기의 기본 모델은 **SAFE13 53,248스텝 + B**입니다. `--run`·`--model`로 다른 모델을 지정할 수 있지만, Y·S 전체를 활성화하는 CLI는 없으며 특히 S는 수집 코드에서 기본적으로 꺼집니다.

따라서 위 명령은 **팀 최종 7:3 + B·Y·S 또는 발표 데이터셋과 동일한 조건의 재생성을 보장하지 않습니다.** 최종 주행 재현에는 [Evaluation](../Evaluation/README.md)의 실행 파일을 사용합니다. 수집 코드 원본의 내용은 보존했습니다.

모델·출력의 상대 경로는 `ppo_v1_parallel/` 기준이며, 절대 경로도 사용할 수 있습니다. 상세 옵션은 `--help`에서 확인합니다.

## 발표 데이터셋

| 항목 | 값 |
|---|---:|
| 행 수 | 10,417,849 |
| 에피소드 | 6,314 |
| 칼럼 | 45 |
| 관측 차원 | 43 |
| 기록 간격 | 0.5초 |

최종 발표자료 43\~44쪽의 보고값이며, 전체 데이터셋은 이 저장소에 포함되어 있지 않습니다. 데이터는 실제 도로 측정값이 아니라 SUMO 정책 주행 기록입니다.

## 데이터 구성

한 행에는 에피소드·스텝·시뮬레이션 시각, 관측, raw 행동·차로 변경 명령과 적용 여부, 보상, 종료 여부, 주변 차량 상태와 사건을 기록합니다. JSON 형태 칼럼이 포함된 압축 CSV이며 `pandas.read_csv()`로 읽을 수 있습니다(pandas 별도 설치).

| 발표 시나리오 | 칼럼 접두어 | 주요 항목 |
|---|---|---|
| S1 대기줄 끼어들기 | `s3.*` | 목표 거리, 남은 차로 변경 수, 대기 차량 수, 목표 차로 앞뒤 간격 |
| S2 합류·차로 감소 | `s2.*` | 차로 끝 거리, 목표 앞뒤 간격, 뒤차 상대속도, 실제 가속도 |
| S3 저속 차량 추월 | `s1.*` | 앞차 간격·상대속도, 목표 차로 앞뒤 간격 |
| S4 회전교차로 | `s4.*` | 진입 거리, 접근 회전 차량 거리·속도 |

`s1/s2/s3.cutin_failed`는 차로 변경 명령이 안전 게이트에 막혔는지를 나타냅니다. 상황 번호와 접두어가 다르므로 위 대응을 사용합니다. 정확한 정의는 [step_recorder.py](../ppo_v1_parallel/env/step_recorder.py), 실제 수치 사례는 [데이터셋 보고서](../docs/PROJECT_REPORT.md#9-final-dataset)에 있습니다.
