# Environment

[프로젝트 홈](../README.md) · [Training](../Training/README.md) · [Evaluation](../Evaluation/README.md) · [Data Collection](../Data_Collection/README.md)

## 설치

저장소 루트에서 가상환경을 활성화하고 `python -m pip install -r requirements.txt`를 실행합니다. SUMO와 `sumo-gui`, `netconvert` 실행 파일은 별도로 필요합니다. 발표 환경은 SUMO 1.27.1이며 `traci`, `sumolib`, 선택 의존성 `libsumo`는 설치된 SUMO와 버전을 맞춥니다.

```powershell
# 실제 설치 경로로 수정
$env:SUMO_HOME = 'C:\Program Files (x86)\Eclipse\Sumo'
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
sumo --version
netconvert --version
```

`SUMO_HOME`을 설정하면 원본 환경 코드에서 해당 설치의 `tools/`를 Python 검색 경로에 추가합니다. GUI는 TraCI, 화면 없는 실행은 설치되어 있으면 libsumo를 사용합니다. 학습·평가 명령의 `--no-libsumo`로 TraCI를 선택할 수 있습니다.

## 도로 생성

아래 명령은 저장소 루트에서 실행합니다.

```powershell
python Environment/build.py
sumo-gui -c Environment/generated/highway.sumocfg
```

기본 출력은 `Environment/generated/`이며 `--output`으로 변경합니다. `highway.net.xml`, 차량 경로, 연결 정보와 SUMO 설정 파일을 생성합니다. 학습·평가 실행 시에도 원본 `road_builder`가 필요한 도로를 생성합니다.

## 구현 위치

| 항목 | 공통 소스 | 수정 내용 |
|---|---|---|
| 도로·차량·교통 | [road_config.py](../ppo_v1_parallel/env/road_config.py) | 레이아웃, 길이 축소, ego 성능, 배경 교통 |
| 도로 생성 | [road_builder.py](../ppo_v1_parallel/env/road_builder.py), [roads/](../ppo_v1_parallel/env/roads/) | 양재IC·서초IC·공사·합류·회전교차로 |
| 도로 입력 | [OSM JSON](../ppo_v1_parallel/env/roads/osm/) | 포함된 `loop_north.json`, `loop_south.json` 사용 |
| MDP 설정 | [mdp_config.py](../ppo_v1_parallel/env/mdp_config.py) | 시간·관측·행동·보상 |
| Gymnasium 환경 | [sumo_env.py](../ppo_v1_parallel/env/sumo_env.py) | reset·step, SUMO 연결, 양보·안전 판정 |

## 환경 요약

| 항목 | 설정 |
|---|---|
| 주행 경로 | 현대차 양재사옥 → 양재IC → 서초IC → 매봉역 → 서초종합체육관, 약 9.4 km |
| 도로망 | 156개 도로 조각, 전체 차로 길이 97.8 km, 신호 교차로 11곳 |
| 상호작용 | S1 대기줄 진입, S2 합류·차로 감소, S3 저속 차량 추월, S4 회전교차로 |
| HV | 일반 A·B 각 25%, 저속·비양보·늦은 진입·회전교차로·대기줄 차량 각 10% |
| 제어 간격 | 0.5초 |
| ego 최대 가감속 | ±5.4 m/s² |
| 최종 PPO 교통 형성 | 출발 전 300~600초 |
| 최종 데이터셋 관측 | 43차원 |

정책은 가감속과 차로 선택을 출력합니다. 최종 차로 변경 게이트는 앞 간격 `5 m + 0.3 s × 자차 속도`, 뒤 간격 `5 m + 0.5 s × 뒤차 속도`, 접근 위험 1초를 확인하며 정문 진입로 60 m 이내 변경을 제한합니다. 실험별 설정은 결과 폴더의 설정 사본과 [상세 보고서](../docs/PROJECT_REPORT.md#2-simulation-environment)를 함께 확인합니다.
