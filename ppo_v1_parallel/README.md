# PPO 병렬 학습 · 평가 · 주행 데이터 수집

로컬 `ppo_v1_parallel` 작업 폴더의 소스와 학습 결과를 정리한 독립 실행 폴더입니다. SUMO의 양재IC 주변 순환 도로에서 자율주행 차량의 가감속과 차로 변경을 PPO로 학습하고, 저장된 정책을 평가하거나 주행 데이터를 수집합니다.

## 최종 발표 모델

최종 발표자료 `K-Driver_final 박제민 수정본.pptx`(33~37쪽)에서 선정한 팀 최종 모델은 **SAFE13 × 0.7 + SAFE17 × 0.3 가중치 평균 + B·Y·S 관측 수정**입니다. 가중치 평균에는 추가 학습을 수행하지 않았습니다.

- 가중치: [`results/MIX13x17_w70/model.pt`](results/MIX13x17_w70/model.pt)
- B (`--yield-cut on`): 상대 차량 도착까지 8초 이상 남으면 양보 필요 값을 0으로 설정
- Y (`--yellow-obs on`): 노란불에서 필요 감속 4 m/s²를 기준으로 멈춤·통과 관측
- S (`--stalled-block on`): 교차로 내부 정지차가 자차 진로 2 m 안에 있으면 양보 필요 값을 1로 설정

| 최종 발표 평가 조건 | 횟수 | 완주율 | 충돌률 | 신호 위반률 | 경로 이탈률 |
|---|---:|---:|---:|---:|---:|
| 기본 교통, 출발 전 교통 형성 300~600초 | 400 | 85.25% | 3.25% | 9.75% | 1.5% |
| 출발 구간 유지, 나머지 도로 ×2·고속도로 ×3 | 100 | 84% | 5% | 6% | 3% |

기본 교통은 시드 1000·2000·50000·60000번대, 증가 교통은 1000번대를 사용한 발표 결과입니다. 기본 교통에서 충돌률 5% 미만 목표를 달성했습니다. 이번 문서 수정에서 해당 평가를 새로 수행하지는 않았습니다.

## 구성

| 경로 | 내용 |
|---|---|
| `train.py` | PPO 학습, 병렬 환경, 기존 모델에서 추가 학습, 중간 평가와 기록 |
| `test.py` | 저장 모델 GUI 재생 및 화면 없는 평가, PPO/BC 모델 판별 |
| `collect_data.py` | 고정 PPO 모델로 여러 작업자에서 주행 데이터 수집, part별 시드 분리 |
| `algorithms/ppo.py` | PPO 업데이트 및 정책 관리 |
| `algorithms/bc.py` | BC 정책 정의·저장·불러오기; 별도의 BC 학습 실행 파일은 포함되어 있지 않음 |
| `env/sumo_env.py` | Gymnasium 환경, 관측·보상·행동 적용, SUMO 연결 |
| `env/road_config.py` | 도로 레이아웃, ego 차량, 배경 교통 설정 |
| `env/mdp_config.py` | 시뮬레이션, 관측, 행동, 보상 설정 |
| `env/roads/` | 양재IC·서초IC·연결 도로·공사 구간 등 도로 생성 모듈 |
| `env/roads/osm/loop_*.json` | 순환 도로 생성에 필요한 OSM 입력 데이터 |
| `env/step_recorder.py` | 관측·행동·보상·주변 차량·상황별 지표를 압축 CSV로 기록 |
| `utils/` | 정책 네트워크, rollout buffer, 병렬 환경, 평가, 로깅, 교통량 조절 |
| `results/` | 기존 실험의 모델, 설정 사본, 학습·평가 로그와 평가 경로 |
| `source_manifest.json` | 복사한 원본 파일 목록·크기·SHA-256 |

정책 기본값은 연속 가감속과 이산 차로 변경을 결합한 hybrid actor-critic입니다. 관측에는 주변 차량, 경로, 신호와 비보호 교차로 관련 정보가 포함되며, 차로 변경 시 앞뒤 간격을 확인합니다. 세부 옵션과 실험별 관측 설정은 소스 및 각 결과 폴더의 설정 사본을 기준으로 확인합니다.

## 실행 준비

저장소를 내려받은 뒤 **이 폴더 안에서** 실행합니다. 상위 저장소의 다른 구현과 import 경로가 섞이지 않도록 별도 가상환경을 사용합니다.

```powershell
cd ppo_v1_parallel
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

SUMO를 별도로 설치하고 `sumo`, `sumo-gui`, `netconvert` 실행 파일을 사용할 수 있도록 설정해야 합니다. Windows 기본 설치 위치를 자동 탐색하지만, 다른 경로에 설치했다면 실제 설치 위치로 `SUMO_HOME`과 `PATH`를 지정합니다.

```powershell
# 실제 설치 경로에 맞게 수정
$env:SUMO_HOME = 'C:\Program Files (x86)\Eclipse\Sumo'
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
sumo --version
netconvert --version
```

화면 없는 실행을 빠르게 하려면 설치된 SUMO와 호환되는 `libsumo`를 추가 설치할 수 있습니다. 설치하지 않으면 TraCI 연결을 사용하며, `--no-libsumo`로 TraCI를 명시할 수도 있습니다. `requirements.txt`는 코드에서 확인한 직접 의존성 목록이며, 원래 환경의 버전 잠금 파일은 없으므로 SUMO·TraCI·sumolib·libsumo의 버전을 맞춰 사용해야 합니다.

## PPO 학습

```powershell
# 기본 설정으로 학습
python train.py

# 환경 8개를 병렬 실행
python train.py --name PPO_parallel --n-envs 8 --timesteps 200000 --lr 2e-5 --seed 0

# 저장된 실험 이어서 학습
python train.py --resume results/SAFE13_20261125_185902

# 지정한 모델 가중치에서 새 실험 시작
python train.py --name PPO_finetune --init-model results/SAFE13_20261125_185902/eval_models/model_00053248.pt --n-envs 8
```

새 실험에서는 `train.py`의 하이퍼파라미터·보상 설정을 확인합니다. 기존 실험을 이어서 실행하면 해당 결과 폴더에 추가 기록될 수 있습니다. `--record`를 추가하면 학습 중 주행 데이터를 기록합니다. 전체 옵션은 `python train.py --help`로 확인합니다.

## 모델 평가와 GUI 재생

최종 선정 모델을 재생·평가할 때는 혼합 가중치와 B·Y·S 옵션을 함께 지정합니다.

```powershell
# 팀 최종 모델 GUI 재생
python test.py results/MIX13x17_w70/model.pt --yield-cut on --yellow-obs on --stalled-block on --warmup 300-600 --episodes 1

# 기본 교통 평가 예시 (100회)
python test.py results/MIX13x17_w70/model.pt --yield-cut on --yellow-obs on --stalled-block on --warmup 300-600 --nogui --episodes 100 --seed-start 1000

# 교통량 증가 평가 예시 (100회)
python test.py results/MIX13x17_w70/model.pt --yield-cut on --yellow-obs on --stalled-block on --warmup 300-600 --heavy-traffic 2 3 --nogui --episodes 100 --seed-start 1000
```

위 기본 교통 명령은 100회 실행 예시이며, 발표의 400회 전체 평가 결과를 단독으로 재현하는 명령은 아닙니다. `--heavy-traffic 2 3`은 출발 구간을 유지하면서 나머지 도로·고속도로의 교통량을 조절합니다.

기존 `collect_data.py`의 기본 모델은 **SAFE13의 53,248스텝 모델 + yield-cut 관측 수정(B)**입니다. 이 개별 모델의 평가 예시는 다음과 같습니다.

```powershell
# GUI로 주행 재생
python test.py results/SAFE13_20261125_185902/eval_models/model_00053248.pt --yield-cut on --episodes 1

# 고정 시드로 화면 없이 평가
python test.py results/SAFE13_20261125_185902/eval_models/model_00053248.pt --yield-cut on --nogui --episodes 10 --seed-start 1000

# 학습 당시 보상·행동·시뮬레이션 설정을 적용해 평가
python test.py results/SAFE13_20261125_185902/eval_models/model_00053248.pt --cfg train --nogui --episodes 10 --seed-start 1000
```

`--obs auto`는 모델 옆의 학습 설정 사본을 이용해 관측을 복원합니다. `--cfg current`가 기본이며, `--cfg train`은 학습 당시 보상·행동·시뮬레이션 설정을 불러옵니다. `--yield-cut`, `--yellow-obs`, `--warmup` 등 추가 옵션은 주행 조건을 바꾸므로 모델 비교 시 조건과 시드를 함께 기록해야 합니다. 도로는 현재 `env/road_config.py`를 바탕으로 다시 생성됩니다.

## 주행 데이터 수집

현재 제공된 수집 코드의 기본값은 SAFE13 + B입니다. `--run`과 `--model`로 모델을 지정할 수 있지만, Y·S를 모두 켜는 CLI 옵션은 없으며 특히 S는 코드에서 기본적으로 꺼집니다. 따라서 아래 명령은 **팀 최종 혼합 모델 + B·Y·S 또는 발표 데이터셋의 동일 조건 재생성을 보장하는 명령이 아닙니다.** 수집 코드 원본은 유지했습니다.

```powershell
python collect_data.py --part 1 --rows 3334000 --workers 8
```

여러 사람이 나눠 수집하면 `--part 1`, `--part 2`, `--part 3`처럼 서로 다른 번호를 사용합니다. 기본 출력은 `data_collect/part<번호>/w<작업자>/data_partNNNN.csv.gz`입니다. 같은 part와 출력 경로로 재실행하면 기존 기록을 기준으로 이어서 수집합니다. 에피소드 단위로 종료하므로 최종 행 수는 요청한 목표를 넘을 수 있습니다.

기록에는 관측, 정책의 raw 행동과 적용 명령, 보상, 종료 여부, 주변 차량 상태, 차로 변경 거절·우회·신호 위반 사건, 추월·합류·대기줄·회전교차로 관련 지표가 포함됩니다. 정확한 칼럼 정의는 `env/step_recorder.py`를 참고합니다. 수집 결과를 분석할 때는 `pandas.read_csv()`로 압축 CSV를 읽을 수 있습니다(pandas 별도 설치).

최종 발표자료(43~44쪽)의 데이터셋 규모는 **10,417,849행, 6,314에피소드, 45칼럼, 관측 43차원**, 기록 간격은 0.5초입니다. 발표 시나리오와 데이터 칼럼 접두어는 S1 대기줄=`s3.*`, S2 합류=`s2.*`, S3 추월=`s1.*`, S4 회전교차로=`s4.*`로 대응합니다. 전체 데이터셋은 이 코드 폴더에 포함되어 있지 않습니다.

## 포함한 학습 결과

| 결과 폴더 | 내용 |
|---|---|
| `SAFE5_20261124_201506` | 학습 모델, 중간 평가 모델, 로그·설정·평가 경로 |
| `SAFE11_20261125_012159` | 학습 모델, 중간 평가 모델, 로그·설정·평가 경로 |
| `SAFE13_20261125_185902` | 데이터 수집 기본 모델 `eval_models/model_00053248.pt` 포함 |
| `SAFE17_20261126_024350` | 학습 모델, 중간 평가 모델, 로그·설정·평가 경로 |
| `MIX13x17_w50` | SAFE13 53,248스텝 모델과 SAFE17 `model.pt` 가중치 50:50 평균 |
| `MIX13x17_w70` | 같은 두 모델의 가중치 70:30 평균; B·Y·S 적용 시 최종 발표 선정 모델 |

MIX 모델은 추가 학습 없이 가중치를 평균한 실험입니다. 폴더 안 원본 `README.txt`는 B·Y 옵션을 안내하며, **최종 발표 선정 조건에는 S(`--stalled-block on`)도 추가해야 합니다.** SAFE 실험의 `best_eval.json`·`eval_log.csv`는 개별 학습 당시 기록이고, 위 최종 혼합 모델의 400회 성능은 발표자료 보고값입니다. 실험 설정 JSON 안의 로컬 절대 경로는 원본 기록으로 보존했으므로 다른 PC에서 재학습할 때 실제 모델 위치로 지정해야 합니다.

## 보존 범위와 확인 사항

- 소스, 도로 입력 JSON, 모델, 설정 사본과 실험 기록은 원본 내용 그대로 복사했습니다.
- `__pycache__`, 약 632 MiB의 로컬 `data_collect/`, 재생성 가능한 실험별 `sumo/`와 TensorBoard `tb/`는 제외했습니다.
- 새로운 실행이 만드는 도로·수집 데이터·실험 출력은 `.gitignore`로 제외하며, 이미 포함한 실험 파일은 계속 추적됩니다.
- 업로드 전 Python 문법과 JSON 파싱, Git에 추가한 원본 파일 140개의 SHA-256 일치, `train.py`·`test.py`·`collect_data.py`의 `--help` 실행을 확인했습니다. SUMO에서 실제 학습·평가·수집을 실행한 검증은 수행하지 않았습니다.
