# Behavioral Cloning + PPO

[프로젝트 홈](../../README.md) · [전체 코드 안내](../README.md) · [DQN·초기 PPO](../DQN/README.md)

`BC_PPO model.zip`의 **v118 round004** 모델 패키지입니다. 기존 `code/BC_PPO_13p33_share/`에서 위치만 변경했습니다. BC로 확보한 가감속·주행망을 고정하고 PPO로 차로 변경 보정층을 학습한 구현입니다.

## 실행 준비와 주행 재생

SUMO를 설치하고 `SUMO_HOME`을 지정한 뒤 저장소 루트에서 실행합니다. 원본 설명서의 평가 환경은 SUMO 1.27.1입니다.

```powershell
cd code/BC_PPO
python -m pip install -r requirements.txt

# 당시 고정 환경에서 GUI 재생
python run_model.py --seed 1024000

# 화면 없이 재생
python run_model.py --nogui --seed 1024000 --episodes 2
```

현재 Python과 원본 `requirements.txt`의 고정 버전이 호환되는지 확인합니다. 실행기는 폴더 내부 상대 경로를 사용하고 고정 환경의 해시를 검사하므로 `model/`과 `fixture/`를 함께 유지합니다. 원본 설명은 [READ_ME_KO.txt](READ_ME_KO.txt)에 있습니다.

## 가중치와 평가 기록

| 항목 | 내용 |
|---|---|
| 선택 가중치 | [model/model.pt](model/model.pt) |
| 실행 설정 | [model/run_config.json](model/run_config.json) |
| 고정 환경 | [fixture/](fixture/) — 당시 도로·교통 수요·차량 특성 |
| PPO 초기 가중치 | [ppo_initialization/](ppo_initialization/) |
| 개발 평가 | 300회: 완주 255회(85%), 충돌 40회(13.33%) |
| 별도 평가 | 100회: 완주 86회(86%), 충돌 11회(11%) |
| 평가 상세 | [evaluation/](evaluation/) |

위 수치는 원본 패키지의 과거 기록입니다. 당시 교통 사전 형성은 꺼져 있었으며, 팀 최종 PPO의 300~600초 교통 형성 조건과 다릅니다. 이번 정리에서 주행 성능을 재측정하지 않았습니다.

## 학습·수집 코드

| 코드 | 역할 |
|---|---|
| [train_local_bc.py](train_local_bc.py) | 기본 BC 학습 |
| [train_trust_region_correction.py](train_trust_region_correction.py) | 실패 시연 기반 보완 |
| [train_warmstart_ppo.py](train_warmstart_ppo.py) | PPO 초기화·보완학습 |
| [train_bc_ppo.py](train_bc_ppo.py) | BC+PPO 관련 학습·시연 수집 |
| [collect_audited_course_data.py](collect_audited_course_data.py) | 시연 데이터 수집 |
| [training_logs/](training_logs/) | 당시 학습 기록 |
| [provenance/](provenance/) | 원본 실행 명령·설정·소스 해시·데이터 보유 현황 |

학습용 시연 NPZ는 포함되어 있지 않으며 일부 초기 시연은 원본에도 남아 있지 않아 당시 학습 전체의 동일 재현에는 제한이 있습니다. `provenance/`의 이전 PC 절대 경로는 원본 기록으로 보존했습니다. 학습하려면 데이터를 준비하고 실제 위치로 경로를 지정해야 합니다.
