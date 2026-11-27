# K-Driver | Code & Reproduction

성균관대학교 자율주행인공지능제어 9조 — 양재 IC 인근 SUMO 기반 자율주행 프로젝트.

[팀 프로젝트 저장소](https://github.com/gofltms33-maker/Team-9_K-driver) · [프로젝트 소개](../README.md)

## 구성 및 실행 안내

| 구분 | 안내 |
|---|---|
| Environment | [도로 환경 및 설치](Environment/README.md) |
| Training | [학습 코드와 설정](Training/README.md) |
| Evaluation | [모델 실행 및 평가](Evaluation/README.md) |
| Data Collection | [데이터 수집 코드](Data_Collection/README.md) |

## 모델별 소스

```text
code/
├── DQN/               # K-Driver_P1-P9, DQN.zip: DQN + 초기 PPO P1~P9
│   ├── README.md
│   ├── train.py / test.py
│   ├── algorithms/ / env/ / utils/
│   ├── models/        # 최종 DQN 및 PPO 비교 모델
│   └── experiments/   # 실험별 설정·모델·기록
├── BC_PPO/            # BC_PPO model.zip: 모방학습 + PPO 보완학습
│   ├── README.md
│   ├── run_model.py
│   ├── model/ / ppo_initialization/
│   ├── fixture/       # 당시 고정 도로 환경
│   └── evaluation/ / training_logs/ / provenance/
├── Environment/       # 두 패키지의 환경 안내
├── Training/          # 학습 색인
├── Evaluation/        # 평가 색인
└── Data_Collection/   # 수집 색인
```

| 패키지 | 내용 | 실행 안내 |
|---|---|---|
| [DQN](DQN/) | Double DQN, 초기 PPO P1~P9 실험, 최종 가중치·기록 | [DQN README](DQN/README.md) |
| [BC_PPO](BC_PPO/) | v118 round004, 가감속 고정·차로 변경 PPO, 고정 fixture | [BC+PPO README](BC_PPO/README.md) |
| [팀 최종 순수 PPO](../ppo_v1_parallel/) | SAFE13·17의 7:3 혼합 + B·Y·S | [최종 모델 평가](../Evaluation/README.md) |

첨부 ZIP과 대조한 원본 파일은 DQN 패키지 213개, BC+PPO 패키지 82개입니다. 기존 `K-Driver/`는 `DQN/`, `BC_PPO_13p33_share/`는 `BC_PPO/`로 이동했습니다. 소스·모델·환경·실험 기록의 내용은 보존하고 안내 문서와 경로만 정리했습니다.

각 모델의 코드와 설정은 독립적으로 유지한다. 네 가지 기능별 폴더는 코드 위치와 명령을 안내하는 색인이다. 명령은 Windows CMD 기준이며 저장소 루트에서 시작한다. 모델별 디렉터리로 이동한 후 실행한다.

## 재현 범위

BC+PPO는 과거 평가 환경(prefill OFF), P1~P9/DQN은 별도 환경을 사용하므로 결과를 동일 조건의 비교로 간주하지 않는다. BC+PPO 원본 시연 NPZ 일부가 유실되었고 이 묶음에도 시연 NPZ가 없으므로 당시 전체 학습의 완전한 재현은 불가능하다. 제공된 가중치로 주행 실행을 시도할 수 있다. 원본 코드는 변경하지 않았으며 이번 정리에서 SUMO 전체 주행 및 재학습은 검증하지 않았다.

발표 PDF와 Hugging Face 데이터셋 링크는 실제 공개 링크 확정 후 프로젝트 페이지에 추가한다.
