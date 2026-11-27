# K-Driver

> 추가 구현: [PPO 병렬 학습·평가·주행 데이터 수집 (`ppo_v1_parallel/`)](ppo_v1_parallel/README.md)

### 성균관대학교 자율주행인공지능제어 9조 프로젝트

SUMO 기반 도심 자율주행 환경을 구축하고, 다양한 일반 차량(HV)과의 상호작용 속에서 자율주행 차량(AV)의 가감속 및 차로 변경 판단을 학습하는 프로젝트입니다.

PPO, BC+PPO, DQN의 학습 과정과 실패 원인을 분석하고, 교통 혼잡 상황에서의 주행 안전성과 완주 성능을 개선하였습니다.


---

## 1. Project Overview

### 프로젝트 목표

실제 도로에서 발생하는 대기줄 끼어들기, 공사 구간 합류, 저속 차량 추월, 회전교차로 진입을 하나의 SUMO 주행 환경에 구현하였습니다.

단순한 목적지 도착뿐만 아니라 주변 차량과의 상호작용을 고려한 안전하고 효율적인 자율주행 정책을 개발하는 것을 목표로 합니다.

- AV (Autonomous Vehicle): 학습 및 평가 대상 자율주행 차량
- HV (Human-driven Vehicle): 일반 운전자의 주행 행동을 모사하는 차량

### Team Members

| 팀원 | 소속 | 주요 수행 분야 |
|---|---|---|
| 박제민 | 전자전기공학과 | SUMO 도로 구축 및 수정, PPO 개선 학습 |
| 김규환 | 화학공학과 | 전체 도로 구조 정비, 양재 IC 및 서초 IC 구현, 공사 구간 구현, 도로 연결 단절 및 지하차도·고가도로 간 차량 인식 문제 해결, P1~P9 보상 가중치 조정 및 PPO 학습, DQN 알고리즘 구현 및 보상 가중치 최적화, 최종 발표자료 제작 및 발표 진행 |
| 권도균 | 전자전기공학과 | Behavioral Cloning 및 PPO 보완학습 |

---

## 2. Simulation Environment

### 2.1 Road Modeling

양재 IC 인근 실제 도로 구조를 기반으로 SUMO 시뮬레이션 환경을 구축하였습니다.

**현대자동차 양재사옥 → 양재 IC → 서초 IC → 매봉역 → 서초종합체육관**

실제 도로의 주요 연결 구조를 반영하되, 긴 직선 구간을 압축하고 불필요한 골목길을 제외하여 차량 간 상호작용이 발생하는 핵심 도로를 중심으로 구성하였습니다.

| 항목 | 설정 |
|---|---|
| AV 주행 경로 | 9.4 km |
| 전체 차로 길이 | 97.8 km |
| 도로 조각 | 156개 |
| 신호 교차로 | 11곳 |
| 지퍼 합류 | 9곳 |
| 회전교차로 | 1곳 |
| 유턴 연결 | 66곳 |
| 제한속도 | 20~120 km/h |
| 공사 구간 | 언주로 남행 |
| 공사 구간 차로 감소 | 3차로 → 2차로 → 1차로 |

### 2.2 V2V Interaction Scenarios

단순한 직선 주행 환경에서 벗어나 AV와 HV 사이의 상호작용이 발생하도록 네 가지 핵심 시나리오를 구성하였습니다.

| Scenario | 주행 상황 | AV 판단 목표 |
|---|---|---|
| S1 | 대기줄 끼어들기 | 목표 차로 사전 확보 및 안전한 진입 |
| S2 | 합류 및 차로 감소 | 감속, 양보, 적절한 합류 간격 판단 |
| S3 | 1차로 저속 정속 차량 | 안전거리 유지 및 추월 판단 |
| S4 | 회전교차로 진입 | 회전 차량 우선권 및 진입 시점 판단 |

#### S1. Queue Cut-in

좌회전 및 출구 대기행렬에서 AV가 목표 차로에 진입해야 하는 상황입니다.

주변 차량의 속도와 간격을 고려하여 필요한 차로를 확보하고, 충분한 공간이 없는 경우 감속하거나 대기하도록 구성하였습니다.

#### S2. Merging and Lane Reduction

고속도로 합류 구간과 공사로 인한 차로 감소 상황입니다.

차로가 단절되는 위치와 목표 차로의 전후방 차량을 고려하여 안전하게 합류해야 합니다.

#### S3. Overtaking

저속 정속 주행 차량이 AV의 진행을 방해하는 상황입니다.

앞차와의 상대속도, 목표 차로 차량과의 간격을 고려하여 추종 또는 추월 여부를 판단합니다.

#### S4. Roundabout

회전교차로 내부 차량에 우선권을 부여하고, AV가 접근 차량의 거리 및 속도를 고려하여 진입 시점을 판단하도록 구성하였습니다.

### 2.3 HV Configuration

다양한 운전자 성향을 반영하기 위해 일반 차량과 상황별 차량의 주행 특성을 구분하였습니다.

| HV 유형 | 비율 | 주행 특성 |
|---|---:|---|
| 일반 차량 A | 25% | Krauss 기반 일반적인 주행 |
| 일반 차량 B | 25% | IDM 기반 부드러운 앞차 추종 |
| 저속 정속 차량 | 10% | 제한속도 대비 낮은 속도로 주행 |
| 양보하지 않는 차량 | 10% | 높은 희망 속도 및 작은 차간거리 |
| 늦게 끼어드는 차량 | 10% | 늦은 차로 변경 및 작은 간격 수용 |
| 회전교차로 통과 차량 | 10% | 회전교차로 통과 경로 이용 |
| 대기줄 차량 | 10% | 좌회전 및 출구 대기행렬 형성 |

### 2.4 AV Configuration

| Parameter | Value |
|---|---|
| Simulator | SUMO 1.27.1 |
| Control Interval | 0.5 s |
| Maximum Acceleration | 5.4 m/s² |
| Maximum Deceleration | -5.4 m/s² |
| Maximum Speed | 33.3 m/s |
| Lane Change Duration | 1 s |
| Default Maximum Steps | 2,000 |
| DQN Final Maximum Steps | 3,000 |
| Traffic Initialization | 모델별 교통 사전 형성 적용 |

차로 변경 안전 게이트를 적용하여 주변 차량과의 간격 및 접근 위험을 확인합니다.

에피소드 종료 조건은 목적지 도착, 충돌, 신호 위반, 시간 초과, 장시간 정지 등을 포함하며 세부 설정은 모델별로 상이합니다.

---

## 3. Observation, Action and Reward

### 3.1 Observation Space

자율주행 차량은 전체 교통 환경을 직접 관측하지 않고, 주변 차량 및 도로에 관한 부분 관측 정보를 활용합니다.

주요 관측 항목은 다음과 같습니다.

- 자차 속도
- 주변 5개 차로의 전후방 차량 거리
- 주변 차량과의 상대속도
- 차로 연결 가능 여부
- 주변 차량 밀도
- 신호등 상태 및 정지선까지의 거리
- 목적지 및 내비게이션 정보
- 공사 구간 및 차로 감소 지점까지의 거리
- 도로 제한속도
- 교차로 양보 및 진입 관련 정보

초기 기본 관측은 40차원으로 구성하였으며, 이후 학습 과정에서 발생하는 문제를 분석하여 모델별 관측 정보를 추가하였습니다.

최종 PPO 주행 데이터셋에는 43차원 관측 정보가 기록되어 있습니다.

### 3.2 Action Space

| Model | Acceleration | Lane Change |
|---|---|---|
| PPO | 연속 행동 | 오른쪽 / 유지 / 왼쪽 |
| BC + PPO | 연속 행동 | 오른쪽 / 유지 / 왼쪽 |
| DQN | 5단계 이산 행동 | 오른쪽 / 유지 / 왼쪽 |

PPO 및 BC 기반 모델의 가감속은 정규화된 연속 행동값을 실제 가감속으로 변환합니다.

DQN의 경우 가감속을 다음 다섯 단계로 이산화하였습니다.

- 강한 감속
- 약한 감속
- 유지
- 약한 가속
- 강한 가속

가감속 5단계와 차로 선택 3개를 조합하여 총 15개의 이산 행동을 구성하였습니다.

### 3.3 Reward Design

안전성과 주행 효율을 동시에 고려하여 보상함수를 설계하였습니다.

| Reward Component | 목적 |
|---|---|
| Speed Reward | 효율적인 주행 유도 |
| Progress Reward | 목적지 방향으로의 주행 유도 |
| Arrival Reward | 최종 목적지 도착 유도 |
| Collision Penalty | 차량 충돌 방지 |
| TTC Penalty | 위험한 차량 접근 방지 |
| Lane Change Penalty | 불필요한 차로 변경 억제 |
| Route Deviation Penalty | 경로 이탈 방지 |
| Red Light Penalty | 신호 위반 방지 |
| Speed Limit Penalty | 제한속도 준수 |
| Stopping Penalty | 불필요한 장시간 정지 방지 |

학습 과정에서 발생한 충돌, 과도한 차로 변경, 장시간 정지 및 경로 이탈 문제를 분석하여 보상함수를 단계적으로 개선하였습니다.

---

## 4. AV Training and Problem Solving

본 프로젝트에서는 세 가지 자율주행 학습방식을 개발하고 평가하였습니다.

| Model | Algorithm | 특징 |
|---|---|---|
| Model A | BC + PPO | 전문가 주행 모방 후 강화학습을 통한 보완 |
| Model B | Double DQN | 이산 행동 및 경험 재사용 기반 강화학습 |
| Model C | PPO | 모방학습 없는 강화학습 기반 정책 개선 |

### 4.1 Baseline PPO: P1~P9

초기 PPO 학습에서는 학습률과 할인율을 변경하며 자율주행 성능을 비교하였습니다.

#### Hyperparameter Search

| Parameter | Candidates |
|---|---|
| Learning Rate | 0.001, 0.0003, 0.0001 |
| Gamma | 0.95, 0.99, 0.995 |
| Training Steps | 약 1,120,000 |
| Evaluation Interval | 50,000 steps |

#### Training Progress

| Version | 주요 개선 사항 | 주요 결과 |
|---|---|---|
| P1~P3 | 하이퍼파라미터 비교, 정지 감점, 유턴 연결 | 완주 0%, 과도한 정지 및 차로 변경 |
| P4 | 과속 및 경로 이탈 감점, 유턴 연결 30곳 | 완주 38%, 충돌 34% |
| P5~P7 | 유턴 연결 66곳, IDM 도입, 진행 거리 보상, 차로 변경 감점 | 대기줄 진입 및 경로 이탈 문제 |
| P8 | 도로 길이 보정, 경로 이탈 허용 횟수 조정 | 완주 54%, 충돌 40% |
| P9 | 충돌 및 TTC 감점 강화 | 완주 0%, 시간 초과 발생 |

P3의 충돌 사례를 분석한 결과, 전체 충돌의 약 75%가 충돌 직전 2초 이내의 차로 변경과 관련되어 있었습니다.

P9에서는 충돌 감점을 -5에서 -30으로 증가시켰지만, 차량이 위험을 회피하기 위해 지나치게 정지하는 문제가 발생하였습니다.

이를 통해 단순히 충돌 감점을 증가시키는 방식은 안전성과 주행 효율을 동시에 확보하기 어렵다는 것을 확인하였습니다.

---

## 5. Model A: Behavioral Cloning + PPO

### 5.1 Behavioral Cloning

SUMO 기반 교사 정책에서 수집한 주행 데이터를 활용하여 AV가 전문가의 행동을 모방하도록 학습하였습니다.

기본적인 가감속 및 차로 변경 능력을 먼저 확보함으로써 초기 강화학습의 탐험 부담을 줄이는 것을 목표로 하였습니다.

### 5.2 Failure Recovery

BC 모델의 평가 과정에서 발생한 실패 시드를 분석하고, 충돌 및 주행 실패 상황에 대한 추가 시연 데이터를 수집하였습니다.

이를 기반으로 DAgger와 유사한 방식의 보완학습을 진행하였습니다.

### 5.3 PPO Fine-tuning

BC를 통해 확보한 가감속 정책을 유지하고, PPO를 활용하여 차로 변경 판단을 추가적으로 학습하였습니다.

학습 과정은 다음과 같습니다.

1. SUMO 교사 주행 데이터 수집
2. Behavioral Cloning 학습
3. 실패 시드 및 충돌 상황 분석
4. 실패 상황 추가 데이터 수집
5. BC 보완학습
6. PPO 기반 차로 변경 학습
7. 최종 주행 평가

### 5.4 Evaluation Results

| Model | Completion Rate | Collision Rate |
|---|---:|---:|
| Baseline BC | 83.33% | 15.00% |
| 차로 변경 추가 BC | 80.33% | 16.33% |
| 가감속 및 차로 변경 추가 BC | 79.33% | 17.33% |
| 가감속 고정 + 차로 변경 PPO | **85.00%** | **13.33%** |

PPO 보완학습을 통해 기본 BC 대비 완주율을 개선하고 충돌률을 감소시켰습니다.

다만 초기 교통이 형성되지 않은 평가 환경이라는 한계가 있으며, 충돌률 5% 미만의 목표는 달성하지 못하였습니다.

---

## 6. Model B: Double DQN

### 6.1 Motivation

PPO의 경우 현재 정책으로 수집한 경험을 활용하여 행동을 선택하는 정책을 학습합니다.

반면 DQN은 현재 상태에서 각 행동을 수행했을 때 기대되는 미래 누적 보상을 Q-value로 학습하고, 평가 시 가장 높은 Q-value를 갖는 행동을 선택합니다.

DQN을 선정한 주요 이유는 다음과 같습니다.

- 차로 변경은 오른쪽, 유지, 왼쪽 중 하나를 선택하는 이산적인 의사결정 문제입니다.
- Experience Replay를 통해 과거의 주행 경험을 반복적으로 학습할 수 있습니다.
- 상대적으로 드물게 발생하는 충돌 및 끼어들기 실패 경험을 재사용할 수 있습니다.
- 동일한 보상 조건에서 PPO P9는 완주율 0%, 초기 DQN D1은 완주율 44%를 기록하였습니다.

이를 바탕으로 DQN 기반 자율주행 코드를 별도로 구성하였습니다.

### 6.2 Network Architecture

기본 DQN 신경망은 다음과 같이 구성하였습니다.

```text
Input Layer (40)
       ↓
Hidden Layer (256)
       ↓
     ReLU
       ↓
Hidden Layer (256)
       ↓
     ReLU
       ↓
Output Layer (15)
       ↓
   Action Selection
```

출력층의 15개 값은 각각의 가감속 및 차로 선택 조합에 대한 Q-value를 나타냅니다.

Double DQN 구조를 적용하여 온라인 네트워크에서 다음 행동을 선택하고, 타깃 네트워크에서 해당 행동의 가치를 평가하도록 구성하였습니다.

이를 통해 일반 DQN에서 발생할 수 있는 Q-value 과대평가 문제를 완화하였습니다.

### 6.3 Hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | Double DQN |
| Action Space | 15 |
| Hidden Layers | 256, 256 |
| Activation Function | ReLU |
| Replay Buffer | 200,000 |
| Batch Size | 128 |
| Learning Rate | 0.0001 |
| Discount Factor | 0.95 |
| Loss Function | Huber Loss |
| Training Frequency | 4 steps |
| Target Network Update | 2,000 updates |
| Exploration | Epsilon-greedy |
| Initial Epsilon | 1.0 |
| Final Epsilon | 0.05 |
| Epsilon Decay | 300,000 steps |
| Parallel Environments | 4 |
| Simulator Interface | libsumo |
| Final Checkpoint | 870,000 steps |

### 6.4 DQN Improvement Process

#### D1: Initial DQN

초기 PPO와 동일한 보상 조건을 기반으로 DQN을 적용하였습니다.

- 완주율: 44%
- 충돌률: 36%

완주 성능은 개선되었지만 불필요한 차로 변경으로 인한 충돌이 지속적으로 발생하였습니다.

#### D2: Lane Change Penalty

불필요한 차로 변경을 억제하기 위해 차로 변경 감점을 강화하였습니다.

- 기존 차로 변경 감점: -0.02
- 변경 차로 변경 감점: -0.1
- 완주율: 65%
- 충돌률: 19%

차로 변경 빈도를 억제함으로써 초기 DQN 대비 충돌률을 감소시켰습니다.

#### D3: Side Vehicle Detection

기존 관측 방식에서는 서로 다른 도로 조각이나 교차로 내부에 위치한 인접 차량을 제대로 감지하지 못하는 문제가 발생하였습니다.

이를 해결하기 위해 실제 차량 위치를 기반으로 전후방 12m 범위의 인접 차량을 추가 감지하도록 관측 구조를 개선하였습니다.

교통이 형성된 동일 평가 환경에서 충돌률은 27%에서 17%로 감소하였습니다.

#### D4: Traffic Warm-up

초기 교통이 형성되지 않은 상태에서 학습한 모델은 혼잡한 실제 교통 상황에 충분히 대응하지 못하였습니다.

이를 해결하기 위해 AV 출발 전에 100~300초 동안 교통을 사전에 형성하도록 학습 환경을 개선하였습니다.

이를 통해 다음과 같은 상황을 보다 다양하게 학습하도록 구성하였습니다.

- 이미 형성된 교차로 대기행렬
- 혼잡한 합류 구간
- 주변 차량의 연속적인 차로 변경
- 다양한 초기 신호 상태
- 회전교차로 내 차량 간 상호작용

### 6.5 Final Evaluation

최종 D4 모델을 교통이 사전에 형성된 환경에서 100회 평가하였습니다.

| Result | Rate |
|---|---:|
| Completion | **58%** |
| Collision | **7%** |
| Red Light Violation | 28% |
| Long Stop | 5% |
| No Route | 2% |

| Driving Metric | Value |
|---|---:|
| 평균 완주 시간 | 771 s |
| 평균 완주 주행 속도 | 44.0 km/h |
| 최종 학습 체크포인트 | 870,000 steps |

#### Collision Rate Improvement

| Version | Collision Rate |
|---|---:|
| D1 | 36% |
| D2 | 19% |
| D4 | 7% |

DQN의 학습 및 관측 구조를 개선하면서 충돌률을 감소시킬 수 있었습니다.

다만 모델별 교통 초기화 조건에 차이가 있으므로 모든 수치를 동일 조건의 직접적인 성능 비교로 해석할 수는 없습니다.

최종적으로 신호 위반 및 대기줄 진입 판단은 추가적인 개선이 필요한 부분으로 확인하였습니다.

---

## 7. Model C: PPO Update

모방학습을 사용하지 않고 PPO를 기반으로 자율주행 정책을 직접 학습하는 방식입니다.

초기 PPO 학습 과정에서 발생한 충돌, 불필요한 정지, 차로 변경 문제 및 신호 위반을 분석하여 추가적인 개선 학습을 수행하였습니다.

주요 개선 목표는 다음과 같습니다.

- 주변 차량과의 충돌 방지
- 신호 교차로에서의 안전한 정차
- 회전교차로 및 비보호 교차로 대응
- 불필요한 차로 변경 억제
- 정차 시 적절한 앞차 간격 확보
- 안전성과 주행 효율 사이의 균형 확보

### Final PPO Evaluation

최종 모델의 정량적인 평가 결과 및 체크포인트는 확정 후 추가할 예정입니다.

---

## 8. Final Model Comparison

| Algorithm | Completion Rate | Collision Rate | Evaluation Condition |
|---|---:|---:|---|
| Baseline BC | 83.33% | 15.00% | 기존 BC 평가 환경 |
| BC + PPO | 85.00% | 13.33% | 초기 교통 형성 없음 |
| Double DQN D4 | 58.00% | 7.00% | 교통 사전 형성 |
| PPO Update | TBD | TBD | 최종 평가 조건 추가 예정 |

각 학습방식은 서로 다른 관측 및 보상 설정과 교통 초기화 조건을 적용하였습니다.

따라서 단순히 완주율만 비교하기보다는 충돌률, 신호 위반률, 교통 혼잡도 및 학습 구조를 함께 고려해야 합니다.

---

## 9. Final Dataset

### 9.1 Dataset Overview

최종 PPO 학습 과정에서 발생한 실제 주행 데이터를 수집하였습니다.

초기 데이터셋은 기본 주행 정보 및 상황별 정보를 포함한 38개 칼럼으로 구성하였으며, 이후 학습 환경과 평가 항목을 확장하면서 최종 45개 칼럼으로 구성하였습니다.

| Dataset Information | Value |
|---|---:|
| Total Rows | 10,417,849 |
| Episodes | 6,314 |
| Columns | 45 |
| Observation Dimension | 43 |
| Sampling Interval | 0.5 s |

### 9.2 Dataset Structure

#### Basic Driving Data

| Column | Description |
|---|---|
| episode | 주행 에피소드 번호 |
| t | 에피소드 내 스텝 번호 |
| sim_time_s | 시뮬레이션 시간 |
| observation | AV의 부분 관측 정보 |
| action_raw | 정책의 행동 정보 |
| reward | 스텝별 보상 |
| state | 시뮬레이션 차량 상태 |
| terminated | 에피소드 종료 여부 |
| truncated | 시간 제한 종료 여부 |
| event.collided | 충돌 여부 |
| event.arrived | 목적지 도착 여부 |

#### S1. Queue Cut-in

| Column | Description |
|---|---|
| s3.target_distance_m | 목표 지점까지 남은 거리 |
| s3.lane_changes_remaining | 목표 차로까지 필요한 차로 변경 횟수 |
| s3.queue_vehicle_count | 대기 차량 수 |
| s3.target_front_gap_m | 목표 차로 앞차 간격 |
| s3.target_rear_gap_m | 목표 차로 뒤차 간격 |
| s3.cutin_failed | 안전 게이트에 의한 차로 변경 차단 여부 |

#### S2. Merging and Lane Reduction

| Column | Description |
|---|---|
| s2.lane_end_distance_m | 차로 단절 지점까지의 거리 |
| s2.target_front_gap_m | 합류 목표 차로 앞차 간격 |
| s2.target_rear_gap_m | 합류 목표 차로 뒤차 간격 |
| s2.target_rear_relative_speed_mps | 목표 차로 뒤차 상대속도 |
| s2.ego_acceleration_mps2 | AV의 실제 가속도 |
| s2.cutin_failed | 안전 게이트에 의한 차로 변경 차단 여부 |

#### S3. Overtaking

| Column | Description |
|---|---|
| s1.leader_gap_m | 현재 차로 앞차 간격 |
| s1.leader_relative_speed_mps | 앞차와의 상대속도 |
| s1.target_front_gap_m | 추월 목표 차로 앞차 간격 |
| s1.target_rear_gap_m | 추월 목표 차로 뒤차 간격 |
| s1.target_rear_relative_speed_mps | 목표 차로 뒤차 상대속도 |
| s1.cutin_failed | 안전 게이트에 의한 차로 변경 차단 여부 |

#### S4. Roundabout

| Column | Description |
|---|---|
| s4.entry_distance_m | 회전교차로 진입 지점까지의 거리 |
| s4.circulating_distance_m | 접근하는 회전 차량의 거리 |
| s4.circulating_speed_mps | 접근하는 회전 차량의 속도 |

위 표는 최종 45개 칼럼 중 주요 분석 항목을 정리한 것입니다.

발표 자료의 상황 번호와 기존 데이터셋의 칼럼 접두어는 일부 다르게 구성되어 있습니다.

- S1: s3.*
- S2: s2.*
- S3: s1.*
- S4: s4.*

### 9.3 Scenario Data Analysis

수집된 실제 주행 데이터를 기반으로 상황별 AV의 판단 과정을 분석하였습니다.

| Scenario | Episode | 분석 결과 |
|---|---:|---|
| 대기줄 끼어들기 | 598 | 좌회전 80m 전 필요한 차로 변경 3회에서 3스텝 후 0회로 감소 |
| 공사 구간 합류 | 598 | 합류 간격 0m에서 차로 변경 차단 후 감속하여 14.6m 간격 확보 |
| 저속 차량 추월 | 77 | 목표 차로 앞차 간격 3.2m에서 차단 후 9.2m 간격 확보 및 차로 변경 |
| 회전교차로 진입 | 598 | 접근하는 회전 차량을 고려하여 0.24m/s까지 감속 |

위 데이터는 AV가 주변 차량의 거리와 속도를 고려하여 주행하는 과정의 개별 사례를 나타냅니다.

차로 변경 성공률이나 상황별 전체 성공률을 의미하는 것은 아닙니다.

---

## 10. Code and Reproduction

최종 SUMO 환경 및 학습된 AV를 재현할 수 있도록 모델별 학습 코드, 평가 코드, 모델 가중치 및 환경 설정을 제공합니다.

### Repository Structure

```text
Team-9_K-driver/
│
├── README.md
│
└── code/
    ├── README.md
    │
    ├── Environment/
    ├── Training/
    ├── Evaluation/
    ├── Data_Collection/
    │
    ├── BC_PPO_13p33_share/
    │   └── BC + PPO Model
    │
    └── K-Driver/
        └── P1~P9 and DQN Model
```

별도의 순수 PPO 최종 모델은 추후 추가할 예정입니다.

### Code Description

| Category | Description |
|---|---|
| [Environment](code/Environment/README.md) | SUMO 도로 구조, AV 및 HV 환경, 관측 및 행동 설정 |
| [Training](code/Training/README.md) | BC, PPO, DQN 학습 코드 |
| [Evaluation](code/Evaluation/README.md) | 학습 모델 평가 및 SUMO 주행 재현 |
| [Data Collection](code/Data_Collection/README.md) | 주행 데이터 수집 및 데이터셋 생성 |

### Model Source Code

| Model | Source Code |
|---|---|
| BC + PPO | [BC_PPO_13p33_share](code/BC_PPO_13p33_share/) |
| Initial PPO and DQN | [K-Driver](code/K-Driver/) |
| Final PPO | 추가 예정 |

### 10.1 Environment Setup

프로젝트 저장소를 복제합니다.

```cmd
git clone https://github.com/gofltms33-maker/Team-9_K-driver.git
cd Team-9_K-driver
```

기본 실행 환경은 다음과 같습니다.

- Python
- PyTorch
- SUMO 1.27.1
- NumPy
- TraCI
- libsumo

모델별 세부 라이브러리 및 실행 환경은 Environment 안내와 각 모델의 README를 참고합니다.

### 10.2 BC + PPO Evaluation

```cmd
cd code\BC_PPO_13p33_share
python run_model.py --seed 1024000
```

제공된 학습 모델을 이용하여 SUMO 환경에서 AV의 주행을 재현할 수 있습니다.

일부 원본 교사 시연 데이터는 포함되어 있지 않으므로 당시 BC 학습 과정 전체를 동일하게 재현하는 데에는 제한이 있습니다.

### 10.3 DQN Evaluation

저장소 루트에서 다음 명령어를 실행합니다.

```cmd
cd code\K-Driver
python test.py models\DQN_final_C4_870k.pt --algorithm dqn --side-scan on --warmup 200 --warmup-jitter 100 --episodes 1
```

100회 평가를 진행하는 경우 다음 명령어를 실행합니다.

```cmd
python test.py models\DQN_final_C4_870k.pt --algorithm dqn --side-scan on --warmup 200 --warmup-jitter 100 --nogui --episodes 100
```

### 10.4 Training and Data Collection

모델별 학습 및 데이터 수집 방법은 아래 문서를 참고합니다.

- [Environment Setup](code/Environment/README.md)
- [Model Training](code/Training/README.md)
- [Model Evaluation](code/Evaluation/README.md)
- [Data Collection](code/Data_Collection/README.md)

각 모델의 관측 차원 및 환경 설정이 서로 다르므로 모델별 소스 코드와 실행 환경을 독립적으로 유지하였습니다.

---

## 11. Conclusion

본 프로젝트에서는 실제 양재 IC 인근 도로를 기반으로 차량 간 상호작용이 발생하는 SUMO 시뮬레이션 환경을 구축하였습니다.

PPO, Behavioral Cloning 기반 PPO, Double DQN 등 다양한 학습방식을 적용하고, 각 학습 과정에서 발생한 실패 원인을 분석하여 자율주행 성능을 개선하였습니다.

주요 성과는 다음과 같습니다.

- 실제 도로 구조를 반영한 9.4km 자율주행 경로 구축
- 다양한 HV 주행 성향 및 차량 상호작용 구현
- 세 가지 자율주행 학습방식 개발 및 비교
- BC + PPO 기반 완주율 85% 달성
- 교통이 형성된 환경에서 최종 DQN 충돌률 7% 달성
- 최종 45개 칼럼, 약 1,042만 행 규모의 주행 데이터셋 구축
- 실제 주행 데이터를 활용한 상황별 AV 판단 분석

학습 과정에서 안전성만을 지나치게 강조하면 장시간 정지 및 완주 실패가 발생할 수 있음을 확인하였습니다.

또한 차량의 안전한 주행을 위해서는 보상함수뿐만 아니라 적절한 관측 정보, 교통 환경 구성 및 학습 알고리즘의 선택이 중요하다는 것을 확인하였습니다.

향후 신호 위반 감소, 혼잡한 대기행렬 진입 및 다양한 교통 조건에서의 일반화 성능 개선이 필요합니다.

---

## 12. Project Materials

| Materials | Link |
|---|---|
| Project Code | [GitHub Repository](https://github.com/gofltms33-maker/Team-9_K-driver) |
| Code Documentation | [Code & Reproduction](code/README.md) |
| Presentation PDF | 추가 예정 |
| Dataset | Hugging Face 링크 추가 예정 |
| Project Page | 추가 예정 |
| Driving Demonstration | 추가 예정 |

---

## 13. PPO Parallel Implementation

로컬에서 개발한 PPO 병렬 학습·평가·데이터 수집 구현을 [`ppo_v1_parallel/`](ppo_v1_parallel/)에 추가하였습니다. 이 구현은 독립된 실행 폴더로 구성되어 있으며, 관련 설명은 [별도 README](ppo_v1_parallel/README.md)를 참고합니다.

| 구성 | 내용 |
|---|---|
| PPO 학습 | 연속 가감속 + 이산 차로 변경 hybrid 정책, 여러 SUMO 환경의 병렬 rollout, 기존 모델에서 추가 학습 |
| 주행 환경 | 양재IC·서초IC 주변 순환 도로, 교차로·공사·합류 구간 및 도로 생성용 OSM 입력 데이터 |
| 평가 | GUI 주행 재생, 화면 없는 평가, 고정 시드 및 학습 당시 설정 복원 옵션 |
| 데이터 수집 | SAFE13 53,248스텝 모델을 기본 사용, part·작업자별 시드 분리, 압축 CSV 기록 |
| 실험 기록 | SAFE5·SAFE11·SAFE13·SAFE17 모델·설정·로그·평가 경로, MIX13x17 가중치 평균 모델 |

저장소 루트에서 다음과 같이 실행합니다. SUMO 설치 및 `SUMO_HOME` 설정은 [실행 준비](ppo_v1_parallel/README.md#실행-준비)를 참고합니다.

```powershell
cd ppo_v1_parallel
python -m pip install -r requirements.txt

# 병렬 학습
python train.py --name PPO_parallel --n-envs 8 --timesteps 200000

# 데이터 수집 기본 모델 평가
python test.py results/SAFE13_20261125_185902/eval_models/model_00053248.pt --yield-cut on --nogui --episodes 10 --seed-start 1000

# 주행 데이터 수집
python collect_data.py --part 1 --rows 3334000 --workers 8
```

소스와 실험 파일은 원본 내용을 유지하였으며, 로컬 대용량 수집 데이터와 자동 생성 SUMO 파일, TensorBoard 출력 및 Python 캐시는 포함하지 않았습니다. 복사 파일 목록과 SHA-256은 [`source_manifest.json`](ppo_v1_parallel/source_manifest.json)에 기록하였습니다. 저장된 성능 지표는 각 실험 당시의 결과이며 이번 정리 작업에서 SUMO 주행을 새로 실행하지는 않았습니다.

---

**K-Driver — Team 9**

Artificial Intelligence and Control for Autonomous Driving

Sungkyunkwan University (SKKU), 2026
