# Evaluation

## BC + PPO

저장소 루트에서:

```cmd
cd code\BC_PPO
python run_model.py --seed 1024000
python run_model.py --nogui --seed 1024000 --episodes 2
```

[가중치](../BC_PPO/model/model.pt) · [평가 기록](../BC_PPO/evaluation/) · [평가 코드](../BC_PPO/evaluate_imitation.py) · [원본 설명](../BC_PPO/READ_ME_KO.txt)

과거 개발 평가 300회: 완주 255회(85%), 충돌 40회(13.33%). 별도 평가 100회: 완주 86회, 충돌 11회. 출처는 원본 설명서이며 본 정리에서 재측정하지 않았다.

## P1\~P9 / DQN

새 CMD 또는 저장소 루트에서:

```cmd
cd code\DQN
python test.py models\DQN_final_C4_870k.pt --algorithm dqn --side-scan on --warmup 200 --warmup-jitter 100 --episodes 1
python test.py models\DQN_final_C4_870k.pt --algorithm dqn --side-scan on --warmup 200 --warmup-jitter 100 --nogui --episodes 100
```

C4 저장 설정에 맞춰 교통 채움 200±100초와 side-scan 활성화를 명시했다. 이 명령은 실행 예시이며 과거 평가의 모든 조건을 동일하게 복원한 것은 아니다. libsumo 문제가 있으면 `--no-libsumo`를 추가한다.

[PPO 비교 가중치](../DQN/models/PPO_v1_best_720k.pt) · [DQN 최종 가중치](../DQN/models/DQN_final_C4_870k.pt) · [test.py](../DQN/test.py) · [실험 기록](../DQN/experiments/)

각 체크포인트의 관측 차원과 실험 설정이 현재 환경과 일치하는지 확인한 뒤 실행한다.
