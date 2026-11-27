# Data Collection

| 모델 | 데이터 수집 코드 |
|---|---|
| BC + PPO | [collect_audited_course_data.py](../BC_PPO/collect_audited_course_data.py), [train_bc_ppo.py](../BC_PPO/train_bc_ppo.py) |
| P1\~P9 / DQN | [collect_pomdp_data.py](../DQN/collect_pomdp_data.py), [step_recorder.py](../DQN/env/step_recorder.py) |

P1\~P9/DQN 모델 폴더에서 수집 코드 옵션을 확인한다.

```cmd
python collect_pomdp_data.py --help
```

수집 정책 옵션은 random, keep-lane, ppo, bc이며 현재 수집 코드에 DQN 정책 옵션은 없다. 정책과 모델 경로를 지정해 데이터를 수집할 수 있다.

BC+PPO의 [데이터 보유 현황](../BC_PPO/provenance/training_data_availability.json)과 [원본 설명](../BC_PPO/READ_ME_KO.txt)을 참고한다. 원본 시연 NPZ는 제공되지 않았다. Hugging Face 제출 데이터는 공개 URL이 확정되면 프로젝트 페이지에 연결한다.
