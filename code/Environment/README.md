# Environment

## 설치

SUMO와 Python(사용할 PyTorch 버전이 지원하는 버전)을 설치한다. BC+PPO 원본 설명서의 평가 SUMO 버전은 1.27.1이다. Windows CMD에서 SUMO 설치 경로에 맞춰 환경변수를 설정한다(예시는 기본 설치 경로이며 필요시 수정).

```cmd
set "SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo"
set "PATH=%SUMO_HOME%\bin;%PATH%"
sumo --version
```

BC+PPO: 저장소 루트에서 `cd code\BC_PPO_13p33_share` 후 `python -m pip install -r requirements.txt`. 원본 고정 버전의 설치 가능 여부를 확인한다. SUMO의 Python 도구도 필요하다.

P1~P9/DQN: 저장소 루트에서 `cd code\K-Driver` 후 `python -m pip install torch numpy gymnasium matplotlib tensorboard traci sumolib`. GUI 없는 libsumo 실행을 원하면 호환되는 libsumo를 별도로 설치한다. 이 의존성 목록은 소스 import를 바탕으로 정리했으며 버전 호환성은 검증하지 않았다.

## 도로 및 환경 코드

| 파일 | BC+PPO | P1~P9/DQN |
|---|---|---|
| 환경 구현 | [sumo_env.py](../BC_PPO_13p33_share/env/sumo_env.py) | [sumo_env.py](../K-Driver/env/sumo_env.py) |
| 도로 생성 | [road_builder.py](../BC_PPO_13p33_share/env/road_builder.py) | [road_builder.py](../K-Driver/env/road_builder.py) |
| 도로 설정 | [road_config.py](../BC_PPO_13p33_share/env/road_config.py) | [road_config.py](../K-Driver/env/road_config.py) |
| 관측·행동·보상 | [mdp_config.py](../BC_PPO_13p33_share/env/mdp_config.py) | [mdp_config.py](../K-Driver/env/mdp_config.py) |
| SUMO 파일 | [fixture/sumo](../BC_PPO_13p33_share/fixture/sumo/) | [env/sumo](../K-Driver/env/sumo/) |

P1~P9/DQN 도로 확인: 해당 모델 폴더에서 `python view_road.py --play`. BC+PPO는 고정 fixture를 사용한다.
