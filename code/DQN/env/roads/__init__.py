"""순환 코스 도로 부품. env/road_builder.py 가 ROAD["layout"] 에 따라 불러 쓴다.

  loop_full.py  전체 조립 (아래 부품을 순서대로 붙임)
  yangjae_ic.py, yangjae_fix.py, yangjae_data.py  양재IC
  seocho_ic.py  서초IC
  gyeongbu.py   양재IC <-> 서초IC 경부고속도로
  arterials.py  남부순환로, 언주로, 양재대로와 사거리
  connectors.py 가운데 연결 도로
  hq.py         현대차 본사, 정문
  wonji.py      원지동 회전교차로, 지하차도, 체육관
  construction.py 공사 구간
  taper.py      차로 수 바뀌는 곳 다듬기 (화면용)
  osm_corridor.py, osm/  주변 실측 도로
  traffic_mix.py 일반 차량 구성
"""
