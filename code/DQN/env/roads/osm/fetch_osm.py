"""양재 순환 코스 일대 OSM 도로(고속도로~2차로) 실측 데이터를 내려받아 JSON으로 저장."""
import json, os, time, urllib.parse, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
TYPES = "motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link"
PARTS = {"north": (37.4700, 127.0215, 37.4905, 127.0615), "south": (37.4550, 127.0215, 37.4705, 127.0615)}
for name, (s, w, n, e) in PARTS.items():
    out = os.path.join(HERE, f"loop_{name}.json")
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        print(name, "이미 있음"); continue
    q = f'[out:json][timeout:120];way["highway"~"^({TYPES})$"]({s},{w},{n},{e});out body geom;'
    for url in ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
                "https://maps.mail.ru/osm/tools/overpass/api/interpreter"):
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": "yangjae-loop"})
            body = urllib.request.urlopen(req, timeout=180).read()
            d = json.loads(body)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False)
            print(name, "저장", len(d["elements"]), "ways"); break
        except Exception as ex:
            print(name, url, "실패", ex); time.sleep(5)
