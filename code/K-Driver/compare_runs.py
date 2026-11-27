"""학습률·감마를 바꿔 돌린 PPO 실험들을 한 표와 그래프로 비교한다.

사용법:
    python compare_runs.py                 # results/ 아래 실험 전부
    python compare_runs.py A1 B2 C3        # 이름이 이걸로 시작하는 실험만

팀원들의 결과 폴더(results/<이름>_날짜_시각/)를 한 컴퓨터의 results/ 에 모아 두고 실행하면 된다.

출력:
    results/compare_summary.csv   실험별 요약 (최종·최고 성적)
    results/compare_curves.png    학습 곡선 (matplotlib 이 있을 때)
"""
import csv
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        try:
            out.append({k: float(v) for k, v in r.items() if v not in ("", None)})
        except ValueError:
            continue
    return out


def load_runs(prefixes):
    runs = []
    for d in sorted(os.listdir(RES)):
        p = os.path.join(RES, d)
        cfg_path = os.path.join(p, "run_config.json")
        if not os.path.isfile(cfg_path):
            continue                       # 이 비교 기능 이전의 실험은 건너뜀
        if prefixes and not any(d.startswith(x) for x in prefixes):
            continue
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        runs.append(dict(dir=d, cfg=cfg,
                         train=read_csv(os.path.join(p, "training_log.csv")),
                         ev=read_csv(os.path.join(p, "eval_log.csv"))))
    return runs


def summarize(run, last_k=3):
    ev, tr, cfg = run["ev"], run["train"], run["cfg"]
    row = {"실험": cfg.get("name") or run["dir"], "학습률": cfg["lr"], "감마": cfg["gamma"],
           "진행 스텝": int(tr[-1]["steps"]) if tr else 0, "목표 스텝": cfg["total_timesteps"]}
    if ev:
        tail = ev[-last_k:]
        mean = lambda k: sum(r.get(k, 0.0) for r in tail) / len(tail)
        best = max(ev, key=lambda r: (r.get("success_rate", 0), -r.get("collision_rate", 1)))
        row.update({
            "최근 완주율": round(mean("success_rate"), 3),
            "최근 충돌률": round(mean("collision_rate"), 3),
            "최근 시간초과율": round(mean("timeout_rate"), 3),
            "최근 신호위반율": round(mean("red_light_rate"), 3),
            "최근 평균속도(m/s)": round(mean("mean_speed"), 2),
            "최근 판당 보상": round(mean("ep_return"), 2),
            "최고 완주율": round(best.get("success_rate", 0), 3),
            "최고 완주율 시점(스텝)": int(best["steps"]),
        })
    if tr:
        row["최근 학습 보상"] = round(sum(r["ep_rew_mean"] for r in tr[-5:]) / len(tr[-5:]), 2)
    return row


def plot(runs, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import logging
        import warnings
        warnings.filterwarnings("ignore")
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    except ImportError:
        print("(matplotlib 이 없어 그래프는 건너뜀: pip install matplotlib)")
        return
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    kor = [f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic") if f in have]
    if kor:                                   # 한글 글꼴이 있으면 사용 (Windows: 맑은 고딕)
        plt.rcParams["font.family"] = kor[0]
    plt.rcParams["axes.unicode_minus"] = False
    panels = [("train", "ep_rew_mean", "학습 중 판당 보상"),
              ("ev", "success_rate", "완주율 (평가)"),
              ("ev", "collision_rate", "충돌률 (평가)"),
              ("ev", "mean_speed", "평균 속도 m/s (평가)")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (src, key, title) in zip(axes.flat, panels):
        for r in runs:
            data = [x for x in r[src] if key in x]
            if data:
                label = f"{r['cfg'].get('name') or r['dir']} (lr {r['cfg']['lr']:g}, γ {r['cfg']['gamma']:g})"
                ax.plot([x["steps"] for x in data], [x[key] for x in data], label=label)
        ax.set_title(title)
        ax.set_xlabel("학습 스텝")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"그래프: {path}")


def main():
    runs = load_runs(sys.argv[1:])
    if not runs:
        print("비교할 실험이 없습니다 (results/ 아래에 run_config.json 이 있는 폴더).")
        return
    rows = [summarize(r) for r in runs]
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    out = os.path.join(RES, "compare_summary.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(" | ".join(f"{k} {r[k]}" for k in cols if k in r))
    print(f"\n요약표: {out}")
    plot(runs, os.path.join(RES, "compare_curves.png"))


if __name__ == "__main__":
    main()
