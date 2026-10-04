"""フェーズ3：特徴量の集計（目的変数との関係は見ない。DECISIONS.md 2026-10-04）。

- 対象：各週の予測日（前の週の最終営業日）のユニバースの銘柄。期間は特徴量がそろう 2017-01 以降〜ホールドアウトの前
- 値のある割合（全体・年ごと・最小の週）、分布の分位点、予測日ごとの特徴量どうしの順位相関（スピアマン）の平均

出力：reports/phase3_features.json、reports/phase3_corr.png
実行：python -m src.analysis.phase3_features
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd

from src.backtest.market import JST, load_market
from src.backtest.target import make_weeks
from src.backtest.universe import universe_mask
from src.config import ROOT, load_config
from src.features.build import load_features
from src.features.registry import FEATURE_SETS, get

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

START = "2017-01-01"
FEATURE_SET = "all_v1"


def main() -> None:
    cfg = load_config()
    m = load_market(cfg)
    u = universe_mask(m, cfg)
    names = FEATURE_SETS[FEATURE_SET]
    F = load_features(names, cfg, m)
    preds = np.array([w.pred for w in make_weeks(m.dates) if w.pred >= 0 and m.dates[w.pred] >= START])
    years = pd.Series(m.dates[preds]).str[:4].to_numpy()
    res: dict = {"meta": {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                                       cwd=ROOT).stdout.strip(),
                          "run_at_jst": datetime.now(JST).isoformat(timespec="seconds"),
                          "data_latest_fetch_jst": m.meta.get("data_latest_fetch_jst"),
                          "market_cache_version": m.meta.get("cache_version"),
                          "feature_set": FEATURE_SET, "period": [str(m.dates[preds[0]]), str(m.dates[preds[-1]])],
                          "pred_dates": int(len(preds)),
                          "universe_size_median": float(np.median(u[preds].sum(1)))},
                 "features": {}}
    for n in names:
        a = F[n]
        cov = np.array([np.isfinite(a[p][u[p]]).mean() for p in preds])
        v = a[preds][u[preds]]
        v = v[np.isfinite(v)]
        res["features"][n] = {
            "group": get(n).group, "description": get(n).description,
            "coverage_mean": float(cov.mean()), "coverage_min": float(cov.min()),
            "coverage_min_date": str(m.dates[preds[cov.argmin()]]),
            "coverage_by_year": pd.Series(cov).groupby(years).mean().round(3).to_dict(),
            "quantiles": {str(q): float(np.quantile(v, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
        }
    # 予測日ごとの順位相関の平均
    corr_sum = np.zeros((len(names), len(names)))
    k = 0
    for p in preds:
        df = pd.DataFrame({n: F[n][p][u[p]] for n in names})
        c = df.rank().corr().to_numpy()
        if np.isfinite(c).all():
            corr_sum += c
            k += 1
    corr = corr_sum / k
    res["rank_corr_mean"] = {"dates_used": k, "names": names, "matrix": np.round(corr, 3).tolist()}
    pairs = [(names[i], names[j], corr[i, j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    res["rank_corr_high_pairs"] = [{"a": a, "b": b, "corr": round(float(c), 3)}
                                   for a, b, c in sorted(pairs, key=lambda x: -abs(x[2])) if abs(c) >= 0.5]
    out = ROOT / "reports" / "phase3_features.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"特徴量どうしの順位相関（予測日ごとのユニバース内、{k}日の平均）")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "phase3_corr.png", dpi=110)
    plt.close(fig)
    print(json.dumps({n: {"cov": round(d["coverage_mean"], 3), "min": round(d["coverage_min"], 3)}
                      for n, d in res["features"].items()}, ensure_ascii=False))
    print(json.dumps(res["rank_corr_high_pairs"], ensure_ascii=False))


if __name__ == "__main__":
    main()
