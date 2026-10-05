"""運用時（Light プラン：5年前までのデータ）に、直近4年のローリングの学習を同じ条件で再現できるかの確認（ユーザーの指示4、2026-10-05）。

評価の月 M の再学習を、その月の最初の予測日 D の引け後に行うとする。Light では D の5年前（暦）の日付以降のデータしか取れない
（株価・銘柄一覧・決算短信・決算発表予定日。PLAN.md 1.2）。そこで、D の5年前以降のデータだけで Market・ユニバース・特徴量を
作り直し、全期間のデータ（2016-10-05 以降）で作った値と、学習に使う起点（D の4年前〜D の6営業日前）で比べる。

- 比べるのは、全期間のデータでのユニバースの行（学習に使う行）。ユニバースの違いも数える
- 特徴量ごとに：全期間では値があるのに Light では欠ける行、両方に値があるが値が違う行、Light だけに値がある行
- 欠ける行は、起点の月ごとにも数える（学習期間の初めに偏るかを見る）
- 目的変数・リターン・成績は計算しない（評価役が計画を確認する前に結果を見ないため）

月の選び方：2021-09 以前の月は、D の5年前が手元のデータの開始日（2016-10-05）より前になり、全期間と同じデータになるため比べない
（2021-10 で確認して、すべて一致）。2022-10 以降の4つの月を比べる

実行：python -m src.analysis.phase4_light_check --months 2022-10 2023-10 2024-10 2025-09
出力：reports/phase4_light_check.json
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import pandas as pd

from src.backtest.market import build_market, load_market
from src.backtest.universe import universe_mask
from src.config import ROOT, load_config
from src.features.build import compute, feature_dir
from src.features.data import load_feature_data
from src.features.registry import get, resolve
from src.models.dataset import walk_forward_splits

LIGHT_YEARS = 5


def full_feature(cfg: dict, m, name: str) -> np.ndarray:
    """保存済みの特徴量（全期間）。関数のハッシュと Market の作成日時が合わなければ止める。"""
    d = feature_dir(cfg)
    meta = json.loads((d / f"{name}.json").read_text(encoding="utf-8"))
    if meta["source_hash"] != get(name).source_hash() or \
            meta["market"].get("built_at_jst") != m.meta.get("built_at_jst"):
        raise SystemExit(f"{name} のキャッシュが古い（python -m src.features.build で作り直す）")
    return np.load(d / f"{name}.npy", mmap_mode="r")


def check_month(cfg: dict, m, u, names: list[str], month: str) -> dict:
    v = cfg["validation"]
    splits = walk_forward_splits(m.dates, v["train_start"], cfg["backtest"]["trade_start_date"], str(m.dates[-1]),
                                 v["gap_days"], cfg["target"]["horizon_days"], v.get("train_window_years"))
    sp = next(s for s in splits if s.month == month)
    e0 = int(sp.eval_preds.min())
    D = str(m.dates[e0])
    light_start = (pd.Timestamp(D) - pd.DateOffset(years=LIGHT_YEARS)).date().isoformat()
    t0 = time.time()
    cfg2 = copy.deepcopy(cfg)
    cfg2["data"]["start_date"] = light_start
    m2 = build_market(cfg2, end=D)
    u2 = universe_mask(m2, cfg2)
    feats2 = compute(load_feature_data(m2, cfg2, end_date=D), names)
    elapsed = time.time() - t0

    tt = sp.train_t
    t2 = np.array([m2.date_index[str(m.dates[t])] for t in tt])
    jmap = np.full(len(m.codes), -1)
    for j2, c in enumerate(m2.codes):
        jmap[m.code_index[c]] = j2
    rows_t, rows_j = np.nonzero(u[tt])          # 全期間のユニバースの行（学習に使う行）
    j2 = jmap[rows_j]
    has2 = j2 >= 0
    in_u2 = np.zeros(len(rows_t), dtype=bool)
    in_u2[has2] = u2[t2[rows_t[has2]], j2[has2]]
    u2_rows = int(u2[t2].sum())
    month_of = pd.Series(m.dates[tt[rows_t]]).str[:7].to_numpy()
    out = {"month": month, "retrain_date": D, "light_start": light_start,
           "train_first": str(m.dates[tt[0]]), "train_last": str(m.dates[tt[-1]]),
           "rows_full_universe": int(len(rows_t)), "rows_light_universe": u2_rows,
           "rows_full_only": int((~in_u2).sum()), "rows_light_only": int(u2_rows - in_u2.sum()),
           "full_only_by_month": {k: int(v) for k, v in pd.Series(~in_u2).groupby(month_of).sum().items() if v},
           "elapsed_sec": round(elapsed, 1), "features": {}}
    for n in names:
        a = np.asarray(full_feature(cfg, m, n)[tt])[rows_t, rows_j]
        b = np.full(len(rows_t), np.nan)
        b[has2] = feats2[n][t2[rows_t[has2]], j2[has2]]
        fa, fb = np.isfinite(a), np.isfinite(b)
        missing = fa & ~fb
        with np.errstate(invalid="ignore"):
            differ = fa & fb & (np.abs(a - b) > 1e-9 * np.maximum(1.0, np.abs(a)))
        # モデルが見るのは日ごとの順位（0〜1）なので、順位に直した値の違いも見る（ユニバースは同じであることを上で確かめる）
        ra = pd.Series(a).groupby(rows_t).rank(pct=True).to_numpy()
        rb = pd.Series(b).groupby(rows_t).rank(pct=True).to_numpy()
        both = fa & fb
        absdiff = np.abs(a - b)[both]
        rdiff = np.abs(ra - rb)[both]
        by_month = pd.Series(missing).groupby(month_of).agg(["sum", "size"])
        by_month = by_month[by_month["sum"] > 0]
        out["features"][n] = {
            "full_finite": int(fa.sum()), "missing_in_light": int(missing.sum()),
            "missing_ratio": float(missing.sum() / max(fa.sum(), 1)),
            "differ": int(differ.sum()), "light_only": int((~fa & fb).sum()),
            "abs_diff_max": float(absdiff.max()) if len(absdiff) else 0.0,
            "abs_diff_p99": float(np.quantile(absdiff, 0.99)) if len(absdiff) else 0.0,
            "rank_diff_max": float(np.nanmax(rdiff)) if len(rdiff) else 0.0,
            "rank_diff_mean": float(np.nanmean(rdiff)) if len(rdiff) else 0.0,
            "rank_diff_gt_0.01": int((rdiff > 0.01).sum()),
            "missing_by_month": {k: [int(r["sum"]), int(r["size"])] for k, r in by_month.iterrows()},
            "last_missing_date": str(m.dates[tt[rows_t[missing]].max()]) if missing.any() else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=["2022-10", "2023-10", "2024-10", "2025-09"])
    args = ap.parse_args()
    cfg = load_config()
    out_path = ROOT / "reports" / "phase4_light_check.json"
    if out_path.exists():
        raise SystemExit(f"既にあります（上書きしません）: {out_path}")
    m = load_market(cfg)
    u = universe_mask(m, cfg)
    names = resolve(cfg["features"]["set"])
    res = {"note": "目的変数・リターン・成績は計算していない。特徴量の値（正規化の前）とユニバースだけを比べる",
           "light_years": LIGHT_YEARS, "months": []}
    for month in args.months:
        r = check_month(cfg, m, u, names, month)
        res["months"].append(r)
        print(f"{month}: 再学習日 {r['retrain_date']}、Light の開始 {r['light_start']}、"
              f"学習 {r['train_first']}〜{r['train_last']}、{r['elapsed_sec']}秒", flush=True)
        for n, f in r["features"].items():
            if f["missing_in_light"] or f["differ"] or f["light_only"]:
                print(f"  {n}: 欠ける {f['missing_in_light']}（{f['missing_ratio']:.2%}）、違う {f['differ']}"
                      f"（順位の差の最大 {f['rank_diff_max']:.4f}）、"
                      f"Light だけ {f['light_only']}、最後に欠ける日 {f['last_missing_date']}", flush=True)
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()
