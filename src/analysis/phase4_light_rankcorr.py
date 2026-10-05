"""運用時（Light）の学習の再現の確認・案B（experiments/README.md 第2章。2026-10-05 ユーザーが承認）。

確定したモデルを、評価の月 M（2023-10・2024-10・2025-09）について、次の2通りの特徴量で学習し、同じ日の点数の順位相関を比べる。
成績（リターン・Rank IC・売買）は計算しない。

- 全期間：バックテストと同じ（2016-10-05 以降のデータで作った特徴量）
- Light の範囲：月 M の最初の予測日 D の5年前（暦）以降のデータだけで Market・ユニバース・特徴量を作り直し（D まで）、学習の行
  （D の4年前〜D の6営業日前）の特徴量とユニバースをこれに置き換える。目的変数は同じ株価のリターンなので全期間の Market で計算する
  （Light でも、学習の行の起点から5営業日後までの株価は D までに取れる）

点数を付ける行（月 M の予測日・継続の判断の日のユニバース）は、どちらも全期間の特徴量を使う（学習の違いの影響だけを見るため）。
D の日の特徴量が Light の範囲でも同じかは、別に数える。全期間で学習した点数は、実験の scores.parquet と一致することを確かめる。
判定：どれかの月で、点数を付ける日ごとの順位相関（スピアマン）の最小値が 0.98 未満なら、案A への切り替えを提案して止まる。

実行：python -m src.analysis.phase4_light_rankcorr --exp EXP-xxx
出力：reports/phase4_light_rankcorr.json
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import pandas as pd

from src.backtest.market import build_market, load_market
from src.backtest.universe import avg_turnover, bad_factor_events, universe_mask
from src.config import ROOT, load_config
from src.features.build import compute, load_features
from src.features.data import load_feature_data
from src.features.registry import resolve
from src.models.dataset import daily_panel
from src.models.train import Model
from src.models.walkforward import score_days, splits_for

LIGHT_YEARS = 5
THRESHOLD = 0.98
MONTHS = ["2023-10", "2024-10", "2025-09"]


def light_inputs(cfg: dict, m, names: list[str], D: str):
    """D の5年前以降・D までのデータで作り直したユニバースと特徴量を、全期間の Market の位置 [T, N] に並べ直す（範囲外は NaN・False）。"""
    light_start = (pd.Timestamp(D) - pd.DateOffset(years=LIGHT_YEARS)).date().isoformat()
    cfg2 = copy.deepcopy(cfg)
    cfg2["data"]["start_date"] = light_start
    m2 = build_market(cfg2, end=D)
    u2 = universe_mask(m2, cfg2)
    f2 = compute(load_feature_data(m2, cfg2, end_date=D), names)
    tmap = np.array([m2.date_index.get(str(d), -1) for d in m.dates])
    jmap = np.array([m2.code_index.get(str(c), -1) for c in m.codes])
    ok_t, ok_j = tmap >= 0, jmap >= 0
    u = np.zeros((len(m.dates), len(m.codes)), dtype=bool)
    u[np.ix_(ok_t, ok_j)] = u2[np.ix_(tmap[ok_t], jmap[ok_j])]
    feats = {}
    for n in names:
        a = np.full(u.shape, np.nan, dtype=np.float32)
        a[np.ix_(ok_t, ok_j)] = f2[n][np.ix_(tmap[ok_t], jmap[ok_j])]
        feats[n] = a
    return light_start, u, feats


def check_month(cfg, m, u, feats, bad_event, sp, sd, saved: pd.DataFrame, names) -> dict:
    t_start = time.time()
    mcfg, seed = cfg["model"], int(cfg["project"]["random_seed"])
    horizon, how = cfg["target"]["horizon_days"], cfg["features"]["normalize"]
    e0 = int(sp.eval_preds.min())
    D = str(m.dates[e0])
    tt = sp.train_t
    light_start, u_l, f_l = light_inputs(cfg, m, names, D)

    # 学習：全期間 と Light の範囲（学習の行の特徴量・ユニバースだけを置き換える）
    tr_full = daily_panel(m, u, feats, horizon, how, t_index=tt, bad_event=bad_event)
    u_mix = u.copy()
    u_mix[tt] = u_l[tt]
    f_mix = {}
    for n in names:
        a = np.array(feats[n], dtype=np.float32, copy=True)
        a[tt] = f_l[n][tt]
        f_mix[n] = a
    tr_light = daily_panel(m, u_mix, f_mix, horizon, how, t_index=tt, bad_event=bad_event)

    # 点数を付ける行（全期間の特徴量）
    days = sd[sd["split"] == sp.month]["t"].to_numpy()
    sc = daily_panel(m, u, feats, horizon, how, t_index=days, bad_event=bad_event)
    out_scores = {}
    for label, tr in (("full", tr_full), ("light", tr_light)):
        sel = np.isfinite(tr.y)
        model = Model(mcfg, seed).fit(tr.X[sel], tr.y[sel].astype(float), tr.t[sel])
        out_scores[label] = model.predict(sc.X)
    df = pd.DataFrame({"date": m.dates[sc.t].astype(str), "code": m.codes[sc.j].astype(str),
                       "full": out_scores["full"], "light": out_scores["light"]})
    # 全期間で学習した点数が、実験の scores.parquet と一致すること（同じ計算の再現）
    mg = df.merge(saved, on=["date", "code"], how="left")
    max_diff_saved = float(np.nanmax(np.abs(mg["full"] - mg["score"])))
    if mg["score"].isna().any() or max_diff_saved > 1e-6:
        raise AssertionError(f"{sp.month}: 全期間で学習した点数が scores.parquet と合いません（最大の差 {max_diff_saved}）")
    by_day = {d: float(g["full"].rank().corr(g["light"].rank())) for d, g in df.groupby("date")}

    # D の日の特徴量が Light の範囲でも同じか（点数を付ける最初の日）
    rows = np.flatnonzero(u[e0])
    d_diff = {}
    for n in names:
        a, b = np.asarray(feats[n][e0, rows], dtype=float), np.asarray(f_l[n][e0, rows], dtype=float)
        with np.errstate(invalid="ignore"):
            diff = (np.isfinite(a) != np.isfinite(b)) | (np.isfinite(a) & np.isfinite(b)
                                                         & (np.abs(a - b) > 1e-6 * np.maximum(1.0, np.abs(a))))
        if diff.any():
            d_diff[n] = int(diff.sum())
    return {"month": sp.month, "retrain_date": D, "light_start": light_start,
            "train_first": str(m.dates[tt[0]]), "train_last": str(m.dates[tt[-1]]),
            "train_rows_full": int(np.isfinite(tr_full.y).sum()), "train_rows_light": int(np.isfinite(tr_light.y).sum()),
            "score_days": len(by_day), "score_rows": int(len(df)),
            "rank_corr_by_day": by_day, "rank_corr_min": min(by_day.values()),
            "rank_corr_mean": float(np.mean(list(by_day.values()))),
            "rank_corr_all_rows": float(df["full"].rank().corr(df["light"].rank())),
            "full_matches_saved_scores_max_abs_diff": max_diff_saved,
            "universe_rows_differ_in_train": int((u[tt] != u_l[tt]).sum()),
            "features_differ_on_retrain_date": d_diff,
            "universe_differs_on_retrain_date": int((u[e0] != u_l[e0]).sum()),
            "elapsed_sec": round(time.time() - t_start, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="確定したモデルの実験ID")
    ap.add_argument("--months", nargs="+", default=MONTHS)
    args = ap.parse_args()
    out_path = ROOT / "reports" / "phase4_light_rankcorr.json"
    if out_path.exists():
        raise SystemExit(f"既にあります（上書きしません）: {out_path}")
    cfg = load_config(args.exp)
    if cfg["model"].get("train_universe", "all") != "all" or cfg["model"].get("target_shuffle"):
        raise SystemExit("この確認は、全銘柄で学習するモデルだけ")
    m = load_market(cfg)
    adv = avg_turnover(m, cfg["universe"]["turnover_window"])
    u = universe_mask(m, cfg, adv)
    names = resolve(cfg["features"]["set"])
    feats = load_features(names, cfg, m)
    thr = cfg["universe"].get("bad_adjfactor_threshold")
    bad_event = bad_factor_events(m, thr)[1] if thr is not None else None
    splits = {s.month: s for s in splits_for(m, cfg)}
    sd = score_days(m, cfg)
    e0s = np.array([int(s.eval_preds.min()) for s in splits.values()])
    months_sorted = list(splits)
    sd["split"] = [months_sorted[i] for i in np.searchsorted(e0s, sd["t"].to_numpy(), side="right") - 1]
    saved = pd.read_parquet(ROOT / "logs" / "backtest" / args.exp / "scores.parquet")[["date", "code", "score"]]
    res = {"experiment": args.exp, "threshold": THRESHOLD, "light_years": LIGHT_YEARS,
           "note": "成績は計算していない。学習の行の特徴量・ユニバースを Light の範囲で作り直した場合の、点数の順位相関だけ",
           "months": []}
    for month in args.months:
        r = check_month(cfg, m, u, feats, bad_event, splits[month], sd, saved, names)
        res["months"].append(r)
        print(f"{month}: 学習 {r['train_first']}〜{r['train_last']}、Light の開始 {r['light_start']}、"
              f"順位相関 最小 {r['rank_corr_min']:.5f}・平均 {r['rank_corr_mean']:.5f}（{r['score_days']}日）、"
              f"学習の行 {r['train_rows_full']} / {r['train_rows_light']}、{r['elapsed_sec']}秒", flush=True)
    res["min_rank_corr"] = min(r["rank_corr_min"] for r in res["months"])
    res["pass"] = bool(res["min_rank_corr"] >= THRESHOLD)
    res["decision"] = ("案B のまま進める（すべての月で 0.98 以上）" if res["pass"]
                       else "0.98 未満の月がある：案A への切り替えを提案して止まる")
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(res["decision"])
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()
