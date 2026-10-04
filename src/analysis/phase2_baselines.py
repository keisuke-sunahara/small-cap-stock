"""フェーズ2：4つのベースラインの成績と、選ばれた銘柄の上場廃止の割合。

- 期間：売買は 2018-10-01 の週から、ホールドアウト開始の前の週（2025-09-22〜26）まで
- 主の条件：コスト片道0.3%、継続保有の判断は config の backtest.continuation（暫定 prev_day）
- 予算の決め方は2通り：min_equity（基本ルール）と fixed（30万円 ÷ N に固定。戦略の比較用。2026-10-04 ユーザーの依頼）
- ランダム選択は乱数シードを変えて1,000回
- 上場廃止の集計（ユーザーの依頼、2026-10-04）：選ばれた（注文を出した）銘柄のうち、予測日から30営業日以内に
  上場廃止になった割合。判断材料にだけ使い、売買のルールには使わない

出力：reports/phase2_baselines.json、reports/phase2_equity.png、reports/phase2_random.png
実行：python -m src.analysis.phase2_baselines [--random-runs 1000]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd

from src.backtest.baselines import RandomScore, momentum, past_return, reversal, universe_average
from src.backtest.engine import BUDGET_MODES, CONTINUATION_MODES, Backtester, Rules
from src.backtest.market import JST, load_market
from src.backtest.metrics import regime_by_quarter, summarize, t_stat, weekly_rank_ic
from src.backtest.target import make_weeks, weekly_realized
from src.backtest.universe import avg_turnover, universe_mask
from src.config import ROOT, load_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DELIST_WINDOW = 30
RANDOM_KEYS = ["annual_return", "annual_excess_vs_universe", "sharpe", "max_drawdown", "fill_rate",
               "cash_week_ratio", "mean_weekly_excess"]
TITLES = {"min_equity": "予算 = min(30万円, 総資産) ÷ N（基本ルール）", "fixed": "予算を30万円 ÷ N に固定（比較用）"}


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def delisting_stats(m, picks: pd.DataFrame, window: int) -> dict:
    """picks：pred（予測日の位置）と j（銘柄の位置）。予測日から window 営業日以内に上場廃止になった割合。

    データの最終日より後に上場廃止かどうかは分からないため、予測日 + window がデータの外に出る選択は数えない。
    """
    T = len(m.dates)
    p = picks[picks["pred"] + window <= T - 1]
    last = m.last_listed[p["j"].to_numpy()]
    delisted = (last < T - 1) & (last <= p["pred"].to_numpy() + window)
    out = {"picks": int(len(p)), "delisted_within_window": int(delisted.sum()),
           "ratio": float(delisted.mean()) if len(p) else None}
    if "filled" in p:
        out["filled_and_delisted"] = int((delisted & p["filled"].to_numpy(dtype=bool)).sum())
    if delisted.any():
        d = p[delisted]
        qc = m.Qc_ff
        # 予測日の終値 → 最後の終値 の変化（分割調整済み）
        chg = qc[m.last_listed[d["j"].to_numpy()], d["j"].to_numpy()] / qc[d["pred"].to_numpy(), d["j"].to_numpy()] - 1
        out["examples"] = [{"pred_date": str(m.dates[a]), "code": str(m.codes[b]),
                            "last_listed": str(m.dates[m.last_listed[b]]), "change_to_last_close": round(float(c), 4)}
                           for a, b, c in zip(d["pred"], d["j"], chg)][:20]
        out["change_to_last_close_median"] = float(np.median(chg))
    return out


def order_picks(m, orders: pd.DataFrame) -> pd.DataFrame:
    if not len(orders):
        return pd.DataFrame({"pred": [], "j": []}, dtype=int)
    return pd.DataFrame({"pred": orders["pred_date"].map(m.date_index).to_numpy(), "j": orders["j"].to_numpy(),
                         "filled": orders["filled"].to_numpy()})


def top_n_picks(m, universe, score_arr, weeks, n) -> pd.DataFrame:
    """順位の上位 n（100株を買えるかを問わない）。"""
    rows = []
    for w in weeks:
        s = np.where(universe[w.pred], score_arr[w.pred], np.nan)
        ok = np.flatnonzero(~np.isnan(s))
        top = ok[np.argsort(-s[ok], kind="stable")[:n]]
        rows += [(w.pred, int(j)) for j in top]
    return pd.DataFrame(rows, columns=["pred", "j"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--random-runs", type=int, default=1000)
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config()
    m = load_market(cfg)
    adv = avg_turnover(m, cfg["universe"]["turnover_window"])
    u = universe_mask(m, cfg, adv)
    start = cfg["backtest"]["trade_start_date"]
    end = str(m.dates[-1])
    weeks = [w for w in make_weeks(m.dates) if m.dates[w.first] >= start and w.pred >= 0]
    preds = [w.pred for w in weeks]
    week_last = pd.Series([m.dates[w.last] for w in weeks])
    tax = cfg["tax"]["rate"]
    regime = regime_by_quarter(m.dates, m.topix)
    costs = {"cost_0.3%": cfg["cost"]["one_way"], "cost_0.5%": cfg["cost"]["one_way_stress"], "no_cost": 0.0}
    main_mode = cfg["backtest"]["continuation"]
    n = cfg["capital"]["n_holdings"]

    def rules(**kw) -> Rules:
        kw.setdefault("continuation", main_mode)
        return Rules.from_config(cfg, **kw)

    # ユニバース平均（ベースライン4、超過リターンの基準）
    ua = {name: universe_average(m, u, weeks, cfg["order"]["limit_up_pct"], c)[1] for name, c in costs.items()}
    bench = pd.Series(ua["no_cost"])
    start_pred = str(m.dates[weeks[0].pred])

    def summarize_run(res):
        assert list(res.weekly["last"]) == list(week_last)
        return summarize(res.weekly, res.curve, bench, start_pred=start_pred, tax_rate=tax, regime=regime,
                         orders=res.orders)

    usize = u[preds].sum(1)
    report: dict = {"meta": {"git_commit": git_commit(), "run_at_jst": datetime.now(JST).isoformat(timespec="seconds"),
                             "data_latest_fetch_jst": m.meta.get("data_latest_fetch_jst"),
                             "period": [start_pred, end], "weeks": len(weeks),
                             "one_day_weeks": int(sum(len(w.days) == 1 for w in weeks)),
                             "continuation_main": main_mode, "n_holdings": n,
                             "capital": cfg["capital"]["initial_capital_jpy"],
                             "random_runs": args.random_runs,
                             "universe_size_per_pred": {"min": int(usize.min()), "median": float(np.median(usize)),
                                                        "max": int(usize.max())}}}
    uw = pd.DataFrame({"last": week_last, "invested": True})
    report["universe_average"] = {}
    for name in costs:
        uw["ret"] = ua[name]
        eq = np.concatenate([[1.0], np.cumprod(1 + ua[name])])
        report["universe_average"][name] = summarize(uw, eq, bench, start_pred=start_pred, tax_rate=tax,
                                                     regime=regime)
    fill = universe_average(m, u, weeks, cfg["order"]["limit_up_pct"], 0.0)[2]
    report["universe_average"]["fill_rate_mean"] = float(np.nanmean(fill))

    # モメンタム・リバーサル（予算の決め方2通り × コスト3通り）
    scores = {"momentum_20d": momentum(m, 20), "reversal_5d": reversal(m, 5)}
    report["baselines"] = {}
    curves: dict[str, dict[str, np.ndarray]] = {b: {} for b in BUDGET_MODES}
    picks_pool: dict[str, pd.DataFrame] = {}
    for sname, sfn in scores.items():
        report["baselines"][sname] = {}
        for budget in BUDGET_MODES:
            report["baselines"][sname][budget] = {}
            for cname, c in costs.items():
                res = Backtester(m, u, adv, rules(cost=c, budget_mode=budget)).run(sfn, start, end)
                sm = summarize_run(res)
                sm["order_rank_mean"] = float(res.orders["rank"].mean())
                sm["order_rank_median"] = float(res.orders["rank"].median())
                sm["order_limit_price_median"] = float(res.orders["limit"].median())
                report["baselines"][sname][budget][cname] = sm
                if cname == "cost_0.3%":
                    curves[budget][sname] = np.cumprod(1 + res.weekly["ret"].to_numpy())
                    picks_pool[f"{sname}_{budget}"] = order_picks(m, res.orders)

    # 継続保有の方式の比較（コスト0.3%）
    report["continuation_modes"] = {}
    for budget in BUDGET_MODES:
        report["continuation_modes"][budget] = {}
        for sname, sfn in scores.items():
            report["continuation_modes"][budget][sname] = {}
            for mode in CONTINUATION_MODES:
                res = Backtester(m, u, adv, rules(continuation=mode, budget_mode=budget)).run(sfn, start, end)
                sm = summarize_run(res)
                d = {k: sm[k] for k in RANDOM_KEYS}
                d["trades"] = int(len(res.trades))
                report["continuation_modes"][budget][sname][mode] = d

    # Rank IC（週次・重ならない期間）
    realized = weekly_realized(m, weeks)
    urows = u[preds]
    report["rank_ic"] = {}
    for sname, window, sign in [("momentum_20d", 20, 1), ("reversal_5d", 5, -1)]:
        sc = sign * past_return(m, window)[preds]
        ic = pd.Series(weekly_rank_ic(sc, realized, urows))
        report["rank_ic"][sname] = {"mean": float(ic.mean()), "std": float(ic.std()),
                                    "mean_over_std": float(ic.mean() / ic.std()), "t": t_stat(ic),
                                    "weeks": int(ic.notna().sum())}

    # ランダム選択（1,000回 × 予算の決め方2通り × コスト3通り）。継続保有の方式の比較は100回
    rnd = {b: {k: [] for k in costs} for b in BUDGET_MODES}
    rnd_modes = {b: {k: [] for k in CONTINUATION_MODES} for b in BUDGET_MODES}
    rnd_picks = []
    for seed in range(args.random_runs):
        for budget in BUDGET_MODES:
            for cname, c in costs.items():
                res = Backtester(m, u, adv, rules(cost=c, budget_mode=budget)).run(
                    RandomScore(len(m.codes), seed), start, end)
                sm = summarize_run(res)
                rnd[budget][cname].append({k: sm[k] for k in RANDOM_KEYS})
                if cname == "cost_0.3%":
                    if budget == "min_equity":
                        rnd_picks.append(order_picks(m, res.orders))
                    if seed == 0:
                        curves[budget]["random_seed0"] = np.cumprod(1 + res.weekly["ret"].to_numpy())
            if seed < 100:
                for mode in CONTINUATION_MODES:
                    res = Backtester(m, u, adv, rules(continuation=mode, budget_mode=budget)).run(
                        RandomScore(len(m.codes), seed), start, end)
                    rnd_modes[budget][mode].append(summarize_run(res)["annual_return"])
        if (seed + 1) % 100 == 0:
            print(f"ランダム {seed + 1}/{args.random_runs}（{time.time() - t0:.0f}秒）", flush=True)
    report["random"] = {}
    rnd_ann = {}
    for budget in BUDGET_MODES:
        report["random"][budget] = {}
        for cname in costs:
            df = pd.DataFrame(rnd[budget][cname])
            report["random"][budget][cname] = {
                "runs": len(df), "mean": df.mean().to_dict(),
                "percentiles_annual_return": {str(q): float(df["annual_return"].quantile(q))
                                              for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}
            for sname in scores:
                ar = report["baselines"][sname][budget][cname]["annual_return"]
                report["baselines"][sname][budget][cname]["percentile_in_random"] = float(
                    (df["annual_return"] < ar).mean())
        rnd_ann[budget] = pd.DataFrame(rnd[budget]["cost_0.3%"])["annual_return"]
        report["continuation_modes"][budget]["random_100runs_mean_annual_return"] = {
            k: float(np.mean(v)) for k, v in rnd_modes[budget].items()}

    # 上場廃止の集計
    report["delisting"] = {"window_bdays": DELIST_WINDOW,
                           "note": "予測日から30営業日以内に上場廃止。予測日+30営業日がデータの外（2025-08以降の予測日）の選択は数えない"}
    for key, picks in picks_pool.items():
        report["delisting"][f"{key}_orders"] = delisting_stats(m, picks, DELIST_WINDOW)
    for sname, sfn in scores.items():
        report["delisting"][f"{sname}_top{n}_any_price"] = delisting_stats(
            m, top_n_picks(m, u, sfn.arr, weeks, n), DELIST_WINDOW)
    rs = delisting_stats(m, pd.concat(rnd_picks, ignore_index=True), DELIST_WINDOW)
    rs.pop("examples", None)
    report["delisting"]["random_min_equity_orders_all_runs"] = rs
    allu = pd.DataFrame([(w.pred, int(j)) for w in weeks for j in np.flatnonzero(u[w.pred])], columns=["pred", "j"])
    us = delisting_stats(m, allu, DELIST_WINDOW)
    us.pop("examples", None)
    report["delisting"]["universe_all"] = us
    report["meta"]["elapsed_sec"] = round(time.time() - t0, 1)

    out = ROOT / "reports" / "phase2_baselines.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # グラフ
    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
    x = pd.to_datetime(week_last)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, budget in zip(axes, BUDGET_MODES):
        for name in ["no_cost", "cost_0.3%"]:
            ax.plot(x, np.cumprod(1 + ua[name]), label=f"ユニバース平均（{name}）", lw=1.2)
        for name, cv in curves[budget].items():
            ax.plot(x, cv, label=f"{name}（コスト0.3%）", lw=1)
        ax.set_yscale("log")
        ax.axhline(1, color="gray", lw=0.5)
        ax.set_title(TITLES[budget], fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("ベースラインの資産の推移（初期値=1、対数目盛り）")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "phase2_equity.png", dpi=110)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, budget in zip(axes, BUDGET_MODES):
        ax.hist(rnd_ann[budget] * 100, bins=40, color="#8899aa")
        for sname, col in [("momentum_20d", "C3"), ("reversal_5d", "C2")]:
            ax.axvline(report["baselines"][sname][budget]["cost_0.3%"]["annual_return"] * 100, color=col,
                       label=sname)
        ax.axvline(report["universe_average"]["no_cost"]["annual_return"] * 100, color="k", ls="--",
                   label="ユニバース平均（コストなし）")
        ax.set_xlabel("年率リターン（%、コスト0.3%）")
        ax.set_title(f"ランダム選択 {args.random_runs} 回の分布：{TITLES[budget]}", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "phase2_random.png", dpi=110)
    plt.close(fig)
    print(f"保存: {out}（{time.time() - t0:.0f}秒）")


if __name__ == "__main__":
    main()
