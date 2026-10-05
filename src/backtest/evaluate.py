"""フェーズ4の実験の評価（experiments/README.md 第3〜5章、CLAUDE.md 第9章）。src/run.py から使う。

- 売買：予算の決め方2通り（fixed：予算固定＝主、min_equity：基本ルール）× コスト4通り（一律0.3%・一律0.5%・銘柄ごと・なし）。
  保有期間 H > 1（候補9）は開始週 H 通り（offset 0〜H−1）を計算し、指標は H 通りの平均、週次リターンは週ごとの平均を使う
- 超過リターンの基準：ユニバース平均（コストなし）。ベースラインとの比較は reports/phase3r_baselines.json（同じ期間・同じ売買ルール・
  同じコスト）。ランダムの分布の中での位置は、ランダム1,000回の年率の超過リターン（logs/backtest/random/。無ければ計算して保存）
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.baselines import RandomScore, universe_average
from src.backtest.engine import BUDGET_MODES, Backtester, Result, Rules
from src.backtest.execution import limit_price_array
from src.backtest.market import Market
from src.backtest.metrics import (breakeven_one_way_cost, regime_by_quarter, summarize, t_stat,
                                  weekly_rank_ic)
from src.backtest.target import Week, weekly_realized
from src.config import ROOT
from src.models.walkforward import eval_end, eval_weeks

COST_NAMES = ("cost_0.3%", "cost_0.5%", "cost_tick", "no_cost")
MAIN_COSTS = ("cost_0.3%", "cost_tick")      # 判断に使う2つのコスト（一律0.3%と銘柄ごと）
DELIST_WINDOW = 30
RANDOM_RUNS = 1000
MEAN_KEYS = ("annual_return", "annual_excess_vs_universe", "total_return", "sharpe", "max_drawdown", "worst_week",
             "best_week", "mean_weekly_excess", "cash_week_ratio", "fill_rate", "years_with_positive_excess",
             "breakeven_one_way_cost_vs_universe", "turnover_per_year")


def cost_variants(cfg: dict) -> dict[str, dict]:
    c = cfg["cost"]
    return {"cost_0.3%": {"cost": c["one_way"], "cost_model": "flat"},
            "cost_0.5%": {"cost": c["one_way_stress"], "cost_model": "flat"},
            "cost_tick": {"cost": c["one_way"], "cost_model": "tick"},
            "no_cost": {"cost": 0.0, "cost_model": "flat"}}


@dataclass
class Context:
    m: Market
    u: np.ndarray
    adv: np.ndarray
    cfg: dict
    weeks: list[Week]
    start: str
    end: str
    start_pred: str
    ua: dict[str, np.ndarray]          # ユニバース平均の週次リターン（コストごと）
    regime: pd.Series
    extras: dict = field(default_factory=dict)   # Backtester に渡す market_ok・avoid・vol

    @property
    def bench(self) -> pd.Series:
        return pd.Series(self.ua["no_cost"])

    @property
    def week_last(self) -> list[str]:
        return [str(self.m.dates[w.last]) for w in self.weeks]


def make_context(m: Market, u: np.ndarray, adv: np.ndarray, cfg: dict, extras: dict | None = None) -> Context:
    weeks = eval_weeks(m, cfg)
    ua = {name: universe_average(m, u, weeks, cfg["order"]["limit_up_pct"], **c)[1]
          for name, c in cost_variants(cfg).items()}
    return Context(m=m, u=u, adv=adv, cfg=cfg, weeks=weeks, start=cfg["backtest"]["trade_start_date"],
                   end=eval_end(cfg), start_pred=str(m.dates[weeks[0].pred]), ua=ua,
                   regime=regime_by_quarter(m.dates, m.topix), extras=extras or {})


# ---- 売買 ----
def run_backtests(ctx: Context, score_fn, cfg: dict) -> dict[str, dict[str, list[Result]]]:
    """予算の決め方 × コスト ×（H > 1 なら）開始週 の売買の結果。"""
    out: dict[str, dict[str, list[Result]]] = {}
    base_rules = Rules.from_config(cfg)
    for budget in ("fixed", "min_equity"):
        out[budget] = {}
        for cname, c in cost_variants(cfg).items():
            rules = Rules.from_config(cfg, budget_mode=budget, **c)
            bt = Backtester(ctx.m, ctx.u, ctx.adv, rules, **ctx.extras)
            out[budget][cname] = [bt.run(score_fn, ctx.start, ctx.end, offset=k)
                                  for k in range(base_rules.holding_weeks)]
    return out


def summarize_result(ctx: Context, res: Result) -> dict:
    if list(res.weekly["last"]) != ctx.week_last:
        raise AssertionError("売買の週がユニバース平均の週と合いません")
    sm = summarize(res.weekly, res.curve, ctx.bench, start_pred=ctx.start_pred, tax_rate=ctx.cfg["tax"]["rate"],
                   regime=ctx.regime, orders=res.orders)
    years = (pd.Timestamp(ctx.week_last[-1]) - pd.Timestamp(ctx.start_pred)).days / 365.25
    sm["turnover_per_year"] = float(res.weekly["traded_value"].sum() / res.rules.capital / years)
    if res.rules.budget_mode == "fixed":
        be = breakeven_one_way_cost(res.weekly, res.rules.capital, ctx.bench, start_pred=ctx.start_pred)
        sm["breakeven_one_way_cost_vs_universe"] = be
        sm["breakeven_one_way_cost_vs_zero"] = breakeven_one_way_cost(res.weekly, res.rules.capital, None,
                                                                      start_pred=ctx.start_pred)
    if len(res.orders):
        sm["order_rank_mean"] = float(res.orders["rank"].mean())
        sm["order_limit_price_median"] = float(res.orders["limit"].median())
    if len(res.trades):
        sm["trade_reasons"] = {k: int(v) for k, v in res.trades["reason"].value_counts().items()}
    return sm


def mean_of(summaries: list[dict]) -> dict:
    """開始週ごとの指標の平均（候補9）。"""
    out = {}
    for k in MEAN_KEYS:
        vals = [s.get(k) for s in summaries]
        vals = [np.nan if v is None else v for v in vals]
        arr = np.array(vals, dtype=float)
        if len(arr) and np.isfinite(arr).any():
            out[k] = float(np.nanmean(arr))
    out["years"] = summaries[0]["years"]
    return out


def summarize_grid(ctx: Context, grid: dict) -> tuple[dict, pd.DataFrame]:
    """指標（budget → cost → 指標。H > 1 なら "mean"・"offsets"・最も良い/悪い開始週）と、週次リターンの表。"""
    metrics: dict = {}
    weekly = pd.DataFrame({"last": ctx.week_last})
    for budget, by_cost in grid.items():
        metrics[budget] = {}
        for cname, results in by_cost.items():
            sms = [summarize_result(ctx, r) for r in results]
            rets = np.array([r.weekly["ret"].to_numpy() for r in results])
            weekly[f"{budget}|{cname}"] = rets.mean(axis=0)
            if len(results) == 1:
                metrics[budget][cname] = sms[0]
            else:
                for k, r in enumerate(rets):
                    weekly[f"{budget}|{cname}|o{k}"] = r
                ex = [s["annual_excess_vs_universe"] for s in sms]
                metrics[budget][cname] = {**mean_of(sms), "offsets": sms,
                                          "best_offset": int(np.argmax(ex)), "worst_offset": int(np.argmin(ex)),
                                          "note": "開始週の平均（候補9）。offsets に開始週ごとの指標"}
    return metrics, weekly


# ---- 予測の評価 ----
def rank_ic_stats(ctx: Context, scores: np.ndarray, rows: np.ndarray | None = None) -> dict:
    """予測日の点数 scores [週, N] と翌週の実現リターンの Rank IC（rows はユニバースの代わりに使う銘柄 [週, N]）。"""
    realized = weekly_realized(ctx.m, ctx.weeks)
    urows = ctx.u[[w.pred for w in ctx.weeks]] if rows is None else rows
    ic = pd.Series(weekly_rank_ic(scores, realized, urows))
    year = pd.Series([s[:4] for s in ctx.week_last])
    return {"mean": float(ic.mean()), "std": float(ic.std()), "mean_over_std": float(ic.mean() / ic.std()),
            "t": t_stat(ic), "weeks": int(ic.notna().sum()),
            "by_year": {y: float(v) for y, v in ic.groupby(year).mean().items()},
            "weekly": [None if v != v else float(v) for v in ic]}


def decile_returns(ctx: Context, scores: np.ndarray, n_q: int = 10) -> list[float]:
    """予測の分位（1 が点数の最も低い組）ごとの翌週の平均リターン（ユニバース全体、コストなし。週ごとの平均の平均）。"""
    realized = weekly_realized(ctx.m, ctx.weeks)
    sums, counts = np.zeros(n_q), np.zeros(n_q)
    for k, w in enumerate(ctx.weeks):
        ok = ctx.u[w.pred] & np.isfinite(scores[k]) & np.isfinite(realized[k])
        if ok.sum() < n_q:
            continue
        q = pd.qcut(pd.Series(scores[k][ok]).rank(method="first"), n_q, labels=False).to_numpy()
        r = realized[k][ok]
        for g in range(n_q):
            sums[g] += r[q == g].mean()
            counts[g] += 1
    return [float(s / c) if c else None for s, c in zip(sums, counts)]


def top_n_unaffordable(ctx: Context, scores: np.ndarray, cfg: dict) -> dict:
    """予測日のユニバースで点数の上位N（買えるかを問わない）のうち、100株を買えない（指値 × 100株 > 運用資金 ÷ N）割合。"""
    n = cfg["capital"]["n_holdings"]
    budget = cfg["capital"]["initial_capital_jpy"] / n
    tot = bad = 0
    for k, w in enumerate(ctx.weeks):
        s = np.where(ctx.u[w.pred], scores[k], np.nan)
        ok = np.flatnonzero(~np.isnan(s))
        top = ok[np.lexsort((ok, -s[ok]))][:n]
        lim = limit_price_array(ctx.m.C[w.pred, top], cfg["order"]["limit_up_pct"])
        bad += int((lim * cfg["order"]["lot_size"] > budget).sum())
        tot += len(top)
    return {"top_n": n, "picks": tot, "unaffordable": bad, "ratio": bad / tot if tot else None}


# ---- ベースラインとランダムの分布 ----
def load_baselines(cfg: dict, ctx: Context) -> dict:
    path = ROOT / cfg["evaluation"]["baselines_report"]
    rep = json.loads(path.read_text(encoding="utf-8"))
    if rep["meta"]["period"] != [ctx.start_pred, ctx.week_last[-1]] or rep["meta"]["weeks"] != len(ctx.weeks):
        raise AssertionError(f"ベースラインの期間が違います: {rep['meta']['period']}, {rep['meta']['weeks']}週")
    out = {"source": str(cfg["evaluation"]["baselines_report"]), "git_commit": rep["meta"]["git_commit"]}
    for budget in BUDGET_MODES:
        out[budget] = {}
        for cname in COST_NAMES:
            out[budget][cname] = {
                "momentum_20d": rep["baselines"]["momentum_20d"][budget][cname]["annual_excess_vs_universe"],
                "reversal_5d": rep["baselines"]["reversal_5d"][budget][cname]["annual_excess_vs_universe"],
                "random_mean": rep["random"][budget][cname]["mean"]["annual_excess_vs_universe"],
                "universe_average": rep["universe_average"][cname]["annual_excess_vs_universe"],
                "universe_average_annual_return": rep["universe_average"][cname]["annual_return"],
            }
    return out


def random_distribution(ctx: Context, cfg: dict, path=None, runs: int = RANDOM_RUNS, log=print) -> pd.DataFrame:
    """ランダム（乱数シード 0〜runs−1）の年率リターン・年率の超過リターン（予算の決め方2通り × 判断に使うコスト2通り）。

    基本ルール（N は cfg の値）で計算し、path に保存して使い回す（同じ設定なら同じ値。ベースラインの計算と同じコード）。
    """
    if path is not None and path.exists():
        return pd.read_csv(path)
    rows = []
    for seed in range(runs):
        for budget in BUDGET_MODES:
            for cname in MAIN_COSTS:
                rules = Rules.from_config(cfg, budget_mode=budget, **cost_variants(cfg)[cname])
                res = Backtester(ctx.m, ctx.u, ctx.adv, rules).run(RandomScore(len(ctx.m.codes), seed), ctx.start,
                                                                   ctx.end)
                sm = summarize_result(ctx, res)
                rows.append({"seed": seed, "budget": budget, "cost": cname, "annual_return": sm["annual_return"],
                             "annual_excess_vs_universe": sm["annual_excess_vs_universe"]})
        if log is not None and (seed + 1) % 100 == 0:
            log(f"ランダム {seed + 1}/{runs}")
    df = pd.DataFrame(rows)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


def check_random_against_report(dist: pd.DataFrame, baselines_report: dict) -> None:
    """保存したランダムの分布の平均が、ベースラインの報告（同じコード・同じ乱数シード）の平均と一致することを確かめる。"""
    for (budget, cname), g in dist.groupby(["budget", "cost"]):
        want = baselines_report["random"][budget][cname]["mean"]["annual_excess_vs_universe"]
        if len(g) == baselines_report["random"][budget][cname]["runs"] and abs(g["annual_excess_vs_universe"].mean() - want) > 1e-9:
            raise AssertionError(f"ランダムの分布がベースラインの報告と合いません: {budget} {cname}")


def percentile_in(dist: pd.DataFrame, budget: str, cname: str, value: float) -> float:
    d = dist[(dist["budget"] == budget) & (dist["cost"] == cname)]["annual_excess_vs_universe"]
    return float((d < value).mean())


# ---- 判定 ----
def beats_baselines(summary: dict, base: dict) -> dict:
    ex = summary["annual_excess_vs_universe"]
    vs = {k: bool(ex > base[k]) for k in ("momentum_20d", "reversal_5d", "random_mean", "universe_average")}
    return {"annual_excess": ex, "baselines": {k: base[k] for k in vs}, "beats": vs, "all": all(vs.values())}


def criteria(metrics: dict, ic: dict, baselines: dict, cfg: dict) -> dict:
    """CLAUDE.md 第9章の合格基準の各項目（ホールドアウトはフェーズ5）。予算の決め方 × 判断に使うコストごと。"""
    data_cost = cfg["evaluation"]["data_cost_annual_jpy"] / cfg["capital"]["initial_capital_jpy"]
    out = {"rank_ic_positive_t2": bool(ic["mean"] > 0 and ic["t"] >= 2), "data_cost_ratio": data_cost,
           "holdout": "フェーズ5で判定"}
    for budget in BUDGET_MODES:
        out[budget] = {}
        for cname in MAIN_COSTS:
            s = metrics[budget][cname]
            b = baselines[budget][cname]
            ann = s["annual_return"]
            years = s.get("years") or len(s.get("by_year", {})) or None
            ypos = s["years_with_positive_excess"]
            out[budget][cname] = {
                "excess_positive": bool(s["annual_excess_vs_universe"] > 0),
                "majority_years_positive": bool(years and ypos > years / 2) if years else None,
                "beats_all_baselines": beats_baselines(s, b)["all"],
                "after_data_cost_beats_universe_same_cost": bool(ann - data_cost > b["universe_average_annual_return"]),
                "after_data_cost_beats_universe_no_cost": bool(
                    ann - data_cost > baselines[budget]["no_cost"]["universe_average_annual_return"]),
                "after_all_costs_positive": bool(ann - data_cost > 0),
                "annual_return_after_data_cost": ann - data_cost,
            }
    return out


def model_candidate(metrics: dict, ic: dict, baselines: dict) -> dict:
    """README 第4章の採用候補の条件：Rank IC の平均 > 0 かつ t ≥ 2、予算固定の年率の超過リターンが一律0.3%と銘柄ごとの両方で
    4つのベースラインをすべて上回る。"""
    bb = {c: beats_baselines(metrics["fixed"][c], baselines["fixed"][c]) for c in MAIN_COSTS}
    ok_ic = bool(ic["mean"] > 0 and ic["t"] >= 2)
    return {"rank_ic_ok": ok_ic, "beats_baselines": bb, "candidate": bool(ok_ic and all(v["all"] for v in bb.values()))}


def weekly_diff_t(a: pd.Series, b: pd.Series) -> float:
    return t_stat(pd.Series(a.to_numpy() - b.to_numpy()))


def rule_adoption(cand_metrics: dict, cand_weekly: pd.DataFrame, base_metrics: dict, base_weekly: pd.DataFrame,
                  t_min: float = 2.0, mdd_tol: float = 0.05) -> dict:
    """README 第5章の採用の条件（候補 vs 確定したモデル ＋ 基本ルール。予算固定）。"""
    better = {c: bool(cand_metrics["fixed"][c]["annual_excess_vs_universe"]
                      > base_metrics["fixed"][c]["annual_excess_vs_universe"]) for c in MAIN_COSTS}
    t = weekly_diff_t(cand_weekly["fixed|cost_0.3%"], base_weekly["fixed|cost_0.3%"])
    mdd_c = cand_metrics["fixed"]["cost_0.3%"]["max_drawdown"]
    mdd_b = base_metrics["fixed"]["cost_0.3%"]["max_drawdown"]
    ok_mdd = bool(mdd_c > mdd_b - mdd_tol)
    out = {"better_excess": better, "weekly_diff_t": t, "t_min": t_min, "max_drawdown": [mdd_c, mdd_b],
           "mdd_ok": ok_mdd,
           "excess": {c: [cand_metrics["fixed"][c]["annual_excess_vs_universe"],
                          base_metrics["fixed"][c]["annual_excess_vs_universe"]] for c in MAIN_COSTS}}
    adopt = bool(all(better.values()) and t == t and t >= t_min and ok_mdd)
    if "offsets" in cand_metrics["fixed"]["cost_0.3%"]:
        # 候補9：H 通りの開始週のすべてで、一律0.3%と銘柄ごとのコストの両方で基本ルールを上回る（評価役のフェーズ4計画の指摘 低2）
        by_off = {c: [bool(o["annual_excess_vs_universe"] > base_metrics["fixed"][c]["annual_excess_vs_universe"])
                      for o in cand_metrics["fixed"][c]["offsets"]] for c in MAIN_COSTS}
        out["better_excess_all_offsets"] = by_off
        adopt = adopt and all(all(v) for v in by_off.values())
    out["adopt"] = adopt
    return out


def leak_check(ic: dict, metrics: dict, dist: pd.DataFrame) -> dict:
    """EXP-003 の判断基準：Rank IC の t値の絶対値 < 2（平均の絶対値 < 0.005 は参考。評価役のフェーズ4計画の指摘 低1）、
    予算固定・一律0.3%の年率の超過リターンがランダムの分布の 5%点〜95%点 の間。"""
    d = dist[(dist["budget"] == "fixed") & (dist["cost"] == "cost_0.3%")]["annual_excess_vs_universe"]
    lo, hi = float(d.quantile(0.05)), float(d.quantile(0.95))
    ex = metrics["fixed"]["cost_0.3%"]["annual_excess_vs_universe"]
    c1 = bool(abs(ic["t"]) < 2)
    c2 = bool(lo <= ex <= hi)
    return {"ic_ok": c1, "ic_mean_abs_below_0.005_reference": bool(abs(ic["mean"]) < 0.005), "random_range": [lo, hi],
            "annual_excess": ex, "random_ok": c2, "no_leak_found": c1 and c2}


def stop_loss_stats(m: Market, res: Result) -> dict:
    """損切りの報告（候補2。評価役のフェーズ4計画の指摘 中1）：当たった件数、窓を開けた割合、ストップ安に張り付いて
    その日に売れなかった件数、売った価格とその日の終値の差、張り付いた銘柄を後日の始値で売った価格と逆指値の価格の差。"""
    ev_ = res.meta.get("stop_events")
    tr = res.trades
    if ev_ is None or not len(ev_):
        return {"stops": 0, "buys": int((tr["side"] == "buy").sum()) if len(tr) else 0}
    sold = tr[tr["reason"].isin(["stop_loss", "stop_loss_gap"])]
    t = sold["date"].map(m.date_index).to_numpy()
    close = m.C[t, sold["j"].to_numpy()]
    retry = tr[tr["reason"] == "stop_loss_retry"]
    stuck = ev_[ev_["stuck"]]
    out = {"stops": int(len(ev_)), "gap": int(ev_["gap"].sum()), "gap_ratio": float(ev_["gap"].mean()),
           "stuck_limit_down": int(len(stuck)), "sold_same_day": int(len(sold)), "sold_later_open": int(len(retry)),
           "close_vs_sell_price_mean": float(np.nanmean(close / sold["price"].to_numpy() - 1)) if len(sold) else None,
           "buys": int((tr["side"] == "buy").sum())}
    if len(retry):
        # 売った日より前で最も新しい、同じ銘柄の張り付いた日の逆指値の価格（分割・併合の調整前。件数が少ないので目安）
        stops = [stuck[(stuck["j"] == j) & (stuck["date"] < d)]["stop"].iloc[-1] for d, j in zip(retry["date"], retry["j"])]
        out["later_open_vs_stop_mean"] = float(np.nanmean(retry["price"].to_numpy() / np.array(stops) - 1))
    return out


def delisting_of_orders(m: Market, orders: pd.DataFrame) -> dict:
    from src.analysis.phase2_baselines import delisting_stats, order_picks
    d = delisting_stats(m, order_picks(m, orders), DELIST_WINDOW)
    d.pop("examples", None)
    return d
