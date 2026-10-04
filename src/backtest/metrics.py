"""評価指標（CLAUDE.md 第9章）。週次リターンを主に使う。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

WEEKS_PER_YEAR = 52


def max_drawdown(equity: np.ndarray) -> float:
    e = equity[~np.isnan(equity)]
    if len(e) == 0:
        return float("nan")
    peak = np.maximum.accumulate(e)
    return float((e / peak - 1).min())


def annualized(weekly_ret: pd.Series, first_date: str, last_date: str) -> float:
    growth = float(np.prod(1 + weekly_ret.to_numpy()))
    years = (pd.Timestamp(last_date) - pd.Timestamp(first_date)).days / 365.25
    return growth ** (1 / years) - 1 if years > 0 and growth > 0 else float("nan")


def sharpe(weekly_ret: pd.Series) -> float:
    s = weekly_ret.std(ddof=1)
    return float(weekly_ret.mean() / s * np.sqrt(WEEKS_PER_YEAR)) if s > 0 else float("nan")


def t_stat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def compound_by(ret: pd.Series, key: pd.Series) -> pd.Series:
    return (1 + ret).groupby(key.to_numpy()).prod() - 1


def regime_by_quarter(dates: np.ndarray, topix: np.ndarray, threshold: float = 0.05) -> pd.Series:
    """暦の四半期ごとの TOPIX の騰落率で、相場環境を「上昇・下落・横ばい」に分ける（±threshold）。

    分析用の区分で、売買の判断には使わない（その四半期の終わりまで分からないため）。
    """
    s = pd.Series(topix, index=pd.to_datetime(pd.Series(dates)))
    q = s.groupby(s.index.to_period("Q")).agg(["first", "last"])
    prev_last = q["last"].shift(1).fillna(q["first"])
    r = q["last"] / prev_last - 1
    return r.map(lambda x: "上昇" if x > threshold else ("下落" if x < -threshold else "横ばい"))


def summarize(weekly: pd.DataFrame, equity: np.ndarray, bench_ret: pd.Series, *, start_pred: str,
              tax_rate: float, regime: pd.Series | None = None, orders: pd.DataFrame | None = None) -> dict:
    """weekly：週ごとの ret（戦略）と last（最終営業日）。bench_ret：同じ週のユニバース平均（コストなし）。"""
    ret = weekly["ret"].reset_index(drop=True)
    bench = bench_ret.reset_index(drop=True)
    excess = ret - bench
    year = weekly["last"].str[:4].reset_index(drop=True)
    y_ret = compound_by(ret, year)
    y_bench = compound_by(bench, year)
    out = {
        "weeks": int(len(ret)),
        "period": [start_pred, str(weekly["last"].iloc[-1])],
        "annual_return": annualized(ret, start_pred, weekly["last"].iloc[-1]),
        "total_return": float(np.prod(1 + ret) - 1),
        "sharpe": sharpe(ret),
        "max_drawdown": max_drawdown(equity),
        "worst_week": float(ret.min()),
        "best_week": float(ret.max()),
        "mean_weekly_excess": float(excess.mean()),
        "excess_t": t_stat(excess),
        "annual_excess_vs_universe": annualized(ret, start_pred, weekly["last"].iloc[-1])
        - annualized(bench, start_pred, weekly["last"].iloc[-1]),
        "cash_week_ratio": float((~weekly["invested"]).mean()) if "invested" in weekly else None,
        "by_year": {y: {"return": float(y_ret[y]), "universe": float(y_bench[y]),
                        "excess": float(y_ret[y] - y_bench[y]),
                        "after_tax_return": float(y_ret[y] * (1 - tax_rate) if y_ret[y] > 0 else y_ret[y])}
                    for y in y_ret.index},
    }
    out["years_with_positive_excess"] = int(sum(v["excess"] > 0 for v in out["by_year"].values()))
    out["years"] = len(out["by_year"])
    if orders is not None and len(orders):
        out["orders"] = int(len(orders))
        out["fill_rate"] = float(orders["filled"].mean())
    if regime is not None:
        q = pd.to_datetime(weekly["last"]).dt.to_period("Q").reset_index(drop=True)
        reg = q.map(regime)
        out["by_regime"] = {k: {"weeks": int((reg == k).sum()),
                                "mean_weekly": float(ret[reg == k].mean()),
                                "mean_weekly_excess": float(excess[reg == k].mean())}
                            for k in ["上昇", "横ばい", "下落"] if (reg == k).any()}
    return out


def weekly_rank_ic(scores: np.ndarray, realized: np.ndarray, universe_rows: np.ndarray) -> np.ndarray:
    """週ごとの Rank IC（予測日の点数と、翌週の実現リターンの順位相関）。scores・realized・universe_rows は [週, N]。"""
    out = np.full(len(scores), np.nan)
    for k in range(len(scores)):
        ok = universe_rows[k] & ~np.isnan(scores[k]) & ~np.isnan(realized[k])
        if ok.sum() >= 10:
            out[k] = stats.spearmanr(scores[k][ok], realized[k][ok]).statistic
    return out
