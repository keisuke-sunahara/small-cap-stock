"""学習用データの組み立てと、ウォークフォワード検証の期間の分割（CLAUDE.md 第5章「検証方法」）。

学習用データ（daily_panel）
- 行：各営業日 t（起点）× その日のユニバースの銘柄
- 特徴量：t の値（t の引け後に入手できる情報だけ。src.features）。normalize = "rank" なら、日ごとにユニバース内の
  順位（0〜1）に直す（外れ値と、相場全体の水準の違いの影響を除くため）。NaN は NaN のまま
- 目的変数：t の翌営業日の始値から t+5 の終値までのリターンの、t のユニバース内での順位（0〜1。src.backtest.target）
  bad_event（src.backtest.universe.bad_factor_events。株価の動きと合わない調整係数の日）を渡すと、
  t+2 〜 t+horizon にその日がある行の目的変数を NaN にし、順位の計算からも外す（2026-10-04 承認、案A）。
  学習と評価では、目的変数が NaN の行を使わない

ウォークフォワードの分割（walk_forward_splits）
- 評価期間を暦の月ごとに区切り、月ごとに学習し直す（retrain_frequency: monthly）
- 評価の予測日は、その月に入る週の予測日（前の週の最終営業日）
- 学習に使う起点 t は、学習の開始日以降で、t ≤ e0 − gap − 1（e0 = その月の最初の予測日の位置）。
  window_years があれば、さらに e0 の日付の window_years 年前（暦）以降に限る（直近4年のローリング。
  2026-10-04 承認の PLAN.md。運用時に Light の5年分のデータで同じ再学習ができるようにするため）
  t の目的変数は t+horizon の終値で決まるので、gap ≥ horizon なら e0 の時点で答えが分かっているものだけを使う。
  さらに「学習期間の終わりと評価期間の始まりの間に gap 営業日以上の空白」（CLAUDE.md）を満たす
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.market import Market
from src.backtest.target import cross_section_rank, forward_return, make_weeks


@dataclass
class Panel:
    t: np.ndarray          # 起点の日付の位置 [行]
    j: np.ndarray          # 銘柄の位置 [行]
    X: np.ndarray          # 特徴量 [行, 特徴量]（float32）
    y: np.ndarray          # 目的変数（順位 0〜1。データの外にはみ出す起点は NaN）[行]
    names: list[str]


def normalize(arr: np.ndarray, universe: np.ndarray, how: str) -> np.ndarray:
    if how == "none":
        return np.where(universe, arr, np.nan)
    if how == "rank":
        return cross_section_rank(arr, universe & np.isfinite(arr))
    raise ValueError(f"normalize は none / rank: {how}")


def label_spans_bad_event(bad_event: np.ndarray, horizon: int) -> np.ndarray:
    """起点 t の目的変数の期間（t+1 の始値 → t+horizon の終値）の間に bad_event の日がある [T, N]。

    t+1 の係数は始値と終値の両方に同じように効くので、見るのは t+2 〜 t+horizon。
    """
    T = bad_event.shape[0]
    c = np.cumsum(bad_event, axis=0)
    out = np.zeros(bad_event.shape, dtype=bool)
    if T > horizon:
        out[: T - horizon] = (c[horizon:] - c[1: T - horizon + 1]) > 0
    return out


def daily_panel(m: Market, universe: np.ndarray, features: dict[str, np.ndarray], horizon: int,
                how: str = "rank", t_index: np.ndarray | None = None,
                bad_event: np.ndarray | None = None) -> Panel:
    """t_index の各日（省略時は全営業日）のユニバースの行。"""
    names = list(features)
    t_index = np.arange(len(m.dates)) if t_index is None else np.asarray(t_index)
    sub = universe[t_index]
    ti, jj = np.nonzero(sub)
    X = np.empty((len(ti), len(names)), dtype=np.float32)
    for k, n in enumerate(names):
        X[:, k] = normalize(features[n][t_index], sub, how)[ti, jj]
    label_ok = universe if bad_event is None else universe & ~label_spans_bad_event(bad_event, horizon)
    y = cross_section_rank(forward_return(m, horizon), label_ok)[t_index][ti, jj]
    return Panel(t=t_index[ti], j=jj, X=X, y=y.astype(np.float32), names=names)


@dataclass(frozen=True)
class Split:
    month: str                 # "YYYY-MM"
    train_t: np.ndarray        # 学習に使う起点の日付の位置
    eval_preds: np.ndarray     # 評価する予測日の位置（週の予測日）


def walk_forward_splits(dates: np.ndarray, train_start: str, eval_start: str, eval_end: str,
                        gap: int, horizon: int, window_years: int | None = None) -> list[Split]:
    """eval_start〜eval_end に最初の営業日がある週の予測日を、月ごとにまとめて分割する。"""
    if gap < horizon:
        raise ValueError(f"gap（{gap}）は目的変数の期間（{horizon}）以上にする")
    weeks = [w for w in make_weeks(dates)
             if w.pred >= 0 and eval_start <= dates[w.first] and dates[w.last] <= eval_end]
    by_month: dict[str, list[int]] = {}
    for w in weeks:
        by_month.setdefault(str(dates[w.pred])[:7], []).append(w.pred)
    t0 = int(np.searchsorted(dates, train_start))
    out = []
    for month, preds in sorted(by_month.items()):
        e0 = min(preds)
        last = e0 - gap - 1
        first = t0
        if window_years is not None:
            lower = (pd.Timestamp(str(dates[e0])) - pd.DateOffset(years=window_years)).date().isoformat()
            first = max(t0, int(np.searchsorted(dates, lower)))
        out.append(Split(month=month, train_t=np.arange(first, last + 1), eval_preds=np.array(sorted(preds))))
    return out


def splits_frame(dates: np.ndarray, splits: list[Split]) -> pd.DataFrame:
    """分割の一覧（確認用）。"""
    return pd.DataFrame([{"month": s.month, "train_first": dates[s.train_t[0]], "train_last": dates[s.train_t[-1]],
                          "eval_first_pred": dates[s.eval_preds[0]], "eval_last_pred": dates[s.eval_preds[-1]],
                          "n_eval_weeks": len(s.eval_preds)} for s in splits])
