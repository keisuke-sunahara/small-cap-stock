"""ユニバース（対象銘柄）の判定。各日付の値は、その日の引けまでのデータだけで決まる。

条件（config/base.yaml の universe）
- 普通株としてその日に上場している
- 時価総額（開示済みの発行済株式数 × その日の終値）が上限以下。発行済株式数がまだ開示されていない銘柄は除く
- 過去 turnover_window 営業日（その日を含む）の平均売買代金が下限以上。売買不成立の日は0として平均する。
  上場してから turnover_window 営業日に満たない銘柄は除く
- その日の終値が下限以上。その日に売買が成立していない銘柄は除く（翌営業日の指値を決められないため）
- exclude_margin_other が true なら、その日の銘柄一覧で貸借信用区分が「その他（3）」の銘柄を除く
  （監理・整理銘柄、TOB後の銘柄などを、その時点の情報で外す。翌営業日の一覧は使わない。2026-10-04 承認）
"""
from __future__ import annotations

import numpy as np

from src.backtest.market import Market


def rolling_sum(a: np.ndarray, window: int) -> np.ndarray:
    """縦方向（日付）の過去 window 行の合計。window 行に満たない先頭は NaN。"""
    c = np.cumsum(np.nan_to_num(a, nan=0.0), axis=0, dtype=float)
    out = np.full(a.shape, np.nan)
    out[window - 1] = c[window - 1]
    out[window:] = c[window:] - c[:-window]
    return out


def avg_turnover(m: Market, window: int) -> np.ndarray:
    """過去 window 営業日の平均売買代金。window 日すべてで普通株として上場していない場合は NaN。"""
    adv = rolling_sum(m.Va, window) / window
    full = rolling_sum(m.common.astype(float), window) == window
    return np.where(full, adv, np.nan)


def market_cap(m: Market) -> np.ndarray:
    return m.shares * m.C


def universe_mask(m: Market, cfg: dict, adv: np.ndarray | None = None) -> np.ndarray:
    u = cfg["universe"]
    if adv is None:
        adv = avg_turnover(m, u["turnover_window"])
    mcap = market_cap(m)
    with np.errstate(invalid="ignore"):
        mask = (m.common
                & (m.C >= u["min_price_jpy"])
                & (mcap <= u["max_market_cap_jpy"])
                & (adv >= u["min_avg_turnover_jpy"]))
    if u.get("exclude_margin_other", False):
        mask &= ~m.margin_other
    return mask & ~np.isnan(m.C) & ~np.isnan(mcap) & ~np.isnan(adv)
