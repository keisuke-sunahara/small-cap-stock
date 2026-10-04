"""ユニバース（対象銘柄）の判定。各日付の値は、その日の引けまでのデータだけで決まる。

条件（config/base.yaml の universe）
- 普通株としてその日に上場している
- 時価総額（開示済みの発行済株式数 × その日の終値）が上限以下。発行済株式数がまだ開示されていない銘柄は除く
- 過去 turnover_window 営業日（その日を含む）の平均売買代金が下限以上。売買不成立の日は0として平均する。
  上場してから turnover_window 営業日に満たない銘柄は除く
- その日の終値が下限以上。その日に売買が成立していない銘柄は除く（翌営業日の指値を決められないため）
- exclude_margin_other が true なら、その日の銘柄一覧で貸借信用区分が「その他（3）」の銘柄を除く
  （監理・整理銘柄、TOB後の銘柄などを、その時点の情報で外す。翌営業日の一覧は使わない。2026-10-04 承認）
- bad_adjfactor_threshold・bad_adjfactor_exclude_days があれば、調整係数のある日をまたぐ調整後リターンが
  ±threshold を超えた銘柄を、それが分かった日（係数のある日以降で最初に売買が成立した日）から exclude_days 営業日除く
  （係数が実際の株価の動きと合わないものを外すため。判定はその日の終値と係数だけで決まる。2026-10-04 承認、
  reports/phase1_review.md 3.3 の案A）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

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


def bad_factor_events(m: Market, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """調整係数が株価の動きと合わないとみなす日。

    戻り値
    - known [T, N]：係数のある日（売買不成立の日を含む）のあとで最初に売買が成立した日 t に、直前に売買が成立した日の
      終値からの調整後リターンが ±threshold を超えたら True。t の終値と、t までの係数だけで決まる
    - bad_event [T, N]：その判定の対象になった係数のある日（直前の約定から t までの間の、係数が 1 でない日）
    """
    T, N = m.C.shape
    traded = ~np.isnan(m.C)
    is_event = np.abs(m.adj - 1.0) > 1e-12
    n_events = np.cumsum(is_event, axis=0)
    # 各日の「直前に売買が成立した日」までの係数の数と、その日の調整後の終値
    prev_events = np.full((T, N), np.nan)
    prev_events[1:] = np.where(traded[:-1], n_events[:-1], np.nan)
    prev_events = pd.DataFrame(prev_events).ffill().to_numpy()
    prev_q = np.full((T, N), np.nan)
    prev_q[1:] = m.Qc_ff[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = m.Qc / prev_q - 1
        spans_event = n_events - np.nan_to_num(prev_events, nan=0.0) > 0
        known = traded & spans_event & ~np.isnan(prev_events) & (np.abs(ret) > threshold)
    # 判定の対象の係数の日：known の日から遡って、直前に売買が成立した日の翌日まで
    bad_event = np.zeros((T, N), dtype=bool)
    for t, j in zip(*np.nonzero(known)):
        k = t
        while k >= 0 and (k == t or not traded[k, j]):
            if is_event[k, j]:
                bad_event[k, j] = True
            k -= 1
    return known, bad_event


def bad_factor_exclusion(m: Market, cfg: dict) -> np.ndarray | None:
    """設定があれば、bad_factor_events の known の日から exclude_days 営業日（その日を含む）True。無ければ None。"""
    u = cfg["universe"]
    if u.get("bad_adjfactor_threshold") is None:
        return None
    known, _ = bad_factor_events(m, u["bad_adjfactor_threshold"])
    c = np.cumsum(known, axis=0)
    days = int(u["bad_adjfactor_exclude_days"])
    before = np.zeros_like(c)
    before[days:] = c[:-days]
    return (c - before) > 0


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
    bad = bad_factor_exclusion(m, cfg)
    if bad is not None:
        mask &= ~bad
    return mask & ~np.isnan(m.C) & ~np.isnan(mcap) & ~np.isnan(adv)
