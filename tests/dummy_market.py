"""テスト用のダミーの Market（本物の株価ではない）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.market import Market


def dummy_dates(start: str, n: int, holidays: tuple[str, ...] = ()) -> np.ndarray:
    days = [d.date().isoformat() for d in pd.bdate_range(start, periods=n + len(holidays))]
    return np.array([d for d in days if d not in holidays][:n])


def dummy_market(dates: np.ndarray, close: dict[str, list[float]], open_: dict[str, list[float]] | None = None,
                 va: float = 1e9, shares: float = 1e6) -> Market:
    """close・open_ は銘柄コード → 日ごとの値（NaN は売買不成立）。高値・安値は始値と終値から作る。"""
    codes = np.array(sorted(close))
    T, N = len(dates), len(codes)
    C = np.array([close[c] for c in codes], dtype=float).T
    O = np.array([(open_ or close)[c] for c in codes], dtype=float).T
    H = np.fmax(O, C)
    L = np.fmin(O, C)
    Va = np.where(np.isnan(C), 0.0, va)
    Vo = np.where(np.isnan(C), 0.0, va / np.nan_to_num(C, nan=1.0))
    return Market(dates=dates, codes=codes, O=O, H=H, L=L, C=C, Va=Va, Vo=Vo, adj=np.ones((T, N)),
                  UL=np.zeros((T, N), bool), LL=np.zeros((T, N), bool), common=np.ones((T, N), bool),
                  shares_base=np.full((T, N), shares), last_listed=np.full(N, T - 1),
                  topix=np.full(T, 1000.0))


CFG = {
    "capital": {"n_holdings": 1, "initial_capital_jpy": 100_000},
    "order": {"limit_up_pct": 0.02, "lot_size": 100, "max_order_to_adv": 0.01, "adv_window": 20},
    "cost": {"one_way": 0.0},
    "backtest": {"continuation": "none"},
    "universe": {"min_price_jpy": 50, "max_market_cap_jpy": 5e10, "min_avg_turnover_jpy": 3e7,
                 "turnover_window": 2},
}
