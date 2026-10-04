"""週の区切りと目的変数。

- 週：東証の営業日を、月曜始まりの暦の週（ISO 週）でまとめる。予測日は前の週の最終営業日
- 学習用の目的変数（日次起点）：起点 t の翌営業日の始値から、t の5営業日後の終値までのリターン（分割調整済み）
  の、t のユニバース内での順位（0〜1）。終値が無い日は直前の約定の終値を使う（上場廃止なら最後の終値）
- 週次の実現リターン：週の最初の営業日の始値から、最終営業日の終値まで（評価・Rank IC 用）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.market import Market


@dataclass(frozen=True)
class Week:
    days: tuple[int, ...]   # 週の営業日の位置
    pred: int               # 予測日（前の週の最終営業日）の位置。期間の先頭の週は -1

    @property
    def first(self) -> int:
        return self.days[0]

    @property
    def last(self) -> int:
        return self.days[-1]


def make_weeks(dates: np.ndarray) -> list[Week]:
    d = pd.to_datetime(pd.Series(dates))
    iso = d.dt.isocalendar()
    key = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()
    weeks: list[Week] = []
    start = 0
    for i in range(1, len(dates) + 1):
        if i == len(dates) or key[i] != key[start]:
            weeks.append(Week(days=tuple(range(start, i)), pred=start - 1))
            start = i
    return weeks


def forward_return(m: Market, horizon: int) -> np.ndarray:
    """起点 t の翌営業日の始値 → t+horizon の終値 のリターン [T, N]。データの外にはみ出す起点は NaN。"""
    T = len(m.dates)
    qo = m.Qo
    qc = m.Qc_ff
    out = np.full(qo.shape, np.nan)
    if T > horizon:
        out[: T - horizon] = qc[horizon:] / qo[1: T - horizon + 1] - 1
    return out


def cross_section_rank(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """各日、mask の銘柄の中での順位（0〜1、同順位は平均）。mask 外・NaN は NaN。"""
    v = np.where(mask, values, np.nan)
    return pd.DataFrame(v).rank(axis=1, pct=True).to_numpy()


def weekly_realized(m: Market, weeks: list[Week]) -> np.ndarray:
    """各週の、最初の営業日の始値 → 最終営業日の終値 のリターン [週の数, N]。"""
    qo = m.Qo
    qc = m.Qc_ff
    return np.array([qc[w.last] / qo[w.first] - 1 for w in weeks])
