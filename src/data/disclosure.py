"""開示情報の利用開始日（CLAUDE.md 第5章）。

開示日時がその日の大引けより前なら、その日の引け後の予測から使う。大引け以降（ちょうど大引けの時刻を含む）、
または休日の開示なら、次の営業日の引け後の予測から使う。大引けは 2024-11-05 以降は 15:30、それより前は 15:00。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def close_time(disc_date: pd.Series, cfg: dict) -> pd.Series:
    m = cfg["market"]
    return pd.Series(np.where(disc_date.astype(str) >= m["close_time_change_date"],
                              m["close_time_after"], m["close_time_before"]), index=disc_date.index)


def available_index(disc_date: pd.Series, disc_time: pd.Series, bdays: list[str], cfg: dict) -> np.ndarray:
    """各開示を最初に使える営業日の、bdays の中の位置（len(bdays) なら期間内では使えない）。"""
    days = np.asarray(bdays)
    d = disc_date.astype(str).to_numpy()
    hhmm = disc_time.astype(str).str[:5].to_numpy()
    before_close = hhmm < close_time(disc_date, cfg).to_numpy()
    left = np.searchsorted(days, d, side="left")
    is_bday = (left < len(days)) & (days[np.minimum(left, len(days) - 1)] == d)
    right = np.searchsorted(days, d, side="right")
    return np.where(is_bday & before_close, left, right)
