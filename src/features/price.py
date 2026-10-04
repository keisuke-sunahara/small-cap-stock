"""第1段階の特徴量（値動き）。株価は分割調整済み（Market.Qc・Qc_ff）を使う。

- 位置 t の値は、t の終値までのデータだけで決まる
- 売買不成立の日は、直前の終値（Qc_ff）でつなぐ。上場してから必要な日数に満たない銘柄は NaN
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.registry import register


def _ret(fd, window: int) -> np.ndarray:
    q = fd.m.Qc_ff
    out = np.full(q.shape, np.nan)
    out[window:] = q[window:] / q[:-window] - 1
    return out


@register("ret_5d", "price", "過去5営業日の上昇率（t−5 の終値 → t の終値）")
def ret_5d(fd) -> np.ndarray:
    return _ret(fd, 5)


@register("ret_20d", "price", "過去20営業日の上昇率")
def ret_20d(fd) -> np.ndarray:
    return _ret(fd, 20)


@register("ret_60d", "price", "過去60営業日の上昇率")
def ret_60d(fd) -> np.ndarray:
    return _ret(fd, 60)


@register("vol_ratio_5_60", "price",
          "出来高の変化率：log((過去5営業日の平均出来高 + 1) ÷ (過去60営業日の平均出来高 + 1))。出来高は分割調整済み")
def vol_ratio_5_60(fd) -> np.ndarray:
    v = pd.DataFrame(fd.m.Vq)
    v5 = v.rolling(5, min_periods=5).mean()
    v60 = v.rolling(60, min_periods=60).mean()
    return np.log((v5 + 1) / (v60 + 1)).to_numpy()


@register("volatility_20d", "price", "値動きの大きさ：過去20営業日の日次の対数リターンの標準偏差")
def volatility_20d(fd) -> np.ndarray:
    lr = pd.DataFrame(np.log(fd.m.Qc_ff)).diff()
    return lr.rolling(20, min_periods=20).std().to_numpy()


@register("dist_high_60d", "price", "直近高値からの距離：t の終値 ÷ 過去60営業日（t を含む）の高値の最大 − 1")
def dist_high_60d(fd) -> np.ndarray:
    m = fd.m
    high = pd.DataFrame(m.H / m.cumF).rolling(60, min_periods=1).max().to_numpy()
    # 60営業日の間、上場していたことを条件にする（Qc_ff の60日前が NaN なら NaN）
    q = m.Qc_ff
    listed60 = np.full(q.shape, False)
    listed60[59:] = ~np.isnan(q[:-59])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = q / high - 1
    return np.where(listed60, out, np.nan)


@register("log_turnover_20d", "price", "売買代金：log(過去20営業日の平均売買代金 + 1)")
def log_turnover_20d(fd) -> np.ndarray:
    va = pd.DataFrame(fd.m.Va).rolling(20, min_periods=20).mean().to_numpy()
    return np.log1p(va)
