"""ベースライン（CLAUDE.md 第5章）。すべて同じ売買ルール・資金制約で計算する。

1. ランダム：予測日ごとに、ユニバースの銘柄に乱数で順位を付ける
2. モメンタム：過去20営業日の上昇率が高い順
3. リバーサル：過去5営業日の下落率が大きい順
4. ユニバース全体の等金額平均：30万円では全銘柄を買えないため、資金制約は適用できない。
   売買のタイミング（寄付の指値で買い、最終営業日の引けで売る）と約定の判定は同じにし、
   約定しなかった銘柄の分は現金（0%）として等金額で平均する（DECISIONS.md）
   売りの判定もエンジンと同じ：ストップ安で引けた・売買不成立の銘柄は、次に寄った日の始値で売る（その前に上場廃止に
   なれば最後の終値）。その週のリターンに含める（評価役のフェーズ2の指摘4）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.execution import close_sell_filled, floor_to_tick, tick_size
from src.backtest.market import Market
from src.backtest.target import Week


def past_return(m: Market, window: int) -> np.ndarray:
    """t の終値の、window 営業日前の終値に対するリターン（分割調整済み。売買不成立の日は直前の終値）。"""
    q = m.Qc_ff
    out = np.full(q.shape, np.nan)
    out[window:] = q[window:] / q[:-window] - 1
    return out


class ArrayScore:
    def __init__(self, arr: np.ndarray):
        self.arr = arr

    def __call__(self, t: int) -> np.ndarray:
        return self.arr[t]


def momentum(m: Market, window: int = 20) -> ArrayScore:
    return ArrayScore(past_return(m, window))


def reversal(m: Market, window: int = 5) -> ArrayScore:
    return ArrayScore(-past_return(m, window))


class RandomScore:
    """呼ばれるたびに新しい乱数で順位を付ける（シードを固定すれば同じ結果になる）。"""

    def __init__(self, n_codes: int, seed: int):
        self.n = n_codes
        self.rng = np.random.default_rng(seed)

    def __call__(self, t: int) -> np.ndarray:
        return self.rng.random(self.n)


_floor_vec = np.vectorize(floor_to_tick, otypes=[float])


def _exit_price(m: Market, w: Week, j: int) -> tuple[float, float]:
    """エンジンと同じ売りの判定での (調整後の売値, 調整前の売値)。最終日に売れなければ、次に寄った日の始値。"""
    t = w.last
    if close_sell_filled(m.C[t, j], m.L[t, j], bool(m.LL[t, j])):
        return m.C[t, j] / m.cumF[t, j], m.C[t, j]
    for k in range(t + 1, len(m.dates)):
        if m.last_listed[j] < k:
            last = m.last_listed[j]
            px = pd.Series(m.C[: last + 1, j]).ffill().iloc[-1]
            return px / m.cumF[last, j], px
        if m.O[k, j] == m.O[k, j]:
            return m.O[k, j] / m.cumF[k, j], m.O[k, j]
    # データの最終日まで売れない：最終日の評価額（直前の終値）
    px = pd.Series(m.C[:, j]).ffill().iloc[-1]
    return px / m.cumF[-1, j], px


def universe_average(m: Market, universe: np.ndarray, weeks: list[Week], limit_up_pct: float,
                     cost: float, cost_model: str = "flat") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ユニバースの等金額平均の週次リターン（コストなし・コストあり）と、指値で約定した割合。

    予測日のユニバースの各銘柄を、翌週の最初の営業日の寄付で指値で買い（同じ約定の判定）、
    最終営業日の引けで売ったときのリターン（売りの判定はエンジンと同じ。_exit_price）。約定しなかった分は0%。
    営業日が1日しかない週は買わない（0%）。cost_model が "tick" なら片道 = max(cost, 1呼値 ÷ 約定価格)。
    """
    def rate(px: np.ndarray) -> np.ndarray:
        if cost_model == "tick":
            ticks = np.array([tick_size(x) if x == x else np.nan for x in px], dtype=float)
            return np.maximum(cost, ticks / px)
        return np.full(len(px), cost)

    qo = m.Qo
    gross = np.zeros(len(weeks))
    net = np.zeros(len(weeks))
    fill = np.full(len(weeks), np.nan)
    for k, w in enumerate(weeks):
        if len(w.days) < 2 or w.pred < 0:
            continue
        js = np.flatnonzero(universe[w.pred])
        if len(js) == 0:
            continue
        lim = _floor_vec(m.C[w.pred, js] * (1 + limit_up_pct))
        o = m.O[w.first, js]
        with np.errstate(invalid="ignore"):
            ok = (o < lim - 1e-9) & (np.abs(m.adj[w.first, js] - 1) < 1e-9)
        exit_q = np.full(len(js), np.nan)
        exit_px = np.full(len(js), np.nan)
        for i in np.flatnonzero(ok):
            exit_q[i], exit_px[i] = _exit_price(m, w, int(js[i]))
        with np.errstate(invalid="ignore"):
            ret = np.where(ok, exit_q / qo[w.first, js] - 1, 0.0)
            ret = np.nan_to_num(ret, nan=0.0)
            c_buy, c_sell = rate(o), rate(exit_px)
            gross[k] = ret.mean()
            net[k] = np.where(ok, (1 + ret) * (1 - c_sell) / (1 + c_buy) - 1, 0.0).mean()
        fill[k] = ok.mean()
    return gross, net, fill
