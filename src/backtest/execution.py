"""約定判定と株数の計算（CLAUDE.md 第5章の基本の売買ルール）。純粋な関数だけを置き、テストで確認する。"""
from __future__ import annotations

import math

# 東証の呼値の単位（TOPIX100 構成銘柄以外。2014-01-14 以降）。(この価格以下, 呼値)
# 時価総額500億円以下のユニバースには TOPIX100 構成銘柄は入らないため、この表だけを使う（DECISIONS.md）
TICK_TABLE = [
    (3_000, 1), (5_000, 5), (30_000, 10), (50_000, 50), (300_000, 100), (500_000, 500),
    (3_000_000, 1_000), (5_000_000, 5_000), (30_000_000, 10_000), (50_000_000, 50_000),
]
TICK_ABOVE = 100_000
EPS = 1e-9


def tick_size(price: float) -> int:
    for upper, tick in TICK_TABLE:
        if price <= upper + EPS:
            return tick
    return TICK_ABOVE


def floor_to_tick(price: float) -> float:
    """price 以下で、呼値の単位に合う最大の価格。"""
    tick = tick_size(price)
    return math.floor(price / tick + EPS) * tick


def limit_price(prev_close: float, limit_up_pct: float) -> float:
    """買いの指値 = 前営業日の終値 × (1 + 上限乖離率) を呼値の単位に合わせて切り下げた価格。"""
    return floor_to_tick(prev_close * (1 + limit_up_pct))


def order_shares(budget: float, price: float, lot: int) -> int:
    """予算内で lot 単位で買える最大の株数（price は指値。買えなければ 0）。"""
    if not (price > 0) or not (budget > 0):
        return 0
    return int(math.floor(budget / (price * lot) + EPS)) * lot


def within_turnover_cap(shares: int, price: float, adv: float, max_ratio: float) -> bool:
    """注文金額が、過去の平均売買代金の max_ratio 以下か。"""
    return adv > 0 and shares * price <= max_ratio * adv + EPS


def buy_filled(open_price: float, limit: float, adj_factor: float = 1.0) -> bool:
    """寄付の指値の買いが約定したか。

    - 寄らなかった（始値が NaN：ストップ高で寄らず、売買停止、終日の売買停止など）→ 約定しない
    - 始値 ≥ 指値 → 約定しない（同じ値も、全部は約定しないことがあるため約定しないとする）
    - 買う日の朝に分割・併合が効く（adj_factor ≠ 1）→ 約定しない。分割前の価格で出した指値は
      新しい値幅制限の外になり失効するため（DECISIONS.md）
    """
    if open_price != open_price:  # NaN
        return False
    if abs(adj_factor - 1.0) > EPS:
        return False
    return open_price < limit - EPS


def close_sell_filled(close: float, low: float, limit_down_flag: bool) -> bool:
    """引成の売りが約定したか。終値が無い（売買不成立）か、ストップ安で引けた（ストップ安のフラグがあり、
    終値がその日の安値）場合は、売れなかったものとする（次に売買が成立した日の始値で売る）。"""
    if close != close:
        return False
    if limit_down_flag and close <= low + EPS:
        return False
    return True


def open_sell_filled(open_price: float) -> bool:
    """売れ残りを始値で売る。寄らなかった日は売れない。"""
    return open_price == open_price


def market_buy_filled(open_price: float, high: float, limit_up_flag: bool, adj_factor: float = 1.0) -> bool:
    """寄付の成行の買いが約定したか（比較する売買ルールの候補4。EXP-009 の plan.md）。

    - 寄らなかった（始値が NaN）→ 約定しない
    - 買う日の朝に分割・併合が効く → 約定しない（注文の株数が分割前の基準のため。指値と同じ）
    - 始値がストップ高（ストップ高のフラグがあり、始値がその日の高値と同じ）→ 比例配分で買えないことがあるため、約定しない（保守的）
    """
    if open_price != open_price:
        return False
    if abs(adj_factor - 1.0) > EPS:
        return False
    if limit_up_flag and open_price >= high - EPS:
        return False
    return True


def stop_price(buy_price: float, stop_loss: float) -> float:
    """損切りの逆指値の価格 = 買値 × (1 − stop_loss) を呼値の単位に合わせて切り下げた価格（候補2。EXP-007 の plan.md）。"""
    return floor_to_tick(buy_price * (1 - stop_loss))


STOP_FILLS = ("tick_below", "low")


def stop_triggered(open_price: float, low: float, stop: float, bought_today: bool,
                   fill: str = "tick_below") -> tuple[float, bool] | None:
    """損切りの逆指値が当たったときの (約定価格, 窓を開けたか)。当たらなければ None。日足で判定する。

    - 始値が無い日（売買不成立・ストップ安で寄らない）は当たらない（次の営業日に同じ判定をする）
    - 窓を開けて下回った（買った日以外で 始値 ≤ 逆指値の価格）：始値で売る
    - 場中に触れた（安値 ≤ 逆指値の価格。買った日は始値（買値）より後の値動きなので、こちらだけで判定する）：
      逆指値の価格 − 1呼値 で売る（逆指値の成行が、触れた価格より1呼値下で約定するとみなす。評価役のフェーズ4計画の指摘 中1）
    - fill = "low"：参考の計算。どちらの場合も、その日の安値で売れたとする
    - ストップ安で張り付いた日は売れない。この判定は呼び出し側（engine）で行う
    """
    if fill not in STOP_FILLS:
        raise ValueError(f"fill は {STOP_FILLS} のどれか: {fill}")
    if open_price != open_price:
        return None
    if not bought_today and open_price <= stop + EPS:
        return (low if fill == "low" and low == low else open_price), True
    if low == low and low <= stop + EPS:
        return (low if fill == "low" else stop - tick_size(stop)), False
    return None


def limit_price_array(prev_close: "np.ndarray", limit_up_pct: float) -> "np.ndarray":
    """limit_price の配列版（NaN は NaN）。結果は limit_price と同じ（tests/test_execution.py で確認する）。"""
    import numpy as np
    raw = np.asarray(prev_close, dtype=float) * (1 + limit_up_pct)
    tick = np.full(raw.shape, float(TICK_ABOVE))
    for upper, t in reversed(TICK_TABLE):
        tick = np.where(raw <= upper + EPS, float(t), tick)
    with np.errstate(invalid="ignore"):
        return np.floor(raw / tick + EPS) * tick
