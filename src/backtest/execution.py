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
