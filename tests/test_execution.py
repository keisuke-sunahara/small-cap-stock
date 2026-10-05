"""約定判定・株数計算・呼値・開示の利用開始日のテスト。"""
import math

import numpy as np
import pandas as pd

from src.backtest.execution import (buy_filled, close_sell_filled, floor_to_tick, limit_price, open_sell_filled,
                                    order_shares, tick_size, within_turnover_cap)
from src.data.disclosure import available_index


def test_tick_size_boundaries():
    assert tick_size(3000) == 1
    assert tick_size(3001) == 5
    assert tick_size(5000) == 5
    assert tick_size(5001) == 10
    assert tick_size(30000) == 10
    assert tick_size(30001) == 50
    assert tick_size(60_000_000) == 100_000


def test_limit_price_floors_to_tick():
    assert limit_price(980, 0.02) == 999          # 999.6 → 999
    assert limit_price(1000, 0.02) == 1020
    assert limit_price(2950, 0.02) == 3005        # 3009 → 呼値5円で 3005
    assert limit_price(2941, 0.02) == 2999        # 2999.82 → 2999
    assert limit_price(4950, 0.02) == 5040        # 5049 → 呼値10円で 5040
    assert limit_price(50, 0.02) == 51
    # 浮動小数点の誤差で1円下がらない（100 × 1.02 = 102.00000000000001 など）
    assert limit_price(100, 0.02) == 102
    assert floor_to_tick(3000.0) == 3000


def test_order_shares_uses_lot_and_budget():
    assert order_shares(100_000, 999, 100) == 100
    assert order_shares(100_000, 1000, 100) == 100   # ちょうど予算どおり
    assert order_shares(100_000, 1001, 100) == 0     # 100株も買えない
    assert order_shares(100_000, 333, 100) == 300
    assert order_shares(100_000, 0, 100) == 0
    assert order_shares(100_000, math.nan, 100) == 0


def test_turnover_cap():
    assert within_turnover_cap(100, 999, 10_000_000, 0.01)        # 99,900 ≤ 100,000
    assert not within_turnover_cap(100, 1001, 10_000_000, 0.01)   # 100,100 > 100,000
    assert not within_turnover_cap(100, 500, math.nan, 0.01)
    assert not within_turnover_cap(100, 500, 0, 0.01)


def test_buy_fill_rules():
    assert buy_filled(998, 999)
    assert not buy_filled(999, 999)          # 始値 = 指値は買えなかったものとする
    assert not buy_filled(1000, 999)
    assert not buy_filled(math.nan, 999)     # 寄らず（ストップ高・売買停止・終日停止）
    assert not buy_filled(500, 999, adj_factor=0.5)   # 当日朝に分割


def test_sell_fill_rules():
    assert close_sell_filled(100, 95, False)
    assert not close_sell_filled(math.nan, math.nan, False)   # 売買不成立
    assert not close_sell_filled(95, 95, True)                # ストップ安で引けた
    assert close_sell_filled(97, 95, True)                    # 日中にストップ安をつけたが戻した
    assert open_sell_filled(90)
    assert not open_sell_filled(math.nan)


CFG_MARKET = {"market": {"close_time_change_date": "2024-11-05", "close_time_before": "15:00",
                         "close_time_after": "15:30"}}


def test_disclosure_available_index():
    bdays = ["2024-11-01", "2024-11-05", "2024-11-06", "2024-11-08"]
    df = pd.DataFrame([
        ("2024-11-01", "14:59:00"),   # 15:00 より前 → 当日（0）
        ("2024-11-01", "15:00:00"),   # ちょうど大引け → 翌営業日（1）
        ("2024-11-01", "15:20:00"),   # 2024-11-05 より前は 15:00 が大引け → 翌営業日（1）
        ("2024-11-05", "15:20:00"),   # 2024-11-05 以降は 15:30 が大引け → 当日（1）
        ("2024-11-05", "15:30:00"),   # → 翌営業日（2）
        ("2024-11-03", "10:00:00"),   # 休日 → 次の営業日（1）
        ("2024-11-07", "10:00:00"),   # 営業日の一覧にない日 → 次の営業日（3）
        ("2024-11-08", "18:00:00"),   # 最終日の引け後 → 期間内では使えない（4）
    ], columns=["DiscDate", "DiscTime"])
    got = available_index(df["DiscDate"], df["DiscTime"], bdays, CFG_MARKET)
    assert got.tolist() == [0, 1, 1, 1, 2, 1, 3, 4]


def test_limit_price_array_matches_scalar():
    from src.backtest.execution import limit_price_array
    rng = np.random.default_rng(0)
    prices = np.concatenate([rng.uniform(30, 60_000, 5000), [2941.0, 2942.0, 4901.0, 29411.0, np.nan],
                             np.array([3000, 5000, 30000, 50000]) / 1.02])
    got = limit_price_array(prices, 0.02)
    want = np.array([limit_price(p, 0.02) if p == p else np.nan for p in prices])
    assert np.array_equal(got, want, equal_nan=True)


def test_market_buy_filled():
    from src.backtest.execution import market_buy_filled
    assert market_buy_filled(520.0, 530.0, False)              # 指値（前日の終値 + 2%）を超えて始まっても買える
    assert not market_buy_filled(float("nan"), float("nan"), False)
    assert not market_buy_filled(580.0, 580.0, True)          # ストップ高で寄った（始値 = 高値）
    assert market_buy_filled(570.0, 580.0, True)              # 寄った後にストップ高になった
    assert not market_buy_filled(500.0, 510.0, False, adj_factor=0.5)


def test_stop_price_and_trigger():
    from src.backtest.execution import stop_price, stop_triggered
    assert stop_price(500.0, 0.08) == 460.0
    assert stop_price(3010.0, 0.08) == 2769.0                 # 2769.2 → 1円単位で切り下げ
    assert stop_price(3500.0, 0.08) == 3220.0                 # 3220 は 5円単位
    assert stop_triggered(470.0, 455.0, 460.0, False) == 460.0   # 場中に逆指値の価格に届いた
    assert stop_triggered(450.0, 440.0, 460.0, False) == 450.0   # 窓を開けて下回った → 始値
    assert stop_triggered(450.0, 440.0, 460.0, True) == 460.0    # 買った日は安値だけで判定
    assert stop_triggered(470.0, 461.0, 460.0, False) is None
    assert stop_triggered(float("nan"), float("nan"), 460.0, False) is None   # 寄らない日は売れない
