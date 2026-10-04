"""バックテストの売買ルールの再現のテスト（ダミーデータ）。"""
import copy

import numpy as np
import pandas as pd
import pytest

from src.backtest.baselines import ArrayScore, past_return, universe_average
from src.backtest.engine import Backtester, Rules
from src.backtest.target import forward_return, make_weeks
from src.backtest.universe import avg_turnover, universe_mask
from tests.dummy_market import CFG, dummy_dates, dummy_market

NAN = float("nan")


def run(m, score, cfg=CFG, **rule_kw):
    adv = avg_turnover(m, cfg["universe"]["turnover_window"])
    u = universe_mask(m, cfg, adv)
    bt = Backtester(m, u, adv, Rules.from_config(cfg, **rule_kw))
    return bt.run(ArrayScore(score), m.dates[0], m.dates[-1])


def const_score(m, order):
    """銘柄コード → 点数（全日同じ）。"""
    s = np.zeros((len(m.dates), len(m.codes)))
    for j, c in enumerate(m.codes):
        s[:, j] = order[c]
    return s


def test_make_weeks_and_one_day_week():
    # 2024-01-15 の週は月曜だけ営業日にする
    dates = dummy_dates("2024-01-08", 11, holidays=("2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19"))
    weeks = make_weeks(dates)
    assert [len(w.days) for w in weeks] == [5, 1, 5]
    assert weeks[1].pred == 4 and weeks[2].pred == 5


def test_buy_at_open_below_limit_and_sell_at_close():
    dates = dummy_dates("2024-01-08", 10)
    c = [500.0] * 9 + [600.0]
    o = [500.0] * 5 + [505.0] + [500.0] * 4
    m = dummy_market(dates, {"11110": c}, {"11110": o})
    res = run(m, const_score(m, {"11110": 1}))
    # 指値 = 500 × 1.02 = 510、予算10万円 → 100株、始値505で買い、終値600で売り
    assert res.orders["shares"].tolist() == [100]
    assert res.orders["limit"].tolist() == [510]
    assert res.orders["filled"].tolist() == [True]
    assert res.equity[9] == pytest.approx(100_000 - 50_500 + 60_000)
    assert res.weekly["ret"].iloc[-1] == pytest.approx(0.095)


def test_open_equal_to_limit_is_not_filled_and_cost_is_charged():
    dates = dummy_dates("2024-01-08", 10)
    m = dummy_market(dates, {"11110": [500.0] * 10}, {"11110": [500.0] * 5 + [510.0] + [500.0] * 4})
    res = run(m, const_score(m, {"11110": 1}))
    assert res.orders["filled"].tolist() == [False]
    assert res.equity[9] == 100_000
    assert not res.weekly["invested"].iloc[-1]
    # コスト：片道0.3%
    m2 = dummy_market(dates, {"11110": [500.0] * 10})
    cfg = copy.deepcopy(CFG)
    cfg["cost"]["one_way"] = 0.003
    res2 = run(m2, const_score(m2, {"11110": 1}), cfg)
    assert res2.equity[9] == pytest.approx(100_000 - 50_000 * 1.003 + 50_000 * 0.997)


def test_one_day_week_does_not_buy():
    dates = dummy_dates("2024-01-08", 11, holidays=("2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19"))
    m = dummy_market(dates, {"11110": [500.0] * 11})
    res = run(m, const_score(m, {"11110": 1}))
    w = res.weekly.set_index("first")
    assert w.loc["2024-01-15", "n_orders"] == 0
    assert w.loc["2024-01-22", "n_filled"] == 1


def test_market_halt_on_buy_day_means_no_fill():
    dates = dummy_dates("2024-01-08", 10)
    c = [500.0] * 5 + [NAN] + [500.0] * 4
    m = dummy_market(dates, {"11110": c}, {"11110": c})
    res = run(m, const_score(m, {"11110": 1}))
    assert res.orders["filled"].tolist() == [False]


def test_unaffordable_and_turnover_cap_go_to_next_rank():
    dates = dummy_dates("2024-01-08", 10)
    m = dummy_market(dates, {"11110": [2000.0] * 10, "22220": [400.0] * 10, "33330": [300.0] * 10})
    # 22220：指値408円、予算10万円 → 200株（81,600円）。平均売買代金5,000万円の1% = 50万円以下
    m.Va[:, 1] = 5e7
    res = run(m, const_score(m, {"11110": 3, "22220": 2, "33330": 1}))
    assert res.orders["code"].tolist() == ["22220"]     # 11110 は100株買えない（2,040円 × 100 > 10万円）
    assert res.orders["shares"].tolist() == [200]
    cfg = copy.deepcopy(CFG)
    cfg["order"]["max_order_to_adv"] = 0.001             # 上限 5万円 → 22220 は除外
    res2 = run(m, const_score(m, {"11110": 3, "22220": 2, "33330": 1}), cfg)
    assert res2.orders["code"].tolist() == ["33330"]


def test_sell_failure_at_limit_down_sells_next_open():
    dates = dummy_dates("2024-01-08", 15)
    c = [500.0] * 9 + [400.0] + [380.0] * 5
    o = [500.0] * 10 + [390.0] + [380.0] * 4
    m = dummy_market(dates, {"11110": c}, {"11110": o})
    m.L[9, 0] = 400.0
    m.LL[9, 0] = True                                   # 金曜にストップ安で引けた
    score = const_score(m, {"11110": 1})
    score[9:, 0] = np.nan                               # 翌週は選ばれない
    res = run(m, score)
    sells = res.trades[res.trades["side"] == "sell"]
    assert sells["reason"].tolist() == ["open_retry"]
    assert sells["date"].tolist() == [dates[10]]
    assert res.equity[14] == pytest.approx(100_000 - 50_000 + 39_000)


def test_continuation_modes():
    # 15営業日（3週）。11110 は木曜の時点で1位、金曜の引け後は2位になる週がある
    dates = dummy_dates("2024-01-08", 15)
    m = dummy_market(dates, {"11110": [500.0] * 15, "22220": [400.0] * 15})
    score = const_score(m, {"11110": 2, "22220": 1})
    score[9, 0] = 0                                     # 2週目の金曜の引け後だけ 11110 が下がる
    cfg = copy.deepcopy(CFG)
    cfg["cost"]["one_way"] = 0.003
    none = run(m, score, cfg, continuation="none")
    prev = run(m, score, cfg, continuation="prev_day")
    same = run(m, score, cfg, continuation="same_day")
    # none：毎週売って買い直す（2週 × 往復）
    assert (none.trades["side"] == "buy").sum() == 2
    # prev_day：木曜（2週目）の順位で 11110 が上位 → 売らずに持ち越す。3週目は買わない
    assert (prev.trades["side"] == "buy").sum() == 1
    assert (prev.trades["side"] == "sell").sum() == 0
    # same_day：2週目の金曜の引け後の順位で 11110 が外れる → 売り、その順位で1位の 22220 を3週目に買う。
    # 3週目の金曜は 11110 が1位に戻るため 22220 も売る
    assert (same.trades["side"] == "sell").sum() == 2
    assert same.trades[same.trades["side"] == "buy"]["code"].tolist() == ["11110", "22220"]
    assert prev.equity[14] > none.equity[14]


def test_split_while_holding_keeps_value():
    dates = dummy_dates("2024-01-08", 10)
    c = [500.0] * 7 + [250.0, 250.0, 260.0]
    m = dummy_market(dates, {"11110": c})
    m.adj[7, 0] = 0.5                                   # 水曜の朝に2分割
    res = run(m, const_score(m, {"11110": 1}))
    assert res.equity[7] == pytest.approx(100_000)
    assert res.equity[9] == pytest.approx(100_000 + 200 * 10)


def test_delisting_sells_at_last_close():
    dates = dummy_dates("2024-01-08", 10)
    c = [500.0] * 7 + [450.0, NAN, NAN]
    m = dummy_market(dates, {"11110": c, "22220": [100.0] * 10})
    m.last_listed[0] = 7                                # 水曜が最終売買日
    m.common[8:, 0] = False
    res = run(m, const_score(m, {"11110": 2, "22220": 1}))
    sells = res.trades[res.trades["side"] == "sell"]
    assert sells["reason"].tolist() == ["delisted"]
    assert res.equity[9] == pytest.approx(100_000 - 5_000)


def test_no_future_information_in_decisions():
    """予測日 p の注文・ユニバース・点数は、p より後のデータを消しても変わらない。"""
    rng = np.random.default_rng(0)
    dates = dummy_dates("2024-01-08", 30)
    close = {f"{k}0": list(300 + rng.normal(0, 10, 30).cumsum()) for k in range(1111, 1121)}
    m = dummy_market(dates, close)
    cfg = CFG
    p = 19
    cut = m.truncated(p)
    for mm in (m, cut):
        mm._adv = avg_turnover(mm, cfg["universe"]["turnover_window"])
        mm._u = universe_mask(mm, cfg, mm._adv)
    assert np.array_equal(m._u[: p + 1], cut._u)
    assert np.allclose(past_return(m, 5)[: p + 1], past_return(cut, 5), equal_nan=True)
    orders = []
    for mm in (m, cut):
        bt = Backtester(mm, mm._u, mm._adv, Rules.from_config(cfg))
        orders.append(bt.build_orders(p, set(), 1, 100_000, 100_000, ArrayScore(past_return(mm, 5))))
    assert orders[0] == orders[1] and len(orders[0]) == 1
    # 目的変数は未来を使う（p の目的変数は p+1〜p+5 で決まる）。データの外の起点は NaN
    fr = forward_return(m, 5)
    assert np.isnan(fr[-5:]).all() and not np.isnan(fr[: -5]).any()


def test_universe_average_uses_same_fill_rule():
    dates = dummy_dates("2024-01-08", 10)
    m = dummy_market(dates, {"11110": [500.0] * 9 + [550.0], "22220": [100.0] * 10},
                     {"11110": [500.0] * 10, "22220": [100.0] * 5 + [102.0] + [100.0] * 4})
    adv = avg_turnover(m, 2)
    u = universe_mask(m, CFG, adv)
    weeks = [w for w in make_weeks(m.dates) if w.pred >= 0]
    gross, net, fill = universe_average(m, u, weeks, 0.02, 0.003)
    # 11110 は約定して +10%、22220 は始値 = 指値で約定しない（0%）
    assert gross[0] == pytest.approx(0.05)
    assert fill[0] == pytest.approx(0.5)
    assert net[0] == pytest.approx((1.1 * 0.997 / 1.003 - 1) / 2)


def test_fixed_budget_mode_keeps_budget_after_losses():
    # 2週目に半値になって損をした後、3週目の予算が縮むか（min_equity）、固定のままか（fixed）
    dates = dummy_dates("2024-01-08", 15)
    c = [300.0] * 5 + [300.0] * 4 + [150.0] + [150.0] * 4 + [160.0]
    m = dummy_market(dates, {"11110": c})
    score = const_score(m, {"11110": 1})
    base = run(m, score, budget_mode="min_equity")
    fixed = run(m, score, budget_mode="fixed")
    # 2週目：指値306円・300株（91,800円）。終値150円で売り → 総資産 100,000 − 45,000 = 55,000
    assert base.orders["shares"].tolist() == [300, 300]   # 3週目：予算 55,000 → 指値153円で300株
    assert fixed.orders["shares"].tolist() == [300, 600]  # 3週目：予算 100,000 → 600株
    # weekly の1行目が2週目（1週目は予測日が無いため対象外）
    assert base.weekly["ret"].iloc[0] == pytest.approx(-0.45)
    assert fixed.weekly["ret"].iloc[0] == pytest.approx(-0.45)
    assert base.weekly["ret"].iloc[1] == pytest.approx(300 * 10 / 55_000)
    assert fixed.weekly["ret"].iloc[1] == pytest.approx(600 * 10 / 100_000)
    assert fixed.curve[-1] == pytest.approx(0.55 * 1.06)


def test_continuation_budget_values_holdings_at_decision_day_after_consolidation():
    """最終日の朝に併合が効く保有銘柄を、継続の判断（前日の順位・基本ルールの予算）で正しく評価する（評価役の指摘3）。

    N=1。2週目に 11110 を900円で100株。最終日（金曜）の朝に10株→1株の併合（株価9,000円）。
    木曜の順位は 22220（500円）が1位。木曜の総資産は 10,000 + 100株 × 900円 = 100,000円なので予算は10万円で、
    22220 は買える → 上位1に入る → 11110 は継続せず売る。
    併合後の株数（10株）× 木曜の併合前の株価（900円）で評価すると総資産が19,000円になり、22220 が買えないと判定されて
    11110 を誤って持ち越す。
    """
    dates = dummy_dates("2024-01-08", 15)
    a = [900.0] * 9 + [9000.0] * 6
    m = dummy_market(dates, {"11110": a, "22220": [500.0] * 15})
    m.adj[9, 0] = 10.0
    m.cumF = np.cumprod(m.adj, axis=0)
    score = const_score(m, {"11110": 2, "22220": 1})
    score[8:, 0] = 0                                    # 木曜から 11110 は2位
    res = run(m, score, continuation="prev_day", budget_mode="min_equity")
    sells = res.trades[res.trades["side"] == "sell"]
    assert sells["code"].tolist() == ["11110"]
    assert sells["date"].tolist() == [dates[9]]
    assert sells["shares"].tolist() == [pytest.approx(10.0)]
    assert res.equity[9] == pytest.approx(100_000)      # 併合の前後で資産は変わらない（コスト0）


def test_tick_cost_model():
    """cost_model="tick"：片道 = max(cost, 1呼値 ÷ 約定価格)。80円の株は1円 ÷ 80円 = 1.25%。"""
    dates = dummy_dates("2024-01-08", 10)
    m = dummy_market(dates, {"11110": [80.0] * 10})
    cfg = copy.deepcopy(CFG)
    cfg["cost"]["one_way"] = 0.003
    flat = run(m, const_score(m, {"11110": 1}), cfg)
    tick = run(m, const_score(m, {"11110": 1}), cfg, cost_model="tick")
    # 指値 81円、予算10万円 → 1,200株（97,200円）。始値80円で買い、終値80円で売る
    assert flat.equity[9] == pytest.approx(100_000 - 96_000 * 0.003 * 2)
    assert tick.equity[9] == pytest.approx(100_000 - 96_000 * (1 / 80) * 2)
    assert tick.weekly["cost_paid"].iloc[-1] == pytest.approx(96_000 * (1 / 80) * 2)
    assert tick.weekly["traded_value"].iloc[-1] == pytest.approx(96_000 * 2)
    # 高い株では一律のコストと同じ（2,000円の株の1呼値は 1円 ÷ 2,000円 = 0.05% < 0.3%）
    from src.backtest.engine import Rules as R
    r = R.from_config(cfg, cost_model="tick")
    assert r.cost_rate(2000.0) == pytest.approx(0.003)
    assert r.cost_rate(150.0) == pytest.approx(1 / 150)


def test_universe_average_sells_next_open_after_limit_down_close():
    """ユニバース平均も、ストップ安で引けた銘柄は次に寄った日の始値で売る（評価役の指摘4。エンジンと同じ）。"""
    dates = dummy_dates("2024-01-08", 15)
    c = [500.0] * 9 + [400.0] + [380.0] * 5
    o = [500.0] * 10 + [390.0] + [380.0] * 4
    m = dummy_market(dates, {"11110": c}, {"11110": o})
    m.L[9, 0] = 400.0
    m.LL[9, 0] = True
    adv = avg_turnover(m, 2)
    u = universe_mask(m, CFG, adv)
    weeks = [w for w in make_weeks(m.dates) if w.pred >= 0]
    gross, net, fill = universe_average(m, u, weeks, 0.02, 0.0)
    assert gross[0] == pytest.approx(390 / 500 - 1)
    # エンジンの結果と一致する
    res = run(m, const_score(m, {"11110": 1}))
    assert res.equity[14] == pytest.approx(100_000 - 50_000 + 39_000)


def test_breakeven_one_way_cost_makes_excess_zero():
    from src.backtest.metrics import annualized, breakeven_one_way_cost
    rng = np.random.default_rng(1)
    dates = dummy_dates("2024-01-08", 60)
    close = {f"{k}0": list(300 + rng.normal(0, 8, 60).cumsum()) for k in range(1111, 1117)}
    m = dummy_market(dates, close)
    score = past_return(m, 5)
    res = run(m, score, budget_mode="fixed")
    sp = str(m.dates[make_weeks(m.dates)[1].pred])
    bench = pd.Series(np.full(len(res.weekly), 0.001))
    c = breakeven_one_way_cost(res.weekly, CFG["capital"]["initial_capital_jpy"], bench, start_pred=sp)
    assert c is not None
    cfg = copy.deepcopy(CFG)
    cfg["cost"]["one_way"] = c
    res_c = run(m, score, cfg, budget_mode="fixed")
    last = res_c.weekly["last"].iloc[-1]
    assert annualized(res_c.weekly["ret"], sp, last) - annualized(bench, sp, last) == pytest.approx(0, abs=1e-9)
