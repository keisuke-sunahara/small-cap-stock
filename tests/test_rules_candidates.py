"""比較する売買ルールの候補（EXP-006〜016）の売買の再現のテスト（ダミーデータ）。"""
import copy

import numpy as np
import pandas as pd
import pytest

from src.backtest.baselines import ArrayScore
from src.backtest.engine import Backtester, Rules, allocation_weights
from src.backtest.rules import EarningsCalendar, market_ok, rank_return_trend
from src.backtest.target import make_weeks
from src.backtest.universe import avg_turnover, universe_mask
from tests.dummy_market import CFG, dummy_dates, dummy_market


def run(m, score, cfg=CFG, offset=0, extras=None, **rule_kw):
    adv = avg_turnover(m, cfg["universe"]["turnover_window"])
    u = universe_mask(m, cfg, adv)
    bt = Backtester(m, u, adv, Rules.from_config(cfg, **rule_kw), **(extras or {}))
    return bt.run(score if callable(score) else ArrayScore(score), m.dates[0], m.dates[-1], offset=offset)


def const_score(m, order):
    s = np.zeros((len(m.dates), len(m.codes)))
    for j, c in enumerate(m.codes):
        s[:, j] = order[c]
    return s


def cfg_n(n, capital):
    cfg = copy.deepcopy(CFG)
    cfg["capital"] = {"n_holdings": n, "initial_capital_jpy": capital}
    return cfg


# ---- 候補1 地合いフィルター ----
def test_market_ok_moving_average():
    topix = np.array([100.0, 100, 100, 90, np.nan, 120])
    ok = market_ok(topix, 3)
    # 3日目までは平均が無い → True。4日目 90 < 96.7 → False。5日目は前の値（90）でつなぐ → 90 < 93.3 → False
    assert ok.tolist() == [True, True, True, False, False, True]


def test_market_filter_no_buy_and_sell_at_continuation():
    dates = dummy_dates("2024-01-08", 20)
    m = dummy_market(dates, {"11110": [500.0] * 20})
    score = const_score(m, {"11110": 1})
    ok = np.ones(20, dtype=bool)
    ok[4] = False                                # 1週目の金曜（2週目の予測日）は TOPIX が平均の下 → 2週目は買わない
    res = run(m, score, CFG, extras={"market_ok": ok}, continuation="prev_day")
    w = res.weekly.set_index("first")
    assert w.loc[dates[5], "n_orders"] == 0 and w.loc[dates[5], "skip_reason"] == "market_filter"
    assert w.loc[dates[10], "n_filled"] == 1     # 3週目は買う
    # 継続の判断の日（4週目の木曜）が平均の下 → 継続せずに4週目の金曜の引けで売る
    ok2 = np.ones(20, dtype=bool)
    ok2[13] = False                              # 3週目の木曜
    res2 = run(m, score, CFG, extras={"market_ok": ok2}, continuation="prev_day")
    sells = res2.trades[res2.trades["side"] == "sell"]
    assert sells["date"].tolist() == [dates[14]] and sells["reason"].tolist() == ["close"]
    buys = res2.trades[res2.trades["side"] == "buy"]
    assert buys["date"].tolist() == [dates[5], dates[15]]   # 継続しなかったので次の週に買い直す


# ---- 候補2 損切り ----
def test_stop_loss_intraday_gap_and_buy_day():
    dates = dummy_dates("2024-01-08", 15)
    c = [500.0] * 15
    m = dummy_market(dates, {"11110": c, "22220": [400.0] * 15, "33330": [300.0] * 15})
    cfg = cfg_n(3, 300_000)
    # 11110：2週目の火曜に安値455（逆指値 460 に届く）。22220：火曜に窓を開けて 360 で始まる。33330：買った日に安値 270
    m.O[5, 0] = 500.0
    m.L[6, 0] = 455.0
    m.O[6, 1] = 360.0
    m.L[6, 1] = 350.0
    m.L[5, 2] = 270.0
    res = run(m, const_score(m, {"11110": 3, "22220": 2, "33330": 1}), cfg, stop_loss=0.08, continuation="prev_day")
    st = res.trades[res.trades["reason"].str.startswith("stop")].set_index("code")
    assert st.loc["11110", "price"] == 460.0 and st.loc["11110", "reason"] == "stop_loss"
    assert st.loc["22220", "price"] == 360.0 and st.loc["22220", "reason"] == "stop_loss_gap"
    assert st.loc["33330", "price"] == 276.0 and st.loc["33330", "date"] == dates[5]   # 300 × 0.92 = 276
    assert res.weekly.loc[0, "n_stop"] == 3
    # 損切りの後、その週は現金のまま（代わりの銘柄は買わない）
    assert (res.trades["side"] == "buy").groupby(res.trades["date"]).sum().get(dates[6], 0) == 0


def test_stop_loss_on_last_day_counts_as_held_for_continuation():
    # N=2、予算5万円。A は買った後に値上がりして木曜には100株買えない価格になり、金曜に損切りされる。
    # 木曜の順位：A、C、B。A は木曜の時点では保有していたので「上位2」に数える → C までで2つ → B は継続しない
    dates = dummy_dates("2024-01-08", 15)
    a = [400.0] * 5 + [400.0, 450.0, 520.0, 600.0, 300.0] + [300.0] * 5
    m = dummy_market(dates, {"11110": a, "22220": [300.0] * 15, "33330": [350.0] * 15})
    m.O[9, 0] = 600.0
    m.L[9, 0] = 300.0
    score = const_score(m, {"11110": 3, "22220": 2, "33330": 1})
    score[8, :] = [3, 1, 2]                      # 木曜の順位：A、C、B
    cfg = cfg_n(2, 100_000)
    res = run(m, score, cfg, stop_loss=0.08, continuation="prev_day")
    fri = res.trades[res.trades["date"] == dates[9]].set_index("code")
    assert fri.loc["11110", "reason"] == "stop_loss" and fri.loc["11110", "price"] == 368.0
    assert fri.loc["22220", "reason"] == "close"           # B は継続しない


# ---- 候補3 決算の回避 ----
def sched_frame(rows):
    df = pd.DataFrame(rows, columns=["j", "avail", "PubDate", "FYE", "FQName", "SchDate"])
    df["SchDate"] = pd.to_datetime(df["SchDate"])
    return df


def test_earnings_calendar_known_and_undecided():
    dates = dummy_dates("2024-01-08", 20)
    sched = sched_frame([
        (0, 2, dates[1], "2024-03-31", "3Q", "2024-01-31"),
        (0, 6, dates[5], "2024-03-31", "3Q", "2024-01-25"),   # 同じ四半期の変更 → 最新の 1/25
        (1, 3, dates[2], "2024-03-31", "3Q", None),           # 「未定」
        (2, 3, dates[2], "2024-03-31", "3Q", "2024-01-19"),
        (2, 8, dates[7], "2024-03-31", "3Q", None),           # 予定日が「未定」に変わった
    ])
    st = pd.DataFrame({"j": [1], "avail": [12]})              # 1 の決算短信を 12 から使える
    cal = EarningsCalendar(sched, st, 3, dates)
    assert cal.mask(4, "2024-01-29", "2024-02-02", include_undecided=False).tolist() == [True, False, False]
    assert cal.mask(6, "2024-01-29", "2024-02-02", include_undecided=False).tolist() == [False, False, False]
    assert cal.mask(6, "2024-01-22", "2024-01-26", include_undecided=False).tolist() == [True, False, False]
    assert cal.mask(1, "2024-01-01", "2024-12-31").tolist() == [False, False, False]   # まだ使えない
    assert cal.mask(5, "2024-01-15", "2024-01-19").tolist() == [False, True, True]
    assert cal.mask(9, "2024-01-15", "2024-01-19", include_undecided=False).tolist() == [False, False, False]
    assert cal.mask(9, "2030-01-01", "2030-01-02").tolist() == [False, True, True]    # 「未定」の状態
    assert cal.mask(12, "2030-01-01", "2030-01-02").tolist() == [False, False, True]  # 決算短信が出た → 「未定」でない


def test_earnings_avoid_in_engine():
    dates = dummy_dates("2024-01-08", 15)
    m = dummy_market(dates, {"11110": [500.0] * 15, "22220": [400.0] * 15})
    score = const_score(m, {"11110": 2, "22220": 1})
    calls = []

    def avoid(d, lo, hi):
        calls.append((d, lo, hi))
        out = np.zeros(2, dtype=bool)
        if d == 4:
            out[0] = True                        # 2週目の予測日：11110 は保有期間中に決算
        return out

    res = run(m, score, CFG, extras={"avoid": avoid}, continuation="prev_day")
    assert res.orders["code"].tolist()[:1] == ["22220"]
    assert (4, dates[5], dates[9]) in calls                          # 予測日：その週の最初〜最後の営業日
    assert (8, dates[9], dates[14]) in calls                         # 継続の判断：最終営業日〜翌週の最終営業日


def test_earnings_avoid_sells_held_at_continuation():
    dates = dummy_dates("2024-01-08", 15)
    m = dummy_market(dates, {"11110": [500.0] * 15, "22220": [400.0] * 15})
    score = const_score(m, {"11110": 2, "22220": 1})

    def avoid(d, lo, hi):
        return np.array([d == 8, False])         # 2週目の木曜：11110 は次の保有期間に決算

    res = run(m, score, CFG, extras={"avoid": avoid}, continuation="prev_day")
    sells = res.trades[res.trades["side"] == "sell"]
    assert sells["code"].tolist() == ["11110"] and sells["date"].tolist() == [dates[9]]


# ---- 候補4 指値なし ----
def test_market_order_buys_above_limit():
    dates = dummy_dates("2024-01-08", 10)
    o = [500.0] * 5 + [530.0] + [500.0] * 4      # 指値 510 を超えて始まる
    m = dummy_market(dates, {"11110": [500.0] * 10}, {"11110": o})
    score = const_score(m, {"11110": 1})
    assert run(m, score).orders["filled"].tolist() == [False]
    res = run(m, score, buy_order="market")
    assert res.orders["filled"].tolist() == [True]
    assert res.trades.iloc[0]["price"] == 530.0 and res.trades.iloc[0]["reason"] == "open_market"
    m.UL[5, 0] = True
    m.H[5, 0] = 530.0                            # ストップ高で寄った
    assert run(m, score, buy_order="market").orders["filled"].tolist() == [False]


# ---- 候補6・7 配分 ----
def test_allocation_weights():
    r = Rules.from_config(cfg_n(3, 300_000), weighting="rank", rank_weights=[0.4, 0.35, 0.25])
    score = np.array([1.0, 3.0, 2.0, np.nan])
    assert allocation_weights(r, [0, 1, 2], score, None) == {1: 0.4, 2: 0.35, 0: 0.25}
    assert allocation_weights(r, [3, 0], score, None) == {0: 0.4, 3: 0.35}      # NaN は最後
    rv = Rules.from_config(cfg_n(3, 300_000), weighting="inverse_vol")
    vol = np.array([0.01, 0.02, 0.04, np.nan])
    w = allocation_weights(rv, [0, 1, 2], score, vol)
    assert w[0] == 0.5                                           # 4/7 → 上限 0.5（超えた分は現金）
    assert w[1] == pytest.approx(2 / 7) and w[2] == pytest.approx(1 / 7)
    w2 = allocation_weights(rv, [1, 3], score, vol)                 # 3 は値動きの大きさが無い → 1/3
    assert w2[3] == pytest.approx(1 / 3) and w2[1] == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        Rules.from_config(cfg_n(2, 300_000), weighting="rank", rank_weights=[0.4, 0.35, 0.25])


def test_rank_weighting_resizes_new_orders():
    dates = dummy_dates("2024-01-08", 10)
    m = dummy_market(dates, {"11110": [100.0] * 10, "22220": [100.0] * 10, "33330": [100.0] * 10})
    cfg = cfg_n(3, 300_000)
    res = run(m, const_score(m, {"11110": 3, "22220": 2, "33330": 1}), cfg, budget_mode="fixed",
              weighting="rank", rank_weights=[0.4, 0.35, 0.25])
    # 指値 102 円。予算 30万円 × 40%・35%・25% → 1,100株・1,000株・700株（100株単位で切り下げ）
    assert dict(zip(res.orders["code"], res.orders["shares"])) == {"11110": 1100, "22220": 1000, "33330": 700}
    # 売買代金の上限：注文金額 ≤ 平均売買代金の1%。22220 の売買代金を下げると、上限まで減らし、残りは現金
    cfg["universe"]["min_avg_turnover_jpy"] = 1e6
    m.Va[:, 1] = 1e7                             # 上限 10万円：均等の予算（900株・91,800円）では選ばれ、35%（1,000株）では超える
    res2 = run(m, const_score(m, {"11110": 3, "22220": 2, "33330": 1}), cfg, budget_mode="fixed",
               weighting="rank", rank_weights=[0.4, 0.35, 0.25])
    assert dict(zip(res2.orders["code"], res2.orders["shares"]))["22220"] == 900


def test_inverse_vol_requires_vol_and_uses_it():
    dates = dummy_dates("2024-01-08", 10)
    m = dummy_market(dates, {"11110": [100.0] * 10, "22220": [100.0] * 10})
    cfg = cfg_n(2, 200_000)
    with pytest.raises(ValueError):
        run(m, const_score(m, {"11110": 2, "22220": 1}), cfg, weighting="inverse_vol")
    vol = np.tile([0.01, 0.03], (10, 1))
    res = run(m, const_score(m, {"11110": 2, "22220": 1}), cfg, extras={"vol": vol}, budget_mode="fixed",
              weighting="inverse_vol")
    # 配分 0.75 → 上限 0.5（10万円 → 900株）、0.25（5万円 → 400株）
    assert dict(zip(res.orders["code"], res.orders["shares"])) == {"11110": 900, "22220": 400}


# ---- 候補9 保有期間 ----
def test_holding_weeks_two_and_offsets():
    dates = dummy_dates("2024-01-08", 30)
    m = dummy_market(dates, {"11110": [500.0] * 30, "22220": [400.0] * 30})
    score = const_score(m, {"11110": 2, "22220": 1})
    score[13, 0] = 0                             # 1回目の保有期間の2週目の木曜は 11110 が2位 → 継続しない
    called = []

    class Recorder:
        def __call__(self, t):
            called.append(t)
            return score[t]

    res = run(m, Recorder(), CFG, continuation="prev_day", holding_weeks=2)
    w = res.weekly
    # 週 = 2024-01-15〜（予測日のある週）。0週目に買い、1週目の最終営業日に継続の判断と売り。2週目に買い直す
    assert w["skip_reason"].tolist()[:4] == ["", "idle", "", "idle"]
    assert sorted(set(called)) == [4, 13, 14, 23]   # 点数を使うのは買いの週の予測日と、売りの週の継続の判断の日だけ
    buys = res.trades[res.trades["side"] == "buy"]
    assert buys["date"].tolist() == [dates[5], dates[15]]
    assert res.trades[res.trades["side"] == "sell"]["date"].tolist() == [dates[14]]
    res1 = run(m, score, CFG, continuation="prev_day", holding_weeks=2, offset=1)
    assert res1.weekly["skip_reason"].tolist()[:3] == ["idle", "", "idle"]
    assert res1.weekly.loc[0, "ret"] == 0
    with pytest.raises(ValueError):
        run(m, score, CFG, holding_weeks=2, offset=2)


def test_holding_weeks_continuation_decided_on_second_week():
    dates = dummy_dates("2024-01-08", 30)
    m = dummy_market(dates, {"11110": [500.0] * 30, "22220": [400.0] * 30})
    score = const_score(m, {"11110": 2, "22220": 1})
    score[18, 0] = 0                             # 間の週の木曜（判断しない日）に 11110 が2位になっても売らない
    score[23, 0] = 0                             # 2回目の保有期間の2週目の木曜に 11110 が2位 → 継続しない
    res = run(m, score, CFG, continuation="prev_day", holding_weeks=2)
    sells = res.trades[res.trades["side"] == "sell"]
    assert sells["date"].tolist() == [dates[24]]
    buys = res.trades[res.trades["side"] == "buy"]
    assert buys["date"].tolist() == [dates[5], dates[25]]


# ---- 候補6・7 の事前の確認 ----
def test_rank_return_trend_detects_ordering():
    rng = np.random.default_rng(1)
    dates = dummy_dates("2024-01-08", 60)
    n = 12
    close = {f"{1000 + k}0": [300.0] * 60 for k in range(n)}
    m = dummy_market(dates, close)
    weeks = [w for w in make_weeks(m.dates) if w.pred >= 0]
    score = np.zeros((60, n))
    for w in weeks:
        s = rng.permutation(n).astype(float)
        score[w.pred] = s
        # 点数の高い銘柄ほど翌週に上がる（ばらつきを加える）
        for j in range(n):
            m.C[w.last, j] = 300.0 * (1 + 0.01 * s[j] + rng.normal(0, 0.03))
    m.__post_init__()
    adv = avg_turnover(m, CFG["universe"]["turnover_window"])
    u = universe_mask(m, CFG, adv)
    bt = Backtester(m, u, adv, Rules.from_config(cfg_n(3, 300_000)))
    out = rank_return_trend(bt, m, weeks, ArrayScore(score), 100_000)
    assert out["trend"] and out["mean"] < -0.3
    out2 = rank_return_trend(bt, m, weeks, ArrayScore(-score), 100_000)
    assert not out2["trend"]
