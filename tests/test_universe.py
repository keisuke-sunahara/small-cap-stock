"""ユニバースの条件のテスト（ダミーデータ）。貸借信用区分「その他」の除外（2026-10-04 承認）。"""
import copy

import numpy as np

from src.backtest.universe import avg_turnover, universe_mask
from tests.dummy_market import CFG, dummy_dates, dummy_market
from tests.test_engine import const_score, run


def cfg_with(exclude: bool) -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["universe"]["exclude_margin_other"] = exclude
    return cfg


def test_margin_other_is_excluded_only_on_flagged_days():
    dates = dummy_dates("2024-01-08", 10)
    m = dummy_market(dates, {"11110": [500.0] * 10, "22220": [400.0] * 10})
    m.margin_other[4:7, 0] = True
    adv = avg_turnover(m, 2)
    on = universe_mask(m, cfg_with(True), adv)
    off = universe_mask(m, cfg_with(False), adv)
    assert not on[4:7, 0].any() and on[7:, 0].all() and on[1:4, 0].all()
    assert off[1:, 0].all()
    assert np.array_equal(on[:, 1], off[:, 1])
    # 設定が無い場合は除外しない（フェーズ2の結果の再現用）
    assert np.array_equal(universe_mask(m, CFG, adv), off)


def test_default_market_has_no_margin_other_flags():
    dates = dummy_dates("2024-01-08", 5)
    m = dummy_market(dates, {"11110": [500.0] * 5})
    assert m.margin_other.shape == m.C.shape and not m.margin_other.any()


def test_held_position_flagged_other_is_sold_and_not_bought_again():
    # 3週。11110 が常に1位。2週目の木曜（判断日）から「その他」になる
    dates = dummy_dates("2024-01-08", 15)
    m = dummy_market(dates, {"11110": [500.0] * 15, "22220": [400.0] * 15})
    m.margin_other[8:, 0] = True
    score = const_score(m, {"11110": 2, "22220": 1})
    res = run(m, score, cfg_with(True), continuation="prev_day")
    sells = res.trades[res.trades["side"] == "sell"]
    assert sells["code"].tolist() == ["11110"]
    assert sells["date"].tolist() == [dates[9]] and sells["reason"].tolist() == ["close"]
    assert res.trades[res.trades["side"] == "buy"]["code"].tolist() == ["11110", "22220"]
    # 除外しなければ持ち続ける
    res_off = run(m, score, cfg_with(False), continuation="prev_day")
    assert (res_off.trades["side"] == "sell").sum() == 0


def test_margin_other_uses_only_data_up_to_the_date():
    """日付 p のユニバースは、p より後の「その他」の印を消しても（未来を切っても）変わらない。"""
    rng = np.random.default_rng(1)
    dates = dummy_dates("2024-01-08", 30)
    close = {f"{k}0": list(300 + rng.normal(0, 10, 30).cumsum()) for k in range(1111, 1121)}
    m = dummy_market(dates, close)
    m.margin_other[rng.random(m.C.shape) < 0.2] = True
    p = 19
    cut = m.truncated(p)
    assert np.array_equal(cut.margin_other, m.margin_other[: p + 1])
    cfg = cfg_with(True)
    u_full = universe_mask(m, cfg, avg_turnover(m, 2))
    u_cut = universe_mask(cut, cfg, avg_turnover(cut, 2))
    assert np.array_equal(u_full[: p + 1], u_cut)
    # p より後の印を変えても p までのユニバースは同じ
    m2 = copy.deepcopy(m)
    m2.margin_other[p + 1:] = ~m2.margin_other[p + 1:]
    assert np.array_equal(universe_mask(m2, cfg, avg_turnover(m2, 2))[: p + 1], u_full[: p + 1])
