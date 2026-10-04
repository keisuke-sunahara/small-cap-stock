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


# ---- 係数が株価の動きと合わない銘柄の除外（2026-10-04 承認、案A） ----

def cfg_bad(days: int = 5) -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["universe"]["bad_adjfactor_threshold"] = 0.3
    cfg["universe"]["bad_adjfactor_exclude_days"] = days
    return cfg


def test_bad_factor_is_excluded_from_the_day_it_is_known():
    from src.backtest.universe import bad_factor_events
    dates = dummy_dates("2024-01-08", 20)
    nan = float("nan")
    # ダミー：11110 は位置5に正しい2分割（500円 → 250円）。22220 は位置5に係数0.6が付くが株価は −5%（誤り）。
    # 33330 は売買不成立の位置8に係数0.5が付き、位置9に約定（400円 → 380円。誤り）
    close = {"11110": [500.0] * 5 + [250.0] * 15,
             "22220": [500.0] * 5 + [475.0] * 15,
             "33330": [400.0] * 8 + [nan] + [380.0] * 11}
    m = dummy_market(dates, close)
    m.adj[5, 0] = 0.5
    m.adj[5, 1] = 0.6
    m.adj[8, 2] = 0.5
    m.__post_init__()   # cumF を作り直す
    known, bad_event = bad_factor_events(m, 0.3)
    assert np.flatnonzero(known.any(axis=0)).tolist() == [1, 2]
    assert np.flatnonzero(known[:, 1]).tolist() == [5] and np.flatnonzero(known[:, 2]).tolist() == [9]
    assert np.flatnonzero(bad_event[:, 2]).tolist() == [8] and not bad_event[:, 0].any()
    adv = avg_turnover(m, 2)
    u = universe_mask(m, cfg_bad(5), adv)
    base = universe_mask(m, CFG, adv)
    assert np.array_equal(u[:, 0], base[:, 0])
    assert not u[5:10, 1].any() and u[10:, 1].all() and u[1:5, 1].all()
    assert not u[9:14, 2].any() and u[14:, 2].all()
    # 設定が無ければ除かない（フェーズ2の結果の再現用）
    assert base[5:10, 1].all()


def test_bad_factor_uses_only_data_up_to_the_date():
    """位置 p までのユニバースは、p より後のデータを切っても変わらない。"""
    dates = dummy_dates("2024-01-08", 20)
    m = dummy_market(dates, {"22220": [500.0] * 5 + [475.0] * 15, "11110": [300.0] * 20})
    m.adj[5, m.code_index["22220"]] = 0.6
    m.__post_init__()
    assert not universe_mask(m, cfg_bad(5), avg_turnover(m, 2))[5:10, m.code_index["22220"]].any()
    adv = avg_turnover(m, 2)
    full = universe_mask(m, cfg_bad(5), adv)
    for p in (4, 5, 7):
        cut = m.truncated(p)
        assert np.array_equal(full[: p + 1], universe_mask(cut, cfg_bad(5), avg_turnover(cut, 2)))


def test_labels_spanning_bad_factor_are_dropped():
    from src.backtest.universe import bad_factor_events
    from src.models.dataset import daily_panel, label_spans_bad_event
    dates = dummy_dates("2024-01-08", 20)
    m = dummy_market(dates, {"22220": [500.0] * 10 + [475.0] * 10, "11110": [300.0] * 20,
                             "33330": [200.0] * 20})
    b, ok = m.code_index["22220"], m.code_index["11110"]
    m.adj[10, b] = 0.6
    m.__post_init__()
    _, bad_event = bad_factor_events(m, 0.3)
    span = label_spans_bad_event(bad_event, 5)
    # 起点 t の期間は t+1 の始値 → t+5 の終値。位置10の係数が効くのは t = 5〜8（t = 9 は t+1 の始値から調整済み）
    assert np.flatnonzero(span[:, b]).tolist() == [5, 6, 7, 8]
    assert not span[:, ok].any()
    u = np.ones(m.C.shape, dtype=bool)
    p = daily_panel(m, u, {"x": np.zeros(m.C.shape)}, 5, bad_event=bad_event)
    y = {(t, j): v for t, j, v in zip(p.t, p.j, p.y)}
    assert all(np.isnan(y[(t, b)]) for t in range(5, 9))
    assert not np.isnan(y[(4, b)]) and not np.isnan(y[(9, b)])
    assert not np.isnan(y[(6, ok)])
