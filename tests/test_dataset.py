"""学習用データとウォークフォワードの分割のテスト（ダミーデータ）。"""
import numpy as np
import pandas as pd
import pytest

from src.backtest.target import forward_return
from src.models.dataset import daily_panel, normalize, walk_forward_splits
from tests.dummy_market import dummy_dates, dummy_market


def test_walk_forward_gap_and_no_overlap():
    dates = dummy_dates("2024-01-01", 200)
    splits = walk_forward_splits(dates, "2024-01-01", "2024-04-01", "2024-09-30", gap=5, horizon=5)
    assert [s.month for s in splits][:2] == ["2024-03", "2024-04"]   # 4/1 の週の予測日は 3/29
    for s in splits:
        e0 = s.eval_preds.min()
        t_end = s.train_t.max()
        assert e0 - t_end - 1 >= 5                    # 5営業日以上の空白
        assert t_end + 5 < e0                         # 学習の目的変数は評価の最初の予測日より前に確定
        assert s.train_t.min() == 0
        assert all(str(dates[p])[:7] == s.month for p in s.eval_preds)
    # 予測日は重複も抜けもない
    allp = np.concatenate([s.eval_preds for s in splits])
    assert len(allp) == len(set(allp.tolist()))
    with pytest.raises(ValueError):
        walk_forward_splits(dates, "2024-01-01", "2024-04-01", "2024-09-30", gap=3, horizon=5)


def test_daily_panel_rows_target_and_rank_normalization():
    rng = np.random.default_rng(0)
    dates = dummy_dates("2024-01-08", 30)
    close = {f"{k}0": list(300 + rng.normal(0, 10, 30).cumsum()) for k in range(1111, 1116)}
    m = dummy_market(dates, close)
    u = np.ones((30, 5), dtype=bool)
    u[10, 2] = False
    f = rng.normal(size=(30, 5))
    f[12, 1] = np.nan
    p = daily_panel(m, u, {"f": f, "g": -f}, horizon=5)
    assert len(p.y) == u.sum()
    r = (p.t == 10)
    assert 2 not in p.j[r]
    # 目的変数：その日のユニバース内の順位
    fr = forward_return(m, 5)
    k = np.flatnonzero((p.t == 3) & (p.j == int(np.argmax(fr[3]))))[0]
    assert p.y[k] == pytest.approx(1.0)
    assert np.isnan(p.y[p.t >= 25]).all()
    # 正規化：順位（0〜1）、NaN は NaN のまま、符号を反転した特徴量は順位も反転
    assert np.nanmax(p.X[:, 0]) == pytest.approx(1.0) and np.nanmin(p.X[:, 0]) > 0
    assert np.isnan(p.X[(p.t == 12) & (p.j == 1), 0]).all()
    rows = p.t == 5
    assert np.argmax(p.X[rows, 0]) == np.argmin(p.X[rows, 1])
    with pytest.raises(ValueError):
        normalize(f, u, "zscore")


def test_panel_uses_only_data_up_to_t():
    """起点 t の特徴量の行は、t より後を切った Market で作っても同じ（目的変数は未来を使うので除く）。"""
    rng = np.random.default_rng(1)
    dates = dummy_dates("2024-01-08", 30)
    close = {f"{k}0": list(300 + rng.normal(0, 10, 30).cumsum()) for k in range(1111, 1116)}
    m = dummy_market(dates, close)
    u = np.ones((30, 5), dtype=bool)
    f = rng.normal(size=(30, 5))
    p_full = daily_panel(m, u, {"f": f}, horizon=5, t_index=np.arange(20))
    cut = m.truncated(19)
    p_cut = daily_panel(cut, u[:20], {"f": f[:20]}, horizon=5)
    assert np.array_equal(p_full.t, p_cut.t) and np.allclose(p_full.X, p_cut.X, equal_nan=True)


def test_walk_forward_rolling_window_years():
    """直近 window_years 年のローリング：学習の最初の起点は、評価の月の最初の予測日の window_years 年前以降。"""
    dates = np.array([d.date().isoformat() for d in pd.bdate_range("2019-01-01", "2024-12-31")])
    expanding = walk_forward_splits(dates, "2019-01-01", "2024-03-01", "2024-06-30", gap=5, horizon=5)
    rolling = walk_forward_splits(dates, "2019-01-01", "2024-03-01", "2024-06-30", gap=5, horizon=5, window_years=4)
    for e, r in zip(expanding, rolling):
        assert dates[e.train_t[0]] == "2019-01-01"
        e0 = str(dates[r.eval_preds[0]])
        lower = (pd.Timestamp(e0) - pd.DateOffset(years=4)).date().isoformat()
        assert dates[r.train_t[0]] >= lower and dates[r.train_t[0] - 1] < lower
        assert r.train_t[-1] == e.train_t[-1]               # 終わり（空白の前）は同じ
    # 学習の開始日から4年たつまでは、拡大窓と同じ
    early = walk_forward_splits(dates, "2019-01-01", "2020-03-01", "2020-04-30", gap=5, horizon=5, window_years=4)
    assert all(dates[s.train_t[0]] == "2019-01-01" for s in early)
