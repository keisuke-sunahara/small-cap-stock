"""モデルの学習とウォークフォワードの点数のテスト（ダミーデータ）。"""
import copy

import numpy as np
import pandas as pd
import pytest

from src.backtest.target import cross_section_rank
from src.config import load_config
from src.models.train import Model, decile_labels, shuffle_within_day
from src.models.walkforward import (ScoreTable, affordable_mask, rank_against, score_days, splits_for,
                                    walk_forward_scores)
from tests.dummy_market import dummy_dates, dummy_market


def small_cfg(dates, **model):
    """base.yaml を、ダミーの期間と小さいモデルに合わせたもの（本物の設定ではない）。"""
    cfg = copy.deepcopy(load_config())
    cfg["data"]["holdout_start"] = (pd.Timestamp(str(dates[-1])) + pd.Timedelta(days=1)).date().isoformat()
    cfg["backtest"]["trade_start_date"] = str(dates[120])
    cfg["validation"].update({"train_start": str(dates[0]), "train_window": "rolling", "train_window_years": 1})
    cfg["model"]["lightgbm"].update({"n_estimators": 15, "min_data_in_leaf": 20, "num_threads": 1})
    cfg["model"].update(model)
    return cfg


def dummy_env(T=220, N=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = dummy_dates("2023-01-02", T)
    close = {f"{1000 + k}0": list(rng.choice([150, 400, 900, 2500]) * np.exp(np.cumsum(rng.normal(0, 0.02, T))))
             for k in range(N)}
    m = dummy_market(dates, close)
    u = np.ones((T, N), dtype=bool)
    u[:5] = False
    feats = {"f1": rng.normal(size=(T, N)), "f2": rng.normal(size=(T, N))}
    feats["f2"][rng.random((T, N)) < 0.05] = np.nan
    return m, u, feats


def test_decile_labels():
    y = np.array([0.01, 0.1, 0.1001, 0.5, 0.95, 1.0])
    assert decile_labels(y).tolist() == [0, 0, 1, 4, 9, 9]


def test_shuffle_within_day_keeps_values_per_day():
    day = np.repeat([0, 1, 2], 50)
    y = np.arange(150, dtype=float)
    y[[3, 70]] = np.nan
    s = shuffle_within_day(y, day, 42)
    assert np.isnan(s[3]) and np.isnan(s[70])
    for d in range(3):
        a, b = y[day == d], s[day == d]
        assert sorted(a[~np.isnan(a)]) == sorted(b[~np.isnan(b)])
    assert not np.array_equal(np.nan_to_num(s), np.nan_to_num(y))
    assert np.array_equal(shuffle_within_day(y, day, 42), s, equal_nan=True)   # 同じシードなら同じ


def test_rank_against_matches_cross_section_rank_and_places_outsiders():
    rng = np.random.default_rng(0)
    v = rng.integers(0, 20, size=(5, 40)).astype(float)
    v[0, 3] = np.nan
    mask = rng.random((5, 40)) < 0.7
    same = rank_against(v, mask, mask)
    want = cross_section_rank(v, mask & np.isfinite(v))
    assert np.allclose(same, want, equal_nan=True)
    ref = np.zeros((1, 4), dtype=bool)
    ref[0, :3] = True
    x = np.array([[10.0, 20.0, 30.0, 25.0]])
    out = rank_against(x, np.ones((1, 4), dtype=bool), ref)
    assert out[0, :3].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert out[0, 3] == pytest.approx(2.5 / 3)                    # 20 と 30 の間


@pytest.mark.parametrize("model", [{"type": "lightgbm", "objective": "regression"},
                                   {"type": "lightgbm", "objective": "lambdarank"},
                                   {"type": "ridge"}])
def test_models_fit_and_predict(model):
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(20), 50)
    X = rng.normal(size=(1000, 3)).astype(np.float32)
    X[rng.random((1000, 3)) < 0.05] = np.nan
    y = pd.Series(np.nan_to_num(X[:, 0]) + rng.normal(0, 0.5, 1000)).groupby(day).rank(pct=True).to_numpy()
    cfg = load_config()["model"]
    cfg = {**cfg, **model, "lightgbm": {**cfg["lightgbm"], "n_estimators": 20, "min_data_in_leaf": 20}}
    p = Model(cfg, 42).fit(X, y, day).predict(X)
    assert np.corrcoef(p, np.nan_to_num(X[:, 0]))[0, 1] > 0.5
    p2 = Model(cfg, 42).fit(X, y, day).predict(X)
    assert np.array_equal(p, p2)                                   # 同じ設定なら同じ結果


def test_score_days_and_model_assignment():
    m, u, feats = dummy_env()
    cfg = small_cfg(m.dates)
    sd = score_days(m, cfg)
    assert set(sd["kind"]) <= {"pred", "cont", "pred+cont"}
    splits = splits_for(m, cfg)
    res = walk_forward_scores(m, u, feats, cfg, None, log=None)
    sc = res.scores
    assert set(sc["date"]) == set(m.dates[sd["t"]])
    assert sc["score"].notna().all()
    assert len(res.splits) == len(splits)
    # 学習の最後の起点 + 目的変数の期間 < その月のモデルで点数を付ける最初の日
    for s, row in zip(splits, res.splits.itertuples()):
        assert m.date_index[row.train_last] + cfg["target"]["horizon_days"] < int(s.eval_preds.min())
    # 点数の表：点数の無い日はエラー
    st = ScoreTable(m, sc)
    t = int(sd["t"].iloc[0])
    assert np.isfinite(st(t)[u[t]]).all()
    with pytest.raises(KeyError):
        st(0)


@pytest.mark.parametrize("model", [{"type": "lightgbm", "objective": "regression"}, {"type": "ridge"}])
def test_no_future_information_in_scores(model):
    """日 X 以降の株価と特徴量を変えても、X より前に付けた点数（X より前に学習を始めた月のモデル）は変わらない。"""
    m, u, feats = dummy_env(seed=1)
    cfg = small_cfg(m.dates, **model)
    base = walk_forward_scores(m, u, feats, cfg, None, log=None).scores
    X = 180
    m2 = copy.deepcopy(m)
    rng = np.random.default_rng(9)
    m2.C[X:] *= np.exp(rng.normal(0, 0.2, m2.C[X:].shape))
    m2.O[X:] *= np.exp(rng.normal(0, 0.2, m2.O[X:].shape))
    m2.__post_init__()
    feats2 = {k: v.copy() for k, v in feats.items()}
    feats2["f1"][X:] = rng.normal(size=feats2["f1"][X:].shape)
    other = walk_forward_scores(m2, u, feats2, cfg, None, log=None).scores
    before = base["date"] < m.dates[X]
    assert before.sum() > 100
    pd.testing.assert_frame_equal(base[before].reset_index(drop=True), other[before].reset_index(drop=True))
    assert not np.allclose(base.loc[~before, "score"], other.loc[~before, "score"])


def test_target_shuffle_changes_scores():
    m, u, feats = dummy_env(seed=2)
    a = walk_forward_scores(m, u, feats, small_cfg(m.dates), None, log=None).scores
    b = walk_forward_scores(m, u, feats, small_cfg(m.dates, target_shuffle=True), None, log=None).scores
    assert not np.allclose(a["score"], b["score"])


def test_affordable_training_universe():
    m, u, feats = dummy_env(seed=3)
    cfg = small_cfg(m.dates, train_universe="affordable")
    cfg["capital"].update({"initial_capital_jpy": 300_000, "n_holdings": 3})
    aff = affordable_mask(m, cfg)
    # 指値 × 100株 ≤ 10万円 → 終値がおよそ 980 円以下
    assert aff[m.C < 900].all() and not aff[m.C > 1000].any()
    res = walk_forward_scores(m, u, feats, cfg, None, log=None)
    # 点数はユニバースの全銘柄（買えない銘柄にも）に付く
    sc = res.scores
    t = m.date_index[sc["date"].iloc[0]]
    day = sc[sc["date"] == sc["date"].iloc[0]]
    assert len(day) == u[t].sum()
    # 学習の行は「ユニバース ∩ 買える」だけ
    full = walk_forward_scores(m, u, feats, small_cfg(m.dates), None, log=None)
    assert (res.splits["train_rows"] < full.splits["train_rows"]).all()
