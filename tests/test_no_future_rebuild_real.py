"""未来情報の混入テスト（実データ、作り直し）。--runslow を付けたときだけ実行する（約15分）。

予測日 p より後のデータ（株価・銘柄一覧・開示・決算発表予定日の公表）を data/raw から読まずに、Market（発行済株式数の
補正を含む）・ユニバース・特徴量をすべて作り直しても、全期間のデータで作った値と p まで一致することを確かめる
（2026-10-05 ユーザーの指示。tests/test_features_real.py は Market の配列を切り詰めるだけなので、株式数の補正や
銘柄一覧から作る部分は確かめられない）。保存した特徴量（logs/backtest/features/<セット名>.parquet）の p の行とも比べる。
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest.market import build_market, cache_path, load_market
from src.backtest.universe import universe_mask
from src.config import ROOT, load_config
from src.features.build import load_features
from src.features.data import load_feature_data
from src.features.registry import get, resolve

cfg = load_config()
SET = cfg["features"]["set"]
NAMES = resolve(SET)
ARRAYS = ["O", "H", "L", "C", "Va", "Vo", "adj", "UL", "LL", "common", "shares_base", "margin_other"]
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not cache_path(cfg).exists(), reason="Market のキャッシュがありません")]


@pytest.fixture(scope="module")
def full():
    m = load_market(cfg)
    return m, universe_mask(m, cfg), load_features(NAMES, cfg, m)


# 予測日（週の最終営業日）：売買の開始の直後、コロナ禍、大引けの時刻の変更の後、ホールドアウトの前の最後の予測日
@pytest.mark.parametrize("p_date", ["2018-12-28", "2020-04-03", "2024-11-08", "2025-09-19"])
def test_rebuild_up_to_prediction_date(full, p_date):
    m, u, feats = full
    p = m.date_index[p_date]
    cut = build_market(cfg, end=p_date)
    assert np.array_equal(cut.dates, m.dates[: p + 1])
    jm = np.array([m.code_index[c] for c in cut.codes])
    # p までに一度も普通株でなかった銘柄は、全期間のデータで作ってもユニバースに入らない
    others = np.setdiff1d(np.arange(len(m.codes)), jm)
    assert not u[: p + 1][:, others].any()
    for a in ARRAYS:
        assert np.array_equal(getattr(cut, a), getattr(m, a)[: p + 1][:, jm], equal_nan=True), a
    u_cut = universe_mask(cut, cfg)
    assert np.array_equal(u_cut, u[: p + 1][:, jm])
    fd_cut = load_feature_data(cut, cfg, end_date=p_date)
    for name in NAMES:
        b = get(name).func(fd_cut)
        a = feats[name][: p + 1][:, jm]
        assert np.allclose(a, b, rtol=1e-10, atol=0, equal_nan=True), name
        assert np.allclose(a[p][u_cut[p]], b[p][u_cut[p]], rtol=1e-10, atol=0, equal_nan=True), name
    # 保存した特徴量の p の行（ユニバースの銘柄と値）
    path = ROOT / "logs" / "backtest" / "features" / f"{SET}.parquet"
    if not path.exists():
        pytest.skip(f"{path} がまだありません")
    saved = pd.read_parquet(path, filters=[("date", "==", p_date)]).sort_values("code")
    codes = cut.codes[u_cut[p]]
    assert saved["code"].tolist() == sorted(codes.tolist())
    idx = np.array([cut.code_index[c] for c in saved["code"]])
    for name in NAMES:
        b = get(name).func(fd_cut)[p, idx]
        assert np.allclose(saved[name].to_numpy(), b, rtol=1e-10, atol=0, equal_nan=True), name
