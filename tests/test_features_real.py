"""特徴量のテスト（実データ。約7分かかるため --runslow を付けたときだけ実行。Market のキャッシュが無ければ飛ばす）。

1. 未来情報の混入：予測日 p より後の株価・開示・予定日の公表を消して計算しても、p までの値が同じ
2. 独立な計算との照合：無作為に選んだ（日付, 銘柄）について、data/raw の開示を素朴な方法で絞り込んで計算した値と一致する
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest.market import cache_path, load_market
from src.backtest.universe import universe_mask
from src.config import load_config
from src.data.load import read_raw
from src.features.data import FIN_COLUMNS, load_feature_data
from src.features.registry import FEATURE_SETS, get

cfg = load_config()
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not cache_path(cfg).exists(), reason="Market のキャッシュがありません")]


@pytest.fixture(scope="module")
def full():
    m = load_market(cfg)
    return m, load_feature_data(m, cfg)


@pytest.mark.parametrize("p_date", ["2019-03-29", "2024-11-06"])
def test_no_future_information_real(full, p_date):
    m, fd = full
    p = m.date_index[p_date]
    cut_m = m.truncated(p)
    cut = load_feature_data(cut_m, cfg, end_date=p_date)
    u = universe_mask(m, cfg)
    for name in FEATURE_SETS["all_v1"]:
        a = get(name).func(fd)[: p + 1]
        b = get(name).func(cut)
        # 全体と、予測日のユニバースの行の両方で比べる
        assert np.allclose(a, b, equal_nan=True), name
        assert np.allclose(a[p][u[p]], b[p][u[p]], equal_nan=True), name


def _close_time(d: str) -> str:
    return "15:30" if d >= cfg["market"]["close_time_change_date"] else "15:00"


def test_matches_naive_recomputation(full):
    """開示の利用条件を「開示日 < 予測日、または 開示日 = 予測日 かつ 開示時刻 < 大引け」として素朴に絞り込む。"""
    m, fd = full
    u = universe_mask(m, cfg)
    feats = {n: get(n).func(fd) for n in ["bp", "ep_fcst", "bdays_since_report", "cdays_to_next_earnings"]}
    raw = read_raw(cfg, "summary", columns=FIN_COLUMNS)
    raw["Code"] = raw["Code"].astype(str)
    sched = read_raw(cfg, "earnings_date", columns=["Code", "SchDate", "FQName", "FYE"])
    sched["Code"] = sched["Code"].astype(str)
    rng = np.random.default_rng(7)
    ts = rng.integers(m.date_index["2018-10-01"], len(m.dates), 60)
    checked = 0
    for t in ts:
        d = str(m.dates[t])
        js = np.flatnonzero(u[t])
        for j in rng.choice(js, 5, replace=False):
            code = str(m.codes[j])
            r = raw[(raw["Code"] == code) & ((raw["DiscDate"] < d) |
                                             ((raw["DiscDate"] == d) & (raw["DiscTime"].str[:5] < _close_time(d))))]
            r = r.sort_values(["DiscDate", "DiscTime", "DiscNo"])
            mcap = m.shares[t, j] * m.C_ff[t, j]
            st = r[r["DocType"].str.contains("FinancialStatements")]
            if len(st):
                eq = pd.to_numeric(st["Eq"].iloc[-1], errors="coerce")
                assert np.isclose(feats["bp"][t, j], eq / mcap, equal_nan=True), (d, code)
                last_day = st["DiscDate"].iloc[-1]
                last_time = st["DiscTime"].iloc[-1][:5]
                first_use = m.date_index.get(last_day) if last_day in m.date_index and \
                    last_time < _close_time(last_day) else int(np.searchsorted(m.dates, last_day, side="right"))
                assert feats["bdays_since_report"][t, j] == t - first_use, (d, code)
            # 予想：通期予想を含む最新の記録（本決算は NxF*、それ以外は F*）
            fc = r[r["DocType"].str.contains("FinancialStatements") | (r["DocType"] == "EarnForecastRevision")].copy()
            is_fy = fc["DocType"].str.contains("FinancialStatements") & (fc["CurPerType"] == "FY")
            fc["fy_end"] = np.where(is_fy, fc["NxtFYEn"], fc["CurFYEn"])
            fc["fnp"] = pd.to_numeric(np.where(is_fy, fc["NxFNp"], fc["FNP"]), errors="coerce")
            fc["fsales"] = pd.to_numeric(np.where(is_fy, fc["NxFSales"], fc["FSales"]), errors="coerce")
            fc["fop"] = pd.to_numeric(np.where(is_fy, fc["NxFOP"], fc["FOP"]), errors="coerce")
            empty = (fc["DocType"] == "EarnForecastRevision") & fc[["fnp", "fsales", "fop"]].isna().all(axis=1)
            fc = fc[~empty & (fc["fy_end"].str.len() == 10)]
            fc = fc[fc["fy_end"] >= fc["fy_end"].cummax()]
            if len(fc):
                assert np.isclose(feats["ep_fcst"][t, j], fc["fnp"].iloc[-1] / mcap, equal_nan=True), (d, code)
            # 次の決算発表予定：公表日 < 予測日 の記録で、同じ四半期は最新の公表、予定日 > 予測日 の最も早いもの
            s = sched[(sched["Code"] == code) & (sched["PubDate"] < d)].sort_values("PubDate")
            s = s.drop_duplicates(["FYE", "FQName"], keep="last")
            s = s[(s["SchDate"].str.len() == 10) & (s["SchDate"] > d)]
            want = (pd.Timestamp(s["SchDate"].min()) - pd.Timestamp(d)).days if len(s) else np.nan
            assert np.isclose(feats["cdays_to_next_earnings"][t, j], want, equal_nan=True), (d, code)
            checked += 1
    assert checked == 300
