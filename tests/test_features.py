"""特徴量のテスト（ダミーデータ）。登録簿、計算、開示情報の利用開始日、未来情報の混入。"""
import copy

import numpy as np
import pandas as pd
import pytest

from src.features import registry
from src.features.data import FIN_COLUMNS, FeatureData, prepare_fins, prepare_sched
from src.features.registry import FEATURE_SETS, get, register
from tests.dummy_market import dummy_dates, dummy_market

NAN = float("nan")
CFG = {"market": {"close_time_change_date": "2024-11-05", "close_time_before": "15:00",
                  "close_time_after": "15:30"}}


def fin_row(code, disc_date, disc_time, doc="FYFinancialStatements_Consolidated_JP", per="FY", per_end="2024-03-31",
            fy_st="2023-04-01", fy_end="2024-03-31", nxt_fy_end="2025-03-31", no=None, **vals):
    row = {c: "" for c in FIN_COLUMNS}
    row.update({"DiscDate": disc_date, "DiscTime": disc_time, "Code": code, "DocType": doc, "CurPerType": per,
                "CurPerEn": per_end, "CurFYSt": fy_st, "CurFYEn": fy_end, "NxtFYEn": nxt_fy_end,
                "DiscNo": no or f"{disc_date}{disc_time}{code}"})
    row.update({k: str(v) for k, v in vals.items()})
    return row


def make_fd(m, fins_rows=(), sched_rows=()):
    fins = pd.DataFrame(list(fins_rows), columns=["DiscDate", *FIN_COLUMNS])
    sched = pd.DataFrame(list(sched_rows), columns=["PubDate", "Code", "SchDate", "FQName", "FYE"])
    return FeatureData(m=m, cfg=CFG, fins=prepare_fins(fins, m, CFG), sched=prepare_sched(sched, m))


def feat(name, fd):
    return get(name).func(fd)


# ---- 登録簿 ----

def test_all_feature_sets_are_registered_and_duplicates_are_rejected():
    for names in FEATURE_SETS.values():
        for n in names:
            assert get(n).name == n
    with pytest.raises(ValueError):
        register("ret_5d", "price", "二重登録")(lambda fd: None)
    assert registry.REGISTRY["ret_5d"].description.startswith("過去5営業日")


# ---- 値動き ----

def test_price_features_are_split_adjusted():
    dates = dummy_dates("2024-01-08", 80)
    c = [1000.0] * 70 + [500.0] * 10                   # 70日目の朝に2分割（実質の値動きなし）
    m = dummy_market(dates, {"11110": c})
    m.adj[70, 0] = 0.5
    m.__post_init__()
    m.Vo[:70, 0] = 1000.0
    m.Vo[70:, 0] = 2000.0                              # 分割後は出来高も2倍（実質は同じ）
    fd = make_fd(m)
    assert np.allclose(feat("ret_5d", fd)[72:, 0], 0.0)
    assert np.allclose(feat("ret_60d", fd)[72:, 0], 0.0)
    assert np.allclose(feat("vol_ratio_5_60", fd)[72:, 0], 0.0)
    assert np.allclose(feat("volatility_20d", fd)[72:, 0], 0.0)
    assert np.allclose(feat("dist_high_60d", fd)[72:, 0], 0.0)
    assert np.isnan(feat("ret_60d", fd)[59, 0]) and np.isfinite(feat("ret_60d", fd)[60, 0])


def test_price_feature_values():
    dates = dummy_dates("2024-01-08", 25)
    c = [100.0] * 20 + [110.0, NAN, 121.0, 100.0, 100.0]
    m = dummy_market(dates, {"11110": c})
    fd = make_fd(m)
    r5 = feat("ret_5d", fd)[:, 0]
    assert r5[20] == pytest.approx(0.10)
    assert r5[21] == pytest.approx(0.10)                # 売買不成立の日は直前の終値
    assert r5[22] == pytest.approx(0.21)
    m2 = dummy_market(dummy_dates("2024-01-08", 70), {"11110": [100.0] * 65 + [200.0] + [150.0] * 4})
    d = feat("dist_high_60d", make_fd(m2))
    assert d[69, 0] == pytest.approx(150 / 200 - 1)
    assert feat("log_turnover_20d", fd)[19, 0] == pytest.approx(np.log1p(1e9))


# ---- 開示情報の利用開始日（CLAUDE.md 第5章） ----

def test_disclosure_availability_by_close_time():
    # 2024-10-28（月）〜。大引けは 2024-11-05 から 15:30、それより前は 15:00
    dates = dummy_dates("2024-10-28", 15, holidays=("2024-11-04",))
    codes = ["11110", "22220", "33330", "44440", "55550"]
    m = dummy_market(dates, {c: [1000.0] * 15 for c in codes}, shares=1e6)
    rows = [fin_row("11110", "2024-10-29", "14:59", Eq=1e8),   # 15:00 より前 → 当日から
            fin_row("22220", "2024-10-29", "15:00", Eq=1e8),   # 15:00 ちょうど → 翌営業日から
            fin_row("33330", "2024-11-06", "15:20", Eq=1e8),   # 15:30 より前 → 当日から
            fin_row("44440", "2024-11-06", "15:30", Eq=1e8),   # 15:30 → 翌営業日から
            fin_row("55550", "2024-11-02", "10:00", Eq=1e8)]   # 土曜 → 次の営業日（11/4 は休日 → 11/5）から
    b = feat("bp", make_fd(m, rows))
    first = {c: dates[np.flatnonzero(np.isfinite(b[:, j]))[0]] for j, c in enumerate(m.codes)}
    assert first == {"11110": "2024-10-29", "22220": "2024-10-30", "33330": "2024-11-06",
                     "44440": "2024-11-07", "55550": "2024-11-05"}
    assert b[-1, 0] == pytest.approx(1e8 / 1e9)


def test_forecast_events_and_revisions():
    dates = dummy_dates("2024-04-01", 60)
    m = dummy_market(dates, {"11110": [1000.0] * 60}, shares=1e6)   # 時価総額 10億円
    rows = [
        # 本決算：翌期の予想（NxF*）を使う。今期の予想（F*）の列は空
        fin_row("11110", "2024-04-02", "10:00", Sales=9e9, OP=4e8, NxFSales=1e10, NxFOP=5e8, NxFNp=1e8),
        # 第2四半期の予想だけの修正（通期の予想が空）→ 使わない
        fin_row("11110", "2024-04-10", "10:00", doc="EarnForecastRevision", per="2Q", per_end="2024-09-30",
                fy_st="2024-04-01", fy_end="2025-03-31", nxt_fy_end=""),
        # 通期の予想の修正：営業利益 5億 → 6億
        fin_row("11110", "2024-04-16", "10:00", doc="EarnForecastRevision", per="FY", per_end="2025-03-31",
                fy_st="2024-04-01", fy_end="2025-03-31", nxt_fy_end="", FSales=1e10, FOP=6e8, FNP=2e8),
        # 終わった期（2024-03 期）の予想の修正 → 最新の予想として使わない
        fin_row("11110", "2024-04-22", "10:00", doc="EarnForecastRevision", per="FY", per_end="2024-03-31",
                fy_st="2023-04-01", fy_end="2024-03-31", nxt_fy_end="", FSales=9e9, FOP=1e8, FNP=-5e8),
    ]
    fd = make_fd(m, rows)
    ep = feat("ep_fcst", fd)[:, 0]
    rev = feat("op_fcst_rev", fd)[:, 0]
    i = {d: k for k, d in enumerate(dates)}
    assert np.isnan(ep[i["2024-04-01"]])
    assert ep[i["2024-04-02"]] == pytest.approx(0.1)
    assert ep[i["2024-04-15"]] == pytest.approx(0.1)
    assert ep[i["2024-04-16"]] == pytest.approx(0.2)
    assert ep[i["2024-04-22"]] == pytest.approx(0.2)    # 終わった期の修正は無視
    assert np.isnan(rev[i["2024-04-02"]])                # その期の最初の予想
    assert rev[i["2024-04-16"]] == pytest.approx(1e8 / 1e10)


def test_yoy_growth_surprise_and_days_since_report():
    dates = dummy_dates("2024-04-01", 300)
    m = dummy_market(dates, {"11110": [1000.0] * 300}, shares=1e6)
    rows = [
        fin_row("11110", "2024-04-05", "15:30", Sales=1e10, OP=5e8, NxFSales=1.1e10, NxFOP=6e8, NxFNp=3e8),
        fin_row("11110", "2024-07-30", "15:30", doc="1QFinancialStatements_Consolidated_JP", per="1Q",
                per_end="2024-06-30", fy_st="2024-04-01", fy_end="2025-03-31", nxt_fy_end="",
                Sales=2.5e9, OP=1e8, FSales=1.1e10, FOP=7e8, FNP=3e8),
        # 前年の第1四半期（データの都合で後から開示された訂正の扱いは別のテスト）。期末の月が12か月前
        fin_row("11110", "2024-04-03", "15:30", doc="1QFinancialStatements_Consolidated_JP", per="1Q",
                per_end="2023-06-30", fy_st="2023-04-01", fy_end="2024-03-31", nxt_fy_end="",
                Sales=2e9, OP=2e8, FSales=1e10, FOP=4.5e8, FNP=2e8),
        # 翌年の本決算：実績の営業利益 8億、直前の予想 7億 → 上振れ 1億 ÷ 売上 120億
        fin_row("11110", "2025-04-04", "15:30", per_end="2025-03-31", fy_st="2024-04-01", fy_end="2025-03-31",
                nxt_fy_end="2026-03-31", Sales=1.2e10, OP=8e8, NxFSales=1.3e10, NxFOP=9e8, NxFNp=4e8),
    ]
    fd = make_fd(m, rows)
    i = {d: k for k, d in enumerate(dates)}
    g = feat("sales_growth_yoy", fd)[:, 0]
    o = feat("op_chg_to_sales_yoy", fd)[:, 0]
    s = feat("op_surprise_fy", fd)[:, 0]
    days = feat("bdays_since_report", fd)[:, 0]
    t1 = i["2024-07-31"]
    assert g[t1] == pytest.approx(2.5e9 / 2e9 - 1)
    assert o[t1] == pytest.approx((1e8 - 2e8) / 2e9)
    assert np.isnan(g[i["2024-04-08"]])                   # 本決算の前年の記録が無い
    # 2024-03 期の本決算：直前の予想は 2024-04-03 の四半期の短信の通期予想（営業利益 4.5億）
    assert s[i["2024-04-08"]] == pytest.approx((5e8 - 4.5e8) / 1e10)
    t2 = i["2025-04-07"]
    assert s[t2] == pytest.approx((8e8 - 7e8) / 1.2e10)
    assert days[t1] == 0 and days[t1 + 3] == 3


def test_next_earnings_uses_only_published_schedule():
    dates = dummy_dates("2024-04-01", 40)
    m = dummy_market(dates, {"11110": [1000.0] * 40})
    sched = [
        {"PubDate": "2024-04-03", "Code": "11110", "SchDate": "2024-05-10", "FQName": "FY", "FYE": "0331"},
        # 予定日の変更（同じ四半期の後の公表が優先）
        {"PubDate": "2024-04-10", "Code": "11110", "SchDate": "2024-05-14", "FQName": "FY", "FYE": "0331"},
    ]
    x = feat("cdays_to_next_earnings", make_fd(m, sched_rows=sched))[:, 0]
    i = {d: k for k, d in enumerate(dates)}
    assert np.isnan(x[i["2024-04-03"]])                   # 公表日の当日は使わない
    assert x[i["2024-04-04"]] == (pd.Timestamp("2024-05-10") - pd.Timestamp("2024-04-04")).days
    assert x[i["2024-04-10"]] == (pd.Timestamp("2024-05-10") - pd.Timestamp("2024-04-10")).days
    assert x[i["2024-04-11"]] == (pd.Timestamp("2024-05-14") - pd.Timestamp("2024-04-11")).days
    assert x[i["2024-05-13"]] == 1
    assert np.isnan(x[i["2024-05-14"]])                   # 予定日の当日以降は「次」が分からない


# ---- 未来情報の混入 ----

def test_no_future_information_dummy():
    """予測日 p より後のデータ（株価・開示・予定日の公表）を消しても、p までの特徴量は変わらない。"""
    rng = np.random.default_rng(3)
    dates = dummy_dates("2024-10-01", 90)
    codes = [f"{k}0" for k in range(1111, 1117)]
    close = {c: list(500 + rng.normal(0, 10, 90).cumsum()) for c in codes}
    m = dummy_market(dates, close, shares=1e6)
    rows, sched = [], []
    for c in codes:
        for k in range(6):
            d = str(dates[rng.integers(0, 90)])
            t = str(rng.choice(["09:00", "14:59", "15:00", "15:10", "15:30", "18:00"]))
            doc, per = [("FYFinancialStatements_Consolidated_JP", "FY"),
                        ("1QFinancialStatements_Consolidated_JP", "1Q"),
                        ("EarnForecastRevision", "FY")][rng.integers(0, 3)]
            # 期末を2年のどちらかにして、前年同期との比較（YoY）も起きるようにする
            y = int(rng.integers(2023, 2025))
            per_end = f"{y}-03-31" if per == "FY" else f"{y}-06-30"
            rows.append(fin_row(c, d, t, doc=doc, per=per, per_end=per_end, fy_end=f"{y + (per != 'FY')}-03-31",
                                nxt_fy_end=f"{y + 1 + (per != 'FY')}-03-31",
                                Sales=rng.uniform(1e9, 2e9), OP=rng.normal(1e8, 5e7),
                                Eq=rng.uniform(1e8, 1e9), FSales=2e9, FOP=rng.normal(1e8, 5e7), FNP=5e7,
                                NxFSales=2e9, NxFOP=rng.normal(1e8, 5e7), NxFNp=5e7))
            sched.append({"PubDate": str(dates[rng.integers(0, 90)]), "Code": c,
                          "SchDate": (pd.Timestamp(str(dates[0])) + pd.Timedelta(days=int(rng.integers(0, 150)))
                                      ).date().isoformat(), "FQName": str(rng.choice(["1Q", "2Q", "3Q", "FY"])),
                          "FYE": "0331"})
    full = make_fd(m, rows, sched)
    for p in (30, 61, 75):
        cut_m = m.truncated(p)
        cut = make_fd(cut_m, [r for r in rows if r["DiscDate"] <= dates[p]],
                      [s for s in sched if s["PubDate"] <= dates[p]])
        for name in FEATURE_SETS["all_v1"]:
            a = feat(name, full)[: p + 1]
            b = feat(name, cut)
            assert np.allclose(a, b, equal_nan=True), name
        # 予測日の当日の大引け以降の開示が、p の値に効いていないこと（p の行は p の翌日の開示を加えても同じ）
        late = copy.deepcopy(rows) + [fin_row(codes[0], dates[p], "15:30", Eq=9e9)]
        assert np.allclose(feat("bp", make_fd(m, late))[: p + 1], feat("bp", full)[: p + 1], equal_nan=True)
