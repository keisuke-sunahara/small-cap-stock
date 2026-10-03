"""フェーズ1：取得したデータの品質チェック。

ホールドアウト開始日より前のデータだけを読む（src.data.load.read_raw）。
結果は reports/phase1_quality.json と reports/phase1_counts.png に保存する（集計値のみ。株価そのものは出さない）。

実行: python -m src.analysis.phase1_quality
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import matplotlib
import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.data.fetch import TSE_BUSINESS_HOLDIV, target_dates, DATASETS
from src.data.load import read_raw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 普通株の判定（DECISIONS.md 2026-10-04）
COMMON_PRODCAT = "011"
COMMON_MKT = {"0101", "0102", "0104", "0106", "0107", "0111", "0112", "0113"}
EXTREME_RET = 0.5


def is_common(master: pd.DataFrame) -> pd.Series:
    return (master["ProdCat"].astype(str).eq(COMMON_PRODCAT)
            & master["Mkt"].astype(str).isin(COMMON_MKT)
            & master["Code"].astype(str).str.endswith("0"))


def check_coverage(cfg: dict, start: date, end: date) -> dict:
    out = {}
    raw = ROOT / cfg["data"]["raw_dir"]
    for name, ds in DATASETS.items():
        if ds.kind == "range":
            continue
        dates = target_dates(cfg, ds, start, end)
        files = {p.stem for p in (raw / name).glob("????-??-??.parquet")}
        missing = [d for d in dates if d not in files]
        out[name] = {"target_days": len(dates), "missing_days": len(missing), "missing_sample": missing[:10]}
    return out


def check_calendar(cal: pd.DataFrame) -> dict:
    cal = cal.copy()
    cal["year"] = cal["Date"].str[:4]
    bd = cal[cal["HolDiv"].astype(str).isin(TSE_BUSINESS_HOLDIV)]
    return {"holdiv_counts": cal["HolDiv"].astype(str).value_counts().to_dict(),
            "business_days_per_year": bd.groupby("year").size().to_dict()}


def check_master(master: pd.DataFrame) -> tuple[dict, pd.Series]:
    master = master.copy()
    master["common"] = is_common(master)
    dates = sorted(master["Date"].unique())
    samples = [dates[0]] + [d for d in dates if d.endswith("-12-28") or d.endswith("-12-30")] + [dates[-1]]
    snap = {}
    for d in sorted(set(samples)):
        m = master[master["Date"] == d]
        snap[d] = {"rows": len(m), "common": int(m["common"].sum()),
                   "ProdCat": m["ProdCat"].astype(str).value_counts().to_dict(),
                   "Mkt": m["MktNm"].astype(str).value_counts().to_dict()}
    last = master[master["Date"] == dates[-1]]
    pref = last[last["ProdCat"].astype(str).eq(COMMON_PRODCAT) & ~last["Code"].astype(str).str.endswith("0")]
    other_mkt = last[last["ProdCat"].astype(str).eq(COMMON_PRODCAT) & ~last["Mkt"].astype(str).isin(COMMON_MKT)]
    common_per_day = master[master["common"]].groupby("Date").size()
    result = {
        "snapshots": snap,
        "last_date": dates[-1],
        "prodcat011_code_not_ending_0": {"count": len(pref), "names": pref["CoName"].head(10).tolist()},
        "prodcat011_other_market": {"count": len(other_mkt),
                                    "markets": other_mkt["MktNm"].astype(str).value_counts().to_dict()},
        "common_per_day": {"min": int(common_per_day.min()), "max": int(common_per_day.max()),
                           "first": int(common_per_day.iloc[0]), "last": int(common_per_day.iloc[-1])},
    }
    return result, common_per_day


def adjusted_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """AdjFactor の累積積で終値を割った Q = C / cumprod(AdjFactor) から、売買が成立した日どうしのリターンを出す。

    Q はその日までの調整係数だけを使う（後の分割の情報を含まない）。売買不成立の日の係数も累積に含める。
    """
    b = bars.sort_values(["Code", "Date"]).copy()
    b["AdjFactor"] = b["AdjFactor"].fillna(1.0)
    b["cumF"] = b.groupby("Code")["AdjFactor"].cumprod()
    b["Q"] = b["C"] / b["cumF"]
    traded = b[b["C"].notna()].copy()
    traded["prevQ"] = traded.groupby("Code")["Q"].shift(1)
    traded["prevC"] = traded.groupby("Code")["C"].shift(1)
    traded["ret_adj"] = traded["Q"] / traded["prevQ"] - 1
    traded["ret_raw"] = traded["C"] / traded["prevC"] - 1
    return traded


def check_bars(bars: pd.DataFrame, master: pd.DataFrame, bdays: list[str]) -> tuple[dict, pd.DataFrame]:
    res: dict = {}
    per_day = bars.groupby("Date").agg(rows=("Code", "size"), no_trade=("C", lambda s: int(s.isna().sum())))
    res["rows_per_day"] = {"min": int(per_day["rows"].min()), "max": int(per_day["rows"].max()),
                           "median": float(per_day["rows"].median())}
    res["days_missing_in_bars"] = [d for d in bdays if d not in per_day.index][:20]
    # 前日比で10%以上銘柄数が減った日
    drop = per_day["rows"] / per_day["rows"].shift(1) - 1
    res["days_rows_drop_over_10pct"] = drop[drop < -0.10].round(3).to_dict()

    ohlc = ["O", "H", "L", "C", "Vo", "Va"]
    nulls = bars[ohlc].isna()
    res["no_trade_rows"] = int(nulls["C"].sum())
    res["no_trade_ratio"] = round(float(nulls["C"].mean()), 5)
    res["partial_null_rows"] = int((nulls.any(axis=1) & ~nulls.all(axis=1)).sum())
    t = bars[bars["C"].notna()]
    res["invalid"] = {
        "nonpositive_price": int((t[["O", "H", "L", "C"]] <= 0).any(axis=1).sum()),
        "low_gt_high": int((t["L"] > t["H"]).sum()),
        "open_or_close_outside_range": int(((t["O"] > t["H"]) | (t["O"] < t["L"])
                                            | (t["C"] > t["H"]) | (t["C"] < t["L"])).sum()),
        "zero_volume_with_price": int((t["Vo"] <= 0).sum()),
    }
    vwap = t["Va"] / t["Vo"]
    outside = (vwap < t["L"] * 0.99) | (vwap > t["H"] * 1.01)
    res["vwap_outside_range_rows"] = int(outside.sum())
    res["limit_flags"] = {"UL_rows": int((bars["UL"].astype(str) == "1").sum()),
                          "LL_rows": int((bars["LL"].astype(str) == "1").sum())}

    # 分割・併合
    adj = bars[bars["AdjFactor"].notna() & (bars["AdjFactor"] != 1.0)]
    res["adjfactor_events"] = {"count": len(adj), "codes": int(adj["Code"].nunique()),
                               "values_top": adj["AdjFactor"].round(4).value_counts().head(10).to_dict()}
    traded = adjusted_returns(bars)
    ev = traded[traded["Code"].astype(str).isin(adj["Code"].astype(str))]
    ev = ev.merge(adj[["Date", "Code"]].assign(event=True), on=["Date", "Code"], how="left")
    on_event = ev[ev["event"].eq(True) & ev["ret_adj"].notna()]
    res["split_days"] = {
        "n": len(on_event),
        "abs_raw_return_median": round(float(on_event["ret_raw"].abs().median()), 4) if len(on_event) else None,
        "abs_adj_return_median": round(float(on_event["ret_adj"].abs().median()), 4) if len(on_event) else None,
        "adj_return_over_50pct": int((on_event["ret_adj"].abs() > EXTREME_RET).sum()),
    }
    r = traded["ret_adj"].dropna()
    ext = traded[traded["ret_adj"].abs() > EXTREME_RET]
    res["adjusted_daily_returns"] = {
        "n": len(r), "abs_over_50pct": len(ext),
        "quantiles": {str(q): round(float(r.quantile(q)), 4) for q in (0.001, 0.01, 0.5, 0.99, 0.999)},
        # 大きく動いた日が、売買不成立明け（前回の約定から日が空いている）かどうか
        "extreme_by_year": ext["Date"].str[:4].value_counts().sort_index().to_dict(),
    }
    raw_ext = traded[(traded["ret_raw"].abs() > EXTREME_RET)]
    raw_ext_no_event = raw_ext.merge(adj[["Date", "Code"]].assign(event=True), on=["Date", "Code"], how="left")
    res["raw_return_over_50pct_without_adjfactor"] = int(raw_ext_no_event["event"].isna().sum())

    # 上場廃止：期間の途中で株価が終わった銘柄
    last_day = bars["Date"].max()
    last_seen = bars.groupby("Code")["Date"].max()
    first_seen = bars.groupby("Code")["Date"].min()
    ended = last_seen[last_seen < last_day]
    m_codes = master.groupby("Date")["Code"].apply(set)
    ended_still_listed = sum(1 for c, d in ended.items()
                             if d in m_codes.index and c not in m_codes.loc[d])
    res["delisted"] = {
        "codes_total": int(last_seen.size),
        "codes_ended_before_last_day": int(ended.size),
        "ended_by_year": ended.str[:4].value_counts().sort_index().to_dict(),
        "started_after_first_day_by_year": first_seen[first_seen > bars["Date"].min()]
        .str[:4].value_counts().sort_index().to_dict(),
        "ended_but_missing_from_master_on_last_day": ended_still_listed,
    }

    # 銘柄一覧と株価の対応（普通株について）
    common = master[is_common(master)][["Date", "Code"]]
    j = common.merge(bars[["Date", "Code"]].assign(in_bars=True), on=["Date", "Code"], how="left")
    res["common_in_master_without_bars_rows"] = int(j["in_bars"].isna().sum())
    allm = master[["Date", "Code"]].assign(in_master=True)
    jb = bars[["Date", "Code"]].merge(allm, on=["Date", "Code"], how="left")
    res["bars_without_master_rows"] = int(jb["in_master"].isna().sum())

    # 連続した売買不成立（長期の売買停止など）
    b = bars.sort_values(["Code", "Date"])
    nt = b["C"].isna()
    grp = (nt != nt.groupby(b["Code"]).shift()).cumsum()
    runs = nt.groupby([b["Code"], grp]).sum()
    res["longest_no_trade_runs_days"] = runs.sort_values(ascending=False).head(5).astype(int).tolist()
    return res, per_day


def check_topix(topix: pd.DataFrame, bdays: list[str]) -> dict:
    t = topix.sort_values("Date")
    r = t["C"].pct_change()
    return {"rows": len(t), "missing_bdays": [d for d in bdays if d not in set(t["Date"])][:20],
            "abs_daily_return_max": round(float(r.abs().max()), 4),
            "null_close": int(t["C"].isna().sum())}


def check_summary(s: pd.DataFrame, master: pd.DataFrame) -> dict:
    res: dict = {"rows": len(s), "rows_per_year": s["DiscDate"].str[:4].value_counts().sort_index().to_dict()}
    res["DocType_top"] = s["DocType"].value_counts().head(15).to_dict()
    res["DocType_kinds"] = int(s["DocType"].nunique())
    tm = s["DiscTime"].astype(str)
    res["DiscTime_bad_format"] = int((~tm.str.fullmatch(r"\d{2}:\d{2}(:\d{2})?")).sum())
    hhmm = tm.str[:5]
    res["DiscTime_share"] = {
        "before_15:00": round(float((hhmm < "15:00").mean()), 4),
        "15:00_to_15:30": round(float(((hhmm >= "15:00") & (hhmm < "15:30")).mean()), 4),
        "15:30_or_later": round(float((hhmm >= "15:30").mean()), 4),
    }
    res["duplicate_DiscNo"] = int(s["DiscNo"].duplicated().sum())
    res["dates_with_weekend_disclosure"] = int(pd.to_datetime(s["DiscDate"]).dt.dayofweek.ge(5).sum())
    # 時価総額の計算に使う発行済株式数（ShOutFY）の空欄の割合（決算短信のみ）
    fin = s[s["DocType"].astype(str).str.contains("FinancialStatements")]
    res["ShOutFY_blank_ratio_in_FinancialStatements"] = round(float(
        (fin["ShOutFY"].astype(str).str.strip() == "").mean()), 4) if len(fin) else None
    # 普通株のうち、直近1年に決算短信がある銘柄の割合（最終日時点）
    last = master["Date"].max()
    common_codes = set(master[(master["Date"] == last) & is_common(master)]["Code"])
    one_year = (date.fromisoformat(last) - timedelta(days=365)).isoformat()
    recent = set(fin[fin["DiscDate"] >= one_year]["Code"])
    res["common_with_statement_in_last_365d"] = round(len(common_codes & recent) / len(common_codes), 4)
    return res


def check_earnings_date(e: pd.DataFrame, master: pd.DataFrame) -> dict:
    sch = e["SchDate"].astype(str)
    filled = e[sch != ""]
    res = {"rows": len(e), "rows_per_year": e["PubDate"].str[:4].value_counts().sort_index().to_dict(),
           "SchDate_blank_ratio": round(float((sch == "").mean()), 4),
           "SchDate_before_PubDate": int((filled["SchDate"] < filled["PubDate"]).sum()),
           "FQName": e["FQName"].value_counts().to_dict(),
           "lead_days_median": float((pd.to_datetime(filled["SchDate"]) - pd.to_datetime(filled["PubDate"]))
                                     .dt.days.median())}
    last = master["Date"].max()
    common_codes = set(master[(master["Date"] == last) & is_common(master)]["Code"])
    one_year = (date.fromisoformat(last) - timedelta(days=365)).isoformat()
    recent = set(e[e["PubDate"] >= one_year]["Code"])
    res["common_with_schedule_in_last_365d"] = round(len(common_codes & recent) / len(common_codes), 4)
    return res


def plot_counts(per_day: pd.DataFrame, common_per_day: pd.Series, path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    idx = pd.to_datetime(per_day.index)
    ax.plot(idx, per_day["rows"], label="株価の行数（全銘柄）", lw=1)
    ax.plot(pd.to_datetime(common_per_day.index), common_per_day.values, label="普通株（銘柄一覧）", lw=1)
    ax.plot(idx, per_day["no_trade"], label="売買不成立", lw=1)
    ax.set_title("日ごとの銘柄数")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
    start = date.fromisoformat(cfg["data"]["start_date"])
    end = date.fromisoformat(cfg["data"]["holdout_start"]) - timedelta(days=1)
    report: dict = {"period": [start.isoformat(), end.isoformat()]}

    report["coverage"] = check_coverage(cfg, start, end)
    cal = read_raw(cfg, "calendar")
    bdays = sorted(cal.loc[cal["HolDiv"].astype(str).isin(TSE_BUSINESS_HOLDIV), "Date"])
    bdays = [d for d in bdays if d >= start.isoformat()]
    report["calendar"] = check_calendar(cal)
    print("calendar ok", flush=True)

    master = read_raw(cfg, "master", columns=["Code", "CoName", "Mkt", "MktNm", "ProdCat", "S17"])
    report["master"], common_per_day = check_master(master)
    print("master ok", flush=True)

    bars = read_raw(cfg, "bars", columns=["Code", "O", "H", "L", "C", "Vo", "Va", "UL", "LL", "AdjFactor"])
    bars["Code"] = bars["Code"].astype(str)
    report["bars"], per_day = check_bars(bars, master, bdays)
    print("bars ok", flush=True)
    del bars

    report["topix"] = check_topix(read_raw(cfg, "topix"), bdays)
    summary = read_raw(cfg, "summary", columns=["Code", "DiscTime", "DiscNo", "DocType", "ShOutFY"])
    report["summary"] = check_summary(summary, master)
    edate = read_raw(cfg, "earnings_date", columns=["Code", "SchDate", "FQName"])
    report["earnings_date"] = check_earnings_date(edate, master)

    out = ROOT / "reports" / "phase1_quality.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    plot_counts(per_day, common_per_day, ROOT / "reports" / "phase1_counts.png")
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
