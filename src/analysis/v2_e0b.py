"""v2 の E0b：かぶミニの今の取扱一覧（2026-10-05）での測り直し（2026-10-05 ユーザーの指示）。

成績・Rank IC・目的変数・リターンは計算しない。使う株価のデータは 2025-09-26 以前だけ（Market）。
今の一覧の CSV の「現在値」「前日比」の列（2026-10-05 の値。ホールドアウトの期間）は読み込みの時点で捨てる。
写し（logs/v2_e0/）は コード・銘柄名・市場 の3列だけにする（2026-10-05 ユーザーの指示）。

(0) 照合：2024-12-17 の一覧（logs/v2_e0/kabumini_lineup_2024-12-17.csv の寄付取引○）と今の一覧
(1) E0 (1) を今の一覧で：2025-09-26 のユニバースの割合（時価総額の帯と5分位、売買代金の帯と5分位）、各予測日に当てた割合（未来の情報を含む参考値）
(2) v1 の EXP-002（リッジ）のコミット済みの点数（logs/backtest/EXP-002/scores.parquet）で、2024-10〜2025-09 の各月末以前の最後の予測日の
    点数の上位20銘柄と上位20%のうち、今の一覧・2024-12-17 の一覧で買える割合（点数を読むだけ。試行回数に数えない）

実行：python -m src.analysis.v2_e0b [--source external|copy]
  external：data/external/ のユーザーの CSV（コミットしない。5列）から読み、3列の写しを logs/v2_e0/ に書く（既定。ファイルが無ければ copy）
  copy：logs/v2_e0/ の3列の写しから読む（評価役の再現用。写しは書き換えない）
出力：reports/v2_e0b.json、logs/v2_e0/kabumini_yoritsuki_2026-10-05.csv（ユーザーの CSV の3列の写し）
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.analysis.v2_e0 import CHECK_DATE, JST, coverage_table, month_end_dates, universe_frame
from src.backtest.market import load_market
from src.backtest.universe import avg_turnover, market_cap, universe_mask
from src.config import ROOT, load_config

NEW_DATE = "2026-10-05"
NEW_SRC = ROOT / "data" / "external" / f"kabumini_yoritsuki_{NEW_DATE}.csv"
NEW_COPY = ROOT / "logs" / "v2_e0" / f"kabumini_yoritsuki_{NEW_DATE}.csv"
OLD_LIST = ROOT / "logs" / "v2_e0" / "kabumini_lineup_2024-12-17.csv"
SCORES = ROOT / "logs" / "backtest" / "EXP-002" / "scores.parquet"
NEW_COLUMNS = ["コード", "銘柄名", "市場", "現在値", "前日比(%)"]   # ユーザーが出力した CSV の列（記録用）
KEEP_COLUMNS = ["コード", "銘柄名", "市場"]                          # 読み込みで残す列。価格の列は捨てる
SCORE_MONTHS = [str(p) for p in pd.period_range("2024-10", "2025-09", freq="M")]
TOP_N = 20
TOP_FRAC = 0.2


def read_lineup(path) -> pd.DataFrame:
    """かぶミニの取扱一覧を読む。コード・銘柄名・市場 の3列だけを残し、価格などの他の列は読み込みの時点で捨てる。"""
    head = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.tolist()
    assert set(KEEP_COLUMNS) <= set(head), head
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, usecols=KEEP_COLUMNS)[KEEP_COLUMNS]
    return df


def read_new_list(source: str) -> pd.DataFrame:
    df = read_lineup(NEW_SRC if source == "external" else NEW_COPY)
    if source == "external":
        # 3列の写しを書く（価格の列はここで捨てているので写しにも入らない）
        NEW_COPY.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(NEW_COPY, index=False, encoding="utf-8-sig", lineterminator="\n")
    df = df.copy()
    df.columns = ["code4", "name", "market"]
    assert df["code4"].str.fullmatch(r"[0-9][0-9A-Z]{2}[0-9A-Z]").all()
    assert not df["code4"].duplicated().any()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["external", "copy"], default=None)
    args = ap.parse_args()
    source = args.source or ("external" if NEW_SRC.exists() else "copy")
    cfg = load_config()
    out: dict = {"created_at_jst": datetime.now(JST).isoformat(timespec="seconds")}

    new = read_new_list(source)
    old = pd.read_csv(OLD_LIST, dtype={"code4": str})
    out["new_list"] = {
        "source": "楽天証券 スーパースクリーナー、種別「かぶミニ寄付」の出力（ユーザーが 2026-10-05 に取得）",
        "file": str(NEW_COPY.relative_to(ROOT)).replace("\\", "/"), "rows": int(len(new)),
        "columns": NEW_COLUMNS, "code_format": "4文字（数字4桁、または 130A のような英字入り）。末尾に0を付けて5桁のコードと照合",
        "markets": {k: int(v) for k, v in new["market"].value_counts().items()},
        "not_read": "現在値・前日比の列（2026-10-05 の値）は読んでいない",
    }

    m = load_market(cfg)
    assert m.dates[-1] < cfg["data"]["holdout_start"]
    T = len(m.dates) - 1
    idx = {c: j for j, c in enumerate(m.codes)}
    u = universe_mask(m, cfg)
    mcap = market_cap(m)
    adv = avg_turnover(m, cfg["universe"]["turnover_window"])
    t_chk = m.date_index[CHECK_DATE]
    uni_chk = set(m.codes[np.nonzero(u[t_chk])[0]])

    def status(c4: str) -> str:
        j = idx.get(c4 + "0")
        if j is None:
            return "J-Quants の銘柄一覧（〜2025-09-26）に無い"
        return "2025-09-26 に上場" if m.last_listed[j] == T else "2025-09-26 より前に上場廃止"

    # (0) 照合
    old_y = old[old["yoritsuki"]].copy()
    o, n = set(old_y["code4"]), set(new["code4"])
    old_only = old_y[~old_y["code4"].isin(n)].copy()
    old_only["status"] = old_only["code4"].map(status)
    old_only["in_universe_2025_09_26"] = (old_only["code4"] + "0").isin(uni_chk)
    new_only = new[~new["code4"].isin(o)].copy()
    new_only["status"] = new_only["code4"].map(status)
    new_only["in_universe_2025_09_26"] = (new_only["code4"] + "0").isin(uni_chk)
    listed_old_only = old_only[old_only["status"] == "2025-09-26 に上場"]
    out["e0b_0_match"] = {
        "old_yoritsuki": len(o), "new": len(n), "both": len(o & n), "old_only": len(o - n), "new_only": len(n - o),
        "old_only_by_status": {k: int(v) for k, v in old_only["status"].value_counts().items()},
        "old_only_listed_2025_09_26_in_universe": int(listed_old_only["in_universe_2025_09_26"].sum()),
        "old_only_listed_2025_09_26_share_of_old_listed":
            round(len(listed_old_only) / (len(o & n) + len(listed_old_only)), 4),
        "new_only_by_status": {k: int(v) for k, v in new_only["status"].value_counts().items()},
        "new_only_in_universe_2025_09_26": int(new_only["in_universe_2025_09_26"].sum()),
        "old_only_listed_2025_09_26": listed_old_only[["code4", "name", "in_universe_2025_09_26",
                                                       "sell_only_2025_06_10"]].to_dict("records"),
        "old_only_delisted": old_only.loc[old_only["status"] != "2025-09-26 に上場", ["code4", "name", "status"]].to_dict("records"),
        "new_only": new_only[["code4", "name", "market", "status", "in_universe_2025_09_26"]].to_dict("records"),
        "note": "2025-09-29 以降の上場・上場廃止は、ホールドアウトのため J-Quants では確かめていない",
    }

    # (1) 今の一覧でのユニバースの割合
    buy_new = {c + "0" for c in n}
    buy_old = {c + "0" for c in o}
    mc_bands = [("50億円未満", 0, 5e9), ("50〜100億円", 5e9, 1e10), ("100〜200億円", 1e10, 2e10),
                ("200〜500億円", 2e10, 5.0000001e10)]
    adv_bands = [("3,000万〜5,000万円", 3e7, 5e7), ("5,000万〜1億円", 5e7, 1e8), ("1〜3億円", 1e8, 3e8),
                 ("3億円以上", 3e8, np.inf)]
    res1: dict = {}
    for label, codes in [("new_2026_10_05", buy_new), ("old_2024_12_17", buy_old)]:
        df = universe_frame(m, u, mcap, adv, t_chk, codes)
        r = {"universe": int(len(df)), "kabumini_buyable": int(df["buyable"].sum()),
             "share": round(float(df["buyable"].mean()), 4),
             "by_market_cap": coverage_table(df, "mcap", mc_bands),
             "by_turnover": coverage_table(df, "adv", adv_bands)}
        for col, key in [("mcap", "by_market_cap_quintile"), ("adv", "by_turnover_quintile")]:
            q = pd.qcut(df[col], 5, labels=[f"Q{i}" for i in range(1, 6)])
            r[key] = [{"quintile": str(k), "n": int(len(g)), "share": round(float(g["buyable"].mean()), 4),
                       "range_oku_yen": [round(float(g[col].min()) / 1e8, 1), round(float(g[col].max()) / 1e8, 1)]}
                      for k, g in df.groupby(q, observed=True)]
        res1[label] = r
    P = month_end_dates(m.dates)
    sb = pd.Series([float(np.isin(m.codes[np.nonzero(u[t])[0]], list(buy_new)).mean()) for t in P],
                   index=[str(m.dates[t]) for t in P])
    res1["by_prediction_date_new_list"] = {
        "note": "今の一覧（2026-10-05）を過去の予測日に当てた参考値。過去の取扱いではなく、未来の情報を含む"
                "（後で取扱いに加わった銘柄が入り、それより前に上場廃止した銘柄は入らない）",
        "n_dates": len(P), "min": round(float(sb.min()), 4), "median": round(float(sb.median()), 4),
        "max": round(float(sb.max()), 4), "first": [str(sb.index[0]), round(float(sb.iloc[0]), 4)],
        "by_year_median": {y: round(float(g.median()), 4) for y, g in sb.groupby(sb.index.str[:4])}}
    out["e0b_1_coverage"] = res1

    # (2) EXP-002 の点数の上位
    s = pd.read_parquet(SCORES)
    s = s[s["kind"].isin(["pred", "pred+cont"])]
    pdates = sorted(s["date"].unique())
    rows = []
    for mo in SCORE_MONTHS:
        end = (pd.Period(mo, "M").end_time).strftime("%Y-%m-%d")
        d = max(x for x in pdates if x <= end)
        g = s[s["date"] == d].sort_values(["score", "code"], ascending=[False, True]).reset_index(drop=True)
        k20 = int(np.floor(len(g) * TOP_FRAC + 1e-9))
        row = {"month": mo, "prediction_date": d, "universe": int(len(g)), "top20pct_n": k20}
        for label, codes in [("new", buy_new), ("old", buy_old)]:
            b = g["code"].isin(codes)
            row[f"top{TOP_N}_{label}"] = round(float(b.iloc[:TOP_N].mean()), 4)
            row[f"top20pct_{label}"] = round(float(b.iloc[:k20].mean()), 4)
            row[f"universe_{label}"] = round(float(b.mean()), 4)
        rows.append(row)
    sc = pd.DataFrame(rows)
    summ = {c: {"mean": round(float(sc[c].mean()), 4), "min": round(float(sc[c].min()), 4),
                "max": round(float(sc[c].max()), 4)}
            for c in sc.columns if c.startswith(("top", "universe_")) and c != "top20pct_n"}
    out["e0b_2_exp002_top"] = {
        "note": "v1 の EXP-002（リッジ、週次・5営業日の目的変数）の点数。v2 のモデルではない。各月末以前の最後の予測日の、"
                "点数の高い順（同点は銘柄コードの順）の上位20銘柄と上位20%（順位 ≤ 銘柄数 × 0.2）のうち、一覧にある割合。"
                "リターン・成績・Rank IC は計算していない。試行回数に数えない",
        "months": rows, "summary": summ}

    path = ROOT / "reports" / "v2_e0b.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    brief = {k: out[k] for k in ["new_list"]}
    brief["match"] = {k: v for k, v in out["e0b_0_match"].items() if not isinstance(v, list)}
    brief["coverage_2025_09_26"] = {k: {kk: vv for kk, vv in v.items() if kk in ("universe", "kabumini_buyable", "share")}
                                    for k, v in res1.items() if k.startswith(("new", "old"))}
    brief["by_prediction_date_new_list"] = res1["by_prediction_date_new_list"]
    brief["exp002_summary"] = summ
    print(json.dumps(brief, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
