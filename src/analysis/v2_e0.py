"""v2 の E0（実現の確認）。成績・Rank IC・目的変数・将来のリターンの平均は計算しない（2026-10-05 ユーザーの指示）。

使うデータは 2025-09-26 以前だけ（ホールドアウト 2025-09-29 以降は Market に入っていない）。

(1) かぶミニの取扱銘柄が、ユニバースの何割か（時価総額・売買代金の大きさ別にも）
    取扱銘柄一覧は楽天証券の公開 PDF（2024-12-17 時点。data/external/rakuten_kabumini/。取得は1回、ユーザーの承認）
(2) 資金 × N ごとに、100株を買えるユニバースの割合（v1 の方法：予算 = 資金 ÷ N、指値 = 終値 × 1.02 を呼値で切り下げ、
    100株 × 指値 ≤ 予算、注文金額 ≤ 20日平均売買代金の1%）
(3) ユニバースの銘柄の、市場と関係ない月の値動きの大きさ：月ごとの、ユニバース内の月次リターンの標準偏差
    （= ユニバース平均との差の標準偏差）の、月をまたいだ中央値。標準偏差だけを出力し、リターンの平均や、
    特徴量・点数との関係は計算しない。月次リターンは v2 の売買の期間（予測日の2営業日後の始値 → 翌々月の第1営業日の始値）
(付) かぶミニの買い注文の余力拘束（ストップ高の値段 × 1.0022 × 株数）で、1回の注文で使える資金の割合（株価の水準だけで決まる）
(付) 長い期間の特徴量（220営業日）が、Light の5年分のデータで、4年の学習期間の最初の行まで計算できるか（カレンダーだけ）

実行：python -m src.analysis.v2_e0
出力：reports/v2_e0.json、logs/v2_e0/kabumini_lineup_2024-12-17.csv
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.backtest.execution import limit_price
from src.backtest.market import load_market
from src.backtest.universe import avg_turnover, market_cap, universe_mask
from src.config import ROOT, load_config

JST = ZoneInfo("Asia/Tokyo")
LIST_DIR = ROOT / "data" / "external" / "rakuten_kabumini"
LIST_DATE = "2024-12-17"
CHECK_DATE = "2025-09-26"
CAPITALS = [300_000, 500_000, 1_000_000, 3_000_000]
NS_LOT = [3, 10, 20]
NS_NOISE = [3, 10, 20, 30]
FIRST_P_MONTH = "2018-09"
LAST_P_MONTH = "2025-08"   # 2025-09 の月末（09-30）はホールドアウトに入る
SPREAD = 0.0022
LONG_WINDOW = 220

# 東証の制限値幅（基準値段 未満 → 値幅）。2014-01 以降の表（基準値段 1,000万円未満まで）
LIMIT_WIDTH = [
    (100, 30), (200, 50), (500, 80), (700, 100), (1_000, 150), (1_500, 300), (2_000, 400), (3_000, 500),
    (5_000, 700), (7_000, 1_000), (10_000, 1_500), (15_000, 3_000), (20_000, 4_000), (30_000, 5_000),
    (50_000, 7_000), (70_000, 10_000), (100_000, 15_000), (150_000, 30_000), (200_000, 40_000),
    (300_000, 50_000), (500_000, 70_000), (700_000, 100_000), (1_000_000, 150_000),
]


def limit_width(base: float) -> float:
    for upper, w in LIMIT_WIDTH:
        if base < upper:
            return w
    return float("nan")


# ---------------------------------------------------------------- 取扱銘柄一覧
MAIN_ROW = re.compile(r"([0-9][0-9A-Z]{2}[0-9A-Z])\s+(.+?)\s+([○\-－])\s+([○\-－])(?=\s|$)")
SELL_ROW = re.compile(r"^([0-9][0-9A-Z]{2}[0-9A-Z])\s+(\S.*)$")


def parse_lineup(pdf_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    main, sell = [], []
    header = {}
    for page in reader.pages:
        text = page.extract_text()
        if not header:
            m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日時点", text)
            n = re.search(r"寄付銘柄数 リアルタイム銘柄数\s*\n?([\d,]+)\s+([\d,]+)", text)
            if m and n:
                header = {"as_of": f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}",
                          "n_yoritsuki": int(n[1].replace(",", "")), "n_realtime": int(n[2].replace(",", ""))}
        if "寄付取引 リアルタイム取引" in text:
            for line in text.splitlines():
                for code, name, yori, rt in MAIN_ROW.findall(line):
                    main.append({"code4": code, "name": name.strip(), "yoritsuki": yori == "○", "realtime": rt == "○"})
        elif "売り注文のみ" in text:
            for line in text.splitlines():
                mm = SELL_ROW.match(line.strip())
                if mm:
                    sell.append({"code4": mm[1], "name": mm[2].strip()})
    return pd.DataFrame(main).drop_duplicates("code4"), pd.DataFrame(sell).drop_duplicates("code4"), header


def parse_sell_only(pdf_path: Path) -> tuple[pd.DataFrame, str]:
    """売り注文のみ受付銘柄の一覧（2列）。"""
    from pypdf import PdfReader
    text = "\n".join(p.extract_text() for p in PdfReader(str(pdf_path)).pages)
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日現在", text)
    as_of = f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}" if m else ""
    rows = re.findall(r"([0-9][0-9A-Z]{2}[0-9A-Z])\s+(\S+)", text)
    return pd.DataFrame(rows, columns=["code4", "name"]).drop_duplicates("code4"), as_of


# ---------------------------------------------------------------- 日付
def month_end_dates(dates: np.ndarray) -> list[int]:
    """各暦月の最終営業日の位置（FIRST_P_MONTH〜LAST_P_MONTH）。"""
    months = pd.Series(dates).str[:7].to_numpy()
    out = []
    for mo in sorted(set(months)):
        if FIRST_P_MONTH <= mo <= LAST_P_MONTH:
            out.append(int(np.nonzero(months == mo)[0].max()))
    return out


def first_bday_of_next_month(dates: np.ndarray, t: int) -> int | None:
    mo = dates[t][:7]
    for k in range(t + 1, len(dates)):
        if dates[k][:7] != mo:
            return k
    return None


# ---------------------------------------------------------------- 本体
def coverage_table(df: pd.DataFrame, col: str, bands: list[tuple[str, float, float]]) -> list[dict]:
    rows = []
    for label, lo, hi in bands:
        sub = df[(df[col] >= lo) & (df[col] < hi)]
        rows.append({"band": label, "n": int(len(sub)),
                     "kabumini_buyable": int(sub["buyable"].sum()),
                     "share": round(float(sub["buyable"].mean()), 4) if len(sub) else None})
    return rows


def universe_frame(m, u, mcap, adv, t: int, buyable_codes: set[str]) -> pd.DataFrame:
    js = np.nonzero(u[t])[0]
    df = pd.DataFrame({"code": m.codes[js], "close": m.C[t, js], "mcap": mcap[t, js], "adv": adv[t, js]})
    df["buyable"] = df["code"].isin(buyable_codes)
    return df


def main() -> None:
    cfg = load_config()
    out: dict = {"created_at_jst": datetime.now(JST).isoformat(timespec="seconds")}

    # 取扱銘柄一覧
    lineup, sell01, header = parse_lineup(LIST_DIR / "pdf_01.pdf")
    sell02, sell02_asof = parse_sell_only(LIST_DIR / "pdf_02.pdf")
    fetched = (LIST_DIR / "fetched_at_jst.txt").read_text(encoding="utf-8").strip()
    lineup["sell_only_2024_12_17"] = lineup["code4"].isin(sell01["code4"])
    lineup["sell_only_2025_06_10"] = lineup["code4"].isin(sell02["code4"])
    out_dir = ROOT / "logs" / "v2_e0"
    out_dir.mkdir(parents=True, exist_ok=True)
    lineup.to_csv(out_dir / f"kabumini_lineup_{LIST_DATE}.csv", index=False, encoding="utf-8")
    out["lineup"] = {
        "source": "https://www.rakuten-sec.co.jp/web/domestic/ols/lineup/pdf/pdf_01.pdf（2024-12-16 更新。今の取扱銘柄のページからはリンクされていない）、"
                  "pdf_02.pdf（売り注文のみ受付銘柄、2025-06-10 現在）",
        "fetched_at_jst": fetched, "header": header,
        "parsed_rows": int(len(lineup)), "parsed_yoritsuki": int(lineup["yoritsuki"].sum()),
        "parsed_realtime": int(lineup["realtime"].sum()),
        "sell_only_2024_12_17": int(len(sell01)), "sell_only_2025_06_10": int(len(sell02)),
        "sell_only_2025_06_10_as_of": sell02_asof,
        "sell_only_2025_06_10_not_in_lineup": int((~sell02["code4"].isin(lineup["code4"])).sum()),
    }
    # 買える銘柄：寄付取引が○で、2024-12-17 の「売り注文のみ」に入っていない（主）
    buy4 = set(lineup.loc[lineup["yoritsuki"] & ~lineup["sell_only_2024_12_17"], "code4"])
    buy4_strict = buy4 - set(sell02["code4"])
    to5 = lambda s: {c + "0" for c in s}  # noqa: E731
    buyable, buyable_strict = to5(buy4), to5(buy4_strict)

    m = load_market(cfg)
    assert m.dates[-1] < cfg["data"]["holdout_start"]
    u = universe_mask(m, cfg)
    mcap = market_cap(m)
    adv = avg_turnover(m, cfg["universe"]["turnover_window"])
    P = month_end_dates(m.dates)
    out["prediction_dates"] = {"n": len(P), "first": str(m.dates[P[0]]), "last": str(m.dates[P[-1]])}

    # (1) 取扱銘柄の割合
    mc_bands = [("50億円未満", 0, 5e9), ("50〜100億円", 5e9, 1e10), ("100〜200億円", 1e10, 2e10),
                ("200〜500億円", 2e10, 5.0000001e10)]
    adv_bands = [("3,000万〜5,000万円", 3e7, 5e7), ("5,000万〜1億円", 5e7, 1e8), ("1〜3億円", 1e8, 3e8),
                 ("3億円以上", 3e8, np.inf)]
    res1 = {}
    for d in [CHECK_DATE, LIST_DATE]:
        t = m.date_index[d]
        df = universe_frame(m, u, mcap, adv, t, buyable)
        strict = df["code"].isin(buyable_strict).mean()
        res1[d] = {"universe": int(len(df)), "kabumini_buyable": int(df["buyable"].sum()),
                   "share": round(float(df["buyable"].mean()), 4),
                   "share_excluding_sell_only_2025_06_10": round(float(strict), 4),
                   "by_market_cap": coverage_table(df, "mcap", mc_bands),
                   "by_turnover": coverage_table(df, "adv", adv_bands)}
        # 時価総額・売買代金の5分位（その日のユニバース内）
        for col, key in [("mcap", "by_market_cap_quintile"), ("adv", "by_turnover_quintile")]:
            q = pd.qcut(df[col], 5, labels=[f"Q{i}" for i in range(1, 6)])
            res1[d][key] = [{"quintile": str(k), "n": int(len(g)), "share": round(float(g["buyable"].mean()), 4),
                             "range_oku_yen": [round(float(g[col].min()) / 1e8, 1), round(float(g[col].max()) / 1e8, 1)]}
                            for k, g in df.groupby(q, observed=True)]
    shares_by_month = []
    for t in P:
        js = np.nonzero(u[t])[0]
        shares_by_month.append(float(np.isin(m.codes[js], list(buyable)).mean()))
    sb = pd.Series(shares_by_month, index=[str(m.dates[t]) for t in P])
    res1["by_prediction_date"] = {"min": round(float(sb.min()), 4), "median": round(float(sb.median()), 4),
                                  "max": round(float(sb.max()), 4),
                                  "first": [str(sb.index[0]), round(float(sb.iloc[0]), 4)],
                                  "by_year_median": {y: round(float(g.median()), 4)
                                                     for y, g in sb.groupby(sb.index.str[:4])}}
    out["e0_1_kabumini_coverage"] = res1

    # (2) 100株を買えるユニバースの割合
    lot = cfg["order"]["lot_size"]
    cap_ratio = cfg["order"]["max_order_to_adv"]
    up = cfg["order"]["limit_up_pct"]

    def lot_share(t: int, capital: float, n: int) -> tuple[float, float]:
        js = np.nonzero(u[t])[0]
        lim = np.array([limit_price(c, up) for c in m.C[t, js]])
        budget = capital / n
        shares = np.floor(budget / (lim * lot) + 1e-9) * lot
        price_ok = shares >= lot
        ok = price_ok & (shares * lim <= adv[t, js] * cap_ratio + 1e-9)
        return float(ok.mean()), float(price_ok.mean())

    res2 = {"rule": "予算 = 資金 ÷ N、指値 = 終値 × 1.02 を呼値で切り下げ、100株 × 指値 ≤ 予算、かつ 注文金額 ≤ 20日平均売買代金の1%",
            "rows": []}
    for capital in CAPITALS:
        for n in NS_LOT:
            a, b = lot_share(m.date_index[CHECK_DATE], capital, n)
            per = [lot_share(t, capital, n)[0] for t in P]
            res2["rows"].append({"capital": capital, "N": n, "budget_per_stock": round(capital / n),
                                 "share_2025_09_26": round(a, 4), "share_price_only_2025_09_26": round(b, 4),
                                 "share_median_prediction_dates": round(float(np.median(per)), 4),
                                 "share_min_prediction_dates": round(float(np.min(per)), 4)})
    out["e0_2_lot100_affordable"] = res2

    # (付) かぶミニ（50万円・N=20）：1株も買えない銘柄、100株以上になり通常の注文が要る銘柄の割合、余力拘束
    res_k = {}
    widths = np.vectorize(limit_width)
    for d in [CHECK_DATE]:
        t = m.date_index[d]
        js = np.nonzero(u[t])[0]
        c = m.C[t, js]
        budget = 500_000 * 0.97 / 20
        res_k[d] = {"budget_per_stock": round(budget), "share_price_above_budget": round(float((c > budget).mean()), 4),
                    "share_100_or_more_shares": round(float((np.floor(budget / c) >= 100).mean()), 4)}
    fracs, fr_all = [], []
    for t in P:
        js = np.nonzero(u[t])[0]
        c = m.C[t, js]
        bind = np.ceil((c + widths(c)) * (1 + SPREAD))     # 拘束の単価（1円未満は切上げ）
        f = c / bind                                       # 拘束の単価に対する、寄付が前日終値と同じときの約定金額の割合
        fracs.append(float(np.mean(f)))
        fr_all.append(f)
    f_all = np.concatenate(fr_all)
    f1 = float(np.median(fracs))
    res_k["binding"] = {
        "rule": "かぶミニの成行（寄付取引を含む）の買い：ストップ高の値段（基準値段 + 制限値幅）× 1.0022（1円未満切上げ）× 株数 で余力を拘束",
        "invested_fraction_one_stage_median_of_monthly_means": round(f1, 4),
        "invested_fraction_quantiles_all_rows": {q: round(float(np.quantile(f_all, q)), 4) for q in [0.05, 0.25, 0.5, 0.75, 0.95]},
        "invested_fraction_two_stage_approx": round(f1 + (1 - f1) * f1, 4),
        "note": "株価の水準と制限値幅の表だけで決まる値（基準値段＝予測日の終値で近似）。リターンは使っていない",
    }
    out["kabumini_order_checks"] = res_k

    # (3) 市場と関係ない月の値動きの大きさ（標準偏差だけ）
    Qo, Qc_ff = m.Qo, m.Qc_ff
    stds, iqrs, ns = [], [], []
    for t in P[:-1]:
        S = first_bday_of_next_month(m.dates, t)
        B = S + 1
        S2 = first_bday_of_next_month(m.dates, S)
        js = np.nonzero(u[t])[0]
        entry = Qo[B, js]
        exit_ = np.where(np.isnan(Qo[S2, js]), Qc_ff[S2 - 1, js], Qo[S2, js])
        ok = ~np.isnan(entry) & ~np.isnan(exit_)
        r = exit_[ok] / entry[ok] - 1
        stds.append(float(np.std(r, ddof=1)))
        q75, q25 = np.quantile(r, [0.75, 0.25])
        iqrs.append(float((q75 - q25) / 1.349))
        ns.append(int(ok.sum()))
        del r  # 平均は計算しない・出力しない
    sigma = float(np.median(stds))
    sigma_iqr = float(np.median(iqrs))
    n_months = len(stds)
    years = n_months / 12
    res3 = {"months": n_months, "first_holding_start": str(m.dates[first_bday_of_next_month(m.dates, P[0]) + 1]),
            "last_holding_end": str(m.dates[first_bday_of_next_month(m.dates, first_bday_of_next_month(m.dates, P[-2]))]),
            "stocks_per_month_median": int(np.median(ns)),
            "monthly_idio_std_median": round(sigma, 4),
            "monthly_idio_std_quantiles": {q: round(float(np.quantile(stds, q)), 4) for q in [0.0, 0.25, 0.5, 0.75, 1.0]},
            "monthly_idio_std_iqr_based_median": round(sigma_iqr, 4),
            "noise": []}
    for n in NS_NOISE:
        one_year = sigma * math.sqrt(12 / n)
        res3["noise"].append({"N": n, "one_year_sd": round(one_year, 4), "one_year_2sd": round(2 * one_year, 4),
                              "one_year_sd_iqr_based": round(sigma_iqr * math.sqrt(12 / n), 4),
                              "annualized_mean_se_over_period": round(one_year / math.sqrt(years), 4)})
    res3["note"] = ("月次リターン = 予測日の2営業日後の始値 → 翌々月の第1営業日の始値（調整後。始値が無ければ前の営業日までの最後の終値）。"
                    "月ごとのユニバース内の標準偏差 = ユニバース平均との差の標準偏差。N銘柄の等金額で、銘柄どうしの差が独立なら"
                    "1年の超過リターンの標準偏差 ≈ σ × √(12 ÷ N)。平均・特徴量・点数との関係は計算していない")
    out["e0_3_idiosyncratic_volatility"] = res3

    # (付) 長い期間の特徴量と Light の5年分
    dates = pd.to_datetime(pd.Series(m.dates))
    gaps = []
    for t in P:
        D = dates[t]
        if D - pd.DateOffset(years=5) < dates.iloc[0]:
            continue
        lo, hi = D - pd.DateOffset(years=5), D - pd.DateOffset(years=4)
        gaps.append(int(((dates >= lo) & (dates < hi)).sum()))
    out["light_long_feature_check"] = {
        "rule": f"予測日 D の5年前〜4年前（暦）の営業日数が {LONG_WINDOW} 以上なら、4年の学習期間の最初の行で {LONG_WINDOW} 営業日前の値が Light の範囲にある",
        "long_window": LONG_WINDOW, "months_checked": len(gaps), "min_bdays": min(gaps), "max_bdays": max(gaps),
        "margin_bdays": min(gaps) - LONG_WINDOW, "ok": bool(min(gaps) >= LONG_WINDOW)}

    path = ROOT / "reports" / "v2_e0.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
