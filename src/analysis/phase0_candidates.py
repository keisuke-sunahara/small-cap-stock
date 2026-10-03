"""フェーズ0：運用資金と保有銘柄数Nに対して、100株単位で買える候補銘柄数を概算する。

ホールドアウト（直近12か月）に触れないよう、対象日は SAFE_MAX_DATE 以前に限定する
（ホールドアウト開始日はフェーズ1で決めるが、どう決めても2025-07以降になるため）。
取得したデータは保存せず、集計結果（銘柄数）だけを出力する。

実行: python -m src.analysis.phase0_candidates
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import pandas as pd

from src.config import ROOT, load_config
from src.data.jquants import client_from_config

SAFE_MAX_DATE = date(2025, 6, 30)
DEFAULT_SAMPLE_DATES = ["2024-09-30", "2024-12-27", "2025-03-31", "2025-06-30"]
COMMON_MARKET_NAMES = ("プライム", "スタンダード", "グロース")
N_CANDIDATES = (2, 3, 5)


def business_days(calendar_rows: list[dict]) -> list[str]:
    # HolDiv: "0" は非営業日。それ以外（営業日・半日立会）を営業日とする
    return sorted(r["Date"] for r in calendar_rows if str(r["HolDiv"]) != "0")


def screen(master: pd.DataFrame, bars: pd.DataFrame, valuation: pd.DataFrame, as_of: str,
           cfg: dict) -> pd.DataFrame:
    """as_of 時点のユニバース条件を当てはめた銘柄表を返す（概算用）。

    master: Code, MktNm, S17 / bars: Date, Code, C, Va（as_of までの20営業日分）/ valuation: Code, MktCap(百万円)
    """
    ucfg = cfg["universe"]
    common = master[master["MktNm"].astype(str).str.contains("|".join(COMMON_MARKET_NAMES))
                    & (master["S17"].astype(str) != "99")]
    window = bars[bars["Date"] <= as_of]
    adv = window.groupby("Code")["Va"].mean().rename("adv20")
    days = window.groupby("Code")["Va"].count().rename("n_days")
    close = window[window["Date"] == as_of].set_index("Code")["C"].rename("close")
    df = (common[["Code"]].drop_duplicates().set_index("Code")
          .join([close, adv, days], how="left")
          .join(valuation.set_index("Code")["MktCap"].rename("mktcap_mn"), how="left"))
    df = df.dropna(subset=["close", "adv20", "mktcap_mn"])
    df = df[(df["n_days"] >= ucfg["turnover_window"])
            & (df["mktcap_mn"] * 1e6 <= ucfg["max_market_cap_jpy"])
            & (df["adv20"] >= ucfg["min_avg_turnover_jpy"])
            & (df["close"] >= ucfg["min_price_jpy"])]
    return df


def count_affordable(universe: pd.DataFrame, capital: float, n: int, lot: int, max_adv_ratio: float) -> int:
    """予算内で買える最大株数（売買単位の倍数）が1単位以上で、その注文金額が平均売買代金の上限以下の銘柄数。

    概算のため、指値ではなく as_of の終値で計算する。
    """
    budget = capital / n
    shares = (budget // (universe["close"] * lot)) * lot
    amount = shares * universe["close"]
    ok = (shares >= lot) & (amount <= universe["adv20"] * max_adv_ratio)
    return int(ok.sum())


def to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", nargs="*", default=DEFAULT_SAMPLE_DATES)
    parser.add_argument("--rate", type=float, default=4.0, help="1分あたりのリクエスト数（Freeプランは上限5）")
    args = parser.parse_args()
    for d in args.dates:
        if date.fromisoformat(d) > SAFE_MAX_DATE:
            raise SystemExit(f"{d} はホールドアウトに入る可能性があるため使えません（上限 {SAFE_MAX_DATE}）")

    cfg = load_config()
    client = client_from_config(cfg, rate_limit_per_min=args.rate)
    # Free プランの取得範囲（2年12週前〜）に収まるよう、開始は 2024-08-01 とする
    cal = client.get_all("/markets/calendar", **{"from": "2024-08-01", "to": SAFE_MAX_DATE.isoformat()})
    bdays = business_days(cal)

    results = []
    for as_of in args.dates:
        if as_of not in bdays:
            raise SystemExit(f"{as_of} は営業日ではありません")
        idx = bdays.index(as_of)
        window_days = bdays[idx - cfg["universe"]["turnover_window"] + 1: idx + 1]
        master = pd.DataFrame(client.get_all("/equities/master", date=as_of))
        if not results:
            print("市場区分の内訳:", master["MktNm"].value_counts().to_dict())
            print("商品区分の内訳:", master.get("ProdCat", pd.Series(dtype=str)).value_counts().to_dict())
        bars = pd.concat([pd.DataFrame(client.get_all("/equities/bars/daily", date=d)) for d in window_days])
        bars = to_num(bars, ["C", "Va"])
        valuation = to_num(pd.DataFrame(client.get_all("/equities/valuation", date=as_of)), ["MktCap"])
        uni = screen(master, bars, valuation, as_of, cfg)
        row = {"date": as_of, "listed_common": int(master["MktNm"].astype(str)
                                                   .str.contains("|".join(COMMON_MARKET_NAMES)).sum()),
               "universe": len(uni),
               "price_quantiles": uni["close"].quantile([0.25, 0.5, 0.75]).round(0).tolist()}
        for n in N_CANDIDATES:
            row[f"affordable_N{n}"] = count_affordable(uni, cfg["capital"]["initial_capital_jpy"], n,
                                                       cfg["order"]["lot_size"], cfg["order"]["max_order_to_adv"])
        print(json.dumps(row, ensure_ascii=False))
        results.append(row)

    out = ROOT / "reports" / "phase0_candidates.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
