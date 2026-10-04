"""破綻などで株価が大きく下がってから上場廃止になった銘柄が、上場廃止の前にユニバースに残っていたかの確認（評価役の指摘）。

対象：期間中に上場廃止になった普通株のうち、次のどちらかに当てはまる銘柄
- 最後に売買が成立した日の終値（調整前）が、ユニバースの株価の下限（50円）未満
- 最後の終値が、上場廃止前の250営業日の最高値（分割調整後）の20%以下（−80%以上の下落）

銘柄ごとに出すもの
- 上場廃止前の最後の「その他」の区間の開始日（貸借信用区分が「その他（3）」になった日。整理銘柄の指定などで変わる）。
  日付 d の銘柄一覧は d の前の営業日の 17:30 頃に公開される（公式仕様）。ユニバースでは d の一覧を d の引け後に使う
- ユニバース（「その他」の除外あり＝現在の基本ルール）に最後に入っていた日と、その翌営業日に外れた理由
- 除外なしの場合の、最後に入っていた日（除外の効果を見るため）
- 最後にユニバースに入っていた日の終値から、最後の終値までの下落率（その日の予測で選ばれて買った場合の損失の目安）

あわせて、ベースライン（モメンタム・リバーサルは予算の決め方2通り、ランダムは予算固定で100回）をコスト0.3%で実行し、
対象の銘柄を上場廃止の WINDOW 営業日前以内に買った取引と、その損益を出す。

この集計は報告のためのもので、売買のルールには使わない。
出力：reports/phase3_distressed.json
実行：python -m src.analysis.phase3_distressed
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

from src.backtest.baselines import RandomScore, momentum, reversal
from src.backtest.engine import Backtester, Rules
from src.backtest.market import load_market
from src.backtest.universe import avg_turnover, market_cap, universe_mask
from src.config import ROOT, load_config
from src.data.load import read_raw

LOOKBACK = 250
DROP_RATIO = 0.2
WINDOW = 30       # 上場廃止の何営業日前までを「直前」とみなすか（phase2_delisting_signal と同じ）
RANDOM_SEEDS = 100


def reasons(m, cfg, adv, mcap, t: int, j: int) -> list[str]:
    """位置 t で銘柄 j がユニバースに入らない理由。"""
    u = cfg["universe"]
    out = []
    if not m.common[t, j]:
        out.append("上場していない（上場廃止）")
        return out
    if np.isnan(m.C[t, j]):
        out.append("売買不成立")
    elif m.C[t, j] < u["min_price_jpy"]:
        out.append(f"株価{u['min_price_jpy']}円未満")
    if m.margin_other[t, j]:
        out.append("貸借信用区分「その他」")
    if not adv[t, j] >= u["min_avg_turnover_jpy"]:
        out.append("売買代金の下限未満")
    if not mcap[t, j] <= u["max_market_cap_jpy"]:
        out.append("時価総額の上限超え（または株式数が未開示）")
    return out


def last_true(a: np.ndarray, end: int) -> int:
    idx = np.flatnonzero(a[: end + 1])
    return int(idx[-1]) if len(idx) else -1


def baseline_buys(m, u, adv, cfg, targets: dict[int, int]) -> dict:
    """ベースラインで、対象の銘柄（位置 j → 最後の上場日の位置）を上場廃止の WINDOW 営業日前以内に買った取引。"""
    start, end = cfg["backtest"]["trade_start_date"], str(m.dates[-1])
    runs = [(f"{name}_{budget}", fn, budget) for name, fn in (("momentum_20d", momentum(m, 20)),
                                                              ("reversal_5d", reversal(m, 5)))
            for budget in ("fixed", "min_equity")]
    runs += [(f"random_seed{seed}_fixed", RandomScore(len(m.codes), seed), "fixed") for seed in range(RANDOM_SEEDS)]
    out: dict = {"runs": {}, "buys": []}
    for name, fn, budget in runs:
        res = Backtester(m, u, adv, Rules.from_config(cfg, budget_mode=budget)).run(fn, start, end)
        tr = res.trades
        n = 0
        for b in tr[(tr["side"] == "buy") & tr["j"].isin(list(targets))].itertuples():
            t = m.date_index[b.date]
            if targets[b.j] - t >= WINDOW:
                continue
            sells = tr[(tr["side"] == "sell") & (tr["j"] == b.j) & (tr["date"] >= b.date)]
            sell = sells.iloc[0] if len(sells) else None
            n += 1
            rec = {"run": name, "Code": str(m.codes[b.j]), "buy_date": str(b.date), "buy_price": float(b.price),
                   "shares": float(b.shares)}
            if sell is not None:
                # 価格は調整前。分割があっても株数の側で直るよう、金額どうしで比べる（コストは含めない）
                rec.update({"sell_date": str(sell["date"]), "sell_price": float(sell["price"]),
                            "sell_reason": sell["reason"],
                            "ret": round(float(sell["price"] * sell["shares"] / (b.price * b.shares) - 1), 4)})
            out["buys"].append(rec)
        out["runs"][name] = n
    rnd = [v for k, v in out["runs"].items() if k.startswith("random")]
    out["random_buys_per_run_mean"] = float(np.mean(rnd))
    out["random_runs_with_any_buy"] = int(sum(v > 0 for v in rnd))
    return out


def main() -> None:
    cfg = load_config()
    m = load_market(cfg)
    T = len(m.dates)
    adv = avg_turnover(m, cfg["universe"]["turnover_window"])
    mcap = market_cap(m)
    u_ex = universe_mask(m, cfg, adv)
    cfg_no = copy.deepcopy(cfg)
    cfg_no["universe"]["exclude_margin_other"] = False
    u_no = universe_mask(m, cfg_no, adv)
    qc_ff = m.Qc_ff
    c_ff = m.C_ff

    master = read_raw(cfg, "master", columns=["Code", "CoName"])
    names = (master.assign(Code=master["Code"].astype(str)).drop_duplicates("Code", keep="last")
             .set_index("Code")["CoName"])

    delisted = np.flatnonzero((m.last_listed < T - 1) & m.common.any(axis=0))
    rows = []
    for j in delisted:
        last = int(m.last_listed[j])
        lt = last_true(~np.isnan(m.C[:, j]), last)
        if lt < 0:
            continue
        last_c = float(m.C[lt, j])
        peak = float(np.nanmax(m.Qc[max(0, last - LOOKBACK): last + 1, j]))
        ratio = float(qc_ff[lt, j] / peak)
        if not (last_c < cfg["universe"]["min_price_jpy"] or ratio <= DROP_RATIO):
            continue
        # 上場廃止前の最後の「その他」の区間の開始日
        other_start = None
        if m.margin_other[last, j]:
            t = last
            while t > 0 and m.margin_other[t - 1, j] and m.common[t - 1, j]:
                t -= 1
            other_start = str(m.dates[t])
        lu = last_true(u_ex[:, j], last)
        ln = last_true(u_no[:, j], last)
        rec = {"Code": str(m.codes[j]), "CoName": str(names.get(str(m.codes[j]), "")),
               "last_listed": str(m.dates[last]), "last_traded": str(m.dates[lt]), "last_close": last_c,
               "last_close_to_250d_high": round(ratio, 4), "other_start": other_start,
               "other_bdays_before_delisting": (last - int(m.date_index[other_start])) if other_start else None}
        if lu >= 0:
            rec.update({"last_in_universe": str(m.dates[lu]), "bdays_from_last_universe_to_delisting": last - lu,
                        "close_on_last_universe_day": float(m.C[lu, j]),
                        "ret_last_universe_close_to_last_close": round(float(qc_ff[lt, j] / m.Qc[lu, j] - 1), 4),
                        "reasons_next_day": reasons(m, cfg, adv, mcap, lu + 1, j) if lu + 1 < T else []})
        else:
            rec.update({"last_in_universe": None})
        rec["last_in_universe_without_exclusion"] = str(m.dates[ln]) if ln >= 0 else None
        rec["exclusion_removed_bdays"] = (int(u_no[: last + 1, j].sum()) - int(u_ex[: last + 1, j].sum()))
        # 上場廃止の直前（WINDOW 営業日以内）にユニバースに入っていた日数
        rec["universe_days_within_window"] = int(u_ex[max(0, last - WINDOW + 1): last + 1, j].sum())
        rec["universe_days_within_window_without_exclusion"] = int(u_no[max(0, last - WINDOW + 1): last + 1, j].sum())
        # 最後にユニバースに入っていた日の終値（調整前）と、そこからの最安値
        if lu >= 0:
            rec["min_close_after_last_universe"] = float(np.nanmin(m.C[lu: last + 1, j]))
        rec["last_close_ff"] = float(c_ff[last, j])
        rows.append(rec)

    df = pd.DataFrame(rows)
    in_win = df["universe_days_within_window"] > 0
    summary = {
        "delisted_common": int(len(delisted)),
        "distressed": int(len(df)),
        "criteria": {"last_close_below_jpy": cfg["universe"]["min_price_jpy"],
                     "or_last_close_to_high_at_most": DROP_RATIO, "high_lookback_bdays": LOOKBACK},
        "ever_in_universe": int(df["last_in_universe"].notna().sum()),
        "in_universe_within_last_30bdays": int(in_win.sum()),
        "in_universe_within_last_30bdays_without_exclusion": int((df["universe_days_within_window_without_exclusion"] > 0).sum()),
        "other_flag_on_last_listed_day": int(df["other_start"].notna().sum()),
    }
    targets = {m.code_index[c]: m.date_index[d] for c, d in zip(df["Code"], df["last_listed"])}
    bb = baseline_buys(m, u_ex, adv, cfg, targets)
    summary["baseline_buys_within_window"] = {k: v for k, v in bb["runs"].items() if not k.startswith("random")}
    summary["random_buys_within_window_per_run_mean"] = bb["random_buys_per_run_mean"]
    summary["random_runs_with_any_buy_within_window"] = f"{bb['random_runs_with_any_buy']}/{RANDOM_SEEDS}"
    out = {"summary": summary, "baseline_buys": bb["buys"],
           "examples": df[df["Code"].isin(["16060", "57590"])].to_dict(orient="records"),
           "in_universe_within_last_30bdays": df[in_win].to_dict(orient="records"),
           "all": df.to_dict(orient="records")}
    path = ROOT / "reports" / "phase3_distressed.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"保存: {path}")


if __name__ == "__main__":
    main()
