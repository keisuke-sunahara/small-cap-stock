"""フェーズ2：上場廃止が決まった銘柄（整理銘柄など）を、その時点の情報だけで見分けられるかの確認。

調べる情報：J-Quants の上場銘柄一覧の貸借信用区分（Mrgn。1:信用、2:貸借、3:その他）。
東証は監理銘柄・整理銘柄に指定した銘柄の制度信用取引を止めるため、「その他（3）」に変わると考えられる。
銘柄一覧は日付を指定するとその時点の情報が返る（公式仕様）。日付 d の一覧は d の前の営業日の 17:30 頃に公開される
ため、予測日 p の引け後には p（と p の翌営業日）の一覧が分かっている。ここでは保守的に p の一覧だけを使う。

集計（普通株のみ。ホールドアウトより前のデータだけ）
1. 見逃し：期間中に上場廃止になった銘柄のうち、最後の上場日に区分が「その他」だった割合と、
   「その他」に変わってから最後の上場日までの営業日数（何日前に分かるか）
2. 誤検出：「その他」になった期間（銘柄ごとの連続した区間）を、上場廃止で終わったもの・上場の直後のもの・
   その他（「その他」から戻ったもの）に分ける
3. ユニバース（予測日）に入っていた「その他」の銘柄の数と、そのうち30営業日以内に上場廃止になった割合
4. ベースラインで選ばれた銘柄のうち、30営業日以内に上場廃止になったものが、予測日に「その他」だったか

この集計は判断材料で、売買のルールには使わない（ルールに加えるにはユーザーの承認が必要）。
出力：reports/phase2_delisting_signal.json
実行：python -m src.analysis.phase2_delisting_signal
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.backtest.baselines import past_return
from src.backtest.market import load_market
from src.backtest.target import make_weeks
from src.backtest.universe import avg_turnover, universe_mask
from src.config import ROOT, load_config
from src.data.load import read_raw

WINDOW = 30


def episodes(flag: np.ndarray, listed: np.ndarray, j: int) -> list[tuple[int, int]]:
    """銘柄 j の、区分が「その他」の連続した区間 [start, end]（上場している日だけを見る）。"""
    days = np.flatnonzero(listed[:, j])
    out = []
    start = None
    prev = None
    for t in days:
        if flag[t, j]:
            if start is None:
                start = t
            prev = t
        elif start is not None:
            out.append((start, prev))
            start = None
    if start is not None:
        out.append((start, prev))
    return out


def main() -> None:
    cfg = load_config()
    # 除外前のユニバースで集計する（2026-10-04 の結果の再現のため。除外の判断材料を作る集計なので）
    cfg["universe"]["exclude_margin_other"] = False
    m = load_market(cfg)
    T, N = len(m.dates), len(m.codes)
    master = read_raw(cfg, "master", columns=["Code", "Mrgn"])
    master["Code"] = master["Code"].astype(str)
    master = master[master["Code"].isin(m.code_index)]
    ti = master["Date"].map(m.date_index).to_numpy()
    tj = master["Code"].map(m.code_index).to_numpy()
    other = np.zeros((T, N), dtype=bool)
    other[ti, tj] = master["Mrgn"].astype(str).eq("3").to_numpy()
    listed = np.zeros((T, N), dtype=bool)
    listed[ti, tj] = True
    other &= m.common
    first_listed = np.argmax(listed, axis=0)

    # 1. 上場廃止になった普通株（期間の最初から上場していた銘柄も含む）
    delisted = np.flatnonzero((m.last_listed < T - 1) & m.common.any(axis=0))
    leads = []
    detected = 0
    for j in delisted:
        last = m.last_listed[j]
        if other[last, j]:
            detected += 1
            ep = [e for e in episodes(other, listed, j) if e[1] == last][0]
            leads.append(int(last - ep[0]))
    leads = np.array(leads)
    # 最後の上場日の前20営業日の値動き（分割調整済み）。TOB・合併による廃止は値動きが小さいと考えられる
    qc = m.Qc_ff
    move = {True: [], False: []}
    for j in delisted:
        last = m.last_listed[j]
        if last >= 20:
            move[bool(other[last, j])].append(abs(qc[last, j] / qc[last - 20, j] - 1))
    res: dict = {"delisted_common": {
        "codes": int(len(delisted)), "other_on_last_day": int(detected),
        "ratio": float(detected / len(delisted)),
        "lead_bdays_quantiles": {str(q): float(np.quantile(leads, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "lead_bdays_ge_5_ratio": float((leads >= 5).mean()),
        "lead_bdays_ge_10_ratio": float((leads >= 10).mean()),
        "abs_move_last20_median_flagged": float(np.nanmedian(move[True])),
        "abs_move_last20_median_not_flagged": float(np.nanmedian(move[False])),
        "abs_move_last20_over_20pct_flagged": float(np.nanmean(np.array(move[True]) > 0.2)),
        "abs_move_last20_over_20pct_not_flagged": float(np.nanmean(np.array(move[False]) > 0.2)),
    }}

    # 2. 「その他」の区間の分類
    kinds = {"ends_with_delisting": [], "right_after_listing": [], "returned": [], "ongoing_at_data_end": []}
    for j in range(N):
        for s, e in episodes(other, listed, j):
            length = int(e - s + 1)
            if e == m.last_listed[j] and e < T - 1:
                kinds["ends_with_delisting"].append(length)
            elif e == T - 1:
                kinds["ongoing_at_data_end"].append(length)
            elif s - first_listed[j] <= 2 and first_listed[j] > 0:
                kinds["right_after_listing"].append(length)
            else:
                kinds["returned"].append(length)
    res["other_episodes"] = {k: {"count": len(v), "median_length_bdays": float(np.median(v)) if v else None}
                             for k, v in kinds.items()}

    # 3. 予測日のユニバースの中の「その他」
    adv = avg_turnover(m, cfg["universe"]["turnover_window"])
    u = universe_mask(m, cfg, adv)
    weeks = [w for w in make_weeks(m.dates) if m.dates[w.first] >= cfg["backtest"]["trade_start_date"]
             and w.pred >= 0]
    preds = np.array([w.pred for w in weeks])
    n_other = (u[preds] & other[preds]).sum(axis=1)
    # 除外した場合に外れる銘柄の数（毎週）。100株を買える銘柄（指値 ≤ 予算）に限った数も出す
    budget = cfg["capital"]["initial_capital_jpy"] / cfg["capital"]["n_holdings"]
    lot = cfg["order"]["lot_size"]
    affordable = m.C[preds] * (1 + cfg["order"]["limit_up_pct"]) * lot <= budget
    n_other_aff = (u[preds] & other[preds] & affordable).sum(axis=1)
    years = pd.Series(m.dates[preds]).str[:4]
    res["excluded_per_week"] = {
        "universe": {"mean": float(n_other.mean()), "median": float(np.median(n_other)),
                     "p90": float(np.quantile(n_other, 0.9)), "max": int(n_other.max()),
                     "share_of_universe_mean": float((n_other / u[preds].sum(axis=1)).mean()),
                     "by_year_mean": pd.Series(n_other).groupby(years).mean().round(2).to_dict()},
        "affordable_at_n": {"mean": float(n_other_aff.mean()), "median": float(np.median(n_other_aff)),
                            "p90": float(np.quantile(n_other_aff, 0.9)), "max": int(n_other_aff.max()),
                            "share_of_affordable_mean": float(
                                (n_other_aff / (u[preds] & affordable).sum(axis=1)).mean())},
    }
    rows = [(p, j) for p in preds for j in np.flatnonzero(u[p] & other[p])]
    ok = [(p, j) for p, j in rows if p + WINDOW <= T - 1]
    dl = [(m.last_listed[j] < T - 1) and (m.last_listed[j] <= p + WINDOW) for p, j in ok]
    allu = [(p, j) for p in preds if p + WINDOW <= T - 1 for j in np.flatnonzero(u[p])]
    dl_all = np.array([(m.last_listed[j] < T - 1) and (m.last_listed[j] <= p + WINDOW) for p, j in allu])
    oth_all = np.array([other[p, j] for p, j in allu])
    later = {h: float(np.mean([(m.last_listed[j] < T - 1) and (m.last_listed[j] <= p + h) for p, j in ok]))
             for h in (60, 120, 250)}
    res["universe_on_pred_dates"] = {
        "other_rows_delisted_within": later,
        "other_per_week": {"mean": float(n_other.mean()), "max": int(n_other.max()),
                           "weeks_with_any": int((n_other > 0).sum()), "weeks": int(len(preds))},
        "other_rows": len(ok), "other_rows_delisted_within_30": int(np.sum(dl)),
        "other_rows_delisted_ratio": float(np.mean(dl)) if ok else None,
        "universe_rows_delisted_within_30": int(dl_all.sum()),
        "share_of_delisted_rows_flagged_other": float(oth_all[dl_all].mean()) if dl_all.any() else None,
        "non_other_rows_delisted_ratio": float(dl_all[~oth_all].mean()),
    }

    # 4. ベースラインの上位3（100株を買えるかを問わない）で、30営業日以内に上場廃止になった選択
    n = cfg["capital"]["n_holdings"]
    res["baseline_top_picks"] = {}
    for name, arr in [("momentum_20d", past_return(m, 20)), ("reversal_5d", -past_return(m, 5))]:
        picks = []
        for p in preds:
            s = np.where(u[p], arr[p], np.nan)
            cand = np.flatnonzero(~np.isnan(s))
            picks += [(p, int(j)) for j in cand[np.argsort(-s[cand], kind="stable")[:n]]]
        picks = [(p, j) for p, j in picks if p + WINDOW <= T - 1]
        d = [(p, j) for p, j in picks if m.last_listed[j] < T - 1 and m.last_listed[j] <= p + WINDOW]
        res["baseline_top_picks"][name] = {
            "picks": len(picks), "picks_flagged_other": int(sum(other[p, j] for p, j in picks)),
            "delisted_within_30": len(d), "delisted_flagged_other_on_pred_date": int(sum(other[p, j] for p, j in d)),
            "examples": [{"pred_date": str(m.dates[p]), "code": str(m.codes[j]), "other_on_pred": bool(other[p, j]),
                          "last_listed": str(m.dates[m.last_listed[j]])} for p, j in d][:20]}

    out = ROOT / "reports" / "phase2_delisting_signal.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
