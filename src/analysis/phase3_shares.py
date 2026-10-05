"""評価役のフェーズ3の指摘・低1（発行済株式数の基準のずれ）と低2（決算発表予定日の「未定」）の前後の比較。

1. 検出（診断用。前後の短信を両方使うので、売買には使わない）：短信ごとの基準値（期末の株数 × 期末日までの累積積）が、
   同じ銘柄の直前と直後の短信の両方から10%超ずれ、そのずれの比率が直前と直後で2%以内でそろっているもの
   （その短信だけが前後と食い違っている）。補正の前と後で数える
2. 補正（src.backtest.market.share_basis）の件数と種類
3. 予測日（364日）のユニバースと、時価総額・E/P・B/P の変化（補正前 = Market v3、補正後 = v4）
4. 「未定」の記録の件数と、cdays_to_next_earnings が変わった銘柄日

出力：reports/phase3_shares.json、reports/phase3_shares_bad_reports.csv（検出した短信の一覧）
実行：python -m src.analysis.phase3_shares
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.backtest.market import Market, load_market, share_basis
from src.backtest.target import make_weeks
from src.backtest.universe import universe_mask
from src.config import ROOT, load_config
from src.data.disclosure import available_index
from src.data.load import read_raw
from src.features.build import load_features

DEV = np.log(1.10)      # 前後の短信からのずれ（10%超）
AGREE = 0.02            # 直前と直後のずれの比率の一致（log で 0.02 以内）
NEAR_BEFORE = 10        # 期末日の前の何営業日以内の分割を「期末日の直前の分割」とみなすか（分類用）


def report_rows(m: Market, cfg: dict) -> pd.DataFrame:
    """Market と同じ手順で、短信ごとの株数・期末日・開示日の位置を作る（src.backtest.market.build_market と同じ）。"""
    bdays = list(m.dates)
    T = len(bdays)
    s = read_raw(cfg, "summary", columns=["Code", "DiscTime", "DiscNo", "DocType", "ShOutFY", "CurPerEn"])
    s["Code"] = s["Code"].astype(str)
    s = s[s["Code"].isin(m.code_index) & (s["ShOutFY"].astype(str).str.strip() != "")].copy()
    s["sh"] = pd.to_numeric(s["ShOutFY"], errors="coerce")
    s = s[s["sh"] > 0]
    s["avail"] = available_index(s["DiscDate"], s["DiscTime"], bdays, cfg)
    s = s[s["avail"] < T]
    ref = np.searchsorted(np.asarray(bdays), s["CurPerEn"].astype(str).to_numpy(), side="right") - 1
    s["ref"] = np.clip(ref, 0, T - 1)
    s["j"] = s["Code"].map(m.code_index)
    s["disc"] = np.searchsorted(np.asarray(bdays), s["DiscDate"].astype(str).to_numpy(), side="right") - 1
    return s.sort_values(["avail", "DiscDate", "DiscTime"])


def detect(s: pd.DataFrame, col: str) -> pd.Series:
    g = s.groupby("j", sort=False)[col]
    lp = np.log(s[col] / g.shift(1))
    ln = np.log(s[col] / g.shift(-1))
    return (lp.abs() > DEV) & ((lp - ln).abs() < AGREE)


def classify(s: pd.DataFrame, m: Market) -> pd.Series:
    """検出した短信の分類：期末日の後〜開示日の分割 / 期末日の直前の分割 / 近くに分割なし。"""
    ev = np.abs(m.adj - 1) > 1e-12
    out = []
    for r in s.itertuples():
        if ev[r.ref + 1: r.disc + 1, r.j].any():
            out.append("期末日の後〜開示日に分割・併合")
        elif ev[max(r.ref - NEAR_BEFORE + 1, 0): r.ref + 1, r.j].any():
            out.append(f"期末日の前{NEAR_BEFORE}営業日以内に分割・併合")
        else:
            out.append("近くに分割・併合なし")
    return pd.Series(out, index=s.index)


def main() -> None:
    cfg = load_config()
    m = load_market(cfg)
    assert m.meta["cache_version"] == "v4", m.meta
    tol = cfg["data"]["shares_split_fix_tol"]
    s = report_rows(m, cfg)
    s = share_basis(s, m.cumF, tol).rename(columns={"base": "base_fixed"})
    s["base_raw"] = s["sh"] * m.cumF[s["ref"].to_numpy(), s["j"].to_numpy()]
    s["code"] = s["Code"]
    bad_before = detect(s, "base_raw")
    bad_after = detect(s, "base_fixed")
    s["class"] = ""
    s.loc[bad_before, "class"] = classify(s[bad_before], m)
    out: dict = {"definition": {"deviation_from_both_neighbors": "> 10%", "neighbor_agreement_log": AGREE,
                                "fix_tol": tol}}
    out["detected_before_fix"] = {"reports": int(bad_before.sum()), "codes": int(s.loc[bad_before, "j"].nunique()),
                                  "by_class": s.loc[bad_before, "class"].value_counts().to_dict()}
    out["detected_after_fix"] = {"reports": int(bad_after.sum()), "codes": int(s.loc[bad_after, "j"].nunique())}
    fixed = s["fix"] != ""
    out["fixed"] = {"reports": int(fixed.sum()), "codes": int(s.loc[fixed, "j"].nunique()),
                    "by_type": s.loc[fixed, "fix"].value_counts().to_dict(),
                    "detected_and_fixed": int((fixed & bad_before).sum()),
                    "fixed_not_detected": int((fixed & ~bad_before).sum()),
                    "detected_not_fixed_by_class": s.loc[bad_before & ~fixed, "class"].value_counts().to_dict()}
    lst = s.loc[bad_before | fixed | bad_after, ["code", "DiscDate", "DiscTime", "DocType", "CurPerEn", "sh",
                                                 "base_raw", "base_fixed", "fix", "class"]].copy()
    lst["detected_before"] = bad_before[lst.index]
    lst["detected_after"] = bad_after[lst.index]
    lst.sort_values(["code", "DiscDate"]).to_csv(ROOT / "reports" / "phase3_shares_bad_reports.csv", index=False,
                                                 encoding="utf-8")

    # 予測日のユニバースと特徴量の変化（補正前 = v3 のキャッシュ、補正後 = v4）
    old_path = ROOT / "data" / "processed" / "market_v3_2016-10-05_2025-09-28.npz"
    z = np.load(old_path, allow_pickle=False)
    assert np.array_equal(z["codes"], m.codes) and np.array_equal(z["dates"], m.dates)
    old = Market(dates=z["dates"], codes=z["codes"], last_listed=z["last_listed"], topix=z["topix"],
                 meta=json.loads(str(z["meta"])), **{a: z[a] for a in
                                                     ["O", "H", "L", "C", "Va", "Vo", "adj", "UL", "LL", "common",
                                                      "shares_base", "margin_other"]})
    for a in ["O", "H", "L", "C", "Va", "Vo", "adj", "common", "margin_other"]:
        assert np.array_equal(getattr(old, a), getattr(m, a), equal_nan=True), a
    weeks = [w for w in make_weeks(m.dates) if w.pred >= 0 and m.dates[w.first] >= cfg["backtest"]["trade_start_date"]]
    preds = np.array([w.pred for w in weeks])
    u_new = universe_mask(m, cfg)[preds]
    u_old = universe_mask(old, cfg)[preds]
    mc_new = (m.shares * m.C)[preds]
    mc_old = (old.shares * old.C)[preds]
    with np.errstate(invalid="ignore", divide="ignore"):
        changed = np.abs(mc_new / mc_old - 1) > 1e-9
    cap = cfg["universe"]["max_market_cap_jpy"]
    out["prediction_dates"] = {
        "n_dates": int(len(preds)), "first": str(m.dates[preds[0]]), "last": str(m.dates[preds[-1]]),
        "universe_rows_before": int(u_old.sum()), "universe_rows_after": int(u_new.sum()),
        "left_universe": int((u_old & ~u_new).sum()),
        "left_universe_mcap_over_cap": int((u_old & ~u_new & (mc_new > cap)).sum()),
        "entered_universe": int((~u_old & u_new).sum()),
        "entered_universe_mcap_was_over_cap": int((~u_old & u_new & (mc_old > cap)).sum()),
        "mcap_changed_in_either_universe": int((changed & (u_old | u_new)).sum()),
        "dates_with_universe_change": int(((u_old != u_new).any(axis=1)).sum()),
    }
    # 特徴量（E/P・B/P・時価総額は株数で、cdays_to_next_earnings は「未定」で変わる）
    names = ["log_mcap", "ep_fcst", "bp", "cdays_to_next_earnings"]
    old_dir = ROOT / "data" / "processed" / "features" / "market_v3_2016-10-05_2025-09-28"
    new_f = load_features(names, cfg, m)
    feat_cmp = {}
    for n in names:
        a = np.load(old_dir / f"{n}.npy")[preds]
        b = new_f[n][preds]
        diff = ~np.isclose(a, b, rtol=1e-12, atol=0, equal_nan=True)
        feat_cmp[n] = {"changed_in_new_universe": int((diff & u_new).sum()),
                       "changed_in_either_universe": int((diff & (u_old | u_new)).sum())}
    out["features_on_prediction_dates"] = feat_cmp
    # 「未定」の記録（全営業日、補正後のユニバース）
    sched = read_raw(cfg, "earnings_date", columns=["Code", "SchDate", "FQName", "FYE"])
    sched["Code"] = sched["Code"].astype(str)
    blank = sched[sched["Code"].isin(m.code_index) & (sched["SchDate"].astype(str).str.len() != 10)]
    avail = np.searchsorted(np.asarray(m.dates), blank["PubDate"].astype(str).to_numpy(), side="right")
    blank = blank[avail < len(m.dates)]
    a = np.load(old_dir / "cdays_to_next_earnings.npy")
    b = new_f["cdays_to_next_earnings"]
    u_all = universe_mask(m, cfg)
    diff = ~np.isclose(a, b, rtol=0, atol=0, equal_nan=True)
    out["undecided_schedule"] = {
        "records": int(len(blank)), "codes": int(blank["Code"].nunique()),
        "records_by_year": blank["PubDate"].str[:4].value_counts().sort_index().to_dict(),
        "stock_days_changed_in_universe": int((diff & u_all).sum()),
        "stock_days_changed_in_universe_to_nan": int((diff & u_all & np.isnan(b)).sum()),
        "prediction_dates_changed_in_universe": int((diff[preds] & u_new).sum()),
    }
    path = ROOT / "reports" / "phase3_shares.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
