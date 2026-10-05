"""第2段階の特徴量（財務・イベント）。

- 開示情報は、使える日（avail。CLAUDE.md 第5章）から次の開示まで値を引き継ぐ（src.features.data.asof）
- 決算短信：DocType に "FinancialStatements" を含む行。実績（Sales・OP・NP・Eq）は期首からの累計
- 会社予想（通期）：
  - 本決算の短信（CurPerType = FY）は、翌期の予想（NxF*、対象の期末 = NxtFYEn）
  - 四半期の短信と業績予想の修正（EarnForecastRevision）は、今期の予想（F*、対象の期末 = CurFYEn）
  - 業績予想の修正のうち、通期の予想が空のもの（第2四半期の予想だけの修正）は使わない
  - 対象の期末が、その銘柄でそれまでに出た予想の期末より前のもの（終わった期の予想）は、最新の予想として使わない
- 時価総額は、開示済みの発行済株式数 × その日の終値（売買不成立の日は直前の終値）。ユニバースの判定と同じ株数
- 決算発表予定日は、公表日の次の営業日から使う（src.features.data）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.data import FeatureData, asof
from src.features.registry import register


def _mcap(fd: FeatureData) -> np.ndarray:
    return fd.m.shares * fd.m.C_ff


def statements(fd: FeatureData) -> pd.DataFrame:
    f = fd.fins
    st = f[f["DocType"].astype(str).str.contains("FinancialStatements")].copy()
    st["order"] = st.index
    return st


def forecast_events(fd: FeatureData) -> pd.DataFrame:
    """通期の会社予想の記録（order = fins の行番号の順）。列 fy_end・f_sales・f_op・f_np。"""
    f = fd.fins
    is_st = f["DocType"].astype(str).str.contains("FinancialStatements")
    is_rev = f["DocType"].astype(str).eq("EarnForecastRevision")
    is_fy = is_st & f["CurPerType"].astype(str).eq("FY")
    ev = pd.DataFrame({
        "order": f.index, "avail": f["avail"], "j": f["j"],
        "fy_end": np.where(is_fy, f["NxtFYEn"].astype(str), f["CurFYEn"].astype(str)),
        "f_sales": np.where(is_fy, f["NxFSales"], f["FSales"]),
        "f_op": np.where(is_fy, f["NxFOP"], f["FOP"]),
        "f_np": np.where(is_fy, f["NxFNp"], f["FNP"]),
    })
    empty_rev = is_rev & ev[["f_sales", "f_op", "f_np"]].isna().all(axis=1)
    ev = ev[(is_st | is_rev) & ~empty_rev & (ev["fy_end"].str.len() == 10)].copy()
    # 終わった期の予想（それまでの最大の期末より前）は除く
    fy_num = pd.to_datetime(ev["fy_end"], errors="coerce").astype("int64")
    ev = ev[fy_num >= fy_num.groupby(ev["j"]).cummax()]
    return ev.reset_index(drop=True)


@register("log_mcap", "fundamental", "時価総額の対数：log(開示済みの発行済株式数 × 終値)")
def log_mcap(fd) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(_mcap(fd))


@register("ep_fcst", "fundamental",
          "予想PERの逆数：最新の会社予想の通期純利益 ÷ 時価総額（PER は赤字で符号が反転するため逆数にする）")
def ep_fcst(fd) -> np.ndarray:
    return asof(fd, forecast_events(fd), "f_np") / _mcap(fd)


@register("bp", "fundamental", "PBR の逆数：最新の決算短信の純資産 ÷ 時価総額")
def bp(fd) -> np.ndarray:
    return asof(fd, statements(fd), "Eq") / _mcap(fd)


def _yoy(fd: FeatureData) -> pd.DataFrame:
    """決算短信ごとに、前年の同じ期間（四半期の種類が同じで、期末の月が12か月前）の短信の実績を付ける。

    前年の短信は、その短信より前に開示されたもののうち最新（訂正があれば訂正後）を使う。
    """
    st = statements(fd)
    st = st[st["CurPerEn"].astype(str).str.len() == 10].copy()
    st["ym"] = pd.PeriodIndex(st["CurPerEn"].astype(str), freq="M").astype("int64")
    st["per"] = st["CurPerType"].astype(str)
    prev = st[["order", "j", "per", "ym", "Sales", "OP"]].rename(columns={"Sales": "Sales_prev", "OP": "OP_prev"})
    prev["ym"] = prev["ym"] + 12
    cur = st.sort_values("order")
    prev = prev.sort_values("order")
    out = pd.merge_asof(cur, prev, on="order", by=["j", "per", "ym"], allow_exact_matches=False)
    return out.sort_values("order").reset_index(drop=True)


@register("sales_growth_yoy", "fundamental", "売上高の前年同期比の伸び率（最新の決算短信の期首からの累計）")
def sales_growth_yoy(fd) -> np.ndarray:
    y = _yoy(fd)
    y["v"] = np.where(y["Sales_prev"] > 0, y["Sales"] / y["Sales_prev"] - 1, np.nan)
    return asof(fd, y, "v")


@register("op_chg_to_sales_yoy", "fundamental",
          "営業利益の伸び：(営業利益 − 前年同期の営業利益) ÷ 前年同期の売上高（赤字をまたいでも計算できるよう売上高で割る）")
def op_chg_to_sales_yoy(fd) -> np.ndarray:
    y = _yoy(fd)
    y["v"] = np.where(y["Sales_prev"] > 0, (y["OP"] - y["OP_prev"]) / y["Sales_prev"], np.nan)
    return asof(fd, y, "v")


@register("bdays_since_report", "fundamental", "決算発表からの日数：最新の決算短信を使えるようになってからの営業日数")
def bdays_since_report(fd) -> np.ndarray:
    st = statements(fd).copy()
    st["a"] = st["avail"].astype(float)
    last = asof(fd, st, "a")
    return np.arange(len(fd.m.dates))[:, None] - last


@register("op_surprise_fy", "fundamental",
          "本決算の上振れ・下振れ：(実績の営業利益 − 直前の会社予想の営業利益) ÷ 実績の売上高。次の本決算まで引き継ぐ")
def op_surprise_fy(fd) -> np.ndarray:
    st = statements(fd)
    fy = st[st["CurPerType"].astype(str).eq("FY")][["order", "avail", "j", "CurFYEn", "OP", "Sales"]].copy()
    fy = fy.rename(columns={"CurFYEn": "fy_end"}).sort_values("order")
    ev = forecast_events(fd)[["order", "j", "fy_end", "f_op"]].sort_values("order")
    out = pd.merge_asof(fy, ev, on="order", by=["j", "fy_end"], allow_exact_matches=False)
    out["v"] = np.where(out["Sales"].abs() > 0, (out["OP"] - out["f_op"]) / out["Sales"].abs(), np.nan)
    return asof(fd, out.sort_values("order"), "v")


@register("op_fcst_rev", "fundamental",
          "会社予想の修正：最新の予想の通期営業利益 − 同じ期の1つ前の予想 を、予想の売上高で割ったもの。"
          "その期の最初の予想なら NaN")
def op_fcst_rev(fd) -> np.ndarray:
    ev = forecast_events(fd).sort_values("order").copy()
    prev = ev.groupby(["j", "fy_end"])["f_op"].shift(1)
    ev["v"] = np.where(ev["f_sales"] > 0, (ev["f_op"] - prev) / ev["f_sales"], np.nan)
    return asof(fd, ev, "v")


@register("cdays_to_next_earnings", "fundamental",
          "次の決算発表予定までの暦日数：その日までに公表された予定日（同じ四半期の変更は最新の公表。"
          "最新の公表が「未定」ならその四半期の予定日は不明）のうち、その日より後で最も早いもの。分からなければ NaN")
def cdays_to_next_earnings(fd) -> np.ndarray:
    m = fd.m
    T, N = len(m.dates), len(m.codes)
    day = np.asarray(m.dates, dtype="datetime64[D]").astype(np.int64)
    out = np.full((T, N), np.nan)
    s = fd.sched
    has = s["SchDate"].notna().to_numpy()
    sch = np.where(has, s["SchDate"].to_numpy(dtype="datetime64[D]").astype(np.int64), 0)
    keys = (s["FYE"].astype(str) + "_" + s["FQName"].astype(str)).to_numpy()
    av = s["avail"].to_numpy()
    for j, idx in s.groupby("j").indices.items():
        known: dict[str, int] = {}
        for k, i in enumerate(idx):
            if has[i]:
                known[keys[i]] = sch[i]
            else:
                known.pop(keys[i], None)   # 「未定」：その四半期の予定日は不明
            start = av[i]
            end = av[idx[k + 1]] if k + 1 < len(idx) else T
            if end <= start:
                continue
            v = np.sort(np.fromiter(known.values(), dtype=np.int64))
            d = day[start:end]
            pos = np.searchsorted(v, d, side="right")
            ok = pos < len(v)
            seg = np.full(len(d), np.nan)
            seg[ok] = v[pos[ok]] - d[ok]
            out[start:end, j] = seg
    return out
