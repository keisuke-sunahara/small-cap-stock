"""特徴量の計算に使うデータ（Market と、開示情報・決算発表予定日）。

開示情報（/fins/summary）
- 各開示を最初に使える営業日の位置 avail は src.data.disclosure.available_index（CLAUDE.md 第5章）で決める
- 同じ日に使える開示が複数あれば、開示日・開示時刻・開示番号の順で後のものを優先する

決算発表予定日（/fins/earnings-date）
- 公表日（PubDate）の記録は、J-Quants では公表日の翌日の朝（10:05 頃）に更新されるため、
  **公表日の次の営業日の引け後の予測から** 使う（公表時刻が無いので、公表日当日の引け後には使わない。DECISIONS.md）

ここでは数値への変換と、使える日の位置の計算だけを行い、値の意味づけは src.features.fundamental で行う。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.market import Market
from src.data.disclosure import available_index
from src.data.load import read_raw

FIN_NUMERIC = ["Sales", "OP", "NP", "Eq", "FSales", "FOP", "FNP", "NxFSales", "NxFOP", "NxFNp"]
FIN_COLUMNS = ["Code", "DiscTime", "DiscNo", "DocType", "CurPerType", "CurPerEn", "CurFYSt", "CurFYEn",
               "NxtFYEn", *FIN_NUMERIC]
SCHED_COLUMNS = ["Code", "SchDate", "FQName", "FYE"]


@dataclass
class FeatureData:
    m: Market
    cfg: dict
    fins: pd.DataFrame     # 開示情報（avail < T のもの）。列 j・avail を持つ
    sched: pd.DataFrame    # 決算発表予定日（avail < T のもの）。列 j・avail を持つ


def prepare_fins(raw: pd.DataFrame, m: Market, cfg: dict) -> pd.DataFrame:
    df = raw[raw["Code"].astype(str).isin(m.code_index)].copy()
    df["Code"] = df["Code"].astype(str)
    for c in FIN_NUMERIC:
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    df["avail"] = available_index(df["DiscDate"], df["DiscTime"], list(m.dates), cfg)
    df = df[df["avail"] < len(m.dates)].copy()
    df["j"] = df["Code"].map(m.code_index).astype(int)
    df = df.sort_values(["avail", "DiscDate", "DiscTime", "DiscNo"], kind="stable").reset_index(drop=True)
    return df


def prepare_sched(raw: pd.DataFrame, m: Market) -> pd.DataFrame:
    df = raw[raw["Code"].astype(str).isin(m.code_index)].copy()
    df["Code"] = df["Code"].astype(str)
    # 公表日の次の営業日から使う（公表日が休日でも、その後の最初の営業日）
    df["avail"] = np.searchsorted(np.asarray(m.dates), df["PubDate"].astype(str).to_numpy(), side="right")
    df = df[(df["avail"] < len(m.dates)) & (df["SchDate"].astype(str).str.len() == 10)].copy()
    df["j"] = df["Code"].map(m.code_index).astype(int)
    df = df.sort_values(["avail", "PubDate"], kind="stable").reset_index(drop=True)
    return df


def load_feature_data(m: Market, cfg: dict, end_date: str | None = None) -> FeatureData:
    """end_date を指定すると、その日までに開示・公表された記録だけを使う（未来情報の混入テスト用）。"""
    fins = read_raw(cfg, "summary", end=end_date, columns=FIN_COLUMNS)
    sched = read_raw(cfg, "earnings_date", end=end_date, columns=SCHED_COLUMNS)
    return FeatureData(m=m, cfg=cfg, fins=prepare_fins(fins, m, cfg), sched=prepare_sched(sched, m))


def asof(fd: FeatureData, events: pd.DataFrame, value: str) -> np.ndarray:
    """events（avail・j・value の列。avail の順に並んでいること）の値を、使える日から次の記録まで引き継いだ [T, N]。

    同じ日・同じ銘柄に複数あれば、後の行を使う。値が NaN の記録も「最新の記録」として NaN を引き継ぐ。
    """
    T, N = len(fd.m.dates), len(fd.m.codes)
    out = np.full((T, N), np.nan)
    has = np.zeros((T, N), dtype=bool)
    ev = events.drop_duplicates(["avail", "j"], keep="last")
    out[ev["avail"].to_numpy(), ev["j"].to_numpy()] = ev[value].to_numpy(dtype=float)
    has[ev["avail"].to_numpy(), ev["j"].to_numpy()] = True
    # NaN の記録も引き継ぐため、記録の番号を前に埋めてから値を引く
    idx = np.where(has, np.arange(T)[:, None], -1)
    idx = np.maximum.accumulate(idx, axis=0)
    res = np.take_along_axis(out, np.maximum(idx, 0), axis=0)
    return np.where(idx >= 0, res, np.nan)
