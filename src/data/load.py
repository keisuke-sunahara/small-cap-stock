"""data/raw/jquants/ の読み込み。ホールドアウト期間のデータは読み込まない。

- 日付で取ったデータ（1日1ファイル）は、ファイル名の日付で絞ってから読む
- 期間で取ったデータ（calendar・topix）は、読んだ後に Date で絞る
- ホールドアウト開始日以降を含む指定は、allow_holdout=True（フェーズ5でユーザーの承認後のみ）でなければ拒否する
- 株価（bars）の Adj* 列（AdjO・AdjH・AdjL・AdjC・AdjVo）と MktCap 列は読み込まない（評価役の指摘、DECISIONS.md）。
  Adj* は取得日までの分割を反映して後から計算し直された値で、AdjC ÷ C からホールドアウト期間の分割が分かってしまう。
  MktCap は計算に使った株式数の時点が未確認のため使わない（時価総額は開示済みの株式数から自分で計算する）。
  分割の調整は、その日の調整係数 AdjFactor の累積積で行う（src/backtest/market.py）
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.config import ROOT
from src.data.fetch import DATASETS, HoldoutError

DATE_COLUMN = {"calendar": "Date", "topix": "Date", "master": "Date", "bars": "Date",
               "summary": "DiscDate", "earnings_date": "PubDate"}
# 読み込まない列（指定されたらエラー。columns=None でも読まない）
EXCLUDED_COLUMNS = {"bars": ("AdjO", "AdjH", "AdjL", "AdjC", "AdjVo", "MktCap")}


def _holdout_start(cfg: dict) -> str:
    return cfg["data"]["holdout_start"]


def read_raw(cfg: dict, dataset: str, start: str | None = None, end: str | None = None,
             columns: list[str] | None = None, allow_holdout: bool = False,
             base_dir: Path | None = None) -> pd.DataFrame:
    """dataset の [start, end] の行を返す。end の省略時はホールドアウト開始の前日まで。"""
    ds = DATASETS[dataset]
    holdout = _holdout_start(cfg)
    if end is None:
        end = (date.fromisoformat(holdout) - pd.Timedelta(days=1)).isoformat()
    if end >= holdout and not allow_holdout:
        raise HoldoutError(f"{end} はホールドアウト（{holdout} 以降）に入るため読み込めません")
    ds_dir = (base_dir or ROOT / cfg["data"]["raw_dir"]) / dataset
    date_col = DATE_COLUMN[dataset]
    if ds.kind == "range":
        files = sorted(ds_dir.glob("*_*.parquet"))
    else:
        files = sorted(p for p in ds_dir.glob("????-??-??.parquet")
                       if (start is None or p.stem >= start) and p.stem <= end)
    if not files:
        return pd.DataFrame(columns=columns or [date_col])
    excluded = set(EXCLUDED_COLUMNS.get(dataset, ()))
    if columns is not None and excluded & set(columns):
        raise ValueError(f"{dataset} の列 {sorted(excluded & set(columns))} は読み込めません（未来の分割の情報を含むため）")
    if columns is not None and date_col not in columns:
        columns = [date_col, *columns]
    frames = []
    for f in files:
        if columns is None:
            names = pq.read_schema(f).names
            df = pd.read_parquet(f, columns=[c for c in names if c not in excluded])
        else:
            # 日によっては列が無いことがある（0件の日など）。ある列だけ読み、無い列は欠損にする
            present = set(pq.read_schema(f).names)
            df = pd.read_parquet(f, columns=[c for c in columns if c in present]).reindex(columns=columns)
        if len(df) == 0:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=columns or [date_col])
    out = pd.concat(frames, ignore_index=True)
    mask = out[date_col] <= end
    if start is not None:
        mask &= out[date_col] >= start
    return out[mask].reset_index(drop=True)
