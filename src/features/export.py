"""評価する予測日のユニバースの特徴量の値（正規化の前）を、特徴量のセットごとに1回だけ保存する（評価役の照合用）。

評価役のフェーズ3の依頼4、2026-10-05 ユーザーの指示：実験ごとではなく特徴量のセットごとに保存し、コミットする
（同じデータが実験の数だけ溜まるのを避けるため）。実験の予測（logs/backtest/<実験ID>/）は、このファイルを参照する。

- 行：評価する予測日（売買の開始の週〜ホールドアウトの前の週の、各週の予測日。364日）× その日のユニバースの銘柄
  （継続の判断の日も加えると約63MBになり、GitHub の1ファイルの警告の目安（50MB）を超えるため、評価役の依頼どおり
  予測日だけにする。約33MB）
- 列：date・code・特徴量（float64、正規化の前）
- 保存先：logs/backtest/features/<セット名>.parquet と、作成の情報 <セット名>.json（Market の作成日時、データの取得日、
  gitのコミットID、特徴量の関数のハッシュ）
- 既にあるファイルは上書きしない（記録の書き換えの禁止。作り直すときは別のセット名か、ユーザーの承認を得て削除する）

実行：python -m src.features.export --set all_v1
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd

from src.backtest.market import JST, Market, load_market
from src.backtest.target import make_weeks
from src.backtest.universe import universe_mask
from src.config import ROOT, load_config
from src.features.build import load_features
from src.features.registry import get, resolve


def evaluation_days(m: Market, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """評価する予測日と、継続の判断の日（prev_day：各週の最終営業日の前の営業日）の位置。"""
    end = (pd.Timestamp(cfg["data"]["holdout_start"]) - pd.Timedelta(days=1)).date().isoformat()
    weeks = [w for w in make_weeks(m.dates)
             if w.pred >= 0 and m.dates[w.first] >= cfg["backtest"]["trade_start_date"] and m.dates[w.last] <= end]
    preds = np.array(sorted({w.pred for w in weeks}))
    conts = np.array(sorted({w.last - 1 for w in weeks if w.last - 1 >= 0}))
    return preds, conts


def feature_frame(m: Market, cfg: dict, names: list[str], feats: dict[str, np.ndarray],
                  universe: np.ndarray) -> pd.DataFrame:
    days, _ = evaluation_days(m, cfg)
    ti, jj = np.nonzero(universe[days])
    t = days[ti]
    df = pd.DataFrame({"date": m.dates[t].astype(str), "code": m.codes[jj].astype(str)})
    for n in names:
        df[n] = feats[n][t, jj].astype(np.float64)
    return df


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=None)
    args = ap.parse_args()
    cfg = load_config()
    set_name = args.set or cfg["features"]["set"]
    names = resolve(set_name)
    out_dir = ROOT / "logs" / "backtest" / "features"
    path, meta_path = out_dir / f"{set_name}.parquet", out_dir / f"{set_name}.json"
    if path.exists() or meta_path.exists():
        raise SystemExit(f"既にあります（上書きしません）: {path}")
    m = load_market(cfg)
    feats = load_features(names, cfg, m)
    df = feature_frame(m, cfg, names, feats, universe_mask(m, cfg))
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    meta = {"feature_set": set_name, "features": {n: {"description": get(n).description,
                                                      "source_hash": get(n).source_hash()} for n in names},
            "rows": int(len(df)), "pred_dates": int(df["date"].nunique()),
            "market": m.meta, "git_commit": git_commit(),
            "created_at_jst": datetime.now(JST).isoformat(timespec="seconds"),
            "note": "値は正規化の前（src.features の計算結果そのもの）。ユニバースは src.backtest.universe.universe_mask"}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{path}: {len(df)} 行、{path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
