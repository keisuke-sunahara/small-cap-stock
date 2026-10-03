"""J-Quants のデータを data/raw/jquants/ に保存する。再実行すると未取得の分だけを取得する。

保存形式（API の応答 data 配列をそのまま表にした Parquet。加工しない）
- 期間で取るデータ（calendar・topix）: <dataset>/<from>_<to>.parquet
- 日付で取るデータ（master・bars・summary・earnings_date）: <dataset>/<YYYY-MM-DD>.parquet（1日1ファイル。0件の日も空のファイルを残す）

守ること
- 既にあるファイルは上書きしない（data/raw は変更禁止）
- 書き込みは一時ファイル → 名前の変更で行い、途中で止まっても壊れたファイルを残さない
- ホールドアウト開始日以降は取得しない。--allow-holdout はフェーズ5でユーザーの承認後にだけ使う
- 銘柄コード指定の取得は期間を区切れずホールドアウトの記録まで返るため、使わない
- 取得の記録（日付、件数、取得日時）を <dataset>/_fetch_log.csv に追記する

実行例: python -m src.data.fetch --datasets calendar topix master bars
        python -m src.data.fetch --datasets summary earnings_date
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import ROOT, load_config
from src.data.jquants import JQuantsClient, client_from_config

JST = ZoneInfo("Asia/Tokyo")
TSE_BUSINESS_HOLDIV = {"1", "2"}  # 1: 営業日、2: 東証半日立会日（"3" は東証休み・大阪のみ祝日取引）


@dataclass(frozen=True)
class Dataset:
    name: str
    path: str
    kind: str       # "range"（from/to で取る） / "bday"（営業日ごとに date で取る） / "cday"（暦日ごとに date で取る）
    fins: bool      # /fins/ 以下（個別のレート制限）


DATASETS = {
    "calendar": Dataset("calendar", "/markets/calendar", "range", False),
    "topix": Dataset("topix", "/indices/bars/daily/topix", "range", False),
    "master": Dataset("master", "/equities/master", "bday", False),
    "bars": Dataset("bars", "/equities/bars/daily", "bday", False),
    # 開示・公表は休日にもあり得るため、暦日すべてを取る（0件の日は空のファイル）
    "summary": Dataset("summary", "/fins/summary", "cday", True),
    "earnings_date": Dataset("earnings_date", "/fins/earnings-date", "cday", True),
}


class HoldoutError(RuntimeError):
    pass


def resolve_end(cfg: dict, end: date | None, allow_holdout: bool) -> date:
    """取得の最終日。指定がなければホールドアウト開始の前日。ホールドアウトに入る指定は拒否する。"""
    holdout_start = date.fromisoformat(cfg["data"]["holdout_start"])
    last_allowed = holdout_start - timedelta(days=1)
    if end is None:
        return last_allowed
    if end > last_allowed and not allow_holdout:
        raise HoldoutError(f"{end} はホールドアウト（{holdout_start} 以降）に入るため取得できません")
    return end


def raw_dir(cfg: dict) -> Path:
    return ROOT / cfg["data"]["raw_dir"]


def write_new_parquet(df: pd.DataFrame, path: Path) -> None:
    """新しいファイルとして保存する。既にあれば例外（上書きしない）。"""
    if path.exists():
        raise FileExistsError(f"上書きしません: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def append_log(ds_dir: Path, key: str, n_rows: int) -> None:
    log = ds_dir / "_fetch_log.csv"
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["key", "rows", "fetched_at_jst"])
        w.writerow([key, n_rows, datetime.now(JST).isoformat(timespec="seconds")])


def covered_until(ds_dir: Path) -> date | None:
    """期間で取るデータの、取得済みの最終日（ファイル名 <from>_<to>.parquet から）。"""
    ends = [date.fromisoformat(p.stem.split("_")[1]) for p in ds_dir.glob("*_*.parquet")]
    return max(ends) if ends else None


def fetch_range(client: JQuantsClient, ds: Dataset, ds_dir: Path, start: date, end: date) -> int:
    last = covered_until(ds_dir)
    frm = start if last is None else last + timedelta(days=1)
    if frm > end:
        return 0
    rows = client.get_all(ds.path, **{"from": frm.isoformat(), "to": end.isoformat()})
    df = pd.DataFrame(rows)
    if len(df) and (df["Date"].min() < frm.isoformat() or df["Date"].max() > end.isoformat()):
        raise RuntimeError(f"{ds.name}: 指定した期間外の行が返りました")
    key = f"{frm.isoformat()}_{end.isoformat()}"
    write_new_parquet(df, ds_dir / f"{key}.parquet")
    append_log(ds_dir, key, len(df))
    return 1


def load_calendar(cfg: dict) -> pd.DataFrame:
    files = sorted((raw_dir(cfg) / "calendar").glob("*_*.parquet"))
    if not files:
        raise FileNotFoundError("取引カレンダーが未取得です（--datasets calendar を先に実行）")
    cal = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates("Date").sort_values("Date")
    return cal


def business_days(cal: pd.DataFrame) -> list[str]:
    return sorted(cal.loc[cal["HolDiv"].astype(str).isin(TSE_BUSINESS_HOLDIV), "Date"])


def target_dates(cfg: dict, ds: Dataset, start: date, end: date) -> list[str]:
    cal = load_calendar(cfg)
    covered = covered_until(raw_dir(cfg) / "calendar")
    if covered is None or covered < end:
        raise RuntimeError(f"取引カレンダーが {end} まで取得されていません（--datasets calendar を先に実行）")
    if ds.kind == "bday":
        days = business_days(cal)
    else:
        days = [d.date().isoformat() for d in pd.date_range(start, end, freq="D")]
    return [d for d in days if start.isoformat() <= d <= end.isoformat()]


def fetch_by_date(client: JQuantsClient, ds: Dataset, ds_dir: Path, dates: list[str]) -> int:
    todo = [d for d in dates if not (ds_dir / f"{d}.parquet").exists()]
    print(f"[{ds.name}] 対象 {len(dates)} 日、未取得 {len(todo)} 日", flush=True)
    for i, d in enumerate(todo, 1):
        rows = client.get_all(ds.path, date=d)
        df = pd.DataFrame(rows)
        date_col = {"master": "Date", "bars": "Date", "summary": "DiscDate", "earnings_date": "PubDate"}[ds.name]
        if len(df) and set(df[date_col].unique()) != {d}:
            raise RuntimeError(f"{ds.name} {d}: 指定日と異なる日付の行が返りました")
        write_new_parquet(df, ds_dir / f"{d}.parquet")
        append_log(ds_dir, d, len(df))
        if i % 50 == 0 or i == len(todo):
            print(f"[{ds.name}] {i}/{len(todo)} 日（{d}、{len(df)}件）", flush=True)
    return len(todo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--end", type=date.fromisoformat, default=None,
                        help="取得の最終日（省略時はホールドアウト開始の前日）")
    parser.add_argument("--allow-holdout", action="store_true",
                        help="ホールドアウト期間の取得を許可する（フェーズ5でユーザーの承認後のみ）")
    parser.add_argument("--rate", type=float, default=None, help="1分あたりのリクエスト数（省略時は設定値×安全率）")
    args = parser.parse_args()

    cfg = load_config()
    start = date.fromisoformat(cfg["data"]["start_date"])
    end = resolve_end(cfg, args.end, args.allow_holdout)
    safety = cfg["data"]["rate_limit_safety"]
    print(f"取得期間: {start} 〜 {end}", flush=True)

    for name in args.datasets:
        ds = DATASETS[name]
        limit = args.rate or (cfg["data"]["fins_rate_limit_per_min"] if ds.fins
                              else cfg["data"]["rate_limit_per_min"]) * safety
        client = client_from_config(cfg, rate_limit_per_min=limit)
        ds_dir = raw_dir(cfg) / ds.name
        ds_dir.mkdir(parents=True, exist_ok=True)
        if ds.kind == "range":
            n = fetch_range(client, ds, ds_dir, start, end)
            print(f"[{ds.name}] 新しく取得したファイル {n}", flush=True)
        else:
            fetch_by_date(client, ds, ds_dir, target_dates(cfg, ds, start, end))
    print("完了", flush=True)


if __name__ == "__main__":
    main()
