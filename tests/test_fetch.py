"""データ取得（src/data/fetch.py）のテスト。ネットワークには接続しない。"""
from datetime import date

import pandas as pd
import pytest

from src.data import fetch
from src.data.fetch import DATASETS, HoldoutError, fetch_by_date, fetch_range, resolve_end, write_new_parquet

CFG = {"data": {"holdout_start": "2025-09-29"}}


class FakeClient:
    """テスト用の偽クライアント。呼ばれた引数を記録し、決められた行を返す。"""

    def __init__(self, rows_by_date=None, range_rows=None):
        self.rows_by_date = rows_by_date or {}
        self.range_rows = range_rows or []
        self.calls = []

    def get_all(self, path, **params):
        self.calls.append(params)
        if "date" in params:
            return self.rows_by_date.get(params["date"], [])
        return [r for r in self.range_rows if params["from"] <= r["Date"] <= params["to"]]


def test_default_end_is_day_before_holdout():
    assert resolve_end(CFG, None, False) == date(2025, 9, 28)


def test_end_inside_holdout_is_refused():
    with pytest.raises(HoldoutError):
        resolve_end(CFG, date(2025, 9, 29), False)
    assert resolve_end(CFG, date(2025, 9, 26), False) == date(2025, 9, 26)
    # フェーズ5で承認後に明示した場合だけ許可
    assert resolve_end(CFG, date(2026, 9, 30), True) == date(2026, 9, 30)


def test_write_new_parquet_never_overwrites(tmp_path):
    p = tmp_path / "a.parquet"
    write_new_parquet(pd.DataFrame({"x": [1]}), p)
    with pytest.raises(FileExistsError):
        write_new_parquet(pd.DataFrame({"x": [2]}), p)
    assert pd.read_parquet(p)["x"].tolist() == [1]
    assert not list(tmp_path.glob("*.tmp"))


def test_fetch_by_date_skips_existing_and_keeps_empty_days(tmp_path):
    ds = DATASETS["bars"]
    client = FakeClient({"2024-01-04": [{"Date": "2024-01-04", "Code": "11110", "C": 100.0}]})
    write_new_parquet(pd.DataFrame({"Date": ["2024-01-05"]}), tmp_path / "2024-01-05.parquet")
    n = fetch_by_date(client, ds, tmp_path, ["2024-01-04", "2024-01-05", "2024-01-09"])
    assert n == 2
    assert [c["date"] for c in client.calls] == ["2024-01-04", "2024-01-09"]
    assert len(pd.read_parquet(tmp_path / "2024-01-09.parquet")) == 0  # 0件の日も記録して再取得しない
    log = pd.read_csv(tmp_path / "_fetch_log.csv")
    assert log["key"].tolist() == ["2024-01-04", "2024-01-09"]
    # 2回目は何も取得しない
    assert fetch_by_date(client, ds, tmp_path, ["2024-01-04", "2024-01-05", "2024-01-09"]) == 0


def test_fetch_by_date_rejects_rows_of_other_dates(tmp_path):
    client = FakeClient({"2024-01-04": [{"Date": "2024-01-05", "Code": "11110"}]})
    with pytest.raises(RuntimeError):
        fetch_by_date(client, DATASETS["bars"], tmp_path, ["2024-01-04"])
    assert not (tmp_path / "2024-01-04.parquet").exists()


def test_fetch_range_continues_after_covered_period(tmp_path):
    rows = [{"Date": d, "C": 1.0} for d in ["2024-01-04", "2024-01-05", "2024-01-09", "2024-01-10"]]
    client = FakeClient(range_rows=rows)
    ds = DATASETS["topix"]
    assert fetch_range(client, ds, tmp_path, date(2024, 1, 1), date(2024, 1, 5)) == 1
    assert fetch_range(client, ds, tmp_path, date(2024, 1, 1), date(2024, 1, 5)) == 0
    assert fetch_range(client, ds, tmp_path, date(2024, 1, 1), date(2024, 1, 10)) == 1
    assert client.calls[-1] == {"from": "2024-01-06", "to": "2024-01-10"}
    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == [
        "2024-01-01_2024-01-05.parquet", "2024-01-06_2024-01-10.parquet"]


def test_business_days_exclude_holiday_trading_days():
    cal = pd.DataFrame({"Date": ["2025-03-19", "2025-03-20", "2025-03-21", "2025-03-22"],
                        "HolDiv": ["1", "3", "1", "0"]})
    assert fetch.business_days(cal) == ["2025-03-19", "2025-03-21"]
