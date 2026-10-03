"""data/raw の読み込み（src/data/load.py）のテスト。ホールドアウト期間を読まないことを確認する。"""
import pandas as pd
import pytest

from src.data.fetch import HoldoutError, write_new_parquet
from src.data.load import read_raw

CFG = {"data": {"holdout_start": "2025-09-29", "raw_dir": "unused"}}


@pytest.fixture
def raw(tmp_path):
    # テスト用のダミーデータ（本物の株価ではない）
    for d in ["2025-09-25", "2025-09-26", "2025-09-29"]:
        write_new_parquet(pd.DataFrame({"Date": [d], "Code": ["11110"], "C": [100.0]}),
                          tmp_path / "bars" / f"{d}.parquet")
    write_new_parquet(pd.DataFrame(), tmp_path / "bars" / "2025-09-24.parquet")  # 0件の日
    write_new_parquet(pd.DataFrame({"Date": ["2025-09-26", "2025-09-29"], "C": [1.0, 2.0]}),
                      tmp_path / "topix" / "2025-09-01_2025-09-30.parquet")
    return tmp_path


def test_default_reads_only_before_holdout(raw):
    df = read_raw(CFG, "bars", base_dir=raw)
    assert df["Date"].tolist() == ["2025-09-25", "2025-09-26"]
    topix = read_raw(CFG, "topix", base_dir=raw)
    assert topix["Date"].tolist() == ["2025-09-26"]


def test_end_inside_holdout_is_refused(raw):
    with pytest.raises(HoldoutError):
        read_raw(CFG, "bars", end="2025-09-29", base_dir=raw)
    df = read_raw(CFG, "bars", end="2025-09-29", allow_holdout=True, base_dir=raw)
    assert df["Date"].max() == "2025-09-29"


def test_start_and_columns(raw):
    df = read_raw(CFG, "bars", start="2025-09-26", columns=["Code", "Missing"], base_dir=raw)
    assert list(df.columns) == ["Date", "Code", "Missing"]
    assert df["Date"].tolist() == ["2025-09-26"]
    assert df["Missing"].isna().all()
