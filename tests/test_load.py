"""data/raw の読み込み（src/data/load.py）のテスト。ホールドアウト期間を読まないことを確認する。"""
from pathlib import Path

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


ADJ_COLUMNS = ["AdjO", "AdjH", "AdjL", "AdjC", "AdjVo", "MktCap"]


@pytest.fixture
def raw_with_adj(tmp_path):
    # テスト用のダミーデータ（本物の株価ではない）。J-Quants の株価と同じく Adj* と MktCap の列を持つ
    for d in ["2025-09-25", "2025-09-26"]:
        write_new_parquet(pd.DataFrame({"Date": [d], "Code": ["11110"], "C": [100.0], "Vo": [1000.0],
                                        "AdjFactor": [1.0], **{c: [50.0] for c in ADJ_COLUMNS}}),
                          tmp_path / "bars" / f"{d}.parquet")
    write_new_parquet(pd.DataFrame(), tmp_path / "bars" / "2025-09-24.parquet")  # 0件の日
    return tmp_path


def test_bars_never_include_adj_or_mktcap_columns(raw_with_adj):
    df = read_raw(CFG, "bars", base_dir=raw_with_adj)
    assert len(df) == 2
    assert not [c for c in df.columns if c in ADJ_COLUMNS]
    # 当日の調整係数（AdjFactor）と調整前の出来高（Vo）は読む
    assert {"C", "Vo", "AdjFactor"} <= set(df.columns)
    for c in ADJ_COLUMNS:
        with pytest.raises(ValueError):
            read_raw(CFG, "bars", columns=["Code", c], base_dir=raw_with_adj)


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "data/raw/jquants/bars").exists(),
                    reason="実データがありません")
def test_real_bars_have_no_adj_columns():
    from src.config import ROOT, load_config
    cfg = load_config()
    df = read_raw(cfg, "bars", start="2019-03-28", end="2019-03-29")
    assert len(df) > 0
    assert not [c for c in df.columns if c.startswith("Adj") and c != "AdjFactor"]
    assert "MktCap" not in df.columns
    raw_cols = pd.read_parquet(ROOT / cfg["data"]["raw_dir"] / "bars" / "2019-03-29.parquet").columns
    assert "AdjC" in raw_cols   # 元のファイルにはある列を、読み込みで落としていること
