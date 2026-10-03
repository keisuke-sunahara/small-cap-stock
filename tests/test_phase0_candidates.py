"""フェーズ0の候補数集計のテスト（テスト用のダミーデータのみ使用）。"""
import pandas as pd

from src.analysis.phase0_candidates import business_days, count_affordable, screen
from src.config import load_config


def test_business_days_excludes_holidays_including_holiday_trading_days():
    # 2025-03-20（春分の日）は HolDiv "3"：東証は休み、大阪取引所のみ祝日取引
    cal = [{"Date": "2025-03-19", "HolDiv": "1"}, {"Date": "2025-03-20", "HolDiv": "3"},
           {"Date": "2025-03-21", "HolDiv": "1"}, {"Date": "2025-03-22", "HolDiv": "0"},
           {"Date": "2025-03-24", "HolDiv": "2"}]
    assert business_days(cal) == ["2025-03-19", "2025-03-21", "2025-03-24"]


def test_count_affordable_lot_and_adv_limits():
    dummy_universe = pd.DataFrame({
        "close": [900.0, 1100.0, 500.0, 300.0],
        "adv20": [1e8, 1e8, 1e8, 5e6],
    })
    # 予算10万円：900円→100株(9万円)可、1100円→不可、500円→200株(10万円)可、
    # 300円→300株(9万円)だが売買代金500万円の1%(5万円)を超えるため不可
    assert count_affordable(dummy_universe, 300000, 3, 100, 0.01) == 2


def test_screen_applies_universe_filters():
    cfg = load_config()
    dates = [f"2025-01-{d:02d}" for d in range(1, 21)]
    dummy_master = pd.DataFrame({"Code": ["A", "B", "C", "D", "E"],
                                 "MktNm": ["グロース", "プライム", "その他", "スタンダード", "スタンダード"],
                                 "S17": ["1", "1", "99", "1", "1"]})
    rows = []
    for d in dates:
        rows += [{"Date": d, "Code": "A", "C": 500.0, "Va": 5e7},
                 {"Date": d, "Code": "B", "C": 500.0, "Va": 5e7},
                 {"Date": d, "Code": "C", "C": 500.0, "Va": 5e7},
                 {"Date": d, "Code": "D", "C": 40.0, "Va": 5e7},
                 {"Date": d, "Code": "E", "C": 500.0, "Va": 1e7}]
    dummy_bars = pd.DataFrame(rows)
    dummy_val = pd.DataFrame({"Code": ["A", "B", "C", "D", "E"],
                              "MktCap": [10000.0, 80000.0, 10000.0, 10000.0, 10000.0]})
    uni = screen(dummy_master, dummy_bars, dummy_val, dates[-1], cfg)
    # B: 時価総額800億円で除外 / C: その他市場で除外 / D: 株価50円未満で除外 / E: 売買代金不足で除外
    assert list(uni.index) == ["A"]
