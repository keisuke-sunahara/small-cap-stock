"""フェーズ1の品質チェックで使う計算のテスト（ダミーデータ）。"""
import numpy as np
import pandas as pd

from src.analysis.phase1_quality import adjusted_returns, is_common


def test_adjusted_returns_remove_split_jump_and_skip_no_trade_days():
    # ダミー：2分割（係数0.5）が売買不成立の日に起きたケースを含む
    bars = pd.DataFrame({
        "Date": ["2024-01-04", "2024-01-05", "2024-01-09", "2024-01-10"],
        "Code": ["11110"] * 4,
        "C": [1000.0, 1100.0, np.nan, 560.0],
        "AdjFactor": [1.0, 1.0, 0.5, 1.0],
    })
    t = adjusted_returns(bars).set_index("Date")
    assert np.isclose(t.loc["2024-01-05", "ret_adj"], 0.1)
    # 1100円 → 分割後560円は、分割前の基準で1120円 → +1.818...%
    assert np.isclose(t.loc["2024-01-10", "ret_adj"], 1120 / 1100 - 1)
    assert "2024-01-09" not in t.index


def test_adjusted_returns_do_not_depend_on_later_factors():
    base = pd.DataFrame({"Date": ["2024-01-04", "2024-01-05", "2024-01-09"], "Code": ["11110"] * 3,
                         "C": [100.0, 110.0, 55.0], "AdjFactor": [1.0, 1.0, 0.5]})
    full = adjusted_returns(base).set_index("Date")["ret_adj"]
    cut = adjusted_returns(base.iloc[:2]).set_index("Date")["ret_adj"]
    assert np.isclose(full["2024-01-05"], cut["2024-01-05"])


def test_is_common():
    m = pd.DataFrame({"Code": ["72030", "25935", "13060", "11110", "99990"],
                      "ProdCat": ["011", "011", "014", "011", "011"],
                      "Mkt": ["0111", "0112", "0111", "0105", "0113"]})
    assert is_common(m).tolist() == [True, False, False, False, True]
