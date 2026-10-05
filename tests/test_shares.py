"""発行済株式数の基準のずれの補正（src.backtest.market.share_basis）のテスト（ダミーデータ）。

評価役のフェーズ3の指摘・低1（2026-10-05 ユーザーの承認）。短信の株数 × 期末日までの累積積（基準値）が、
直前の短信と分割の比率だけずれ、その間に分割がある場合は、直前に最も近くなる時点の累積積を使う。
"""
import numpy as np
import pandas as pd

from src.backtest.market import share_basis

TOL = 0.05


def cum(adj_by_day: dict[int, float], T: int = 200) -> np.ndarray:
    a = np.ones((T, 1))
    for t, v in adj_by_day.items():
        a[t, 0] = v
    return np.cumprod(a, axis=0)


def rows(*reports) -> pd.DataFrame:
    """reports：(株数, 期末日の位置, 開示日の位置)。使える順に並べる。"""
    return pd.DataFrame([{"j": 0, "sh": sh, "ref": r, "disc": d} for sh, r, d in reports])


def test_split_after_period_end_reflected_in_report_uses_disclosure_time():
    # 期末日 100 の後、105 の朝に 1→2 分割。110 に開示した短信の株数は分割後（200万株）
    cumF = cum({105: 0.5})
    s = share_basis(rows((1e6, 40, 80), (2e6, 100, 110)), cumF, TOL)
    assert s["base"].tolist() == [1e6, 2e6 * 0.5]
    assert s["fix"].tolist() == ["", "after"]
    # 補正しない場合は2倍に過大
    raw = share_basis(rows((1e6, 40, 80), (2e6, 100, 110)), cumF, None)
    assert raw["base"].tolist() == [1e6, 2e6] and raw["fix"].tolist() == ["", ""]


def test_ex_date_before_period_end_with_pre_split_count():
    # 98 の朝に 1→4 分割の権利落ち（効力発生は期末日 100 の後）。短信の株数は分割前（100万株）のまま
    cumF = cum({98: 0.25})
    s = share_basis(rows((1e6, 40, 80), (1e6, 100, 130)), cumF, TOL)
    assert s["base"].tolist() == [1e6, 1e6]
    assert s["fix"].tolist() == ["", "before"]


def test_split_reflected_correctly_is_not_changed():
    # 期中（60 の朝）に 1→2 分割し、次の短信（期末日 100）の株数は分割後。基準値は変わらないので補正しない
    cumF = cum({60: 0.5})
    s = share_basis(rows((1e6, 40, 50), (2e6, 100, 110)), cumF, TOL)
    assert s["base"].tolist() == [1e6, 1e6] and s["fix"].tolist() == ["", ""]


def test_real_change_without_split_is_not_changed():
    # 増資で株数が30%増えた（分割なし）。補正しない
    cumF = cum({})
    s = share_basis(rows((1e6, 40, 50), (1.3e6, 100, 110)), cumF, TOL)
    assert s["base"].tolist() == [1e6, 1.3e6] and s["fix"].tolist() == ["", ""]
    # 分割はあるが、比率が合わない（1→2 分割と同時に大きな増資）ときも補正しない
    cumF = cum({105: 0.5})
    s = share_basis(rows((1e6, 40, 50), (3e6, 100, 110)), cumF, TOL)
    assert s["fix"].tolist() == ["", ""]


def test_report_ahead_of_split_does_not_drag_the_next_report():
    """分割の効力発生（係数の日 120）より前に、分割後の株数（300万株）を載せた短信（開示 110）は、開示の時点では直せない。
    その次の正しい短信（期末日 160）を、誤った直前の値に合わせて「補正」しない（期末日の直前の範囲だけを見るため）。"""
    cumF = cum({120: 1 / 3})
    s = share_basis(rows((1e6, 40, 80), (3e6, 100, 110), (3e6, 160, 170)), cumF, TOL)
    assert s["fix"].tolist() == ["", "", ""]
    assert np.allclose(s["base"], [1e6, 3e6, 1e6])


def test_previous_report_ahead_of_the_same_split():
    """直前の短信（期末日 100）が、効力発生（係数の日 157）の前に分割後の株数を載せていた。次の短信（期末日 160）は
    権利落ちが期末日の直前でも、正しい（分割後の）株数なので補正しない（直前の値を分割の比率で戻すと、その前と合うため）。"""
    cumF = cum({157: 0.5})
    s = share_basis(rows((1e6, 40, 50), (2e6, 100, 110), (2e6, 160, 170)), cumF, TOL)
    assert s["fix"].tolist() == ["", "", ""]
    assert np.allclose(s["base"], [1e6, 2e6, 1e6])
    # 直前の跳びが別の理由（1.5倍の増資）なら、期末日の後の分割の補正は行う
    cumF = cum({165: 0.5})
    s = share_basis(rows((1e6, 40, 50), (1.5e6, 100, 110), (3e6, 160, 170)), cumF, TOL)
    assert s["fix"].tolist() == ["", "", "after"]


def test_before_type_only_near_period_end():
    # 期末日 100 の3営業日前の権利落ちは「分割前の株数」とみなせるが、10営業日前ならみなさない
    near = share_basis(rows((1e6, 40, 80), (1e6, 100, 130)), cum({97: 0.5}), TOL)
    far = share_basis(rows((1e6, 40, 80), (1e6, 100, 130)), cum({90: 0.5}), TOL)
    assert near["fix"].tolist() == ["", "before"] and far["fix"].tolist() == ["", ""]


def test_first_report_and_later_reports_chain_from_corrected_value():
    # 最初の短信は比べる相手が無いのでそのまま。補正した短信の次の短信は、補正後の値と比べる
    cumF = cum({105: 0.5})
    s = share_basis(rows((1e6, 40, 80), (2e6, 100, 110), (2e6, 160, 170)), cumF, TOL)
    assert s["base"].tolist() == [1e6, 1e6, 1e6]
    assert s["fix"].tolist() == ["", "after", ""]


def test_uses_only_information_up_to_disclosure():
    """開示日より後の係数や短信を消しても、それまでの短信の基準値は変わらない。"""
    rng = np.random.default_rng(0)
    T = 400
    adj = np.ones((T, 3))
    for _ in range(12):
        adj[rng.integers(1, T), rng.integers(0, 3)] = rng.choice([0.5, 0.25, 2.0, 0.2])
    cumF = np.cumprod(adj, axis=0)
    reps = []
    for j in range(3):
        for k in range(6):
            ref = 30 + 60 * k
            disc = ref + int(rng.integers(5, 40))
            # 期末の株数（たまに、開示日の時点の株数で書かれた短信を混ぜる）
            x = disc if rng.random() < 0.3 else ref
            reps.append({"j": j, "sh": 1e6 / cumF[x, j] * (1 + 0.01 * k), "ref": ref, "disc": disc})
    s = pd.DataFrame(reps).sort_values(["disc", "j"], kind="stable").reset_index(drop=True)
    full = share_basis(s, cumF, TOL)
    for p in (120, 250, 330):
        cut = share_basis(s[s["disc"] <= p], cumF[: p + 1], TOL)
        assert np.allclose(cut["base"], full.loc[full["disc"] <= p, "base"])
        assert cut["fix"].tolist() == full.loc[full["disc"] <= p, "fix"].tolist()
