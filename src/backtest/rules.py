"""比較する売買ルールの候補（CLAUDE.md 第5章）で使うデータの判定。売買そのものは src/backtest/engine.py。

- market_ok：候補1 地合いフィルター（EXP-006）。TOPIX の終値 ≥ 過去 ma_days 営業日（その日を含む）の終値の平均 なら True
- EarningsCalendar：候補3 決算の回避（EXP-008）。各日の時点で公表済みの決算発表予定日と「未定」の状態
- rank_return_trend：候補6・7 の事前の確認（EXP-012・013 の plan.md）。予測上位10銘柄の順位と翌週のリターンの順位相関
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest.engine import Backtester, ranking
from src.backtest.market import Market
from src.backtest.metrics import t_stat
from src.backtest.target import Week, weekly_realized


def market_ok(topix: np.ndarray, ma_days: int) -> np.ndarray:
    """[T] の bool。TOPIX の終値が過去 ma_days 営業日の平均を下回る日だけ False。

    TOPIX が無い日（2020-10-01 の終日の売買停止など）は直前の値でつなぐ。平均を計算できない最初の日は True（判定しない）。
    """
    s = pd.Series(topix, dtype=float).ffill()
    ma = s.rolling(ma_days, min_periods=ma_days).mean()
    return ~(s < ma).to_numpy()


class EarningsCalendar:
    """決算発表予定日（src.features.data.prepare_sched の結果）と、決算短信を使える日から、各日の状態を返す。

    - 予定日：日 d の時点で使える記録（avail ≤ d。公表日の次の営業日から）のうち、同じ四半期（決算期末と四半期の名前）の最新のもの。
      最新が「未定」（予定日が空欄）なら、その四半期の予定日は不明（src.features.fundamental.cdays_to_next_earnings と同じ）
    - 「未定」の状態：その銘柄の最新の記録が「未定」で、その記録を使えるようになってから（その日を含む）d までに、
      使える決算短信が無い（EXP-008 の plan.md）
    """

    def __init__(self, sched: pd.DataFrame, statement_avail: pd.DataFrame, n_codes: int, dates: np.ndarray):
        s = sched.sort_values(["avail", "PubDate"], kind="stable").reset_index(drop=True)
        self.avail = s["avail"].to_numpy()
        self.j = s["j"].to_numpy()
        self.key = (s["FYE"].astype(str) + "_" + s["FQName"].astype(str)).to_numpy()
        self.sch = s["SchDate"].to_numpy(dtype="datetime64[D]")
        self.n = n_codes
        self.dates = dates
        st = statement_avail.sort_values("avail")
        self.st = {j: g["avail"].to_numpy() for j, g in st.groupby("j")}

    @classmethod
    def from_feature_data(cls, fd) -> "EarningsCalendar":
        f = fd.fins
        st = f[f["DocType"].astype(str).str.contains("FinancialStatements")][["j", "avail"]]
        return cls(fd.sched, st, len(fd.m.codes), fd.m.dates)

    def state(self, d: int) -> tuple[pd.DataFrame, np.ndarray]:
        """日 d の時点の (四半期ごとの最新の予定日 [列 j・sch]、「未定」の状態の銘柄 bool[N])。"""
        k = int(np.searchsorted(self.avail, d, side="right"))
        df = pd.DataFrame({"j": self.j[:k], "key": self.key[:k], "sch": self.sch[:k], "avail": self.avail[:k]})
        latest_q = df.drop_duplicates(["j", "key"], keep="last")
        known = latest_q[latest_q["sch"].notna()][["j", "sch"]]
        latest = df.drop_duplicates("j", keep="last")
        undecided = np.zeros(self.n, dtype=bool)
        for j, a in latest.loc[latest["sch"].isna(), ["j", "avail"]].itertuples(index=False):
            sa = self.st.get(j)
            if sa is None or not ((sa >= a) & (sa <= d)).any():
                undecided[j] = True
        return known, undecided

    def mask(self, d: int, lo: str, hi: str, include_undecided: bool = True) -> np.ndarray:
        """日 d の時点で、予定日が暦日 [lo, hi] に入る銘柄（include_undecided なら「未定」の状態の銘柄も）を True。"""
        known, undecided = self.state(d)
        lo_d, hi_d = np.datetime64(lo, "D"), np.datetime64(hi, "D")
        hit = known[(known["sch"] >= lo_d) & (known["sch"] <= hi_d)]["j"].to_numpy()
        out = np.zeros(self.n, dtype=bool)
        out[hit] = True
        if include_undecided:
            out |= undecided
        return out


def rank_return_trend(bt: Backtester, m: Market, weeks: list[Week], score_fn, budget: float, top: int = 10) -> dict:
    """候補6・7 の事前の確認：各予測日に、選定の条件（ユニバース、100株を買える、売買代金の上限）を満たす銘柄を点数の高い順に
    top 銘柄並べ、順位（1〜top）と翌週の実現リターン（最初の営業日の始値→最終営業日の終値、調整後。コストなし）の順位相関を計算する。

    営業日が1日の週（買わない週）は除く。364週の平均がマイナスで t値 ≤ −2 なら「傾向がある」（EXP-012 の plan.md）。
    """
    realized = weekly_realized(m, weeks)
    rows = []
    for k, w in enumerate(weeks):
        if len(w.days) < 2:
            continue
        p = w.pred
        picks = []
        for j in ranking(score_fn(p), bt.universe[p]):
            if bt._affordable(p, int(j), budget)[0] > 0:
                picks.append(int(j))
            if len(picks) >= top:
                break
        r = realized[k, picks]
        ok = ~np.isnan(r)
        corr = stats.spearmanr(np.arange(1, len(picks) + 1)[ok], r[ok]).statistic if ok.sum() >= 3 else np.nan
        rows.append({"pred": str(m.dates[p]), "n": int(ok.sum()), "corr": corr})
    df = pd.DataFrame(rows)
    c = df["corr"].dropna()
    t = t_stat(c)
    return {"weeks": int(len(c)), "mean": float(c.mean()), "std": float(c.std()), "t": t,
            "trend": bool(c.mean() < 0 and t <= -2),
            "rule": "順位（1が最上位）とリターンの順位相関の平均がマイナスで t値 ≤ −2 なら「傾向がある」"}
