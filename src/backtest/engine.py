"""週次サイクルのバックテスト（CLAUDE.md 第5章の基本の売買ルール）。

1週間の流れ（週の営業日 d1..dk、予測日 p = 前の週の最終営業日）
- p の引け後：順位の上から、保有していない銘柄を空き枠（N − 継続保有の数）だけ選ぶ。
  予算 = min(運用資金, p の引け時点の総資産) ÷ N。指値 = p の終値 × (1 + 上限乖離率) を呼値で切り下げ。
  100株も買えない銘柄、注文金額が平均売買代金の上限を超える銘柄、現金が足りない銘柄は飛ばして次の順位へ
- d1 の寄付：始値 < 指値なら始値で買う（コストを加える）。約定しなければその週は現金のまま（予備の銘柄は買わない）
- 営業日が1日しかない週（k = 1）は買わない。継続保有中の銘柄は、通常どおり最終営業日に継続か売却かを判断する
- dk の大引け：予算（budget_mode）
- "min_equity"：予算 = min(運用資金, 予測日の総資産) ÷ N（承認済みの基本ルール。実際の口座に近い）
- "fixed"：予算 = 運用資金 ÷ N に固定し、毎週の損益 ÷ 運用資金 を週次リターンとする（戦略どうしの比較用。
  負けると予算が縮んで超低位株しか買えなくなる連鎖を除くため。2026-10-04 ユーザーの依頼）

継続保有の判断（continuation）に入らなかった保有銘柄を引成で売る。売れなければ、次に売買が成立した日の始値で売る
- 保有中の分割・併合は株数を直す。上場廃止は最後に売買が成立した日の終値で売ったものとする
- 評価額は毎営業日の終値（売買不成立の日は直前の終値）で計算する

予算（budget_mode）
- "min_equity"：予算 = min(運用資金, 予測日の総資産) ÷ N（承認済みの基本ルール。実際の口座に近い）
- "fixed"：予算 = 運用資金 ÷ N に固定し、毎週の損益 ÷ 運用資金 を週次リターンとする（戦略どうしの比較用。
  負けると予算が縮んで超低位株しか買えなくなる連鎖を除くため。2026-10-04 ユーザーの依頼）

継続保有の判断（continuation）
- "none"：継続保有しない（毎週すべて売る）
- "prev_day"：dk の前の営業日の引け後の順位で、上位N銘柄（保有銘柄は買えるかの判定なしで数える）に入っていれば売らない。
  引成の注文を dk の 11:30〜15:00 に入れる前に判断できる（暫定の初期値。DECISIONS.md）
- "same_day"：dk の引け後の順位で判断する。基本ルールの文面どおりだが、引けで売るかを引け値を見て決めることになり
  実際には発注できない（参考値）
- 判断に使う予算（基本ルール）の総資産は、判断する日 c の終値と c の時点の株数で計算する。dk の朝に分割・併合が
  効いた保有銘柄は、株数を c の時点に戻して評価する（評価役のフェーズ2の指摘3）

コスト（cost_model）
- "flat"：片道 cost を一律に差し引く（基本ルール。手数料0円＋自分の注文による寄付・引けの値動き。DECISIONS.md）
- "tick"：片道 = max(cost, 1呼値 ÷ 約定価格)。安い株ほど1呼値の値動きの割合が大きいことを反映する
  （評価役のフェーズ2の指摘1。フェーズ4の比較では flat と併記する）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.backtest.execution import (buy_filled, close_sell_filled, limit_price, open_sell_filled,
                                    order_shares, tick_size, within_turnover_cap)
from src.backtest.market import Market
from src.backtest.target import Week, make_weeks

ScoreFn = Callable[[int], np.ndarray]   # 日付の位置 → 銘柄ごとの点数（大きいほど上位。NaN は対象外）
CONTINUATION_MODES = ("none", "prev_day", "same_day")
BUDGET_MODES = ("min_equity", "fixed")
COST_MODELS = ("flat", "tick")


@dataclass
class Rules:
    n_holdings: int
    capital: float
    limit_up_pct: float
    lot: int
    max_order_to_adv: float
    cost: float
    continuation: str = "prev_day"
    budget_mode: str = "min_equity"
    cost_model: str = "flat"

    @classmethod
    def from_config(cls, cfg: dict, **override) -> "Rules":
        kw = dict(n_holdings=cfg["capital"]["n_holdings"], capital=cfg["capital"]["initial_capital_jpy"],
                  limit_up_pct=cfg["order"]["limit_up_pct"], lot=cfg["order"]["lot_size"],
                  max_order_to_adv=cfg["order"]["max_order_to_adv"], cost=cfg["cost"]["one_way"],
                  continuation=cfg["backtest"].get("continuation", "prev_day"),
                  budget_mode=cfg["backtest"].get("budget_mode", "min_equity"),
                  cost_model=cfg["cost"].get("model", "flat"))
        kw.update(override)
        if kw["continuation"] not in CONTINUATION_MODES:
            raise ValueError(f"continuation は {CONTINUATION_MODES} のどれか: {kw['continuation']}")
        if kw["budget_mode"] not in BUDGET_MODES:
            raise ValueError(f"budget_mode は {BUDGET_MODES} のどれか: {kw['budget_mode']}")
        if kw["cost_model"] not in COST_MODELS:
            raise ValueError(f"cost_model は {COST_MODELS} のどれか: {kw['cost_model']}")
        return cls(**kw)

    def cost_rate(self, price: float) -> float:
        """片道のコストの割合（price は約定価格）。"""
        if self.cost_model == "tick" and price > 0:
            return max(self.cost, tick_size(price) / price)
        return self.cost


@dataclass
class Result:
    dates: np.ndarray
    equity: np.ndarray                 # 各営業日の引け後の総資産（期間外は NaN）
    curve: np.ndarray                  # 最大ドローダウンの計算に使う資産の推移（budget_mode により異なる）
    weekly: pd.DataFrame
    orders: pd.DataFrame
    trades: pd.DataFrame
    rules: Rules
    meta: dict = field(default_factory=dict)


def ranking(score: np.ndarray, universe_row: np.ndarray) -> np.ndarray:
    """ユニバース内で点数が NaN でない銘柄の位置を、点数の高い順に並べる（同点は銘柄コードの順）。"""
    idx = np.flatnonzero(universe_row & ~np.isnan(score))
    order = np.lexsort((idx, -score[idx]))
    return idx[order]


class Backtester:
    def __init__(self, m: Market, universe: np.ndarray, adv: np.ndarray, rules: Rules):
        self.m = m
        self.universe = universe
        self.adv = adv
        self.r = rules
        self.C_ff = m.C_ff

    # ---- 選定 ----
    def _affordable(self, t: int, j: int, budget: float) -> tuple[int, float]:
        lim = limit_price(self.m.C[t, j], self.r.limit_up_pct)
        shares = order_shares(budget, lim, self.r.lot)
        if shares == 0 or not within_turnover_cap(shares, lim, self.adv[t, j], self.r.max_order_to_adv):
            return 0, lim
        return shares, lim

    def kept_positions(self, t: int, held: set[int], score_fn: ScoreFn, budget: float) -> set[int]:
        """t の引け後の上位N（保有銘柄は買えるかの判定なしで数える）に入る保有銘柄。"""
        kept: set[int] = set()
        count = 0
        for j in ranking(score_fn(t), self.universe[t]):
            j = int(j)
            if j in held:
                kept.add(j)
                count += 1
            elif self._affordable(t, j, budget)[0] > 0:
                count += 1
            if count >= self.r.n_holdings:
                break
        return kept

    def build_orders(self, t: int, skip: set[int], slots: int, budget: float, cash: float,
                     score_fn: ScoreFn) -> list[dict]:
        orders: list[dict] = []
        if slots <= 0:
            return orders
        remaining = cash
        for pos, j in enumerate(ranking(score_fn(t), self.universe[t]), start=1):
            j = int(j)
            if j in skip:
                continue
            shares, lim = self._affordable(t, j, budget)
            if shares == 0:
                continue
            need = shares * lim * (1 + self.r.cost_rate(lim))
            if need > remaining:
                continue
            remaining -= need
            orders.append({"j": j, "rank": pos, "shares": shares, "limit": lim})
            if len(orders) >= slots:
                break
        return orders

    # ---- 実行 ----
    def run(self, score_fn: ScoreFn, start_date: str, end_date: str) -> Result:
        m, r = self.m, self.r
        weeks = [w for w in make_weeks(m.dates)
                 if m.dates[w.first] >= start_date and m.dates[w.last] <= end_date and w.pred >= 0]
        if not weeks:
            raise ValueError("対象の週がありません")
        T = len(m.dates)
        equity = np.full(T, np.nan)
        cash = float(r.capital)
        pos: dict[int, float] = {}       # 銘柄の位置 → 株数（分割で小数になりうる）
        pending: dict[int, float] = {}   # 売れ残り（次に売買が成立した日の始値で売る）
        trades: list[tuple] = []
        order_rows: list[dict] = []
        week_rows: list[dict] = []
        flow = {"traded": 0.0, "cost": 0.0}   # その週の売買代金とコスト（損益分岐のコストの計算用）

        def value(t: int, now: int | None = None) -> float:
            """t の終値での総資産。now（≥ t）を渡すと、now の朝までに効いた分割・併合を戻して t の時点の株数で評価する。"""
            v = cash
            for j, sh in list(pos.items()) + list(pending.items()):
                if now is not None and now != t:
                    sh = sh * m.cumF[now, j] / m.cumF[t, j]
                v += sh * self.C_ff[t, j]
            return v

        def trade(t: int, j: int, side: str, sh: float, px: float, reason: str) -> float:
            """売買を記録し、現金の増減（コスト込み）を返す。"""
            c = r.cost_rate(px)
            flow["traded"] += sh * px
            flow["cost"] += sh * px * c
            trades.append((m.dates[t], j, side, sh, px, reason))
            return -sh * px * (1 + c) if side == "buy" else sh * px * (1 - c)

        equity[weeks[0].pred] = cash
        for w in weeks:
            p = w.pred
            eq_p = value(p)
            fixed = r.budget_mode == "fixed"
            budget = (r.capital if fixed else min(r.capital, eq_p)) / r.n_holdings
            orders: list[dict] = []
            if len(w.days) >= 2:
                orders = self.build_orders(p, set(pos) | set(pending), r.n_holdings - len(pos),
                                           budget, float("inf") if fixed else cash, score_fn)
            n_filled = 0
            invested_after_open = False
            for t in w.days:
                # 分割・併合（その日の朝に効く）
                for book in (pos, pending):
                    for j in book:
                        a = m.adj[t, j]
                        if a != 1.0:
                            book[j] = book[j] / a
                # 上場廃止：最後に売買が成立した日の終値で売ったものとする
                for book in (pos, pending):
                    for j in [j for j in book if m.last_listed[j] < t]:
                        px = self.C_ff[m.last_listed[j], j]
                        sh = book.pop(j)
                        cash += trade(t, j, "sell", sh, px, "delisted")
                # 寄付：売れ残りの売り
                for j in list(pending):
                    if open_sell_filled(m.O[t, j]):
                        sh = pending.pop(j)
                        cash += trade(t, j, "sell", sh, m.O[t, j], "open_retry")
                # 寄付：買い
                if t == w.first:
                    for o in orders:
                        j = o["j"]
                        ok = buy_filled(m.O[t, j], o["limit"], m.adj[t, j])
                        order_rows.append({"pred_date": m.dates[p], "buy_date": m.dates[t], "j": j,
                                           "rank": o["rank"], "shares": o["shares"], "limit": o["limit"],
                                           "open": m.O[t, j], "filled": ok})
                        if ok:
                            cash += trade(t, j, "buy", float(o["shares"]), m.O[t, j], "open_limit")
                            pos[j] = float(o["shares"])
                            n_filled += 1
                    invested_after_open = len(pos) > 0
                # 大引け：継続保有の判断と引成の売り
                if t == w.last and pos:
                    held = set(pos)
                    if r.continuation == "none":
                        kept: set[int] = set()
                    else:
                        c = t - 1 if r.continuation == "prev_day" else t
                        b = r.capital if r.budget_mode == "fixed" else min(r.capital, value(c, now=t))
                        kept = self.kept_positions(c, held, score_fn, b / r.n_holdings)
                    for j in held - kept:
                        sh = pos.pop(j)
                        if close_sell_filled(m.C[t, j], m.L[t, j], bool(m.LL[t, j])):
                            cash += trade(t, j, "sell", sh, m.C[t, j], "close")
                        else:
                            pending[j] = sh
                equity[t] = value(t)
            week_rows.append({"first": m.dates[w.first], "last": m.dates[w.last], "n_days": len(w.days),
                              "equity": equity[w.last], "n_orders": len(orders), "n_filled": n_filled,
                              "invested": invested_after_open, "n_held_end": len(pos),
                              "n_pending_end": len(pending), "traded_value": flow["traded"],
                              "cost_paid": flow["cost"]})
            flow["traded"] = flow["cost"] = 0.0
        weekly = pd.DataFrame(week_rows)
        prev = np.concatenate([[equity[weeks[0].pred]], weekly["equity"].to_numpy()[:-1]])
        if r.budget_mode == "fixed":
            # 毎週、資金を運用資金（30万円）に戻したとみなす：その週の損益 ÷ 運用資金。現金のマイナスは補充したとみなす
            weekly["ret"] = (weekly["equity"].to_numpy() - prev) / r.capital
            curve = np.concatenate([[1.0], np.cumprod(1 + weekly["ret"].to_numpy())])
        else:
            weekly["ret"] = weekly["equity"].to_numpy() / prev - 1
            curve = equity
        orders_df = pd.DataFrame(order_rows)
        if len(orders_df):
            orders_df["code"] = m.codes[orders_df["j"].to_numpy()]
        trades_df = pd.DataFrame(trades, columns=["date", "j", "side", "shares", "price", "reason"])
        if len(trades_df):
            trades_df["code"] = m.codes[trades_df["j"].to_numpy()]
        return Result(dates=m.dates, equity=equity, curve=curve, weekly=weekly, orders=orders_df, trades=trades_df,
                      rules=r, meta={"start_pred": m.dates[weeks[0].pred], "end": m.dates[weeks[-1].last]})
