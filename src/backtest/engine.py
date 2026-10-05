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

比較する売買ルールの候補（CLAUDE.md 第5章、experiments/EXP-006〜016 の plan.md。初期値はどれも使わない＝基本ルール）
- 候補1 地合いフィルター（market_ok）：予測日 p の判定が False なら新しく買わない。継続の判断の日 c の判定が False なら、
  保有銘柄を継続せずに最終営業日の引けで売る（c の時点で分かる情報だけで判断する）
- 候補2 損切り（stop_loss）：買いの約定の後、逆指値 = 買値 × (1 − stop_loss) を呼値で切り下げた価格。毎営業日、日足で判定する
  （execution.stop_triggered）。継続保有中も最初の買値の逆指値を使い、分割・併合があれば係数で直す。最終営業日に当たった場合も
  損切りで売ったものとする（保守的）。損切りで売った後、その週は現金のまま。最終営業日に損切りで売った銘柄は、継続の判断の日 c
  の時点では保有していたので、継続の判断の「上位N」では保有銘柄として数える（prev_day のとき）
- 候補3 決算の回避（avoid）：avoid(日, 期間の最初の日, 最後の日) が True の銘柄を、p では順位付けの対象から除き（期間 = 保有する週の
  最初の営業日〜最後の営業日の暦日）、c では継続しない・「上位N」に数えない（期間 = その週の最終営業日〜次の保有期間の最終営業日）
- 候補4 指値なし（buy_order = "market"）：寄付の成行（execution.market_buy_filled）。株数・選定・現金の判定は指値と同じ
- 候補6・7 配分（weighting = "rank" / "inverse_vol"）：選定は基本ルールと同じ（均等の予算）。そのあと、その週に持つ銘柄
  （継続保有 ＋ 新しく買う銘柄）の配分を計算し、新しく買う銘柄の株数を 予算の基準（予算固定なら運用資金、基本ルールなら
  min(運用資金, 総資産)）× 配分 で決め直す（継続保有の銘柄の株数は変えない）。100株単位・売買代金の上限・（基本ルールでは）現金で
  買えない分は、他の銘柄に回さず現金のまま。1銘柄の配分は max_weight まで
  - rank：p の点数の高い順に rank_weights（N と同じ長さ）
  - inverse_vol：vol[p]（過去20営業日の値動きの大きさ）の逆数に比例し、合計は（銘柄の数 ÷ N）。値動きの大きさが無い銘柄は 1 ÷ N
- 候補9 保有期間（holding_weeks = H）：H 週ごとに予測と買いを行い、買った週から H 週目の最終営業日の引けで、継続の判断と売りを行う。
  間の週は何もしない（売れ残りの売り、分割・併合、上場廃止の処理だけ）。run の offset で最初の買いの週をずらす（0〜H−1。
  最初の offset 週は現金のまま）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from src.backtest.execution import (EPS, buy_filled, close_sell_filled, limit_price, market_buy_filled,
                                    open_sell_filled, order_shares, stop_price, stop_triggered, tick_size,
                                    within_turnover_cap, floor_to_tick)
from src.backtest.market import Market
from src.backtest.target import Week, make_weeks

ScoreFn = Callable[[int], np.ndarray]   # 日付の位置 → 銘柄ごとの点数（大きいほど上位。NaN は対象外）
AvoidFn = Callable[[int, str, str], np.ndarray]   # (判断する日の位置, 期間の最初の日, 最後の日) → 除く銘柄（bool [N]）
CONTINUATION_MODES = ("none", "prev_day", "same_day")
BUDGET_MODES = ("min_equity", "fixed")
COST_MODELS = ("flat", "tick")
BUY_ORDERS = ("limit", "market")
WEIGHTINGS = ("equal", "rank", "inverse_vol")


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
    buy_order: str = "limit"
    stop_loss: float | None = None
    weighting: str = "equal"
    rank_weights: tuple[float, ...] = ()
    max_weight: float = 0.5
    holding_weeks: int = 1

    @classmethod
    def from_config(cls, cfg: dict, **override) -> "Rules":
        rl = cfg.get("rules", {})
        kw = dict(n_holdings=cfg["capital"]["n_holdings"], capital=cfg["capital"]["initial_capital_jpy"],
                  limit_up_pct=cfg["order"]["limit_up_pct"], lot=cfg["order"]["lot_size"],
                  max_order_to_adv=cfg["order"]["max_order_to_adv"], cost=cfg["cost"]["one_way"],
                  continuation=cfg["backtest"].get("continuation", "prev_day"),
                  budget_mode=cfg["backtest"].get("budget_mode", "min_equity"),
                  cost_model=cfg["cost"].get("model", "flat"),
                  buy_order=rl.get("buy_order", "limit"), stop_loss=rl.get("stop_loss"),
                  weighting=rl.get("weighting", "equal"), rank_weights=tuple(rl.get("rank_weights") or ()),
                  max_weight=rl.get("max_weight", 0.5), holding_weeks=int(rl.get("holding_weeks", 1)))
        kw.update(override)
        kw["rank_weights"] = tuple(kw["rank_weights"])
        for key, allowed in (("continuation", CONTINUATION_MODES), ("budget_mode", BUDGET_MODES),
                             ("cost_model", COST_MODELS), ("buy_order", BUY_ORDERS), ("weighting", WEIGHTINGS)):
            if kw[key] not in allowed:
                raise ValueError(f"{key} は {allowed} のどれか: {kw[key]}")
        if kw["weighting"] == "rank" and len(kw["rank_weights"]) != kw["n_holdings"]:
            raise ValueError(f"rank_weights の長さ（{len(kw['rank_weights'])}）は N（{kw['n_holdings']}）と同じにする")
        if kw["weighting"] == "rank" and max(kw["rank_weights"]) > kw["max_weight"] + EPS:
            raise ValueError(f"rank_weights が max_weight（{kw['max_weight']}）を超えています")
        if kw["holding_weeks"] < 1:
            raise ValueError("holding_weeks は1以上")
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


def allocation_weights(rules: Rules, items: list[int], score: np.ndarray, vol: np.ndarray | None) -> dict[int, float]:
    """その週に持つ銘柄 items（継続保有 ＋ 新しく買う銘柄）の配分（資金に対する割合。候補6・7）。

    rank：点数の高い順（NaN は最後、同点は銘柄の位置の順）に rank_weights。inverse_vol：vol の逆数に比例し、合計は
    len(items) ÷ N。vol が無い（NaN・0以下）銘柄は 1 ÷ N とし、残りを vol のある銘柄で分ける。どちらも max_weight で頭打ち
    （超えた分は現金のまま）。
    """
    n = rules.n_holdings
    if rules.weighting == "rank":
        s = np.array([score[j] if score[j] == score[j] else -np.inf for j in items])
        order = np.lexsort((np.array(items), -s))
        w = {items[k]: rules.rank_weights[pos] for pos, k in enumerate(order)}
    elif rules.weighting == "inverse_vol":
        v = np.array([vol[j] if vol is not None else np.nan for j in items], dtype=float)
        ok = np.isfinite(v) & (v > 0)
        w = {j: 1.0 / n for j, good in zip(items, ok) if not good}
        if ok.any():
            inv = 1.0 / v[ok]
            share = inv / inv.sum() * ok.sum() / n
            w.update({j: float(x) for j, x in zip(np.array(items)[ok], share)})
    else:
        w = {j: 1.0 / n for j in items}
    return {j: min(x, rules.max_weight) for j, x in w.items()}


class Backtester:
    """market_ok：[T] の bool（候補1。False の日は買わない・継続しない）。avoid：候補3。vol：[T, N]（候補7）。"""

    def __init__(self, m: Market, universe: np.ndarray, adv: np.ndarray, rules: Rules,
                 market_ok: np.ndarray | None = None, avoid: AvoidFn | None = None, vol: np.ndarray | None = None):
        self.m = m
        self.universe = universe
        self.adv = adv
        self.r = rules
        self.C_ff = m.C_ff
        self.market_ok = market_ok
        self.avoid = avoid
        self.vol = vol
        if rules.weighting == "inverse_vol" and vol is None:
            raise ValueError("weighting = inverse_vol には vol が必要")

    # ---- 選定 ----
    def _affordable(self, t: int, j: int, budget: float) -> tuple[int, float]:
        lim = limit_price(self.m.C[t, j], self.r.limit_up_pct)
        shares = order_shares(budget, lim, self.r.lot)
        if shares == 0 or not within_turnover_cap(shares, lim, self.adv[t, j], self.r.max_order_to_adv):
            return 0, lim
        return shares, lim

    def kept_positions(self, t: int, held: set[int], score_fn: ScoreFn, budget: float,
                       exclude: np.ndarray | None = None) -> set[int]:
        """t の引け後の上位N（保有銘柄は買えるかの判定なしで数える）に入る保有銘柄。exclude の銘柄は数えない。"""
        kept: set[int] = set()
        count = 0
        row = self.universe[t] if exclude is None else self.universe[t] & ~exclude
        for j in ranking(score_fn(t), row):
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
                     score_fn: ScoreFn, exclude: np.ndarray | None = None) -> list[dict]:
        orders: list[dict] = []
        if slots <= 0:
            return orders
        remaining = cash
        score = score_fn(t)
        row = self.universe[t] if exclude is None else self.universe[t] & ~exclude
        for pos, j in enumerate(ranking(score, row), start=1):
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
        self._last_score = score
        return orders

    def reweight_orders(self, t: int, orders: list[dict], held: list[int], base: float, cash: float) -> list[dict]:
        """候補6・7：新しく買う銘柄の株数を 予算の基準 base × 配分 で決め直す。買えない分は現金のまま。"""
        if not orders:
            return orders
        w = allocation_weights(self.r, held + [o["j"] for o in orders], self._last_score,
                               None if self.vol is None else self.vol[t])
        out, remaining = [], cash
        for o in orders:
            j, lim = o["j"], o["limit"]
            shares = order_shares(base * w[j], lim, self.r.lot)
            cap = self.r.max_order_to_adv * self.adv[t, j]
            shares = min(shares, int(np.floor(cap / (lim * self.r.lot) + EPS)) * self.r.lot)
            unit = lim * (1 + self.r.cost_rate(lim))
            if shares * unit > remaining:
                shares = int(np.floor(remaining / (unit * self.r.lot) + EPS)) * self.r.lot
            if shares <= 0:
                continue
            remaining -= shares * unit
            out.append({**o, "shares": shares, "weight": w[j]})
        return out

    # ---- 実行 ----
    def run(self, score_fn: ScoreFn, start_date: str, end_date: str, offset: int = 0) -> Result:
        m, r = self.m, self.r
        H = r.holding_weeks
        if not 0 <= offset < H:
            raise ValueError(f"offset は 0〜{H - 1}: {offset}")
        weeks = [w for w in make_weeks(m.dates)
                 if m.dates[w.first] >= start_date and m.dates[w.last] <= end_date and w.pred >= 0]
        if not weeks:
            raise ValueError("対象の週がありません")
        T = len(m.dates)
        equity = np.full(T, np.nan)
        cash = float(r.capital)
        pos: dict[int, float] = {}       # 銘柄の位置 → 株数（分割で小数になりうる）
        pending: dict[int, float] = {}   # 売れ残り（次に売買が成立した日の始値で売る）
        stops: dict[int, float] = {}     # 損切りの逆指値の価格（候補2）
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

        def cycle_end(i: int) -> str:
            """i 番目の週に買ったときの保有期間の最終営業日（データの外なら、最後の週の最終営業日 + 7日 × 足りない週数）。"""
            k = i + H - 1
            if k < len(weeks):
                return str(m.dates[weeks[k].last])
            last = date.fromisoformat(str(m.dates[weeks[-1].last]))
            return (last + timedelta(days=7 * (k - len(weeks) + 1))).isoformat()

        equity[weeks[0].pred] = cash
        for i, w in enumerate(weeks):
            p = w.pred
            buy_week = i >= offset and (i - offset) % H == 0
            sell_week = i >= offset and (i - offset) % H == H - 1
            orders: list[dict] = []
            skip_reason = ""
            if not buy_week:
                skip_reason = "idle"
            elif len(w.days) < 2:
                skip_reason = "one_day_week"
            elif self.market_ok is not None and not self.market_ok[p]:
                skip_reason = "market_filter"
            else:
                eq_p = value(p)
                fixed = r.budget_mode == "fixed"
                base = r.capital if fixed else min(r.capital, eq_p)
                exclude = None if self.avoid is None else self.avoid(p, str(m.dates[w.first]), cycle_end(i))
                orders = self.build_orders(p, set(pos) | set(pending), r.n_holdings - len(pos),
                                           base / r.n_holdings, float("inf") if fixed else cash, score_fn, exclude)
                if r.weighting != "equal":
                    orders = self.reweight_orders(p, orders, sorted(pos), base, float("inf") if fixed else cash)
            n_filled = n_stop = 0
            invested_after_open = False
            stopped_last_day: set[int] = set()
            for t in w.days:
                # 分割・併合（その日の朝に効く）
                for book in (pos, pending):
                    for j in book:
                        a = m.adj[t, j]
                        if a != 1.0:
                            book[j] = book[j] / a
                            if book is pos and j in stops:
                                stops[j] = floor_to_tick(stops[j] / a)
                # 上場廃止：最後に売買が成立した日の終値で売ったものとする
                for book in (pos, pending):
                    for j in [j for j in book if m.last_listed[j] < t]:
                        px = self.C_ff[m.last_listed[j], j]
                        sh = book.pop(j)
                        stops.pop(j, None)
                        cash += trade(t, j, "sell", sh, px, "delisted")
                # 寄付：売れ残りの売り
                for j in list(pending):
                    if open_sell_filled(m.O[t, j]):
                        sh = pending.pop(j)
                        cash += trade(t, j, "sell", sh, m.O[t, j], "open_retry")
                # 寄付：買い
                bought_today: set[int] = set()
                if t == w.first:
                    for o in orders:
                        j = o["j"]
                        if r.buy_order == "market":
                            ok = market_buy_filled(m.O[t, j], m.H[t, j], bool(m.UL[t, j]), m.adj[t, j])
                        else:
                            ok = buy_filled(m.O[t, j], o["limit"], m.adj[t, j])
                        row = {"pred_date": m.dates[p], "buy_date": m.dates[t], "j": j, "rank": o["rank"],
                               "shares": o["shares"], "limit": o["limit"], "open": m.O[t, j], "filled": ok}
                        if "weight" in o:
                            row["weight"] = o["weight"]
                        order_rows.append(row)
                        if ok:
                            cash += trade(t, j, "buy", float(o["shares"]), m.O[t, j],
                                          "open_market" if r.buy_order == "market" else "open_limit")
                            pos[j] = float(o["shares"])
                            n_filled += 1
                            bought_today.add(j)
                            if r.stop_loss is not None:
                                stops[j] = stop_price(m.O[t, j], r.stop_loss)
                    invested_after_open = len(pos) > 0
                # 損切りの逆指値（候補2）
                if r.stop_loss is not None:
                    for j in list(pos):
                        px = stop_triggered(m.O[t, j], m.L[t, j], stops[j], j in bought_today)
                        if px is None:
                            continue
                        sh = pos.pop(j)
                        stops.pop(j)
                        reason = "stop_loss_gap" if (j not in bought_today and px == m.O[t, j]) else "stop_loss"
                        cash += trade(t, j, "sell", sh, px, reason)
                        n_stop += 1
                        if t == w.last:
                            stopped_last_day.add(j)
                # 大引け：継続保有の判断と引成の売り
                if t == w.last and sell_week and (pos or stopped_last_day):
                    held = set(pos)
                    if r.continuation == "none":
                        kept: set[int] = set()
                    else:
                        c = t - 1 if r.continuation == "prev_day" else t
                        if r.continuation == "prev_day":
                            held |= stopped_last_day   # c の時点では保有していた
                        if self.market_ok is not None and not self.market_ok[c]:
                            kept = set()
                        else:
                            b = r.capital if r.budget_mode == "fixed" else min(r.capital, value(c, now=t))
                            exclude = None if self.avoid is None else self.avoid(c, str(m.dates[t]), cycle_end(i + 1))
                            kept = self.kept_positions(c, held, score_fn, b / r.n_holdings, exclude)
                    for j in set(pos) - kept:
                        sh = pos.pop(j)
                        stops.pop(j, None)
                        if close_sell_filled(m.C[t, j], m.L[t, j], bool(m.LL[t, j])):
                            cash += trade(t, j, "sell", sh, m.C[t, j], "close")
                        else:
                            pending[j] = sh
                equity[t] = value(t)
            week_rows.append({"first": m.dates[w.first], "last": m.dates[w.last], "n_days": len(w.days),
                              "equity": equity[w.last], "n_orders": len(orders), "n_filled": n_filled,
                              "invested": invested_after_open, "n_held_end": len(pos),
                              "n_pending_end": len(pending), "traded_value": flow["traded"],
                              "cost_paid": flow["cost"], "skip_reason": skip_reason, "n_stop": n_stop})
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
                      rules=r, meta={"start_pred": m.dates[weeks[0].pred], "end": m.dates[weeks[-1].last],
                                     "offset": offset})
