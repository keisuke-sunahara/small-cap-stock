"""ウォークフォワード検証の点数（experiments/README.md 第2章）。

- 評価の週：売買の開始の週（backtest.trade_start_date）〜ホールドアウトの前の週。予測日は前の週の最終営業日
- 点数を付ける日：各週の予測日（kind = "pred"）と、継続の判断の日（最終営業日の前の営業日。kind = "cont"）。
  営業日が1日の週の継続の判断の日は前の週の予測日と同じ日になる（kind = "pred+cont"）
- 月ごとに学習し直す（src.models.dataset.walk_forward_splits）。日 d の点数には、「その月の最初の予測日 ≤ d」となる最も新しい月の
  モデルを使う（d の時点で学習を終えているモデルだけ。学習の起点は、その月の最初の予測日の gap + 1 営業日前まで）
- 学習期間：validation.train_window = rolling（直近 train_window_years 年。基本）/ expanding（拡大窓。EXP-005）
- model.train_universe = affordable（候補8。EXP-014）：「買える」= 起点の日の終値から計算した指値で100株の金額が 運用資金 ÷ N 以下。
  学習の行は「ユニバース ∩ 買える」で、特徴量の順位と目的変数の順位もその中で計算する。点数はユニバースの全銘柄に付け、
  特徴量は「ユニバース ∩ 買える」の分布の中での位置（rank_against）に直す
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.execution import EPS, limit_price_array
from src.backtest.market import Market
from src.backtest.target import Week, make_weeks
from src.models.dataset import daily_panel, walk_forward_splits
from src.models.train import Model, shuffle_within_day


def eval_end(cfg: dict) -> str:
    return (pd.Timestamp(cfg["data"]["holdout_start"]) - pd.Timedelta(days=1)).date().isoformat()


def eval_weeks(m: Market, cfg: dict) -> list[Week]:
    start, end = cfg["backtest"]["trade_start_date"], eval_end(cfg)
    return [w for w in make_weeks(m.dates) if w.pred >= 0 and m.dates[w.first] >= start and m.dates[w.last] <= end]


def empty_score_days(m: Market, cfg: dict, universe: np.ndarray) -> list[int]:
    """点数を付ける日のうち、ユニバースが空の日（点数の行が無い日）。"""
    return [int(t) for t in score_days(m, cfg)["t"] if not universe[t].any()]


def score_days(m: Market, cfg: dict) -> pd.DataFrame:
    """点数を付ける日（列 t・kind）。"""
    kinds: dict[int, set[str]] = {}
    for w in eval_weeks(m, cfg):
        kinds.setdefault(w.pred, set()).add("pred")
        if w.last - 1 >= 0:
            kinds.setdefault(w.last - 1, set()).add("cont")
    rows = [(t, "+".join(k for k in ("pred", "cont") if k in ks)) for t, ks in sorted(kinds.items())]
    return pd.DataFrame(rows, columns=["t", "kind"])


def affordable_mask(m: Market, cfg: dict) -> np.ndarray:
    """起点の日の終値から計算した指値で、100株（売買単位）の金額が 運用資金 ÷ N 以下（候補8の「買える」。予算固定の基準）。"""
    lim = limit_price_array(m.C, cfg["order"]["limit_up_pct"])
    budget = cfg["capital"]["initial_capital_jpy"] / cfg["capital"]["n_holdings"]
    with np.errstate(invalid="ignore"):
        return np.isfinite(lim) & (lim * cfg["order"]["lot_size"] <= budget + EPS)


def rank_against(values: np.ndarray, row_mask: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    """各日（行）の row_mask の銘柄の値を、ref_mask の銘柄の値の分布の中での順位（0〜1）に直す。

    順位 = (ref の中で x より小さい数 + x 以下の数 + 1) ÷ 2 ÷ ref の数。ref に入っている銘柄では、同順位を平均にした
    順位 ÷ 銘柄数（src.backtest.target.cross_section_rank）と一致する。ref の外の値は、ref の間の位置になる
    （ref の最大より大きければ 1 + 0.5 ÷ ref の数）。NaN・row_mask の外は NaN。
    """
    out = np.full(values.shape, np.nan)
    for k in range(values.shape[0]):
        v = values[k]
        ref = np.sort(v[ref_mask[k] & np.isfinite(v)])
        rows = np.flatnonzero(row_mask[k] & np.isfinite(v))
        if len(ref) == 0 or len(rows) == 0:
            continue
        left = np.searchsorted(ref, v[rows], side="left")
        right = np.searchsorted(ref, v[rows], side="right")
        out[k, rows] = (left + right + 1) / 2 / len(ref)
    return out


@dataclass
class WalkForwardResult:
    scores: pd.DataFrame      # date・code・kind・score
    splits: pd.DataFrame      # 月ごとの学習の範囲と行数


def splits_for(m: Market, cfg: dict):
    v = cfg["validation"]
    if v["train_window"] not in ("rolling", "expanding"):
        raise ValueError(f"validation.train_window は rolling / expanding: {v['train_window']}")
    years = v["train_window_years"] if v["train_window"] == "rolling" else None
    return walk_forward_splits(m.dates, v["train_start"], cfg["backtest"]["trade_start_date"], eval_end(cfg),
                               v["gap_days"], cfg["target"]["horizon_days"], years)


def walk_forward_scores(m: Market, universe: np.ndarray, feats: dict[str, np.ndarray], cfg: dict,
                        bad_event: np.ndarray | None, log=print) -> WalkForwardResult:
    mcfg = cfg["model"]
    horizon = cfg["target"]["horizon_days"]
    how = cfg["features"]["normalize"]
    seed = int(cfg["project"]["random_seed"])
    splits = splits_for(m, cfg)
    sd = score_days(m, cfg)
    e0s = np.array([int(s.eval_preds.min()) for s in splits])
    sd["split"] = np.searchsorted(e0s, sd["t"].to_numpy(), side="right") - 1
    if (sd["split"] < 0).any():
        raise ValueError("最初の月の予測日より前に点数を付ける日があります")
    train_days = np.arange(min(int(s.train_t.min()) for s in splits), max(int(s.train_t.max()) for s in splits) + 1)
    score_t = sd["t"].to_numpy()
    names = list(feats)

    if mcfg.get("train_universe", "all") == "affordable":
        tu = universe & affordable_mask(m, cfg)
        train = daily_panel(m, tu, feats, horizon, how, t_index=train_days, bad_event=bad_event)
        sub = universe[score_t]
        si, sj = np.nonzero(sub)
        Xs = np.empty((len(si), len(names)), dtype=np.float32)
        for k, n in enumerate(names):
            if how != "rank":
                raise ValueError("train_universe = affordable は normalize = rank だけ")
            Xs[:, k] = rank_against(feats[n][score_t], sub, tu[score_t])[si, sj]
        score_rows_t, score_rows_j = score_t[si], sj
    elif mcfg.get("train_universe", "all") == "all":
        all_t = np.union1d(train_days, score_t)
        panel = daily_panel(m, universe, feats, horizon, how, t_index=all_t, bad_event=bad_event)
        train = panel
        is_score = np.isin(panel.t, score_t)
        Xs, score_rows_t, score_rows_j = panel.X[is_score], panel.t[is_score], panel.j[is_score]
    else:
        raise ValueError(f"model.train_universe は all / affordable: {mcfg.get('train_universe')}")

    y = train.y.astype(float)
    if mcfg.get("target_shuffle", False):
        y = shuffle_within_day(y, train.t, seed)
    split_of_row = pd.Series(sd["split"].to_numpy(), index=score_t).reindex(score_rows_t).to_numpy()
    scores = np.full(len(score_rows_t), np.nan)
    info = []
    for k, s in enumerate(splits):
        t0 = time.time()
        lo, hi = int(s.train_t.min()), int(s.train_t.max())
        sel = (train.t >= lo) & (train.t <= hi) & np.isfinite(y)
        rows = split_of_row == k
        if rows.any():
            first_score = int(score_rows_t[rows].min())
            if hi + horizon >= first_score or hi + cfg["validation"]["gap_days"] >= e0s[k]:
                raise AssertionError(f"{s.month}: 学習の目的変数の期間が点数を付ける日に重なります")
            model = Model(mcfg, seed).fit(train.X[sel], y[sel], train.t[sel])
            scores[rows] = model.predict(Xs[rows])
        info.append({"month": s.month, "train_first": str(m.dates[lo]), "train_last": str(m.dates[hi]),
                     "train_rows": int(sel.sum()), "score_days": int(sd["split"].eq(k).sum()),
                     "score_rows": int(rows.sum()), "seconds": round(time.time() - t0, 1)})
        if log is not None:
            log(f"{s.month}: 学習 {info[-1]['train_first']}〜{info[-1]['train_last']}（{info[-1]['train_rows']} 行）、"
                f"点数 {info[-1]['score_rows']} 行、{info[-1]['seconds']}秒")
    kind = pd.Series(sd["kind"].to_numpy(), index=score_t).reindex(score_rows_t).to_numpy()
    df = pd.DataFrame({"date": m.dates[score_rows_t].astype(str), "code": m.codes[score_rows_j].astype(str),
                       "kind": kind, "score": scores})
    return WalkForwardResult(scores=df, splits=pd.DataFrame(info))


class ScoreTable:
    """保存した点数（date・code・score）を、engine の点数の関数（日付の位置 → [N]）にする。点数の無い日はエラー。

    empty_days：点数を付ける日のうち、ユニバースが空で点数の行が無い日（2020-10-01 の終日の売買停止など）。この日は点数が
    すべて NaN（順位に入る銘柄が無い）とする。ベースラインの計算と同じく、継続の判断の「上位N」が空になり、保有銘柄は売る
    """

    def __init__(self, m: Market, scores: pd.DataFrame, empty_days=()):
        self.n = len(m.codes)
        self.by_t: dict[int, np.ndarray] = {}
        t = scores["date"].map(m.date_index).to_numpy()
        j = scores["code"].map(m.code_index).to_numpy()
        if np.isnan(t.astype(float)).any() or np.isnan(j.astype(float)).any():
            raise ValueError("点数の日付・銘柄コードが Market にありません")
        for tt, idx in pd.Series(np.arange(len(t))).groupby(t).groups.items():
            arr = np.full(self.n, np.nan)
            arr[j[idx]] = scores["score"].to_numpy()[idx]
            self.by_t[int(tt)] = arr
        for t in empty_days:
            if int(t) in self.by_t:
                raise ValueError(f"ユニバースが空の日に点数があります（位置 {t}）")
            self.by_t[int(t)] = np.full(self.n, np.nan)

    def __call__(self, t: int) -> np.ndarray:
        if t not in self.by_t:
            raise KeyError(f"点数の無い日です（位置 {t}）")
        return self.by_t[t]

    def matrix(self, ts) -> np.ndarray:
        return np.array([self(t) for t in ts])
