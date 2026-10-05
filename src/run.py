"""実験の実行（CLAUDE.md 第8章、experiments/README.md）。1つのコマンドで学習から評価までを行う。

実行：python -m src.run --exp EXP-xxx

- モデルの実験（config.yaml に model_experiment が無い）：ウォークフォワードで学習して点数を付け（logs/backtest/<実験ID>/scores.parquet）、
  基本ルールで売買する
- 売買ルールの候補の実験（model_experiment に確定したモデルの実験ID）：設定は base.yaml ＋ そのモデルの実験の差分 ＋ この実験の差分。
  モデル・特徴量・学習期間・目的変数がモデルの実験と同じなら、その点数を使う（学習し直さない。点数も保存しない）。違えば（候補8）
  学習し直して点数を保存する。比べる相手（確定したモデル ＋ 基本ルール）も同じ期間・同じ計算で出す
- 出力：experiments/<実験>/metrics.json・results.md、logs/backtest/<実験ID>/（weekly_returns.csv、予算固定・一律0.3%の
  weekly.csv・orders.csv・trades.csv、学習した場合は scores.parquet・splits.csv）。既にあるファイルは上書きしない
- 実行の前に確かめること（--no-git-check で外せるが、正式な実行では外さない）：ブランチが exp/<実験ID>、plan.md と config.yaml が
  コミット済みで変更が無く、GitHub に送信済み（事前登録。CLAUDE.md 第8章）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest import evaluate as ev
from src.backtest.engine import Backtester, Rules
from src.backtest.market import JST, Market
from src.config import ROOT, deep_merge, find_experiment_dir, load_config
from src.models.walkforward import ScoreTable, affordable_mask, walk_forward_scores

TRAIN_KEYS = ("model", "features", "validation", "target", "universe", "project")


@dataclass
class Inputs:
    m: Market
    u: np.ndarray
    adv: np.ndarray
    feats: dict[str, np.ndarray]
    bad_event: np.ndarray | None = None
    fd: object | None = None          # FeatureData（候補3の決算発表予定日）
    vol: np.ndarray | None = None     # 過去20営業日の値動きの大きさ（候補7）


def experiment_config(exp_id: str) -> tuple[dict, dict | None, str | None]:
    """(この実験の設定, 確定したモデルの実験の設定（モデルの実験なら None）, モデルの実験ID)。"""
    diff = yaml.safe_load((find_experiment_dir(exp_id) / "config.yaml").read_text(encoding="utf-8")) or {}
    if "model_experiment" not in diff:
        return load_config(exp_id), None, None
    me = diff["model_experiment"]
    if not me:
        raise SystemExit(f"{exp_id}：config.yaml の model_experiment が未記入です（モデルの確定の後に書いてコミットする）")
    model_cfg = load_config(me)
    return deep_merge(model_cfg, diff), model_cfg, me


def needs_training(cfg: dict, model_cfg: dict | None) -> bool:
    return model_cfg is None or any(cfg.get(k) != model_cfg.get(k) for k in TRAIN_KEYS)


def make_extras(cfg: dict, inp: Inputs) -> dict:
    """設定の売買ルールの候補に必要なデータを、Backtester の引数にする。"""
    from src.backtest.rules import EarningsCalendar, market_ok
    rl = cfg.get("rules", {})
    extras: dict = {}
    if rl.get("market_filter_ma_days"):
        extras["market_ok"] = market_ok(inp.m.topix, int(rl["market_filter_ma_days"]))
    if rl.get("earnings_avoid"):
        cal = EarningsCalendar.from_feature_data(inp.fd)
        undecided = bool(rl.get("earnings_avoid_undecided", True))
        extras["avoid"] = lambda d, lo, hi: cal.mask(d, lo, hi, include_undecided=undecided)
    if rl.get("weighting") == "inverse_vol":
        extras["vol"] = inp.vol
    return extras


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def check_preregistered(exp_id: str) -> None:
    """ブランチ、plan.md・config.yaml のコミット・送信を確かめる（事前登録）。"""
    d = find_experiment_dir(exp_id).relative_to(ROOT).as_posix()
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != f"exp/{exp_id}":
        raise SystemExit(f"ブランチが exp/{exp_id} ではありません: {branch}")
    files = [f"{d}/plan.md", f"{d}/config.yaml"]
    if git("status", "--porcelain", "--", *files):
        raise SystemExit(f"plan.md・config.yaml にコミットしていない変更があります: {files}")
    commit = git("log", "-1", "--format=%H", "--", *files)
    if not commit:
        raise SystemExit("plan.md・config.yaml がコミットされていません")
    if not git("branch", "-r", "--contains", commit):
        raise SystemExit(f"plan.md・config.yaml のコミット {commit[:7]} が GitHub に送信されていません")


def feature_hash(cfg: dict) -> str:
    from src.features.registry import get, resolve
    names = resolve(cfg["features"]["set"])
    return hashlib.sha256("|".join(f"{n}:{get(n).source_hash()}" for n in names).encode()).hexdigest()[:16]


def load_inputs(cfg: dict) -> Inputs:
    """実データ（data/processed の Market と特徴量。ホールドアウト期間は読まない）。"""
    from src.backtest.market import load_market
    from src.backtest.universe import avg_turnover, bad_factor_events, universe_mask
    from src.features.build import load_features
    from src.features.data import load_feature_data
    from src.features.registry import resolve
    m = load_market(cfg)
    adv = avg_turnover(m, cfg["universe"]["turnover_window"])
    u = universe_mask(m, cfg, adv)
    feats = load_features(resolve(cfg["features"]["set"]), cfg, m)
    thr = cfg["universe"].get("bad_adjfactor_threshold")
    bad_event = bad_factor_events(m, thr)[1] if thr is not None else None
    rl = cfg.get("rules", {})
    fd = load_feature_data(m, cfg) if rl.get("earnings_avoid") else None
    vol = load_features(["volatility_20d"], cfg, m)["volatility_20d"] if rl.get("weighting") == "inverse_vol" else None
    return Inputs(m=m, u=u, adv=adv, feats=feats, bad_event=bad_event, fd=fd, vol=vol)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if x != x else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def run_experiment(exp_id: str, cfg: dict, model_cfg: dict | None, model_exp: str | None, inp: Inputs, *,
                   exp_dir: Path, log_root: Path, base_cfg: dict, random_path: Path | None,
                   random_runs: int = ev.RANDOM_RUNS, meta: dict | None = None, log=print) -> dict:
    """学習（必要なら）・売買・評価を行い、metrics を保存して返す。base_cfg はランダムの分布（基本ルール）に使う設定。"""
    t0 = time.time()
    m = inp.m
    out_files = [exp_dir / "metrics.json", exp_dir / "results.md"]
    log_dir = log_root / exp_id
    if any(p.exists() for p in out_files) or (log_dir / "weekly_returns.csv").exists() or \
            (needs_training(cfg, model_cfg) and (log_dir / "scores.parquet").exists()):
        raise SystemExit(f"{exp_id} の結果が既にあります（上書きしません）")
    ctx = ev.make_context(m, inp.u, inp.adv, cfg, make_extras(cfg, inp))
    base_ctx = ev.Context(**{**ctx.__dict__, "extras": {}})
    metrics: dict = {"meta": {"experiment": exp_id, "model_experiment": model_exp, **(meta or {}),
                              "market_built_at_jst": m.meta.get("built_at_jst"),
                              "data_latest_fetch_jst": m.meta.get("data_latest_fetch_jst"),
                              "period": [ctx.start_pred, ctx.week_last[-1]], "weeks": len(ctx.weeks),
                              "config": cfg}}

    # 点数
    trained = needs_training(cfg, model_cfg)
    if trained:
        wf = walk_forward_scores(m, inp.u, inp.feats, cfg, inp.bad_event, log=log)
        scores = wf.scores
        log_dir.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(log_dir / "scores.parquet", index=False, compression="zstd")
        wf.splits.to_csv(log_dir / "splits.csv", index=False)
        metrics["model"] = {"months": len(wf.splits), "train_rows_total": int(wf.splits["train_rows"].sum()),
                            "first_split": wf.splits.iloc[0].to_dict(), "last_split": wf.splits.iloc[-1].to_dict()}
    else:
        scores = pd.read_parquet(log_root / model_exp / "scores.parquet")
        metrics["model"] = {"scores_from": f"logs/backtest/{model_exp}/scores.parquet"}
    st = ScoreTable(m, scores)
    s_pred = st.matrix([w.pred for w in ctx.weeks])

    # 予測の評価
    ic = ev.rank_ic_stats(ctx, s_pred)
    metrics["rank_ic"] = ic
    metrics["deciles_mean_weekly_return"] = ev.decile_returns(ctx, s_pred, cfg["evaluation"]["top_quantiles"])
    metrics["top_n_unaffordable"] = ev.top_n_unaffordable(ctx, s_pred, cfg)
    if cfg["model"].get("train_universe") == "affordable":
        aff_rows = (inp.u & affordable_mask(m, cfg))[[w.pred for w in ctx.weeks]]
        metrics["rank_ic_affordable"] = {"this": ev.rank_ic_stats(ctx, s_pred, aff_rows)}
        if model_exp is not None:
            sm = ScoreTable(m, pd.read_parquet(log_root / model_exp / "scores.parquet"))
            metrics["rank_ic_affordable"][model_exp] = ev.rank_ic_stats(ctx, sm.matrix([w.pred for w in ctx.weeks]),
                                                                        aff_rows)

    # 候補6・7 の事前の確認
    rl = cfg.get("rules", {})
    if rl.get("weighting", "equal") != "equal":
        from src.backtest.rules import rank_return_trend
        fixed_rules = Rules.from_config(cfg, budget_mode="fixed", weighting="equal")
        bt = Backtester(m, inp.u, inp.adv, fixed_rules)
        pre = rank_return_trend(bt, m, ctx.weeks, st, fixed_rules.capital / fixed_rules.n_holdings)
        metrics["precheck_rank_trend"] = pre
        if rl["weighting"] == "rank" and not pre["trend"]:
            metrics["executed"] = False
            metrics["note"] = "事前の確認で傾向が無かったため、売買ルールの候補6は実行しない（試行回数に数えない。EXP-012 の plan.md）"
            _save(exp_id, metrics, None, exp_dir, log_dir, ctx, None, log, t0)
            return metrics
    metrics["executed"] = True

    # 売買
    grid = ev.run_backtests(ctx, st, cfg)
    bt_metrics, weekly = ev.summarize_grid(ctx, grid)
    metrics["backtest"] = bt_metrics

    # ベースライン・ランダム
    n_base = base_cfg["capital"]["n_holdings"]
    baselines = ev.load_baselines(cfg, ctx)
    metrics["baselines"] = baselines
    if cfg["capital"]["n_holdings"] != n_base:
        metrics["baselines_note"] = f"ベースラインは N = {n_base}（基本ルール）の値。N が違うので参考"
    dist = ev.random_distribution(base_ctx, base_cfg, random_path, runs=random_runs, log=log)
    rep = json.loads((ROOT / cfg["evaluation"]["baselines_report"]).read_text(encoding="utf-8"))
    ev.check_random_against_report(dist, rep)
    metrics["random_runs"] = int(dist["seed"].nunique())
    metrics["random_percentile"] = {b: {c: ev.percentile_in(dist, b, c, bt_metrics[b][c]["annual_excess_vs_universe"])
                                        for c in ev.MAIN_COSTS} for b in ("fixed", "min_equity")}
    metrics["criteria"] = ev.criteria(bt_metrics, ic, baselines, cfg)
    if model_exp is None:
        metrics["model_candidate"] = ev.model_candidate(bt_metrics, ic, baselines)
        if cfg["validation"]["train_window"] == "expanding":
            metrics["model_candidate"]["note"] = "参考：EXP-005（拡大窓）は採用候補にしない（README 第4章）"
    if cfg["model"].get("target_shuffle"):
        metrics["leak_check"] = ev.leak_check(ic, bt_metrics, dist)

    # 売買ルールの候補：確定したモデル ＋ 基本ルールとの比較
    base_weekly = None
    if model_exp is not None:
        st_model = st if not trained else ScoreTable(m, pd.read_parquet(log_root / model_exp / "scores.parquet"))
        base_grid = ev.run_backtests(base_ctx, st_model, model_cfg)
        base_metrics, base_weekly = ev.summarize_grid(base_ctx, base_grid)
        metrics["compared_with"] = {"label": f"{model_exp} ＋ 基本ルール",
                                    "fixed": {c: base_metrics["fixed"][c] for c in ev.MAIN_COSTS}}
        metrics["rule_adoption"] = ev.rule_adoption(bt_metrics, weekly, base_metrics, base_weekly)
        if cfg["capital"]["n_holdings"] != model_cfg["capital"]["n_holdings"]:
            metrics["rule_adoption"]["note"] = "候補5（保有銘柄数）は参考として出すだけで、採用の判断はしない（README 第5章）"
            metrics["rule_adoption"]["adopt"] = None

    main = grid["fixed"]["cost_0.3%"]
    if rl.get("stop_loss"):
        metrics["stop_loss"] = ev.stop_loss_stats(m, main[0].trades)
    metrics["delisting_of_orders"] = ev.delisting_of_orders(m, main[0].orders)
    _save(exp_id, metrics, weekly, exp_dir, log_dir, ctx, main, log, t0, base_weekly)
    return metrics


def _save(exp_id, metrics, weekly, exp_dir: Path, log_dir: Path, ctx, main, log, t0, base_weekly=None) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    if weekly is not None:
        weekly.to_csv(log_dir / "weekly_returns.csv", index=False)
        if base_weekly is not None:
            base_weekly.to_csv(log_dir / "weekly_returns_compared.csv", index=False)
        for name in ("weekly", "orders", "trades"):
            frames = []
            for k, res in enumerate(main):
                f = getattr(res, name).copy()
                f.insert(0, "offset", k)
                frames.append(f)
            pd.concat(frames, ignore_index=True).to_csv(log_dir / f"{name}.csv", index=False)
    metrics["meta"]["elapsed_sec"] = round(time.time() - t0, 1)
    (exp_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    (exp_dir / "results.md").write_text(results_markdown(exp_id, metrics), encoding="utf-8")
    if log is not None:
        log(f"保存: {exp_dir / 'metrics.json'}、{exp_dir / 'results.md'}、{log_dir}")


def _pct(x) -> str:
    return "—" if x is None or x != x else f"{x * 100:+.2f}%"


def _ratio(x) -> str:
    return "—" if x is None or x != x else f"{x * 100:.1f}%"


def results_markdown(exp_id: str, mt: dict) -> str:
    """results.md の数字の部分（考察は実行の後に人が追記する）。"""
    me = mt["meta"]
    lines = [f"# {exp_id} の結果", "",
             f"- 実行：{me.get('run_at_jst', '')}／コミット：{me.get('git_commit', '')}／データの取得日：{me.get('data_latest_fetch_jst')}",
             f"- 期間：{me['period'][0]}〜{me['period'][1]}（{me['weeks']}週）／モデルの実験：{me.get('model_experiment') or '（この実験で学習）'}",
             ""]
    ic = mt["rank_ic"]
    lines += ["## 予測", "",
              f"- Rank IC：平均 {ic['mean']:.4f}、標準偏差 {ic['std']:.4f}、平均÷標準偏差 {ic['mean_over_std']:.3f}、t値 {ic['t']:.2f}（{ic['weeks']}週）",
              f"- 予測の10分位ごとの翌週の平均リターン（低→高）：" + "、".join(_pct(x) for x in mt["deciles_mean_weekly_return"]),
              f"- 点数の上位{mt['top_n_unaffordable']['top_n']}のうち100株を買えない割合：{_ratio(mt['top_n_unaffordable']['ratio'])}", ""]
    if "precheck_rank_trend" in mt:
        p = mt["precheck_rank_trend"]
        lines += [f"- 事前の確認（順位と翌週のリターンの順位相関）：平均 {p['mean']:.4f}、t値 {p['t']:.2f}、傾向 {'あり' if p['trend'] else 'なし'}", ""]
    if not mt.get("executed", True):
        lines += ["## 売買", "", mt.get("note", ""), "", "## 考察", "", "（実行の後に追記する）", ""]
        return "\n".join(lines)
    lines += ["## 売買（週次リターン。超過はユニバース平均（コストなし）に対して）", "",
              "| 予算 | コスト | 年率 | 年率の超過 | シャープ | 最大DD | 1週間の最大損失 | 約定率 | 現金の週 | 損益分岐の片道コスト |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for b, label in (("fixed", "予算固定"), ("min_equity", "基本ルール")):
        for c in ev.COST_NAMES:
            s = mt["backtest"][b][c]
            be = s.get("breakeven_one_way_cost_vs_universe")
            lines.append(f"| {label} | {c} | {_pct(s['annual_return'])} | {_pct(s['annual_excess_vs_universe'])} | "
                         f"{s['sharpe']:.2f} | {_pct(s['max_drawdown'])} | {_pct(s['worst_week'])} | "
                         f"{_ratio(s.get('fill_rate'))} | {_ratio(s.get('cash_week_ratio'))} | "
                         f"{_pct(be) if (b, c) == ('fixed', 'cost_0.3%') else '—'} |")
    lines += ["", "## 判定", ""]
    if "model_candidate" in mt:
        mc = mt["model_candidate"]
        lines.append(f"- モデルの採用候補（README 第4章）：{'候補' if mc['candidate'] else '候補ではない'}"
                     f"（Rank IC {'○' if mc['rank_ic_ok'] else '×'}、ベースライン：" +
                     "、".join(f"{c} {'○' if v['all'] else '×'}" for c, v in mc["beats_baselines"].items()) + "）"
                     + (f" {mc['note']}" if "note" in mc else ""))
    if "leak_check" in mt:
        lc = mt["leak_check"]
        lines.append(f"- 情報漏れの確認（EXP-003）：{'見つからなかった' if lc['no_leak_found'] else '**疑われる（止めて報告）**'}"
                     f"（Rank IC {'○' if lc['ic_ok'] else '×'}、ランダムの5〜95%点 {'○' if lc['random_ok'] else '×'}）")
    if "rule_adoption" in mt:
        ra = mt["rule_adoption"]
        adopt = {True: "採用", False: "不採用", None: "判断しない（参考）"}[ra["adopt"]]
        lines.append(f"- 売買ルールの候補（README 第5章）：{adopt}（年率の超過：" +
                     "、".join(f"{c} {_pct(v[0])} vs {_pct(v[1])}" for c, v in ra["excess"].items()) +
                     f"、週次の差の t値 {ra['weekly_diff_t']:.2f}（基準 {ra['t_min']}）、最大DD {_pct(ra['max_drawdown'][0])} vs {_pct(ra['max_drawdown'][1])}）")
    rp = mt["random_percentile"]
    lines.append(f"- ランダム（{mt.get('random_runs')}回）の分布の中での位置（年率の超過）：" +
                 "、".join(f"{b} {c} {rp[b][c] * 100:.1f}%点" for b in rp for c in rp[b]))
    cr = mt["criteria"]
    lines += ["", "### 合格基準（CLAUDE.md 第9章。ホールドアウトはフェーズ5）", "",
              f"- Rank IC の平均がプラスで t値 ≥ 2：{'○' if cr['rank_ic_positive_t2'] else '×'}"]
    names = {"excess_positive": "超過リターンがプラス", "majority_years_positive": "過半数の年で超過がプラス",
             "beats_all_baselines": "4つのベースラインを上回る",
             "after_data_cost_beats_universe_same_cost": "データ費用を引いてユニバース平均（同じコスト）を上回る",
             "after_all_costs_positive": "売買コストとデータ費用を引いた年率がプラス"}
    for b in ("fixed", "min_equity"):
        for c in ev.MAIN_COSTS:
            d = cr[b][c]
            lines.append(f"- {b} {c}：" + "、".join(f"{v} {'○' if d[k] else '×'}" for k, v in names.items()))
    lines += ["", "## 考察", "", "（実行の後に追記する）", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--no-git-check", action="store_true", help="事前登録の確認を外す（正式な実行では使わない）")
    args = ap.parse_args()
    exp_id = args.exp
    if not args.no_git_check:
        check_preregistered(exp_id)
    cfg, model_cfg, model_exp = experiment_config(exp_id)
    base_cfg = load_config()
    meta = {"git_commit": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_check": not args.no_git_check, "run_at_jst": datetime.now(JST).isoformat(timespec="seconds"),
            "feature_hash": feature_hash(cfg)}
    inp = load_inputs(cfg)
    run_experiment(exp_id, cfg, model_cfg, model_exp, inp, exp_dir=find_experiment_dir(exp_id),
                   log_root=ROOT / "logs" / "backtest", base_cfg=base_cfg,
                   random_path=ROOT / base_cfg["evaluation"]["random_runs_file"], meta=meta,
                   log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
