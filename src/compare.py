"""実験の比較（CLAUDE.md 第8章）と、モデルの確定の手順（experiments/README.md 第4章）。

- python -m src.compare EXP-005 EXP-012：同じ期間・同じ指標で並べる（experiments/<実験>/metrics.json と
  logs/backtest/<実験ID>/weekly_returns.csv）。最初の実験に対する週次リターンの差の t値も出す
- python -m src.compare --select-model EXP-002 EXP-001 EXP-004：単純な順に並べた実験から、README 第4章の手順でモデルを決める
  （採用候補のうち最も複雑なものから順に、それより単純なすべての採用候補に対する週次リターンの差の t値がすべて2以上なら選ぶ。
  無ければ最も単純な候補。一律0.3%と銘柄ごとのコストで同じモデルになることを確かめる）。拡大窓（EXP-005）は採用候補にしない
- リリース（release-vN）の比較は、フェーズ6でリリースを作るときに加える
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.backtest.metrics import t_stat
from src.config import ROOT, find_experiment_dir

MAIN_COSTS = ("cost_0.3%", "cost_tick")
T_MIN = 2.0
DEFAULT_ORDER = ["EXP-002", "EXP-001", "EXP-004"]   # 単純な順（リッジ → LightGBM の回帰 → lambdarank）


def load(exp_id: str, log_root: Path | None = None) -> tuple[dict, pd.DataFrame]:
    metrics = json.loads((find_experiment_dir(exp_id) / "metrics.json").read_text(encoding="utf-8"))
    weekly = pd.read_csv((log_root or ROOT / "logs" / "backtest") / exp_id / "weekly_returns.csv")
    return metrics, weekly


def diff_t(a: pd.DataFrame, b: pd.DataFrame, col: str) -> float:
    if list(a["last"]) != list(b["last"]):
        raise ValueError("週が違う実験は比べられません")
    return t_stat(a[col] - b[col])


def select_model(order: list[str], metrics: dict[str, dict], weekly: dict[str, pd.DataFrame],
                 t_min: float = T_MIN) -> dict:
    """README 第4章 3。order は単純な順。戻り値の selected が None なら決められない（候補なし、またはコストで結論が違う）。"""
    excluded = {}
    cands = []
    for e in order:
        mt = metrics[e]
        if mt["meta"]["config"]["validation"]["train_window"] == "expanding":
            excluded[e] = "拡大窓は採用候補にしない（README 第4章）"
        elif not mt.get("model_candidate", {}).get("candidate"):
            excluded[e] = "採用候補の条件を満たさない"
        else:
            cands.append(e)
    out: dict = {"order": order, "candidates": cands, "excluded": excluded, "t_min": t_min, "by_cost": {}}
    if not cands:
        out["selected"] = None
        out["decision"] = "採用候補なし：売買ルールの候補に進まずに止めて報告する（README 第4章 4）"
        return out
    for cost in MAIN_COSTS:
        col = f"fixed|{cost}"
        steps = []
        choice = None
        for k in range(len(cands) - 1, -1, -1):
            c = cands[k]
            ts = {s: diff_t(weekly[c], weekly[s], col) for s in cands[:k]}
            ok = all(t == t and t >= t_min for t in ts.values())
            steps.append({"model": c, "t_vs_simpler": ts, "chosen": ok})
            if ok:
                choice = c
                break
        out["by_cost"][cost] = {"selected": choice, "steps": steps}
    picks = {v["selected"] for v in out["by_cost"].values()}
    if len(picks) == 1:
        out["selected"] = picks.pop()
        out["decision"] = f"{out['selected']} に決める（一律0.3%と銘柄ごとのコストで同じ）"
    else:
        out["selected"] = None
        out["decision"] = "一律0.3%と銘柄ごとのコストで選ばれるモデルが違う：決めずに止めてユーザーに相談する（README 第4章 3）"
    return out


def comparison_table(ids: list[str], metrics: dict[str, dict], weekly: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    ref = ids[0]
    for e in ids:
        mt = metrics[e]
        if not mt.get("executed", True):
            rows.append({"実験": e, "備考": "実行しなかった"})
            continue
        row = {"実験": e, "Rank IC": mt["rank_ic"]["mean"], "IC t値": mt["rank_ic"]["t"]}
        for c in MAIN_COSTS:
            s = mt["backtest"]["fixed"][c]
            row[f"年率の超過（固定・{c}）"] = s["annual_excess_vs_universe"]
            row[f"シャープ（固定・{c}）"] = s["sharpe"]
        row["最大DD（固定・0.3%）"] = mt["backtest"]["fixed"]["cost_0.3%"]["max_drawdown"]
        row["損益分岐の片道コスト"] = mt["backtest"]["fixed"]["cost_0.3%"].get("breakeven_one_way_cost_vs_universe")
        row["年率の超過（基本ルール・0.3%）"] = mt["backtest"]["min_equity"]["cost_0.3%"]["annual_excess_vs_universe"]
        if e != ref:
            for c in MAIN_COSTS:
                row[f"{ref} との週次の差の t値（{c}）"] = diff_t(weekly[e], weekly[ref], f"fixed|{c}")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--select-model", action="store_true")
    args = ap.parse_args()
    ids = args.ids or (DEFAULT_ORDER if args.select_model else [])
    if not ids:
        ap.error("実験IDを指定してください")
    loaded = {e: load(e) for e in ids}
    metrics = {e: v[0] for e, v in loaded.items()}
    weekly = {e: v[1] for e, v in loaded.items()}
    periods = {tuple(mt["meta"]["period"]) for mt in metrics.values()}
    if len(periods) != 1:
        raise SystemExit(f"期間が違う実験は比べられません: {periods}")
    if args.select_model:
        print(json.dumps(select_model(ids, metrics, weekly), ensure_ascii=False, indent=1, default=float))
    else:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(comparison_table(ids, metrics, weekly).to_string(index=False))


if __name__ == "__main__":
    main()
