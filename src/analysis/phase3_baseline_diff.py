"""株式数の補正の前（reports/phase2c_baselines.json）と後（reports/phase3r_baselines.json）のベースラインの比較表。

出力：reports/phase3r_vs_phase2c.json と、標準出力に Markdown の表
実行：python -m src.analysis.phase3_baseline_diff
"""
from __future__ import annotations

import json

from src.config import ROOT

COSTS = ["cost_0.3%", "cost_tick", "cost_0.5%", "no_cost"]
KEYS = ["annual_return", "annual_excess_vs_universe", "sharpe", "max_drawdown"]


def load(name: str) -> dict:
    return json.loads((ROOT / "reports" / f"{name}_baselines.json").read_text(encoding="utf-8"))


def rows(d: dict) -> dict:
    out = {}
    for b in ["momentum_20d", "reversal_5d"]:
        for mode in ["fixed", "min_equity"]:
            for c in COSTS:
                out[f"{b}/{mode}/{c}"] = {k: d["baselines"][b][mode][c][k] for k in KEYS}
    for mode in ["fixed", "min_equity"]:
        for c in COSTS:
            out[f"random_mean/{mode}/{c}"] = {k: d["random"][mode][c]["mean"].get(k) for k in KEYS}
    for c in COSTS:
        out[f"universe_average/{c}"] = {k: d["universe_average"][c][k] for k in KEYS}
    for b in ["momentum_20d", "reversal_5d"]:
        out[f"rank_ic/{b}"] = {"mean": d["rank_ic"][b]["mean"], "t": d["rank_ic"][b]["t"]}
    return out


def main() -> None:
    before, after = rows(load("phase2c")), rows(load("phase3r"))
    diff = {k: {m: {"before": before[k][m], "after": after[k][m]} for m in after[k]} for k in after}
    (ROOT / "reports" / "phase3r_vs_phase2c.json").write_text(json.dumps(diff, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
    print("| 系列 | 年率（前→後） | 年率の超過（前→後） | 最大DD（前→後） |")
    print("|---|---|---|---|")
    for k, v in diff.items():
        if k.startswith("rank_ic"):
            print(f"| {k} | Rank IC {v['mean']['before']:.4f} → {v['mean']['after']:.4f}（t {v['t']['before']:.2f} → "
                  f"{v['t']['after']:.2f}） | | |")
            continue
        f = lambda m: f"{v[m]['before'] * 100:.2f}% → {v[m]['after'] * 100:.2f}%"  # noqa: E731
        print(f"| {k} | {f('annual_return')} | {f('annual_excess_vs_universe')} | {f('max_drawdown')} |")


if __name__ == "__main__":
    main()
