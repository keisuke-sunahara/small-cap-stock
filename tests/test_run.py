"""実験の実行（src/run.py）と比較（src/compare.py）の通しのテスト（ダミーデータ。本物の成績ではない）。"""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import compare
from src.backtest import evaluate as ev
from src.backtest.universe import avg_turnover, universe_mask
from src.config import deep_merge
from src.run import Inputs, experiment_config, needs_training, run_experiment
from tests.test_models import small_cfg
from tests.dummy_market import dummy_dates, dummy_market


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    rng = np.random.default_rng(5)
    T, N = 330, 40
    dates = dummy_dates("2023-01-02", T)
    close = {f"{1000 + k}0": list(rng.choice([150, 400, 900, 2500]) * np.exp(np.cumsum(rng.normal(0, 0.025, T))))
             for k in range(N)}
    m = dummy_market(dates, close)
    m.O[1:] = m.C[:-1] * np.exp(rng.normal(0, 0.01, (T - 1, N)))
    m.H = np.fmax(m.O, m.C) * 1.01
    m.L = np.fmin(m.O, m.C) * 0.97
    m.topix = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, T)))    # 地合いフィルターが働く週があるように
    m.__post_init__()
    cfg = small_cfg(m.dates)
    adv = avg_turnover(m, cfg["universe"]["turnover_window"])
    u = universe_mask(m, cfg, adv)
    lr = np.vstack([np.zeros((1, N)), np.diff(np.log(m.C), axis=0)])
    feats = {"f1": rng.normal(size=(T, N)), "f2": pd.DataFrame(lr).rolling(5).sum().to_numpy()}
    vol = pd.DataFrame(lr).rolling(20).std().to_numpy()
    # 全銘柄の決算発表予定日を dates[262] にする（240 から使える）→ その週は買わず、前の週の保有は継続しない
    sched = pd.DataFrame({"j": np.arange(N), "avail": 240, "PubDate": dates[239], "FYE": "2024-03-31", "FQName": "3Q",
                          "SchDate": pd.to_datetime(str(dates[262]))})
    fins = pd.DataFrame({"j": [0], "avail": [250], "DocType": ["3QFinancialStatements"]})
    fd = SimpleNamespace(m=m, fins=fins, sched=sched)
    inp = Inputs(m=m, u=u, adv=adv, feats=feats, bad_event=None, fd=fd, vol=vol)
    root = tmp_path_factory.mktemp("run")
    ctx = ev.make_context(m, u, adv, cfg)
    rep = {"meta": {"period": [ctx.start_pred, ctx.week_last[-1]], "weeks": len(ctx.weeks), "git_commit": "dummy"},
           "baselines": {}, "random": {}, "universe_average": {}}
    for s in ("momentum_20d", "reversal_5d"):
        rep["baselines"][s] = {b: {c: {"annual_excess_vs_universe": -0.1} for c in ev.COST_NAMES}
                               for b in ("fixed", "min_equity")}
    rep["random"] = {b: {c: {"runs": 1000, "mean": {"annual_excess_vs_universe": -0.2}} for c in ev.COST_NAMES}
                     for b in ("fixed", "min_equity")}
    rep["universe_average"] = {c: {"annual_excess_vs_universe": -0.01, "annual_return": 0.0} for c in ev.COST_NAMES}
    (root / "baselines.json").write_text(json.dumps(rep), encoding="utf-8")
    cfg["evaluation"]["baselines_report"] = str(root / "baselines.json")
    return SimpleNamespace(m=m, cfg=cfg, inp=inp, root=root)


def run(env, exp_id, cfg, model_cfg=None, model_exp=None):
    d = env.root / exp_id
    d.mkdir(exist_ok=True)
    return run_experiment(exp_id, cfg, model_cfg, model_exp, env.inp, exp_dir=d, log_root=env.root / "logs",
                          base_cfg=env.cfg, random_path=env.root / "random.csv", random_runs=3, log=None)


@pytest.fixture(scope="module")
def model_run(env):
    return run(env, "EXP-T01", env.cfg)


def test_model_experiment_outputs(env, model_run):
    mt = model_run
    logs = env.root / "logs" / "EXP-T01"
    for f in ("scores.parquet", "splits.csv", "weekly_returns.csv", "weekly.csv", "orders.csv", "trades.csv"):
        assert (logs / f).exists(), f
    assert (env.root / "EXP-T01" / "metrics.json").exists() and (env.root / "EXP-T01" / "results.md").exists()
    saved = json.loads((env.root / "EXP-T01" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["meta"]["weeks"] == len(ev.eval_weeks(env.m, env.cfg))
    for b in ("fixed", "min_equity"):
        for c in ev.COST_NAMES:
            assert "annual_excess_vs_universe" in saved["backtest"][b][c]
    assert "breakeven_one_way_cost_vs_universe" in saved["backtest"]["fixed"]["cost_0.3%"]
    assert set(saved["model_candidate"]) >= {"rank_ic_ok", "beats_baselines", "candidate"}
    assert set(saved["criteria"]["fixed"]["cost_tick"]) >= {"excess_positive", "after_all_costs_positive"}
    assert len(saved["deciles_mean_weekly_return"]) == 10
    sc = pd.read_parquet(logs / "scores.parquet")
    assert list(sc.columns) == ["date", "code", "kind", "score"]
    wr = pd.read_csv(logs / "weekly_returns.csv")
    assert {"fixed|cost_0.3%", "fixed|cost_tick", "min_equity|no_cost"} <= set(wr.columns)
    # 上書きしない
    with pytest.raises(SystemExit):
        run(env, "EXP-T01", env.cfg)


def test_fixed_budget_cost_zero_vs_flat(env, model_run):
    """コストなし ≥ 一律0.3% ≥ 一律0.5%（予算固定では注文がコストによらない）。"""
    b = model_run["backtest"]["fixed"]
    assert b["no_cost"]["annual_return"] >= b["cost_0.3%"]["annual_return"] >= b["cost_0.5%"]["annual_return"]


def test_leak_check_experiment(env):
    cfg = deep_merge(env.cfg, {"model": {"target_shuffle": True}})
    mt = run(env, "EXP-T03", cfg)
    assert set(mt["leak_check"]) >= {"ic_ok", "random_ok", "no_leak_found"}


@pytest.mark.parametrize("exp_id,diff,key", [
    ("EXP-T06", {"rules": {"market_filter_ma_days": 20}}, None),
    ("EXP-T07", {"rules": {"stop_loss": 0.08}}, "stop_loss"),
    ("EXP-T08", {"rules": {"earnings_avoid": True}}, None),
    ("EXP-T09", {"rules": {"buy_order": "market"}}, None),
    ("EXP-T10", {"capital": {"n_holdings": 2}}, None),
    ("EXP-T13", {"rules": {"weighting": "inverse_vol"}}, "precheck_rank_trend"),
    ("EXP-T15", {"rules": {"holding_weeks": 2}}, None),
])
def test_rule_experiments(env, model_run, exp_id, diff, key):
    cfg = deep_merge(env.cfg, {"model_experiment": "EXP-T01", **diff})
    assert not needs_training(cfg, env.cfg)
    mt = run(env, exp_id, cfg, env.cfg, "EXP-T01")
    assert mt["executed"]
    ra = mt["rule_adoption"]
    assert set(ra) >= {"better_excess", "weekly_diff_t", "mdd_ok", "adopt"}
    assert ra["t_min"] == 2.0
    if key:
        assert key in mt
    assert not (env.root / "logs" / exp_id / "scores.parquet").exists()   # 点数はモデルの実験のものを使う
    if exp_id in ("EXP-T06", "EXP-T08"):
        w = pd.read_csv(env.root / "logs" / exp_id / "weekly.csv", keep_default_na=False)
        if exp_id == "EXP-T06":
            assert (w["skip_reason"] == "market_filter").any()
        else:
            hit = w[(w["first"] <= str(env.m.dates[262])) & (w["last"] >= str(env.m.dates[262]))]
            assert hit["n_orders"].tolist() == [0]
            base = pd.read_csv(env.root / "logs" / "EXP-T01" / "weekly.csv")
            assert base.set_index("first").loc[hit["first"].iloc[0], "n_orders"] > 0
    if exp_id == "EXP-T10":
        assert ra["adopt"] is None                                          # 候補5は参考
    if exp_id == "EXP-T15":
        fixed = mt["backtest"]["fixed"]["cost_0.3%"]
        assert len(fixed["offsets"]) == 2 and "best_offset" in fixed
        wr = pd.read_csv(env.root / "logs" / exp_id / "weekly_returns.csv")
        assert np.allclose(wr["fixed|cost_0.3%"], (wr["fixed|cost_0.3%|o0"] + wr["fixed|cost_0.3%|o1"]) / 2)


def test_rank_weighting_precheck(env, model_run):
    cfg = deep_merge(env.cfg, {"model_experiment": "EXP-T01", "rules": {"weighting": "rank"}})
    mt = run(env, "EXP-T12", cfg, env.cfg, "EXP-T01")
    pre = mt["precheck_rank_trend"]
    if pre["trend"]:
        assert mt["executed"] and "rule_adoption" in mt
    else:
        assert not mt["executed"] and "backtest" not in mt
        assert "実行しない" in (env.root / "EXP-T12" / "results.md").read_text(encoding="utf-8")


def test_affordable_experiment_retrains(env, model_run):
    cfg = deep_merge(env.cfg, {"model_experiment": "EXP-T01", "model": {"train_universe": "affordable"}})
    assert needs_training(cfg, env.cfg)
    mt = run(env, "EXP-T14", cfg, env.cfg, "EXP-T01")
    assert (env.root / "logs" / "EXP-T14" / "scores.parquet").exists()
    assert set(mt["rank_ic_affordable"]) == {"this", "EXP-T01"}
    assert "rule_adoption" in mt


def test_select_model_rule():
    last = [str(i) for i in range(100)]
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.02, 100)

    def wk(shift):
        return pd.DataFrame({"last": last, "fixed|cost_0.3%": base + shift, "fixed|cost_tick": base + shift})

    def mt(cand, window="rolling"):
        return {"model_candidate": {"candidate": cand}, "meta": {"config": {"validation": {"train_window": window}}}}

    noise = rng.normal(0, 0.001, 100)
    weekly = {"R": wk(0.0), "L": wk(0.0005 + noise), "K": wk(0.0002 + noise)}
    metrics = {"R": mt(True), "L": mt(True), "K": mt(True)}
    # L はリッジを明確に上回る（t 値が大きい）。K は L を上回らない → L
    out = compare.select_model(["R", "L", "K"], metrics, weekly)
    assert out["selected"] == "L"
    # K がリッジと回帰の両方を明確に上回るときだけ K
    weekly["K"] = wk(0.002 + noise)
    assert compare.select_model(["R", "L", "K"], metrics, weekly)["selected"] == "K"
    # 差がはっきりしなければ最も単純な候補
    weekly2 = {"R": wk(0.0), "L": wk(rng.normal(0, 0.01, 100)), "K": wk(rng.normal(0, 0.01, 100))}
    assert compare.select_model(["R", "L", "K"], metrics, weekly2)["selected"] == "R"
    # 拡大窓・採用候補でないものは除く。候補が無ければ None
    metrics2 = {"R": mt(False), "L": mt(True, "expanding"), "K": mt(False)}
    out2 = compare.select_model(["R", "L", "K"], metrics2, weekly)
    assert out2["selected"] is None and out2["candidates"] == []
    # コストで結論が違えば決めない
    w3 = {k: v.copy() for k, v in weekly.items()}
    w3["K"]["fixed|cost_tick"] = base
    assert compare.select_model(["R", "L", "K"], metrics, w3)["selected"] is None


def test_experiment_config_reads_registered_experiments():
    cfg, model_cfg, me = experiment_config("EXP-002")
    assert cfg["model"]["type"] == "ridge" and model_cfg is None and me is None
    with pytest.raises(SystemExit):
        experiment_config("EXP-006")        # model_experiment が未記入（モデルの確定の後に書く）
