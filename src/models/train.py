"""モデルの学習と予測（フェーズ4。config/base.yaml の model。パラメータは固定で、結果を見て調整しない）。

- lightgbm・regression：目的変数（ユニバース内の順位 0〜1）の二乗誤差（EXP-001）
- lightgbm・lambdarank：起点の日ごとを1つのグループとし、目的変数を10分位の整数（0〜9）に直したものをラベルにする（EXP-004）
- ridge：特徴量の欠損を nan_fill（0.5）で埋めてリッジ回帰（EXP-002）
- 乱数：LightGBM の seed は project.random_seed、deterministic、スレッド数は固定（同じ設定なら同じ結果）
- target_shuffle：学習用の目的変数を、起点の日ごとに（同じ日の銘柄の間で）乱数で並べ替える（EXP-003。情報漏れの確認）
"""
from __future__ import annotations

import numpy as np

MODEL_TYPES = ("lightgbm", "ridge")
OBJECTIVES = ("regression", "lambdarank")
N_LABELS = 10


def decile_labels(y: np.ndarray) -> np.ndarray:
    """順位（0〜1。上が大きい）→ 10分位の整数（0〜9）。(0, 0.1] → 0、…、(0.9, 1] → 9。"""
    return np.clip(np.ceil(y * N_LABELS) - 1, 0, N_LABELS - 1).astype(int)


def shuffle_within_day(y: np.ndarray, day: np.ndarray, seed: int) -> np.ndarray:
    """同じ日の行の間で y を並べ替える（NaN の行は動かさない）。"""
    out = y.copy()
    rng = np.random.default_rng(seed)
    ok = ~np.isnan(y)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return out
    d = day[idx]
    order = np.argsort(d, kind="stable")
    idx, d = idx[order], d[order]
    bounds = np.flatnonzero(np.diff(d)) + 1
    for g in np.split(idx, bounds):
        out[g] = y[rng.permutation(g)]
    return out


def lgb_params(mcfg: dict, seed: int) -> dict:
    p = dict(mcfg["lightgbm"])
    p.pop("n_estimators")
    trunc = p.pop("lambdarank_truncation_level")
    p.update({"seed": seed, "verbosity": -1, "force_col_wise": True})
    if mcfg["objective"] == "lambdarank":
        p.update({"objective": "lambdarank", "metric": "ndcg", "lambdarank_truncation_level": trunc})
    else:
        p.update({"objective": "regression", "metric": "l2"})
    return p


class Model:
    def __init__(self, mcfg: dict, seed: int):
        if mcfg["type"] not in MODEL_TYPES:
            raise ValueError(f"model.type は {MODEL_TYPES} のどれか: {mcfg['type']}")
        if mcfg["type"] == "lightgbm" and mcfg["objective"] not in OBJECTIVES:
            raise ValueError(f"model.objective は {OBJECTIVES} のどれか: {mcfg['objective']}")
        self.mcfg = mcfg
        self.seed = seed
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, day: np.ndarray) -> "Model":
        """day：各行の起点の日（lambdarank のグループ。行は day の順に並んでいること）。"""
        if self.mcfg["type"] == "ridge":
            from sklearn.linear_model import Ridge
            r = self.mcfg["ridge"]
            self.model = Ridge(alpha=r["alpha"]).fit(np.where(np.isnan(X), r["nan_fill"], X), y)
            return self
        import lightgbm as lgb
        params = lgb_params(self.mcfg, self.seed)
        if self.mcfg["objective"] == "lambdarank":
            if np.any(np.diff(day) < 0):
                raise ValueError("lambdarank では行を起点の日の順に並べる")
            _, sizes = np.unique(day, return_counts=True)
            ds = lgb.Dataset(X, decile_labels(y), group=sizes, free_raw_data=True)
        else:
            ds = lgb.Dataset(X, y, free_raw_data=True)
        self.model = lgb.train(params, ds, num_boost_round=int(self.mcfg["lightgbm"]["n_estimators"]))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.mcfg["type"] == "ridge":
            return self.model.predict(np.where(np.isnan(X), self.mcfg["ridge"]["nan_fill"], X))
        return self.model.predict(X, num_threads=int(self.mcfg["lightgbm"]["num_threads"]))
