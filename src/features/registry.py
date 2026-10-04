"""特徴量の登録簿（CLAUDE.md 第8章）。

- 特徴量は名前で登録する。同じ名前の二重登録はエラーにする（既存の特徴量の計算方法を上書きしない）
- 計算方法を変えるときは、別の名前で新しく登録する（例：ret_20d → ret_20d_v2）。古い実験を同じ結果で再現するため
- 各特徴量は FeatureData を受け取り、[営業日, 銘柄] の配列を返す。位置 t の値は、t の引け後の予測の時点で
  入手できた情報だけで決まる（tests/test_features.py で確認する）
- 特徴量の組（FEATURE_SETS）も名前で管理し、追加のみ行う
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from src.features.data import FeatureData

FeatureFn = Callable[["FeatureData"], np.ndarray]


@dataclass(frozen=True)
class Feature:
    name: str
    group: str            # "price"（値動き）/ "fundamental"（財務・イベント）
    description: str
    func: FeatureFn

    def source_hash(self) -> str:
        """計算の関数のソースのハッシュ（保存した計算結果が古くなっていないかの確認用）。"""
        return hashlib.sha256(inspect.getsource(self.func).encode("utf-8")).hexdigest()[:16]


REGISTRY: dict[str, Feature] = {}


def register(name: str, group: str, description: str) -> Callable[[FeatureFn], FeatureFn]:
    def deco(func: FeatureFn) -> FeatureFn:
        if name in REGISTRY:
            raise ValueError(f"特徴量 {name} は登録済みです（上書きせず、別の名前で登録してください）")
        REGISTRY[name] = Feature(name, group, description, func)
        return func
    return deco


def get(name: str) -> Feature:
    _load_all()
    if name not in REGISTRY:
        raise KeyError(f"未登録の特徴量: {name}")
    return REGISTRY[name]


def _load_all() -> None:
    # 登録はモジュールの読み込み時に行われる
    import src.features.fundamental  # noqa: F401
    import src.features.price  # noqa: F401


# 特徴量の組（追加のみ。既存の組の中身は変えない）
FEATURE_SETS: dict[str, list[str]] = {
    # 第1段階（値動き）
    "price_v1": ["ret_5d", "ret_20d", "ret_60d", "vol_ratio_5_60", "volatility_20d", "dist_high_60d",
                 "log_turnover_20d"],
    # 第2段階（財務・イベント）
    "fundamental_v1": ["log_mcap", "ep_fcst", "bp", "sales_growth_yoy", "op_chg_to_sales_yoy",
                       "bdays_since_report", "op_surprise_fy", "op_fcst_rev", "cdays_to_next_earnings"],
}
FEATURE_SETS["all_v1"] = FEATURE_SETS["price_v1"] + FEATURE_SETS["fundamental_v1"]


def resolve(names_or_set: str | list[str]) -> list[str]:
    if isinstance(names_or_set, str):
        return list(FEATURE_SETS[names_or_set])
    return list(names_or_set)
