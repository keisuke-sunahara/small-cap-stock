"""設定の読み込み。config/base.yaml に実験ごとの差分（experiments/EXP-xxx_*/config.yaml）を重ねる。"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "config" / "base.yaml"
EXPERIMENTS_DIR = ROOT / "experiments"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """base に override を再帰的に重ねた新しい dict を返す（どちらも変更しない）。"""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_experiment_dir(exp_id: str) -> Path:
    matches = sorted(EXPERIMENTS_DIR.glob(f"{exp_id}_*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"実験ディレクトリが一意に見つかりません: {exp_id} -> {matches}")
    return matches[0]


def load_config(exp_id: str | None = None, base_path: Path = BASE_CONFIG) -> dict[str, Any]:
    config = _read_yaml(base_path)
    if exp_id is not None:
        exp_config = find_experiment_dir(exp_id) / "config.yaml"
        config = deep_merge(config, _read_yaml(exp_config))
    return config
