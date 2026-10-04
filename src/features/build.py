"""特徴量の計算と保存（data/processed/features/）。

- 保存先：data/processed/features/<Market のキャッシュ名>/<特徴量名>.npy と、計算の関数のハッシュ（.json）
- 関数のソースが変わっていたら（ハッシュが違えば）計算し直す。ただし、登録済みの特徴量の計算方法は変えない決まり
  （registry.py）なので、通常はバグの修正のときだけ起きる
- 特徴量の組の名前（例：all_v1）か、特徴量の名前を指定する

実行：python -m src.features.build --set all_v1
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src.backtest.market import Market, cache_path, load_market
from src.config import load_config
from src.features.data import FeatureData, load_feature_data
from src.features.registry import FEATURE_SETS, get, resolve


def feature_dir(cfg: dict):
    p = cache_path(cfg)
    return p.parent / "features" / p.stem


def compute(fd: FeatureData, names: list[str]) -> dict[str, np.ndarray]:
    return {n: get(n).func(fd).astype(np.float64) for n in names}


def load_features(names: list[str] | str, cfg: dict | None = None, m: Market | None = None,
                  rebuild: bool = False, verbose: bool = False) -> dict[str, np.ndarray]:
    cfg = cfg or load_config()
    names = resolve(names)
    m = m or load_market(cfg)
    d = feature_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    out: dict[str, np.ndarray] = {}
    fd: FeatureData | None = None
    for n in names:
        f = get(n)
        npy, meta = d / f"{n}.npy", d / f"{n}.json"
        if not rebuild and npy.exists() and meta.exists() and \
                json.loads(meta.read_text(encoding="utf-8")).get("source_hash") == f.source_hash():
            out[n] = np.load(npy)
            continue
        if fd is None:
            fd = load_feature_data(m, cfg)
        t0 = time.time()
        arr = f.func(fd).astype(np.float64)
        if arr.shape != (len(m.dates), len(m.codes)):
            raise ValueError(f"{n} の形が違います: {arr.shape}")
        tmp = d / f"{n}.tmp.npy"
        np.save(tmp, arr)
        tmp.replace(npy)
        meta.write_text(json.dumps({"name": n, "group": f.group, "description": f.description,
                                    "source_hash": f.source_hash(), "market": m.meta}, ensure_ascii=False),
                        encoding="utf-8")
        if verbose:
            print(f"{n}: {time.time() - t0:.1f}秒")
        out[n] = arr
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=None, choices=sorted(FEATURE_SETS))
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    names = args.names or resolve(args.set or "all_v1")
    feats = load_features(names, rebuild=args.rebuild, verbose=True)
    for n, a in feats.items():
        print(f"{n}: 値のある割合 {np.isfinite(a).mean():.3f}")


if __name__ == "__main__":
    main()
