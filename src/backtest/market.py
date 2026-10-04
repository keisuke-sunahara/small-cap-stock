"""バックテスト・特徴量で使う「営業日 × 銘柄」の行列。

data/raw から作り、data/processed/ に保存して再利用する（ホールドアウト期間は読まない）。
- 対象の銘柄：期間中に一度でも普通株だった銘柄（判定は src.analysis.phase1_quality.is_common）
- 株価は調整前の実際の値（O・H・L・C）。分割・併合の調整は cumF（調整係数の累積積）で行う
  （その日までの係数だけを使うため、後の分割の情報を含まない。DECISIONS.md 2026-10-04）
- 発行済株式数は、開示の利用開始日（src.data.disclosure）から使う。期末日の後の分割は cumF で直す
- 貸借信用区分が「その他（3）」の印（margin_other）は、日付 t の銘柄一覧の値。日付 t の一覧は t の前の営業日の
  17:30 頃に公開されるので、t の引け後には確実に分かる（ユニバースからの除外に使う。CLAUDE.md 第5章、2026-10-04 承認）

実行（作り直し）: python -m src.backtest.market --rebuild
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.analysis.phase1_quality import is_common
from src.config import ROOT, load_config
from src.data.disclosure import available_index
from src.data.fetch import business_days, raw_dir
from src.data.load import read_raw

JST = ZoneInfo("Asia/Tokyo")
CACHE_VERSION = "v2"   # v2：margin_other を追加（2026-10-04）
ARRAYS = ["O", "H", "L", "C", "Va", "adj", "UL", "LL", "common", "shares_base", "margin_other"]


@dataclass
class Market:
    dates: np.ndarray          # 営業日（"YYYY-MM-DD"）[T]
    codes: np.ndarray          # 銘柄コード [N]
    O: np.ndarray              # 始値（売買不成立・上場していない日は NaN）[T, N]
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    Va: np.ndarray             # 売買代金（売買不成立は 0、上場していない日は NaN）
    adj: np.ndarray            # 調整係数（その日の朝に効く分割・併合。無ければ 1）
    UL: np.ndarray             # ストップ高のフラグ（bool）
    LL: np.ndarray             # ストップ安のフラグ（bool）
    common: np.ndarray         # その日に普通株として上場している（bool）
    shares_base: np.ndarray    # 発行済株式数 × cumF（開示の利用開始日から。cumF で割ると当日の株数）
    last_listed: np.ndarray    # 銘柄一覧に最後に載っていた日の位置 [N]（データの最終日なら T-1）
    topix: np.ndarray          # TOPIX の終値 [T]
    margin_other: np.ndarray | None = None  # その日の銘柄一覧で貸借信用区分が「その他（3）」（bool）
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.margin_other is None:
            self.margin_other = np.zeros(self.C.shape, dtype=bool)
        self.cumF = np.cumprod(self.adj, axis=0)
        self.date_index = {d: i for i, d in enumerate(self.dates)}
        self.code_index = {c: j for j, c in enumerate(self.codes)}

    @property
    def Qc(self) -> np.ndarray:
        """分割調整した終値（売買不成立は NaN）。"""
        return self.C / self.cumF

    @property
    def Qo(self) -> np.ndarray:
        return self.O / self.cumF

    @property
    def Qc_ff(self) -> np.ndarray:
        """分割調整した終値を、直前に売買が成立した日の値で埋めたもの（上場廃止後も最後の値のまま）。"""
        return pd.DataFrame(self.Qc).ffill().to_numpy()

    @property
    def C_ff(self) -> np.ndarray:
        """調整前の終値を直前の約定で埋めたもの（評価額の計算用。分割は株数の側で直す）。"""
        return pd.DataFrame(self.C).ffill().to_numpy()

    @property
    def shares(self) -> np.ndarray:
        return self.shares_base / self.cumF

    def truncated(self, last: int) -> "Market":
        """位置 last までのデータだけを残した Market（未来情報の混入テスト用）。"""
        kw = {a: getattr(self, a)[: last + 1].copy() for a in ARRAYS}
        return Market(dates=self.dates[: last + 1], codes=self.codes,
                      last_listed=np.minimum(self.last_listed, last), topix=self.topix[: last + 1],
                      meta=dict(self.meta), **kw)


def _period(cfg: dict) -> tuple[str, str]:
    start = cfg["data"]["start_date"]
    end = (date.fromisoformat(cfg["data"]["holdout_start"]) - timedelta(days=1)).isoformat()
    return start, end


def _latest_fetch(cfg: dict) -> str:
    """data/raw の取得日時のうち最も新しいもの（metrics.json の「データの取得日」）。"""
    latest = ""
    for log in (ROOT / cfg["data"]["raw_dir"]).glob("*/_fetch_log.csv"):
        s = pd.read_csv(log)["fetched_at_jst"].max()
        latest = max(latest, str(s))
    return latest


def build_market(cfg: dict) -> Market:
    start, end = _period(cfg)
    cal = read_raw(cfg, "calendar")
    bdays = [d for d in business_days(cal) if start <= d <= end]
    di = {d: i for i, d in enumerate(bdays)}
    T = len(bdays)

    master = read_raw(cfg, "master", columns=["Code", "Mkt", "ProdCat", "Mrgn"])
    master["Code"] = master["Code"].astype(str)
    master["is_common"] = is_common(master)
    codes = np.array(sorted(master.loc[master["is_common"], "Code"].unique()))
    ci = {c: j for j, c in enumerate(codes)}
    N = len(codes)
    master = master[master["Code"].isin(ci)]
    mi = master["Date"].map(di).to_numpy()
    mj = master["Code"].map(ci).to_numpy()
    common = np.zeros((T, N), dtype=bool)
    common[mi[master["is_common"].to_numpy()], mj[master["is_common"].to_numpy()]] = True
    margin_other = np.zeros((T, N), dtype=bool)
    margin_other[mi, mj] = master["Mrgn"].astype(str).eq("3").to_numpy()
    last_listed = np.full(N, -1, dtype=np.int64)
    np.maximum.at(last_listed, mj, mi)

    bars = read_raw(cfg, "bars", columns=["Code", "O", "H", "L", "C", "Va", "UL", "LL", "AdjFactor"])
    bars["Code"] = bars["Code"].astype(str)
    bars = bars[bars["Code"].isin(ci)]
    bi = bars["Date"].map(di).to_numpy()
    bj = bars["Code"].map(ci).to_numpy()
    arr = {}
    for col in ["O", "H", "L", "C"]:
        a = np.full((T, N), np.nan)
        a[bi, bj] = bars[col].to_numpy(dtype=float)
        arr[col] = a
    va = np.full((T, N), np.nan)
    va[bi, bj] = bars["Va"].fillna(0.0).to_numpy(dtype=float)  # 売買不成立の日は売買代金0
    adj = np.ones((T, N))
    adj[bi, bj] = bars["AdjFactor"].fillna(1.0).to_numpy(dtype=float)
    ul = np.zeros((T, N), dtype=bool)
    ul[bi, bj] = bars["UL"].astype(str).eq("1").to_numpy()
    ll = np.zeros((T, N), dtype=bool)
    ll[bi, bj] = bars["LL"].astype(str).eq("1").to_numpy()
    del bars
    cumF = np.cumprod(adj, axis=0)

    s = read_raw(cfg, "summary", columns=["Code", "DiscTime", "ShOutFY", "CurPerEn"])
    s["Code"] = s["Code"].astype(str)
    s = s[s["Code"].isin(ci) & (s["ShOutFY"].astype(str).str.strip() != "")].copy()
    s["sh"] = pd.to_numeric(s["ShOutFY"], errors="coerce")
    s = s[s["sh"] > 0]
    s["avail"] = available_index(s["DiscDate"], s["DiscTime"], bdays, cfg)
    s = s[s["avail"] < T]
    # 株数は期末日時点の数。期末日（の直前の営業日）までの cumF を掛けて基準をそろえる
    ref = np.searchsorted(np.asarray(bdays), s["CurPerEn"].astype(str).to_numpy(), side="right") - 1
    s["ref"] = np.clip(ref, 0, T - 1)
    s["j"] = s["Code"].map(ci)
    s["base"] = s["sh"].to_numpy() * cumF[s["ref"].to_numpy(), s["j"].to_numpy()]
    s = s.sort_values(["avail", "DiscDate", "DiscTime"])  # 同じ日に使えるものは、後の開示を優先
    shares_base = np.full((T, N), np.nan)
    shares_base[s["avail"].to_numpy(), s["j"].to_numpy()] = s["base"].to_numpy()
    shares_base = pd.DataFrame(shares_base).ffill().to_numpy()

    topix = read_raw(cfg, "topix").set_index("Date")["C"].reindex(bdays).to_numpy(dtype=float)
    meta = {"built_at_jst": datetime.now(JST).isoformat(timespec="seconds"),
            "data_latest_fetch_jst": _latest_fetch(cfg), "period": [bdays[0], bdays[-1]],
            "cache_version": CACHE_VERSION}
    return Market(dates=np.array(bdays), codes=codes, O=arr["O"], H=arr["H"], L=arr["L"], C=arr["C"],
                  Va=va, adj=adj, UL=ul, LL=ll, common=common, shares_base=shares_base,
                  last_listed=last_listed, topix=topix, margin_other=margin_other, meta=meta)


def cache_path(cfg: dict) -> Path:
    start, end = _period(cfg)
    return ROOT / "data" / "processed" / f"market_{CACHE_VERSION}_{start}_{end}.npz"


def save_market(m: Market, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, dates=m.dates, codes=m.codes, last_listed=m.last_listed, topix=m.topix,
             meta=np.array(json.dumps(m.meta)), **{a: getattr(m, a) for a in ARRAYS})
    tmp.replace(path)


def load_market(cfg: dict | None = None, rebuild: bool = False) -> Market:
    cfg = cfg or load_config()
    path = cache_path(cfg)
    if rebuild or not path.exists():
        m = build_market(cfg)
        save_market(m, path)
        return m
    z = np.load(path, allow_pickle=False)
    return Market(dates=z["dates"], codes=z["codes"], last_listed=z["last_listed"], topix=z["topix"],
                  meta=json.loads(str(z["meta"])), **{a: z[a] for a in ARRAYS})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    m = load_market(rebuild=args.rebuild)
    print(f"営業日 {len(m.dates)}（{m.dates[0]}〜{m.dates[-1]}）、銘柄 {len(m.codes)}")
    print(json.dumps(m.meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
