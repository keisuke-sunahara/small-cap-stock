"""フェーズ1：品質チェックで見つかった例外の中身の確認（売買が成立しない銘柄、±50%を超える日次リターン、予定日が公表日より前の記録）。

ホールドアウトより前のデータだけを読む。結果は画面に出すだけで保存しない（reports/phase1.md にまとめた）。

実行: python -m src.analysis.phase1_anomalies
"""
import pandas as pd
from src.config import load_config
from src.data.load import read_raw
from src.analysis.phase1_quality import adjusted_returns, is_common

cfg = load_config()
master = read_raw(cfg, "master", columns=["Code", "CoName", "Mkt", "MktNm", "ProdCat"])
master["common"] = is_common(master)
bars = read_raw(cfg, "bars", columns=["Code", "O", "H", "L", "C", "Vo", "Va", "UL", "LL", "AdjFactor"])
bars = bars.merge(master[["Date", "Code", "common", "ProdCat", "CoName", "MktNm"]], on=["Date", "Code"], how="left")

# 1. 一度も売買が成立しなかった・長期の不成立
traded_days = bars.groupby("Code")["C"].count()
total_days = bars.groupby("Code").size()
never = traded_days[traded_days == 0].index
info = bars.drop_duplicates("Code", keep="last").set_index("Code")
print("== 一度も成立しない銘柄 ==")
print(info.loc[never, ["CoName", "ProdCat", "MktNm", "common"]].to_string())
ratio = (traded_days / total_days)
low = ratio[ratio < 0.5].index
print("成立率50%未満の銘柄:", len(low), " うち普通株:", int(info.loc[low, "common"].fillna(False).sum()))
print(info.loc[low][info.loc[low, "common"].fillna(False)][["CoName", "MktNm"]].head(15).to_string())

# 2. 普通株の売買不成立の割合
cs = bars[bars["common"].fillna(False)]
print("== 普通株の不成立割合 ==", round(cs["C"].isna().mean(), 4))
print("年別:", cs.assign(y=cs["Date"].str[:4]).groupby("y")["C"].apply(lambda s: round(s.isna().mean(), 4)).to_dict())

# 3. 調整後の日次リターンが ±50% を超える日
t = adjusted_returns(bars[["Date", "Code", "C", "AdjFactor"]])
t = t.merge(bars[["Date", "Code", "common", "CoName", "UL", "LL"]], on=["Date", "Code"], how="left")
t["prevDate"] = t.groupby("Code")["Date"].shift(1)
ext = t[t["ret_adj"].abs() > 0.5].copy()
bd = sorted(bars["Date"].unique())
pos = {d: i for i, d in enumerate(bd)}
ext["gap_bdays"] = [pos[d] - pos[p] if isinstance(p, str) else None for d, p in zip(ext["Date"], ext["prevDate"])]
print("== ±50%超 ==", len(ext), " 普通株:", int(ext["common"].fillna(False).sum()))
print("前回の約定から2営業日以上空いた:", int((ext["gap_bdays"] > 1).sum()))
print("上昇/下落:", int((ext["ret_adj"] > 0).sum()), int((ext["ret_adj"] < 0).sum()))
print("UL/LLフラグあり:", int(((ext["UL"].astype(str) == "1") | (ext["LL"].astype(str) == "1")).sum()))
c1 = ext[(ext["gap_bdays"] == 1)].sort_values("ret_adj")
print(c1[["Date", "Code", "CoName", "ret_adj", "common"]].head(8).to_string())
print(c1[["Date", "Code", "CoName", "ret_adj", "common"]].tail(8).to_string())
# 前日比の制限値幅を超える動き（翌日の値幅制限を超える = 調整漏れの疑い）: 連続取引で -50% 未満
sus = c1[c1["ret_adj"] < -0.5]
print("連続取引で-50%未満:", len(sus))

# 4. 予定日が公表日より前
e = read_raw(cfg, "earnings_date")
f = e[(e["SchDate"].astype(str) != "") & (e["SchDate"] < e["PubDate"])]
print("== SchDate < PubDate ==")
print(f[["PubDate", "SchDate", "Code", "CoName", "FQName"]].to_string())

