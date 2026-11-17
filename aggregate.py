"""彙整多個獨立種子的結果，檢查結論是否穩定；輸出表格與圖。"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

R = os.path.join(os.path.dirname(__file__), "..", "results")
files = sorted(glob.glob(os.path.join(R, "benchmark_seed*.json")))
data = {os.path.basename(f)[14:-5]: json.load(open(f)) for f in files}
seeds = list(data)
sizes = sorted({r["n"] for r in next(iter(data.values()))})
M = ["qaoa", "cvar", "ws_heur", "ws_poor"]
NAME = {"qaoa": "QAOA", "cvar": "CVaR-QAOA", "ws_heur": "WS(一般起點)", "ws_poor": "WS(較差起點)", "rqaoa": "R-QAOA", "sa": "古典 SA", "ls": "1-flip LS"}


def lift(recs, n, m):
    return np.mean([np.mean([x["lift"] for x in r["runs"][m]]) for r in recs if r["n"] == n])


def hit(recs, n, m):
    if m == "rqaoa":
        return np.mean([np.mean([x["hit"] for x in r["runs"][m]]) for r in recs if r["n"] == n])
    if m in ("sa", "ls"):
        return np.mean([r[m + "_hit_rate"] for r in recs if r["n"] == n])
    return np.mean([np.mean([x["top1_hit"] for x in r["runs"][m]]) for r in recs if r["n"] == n])


L = ["# 跨種子彙整（各種子的問題實例與優化初值皆獨立）\n", f"種子：{seeds}\n"]
L.append("## 基態機率 lift（相對均勻隨機的倍數；每格為各種子的實例平均）\n")
L.append("| n | 方法 | " + " | ".join(f"seed {s}" for s in seeds) + " | 跨種子均值 | 跨種子 SD |")
L.append("|---|---|" + "---|" * (len(seeds) + 2))
for n in sizes:
    for m in M:
        v = [lift(data[s], n, m) for s in seeds]
        L.append(f"| {n} | {NAME[m]} | " + " | ".join(f"{x:.2f}" for x in v) + f" | {np.mean(v):.2f} | {np.std(v):.2f} |")
L.append("\n## 『單一答案』命中最優解的比率（越高越好）\n")
L.append("| n | 方法 | " + " | ".join(f"seed {s}" for s in seeds) + " | 跨種子均值 |")
L.append("|---|---|" + "---|" * (len(seeds) + 1))
for n in sizes:
    for m in ["ls", "sa", "rqaoa"]:
        v = [hit(data[s], n, m) for s in seeds]
        L.append(f"| {n} | {NAME[m]} | " + " | ".join(f"{x:.2f}" for x in v) + f" | {np.mean(v):.2f} |")
L.append("\n（1-flip LS：單次隨機起點的局部搜尋；SA：1024 步單鏈。兩者都是微秒～毫秒級的古典基線。）\n")

# 結論穩定性檢查：每個 n、每個種子，下列命題是否成立
checks = {
    "CVaR 的 lift > 標準 QAOA": lambda r, n: lift(r, n, "cvar") > lift(r, n, "qaoa"),
    "標準 QAOA 的 lift > 1（優於隨機）": lambda r, n: lift(r, n, "qaoa") > 1,
    "Warm-start(一般起點) 的 lift > 標準 QAOA": lambda r, n: lift(r, n, "ws_heur") > lift(r, n, "qaoa"),
    "Warm-start(較差起點) 的 lift < Warm-start(一般起點)": lambda r, n: lift(r, n, "ws_poor") < lift(r, n, "ws_heur"),
    "古典 SA 單鏈命中率 > R-QAOA": lambda r, n: hit(r, n, "sa") > hit(r, n, "rqaoa"),
    "R-QAOA 命中率 > 單次 1-flip LS": lambda r, n: hit(r, n, "rqaoa") > hit(r, n, "ls"),
}
L.append("## 結論穩定性（各 n × 各種子，命題成立的比例）\n")
L.append("| 命題 | 成立 / 總數 |")
L.append("|---|---|")
for k, f in checks.items():
    res = [f(data[s], n) for s in seeds for n in sizes]
    L.append(f"| {k} | {sum(res)}/{len(res)} |")
open(os.path.join(R, "summary_all_seeds.md"), "w").write("\n".join(L))
print("\n".join(L))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
w = 0.2
for i, m in enumerate(M):
    mu = [np.mean([lift(data[s], n, m) for s in seeds]) for n in sizes]
    sd = [np.std([lift(data[s], n, m) for s in seeds]) for n in sizes]
    ax[0].bar(np.arange(len(sizes)) + i * w, mu, w, yerr=sd, label=NAME[m], capsize=2)
ax[0].axhline(1, color="k", ls="--", lw=1)
ax[0].set_xticks(np.arange(len(sizes)) + 1.5 * w); ax[0].set_xticklabels([f"n={n}" for n in sizes])
ax[0].set_ylabel("基態機率 lift（×隨機）"); ax[0].set_title("量子變體相對隨機取樣的增益（虛線=隨機）"); ax[0].legend(fontsize=8)
for i, m in enumerate(["ls", "sa", "rqaoa"]):
    mu = [np.mean([hit(data[s], n, m) for s in seeds]) for n in sizes]
    sd = [np.std([hit(data[s], n, m) for s in seeds]) for n in sizes]
    ax[1].bar(np.arange(len(sizes)) + i * 0.25, mu, 0.25, yerr=sd, label=NAME[m], capsize=2)
ax[1].set_xticks(np.arange(len(sizes)) + 0.25); ax[1].set_xticklabels([f"n={n}" for n in sizes])
ax[1].set_ylabel("單一答案命中最優解的比率"); ax[1].set_title("R-QAOA vs 古典基線"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(R, "summary.png"), dpi=150)
