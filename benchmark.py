"""多種子、多實例的公平比較（精確模擬，無取樣雜訊；隨機性只來自問題實例與優化初值）。

用法：python scripts/benchmark.py --seed 0 --sizes 8 10 12 --instances 8 --inits 4
輸出：results/benchmark_seed{S}.json 與 results/benchmark_seed{S}.md
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qdiag import ising, algos, classical

METHODS = ["qaoa", "cvar", "ws_heur", "ws_poor", "rqaoa"]
LABEL = {"qaoa": "QAOA(p=1)", "cvar": "CVaR-QAOA(α=0.1)", "ws_heur": "Warm-start(一般啟發式起點)",
         "ws_poor": "Warm-start(刻意較差起點)", "rqaoa": "R-QAOA(完整遞迴)"}


def one_instance(n, inst_seed, inits, starts, maxiter):
    rng_inst = np.random.default_rng(inst_seed)
    m, p, C, lam = ising.two_line_scheduling(n, rng_inst)
    E, emin, emax, ground = classical.brute_force(m)
    p0 = float(ground.mean())
    rec = dict(n=n, seed=int(inst_seed), n_ground=int(ground.sum()), p_random=p0, runs={k: [] for k in METHODS})
    # 古典對照：1-flip 局部搜尋與 SA 的命中率（各 200 次獨立執行）
    r = np.random.default_rng(inst_seed + 777)
    ls_hit = np.mean([np.isclose(m.energy_of_spins(classical.local_search(m, r.choice([-1.0, 1.0], n))), emin, atol=1e-9 * max(1, abs(emin))) for _ in range(200)])
    sa_hit = np.mean([np.isclose(classical.simulated_annealing(m, r, steps=1024), emin, atol=1e-9 * max(1, abs(emin))) for _ in range(200)])
    rec.update(ls_hit_rate=float(ls_hit), sa_hit_rate=float(sa_hit))
    for k in range(inits):
        rng = np.random.default_rng([inst_seed, k, 1])
        kw = dict(starts=starts, maxiter=maxiter)
        a = algos.run_qaoa(m, rng, **kw); a.pop("P")
        b = algos.cvar_qaoa(m, rng, alpha=0.1, **kw); b.pop("P")
        s_good, e_good = classical.heuristic_solution(m, rng, 1)
        c = algos.warm_start_qaoa(m, rng, algos.spins_to_c(s_good), **kw); c.pop("P")
        c["start_is_opt"] = bool(np.isclose(e_good, emin, atol=1e-9 * max(1, abs(emin))))
        s_poor, e_poor = classical.heuristic_solution(m, rng, 20, worst=True)
        d = algos.warm_start_qaoa(m, rng, algos.spins_to_c(s_poor), **kw); d.pop("P")
        d["start_is_opt"] = bool(np.isclose(e_poor, emin, atol=1e-9 * max(1, abs(emin))))
        e = algos.r_qaoa(m, rng, n_cut=4, starts=max(2, starts - 1), maxiter=maxiter)
        e = dict(hit=e["hit_ground"], gap=e["gap_norm"], n_eliminated=e["n_eliminated"], max_corrs=e["max_corrs"])
        for name, v in zip(METHODS, (a, b, c, d, e)):
            rec["runs"][name].append(v)
    return rec


def summarise(recs, sizes):
    lines = []
    for n in sizes:
        R = [r for r in recs if r["n"] == n]
        lines.append(f"\n### n = {n} 個變數（{len(R)} 個實例 × 每法多組初值）\n")
        lines.append(f"- 隨機取樣基態機率（平均）：{np.mean([r['p_random'] for r in R]):.4f}；"
                     f"基態簡併度（平均）：{np.mean([r['n_ground'] for r in R]):.1f}")
        lines.append(f"- 古典對照命中率（平均）：1-flip 局部搜尋 {np.mean([r['ls_hit_rate'] for r in R]):.2f}、"
                     f"SA(1024 步) {np.mean([r['sa_hit_rate'] for r in R]):.2f}\n")
        lines.append("| 方法 | 基態機率 lift（×隨機，均值±SD） | 單答案命中率 | 實例勝過隨機(Wilcoxon p) | 對 QAOA 的 p |")
        lines.append("|---|---|---|---|---|")
        per_inst = {}
        for name in METHODS:
            if name == "rqaoa":
                per_inst[name] = np.array([np.mean([x["hit"] for x in r["runs"][name]]) for r in R])
            else:
                per_inst[name] = np.array([np.mean([x["lift"] for x in r["runs"][name]]) for r in R])
        for name in METHODS:
            if name == "rqaoa":
                hit = per_inst[name].mean()
                lines.append(f"| {LABEL[name]} | （單一答案，不適用 lift） | {hit:.2f} | — | — |")
                continue
            lifts = np.concatenate([[x["lift"] for x in r["runs"][name]] for r in R])
            hit = np.mean([x["top1_hit"] for r in R for x in r["runs"][name]])
            d = per_inst[name] - 1.0
            p_rand = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            if name == "qaoa":
                p_q = "—"
            else:
                dq = per_inst[name] - per_inst["qaoa"]
                p_q = f"{wilcoxon(dq).pvalue:.3g}" if np.any(dq != 0) else "1"
            lines.append(f"| {LABEL[name]} | {lifts.mean():.2f} ± {lifts.std():.2f} | {hit:.2f} | {np.mean(d>0):.2f} (p={p_rand:.3g}) | {p_q} |")
        # warm-start 附加分析：只看起點『不是』最優解的情況，QAOA 有沒有把它改得更好
        for name in ("ws_heur", "ws_poor"):
            sub = [(x["start"]["p_ground"], x["p_ground"]) for r in R for x in r["runs"][name] if not x["start_is_opt"]]
            full = sum(1 for r in R for x in r["runs"][name])
            if sub:
                a, b = np.array(sub).T
                lines.append(f"\n- {LABEL[name]}：起點非最優的情形 {len(sub)}/{full} 次；"
                             f"起點基態機率均值 {a.mean():.3f} → QAOA 後 {b.mean():.3f}（提升 {np.mean(b>a+1e-12):.0%} 的次數）")
            else:
                lines.append(f"\n- {LABEL[name]}：所有起點本身已是最優解，無法評估增益")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 10, 12])
    ap.add_argument("--instances", type=int, default=8)
    ap.add_argument("--inits", type=int, default=4)
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs, t0 = [], time.time()
    for n in a.sizes:
        for i in range(a.instances):
            recs.append(one_instance(n, 100000 * a.seed + 1000 * n + i, a.inits, a.starts, a.maxiter))
            print(f"[{time.time()-t0:6.0f}s] n={n} instance {i+1}/{a.instances} done", flush=True)
    base = os.path.join(a.out, f"benchmark_seed{a.seed}")
    json.dump(recs, open(base + ".json", "w"), indent=1, default=float)
    md = f"# Benchmark seed={a.seed}\n參數：sizes={a.sizes} instances={a.instances} inits={a.inits} starts={a.starts} maxiter={a.maxiter}\n" + summarise(recs, a.sizes)
    open(base + ".md", "w").write(md)
    print(md)
