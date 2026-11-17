"""2x2 J1-J2 測試床：掃描 J2，與解析解、隨機基線對照（修正 v3 只測單一簡併點的缺陷）。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qdiag import ising, algos

rows = ["| J2/J1 | E_gs 解析 | E_gs 窮舉 | 基態簡併 | 隨機 p_ground | 隨機 CVaR(0.1) | QAOA p_ground (20 初值均值) | CVaR-QAOA p_ground |",
        "|---|---|---|---|---|---|---|---|"]
for j2 in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
    m = ising.j1j2_2x2(1.0, j2)
    E = m.energies()
    egs, deg = ising.j1j2_analytic(1.0, j2)
    assert abs(E.min() - egs) < 1e-9 and int(np.isclose(E, E.min()).sum()) == deg
    rnd = algos.random_baseline(E)
    cv_rand = algos.cvar(E, np.full(16, 1 / 16), 0.1, np.argsort(E))
    q, c = [], []
    for k in range(20):
        rng = np.random.default_rng([int(j2 * 100), k])
        q.append(algos.run_qaoa(m, rng, p=2)["p_ground"])
        c.append(algos.cvar_qaoa(m, rng, alpha=0.1, p=2)["p_ground"])
    rows.append(f"| {j2:.2f} | {egs:.2f} | {E.min():.2f} | {deg}/16 | {rnd['p_ground']:.3f} | {cv_rand:.3f} | {np.mean(q):.3f} ± {np.std(q):.3f} | {np.mean(c):.3f} ± {np.std(c):.3f} |")
out = "\n".join(rows)
print(out)
open(os.path.join(os.path.dirname(__file__), "..", "results", "j1j2_sweep.md"), "w").write(
    "# J1-J2 2x2 掃描（p=2，精確模擬）\n\n" + out +
    "\n\n解讀：『隨機 CVaR(0.1)』在每個 J2 都等於 E_gs —— 這就是 v3 只印 CVaR 能量卻無法證明任何事的原因。"
    "J2=1.0 為能階交叉（簡併 6/16），不是好的辨識測試點。\n")
