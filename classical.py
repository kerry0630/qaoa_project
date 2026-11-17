"""古典基線：暴力窮舉、1-flip 局部搜尋、模擬退火。"""
from __future__ import annotations
import numpy as np
from .ising import Ising


def brute_force(m: Ising):
    E = m.energies()
    emin, emax = float(E.min()), float(E.max())
    ground = np.isclose(E, emin, atol=1e-9 * max(1.0, abs(emin)))
    return E, emin, emax, ground


def local_search(m: Ising, s0: np.ndarray) -> np.ndarray:
    """最陡下降 1-flip，直到局部最小。"""
    s = s0.astype(float).copy()
    while True:
        # 翻轉 i 的能量變化：dE_i = -2 s_i (h_i + sum_j J_ij s_j)
        dE = -2 * s * (m.h + m.J @ s)
        i = int(np.argmin(dE))
        if dE[i] >= -1e-12:
            return s
        s[i] = -s[i]


def heuristic_solution(m: Ising, rng: np.random.Generator, n_restarts: int = 1, worst: bool = False):
    """模擬企業既有的『快速啟發式』：隨機起點 + 1-flip 局部搜尋。

    n_restarts=1  -> 一般品質的啟發式解
    worst=True    -> 在 n_restarts 次中刻意挑最差的局部最優，用來測試 warm-start 對『差的』起點是否穩健
    """
    cands = []
    for _ in range(n_restarts):
        s0 = rng.choice([-1.0, 1.0], size=m.n)
        s = local_search(m, s0)
        cands.append((m.energy_of_spins(s), s))
    cands.sort(key=lambda t: t[0])
    return (cands[-1] if worst else cands[0])[1], (cands[-1] if worst else cands[0])[0]


def simulated_annealing(m: Ising, rng: np.random.Generator, steps: int = 1024,
                        t0: float | None = None, t1: float | None = None) -> float:
    """單條 SA 鏈（steps 次單點翻轉嘗試），回傳過程中見過的最低能量。"""
    n = m.n
    scale = np.abs(m.J).sum() / max(n, 1)
    t0 = t0 or 1.0 * scale
    t1 = t1 or 0.01 * scale
    s = rng.choice([-1.0, 1.0], size=n)
    e = m.energy_of_spins(s)
    best = e
    for k in range(steps):
        t = t0 * (t1 / t0) ** (k / max(steps - 1, 1))
        i = rng.integers(n)
        de = -2 * s[i] * (m.h[i] + m.J[i] @ s)
        if de <= 0 or rng.random() < np.exp(-de / t):
            s[i] = -s[i]
            e += de
            best = min(best, e)
    return best
