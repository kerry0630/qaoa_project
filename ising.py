"""Ising 問題定義與測試問題產生器。

約定（全專案一致，並由 tests 檢查）：
  * 位元索引 k 的第 i 個 bit = 量子位元 i（little-endian，與 Qiskit 相同）。
  * 自旋 s_i = 1 - 2*bit_i，即 bit 0 -> s=+1，bit 1 -> s=-1。
  * 能量 E(s) = offset + sum_i h_i s_i + sum_{i<j} J_ij s_i s_j  （J 以對稱矩陣儲存、對角為 0）。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Ising:
    J: np.ndarray            # (n,n) 對稱、對角為 0
    h: np.ndarray            # (n,)
    offset: float = 0.0
    name: str = ""

    @property
    def n(self) -> int:
        return self.J.shape[0]

    def spins(self) -> np.ndarray:
        """(2^n, n) 的自旋矩陣。"""
        k = np.arange(2 ** self.n)[:, None]
        bits = (k >> np.arange(self.n)) & 1
        return (1 - 2 * bits).astype(float)

    def energies(self) -> np.ndarray:
        S = self.spins()
        return self.offset + S @ self.h + 0.5 * np.einsum("ki,ij,kj->k", S, self.J, S)

    def energy_of_spins(self, s: np.ndarray) -> float:
        s = np.asarray(s, float)
        return float(self.offset + s @ self.h + 0.5 * s @ self.J @ s)


def from_edges(n, edges, offset=0.0, name="") -> Ising:
    J = np.zeros((n, n))
    for i, j, w in edges:
        J[i, j] += w
        J[j, i] += w
    return Ising(J, np.zeros(n), offset, name)


# ---------------------------------------------------------------- J1-J2 (原 v3 的 2x2 測試床)
J1_EDGES = [(0, 1), (2, 3), (0, 2), (1, 3)]
J2_EDGES = [(0, 3), (1, 2)]


def j1j2_2x2(j1: float, j2: float) -> Ising:
    edges = [(i, j, j1) for i, j in J1_EDGES] + [(i, j, j2) for i, j in J2_EDGES]
    return from_edges(4, edges, name=f"J1J2_2x2(J1={j1},J2={j2})")


def j1j2_analytic(j1: float, j2: float):
    """解析候選：Néel = -4J1+2J2，Stripe = -2J2。回傳 (E_gs, 簡併度)，僅在 J1,J2>0 的區域保證成立。"""
    neel, stripe = -4 * j1 + 2 * j2, -2 * j2
    if np.isclose(neel, stripe):
        return neel, 6
    return (neel, 2) if neel < stripe else (stripe, 4)


# ---------------------------------------------------------------- 兩產線指派（製造情境）
def two_line_scheduling(n: int, rng: np.random.Generator, lam: float = 1.0,
                        conflict_density: float = 0.5, cmax: float = 6.0):
    """把 n 張工單分到兩條產線 (s_i=±1)。回傳 (Ising, p, C, lam) 以便獨立驗證。

    成本 = lam*(sum_i p_i s_i)^2          產線負載不平衡（makespan 的平滑代理）
         + sum_{i<j} c_ij (1+s_i s_j)/2   衝突成本：c_ij>0 的兩張單若同線，需額外換線/共用治具
    這是 number-partitioning + 衝突分割（含 MaxCut 成分），NP-hard。
    """
    p = rng.integers(1, 10, size=n).astype(float)
    C = np.triu(rng.uniform(1, cmax, size=(n, n)) * (rng.random((n, n)) < conflict_density), 1)
    J = 2 * lam * np.outer(p, p) + 0.5 * (C + C.T)
    np.fill_diagonal(J, 0.0)
    offset = lam * float(p @ p) + 0.5 * float(C.sum())
    return Ising(J, np.zeros(n), offset, name=f"two_line_n{n}"), p, C, lam


def scheduling_cost_direct(p, C, lam, s) -> float:
    """不經 Ising 展開、直接計算成本（供測試獨立驗證展開是否正確）。"""
    p = np.asarray(p, float); s = np.asarray(s, float)
    n = len(p)
    cost = lam * float((p @ s) ** 2)
    for i in range(n):
        for j in range(i + 1, n):
            cost += C[i, j] * (1 + s[i] * s[j]) / 2
    return cost
