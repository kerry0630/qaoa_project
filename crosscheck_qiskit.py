"""【選配、需安裝 qiskit】把本專案的 numpy 模擬與 Qiskit 的 Statevector 交叉比對。
注意：此腳本在撰寫時未在作者環境執行（該環境無法安裝 qiskit），請你在自己的環境跑一次並保留輸出作為證據。
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qdiag import ising, sim

rng = np.random.default_rng(0)
worst = 0.0
for n in (4, 6, 8):
    m, *_ = ising.two_line_scheduling(n, rng)
    E = m.energies()
    En = (E - E.mean()) / E.std()
    gam, bet = rng.uniform(0, 1.5, 2), rng.uniform(0, 1.5, 2)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b in zip(gam, bet):
        for i in range(n):
            for j in range(i + 1, n):
                # E_norm 的 ZZ 係數 = J_ij / std；exp(-i g J s_i s_j) = RZZ(2 g J)
                qc.rzz(2 * g * m.J[i, j] / E.std(), i, j)
        qc.rx(2 * b, range(n))
    p_q = np.abs(Statevector(qc).data) ** 2
    p_n = sim.probs(sim.qaoa_state(En, n, gam, bet))
    worst = max(worst, float(np.abs(p_q - p_n).max()))
    print(f"n={n}: max |P_qiskit - P_numpy| = {np.abs(p_q - p_n).max():.2e}")
assert worst < 1e-9, "與 Qiskit 不一致！"
print("OK：兩個實作一致")
