"""QAOA / CVaR-QAOA / Warm-start QAOA (Egger et al.) / R-QAOA (Bravyi et al.) —— 精確模擬版。

所有『效能』指標都用同一套定義（見 metrics），避免 v3 中 CVaR 與期望值互相比較的問題。
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .ising import Ising
from . import sim
from .classical import brute_force, heuristic_solution


# ------------------------------------------------------------------ 共用工具
def normalise(E: np.ndarray) -> np.ndarray:
    """平移並縮放到 std=1，使 gamma 的合理範圍與問題尺度無關（不改變最適解）。"""
    sd = E.std()
    return (E - E.mean()) / (sd if sd > 0 else 1.0)


def cvar(E: np.ndarray, P: np.ndarray, alpha: float, order: np.ndarray) -> float:
    """精確 CVaR_alpha：機率質量最低的前 alpha 部分的條件期望（alpha=1 即期望值）。"""
    Es, Ps = E[order], P[order]
    cum = np.cumsum(Ps)
    w = np.clip(alpha - (cum - Ps), 0.0, Ps)
    return float(w @ Es / alpha)


def metrics(m_E: np.ndarray, P: np.ndarray) -> dict:
    """統一指標（單位無關）：
      p_ground   取樣到任一基態的機率
      lift       p_ground 相對於均勻隨機取樣的倍數（>1 才代表演算法『真的』有幫助）
      exp_shots  期望要取樣幾次才第一次命中基態 = 1/p_ground
      ar         近似比 (Emax - <E>)/(Emax - Emin) ∈[0,1]，1 為最佳
      top1_hit   機率最高的位元串是否為基態；top1_gap 其能量缺口（以 Emax-Emin 正規化）
    """
    emin, emax = m_E.min(), m_E.max()
    ground = np.isclose(m_E, emin, atol=1e-9 * max(1.0, abs(emin)))
    pg = float(P[ground].sum())
    ar = float((emax - P @ m_E) / (emax - emin)) if emax > emin else 1.0
    p0 = float(ground.mean())
    k = int(np.argmax(P))
    return dict(p_ground=pg, lift=pg / p0, exp_shots=1.0 / pg if pg > 0 else float("inf"), ar=ar,
                top1_hit=bool(ground[k]), top1_gap=float((m_E[k] - emin) / (emax - emin)) if emax > emin else 0.0)


def random_baseline(E: np.ndarray) -> dict:
    return metrics(E, np.full(len(E), 1 / len(E)))


def _optimise(f, p: int, rng: np.random.Generator, starts: int, maxiter: int):
    best = None
    for _ in range(starts):
        x0 = np.concatenate([rng.uniform(0, 1.5, p), rng.uniform(0, np.pi / 2, p)])
        r = minimize(f, x0, method="COBYLA", options=dict(maxiter=maxiter, rhobeg=0.3))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, float(best.fun), int(best.nfev)


# ------------------------------------------------------------------ 三種 QAOA 變體
def run_qaoa(m: Ising, rng, objective: str = "expectation", alpha: float = 0.1, p: int = 1,
             starts: int = 4, maxiter: int = 80, init=None, mixer=None, E=None) -> dict:
    """objective: 'expectation'（標準 QAOA）或 'cvar'（CVaR-QAOA）。"""
    n = m.n
    E = m.energies() if E is None else E
    En = normalise(E)
    order = np.argsort(E)

    def state(x):
        return sim.qaoa_state(En, n, x[:p], x[p:], init=init, mixer=mixer)

    def f(x):
        P = sim.probs(state(x))
        return float(P @ En) if objective == "expectation" else cvar(En, P, alpha, order)

    x, fval, nfev = _optimise(f, p, rng, starts, maxiter)
    P = sim.probs(state(x))
    out = metrics(E, P)
    out.update(params=x.tolist(), nfev=nfev, P=P)
    return out


def cvar_qaoa(m, rng, alpha=0.1, **kw):
    return run_qaoa(m, rng, objective="cvar", alpha=alpha, **kw)


# ------------------------------------------------------------------ Warm-start (Egger, Mareček, Woerner 2021)
def spins_to_c(s: np.ndarray, eps: float = 0.25) -> np.ndarray:
    """二元解 -> 帶 epsilon 正則化的 c_i = P(bit_i=1)（bit 1 <-> s=-1）。"""
    x = (1 - np.asarray(s, float)) / 2
    return np.clip(x, eps, 1 - eps)


def convex_relaxation(m: Ising) -> np.ndarray:
    """QUBO 的凸鬆弛（加 mu*I 使其半正定，再於 [0,1]^n 內求解）。

    注意：對『純 ZZ、無外場』問題，Z2 對稱性使唯一最適解為 c=0.5（無資訊）。
    這就是需要 SDP 捨入或啟發式解來打破對稱的原因。
    """
    from scipy.optimize import minimize as _min
    n = m.n
    Q = 2 * m.J
    q = -2 * m.h - 2 * m.J @ np.ones(n)
    mu = max(0.0, -np.linalg.eigvalsh(Q).min()) + 1e-6
    Qc, qc = Q + mu * np.eye(n), q - mu
    f = lambda x: x @ Qc @ x + qc @ x
    g = lambda x: 2 * Qc @ x + qc
    r = _min(f, np.full(n, 0.5), jac=g, bounds=[(0, 1)] * n, method="L-BFGS-B")
    return r.x


def warm_start_qaoa(m: Ising, rng, c: np.ndarray, eps: float = 0.25, p: int = 1, **kw) -> dict:
    c_reg = np.clip(c, eps, 1 - eps)
    thetas = 2 * np.arcsin(np.sqrt(c_reg))
    init = sim.product_state(c_reg)
    out = run_qaoa(m, rng, p=p, init=init, mixer=sim.warm_mixer(thetas), **kw)
    # 起點本身（beta=gamma=0）的品質，用來回答『QAOA 有沒有比它的暖啟動更好』
    E = m.energies()
    out["start"] = metrics(E, sim.probs(init))
    return out


# ------------------------------------------------------------------ R-QAOA (Bravyi, Kliesch, Koenig, Tang 2020)
def r_qaoa(m: Ising, rng, n_cut: int = 4, starts: int = 3, maxiter: int = 60) -> dict:
    """完整遞迴：每輪跑 p=1 QAOA -> 取 |<Z_i Z_j>| 最大的邊 -> 代入 s_i = sigma*s_j -> 降維；
    降到 n_cut 個變數後暴力求解並回推原問題的解。"""
    if np.any(m.h != 0):
        raise NotImplementedError("此實作僅處理無外場（h=0）的 Ising 問題")
    J = m.J.copy()
    offset = m.offset
    idx = list(range(m.n))
    subs, max_corrs = [], []
    n_circuit_qubits = []
    while len(idx) > n_cut:
        n = len(idx)
        sub = Ising(J.copy(), np.zeros(n), 0.0)
        E = sub.energies()
        res = run_qaoa(sub, rng, p=1, starts=starts, maxiter=maxiter, E=E)
        n_circuit_qubits.append(n)
        S = sub.spins()
        M = S.T @ (res["P"][:, None] * S)
        np.fill_diagonal(M, 0.0)
        a, b = np.unravel_index(int(np.argmax(np.abs(M) - 1e-12 * np.arange(n * n).reshape(n, n))), M.shape)
        j, i = min(a, b), max(a, b)                      # 保留 j、消去 i
        sigma = 1.0 if M[i, j] >= 0 else -1.0
        max_corrs.append(float(M[i, j]))
        offset += sigma * J[i, j]
        for k in range(n):
            if k not in (i, j):
                J[j, k] += sigma * J[i, k]
                J[k, j] = J[j, k]
        J = np.delete(np.delete(J, i, 0), i, 1)
        subs.append((idx[i], idx[j], sigma))
        del idx[i]
    resid = Ising(J, np.zeros(len(idx)), offset)
    Er, emin_r, _, ground_r = brute_force(resid)
    k0 = int(np.argmax(ground_r))
    s_res = resid.spins()[k0]
    s_full = np.zeros(m.n)
    for pos, orig in enumerate(idx):
        s_full[orig] = s_res[pos]
    for i_o, j_o, sigma in reversed(subs):
        s_full[i_o] = sigma * s_full[j_o]
    E_found = m.energy_of_spins(s_full)
    E_all, emin, emax, ground = brute_force(m)
    return dict(spins=s_full, energy=E_found, emin=emin, hit_ground=bool(np.isclose(E_found, emin, atol=1e-9 * max(1, abs(emin)))),
                gap_norm=float((E_found - emin) / (emax - emin)), residual_energy=emin_r,
                n_eliminated=len(subs), max_corrs=max_corrs, subs=subs, qubits_per_round=n_circuit_qubits)
