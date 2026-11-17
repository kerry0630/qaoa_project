"""精確狀態向量模擬（對角成本哈密頓量 + 逐量子位元單比特閘）。純 numpy，無隨機性。"""
from __future__ import annotations
import numpy as np

X = np.array([[0, 1], [1, 0]], complex)


def rx(beta: float) -> np.ndarray:
    """exp(-i*beta*X)"""
    c, s = np.cos(beta), np.sin(beta)
    return np.array([[c, -1j * s], [-1j * s, c]])


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], complex)


def rz(phi: float) -> np.ndarray:
    return np.diag([np.exp(-1j * phi / 2), np.exp(1j * phi / 2)])


def apply_1q(psi: np.ndarray, U: np.ndarray, q: int) -> np.ndarray:
    """對量子位元 q 作用 2x2 單比特閘（索引的第 q 個 bit）。"""
    v = psi.reshape(-1, 2, 2 ** q)
    return np.einsum("ab,ibj->iaj", U, v).reshape(-1)


def apply_all(psi: np.ndarray, U: np.ndarray, n: int) -> np.ndarray:
    for q in range(n):
        psi = apply_1q(psi, U, q)
    return psi


def plus_state(n: int) -> np.ndarray:
    return np.full(2 ** n, 2 ** (-n / 2), complex)


def product_state(c: np.ndarray) -> np.ndarray:
    """逐位元 Ry(theta_i)|0> 的乘積態，P(bit_i=1)=c_i。索引的 bit i = qubit i。"""
    c = np.asarray(c, float)
    psi = np.array([1.0 + 0j])
    for ci in c:                       # 後加入的位元放在高位 -> little-endian
        q = np.array([np.sqrt(1 - ci), np.sqrt(ci)], complex)
        psi = np.kron(q, psi)
    return psi


def probs(psi: np.ndarray) -> np.ndarray:
    return np.abs(psi) ** 2


def qaoa_state(E: np.ndarray, n: int, gammas, betas, init=None, mixer=None) -> np.ndarray:
    """|psi> = prod_l [ U_M(beta_l) e^{-i gamma_l E} ] |init>。

    mixer: None -> 標準 exp(-i beta X)；否則為 callable(beta)->長度 n 的 2x2 矩陣 list。
    """
    psi = plus_state(n) if init is None else init.astype(complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * E)
        if mixer is None:
            psi = apply_all(psi, rx(b), n)
        else:
            for q, U in enumerate(mixer(b)):
                psi = apply_1q(psi, U, q)
    return psi


def warm_mixer(thetas: np.ndarray):
    """Egger 等人的 warm-start mixer：U_i(beta) = Ry(theta_i) Rz(-2 beta) Ry(-theta_i)。"""
    def f(beta):
        return [ry(t) @ rz(-2 * beta) @ ry(-t) for t in thetas]
    return f
