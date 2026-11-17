"""執行：python -m unittest discover -s tests -v   （僅需 numpy/scipy）"""
import itertools, os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qdiag import ising, sim, algos, classical


def dense_cost(E):                     # 獨立實作：完整 2^n x 2^n 矩陣
    return np.diag(E)


def dense_x_mixer(n, beta):
    X = np.array([[0, 1], [1, 0]], complex)
    I = np.eye(2)
    Ub = np.cos(beta) * I - 1j * np.sin(beta) * X
    U = np.array([[1.0 + 0j]])
    for _ in range(n):                 # little-endian：後加入者在高位
        U = np.kron(Ub, U)
    return U


class TestIsing(unittest.TestCase):
    def test_scheduling_expansion_matches_direct_cost(self):
        rng = np.random.default_rng(3)
        for n in (4, 6, 7):
            m, p, C, lam = ising.two_line_scheduling(n, rng, lam=0.7)
            E, S = m.energies(), m.spins()
            direct = np.array([ising.scheduling_cost_direct(p, C, lam, s) for s in S])
            np.testing.assert_allclose(E, direct, atol=1e-9)

    def test_bit_convention(self):
        m = ising.j1j2_2x2(1.0, 0.3)
        E = m.energies()
        for k in range(16):
            s = np.array([1 - 2 * ((k >> i) & 1) for i in range(4)], float)
            self.assertAlmostEqual(E[k], m.energy_of_spins(s))

    def test_j1j2_analytic_and_crossing(self):
        for j2 in (0.0, 0.25, 0.5, 0.75, 0.99, 1.0, 1.01, 1.5, 2.0):
            E = ising.j1j2_2x2(1.0, j2).energies()
            egs, deg = ising.j1j2_analytic(1.0, j2)
            self.assertAlmostEqual(E.min(), egs, places=9, msg=f"J2={j2}")
            self.assertEqual(int(np.isclose(E, E.min()).sum()), deg, msg=f"J2={j2}")
        # Néel 態與 Stripe 態能量
        s_neel, s_stripe = np.array([1, -1, -1, 1.0]), np.array([1, 1, -1, -1.0])
        m = ising.j1j2_2x2(1.0, 0.4)
        self.assertAlmostEqual(m.energy_of_spins(s_neel), -4 + 2 * 0.4)
        self.assertAlmostEqual(m.energy_of_spins(s_stripe), -2 * 0.4)
        # 能階交叉在 J2/J1 = 1（有限系統是 level crossing，不是相變）
        self.assertEqual(int(np.isclose(ising.j1j2_2x2(1, 1).energies(), -2).sum()), 6)


class TestSimulator(unittest.TestCase):
    def test_matches_independent_dense_simulator(self):
        rng = np.random.default_rng(0)
        for n in (3, 4, 5):
            E = rng.normal(size=2 ** n)
            g, b = rng.uniform(0, 2, 2), rng.uniform(0, 1.5, 2)
            fast = sim.qaoa_state(E, n, g, b)
            psi = np.full(2 ** n, 2 ** (-n / 2), complex)
            for gi, bi in zip(g, b):
                psi = dense_x_mixer(n, bi) @ (np.diag(np.exp(-1j * gi * E)) @ psi)
            np.testing.assert_allclose(fast, psi, atol=1e-12)

    def test_norm_preserved(self):
        rng = np.random.default_rng(1)
        n = 6
        E = rng.normal(size=2 ** n)
        c = np.clip(rng.random(n), 0.25, 0.75)
        th = 2 * np.arcsin(np.sqrt(c))
        psi = sim.qaoa_state(E, n, [0.9, 0.4], [0.3, 1.1], init=sim.product_state(c), mixer=sim.warm_mixer(th))
        self.assertAlmostEqual(float(np.linalg.norm(psi)), 1.0, places=12)

    def test_zero_params_return_initial_state(self):
        n = 4
        c = np.array([0.8, 0.25, 0.25, 0.75])
        init = sim.product_state(c)
        P = sim.probs(init)
        for k in range(2 ** n):                          # 邊際機率 P(bit_i=1)=c_i 且為乘積分佈
            expect = np.prod([c[i] if (k >> i) & 1 else 1 - c[i] for i in range(n)])
            self.assertAlmostEqual(P[k], expect, places=12)
        th = 2 * np.arcsin(np.sqrt(c))
        out = sim.qaoa_state(np.zeros(2 ** n), n, [0.0], [0.0], init=init, mixer=sim.warm_mixer(th))
        np.testing.assert_allclose(out, init, atol=1e-12)

    def test_warm_mixer_has_initial_state_as_eigenstate(self):
        n = 4
        c = np.array([0.8, 0.25, 0.25, 0.75])
        th = 2 * np.arcsin(np.sqrt(c))
        init = sim.product_state(c)
        for beta in (0.3, 1.0, 2.2):
            psi = init.copy()
            for q, U in enumerate(sim.warm_mixer(th)(beta)):
                psi = sim.apply_1q(psi, U, q)
            overlap = np.vdot(init, psi)
            self.assertAlmostEqual(abs(overlap), 1.0, places=12)        # 只差一個全域相位
            np.testing.assert_allclose(psi, overlap * init, atol=1e-12)

    def test_warm_start_with_c_half_equals_standard_qaoa_with_negated_beta(self):
        rng = np.random.default_rng(5)
        n = 5
        E = rng.normal(size=2 ** n)
        c = np.full(n, 0.5)
        th = 2 * np.arcsin(np.sqrt(c))
        w = sim.qaoa_state(E, n, [0.7], [0.4], init=sim.product_state(c), mixer=sim.warm_mixer(th))
        s = sim.qaoa_state(E, n, [0.7], [-0.4])
        np.testing.assert_allclose(sim.probs(w), sim.probs(s), atol=1e-12)


class TestObjectives(unittest.TestCase):
    def test_cvar_limits(self):
        rng = np.random.default_rng(2)
        E = rng.normal(size=64)
        P = rng.random(64); P /= P.sum()
        order = np.argsort(E)
        self.assertAlmostEqual(algos.cvar(E, P, 1.0, order), float(P @ E), places=12)
        self.assertAlmostEqual(algos.cvar(E, P, 1e-9, order), E.min(), places=6)
        a = [algos.cvar(E, P, al, order) for al in (0.05, 0.1, 0.3, 0.7, 1.0)]
        self.assertTrue(all(x <= y + 1e-12 for x, y in zip(a, a[1:])))      # 對 alpha 單調

    def test_v3_flaw_cvar_of_random_guessing_equals_ground_energy_at_J2_eq_1(self):
        """重現 v3 的缺陷：J1=J2=1 時基態佔 6/16，完全隨機取樣的 CVaR_0.1 也等於基態能量。"""
        E = ising.j1j2_2x2(1, 1).energies()
        P = np.full(16, 1 / 16)
        self.assertAlmostEqual(algos.cvar(E, P, 0.1, np.argsort(E)), E.min(), places=12)

    def test_metrics_random_baseline_lift_is_one(self):
        E = ising.j1j2_2x2(1, 0.5).energies()
        self.assertAlmostEqual(algos.random_baseline(E)["lift"], 1.0)


class TestClassicalAndRQAOA(unittest.TestCase):
    def test_local_search_returns_local_minimum(self):
        rng = np.random.default_rng(4)
        m, *_ = ising.two_line_scheduling(8, rng)
        s = classical.local_search(m, rng.choice([-1.0, 1.0], 8))
        e0 = m.energy_of_spins(s)
        for i in range(8):
            t = s.copy(); t[i] = -t[i]
            self.assertGreaterEqual(m.energy_of_spins(t), e0 - 1e-12)

    def test_convex_relaxation_is_uninformative_without_fields(self):
        m, *_ = ising.two_line_scheduling(6, np.random.default_rng(7))
        np.testing.assert_allclose(algos.convex_relaxation(m), 0.5, atol=1e-3)

    def test_rqaoa_reduction_is_exact(self):
        """每次代入 s_i=sigma*s_j 後，降維問題的能量（含 offset）必須等於還原後原問題的能量。"""
        for seed in range(6):
            rng = np.random.default_rng(seed)
            m, *_ = ising.two_line_scheduling(7, rng)
            r = algos.r_qaoa(m, rng, n_cut=3, starts=2, maxiter=30)
            self.assertAlmostEqual(r["energy"], r["residual_energy"], places=9)
            self.assertEqual(r["n_eliminated"], 4)
            self.assertGreaterEqual(r["energy"], r["emin"] - 1e-9)

    def test_determinism_same_seed_same_result(self):
        m, *_ = ising.two_line_scheduling(6, np.random.default_rng(9))
        a = algos.run_qaoa(m, np.random.default_rng(11))
        b = algos.run_qaoa(m, np.random.default_rng(11))
        self.assertEqual(a["params"], b["params"])
        self.assertEqual(a["p_ground"], b["p_ground"])


if __name__ == "__main__":
    unittest.main()
