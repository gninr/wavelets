from scaling_function import PrimalScalingFunction, DualScalingFunction
from math import ceil, factorial
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import solve_triangular
from scipy.special import binom
import matplotlib.pyplot as plt


class PrimalMRA:
    def __init__(self, d):
        self.d = d
        self.compute_ML()
        self.sf = PrimalScalingFunction(d)

    def basis_functions(self, j, nu=0, from_refine_mat=False):
        d = self.d
        n = 2**j + d - 1
        bs = []

        def bspline(knots, coeffs, d):
            b = BSpline(knots, coeffs, d - 1, extrapolate=False)
            if nu > 0:
                b = b.derivative(nu)
            return lambda x: np.nan_to_num(b(x))

        if from_refine_mat:
            M0 = self.refinement_matrix(j)
            knots = np.concatenate((np.zeros(d - 1),
                                    np.linspace(0, 1, 2**(j+1) + 1),
                                    np.ones(d - 1)))
            for k in range(n):
                coeffs = 2**((j+1)/2) * M0[:, k]
                bs.append(bspline(knots, coeffs, d))
        else:
            knots = np.concatenate((np.zeros(d - 1),
                                    np.linspace(0, 1, 2**j + 1),
                                    np.ones(d - 1)))
            for k in range(n):
                coeffs = np.zeros(n)
                coeffs[k] = 2**(j/2)
                bs.append(bspline(knots, coeffs, d))
        return bs

    def support(self, j, k):
        d = self.d
        return (max(2**(-j) * (k-d+1), 0), min(2**(-j) * (k+1), 1))

    def plot(self, j, k=None, nu=0, from_refine_mat=False):
        bs = self.basis_functions(j, nu, from_refine_mat)
        x = np.linspace(0, 1, 1000)

        if k is None:
            for b in bs:
                plt.plot(x, b(x))
        else:
            plt.plot(x, bs[k](x))
        plt.show()

    def compute_ML(self):
        d = self.d
        knots = np.concatenate((np.zeros(d - 1), np.arange(3 * d - 3)))
        x = np.arange(2 * d - 2)

        B1 = np.empty((2 * d - 2, d - 1))
        for k in range(d - 1):
            coeffs = np.zeros(3 * d - 4)
            coeffs[k] = 1.
            b = BSpline(knots, coeffs, d - 1, extrapolate=False)
            B1[:, k] = b(x / 2)

        B2 = np.empty((2 * d - 2, 2 * d - 2))
        for k in range(2 * d - 2):
            coeffs = np.zeros(3 * d - 4)
            coeffs[k] = 1.
            b = BSpline(knots, coeffs, d - 1, extrapolate=False)
            B2[:, k] = b(x)

        self.ML = solve_triangular(B2, B1)

    def compute_A(self, j):
        d = self.d
        A = np.zeros((2**(j+1) - d + 1, 2**j - d + 1))
        for k in range(2**j - d + 1):
            A[2*k:2*k+d+1, k] = self.sf.refinement_coeffs()
        return A

    def refinement_matrix(self, j):
        d = self.d
        M0 = np.zeros((2**(j+1) + d - 1, 2**j + d - 1))
        M0[:2*d-2, :d-1] = self.ML
        M0[d-1:2**(j+1), d-1:2**j] = self.compute_A(j)
        M0[2-2*d:, 1-d:] = self.ML[::-1, ::-1]

        return 1 / np.sqrt(2) * M0

    def gramian(self, j):
        d = self.d
        G = np.zeros((2**j + d - 1, 2**j + d - 1))

        g = self.sf.gramian()
        for k in range(2**j - d + 1):
            G[k+d-1, k:k+2*d-1] = g

        M0 = self.refinement_matrix(j)
        n = 2 * d - 2
        U = M0[:n, :n]
        L = M0[n:2*n, :n]
        lhs = np.identity(n**2) - np.kron(U.T, U.T)
        rhs = (U.T @ G[:n, n:2*n] @ L
               + L.T @ G[n:2*n, :n] @ U
               + L.T @ G[n:2*n, n:2*n] @ L).reshape(-1, order='F')
        GL = np.linalg.solve(lhs, rhs).reshape((n, n), order='F')
        G[:n, :n] = GL
        G[-n:, -n:] = GL[::-1, ::-1]

        return G

    def inner_product(self, j, nu=0):
        d0 = self.d - nu
        A = PrimalMRA(d0).gramian(j)

        for d in range(d0 + 1, self.d + 1):
            for i in range(d - 2):
                c = (d - 1) / (i + 1)
                A[i, :] *= c
                A[:, i] *= c
                A[-i-1, :] *= c
                A[:, -i-1] *= c
            n = A.shape[0]
            Ad = np.zeros((n + 1, n + 1))
            Ad[:-1, :-1] += A
            Ad[:-1, 1:] -= A
            Ad[1:, :-1] -= A
            Ad[1:, 1:] += A
            A = 2**(2*j) * Ad
        return A


class DualMRA:
    def __init__(self, d, d_t):
        self.d = d
        self.d_t = d_t
        self.j0 = ceil(np.log2(d + 2 * d_t - 3) + 1)
        self.sf = PrimalScalingFunction(d)
        self.sf_t = DualScalingFunction(d, d_t)
        self.mra = PrimalMRA(d)
        self.compute_ML()

    def compute_ML(self):
        d = self.d
        d_t = self.d_t
        l1 = self.sf.l1
        l2 = self.sf.l2
        l1_t = self.sf_t.l1
        l2_t = self.sf_t.l2
        a = self.sf.refinement_coeffs()
        a_t = self.sf_t.refinement_coeffs()
        ML_t = np.zeros((2 * d + 3 * d_t - 5, d + d_t - 2))

        ML = self.mra.ML
        ML_full = np.zeros_like(ML_t)
        ML_full[:2*d-2, :d-1] = ML
        for k in range(d - 1, d + d_t - 2):
            ML_full[2*k-d+1:2*k+2, k] = a
        ML = ML_full

        # Compute block of ML_t corresponding to k = d-2, ..., d+2*d_t-3

        # Compute alpha_{0,r}
        alpha0 = np.zeros(d_t)
        alpha0[0] = 1
        for r in range(1, d_t):
            for k in range(l1, l2 + 1):
                sum = 0
                for s in range(r):
                    sum += binom(r, s) * k**(r-s) * alpha0[s]
                alpha0[r] += a[k-l1] * sum
            alpha0[r] /= (2**(r+1) - 2)

        # Compute alpha_{k,r}
        def alpha(k, r):
            res = 0
            for i in range(r + 1):
                res += binom(r, i) * k**i * alpha0[r-i]
            return res

        # Compute beta_{n,r}
        def beta(n, r):
            res = 0
            for k in range(ceil((n-l2_t) / 2), -l1_t):
                res += alpha(k, r) * a_t[n-2*k-l1_t]
            return res

        def divided_diff(f, t):
            if t.size == 1:
                return f(t[0])
            return (divided_diff(f, t[1:]) - divided_diff(f, t[:-1])) \
                / (t[-1] - t[0])

        D1 = np.zeros((d_t, d_t))
        D2 = np.zeros((d_t, d_t))
        D3 = np.zeros((d_t, d_t))
        k0 = -l1_t - 1
        for n in range(d_t):
            for k in range(n + 1):
                D1[n, k] = binom(n, k) * alpha0[n-k]
                D2[n, k] = binom(n, k) * k0**(n-k) * (-1)**k
                D3[n, k] = factorial(k)\
                    * divided_diff(lambda x: x**n, np.arange(k + 1))
        D_t = D1 @ D2 @ D3
        rhs = np.empty((d_t, d + 3 * d_t - 3))
        rhs[:, :d_t] = \
            np.diag([2**(-r) for r in range(d_t)]) @ D_t[:, ::-1]
        rhs[:, d_t:] = np.array(
            [[beta(n-l1_t, r) for n in range(d + 2 * d_t - 3)]
             for r in range(d_t)])
        ML_t[d-2:, d-2:] = solve_triangular(D_t, rhs, lower=True)[::-1, :].T

        def compute_gramian():
            n = ML.shape[1]
            UL = ML[:n, :]
            LL = ML[n:, :]
            UL_t = ML_t[:n, :]
            LL_t = ML_t[n:, :]
            lhs = 2 * np.identity(n**2) - np.kron(UL_t.T, UL.T)
            rhs = (LL.T @ LL_t).reshape(-1, order='F')
            gamma = np.linalg.solve(lhs, rhs)
            return gamma.reshape((n, n), order='F')

        gramian_full = np.identity(2 * d + 3 * d_t - 5)
        for k in range(d - 3, -1, -1):
            gramian_full[:d+d_t-2, :d+d_t-2] = compute_gramian()
            B_k = ML[:, :k+d].T @ gramian_full[:, k+1:2*k+d+1] / 2.

            delta = np.zeros(k + d)
            delta[k] = 1
            ML_t[k+1:2*k+d+1, k] = np.linalg.solve(B_k, delta)
        ML_t[np.abs(ML_t) < 1e-9] = 0.

        # Biorthogonalize

        gramian = np.triu(compute_gramian())
        gramian[:d-2, :d-2] = np.identity(d - 2)
        ML_t[:d+d_t-2, :d+d_t-2] = gramian @ ML_t[:d+d_t-2, :d+d_t-2]
        ML_t = solve_triangular(gramian.T, ML_t.T, lower=True).T

        """
        # Apply boundary condition
        UL = ML_t[:d+d_t-2, :d+d_t-2]
        w, v = np.linalg.eig(UL.T)
        idx = np.argmax(w)
        assert np.isclose(w[idx], 1.)
        la = v[:, idx]
        la /= la[0]
        """

        self.ML = ML_t

    def compute_A(self, j):
        assert j >= self.j0
        d = self.d
        d_t = self.d_t
        A_t = np.zeros((2**(j+1) - d - 2 * d_t + 3, 2**j - d - 2 * d_t + 3))
        for k in range(2**j - d - 2 * d_t + 3):
            A_t[2*k:2*k+d+2*d_t-1, k] = self.sf_t.refinement_coeffs()
        return A_t

    def refinement_matrix(self, j):
        d = self.d
        d_t = self.d_t
        M0_t = np.zeros((2**(j+1) + d - 1, 2**j + d - 1))
        M0_t[:2*d+3*d_t-5, :d+d_t-2] = self.ML
        M0_t[d+d_t-2:2**(j+1)-d_t+1, d+d_t-2:2**j-d_t+1] = self.compute_A(j)
        M0_t[-(2*d+3*d_t-5):, -(d+d_t-2):] = self.ML[::-1, ::-1]

        return 1 / np.sqrt(2) * M0_t


class WaveletBasis:
    def __init__(self, d, d_t):
        self.d = d
        self.d_t = d_t
        self.sf = PrimalScalingFunction(d)
        self.sf_t = DualScalingFunction(d, d_t)
        self.mra = PrimalMRA(d)
        self.mra_t = DualMRA(d, d_t)

    def basis_functions(self, j, nu=0):
        d = self.d
        n = 2**j
        bs = []

        def bspline(knots, coeffs, d):
            b = BSpline(knots, coeffs, d - 1, extrapolate=False)
            if nu > 0:
                b = b.derivative(nu)
            return lambda x: np.nan_to_num(b(x))

        _, M1, _, _ = self.refinement_matrix(j)
        knots = np.concatenate((np.zeros(d - 1),
                                np.linspace(0, 1, 2**(j+1) + 1),
                                np.ones(d - 1)))
        for k in range(n):
            coeffs = 2**((j+1)/2) * M1[:, k]
            bs.append(bspline(knots, coeffs, d))
        return bs

    def support(self, j, k):
        # primal
        d = self.d
        d_t = self.d_t
        n = (d + d_t - 2) // 2  # number of boundary wavelets
        if k < n:
            return (0, 2**(-j) * (d + d_t - 2))
        elif k >= 2**j - n:
            return (1 - 2**(-j) * (d + d_t - 2), 1)
        else:
            return (max(2**(-j) * (k-n), 0), min(2**(-j) * (k-n+d+d_t-1), 1))

    def plot(self, j, k=None, nu=0, boundary=False):
        bs = self.basis_functions(j, nu)
        if boundary:
            L = self.d + self.d_t - 2
            x = np.linspace(0, 2**(-j) * L, 1000)
            assert k is not None and k < L // 2
            plt.plot(2**j * x, 2**(-j/2) * bs[k](x))
        else:
            x = np.linspace(0, 1, 1000)
            if k is None:
                for b in bs:
                    plt.plot(x, b(x))
            else:
                plt.plot(x, bs[k](x))
        plt.show()

    def initial_completion(self, j):
        d = self.d
        l1 = self.sf.l1
        l2 = self.sf.l2
        p = 2**j - d + 1
        q = 2**(j+1) - d + 1

        ML = self.mra.ML
        P = np.identity(q + 2 * d - 2)
        P[:2*d-2, :d-1] = ML
        P[2-2*d:, 1-d:] = ML[::-1, ::-1]

        ML_inv = np.linalg.inv(P[:2*d-2, :2*d-2])
        P_inv = np.identity(q + 2 * d - 2)
        P_inv[:2*d-2, :2*d-2] = ML_inv
        P_inv[2-2*d:, 2-2*d:] = ML_inv[::-1, ::-1]

        A = self.mra.compute_A(j)
        H = np.identity(q)
        H_inv = np.identity(q)
        for i in range(d):
            if i % 2 == 0:
                m = i // 2
                v = A[m, 0] / A[m+1, 0]
                H_i = np.identity(q)
                H_i_inv = np.identity(q)
                rows = np.arange(m % 2, q, 2)
                cols = np.arange(m % 2 + 1, q, 2)
                rows = rows[:cols.size]
                H_i[rows, cols] = -v
                H_i_inv[rows, cols] = v
            else:
                m = (i - 1) // 2
                v = A[d-m, 0] / A[d-m-1, 0]
                H_i = np.identity(q)
                H_i_inv = np.identity(q)
                rows = np.arange(q - m % 2, 0, -2) - 1
                cols = np.arange(q - m % 2 - 1, 0, -2) - 1
                rows = rows[:cols.size]
                H_i[rows, cols] = -v
                H_i_inv[rows, cols] = v
            A = H_i @ A
            H = H_i @ H
            H_inv = H_inv @ H_i_inv

        b = A[l2, 0]
        rows = np.arange(p) * 2 + l2
        cols = np.arange(p)
        A = np.zeros_like(A)
        A[rows, cols] = b
        B = np.zeros_like(A.T)
        B[cols, rows] = 1 / b
        F = np.zeros_like(A)
        rows -= 1
        F[rows, cols] = 1

        A_hat = np.zeros((q + 2*d - 2, p + 2 * d - 2))
        A_hat[:d-1, :d-1] = np.identity(d - 1)
        A_hat[d-1:q+d-1, d-1:p+d-1] = A
        A_hat[1-d:, 1-d:] = np.identity(d - 1)

        B_hat = np.zeros_like(A_hat.T)
        B_hat[:d-1, :d-1] = np.identity(d - 1)
        B_hat[d-1:p+d-1, d-1:q+d-1] = B
        B_hat[1-d:, 1-d:] = np.identity(d - 1)

        F_hat = np.zeros((q + 2 * d - 2, p + d - 1))
        F_hat[d-1:l2+d-2, :l2-1] = np.identity(l2 - 1)
        F_hat[d-1:q+d-1, l2-1:p+l2-1] = F
        F_hat[l1-d+1:1-d, l1:] = np.identity(-l1)

        H_hat = np.identity(q + 2 * d - 2)
        H_hat[d-1:1-d, d-1:1-d] = H
        H_hat_inv = np.identity(q + 2 * d - 2)
        H_hat_inv[d-1:1-d, d-1:1-d] = H_inv

        M0 = self.mra.refinement_matrix(j)
        M1 = (-1)**(d+1) * np.sqrt(2) / b * P @ H_hat_inv @ F_hat
        G0 = np.sqrt(2) * B_hat @ H_hat @ P_inv
        G1 = (-1)**(d+1) * b / np.sqrt(2) * F_hat.T @ H_hat @ P_inv
        return M0, M1, G0, G1

    def refinement_matrix(self, j):
        M0, M1, _, G1 = self.initial_completion(j)
        M0_t = self.mra_t.refinement_matrix(j)
        M1 = M1 - M0 @ M0_t.T @ M1

        # Symmetrize
        N1 = M1[:, :2**(j-1)]
        N1_t = G1.T[:, :2**(j-1)]
        H1 = np.hstack((N1, N1[::-1, ::-1]))
        H1_t = np.hstack((N1_t, N1_t[::-1, ::-1]))
        M1 = np.linalg.solve(H1.T @ H1_t, H1.T).T
        M1_t = H1_t

        M0[np.abs(M0) < 1e-9] = 0.
        M1[np.abs(M1) < 1e-9] = 0.
        M0_t[np.abs(M0_t) < 1e-9] = 0.
        M1_t[np.abs(M1_t) < 1e-9] = 0.
        return M0, M1, M0_t, M1_t

    def gramian(self, j):
        G = self.mra.gramian(j + 1)
        _, M1, _, _ = self.refinement_matrix(j)
        return M1.T @ G @ M1

    def inner_product(self, j, nu=0):
        pass
