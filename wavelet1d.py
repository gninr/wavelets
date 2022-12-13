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

    def refinement_matrix(self, j):
        d = self.d
        A = np.zeros((2**(j+1) - d + 1, 2**j - d + 1))
        for k in range(2**j - d + 1):
            A[2*k:2*k+d+1, k] = self.sf.refinement_coeffs()

        M0 = np.zeros((2**(j+1) + d - 1, 2**j + d - 1))
        M0[:2*d-2, :d-1] = self.ML
        M0[d-1:2**(j+1), d-1:2**j] = A
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
        D_t = (D1 @ D2 @ D3)[:, ::-1]
        block1 = np.empty((d + 3 * d_t - 3, d_t))
        block1[:d_t, :] = \
            D_t.T @ np.diag([2**(-r) for r in range(d_t)])
        block1[d_t:, :] = np.array(
            [[beta(n-l1_t, r) for r in range(d_t)] for n in range(d+2*d_t-3)]
        )
        ML_t[d-2:, d-2:] = block1 @ np.linalg.inv(D_t.T)

        def compute_gramian():
            n = ML_full.shape[1]
            UL = ML_full[:n, :]
            LL = ML_full[n:, :]
            UL_t = ML_t[:n, :]
            LL_t = ML_t[n:, :]
            lhs = 2 * np.identity(n**2) - np.kron(UL_t.T, UL.T)
            rhs = (LL.T @ LL_t).reshape(-1, order='F')
            gamma = np.linalg.solve(lhs, rhs)
            return gamma.reshape((n, n), order='F')

        gramian_full = np.identity(2 * d + 3 * d_t - 5)
        for k in range(d - 3, -1, -1):
            gramian_full[:d+d_t-2, :d+d_t-2] = compute_gramian()
            B_k = ML_full[:, :k+d].T @ gramian_full[:, k+1:2*k+d+1] / 2.

            delta = np.zeros(k+d)
            delta[k] = 1
            ML_t[k+1:2*k+d+1, k] = np.linalg.solve(B_k, delta)

        # Biorthogonalization

        gramian = compute_gramian()
        ML_t[:d+d_t-2, :d+d_t-2] = gramian @ ML_t[:d+d_t-2, :d+d_t-2]
        ML_t = ML_t @ np.linalg.inv(gramian)
        ML_t[np.abs(ML_t) < 1e-9] = 0.
        self.ML = ML_t

    def refinement_matrix(self, j):
        d = self.d
        d_t = self.d_t
        A_t = np.zeros((2**(j+1) - d - 2 * d_t + 3, 2**j - d - 2 * d_t + 3))
        for k in range(2**j - d - 2 * d_t + 3):
            A_t[2*k:2*k+d+2*d_t-1, k] = self.sf_t.refinement_coeffs()

        M0_t = np.zeros((2**(j+1) + d - 1, 2**j + d - 1))
        M0_t[:2*d+3*d_t-5, :d+d_t-2] = self.ML
        M0_t[d+d_t-2:2**(j+1)-d_t+1, d+d_t-2:2**j-d_t+1] = A_t
        M0_t[-(2*d+3*d_t-5):, -(d+d_t-2):] = self.ML[::-1, ::-1]

        return 1 / np.sqrt(2) * M0_t
