from scaling_function import \
    PrimalScalingFunction, DualScalingFunction, MotherWavelet
from math import ceil, factorial
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import solve_triangular
from scipy.special import binom
import matplotlib.pyplot as plt


class PrimalMRA:
    def __init__(self, d, bc=False):
        assert d > 1
        self.d = d
        self.bc = bc
        self.sf = PrimalScalingFunction(d)
        self.compute_ML()

    def basis_functions(self, j, nu=0, from_refine_mat=False):
        d = self.d
        bc = self.bc
        n = 2**j + d - 1
        bs = []

        def bspline(knots, coeffs, d):
            b = BSpline(knots, coeffs, d - 1, extrapolate=False)
            if nu > 0:
                b = b.derivative(nu)
            return lambda x: np.nan_to_num(b(x))

        if from_refine_mat:
            M0 = self.refinement_matrix(j)
            m = 2**(j+1) + d - 1
            knots = np.concatenate((np.zeros(d - 1),
                                    np.linspace(0, 1, 2**(j+1) + 1),
                                    np.ones(d - 1)))
            if bc:
                for k in range(n - 2):
                    coeffs = np.zeros(m)
                    coeffs[1:-1] = 2**((j+1)/2) * M0[:, k]
                    bs.append(bspline(knots, coeffs, d))
            else:
                for k in range(n):
                    coeffs = 2**((j+1)/2) * M0[:, k]
                    bs.append(bspline(knots, coeffs, d))
        else:
            knots = np.concatenate((np.zeros(d - 1),
                                    np.linspace(0, 1, 2**j + 1),
                                    np.ones(d - 1)))
            for k in range(bc, n - bc):
                coeffs = np.zeros(n)
                coeffs[k] = 2**(j/2)
                bs.append(bspline(knots, coeffs, d))
        return bs

    def support(self, j, k):
        d = self.d
        k += self.bc
        return (max(2**(-j) * (k-d+1), 0), min(2**(-j) * (k+1), 1))

    def plot(self, j, k=None, nu=0, from_refine_mat=False):
        bs = self.basis_functions(j, nu, from_refine_mat)
        x = np.linspace(0, 1, 1000)

        if k is None:
            for b in bs:
                plt.plot(x, b(x))
        else:
            plt.plot(x, bs[k](x))
            supp = self.support(j, k)
            plt.plot(supp, [0, 0], 'ro')
            print("support", supp)
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

        ML = solve_triangular(B2, B1)
        ML[np.abs(ML) < 1e-9] = 0.
        if self.bc:
            self.ML = ML[1:, 1:]
        else:
            self.ML = ML

    def compute_A(self, j):
        d = self.d
        a = self.sf.refinement_coeffs()
        n = a.size
        A = np.zeros((2**(j+1) - d + 1, 2**j - d + 1))
        for k in range(A.shape[1]):
            A[2*k:2*k+n, k] = a
        return A

    def refinement_matrix(self, j):
        d = self.d
        bc = self.bc
        M0 = np.zeros((2**(j+1) + d - 1 - 2 * bc, 2**j + d - 1 - 2 * bc))
        m, n = M0.shape
        ML = self.ML
        mL, nL = ML.shape
        M0[:mL, :nL] = ML
        M0[nL:m-nL, nL:n-nL] = self.compute_A(j)
        M0[m-mL:, n-nL:] = ML[::-1, ::-1]

        return 1 / np.sqrt(2) * M0

    def gramian(self, j):
        d = self.d
        bc = self.bc
        G = np.zeros((2**j + d - 1, 2**j + d - 1))

        g = self.sf.gramian()
        for k in range(2**j - d + 1):
            G[k+d-1, k:k+2*d-1] = g
        if bc:
            G = G[1:-1, 1:-1]

        M0 = self.refinement_matrix(j)
        m = 2 * d - 2 - bc
        n = m + 2 * d - 2
        U = M0[:m, :m]
        L = M0[m:n, :m]
        lhs = np.identity(m**2) - np.kron(U.T, U.T)
        rhs = (U.T @ G[:m, m:n] @ L
               + L.T @ G[m:n, :m] @ U
               + L.T @ G[m:n, m:n] @ L).reshape(-1, order='F')
        GL = np.linalg.solve(lhs, rhs).reshape((m, m), order='F')
        G[:m, :m] = GL
        G[-m:, -m:] = GL[::-1, ::-1]

        return G

    def inner_product(self, j, nu=0):
        d0 = self.d - nu
        if d0 == 1:
            A = np.identity(2**j)
        else:
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
        if self.bc:
            A = A[1:-1, 1:-1]
        return A


class DualMRA:
    def __init__(self, d, d_t, bc=False):
        self.d = d
        self.d_t = d_t
        self.j0 = ceil(np.log2(d + 2 * d_t - 3) + 1)
        self.bc = bc
        self.sf = PrimalScalingFunction(d)
        self.sf_t = DualScalingFunction(d, d_t)
        self.mra = PrimalMRA(d, bc)
        self.compute_ML()

    def compute_ML(self):
        d = self.d
        d_t = self.d_t
        bc = self.bc
        l1 = self.sf.l1
        l2 = self.sf.l2
        l1_t = self.sf_t.l1
        l2_t = self.sf_t.l2
        a = self.sf.refinement_coeffs()
        a_t = self.sf_t.refinement_coeffs()
        if bc and d == 2:
            ML_t = np.zeros((5 * d_t,  2 * d_t))
        else:
            ML_t = np.zeros((2 * d + 3 * d_t - 5 - bc, d + d_t - 2 - bc))
        mL, nL = ML_t.shape

        ML = self.mra.ML
        m, n = ML.shape
        ML_full = np.zeros_like(ML_t)
        ML_full[:m, :n] = ML
        for k in range(n, nL):
            ML_full[2*k-n:2*k-n+d+1, k] = a
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

        if bc and d == 2:
            # Compute beta_{n,r}
            def beta(n, r):
                res = 0
                for k in range(ceil((n-d_t) / 2), d_t + 1):
                    res += alpha(k, r) * a_t[n-2*k+d_t]
                return res

            ML_t[:d_t, :d_t] = np.diag([2**(-r) for r in range(d_t)])
            ML_t[d_t, :d_t] = np.array(
                [2**(-r) * alpha(d_t + 1, r) for r in range(d_t)])
            ML_t[d_t+1:3*d_t, :d_t] = np.array(
                [[beta(n+d_t+2, r) for r in range(d_t)]
                 for n in range(2 * d_t - 1)])
            for k in range(d_t, 2 * d_t):
                ML_t[2*k-d_t+1:2*k+d_t+2, k] = a_t

            # Biorthogonalize

            gramian = compute_gramian()
            ML_t[:2*d_t, :] = gramian @ ML_t[:2*d_t, :]
            ML_t = np.linalg.solve(gramian.T, ML_t.T).T
            ML_t = ML_t[:3*d_t, :d_t]

        else:
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
            ML_t[d-2-bc:, d-2-bc:] = \
                solve_triangular(D_t, rhs, lower=True)[::-1, :].T

            # Compute block of ML_t corresponding to k = 0, ..., d-3

            gramian_full = np.identity(mL)
            for k in range(d - 3 - bc, -1, -1):
                gramian_full[:nL, :nL] = compute_gramian()
                B_k = ML[:, :k+d].T @ gramian_full[:, k+1:2*k+d+1] / 2.

                delta = np.zeros(k + d)
                delta[k] = 1
                ML_t[k+1:2*k+d+1, k] = np.linalg.solve(B_k, delta)

            # Biorthogonalize

            gramian = np.triu(compute_gramian())
            gramian[:d-2-bc, :d-2-bc] = np.identity(d - 2 - bc)
            ML_t[:d+d_t-2-bc, :] = gramian @ ML_t[:d+d_t-2-bc, :]
            ML_t = solve_triangular(gramian.T, ML_t.T, lower=True).T

        ML_t[np.abs(ML_t) < 1e-9] = 0.
        self.ML = ML_t

    def compute_A(self, j):
        assert j >= self.j0
        d = self.d
        d_t = self.d_t
        a_t = self.sf_t.refinement_coeffs()
        n = a_t.size
        if self.bc and d == 2:
            A_t = np.zeros((2**(j+1) - 2 * d_t - 3,
                            2**j - 2 * d_t - 1))
        else:
            A_t = np.zeros((2**(j+1) - d - 2 * d_t + 3,
                            2**j - d - 2 * d_t + 3))
        for k in range(A_t.shape[1]):
            A_t[2*k:2*k+n, k] = a_t
        return A_t

    def refinement_matrix(self, j):
        d = self.d
        bc = self.bc
        M0_t = np.zeros((2**(j+1) + d - 1 - 2 * bc, 2**j + d - 1 - 2 * bc))
        m, n = M0_t.shape
        ML_t = self.ML
        mL, nL = ML_t.shape
        A_t = self.compute_A(j)
        shift = (m - A_t.shape[0]) // 2
        M0_t[:mL, :nL] = ML_t
        M0_t[shift:m-shift, nL:n-nL] = A_t
        M0_t[m-mL:, n-nL:] = ML_t[::-1, ::-1]

        return 1 / np.sqrt(2) * M0_t


class WaveletBasis:
    def __init__(self, d, d_t, bc=False):
        self.d = d
        self.d_t = d_t
        if bc and d == 2:
            self.j0 = ceil(np.log2(3 / 2 * d_t + 1) + 1)
        else:
            self.j0 = ceil(np.log2(d + d_t - 2) + 1)
        self.bc = bc
        self.sf = PrimalScalingFunction(d)
        self.sf_t = DualScalingFunction(d, d_t)
        self.mw = MotherWavelet(d, d_t)
        self.mra = PrimalMRA(d, bc)
        self.mra_t = DualMRA(d, d_t, bc)
        self.compute_GL()

    def basis_functions(self, j, nu=0):
        d = self.d
        bs = []

        def bspline(knots, coeffs, d):
            b = BSpline(knots, coeffs, d - 1, extrapolate=False)
            if nu > 0:
                b = b.derivative(nu)
            return lambda x: np.nan_to_num(b(x))

        M1 = self.refinement_matrix(j)
        knots = np.concatenate((np.zeros(d - 1),
                                np.linspace(0, 1, 2**(j+1) + 1),
                                np.ones(d - 1)))
        if self.bc:
            m = 2**(j+1) + d - 1
            for k in range(2**j):
                coeffs = np.zeros(m)
                coeffs[1:-1] = 2**((j+1)/2) * M1[:, k]
                bs.append(bspline(knots, coeffs, d))
        else:
            for k in range(2**j):
                coeffs = 2**((j+1)/2) * M1[:, k]
                bs.append(bspline(knots, coeffs, d))
        return bs

    def support(self, j, k):
        # primal
        d = self.d
        d_t = self.d_t
        n = (d + d_t - 2) // 2  # number of boundary wavelets
        if self.bc and d == 2:
            if k < d_t // 2:
                return (0, 2**(-j) * (d_t + 1))
            elif k >= 2**j - d_t // 2:
                return (1 - 2**(-j) * (d_t + 1), 1)
            else:
                return (max(2**(-j) * (k-n), 0),
                        min(2**(-j) * (k-n+d+d_t-1), 1))
        else:
            if k < n:
                return (0, 2**(-j) * (d + d_t - 2))
            elif k >= 2**j - n:
                return (1 - 2**(-j) * (d + d_t - 2), 1)
            else:
                return (max(2**(-j) * (k-n), 0),
                        min(2**(-j) * (k-n+d+d_t-1), 1))

    def plot(self, j, k=None, nu=0, boundary=False):
        bs = self.basis_functions(j, nu)
        if boundary:
            d = self.d
            d_t = self.d_t
            if self.bc and d == 2:
                L = d_t + 1
                assert k is not None and k < d_t // 2
            else:
                L = self.d + self.d_t - 2
                assert k is not None and k < L // 2
            x = np.linspace(0, L, 1000)
            plt.plot(x, 2**(-j/2) * bs[k](2**(-j) * x))
        else:
            x = np.linspace(0, 1, 1000)
            if k is None:
                for b in bs:
                    plt.plot(x, b(x))
            else:
                plt.plot(x, bs[k](x))
                supp = self.support(j, k)
                plt.plot(supp, [0, 0], 'ro')
                print("support", supp)
        plt.show()

    def initial_completion(self, j):
        d = self.d
        l1 = self.sf.l1
        l2 = self.sf.l2
        p = 2**j - d + 1
        q = 2**(j+1) - d + 1

        ML = self.mra.ML
        mL, nL = ML.shape
        P = np.identity(q + 2 * nL)
        P[:mL, :nL] = ML
        P[q+2*nL-mL:, q+nL:] = ML[::-1, ::-1]

        ML_inv = np.linalg.inv(P[:mL, :mL])
        P_inv = np.identity(q + 2 * nL)
        P_inv[:mL, :mL] = ML_inv
        P_inv[q+2*nL-mL:, q+2*nL-mL:] = ML_inv[::-1, ::-1]

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

        A_hat = np.zeros((q + 2 * nL, p + 2 * nL))
        A_hat[:nL, :nL] = np.identity(nL)
        A_hat[nL:q+nL, nL:p+nL] = A
        A_hat[q+nL:, p+nL:] = np.identity(nL)

        B_hat = np.zeros_like(A_hat.T)
        B_hat[:nL, :nL] = np.identity(nL)
        B_hat[nL:p+nL, nL:q+nL] = B
        B_hat[p+nL:, q+nL:] = np.identity(nL)

        F_hat = np.zeros((q + 2 * nL, 2**j))
        F_hat[nL:l2-1+nL, :l2-1] = np.identity(l2 - 1)
        F_hat[nL:q+nL, l2-1:p+l2-1] = F
        F_hat[q+nL+l1:q+nL, 2**j+l1:] = np.identity(-l1)

        H_hat = np.identity(q + 2 * nL)
        H_hat[nL:q+nL, nL:q+nL] = H
        H_hat_inv = np.identity(q + 2 * nL)
        H_hat_inv[nL:q+nL, nL:q+nL] = H_inv

        M0 = self.mra.refinement_matrix(j)
        M1 = (-1)**(d+1) * np.sqrt(2) / b * P @ H_hat_inv @ F_hat
        G0 = np.sqrt(2) * B_hat @ H_hat @ P_inv
        G1 = (-1)**(d+1) * b / np.sqrt(2) * F_hat.T @ H_hat @ P_inv
        return M0, M1, G0, G1

    def compute_GL(self):
        d = self.d
        d_t = self.d_t
        bc = self.bc
        j0 = self.mra_t.j0
        M0, M1, _, G1 = self.initial_completion(j0)
        M0_t = self.mra_t.refinement_matrix(j0)
        M1 = M1 - M0 @ M0_t.T @ M1
        M1_t = G1.T

        if bc and d == 2:
            GL = np.sqrt(2) * M1[:2*d_t+1, :d_t//2]
            GL_t = np.sqrt(2) * M1_t[:2*d_t+1, :d_t//2]
        else:
            GL = np.sqrt(2) * M1[:2*(d+d_t-2)-bc, :(d+d_t-2)//2]
            GL_t = np.sqrt(2) * M1_t[:2*d+d_t-3-bc, :(d+d_t-2)//2]

        GL[np.abs(GL) < 1e-9] = 0.
        self.GL = GL

        GL_t[np.abs(GL_t) < 1e-9] = 0.
        self.GL_t = GL_t

    def compute_B(self, j):
        d = self.d
        d_t = self.d_t
        b = self.mw.refinement_coeffs()
        B = np.zeros((2**(j+1) - d + 1, 2**j - d - d_t + 2))
        for k in range(2**(j-1) - (d + d_t - 2) // 2):
            B[2*k:2*k+d+2*d_t-1, k] = b
        B += B[::-1, ::-1]
        return B

    def refinement_matrix(self, j, full=False):
        assert j >= self.j0

        if full:
            M0, M1, _, G1 = self.initial_completion(j)
            M0_t = self.mra_t.refinement_matrix(j)
            M1 = M1 - M0 @ M0_t.T @ M1
            M1_t = G1.T

            # Symmetrize
            N1 = M1[:, :2**(j-1)]
            N1_t = M1_t[:, :2**(j-1)]
            H1 = np.hstack((N1, N1[::-1, ::-1]))
            H1_t = np.hstack((N1_t, N1_t[::-1, ::-1]))
            M1 = np.linalg.solve(H1.T @ H1_t, H1.T).T
            M1_t = H1_t

            M0[np.abs(M0) < 1e-9] = 0.
            M1[np.abs(M1) < 1e-9] = 0.
            M0_t[np.abs(M0_t) < 1e-9] = 0.
            M1_t[np.abs(M1_t) < 1e-9] = 0.
            return M0, M1, M0_t, M1_t

        else:
            d = self.d
            bc = self.bc
            M1 = np.zeros((2**(j+1) + d - 1 - 2 * bc, 2**j))
            m, n = M1.shape
            GL = self.GL
            mL, nL = GL.shape
            B = self.compute_B(j)
            shift = (m - B.shape[0]) // 2
            M1[:mL, :nL] = GL
            M1[shift:m-shift, nL:n-nL] = B
            M1[m-mL:, n-nL:] = GL[::-1, ::-1]

            return 1 / np.sqrt(2) * M1

    def gramian(self, j):
        G = self.mra.gramian(j + 1)
        M1 = self.refinement_matrix(j)
        return M1.T @ G @ M1


class MultiscaleWaveletBasis:
    def __init__(self, d, d_t, bc=False):
        self.d = d
        self.d_t = d_t
        self.bc = bc
        self.mra = PrimalMRA(d, bc)
        self.wb = WaveletBasis(d, d_t, bc)
        self.j0 = self.wb.j0

    def refinement_matrix(self, s, j0=None):
        if j0 is None:
            j0 = self.j0
        else:
            assert j0 >= self.j0

        d = self.d
        T = np.identity(2**(j0+s) + d - 1 - 2 * self.bc)
        for j in range(j0, j0 + s):
            M0 = self.mra.refinement_matrix(j)
            M1 = self.wb.refinement_matrix(j)
            M = np.hstack((M0, M1))
            n = M.shape[0]
            T[:n, :n] = M @ T[:n, :n]
        return T

    def gramian(self, s, j0=None):
        if j0 is None:
            j0 = self.j0
        else:
            assert j0 >= self.j0

        G = self.mra.gramian(j0 + s)
        T = self.refinement_matrix(s, j0)
        return T.T @ G @ T

    def inner_product(self, s, j0=None, nu=0):
        if j0 is None:
            j0 = self.j0
        else:
            assert j0 >= self.j0

        A = self.mra.inner_product(j0 + s, nu)
        T = self.refinement_matrix(s, j0)
        return T.T @ A @ T
