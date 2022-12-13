from scaling_function import *
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt


class PrimalMRA:
    def __init__(self, d):
        self.d = d
        self.compute_ML()
        self.sf = PrimalScalingFunction(d)

    def basis_functions(self, j, from_refine_mat=False):
        d = self.d
        n = 2**j + d - 1
        bs = []

        def bspline(knots, coeffs, d):
            b = BSpline(knots, coeffs, d - 1, extrapolate=False)
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

    def plot(self, j, k=None, from_refine_mat=False):
        bs = self.basis_functions(j, from_refine_mat)
        x = np.linspace(0, 1, 1000)

        if k is None:
            for b in bs:
                plt.plot(x, np.nan_to_num(b(x)))
        else:
            plt.plot(x, np.nan_to_num(bs[k](x)))
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

        M = np.zeros((2**(j+1) + d - 1, 2**j + d - 1))
        M[:2*d-2, :d-1] = self.ML
        M[d-1:2**(j+1), d-1:2**j] = A
        M[2-2*d:, 1-d:] = self.ML[::-1, ::-1]

        return 1 / np.sqrt(2) * M

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
