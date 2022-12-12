from scaling_function import *
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt


class PrimalMRA:
    def __init__(self, d):
        self.d = d
        self.compute_ML()
        self.a = PrimalScalingFunction(d).refinement_coeffs()

    def basis_functions(self, j, from_refine_mat=False):
        d = self.d
        n = 2**j + d - 1
        bs = []

        if from_refine_mat:
            M0 = self.refinement_matrix(j)
            knots = np.concatenate((np.zeros(d - 1),
                                    np.linspace(0, 1, 2**(j+1) + 1),
                                    np.ones(d - 1)))
            for k in range(n):
                coeffs = 2**((j+1)/2) * M0[:, k]
                bs.append(BSpline(knots, coeffs, d - 1, extrapolate=False))
        else:
            knots = np.concatenate((np.zeros(d - 1),
                                    np.linspace(0, 1, 2**j + 1),
                                    np.ones(d - 1)))
            for k in range(n):
                coeffs = np.zeros(n)
                coeffs[k] = 2**(j/2)
                bs.append(BSpline(knots, coeffs, d - 1, extrapolate=False))
        return bs

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
            A[2*k:2*k+d+1, k] = self.a

        M = np.zeros((2**(j+1) + d - 1, 2**j + d - 1))
        M[:2*d-2, :d-1] = self.ML
        M[d-1:2**(j+1), d-1:2**j] = A
        M[2-2*d:, 1-d:] = self.ML[::-1, ::-1]

        return 1 / np.sqrt(2) * M
