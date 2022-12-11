from math import floor, ceil, isclose
import numpy as np
from scipy.interpolate import BSpline
from scipy.special import binom
import matplotlib.pyplot as plt


class ScalingFunction:
    def __init__(self, d):
        self.d = d
        self.l1 = -floor(d / 2)
        self.l2 = ceil(d / 2)

        # knots = np.arange(self.l1, self.l2 + 1)
        # self.spline = BSpline.basis_element(knots, extrapolate=False)
        # print(self.spline.tck)

        knots = np.concatenate((np.zeros(d - 1),
                                np.arange(d + 1),
                                np.full(d - 1, d))) + self.l1
        coeffs = np.zeros(2*d - 1)
        coeffs[d - 1] = 1.
        self.spline = BSpline(knots, coeffs, d - 1, extrapolate=False)
        self.a = 2**(1-d) * np.array([binom(d, k - self.l1)
                                      for k in range(self.l1, self.l2 + 1)])

    def __call__(self, x):
        return np.nan_to_num(self.spline(x))

    def plot(self, scale=1., nu=0):
        f = self
        if nu > 0:
            f = self.derivative(nu)

        x = np.linspace(self.l1, self.l2, 100)
        plt.plot(x, scale * f(x))
        plt.show()

    def derivative(self, nu=1):
        def deriv(x):
            f = self.spline.derivative(nu=nu)
            return np.nan_to_num(f(x))
        return deriv

    def refinement_coeffs(self, nu=0):
        return self.a * 2**nu

    def inner_product(self, nu=0):
        d = self.d
        a = self.refinement_coeffs(nu)
        a_corr = np.correlate(a, a, 'full')  # (2*d+1,) array

        def entry(k, m):
            shift = 2 * k - m
            if abs(shift) > d:
                return 0
            return a_corr[shift+d]
        A = np.array([[entry(k, m) for m in range(-d+1, d)]
                      for k in range(-d+1, d)])
        w, v = np.linalg.eig(A)
        idx = np.where(np.abs(w - 2.) < 1e-9)[0][0]
        g = v[:, idx]
        g /= g.sum()  # integrals of phi(x)*phi(x-k)
        return g
