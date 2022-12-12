from math import floor, ceil
import numpy as np
from scipy.interpolate import BSpline, interp1d
from scipy.special import binom
import matplotlib.pyplot as plt


class ScalingFunction:
    def refinement_coeffs(self):
        return self.a

    def __call__(self, x):
        return self.phi(x)

    def plot(self, scale=1., fct=None):
        if fct is None:
            fct = self
        x = np.linspace(self.l1, self.l2, 1000)
        plt.plot(x, scale * fct(x))
        plt.show()


class PrimalScalingFunction(ScalingFunction):
    def __init__(self, d):
        self.d = d
        self.l1 = -floor(d / 2)
        self.l2 = ceil(d / 2)

        # refinement coefficients
        self.a = 2**(1-d) * np.array([binom(d, k - self.l1)
                                      for k in range(self.l1, self.l2 + 1)])

        # knots = np.arange(self.l1, self.l2 + 1)
        # self.phi = BSpline.basis_element(knots, extrapolate=False)
        # print(self.phi.tck)

        # function
        knots = np.concatenate((np.zeros(d - 1),
                                np.arange(d + 1),
                                np.full(d - 1, d))) + self.l1
        coeffs = np.zeros(2 * d - 1)
        coeffs[d-1] = 1.
        self.phi = BSpline(knots, coeffs, d - 1, extrapolate=False)

    def __call__(self, x):
        return np.nan_to_num(self.phi(x))

    def plot(self, scale=1., nu=0):
        fct = None
        if nu > 0:
            fct = self.derivative(nu)
        super().plot(scale=scale, fct=fct)

    def derivative(self, nu=1):
        def deriv(x):
            f = self.phi.derivative(nu)
            return np.nan_to_num(f(x))
        return deriv

    def gramian(self):
        d = self.d
        a = self.refinement_coeffs()
        a_corr = np.correlate(a, a, 'full')  # (2*d+1,) array

        def entry(k, m):
            shift = 2 * k - m
            if abs(shift) > d:
                return 0
            return a_corr[shift+d]
        A = np.array([[entry(k, m) for m in range(-d + 1, d)]
                      for k in range(-d + 1, d)])
        w, v = np.linalg.eig(A)
        idx = np.argmax(w)
        assert np.isclose(w[idx], 2.)
        g = v[:, idx]
        g /= g.sum()  # integrals of phi(x)*phi(x-k)
        return g

    def inner_product(self, nu=0):
        d0 = self.d - nu
        g = PrimalScalingFunction(d0).gramian()

        for d in range(d0 + 1, self.d + 1):
            gd = np.zeros(2*d-1)
            gd[1:-1] = 2 * g
            gd[:-2] -= g
            gd[2:] -= g
            g = gd
        return g


class DualScalingFunction(ScalingFunction):
    def __init__(self, d, d_t):
        assert (d + d_t) % 2 == 0
        self.d = d_t
        self.l1 = -floor(d / 2) - d_t + 1
        self.l2 = ceil(d / 2) + d_t - 1
        K = (d + d_t) // 2

        # refinement coefficients
        def entry(k):
            res = 0
            for n in range(K):
                for i in range(2 * n + 1):
                    res += 2**(1 - d_t - 2 * n) * (-1)**(n + i)\
                           * binom(d_t, k + floor(d_t / 2) - i + n)\
                           * binom(K - 1 + n, n) * binom(2 * n, i)
            return res
        self.a = \
            np.array([entry(k) for k in range(self.l1, self.l2 + 1)])

        # function
        assert np.isclose(self.a.sum(), 2.)
        level = 10
        eta = np.array([1.])
        for _ in range(level):
            temp = np.zeros(eta.size * 2 + 1)
            temp[1::2] = eta
            eta = np.convolve(temp, self.a)
        X = np.arange(eta.size)
        X = (X - X.mean()) * 2**(-level) + d % 2 / 2
        self.phi = interp1d(X, eta, bounds_error=False, fill_value=0.)


class MotherWavelet(PrimalScalingFunction):
    def __init__(self, d, d_t):
        self.d = d
        self.d_t = d_t
        self.l1 = 1 - (d + d_t) // 2
        self.l2 = (d + d_t) // 2

        # function
        sf_t = DualScalingFunction(d, d_t)
        a_t = sf_t.refinement_coeffs()
        self.b = (-1)**np.arange(a_t.size) * np.flip(a_t)
        knots = np.concatenate((np.zeros(d - 1),
                                np.arange(2 * (d + d_t - 1) + 1) / 2,
                                np.full(d - 1, d + d_t - 1))) + self.l1
        coeffs = np.zeros(3 * d + 2 * d_t - 3)
        coeffs[d-1:-d+1] = self.b
        self.phi = BSpline(knots, coeffs, d - 1, extrapolate=False)

    def refinement_coeffs(self):
        raise NotImplementedError

    def gramian(self):
        return self.inner_product()

    def inner_product(self, nu=0):
        d = self.d
        d_t = self.d_t
        offset = self.b.size - 1
        b_corr = np.correlate(self.b, self.b, 'full')  # (2*offset+1,) array

        def entry(k, m):
            shift = 2 * k - m
            if abs(shift) > offset:
                return 0
            return b_corr[shift+offset]
        B = np.array([[entry(k, m) for m in range(-d + 1, d)]
                      for k in range(-d - d_t + 2, d + d_t - 1)])
        return 2**(2*nu-1) * B @ super().inner_product(nu)
