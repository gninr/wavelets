from math import floor, ceil
import numpy as np
from scipy.interpolate import BSpline
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

    def __call__(self, x):
        return np.nan_to_num(self.spline(x))

    def plot(self):
        x = np.linspace(self.l1, self.l2, 100)
        plt.plot(x, self.spline(x))
        plt.show()
