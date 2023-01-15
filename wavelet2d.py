from wavelet1d import MultiscaleWaveletBasis
from functools import reduce
from itertools import product
import numpy as np
import matplotlib.pyplot as plt


class MultidimensionalWaveletBasis:
    def __init__(self, orders, dual_orders, bcs=None):
        self.dim = len(orders)
        if bcs is None:
            bcs = [False] * self.dim
        self.j0 = -1
        self.w1d = []
        for dim in range(self.dim):
            d = orders[dim]
            d_t = dual_orders[dim]
            bc = bcs[dim]
            w1d = MultiscaleWaveletBasis(d, d_t, bc)
            self.j0 = max(self.j0, w1d.j0)
            self.w1d.append(w1d)

    def basis_functions(self, J, nu=0):
        assert J >= self.j0
        bs = []

        def bspline_tp(b1d):
            def b(x):
                res = np.ones(x[0].shape)
                for dim in range(self.dim):
                    res *= np.nan_to_num(b1d[dim](x[dim]))
                return res
            return b

        # scaling functions
        b1d = [w1d.mra.basis_functions(self.j0) for w1d in self.w1d]
        bs.append(list(map(bspline_tp, product(*b1d))))

        for j in range(self.j0, J):
            b_j = [[w1d.mra.basis_functions(j), w1d.wb.basis_functions(j)]
                   for w1d in self.w1d]
            b_j = list(product(*b_j))
            b_j.pop(0)
            bs.append([list(map(bspline_tp, product(*b1d))) for b1d in b_j])
        return bs

    def support(self, j, e, k):
        supp = []
        for dim in range(self.dim):
            w1d = self.w1d[dim]
            supp1d = w1d.wb.support(j, k[dim]) if e[dim]\
                else w1d.mra.support(j, k[dim])
            supp.append(supp1d)
        return supp

    def plot(self, bs, j, e, k):
        if e == (0,) * self.dim:
            if j == self.j0:
                idx = 0
                for dim in range(self.dim):
                    mra = self.w1d[dim].mra
                    n = 2**j + mra.d - 1 - 2 * mra.bc
                    assert 0 <= k[dim] < n
                    idx += k[dim]
                    if dim < self.dim - 1:
                        idx *= n
                b_tp = bs[0][idx]
            else:
                raise IndexError
        else:
            n = []
            for dim in range(self.dim):
                mra = self.w1d[dim].mra
                n.append([2**j + mra.d - 1 - 2 * mra.bc, 2**j])
            n = list(product(*n))

            s = j - self.j0

            part = e[0]
            for dim in range(self.dim - 1):
                part *= 2
                part += e[dim+1]
            n = n[part]

            idx = 0
            for dim in range(self.dim):
                assert 0 <= k[dim] < n[dim]
                idx += k[dim]
                if dim < self.dim - 1:
                    idx *= n[dim]
            b_tp = bs[s+1][part-1][idx]

        supp = self.support(j, e, k)
        xmin, xmax = supp[0]
        ymin, ymax = supp[1]
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X = np.linspace(xmin, xmax, 100)
        Y = np.linspace(ymin, ymax, 100)
        X, Y = np.meshgrid(X, Y)
        Z = b_tp([X, Y])
        ax.plot_surface(X, Y, Z, linewidth=0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def refinement_matrix(self, s, j0=None):
        def tracy_singh(A, B):
            return [np.kron(a, b) for a, b in product(A, B)]

        n = [2**(j0+s) + w1d.d - 1 - 2 * w1d.bc for w1d in self.w1d]
        n = reduce(lambda x, y: x*y, n)
        T = np.identity(n)

        for j in range(j0, j0 + s):
            M = []
            for dim in range(self.dim):
                w1d = self.w1d[dim]
                M0 = w1d.mra.refinement_matrix(j)
                M1 = w1d.wb.refinement_matrix(j)
                M.append([M0, M1])
            M = reduce(tracy_singh, M)
            M = np.hstack(M)
            n = M.shape[0]
            T[:n, :n] = M @ T[:n, :n]
        return T

    def gramian(self, s, j0=None):
        if j0 is None:
            j0 = self.j0
        else:
            assert j0 >= self.j0

        G = [w1d.mra.gramian(j0 + s) for w1d in self.w1d]
        G = reduce(np.kron, G)
        T = self.refinement_matrix(s, j0)
        return T.T @ G @ T

    def inner_product(self, s, j0=None, nu=0):
        if j0 is None:
            j0 = self.j0
        else:
            assert j0 >= self.j0

        A = [w1d.mra.inner_product(j0 + s) for w1d in self.w1d]
        A = reduce(np.kron, A)
        T = self.refinement_matrix(s, j0)
        return T.T @ A @ T
