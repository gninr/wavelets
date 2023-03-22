# Biorthogonal Spline-Wavelets on the Interval

This repository contains the code generating the numerical results in Chapter 3 of the thesis "Sensitivity-Guided Shape Reconstruction".

## Dependencies

The implementation depends on the following Python packages:

- NumPy

- Matplotlib

- SciPy

## File description

- `scaling_function.py`

    This module implements primal scaling functions, dual scaling functions and mother wavelets on the real line.

- `scaling_function.ipynb`

    This notebook presents the usage and the validation of functions in `scaling_function.py`.

- `wavelet1d.py`

    This module implements primal MRA, dual MRA, single-scale and multi-scale wavelet bases on the interval.

- `wavelet1d.ipynb`

    This notebook presents the usage and the validation of functions in `wavelet1d.py`.

## References

- Primbs, Miriam. 2006. “Stabile biortogonale Spline-Waveletbasen auf dem Intervall.”

- Primbs, Miriam. 2008. “On the Computation of Gramian Matrices for Refinable Bases on the Interval.” International Journal of Wavelets, Multiresolution and Information Processing 06 (03): 459–79. https://doi.org/10.1142/S0219691308002422.

- Primbs, Miriam. 2010. “New Stable Biorthogonal Spline-Wavelets on the Interval.” Results in Mathematics 57 (1–2): 121–62. https://doi.org/10.1007/s00025-009-0008-6.

- Urban, Karsten. 2009. Wavelet Methods for Elliptic Partial Differential Equations. Numerical Mathematics and Scientific Computation. Oxford ; New York: Oxford University Press.