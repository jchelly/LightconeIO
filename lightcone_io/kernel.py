#!/bin/env python

import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate

#
# Wendland C2 kernel as used in FLAMINGO
#
# Taken from Swift's kernel_hydro.h.
#
kernel_degree=5
kernel_ivals=1
kernel_gamma=1.936492
kernel_constant=21. * np.pi / 2.
kernel_coeffs = np.asarray([4., -15., 20., -10., 0.,  1.,
                            0.,   0.,  0.,   0., 0.,  0.], dtype=float)
kernel_gamma_dim=(kernel_gamma * kernel_gamma * kernel_gamma)
kernel_gamma_inv_dim=(1. / (kernel_gamma * kernel_gamma * kernel_gamma))
kernel_gamma_inv=1./kernel_gamma
kernel_ivals_f=float(kernel_ivals)
kernel_root=kernel_coeffs[kernel_degree] * kernel_constant * kernel_gamma_inv_dim
kernel_norm=(4./3*np.pi) * kernel_gamma_dim


def kernel_eval(u):
    """
    Numpy vectorized kernel function
    """
    u = np.asarray(u, dtype=float)

    # Go to the range [0,1[ from [0,H[
    x = u * kernel_gamma_inv
    
    # Pick the correct branch of the kernel
    temp = (x * kernel_ivals_f).astype(int)
    ind = np.where(temp > kernel_ivals, kernel_ivals, temp)

    coeffs_index = ind * (kernel_degree + 1)

    # First two terms of the polynomial ...
    w = kernel_coeffs[coeffs_index] * x + kernel_coeffs[coeffs_index+1]

    # ... and the rest of them
    for k in range(2, kernel_degree+1):
        w = x * w + kernel_coeffs[coeffs_index+k]

    w = np.asarray(w)
    w[w < 0.0] = 0.0

    return w * kernel_constant * kernel_gamma_inv_dim


def projected_kernel_integrate(u):
    
    u = float(u) # not implemented for array input

    # Avoid integration errors if u is out of range
    if u > kernel_gamma:
        return 0.0

    projected_kernel_integrand = lambda qz : kernel_eval(np.sqrt(u**2+qz**2))

    qz_max = np.sqrt(kernel_gamma**2-u**2)
    qz_min = -qz_max

    result, abserror = integrate.quad(projected_kernel_integrand, qz_min, qz_max)
    return result


class ProjectedKernel:
    """
    Evaluate the projected kernel by interpolating a tabulated function
    """
    def __init__(self):
        """Tabulate the projected kernel"""
        npoints = 1000
        x = np.linspace(0.0, kernel_gamma, npoints)
        y = np.zeros_like(x)
        for i in range(npoints):
            y[i] = projected_kernel_integrate(x[i])
        self.interp = interpolate.interp1d(x, y)

    def __call__(self, u):
        """Evaluate the projected kernel at the input radii"""
        u = np.asarray(u)
        if np.any(u < 0.0):
            raise ValueError("Can't evaluate projected kernel at negative radius!")
        in_range = u < kernel_gamma
        result = np.zeros_like(u, dtype=float)
        result[in_range] = self.interp(u[in_range])
        return result
