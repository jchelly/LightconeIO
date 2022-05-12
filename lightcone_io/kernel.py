#!/bin/env python

import numpy as np

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

    # Go to the range [0,1[ from [0,H[
    x = u * kernel_gamma_inv
    
    # Pick the correct branch of the kernel
    temp = int(x * kernel_ivals_f)
    ind = kernel_ivals if temp > kernel_ivals else temp
    coeffs = kernel_coeffs[ind * (kernel_degree + 1):]

    # First two terms of the polynomial ...
    w = coeffs[0] * x + coeffs[1]

    # ... and the rest of them
    for k in range(2, kernel_degree+1):
        w = x * w + coeffs[k]

    w = max(w, 0.)

    return w * kernel_constant * kernel_gamma_inv_dim

