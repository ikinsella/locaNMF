
from typing import List


import numpy as np
from scipy import signal
from scipy import ndimage
import scipy.io as sio



def demix_setting():
    """Create default CNMF_E algorithm demix options dictionary

    :return: default CNMF_E algorithm options dictionary
    """

    options = {}

    # TODO: add more options


    return options


def spatial_filtering(spatial_u, settings):
    """Apply spatial filtering on denoised image data spatial_u
    :parameter
        spatial_u: spatial components matrix, [height, width, channels, max_components]
        settings: settings dictionary returned by demix_setting() fcn
    :return: gaussian-filtered spatial components if gaussian kernel width is greater
        than zero, otherwise return spatial_u
    """

    # Gaussian kernel width (approximating size of one neuron)
    gaussian_kernel_width = settings['gaussian_kernel_width']
    gaussian_kernel_truncate = 2 # this part is different from matlab, equivalent to 4

    if gaussian_kernel_width > 0:
        spatial_u_filtered = np.zeros_like(spatial_u)
        shape = list(spatial_u_filtered.shape)
        for i in range(shape[-1]):
            for j in range(shape[-2]):
                ndimage.gaussian_filter(spatial_u[:, :, j, i], sigma=gaussian_kernel_width,
                                        output=spatial_u_filtered[:, :, j, i],
                                        truncate=gaussian_kernel_truncate,
                                        mode='nearest')
        return spatial_u_filtered
    return spatial_u


def pnr(spatial_u, setting):
    """Calculate PNR (Peak Noise Ratio) of denoised image data spatial_u


    :return: PNR of image data Y
    """


    pass


def correlation(Y, setting):
    """Compute correlation image

    :return: Correlation image
    """

    "function Cn = correlation_image(Y, sz, d1, d2, flag_norm, K)"

    pass


def update_ring_model_w(U, V, A, X, W, d1, d2, T, r):
    """Update W matrix using ring model

    :param U: spatial component matrix from denoiser, R(d=d1xd2, N)
    :param V: temporal component matrix from denoiser, R(N, T)
    :param A: spatial component matrix, R(d, K)
    :param X: temporal component matrix (the actual temporal matrix C = XV),  R(K, N)
    :param W: weighting matrix, R(d, d)
    :param d1: x axis frame size
    :param d2: y axis frame size
    :param T: number to time steps along time axis
    :param r: ring radius of ring model
    :return:
        W: updated weighting matrix
        b0: constant baseline of background image
    """

    if not W:
        W = init_w(d1, d2, r)

    dU = U - np.matmul(A, X)
    b0 = np.matmul(dU, np.mean(V, axis=1)/T)
    W = update_w_1p(dU, W)
    return b0, W


def init_w(d1, d2, r):
    """Compute the initial W weighting matrix

    :param d1: x axis frame size
    :param d2: y axis frame size
    :param r: ring radius of ring model
    :return:
    """

    # Compute XY distance tile
    x_tile = np.tile(range(-(r + 1), (r + 2)), [(2 * r + 3), 1])
    y_tile = np.transpose(np.tile(range(-(r + 1), (r + 2)), [(2 * r + 3), 1]))
    xy_tile = np.sqrt(np.multiply(x_tile, x_tile) + np.multiply(y_tile, y_tile))

    # Find ring index
    r_tile = np.ones((2 * r + 3, 2 * r + 3)) * r
    r1_tile = r_tile + 1
    ring = np.logical_and(np.greater_equal(xy_tile, r_tile),
                          np.less(xy_tile, r1_tile))
    ring_idx = np.argwhere(ring)
    ring_idx_T = np.transpose(ring_idx)
    ring_idx_T = ring_idx_T - (r + 1) # shift index so that center has zero index

    # Create a weighting matrix to store initial value, the matrix size is padded
    # r cells along 2nd and 3rd dimensions to avoid out of index
    d_d1_d2_2r = np.zeros((d1 * d2, d1 + 2 * r + 3, d2 + 2 * r + 3))
    for i in range(d1):
        for j in range(d2):
            ij = i * d2 + j
            x_base = i + r
            y_base = j + r
            d_d1_d2_2r[ij, ring_idx_T[0, :] + x_base, ring_idx_T[1, :] + y_base] = 1

    # Grep the central part and reshape
    d3_sub = d_d1_d2_2r[:, r:(r + d1), r:(r + d2)]
    W = np.reshape(d3_sub, (d1*d2, -1))
    return W


def update_w_1p(dU, W, d1, d2):
    """Update weighting matrix W in place

    :param dU: 2D matrix that is equal to U - AX
    :param W: weighting matrix
    :param d1: x axis frame size
    :param d2: y axis frame size
    :return: None,
    """

    d = d1 * d2
    for i in range(d):
        dU = dU[i, :]
        omega = W[i, :] > 0  #use greater than 0, instead of not-equal to 0
        X = dU[omega, :] # omega is row index vector, check whether this OK
        A = np.matmul(X, np.transpose(X))
        b = np.matmul(X, np.transpose(dU))
        W[i, omega] = np.linalg.solve(A, b)


































