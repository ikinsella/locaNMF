
from typing import List
import copy

import numpy as np
# from scipy import signal

from scipy import ndimage
from scipy.sparse import lil_matrix
# import scipy.io as sio



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

    Y = U - np.matmul(A, X)
    b0 = np.matmul(Y, np.mean(V, axis=1) / T)
    update_w_1p(Y, W, d1, d2)
    return b0, W


def init_w(d1, d2, r):
    """Compute the initial W weighting matrix

    :param d1: x axis frame size
    :param d2: y axis frame size
    :param r: ring radius of ring model
    :return: W weighting matrix of shape [d1, d2]
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
    ring_idx_T = ring_idx_T - (r + 1)  # shift index so that center has zero index

    # Create a weighting matrix to store initial value, the matrix size is padded
    # r cells along 2nd and 3rd dimensions to avoid out of index

    d = d1 * d2
    W = lil_matrix((d, d), dtype=np.float)
    for i in range(d1):
        for j in range(d2):
            ij = i * d2 + j
            x_base, y_base = i + r, j + r
            ring_idx_T2 = copy.deepcopy(ring_idx_T)
            ring_idx_T2[0, :] += x_base
            ring_idx_T2[1, :] += y_base
            selection_0 = np.logical_and(ring_idx_T2[0, :] >= r, ring_idx_T2[0, :] < r + d1)
            selection_1 = np.logical_and(ring_idx_T2[1, :] >= r, ring_idx_T2[1, :] < r + d2)
            selection = np.logical_and(selection_0, selection_1)
            selection_idx = np.argwhere(selection)
            ring_idx_T3 = ring_idx_T2[:, selection_idx[:, 0]]
            ring_idx = (ring_idx_T3[0, :] - r) * d2 + ring_idx_T3[1, :] - r
            W[ij, ring_idx] = 1.0
    return W


def update_w_1p(Y, W, d1, d2):
    """Update weighting matrix W in place

    :param Y: 2D matrix that is equal to U - AX
    :param W: weighting matrix
    :param d1: x axis frame size
    :param d2: y axis frame size
    :return: None, update W in place
    """

    d = d1 * d2
    for i in range(d):
        y = Y[i, :]
        omega = np.argwhere(W[i, :] > 0)[:, 1]  # use greater than 0, instead of not-equal to 0
        X = Y[omega, :]
        A = np.matmul(X, np.transpose(X))
        b = np.matmul(X, np.transpose(y))
        W[i, omega] = np.linalg.solve(A, b)
































