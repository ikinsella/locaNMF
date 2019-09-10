
# from typing import list
import copy
import math
import time

import numpy as np
from scipy import ndimage
from scipy.sparse import lil_matrix
# import skimage
from skimage.morphology import disk as sk_disk
import cv2 as cv

import superpixel_analysis


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


def init_background(U, V, W, d1, d2, r, T):
    """Initialize background

    :param U: spatial component matrix from denoiser, R(d=d1xd2, N)
    :param V: temporal component matrix from denoiser, R(N, T)
    :param W: weighting matrix, R(d, d)
    :param d1: x axis frame size
    :param d2: y axis frame size
    :param r: ring radius of ring model
    :param T:
    :return:
    """

    U_ax, W = init_ring_model(U, d1, d2, r)

    P1 = np.reshape(U - W.dot(U - U_ax), (d1, d2, -1))
    P1 = np.reshape(P1, (d1*d2, -1), order='F') # init_neurons() use F order
    P2 = np.transpose(V)
    P0 = np.reshape(np.matmul(U - W.dot(U - U_ax), V), (d1, d2, T), order='F')
    A, X = init_neurons(P0, P1, P2)
    W, b0 = update_ring_model_w(U, V, A, X, W, d1, d2, T, r)
    return A, X, W, b0




def init_ring_model(U, d1, d2, r):
    """Initialize the ring model background

    :param U: spatial component matrix from denoiser, R(d=d1xd2, N)
    :param d1: x axis frame size
    :param d2: y axis frame size
    :param r: ring radius of ring model
    :return: initialized spatial components and weighting matrix
    """

    W = init_w(d1, d2, r)
    _, N = U.shape
    Z = np.zeros((d1*d2, N))
    # kernel = skimage.morphology.disk(math.ceil(r/2.0))
    kernel = sk_disk(math.ceil(r / 2.0))

    for i in range(N):
        res = cv.morphologyEx(U[:, i].reshape(d1, d2), cv.MORPH_OPEN, kernel=kernel)
        Z[:, i] = res.reshape((-1))

    update_w_1p(Z, W, d1, d2)
    return U - Z, W

# init_neurons
def init_neurons(Yd, U, V, cut_off_point=[0.9,0.9], length_cut=[10,10], th=[2,1], pass_num=1, residual_cut = [0.6,0.6],
                    corr_th_fix=0.31, max_allow_neuron_size=0.3, merge_corr_thr=0.6, merge_overlap_thr=0.6, num_plane=1, patch_size=[100,100],
                    plot_en=False, TF=False, fudge_factor=1, text=True, bg=False, max_iter=35, max_iter_fin=50, update_after=4):
    """
    -------------------------------------------------
    This function is the entire demixing pipeline for low rank data Yd, which can be decomposed as U*V.

    Parameters:

    *** input data: ***
    Yd: 3D np.ndarray, shape: d1 x d2 x T
        input movie
    U: 2D np.ndarray, shape: (d1 x d2) x r
        low rank decomposition of Yd (rank r)
    V: 2D np.ndarray, shape: T x r
        low rank decomposition of Yd (rank r)
    *************************************************
    *** parameters for superpixel initialization: ***
    cut_off_point: list, length = number of pass
        correlation threshold for finding superpixels
    length_cut: list, length = number of pass
        size cut-off for finding superpixels
    th: list, length = number of pass
        MAD threshold for soft-thresholding Yd
    pass_num: integer
        number of pass
    residual_cut: list, length = number of pass
        sqrt(1 - r_sqare of SPA)
        this usually set to 0.6, that is to say, r_square of SPA is 0.8
    bg: boolean
        having fluctuate background or not
    num_plane: integer
        if num_plane > 1: then it's 3D data; Yd should be reshaped as Yd.reshape(dims[0],dims[1]*num_plane, -1, order="F")
    patch_size: list, length = 2
        small patch size used to find pure superpixels, usually set to [100,100]. If d1 (or d2) is smaller than 100, then the patch size will automatically adjust to [d1 (or d2),100]
    **************************************************
    *** parameters for local NMF: ***
    corr_th_fix: float
        correlation threshold for updating spatial support, i.e. supp(ai) = corr(Yd, ci) > corr_th_fix
    max_allow_neuron_size: float
        max allowed max_i supp(ai) / (d1 x d2).
        If neuron i exceed this range, then when updating spatial support of ai, corr_th_fix will automatically increase 0.1; and will print("corr too low!") on screen.
        If there're too many corr too low on screen, you should consider increasing corr_th_fix.
    merge_corr_thr: float
        correlation threshold for truncating corr(Yd, ci) when merging
    merge_overlap_thr: float
        overlapped threshold for truncated correlation images (corr(Yd, ci)) when merging
    max_iter_fin: integer
        iteraltion times for final pass
    max_iter: integer
        iteration times for pre-final pass if you use multiple passes.
    update_after: integer
        merge and update spatial support every 'update_after' iterations
    **************************************************
    *** parameters for l1_TF on temporal components after local NMF (optional): ***
    TF: boolean
        if True, then run l1_TF on temporal components after local NMF
    fudge_factor: float, usually set to 1
        do l1_TF up to fudge_factor*noise level i.e.
        min_ci' |ci'|_1 s.t. |ci' - ci|_F <= fudge_factor*sigma_i\sqrt(T)
    **************************************************
    *** parameters for plot: ***
    plot_en: boolean
        if True, then will plot superpixels, pure superpixels, local corr image, and merge procedure
    text: boolean
        if True, then will add numbers on each superpixels.
    --------------------------------------------------
    Output:

    If multiple passes: return {'rlt':rlt, 'fin_rlt':fin_rlt, "superpixel_rlt":superpixel_rlt}
    - rlt is a dictionary containing results for first pass: {'a', 'c', 'b', "fb", "ff" (if having fluctuate background, otherwise is null),
                                            'res' (residual for NMF iterations, 0 in current code since not calculate it), 'corr_img_all_r'(correlation images),
                                            'num_list' (current component corresponds to which superpixel)}.

    - fin_rlt is a dictionary containing results for final pass: {'a', 'c', 'c_tf'(if apply TF, otherwise is null), b', "fb", "ff" (if having fluctuate background, otherwise is null),
                                            'res' (residual for NMF iterations, 0 in current code since not calculate it),
                                            'corr_img_all_r'(correlation images), 'num_list' (current component corresponds to which superpixel)}.

    - superpixel_rlt is a list (length = number of pass) containing pure superpixel information for each pass (this result is mainly for plot):
    each element of this list is a dictionary containing {'connect_mat_1'(matrix containing all the superpixels, different number represents different superpixels),
                                                            'pure_pix'(numbers for pure superpixels), 'brightness_rank'(brightness rank of each pure superpixel)}
    You can use function 'pure_superpixel_single_plot' to plot these pure superpixels.

    If only one pass: return {'fin_rlt':fin_rlt, "superpixel_rlt":superpixel_rlt}, details are same as above.
    """
    dims = Yd.shape[:2];
    T = Yd.shape[2];

    ## if data has negative values then do pixel-wise minimum subtraction ##
    Yd_min = Yd.min();
    if Yd_min < 0:
        Yd_min_pw = Yd.min(axis=2, keepdims=True);
        Yd -= Yd_min_pw;
        U = np.hstack((U,Yd_min_pw.reshape(np.prod(dims),1,order="F")));
        V = np.hstack((V,-np.ones([T,1])));

    superpixel_rlt = [];
    ## cut image into small parts to find pure superpixels ##

    patch_height = patch_size[0];
    patch_width = patch_size[1];
    height_num = int(np.ceil(dims[0]/patch_height));  ########### if need less data to find pure superpixel, change dims[0] here #################
    width_num = int(np.ceil(dims[1]/(patch_width*num_plane)));
    num_patch = height_num*width_num;
    patch_ref_mat = np.array(range(num_patch)).reshape(height_num, width_num, order="F");

    ii = 0;
    while ii < pass_num:
        print("start " + str(ii+1) + " pass!");
        if ii > 0:
            if bg:
                Yd_res = superpixel_analysis.reconstruct(Yd, a, c, b, fb, ff);
            else:
                Yd_res = superpixel_analysis.reconstruct(Yd, a, c, b);
            Yt = superpixel_analysis.threshold_data(Yd_res, th=th[ii]);
        else:
            if th[ii] >= 0:
                Yt = superpixel_analysis.threshold_data(Yd, th=th[ii]);
            else:
                Yt = Yd.copy();

        start = time.time();
        if num_plane > 1:
            print("3d data!");
            connect_mat_1, idx, comps, permute_col = superpixel_analysis.find_superpixel_3d(Yt,num_plane,cut_off_point[ii],length_cut[ii],eight_neighbours=True);
        else:
            print("find superpixels!")
            connect_mat_1, idx, comps, permute_col = superpixel_analysis.find_superpixel(Yt,cut_off_point[ii],length_cut[ii],eight_neighbours=True);
        print("time: " + str(time.time()-start));

        start = time.time();
        print("rank 1 svd!")
        if ii > 0:
            c_ini, a_ini, _, _ = superpixel_analysis.spatial_temporal_ini(Yt, comps, idx, length_cut[ii], bg=False);
        else:
            c_ini, a_ini, ff, fb = superpixel_analysis.spatial_temporal_ini(Yt, comps, idx, length_cut[ii], bg=bg);
            #return ff
        print("time: " + str(time.time()-start));
        unique_pix = np.asarray(np.sort(np.unique(connect_mat_1)),dtype="int");
        unique_pix = unique_pix[np.nonzero(unique_pix)];
        #unique_pix = np.asarray(np.sort(np.unique(connect_mat_1))[1:]); #search_superpixel_in_range(connect_mat_1, permute_col, V_mat);
        brightness_rank_sup = superpixel_analysis.order_superpixels(permute_col, unique_pix, a_ini, c_ini);

        #unique_pix = np.asarray(unique_pix);
        pure_pix = [];

        start = time.time();
        print("find pure superpixels!")
        for kk in range(num_patch):
            pos = np.where(patch_ref_mat==kk);
            up=pos[0][0]*patch_height;
            down=min(up+patch_height, dims[0]);
            left=pos[1][0]*patch_width;
            right=min(left+patch_width, dims[1]);
            unique_pix_temp, M = superpixel_analysis.search_superpixel_in_range((connect_mat_1.reshape(dims[0],int(dims[1]/num_plane),num_plane,order="F"))[up:down,left:right], permute_col, c_ini);
            pure_pix_temp = superpixel_analysis.fast_sep_nmf(M, M.shape[1], residual_cut[ii]);
            if len(pure_pix_temp)>0:
                pure_pix = np.hstack((pure_pix, unique_pix_temp[pure_pix_temp]));
        pure_pix = np.unique(pure_pix);

        print("time: " + str(time.time()-start));

        start = time.time();
        print("prepare iteration!")
        if ii > 0:
            a_ini, c_ini, brightness_rank = superpixel_analysis.prepare_iteration(Yd_res, connect_mat_1, permute_col, pure_pix, a_ini, c_ini);
            a = np.hstack((a, a_ini));
            c = np.hstack((c, c_ini));
        else:
            a, c, b, normalize_factor, brightness_rank = superpixel_analysis.prepare_iteration(Yd, connect_mat_1, permute_col, pure_pix, a_ini, c_ini, more=True);
        print("time: " + str(time.time()-start));

        # The above codes are copied from Ding's funimag/superpixel_analysis.py
        # C_centered = c - c.mean(axis=1).reshape(-1, 1)
        # VV = V @ V.transpose()
        # X = np.linalg.solve(VV, V @ C_centered.transpose()).transpose()
        # return a, X

        c = np.transpose(c)
        if Yd_min < 0:
            V = V[:, :-1]
        V = np.transpose(V)
        X = np.linalg.solve(np.matmul(V, np.transpose(V)), np.matmul(V, np.transpose(c))).transpose()
        return a, X



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

    if W is None:
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


def update_temporal(U, V, A, X, W, d1, d2, T, iter=5):
    """
    :param U: spatial component matrix from denoiser, R(d=d1xd2, N)
    :param V: temporal component matrix from denoiser, R(N, T)
    :param A: spatial component matrix, R(d, K)
    :param X: temporal component matrix (the actual temporal matrix C = XV),  R(K, N)
    :param W: weighting matrix, R(d, d)
    :param d1: x axis frame size
    :param d2: y axis frame size
    :param T: number to time steps along time axis
    :param iter: number of iterations to update temporal components
    :return:
        constant baseline of background image in place
    """

    U_tilde = U - W.dot(U - np.matmul(A, X))
    P = np.matmul(np.transpose(A), U_tilde)
    Q = np.matmul(np.transpose(A), A)

    _, k = A.shape
    for i in range(iter):
        for j in range(k):
            x_j = X[j, :]
            x_j += (P[j, :] - np.matmul(Q[j, :], X)) / Q[j, j]
            c_j = np.matmul(x_j, V)
            # c_j = denoise_fcn(c_j) # denoise_fcn to be incorporated
            A_part = np.matmul(V, np.transpose(V))
            b_part = np.matmul(V, np.transpose(c_j))
            X[j, :] = np.transpose(np.linalg.solve(A_part, b_part))
    b0 = np.matmul(U, np.mean(V, axis=1)/T) - np.matmul(A, np.mean(X, axis=1)/T)
    return b0


def dilate_A(A, d1, d2, pixels=3):
    """Dilate spatial component matrix A

    :param A: spatial component matrix, R(d, K)
    :param pixels: dilation kernel size, default 3 pixels
    :return: dilated spatial component index matrix
    """

    M = np.zeros_like(A)
    _, k = A.shape
    kernel = np.ones((pixels, pixels), np.uint8)
    for j in range(k):
        a_j = (A[:, j].reshape(d1, d2) > 0.0).astype('uint8')
        a_j_d = cv.dilate(a_j, kernel=kernel, iterations=1)
        a_j_d = np.reshape(a_j_d, [-1, 1])
        M[:, j] = a_j_d[:, 0]
    return M


def update_spatial(U, V, A, X, W, d1, d2, T, iter=5):
    """
    :param U: spatial component matrix from denoiser, R(d=d1xd2, N)
    :param V: temporal component matrix from denoiser, R(N, T)
    :param A: spatial component matrix, R(d, K)
    :param X: temporal component matrix (the actual temporal matrix C = XV),  R(K, N)
    :param W: weighting matrix, R(d, d)
    :param d1: x axis frame size
    :param d2: y axis frame size
    :param T: number to time steps along time axis
    :param iter: number of iterations to update temporal components
    :return:
        constant baseline of background image in place
    """

    U_tilde = U - W.dot(U - np.matmul(A, X))
    M = dilate_A(A, d1, d2, 3)
    P = np.matmul(U_tilde, np.transpose(X))
    Q = np.matmul(X, np.transpose(X))

    _, k = A.shape
    for i in range(iter):
        for j in range(k):
            b_part = A[:, j] + (P[:, j] - np.matmul(A, Q[:, j])) / Q[j, j]
            A[:, j] = np.multiply(M[:, j], np.maximum(0.0, b_part))

    b0 = np.matmul(U, np.mean(V, axis=1)/T) - np.matmul(A, np.mean(X, axis=1)/T)
    return b0































