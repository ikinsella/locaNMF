
import numpy as np
# from scipy import signal
# from scipy import ndimage
import scipy.io as sio

import superpixel_analysis


from locanmf.cnmf_e import spatial_filtering, init_w, update_ring_model_w, \
    update_temporal, update_spatial, init_ring_model, init_background


def test_spatial_filtering():
    settings = {'gaussian_kernel_width': 3}
    # settings[] = 3
    mat_filepath = '/mnt/data/test_CNMFE/from_matlab.mat'
    mat_data = sio.loadmat(mat_filepath)
    spatial_u = mat_data['Y_4d'].astype(np.float64)
    print(spatial_u.dtype)
    spatial_u_filtered = spatial_filtering(spatial_u, settings)
    sio.savemat('/mnt/data/test_CNMFE/from_python.mat', {'spatial_u_filtered': spatial_u_filtered})
    # pass


def test_init_w():
    d1, d2, r = 41, 42, 13
    w = init_w(d1, d2, r)
    print(w.shape)


def test_init_background():

    data = np.load("cnmfe_init_output.npz")
    print(data.files)
    d1, d2, T, r = data['arr_4']
    # print(d1, d2, r, T)
    U = data['arr_0']
    V = data['arr_1']
    A = data['arr_2']
    X = data['arr_3']
    print(U.shape, V.shape, A.shape, X.shape)
    A, X, W, b0 = init_background(U, V, [], d1, d2, r, T)
    # print(U_ax.shape, W.shape)


def test_init_ring_model():

    data = np.load("cnmfe_init_output.npz")
    print(data.files)
    d1, d2, T, r = data['arr_4']
    # print(d1, d2, r, T)
    U = data['arr_0']
    V = data['arr_1']
    A = data['arr_2']
    X = data['arr_3']
    print(U.shape, V.shape, A.shape, X.shape)
    U_ax, W = init_ring_model(U, d1, d2, r)
    print(U_ax.shape, W.shape)




def test_update_ring_model_w():
    # test update_ring_model_w
    data = np.load("cnmfe_init_output.npz")
    print(data.files)
    d1, d2, T, r = data['arr_4']
    # print(d1, d2, r, T)
    U = data['arr_0']
    V = data['arr_1']
    A = data['arr_2']
    X = data['arr_3']
    print(U.shape, V.shape, A.shape, X.shape)
    b0, W = update_ring_model_w(U, V, A, X, [], d1, d2, T, r)
    print(b0.shape, W.shape)
    np.savez('cnmfe_update_ring_output.npz',
             U=U, V=V, A=A, X=X, W=W, d1=d1, d2=d2, T=T, b0=b0)

    # test update_temporal
    # b0_T = update_temporal(U, V, A, X, W, d1, d2, T, iter=5)
    # print(b0_T.shape)

    # test update_spatial
    b0_S = update_spatial(U, V, A, X, W, d1, d2, T, iter=5)





if __name__ == "__main__":
    # test_update_ring_model_w()
    # test_init_w()
    # test_init_ring_model()
    test_init_background()