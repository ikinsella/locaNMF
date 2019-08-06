
import numpy as np
from scipy import signal
from scipy import ndimage
import scipy.io as sio

from .cnmf_e import spatial_filtering, init_w, update_ring_model_w


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


def test_update_ring_model_w():
    data = np.load("init_W.npz")
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


    pass




if __name__ == "__main__":
    test_update_ring_model_w()
