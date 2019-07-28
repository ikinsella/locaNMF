


import numpy as np
from scipy import signal
from scipy import ndimage
import scipy.io as sio

from .cnmf_e import spatial_filtering


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
