
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




def pnr(Y, setting):
    """Calculate PNR (Peak Noise Ratio) of image data Y

    :return: PNR of image data Y
    """


    pass


def correlation(Y, setting):
    """Compute correlation image

    :return: Correlation image
    """

    "function Cn = correlation_image(Y, sz, d1, d2, flag_norm, K)"

    pass

