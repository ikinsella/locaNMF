import numpy as np
import torch

from .base import FactorCollection
from .base import VideoWrapper
from .factor import TensorFactor


class Video(VideoWrapper):
    """ Manages Unrolled Tensor and multiplication with a Full Video """
    def __init__(self,
                 video_shape,
                 device='cuda'):
        """ Allocate required tensors on device

        Parameter:
            video_shape: shape of video
            device: computation device

        """
        super().__init__(video_shape[1], device=device)
        self.mov = TensorFactor(video_shape[0],
                                scratch=False,
                                device=self.device,
                                dtype=torch.float32)

    @property
    def fov(self):
        """ Shape of the field of view in pixels/voxels """
        return self.mov.shape[:-1]

    @property
    def frames(self):
        """ Length of the video in frames """
        return self.mov.shape[-1]

    def _resize(self, num_components):
        """ Resize each factor axis inplace to represent N components """
        return None

    def permute(self, permutation):
        """ Reorder component in factors according to provided permutation """
        return None

    def set(self, h_mov):
        """ Initializes Factors From ndarrays on host

        Parameter:
            h_mov: movie on host

        """
        self.mov.data.copy_(h_mov)

    def forward(self, spatial, temporal, **kwargs):
        """ Computes temporal.scratch = matmul(spatial.data, mov.t())

        Parameter:
            spatial: spatial components
            temporal: temporal components
            **kwargs: optional additional input arguments

        """

        torch.matmul(spatial.data, self.mov.data, out=temporal.scratch)

    def backward(self, temporal, spatial, **kwargs):
        """ Computes spatial.scratch = matmul(temporal.data, mov.t())

        Parameter:
            temporal: temporal components
            spatial: spatial components
            **kwargs: optional additional input arguments

        """
        torch.matmul(temporal.data, self.mov.data.t(), out=spatial.scratch)


class LowRankVideo(VideoWrapper):
    """ Manages Tensors and multiplication with a Low-Rank Factored Video """

    def __init__(self,
                 video_shape,
                 spatial_scratch=False,
                 temporal_scratch=False,
                 device='cuda'):
        """ Allocate required tensors on device

        Parameter:
            video_shape: video matrix shape
            spatial_scratch:
            temporal_scratch:
            device: computation device
        """

        super().__init__(video_shape[1], device=device)
        self.spatial = TensorFactor((self.num_components, video_shape[0]),
                                    scratch=spatial_scratch,
                                    device=self.device,
                                    dtype=torch.float32)
        self.temporal = TensorFactor((self.num_components, video_shape[2]),
                                     scratch=temporal_scratch,
                                     device=self.device,
                                     dtype=torch.float32)

    @property
    def fov(self):
        """ Shape of the field of view in pixels/voxels """
        return self.spatial.shape[1:]

    @property
    def frames(self):
        """ Length of the video in frames """
        return self.temporal.shape[1:]

    def _resize(self, num_components):
        """ Resize each factor axis inplace to represent N components

        Parameter:
            num_components: number of components

        """

        self.spatial.resize_axis_(0, num_components)
        self.temporal.resize_axis_(0, num_components)

    def permute(self, permutation):
        """ Reorder component in factors according to provided permutation

        Parameter:
            permutation:

        """

        self.spatial.permute_(permutation)
        self.temporal.permute_(permutation)

    def set(self, h_spatial, h_temporal):
        """ Initializes Factors From ndarrays on host

        Parameter:
            h_spatial: spatial video component
            h_temporal: temporal video component
        """

        self.spatial.data.copy_(h_spatial)
        self.temporal.data.copy_(h_temporal)

    def forward(self, spatial, temporal, **kwargs):
        """ Computes temporal.scratch = matmul(spatial.data, mov.t())

        Parameter:
            spatial: spatial video components
            temporal: temporal video components
            **kwargs: optional additional input arguments

        """

        if 'intermediate' not in kwargs:
            raise TypeError(
                "Missing 1 required keyword argument: 'intermediate'"
            )
        torch.matmul(self.spatial.data,
                     spatial.data.t(),
                     out=kwargs['intermediate'].data)
        torch.matmul(kwargs['intermediate'].data.t(),
                     self.temporal.data,
                     out=temporal.scratch)

    def backward(self, temporal, spatial, **kwargs):
        """ Computes spatial.scratch = matmul(temporal.data, mov.t())

        Parameter:
            temporal: temporal video components
            spatial: spatial video components
            **kwargs: optional additional input arguments

        """

        if 'intermediate' not in kwargs:
            raise TypeError(
                "Missing 1 required keyword argument: 'intermediate'"
            )
        torch.matmul(self.temporal.data,
                     temporal.data.t(),
                     out=kwargs['intermediate'].data)
        torch.matmul(kwargs['intermediate'].data.t(),
                     self.spatial.data,
                     out=spatial.scratch)


class RegionMetadata(FactorCollection):
    """ Manages Metadata For Each Region In Localized Semi-NMF """

    def __init__(self, max_num_components, region_shape, device='cuda'):
        """ Allocate required tensors on device

        Parameter:
            max_num_components: maximum number of components
            region_shape: region shape
            device: computation device

        """

        super().__init__(max_num_components, device)
        self.support = TensorFactor((max_num_components, region_shape[0]),
                                    device=device,
                                    dtype=torch.uint8)
        self.distance = TensorFactor((max_num_components, region_shape[0]),
                                     device=device,
                                     dtype=torch.float32)
        self.labels = TensorFactor((max_num_components,),
                                   device=device,
                                   dtype=torch.long)

    def _resize(self, num_components):
        """ Resize each factor axis inplace to represent N components

        Parameter:
            num_components: number of components

        """

        self.support.resize_axis_(0, num_components)
        self.distance.resize_axis_(0, num_components)
        self.labels.resize_axis_(0, num_components)

    def permute(self, permutation):
        """ Reorder component in factors according to provided permutation

        Parameter:
            permutation:

        """

        self.support.permute_(permutation)
        self.distance.permute_(permutation)
        self.labels.permute_(permutation)

    def set(self, h_support, h_distance, h_labels):
        """ Initializes Factors From ndarrays on host

        Parameter:
            h_support: the mask of each region
            h_distance: the distance penalty of each region
            h_labels: area code

        Each input parameter comes from LocaNMF.extract_region_metadata fcn
        output.

        """

        self.support.data.copy_(h_support)
        self.distance.data.copy_(h_distance)
        self.labels.data.copy_(h_labels)
