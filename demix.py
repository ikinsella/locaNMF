from abc import abstractmethod

import torch
import numpy as np

import cuhals

from base import VideoFactorizer
from factor import TensorFactor


class SpatialHals():
    """ Mixin Class Carrying 'private' Spatial HALS Update """
    def _spatial_update(self,
                        nonnegative=True,
                        batch=32,
                        **kwargs):
        """ Fast Hals Update Of Spatial Components """
        cuhals.update(self.spatial.data,
                      self.spatial.scratch,
                      self.covariance.data,
                      nonnegative,
                      batch)


class TemporalHals():
    """ Mixin Class Carrying 'private' Temporal HALS Update """
    def _temporal_update(self,
                         nonnegative=True,
                         batch=32,
                         **kwargs):
        """ Fast Hals Update Of Temporal Components """
        cuhals.update(self.temporal.data,
                      self.temporal.scratch,
                      self.covariance.data,
                      nonnegative,
                      batch)


class TemporalLS():
    """ Mixin Class Carrying 'private' Temporal LS Update """
    def _temporal_update(self, **kwargs):
        """ Least Squares Update Of Temporal Components """
        torch.inverse(self.covariance.data,
                      out=self.covariance.data)
        torch.matmul(self.covariance.data,
                     self.temporal.scratch,
                     out=self.temporal.data)


class BaseNMF(VideoFactorizer):
    """ Manages Factor/Buffer Lifetime & Manipulation For HALS NMF """
    def __init__(self,
                 max_num_components,
                 video_shape,
                 device='cuda'):
        """ allocate neccesary buffers """
        super().__init__(max_num_components, device)
        self.spatial = TensorFactor((self.num_components, video_shape[0]),
                                    scratch=True,
                                    device=self.device,
                                    dtype=torch.float32)
        self.intermediate = TensorFactor((video_shape[1], self.num_components),
                                         device=self.device,
                                         dtype=torch.float32)
        self.temporal = TensorFactor((self.num_components, video_shape[2]),
                                     scratch=True,
                                     device=self.device,
                                     dtype=torch.float32)
        self.covariance = TensorFactor((self.num_components,)*2,
                                       device=self.device,
                                       dtype=torch.float32)
        self.scale = TensorFactor((self.num_components,),
                                  scratch=True,
                                  device=self.device,
                                  dtype=torch.float32)
        self.index = TensorFactor((self.num_components,),
                                  device=self.device,
                                  dtype=torch.long)

    def _resize(self, num_components):
        """ Resize each factor axis inplace to represent N components """
        self.spatial.resize_axis_(0, num_components)
        self.temporal.resize_axis_(0, num_components)
        self.intermediate.resize_axis_(1, num_components)
        self.covariance.data.resize_((num_components,)*2)
        self.scale.resize_axis_(0, num_components)
        self.index.resize_axis_(0, num_components)

    def permute(self, permutation):
        """ Reorder component in factors according to provided permutation """
        self.spatial.permute_(permutation)
        self.temporal.permute_(permutation)

    def prune_empty_components(self, p_norm=2):
        """ Remove components with spatial empty factors """
        torch.norm(self.spatial.data,
                   p=p_norm,
                   dim=-1,
                   out=self.scale.data)
        self.scale.scratch.copy_(self.scale.data)
        self.scale.scratch.gt_(0)
        nnz = int(torch.sum(self.scale.scratch).item())
        if nnz < self.num_components:
            print("Removing components, num remaining: {}".format(nnz))
            self.index.data[:] = torch.argsort(
                self.scale.data, descending=True
            )
            self.permute(self.index.data)
            self.num_components = nnz

    def _spatial_precompute(self, video_wrapper, **kwargs):
        """ Precompute Sigma, Delta For Use In Spatial Update """
        # Sigma = A'A
        torch.matmul(self.temporal.data,
                     self.temporal.data.t(),
                     out=self.covariance.data)
        # Delta = C'Y'
        video_wrapper.backward(self.temporal,
                               self.spatial,
                               intermediate=self.intermediate,
                               **kwargs)

    @abstractmethod
    def _spatial_update(self, **kwargs):
        """ Use Precomputed Quantities To Update Spatial Factors """
        ...

    def update_spatial(self, video_wrapper, **kwargs):
        """ Perform the HALS update of the spatial factor """
        self._spatial_precompute(video_wrapper, **kwargs)
        self._spatial_update(**kwargs)

    def normalize_spatial(self, p_norm=float('inf')):
        """ Scale each spatial component to have unit p-norm """
        torch.norm(self.spatial.data,
                   p=p_norm,
                   dim=-1,
                   out=self.scale.data)
        torch.div(self.spatial.data,
                  self.scale.data[..., None],
                  out=self.spatial.data)
        torch.mul(self.temporal.data,
                  self.scale.data[..., None],
                  out=self.temporal.data)

    def _temporal_precompute(self, video_wrapper, **kwargs):
        """ Precompute Sigma, Delta For Use In Temporal Update """
        # Sigma = C'C
        torch.matmul(self.spatial.data,
                     self.spatial.data.t(),
                     out=self.covariance.data)
        # Delta = A'Y
        video_wrapper.forward(self.spatial,
                              self.temporal,
                              intermediate=self.intermediate,
                              **kwargs)

    @abstractmethod
    def _temporal_update(self, **kwargs):
        """ Use Precomputed Quantities To Update Temporal Factors """
        ...

    def update_temporal(self, video_wrapper, **kwargs):
        """ Perform the HALS update of the temporal factor """
        self._temporal_precompute(video_wrapper, **kwargs)
        self._temporal_update(**kwargs)

    def normalize_temporal(self, p_norm=float('inf')):
        """ Scale each temporal component to have unit p-norm """
        torch.norm(self.temporal.data,
                   p=p_norm,
                   dim=-1,
                   out=self.scale.data)
        torch.div(self.temporal.data,
                  self.scale.data[..., None],
                  out=self.temporal.data)
        torch.mul(self.spatial.data,
                  self.scale.data[..., None],
                  out=self.spatial.data)


class HalsNMF(SpatialHals, TemporalHals, BaseNMF):
    """ """
    pass


class LocalizedNMF(SpatialHals, TemporalHals, BaseNMF):
    """ Adds Region-Localization Factors & Updates To Hals Factorization """
    def __init__(self,
                 max_num_components,
                 video_shape,
                 device='cuda'):
        """ allocate neccesary buffers & copy initialization from host """
        super().__init__(max_num_components, video_shape, device=device)
        self.distance = TensorFactor((self.num_components, video_shape[0]),
                                     scratch=True,
                                     device=self.device,
                                     dtype=torch.float32)
        self.regions = TensorFactor((self.num_components,),
                                    device=self.device,
                                    dtype=torch.long)
        self.lambdas = TensorFactor((self.num_components,),
                                    device=self.device,
                                    dtype=torch.float32)

    def _resize(self, num_components):
        """ Resize each factor axis inplace to represent N components """
        super()._resize(num_components)
        self.distance.resize_axis_(0, num_components)
        self.regions.resize_axis_(0, num_components)
        self.lambdas.resize_axis_(0, num_components)

    def permute(self, permutation):
        """ Reorder component in factors according to provided permutation """
        super().permute(permutation)
        self.distance.permute_(permutation, scratch=True)
        self.regions.permute_(permutation)
        self.lambdas.permute_(permutation)

    def _spatial_precompute(self, video_wrapper, **kwargs):
        """ """
        super()._spatial_precompute(video_wrapper, **kwargs)
        torch.sub(self.spatial.scratch,
                  self.distance.scratch,
                  out=self.spatial.scratch)

    def set_from_regions(self, region_factorizations, region_metadata):
        """ Use Within-Region Factorizations To Init Full FOV Factors """
        ranks = [len(factorization) for factorization in region_factorizations]
        self.num_components = np.sum(ranks)
        sdx = 0
        self.spatial.data.fill_(0.0)
        for rdx, rank in enumerate(ranks):
            self.spatial.data[sdx:sdx+rank].masked_scatter_(
                region_metadata.support.data[rdx],
                region_factorizations[rdx].spatial.data
            )
            self.temporal.data[sdx:sdx+rank].copy_(
                region_factorizations[rdx].temporal.data
            )
            self.distance.data[sdx:sdx+rank].copy_(
                region_metadata.distance.data[rdx]
            )
            self.regions.data[sdx:sdx+rank].fill_(rdx)
            sdx += rank
