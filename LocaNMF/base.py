from abc import ABCMeta, abstractmethod


class FactorCollection(metaclass=ABCMeta):
    """ ABC For Managing Factor/Buffer Lifetime & Manipulation """
    def __init__(self, max_num_components, device):
        """ Set Object Attributes """
        self._max_components = max_num_components
        self._num_components = max_num_components
        self._device = device

    def __len__(self):
        return self.num_components

    @property
    def num_components(self):
        """ Number of active compnents in each factor """
        return self._num_components

    @num_components.setter
    def num_components(self, new_value):
        """ Change the number of factors, modifying tensors accoringly """
        if new_value == self.__len__():
            return None
        elif 0 < new_value <= self._max_components:
            self._resize(new_value)
            self._num_components = new_value
            return None
        raise ValueError(
            "Number of components {} out of range:[1,{}].".format(
                new_value, self._max_components
            ))

    @property
    def device(self):
        """ ID, string, or wrapper for the device hosting the factors """
        return self._device

    @abstractmethod
    def _resize(self, num_components):
        """ Resize each factor axis inplace to represent N components """
        ...

    @abstractmethod
    def permute(self, permutation):
        """ Reorder component in factors according to provided permutation """
        ...


class VideoMixin(metaclass=ABCMeta):
    """ Abstract Mixin To Force Behavior Like A Video"""
    @property
    @abstractmethod
    def fov(self):
        """ Shape of the field of view in pixels/voxels """
        ...

    @property
    @abstractmethod
    def frames(self):
        """ Length of the video in frames """
        ...

    @property
    def shape(self):
        """ Shape of factorized video """
        return self.fov + self.frames


class VideoWrapper(FactorCollection, VideoMixin):
    """ """
    @abstractmethod
    def forward(self, spatial, temporal, **kwargs):
        """ Computes tensor_like(C) = matmul(tensor_like(A), Y) """
        ...

    @abstractmethod
    def backward(self, temporal, spatial, **kwargs):
        """ Computes tensor_like(A) = matmul(tensor_like(C), Y) """
        ...


class VideoFactorizer(FactorCollection, VideoMixin):
    """ ABC For Managing Factor/Buffer Lifetime & Manipulation """

    @property
    def fov(self):
        """ Shape of the field of view in pixels/voxels """
        return self.spatial.shape[1:]

    @property
    def frames(self):
        """ Length of the video in frames """
        return self.temporal.shape[1:]

    @abstractmethod
    def prune_empty_components(self, p_norm=2):
        """ Remove components with empty factors """
        ...

    @abstractmethod
    def update_spatial(self, video_wrapper, **kwargs):
        """ Perform an update of the factorization's spatial factor """
        ...

    @abstractmethod
    def normalize_spatial(self, p_norm=float('inf')):
        """ Scale each spatial component to have unit p-norm """
        ...

    @abstractmethod
    def update_temporal(self, video_wrapper, **kwargs):
        """ Perform an update of the factorization's temporal factor """
        ...

    @abstractmethod
    def normalize_temporal(self, p_norm=float('inf')):
        """ Scale each temporal component to have unit p-norm """
        ...
