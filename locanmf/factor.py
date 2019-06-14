import torch


class TensorFactor():
    """ Wrapper For Matrix Factor Represented By A PyTorch Tensor"""

    def __init__(self, shape, scratch=False, **kwargs):
        """ Allocates Component & Related Buffer PyTorch Tensors

        Parameter:
            shape: tensor shape
            scratch:
            **kwargs: optional additional input arguments

        """

        self._data = torch.empty(shape, **kwargs)
        self._scratch = None
        if scratch:
            self._scratch = torch.empty(shape, **kwargs)

    @property
    def shape(self):
        """ Return shape in form of (num_components, num_frames) """
        return self.data.shape

    @property
    def data(self):
        """ Provide access to wrapped factor tensor """
        return self._data

    @property
    def scratch(self):
        """ Provide access to wrapped buffer tensor """
        return self._scratch

    def resize_axis_(self, dim, new_size):
        """ Shrink Component Axis Of Managed Tensors Inplace

        Parameter:
            dim:
            new_size:

        """
        # Create New torch.Size
        new_shape = list(self.data.size())
        new_shape[dim] = new_size
        new_shape = torch.Size(new_shape)
        # Reset Tensor Size Info Keeping Everything Else The Same
        self.data.set_(source=self.data.storage(),
                       storage_offset=self.data.storage_offset(),
                       size=new_shape,
                       stride=self.data.stride())
        # If It Exists, Do The Same To The Scratch Buffer
        if self._scratch is not None:
            self.scratch.set_(source=self.scratch.storage(),
                              storage_offset=self.scratch.storage_offset(),
                              size=new_shape,
                              stride=self.scratch.stride())

    def permute_(self, permutation, dim=0, scratch=False):
        """ Reindex Component Axis Of Managed Tensors Inplace

        Parameter:
            permutation:
            dim=0:
            scratch=False

        """
        torch.index_select(self.data, dim, permutation, out=self.data)
        if scratch and (self._scratch is not None):
            torch.index_select(self.scratch,
                               dim,
                               permutation,
                               out=self.scratch)