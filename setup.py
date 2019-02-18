import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ENV = os.environ["CONDA_PREFIX"]  # Absolute path of active conda env root
LIBRARY_DIRS = [os.path.join(ENV, "lib")]  # .[so,dylib]
INCLUDE_DIRS = [os.path.join(ENV, "include")]
LIBRARIES = ["magma"]

setup(name='fasthals',
      ext_modules=[
          CUDAExtension('cuhals',
                        ['cuhals.cpp', 'cuhals_kernels.cu'],
                        extra_compile_args={'cxx': ["-fopenmp"],
                                            'nvcc': []}),
      ],
      cmdclass={
          'build_ext': BuildExtension
      })
