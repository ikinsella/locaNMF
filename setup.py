import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ENV = os.environ["CONDA_PREFIX"]  # Absolute path of active conda env root
LIBRARY_DIRS = [os.path.join(ENV, "lib")]  # .[so,dylib]
INCLUDE_DIRS = [os.path.join(ENV, "include")]

ext_modules = []
if '--with-extension' in sys.argv:
    ext_modules.append(
        CUDAExtension('cuhals',
                      ['cuhals.cpp', 'cuhals_kernels.cu'],
                      extra_compile_args={'cxx': ["-fopenmp"],
                                          'nvcc': []}))
    sys.argv.remove('--with-extension')

setup(name='locaNMF',
      ext_modules=ext_modules,
      cmdclass={
          'build_ext': BuildExtension
      })
