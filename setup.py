import os
import sys
import setuptools
# from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

with open("README.md", "r") as fh:
    long_description = fh.read()

ENV = os.environ["CONDA_PREFIX"]  # Absolute path of active conda env root
LIBRARY_DIRS = [os.path.join(ENV, "lib")]  # .[so,dylib]
INCLUDE_DIRS = [os.path.join(ENV, "include")]

ext_modules = []
if '--with-extension' in sys.argv:
    ext_modules.append(
        CUDAExtension('cuhals',
                      ['./locanmf/cuhals.cpp', './locanmf/cuhals_kernels.cu'],
                      extra_compile_args={'cxx': ["-fopenmp", "-I"+INCLUDE_DIRS[0]],
                                          'nvcc': []}))
    sys.argv.remove('--with-extension')

# setup(name='locaNMF',
#       ext_modules=ext_modules,
#       cmdclass={
#           'build_ext': BuildExtension
#       })


setuptools.setup(
    name="locanmf",
    version="1.0",
    author="Ian August Kinsella",
    author_email="iak2199@columbia.edu",
    description="Demixer using Localized semi-Nonnegative Matrix Factorization method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ikinsella/locaNMF",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
)
