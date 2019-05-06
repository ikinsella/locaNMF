# LocaNMF

LocaNMF toolkit (Localized semi-Nonnegative Matrix Factorization) can eﬃciently decompose wideﬁeld video 
data and allows user to directly compare activity across multiple mice by outputting mouse-speciﬁc 
localized functional regions. LocaNMF uses a fast low-rank version of Hierarchical Alternating Least 
Squares (HALS), and outputs components that are signiﬁcantly more interpretable than more traditional NMF
or SVD-based techniques.
 
It is written in Python and C++, built on top of PyTorch and can run on both CPU and Nvidia GPU. 
The following describes how to install LocaNMF using conda.


<!--User can choose to run LocaNMF on either CPU or GPU by installing 
CPU or GPU version PyTorch. However, user can choose to run LocaNMF on CPU by setting parameters in LocaNMF
even though GPU version PyTorch is installed. User can choose whether to enable GPU for acceleration. -->


Due to the complexity of installing Nvidia graphic driver and CUDA toolkit, 
an alternative way to use LocaNMF with Nvidia GPU 
is through the prebuilt [Docker](https://www.docker.com/why-docker) image which contains 
both LocaNMF and all of its 
dependency. Follow this [guide](https://github.com/ikinsella/locaNMF/blob/container/README-docker.md) to use 
the prebuilt Docker image. 

## Dependencies

- python3
- numpy
- scipy
- sklearn
- matplotlib
- pytorch
- mkl
- mkl-include

It is recommended to use [conda](https://docs.conda.io/en/latest/miniconda.html) to manage the 
dependencies for LocaNMF in it's own environment. First, install conda or miniconda and then 
create a new environment for LocaNMF with the required dependencies:
```
conda create -n locaNMF python=3 numpy scipy scikit-learn matplotlib mkl mkl-include
conda activate locaNMF
```
When the proceeding installation is completed, [PyTorch](https://pytorch.org/) 
needs to be installed. LocaNMF is implemented using PyTorch in order to take 
advantage of abstractions that allows us to provide one implementation 
that is capable of being run using either CPU or GPU. Use either command below to properly install PyTorch.
Other version of CUDA toolkit may be installed basing on [PyTorch](https://pytorch.org/).
```
# Without GPU acceleration
conda install pytorch-cpu torchvision-cpu -c pytorch
# Or with GPU acceleration 
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

To enable the use of Nvidia GPU you must have a compatible Nvidia GPU, Nvidia graphics driver, **CUDA** 
and PyTorch installation.
Please reference [CUDA](https://developer.nvidia.com/cuda-zone) documentation in order to install Nvidia
graphics driver and CUDA.



<!--TODO: -->

## Installation

With your conda environment active and dependencies installed, you can install locaNMF by cloning the repository and 
running the installation script
```
git clone https://github.com/ikinsella/locaNMF.git
cd locaNMF
python setup.py install
```
LocaNMF is ready to run on CPU or GPU basing on the PyTorch version installed above. 
Please follow the demo notebook in the current folder to try out LocaNMF.


## (OPTIONAL) Compiling The Cuda Extension

This section is optional and for computation performance benefit. 
PyTorch provides an excellent set of general programming abstractions for writing high level code that can use both CPUs & GPUs.
However, these abstractions do not provide the flexibility required to implement certain computations efficiently on the GPU.
In order to mitigate a significant bottleneck, we provide a cuda implementation and use a 
[c++/cuda extension](https://pytorch.org/tutorials/advanced/cpp_extension.html) to integrate it with PyTorch.
To enable use of this feature, an additional argument ```--with-extension``` must be added while invoking the installation script

```python setup.py install --with-extension```

Note that for this to succeed you must have properly configured cuda installation and a compatible c++ compiler availble. 
Below, we provide tips for achieving this on Linux systems.

### (Tip) Environment Variables

To use the ```nvcc``` compiler that comes with your cuda installation, you must add it's location to your path and have a proper set of environment variables. 
For example, this can be done by adding the following lines to your ```~/.bashrc```

```
export CUDA_HOME=/path/to/cuda
export CUDADIR=/path/to/cuda
export PATH=$PATH:$CUDA_HOME/bin
```
### (Tip) Debugging A Cuda 9.0 Installation On Newer Systems

A known issue is that the ```nvcc``` compiler from Cuda 9.0 requires the availability of ```gcc``` older than the default on newer systems.
In order to avoid compatibility issues, you should create a symbolic link to a compatible ```gcc``` into directory containing ```nvcc``` 
For example, if you wish to use ```gcc-5``` (recommended) you would use the line 
```ln -s /usr/bin/gcc-5 /path/to/cuda-9.0/bin/gcc```.
