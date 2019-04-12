# LocaNMF



## Dependencies

- python3
- numpy
- scipy
- sklearn
- pytorch

We encourage the use of conda to manage the dependencies for LocaNMF in it's own environment. 
The following lines will create a new environment with the required dependencies:
```
conda create -n locaNMF python=3 numpy scipy scikit-learn
conda activate locaNMF
```
when the preceeding installation is completed, you must also follow (Pythorch's)[https://pytorch.org/] installation instructions.

### Enabling GPU Support

LocaNMF is implemented using pytorch in order to take advantage of abstractions that allow the same code to execute on CPU and GPU. 
To enable the use of your GPU you must have a a compatible CUDA installation, pytorch installation, and GPU available.
Please reference (Pythorch's)[https://pytorch.org/] documentation in order to achieve this.

## Installation

With your conda environment active and dependencies installed, you can install locaNMF by cloning the repository and running the installation script
```
git clone https://github.com/ikinsella/locaNMF.git
cd locaNMF
python setup.py install
```

## (OPTIONAL) Compiling The Cuda Extension

Pytorch provides an excellent set of general programming abstractions for writing high level code to execute on both CPU & GPU.
However, these abstractions do not provide the flexibility required to implement certain computations efficiently on the GPU.
In order to mitigate a significant bottleneck, we provide a cuda implementation and use a (c++/cuda extension)[https://pytorch.org/tutorials/advanced/cpp_extension.html] to integrate it with pytorch.
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
