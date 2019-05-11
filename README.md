# LocaNMF

LocaNMF toolkit (Localized semi-Nonnegative Matrix Factorization) can eﬃciently decompose wideﬁeld video 
data and allows user to directly compare activity across multiple mice by outputting mouse-speciﬁc 
localized functional regions. LocaNMF uses a fast low-rank version of Hierarchical Alternating Least 
Squares (HALS), and outputs components that are signiﬁcantly more interpretable than traditional NMF
or SVD-based techniques.
 
It is built on top of PyTorch, written in Python and C++, and is capable to run on either CPU or
Nvidia CUDA-enabled GPU. To run LocaNMF on Nvidia GPU, a Nvidia 
[CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus) 
is required and the latest version 
[Nvidia Driver](https://www.nvidia.com/Download/index.aspx?lang=en-us) (version > 418.x)
is required to be properly installed before installing LocaNMF software.

## Install

It is recommended to use [conda](https://www.anaconda.com/) to manage the 
dependencies for LocaNMF in it's own Python environment.
First, download and install [conda](https://www.anaconda.com/distribution/). Verify conda installation
by executing the following scripts. A list of base environment packages will be displayed.
```
conda list
```

<!-- pytorch only requires nvidia driver, doesn't require to install cuda. -->
Create a new environment for LocaNMF and install LocaNMF software and all of its dependencies.
```
conda create -n locanmf python=3.6 locanmf -c jw3132 -c pytorch
```

## Use LocaNMF

Activate `locanmf` conda environment.
```
conda activate locanmf
```

To test the proper installation of LocaNMF software, execute the following script.
```
python
from locanmf import LocaNMF
LocaNMF.version()
```

The following output will be displayed. It implies that LocaNMF and its dependencies have been 
properly installed.
```
python
Python 3.6.8 |Anaconda, Inc.| (default, Dec 30 2018, 01:22:34) 
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from  locanmf import LocaNMF
>>> LocaNMF.version()
version = 1.0
>>>
```

Please download the **demo** folder in this repository to your computer and 
walk through the `demo_simulation.ipynb` notebook to try out the software. 

To run the notebook, in terminal, change directory to the downloaded **demo** folder,
then execute,
```
jupyter-lab &
```
to start jupyter-lab server. In the pop-out web browser, double-click the notebook to open it.
The code is editable and can be run cell by cell by pressing "shift + enter".




<!--TODO: 

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

-->