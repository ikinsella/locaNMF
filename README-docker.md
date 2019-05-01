# Nvidia-docker


This software can be installed from source by downloading the repository and set up
the dependency accordingly. The recommended way to install this 
software is using [Docker](https://www.docker.com/why-docker) container which contains both this software and all of its 
dependency. User will only need to install Nvidia Driver if planning to use GPU 
accelerator.

The following sections describe how to properly install Docker container in Ubuntu 
16.04 & 18.04. It is not tested on CentOS 7. However, it should be 
straightforward to modify the following scripts and install accordingly. 


## Install Nvidia Driver

This section will describe how to install Nvidia driver. Skip to next section if no GPU 
will be used. First switch to `multi-user.target` mode, add Nvidia driver repository and then reboot.

```
sudo apt-get install -y linux-headers-$(uname -r)
sudo systemctl set-default multi-user.target
sudo apt-add-repository ppa:graphics-drivers/ppa
sudo reboot
```

Press Ctrl-Alt-F1 to input username and password. Install Nvidia driver and then
switch back to `graphical.target` mode and reboot.

```
sudo apt-get update
# Ubuntu 16.04
sudo apt-get install -y nvidia-418
# or Ubuntu 18.04
sudo apt-get install -y nvidia-driver-418
sudo systemctl set-default graphical.target
sudo reboot
```

Execute `nvidia-smi` command in bash and it should display the proper Nvidia driver and 
GPU installed in the machine. Otherwise, double check and execute the scripts.

## Install Docker software

Install the dependency, Docker GPG key and repository.

```
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) \
    stable"
```

Install Docker CE and add username to Docker group.

```
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker $USER
```

Log out and then back in to activate the changes in Docker group.

Run Docker version "Hello World". This command downloads a test image and runs it in 
a container. When the container runs, it prints an informational message and exits.
Make sure that the "Hello World" message is displayed properly.

```
docker run hello-world
```

## Nvidia Docker Runtime 

This section will install Nvidia Docker runtime which is required to use GPU accelerator.
You may skip this section if no GPU accelerator will be used.

```
# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
    sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
```

To test the installation, download and run Nvidia CUDA container image. `nvidia-smi` command will be
executed in Docker container and then the GPU usage info will be displayed. 

```
docker pull nvidia/cuda:10.1-base
docker run --runtime=nvidia --rm nvidia/cuda:10.1-base nvidia-smi
```

## Run LocaNMF


