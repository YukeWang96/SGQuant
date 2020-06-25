FROM nvidia/cuda:10.2-base-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.1 \
 && conda clean -ya

# CUDA 10.2-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=10.2 \
    "pytorch=1.5.0=py3.8_cuda10.2.89_cudnn7.6.5_0" \
    "torchvision=0.6.0=py38_cu102" \
 && conda clean -ya

# # Set the default command to python3
# CMD ["python3"]

# Install HDF5 Python bindings.
RUN conda install -y h5py \
 && conda clean -ya
RUN pip install h5py-cache

# Install TorchNet, a high-level framework for PyTorch.
RUN pip install torchnet

# Install Requests, a Python library for making HTTP requests.
RUN conda install -y requests \
 && conda clean -ya

# Install Graphviz.
RUN conda install -y graphviz python-graphviz \
 && conda clean -ya

# Install OpenCV3 Python bindings.
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
 && sudo rm -rf /var/lib/apt/lists/*
# RUN conda install -y -c menpo opencv3 \
#  && conda clean -ya

# Install PyTorch Geometric.
RUN CPATH=/usr/local/cuda/include:$CPATH \
 && LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
 && DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH

RUN pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html \
 && pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html \
 && pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html \
 && pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html \
 && pip install torch-geometric

# Set the default command to python3.
CMD ["python3"]
