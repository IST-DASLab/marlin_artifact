FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir /projects
WORKDIR /projects

ADD . /projects

ENV CUDA_INSTALL_PATH=/usr/local/cuda-11.8
ENV PATH=$CUDA_INSTALL_PATH/bin:$PATH

RUN apt-get update && apt-get install -y \
    software-properties-common
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    wget \
    git \
    vim

ENV TORCH_CUDA_ARCH_LIST="8.6"

#RUN python3 -m pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118/
#RUN python3 -m pip install "pybind11>=2.12"
#RUN python3 -m pip install "numpy<2"

# Install miniconda
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh
RUN ~/miniconda3/bin/conda init bash
ENV PATH=/root/miniconda3/bin:$PATH

# Install MARLIN
#RUN python3 -m pip install .
RUN conda create -y --name marlin python=3.10 -y
SHELL ["conda", "run", "-n", "marlin", "/bin/bash", "-c"]
RUN pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118/
RUN conda install nvidia/label/cuda-11.8.0::cuda-nvcc
RUN conda install nvidia/label/cuda-11.8.0::cuda-toolkit
RUN conda env config vars set CUDA_HOME="/root/miniconda3/envs/marlin"
RUN conda env config vars set CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
RUN conda env config vars set LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
RUN conda env config vars set LIBRARY_PATH=$CUDA_HOME/lib:$LIBRARY_PATH
RUN conda env config vars set LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
RUN conda env config vars set CPATH=/root/miniconda3/envs/marlin/targets/x86_64-linux/include/:$CPATH
RUN conda env config vars set LD_LIBRARY_PATH=/root/miniconda3/envs/marlin/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
RUN conda env config vars set PATH=$CUDA_HOME/bin:$PATH
RUN source deactivate
SHELL ["conda", "run", "-n", "marlin", "/bin/bash", "-c"]
RUN python3 -m pip install "numpy<2"
RUN pip install .
#SHELL ["conda", "deactivate"]
RUN source deactivate

# Install llm-awq
RUN conda create -y --name awq python=3.10 -y
SHELL ["conda", "run", "-n", "awq", "/bin/bash", "-c"]
#RUN git clone ...
#RUN git checkout ...
RUN conda install nvidia/label/cuda-12.4.1::cuda
RUN pip install --upgrade pip
WORKDIR /projects/baselines/llm-awq
RUN pip install -e .
WORKDIR /projects/baselines/llm-awq/awq/kernels
RUN python3 setup.py install
RUN source deactivate

# Install exllamav2
RUN conda create -y --name exllamav2 python=3.8 -y
SHELL ["conda", "run", "-n", "exllamav2", "/bin/bash", "-c"]
#RUN git clone ...
#RUN git checkout v0.0.10
WORKDIR /projects/baselines/exllamav2
RUN conda install nvidia/label/cuda-12.1.0::cuda-nvcc
RUN conda install nvidia/label/cuda-12.1.0::cuda-toolkit
RUN pip install -r requirements.txt
#
#RUN conda install -c conda-forge gxx_linux-64
#RUN conda install -c conda-forge gcc=12.1.0
#RUN ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /root/miniconda3/envs/exllamav2/bin/../lib/libstdc++.so.6
#
RUN conda env config vars set CUDA_HOME="/root/miniconda3/envs/exllamav2"
RUN conda env config vars set CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
RUN conda env config vars set LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
RUN conda env config vars set LIBRARY_PATH=$CUDA_HOME/lib:$LIBRARY_PATH
RUN conda env config vars set LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
RUN conda env config vars set CPATH=/root/miniconda3/envs/exllamav2/targets/x86_64-linux/include/:$CPATH
RUN conda env config vars set LD_LIBRARY_PATH=/root/miniconda3/envs/exllamav2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
RUN conda env config vars set PATH=$CUDA_HOME/bin:$PATH
RUN source deactivate
#SHELL ["conda", "run", "-n", "exllamav2", "/bin/bash", "-c"]
#
#RUN python setup.py install --user
#SHELL ["conda", "deactivate"]

# Install bitsandbytes
WORKDIR /projects/baselines/
RUN conda env create -f env.yml
RUN source deactivate

# Install AutoAWQ_kernels
#WORKDIR /projects/baselines/AutoAWQ_kernels
#RUN conda create -y --name autoAWQ
#SHELL ["conda", "run", "-n", "autoAWQ", "/bin/bash", "-c"]
#RUN conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
#RUN pip install autoawq-kernels
#RUN pip install transformers accelerate datasets
#SHELL ["conda", "deactivate"]

# Install torch-nightly
WORKDIR /projects/baselines/ao
RUN conda create -y --name torchao python=3.10 -y
SHELL ["conda", "run", "-n", "torchao", "/bin/bash", "-c"]
RUN pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu121
RUN pip install hqq pandas
RUN conda env config vars set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN source deactivate

RUN source deactivate
#SHELL ["conda", "deactivate"]

#WORKDIR /

WORKDIR /projects