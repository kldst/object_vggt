# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    SAS_URL="https://zbstore2026.blob.core.windows.net/g-019c8c93-9d1d-765e-9d2f-e3eb9ccb33f0?se=2026-09-14T06%3A46%3A42Z&sp=rwdl&sv=2026-02-06&sr=c&sig=MBLdKaJOufr94Upl8qshasWnhZlO4eJzLSBCke8G2JE%3D"

# --- OS deps (build tools + common libs) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    build-essential cmake pkg-config \
    python3-dev \
    libgl1 libglib2.0-0 \
    zip unzip vim wget tar tmux \
    && rm -rf /var/lib/apt/lists/*

# --- Miniconda ---
# --- AzCopy ---
RUN wget -O /tmp/azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux \
    && tar -xzf /tmp/azcopy.tar.gz -C /tmp \
    && install /tmp/azcopy_linux_amd64_*/azcopy /usr/local/bin/azcopy \
    && rm -rf /tmp/azcopy.tar.gz /tmp/azcopy_linux_amd64_*

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda config --set channel_priority strict \
    && conda update -n base -c defaults conda -y

# --- Create env ---
ARG ENV_NAME=vggt
RUN conda create -n ${ENV_NAME} python=3.10 -y

# Use bash so "conda activate" works
SHELL ["bash", "-lc"]

# Auto-activate the conda env for interactive shells
RUN printf '\n# Auto-activate vggt\nsource /opt/conda/etc/profile.d/conda.sh\nconda activate vggt\n' >> /root/.bashrc

# --- Install PyTorch (cu124) ---
RUN conda run -n ${ENV_NAME} python -m pip install -U pip setuptools wheel \
    && conda run -n ${ENV_NAME} python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# --- (Optional) speed / common deps you listed ---
RUN conda run -n ${ENV_NAME} python -m pip install hydra-core omegaconf iopath wcmatch yacs fvcore tensorboard rich smplx

# --- chumpy via conda-forge (as you did) ---
RUN conda run -n ${ENV_NAME} conda install -c conda-forge chumpy -y

# --- Copy vggt repo into image (OPTION 1) ---
WORKDIR /workspace
COPY . /workspace/vggt

# --- Install vggt requirements + editable install ---
RUN conda run -n ${ENV_NAME} python -m pip install -r /workspace/vggt/requirements.txt \
    && conda run -n ${ENV_NAME} python -m pip install -r /workspace/vggt/requirements_demo.txt \
    && conda run -n ${ENV_NAME} python -m pip install -e /workspace/vggt

# Default command
CMD ["bash"]
