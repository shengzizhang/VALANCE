# VALANCE - HIV-1 bNAb resistance prediction
# Base image matches the CUDA 12.6 build of torch used in requirements.txt
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System dependencies:
#   python3.10   - matches the environment VALANCE was developed in
#   mafft        - required for --addfull --keeplength alignment against the
#                  antibody-specific reference before embedding/prediction
#   git/wget/ca-certificates - misc utilities
#   build-essential - a couple of the pinned packages build from source on install
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3-dev \
        mafft \
        git \
        wget \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Install Python dependencies first so this layer is cached across code changes.
# The +cu126 torch/torchaudio/torchvision builds live on the PyTorch index, not PyPI,
# hence the --extra-index-url.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu126

# VALANCE scripts (generate_embedding.py, training_finetuned_tabpfn.py,
# training_finetuned_tabpfn_Nested.py, prediction_finetuned_tabpfn.py)
COPY scripts/ ./scripts/

# Antibody-specific reference alignments used for MAFFT alignment before
# embedding/prediction (from github.com/shengzizhang/VALANCE/reference_alignment)
COPY reference_alignment/ ./reference_alignment/

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /app/scripts

# Pretrained models (.joblib), input FASTA files, and output directories are
# expected to be bind-mounted at runtime - see README for volume examples.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
