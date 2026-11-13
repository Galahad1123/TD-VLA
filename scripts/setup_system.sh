#!/usr/bin/env bash
# Ubuntu 22.04 only. Installs userspace libraries/toolkit, never an NVIDIA driver.
set -euo pipefail
source "$(dirname "$0")/env.sh"
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 22.04 ]] || exit 2
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential pkg-config libvulkan1 mesa-vulkan-drivers vulkan-tools \
    libgl1 libegl1 libglib2.0-0 libxrender1 libsm6 ffmpeg curl ca-certificates unzip aria2
if [[ ! -x /usr/local/cuda-12.8/bin/nvcc ]]; then
    curl --fail --location --retry 3 \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
        -o "$TMPDIR/cuda-keyring.deb"
    dpkg --force-confmiss -i "$TMPDIR/cuda-keyring.deb"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-cccl-12-8 libcurand-dev-12-8
fi
dpkg-query -W > "$TD_VLA_STORAGE_ROOT/logs/p1/system-packages.tsv"
