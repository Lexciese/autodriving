#!/usr/bin/env bash

set -e  # Exit on error

ENV_NAME="autodriving"
YAML_FILE="environment.yml"
NVIDIA_TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"

echo "=== 1. System Dependencies & NVIDIA Prerequisites ==="
sudo apt-get update && sudo apt-get install -y \
    build-essential cmake git pkg-config \
    python3-pip python3-dev \
    libopenblas-dev \
    libjpeg-dev zlib1g-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libhdf5-dev libffi-dev \
    libgtk-3-dev libcanberra-gtk* \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libv4l-dev v4l-utils wget

echo "=== 2. Installing cuSPARSELt (NVIDIA Requirement for JetPack 6 / JP 24.06+) ==="
if [ ! -f "/usr/local/cuda/include/cusparseLt.h" ]; then
    wget https://raw.githubusercontent.com/pytorch/pytorch/5c6af2b583709f6176898c017424dc9981023c28/.ci/docker/common/install_cusparselt.sh
    export CUDA_VERSION=12.2
    bash ./install_cusparselt.sh
    rm install_cusparselt.sh
fi

echo "=== 3. Creating / Updating Conda Environment ==="
eval "$(conda shell.bash hook)"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Updating existing environment '${ENV_NAME}'..."
    conda env update -n ${ENV_NAME} -f ${YAML_FILE} --prune
else
    echo "Creating new environment '${ENV_NAME}'..."
    conda env create -f ${YAML_FILE}
fi

echo "=== 4. Activating Environment ==="
conda activate ${ENV_NAME}

echo "=== 5. Setting CUDA Paths ==="
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export FORCE_CUDA="1"

echo "=== 6. Installing NVIDIA PyTorch & NumPy Pin ==="
# NVIDIA documentation requires NumPy < 2.0.0 (e.g. 1.26.1) for binary compatibility
pip install "numpy<2.0.0"
pip install --no-cache-dir "${NVIDIA_TORCH_URL}"

echo "=== 7. Building TorchVision v0.20.0 from Source ==="
if [ ! -d "vision" ]; then
    git clone --branch v0.20.0 https://github.com/pytorch/vision.git
fi
cd vision
export BUILD_VERSION=0.20.0
python3 setup.py install
cd ..  # Move back out to prevent namespace conflicts during import

echo "=== 8. Compiling CUDA Extensions (torch-scatter) ==="
pip install --no-build-isolation --no-cache-dir torch-scatter hdf5plugin dropblock

echo "=== 9. Building OpenCV with CUDA 8.7 (AGX Orin Architecture) ==="
OPENCV_VERSION="4.9.0"
if [ ! -d "opencv" ]; then
    git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git
    git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git
fi

mkdir -p opencv/build && cd opencv/build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=8.7 \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_LIBV4L=ON \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_EXECUTABLE=$(which python) ..

make -j$(nproc)
make install
cd ../..

echo "=== 10. Verification ==="
python -c "
import torch
import torchvision
import cv2

print('--- Environment Status ---')
print(f'PyTorch Version:     {torch.__version__}')
print(f'TorchVision Version: {torchvision.__version__}')
print(f'CUDA Available:      {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device:          {torch.cuda.get_device_name(0)}')
print(f'OpenCV Version:      {cv2.__version__}')
print(f'OpenCV CUDA Devices: {cv2.cuda.getCudaEnabledDeviceCount()}')
"

echo "=== Cross-check verified and setup completed successfully! ==="