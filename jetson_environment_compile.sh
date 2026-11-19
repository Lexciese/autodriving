#!/usr/bin/env bash
set -e

SOURCE_ENV="autodriving"
TARGET_ENV="compile"
ORT_CPP_DIR="/opt/onnxruntime"

eval "$(conda shell.bash hook)"

echo "=== 1. Cloning '${SOURCE_ENV}' -> '${TARGET_ENV}' ==="
if conda env list | grep -q "^${TARGET_ENV} "; then
    echo "Environment '${TARGET_ENV}' already exists. Remove it first if you want a fresh clone:"
    echo "  conda env remove -n ${TARGET_ENV}"
    exit 1
fi
conda create -n "${TARGET_ENV}" --clone "${SOURCE_ENV}" -y

echo "=== 2. Activating '${TARGET_ENV}' & Persisting Environment Hooks ==="
conda activate "${TARGET_ENV}"

CONSTRAINT_FILE="${CONDA_PREFIX}/numpy-constraint.txt"
echo "numpy<2.0.0" > "${CONSTRAINT_FILE}"

# Persist PIP_CONSTRAINT and LD_LIBRARY_PATH on every `conda activate compile`
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d" "${CONDA_PREFIX}/etc/conda/deactivate.d"
cat <<EOF > "${CONDA_PREFIX}/etc/conda/activate.d/compile_env_hooks.sh"
export PIP_CONSTRAINT="${CONSTRAINT_FILE}"
export _OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${ORT_CPP_DIR}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
EOF

cat <<EOF > "${CONDA_PREFIX}/etc/conda/deactivate.d/compile_env_hooks.sh"
unset PIP_CONSTRAINT
export LD_LIBRARY_PATH="\${_OLD_LD_LIBRARY_PATH:-}"
unset _OLD_LD_LIBRARY_PATH
EOF

export PIP_CONSTRAINT="${CONSTRAINT_FILE}"
export LD_LIBRARY_PATH="${ORT_CPP_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "=== 3. Downgrading NumPy & Installing ONNX Dependencies ==="
pip install --no-cache-dir --force-reinstall "numpy<2.0.0"
pip install --no-cache-dir "onnx<1.18" "onnxruntime<1.20"

echo "=== 4. Building onnxsim from Source (linux-aarch64) ==="
if [[ "$(uname -m)" == "aarch64" ]]; then
    sudo apt-get update && sudo apt-get install -y \
        build-essential cmake ninja-build protobuf-compiler libprotobuf-dev wget

    ORT_VER=$(python -c "import onnxruntime; print(onnxruntime.__version__)")
    ORT_TAR="onnxruntime-linux-aarch64-${ORT_VER}.tgz"
    ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/${ORT_TAR}"

    echo "Downloading C++ ONNX Runtime v${ORT_VER}..."
    if wget -q --spider "${ORT_URL}"; then
        wget -q --show-progress "${ORT_URL}" -O "/tmp/${ORT_TAR}"
        sudo rm -rf "${ORT_CPP_DIR}"
        sudo mkdir -p "${ORT_CPP_DIR}"
        sudo tar -xzf "/tmp/${ORT_TAR}" -C "${ORT_CPP_DIR}" --strip-components=1
        rm -f "/tmp/${ORT_TAR}"

        pip install --no-cache-dir scikit-build cmake ninja "pybind11[global]"
        CMAKE_ARGS="-DONNXSIM_BUILTIN_ORT=OFF -DONNXRUNTIME_ROOTDIR=${ORT_CPP_DIR}" \
            pip install --no-build-isolation --no-cache-dir onnxsim || \
            echo "Warning: onnxsim C++ build failed. Use --no-simplify with compile.py."
    else
        echo "Warning: No official C++ tarball found at ${ORT_URL}. Skipping onnxsim (use --no-simplify)."
    fi
else
    pip install --no-cache-dir onnxsim
fi

echo "=== 5. Verification ==="
python -c "
import numpy as np
import torch
import onnx

assert int(np.__version__.split('.')[0]) < 2, f'NumPy 2.x detected: {np.__version__}'
print('NumPy:      ', np.__version__)
print('PyTorch:    ', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('ONNX:       ', onnx.__version__)
try:
    import onnxruntime as ort
    print('ONNXRuntime:', ort.__version__)
except Exception as e:
    print('ONNXRuntime: unavailable (use --no-simplify):', e)
try:
    import onnxsim
    print('onnxsim:    ', onnxsim.__version__)
except Exception as e:
    print('onnxsim:     not installed (use --no-simplify):', e)
"
