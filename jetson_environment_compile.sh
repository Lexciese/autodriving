#!/usr/bin/env bash
set -e

SOURCE_ENV="autodriving"
TARGET_ENV="compile"

eval "$(conda shell.bash hook)"

echo "=== 1. Cloning '${SOURCE_ENV}' -> '${TARGET_ENV}' ==="
if conda env list | grep -q "^${TARGET_ENV} "; then
    echo "Environment '${TARGET_ENV}' already exists. Remove it first if you want a fresh clone:"
    echo "  conda env remove -n ${TARGET_ENV}"
    exit 1
fi
conda create -n "${TARGET_ENV}" --clone "${SOURCE_ENV}" -y

echo "=== 2. Activating '${TARGET_ENV}' & Persisting NumPy < 2 Lock ==="
conda activate "${TARGET_ENV}"

CONSTRAINT_FILE="${CONDA_PREFIX}/numpy-constraint.txt"
echo "numpy<2.0.0" > "${CONSTRAINT_FILE}"

# Persist PIP_CONSTRAINT on every `conda activate compile` so future pip installs cannot upgrade NumPy
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d" "${CONDA_PREFIX}/etc/conda/deactivate.d"
cat <<EOF > "${CONDA_PREFIX}/etc/conda/activate.d/numpy_pin.sh"
export PIP_CONSTRAINT="${CONSTRAINT_FILE}"
EOF
cat <<EOF > "${CONDA_PREFIX}/etc/conda/deactivate.d/numpy_pin.sh"
unset PIP_CONSTRAINT
EOF

export PIP_CONSTRAINT="${CONSTRAINT_FILE}"

echo "=== 3. Installing ONNX Export Dependencies ==="
pip install --no-cache-dir "onnx<1.18" "onnxruntime<1.20"

# Optional: onnxsim often compiles C++ from source on aarch64; keep non-fatal
sudo apt-get update && sudo apt-get install -y protobuf-compiler libprotobuf-dev
CMAKE_ARGS="-DONNXSIM_BUILTIN_ORT=OFF" pip install --no-build-isolation --no-cache-dir onnxsim || \
    echo "Warning: onnxsim failed to build on aarch64. Use --no-simplify with compile.py."

echo "=== 4. Verification ==="
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
except Exception:
    print('onnxsim:     not installed (use --no-simplify)')
"
