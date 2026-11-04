#!/usr/bin/env bash
# Build edge-dit.cpp (pinned commit) with the CUDA performance profile on the Jetson Orin Nano.
# Run on the Jetson, ideally inside tmux:  bash jetson_build_edgedit.sh
# Matches the A100 build profile (performance: cuDNN SDPA, CUDA norm/RoPE/modulation, benchmark
# markers) except for the GPU architecture (sm_87). Multi-GPU options stay off, as on the A100.
set -euo pipefail

EDGE_COMMIT=97cc585d9dbd9ac5e30b78ef2e2c4a0d8bfbe9f2
GGML_COMMIT=f270346d688c7bec7c56bae2ae73343aba6817ad
SRC=${SRC:-$HOME/tools/edge-dit-20261010}
BUILD_DIR=${BUILD_DIR:-build-cuda-performance}
CUDNN_SHIM=${CUDNN_SHIM:-$HOME/tools/cudnn-root}
# cudnn-frontend needs CMake >= 3.23; Ubuntu 22.04 ships 3.22. Use a venv CMake for this build only.
REAL_CMAKE=${REAL_CMAKE:-$HOME/venvs/edgedit-build/bin/cmake}
# build_cuda.sh runs `cmake --build <dir> -j` with no job limit; on the 8 GB Orin unbounded nvcc jobs
# exhaust shared memory and the kernel kills the compiler (observed 2026-10-11). Cap the job count.
BUILD_JOBS=${BUILD_JOBS:-2}
CMAKE_BIN=$HOME/tools/cmake-capped-jobs
cat > "$CMAKE_BIN" <<EOF
#!/usr/bin/env bash
args=()
for a in "\$@"; do [[ "\$a" == "-j" ]] && args+=("-j" "$BUILD_JOBS") || args+=("\$a"); done
exec "$REAL_CMAKE" "\${args[@]}"
EOF
chmod +x "$CMAKE_BIN"
export PATH=/usr/local/cuda/bin:$PATH
"$CMAKE_BIN" --version | head -1

if [[ ! -d "$SRC/.git" ]]; then
  git clone -q https://github.com/THU-MIG/edge-dit.cpp "$SRC"
fi
cd "$SRC"
git checkout -q "$EDGE_COMMIT"
# oneDNN is not referenced by the CUDA build; only ggml is needed.
git submodule update --init third_party/ggml
[[ "$(git rev-parse HEAD)" == "$EDGE_COMMIT" ]] || { echo "edge-dit commit mismatch" >&2; exit 1; }
[[ "$(git -C third_party/ggml rev-parse HEAD)" == "$GGML_COMMIT" ]] || { echo "ggml commit mismatch" >&2; exit 1; }

# JetPack installs cuDNN in the Debian multiarch layout; build_cuda.sh expects <root>/include and <root>/lib.
mkdir -p "$CUDNN_SHIM/include" "$CUDNN_SHIM/lib"
ln -sf /usr/include/cudnn*.h "$CUDNN_SHIM/include/"
ln -sf /usr/include/aarch64-linux-gnu/cudnn*.h "$CUDNN_SHIM/include/"
ln -sf /usr/lib/aarch64-linux-gnu/libcudnn*.so* "$CUDNN_SHIM/lib/"

# nvidia-smi on Jetson does not report compute_cap, so the script's fallback list would omit sm_87.
CMAKE_BIN="$CMAKE_BIN" \
CUDA_ARCHITECTURES=87 \
CUDNN_ROOT="$CUDNN_SHIM" \
ED_BUILD_PROFILE=performance \
ED_AUTO_INSTALL_DEPS=OFF \
ED_FETCH_CUDNN_FRONTEND=ON \
BUILD_DIR="$BUILD_DIR" \
bash scripts/build_cuda.sh

echo "--- build-config.txt"
cat "$BUILD_DIR/build-config.txt" || true
sha256sum "$BUILD_DIR/bin/ed-sample" "$BUILD_DIR/bin/ed-cli"
"$BUILD_DIR/bin/ed-sample" --help > "$BUILD_DIR/ed-sample-help.txt" 2>&1 || true
"$BUILD_DIR/bin/ed-cli" --help > "$BUILD_DIR/ed-cli-help.txt" 2>&1 || true
echo "BUILD_OK"
