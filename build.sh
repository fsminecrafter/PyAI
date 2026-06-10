set -e

echo "Dependencies (apt)"
apt install -y --no-install-recommends \
    build-essential \
    cmake \
    python3-dev \
    python3-pip \
    zlib1g-dev 

echo "Compilation..."
rm -rf build
mkdir build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON
cmake --build build --parallel $(nproc)
mv build/pyai "$PWD"
