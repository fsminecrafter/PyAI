#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// neurallm.h  —  shared types, constants, platform abstractions
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <functional>
#include <atomic>

// ── Platform ─────────────────────────────────────────────────────────────────
#ifdef _WIN32
#  define NLM_WINDOWS 1
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  include <io.h>
#else
#  define NLM_LINUX 1
#  include <unistd.h>
#  include <sys/sysinfo.h>
#  include <sys/ioctl.h>
#endif

// ── CUDA guard ───────────────────────────────────────────────────────────────
#ifdef WITH_CUDA
#  include <cuda_runtime.h>
#  define CUDA_CHECK(expr)                                                       \
     do {                                                                        \
       cudaError_t _e = (expr);                                                  \
       if (_e != cudaSuccess) {                                                  \
         fprintf(stderr, "CUDA error %s:%d — %s\n",                             \
                 __FILE__, __LINE__, cudaGetErrorString(_e));                    \
         std::exit(1);                                                           \
       }                                                                         \
     } while(0)
#endif

// ── Float type (change to double for higher precision) ───────────────────────
using Float = float;

// ── Special token strings ─────────────────────────────────────────────────────
inline constexpr const char* PAD_TOKEN = "<PAD>";
inline constexpr const char* UNK_TOKEN = "<UNK>";

// ── Gutenberg constants ───────────────────────────────────────────────────────
inline constexpr int    GB_ID_MIN         = 1;
inline constexpr int    GB_ID_MAX         = 75000;
inline constexpr size_t GB_MIN_BYTES      = 40 * 1024;
inline constexpr int    GB_PROBE_WORKERS  = 6;
inline constexpr int    GB_PROBE_TIMEOUT  = 10;   // seconds
inline constexpr int    GB_DL_TIMEOUT     = 60;   // seconds
inline constexpr double CHUNK_RAM_FRACTION = 0.40;

// ── A flat 2-D array (row-major, heap-allocated, RAII) ───────────────────────
// Used everywhere instead of vector-of-vectors to enable contiguous memcpy
// and pointer arithmetic compatible with CUDA device memcpy.
template<typename T>
struct Matrix2D {
    std::vector<T> data;
    size_t rows = 0, cols = 0;

    Matrix2D() = default;
    Matrix2D(size_t r, size_t c, T init = T{})
        : data(r * c, init), rows(r), cols(c) {}

    T*       ptr()       { return data.data(); }
    const T* ptr() const { return data.data(); }
    T&       at(size_t r, size_t c)       { return data[r * cols + c]; }
    const T& at(size_t r, size_t c) const { return data[r * cols + c]; }
    size_t   numel() const { return rows * cols; }
    void     resize(size_t r, size_t c, T init = T{}) {
        rows = r; cols = c;
        data.assign(r * c, init);
    }
};

// ── Hyper-parameters (subset needed by model) ────────────────────────────────
struct HParams {
    int vocab_size  = 20000;
    int embed_dim   = 64;
    int ctx_len     = 6;
    int hidden_dim  = 256;
};

// ── GPU device info ───────────────────────────────────────────────────────────
struct GpuDevice {
    int         id   = -1;
    std::string name;
    size_t      total_mem = 0;
};
