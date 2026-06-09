#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cuda_ops.h  —  GPU forward/backward declarations (WITH_CUDA builds only)
// ─────────────────────────────────────────────────────────────────────────────
#ifdef WITH_CUDA

#include "model.h"
#include <vector>

// List available CUDA devices; returns empty vec if none found.
struct GpuDevice;
std::vector<GpuDevice> cuda_enumerate_devices();

// Set active device
void cuda_set_device(int device_id);

// Upload/download helpers
void cuda_upload_params(Params& p, const HParams& hp);
void cuda_download_params(Params& p, const HParams& hp);
void cuda_free_params(Params& p);

// Forward pass on GPU: ctx_ids [B×C] device ptr → logits [B×V] device ptr
void cuda_forward(const Params& p, const HParams& hp,
                  const int32_t* ctx_ids_host, int B,
                  float* logits_out_host, FwdCache& cache);

// Backward pass on GPU
float cuda_backward(const Params& p, const HParams& hp,
                    const float* logits_host, const int32_t* targets_host, int B,
                    const FwdCache& cache, Params& grads);

#endif // WITH_CUDA
