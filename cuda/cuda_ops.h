#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// cuda_ops.h  —  GPU declarations (WITH_CUDA builds only)
// ─────────────────────────────────────────────────────────────────────────────
#ifdef WITH_CUDA

#include "model.h"
#include <vector>

struct GpuDevice;
std::vector<GpuDevice> cuda_enumerate_devices();

struct CudaWorkspace {
    // ── Forward activation buffers ────────────────────────────────────────────
    int32_t* d_ctx     = nullptr;  // [B×C]
    int32_t* d_targets = nullptr;  // [B]
    float*   d_x       = nullptr;  // [B×I]
    float*   d_pre     = nullptr;  // [B×H]
    float*   d_xhat    = nullptr;  // [B×H]
    float*   d_mu      = nullptr;  // [B]
    float*   d_var     = nullptr;  // [B]
    float*   d_h       = nullptr;  // [B×H]
    float*   d_logits  = nullptr;  // [B×V]

    // ── Backward intermediate buffers ─────────────────────────────────────────
    float*   d_dlogits = nullptr;  // [B×V]
    float*   d_losses  = nullptr;  // [B]
    float*   d_dh      = nullptr;  // [B×H]
    float*   d_dln     = nullptr;  // [B×H]
    float*   d_dxhat   = nullptr;  // [B×H]
    float*   d_dpre    = nullptr;  // [B×H]
    float*   d_dx_emb  = nullptr;  // [B×I]

    // ── Gradient buffers (zeroed each step) ───────────────────────────────────
    float*   d_gE     = nullptr;
    float*   d_gW1    = nullptr;
    float*   d_gb1    = nullptr;
    float*   d_gln_g  = nullptr;
    float*   d_gln_b  = nullptr;
    float*   d_gW2    = nullptr;
    float*   d_gb2    = nullptr;

    // ── Adam moment buffers (persistent across batches) ───────────────────────
    float *d_mE=nullptr,   *d_vE=nullptr;
    float *d_mW1=nullptr,  *d_vW1=nullptr;
    float *d_mb1=nullptr,  *d_vb1=nullptr;
    float *d_mln_g=nullptr,*d_vln_g=nullptr;
    float *d_mln_b=nullptr,*d_vln_b=nullptr;
    float *d_mW2=nullptr,  *d_vW2=nullptr;
    float *d_mb2=nullptr,  *d_vb2=nullptr;

    int alloc_B = 0;
    int alloc_V = 0;
    int alloc_H = 0;
    int alloc_I = 0;
    int adam_t  = 0;  // unused — caller passes step counter

    void alloc(int B, int C, int I, int H, int V);
    void alloc_adam(const HParams& hp);
    void free_all();
    ~CudaWorkspace() { free_all(); }
};

void cuda_set_device(int device_id);

void cuda_upload_params  (Params& p, const HParams& hp);
void cuda_download_params(Params& p, const HParams& hp);
void cuda_free_params    (Params& p);

// Inference forward: downloads logits to host
void cuda_forward(const Params& p, const HParams& hp,
                  const int32_t* ctx_ids_host, int B,
                  float* logits_out_host, FwdCache& cache,
                  CudaWorkspace& ws);

// Full training step: forward + backward + Adam, all on GPU.
// Returns mean cross-entropy loss.
float cuda_train_step(Params& p, const HParams& hp,
                      const int32_t* ctx_host, const int32_t* targets_host, int B,
                      CudaWorkspace& ws,
                      float lr, float b1, float b2, float eps, int adam_t);

// Link-compat stub (unused in training)
float cuda_backward(const Params& p, const HParams& hp,
                    const float* logits_host, const int32_t* targets_host, int B,
                    const FwdCache& cache, Params& grads);

#endif // WITH_CUDA