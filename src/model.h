#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// model.h  —  Embedding → LayerNorm-tanh hidden → softmax output
//
// CPU path:   hand-rolled SIMD-friendly row-major matrix math (float)
// CUDA path:  same kernels launched via cuda_ops.h when WITH_CUDA defined
// ─────────────────────────────────────────────────────────────────────────────

#include "neurallm.h"
#include "vocab.h"
#include <iosfwd>
#include <unordered_map>
#include <string>

// ── Parameter pack (lives on whichever device is active) ─────────────────────
struct Params {
    // CPU storage (always allocated)
    std::vector<float> E;      // [vocab_size × embed_dim]
    std::vector<float> W1;     // [input_dim   × hidden_dim]
    std::vector<float> b1;     // [hidden_dim]
    std::vector<float> ln_g;   // [hidden_dim]
    std::vector<float> ln_b;   // [hidden_dim]
    std::vector<float> W2;     // [hidden_dim  × vocab_size]
    std::vector<float> b2;     // [vocab_size]

#ifdef WITH_CUDA
    // Device mirrors (allocated on demand by to_device())
    float* d_E    = nullptr;
    float* d_W1   = nullptr;
    float* d_b1   = nullptr;
    float* d_ln_g = nullptr;
    float* d_ln_b = nullptr;
    float* d_W2   = nullptr;
    float* d_b2   = nullptr;
    void   free_device();
    void   upload();    // CPU → GPU
    void   download();  // GPU → CPU
#endif
};

// ── Forward-pass cache (for backprop) ────────────────────────────────────────
struct FwdCache {
    std::vector<int32_t> ctx_ids;  // [B × ctx_len]
    std::vector<float>   x;        // [B × input_dim]
    std::vector<float>   pre;      // [B × hidden_dim]
    std::vector<float>   mu;       // [B]
    std::vector<float>   var;      // [B]
    std::vector<float>   xhat;     // [B × hidden_dim]
    std::vector<float>   h;        // [B × hidden_dim]
    // (logits stored in the output buffer passed to forward())
};

// ── Adam state ────────────────────────────────────────────────────────────────
struct AdamState {
    float lr    = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps   = 1e-8f;
    int   t     = 0;

    std::vector<float> m_E,   v_E;
    std::vector<float> m_W1,  v_W1;
    std::vector<float> m_b1,  v_b1;
    std::vector<float> m_W1g, v_W1g;  // ln_g
    std::vector<float> m_W1b, v_W1b;  // ln_b
    std::vector<float> m_W2,  v_W2;
    std::vector<float> m_b2,  v_b2;

    void init(const Params& p);
    void step(Params& params, const Params& grads);
};

// ── Neural Language Model ─────────────────────────────────────────────────────
class NeuralLM {
public:
    NeuralLM() = default;
    NeuralLM(const HParams& hp);

    // Move weights to/from GPU
    void to_device();   // upload to GPU (no-op on CPU build)
    void to_cpu();      // download from GPU (no-op on CPU build)

    // Forward: ctx_ids [B × ctx_len] → logits [B × vocab_size]
    void forward(const int32_t* ctx_ids, int B,
                 float* logits_out, FwdCache& cache) const;

    // Backward: returns mean cross-entropy loss, fills grads
    float backward(const float* logits, const int32_t* targets, int B,
                   const FwdCache& cache, Params& grads) const;

    // Single-sample inference (always on CPU; used in generate)
    std::vector<float> predict_probs(const int32_t* ctx_ids) const;

    // Serialisation
    void   write(std::ostream& out) const;
    static NeuralLM read(std::istream& in, const HParams& hp);

    // Accessors
    const HParams& hp()     const { return hp_; }
    const Params&  params() const { return p_; }
    Params&        params()       { return p_; }

    int64_t num_params() const;

private:
    HParams hp_;
    Params  p_;
    bool    on_device_ = false;

    void init_weights();
};
