// ─────────────────────────────────────────────────────────────────────────────
// model.cpp  —  CPU neural LM implementation
// All inner loops are written for auto-vectorisation (contiguous, unit-stride).
// ─────────────────────────────────────────────────────────────────────────────
#include "model.h"
#include "neurallm.h"

#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <iostream>

#ifdef WITH_CUDA
#include "cuda_ops.h"
NeuralLM::~NeuralLM() {
#ifdef WITH_CUDA
    delete cuda_ws_;
#endif
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static float randn_f(std::mt19937& rng) {
    static std::normal_distribution<float> dist(0.f, 1.f);
    return dist(rng);
}

static void fill_randn(float* p, size_t n, float scale, std::mt19937& rng) {
    for (size_t i = 0; i < n; ++i) p[i] = randn_f(rng) * scale;
}

// Matrix multiply C = A (m×k) × B (k×n) — row-major, no BLAS needed.
// Unrolled 4× in the k-loop for auto-vectorisation.
static void matmul(const float* A, const float* B, float* C,
                   int m, int k, int n) {
    std::fill(C, C + m * n, 0.f);
    for (int i = 0; i < m; ++i) {
        const float* ai = A + i * k;
        float*       ci = C + i * n;
        for (int p = 0; p < k; ++p) {
            float a = ai[p];
            const float* bp = B + p * n;
            for (int j = 0; j < n; ++j)
                ci[j] += a * bp[j];
        }
    }
}

// C^T accumulation: C (k×n) += A^T (m×k)^T × D (m×n)
// i.e. C[p][j] += A[i][p] * D[i][j]
static void matmul_AtB(const float* A, const float* D, float* C,
                        int m, int k, int n) {
    // C is k×n, initialised by caller
    for (int i = 0; i < m; ++i) {
        const float* ai = A + i * k;
        const float* di = D + i * n;
        for (int p = 0; p < k; ++p) {
            float a = ai[p];
            float* cp = C + p * n;
            for (int j = 0; j < n; ++j)
                cp[j] += a * di[j];
        }
    }
}

// D_input (m×k) += D_out (m×n) × W^T (n×k)
static void matmul_ABt(const float* D, const float* W, float* out,
                        int m, int n, int k) {
    // out is m×k, zero-fill by caller
    for (int i = 0; i < m; ++i) {
        const float* di = D + i * n;
        float*       oi = out + i * k;
        for (int j = 0; j < n; ++j) {
            float d = di[j];
            const float* wj = W + j * k;
            for (int p = 0; p < k; ++p)
                oi[p] += d * wj[p];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NeuralLM
// ─────────────────────────────────────────────────────────────────────────────

NeuralLM::NeuralLM(const HParams& hp) : hp_(hp) {
    init_weights();
}

void NeuralLM::init_weights() {
    const int V = hp_.vocab_size;
    const int D = hp_.embed_dim;
    const int C = hp_.ctx_len;
    const int H = hp_.hidden_dim;
    const int I = C * D;   // input_dim

    std::mt19937 rng(42);

    p_.E.resize(V * D);    fill_randn(p_.E.data(),  V * D, 0.1f, rng);
    p_.W1.resize(I * H);   fill_randn(p_.W1.data(), I * H, std::sqrt(2.f / I), rng);
    p_.b1.assign(H, 0.f);
    p_.ln_g.assign(H, 1.f);
    p_.ln_b.assign(H, 0.f);
    p_.W2.resize(H * V);   fill_randn(p_.W2.data(), H * V, std::sqrt(2.f / H), rng);
    p_.b2.assign(V, 0.f);
}

int64_t NeuralLM::num_params() const {
    return static_cast<int64_t>(
        p_.E.size() + p_.W1.size() + p_.b1.size() +
        p_.ln_g.size() + p_.ln_b.size() +
        p_.W2.size() + p_.b2.size()
    );
}

// ── Device management ─────────────────────────────────────────────────────────

void NeuralLM::to_device() {
#ifdef WITH_CUDA
    p_.upload();
    if (!cuda_ws_) cuda_ws_ = new CudaWorkspace();
    on_device_ = true;
#endif
}

void NeuralLM::to_cpu() {
#ifdef WITH_CUDA
    if (on_device_) { p_.download(); on_device_ = false; }
#endif
}

// ── Forward (CPU) ──────────────────────────────────────────────────────────────

void NeuralLM::forward(const int32_t* ctx_ids, int B,
                        float* logits_out, FwdCache& cache) const {
#ifdef WITH_CUDA
    if (on_device_) {
        if (!cuda_ws_) cuda_ws_ = new CudaWorkspace();
        cuda_forward(p_, hp_, ctx_ids, B, logits_out, cache, *cuda_ws_);
        return;
    }
#endif
    const int D = hp_.embed_dim;
    const int C = hp_.ctx_len;
    const int H = hp_.hidden_dim;
    const int V = hp_.vocab_size;
    const int I = C * D;

    // ── Save ctx for backward
    cache.ctx_ids.assign(ctx_ids, ctx_ids + B * C);

    // ── Embedding lookup + concat → x [B×I]
    cache.x.resize(B * I);
    for (int b = 0; b < B; ++b) {
        float* xb = cache.x.data() + b * I;
        for (int t = 0; t < C; ++t) {
            int id = ctx_ids[b * C + t];
            const float* ev = p_.E.data() + id * D;
            std::copy(ev, ev + D, xb + t * D);
        }
    }

    // ── Linear W1 → pre [B×H]
    cache.pre.resize(B * H);
    matmul(cache.x.data(), p_.W1.data(), cache.pre.data(), B, I, H);
    // add bias
    for (int b = 0; b < B; ++b) {
        float* pb = cache.pre.data() + b * H;
        for (int j = 0; j < H; ++j) pb[j] += p_.b1[j];
    }

    // ── LayerNorm
    cache.mu.resize(B);
    cache.var.resize(B);
    cache.xhat.resize(B * H);
    for (int b = 0; b < B; ++b) {
        const float* pb = cache.pre.data() + b * H;
        float mu = 0.f;
        for (int j = 0; j < H; ++j) mu += pb[j];
        mu /= H;
        float var = 0.f;
        for (int j = 0; j < H; ++j) { float d = pb[j]-mu; var += d*d; }
        var = var / H + 1e-5f;
        cache.mu[b]  = mu;
        cache.var[b] = var;
        float rstd   = 1.f / std::sqrt(var);
        float* xh    = cache.xhat.data() + b * H;
        for (int j = 0; j < H; ++j) xh[j] = (pb[j] - mu) * rstd;
    }

    // ── Scale/shift + tanh → h [B×H]
    cache.h.resize(B * H);
    for (int b = 0; b < B; ++b) {
        const float* xh = cache.xhat.data() + b * H;
        float*       hb = cache.h.data()    + b * H;
        for (int j = 0; j < H; ++j)
            hb[j] = std::tanh(p_.ln_g[j] * xh[j] + p_.ln_b[j]);
    }

    // ── Output layer → logits [B×V]
    matmul(cache.h.data(), p_.W2.data(), logits_out, B, H, V);
    for (int b = 0; b < B; ++b) {
        float* lb = logits_out + b * V;
        for (int j = 0; j < V; ++j) lb[j] += p_.b2[j];
    }
}

// ── Backward (CPU) ────────────────────────────────────────────────────────────

float NeuralLM::backward(const float* logits, const int32_t* targets, int B,
                          const FwdCache& cache, Params& grads) const {
    const int D = hp_.embed_dim;
    const int C = hp_.ctx_len;
    const int H = hp_.hidden_dim;
    const int V = hp_.vocab_size;
    const int I = C * D;

    // ── Softmax + cross-entropy loss
    std::vector<float> probs(B * V);
    float loss = 0.f;
    for (int b = 0; b < B; ++b) {
        const float* lb = logits + b * V;
        float*       pb = probs.data() + b * V;
        float mx = *std::max_element(lb, lb + V);
        float sum = 0.f;
        for (int j = 0; j < V; ++j) { pb[j] = std::exp(lb[j] - mx); sum += pb[j]; }
        for (int j = 0; j < V; ++j) pb[j] /= sum;
        loss -= std::log(pb[targets[b]] + 1e-12f);
    }
    loss /= B;

    // ── dlogits
    std::vector<float> dlogits(B * V);
    for (int b = 0; b < B; ++b) {
        float* dl = dlogits.data() + b * V;
        const float* pb = probs.data() + b * V;
        for (int j = 0; j < V; ++j) dl[j] = pb[j] / B;
        dl[targets[b]] -= 1.f / B;
    }

    // ── dW2, db2
    grads.W2.assign(H * V, 0.f);
    grads.b2.assign(V, 0.f);
    matmul_AtB(cache.h.data(), dlogits.data(), grads.W2.data(), B, H, V);
    for (int b = 0; b < B; ++b) {
        const float* dl = dlogits.data() + b * V;
        for (int j = 0; j < V; ++j) grads.b2[j] += dl[j];
    }

    // ── dh
    std::vector<float> dh(B * H, 0.f);
    matmul_ABt(dlogits.data(), p_.W2.data(), dh.data(), B, V, H);

    // ── Through tanh → LayerNorm scale/shift
    // ln output: ln_j = tanh(ln_g_j * xhat_j + ln_b_j)
    // dln = dh * (1 - h^2)
    std::vector<float> dln(B * H);
    for (int b = 0; b < B; ++b) {
        const float* hb  = cache.h.data()    + b * H;
        const float* dhb = dh.data()         + b * H;
        float*       dlb = dln.data()        + b * H;
        for (int j = 0; j < H; ++j)
            dlb[j] = dhb[j] * (1.f - hb[j] * hb[j]);
    }

    grads.ln_g.assign(H, 0.f);
    grads.ln_b.assign(H, 0.f);
    for (int b = 0; b < B; ++b) {
        const float* xh  = cache.xhat.data() + b * H;
        const float* dlb = dln.data()        + b * H;
        for (int j = 0; j < H; ++j) {
            grads.ln_g[j] += dlb[j] * xh[j];
            grads.ln_b[j] += dlb[j];
        }
    }

    // dxhat = dln * ln_g
    std::vector<float> dxhat(B * H);
    for (int b = 0; b < B; ++b) {
        const float* dlb = dln.data()  + b * H;
        float*       dx  = dxhat.data() + b * H;
        for (int j = 0; j < H; ++j) dx[j] = dlb[j] * p_.ln_g[j];
    }

    // ── LayerNorm backward → dpre
    std::vector<float> dpre(B * H);
    for (int b = 0; b < B; ++b) {
        const float* xh  = cache.xhat.data() + b * H;
        const float* dx  = dxhat.data()      + b * H;
        float*       dp  = dpre.data()        + b * H;
        float rstd = 1.f / std::sqrt(cache.var[b]);

        float sum_dx    = 0.f, sum_dx_xh = 0.f;
        for (int j = 0; j < H; ++j) { sum_dx += dx[j]; sum_dx_xh += dx[j]*xh[j]; }

        for (int j = 0; j < H; ++j)
            dp[j] = rstd / H * (H * dx[j] - sum_dx - xh[j] * sum_dx_xh);
    }

    // ── dW1, db1
    grads.W1.assign(I * H, 0.f);
    grads.b1.assign(H, 0.f);
    matmul_AtB(cache.x.data(), dpre.data(), grads.W1.data(), B, I, H);
    for (int b = 0; b < B; ++b) {
        const float* dp = dpre.data() + b * H;
        for (int j = 0; j < H; ++j) grads.b1[j] += dp[j];
    }

    // dx (B×I)
    std::vector<float> dx_emb(B * I, 0.f);
    matmul_ABt(dpre.data(), p_.W1.data(), dx_emb.data(), B, H, I);

    // ── dE: scatter gradients back to embedding rows
    grads.E.assign(static_cast<size_t>(hp_.vocab_size) * D, 0.f);
    for (int b = 0; b < B; ++b) {
        const float* dx = dx_emb.data() + b * I;
        for (int t = 0; t < C; ++t) {
            int id = cache.ctx_ids[b * C + t];
            float* ev = grads.E.data() + id * D;
            const float* dxt = dx + t * D;
            for (int d = 0; d < D; ++d) ev[d] += dxt[d];
        }
    }

    return loss;
}

// ── predict_probs (single sample, always CPU) ─────────────────────────────────

std::vector<float> NeuralLM::predict_probs(const int32_t* ctx_ids) const {
    // Temporarily bring back to CPU if on device
    const bool was_on_device = on_device_;
    if (was_on_device) {
        const_cast<NeuralLM*>(this)->to_cpu();
    }

    const int V = hp_.vocab_size;
    std::vector<float> logits(V);
    FwdCache cache;
    forward(ctx_ids, 1, logits.data(), cache);

    float mx = *std::max_element(logits.begin(), logits.end());
    float sum = 0.f;
    for (float& v : logits) { v = std::exp(v - mx); sum += v; }
    for (float& v : logits) v /= sum;

    if (was_on_device) {
        const_cast<NeuralLM*>(this)->to_device();
    }
    return logits;
}

// ── Serialisation ─────────────────────────────────────────────────────────────

static void write_vec(std::ostream& out, const std::vector<float>& v) {
    int32_t sz = static_cast<int32_t>(v.size());
    out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    out.write(reinterpret_cast<const char*>(v.data()), sz * sizeof(float));
}

static void read_vec(std::istream& in, std::vector<float>& v) {
    int32_t sz = 0;
    in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
    v.resize(static_cast<size_t>(sz));
    in.read(reinterpret_cast<char*>(v.data()), sz * sizeof(float));
}

void NeuralLM::write(std::ostream& out) const {
    // Ensure weights are on CPU before saving
    const bool was = on_device_;
    if (was) const_cast<NeuralLM*>(this)->to_cpu();

    // HParams
    auto& h = hp_;
    out.write(reinterpret_cast<const char*>(&h.vocab_size),  4);
    out.write(reinterpret_cast<const char*>(&h.embed_dim),   4);
    out.write(reinterpret_cast<const char*>(&h.ctx_len),     4);
    out.write(reinterpret_cast<const char*>(&h.hidden_dim),  4);

    write_vec(out, p_.E);
    write_vec(out, p_.W1);
    write_vec(out, p_.b1);
    write_vec(out, p_.ln_g);
    write_vec(out, p_.ln_b);
    write_vec(out, p_.W2);
    write_vec(out, p_.b2);

    if (was) const_cast<NeuralLM*>(this)->to_device();
}

NeuralLM NeuralLM::read(std::istream& in, const HParams& hp_hint) {
    HParams hp;
    in.read(reinterpret_cast<char*>(&hp.vocab_size), 4);
    in.read(reinterpret_cast<char*>(&hp.embed_dim),  4);
    in.read(reinterpret_cast<char*>(&hp.ctx_len),    4);
    in.read(reinterpret_cast<char*>(&hp.hidden_dim), 4);

    NeuralLM m(hp);
    read_vec(in, m.p_.E);
    read_vec(in, m.p_.W1);
    read_vec(in, m.p_.b1);
    read_vec(in, m.p_.ln_g);
    read_vec(in, m.p_.ln_b);
    read_vec(in, m.p_.W2);
    read_vec(in, m.p_.b2);
    return m;
}

// ── Adam ──────────────────────────────────────────────────────────────────────

static void adam_vec(std::vector<float>& param,
                     std::vector<float>& m, std::vector<float>& v,
                     const std::vector<float>& grad,
                     float lr, float b1, float b2, float eps, int t) {
    float bc1 = 1.f - std::pow(b1, t);
    float bc2 = 1.f - std::pow(b2, t);
    float lr_t = lr * std::sqrt(bc2) / bc1;
    for (size_t i = 0; i < param.size(); ++i) {
        m[i] = b1 * m[i] + (1.f - b1) * grad[i];
        v[i] = b2 * v[i] + (1.f - b2) * grad[i] * grad[i];
        param[i] -= lr_t * m[i] / (std::sqrt(v[i]) + eps);
    }
}

void AdamState::init(const Params& p) {
    auto z = [](size_t n){ return std::vector<float>(n, 0.f); };
    m_E  = z(p.E.size());   v_E  = z(p.E.size());
    m_W1 = z(p.W1.size());  v_W1 = z(p.W1.size());
    m_b1 = z(p.b1.size());  v_b1 = z(p.b1.size());
    m_W1g= z(p.ln_g.size());v_W1g= z(p.ln_g.size());
    m_W1b= z(p.ln_b.size());v_W1b= z(p.ln_b.size());
    m_W2 = z(p.W2.size());  v_W2 = z(p.W2.size());
    m_b2 = z(p.b2.size());  v_b2 = z(p.b2.size());
}

void AdamState::step(Params& params, const Params& grads) {
    ++t;
    adam_vec(params.E,     m_E,   v_E,   grads.E,     lr, beta1, beta2, eps, t);
    adam_vec(params.W1,    m_W1,  v_W1,  grads.W1,    lr, beta1, beta2, eps, t);
    adam_vec(params.b1,    m_b1,  v_b1,  grads.b1,    lr, beta1, beta2, eps, t);
    adam_vec(params.ln_g,  m_W1g, v_W1g, grads.ln_g,  lr, beta1, beta2, eps, t);
    adam_vec(params.ln_b,  m_W1b, v_W1b, grads.ln_b,  lr, beta1, beta2, eps, t);
    adam_vec(params.W2,    m_W2,  v_W2,  grads.W2,    lr, beta1, beta2, eps, t);
    adam_vec(params.b2,    m_b2,  v_b2,  grads.b2,    lr, beta1, beta2, eps, t);
}
