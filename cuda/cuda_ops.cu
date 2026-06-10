// ─────────────────────────────────────────────────────────────────────────────
// cuda_ops.cu  —  Full GPU forward + backward + Adam in one train step.
//
// cuda_train_step() is the single entry point for training:
//   - uploads ctx+targets (two small H2D transfers)
//   - runs forward entirely on GPU, keeps activations in workspace
//   - runs backward entirely on GPU using those activations
//   - runs Adam entirely on GPU, updates d_E/d_W1/d_W2 etc. in-place
//   - downloads one scalar (mean loss) back to host
//   Zero PCIe traffic except ctx/targets in and loss out.
//
// cuda_forward() is kept for inference (generate/chat path).
// ─────────────────────────────────────────────────────────────────────────────
#ifdef WITH_CUDA

#include "cuda_ops.h"
#include "neurallm.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <algorithm>

// ── Error helpers ─────────────────────────────────────────────────────────────
#define CK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d  %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
    throw std::runtime_error(cudaGetErrorString(_e)); } } while(0)

#define CBK(x) do { cublasStatus_t _e=(x); if(_e!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cuBLAS error %s:%d  status=%d\n",__FILE__,__LINE__,(int)_e); \
    throw std::runtime_error("cuBLAS error"); } } while(0)

static cublasHandle_t s_cublas = nullptr;
static void ensure_cublas() {
    if (!s_cublas) CBK(cublasCreate(&s_cublas));
}

// ── Device enumeration ────────────────────────────────────────────────────────
std::vector<GpuDevice> cuda_enumerate_devices() {
    std::vector<GpuDevice> devs;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return devs;
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
        GpuDevice g; g.id = i; g.name = prop.name; g.total_mem = prop.totalGlobalMem;
        devs.push_back(g);
    }
    return devs;
}

void cuda_set_device(int id) { CK(cudaSetDevice(id)); }

// ── Param upload / download ───────────────────────────────────────────────────
static void alloc_upload(float** dst, const std::vector<float>& src) {
    CK(cudaMalloc(dst, src.size()*sizeof(float)));
    CK(cudaMemcpy(*dst, src.data(), src.size()*sizeof(float), cudaMemcpyHostToDevice));
}
static void download_sync(float* d, std::vector<float>& h) {
    if (!d) return;
    CK(cudaMemcpy(h.data(), d, h.size()*sizeof(float), cudaMemcpyDeviceToHost));
}

void cuda_upload_params(Params& p, const HParams&) {
    alloc_upload(&p.d_E,    p.E);   alloc_upload(&p.d_W1,   p.W1);
    alloc_upload(&p.d_b1,   p.b1);  alloc_upload(&p.d_ln_g, p.ln_g);
    alloc_upload(&p.d_ln_b, p.ln_b);alloc_upload(&p.d_W2,   p.W2);
    alloc_upload(&p.d_b2,   p.b2);
}
void cuda_download_params(Params& p, const HParams&) {
    download_sync(p.d_E,    p.E);   download_sync(p.d_W1,   p.W1);
    download_sync(p.d_b1,   p.b1);  download_sync(p.d_ln_g, p.ln_g);
    download_sync(p.d_ln_b, p.ln_b);download_sync(p.d_W2,   p.W2);
    download_sync(p.d_b2,   p.b2);
}
void cuda_free_params(Params& p) {
    auto f=[](float*& x){ if(x){cudaFree(x);x=nullptr;} };
    f(p.d_E); f(p.d_W1); f(p.d_b1); f(p.d_ln_g); f(p.d_ln_b); f(p.d_W2); f(p.d_b2);
}

// ── Params member methods ─────────────────────────────────────────────────────
void Params::upload()      { cuda_upload_params(*this, {}); }
void Params::download()    { cuda_download_params(*this, {}); }
void Params::free_device() { cuda_free_params(*this); }

// ── CudaWorkspace ─────────────────────────────────────────────────────────────
void CudaWorkspace::alloc(int B, int C, int I, int H, int V) {
    // Reallocate only if batch or model dimensions changed
    if (B == alloc_B && V == alloc_V && H == alloc_H && I == alloc_I) return;
    free_all();

    // Forward activations
    CK(cudaMalloc(&d_ctx,    B*C*sizeof(int32_t)));
    CK(cudaMalloc(&d_targets,B*sizeof(int32_t)));
    CK(cudaMalloc(&d_x,      B*I*sizeof(float)));
    CK(cudaMalloc(&d_pre,    B*H*sizeof(float)));
    CK(cudaMalloc(&d_xhat,   B*H*sizeof(float)));
    CK(cudaMalloc(&d_mu,     B*sizeof(float)));
    CK(cudaMalloc(&d_var,    B*sizeof(float)));
    CK(cudaMalloc(&d_h,      B*H*sizeof(float)));
    CK(cudaMalloc(&d_logits, B*V*sizeof(float)));

    // Backward intermediates
    CK(cudaMalloc(&d_dlogits, B*V*sizeof(float)));
    CK(cudaMalloc(&d_losses,  B*sizeof(float)));
    CK(cudaMalloc(&d_dh,      B*H*sizeof(float)));
    CK(cudaMalloc(&d_dln,     B*H*sizeof(float)));
    CK(cudaMalloc(&d_dxhat,   B*H*sizeof(float)));
    CK(cudaMalloc(&d_dpre,    B*H*sizeof(float)));
    CK(cudaMalloc(&d_dx_emb,  B*I*sizeof(float)));

    // Gradient buffers (persistent, zeroed at start of each backward)
    CK(cudaMalloc(&d_gE,    (size_t)V*(I/C/* =D */)*sizeof(float)));  // V*D — I/C = D
    // Actually store full sizes separately:
    alloc_B = B; alloc_V = V; alloc_H = H; alloc_I = I;

    // d_gE size = V*D where D = I/C  — we don't have C here, store with alloc_adam
}

// Separate alloc for grad + adam buffers (needs full HParams)
void CudaWorkspace::alloc_adam(const HParams& hp) {
    const int V=hp.vocab_size, D=hp.embed_dim, H=hp.hidden_dim;
    const int I=hp.ctx_len*D;

    auto ga=[](float** p, size_t n){
        CK(cudaMalloc(p, n*sizeof(float)));
        CK(cudaMemset(*p, 0, n*sizeof(float)));
    };

    // Grad buffers
    if (!d_gE)    ga(&d_gE,    (size_t)V*D);
    if (!d_gW1)   ga(&d_gW1,   (size_t)I*H);
    if (!d_gb1)   ga(&d_gb1,   H);
    if (!d_gln_g) ga(&d_gln_g, H);
    if (!d_gln_b) ga(&d_gln_b, H);
    if (!d_gW2)   ga(&d_gW2,   (size_t)H*V);
    if (!d_gb2)   ga(&d_gb2,   V);

    // Adam moment buffers (zero-initialised, never reset between batches)
    if (!d_mE)    ga(&d_mE,    (size_t)V*D);  if (!d_vE)    ga(&d_vE,    (size_t)V*D);
    if (!d_mW1)   ga(&d_mW1,   (size_t)I*H);  if (!d_vW1)   ga(&d_vW1,   (size_t)I*H);
    if (!d_mb1)   ga(&d_mb1,   H);             if (!d_vb1)   ga(&d_vb1,   H);
    if (!d_mln_g) ga(&d_mln_g, H);             if (!d_vln_g) ga(&d_vln_g, H);
    if (!d_mln_b) ga(&d_mln_b, H);             if (!d_vln_b) ga(&d_vln_b, H);
    if (!d_mW2)   ga(&d_mW2,   (size_t)H*V);  if (!d_vW2)   ga(&d_vW2,   (size_t)H*V);
    if (!d_mb2)   ga(&d_mb2,   V);             if (!d_vb2)   ga(&d_vb2,   V);
}

void CudaWorkspace::free_all() {
    auto f=[](auto*& x){ if(x){cudaFree(x);x=nullptr;} };
    f(d_ctx);     f(d_targets); f(d_x);      f(d_pre);    f(d_xhat);
    f(d_mu);      f(d_var);     f(d_h);      f(d_logits);
    f(d_dlogits); f(d_losses);  f(d_dh);     f(d_dln);    f(d_dxhat);
    f(d_dpre);    f(d_dx_emb);
    f(d_gE);  f(d_gW1);  f(d_gb1);  f(d_gln_g); f(d_gln_b); f(d_gW2);  f(d_gb2);
    f(d_mE);  f(d_vE);   f(d_mW1);  f(d_vW1);   f(d_mb1);   f(d_vb1);
    f(d_mln_g);f(d_vln_g);f(d_mln_b);f(d_vln_b);
    f(d_mW2); f(d_vW2);  f(d_mb2);  f(d_vb2);
    alloc_B=0; alloc_V=0; alloc_H=0; alloc_I=0; adam_t=0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernels
// ─────────────────────────────────────────────────────────────────────────────

// ── Embedding lookup: one thread per (b,t,d) ─────────────────────────────────
__global__ void k_embed(
        const int32_t* ctx, const float* E, float* x, int C, int D) {
    int b=blockIdx.x, t=blockIdx.y, d=threadIdx.x;
    if (d>=D) return;
    x[b*(C*D)+t*D+d] = E[ctx[b*C+t]*D+d];
}

// ── Bias add: out[i] += bias[i % N] ──────────────────────────────────────────
__global__ void k_add_bias(float* out, const float* bias, int BN) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<BN) out[i] += bias[i % blockDim.x];  // blockDim.x == N when called correctly
}
// Generic version (bias stride passed explicitly)
__global__ void k_add_bias2(float* out, const float* bias, int total, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<total) out[i] += bias[i%N];
}

// ── LayerNorm forward: one block per row, warp shuffle reductions ─────────────
__global__ void k_layernorm_fwd(
        const float* pre, const float* ln_g, const float* ln_b,
        float* xhat, float* mu, float* var_out, float* h_out, int H) {
    int b=blockIdx.x;
    const float* row=pre+b*H;
    float* xh=xhat+b*H, *ho=h_out+b*H;

    // Mean
    float s=0.f;
    for (int j=threadIdx.x;j<H;j+=blockDim.x) s+=row[j];
    for (int m=16;m>0;m>>=1) s+=__shfl_down_sync(0xffffffff,s,m);
    __shared__ float smem[32];
    if (threadIdx.x%32==0) smem[threadIdx.x/32]=s;
    __syncthreads();
    if (threadIdx.x==0){
        float t=0.f; for(int i=0;i<(blockDim.x+31)/32;++i) t+=smem[i];
        smem[0]=t/H;
    }
    __syncthreads();
    float mean=smem[0];
    if (threadIdx.x==0) mu[b]=mean;

    // Variance
    float v=0.f;
    for (int j=threadIdx.x;j<H;j+=blockDim.x){ float d=row[j]-mean; v+=d*d; }
    for (int m=16;m>0;m>>=1) v+=__shfl_down_sync(0xffffffff,v,m);
    if (threadIdx.x%32==0) smem[threadIdx.x/32]=v;
    __syncthreads();
    if (threadIdx.x==0){
        float t=0.f; for(int i=0;i<(blockDim.x+31)/32;++i) t+=smem[i];
        smem[0]=t/H+1e-5f;
    }
    __syncthreads();
    float var=smem[0];
    if (threadIdx.x==0) var_out[b]=var;
    float rstd=rsqrtf(var);

    for (int j=threadIdx.x;j<H;j+=blockDim.x){
        float x=(row[j]-mean)*rstd;
        xh[j]=x;
        ho[j]=tanhf(ln_g[j]*x+ln_b[j]);
    }
}

// ── Softmax + cross-entropy fused: one block per sample ──────────────────────
// Overwrites logits with probs, writes per-sample loss to d_losses.
__global__ void k_softmax_xent(
        float* logits, const int32_t* Y, float* losses, int V) {
    int b=blockIdx.x;
    float* lb=logits+b*V;

    float mx=-1e38f;
    for (int j=threadIdx.x;j<V;j+=blockDim.x) mx=fmaxf(mx,lb[j]);
    for (int m=16;m>0;m>>=1) mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,m));
    __shared__ float smx, ssum;
    if (threadIdx.x==0) smx=mx;
    __syncthreads();

    float s=0.f;
    for (int j=threadIdx.x;j<V;j+=blockDim.x){ lb[j]=expf(lb[j]-smx); s+=lb[j]; }
    for (int m=16;m>0;m>>=1) s+=__shfl_down_sync(0xffffffff,s,m);
    if (threadIdx.x==0) ssum=s;
    __syncthreads();

    int y=Y[b];
    for (int j=threadIdx.x;j<V;j+=blockDim.x) lb[j]/=ssum;
    __syncthreads();
    if (threadIdx.x==0) losses[b]=-logf(lb[y]+1e-12f);
}

// ── dlogits = (probs - onehot) / B ───────────────────────────────────────────
// Reuses the probs already in d_logits (written by k_softmax_xent).
__global__ void k_dlogits(float* probs, const int32_t* Y, int B, int V) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=B*V) return;
    probs[i]/=B;
    if (i%V==Y[i/V]) probs[i]-=1.f/B;
}

// ── tanh backward: dln[i] = dh[i] * (1 - h[i]^2) ────────────────────────────
__global__ void k_dtanh(const float* dh, const float* h, float* dln, int N) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N) dln[i]=dh[i]*(1.f-h[i]*h[i]);
}

// ── LayerNorm backward: one block per row ─────────────────────────────────────
// Input:  dln [B×H] = upstream grad after tanh
// Output: dxhat [B×H], dpre [B×H]
// Accumulates into d_gln_g, d_gln_b atomically.
__global__ void k_layernorm_bwd(
        const float* dln, const float* xhat, const float* var,
        const float* ln_g,
        float* dxhat, float* dpre,
        float* d_gln_g, float* d_gln_b, int H) {
    int b=blockIdx.x;
    const float* dlb=dln +b*H, *xhb=xhat+b*H;
    float* dxhb=dxhat+b*H, *dpb=dpre+b*H;
    float rstd=rsqrtf(var[b]);

    // Accumulate ln_g / ln_b grads and compute dxhat
    for (int j=threadIdx.x;j<H;j+=blockDim.x){
        atomicAdd(&d_gln_g[j], dlb[j]*xhb[j]);
        atomicAdd(&d_gln_b[j], dlb[j]);
        dxhb[j]=dlb[j]*ln_g[j];
    }
    __syncthreads();

    // Two reductions: sum(dxhat), sum(dxhat*xhat)
    float sdx=0.f, sdxxh=0.f;
    for (int j=threadIdx.x;j<H;j+=blockDim.x){
        sdx   +=dxhb[j];
        sdxxh +=dxhb[j]*xhb[j];
    }
    for (int m=16;m>0;m>>=1){
        sdx   +=__shfl_down_sync(0xffffffff,sdx,   m);
        sdxxh +=__shfl_down_sync(0xffffffff,sdxxh, m);
    }
    __shared__ float ss[2];
    if (threadIdx.x==0){ ss[0]=sdx; ss[1]=sdxxh; }
    __syncthreads();
    sdx=ss[0]; sdxxh=ss[1];

    for (int j=threadIdx.x;j<H;j+=blockDim.x)
        dpb[j]=rstd/H*(H*dxhb[j]-sdx-xhb[j]*sdxxh);
}

// ── db accumulation: db[j] += sum_b dpre[b][j] ───────────────────────────────
__global__ void k_accum_bias_grad(const float* dpre, float* db, int B, int H) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if (j>=H) return;
    float s=0.f;
    for (int b=0;b<B;++b) s+=dpre[b*H+j];
    db[j]+=s;
}

// ── Embedding backward: atomicAdd grad into d_gE ─────────────────────────────
__global__ void k_embed_bwd(
        const int32_t* ctx, const float* dx_emb, float* d_gE, int C, int D) {
    int b=blockIdx.x, t=blockIdx.y, d=threadIdx.x;
    if (d>=D) return;
    atomicAdd(&d_gE[ctx[b*C+t]*D+d], dx_emb[b*(C*D)+t*D+d]);
}

// ── Adam parameter update ─────────────────────────────────────────────────────
__global__ void k_adam(
        float* param, float* m, float* v, const float* grad, int N,
        float lr_t, float b1, float b2, float eps) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=N) return;
    float g=grad[i];
    m[i]=b1*m[i]+(1.f-b1)*g;
    v[i]=b2*v[i]+(1.f-b2)*g*g;
    param[i]-=lr_t*m[i]/(sqrtf(v[i])+eps);
}

// ── Zero a buffer ─────────────────────────────────────────────────────────────
__global__ void k_zero(float* p, int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<N) p[i]=0.f;
}
static void gpu_zero(float* p, size_t n){
    int blocks=(n+255)/256;
    k_zero<<<blocks,256>>>(p,n);
}

// ─────────────────────────────────────────────────────────────────────────────
// cuda_train_step  —  full forward + backward + Adam, GPU-only
// Returns mean cross-entropy loss (one float copied to host).
// ─────────────────────────────────────────────────────────────────────────────
float cuda_train_step(Params& p, const HParams& hp,
                      const int32_t* ctx_host, const int32_t* targets_host, int B,
                      CudaWorkspace& ws,
                      float lr, float b1, float b2, float eps, int adam_t) {
    ensure_cublas();

    const int C = hp.ctx_len;
    const int D = hp.embed_dim;
    const int H = hp.hidden_dim;
    const int V = hp.vocab_size;
    const int I = C * D;

    const float one  = 1.f;
    const float zero = 0.f;

    // Workspace
    ws.alloc(B, C, I, H, V);
    ws.alloc_adam(hp);

    // Host -> device
    CK(cudaMemcpy(ws.d_ctx,     ctx_host,     B * C * sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ws.d_targets,  targets_host, B * sizeof(int32_t),     cudaMemcpyHostToDevice));

    // ─────────────────────────────────────────────────────────────────────
    // FORWARD
    // Row-major tensors are handled through cuBLAS as column-major views.
    // ─────────────────────────────────────────────────────────────────────

    // x[B×I] = embedding lookup
    k_embed<<<dim3(B, C), D>>>(ws.d_ctx, p.d_E, ws.d_x, C, D);
    CK(cudaGetLastError());

    // pre[B×H] = x[B×I] @ W1[I×H]
    // cuBLAS computes pre^T[H×B] = W1^T[H×I] @ x^T[I×B]
    CBK(cublasSgemm(s_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    H, B, I,
                    &one,
                    p.d_W1, H,
                    ws.d_x,  I,
                    &zero,
                    ws.d_pre, H));

    k_add_bias2<<<(B * H + 255) / 256, 256>>>(ws.d_pre, p.d_b1, B * H, H);
    CK(cudaGetLastError());

    // h[B×H] = layernorm + tanh
    k_layernorm_fwd<<<B, min(H, 1024)>>>(
        ws.d_pre, p.d_ln_g, p.d_ln_b,
        ws.d_xhat, ws.d_mu, ws.d_var, ws.d_h, H);
    CK(cudaGetLastError());

    // logits[B×V] = h[B×H] @ W2[H×V]
    // cuBLAS computes logits^T[V×B] = W2^T[V×H] @ h^T[H×B]
    CBK(cublasSgemm(s_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    V, B, H,
                    &one,
                    p.d_W2, V,
                    ws.d_h,  H,
                    &zero,
                    ws.d_logits, V));

    k_add_bias2<<<(B * V + 255) / 256, 256>>>(ws.d_logits, p.d_b2, B * V, V);
    CK(cudaGetLastError());

    // Softmax + xent
    k_softmax_xent<<<B, min(V, 1024)>>>(ws.d_logits, ws.d_targets, ws.d_losses, V);
    CK(cudaGetLastError());

    // dlogits = (probs - onehot) / B
    CK(cudaMemcpy(ws.d_dlogits, ws.d_logits, B * V * sizeof(float), cudaMemcpyDeviceToDevice));
    k_dlogits<<<(B * V + 255) / 256, 256>>>(ws.d_dlogits, ws.d_targets, B, V);
    CK(cudaGetLastError());

    // ─────────────────────────────────────────────────────────────────────
    // BACKWARD
    // ─────────────────────────────────────────────────────────────────────

    gpu_zero(ws.d_gE,    (size_t)V * D);
    gpu_zero(ws.d_gW1,   (size_t)I * H);
    gpu_zero(ws.d_gb1,   H);
    gpu_zero(ws.d_gln_g, H);
    gpu_zero(ws.d_gln_b, H);
    gpu_zero(ws.d_gW2,   (size_t)H * V);
    gpu_zero(ws.d_gb2,   V);

    // gW2[H×V] = h^T[H×B] @ dlogits[B×V]
    // In cuBLAS column-major form:
    // gW2^T[V×H] = dlogits^T[V×B] @ h^T[B×H]
    CBK(cublasSgemm(s_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                    V, H, B,
                    &one,
                    ws.d_dlogits, V,
                    ws.d_h,       H,
                    &one,
                    ws.d_gW2,     V));

    // gb2 = sum_b dlogits[b]
    k_accum_bias_grad<<<(V + 255) / 256, 256>>>(ws.d_dlogits, ws.d_gb2, B, V);
    CK(cudaGetLastError());

    // dh[B×H] = dlogits[B×V] @ W2^T[V×H]
    // cuBLAS: dh^T[H×B] = W2^T[H×V] @ dlogits^T[V×B]
    CBK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    H, B, V,
                    &one,
                    p.d_W2,       V,
                    ws.d_dlogits, V,
                    &zero,
                    ws.d_dh,      H));

    // dln = dh * (1 - h^2)
    k_dtanh<<<(B * H + 255) / 256, 256>>>(ws.d_dh, ws.d_h, ws.d_dln, B * H);
    CK(cudaGetLastError());

    // LayerNorm backward
    k_layernorm_bwd<<<B, min(H, 1024)>>>(
        ws.d_dln, ws.d_xhat, ws.d_var, p.d_ln_g,
        ws.d_dxhat, ws.d_dpre,
        ws.d_gln_g, ws.d_gln_b, H);
    CK(cudaGetLastError());

    // gW1[H×I] = dpre^T[H×B] @ x[B×I]
    // cuBLAS result is column-major H×I, which is exactly row-major I×H in memory.
    CBK(cublasSgemm(s_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    H, I, B,
                    &one,
                    ws.d_dpre, H,
                    ws.d_x,    I,
                    &one,
                    ws.d_gW1,   H));

    // gb1 = sum_b dpre[b]
    k_accum_bias_grad<<<(H + 255) / 256, 256>>>(ws.d_dpre, ws.d_gb1, B, H);
    CK(cudaGetLastError());

    // dx_emb[B×I] = dpre[B×H] @ W1^T[H×I]
    // cuBLAS: dx_emb^T[I×B] = W1^T[I×H] @ dpre^T[H×B]
    CBK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    I, B, H,
                    &one,
                    p.d_W1,    H,
                    ws.d_dpre, H,
                    &zero,
                    ws.d_dx_emb, I));

    // Scatter embedding gradients back into gE
    k_embed_bwd<<<dim3(B, C), D>>>(ws.d_ctx, ws.d_dx_emb, ws.d_gE, C, D);
    CK(cudaGetLastError());

    // ─────────────────────────────────────────────────────────────────────
    // ADAM UPDATE
    // ─────────────────────────────────────────────────────────────────────

    float bc1  = 1.f - (float)pow((double)b1, adam_t);
    float bc2  = 1.f - (float)pow((double)b2, adam_t);
    float lr_t = lr * (float)sqrt((double)bc2) / bc1;

    auto adam_step = [&](float* param, float* m, float* v, float* grad, size_t n) {
        k_adam<<<((int)n + 255) / 256, 256>>>(param, m, v, grad, (int)n, lr_t, b1, b2, eps);
        CK(cudaGetLastError());
    };

    adam_step(p.d_E,     ws.d_mE,     ws.d_vE,     ws.d_gE,    (size_t)V * D);
    adam_step(p.d_W1,    ws.d_mW1,    ws.d_vW1,    ws.d_gW1,   (size_t)I * H);
    adam_step(p.d_b1,    ws.d_mb1,    ws.d_vb1,    ws.d_gb1,   H);
    adam_step(p.d_ln_g,  ws.d_mln_g,  ws.d_vln_g,  ws.d_gln_g, H);
    adam_step(p.d_ln_b,  ws.d_mln_b,  ws.d_vln_b,  ws.d_gln_b, H);
    adam_step(p.d_W2,    ws.d_mW2,    ws.d_vW2,    ws.d_gW2,   (size_t)H * V);
    adam_step(p.d_b2,    ws.d_mb2,    ws.d_vb2,    ws.d_gb2,   V);

    // Return mean loss
    std::vector<float> losses_h(B);
    CK(cudaMemcpy(losses_h.data(), ws.d_losses, B * sizeof(float), cudaMemcpyDeviceToHost));

    float mean_loss = 0.f;
    for (float l : losses_h) mean_loss += l;
    return mean_loss / B;
}

// ─────────────────────────────────────────────────────────────────────────────
// cuda_forward  —  inference only (generate/chat), downloads logits to host
// ─────────────────────────────────────────────────────────────────────────────
void cuda_forward(const Params& p, const HParams& hp,
                  const int32_t* ctx_ids_host, int B,
                  float* logits_out_host, FwdCache& cache,
                  CudaWorkspace& ws) {
    ensure_cublas();
    const int C=hp.ctx_len, D=hp.embed_dim, H=hp.hidden_dim, V=hp.vocab_size, I=C*D;
    const float one=1.f, zero=0.f;

    ws.alloc(B, C, I, H, V);
    CK(cudaMemcpy(ws.d_ctx, ctx_ids_host, B*C*sizeof(int32_t), cudaMemcpyHostToDevice));

    k_embed<<<dim3(B,C), D>>>(ws.d_ctx, p.d_E, ws.d_x, C, D);

    CBK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        H, B, I, &one, p.d_W1, I, ws.d_x, I, &zero, ws.d_pre, H));
    k_add_bias2<<<(B*H+255)/256,256>>>(ws.d_pre, p.d_b1, B*H, H);

    k_layernorm_fwd<<<B, min(H,1024)>>>(
        ws.d_pre, p.d_ln_g, p.d_ln_b,
        ws.d_xhat, ws.d_mu, ws.d_var, ws.d_h, H);

    CBK(cublasSgemm(s_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        V, B, H, &one, p.d_W2, H, ws.d_h, H, &zero, ws.d_logits, V));
    k_add_bias2<<<(B*V+255)/256,256>>>(ws.d_logits, p.d_b2, B*V, V);

    CK(cudaMemcpy(logits_out_host, ws.d_logits, B*V*sizeof(float), cudaMemcpyDeviceToHost));

    // Populate cache (needed if caller uses it — inference path ignores it)
    cache.ctx_ids.resize(B*C);
    memcpy(cache.ctx_ids.data(), ctx_ids_host, B*C*sizeof(int32_t));
}

// ── Stub kept for link compatibility ─────────────────────────────────────────
float cuda_backward(const Params&, const HParams&,
                    const float*, const int32_t*, int,
                    const FwdCache&, Params&) {
    return 0.f;
}

#endif // WITH_CUDA
