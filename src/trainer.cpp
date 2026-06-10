// ─────────────────────────────────────────────────────────────────────────────
// trainer.cpp
// ─────────────────────────────────────────────────────────────────────────────
#include "trainer.h"
#include "tokenizer.h"
#include "utils.h"
#include "neurallm.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <thread>
#include <mutex>
#include <future>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <filesystem>

// zlib for gzip
#include <zlib.h>

#ifdef WITH_CUDA
#  include "cuda_ops.h"
#endif
 

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Vocab streaming build
// ─────────────────────────────────────────────────────────────────────────────

VocabBuildResult build_vocab_streaming(const std::vector<std::string>& files,
                                        bool lowercase, int vocab_size,
                                        int workers, bool single_thread,
                                        bool show_progress) {
    VocabBuildResult res;
    std::unordered_map<std::string, int64_t> counts;
    counts.reserve(1 << 20);

    int n_files = static_cast<int>(files.size());
    std::atomic<int> done_count{0};

    if (show_progress) {
        printf("  Pass 1/2 — counting tokens for vocabulary…\n");
        fflush(stdout);
    }

    if (single_thread || workers <= 1) {
        for (int i = 0; i < n_files; ++i) {
            auto toks = tokenize_file(files[i], lowercase);
            for (auto& t : toks) counts[t]++;
            res.total_tokens += static_cast<int64_t>(toks.size());
            if (show_progress) {
                printf("\r  Loading %s %d/%d files  (%s tokens)",
                       format_bar(i+1, n_files, 28).c_str(), i+1, n_files,
                       human_num(static_cast<double>(res.total_tokens)).c_str());
                fflush(stdout);
            }
        }
    } else {
        // Per-worker partial counts merged under mutex
        int actual_workers = std::min(workers, n_files);
        std::vector<std::unordered_map<std::string,int64_t>> partial(actual_workers);
        std::vector<int64_t> worker_tokens(actual_workers, 0);
        std::vector<std::future<void>> futures;
        std::mutex merge_mtx;
        std::atomic<int> file_cursor{0};

        for (int w = 0; w < actual_workers; ++w) {
            futures.push_back(std::async(std::launch::async,
                [&, w]() {
                    while (true) {
                        int idx = file_cursor.fetch_add(1);
                        if (idx >= n_files) break;
                        auto toks = tokenize_file(files[idx], lowercase);
                        for (auto& t : toks) partial[w][t]++;
                        worker_tokens[w] += static_cast<int64_t>(toks.size());
                        done_count.fetch_add(1);
                    }
                }));
        }

        // Progress display while waiting
        while (done_count.load() < n_files) {
            if (show_progress) {
                int dc = done_count.load();
                int64_t total_so_far = 0;
                for (auto& wt : worker_tokens) total_so_far += wt;
                printf("\r  Loading %s %d/%d files  (%s tokens)",
                       format_bar(dc, n_files, 28).c_str(), dc, n_files,
                       human_num(static_cast<double>(total_so_far)).c_str());
                fflush(stdout);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        for (auto& f : futures) f.get();

        for (int w = 0; w < actual_workers; ++w) {
            for (auto& [tok, cnt] : partial[w]) counts[tok] += cnt;
            res.total_tokens += worker_tokens[w];
        }
    }

    if (show_progress) printf("\n");

    res.vocab = Vocabulary(vocab_size);
    res.vocab.build_from_counts(counts);
    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// DatasetChunk
// ─────────────────────────────────────────────────────────────────────────────

void DatasetChunk::build(const std::vector<int32_t>& ids, int ctx_len, int pad_id) {
    int n = static_cast<int>(ids.size());
    N = n;
    X.resize(static_cast<size_t>(n) * ctx_len);
    Y.resize(n);

    // Prepend ctx_len pad tokens
    for (int i = 0; i < n; ++i) {
        Y[i] = ids[i];
        for (int t = 0; t < ctx_len; ++t) {
            int src = i - ctx_len + t;
            X[static_cast<size_t>(i) * ctx_len + t] =
                (src < 0) ? pad_id : ids[src];
        }
    }
}

void DatasetChunk::shuffle() {
    // Fisher-Yates on indices, then apply in-place
    static std::mt19937 rng(std::random_device{}());
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    for (int i = N - 1; i > 0; --i) {
        std::uniform_int_distribution<int> d(0, i);
        std::swap(idx[i], idx[d(rng)]);
    }
    const int C = static_cast<int>(X.size()) / N;
    std::vector<int32_t> Xn(X.size()), Yn(N);
    for (int i = 0; i < N; ++i) {
        int j = idx[i];
        Yn[i] = Y[j];
        std::copy(X.begin() + j*C, X.begin() + j*C+C, Xn.begin() + i*C);
    }
    X = std::move(Xn);
    Y = std::move(Yn);
}

// ─────────────────────────────────────────────────────────────────────────────
// Model I/O (gzip-compressed)
// ─────────────────────────────────────────────────────────────────────────────

bool save_model(const NeuralLM& model, const Vocabulary& vocab, const std::string& path) {
    // Serialize to memory buffer first
    std::ostringstream ss(std::ios::binary);
    vocab.write(ss);
    model.write(ss);
    std::string buf = ss.str();

    // gzip compress
    gzFile gz = gzopen(path.c_str(), "wb9");
    if (!gz) return false;
    gzwrite(gz, buf.data(), static_cast<unsigned>(buf.size()));
    gzclose(gz);
    return true;
}

bool load_model(const std::string& path, NeuralLM& model_out, Vocabulary& vocab_out) {
    if (!fs::exists(path)) return false;

    gzFile gz = gzopen(path.c_str(), "rb");
    if (!gz) return false;

    std::string buf;
    buf.reserve(8 * 1024 * 1024);
    char tmp[65536];
    int n;
    while ((n = gzread(gz, tmp, sizeof(tmp))) > 0)
        buf.append(tmp, n);
    gzclose(gz);

    std::istringstream ss(buf, std::ios::binary);
    vocab_out = Vocabulary::read(ss);
    HParams hp;  // will be overwritten inside NeuralLM::read
    model_out = NeuralLM::read(ss, hp);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunk iterator: yields flat encoded token lists of ≤ chunk_size tokens
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<int32_t>
encode_file(const std::string& path, const Vocabulary& vocab, bool lowercase) {
    auto toks = tokenize_file(path, lowercase);
    std::vector<int32_t> ids;
    ids.reserve(toks.size());
    for (auto& t : toks) ids.push_back(vocab.encode(t));
    return ids;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main training loop
// ─────────────────────────────────────────────────────────────────────────────

void run_train(Settings& s) {
    const bool single = s.single_thread;
    const bool show   = s.show_progress;

    // ── GPU backend ──────────────────────────────────────────────────────────
#ifdef WITH_CUDA
    if (s.use_gpu) {
        auto devs = cuda_enumerate_devices();
        if (s.gpu_device < static_cast<int>(devs.size())) {
            cuda_set_device(s.gpu_device);
            printf("[GPU] Using CUDA device %d: %s\n",
                   s.gpu_device, devs[s.gpu_device].name.c_str());
        } else {
            printf("[GPU] Device %d not found — falling back to CPU.\n", s.gpu_device);
            s.use_gpu = false;
        }
    }
#else
    if (s.use_gpu) printf("[CPU] CUDA not compiled in — running on CPU.\n");
    s.use_gpu = false;
#endif

    ensure_folder(s.input_folder);
    auto files = find_text_files(s.input_folder);
    if (files.empty()) {
        printf("No .txt files found in '%s'.\n", s.input_folder.c_str());
        return;
    }

    printf("\n=== Neural LM Training  [%s] ===\n",
           s.use_gpu ? ("GPU:" + std::to_string(s.gpu_device)).c_str() : "CPU");
    printf("Files: %d  |  Vocab: %d  |  ctx=%d  embed=%d  hidden=%d\n",
           (int)files.size(), s.vocab_size, s.ctx_len, s.embed_dim, s.hidden_dim);
    printf("Epochs: %d  |  Batch: %d  |  LR: %.4f\n",
           s.epochs, s.batch_size, s.learning_rate);
    {
        size_t avail = available_ram_bytes();
        size_t total = total_ram_bytes();
        printf("System RAM: %s available of %s total\n\n",
               human_bytes(avail).c_str(), human_bytes(total).c_str());
    }

    // ── Pass 1: build vocab ──────────────────────────────────────────────────
    auto vr = build_vocab_streaming(files, s.lowercase, s.vocab_size,
                                    s.workers, single, show);
    auto& vocab = vr.vocab;
    printf("  Vocab size: %d  |  Total tokens: %s\n",
           vocab.size(), human_num(static_cast<double>(vr.total_tokens)).c_str());

    size_t chunk_tokens   = estimate_chunk_tokens(s.ctx_len);
    int    chunks_approx  = static_cast<int>(
        std::max<int64_t>(1, vr.total_tokens / static_cast<int64_t>(chunk_tokens)));
    printf("  Chunk size: ~%s tokens  (≈%d chunk(s) per epoch)\n\n",
           human_num(static_cast<double>(chunk_tokens)).c_str(), chunks_approx);

    // ── Init model + Adam ────────────────────────────────────────────────────
    HParams hp;
    hp.vocab_size = vocab.size();
    hp.embed_dim  = s.embed_dim;
    hp.ctx_len    = s.ctx_len;
    hp.hidden_dim = s.hidden_dim;

    NeuralLM model(hp);
    if (s.use_gpu) model.to_device();

    AdamState adam;
    adam.lr    = s.learning_rate;
    adam.init(model.params());

    printf("  Parameters: %s\n", human_num(static_cast<double>(model.num_params())).c_str());

    double global_start = now_sec();

    // ── Training epochs ───────────────────────────────────────────────────────
    for (int epoch = 1; epoch <= s.epochs; ++epoch) {
        double epoch_start   = now_sec();
        double epoch_loss    = 0.0;
        int    epoch_batches = 0;
        int64_t epoch_samples = 0;

        printf("\n── Epoch %d/%d ──────────────────────────────────────\n",
               epoch, s.epochs);

        // Accumulate tokens across files into chunks
        std::vector<int32_t> buf;
        buf.reserve(chunk_tokens + 100000);
        int chunk_idx = 0;

        auto process_chunk = [&](std::vector<int32_t>& ids) {
            ++chunk_idx;
            double chunk_start = now_sec();

            DatasetChunk dc;
            dc.build(ids, s.ctx_len, vocab.pad_id());
            ids.clear();
            dc.shuffle();

            const int N  = dc.N;
            const int C  = s.ctx_len;
            const int V  = vocab.size();
            int batches_in_chunk = (N + s.batch_size - 1) / s.batch_size;

            double chunk_loss    = 0.0;
            int    chunk_batches = 0;

            FwdCache cache;
            Params   grads;

            std::vector<float> logits(static_cast<size_t>(s.batch_size) * V);

            for (int b_start = 0; b_start < N; b_start += s.batch_size) {
                int B = std::min(s.batch_size, N - b_start);
                const int32_t* Xb = dc.X.data() + static_cast<size_t>(b_start) * C;
                const int32_t* Yb = dc.Y.data() + b_start;
 
                float loss;
#ifdef WITH_CUDA
                if (s.use_gpu) {
                    ++adam.t;  // keep CPU adam_t in sync for checkpoint consistency
                    loss = cuda_train_step(model.params(), model.hp(),
                                           Xb, Yb, B,
                                           *model.cuda_ws_,
                                           adam.lr, adam.beta1, adam.beta2, adam.eps,
                                           adam.t);
                } else
#endif
                {
                    if (static_cast<int>(logits.size()) < B * V)
                        logits.resize(static_cast<size_t>(B) * V);
                    model.forward(Xb, B, logits.data(), cache);
                    loss = model.backward(logits.data(), Yb, B, cache, grads);
                    adam.step(model.params(), grads);
                }


                chunk_loss    += loss;
                chunk_batches += 1;
                epoch_loss    += loss;
                epoch_batches += 1;
                epoch_samples += B;

                if (show && (chunk_batches % 20 == 0 || chunk_batches == batches_in_chunk)) {
                    double elapsed  = std::max(0.001, now_sec() - chunk_start);
                    double avg_loss = chunk_loss / chunk_batches;
                    double samp_sec = (chunk_batches * s.batch_size) / elapsed;
                    std::string bar = format_bar(chunk_batches, batches_in_chunk, 24);
                    std::string usage = sys_usage_str();
                    printf("\r  Chunk %d %s %5.1f%%  loss=%.4f  %s/s  |  %s    ",
                           chunk_idx, bar.c_str(),
                           100.0 * chunk_batches / batches_in_chunk,
                           avg_loss, human_num(samp_sec).c_str(),
                           usage.c_str());
                    fflush(stdout);
                }
            }
            if (show) printf("\n");
        };

        for (auto& file : files) {
            auto ids = encode_file(file, vocab, s.lowercase);
            buf.insert(buf.end(), ids.begin(), ids.end());

            while (buf.size() >= chunk_tokens) {
                std::vector<int32_t> chunk(buf.begin(),
                                           buf.begin() + static_cast<ptrdiff_t>(chunk_tokens));
                buf.erase(buf.begin(), buf.begin() + static_cast<ptrdiff_t>(chunk_tokens));
                process_chunk(chunk);
            }
        }
        if (!buf.empty()) process_chunk(buf);

        double epoch_elapsed = now_sec() - epoch_start;
        double avg_loss = epoch_loss / std::max(1, epoch_batches);
        printf("  Epoch %d done — %d chunk(s)  avg loss=%.4f  samples=%s  time=%.1fs\n",
               epoch, chunk_idx, avg_loss,
               human_num(static_cast<double>(epoch_samples)).c_str(), epoch_elapsed);

        // Save checkpoint
        if (model.to_cpu(), save_model(model, vocab, s.model_file)) {
            printf("  Checkpoint saved → %s\n", s.model_file.c_str());
        }
        if (s.use_gpu) model.to_device();
    }

    double total = now_sec() - global_start;
    printf("\nTraining complete in %.1fs\n", total);
    printf("Model saved to: %s\n", s.model_file.c_str());
}
