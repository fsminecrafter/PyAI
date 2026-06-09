#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// settings.h  —  all runtime configuration
// ─────────────────────────────────────────────────────────────────────────────

#include <string>
#include <cstdint>

struct Settings {
    // ── Files ─────────────────────────────────────────────────────────────
    std::string input_folder   = "input_files";
    std::string model_file     = "model.nlm";
    std::string settings_file  = "settings.json";

    // ── Vocabulary ────────────────────────────────────────────────────────
    int  vocab_size  = 20000;
    bool lowercase   = true;

    // ── Architecture ──────────────────────────────────────────────────────
    int ctx_len     = 6;
    int embed_dim   = 64;
    int hidden_dim  = 256;

    // ── Training ──────────────────────────────────────────────────────────
    int   epochs       = 3;
    int   batch_size   = 256;
    float learning_rate = 0.001f;
    int   workers      = 4;
    bool  single_thread = false;
    bool  show_progress = true;

    // ── Generation ────────────────────────────────────────────────────────
    int   max_generate_tokens = 80;
    float temperature         = 0.8f;
    int   top_k               = 20;

    // ── GPU ───────────────────────────────────────────────────────────────
    bool use_gpu   = false;
    int  gpu_device = 0;

    // ── Auto-downloader ───────────────────────────────────────────────────
    bool   auto_download    = false;
    int    ad_max_books     = 10;
    size_t ad_max_bytes     = 100 * 1024 * 1024; // 100 MB
    int    ad_refill_below  = 5;
    // Data sources bitmask (DS_* constants from neurallm.h)
    int    ad_sources       = 0xF;  // all sources by default
};

// Load from JSON file (fills defaults for missing keys)
Settings load_settings(const std::string& path);
void     save_settings(const Settings& s);
void     ensure_folder(const std::string& path);