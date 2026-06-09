#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// downloader.h  —  Gutenberg auto-downloader (multi-threaded, cross-platform)
//
// Cross-platform HTTP:
//   Linux:   libcurl-style via POSIX sockets + simple HTTP/1.1
//   Windows: WinHTTP
//
// Both paths are contained in downloader.cpp with platform guards.
// ─────────────────────────────────────────────────────────────────────────────

#include "settings.h"
#include <string>
#include <vector>
#include <set>

// Download books up to the caps in settings.
// multithreaded: true=parallel, false=serial.
void auto_download_blocking(Settings& s,
                             bool multithreaded = true,
                             const std::string& label = "");

// Background downloader thread (used during training when auto_download=true).
class AutoDownloader {
public:
    explicit AutoDownloader(Settings& s);
    void start();
    void stop();
    ~AutoDownloader();

private:
    Settings& s_;
    std::thread worker_;
    std::atomic<bool> stop_flag_{false};
    void run();
};
