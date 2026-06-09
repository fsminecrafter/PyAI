// ─────────────────────────────────────────────────────────────────────────────
// downloader.cpp  —  cross-platform Gutenberg book downloader
// ─────────────────────────────────────────────────────────────────────────────
#include "downloader.h"
#include "utils.h"
#include "neurallm.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Platform HTTP layer
// ─────────────────────────────────────────────────────────────────────────────

#ifdef NLM_WINDOWS
// ── Windows: WinHTTP ─────────────────────────────────────────────────────────
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")

static std::string http_get(const std::string& url,
                             int timeout_sec,
                             bool head_only = false) {
    // Parse URL
    wchar_t buf[2048];
    MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, buf, 2048);

    URL_COMPONENTS uc{};
    uc.dwStructSize = sizeof(uc);
    wchar_t host[512]{}, path[1024]{};
    uc.lpszHostName    = host; uc.dwHostNameLength    = 512;
    uc.lpszUrlPath     = path; uc.dwUrlPathLength     = 1024;
    if (!WinHttpCrackUrl(buf, 0, 0, &uc)) return {};

    HINTERNET hSes = WinHttpOpen(L"neurallm/1.0",
                                  WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                  WINHTTP_NO_PROXY_NAME,
                                  WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSes) return {};
    HINTERNET hCon = WinHttpConnect(hSes, host, uc.nPort, 0);
    HINTERNET hReq = WinHttpOpenRequest(hCon, head_only ? L"HEAD" : L"GET",
                                         path, nullptr, WINHTTP_NO_REFERER,
                                         WINHTTP_DEFAULT_ACCEPT_TYPES,
                                         (uc.nScheme == INTERNET_SCHEME_HTTPS)
                                         ? WINHTTP_FLAG_SECURE : 0);
    DWORD to_ms = static_cast<DWORD>(timeout_sec * 1000);
    WinHttpSetTimeouts(hReq, to_ms, to_ms, to_ms, to_ms);
    WinHttpSendRequest(hReq, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                       WINHTTP_NO_REQUEST_DATA, 0, 0, 0);
    WinHttpReceiveResponse(hReq, nullptr);

    std::string body;
    DWORD avail = 0, read = 0;
    do {
        WinHttpQueryDataAvailable(hReq, &avail);
        if (!avail) break;
        std::string chunk(avail, 0);
        WinHttpReadData(hReq, chunk.data(), avail, &read);
        body.append(chunk.data(), read);
    } while (avail > 0 && !head_only);

    WinHttpCloseHandle(hReq);
    WinHttpCloseHandle(hCon);
    WinHttpCloseHandle(hSes);
    return body;
}

#else
// ── Linux/macOS: POSIX sockets + HTTP/1.1 ────────────────────────────────────
#include <sys/socket.h>
#include <sys/time.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>

static std::string http_get(const std::string& url,
                             int timeout_sec,
                             bool head_only = false) {
    // Parse http[s]://host[:port]/path
    std::regex re(R"(https?://([^/:]+)(?::(\d+))?(/.*)?)");
    std::smatch m;
    if (!std::regex_match(url, m, re)) return {};
    std::string host = m[1];
    std::string port = m[2].matched ? m[2].str() : "80";
    std::string path = m[3].matched ? m[3].str() : "/";
    if (path.empty()) path = "/";

    // Resolve + connect
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) return {};

    int fd = -1;
    for (auto* r = res; r; r = r->ai_next) {
        fd = socket(r->ai_family, r->ai_socktype, r->ai_protocol);
        if (fd < 0) continue;
        // Set send/recv timeout
        struct timeval tv{ timeout_sec, 0 };
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        if (connect(fd, r->ai_addr, r->ai_addrlen) == 0) break;
        ::close(fd); fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) return {};

    // Send request
    std::string req = (head_only ? "HEAD " : "GET ") + path + " HTTP/1.1\r\n"
                    + "Host: " + host + "\r\n"
                    + "User-Agent: neurallm/1.0\r\n"
                    + "Connection: close\r\n\r\n";
    ::send(fd, req.data(), req.size(), 0);

    // Read response
    std::string raw;
    raw.reserve(256 * 1024);
    char buf[65536];
    int n;
    while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0)
        raw.append(buf, n);
    ::close(fd);

    // Strip HTTP headers
    auto pos = raw.find("\r\n\r\n");
    if (pos == std::string::npos) return {};
    return raw.substr(pos + 4);
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Gutenberg helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::string book_url(int id) {
    // e.g. https://www.gutenberg.org/cache/epub/1/pg1.txt
    return "https://www.gutenberg.org/cache/epub/"
         + std::to_string(id) + "/pg" + std::to_string(id) + ".txt";
}

static bool probe_id(int id) {
    auto body = http_get(book_url(id), GB_PROBE_TIMEOUT, /*head*/false);
    // Accept if we got at least AD_MIN_BYTES bytes (not a redirect/404 page)
    return body.size() >= GB_MIN_BYTES && body.rfind("HTTP/", 0) != 0;
}

static std::set<int> already_downloaded(const std::string& folder) {
    std::set<int> have;
    if (!fs::exists(folder)) return have;
    std::regex re(R"(pg(\d+)\.txt)", std::regex::icase);
    for (auto& e : fs::directory_iterator(folder)) {
        std::smatch m;
        std::string name = e.path().filename().string();
        if (std::regex_match(name, m, re))
            have.insert(std::stoi(m[1]));
    }
    return have;
}

// Download one book; return dest path or "" on failure.
static std::string download_book(int id, const std::string& folder,
                                  std::atomic<size_t>* bytes_done = nullptr) {
    std::string dest = folder + "/pg" + std::to_string(id) + ".txt";
    if (fs::exists(dest)) return {};

    auto body = http_get(book_url(id), GB_DL_TIMEOUT);
    if (body.size() < GB_MIN_BYTES) return {};

    std::ofstream f(dest, std::ios::binary);
    if (!f) return {};
    f.write(body.data(), static_cast<std::streamsize>(body.size()));
    if (bytes_done) bytes_done->fetch_add(body.size());
    return dest;
}

// Discover usable IDs via concurrent probing.
static std::vector<int> discover_ids(int needed,
                                      const std::set<int>& skip,
                                      std::set<int>& discovered,
                                      std::set<int>& rejected,
                                      int probe_workers = GB_PROBE_WORKERS,
                                      bool verbose = false) {
    std::vector<int> all;
    all.reserve(GB_ID_MAX - GB_ID_MIN + 1);
    for (int i = GB_ID_MIN; i <= GB_ID_MAX; ++i)
        if (!skip.count(i) && !rejected.count(i) && !discovered.count(i))
            all.push_back(i);
    std::mt19937 rng(std::random_device{}());
    std::shuffle(all.begin(), all.end(), rng);

    std::vector<int> good;
    std::mutex mu;
    std::atomic<int> cursor{0};

    auto worker = [&]() {
        while (true) {
            int idx = cursor.fetch_add(1);
            if (idx >= static_cast<int>(all.size())) return;
            int id = all[idx];
            bool ok = probe_id(id);
            std::lock_guard<std::mutex> lk(mu);
            (ok ? discovered : rejected).insert(id);
            if (ok) good.push_back(id);
        }
    };

    int batch = probe_workers * 4;
    size_t pos = 0;
    while (static_cast<int>(good.size()) < needed && pos < all.size()) {
        size_t end = std::min(pos + static_cast<size_t>(batch), all.size());
        cursor.store(static_cast<int>(pos));
        std::vector<std::thread> ths;
        for (int i = 0; i < probe_workers; ++i)
            ths.emplace_back(worker);
        for (auto& t : ths) t.join();
        pos = end;
        if (verbose) {
            printf("\r  Probing…  found %d/%d usable IDs   ", (int)good.size(), needed);
            fflush(stdout);
        }
        if (static_cast<int>(good.size()) >= needed) break;
    }
    if (verbose) printf("\n");
    return good;
}

// Provide a list of IDs ready to download (from cache + probe if needed).
static std::vector<int> ready_ids(const std::string& folder, int needed,
                                   bool verbose = false) {
    auto discovered = load_id_cache("discovered.json");
    auto rejected   = load_id_cache("rejected.json");
    auto have       = already_downloaded(folder);

    std::set<int> skip = have;
    skip.insert(rejected.begin(), rejected.end());

    std::vector<int> queued;
    for (int id : discovered)
        if (!have.count(id)) queued.push_back(id);

    std::mt19937 rng(std::random_device{}());
    std::shuffle(queued.begin(), queued.end(), rng);

    if (static_cast<int>(queued.size()) < needed) {
        int more = needed - static_cast<int>(queued.size());
        if (verbose) printf("  Discovery: probing for %d more usable IDs…\n", more);
        auto newids = discover_ids(more, skip, discovered, rejected,
                                    GB_PROBE_WORKERS, verbose);
        for (int id : newids) queued.push_back(id);
        save_id_cache("discovered.json", discovered);
        save_id_cache("rejected.json",   rejected);
    }

    std::shuffle(queued.begin(), queued.end(), rng);
    if (static_cast<int>(queued.size()) > needed)
        queued.resize(needed);
    return queued;
}

// ─────────────────────────────────────────────────────────────────────────────
// Serial download
// ─────────────────────────────────────────────────────────────────────────────

static void download_serial(const std::vector<int>& ids,
                              const std::string& folder,
                              int max_books, size_t max_bytes) {
    std::set<int> rejected_local;
    int n = static_cast<int>(ids.size());
    for (int i = 0; i < n; ++i) {
        auto st = folder_state(folder);
        if (st.count >= max_books || st.bytes >= max_bytes) break;
        printf("\r  %s  Downloading pg%d.txt …%s",
               format_bar(i, n, 28).c_str(), ids[i], "           ");
        fflush(stdout);
        auto path = download_book(ids[i], folder);
        if (!path.empty()) {
            size_t sz = fs::file_size(path);
            printf("\r  ✓ pg%d.txt  (%s)%s\n", ids[i],
                   human_bytes(sz).c_str(), "                    ");
        } else {
            printf("\r  ✗ pg%d.txt  (failed)%s\n", ids[i], "                    ");
            rejected_local.insert(ids[i]);
        }
        fflush(stdout);
    }
    if (!rejected_local.empty()) {
        auto rej  = load_id_cache("rejected.json");
        auto disc = load_id_cache("discovered.json");
        rej.insert(rejected_local.begin(), rejected_local.end());
        for (int id : rejected_local) disc.erase(id);
        save_id_cache("rejected.json",   rej);
        save_id_cache("discovered.json", disc);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel download with live progress
// ─────────────────────────────────────────────────────────────────────────────

static void download_parallel(const std::vector<int>& ids,
                               const std::string& folder,
                               int max_books, size_t max_bytes,
                               int workers) {
    int n = static_cast<int>(ids.size());
    workers = std::min(workers, n);
    printf("  Downloading %d books with %d threads…\n\n", n, workers);

    struct BookStatus { int id; std::string state; size_t done=0, total=0; };
    std::vector<BookStatus> status(n);
    for (int i = 0; i < n; ++i) status[i].id = ids[i];
    std::mutex mu;
    std::atomic<int> finished{0};
    std::set<int> rejected_local;
    double started = now_sec();
    int prev_lines = 0;

    auto render = [&]() {
        if (prev_lines > 0) {
            // Move cursor up
            printf("\033[%dA\033[J", prev_lines);
        }
        std::lock_guard<std::mutex> lk(mu);
        int bar_w = 20;
        int lines = 0;
        for (auto& st : status) {
            if (st.state == "done") {
                printf("  pg%-6d  [%s] ✓  %s\n", st.id,
                       std::string(bar_w,'=').c_str(),
                       human_bytes(st.done).c_str());
            } else if (st.state == "failed" || st.state == "too_small") {
                printf("  pg%-6d  [%-*s] FAILED\n", st.id, bar_w, "✗");
            } else if (st.state == "connecting") {
                printf("  pg%-6d  [%-*s] connecting…\n", st.id, bar_w, "…");
            } else if (st.state == "downloading") {
                int filled = (st.total > 0)
                    ? static_cast<int>((double)st.done / st.total * bar_w)
                    : std::min((int)(st.done / (GB_MIN_BYTES * 4.0) * bar_w), bar_w-1);
                std::string bar(filled, '=');
                bar.push_back('>');
                bar.append(bar_w - filled - 1, '-');
                printf("  pg%-6d  [%s] %s\n", st.id, bar.c_str(),
                       human_bytes(st.done).c_str());
            } else {
                printf("  pg%-6d  [pending]\n", st.id);
            }
            ++lines;
        }
        int done_n = finished.load();
        double elapsed = std::max(0.001, now_sec() - started);
        std::string eta;
        if (done_n > 0 && n > 0) {
            double e = (elapsed / done_n) * (n - done_n);
            char buf[32]; snprintf(buf, sizeof(buf), "ETA %.0fs", e);
            eta = buf;
        } else eta = "ETA…";
        printf("  Overall %s %d/%d books  %s  elapsed %.0fs\n",
               format_bar(done_n, n, 30).c_str(), done_n, n,
               eta.c_str(), elapsed);
        ++lines;
        prev_lines = lines;
        fflush(stdout);
    };

    // Launch workers
    std::vector<std::future<void>> futures;
    std::atomic<int> cursor{0};

    for (int w = 0; w < workers; ++w) {
        futures.push_back(std::async(std::launch::async, [&]() {
            while (true) {
                int idx = cursor.fetch_add(1);
                if (idx >= n) return;
                auto st_ptr = &status[idx];
                {
                    std::lock_guard<std::mutex> lk(mu);
                    st_ptr->state = "connecting";
                }
                // Check cap
                {
                    auto fs = folder_state(folder);
                    if (fs.count >= max_books || fs.bytes >= max_bytes) {
                        std::lock_guard<std::mutex> lk(mu);
                        st_ptr->state = "skipped";
                        finished.fetch_add(1);
                        continue;
                    }
                }
                {
                    std::lock_guard<std::mutex> lk(mu);
                    st_ptr->state = "downloading";
                }
                auto body = http_get(book_url(ids[idx]), GB_DL_TIMEOUT);
                if (body.size() < GB_MIN_BYTES) {
                    std::lock_guard<std::mutex> lk(mu);
                    st_ptr->state = "failed";
                    rejected_local.insert(ids[idx]);
                    finished.fetch_add(1);
                    continue;
                }
                std::string dest = folder + "/pg" + std::to_string(ids[idx]) + ".txt";
                std::ofstream f(dest, std::ios::binary);
                if (f) f.write(body.data(), static_cast<std::streamsize>(body.size()));
                std::lock_guard<std::mutex> lk(mu);
                st_ptr->done  = body.size();
                st_ptr->total = body.size();
                st_ptr->state = "done";
                finished.fetch_add(1);
            }
        }));
    }

    // Render loop
    std::atomic<bool> stop_render{false};
    std::thread render_thread([&]() {
        while (!stop_render.load()) {
            render();
            sleep_ms(250);
        }
    });

    for (auto& f : futures) f.get();
    stop_render.store(true);
    render_thread.join();
    render();  // final frame

    if (!rejected_local.empty()) {
        auto rej  = load_id_cache("rejected.json");
        auto disc = load_id_cache("discovered.json");
        rej.insert(rejected_local.begin(), rejected_local.end());
        for (int id : rejected_local) disc.erase(id);
        save_id_cache("rejected.json",   rej);
        save_id_cache("discovered.json", disc);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

void auto_download_blocking(Settings& s, bool multithreaded,
                             const std::string& label) {
    ensure_folder(s.input_folder);
    auto st = folder_state(s.input_folder);
    if (st.count >= s.ad_max_books || st.bytes >= s.ad_max_bytes) {
        printf("%sCap already reached (%d/%d books).\n",
               label.c_str(), st.count, s.ad_max_books);
        return;
    }

    int slots = s.ad_max_books - st.count;
    printf("%sAuto-download: %d/%d books — fetching up to %d more…\n",
           label.c_str(), st.count, s.ad_max_books, slots);

    auto ids = ready_ids(s.input_folder, slots * 2, /*verbose*/true);
    if (ids.empty()) { printf("%sNo usable IDs found.\n", label.c_str()); return; }
    if (static_cast<int>(ids.size()) > slots) ids.resize(slots);

    if (multithreaded && ids.size() > 1)
        download_parallel(ids, s.input_folder, s.ad_max_books, s.ad_max_bytes, s.workers);
    else
        download_serial(ids, s.input_folder, s.ad_max_books, s.ad_max_bytes);

    st = folder_state(s.input_folder);
    printf("\n%sDone: %d books, %s.\n",
           label.c_str(), st.count, human_bytes(st.bytes).c_str());
}

// ── Background thread ─────────────────────────────────────────────────────────

AutoDownloader::AutoDownloader(Settings& s) : s_(s) {}

void AutoDownloader::start() {
    stop_flag_.store(false);
    worker_ = std::thread([this]{ run(); });
}

void AutoDownloader::stop() {
    stop_flag_.store(true);
    if (worker_.joinable()) worker_.join();
}

AutoDownloader::~AutoDownloader() { stop(); }

void AutoDownloader::run() {
    ensure_folder(s_.input_folder);
    auto ids = ready_ids(s_.input_folder, s_.ad_max_books * 2, false);
    std::set<int> rejected_local;

    while (!stop_flag_.load()) {
        auto st = folder_state(s_.input_folder);
        bool need_more = (st.count < s_.ad_refill_below) ||
                         (st.bytes < s_.ad_max_bytes && st.count < s_.ad_max_books);
        if (need_more) {
            if (ids.empty())
                ids = ready_ids(s_.input_folder, s_.ad_max_books * 2, false);
            while (!ids.empty() && !stop_flag_.load()) {
                st = folder_state(s_.input_folder);
                if (st.count >= s_.ad_max_books || st.bytes >= s_.ad_max_bytes) break;
                int id = ids.back(); ids.pop_back();
                auto path = download_book(id, s_.input_folder);
                if (path.empty()) rejected_local.insert(id);
            }
        }
        for (int i = 0; i < 5 && !stop_flag_.load(); ++i) sleep_ms(1000);
    }

    if (!rejected_local.empty()) {
        auto rej  = load_id_cache("rejected.json");
        auto disc = load_id_cache("discovered.json");
        rej.insert(rejected_local.begin(), rejected_local.end());
        for (int id : rejected_local) disc.erase(id);
        save_id_cache("rejected.json",   rej);
        save_id_cache("discovered.json", disc);
    }
}
