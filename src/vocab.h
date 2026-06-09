#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// vocab.h  —  token ↔ integer dictionary
// ─────────────────────────────────────────────────────────────────────────────

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

class Vocabulary {
public:
    explicit Vocabulary(int max_size = 20000);

    // Build from a flat token list (or multiple lists accumulated externally).
    // Call accumulate() repeatedly, then finalize().
    void accumulate(const std::vector<std::string>& tokens);
    void finalize();

    // Direct build from a single counter (for streaming builds).
    void build_from_counts(const std::unordered_map<std::string, int64_t>& counts);

    int  encode(const std::string& token) const;
    const std::string& decode(int idx) const;

    int  size()   const { return static_cast<int>(id2tok_.size()); }
    int  pad_id() const { return 0; }
    int  unk_id() const { return 1; }

    // Serialisation — raw binary to an open stream
    void   write(std::ostream& out) const;
    static Vocabulary read(std::istream& in);

private:
    int max_size_;
    std::vector<std::string>              id2tok_;
    std::unordered_map<std::string, int>  tok2id_;
    std::unordered_map<std::string, int64_t> counts_;   // transient
};
