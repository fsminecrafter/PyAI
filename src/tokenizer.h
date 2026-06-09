#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// tokenizer.h  —  Unicode-aware word tokenizer + detokenizer
// ─────────────────────────────────────────────────────────────────────────────

#include <string>
#include <vector>
#include <set>

// Split text into tokens (words and punctuation marks).
// If lowercase=true, ASCII letters are lowercased.
// Uses a hand-rolled UTF-8 state machine — no ICU dependency.
std::vector<std::string> tokenize(const std::string& text, bool lowercase = true);

// Re-join tokens into readable text, suppressing spaces around punctuation.
std::string detokenize(const std::vector<std::string>& tokens);

// Load and tokenize a whole file.
std::vector<std::string> tokenize_file(const std::string& path, bool lowercase = true);
