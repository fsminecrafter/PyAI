#!/usr/bin/env python3
"""
Pure-Python + numpy Neural Language Model
==========================================
Architecture
  - Embedding matrix  E : [vocab_size × embed_dim]
  - Context window      : last ctx_len tokens → concat embeddings → flat vec
  - Hidden layer        : Linear → LayerNorm → tanh
  - Output layer        : Linear → softmax over vocab
  - Loss                : cross-entropy
  - Optimiser           : Adam (pure numpy)

Everything else kept from the n-gram version:
  - Project Gutenberg Auto-Data Downloader (dynamic ID discovery)
  - Threaded / single-thread file loading with progress bar
  - Interactive settings menu
  - Save / load (npz weights + json vocab, gzip'd)
  - Chat / generate mode with temperature + top-k sampling
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import pickle
import random
import re
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

GUTENBERG_URL       = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
GUTENBERG_ID_MIN    = 1
GUTENBERG_ID_MAX    = 75000
AD_MIN_BYTES        = 40 * 1024
AD_PROBE_WORKERS    = 6
AD_PROBE_TIMEOUT    = 10
AD_DOWNLOAD_TIMEOUT = 60
DISCOVERED_FILE     = "discovered.json"
REJECTED_FILE       = "rejected.json"

WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", "%", ")", "]", "}", "»", "\u201d", "'"}
NO_SPACE_AFTER  = {"(", "[", "{", "«", "\u201c", "'", "$", "£", "€"}

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

DEFAULT_SETTINGS = {
    "input_folder":        "input_files",
    "model_file":          "model.npz",
    "settings_file":       "settings.json",
    # ── Vocabulary ───────────────────────────────────────────────────────
    "vocab_size":          20000,   # most-frequent tokens kept
    "lowercase":           True,
    # ── Architecture ─────────────────────────────────────────────────────
    "ctx_len":             6,       # context window (tokens fed in)
    "embed_dim":           64,      # embedding dimensionality
    "hidden_dim":          256,     # hidden layer width
    # ── Training ─────────────────────────────────────────────────────────
    "epochs":              3,
    "batch_size":          256,
    "learning_rate":       0.001,
    "workers":             max(1, (os.cpu_count() or 2) // 2),
    "single_thread":       False,
    "show_progress":       True,
    # ── Generation ───────────────────────────────────────────────────────
    "max_generate_tokens": 80,
    "temperature":         0.8,
    "top_k":               20,
    # ── Auto-Data Downloader ─────────────────────────────────────────────
    "auto_download":       False,
    "ad_max_books":        10,
    "ad_max_bytes":        104857600,
    "ad_refill_below":     5,
}


# ─────────────────────────────────────────────────────────────────────────────
# Settings I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_settings(path: str) -> dict:
    s = dict(DEFAULT_SETTINGS)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                s.update(loaded)
        except Exception:
            pass
    return s


def save_settings(s: dict) -> None:
    with open(s["settings_file"], "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)


def ensure_folder(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str, lowercase: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    return WORD_RE.findall(text)


def detokenize(tokens: List[str]) -> str:
    if not tokens:
        return ""
    out = tokens[0]
    for tok in tokens[1:]:
        if tok in NO_SPACE_BEFORE:
            out += tok
        elif out and out[-1] in NO_SPACE_AFTER:
            out += tok
        else:
            out += " " + tok
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

class Vocabulary:
    def __init__(self, max_size: int = 20000) -> None:
        self.max_size  = max_size
        self.tok2id: Dict[str, int] = {}
        self.id2tok: List[str]      = []

    def build(self, token_lists: List[List[str]]) -> None:
        counts: Counter = Counter()
        for toks in token_lists:
            counts.update(toks)

        special = [PAD_TOKEN, UNK_TOKEN]
        most_common = [t for t, _ in counts.most_common(self.max_size - len(special))]
        self.id2tok = special + most_common
        self.tok2id = {t: i for i, t in enumerate(self.id2tok)}

    def encode(self, token: str) -> int:
        return self.tok2id.get(token, self.tok2id[UNK_TOKEN])

    def decode(self, idx: int) -> str:
        if 0 <= idx < len(self.id2tok):
            return self.id2tok[idx]
        return UNK_TOKEN

    @property
    def size(self) -> int:
        return len(self.id2tok)

    def to_dict(self) -> dict:
        return {"max_size": self.max_size, "id2tok": self.id2tok}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocabulary":
        v = cls(max_size=d["max_size"])
        v.id2tok = d["id2tok"]
        v.tok2id = {t: i for i, t in enumerate(v.id2tok)}
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Adam optimiser state
# ─────────────────────────────────────────────────────────────────────────────

class AdamState:
    def __init__(self, lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8) -> None:
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def step(self, params: Dict[str, np.ndarray],
             grads:  Dict[str, np.ndarray]) -> None:
        self.t += 1
        for name, g in grads.items():
            if name not in self.m:
                self.m[name] = np.zeros_like(params[name])
                self.v[name] = np.zeros_like(params[name])
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g * g
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_dict(self) -> dict:
        return {
            "lr": self.lr, "beta1": self.beta1, "beta2": self.beta2,
            "eps": self.eps, "t": self.t,
            "m_keys":   list(self.m.keys()),
            "v_keys":   list(self.v.keys()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Neural Language Model
# ─────────────────────────────────────────────────────────────────────────────

class NeuralLM:
    """
    Embedding lookup → concat → hidden (LayerNorm + tanh) → output (softmax).

    Forward pass returns logits (pre-softmax); loss computed externally so
    the backward pass can be written cleanly.
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 ctx_len: int, hidden_dim: int) -> None:
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.ctx_len    = ctx_len
        self.hidden_dim = hidden_dim
        self.input_dim  = ctx_len * embed_dim

        scale_e = 0.1
        scale_h = np.sqrt(2.0 / self.input_dim)
        scale_o = np.sqrt(2.0 / hidden_dim)

        self.params: Dict[str, np.ndarray] = {
            # Embedding
            "E":       np.random.randn(vocab_size, embed_dim).astype(np.float32) * scale_e,
            # Hidden
            "W1":      np.random.randn(self.input_dim, hidden_dim).astype(np.float32) * scale_h,
            "b1":      np.zeros(hidden_dim, dtype=np.float32),
            # LayerNorm (hidden)
            "ln_g":    np.ones(hidden_dim,  dtype=np.float32),
            "ln_b":    np.zeros(hidden_dim, dtype=np.float32),
            # Output
            "W2":      np.random.randn(hidden_dim, vocab_size).astype(np.float32) * scale_o,
            "b2":      np.zeros(vocab_size, dtype=np.float32),
        }

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, ctx_ids: np.ndarray
                ) -> Tuple[np.ndarray, dict]:
        """
        ctx_ids : [B, ctx_len]  int32
        returns : logits [B, vocab_size], cache for backward
        """
        B = ctx_ids.shape[0]
        p = self.params

        # Embedding lookup + flatten
        emb  = p["E"][ctx_ids]                          # [B, ctx_len, embed_dim]
        x    = emb.reshape(B, self.input_dim)            # [B, input_dim]

        # Hidden linear
        pre  = x @ p["W1"] + p["b1"]                    # [B, hidden_dim]

        # LayerNorm
        mu   = pre.mean(axis=1, keepdims=True)
        var  = pre.var( axis=1, keepdims=True) + 1e-5
        xhat = (pre - mu) / np.sqrt(var)
        ln   = p["ln_g"] * xhat + p["ln_b"]             # [B, hidden_dim]

        # tanh
        h    = np.tanh(ln)                               # [B, hidden_dim]

        # Output
        logits = h @ p["W2"] + p["b2"]                  # [B, vocab_size]

        cache = dict(ctx_ids=ctx_ids, emb=emb, x=x,
                     pre=pre, mu=mu, var=var, xhat=xhat,
                     ln=ln, h=h)
        return logits, cache

    # ── backward ─────────────────────────────────────────────────────────

    def backward(self, logits: np.ndarray,
                 targets: np.ndarray,
                 cache: dict) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        targets : [B]  int32
        returns : (mean cross-entropy loss, grads dict)
        """
        B  = logits.shape[0]
        p  = self.params

        # Softmax + cross-entropy loss
        logits_s = logits - logits.max(axis=1, keepdims=True)   # numerical stability
        exp_l    = np.exp(logits_s)
        probs    = exp_l / exp_l.sum(axis=1, keepdims=True)     # [B, V]
        loss     = -np.log(probs[np.arange(B), targets] + 1e-12).mean()

        # dL/d_logits
        dlogits  = probs.copy()
        dlogits[np.arange(B), targets] -= 1
        dlogits /= B                                             # [B, V]

        # Output layer
        dW2 = cache["h"].T @ dlogits                            # [hidden, V]
        db2 = dlogits.sum(axis=0)
        dh  = dlogits @ p["W2"].T                               # [B, hidden]

        # tanh backward
        dln = dh * (1 - cache["h"] ** 2)                       # [B, hidden]

        # LayerNorm backward
        dln_g  = (dln * cache["xhat"]).sum(axis=0)
        dln_b  = dln.sum(axis=0)
        dxhat  = dln * p["ln_g"]
        std    = np.sqrt(cache["var"])
        dpre   = (1 / (B * std)) * (
            B * dxhat
            - dxhat.sum(axis=1, keepdims=True)
            - cache["xhat"] * (dxhat * cache["xhat"]).sum(axis=1, keepdims=True)
        )

        # Hidden layer
        dW1 = cache["x"].T @ dpre
        db1 = dpre.sum(axis=0)
        dx  = dpre @ p["W1"].T                                  # [B, input_dim]

        # Embedding backward
        dx_emb = dx.reshape(B, self.ctx_len, self.embed_dim)    # [B, ctx, E]
        dE     = np.zeros_like(p["E"])
        np.add.at(dE, cache["ctx_ids"], dx_emb)

        grads = {
            "E": dE, "W1": dW1, "b1": db1,
            "ln_g": dln_g, "ln_b": dln_b,
            "W2": dW2, "b2": db2,
        }
        return float(loss), grads

    # ── predict ──────────────────────────────────────────────────────────

    def predict_probs(self, ctx_ids: np.ndarray) -> np.ndarray:
        """ctx_ids: [ctx_len]  → probs [vocab_size]"""
        logits, _ = self.forward(ctx_ids[np.newaxis])
        logits_s  = logits[0] - logits[0].max()
        exp_l     = np.exp(logits_s)
        return exp_l / exp_l.sum()

    # ── serialisation ────────────────────────────────────────────────────

    def to_npz(self) -> io.BytesIO:
        buf = io.BytesIO()
        np.savez_compressed(buf, **self.params)
        buf.seek(0)
        return buf

    @classmethod
    def from_npz(cls, buf: io.BytesIO,
                 vocab_size: int, embed_dim: int,
                 ctx_len: int, hidden_dim: int) -> "NeuralLM":
        m = cls(vocab_size, embed_dim, ctx_len, hidden_dim)
        data = np.load(buf)
        for k in m.params:
            m.params[k] = data[k]
        return m


# ─────────────────────────────────────────────────────────────────────────────
# Model save / load  (gzip'd archive: JSON header + npz weights)
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model: NeuralLM, vocab: Vocabulary,
               hparams: dict, model_file: str) -> None:
    header = {
        "vocab":   vocab.to_dict(),
        "hparams": hparams,
    }
    header_bytes = json.dumps(header).encode("utf-8")
    weights_buf  = model.to_npz()

    with gzip.open(model_file, "wb") as f:
        # 8-byte little-endian length prefix for header
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        f.write(weights_buf.read())


def load_model(model_file: str) -> Optional[Tuple[NeuralLM, Vocabulary, dict]]:
    if not os.path.exists(model_file):
        return None
    with gzip.open(model_file, "rb") as f:
        raw = f.read()

    hlen        = int.from_bytes(raw[:8], "little")
    header      = json.loads(raw[8: 8 + hlen])
    weights_buf = io.BytesIO(raw[8 + hlen:])

    vocab   = Vocabulary.from_dict(header["vocab"])
    hp      = header["hparams"]
    model   = NeuralLM.from_npz(
        weights_buf,
        vocab_size  = vocab.size,
        embed_dim   = hp["embed_dim"],
        ctx_len     = hp["ctx_len"],
        hidden_dim  = hp["hidden_dim"],
    )
    return model, vocab, hp


# ─────────────────────────────────────────────────────────────────────────────
# Training data builder
# ─────────────────────────────────────────────────────────────────────────────

def load_tokens_from_file(path: str, lowercase: bool) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return tokenize(text, lowercase=lowercase)


def build_dataset(token_lists: List[List[str]],
                  vocab: Vocabulary,
                  ctx_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns X [N, ctx_len] int32, Y [N] int32.
    Each sample: context of ctx_len tokens → next token.
    PAD_ID (0) is used to left-pad the start of each file.
    """
    PAD_ID = vocab.tok2id[PAD_TOKEN]
    xs, ys = [], []

    for toks in token_lists:
        ids = [vocab.encode(t) for t in toks]
        # pad the start so every token gets a training sample
        padded = [PAD_ID] * ctx_len + ids
        for i in range(ctx_len, len(padded)):
            xs.append(padded[i - ctx_len: i])
            ys.append(padded[i])

    X = np.array(xs, dtype=np.int32)
    Y = np.array(ys, dtype=np.int32)
    return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: progress, formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_bar(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + "=" * width + "]"
    ratio  = max(0.0, min(1.0, done / total))
    filled = int(ratio * width)
    return "[" + "=" * filled + "-" * (width - filled) + "]"


def human_num(n: float) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{int(n)}"
        n /= 1000
    return f"{n:.1f}T"


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def train(settings: dict) -> None:
    folder        = settings["input_folder"]
    lowercase     = bool(settings["lowercase"])
    single_thread = bool(settings.get("single_thread", False))
    show_prog     = bool(settings.get("show_progress", True))
    auto_dl       = bool(settings.get("auto_download", False))

    vocab_size  = int(settings["vocab_size"])
    ctx_len     = int(settings["ctx_len"])
    embed_dim   = int(settings["embed_dim"])
    hidden_dim  = int(settings["hidden_dim"])
    epochs      = int(settings["epochs"])
    batch_size  = int(settings["batch_size"])
    lr          = float(settings["learning_rate"])
    workers     = max(1, int(settings["workers"]))

    ensure_folder(folder)

    # ── Auto-download ─────────────────────────────────────────────────────
    dl_thread: Optional[AutoDownloadThread] = None
    if auto_dl:
        if single_thread:
            auto_download_blocking(settings)
        else:
            dl_thread = AutoDownloadThread(settings)
            dl_thread.start()

    # ── Load files ────────────────────────────────────────────────────────
    files = find_text_files(folder)
    if not files:
        if dl_thread:
            print("Waiting for downloader…")
            for _ in range(30):
                time.sleep(2)
                files = find_text_files(folder)
                if files:
                    break
    if not files:
        print(f"No .txt files found in '{folder}'.")
        if dl_thread:
            dl_thread.stop()
        return

    print(f"\n=== Neural LM Training ===")
    print(f"Files: {len(files)}  |  Vocab: {vocab_size}  |  "
          f"ctx={ctx_len}  embed={embed_dim}  hidden={hidden_dim}")
    print(f"Epochs: {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print()

    # ── Tokenise files (threaded or single) ──────────────────────────────
    print("Loading and tokenising files…")
    t0 = time.perf_counter()

    token_lists: List[List[str]] = [None] * len(files)  # type: ignore

    if single_thread:
        for i, path in enumerate(files):
            token_lists[i] = load_tokens_from_file(path, lowercase)
            if show_prog:
                sys.stdout.write(f"\r  {i+1}/{len(files)} files tokenised")
                sys.stdout.flush()
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_map = {
                ex.submit(load_tokens_from_file, p, lowercase): idx
                for idx, p in enumerate(files)
            }
            done = 0
            for fut in as_completed(future_map):
                idx = future_map[fut]
                token_lists[idx] = fut.result()
                done += 1
                if show_prog:
                    sys.stdout.write(f"\r  {done}/{len(files)} files tokenised")
                    sys.stdout.flush()

    if show_prog:
        sys.stdout.write("\n")

    total_tokens = sum(len(t) for t in token_lists)
    print(f"  Total tokens: {human_num(total_tokens)}  "
          f"({time.perf_counter()-t0:.1f}s)")

    # ── Build vocabulary ─────────────────────────────────────────────────
    print("Building vocabulary…")
    vocab = Vocabulary(max_size=vocab_size)
    vocab.build(token_lists)
    print(f"  Vocab size: {vocab.size}")

    # ── Build dataset ────────────────────────────────────────────────────
    print("Building training dataset…")
    X, Y = build_dataset(token_lists, vocab, ctx_len)
    del token_lists   # free RAM
    N = X.shape[0]
    print(f"  Samples: {human_num(N)}")

    # ── Init model + optimiser ───────────────────────────────────────────
    model = NeuralLM(vocab.size, embed_dim, ctx_len, hidden_dim)
    adam  = AdamState(lr=lr)

    param_count = sum(p.size for p in model.params.values())
    print(f"  Parameters: {human_num(param_count)}")
    print()

    # ── Training loop ────────────────────────────────────────────────────
    hparams = {
        "vocab_size": vocab_size, "ctx_len": ctx_len,
        "embed_dim":  embed_dim,  "hidden_dim": hidden_dim,
    }

    global_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        perm    = np.random.permutation(N)
        X_shuf  = X[perm]
        Y_shuf  = Y[perm]

        epoch_loss  = 0.0
        n_batches   = 0
        epoch_start = time.perf_counter()

        batches_total = (N + batch_size - 1) // batch_size

        for b_start in range(0, N, batch_size):
            Xb = X_shuf[b_start: b_start + batch_size]
            Yb = Y_shuf[b_start: b_start + batch_size]

            logits, cache = model.forward(Xb)
            loss, grads   = model.backward(logits, Yb, cache)

            adam.step(model.params, grads)

            epoch_loss += loss
            n_batches  += 1

            if show_prog and (n_batches % 20 == 0 or n_batches == batches_total):
                elapsed   = max(0.001, time.perf_counter() - epoch_start)
                avg_loss  = epoch_loss / n_batches
                pct       = n_batches / batches_total
                bar       = format_bar(n_batches, batches_total)
                samp_sec  = (n_batches * batch_size) / elapsed
                sys.stdout.write(
                    f"\r  Epoch {epoch}/{epochs} {bar} "
                    f"{pct*100:5.1f}%  loss={avg_loss:.4f}  "
                    f"{human_num(samp_sec)}/s"
                )
                sys.stdout.flush()

        if show_prog:
            sys.stdout.write("\n")

        epoch_elapsed = time.perf_counter() - epoch_start
        print(f"  Epoch {epoch} done — "
              f"avg loss={epoch_loss/n_batches:.4f}  "
              f"time={epoch_elapsed:.1f}s")

        # Save checkpoint after each epoch
        save_model(model, vocab, hparams, settings["model_file"])

    if dl_thread:
        dl_thread.stop()

    total_elapsed = time.perf_counter() - global_start
    print(f"\nTraining complete in {total_elapsed:.1f}s")
    print(f"Model saved to: {settings['model_file']}")


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_text(model: NeuralLM, vocab: Vocabulary,
                  prompt: str, max_tokens: int,
                  temperature: float, top_k: int,
                  lowercase: bool) -> str:
    ctx_len = model.ctx_len
    PAD_ID  = vocab.tok2id[PAD_TOKEN]

    # Encode prompt
    prompt_toks = tokenize(prompt, lowercase=lowercase)
    ctx = [vocab.encode(t) for t in prompt_toks]
    # Pad / trim to ctx_len
    if len(ctx) < ctx_len:
        ctx = [PAD_ID] * (ctx_len - len(ctx)) + ctx
    else:
        ctx = ctx[-ctx_len:]

    generated = list(prompt_toks)

    for _ in range(max_tokens):
        ctx_arr = np.array(ctx, dtype=np.int32)
        probs   = model.predict_probs(ctx_arr)          # [vocab_size]

        # Temperature scaling
        temp = max(0.05, float(temperature))
        logits = np.log(probs + 1e-12) / temp
        logits -= logits.max()
        probs   = np.exp(logits)
        probs  /= probs.sum()

        # Top-k filtering
        if top_k and top_k > 0:
            k         = min(top_k, len(probs))
            top_idx   = np.argpartition(probs, -k)[-k:]
            mask      = np.zeros_like(probs)
            mask[top_idx] = probs[top_idx]
            probs     = mask / mask.sum()

        # Sample
        next_id = int(np.random.choice(len(probs), p=probs))
        next_tok = vocab.decode(next_id)

        if next_tok in (PAD_TOKEN, UNK_TOKEN):
            # re-sample once ignoring special tokens
            probs[vocab.tok2id[PAD_TOKEN]] = 0
            probs[vocab.tok2id[UNK_TOKEN]] = 0
            s = probs.sum()
            if s > 0:
                probs /= s
                next_id  = int(np.random.choice(len(probs), p=probs))
                next_tok = vocab.decode(next_id)
            else:
                break

        generated.append(next_tok)
        ctx = ctx[1:] + [next_id]

    return detokenize(generated)


# ─────────────────────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────────────────────

def chat(settings: dict) -> None:
    result = load_model(settings["model_file"])
    if result is None:
        print("No model found. Train first.")
        return

    model, vocab, hparams = result
    print(f"Model loaded — vocab={vocab.size}  "
          f"ctx={model.ctx_len}  embed={model.embed_dim}  "
          f"hidden={model.hidden_dim}")
    print("Type 'exit' to quit.\n")

    while True:
        prompt = input("You> ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break

        out = generate_text(
            model, vocab,
            prompt      = prompt,
            max_tokens  = int(settings["max_generate_tokens"]),
            temperature = float(settings["temperature"]),
            top_k       = int(settings["top_k"]),
            lowercase   = bool(settings["lowercase"]),
        )
        print(f"\nAI> {out}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Auto-Data Downloader  (unchanged from n-gram version)
# ─────────────────────────────────────────────────────────────────────────────

def _load_id_cache(path: str) -> Set[int]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return {int(x) for x in data}
        except Exception:
            pass
    return set()


def _save_id_cache(path: str, ids: Set[int]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sorted(ids), f)
    except Exception:
        pass


def _folder_state(folder: str) -> Tuple[int, int]:
    total_bytes = 0
    count = 0
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(".txt"):
                count += 1
                try:
                    total_bytes += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    return count, total_bytes


def _already_downloaded(folder: str) -> Set[int]:
    present: Set[int] = set()
    for root, _, files in os.walk(folder):
        for name in files:
            m = re.match(r"pg(\d+)\.txt$", name, re.IGNORECASE)
            if m:
                present.add(int(m.group(1)))
    return present


def _probe_id(book_id: int) -> bool:
    url     = GUTENBERG_URL.format(id=book_id)
    headers = {"User-Agent": "neural-lm-probe/1.0"}
    try:
        req = urllib.request.Request(url, headers=headers, method="HEAD")
        with urllib.request.urlopen(req, timeout=AD_PROBE_TIMEOUT) as resp:
            if resp.status == 404:
                return False
            cl = resp.headers.get("Content-Length")
            if cl is not None:
                return int(cl) >= AD_MIN_BYTES
    except Exception:
        return False
    try:
        req = urllib.request.Request(
            url, headers={**headers, "Range": f"bytes=0-{AD_MIN_BYTES}"}
        )
        with urllib.request.urlopen(req, timeout=AD_PROBE_TIMEOUT) as resp:
            chunk = resp.read(AD_MIN_BYTES + 1)
        return len(chunk) >= AD_MIN_BYTES
    except Exception:
        return False


def _download_one_book(book_id: int, folder: str) -> Optional[str]:
    dest = os.path.join(folder, f"pg{book_id}.txt")
    if os.path.exists(dest):
        return None
    url = GUTENBERG_URL.format(id=book_id)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "neural-lm/1.0"})
        with urllib.request.urlopen(req, timeout=AD_DOWNLOAD_TIMEOUT) as resp:
            data = resp.read()
        if len(data) < AD_MIN_BYTES:
            return None
        with open(dest, "wb") as f:
            f.write(data)
        return dest
    except Exception:
        if os.path.exists(dest):
            try:
                os.remove(dest)
            except OSError:
                pass
        return None


def _discover_ids(n_needed: int, already_have: Set[int],
                  discovered: Set[int], rejected: Set[int],
                  probe_workers: int = AD_PROBE_WORKERS,
                  verbose: bool = False) -> List[int]:
    skip      = already_have | rejected | discovered
    all_ids   = list(range(GUTENBERG_ID_MIN, GUTENBERG_ID_MAX + 1))
    random.shuffle(all_ids)
    unchecked = [i for i in all_ids if i not in skip]

    good: List[int] = []
    batch_size = probe_workers * 4
    idx = 0

    while len(good) < n_needed and idx < len(unchecked):
        batch = unchecked[idx: idx + batch_size]
        idx  += batch_size
        if verbose:
            sys.stdout.write(
                f"\r  Probing…  found {len(good)}/{n_needed} usable so far   "
            )
            sys.stdout.flush()
        with ThreadPoolExecutor(max_workers=probe_workers) as ex:
            futures = {ex.submit(_probe_id, bid): bid for bid in batch}
            for fut in as_completed(futures):
                bid = futures[fut]
                ok  = False
                try:
                    ok = fut.result()
                except Exception:
                    pass
                (discovered if ok else rejected).add(bid)
                if ok:
                    good.append(bid)

    if verbose:
        sys.stdout.write("\n")
    return good


def _ready_ids(folder: str, n_needed: int,
               verbose: bool = False) -> List[int]:
    discovered = _load_id_cache(DISCOVERED_FILE)
    rejected   = _load_id_cache(REJECTED_FILE)
    have       = _already_downloaded(folder)

    queued = [i for i in discovered if i not in have]
    random.shuffle(queued)

    if len(queued) < n_needed:
        still_needed = n_needed - len(queued)
        if verbose:
            print(f"  Discovery: probing for {still_needed} more usable IDs…")
        new_good = _discover_ids(still_needed, have, discovered, rejected,
                                 verbose=verbose)
        _save_id_cache(DISCOVERED_FILE, discovered)
        _save_id_cache(REJECTED_FILE,   rejected)
        queued.extend(new_good)

    random.shuffle(queued)
    return queued[:n_needed]


def auto_download_blocking(settings: dict, label_prefix: str = "") -> None:
    folder    = settings["input_folder"]
    max_books = int(settings["ad_max_books"])
    max_bytes = int(settings["ad_max_bytes"])

    ensure_folder(folder)
    count, used = _folder_state(folder)
    if count >= max_books or used >= max_bytes:
        print(f"{label_prefix}Cap already reached ({count}/{max_books} books).")
        return

    slots = max_books - count
    print(f"{label_prefix}Auto-download: {count}/{max_books} books — fetching up to {slots} more…")

    ids = _ready_ids(folder, n_needed=slots * 2, verbose=True)
    if not ids:
        print(f"{label_prefix}No usable IDs found.")
        return

    rejected_local: Set[int] = set()
    for book_id in ids:
        count, used = _folder_state(folder)
        if count >= max_books or used >= max_bytes:
            break
        sys.stdout.write(f"\r  Downloading pg{book_id}.txt …" + " " * 10)
        sys.stdout.flush()
        path = _download_one_book(book_id, folder)
        if path:
            size = os.path.getsize(path)
            sys.stdout.write(f"\r  ✓ pg{book_id}.txt  ({size//1024} KB)\n")
        else:
            sys.stdout.write(f"\r  ✗ pg{book_id}.txt  (failed)\n")
            rejected_local.add(book_id)
        sys.stdout.flush()

    if rejected_local:
        rej  = _load_id_cache(REJECTED_FILE)
        disc = _load_id_cache(DISCOVERED_FILE)
        rej  |= rejected_local
        disc -= rejected_local
        _save_id_cache(REJECTED_FILE,   rej)
        _save_id_cache(DISCOVERED_FILE, disc)

    count, used = _folder_state(folder)
    print(f"{label_prefix}Done: {count} books, {used//1048576}MB.")


class AutoDownloadThread(threading.Thread):
    POLL_INTERVAL = 5

    def __init__(self, settings: dict) -> None:
        super().__init__(daemon=True, name="AutoDownloader")
        self._s        = settings
        self._stop_evt = threading.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        folder       = self._s["input_folder"]
        max_books    = int(self._s["ad_max_books"])
        max_bytes    = int(self._s["ad_max_bytes"])
        refill_below = int(self._s["ad_refill_below"])

        ensure_folder(folder)
        queue: List[int] = _ready_ids(folder, n_needed=max_books * 2, verbose=False)
        rejected_local: Set[int] = set()

        while not self._stop_evt.is_set():
            count, used = _folder_state(folder)
            if (count < refill_below) or (used < max_bytes and count < max_books):
                if not queue:
                    queue = _ready_ids(folder, n_needed=max_books * 2, verbose=False)
                while queue and not self._stop_evt.is_set():
                    count, used = _folder_state(folder)
                    if count >= max_books or used >= max_bytes:
                        break
                    bid  = queue.pop(0)
                    path = _download_one_book(bid, folder)
                    if path is None:
                        rejected_local.add(bid)
            self._stop_evt.wait(timeout=self.POLL_INTERVAL)

        if rejected_local:
            rej  = _load_id_cache(REJECTED_FILE)
            disc = _load_id_cache(DISCOVERED_FILE)
            rej  |= rejected_local
            disc -= rejected_local
            _save_id_cache(REJECTED_FILE,   rej)
            _save_id_cache(DISCOVERED_FILE, disc)


# ─────────────────────────────────────────────────────────────────────────────
# File helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_text_files(folder: str) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(".txt"):
                paths.append(os.path.join(root, name))
    paths.sort()
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Settings menu
# ─────────────────────────────────────────────────────────────────────────────

def show_settings(s: dict) -> None:
    count, used = _folder_state(s["input_folder"])
    disc = _load_id_cache(DISCOVERED_FILE)
    rej  = _load_id_cache(REJECTED_FILE)
    print("\nCurrent settings:")
    groups = [
        ("Files",       ["input_folder", "model_file", "lowercase"]),
        ("Vocabulary",  ["vocab_size"]),
        ("Architecture",["ctx_len", "embed_dim", "hidden_dim"]),
        ("Training",    ["epochs", "batch_size", "learning_rate",
                         "workers", "single_thread", "show_progress"]),
        ("Generation",  ["max_generate_tokens", "temperature", "top_k"]),
    ]
    for label, keys in groups:
        print(f"  ── {label}")
        for k in keys:
            print(f"    {k}: {s[k]}")
    ad_on = s.get("auto_download", False)
    print(f"  ── Auto-Data Downloader")
    print(f"    auto_download:   {'ON' if ad_on else 'OFF'}")
    print(f"    ad_max_books:    {s['ad_max_books']}")
    print(f"    ad_max_bytes:    {s['ad_max_bytes']//1048576} MB")
    print(f"    ad_refill_below: {s['ad_refill_below']}")
    print(f"    folder now:      {count} books, {used//1048576} MB")
    print(f"    discovered IDs:  {len(disc)}   rejected: {len(rej)}")


def _auto_dl_submenu(s: dict) -> None:
    while True:
        count, used = _folder_state(s["input_folder"])
        disc = _load_id_cache(DISCOVERED_FILE)
        rej  = _load_id_cache(REJECTED_FILE)
        ad_on = s.get("auto_download", False)
        print(f"\n── Auto-Data Downloader ─────────────────────────────────")
        print(f"  Status: {'ON' if ad_on else 'OFF'}  |  "
              f"{count}/{s['ad_max_books']} books  "
              f"{used//1048576}/{s['ad_max_bytes']//1048576} MB")
        print(f"  Discovered: {len(disc)}   Rejected: {len(rej)}   "
              f"Probe range: {GUTENBERG_ID_MIN}–{GUTENBERG_ID_MAX}   "
              f"Min size: {AD_MIN_BYTES//1024} KB")
        print()
        print("1) Toggle ON/OFF")
        print("2) Max books")
        print("3) Max total MB")
        print("4) Refill-below threshold")
        print("5) Download now (blocking)")
        print("6) Probe for new IDs now")
        print("7) Clear caches")
        print("0) Back")

        c = input("> ").strip()

        if c == "1":
            s["auto_download"] = not bool(s.get("auto_download", False))
            print("Auto-DL:", "ON" if s["auto_download"] else "OFF")
        elif c == "2":
            v = input("Max books (1-500): ").strip()
            try:
                n = int(v)
                if 1 <= n <= 500:
                    s["ad_max_books"] = n
            except ValueError:
                pass
        elif c == "3":
            v = input("Max MB (1-2000): ").strip()
            try:
                n = int(v)
                if 1 <= n <= 2000:
                    s["ad_max_bytes"] = n * 1048576
            except ValueError:
                pass
        elif c == "4":
            v = input("Refill below N books: ").strip()
            try:
                n = int(v)
                if n >= 0:
                    s["ad_refill_below"] = n
            except ValueError:
                pass
        elif c == "5":
            save_settings(s)
            auto_download_blocking(s)
        elif c == "6":
            folder = s["input_folder"]
            have   = _already_downloaded(folder)
            disc2  = _load_id_cache(DISCOVERED_FILE)
            rej2   = _load_id_cache(REJECTED_FILE)
            print("Probing… (Ctrl-C to stop)")
            try:
                _discover_ids(20, have, disc2, rej2, verbose=True)
                _save_id_cache(DISCOVERED_FILE, disc2)
                _save_id_cache(REJECTED_FILE,   rej2)
                print(f"Done. {len(disc2)} total discovered IDs.")
            except KeyboardInterrupt:
                _save_id_cache(DISCOVERED_FILE, disc2)
                _save_id_cache(REJECTED_FILE,   rej2)
                print("\nInterrupted — partial results saved.")
        elif c == "7":
            if input("Clear both caches? (y/N): ").strip().lower() == "y":
                _save_id_cache(DISCOVERED_FILE, set())
                _save_id_cache(REJECTED_FILE,   set())
                print("Cleared.")
        elif c == "0":
            save_settings(s)
            return
        else:
            print("Invalid.")

        save_settings(s)


def settings_menu(s: dict) -> None:
    while True:
        show_settings(s)
        print("\nSettings menu:")
        print(" Files / vocab")
        print("  1) Input folder")
        print("  2) Model file")
        print("  3) Toggle lowercase")
        print("  4) Vocab size")
        print(" Architecture")
        print("  5) Context length (ctx_len)")
        print("  6) Embedding dim")
        print("  7) Hidden dim")
        print(" Training")
        print("  8) Epochs")
        print("  9) Batch size")
        print("  l) Learning rate")
        print("  w) Worker count")
        print("  t) Toggle single-thread")
        print("  p) Toggle progress display")
        print(" Generation")
        print("  g) Max generated tokens")
        print("  e) Temperature")
        print("  k) Top-k")
        print(" Other")
        print("  a) Auto-Data Downloader")
        print("  0) Back")

        c = input("> ").strip().lower()

        if c == "1":
            v = input("Input folder: ").strip()
            if v:
                s["input_folder"] = v
                ensure_folder(v)
        elif c == "2":
            v = input("Model file: ").strip()
            if v:
                s["model_file"] = v
        elif c == "3":
            s["lowercase"] = not bool(s["lowercase"])
            print("Lowercase:", s["lowercase"])
        elif c == "4":
            v = input("Vocab size (1000-100000): ").strip()
            try:
                n = int(v)
                if 1000 <= n <= 100000:
                    s["vocab_size"] = n
            except ValueError:
                pass
        elif c == "5":
            v = input("ctx_len (2-16): ").strip()
            try:
                n = int(v)
                if 2 <= n <= 16:
                    s["ctx_len"] = n
            except ValueError:
                pass
        elif c == "6":
            v = input("embed_dim (16-512): ").strip()
            try:
                n = int(v)
                if 16 <= n <= 512:
                    s["embed_dim"] = n
            except ValueError:
                pass
        elif c == "7":
            v = input("hidden_dim (64-2048): ").strip()
            try:
                n = int(v)
                if 64 <= n <= 2048:
                    s["hidden_dim"] = n
            except ValueError:
                pass
        elif c == "8":
            v = input("Epochs (1-100): ").strip()
            try:
                n = int(v)
                if 1 <= n <= 100:
                    s["epochs"] = n
            except ValueError:
                pass
        elif c == "9":
            v = input("Batch size (32-4096): ").strip()
            try:
                n = int(v)
                if 32 <= n <= 4096:
                    s["batch_size"] = n
            except ValueError:
                pass
        elif c == "l":
            v = input("Learning rate (e.g. 0.001): ").strip()
            try:
                x = float(v)
                if 0 < x < 1:
                    s["learning_rate"] = x
            except ValueError:
                pass
        elif c == "w":
            v = input("Workers: ").strip()
            try:
                n = int(v)
                if n >= 1:
                    s["workers"] = n
            except ValueError:
                pass
        elif c == "t":
            s["single_thread"] = not bool(s.get("single_thread", False))
            print("Single-thread:", s["single_thread"])
        elif c == "p":
            s["show_progress"] = not bool(s["show_progress"])
            print("Show progress:", s["show_progress"])
        elif c == "g":
            v = input("Max generated tokens: ").strip()
            try:
                n = int(v)
                if n >= 1:
                    s["max_generate_tokens"] = n
            except ValueError:
                pass
        elif c == "e":
            v = input("Temperature (0.1-3.0): ").strip()
            try:
                x = float(v)
                if 0.1 <= x <= 3.0:
                    s["temperature"] = x
            except ValueError:
                pass
        elif c == "k":
            v = input("Top-k (0=off): ").strip()
            try:
                n = int(v)
                if n >= 0:
                    s["top_k"] = n
            except ValueError:
                pass
        elif c == "a":
            _auto_dl_submenu(s)
        elif c == "0":
            save_settings(s)
            print("Settings saved.")
            return
        else:
            print("Invalid choice.")

        save_settings(s)
        print("Saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    s = load_settings(DEFAULT_SETTINGS["settings_file"])
    ensure_folder(s["input_folder"])

    parser = argparse.ArgumentParser(description="Pure Python + numpy Neural LM")
    parser.add_argument(
        "mode", nargs="?",
        choices=["train", "chat", "settings", "download"],
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(s); return
    if args.mode == "chat":
        chat(s);  return
    if args.mode == "settings":
        settings_menu(s); return
    if args.mode == "download":
        auto_download_blocking(s); return

    while True:
        count, used = _folder_state(s["input_folder"])
        ad_ind = (f" [Auto-DL ON | {count} books, {used//1048576}MB]"
                  if s.get("auto_download") else "")
        model_exists = os.path.exists(s["model_file"])
        print(f"\n=== Neural LM ==={ad_ind}")
        print(f"  model: {s['model_file']}"
              + (" ✓" if model_exists else " (not trained yet)"))
        print("1) Train")
        print("2) Chat / Generate")
        print("3) Settings")
        print("4) Download books now")
        print("5) Exit")

        c = input("> ").strip()
        if c == "1":
            train(s)
        elif c == "2":
            chat(s)
        elif c == "3":
            settings_menu(s)
        elif c == "4":
            auto_download_blocking(s)
        elif c == "5":
            save_settings(s)
            break
        else:
            print("Invalid.")


if __name__ == "__main__":
    main()
