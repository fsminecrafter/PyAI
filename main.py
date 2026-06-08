#!/usr/bin/env python3
"""
Pure-Python + numpy Neural Language Model  (GPU-capable via CuPy)
==================================================================
Architecture
  - Embedding matrix  E : [vocab_size × embed_dim]
  - Context window      : last ctx_len tokens → concat embeddings → flat vec
  - Hidden layer        : Linear → LayerNorm → tanh
  - Output layer        : Linear → softmax over vocab
  - Loss                : cross-entropy
  - Optimiser           : Adam (numpy / CuPy)

GPU support
  - Requires CuPy (pip install cupy-cuda11x / cupy-cuda12x).
  - Set  use_gpu=True  and  gpu_device=<id>  in settings.
  - At startup the code queries available CUDA devices and lists them.
  - Random shuffling always uses numpy (avoids curand dependency).
  - If CuPy is unavailable the setting is silently ignored.

Memory-safe training
  - Files are processed in chunks; only one chunk lives in RAM at a time.
  - Chunk size is auto-tuned to stay within available system RAM.

Book downloader
  - Multi-threaded concurrent downloads with per-book progress bars and ETA.
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
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

try:
    import psutil as _psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


# ─────────────────────────────────────────────────────────────────────────────
# GPU / Array-backend  (xp = numpy or cupy)
# ─────────────────────────────────────────────────────────────────────────────

_cupy_available  = False
_curand_available = False
_cupy_devices: List[Tuple[int, str]] = []   # [(device_id, name), …]

try:
    import cupy as cp          # type: ignore

    # Verify basic cupy tensor ops work (no curand needed)
    _test_arr = cp.array([1.0, 2.0], dtype=cp.float32)
    _ = _test_arr + 1
    del _test_arr
    _cupy_available = True

    # Probe curand separately — permutation needs it
    try:
        _ = cp.random.permutation(4)
        _curand_available = True
    except Exception:
        _curand_available = False

    # Enumerate CUDA devices
    n_dev = cp.cuda.runtime.getDeviceCount()
    for _di in range(n_dev):
        cp.cuda.Device(_di).use()
        _props = cp.cuda.runtime.getDeviceProperties(_di)
        _name  = _props["name"].decode() if isinstance(_props["name"], bytes) else str(_props["name"])
        _cupy_devices.append((_di, _name))

except Exception:
    _cupy_available   = False
    _curand_available = False

# Active backend — replaced by _set_backend() at runtime
_xp        = np
_device_id: int = -1


def _set_backend(use_gpu: bool, gpu_device: int = 0) -> None:
    """Switch the global array backend to CuPy (GPU) or NumPy (CPU)."""
    global _xp, _device_id
    if use_gpu and _cupy_available:
        _device_id = gpu_device
        cp.cuda.Device(gpu_device).use()
        _xp = cp
    else:
        _xp = np
        _device_id = -1


def _to_numpy(arr) -> np.ndarray:
    """Move an array to CPU numpy (no-op if already numpy)."""
    if _xp is np:
        return arr
    return cp.asnumpy(arr)


def _to_xp(arr: np.ndarray):
    """Move a numpy array to the active backend device."""
    if _xp is np:
        return arr
    return cp.asarray(arr)


def gpu_info_string() -> str:
    if not _cupy_available:
        return "CuPy not installed — GPU unavailable."
    lines = []
    if not _curand_available:
        lines.append(
            "  WARNING: libcurand.so not found — random shuffling will use numpy "
            "(all other GPU ops are fine)."
        )
    if not _cupy_devices:
        lines.append("CuPy installed but no CUDA devices found.")
        return "\n".join(lines)
    lines.insert(0, "Available CUDA devices:")
    for did, name in _cupy_devices:
        lines.append(f"  [{did}] {name}")
    return "\n".join(lines)


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

# Fraction of available RAM to use for dataset chunk
CHUNK_RAM_FRACTION = 0.40

DEFAULT_SETTINGS = {
    "input_folder":        "input_files",
    "model_file":          "model.npz",
    "settings_file":       "settings.json",
    # ── Vocabulary ───────────────────────────────────────────────────────
    "vocab_size":          20000,
    "lowercase":           True,
    # ── Architecture ─────────────────────────────────────────────────────
    "ctx_len":             6,
    "embed_dim":           64,
    "hidden_dim":          256,
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
    # ── GPU ──────────────────────────────────────────────────────────────
    "use_gpu":             False,
    "gpu_device":          0,
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
        special     = [PAD_TOKEN, UNK_TOKEN]
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
# Adam optimiser state  (backend-agnostic — uses _xp)
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
        self.m: Dict[str, object] = {}
        self.v: Dict[str, object] = {}

    def step(self, params: Dict[str, object],
             grads:  Dict[str, object]) -> None:
        self.t += 1
        xp = _xp
        for name, g in grads.items():
            if name not in self.m:
                self.m[name] = xp.zeros_like(params[name])
                self.v[name] = xp.zeros_like(params[name])
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g * g
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            params[name] -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)

    def state_dict(self) -> dict:
        return {
            "lr": self.lr, "beta1": self.beta1, "beta2": self.beta2,
            "eps": self.eps, "t": self.t,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Neural Language Model  (backend-agnostic — all ops via _xp)
# ─────────────────────────────────────────────────────────────────────────────

class NeuralLM:
    """
    Embedding lookup → concat → hidden (LayerNorm + tanh) → output (softmax).
    All array operations use the global _xp module (numpy CPU or cupy GPU).
    Random shuffling always uses numpy to avoid curand dependency.
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
            "E":    np.random.randn(vocab_size, embed_dim).astype(np.float32) * scale_e,
            "W1":   np.random.randn(self.input_dim, hidden_dim).astype(np.float32) * scale_h,
            "b1":   np.zeros(hidden_dim, dtype=np.float32),
            "ln_g": np.ones(hidden_dim,  dtype=np.float32),
            "ln_b": np.zeros(hidden_dim, dtype=np.float32),
            "W2":   np.random.randn(hidden_dim, vocab_size).astype(np.float32) * scale_o,
            "b2":   np.zeros(vocab_size, dtype=np.float32),
        }

    # ── device management ────────────────────────────────────────────────

    def to_device(self) -> None:
        self.params = {k: _to_xp(v) for k, v in self.params.items()}

    def to_cpu(self) -> None:
        self.params = {k: _to_numpy(v) for k, v in self.params.items()}

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, ctx_ids) -> Tuple[object, dict]:
        xp = _xp
        B  = ctx_ids.shape[0]
        p  = self.params

        emb  = p["E"][ctx_ids]
        x    = emb.reshape(B, self.input_dim)
        pre  = x @ p["W1"] + p["b1"]

        mu   = pre.mean(axis=1, keepdims=True)
        var  = pre.var( axis=1, keepdims=True) + 1e-5
        xhat = (pre - mu) / xp.sqrt(var)
        ln   = p["ln_g"] * xhat + p["ln_b"]

        h      = xp.tanh(ln)
        logits = h @ p["W2"] + p["b2"]

        cache = dict(ctx_ids=ctx_ids, emb=emb, x=x,
                     pre=pre, mu=mu, var=var, xhat=xhat,
                     ln=ln, h=h)
        return logits, cache

    # ── backward ─────────────────────────────────────────────────────────

    def backward(self, logits, targets, cache) -> Tuple[float, Dict[str, object]]:
        xp = _xp
        B  = logits.shape[0]
        p  = self.params

        logits_s = logits - logits.max(axis=1, keepdims=True)
        exp_l    = xp.exp(logits_s)
        probs    = exp_l / exp_l.sum(axis=1, keepdims=True)
        loss     = -xp.log(probs[xp.arange(B), targets] + 1e-12).mean()

        dlogits  = probs.copy()
        dlogits[xp.arange(B), targets] -= 1
        dlogits /= B

        dW2 = cache["h"].T @ dlogits
        db2 = dlogits.sum(axis=0)
        dh  = dlogits @ p["W2"].T

        dln    = dh * (1 - cache["h"] ** 2)
        dln_g  = (dln * cache["xhat"]).sum(axis=0)
        dln_b  = dln.sum(axis=0)
        dxhat  = dln * p["ln_g"]
        std    = xp.sqrt(cache["var"])
        dpre   = (1 / (B * std)) * (
            B * dxhat
            - dxhat.sum(axis=1, keepdims=True)
            - cache["xhat"] * (dxhat * cache["xhat"]).sum(axis=1, keepdims=True)
        )

        dW1 = cache["x"].T @ dpre
        db1 = dpre.sum(axis=0)
        dx  = dpre @ p["W1"].T

        dx_emb = dx.reshape(B, self.ctx_len, self.embed_dim)
        dE     = xp.zeros_like(p["E"])
        xp.add.at(dE, cache["ctx_ids"], dx_emb)

        grads = {
            "E": dE, "W1": dW1, "b1": db1,
            "ln_g": dln_g, "ln_b": dln_b,
            "W2": dW2, "b2": db2,
        }
        return float(_to_numpy(loss)), grads

    # ── predict ──────────────────────────────────────────────────────────

    def predict_probs(self, ctx_ids: np.ndarray) -> np.ndarray:
        ctx_dev = _to_xp(ctx_ids[np.newaxis])
        logits, _ = self.forward(ctx_dev)
        logits_cpu = _to_numpy(logits[0])
        logits_s   = logits_cpu - logits_cpu.max()
        exp_l      = np.exp(logits_s)
        return exp_l / exp_l.sum()

    # ── serialisation ────────────────────────────────────────────────────

    def to_npz(self) -> io.BytesIO:
        buf = io.BytesIO()
        cpu_params = {k: _to_numpy(v) for k, v in self.params.items()}
        np.savez_compressed(buf, **cpu_params)
        buf.seek(0)
        return buf

    @classmethod
    def from_npz(cls, buf: io.BytesIO,
                 vocab_size: int, embed_dim: int,
                 ctx_len: int, hidden_dim: int) -> "NeuralLM":
        m    = cls(vocab_size, embed_dim, ctx_len, hidden_dim)
        data = np.load(buf)
        for k in m.params:
            m.params[k] = data[k]
        return m


# ─────────────────────────────────────────────────────────────────────────────
# Model save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model: NeuralLM, vocab: Vocabulary,
               hparams: dict, model_file: str) -> None:
    header       = {"vocab": vocab.to_dict(), "hparams": hparams}
    header_bytes = json.dumps(header).encode("utf-8")
    weights_buf  = model.to_npz()

    with gzip.open(model_file, "wb") as f:
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

    vocab  = Vocabulary.from_dict(header["vocab"])
    hp     = header["hparams"]
    model  = NeuralLM.from_npz(
        weights_buf,
        vocab_size  = vocab.size,
        embed_dim   = hp["embed_dim"],
        ctx_len     = hp["ctx_len"],
        hidden_dim  = hp["hidden_dim"],
    )
    return model, vocab, hp


# ─────────────────────────────────────────────────────────────────────────────
# Training data builder  (streaming / chunked)
# ─────────────────────────────────────────────────────────────────────────────

def load_tokens_from_file(path: str, lowercase: bool) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return tokenize(text, lowercase=lowercase)


def _estimate_chunk_size(ctx_len: int, sample_tokens: int = 500_000) -> int:
    """
    Estimate how many tokens can fit in one chunk while staying within
    CHUNK_RAM_FRACTION of available RAM.  Each (X,Y) sample = ctx_len+1
    int32 values = (ctx_len+1)*4 bytes.
    """
    if _PSUTIL:
        avail = _psutil.virtual_memory().available
    else:
        avail = 2 * 1024 ** 3   # fallback: assume 2 GB

    bytes_per_sample = (ctx_len + 1) * 4
    budget_bytes     = int(avail * CHUNK_RAM_FRACTION)
    max_samples      = budget_bytes // bytes_per_sample
    # tokens ≈ samples (one sample per token after ctx padding)
    return max(50_000, min(max_samples, 10_000_000))


def build_dataset_from_tokens(tokens: List[str],
                               vocab: Vocabulary,
                               ctx_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build X [N, ctx_len] and Y [N] from a flat token list (CPU numpy)."""
    PAD_ID = vocab.tok2id[PAD_TOKEN]
    ids    = [vocab.encode(t) for t in tokens]
    padded = [PAD_ID] * ctx_len + ids
    n      = len(ids)
    X      = np.empty((n, ctx_len), dtype=np.int32)
    Y      = np.empty(n,            dtype=np.int32)
    for i in range(n):
        X[i] = padded[i: i + ctx_len]
        Y[i] = padded[i + ctx_len]
    return X, Y


def _iter_file_chunks(files: List[str], lowercase: bool,
                      chunk_tokens: int) -> Iterator[List[str]]:
    """
    Yield flat token lists of ≤ chunk_tokens tokens, reading files one by
    one.  Never holds more than ~2 files' worth of tokens in memory.
    """
    buf: List[str] = []
    for path in files:
        toks = load_tokens_from_file(path, lowercase)
        buf.extend(toks)
        while len(buf) >= chunk_tokens:
            yield buf[:chunk_tokens]
            buf = buf[chunk_tokens:]
    if buf:
        yield buf


# ─────────────────────────────────────────────────────────────────────────────
# Progress / display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _term_width() -> int:
    try:
        return os.get_terminal_size().columns
    except Exception:
        return 80


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


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _sys_usage_str() -> str:
    """One-line CPU / RAM usage string."""
    if not _PSUTIL:
        return ""
    cpu  = _psutil.cpu_percent(interval=None)
    mem  = _psutil.virtual_memory()
    used = mem.used  // (1024 ** 2)
    tot  = mem.total // (1024 ** 2)
    s    = f"CPU {cpu:4.1f}%  RAM {used}/{tot} MB"

    # GPU via CuPy mempool (if active)
    if _xp is not np:
        try:
            pool  = cp.get_default_memory_pool()
            g_used = pool.used_bytes()  // (1024 ** 2)
            g_tot  = pool.total_bytes() // (1024 ** 2)
            s     += f"  GPU-mem {g_used}/{g_tot} MB"
        except Exception:
            pass
    return s


def _print_file_loading_progress(done: int, total: int,
                                  token_count: int) -> None:
    bar = format_bar(done, total, width=28)
    sys.stdout.write(
        f"\r  Loading {bar} {done}/{total} files  "
        f"({human_num(token_count)} tokens)"
        + " " * 4
    )
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Two-pass vocab builder (streaming, low memory)
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab_streaming(files: List[str], lowercase: bool,
                           vocab_size: int,
                           workers: int, single_thread: bool,
                           show_prog: bool) -> Tuple[Vocabulary, int]:
    """
    Stream through every file once, count tokens, build vocab.
    Returns (vocab, total_token_count).
    """
    counts: Counter = Counter()
    total_tokens = 0
    n_files = len(files)

    print("  Pass 1/2 — counting tokens for vocabulary…")

    if single_thread:
        for i, path in enumerate(files):
            toks = load_tokens_from_file(path, lowercase)
            counts.update(toks)
            total_tokens += len(toks)
            if show_prog:
                _print_file_loading_progress(i + 1, n_files, total_tokens)
    else:
        token_lists: List[Optional[List[str]]] = [None] * n_files
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_map = {
                ex.submit(load_tokens_from_file, p, lowercase): idx
                for idx, p in enumerate(files)
            }
            done = 0
            for fut in as_completed(future_map):
                idx = future_map[fut]
                toks = fut.result()
                token_lists[idx] = toks
                counts.update(toks)
                total_tokens += len(toks)
                done += 1
                if show_prog:
                    _print_file_loading_progress(done, n_files, total_tokens)

    if show_prog:
        sys.stdout.write("\n")

    special     = [PAD_TOKEN, UNK_TOKEN]
    most_common = [t for t, _ in counts.most_common(vocab_size - len(special))]
    vocab       = Vocabulary(max_size=vocab_size)
    vocab.id2tok = special + most_common
    vocab.tok2id = {t: i for i, t in enumerate(vocab.id2tok)}
    return vocab, total_tokens


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

    use_gpu    = bool(settings.get("use_gpu", False))
    gpu_device = int(settings.get("gpu_device", 0))

    # ── Activate GPU backend ──────────────────────────────────────────────
    _set_backend(use_gpu, gpu_device)
    if _xp is not np:
        dev_name = _cupy_devices[gpu_device][1] if gpu_device < len(_cupy_devices) else "?"
        print(f"[GPU] Using CUDA device {gpu_device}: {dev_name}")
        if not _curand_available:
            print("[GPU] Note: libcurand not found — shuffling via numpy (no impact on training).")
    else:
        if use_gpu and not _cupy_available:
            print("[GPU] CuPy not available — falling back to CPU.")
        elif use_gpu:
            print("[GPU] No valid CUDA device found — falling back to CPU.")
        else:
            print("[CPU] Running on CPU.")

    ensure_folder(folder)

    # ── Auto-download ─────────────────────────────────────────────────────
    dl_thread: Optional[AutoDownloadThread] = None
    if auto_dl:
        if single_thread:
            auto_download_blocking(settings)
        else:
            dl_thread = AutoDownloadThread(settings)
            dl_thread.start()

    # ── Find files ────────────────────────────────────────────────────────
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

    backend_label = f"GPU:{gpu_device}" if _xp is not np else "CPU"
    print(f"\n=== Neural LM Training  [{backend_label}] ===")
    print(f"Files: {len(files)}  |  Vocab: {vocab_size}  |  "
          f"ctx={ctx_len}  embed={embed_dim}  hidden={hidden_dim}")
    print(f"Epochs: {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    if _PSUTIL:
        mem = _psutil.virtual_memory()
        print(f"System RAM: {mem.available // (1024**2)} MB available of "
              f"{mem.total // (1024**2)} MB total")
    print()

    # ── Pass 1: Build vocabulary (streaming) ─────────────────────────────
    vocab, total_tokens = build_vocab_streaming(
        files, lowercase, vocab_size, workers, single_thread, show_prog
    )
    print(f"  Vocab size: {vocab.size}  |  Total tokens: {human_num(total_tokens)}")

    # ── Determine chunk size ──────────────────────────────────────────────
    chunk_tokens = _estimate_chunk_size(ctx_len)
    n_chunks_approx = max(1, total_tokens // chunk_tokens)
    print(f"  Chunk size: ~{human_num(chunk_tokens)} tokens  "
          f"(≈{n_chunks_approx} chunk(s) per epoch)")

    # ── Init model + optimiser ────────────────────────────────────────────
    print("\n  Pass 2/2 — initialising model…")
    model = NeuralLM(vocab.size, embed_dim, ctx_len, hidden_dim)
    model.to_device()
    adam  = AdamState(lr=lr)

    param_count = sum(
        int(np.prod((_to_numpy(v)).shape)) for v in model.params.values()
    )
    print(f"  Parameters: {human_num(param_count)}")

    hparams = {
        "vocab_size": vocab_size, "ctx_len": ctx_len,
        "embed_dim":  embed_dim,  "hidden_dim": hidden_dim,
    }

    global_start = time.perf_counter()

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        epoch_start  = time.perf_counter()
        epoch_loss   = 0.0
        epoch_batches = 0
        epoch_samples = 0

        print(f"\n── Epoch {epoch}/{epochs} ──────────────────────────────────────")

        chunk_idx   = 0
        chunk_start = time.perf_counter()

        for chunk_toks in _iter_file_chunks(files, lowercase, chunk_tokens):
            chunk_idx += 1

            # Build dataset for this chunk (on CPU)
            X_cpu, Y_cpu = build_dataset_from_tokens(chunk_toks, vocab, ctx_len)
            N = X_cpu.shape[0]
            del chunk_toks   # free token list immediately

            # Shuffle indices on CPU (avoids curand dependency)
            perm     = np.random.permutation(N)
            X_cpu    = X_cpu[perm]
            Y_cpu    = Y_cpu[perm]

            # Upload to device if GPU
            if _xp is not np:
                X_dev = _to_xp(X_cpu)
                Y_dev = _to_xp(Y_cpu)
                del X_cpu, Y_cpu
            else:
                X_dev = X_cpu
                Y_dev = Y_cpu

            batches_in_chunk = (N + batch_size - 1) // batch_size
            chunk_loss   = 0.0
            chunk_batches = 0

            for b_start in range(0, N, batch_size):
                Xb = X_dev[b_start: b_start + batch_size]
                Yb = Y_dev[b_start: b_start + batch_size]

                logits, cache = model.forward(Xb)
                loss, grads   = model.backward(logits, Yb, cache)
                adam.step(model.params, grads)

                chunk_loss    += loss
                chunk_batches += 1
                epoch_loss    += loss
                epoch_batches += 1
                epoch_samples += Xb.shape[0]

                if show_prog and (chunk_batches % 20 == 0
                                  or chunk_batches == batches_in_chunk):
                    elapsed   = max(0.001, time.perf_counter() - chunk_start)
                    avg_loss  = chunk_loss / chunk_batches
                    pct       = chunk_batches / batches_in_chunk
                    bar       = format_bar(chunk_batches, batches_in_chunk, 24)
                    samp_sec  = (chunk_batches * batch_size) / elapsed
                    usage     = _sys_usage_str()
                    sys.stdout.write(
                        f"\r  Chunk {chunk_idx} {bar} {pct*100:5.1f}%  "
                        f"loss={avg_loss:.4f}  {human_num(samp_sec)}/s"
                        + (f"  |  {usage}" if usage else "")
                        + " " * 4
                    )
                    sys.stdout.flush()

            if show_prog:
                sys.stdout.write("\n")

            del X_dev, Y_dev

        epoch_elapsed = time.perf_counter() - epoch_start
        avg_epoch_loss = epoch_loss / max(1, epoch_batches)
        print(
            f"  Epoch {epoch} done — "
            f"{chunk_idx} chunk(s)  "
            f"avg loss={avg_epoch_loss:.4f}  "
            f"samples={human_num(epoch_samples)}  "
            f"time={epoch_elapsed:.1f}s"
        )

        # Save checkpoint after each epoch
        save_model(model, vocab, hparams, settings["model_file"])
        print(f"  Checkpoint saved → {settings['model_file']}")

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

    prompt_toks = tokenize(prompt, lowercase=lowercase)
    ctx = [vocab.encode(t) for t in prompt_toks]
    if len(ctx) < ctx_len:
        ctx = [PAD_ID] * (ctx_len - len(ctx)) + ctx
    else:
        ctx = ctx[-ctx_len:]

    generated = list(prompt_toks)

    for _ in range(max_tokens):
        ctx_arr = np.array(ctx, dtype=np.int32)
        probs   = model.predict_probs(ctx_arr)

        temp   = max(0.05, float(temperature))
        logits = np.log(probs + 1e-12) / temp
        logits -= logits.max()
        probs   = np.exp(logits)
        probs  /= probs.sum()

        if top_k and top_k > 0:
            k       = min(top_k, len(probs))
            top_idx = np.argpartition(probs, -k)[-k:]
            mask    = np.zeros_like(probs)
            mask[top_idx] = probs[top_idx]
            probs   = mask / mask.sum()

        next_id  = int(np.random.choice(len(probs), p=probs))
        next_tok = vocab.decode(next_id)

        if next_tok in (PAD_TOKEN, UNK_TOKEN):
            probs[vocab.tok2id[PAD_TOKEN]] = 0
            probs[vocab.tok2id[UNK_TOKEN]] = 0
            s = probs.sum()
            if s > 0:
                probs   /= s
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
    use_gpu    = bool(settings.get("use_gpu", False))
    gpu_device = int(settings.get("gpu_device", 0))
    _set_backend(use_gpu, gpu_device)

    result = load_model(settings["model_file"])
    if result is None:
        print("No model found. Train first.")
        return

    model, vocab, hparams = result
    model.to_device()

    backend_label = f"GPU:{gpu_device}" if _xp is not np else "CPU"
    print(f"Model loaded [{backend_label}] — vocab={vocab.size}  "
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
# Auto-Data Downloader  (multi-threaded + progress bars)
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
    """Download a single book, returning destination path or None on failure."""
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


def _download_one_book_progress(book_id: int, folder: str,
                                 status_dict: dict,
                                 lock: threading.Lock) -> Optional[str]:
    """
    Download a single book with live byte-count reporting into status_dict.
    status_dict[book_id] = {'done': bytes, 'total': bytes, 'state': str}
    """
    dest = os.path.join(folder, f"pg{book_id}.txt")
    if os.path.exists(dest):
        with lock:
            status_dict[book_id] = {'done': 0, 'total': 0, 'state': 'exists'}
        return None

    url = GUTENBERG_URL.format(id=book_id)
    with lock:
        status_dict[book_id] = {'done': 0, 'total': 0, 'state': 'connecting'}

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "neural-lm/1.0"})
        with urllib.request.urlopen(req, timeout=AD_DOWNLOAD_TIMEOUT) as resp:
            cl = resp.headers.get("Content-Length")
            total = int(cl) if cl else 0
            with lock:
                status_dict[book_id]['total'] = total
                status_dict[book_id]['state'] = 'downloading'

            chunks = []
            done   = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                done += len(chunk)
                with lock:
                    status_dict[book_id]['done'] = done

        data = b"".join(chunks)
        if len(data) < AD_MIN_BYTES:
            with lock:
                status_dict[book_id]['state'] = 'too_small'
            return None

        with open(dest, "wb") as f:
            f.write(data)
        with lock:
            status_dict[book_id] = {'done': len(data), 'total': len(data), 'state': 'done'}
        return dest

    except Exception as e:
        if os.path.exists(dest):
            try:
                os.remove(dest)
            except OSError:
                pass
        with lock:
            status_dict[book_id] = {'done': 0, 'total': 0, 'state': 'failed'}
        return None


def _render_download_status(status_dict: dict, lock: threading.Lock,
                              n_done: int, n_total: int,
                              started_at: float) -> None:
    """Print live multi-book download status to stdout."""
    with lock:
        snap = dict(status_dict)

    lines = []
    bar_w = 20
    for bid, info in sorted(snap.items()):
        state = info['state']
        done  = info['done']
        total = info['total']

        if state == 'done':
            bar  = "=" * bar_w
            size = human_bytes(done)
            line = f"  pg{bid:<6}  [{'='*bar_w}] ✓  {size}"
        elif state == 'failed':
            line = f"  pg{bid:<6}  [{'✗':^{bar_w}}] FAILED"
        elif state == 'too_small':
            line = f"  pg{bid:<6}  [{'~':^{bar_w}}] too small, skipped"
        elif state == 'exists':
            line = f"  pg{bid:<6}  [already on disk]"
        elif state == 'connecting':
            line = f"  pg{bid:<6}  [{'…':^{bar_w}}] connecting…"
        else:  # downloading
            if total > 0:
                ratio  = done / total
                filled = int(ratio * bar_w)
                bar    = "=" * filled + "-" * (bar_w - filled)
                pct    = f"{ratio*100:4.0f}%  {human_bytes(done)}/{human_bytes(total)}"
            else:
                filled = int((done / (AD_MIN_BYTES * 4)) * bar_w)
                filled = min(filled, bar_w - 1)
                bar    = "=" * filled + ">"  + "-" * (bar_w - filled - 1)
                pct    = f"~{human_bytes(done)}"
            line = f"  pg{bid:<6}  [{bar}] {pct}"
        lines.append(line)

    # Overall progress + ETA
    elapsed = max(0.001, time.perf_counter() - started_at)
    if n_done > 0 and n_total > 0:
        eta_s = (elapsed / n_done) * (n_total - n_done)
        eta   = f"ETA {eta_s:.0f}s" if eta_s < 3600 else f"ETA {eta_s/60:.1f}m"
    else:
        eta = "ETA …"
    overall = format_bar(n_done, n_total, 30)
    lines.append(f"  Overall {overall} {n_done}/{n_total} books  {eta}  "
                 f"elapsed {elapsed:.0f}s")

    # Move cursor up and redraw
    output = "\n".join(lines)
    n_lines = len(lines)
    # On first call the cursor is already at the right row; subsequent calls
    # move it back up.
    sys.stdout.write(output)
    sys.stdout.write("\n")
    sys.stdout.flush()


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
                f"\r  Probing…  found {len(good)}/{n_needed} usable IDs   "
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


def auto_download_blocking(settings: dict,
                            label_prefix: str = "",
                            multithreaded: Optional[bool] = None) -> None:
    """
    Download books up to the configured cap.
    multithreaded=None → respect settings["single_thread"].
    """
    folder    = settings["input_folder"]
    max_books = int(settings["ad_max_books"])
    max_bytes = int(settings["ad_max_bytes"])
    workers   = max(1, int(settings.get("workers", 4)))
    if multithreaded is None:
        multithreaded = not bool(settings.get("single_thread", False))

    ensure_folder(folder)
    count, used = _folder_state(folder)
    if count >= max_books or used >= max_bytes:
        print(f"{label_prefix}Cap already reached ({count}/{max_books} books).")
        return

    slots = max_books - count
    print(f"{label_prefix}Auto-download: {count}/{max_books} books — "
          f"fetching up to {slots} more…")

    ids = _ready_ids(folder, n_needed=slots * 2, verbose=True)
    if not ids:
        print(f"{label_prefix}No usable IDs found.")
        return

    # Clip to what we actually need
    ids = ids[:slots]

    if multithreaded and len(ids) > 1:
        _download_books_parallel(ids, folder, max_books, max_bytes, workers)
    else:
        _download_books_serial(ids, folder, max_books, max_bytes)

    count, used = _folder_state(folder)
    print(f"\n{label_prefix}Done: {count} books, {used//1048576} MB.")


def _download_books_serial(ids: List[int], folder: str,
                            max_books: int, max_bytes: int) -> None:
    rejected_local: Set[int] = set()
    n_total = len(ids)
    for i, book_id in enumerate(ids):
        count, used = _folder_state(folder)
        if count >= max_books or used >= max_bytes:
            break
        bar = format_bar(i, n_total, 28)
        sys.stdout.write(f"\r  {bar}  Downloading pg{book_id}.txt …" + " " * 10)
        sys.stdout.flush()
        path = _download_one_book(book_id, folder)
        if path:
            size = os.path.getsize(path)
            sys.stdout.write(f"\r  ✓ pg{book_id}.txt  ({size//1024} KB)" + " " * 20 + "\n")
        else:
            sys.stdout.write(f"\r  ✗ pg{book_id}.txt  (failed)" + " " * 20 + "\n")
            rejected_local.add(book_id)
        sys.stdout.flush()

    _flush_rejected(rejected_local)


def _download_books_parallel(ids: List[int], folder: str,
                              max_books: int, max_bytes: int,
                              workers: int) -> None:
    """
    Download books concurrently with a live per-book progress display.
    """
    n_total       = len(ids)
    status_dict: dict = {}
    lock          = threading.Lock()
    n_done        = 0
    rejected_local: Set[int] = set()
    started_at    = time.perf_counter()

    # Number of display lines we've already written (for cursor-up redraw)
    _prev_lines   = [0]

    def _refresh() -> None:
        # Move cursor up past previous output
        if _prev_lines[0] > 0:
            sys.stdout.write(f"\033[{_prev_lines[0]}A\033[J")
        _render_download_status(status_dict, lock,
                                 n_done, n_total, started_at)
        with lock:
            _prev_lines[0] = len(status_dict) + 1   # +1 for overall line

    print(f"  Downloading {n_total} books with {min(workers, n_total)} threads…\n")

    with ThreadPoolExecutor(max_workers=min(workers, n_total)) as ex:
        future_map = {}
        submitted = 0
        for bid in ids:
            count, used = _folder_state(folder)
            if count + submitted >= max_books or used >= max_bytes:
                break
            fut = ex.submit(
                _download_one_book_progress,
                bid, folder, status_dict, lock
            )
            future_map[fut] = bid
            submitted += 1

        # Live display loop
        display_thread_stop = threading.Event()

        def _display_loop():
            while not display_thread_stop.is_set():
                _refresh()
                time.sleep(0.25)

        disp = threading.Thread(target=_display_loop, daemon=True)
        disp.start()

        for fut in as_completed(future_map):
            bid  = future_map[fut]
            try:
                path = fut.result()
            except Exception:
                path = None
            if path is None:
                with lock:
                    st = status_dict.get(bid, {}).get('state', '')
                if st not in ('exists', 'done'):
                    rejected_local.add(bid)
            n_done += 1

        display_thread_stop.set()
        disp.join()

    # Final render
    if _prev_lines[0] > 0:
        sys.stdout.write(f"\033[{_prev_lines[0]}A\033[J")
    _render_download_status(status_dict, lock, n_done, n_total, started_at)

    _flush_rejected(rejected_local)


def _flush_rejected(rejected_local: Set[int]) -> None:
    if rejected_local:
        rej  = _load_id_cache(REJECTED_FILE)
        disc = _load_id_cache(DISCOVERED_FILE)
        rej  |= rejected_local
        disc -= rejected_local
        _save_id_cache(REJECTED_FILE,   rej)
        _save_id_cache(DISCOVERED_FILE, disc)


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

        _flush_rejected(rejected_local)


# ─────────────────────────────────────────────────────────────────────────────
# Settings menu
# ─────────────────────────────────────────────────────────────────────────────

def show_settings(s: dict) -> None:
    count, used = _folder_state(s["input_folder"])
    disc = _load_id_cache(DISCOVERED_FILE)
    rej  = _load_id_cache(REJECTED_FILE)
    print("\nCurrent settings:")
    groups = [
        ("Files",        ["input_folder", "model_file", "lowercase"]),
        ("Vocabulary",   ["vocab_size"]),
        ("Architecture", ["ctx_len", "embed_dim", "hidden_dim"]),
        ("Training",     ["epochs", "batch_size", "learning_rate",
                          "workers", "single_thread", "show_progress"]),
        ("Generation",   ["max_generate_tokens", "temperature", "top_k"]),
    ]
    for label, keys in groups:
        print(f"  ── {label}")
        for k in keys:
            print(f"    {k}: {s[k]}")

    use_gpu    = bool(s.get("use_gpu", False))
    gpu_device = int(s.get("gpu_device", 0))
    print(f"  ── GPU")
    print(f"    use_gpu:    {'ON' if use_gpu else 'OFF'}")
    if _cupy_available:
        if not _curand_available:
            print("    WARNING: libcurand.so missing — shuffle uses numpy fallback.")
        if _cupy_devices:
            for did, name in _cupy_devices:
                marker = " ◀ selected" if (use_gpu and did == gpu_device) else ""
                print(f"    [{did}] {name}{marker}")
        else:
            print("    (CuPy installed but no CUDA devices found)")
    else:
        print("    (CuPy not installed — install cupy-cuda11x or cupy-cuda12x)")

    ad_on = s.get("auto_download", False)
    mt    = not bool(s.get("single_thread", False))
    print(f"  ── Auto-Data Downloader")
    print(f"    auto_download:   {'ON' if ad_on else 'OFF'}")
    print(f"    multi-thread DL: {'ON' if mt else 'OFF'} (workers: {s['workers']})")
    print(f"    ad_max_books:    {s['ad_max_books']}")
    print(f"    ad_max_bytes:    {s['ad_max_bytes']//1048576} MB")
    print(f"    ad_refill_below: {s['ad_refill_below']}")
    print(f"    folder now:      {count} books, {used//1048576} MB")
    print(f"    discovered IDs:  {len(disc)}   rejected: {len(rej)}")


def _gpu_submenu(s: dict) -> None:
    while True:
        use_gpu    = bool(s.get("use_gpu", False))
        gpu_device = int(s.get("gpu_device", 0))

        print(f"\n── GPU Settings ────────────────────────────────────────")
        print(f"  Status: {'ON' if use_gpu else 'OFF'}  |  "
              f"Selected device: {gpu_device}")
        print()
        print(gpu_info_string())
        print()
        print("1) Toggle GPU ON/OFF")
        print("2) Select GPU device")
        print("0) Back")

        c = input("> ").strip()

        if c == "1":
            if not _cupy_available:
                print("CuPy is not installed.  "
                      "Install with:  pip install cupy-cuda11x  "
                      "(or cupy-cuda12x for CUDA 12)")
            elif not _cupy_devices:
                print("No CUDA devices detected.")
            else:
                s["use_gpu"] = not use_gpu
                print("GPU:", "ON" if s["use_gpu"] else "OFF")
        elif c == "2":
            if not _cupy_available or not _cupy_devices:
                print("No CUDA devices available.")
            else:
                print("Device IDs:")
                for did, name in _cupy_devices:
                    print(f"  {did}  {name}")
                v = input("Enter device ID: ").strip()
                try:
                    n = int(v)
                    if any(d == n for d, _ in _cupy_devices):
                        s["gpu_device"] = n
                        print(f"GPU device set to {n}.")
                    else:
                        print("Invalid device ID.")
                except ValueError:
                    print("Invalid input.")
        elif c == "0":
            save_settings(s)
            return
        else:
            print("Invalid.")

        save_settings(s)


def _auto_dl_submenu(s: dict) -> None:
    while True:
        count, used = _folder_state(s["input_folder"])
        disc = _load_id_cache(DISCOVERED_FILE)
        rej  = _load_id_cache(REJECTED_FILE)
        ad_on = s.get("auto_download", False)
        mt    = not bool(s.get("single_thread", False))
        print(f"\n── Auto-Data Downloader ────────────────────────────────")
        print(f"  Status: {'ON' if ad_on else 'OFF'}  |  "
              f"{count}/{s['ad_max_books']} books  "
              f"{used//1048576}/{s['ad_max_bytes']//1048576} MB")
        print(f"  Multi-thread DL: {'ON' if mt else 'OFF'}  |  "
              f"Workers: {s['workers']}")
        print(f"  Discovered: {len(disc)}   Rejected: {len(rej)}   "
              f"Probe range: {GUTENBERG_ID_MIN}–{GUTENBERG_ID_MAX}   "
              f"Min size: {AD_MIN_BYTES//1024} KB")
        print()
        print("1) Toggle ON/OFF")
        print("2) Max books")
        print("3) Max total MB")
        print("4) Refill-below threshold")
        print("5) Download now (multi-thread if enabled)")
        print("6) Download now (single-thread)")
        print("7) Probe for new IDs now")
        print("8) Clear caches")
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
            auto_download_blocking(s, multithreaded=mt)
        elif c == "6":
            save_settings(s)
            auto_download_blocking(s, multithreaded=False)
        elif c == "7":
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
        elif c == "8":
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
        print("  u) GPU settings")
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
        elif c == "u":
            _gpu_submenu(s)
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

    parser = argparse.ArgumentParser(
        description="Pure Python + numpy Neural LM (GPU-capable via CuPy)"
    )
    parser.add_argument(
        "mode", nargs="?",
        choices=["train", "chat", "settings", "download"],
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(s);         return
    if args.mode == "chat":
        chat(s);          return
    if args.mode == "settings":
        settings_menu(s); return
    if args.mode == "download":
        auto_download_blocking(s); return

    while True:
        count, used = _folder_state(s["input_folder"])
        ad_ind = (f" [Auto-DL ON | {count} books, {used//1048576}MB]"
                  if s.get("auto_download") else "")
        use_gpu    = bool(s.get("use_gpu", False))
        gpu_device = int(s.get("gpu_device", 0))
        if use_gpu and _cupy_available and _cupy_devices:
            dev_name = _cupy_devices[gpu_device][1] if gpu_device < len(_cupy_devices) else "?"
            gpu_ind  = f" [GPU:{gpu_device} {dev_name}]"
            if not _curand_available:
                gpu_ind += " (curand missing—shuffle via numpy)"
        elif use_gpu:
            gpu_ind = " [GPU: unavailable — will use CPU]"
        else:
            gpu_ind = " [CPU]"

        model_exists = os.path.exists(s["model_file"])
        print(f"\n=== Neural LM ==={ad_ind}{gpu_ind}")
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
