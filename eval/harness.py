"""Evaluation Harness for Model Quality and Numeric Benchmarking (Task UA.2).

Provides standardized evaluation datasets, metrics, and prompt batteries:
- Pinned WikiText-2 test windows (hashed and reproducible)
- 20-prompt evaluation battery
- Core metrics:
  1. Perplexity (PPL) on teacher-forced token windows
  2. Mean KL Divergence KL(P_fp32 || P_int) per token
  3. Top-1 ArgMax Agreement percentage
  4. Greedy Match Length (number of tokens before first divergence)
"""

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"
DATA_DIR = EVAL_DIR / "data"
WIKITEXT_FILE = DATA_DIR / "wikitext2_test.txt"
PROMPTS_FILE = EVAL_DIR / "prompts.json"
MANIFEST_FILE = DATA_DIR / "manifest.json"

WIKITEXT_URL = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt"

# 20 Standardized Prompts for Generation Battery
STANDARD_PROMPTS = [
    {"id": "p01", "name": "General Knowledge", "text": "The capital of France is Paris. The capital of Germany is"},
    {"id": "p02", "name": "Basic Arithmetic", "text": "Two plus two equals four. Three plus three equals"},
    {"id": "p03", "name": "Code Python", "text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"},
    {"id": "p04", "name": "Science History", "text": "In 1687, Sir Isaac Newton published the Mathematical Principles of"},
    {"id": "p05", "name": "Grammar Completion", "text": "She walked to the library because she needed to find a book about"},
    {"id": "p06", "name": "Geography", "text": "The longest river in South America is the"},
    {"id": "p07", "name": "Astronomy", "text": "The planet closest to the Sun in our solar system is"},
    {"id": "p08", "name": "Proverb Completion", "text": "A bird in the hand is worth two in the"},
    {"id": "p09", "name": "Algorithm Explanation", "text": "Binary search has a logarithmic time complexity of O("},
    {"id": "p10", "name": "Physics Concept", "text": "According to Einstein's theory of relativity, energy is related to mass by the equation E ="},
    {"id": "p11", "name": "Short Context 1", "text": "Once upon a time, there was a quiet village nestled in the"},
    {"id": "p12", "name": "Short Context 2", "text": "The boiling point of water at standard atmospheric pressure is"},
    {"id": "p13", "name": "Logic Reasoning", "text": "All men are mortal. Socrates is a man. Therefore, Socrates is"},
    {"id": "p14", "name": "Hardware Description", "text": "In digital electronics, a flip-flop is a sequential circuit that stores one"},
    {"id": "p15", "name": "Repetition Test", "text": "A B C D E F G H I J K L M N O P Q R S T U V W X Y"},
    {"id": "p16", "name": "Number Series", "text": "1, 2, 4, 8, 16, 32, 64, 128,"},
    {"id": "p17", "name": "Chemistry Fact", "text": "The chemical symbol for gold in the periodic table is"},
    {"id": "p18", "name": "Computer Architecture", "text": "RISC stands for Reduced Instruction Set"},
    {"id": "p19", "name": "Dialogue", "text": "Hello, how can I help you today? I would like to make a reservation for"},
    {"id": "p20", "name": "Philosophy", "text": "Cogito, ergo sum is a Latin philosophical proposition that translates to"},
]


def ensure_dataset_present() -> Path:
    """Download and hash WikiText-2 test set if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not WIKITEXT_FILE.exists():
        print(f"[*] Downloading WikiText-2 test set from {WIKITEXT_URL}...")
        urllib.request.urlretrieve(WIKITEXT_URL, WIKITEXT_FILE)

    raw_bytes = WIKITEXT_FILE.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    manifest = {
        "dataset": "wikitext-2-raw-v1",
        "split": "test",
        "file": WIKITEXT_FILE.name,
        "bytes": len(raw_bytes),
        "sha256": sha256,
        "source_url": WIKITEXT_URL,
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))
    return WIKITEXT_FILE


def ensure_prompts_file_present() -> Path:
    """Save standard prompt battery if missing."""
    if not PROMPTS_FILE.exists():
        PROMPTS_FILE.write_text(json.dumps(STANDARD_PROMPTS, indent=2))
    return PROMPTS_FILE


def get_tokenized_windows(
    model_id: str = "HuggingFaceTB/SmolLM2-135M",
    revision: str = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    num_windows: int = 10,
    window_length: int = 64,
) -> list[list[int]]:
    """Return fixed, reproducible token windows from WikiText-2 test set."""
    txt_path = ensure_dataset_present()
    text = txt_path.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)
    stride = (total_tokens - window_length) // num_windows

    windows = []
    for i in range(num_windows):
        start = i * stride
        windows.append(tokens[start : start + window_length])

    return windows


# ==============================================================================
# Metric Computations
# ==============================================================================


def compute_cross_entropy_and_perplexity(
    logits: np.ndarray, target_tokens: list[int]
) -> tuple[float, float]:
    """Compute mean cross-entropy loss and perplexity over token sequence.
    
    logits: [N, vocab_size]
    target_tokens: list of length N
    """
    assert len(logits) == len(target_tokens)
    n = len(target_tokens)
    if n == 0:
        return 0.0, 1.0

    # Log-softmax along vocab dimension
    m = np.max(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(logits - m)
    log_sum_exp = np.log(np.sum(exp_shifted, axis=-1, keepdims=True))
    log_probs = (logits - m) - log_sum_exp

    # Extract log probability of correct target tokens
    nll = -log_probs[np.arange(n), target_tokens]
    mean_loss = float(np.mean(nll))
    ppl = float(np.exp(mean_loss))
    return mean_loss, ppl


def compute_kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """Compute mean KL divergence KL(P_fp32 || Q_int) across token positions."""
    # Convert logits to probabilities
    def to_probs(logits):
        m = np.max(logits, axis=-1, keepdims=True)
        e = np.exp(logits - m)
        return e / np.sum(e, axis=-1, keepdims=True)

    p = to_probs(p_logits)
    q = to_probs(q_logits)
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    kl = np.sum(p * np.log(p / q), axis=-1)
    return float(np.mean(kl))


def compute_top1_agreement(logits1: np.ndarray, logits2: np.ndarray) -> float:
    """Compute percentage of positions where argmax matches exactly."""
    a1 = np.argmax(logits1, axis=-1)
    a2 = np.argmax(logits2, axis=-1)
    return float(np.mean(a1 == a2) * 100.0)


def compute_greedy_match_length(tokens1: list[int], tokens2: list[int]) -> int:
    """Compute number of consecutive generated tokens matching before first divergence."""
    match_len = 0
    for t1, t2 in zip(tokens1, tokens2, strict=False):
        if t1 == t2:
            match_len += 1
        else:
            break
    return match_len
