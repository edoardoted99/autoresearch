"""
One-time data preparation for autoresearch MNIST experiments.
Downloads MNIST dataset and provides runtime utilities.

Usage:
    python prepare.py

Data is stored in ~/.cache/autoresearch-mnist/.
This file is READ-ONLY — do not modify.
"""

import json
import os
import gzip
import struct
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import torch
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 60          # training time budget in seconds
IMG_SIZE = 28
NUM_CLASSES = 10
EVAL_BATCH_SIZE = 512

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-mnist")
MIRROR_URL = "https://ossci-datasets.s3.amazonaws.com/mnist"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

# ---------------------------------------------------------------------------
# Download & parse
# ---------------------------------------------------------------------------

def download_file(filename):
    """Download a single MNIST file if not already cached."""
    filepath = os.path.join(CACHE_DIR, filename)
    if os.path.exists(filepath):
        return filepath
    url = f"{MIRROR_URL}/{filename}"
    print(f"  Downloading {filename}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(filepath, "wb") as f:
        f.write(response.content)
    print(f"  Saved {filename}")
    return filepath


def _parse_idx_images(filepath):
    with gzip.open(filepath, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).copy().reshape(num, rows, cols)
    return data


def _parse_idx_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8).copy()
    return data


def download_and_parse():
    """Download raw MNIST files, parse, and save as a single .pt file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = {key: download_file(fn) for key, fn in FILES.items()}
    torch.save({
        "train_images": torch.from_numpy(_parse_idx_images(paths["train_images"])).float() / 255.0,
        "train_labels": torch.from_numpy(_parse_idx_labels(paths["train_labels"]).astype(np.int64)),
        "test_images":  torch.from_numpy(_parse_idx_images(paths["test_images"])).float() / 255.0,
        "test_labels":  torch.from_numpy(_parse_idx_labels(paths["test_labels"]).astype(np.int64)),
    }, os.path.join(CACHE_DIR, "mnist.pt"))
    print(f"  Saved processed data to {CACHE_DIR}/mnist.pt")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

_status_path = None

def load_data(device="cpu"):
    """
    Load MNIST data.
    Returns (train_images, train_labels, test_images, test_labels).
    Images are float32 in [0, 1] with shape (N, 1, 28, 28).
    Also starts experiment tracking (status.json) automatically.
    """
    # auto-start experiment status on first call
    global _status_path
    if _status_path is None:
        import atexit
        _status_path = PROGRAM_DIR / "status.json"
        _status_path.write_text(json.dumps({
            "running": True,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "time_budget": TIME_BUDGET,
        }))
        atexit.register(lambda: _status_path.unlink(missing_ok=True))
    data = torch.load(os.path.join(CACHE_DIR, "mnist.pt"), map_location=device, weights_only=True)
    return (
        data["train_images"].unsqueeze(1),
        data["train_labels"],
        data["test_images"].unsqueeze(1),
        data["test_labels"],
    )


def make_dataloader(images, labels, batch_size, shuffle=True):
    """Create a DataLoader from image and label tensors."""
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=shuffle, drop_last=True)


@torch.no_grad()
def evaluate(model, device="cpu"):
    """
    Evaluate model on the MNIST test set.
    Returns (val_loss, val_accuracy).

    DO NOT MODIFY — this is the fixed evaluation metric.
    """
    model.eval()
    _, _, test_images, test_labels = load_data(device)
    loader = make_dataloader(test_images, test_labels, EVAL_BATCH_SIZE, shuffle=False)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        logits = model(images)
        total_loss += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
    val_loss = total_loss / total_samples
    val_accuracy = 100.0 * total_correct / total_samples
    model.train()
    return val_loss, val_accuracy

# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

# Always write results to programs/mnist/, regardless of where this file lives
def _get_program_dir():
    here = Path(__file__).resolve().parent
    candidate = here / "programs" / "mnist"
    if candidate.exists():
        return candidate  # running from repo root
    return here  # running from programs/mnist/

PROGRAM_DIR = _get_program_dir()
RESULTS_FILE = PROGRAM_DIR / "results.tsv"
RESULTS_HEADER = "commit\tval_loss\tmemory_gb\tstatus\tdescription\n"


def log_result(val_loss, memory_gb, status="keep", description=""):
    """Append a row to results.tsv. Auto-creates the file with header if needed."""
    import subprocess
    # get current short commit hash
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    # create file with header if it doesn't exist
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(RESULTS_HEADER)
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{val_loss:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")
    print(f"Logged to {RESULTS_FILE}: {commit} val_loss={val_loss:.6f} [{status}]")


# ---------------------------------------------------------------------------
# Experiment status (heartbeat file for the dashboard)
# ---------------------------------------------------------------------------

@contextmanager
def experiment_status():
    """Write a status.json while training is running. Dashboard reads this."""
    status_path = PROGRAM_DIR / "status.json"
    status_path.write_text(json.dumps({
        "running": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "time_budget": TIME_BUDGET,
    }))
    try:
        yield
    finally:
        status_path.unlink(missing_ok=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Cache directory: {CACHE_DIR}")
    t0 = time.time()
    download_and_parse()
    t1 = time.time()
    print(f"\nDone in {t1 - t0:.1f}s. Ready to train.")
