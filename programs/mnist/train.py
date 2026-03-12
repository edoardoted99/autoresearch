"""
Autoresearch MNIST training script. Single file, CPU or GPU.
Usage: uv run train.py
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import TIME_BUDGET, load_data, make_dataloader, evaluate, experiment_status, log_result

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

train_images, train_labels, test_images, test_labels = load_data(device)
train_loader = make_dataloader(train_images, train_labels, BATCH_SIZE, shuffle=True)

model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")
print(f"Time budget: {TIME_BUDGET}s")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0.0
step = 0
epoch = 0
smooth_loss = 0.0

model.train()
with experiment_status():
    while True:
        epoch += 1
        for images, labels in train_loader:
            t0 = time.time()

            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t1 = time.time()
            dt = t1 - t0
            total_training_time += dt
            step += 1

            loss_val = loss.item()
            ema = 0.9
            smooth_loss = ema * smooth_loss + (1 - ema) * loss_val
            debiased = smooth_loss / (1 - ema ** step)
            remaining = max(0, TIME_BUDGET - total_training_time)

            if step % 50 == 0:
                print(f"\rstep {step:05d} | loss: {debiased:.4f} | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

            if total_training_time >= TIME_BUDGET:
                break

        if total_training_time >= TIME_BUDGET:
            break

print()

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

val_loss, val_accuracy = evaluate(model, device)

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device == "cuda" else 0

print("---")
print(f"val_loss:         {val_loss:.6f}")
print(f"val_accuracy:     {val_accuracy:.2f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params:       {num_params}")
print(f"epochs:           {epoch}")

log_result(val_loss, peak_vram_mb / 1024, description="baseline")
