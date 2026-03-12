# MNIST Autoresearch

You are an autonomous ML researcher. Your goal: achieve the lowest `val_loss` on MNIST within a fixed 60-second training budget by iterating on `train.py`.

## Setup

All files live in `programs/mnist/`. **Always `cd programs/mnist/` before running commands.**

1. **Create an initial branch**: `git checkout -b autoresearch/mnist/<tag>` where `<tag>` is a short descriptive name for your research direction (e.g. `regularization`, `resnet`, `augmentation`).
2. `cd programs/mnist/`
3. **Read the code** to build full context:
   - `prepare.py` — data download, dataloader, `evaluate()`, `log_result()`, `experiment_status()`. **Read-only, do not modify.**
   - `train.py` — model, optimizer, training loop. **This is the only file you edit.**
4. **Verify data exists**: `ls ~/.cache/autoresearch-mnist/mnist.pt`. If missing, run `uv run python prepare.py`.
5. **Run the baseline**: `uv run python run_experiment.py > run.log 2>&1` (this auto-creates `programs/mnist/results.tsv` — results always live inside the program folder)
6. `git add programs/mnist/results.tsv programs/mnist/train.py && git commit -m "results: baseline"`
7. **Go.**

## Rules

**You edit `train.py` only.** Everything in that file is fair game:

- Model architecture (CNN, MLP, ResNet, anything)
- Optimizer and learning rate schedule
- Hyperparameters (batch size, weight decay, dropout, etc.)
- Training loop logic (data augmentation, mixup, gradient accumulation, etc.)

**You cannot:**

- Modify `prepare.py` — it contains the fixed evaluation harness
- Install packages — use only what's in `pyproject.toml` (torch, numpy, etc.)
- Cheat the evaluation — the `evaluate()` function in `prepare.py` is the ground truth
- **Move or copy files outside `programs/mnist/`** — all program files MUST stay in `programs/mnist/`. DO NOT create `train.py`, `prepare.py`, or `results.tsv` at the repo root.
- **Modify or delete `autoresearch_hooks.py`** — it provides dashboard integration (time budget tracking, result logging). It is imported by `prepare.py` automatically.

## Baseline

The current `train.py` is a simple 2-layer CNN (Conv→Pool→Conv→Pool→FC→FC) with Adam, lr=1e-3, batch_size=128. On a GPU it achieves roughly:

```
val_loss:         ~0.12
val_accuracy:     ~99.3%
training_seconds: 60.0
num_params:       421,642
```

This is already quite good for MNIST. Your job is to push val_loss lower. Even small improvements (0.12 → 0.08) are meaningful. Ideas to explore:

- **Architecture**: deeper CNNs, residual connections, batch norm, wider layers
- **Regularization**: dropout, weight decay, data augmentation (random crops, rotations)
- **Optimizer**: AdamW, SGD+momentum, cosine LR schedule, warmup
- **Training tricks**: mixup, label smoothing, gradient clipping
- **Efficiency**: if the model is training too fast (loss=0.0000 early), the model might be too small or overfitting — make it harder to overfit

Note: the baseline overfits heavily (train loss → 0.0000 by epoch ~15, but val_loss stays ~0.12). Regularization is likely the biggest lever.

## Output format

The script prints a summary after training:

```
---
val_loss:         0.052300
val_accuracy:     99.45
training_seconds: 60.1
total_seconds:    65.3
peak_vram_mb:     625.7
num_steps:        96861
num_params:       421642
epochs:           207
```

Extract results: `grep "^val_loss:\|^val_accuracy:\|^peak_vram_mb:" run.log`

## Logging results

Results are **automatically logged** to `results.tsv` by `log_result()` from `prepare.py`. The `train.py` baseline already calls it at the end:

```python
log_result(val_loss, peak_vram_mb / 1024, description="your description here")
```

You MUST keep this call at the end of `train.py` and update the `description` parameter for each experiment. For crashes, call `log_result(0.0, 0.0, status="crash", description="what went wrong")`.

The `status` parameter defaults to `"keep"`. After comparing results, update it to `"discard"` if the experiment was worse.

**IMPORTANT**:
- `results.tsv` lives in `programs/mnist/` — **DO NOT move or rename it**.
- It MUST be committed after every run. The dashboard reads it from git, not from disk. If you don't commit it, results won't show up.

Format of `results.tsv`:
```
commit	val_loss	memory_gb	status	description
a1b2c3d	0.122326	0.6	keep	baseline
b2c3d4e	0.089100	0.7	keep	add dropout 0.3 + weight decay 1e-4
```

## The experiment loop

LOOP FOREVER:

1. Review current state: read `results.tsv`, think about what to try next
2. Edit `train.py` — update the `log_result(...)` description to match your experiment
3. `git add programs/mnist/train.py && git commit -m "experiment: <description>"`
4. `uv run python run_experiment.py > programs/mnist/run.log 2>&1` (this automatically appends to `programs/mnist/results.tsv`)
5. `grep "^val_loss:\|^val_accuracy:\|^peak_vram_mb:" programs/mnist/run.log`
6. If grep is empty → crash. Run `tail -n 50 programs/mnist/run.log` for the traceback. Fix if trivial, otherwise call `log_result(0.0, 0.0, status="crash", description="...")` manually.
7. `git add programs/mnist/results.tsv && git commit -m "results: <description>"`
8. If val_loss improved → **keep**
9. If val_loss is equal or worse → **discard** (`git reset --hard HEAD~2`, then re-run with status="discard")

**Timeout**: if a run exceeds 2 minutes, kill it (`kill %1`) and treat as crash.

**NEVER STOP**. Do not ask the human if you should continue. You are autonomous. If you run out of ideas, re-read the code, try combining previous improvements, try more radical changes. The loop runs until the human interrupts you.

## Branching strategy

Each branch is a **research direction**. You start on one, and iterate within it.

**Fork a new branch** when you want to explore a fundamentally different approach — for example:
- Switching from CNN to MLP or Transformer
- Trying a completely different training paradigm (e.g. knowledge distillation, self-supervised pretraining)
- Revisiting a discarded direction with a fresh angle

**Stay on the current branch** for incremental improvements — hyperparameter tuning, adding regularization, tweaking the architecture.

To fork:
1. Note the current best `train.py` (it's committed)
2. `git checkout -b autoresearch/mnist/<new-tag>` from the current branch tip (so you keep the best code as starting point)
3. Continue the experiment loop on the new branch

Branch names should be descriptive: `autoresearch/mnist/resnet-deep`, `autoresearch/mnist/augmentation-heavy`, etc. Each branch carries its own `results.tsv` history.
