"""
Generate text from a trained checkpoint.
Usage: uv run generate.py [--prompt "your prompt"] [--tokens 200] [--temperature 0.8]
"""

import os
import sys
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import Tokenizer

# ---------------------------------------------------------------------------
# Model architecture (must match train.py)
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.n_kv_head < self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=50, device="mps"):
    model.eval()
    tokens = tokenizer.encode(prompt)
    ids = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop to max sequence length
        idx = ids if ids.size(1) <= model.config.sequence_len else ids[:, -model.config.sequence_len:]
        with torch.amp.autocast(device_type=device if device != "cpu" else "cpu", dtype=torch.bfloat16):
            logits = model(idx)
        logits = logits[:, -1, :] / temperature
        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

    return tokenizer.decode(ids[0].tolist())

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from trained checkpoint")
    parser.add_argument("--checkpoint", default="checkpoint.pt", help="Path to checkpoint")
    parser.add_argument("--prompt", default="The meaning of life is", help="Prompt text")
    parser.add_argument("--tokens", type=int, default=200, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--device", default="mps", help="Device (mps/cpu)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"No checkpoint found at '{args.checkpoint}'. Run train.py first.")
        sys.exit(1)

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = GPTConfig(**ckpt['config'])
    print(f"Model: {cfg.n_layer} layers, {cfg.n_embd} dim, {cfg.n_head} heads")

    model = GPT(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(args.device)
    model.eval()

    tokenizer = Tokenizer.from_directory()

    print(f"Prompt: {args.prompt}")
    print(f"Generating {args.tokens} tokens (temp={args.temperature}, top_k={args.top_k})...")
    print("---")
    output = generate(model, tokenizer, args.prompt, args.tokens, args.temperature, args.top_k, args.device)
    print(output)
