"""Student model: a modernised small GPT.

Diagnosed limitations of the baseline (4 x 128, learned absolute positions, LayerNorm+GELU):
  1. Capacity: ~1.1M parameters, and ~25% of its per-token compute is the 2048-way output head.
  2. Positions are a learned table fitted on only a few million training tokens.
  3. Multi-epoch training on a small corpus overfits once the model or step count grows.

Changes (each one can be switched off from the JSON config for ablations):
  - wider model (config: width/depth), sized to stay inside the 5x CPU-scoring budget
  - rope    : rotary position embeddings replace the learned position table
  - swiglu  : gated MLP (hidden ~ 8/3 * width, so parameters match a 4x GELU MLP)
  - norm    : 'rms' (RMSNorm) or 'layer' (LayerNorm)
  - dropout : residual + embedding dropout (disabled automatically in eval mode)
  - pointer : optional within-window pointer-sentinel cache (see PointerCache notes below). Off by default,
              so checkpoints trained without it (final-a/b/c/d) load and score exactly as before.
Input and output embeddings stay tied, as in the baseline.

Pointer cache (config: pointer=true, ptr_dim=128, ptr_null_bias=6.0):
  For a prediction at position t the model may attend over earlier positions j < t of the SAME window. Each
  slot j stores (final hidden state h_j, the observed next token ids[j+1]), and j+1 <= t, so only the observed
  prefix is used. A learned query/key match gives weights a_j; an extra "null" slot (sentinel) keeps the
  ordinary softmax distribution:  p(v) = a_null * p_lm(v) + sum_{j<t, ids[j+1]=v} a_j.
  This sums to 1 exactly, is stateless (nothing survives between windows) and adds no stored assets.
  Idea: Merity et al., Pointer Sentinel Mixture Models (arXiv:1609.07843); Grave et al., continuous cache.
  forward() then returns log-probabilities of the mixture, which are valid logits for cross-entropy.

The file is self-contained (does not import model.py) so the evaluator can rebuild it from a checkpoint.
Missing config keys fall back to the defaults below, so the unit tests can build a tiny model.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


def make_norm(kind, width):
    if kind == 'rms':
        return RMSNorm(width)
    if kind == 'layer':
        return nn.LayerNorm(width)
    raise ValueError(f"norm must be 'rms' or 'layer', got {kind!r}")


def rope_tables(context, head_dim, base=10000.0):
    inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    angles = torch.outer(torch.arange(context, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """x: [batch, heads, time, head_dim]; rotates each (first-half, second-half) feature pair by position."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[:x.shape[-2]].to(x.dtype)
    sin = sin[:x.shape[-2]].to(x.dtype)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class Attention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, rope):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        # Causal mask: position t attends only to positions <= t of this window.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.drop(self.proj(y.transpose(1, 2).reshape(batch, length, width)))


class MLP(nn.Module):
    def __init__(self, width, swiglu, dropout):
        super().__init__()
        self.swiglu = swiglu
        if swiglu:
            hidden = (int(8 * width / 3) + 15) // 16 * 16
            self.up = nn.Linear(width, 2 * hidden, bias=False)
            self.down = nn.Linear(hidden, width, bias=False)
        else:
            self.up = nn.Linear(width, 4 * width, bias=False)
            self.down = nn.Linear(4 * width, width, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.up(x)
        if self.swiglu:
            gate, value = h.chunk(2, dim=-1)
            h = F.silu(gate) * value
        else:
            h = F.gelu(h)
        return self.drop(self.down(h))


class Block(nn.Module):
    def __init__(self, width, heads, dropout, swiglu, norm):
        super().__init__()
        self.norm1, self.norm2 = make_norm(norm, width), make_norm(norm, width)
        self.attn = Attention(width, heads, dropout)
        self.mlp = MLP(width, swiglu, dropout)

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        return x + self.mlp(self.norm2(x))


class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width, heads, depth = config['width'], config['heads'], config['depth']
        dropout = config.get('dropout', 0.0)
        swiglu = config.get('swiglu', True)
        norm = config.get('norm', 'rms')
        self.use_rope = config.get('rope', True)
        if width % heads or (width // heads) % 2:
            raise ValueError('width must be divisible by heads, and head size must be even (RoPE).')
        self.token = nn.Embedding(config['vocab'], width)
        if self.use_rope:
            cos, sin = rope_tables(self.context, width // heads, config.get('rope_base', 10000.0))
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)
        else:
            self.pos = nn.Embedding(self.context, width)
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(width, heads, dropout, swiglu, norm) for _ in range(depth)])
        self.norm = make_norm(norm, width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.pointer = bool(config.get('pointer', False))
        if self.pointer:
            ptr_dim = config.get('ptr_dim', 128)
            self.ptr_q = nn.Linear(width, ptr_dim, bias=False)
            self.ptr_k = nn.Linear(width, ptr_dim, bias=False)
            self.ptr_null = nn.Linear(width, 1)
            keys_before_queries = torch.tril(torch.ones(self.context, self.context), -1) > 0  # slot j < query t
            self.register_buffer('ptr_mask', keys_before_queries, persistent=False)
        self.apply(self.initialize)
        if self.pointer:
            # Start with most mass on the ordinary LM; Adam then learns when copying helps.
            nn.init.constant_(self.ptr_null.bias, config.get('ptr_null_bias', 6.0))
        # GPT-2 style: shrink the residual output projections so deeper stacks start stable.
        for block in self.blocks:
            for layer in (block.attn.proj, block.mlp.down):
                nn.init.normal_(layer.weight, std=.02 / (2 * depth) ** .5)
        self.head.weight = self.token.weight  # tied embeddings (as in the baseline)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        if self.use_rope:
            rope = (self.rope_cos, self.rope_sin)
        else:
            rope = None
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = self.embed_drop(x)
        for block in self.blocks:
            x = block(x, rope)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: next-token logits [batch, time, vocab].

        With pointer=true the returned values are log-probabilities of the pointer/LM mixture; they are
        normalized already, so log_softmax / cross_entropy on them is the identity / the mixture NLL.
        """
        hidden = self.features(ids)
        logits = self.head(hidden)
        if not self.pointer:
            return logits
        return self.mix_with_cache(ids, hidden, logits)

    def mix_with_cache(self, ids, hidden, logits):
        batch, length = ids.shape
        scores = torch.matmul(self.ptr_q(hidden), self.ptr_k(hidden).transpose(-1, -2)).float()  # [B, t, j]
        scores = scores.masked_fill(~self.ptr_mask[:length, :length], float('-inf'))           # only j < t
        null = self.ptr_null(hidden).float()                                                  # [B, t, 1]
        attention = torch.softmax(torch.cat([scores, null], dim=-1), dim=-1)                   # [B, t, T+1]
        weights, keep_lm = attention[..., :length].contiguous(), attention[..., length:]
        # Slot j holds the observed token ids[j+1]; the last slot is always masked (weight exactly 0).
        target = F.pad(ids[:, 1:], (0, 1)).unsqueeze(1).expand(batch, length, length).contiguous()
        mixture = (torch.softmax(logits.float(), dim=-1) * keep_lm).scatter_add(-1, target, weights)
        return torch.log(mixture.clamp_min(1e-30))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities. Stateless, strictly causal."""
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return StudentGPT(config)
