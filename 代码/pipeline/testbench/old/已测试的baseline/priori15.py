import torch
import torch.nn as nn
from einops import rearrange


class TopKAttention(nn.Module):
    def __init__(self, heads, head_dim, dim, topk=0.25):
        super().__init__()
        self.head_dim = head_dim
        self.heads = heads
        self.inner_dim = self.heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.topk = topk
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3)
        self.to_output = nn.Linear(self.inner_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.prior_proj = (nn.Linear(dim, self.inner_dim) if self.inner_dim != dim
                           else nn.Identity())

    def forward(self, x, prior=None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        Q, K, V = map(lambda t: rearrange(t, 'b l (h dim) -> b h l dim', dim=self.head_dim), qkv)
        if prior is not None:
            prior = self.prior_proj(prior)
            prior_heads = rearrange(
                prior, 'b l (h d) -> b h l d', h=self.heads, d=self.head_dim
            )
            V = V * prior_heads
        K_T = K.transpose(-1, -2)
        att_score = Q @ K_T * self.scale
        n = att_score.shape[-1]
        k = max(1, int(n * self.topk))
        topk_vals, _ = att_score.topk(k, dim=-1)
        threshold = topk_vals[..., -1:]
        mask = att_score < threshold
        att_score = att_score.masked_fill(mask, float('-inf'))
        att = nn.functional.softmax(att_score, dim=-1)
        out = att @ V
        out = rearrange(out, 'b h l dim -> b l (h dim)')
        output = self.to_output(out)
        return output
