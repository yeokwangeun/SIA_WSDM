import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class PreNorm(nn.Module):
    """
    Reference: https://github.com/lucidrains/perceiver-pytorch
    """

    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    """
    Reference: https://github.com/lucidrains/perceiver-pytorch
    """

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    """
    Reference: https://github.com/lucidrains/perceiver-pytorch
    """

    def __init__(self, dim, mult=2, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Reference: https://github.com/lucidrains/perceiver-pytorch
    """

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            # mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b i j -> (b h) i j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class FeatureTransformer(nn.Module):
    def __init__(
        self,
        feat_dim,
        out_dim,
        out_latents,
        inner_dim=64,
        heads=1,
        feat_num_layers=2,
        out_num_layers=2,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()
        get_self_attn = lambda dim: PreNorm(
            dim, Attention(dim, heads=heads, dim_head=inner_dim, dropout=attn_dropout)
        )
        get_ff = lambda dim: PreNorm(dim, FeedForward(dim, dropout=ff_dropout))
        self.item_attn_layers = nn.ModuleList(
            [
                nn.ModuleList([get_self_attn(feat_dim), get_ff(feat_dim)])
                for _ in range(feat_num_layers)
            ]
        )
        self.cross_attn = PreNorm(
            out_dim,
            Attention(query_dim=out_dim, context_dim=feat_dim, dim_head=inner_dim, heads=heads),
        )
        self.cross_ff = PreNorm(out_dim, FeedForward(out_dim))
        self.out_attn_layers = nn.ModuleList(
            [
                nn.ModuleList([get_self_attn(out_dim), get_ff(out_dim)])
                for _ in range(out_num_layers)
            ]
        )
        self.S = nn.Parameter(torch.Tensor(1, out_latents, out_dim))
        nn.init.xavier_uniform_(self.S)

    def forward(self, item_feat, mask=None):
        device = item_feat.device
        out = repeat(self.S, "1 n d -> b n d", b=item_feat.size(0))
        if exists(mask):
            mask_item = mask.unsqueeze(2).to(device)
            mask_item_attn = einsum("b i d, b j d -> b i j", mask_item, mask_item) > 0
            mask_out = torch.ones((out.shape[0], out.shape[1])).unsqueeze(2).to(device)
            mask_out_attn = einsum("b i d, b j d -> b i j", mask_out, mask_item) > 0
        else:
            mask_item_attn = None
            mask_out_attn = None
        for self_attn, ff in self.item_attn_layers:
            item_feat = self_attn(item_feat, mask=mask_item_attn) + item_feat
            item_feat = ff(item_feat) + item_feat
        out = self.cross_attn(out, context=item_feat, mask=mask_out_attn) + out
        out = self.cross_ff(out) + out
        for self_attn, ff in self.out_attn_layers:
            out = self_attn(out, mask=None) + out
            out = ff(out) + out
        return out
