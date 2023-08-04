import torch
from torch import nn, einsum
from einops.layers.torch import Reduce
from einops import repeat
from model.modules import PreNorm, Attention, FeedForward
import random


class SIA(nn.Module):
    def __init__(
        self,
        attn_mode,
        latent_dim,
        feature_dim,
        attn_num_heads,
        attn_dim_head,
        attn_depth,
        attn_self_per_cross,
        attn_dropout,
        attn_ff_dropout,
        dim_item_feats,
        num_items,
        maxlen,
        device,
        latent_random,
        latent_with_pos,
        item_with_pos,
    ):
        """
        Args:
          - attn_mode: Attention mode for cross attention.
          - latent_dim: Dimension of a latent vector.
          - feature_dim: Dimension of features.
          - attn_num_heads: Number of heads for attention operation in the iterative attention.
          - attn_dim_head: Dimension for attention operation in the iterative attention.
          - attn_depth: Depth of the iterative attention.
          - attn_self_per_cross: Number of self attention blocks per cross attention.
          - attn_dropout: Attention dropout in the iterative attention.
          - attn_ff_dropout: Feedforward dropout in the iterative attention.
          - dim_item_feats: List of dimensions for item features.
          - num_items: Number of all items in the dataset.
          - maxlen: Max length of the sequence.
          - device: Device type.
          - latent_random: Use randomly initialized latent.
          - latent_with_pos: Add positional embedding to latent.
          - item_with_pos: Add positional embedding to item features.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.attn_mode = attn_mode
        self.id_embedding = nn.Embedding(
            num_embeddings=(num_items + 1),  # 1~num_items
            embedding_dim=latent_dim,
            padding_idx=0,
        )
        self.pos_embedding = nn.Embedding(
            num_embeddings=(maxlen + 1),  # 1~maxlen
            embedding_dim=latent_dim,
            padding_idx=0,
        )
        self.item_pos_embedding = nn.Embedding(
            num_embeddings=(maxlen + 1),  # 1~maxlen
            embedding_dim=feature_dim,
            padding_idx=0,
        )
        self.feat_embeddings = nn.ModuleList([nn.Linear(feat_dim, feature_dim, bias=False) for feat_dim in dim_item_feats])
        self.feat_embeddings.append(nn.Embedding(num_embeddings=(num_items + 1), embedding_dim=feature_dim, padding_idx=0))

        get_latent_attn = lambda: PreNorm(
            latent_dim,
            Attention(
                query_dim=latent_dim,
                heads=attn_num_heads,
                dim_head=attn_dim_head,
                dropout=attn_dropout,
            ),
        )
        get_cross_attn = lambda context_dim: PreNorm(
            latent_dim,
            Attention(
                query_dim=latent_dim,
                context_dim=context_dim,
                heads=attn_num_heads,
                dim_head=attn_dim_head,
                dropout=attn_dropout,
            ),
        )
        get_ff = lambda: PreNorm(
            latent_dim,
            FeedForward(
                latent_dim,
                dropout=attn_ff_dropout,
            ),
        )
        self.cross_attn = get_cross_attn(feature_dim)
        self.cross_ff = get_ff()
        self_attns = nn.ModuleList([])
        for _ in range(attn_self_per_cross):
            self_attns.append(
                nn.ModuleList(
                    [
                        get_latent_attn(),
                        get_ff(),
                    ]
                )
            )
        self.layers = nn.ModuleList([])
        for _ in range(attn_depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        self.cross_attn,
                        self.cross_ff,
                        self_attns,
                    ]
                )
            )
        self.out_ln = nn.LayerNorm(latent_dim)
        self.latent = nn.Parameter(torch.Tensor(1, maxlen + 1, latent_dim))
        nn.init.xavier_uniform_(self.latent)
        self.latent_random = latent_random
        self.latent_with_pos = latent_with_pos
        self.item_with_pos = item_with_pos

    def forward(self, batch_x):
        # ID sequences -> embedded latent vectors (x)
        seq, pos, item_feats = batch_x
        batch_size, seqlen = seq.shape
        id_emb = self.id_embedding(seq)
        pos_emb = self.pos_embedding(pos)
        x = repeat(self.latent, "1 l d -> b l d", b=batch_size) if self.latent_random else id_emb
        x = (x + pos_emb) if self.latent_with_pos else x

        # Item features -> concatenated item features (item_feat)
        item_feats.append(seq)
        feat_embs = [
            feat_embedding(item_feat)
            for item_feat, feat_embedding
            in zip(item_feats, self.feat_embeddings)
        ]
        feat_emb = torch.cat(feat_embs, dim=1)
        item_pos_emb = self.item_pos_embedding(pos.repeat(1, len(feat_embs)))
        feat_emb = (feat_emb + item_pos_emb) if self.item_with_pos else feat_emb

        # Masks for Attention
        pad_mask = einsum("b i d, b j d -> b i j", pos.unsqueeze(2), pos.unsqueeze(2))
        time_mask = repeat(pos, "b n -> b m n", m=pos.shape[-1]) * repeat(pos, "b n -> b n m", m=pos.shape[-1]).tril()
        mask = (pad_mask * time_mask) if self.attn_mode == "masked" else pad_mask
        item_mask = mask.repeat(1, 1, len(feat_embs))
        mask = (mask > 0).to(self.device)
        item_mask = (item_mask > 0).to(self.device)

        # Iterative Attention
        out = []
        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context=feat_emb, mask=item_mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x, mask=mask) + x
                x = self_ff(x) + x

            out.append(self.out_ln(x[:, -1, :]))

        return out
