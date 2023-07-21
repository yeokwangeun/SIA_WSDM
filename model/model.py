import torch
from torch import nn, einsum
from einops.layers.torch import Reduce
from einops import repeat
from model.modules import PreNorm, Attention, FeedForward
import random


class SIA(nn.Module):
    def __init__(
        self,
        fusion_mode,
        out_token,
        latent_dim,
        feature_dim,
        attn_num_heads,
        attn_dim_head,
        attn_depth,
        attn_self_per_cross,
        attn_dropout,
        attn_ff_dropout,
        feat_mask_ratio,
        dim_item_feats,
        num_items,
        maxlen,
        device,
        latent_random,
        latent_without_pos,
        item_without_pos,
    ):
        """
        Args:
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
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.out_token = out_token
        self.fusion_mode = fusion_mode
        self.feat_mask_ratio = feat_mask_ratio
        self.id_embedding = nn.Embedding(
            num_embeddings=(num_items + 2), # 1~num_items+1
            embedding_dim=latent_dim,
            padding_idx=0,
        )
        self.pos_embedding = nn.Embedding(
            num_embeddings=(maxlen + 2), # 1~maxlen+1
            embedding_dim=latent_dim,
            padding_idx=0,
        )
        self.item_pos_embedding = nn.Embedding(
            num_embeddings=(maxlen + 1), # 1~maxlen
            embedding_dim=feature_dim,
            padding_idx=0,
        )
        self.feat_embeddings = nn.ModuleList([
            nn.Linear(feat_dim, feature_dim, bias=False)
            for feat_dim in dim_item_feats
        ])
        self.feat_embeddings.append(nn.Embedding(
            num_embeddings=(num_items + 1),
            embedding_dim=feature_dim,
            padding_idx=0
        ))

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
        self.latent_without_pos = latent_without_pos
        self.item_without_pos = item_without_pos

    def forward(self, batch_x):
        # ID sequences -> embedded latent vectors (x)
        seq_list, pos_list, item_feat_lists = batch_x
        batch_size, seqlen = seq_list.shape
        seqlen -= 1 # cls token
        id_emb = self.id_embedding(seq_list)
        pos_emb = self.pos_embedding(pos_list)
        x = repeat(self.latent, "1 l d -> b l d", b=batch_size) if self.latent_random else id_emb
        x = x if self.latent_without_pos else (x + pos_emb)
        # x = id_emb + pos_emb

        # Item features -> concatenated item features (item_feat)
        item_feat = []
        pos_items = []
        mask_items = []
        item_feat_lists.append(seq_list[:, :-1]) # append id attributes
        for item_feat_list, feat_embedding in zip(item_feat_lists, self.feat_embeddings):
            item_feat.append(feat_embedding(item_feat_list))
            if self.fusion_mode != "mean":
                pos_items.append(pos_list[:, :-1])
                if self.fusion_mode == "attn":
                    mask_items.append(torch.diag_embed(pos_list)[:, :, :-1])
                elif self.fusion_mode == "masked_attn":
                    mask_items.append((repeat(pos_list, "b n -> b m n", m=pos_list.shape[-1]) * repeat(pos_list, "b n -> b n m", m=pos_list.shape[-1])).tril()[:, :, :-1])
        if self.fusion_mode == "mean":
            item_feat = torch.stack(item_feat).mean(axis=0)
        else:
            item_feat = torch.cat(item_feat, axis=1)
            pos_items = torch.cat(pos_items, axis=1)
            item_pos_emb = self.item_pos_embedding(pos_items)
            item_feat = item_feat if self.item_without_pos else (item_feat + item_pos_emb)

        # Masks for Attention
        mask_latent = pos_list.unsqueeze(2).to(self.device)
        mask_self_attn = einsum("b i d, b j d -> b i j", mask_latent, mask_latent) > 0
        if self.fusion_mode == "mean":
            mask_items = pos_list[:, :-1].unsqueeze(2).to(self.device)
            mask_cross_attn = einsum("b i d, b j d -> b i j", mask_latent, mask_items) > 0
        else:
            mask_cross_attn = torch.cat(mask_items, axis=2) > 0
        
        unmask_feat = torch.ones((batch_size, seqlen + 1, seqlen), dtype=bool)
        mask_feat = torch.zeros((batch_size, seqlen + 1, seqlen), dtype=bool)
        num_feat = len(item_feat_lists)

        # Iterative Attention
        out = []
        out_idx = -1 if self.out_token == "cls" else -2
        for cross_attn, cross_ff, self_attns in self.layers:
            if self.training:
                random_mask = [mask_feat if random.random() < self.feat_mask_ratio else unmask_feat for _ in range(num_feat)]
                random_mask = torch.cat(random_mask, axis=-1).to(self.device)
                x = cross_attn(x, context=item_feat, mask=mask_cross_attn * random_mask) + x
            else:
                x = cross_attn(x, context=item_feat, mask=mask_cross_attn) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x, mask=mask_self_attn) + x
                x = self_ff(x) + x

            out.append(self.out_ln(x[:, out_idx, :]))

        return out