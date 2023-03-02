import torch
from torch import nn, einsum
from einops.layers.torch import Reduce
from model.modules import PreNorm, Attention, FeedForward


class SIA(nn.Module):
    def __init__(
        self,
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
        self.id_embedding = nn.Embedding(
            num_embeddings=(num_items + 1),
            embedding_dim=latent_dim,
            padding_idx=0,
        )
        self.pos_embedding = nn.Embedding(
            num_embeddings=(maxlen + 1),
            embedding_dim=latent_dim,
            padding_idx=0,
        )
        self.feat_embeddings = nn.ModuleList([
            nn.Linear(feat_dim, feature_dim, bias=False)
            for feat_dim in dim_item_feats
        ])

        get_latent_attn = lambda: PreNorm(
            latent_dim,
            Attention(
                query_dim=latent_dim,
                heads=attn_num_heads,
                dim_head=attn_dim_head,
                dropout=attn_dropout,
            ),
        )
        get_cross_attn = lambda: PreNorm(
            latent_dim,
            Attention(
                query_dim=latent_dim,
                context_dim=feature_dim,
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
        self.cross_attn = get_cross_attn()
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
        self.to_logits = nn.Sequential(
            Reduce("b n d -> b d", "mean"),
            nn.LayerNorm(latent_dim),
        )
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, batch_x):
        # ID sequences -> embedded latent vectors (x)
        seq_list, pos_list, item_feat_lists = batch_x
        id_emb = self.id_embedding(seq_list)
        pos_emb = self.pos_embedding(pos_list)
        x = id_emb + pos_emb

        # Item features -> concatenated item features (item_feat)
        item_feat = []
        mask_items = []
        for item_feat_list, fc in zip(item_feat_lists, self.feat_embeddings):
            item_feat.append(fc(item_feat_list))
            mask_items.append(pos_list)
        item_feat = torch.cat(item_feat, axis=1)
        mask_items = torch.cat(mask_items, axis=1)

        mask_latent = pos_list.unsqueeze(2).to(self.device)
        mask_items = mask_items.unsqueeze(2).to(self.device)
        mask_cross_attn = einsum("b i d, b j d -> b i j", mask_latent, mask_items) > 0
        mask_self_attn = einsum("b i d, b j d -> b i j", mask_latent, mask_latent) > 0

        # Iterative Attention
        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context=item_feat, mask=mask_cross_attn) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x, mask=mask_self_attn) + x
                x = self_ff(x) + x

        x = (mask_latent > 0) * x
        return self.to_logits(x)