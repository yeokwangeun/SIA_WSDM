import torch
from torch import nn, einsum
from einops import repeat
from einops.layers.torch import Reduce
from model.modules import SAB, ISAB, PMA, PreNorm, Attention, FeedForward


class SIA(nn.Module):
    def __init__(
        self,
        latent_dim,
        item_num_outputs,
        item_num_heads,
        item_num_latents,
        item_dim_hidden,
        attn_depth,
        attn_self_per_cross,
        attn_dropout,
        attn_ff_dropout,
        attn_num_heads,
        attn_dim_head,
        dim_item_feats,
        num_items,
        maxlen,
        device,
    ):
        """
        Args:
          - latent_dim: Dimension of a latent vector.
          - item_num_output: Number of outputs of the set transformer per item features.
          - item_num_heads: Number of heads used for attention operations in set transformer.
          - item_num_latents: Number of latents used for ISAB in set transformer.
          - item_dim_hidden: Dimension of hidden vectors for attention operations in set transformer.
          - attn_depth: Depth of the iterative attention.
          - attn_self_per_cross: Number of self attention blocks per cross attention.
          - attn_dropout: Attention dropout in the iterative attention.
          - attn_ff_dropout: Feedforward dropout in the iterative attention.
          - attn_num_heads: Number of heads for attention operation in the iterative attention.
          - attn_dim_head: Dimension for attention operation in the iterative attention.
          - dim_item_feats: List of dimensions for item features.
          - num_items: Number of all items in the dataset.
          - maxlen: Max length of the sequence.
          - device: Device type.
        """
        super().__init__()
        self.feat_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    SAB(feat_dim, item_dim_hidden, item_num_heads, ln=True),
                    SAB(item_dim_hidden, item_dim_hidden, item_num_heads, ln=True),
                )
                for feat_dim in dim_item_feats
            ]
        )
        self.feats_decoder = nn.Sequential(
            SAB(item_dim_hidden, item_dim_hidden, item_num_heads, ln=True),
            SAB(item_dim_hidden, item_dim_hidden, item_num_heads, ln=True),
            PMA(item_dim_hidden, item_num_heads, item_num_outputs, ln=True),
            SAB(item_dim_hidden, item_dim_hidden, item_num_heads, ln=True),
            SAB(item_dim_hidden, item_dim_hidden, item_num_heads, ln=True),
            nn.Linear(item_dim_hidden, latent_dim)
        )
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
                context_dim=latent_dim,
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

        self.layers = nn.ModuleList([])
        for _ in range(attn_depth):
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
            self.layers.append(
                nn.ModuleList(
                    [
                        get_cross_attn(),
                        get_ff(),
                        self_attns,
                    ]
                )
            )
        self.to_logits = nn.Sequential(
            Reduce("b n d -> b d", "mean"),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_items + 1),
        )
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, batch_x):
        # ID sequences -> embedded latent vectors (x)
        seq_list, pos_list, *item_feat_lists = batch_x
        id_emb = self.id_embedding(seq_list)
        pos_emb = self.pos_embedding(pos_list)
        x = id_emb + pos_emb

        # Item features -> concatenated item features (item_feat)
        item_feat = []
        encoded = []
        for encoder, item_feat_list in zip(self.feat_encoders, item_feat_lists):
            encoded.append([encoder(feat.unsqueeze(0)) for feat in item_feat_list]) 
        for encs in zip(*encoded):
            encs = torch.cat(encs, axis=1)
            item_feat.append(self.feats_decoder(encs))
        item_feat = torch.cat(item_feat)

        # Masks for attention
        mask_latent = repeat(pos_list, "b n -> b n d", d=self.latent_dim).float()
        mask_items = torch.ones(item_feat.shape)
        mask_latent = mask_latent.to(self.device)
        mask_items = mask_items.to(self.device)
        mask_cross_attn = einsum("b i d, b j d -> b i j", mask_latent, mask_items) > 0
        mask_self_attn = einsum("b i d, b j d -> b i j", mask_latent, mask_latent) > 0

        # Iterative Attention
        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context=item_feat, mask=mask_cross_attn) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x, mask=mask_self_attn) + x
                x = self_ff(x) + x

        # To items
        x = (mask_latent > 0) * x
        return self.to_logits(x)


class SetTransformer(nn.Module):
    """
    Reference: https://github.com/juho-lee/set_transformer/blob/master/modules.py
    """

    def __init__(
        self, dim_input, num_outputs, dim_output, num_inds=32, dim_hidden=128, num_heads=4, ln=False
    ):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            # ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            # ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            SAB(dim_input, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        )
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        return self.dec(self.enc(X))
