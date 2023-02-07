import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import SAB, ISAB, PMA


class SIA(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
    
    def forward(self):
        pass


class SetTransformer(nn.Module):
    """
    Reference: https://github.com/juho-lee/set_transformer/blob/master/modules.py
    """

    def __init__(
        self, dim_input, num_outputs, dim_output, num_inds=32, dim_hidden=128, num_heads=4, ln=False
    ):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        return self.dec(self.enc(X))
